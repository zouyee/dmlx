#include "routing_trace.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct RoutingTraceWriter {
    FILE *stream;
    RoutingTraceRecord *records;
    size_t capacity;
    size_t count;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;
    uint64_t process_nonce;
    uint64_t request_sequence;
    uint64_t active_request_id;
    uint64_t next_event_seq;
    uint64_t request_dropped;
    uint64_t request_invalid;
    RoutingTraceStats stats;
};

static uint64_t mix_u64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t make_process_nonce(void) {
    struct timespec monotonic = {0, 0};
    struct timespec realtime = {0, 0};
    (void)clock_gettime(CLOCK_MONOTONIC, &monotonic);
    (void)clock_gettime(CLOCK_REALTIME, &realtime);
    uint64_t seed = (uint64_t)monotonic.tv_sec ^ ((uint64_t)monotonic.tv_nsec << 17);
    seed ^= ((uint64_t)realtime.tv_sec << 7) ^ (uint64_t)realtime.tv_nsec;
    seed ^= (uint64_t)(unsigned int)getpid() << 32;
    return mix_u64(seed);
}

static int write_failed(RoutingTraceWriter *writer) {
    writer->stats.write_errors++;
    return -1;
}

static int write_header(
    RoutingTraceWriter *writer,
    uint64_t model_fingerprint,
    uint32_t cache_capacity,
    uint64_t expert_size_bytes) {
    if (fprintf(writer->stream,
            "{\"kind\":\"header\",\"schema_version\":%u,"
            "\"model_fingerprint_kind\":\"fnv1a64_artifact_metadata_v1\","
            "\"model_fingerprint_u64\":%" PRIu64 ","
            "\"n_layers\":%u,\"n_experts\":%u,\"top_k\":%u,"
            "\"cache_capacity\":%u,\"expert_size_bytes\":%" PRIu64 ","
            "\"record_capacity\":%zu}\n",
            ROUTING_TRACE_SCHEMA_VERSION,
            model_fingerprint,
            writer->n_layers,
            writer->n_experts,
            writer->top_k,
            cache_capacity,
            expert_size_bytes,
            writer->capacity) < 0) {
        return write_failed(writer);
    }
    if (fflush(writer->stream) != 0) return write_failed(writer);
    return 0;
}

int routing_trace_writer_create(
    RoutingTraceWriter **out_writer,
    const char *path,
    size_t capacity,
    uint64_t model_fingerprint,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t top_k,
    uint32_t cache_capacity,
    uint64_t expert_size_bytes) {
    if (!out_writer || !path || path[0] == '\0' || capacity == 0 ||
        n_layers == 0 || n_experts == 0 || top_k != ROUTING_TRACE_TOP_K) {
        errno = EINVAL;
        return -1;
    }
    *out_writer = NULL;

    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd < 0) return -1;
    if (fchmod(fd, S_IRUSR | S_IWUSR) != 0) {
        const int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }

    FILE *stream = fdopen(fd, "w");
    if (!stream) {
        const int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }

    RoutingTraceWriter *writer = (RoutingTraceWriter *)calloc(1, sizeof(*writer));
    if (!writer) {
        fclose(stream);
        return -1;
    }
    writer->records = (RoutingTraceRecord *)calloc(capacity, sizeof(*writer->records));
    if (!writer->records) {
        const int saved_errno = errno;
        free(writer);
        fclose(stream);
        errno = saved_errno;
        return -1;
    }

    writer->stream = stream;
    writer->capacity = capacity;
    writer->n_layers = n_layers;
    writer->n_experts = n_experts;
    writer->top_k = top_k;
    writer->process_nonce = make_process_nonce();
    writer->stats.capacity = capacity;
    (void)setvbuf(stream, NULL, _IOFBF, 64 * 1024);

    if (write_header(writer, model_fingerprint, cache_capacity, expert_size_bytes) != 0) {
        const int saved_errno = errno;
        routing_trace_writer_destroy(writer);
        errno = saved_errno != 0 ? saved_errno : EIO;
        return -1;
    }

    *out_writer = writer;
    return 0;
}

void routing_trace_writer_destroy(RoutingTraceWriter *writer) {
    if (!writer) return;
    if (writer->stats.request_active) (void)routing_trace_request_end(writer);
    if (writer->stream) fclose(writer->stream);
    free(writer->records);
    memset(writer, 0, sizeof(*writer));
    free(writer);
}

int routing_trace_request_begin(RoutingTraceWriter *writer, uint32_t initial_phase) {
    if (!writer || initial_phase > 2u || writer->stats.request_active) {
        errno = EINVAL;
        return -1;
    }

    writer->count = 0;
    writer->next_event_seq = 0;
    writer->request_dropped = 0;
    writer->request_invalid = 0;
    writer->request_sequence++;
    writer->active_request_id = mix_u64(writer->process_nonce ^ writer->request_sequence);
    writer->stats.request_active = true;
    writer->stats.requests_started++;
    writer->stats.buffered_records = 0;

    if (fprintf(writer->stream,
            "{\"kind\":\"request_begin\",\"schema_version\":%u,"
            "\"request_id\":%" PRIu64 ",\"initial_phase\":%u}\n",
            ROUTING_TRACE_SCHEMA_VERSION,
            writer->active_request_id,
            initial_phase) < 0) {
        writer->stats.request_active = false;
        return write_failed(writer);
    }
    return 0;
}

static bool record_is_valid(const RoutingTraceWriter *writer, const RoutingTraceRecord *record) {
    if (record->phase > 2u || record->layer >= writer->n_layers) return false;
    for (uint32_t index = 0; index < writer->top_k; ++index) {
        if (record->expert_ids[index] < 0 ||
            (uint32_t)record->expert_ids[index] >= writer->n_experts) return false;
        if (!isfinite(record->weights[index])) return false;
        if (record->cache_states[index] < ROUTING_TRACE_CACHE_HOT ||
            record->cache_states[index] > ROUTING_TRACE_CACHE_COLD) return false;
    }
    return true;
}

int routing_trace_append(RoutingTraceWriter *writer, const RoutingTraceRecord *record) {
    if (!writer || !record || !writer->stats.request_active) return 0;

    writer->stats.records_attempted++;
    const uint64_t event_seq = writer->next_event_seq++;
    if (!record_is_valid(writer, record)) {
        writer->request_invalid++;
        writer->request_dropped++;
        writer->stats.invalid_records++;
        writer->stats.records_dropped++;
        return 0;
    }
    if (writer->count >= writer->capacity) {
        writer->request_dropped++;
        writer->stats.records_dropped++;
        return 0;
    }

    RoutingTraceRecord *destination = &writer->records[writer->count++];
    *destination = *record;
    destination->request_id = writer->active_request_id;
    destination->event_seq = event_seq;
    writer->stats.buffered_records = writer->count;
    return 1;
}

static int write_u32_array(FILE *stream, const int32_t *values, uint32_t count) {
    if (fputc('[', stream) == EOF) return -1;
    for (uint32_t index = 0; index < count; ++index) {
        if (index > 0 && fputc(',', stream) == EOF) return -1;
        if (fprintf(stream, "%" PRId32, values[index]) < 0) return -1;
    }
    return fputc(']', stream) == EOF ? -1 : 0;
}

static int write_float_array(FILE *stream, const float *values, uint32_t count) {
    if (fputc('[', stream) == EOF) return -1;
    for (uint32_t index = 0; index < count; ++index) {
        if (index > 0 && fputc(',', stream) == EOF) return -1;
        if (fprintf(stream, "%.9g", (double)values[index]) < 0) return -1;
    }
    return fputc(']', stream) == EOF ? -1 : 0;
}

static int write_u8_array(FILE *stream, const uint8_t *values, uint32_t count) {
    if (fputc('[', stream) == EOF) return -1;
    for (uint32_t index = 0; index < count; ++index) {
        if (index > 0 && fputc(',', stream) == EOF) return -1;
        if (fprintf(stream, "%u", (unsigned int)values[index]) < 0) return -1;
    }
    return fputc(']', stream) == EOF ? -1 : 0;
}

static int write_u64_array(FILE *stream, const uint64_t *values, uint32_t count) {
    if (fputc('[', stream) == EOF) return -1;
    for (uint32_t index = 0; index < count; ++index) {
        if (index > 0 && fputc(',', stream) == EOF) return -1;
        if (fprintf(stream, "%" PRIu64, values[index]) < 0) return -1;
    }
    return fputc(']', stream) == EOF ? -1 : 0;
}

static int write_record(RoutingTraceWriter *writer, const RoutingTraceRecord *record) {
    FILE *stream = writer->stream;
    if (fprintf(stream,
            "{\"kind\":\"route\",\"schema_version\":%u,"
            "\"request_id\":%" PRIu64 ",\"event_seq\":%" PRIu64 ","
            "\"phase\":%u,\"token_position\":%u,\"layer\":%u,"
            "\"expert_ids\":",
            ROUTING_TRACE_SCHEMA_VERSION,
            record->request_id,
            record->event_seq,
            record->phase,
            record->token_position,
            record->layer) < 0) return write_failed(writer);
    if (write_u32_array(stream, record->expert_ids, writer->top_k) != 0 ||
        fputs(",\"weights\":", stream) == EOF ||
        write_float_array(stream, record->weights, writer->top_k) != 0 ||
        fputs(",\"cache_states\":", stream) == EOF ||
        write_u8_array(stream, record->cache_states, writer->top_k) != 0 ||
        fputs(",\"route_bytes\":", stream) == EOF ||
        write_u64_array(stream, record->route_bytes, writer->top_k) != 0 ||
        fprintf(stream,
            ",\"pread_bytes\":%" PRIu64 ",\"io_wait_ns\":%" PRIu64
            ",\"short_reads\":%u,\"io_status\":%" PRId32 "}\n",
            record->pread_bytes,
            record->io_wait_ns,
            record->short_reads,
            record->io_status) < 0) {
        return write_failed(writer);
    }
    writer->stats.records_written++;
    return 0;
}

int routing_trace_request_end(RoutingTraceWriter *writer) {
    if (!writer || !writer->stats.request_active) return 0;

    int status = 0;
    uint64_t request_written = 0;
    for (size_t index = 0; index < writer->count; ++index) {
        if (write_record(writer, &writer->records[index]) != 0) {
            status = -1;
            break;
        }
        request_written++;
    }

    const uint64_t visible_write_errors = writer->stats.write_errors;
    if (fprintf(writer->stream,
            "{\"kind\":\"request_end\",\"schema_version\":%u,"
            "\"request_id\":%" PRIu64 ",\"records_attempted\":%" PRIu64 ","
            "\"records_buffered\":%zu,\"records_written\":%" PRIu64 ","
            "\"records_dropped\":%" PRIu64 ",\"invalid_records\":%" PRIu64 ","
            "\"write_errors\":%" PRIu64 "}\n",
            ROUTING_TRACE_SCHEMA_VERSION,
            writer->active_request_id,
            writer->next_event_seq,
            writer->count,
            request_written,
            writer->request_dropped,
            writer->request_invalid,
            visible_write_errors) < 0) {
        status = write_failed(writer);
    }
    if (fflush(writer->stream) != 0) status = write_failed(writer);

    writer->stats.request_active = false;
    writer->stats.buffered_records = 0;
    writer->count = 0;
    writer->active_request_id = 0;
    return status;
}

void routing_trace_get_stats(const RoutingTraceWriter *writer, RoutingTraceStats *out_stats) {
    if (!out_stats) return;
    memset(out_stats, 0, sizeof(*out_stats));
    if (!writer) return;
    *out_stats = writer->stats;
    out_stats->capacity = writer->capacity;
    out_stats->buffered_records = writer->count;
}
