#include "routing_trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "routing trace self-test failed: %s\n", message); \
        return 1; \
    } \
} while (0)

static size_t count_occurrences(const char *text, const char *needle) {
    size_t count = 0;
    size_t needle_length = strlen(needle);
    while ((text = strstr(text, needle)) != NULL) {
        count++;
        text += needle_length;
    }
    return count;
}

static RoutingTraceRecord make_record(uint32_t token_position, uint32_t layer) {
    RoutingTraceRecord record = {0};
    record.phase = token_position == 9 ? 1 : 0;
    record.token_position = token_position;
    record.layer = layer;
    for (uint32_t index = 0; index < ROUTING_TRACE_TOP_K; ++index) {
        record.expert_ids[index] = (int32_t)(index * 7 + layer);
        record.weights[index] = 0.1f + (float)index * 0.01f;
        record.cache_states[index] = (uint8_t)(1 + index % 4);
        record.route_bytes[index] = index >= 4 ? 13369344u : 0u;
    }
    record.pread_bytes = 2u * 13369344u;
    record.io_wait_ns = 123456u;
    record.short_reads = 0;
    record.io_status = 0;
    return record;
}

int main(void) {
    char path[] = "/tmp/dmlx-routing-trace-self-test-XXXXXX";
    int temporary_fd = mkstemp(path);
    CHECK(temporary_fd >= 0, "mkstemp");
    close(temporary_fd);
    CHECK(unlink(path) == 0, "remove mkstemp placeholder");

    // Default-off semantics: no writer means no allocation, append, or file.
    RoutingTraceRecord record = make_record(7, 3);
    CHECK(routing_trace_append(NULL, &record) == 0, "disabled append is a no-op");
    CHECK(access(path, F_OK) != 0, "disabled mode created a file");

    RoutingTraceWriter *invalid_writer = NULL;
    CHECK(routing_trace_writer_create(
              &invalid_writer,
              "/tmp/dmlx-routing-trace-missing-parent/file.jsonl",
              2,
              UINT64_C(123),
              43,
              256,
              ROUTING_TRACE_TOP_K,
              20,
              13369344) != 0,
          "open failure was not reported");
    CHECK(invalid_writer == NULL, "failed create returned a writer");

    RoutingTraceWriter *writer = NULL;
    CHECK(routing_trace_writer_create(
              &writer,
              path,
              2,
              UINT64_C(1234567890123456789),
              43,
              256,
              ROUTING_TRACE_TOP_K,
              20,
              13369344) == 0,
          "create writer");
    CHECK(writer != NULL, "writer is null");

    struct stat mode_stat;
    CHECK(stat(path, &mode_stat) == 0, "stat trace");
    CHECK((mode_stat.st_mode & 0777) == 0600, "trace permissions are not 0600");

    CHECK(routing_trace_request_begin(writer, 0) == 0, "request begin");
    CHECK(routing_trace_append(writer, &record) == 1, "append first record");
    record = make_record(8, 4);
    CHECK(routing_trace_append(writer, &record) == 1, "append second record");
    record = make_record(9, 5);
    CHECK(routing_trace_append(writer, &record) == 0, "capacity drop not reported");
    record = make_record(10, 6);
    record.weights[2] = NAN;
    CHECK(routing_trace_append(writer, &record) == 0, "invalid record not rejected");
    CHECK(routing_trace_request_end(writer) == 0, "request end");

    RoutingTraceStats stats;
    routing_trace_get_stats(writer, &stats);
    CHECK(stats.capacity == 2, "capacity stat");
    CHECK(stats.requests_started == 1, "request count");
    CHECK(stats.records_attempted == 4, "attempt count");
    CHECK(stats.records_written == 2, "written count");
    CHECK(stats.records_dropped == 2, "drop count");
    CHECK(stats.invalid_records == 1, "invalid count");
    CHECK(stats.write_errors == 0, "unexpected write error");
    CHECK(!stats.request_active, "request still active");
    routing_trace_writer_destroy(writer);

    FILE *trace = fopen(path, "rb");
    CHECK(trace != NULL, "open trace output");
    CHECK(fseek(trace, 0, SEEK_END) == 0, "seek end");
    long length = ftell(trace);
    CHECK(length > 0, "empty trace");
    CHECK(fseek(trace, 0, SEEK_SET) == 0, "seek start");
    char *content = (char *)calloc((size_t)length + 1, 1);
    CHECK(content != NULL, "allocate output buffer");
    CHECK(fread(content, 1, (size_t)length, trace) == (size_t)length, "read output");
    fclose(trace);

    const char *header = strstr(content, "\"kind\":\"header\"");
    const char *begin = strstr(content, "\"kind\":\"request_begin\"");
    const char *route = strstr(content, "\"kind\":\"route\"");
    const char *end = strstr(content, "\"kind\":\"request_end\"");
    CHECK(header && begin && route && end, "missing schema record kind");
    CHECK(header < begin && begin < route && route < end, "record ordering");
    CHECK(count_occurrences(content, "\"kind\":\"route\"") == 2, "bounded route count");
    CHECK(strstr(content, "\"schema_version\":1") != NULL, "schema version");
    CHECK(strstr(content, "\"model_fingerprint_u64\":1234567890123456789") != NULL,
          "model fingerprint");
    CHECK(strstr(content, "\"expert_ids\":[3,10,17,24,31,38]") != NULL,
          "six expert ids");
    CHECK(strstr(content, "\"cache_states\":[1,2,3,4,1,2]") != NULL,
          "cache classes");
    CHECK(strstr(content, "\"route_bytes\":[0,0,0,0,13369344,13369344]") != NULL,
          "route bytes");
    CHECK(strstr(content, "\"records_dropped\":2") != NULL, "visible drops");
    CHECK(strstr(content, "\"invalid_records\":1") != NULL, "visible invalid records");
    CHECK(strstr(content, "\"write_errors\":0") != NULL, "visible write errors");
    CHECK(strstr(content, "prompt") == NULL, "prompt field leaked");
    CHECK(strstr(content, "token_id") == NULL, "token id field leaked");
    CHECK(strstr(content, "token_string") == NULL, "token string field leaked");
    CHECK(strstr(content, "request_body") == NULL, "request body field leaked");
    CHECK(strstr(content, "nan") == NULL && strstr(content, "NaN") == NULL,
          "non-finite JSON leaked");

    free(content);
    CHECK(unlink(path) == 0, "cleanup trace");
    fprintf(stderr, "routing trace self-test: PASS\n");
    return 0;
}
