#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROUTING_TRACE_SCHEMA_VERSION 1u
#define ROUTING_TRACE_TOP_K 6u

typedef enum RoutingTraceCacheState {
    ROUTING_TRACE_CACHE_UNKNOWN = 0,
    ROUTING_TRACE_CACHE_HOT = 1,
    ROUTING_TRACE_CACHE_PREFETCH = 2,
    ROUTING_TRACE_CACHE_BATCH_POOL = 3,
    ROUTING_TRACE_CACHE_COLD = 4,
} RoutingTraceCacheState;

typedef struct RoutingTraceRecord {
    uint64_t request_id;
    uint64_t event_seq;
    uint32_t phase;
    uint32_t token_position;
    uint32_t layer;
    int32_t expert_ids[ROUTING_TRACE_TOP_K];
    float weights[ROUTING_TRACE_TOP_K];
    uint8_t cache_states[ROUTING_TRACE_TOP_K];
    uint8_t reserved[2];
    uint64_t route_bytes[ROUTING_TRACE_TOP_K];
    uint64_t pread_bytes;
    uint64_t io_wait_ns;
    uint32_t short_reads;
    int32_t io_status;
} RoutingTraceRecord;

typedef struct RoutingTraceStats {
    uint64_t requests_started;
    uint64_t records_attempted;
    uint64_t records_written;
    uint64_t records_dropped;
    uint64_t invalid_records;
    uint64_t write_errors;
    size_t capacity;
    size_t buffered_records;
    bool request_active;
} RoutingTraceStats;

typedef struct RoutingTraceWriter RoutingTraceWriter;

// Opens path with mode 0600 and allocates exactly capacity fixed-size records.
// No prompt, token id/string, request body, or content-derived identifier is
// accepted by this API. model_fingerprint is an artifact-metadata fingerprint.
int routing_trace_writer_create(
    RoutingTraceWriter **out_writer,
    const char *path,
    size_t capacity,
    uint64_t model_fingerprint,
    uint32_t n_layers,
    uint32_t n_experts,
    uint32_t top_k,
    uint32_t cache_capacity,
    uint64_t expert_size_bytes);

void routing_trace_writer_destroy(RoutingTraceWriter *writer);

// Request identifiers are opaque process-local numeric correlations generated
// by the writer. They are never derived from prompt or token content.
int routing_trace_request_begin(RoutingTraceWriter *writer, uint32_t initial_phase);
int routing_trace_append(RoutingTraceWriter *writer, const RoutingTraceRecord *record);
int routing_trace_request_end(RoutingTraceWriter *writer);
void routing_trace_get_stats(const RoutingTraceWriter *writer, RoutingTraceStats *out_stats);

#ifdef __cplusplus
}
#endif
