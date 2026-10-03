#ifndef SPARKPIPE_SPARK_BYTE_DRAFT_H
#define SPARKPIPE_SPARK_BYTE_DRAFT_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_BYTE_DRAFT_MAX_TAIL_BYTES 4096u
#define SPARK_BYTE_DRAFT_MAX_FORESTS 8u
#define SPARK_BYTE_DRAFT_MAX_CANDIDATES 256u
#define SPARK_BYTE_DRAFT_MAX_CONTINUATION_BYTES 65536u
#define SPARK_BYTE_DRAFT_MAX_BUDGET 2080768u
#define SPARK_BYTE_DRAFT_MAX_RESPONSE_BYTES 2097152u
#define SPARK_BYTE_DRAFT_BANK_ID_BYTES 64u
#define SPARK_BYTE_DRAFT_DIGEST_HEX_BYTES 65u
#define SPARK_BYTE_DRAFT_MAX_WINDOW_TOKENS 256u

#define SPARK_BYTE_DRAFT_STOP_COMPLETE 1u
#define SPARK_BYTE_DRAFT_STOP_CANDIDATE_LIMIT 2u
#define SPARK_BYTE_DRAFT_STOP_BUDGET_EXHAUSTED 3u
#define SPARK_BYTE_DRAFT_STOP_WORK_EXHAUSTED 4u
#define SPARK_BYTE_DRAFT_STOP_PENDING_INDEX 5u
#define SPARK_BYTE_DRAFT_STOP_CANCELLED 6u

typedef struct SparkByteDraftWorkBudget
{
    uint64_t index_steps;
    uint64_t candidates_checked;
    uint64_t physical_bytes_read;
    uint64_t decoded_bytes;
    uint64_t peak_scratch_bytes;
    uint64_t cache_bytes;
    uint64_t elapsed_ms;
} SparkByteDraftWorkBudget;

typedef struct SparkByteDraftForest
{
    const char *alias;
    const char *bank_id;
    uint64_t snapshot;
} SparkByteDraftForest;

typedef struct SparkByteDraftConfiguration
{
    const char *host;
    uint16_t port;
    const char *path;
    const char *bearer_token;
    const SparkByteDraftForest *forests;
    uint32_t forest_count;
    const char *scope;
    uint32_t min_match_bytes;
    uint32_t max_candidates;
    uint32_t max_continuation_bytes;
    uint32_t budget;
    SparkByteDraftWorkBudget work_budget;
    uint32_t io_timeout_ms;
} SparkByteDraftConfiguration;

typedef struct SparkByteDraftCandidate
{
    char bank_id[SPARK_BYTE_DRAFT_BANK_ID_BYTES];
    char content_sha256[SPARK_BYTE_DRAFT_DIGEST_HEX_BYTES];
    uint64_t event;
    uint64_t match_start;
    uint64_t match_end;
    uint64_t continuation_end;
    uint32_t matched_bytes;
    uint32_t has_more;
    uint8_t *continuation;
    uint32_t continuation_bytes;
} SparkByteDraftCandidate;

typedef struct SparkByteDraftResult
{
    uint32_t complete;
    uint32_t stop_reason;
    int64_t service_error_code;
    uint32_t candidate_count;
    SparkByteDraftCandidate candidates[SPARK_BYTE_DRAFT_MAX_CANDIDATES];
} SparkByteDraftResult;

SparkStatus SparkByteDraftQuery(
    const SparkByteDraftConfiguration *configuration,
    const uint8_t *tail,
    uint32_t tail_bytes,
    SparkByteDraftResult *result);

void SparkByteDraftResultRelease(SparkByteDraftResult *result);

SparkStatus SparkByteDraftTail(
    const SparkTokenizer *tokenizer,
    const uint32_t *committed_ids,
    uint32_t committed_count,
    uint32_t window_tokens,
    uint8_t *tail,
    uint32_t tail_capacity,
    uint32_t *tail_bytes,
    uint32_t *tail_tokens);

SparkStatus SparkByteDraftAlign(
    const SparkTokenizer *tokenizer,
    const uint32_t *committed_ids,
    uint32_t committed_count,
    uint32_t window_tokens,
    const uint8_t *continuation,
    uint32_t continuation_bytes,
    uint32_t max_tokens,
    uint32_t *draft_ids,
    uint32_t *draft_count);

#ifdef __cplusplus
}
#endif

#endif
