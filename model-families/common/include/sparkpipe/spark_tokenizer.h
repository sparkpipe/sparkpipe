#ifndef SPARKPIPE_SPARK_TOKENIZER_H
#define SPARKPIPE_SPARK_TOKENIZER_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TOKENIZER_ABI_VERSION 1u
#define SPARK_TOKENIZER_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizer))
#define SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerHuggingFaceJsonConfiguration))
#define SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerCompiledFileConfiguration))
#define SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerEncoding))
#define SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerWorkspace))
#define SPARK_TOKENIZER_BATCH_ENCODE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerBatchEncodeConfiguration))
#define SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL 1u
// Pretoken cache: a byte-level BPE re-encodes the same short words millions of
// times on natural text (word frequencies are Zipfian), so memoizing the token
// sequence a piece produces converts the dominant merge-loop cost into a hash
// lookup on repeats. The cache is thread-local in the workspace so there is no
// contention. Pieces up to the inline byte bound are cached; longer pieces (rare
// in BPE, which splits on whitespace and punctuation) bypass the cache and encode
// directly. The slot count is a power of two for masking; the token pool holds
// the cached id sequences and is bounded, with new inserts skipped once full.
#define SPARK_TOKENIZER_PIECE_CACHE_INLINE_BYTES 32u
#define SPARK_TOKENIZER_PIECE_CACHE_SLOT_COUNT 16384u
#define SPARK_TOKENIZER_PIECE_CACHE_TOKEN_CAPACITY 262144u
#define SPARK_TOKENIZER_PIECE_CACHE_EMPTY_HASH 0u
#define SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES 256u
#define SPARK_TOKENIZER_NO_TOKEN_ID 0xffffffffu
#define SPARK_TOKENIZER_BYTE_COUNT 256u
#define SPARK_TOKENIZER_MAX_PARALLEL_WORKER_COUNT 16u
#define SPARK_TOKENIZER_COMPILED_FILE_MAGIC 0x314b4f54535053ull
#define SPARK_TOKENIZER_COMPILED_FILE_VERSION 1u

#define SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH 0x00000001u
#define SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE 0x00000002u
#define SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION 0x00000004u
#define SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE 0x00000008u
#define SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS \
    (SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH | \
     SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE | \
     SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION | \
     SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE)
#define SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS 0x00000001u
#define SPARK_TOKENIZER_DECODE_KNOWN_FLAGS \
    SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS

typedef struct SparkTokenizerStringEntry
{
    char *text;
    uint32_t text_bytes;
    uint32_t value;
    uint32_t next_index;
} SparkTokenizerStringEntry;

typedef struct SparkTokenizerSpecialToken
{
    char *text;
    uint32_t text_bytes;
    uint32_t token_id;
    uint32_t reserved0;
} SparkTokenizerSpecialToken;

typedef struct SparkTokenizerFastMergePair
{
    uint32_t left_token_id;
    uint32_t right_token_id;
    uint32_t merged_token_id;
    uint32_t rank;
    uint32_t next_index;
    uint32_t reserved0;
} SparkTokenizerFastMergePair;

typedef struct SparkTokenizerMergeCandidate
{
    uint32_t rank;
    uint32_t left_symbol_index;
    uint32_t left_generation;
    uint32_t right_generation;
} SparkTokenizerMergeCandidate;

typedef struct SparkTokenizerHuggingFaceJsonConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *tokenizer_json_path;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerHuggingFaceJsonConfiguration;

typedef struct SparkTokenizerCompiledFileConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *compiled_tokenizer_path;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerCompiledFileConfiguration;

typedef struct SparkTokenizerEncoding
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t token_count;
    uint32_t overflow_token_count;
    uint32_t invalid_segment_count;
    uint32_t token_capacity;
    uint32_t *token_ids;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerEncoding;

typedef struct SparkTokenizerPieceCacheEntry
{
    uint64_t hash;
    uint32_t piece_bytes;
    uint32_t token_offset;
    uint32_t token_count;
    uint8_t piece[SPARK_TOKENIZER_PIECE_CACHE_INLINE_BYTES];
} SparkTokenizerPieceCacheEntry;

typedef struct SparkTokenizerWorkspace
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    void *arena;
    uint32_t arena_bytes;
    uint32_t symbol_capacity;
    uint32_t heap_capacity;
    uint32_t maximum_symbol_count;
    uint32_t *symbol_token_ids;
    uint32_t *previous_symbol_indices;
    uint32_t *next_symbol_indices;
    uint32_t *symbol_generations;
    SparkTokenizerMergeCandidate *merge_heap;
    uint32_t merge_heap_count;
    uint32_t piece_cache_slot_count;
    uint32_t piece_cache_token_capacity;
    uint32_t piece_cache_token_used;
    struct SparkTokenizerPieceCacheEntry *piece_cache_entries;
    uint32_t *piece_cache_token_pool;
} SparkTokenizerWorkspace;

typedef struct SparkTokenizerBatchEncodeConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *const *texts;
    const uint32_t *text_bytes;
    uint32_t text_count;
    uint32_t encode_flags;
    uint32_t *token_ids;
    uint32_t token_stride;
    uint32_t *token_counts;
    uint32_t *overflow_token_counts;
    uint32_t *invalid_segment_counts;
    uint32_t worker_count;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerBatchEncodeConfiguration;

typedef struct SparkTokenizer
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t model_kind;
    uint32_t add_prefix_space;
    uint32_t byte_fallback;
    uint32_t has_unk_token;
    uint32_t unk_token_id;
    uint32_t maximum_token_id;
    uint32_t byte_level_use_regex;
    char **token_text_by_id;
    uint32_t *token_text_bytes_by_id;
    uint32_t vocabulary_count;
    SparkTokenizerStringEntry *vocabulary_entries;
    uint32_t *vocabulary_buckets;
    uint32_t vocabulary_bucket_count;
    uint32_t merge_count;
    SparkTokenizerStringEntry *merge_entries;
    uint32_t *merge_buckets;
    uint32_t merge_bucket_count;
    uint32_t special_token_count;
    SparkTokenizerSpecialToken *special_tokens;
    uint32_t byte_token_ids[256u];
    uint32_t fast_merge_pair_count;
    SparkTokenizerFastMergePair *fast_merge_pairs;
    uint32_t *fast_merge_buckets;
    uint32_t fast_merge_bucket_count;
    uint32_t default_worker_count;
} SparkTokenizer;

void SparkTokenizerReset(
    SparkTokenizer *tokenizer);

void SparkTokenizerDestroy(
    SparkTokenizer *tokenizer);

void SparkTokenizerEncodingReset(
    SparkTokenizerEncoding *encoding);

void SparkTokenizerWorkspaceReset(
    SparkTokenizerWorkspace *workspace);

void SparkTokenizerWorkspaceDestroy(
    SparkTokenizerWorkspace *workspace);

SparkStatus SparkTokenizerWorkspaceInitialize(
    SparkTokenizerWorkspace *workspace,
    uint32_t maximum_symbol_count);

SparkStatus SparkTokenizerFindTokenId(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t *token_id_out);

SparkStatus SparkTokenizerLoadHuggingFaceJson(
    SparkTokenizer *tokenizer,
    const SparkTokenizerHuggingFaceJsonConfiguration *configuration);

SparkStatus SparkTokenizerLoadCompiledFile(
    SparkTokenizer *tokenizer,
    const SparkTokenizerCompiledFileConfiguration *configuration);

SparkStatus SparkTokenizerSaveCompiledFile(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerCompiledFileConfiguration *configuration);

SparkStatus SparkTokenizerEncodeUtf8(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerEncoding *encoding);

SparkStatus SparkTokenizerEncodeUtf8WithWorkspace(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding);

SparkStatus SparkTokenizerEncodeBatchUtf8(
    const SparkTokenizer *tokenizer,
    const char *const *texts,
    const uint32_t *text_bytes,
    uint32_t text_count,
    uint32_t encode_flags,
    uint32_t *token_ids,
    uint32_t token_stride,
    uint32_t *token_counts,
    uint32_t *overflow_token_counts);

SparkStatus SparkTokenizerEncodeBatchUtf8Configured(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerBatchEncodeConfiguration *configuration);

SparkStatus SparkTokenizerDecodeTokenIds(
    const SparkTokenizer *tokenizer,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t decode_flags,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_out);

#ifdef __cplusplus
}
#endif

#endif
