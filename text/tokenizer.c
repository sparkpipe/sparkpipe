#include "sparkpipe/spark_tokenizer.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"

#define SPARK_TOKENIZER_EMPTY_BUCKET UINT32_MAX
#define SPARK_TOKENIZER_INITIAL_BYTE_SYMBOL_BYTES 4u
#define SPARK_TOKENIZER_HASH_LOAD_FACTOR 2u
#define SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_FACTOR 4u
#define SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_SLACK 16u
#define SPARK_TOKENIZER_SYMBOL_NONE UINT32_MAX

static uint32_t SparkTokenizerNextPowerOfTwo(
    uint32_t value)
{
    uint32_t power_of_two;

    power_of_two = 1u;
    while (power_of_two < value && power_of_two <= UINT32_MAX / 2u)
    {
        power_of_two <<= 1u;
    }
    return power_of_two;
}

static uint32_t SparkTokenizerHashBytes(
    const char *text,
    uint32_t text_bytes)
{
    uint32_t hash_value;
    uint32_t byte_index;

    hash_value = 2166136261u;
    for (byte_index = 0u; byte_index < text_bytes; ++byte_index)
    {
        hash_value ^= (uint8_t)text[byte_index];
        hash_value *= 16777619u;
    }
    return hash_value;
}

static uint32_t SparkTokenizerHashTokenPair(
    uint32_t left_token_id,
    uint32_t right_token_id)
{
    uint32_t hash_value;

    hash_value = 2166136261u;
    hash_value ^= left_token_id;
    hash_value *= 16777619u;
    hash_value ^= right_token_id;
    hash_value *= 16777619u;
    return hash_value;
}

static char *SparkTokenizerDuplicateBytes(
    const char *text,
    uint32_t text_bytes)
{
    char *copy;

    if (text == 0 && text_bytes != 0u)
    {
        return 0;
    }
    copy = (char *)malloc((size_t)text_bytes + 1u);
    if (copy == 0)
    {
        return 0;
    }
    if (text_bytes != 0u)
    {
        memcpy(copy, text, text_bytes);
    }
    copy[text_bytes] = '\0';
    return copy;
}

static uint32_t SparkTokenizerByteToUnicodeCodePoint(
    uint32_t byte_value)
{
    uint32_t candidate;
    uint32_t next_code_point;

    if ((byte_value >= 33u && byte_value <= 126u) ||
        (byte_value >= 161u && byte_value <= 172u) ||
        (byte_value >= 174u && byte_value <= 255u))
    {
        return byte_value;
    }

    next_code_point = 256u;
    for (candidate = 0u; candidate <= byte_value; ++candidate)
    {
        if (!((candidate >= 33u && candidate <= 126u) ||
              (candidate >= 161u && candidate <= 172u) ||
              (candidate >= 174u && candidate <= 255u)))
        {
            if (candidate == byte_value)
            {
                return next_code_point;
            }
            next_code_point += 1u;
        }
    }
    return byte_value;
}


static uint32_t SparkTokenizerUnicodeCodePointToByte(
    uint32_t code_point,
    uint8_t *byte_out)
{
    uint32_t candidate;
    uint32_t next_code_point;

    if (byte_out == 0)
    {
        return 0u;
    }
    if ((code_point >= 33u && code_point <= 126u) ||
        (code_point >= 161u && code_point <= 172u) ||
        (code_point >= 174u && code_point <= 255u))
    {
        *byte_out = (uint8_t)code_point;
        return 1u;
    }

    next_code_point = 256u;
    for (candidate = 0u; candidate <= 255u; ++candidate)
    {
        if (!((candidate >= 33u && candidate <= 126u) ||
              (candidate >= 161u && candidate <= 172u) ||
              (candidate >= 174u && candidate <= 255u)))
        {
            if (code_point == next_code_point)
            {
                *byte_out = (uint8_t)candidate;
                return 1u;
            }
            next_code_point += 1u;
        }
    }
    return 0u;
}

static uint32_t SparkTokenizerReadUtf8CodePoint(
    const char *text,
    uint32_t text_bytes,
    uint32_t *position,
    uint32_t *code_point_out)
{
    uint8_t first_byte;
    uint32_t remaining_bytes;

    if (text == 0 || position == 0 || code_point_out == 0 ||
        *position >= text_bytes)
    {
        return 0u;
    }
    first_byte = (uint8_t)text[*position];
    remaining_bytes = text_bytes - *position;
    if (first_byte < 0x80u)
    {
        *code_point_out = first_byte;
        *position += 1u;
        return 1u;
    }
    if ((first_byte & 0xe0u) == 0xc0u && remaining_bytes >= 2u &&
        (((uint8_t)text[*position + 1u]) & 0xc0u) == 0x80u)
    {
        *code_point_out = ((uint32_t)(first_byte & 0x1fu) << 6u) |
            ((uint32_t)((uint8_t)text[*position + 1u]) & 0x3fu);
        *position += 2u;
        return 1u;
    }
    if ((first_byte & 0xf0u) == 0xe0u && remaining_bytes >= 3u &&
        (((uint8_t)text[*position + 1u]) & 0xc0u) == 0x80u &&
        (((uint8_t)text[*position + 2u]) & 0xc0u) == 0x80u)
    {
        *code_point_out = ((uint32_t)(first_byte & 0x0fu) << 12u) |
            (((uint32_t)((uint8_t)text[*position + 1u]) & 0x3fu) << 6u) |
            ((uint32_t)((uint8_t)text[*position + 2u]) & 0x3fu);
        *position += 3u;
        return 1u;
    }
    if ((first_byte & 0xf8u) == 0xf0u && remaining_bytes >= 4u &&
        (((uint8_t)text[*position + 1u]) & 0xc0u) == 0x80u &&
        (((uint8_t)text[*position + 2u]) & 0xc0u) == 0x80u &&
        (((uint8_t)text[*position + 3u]) & 0xc0u) == 0x80u)
    {
        *code_point_out = ((uint32_t)(first_byte & 0x07u) << 18u) |
            (((uint32_t)((uint8_t)text[*position + 1u]) & 0x3fu) << 12u) |
            (((uint32_t)((uint8_t)text[*position + 2u]) & 0x3fu) << 6u) |
            ((uint32_t)((uint8_t)text[*position + 3u]) & 0x3fu);
        *position += 4u;
        return 1u;
    }
    return 0u;
}

static uint32_t SparkTokenizerAppendUtf8CodePoint(
    char *destination,
    uint32_t code_point)
{
    if (code_point <= 0x7fu)
    {
        destination[0u] = (char)code_point;
        return 1u;
    }
    if (code_point <= 0x7ffu)
    {
        destination[0u] = (char)(0xc0u | (code_point >> 6u));
        destination[1u] = (char)(0x80u | (code_point & 0x3fu));
        return 2u;
    }
    destination[0u] = (char)(0xe0u | (code_point >> 12u));
    destination[1u] = (char)(0x80u | ((code_point >> 6u) & 0x3fu));
    destination[2u] = (char)(0x80u | (code_point & 0x3fu));
    return 3u;
}

static void SparkTokenizerInitializeBuckets(
    uint32_t *buckets,
    uint32_t bucket_count)
{
    uint32_t bucket_index;

    if (buckets == 0)
    {
        return;
    }
    for (bucket_index = 0u; bucket_index < bucket_count; ++bucket_index)
    {
        buckets[bucket_index] = SPARK_TOKENIZER_EMPTY_BUCKET;
    }
}

static SparkStatus SparkTokenizerAllocateStringTable(
    SparkTokenizerStringEntry **entries_out,
    uint32_t **buckets_out,
    uint32_t *bucket_count_out,
    uint32_t entry_count)
{
    SparkTokenizerStringEntry *entries;
    uint32_t *buckets;
    uint32_t bucket_count;

    if (entries_out == 0 || buckets_out == 0 || bucket_count_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *entries_out = 0;
    *buckets_out = 0;
    *bucket_count_out = 0u;
    if (entry_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    bucket_count = SparkTokenizerNextPowerOfTwo(
        entry_count * SPARK_TOKENIZER_HASH_LOAD_FACTOR + 1u);
    entries = (SparkTokenizerStringEntry *)calloc(entry_count, sizeof(*entries));
    buckets = (uint32_t *)malloc((uint64_t)bucket_count * sizeof(*buckets));
    if (entries == 0 || buckets == 0)
    {
        free(entries);
        free(buckets);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkTokenizerInitializeBuckets(buckets, bucket_count);

    *entries_out = entries;
    *buckets_out = buckets;
    *bucket_count_out = bucket_count;
    return SPARK_STATUS_OK;
}

static SparkTokenizerStringEntry *SparkTokenizerFindStringEntryInTable(
    SparkTokenizerStringEntry *entries,
    uint32_t *buckets,
    uint32_t bucket_count,
    const char *text,
    uint32_t text_bytes)
{
    uint32_t entry_index;

    if (entries == 0 || buckets == 0 || bucket_count == 0u || text == 0)
    {
        return 0;
    }
    entry_index = buckets[SparkTokenizerHashBytes(text, text_bytes) & (bucket_count - 1u)];
    while (entry_index != SPARK_TOKENIZER_EMPTY_BUCKET)
    {
        SparkTokenizerStringEntry *entry;

        entry = &entries[entry_index];
        if (entry->text_bytes == text_bytes && memcmp(entry->text, text, text_bytes) == 0)
        {
            return entry;
        }
        entry_index = entry->next_index;
    }
    return 0;
}

static SparkStatus SparkTokenizerInsertStringEntry(
    SparkTokenizerStringEntry *entries,
    uint32_t *buckets,
    uint32_t bucket_count,
    uint32_t entry_index,
    const char *text,
    uint32_t text_bytes,
    uint32_t value)
{
    uint32_t bucket_index;
    SparkTokenizerStringEntry *entry;

    if (entries == 0 || buckets == 0 || bucket_count == 0u || text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkTokenizerFindStringEntryInTable(
            entries,
            buckets,
            bucket_count,
            text,
            text_bytes) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    entry = &entries[entry_index];
    entry->text = SparkTokenizerDuplicateBytes(text, text_bytes);
    if (entry->text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    entry->text_bytes = text_bytes;
    entry->value = value;
    bucket_index = SparkTokenizerHashBytes(text, text_bytes) & (bucket_count - 1u);
    entry->next_index = buckets[bucket_index];
    buckets[bucket_index] = entry_index;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerAllocateMergePairTable(
    SparkTokenizer *tokenizer,
    uint32_t entry_capacity)
{
    if (tokenizer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->fast_merge_pair_count = 0u;
    tokenizer->fast_merge_pairs = 0;
    tokenizer->fast_merge_buckets = 0;
    tokenizer->fast_merge_bucket_count = 0u;
    if (entry_capacity == 0u)
    {
        return SPARK_STATUS_OK;
    }
    tokenizer->fast_merge_bucket_count = SparkTokenizerNextPowerOfTwo(
        entry_capacity * SPARK_TOKENIZER_HASH_LOAD_FACTOR + 1u);
    tokenizer->fast_merge_pairs = (SparkTokenizerFastMergePair *)calloc(
        entry_capacity,
        sizeof(*tokenizer->fast_merge_pairs));
    tokenizer->fast_merge_buckets = (uint32_t *)malloc(
        (uint64_t)tokenizer->fast_merge_bucket_count * sizeof(*tokenizer->fast_merge_buckets));
    if (tokenizer->fast_merge_pairs == 0 || tokenizer->fast_merge_buckets == 0)
    {
        free(tokenizer->fast_merge_pairs);
        free(tokenizer->fast_merge_buckets);
        tokenizer->fast_merge_pairs = 0;
        tokenizer->fast_merge_buckets = 0;
        tokenizer->fast_merge_bucket_count = 0u;
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    SparkTokenizerInitializeBuckets(
        tokenizer->fast_merge_buckets,
        tokenizer->fast_merge_bucket_count);
    return SPARK_STATUS_OK;
}

static SparkTokenizerFastMergePair *SparkTokenizerFindMergePairEntry(
    const SparkTokenizer *tokenizer,
    uint32_t left_token_id,
    uint32_t right_token_id)
{
    uint32_t entry_index;

    if (tokenizer == 0 || tokenizer->fast_merge_pairs == 0 ||
        tokenizer->fast_merge_buckets == 0 || tokenizer->fast_merge_bucket_count == 0u)
    {
        return 0;
    }
    entry_index = tokenizer->fast_merge_buckets[
        SparkTokenizerHashTokenPair(left_token_id, right_token_id) &
        (tokenizer->fast_merge_bucket_count - 1u)];
    while (entry_index != SPARK_TOKENIZER_EMPTY_BUCKET)
    {
        SparkTokenizerFastMergePair *entry;

        entry = &tokenizer->fast_merge_pairs[entry_index];
        if (entry->left_token_id == left_token_id && entry->right_token_id == right_token_id)
        {
            return entry;
        }
        entry_index = entry->next_index;
    }
    return 0;
}

static SparkStatus SparkTokenizerInsertMergePairEntry(
    SparkTokenizer *tokenizer,
    uint32_t left_token_id,
    uint32_t right_token_id,
    uint32_t merged_token_id,
    uint32_t rank)
{
    uint32_t bucket_index;
    SparkTokenizerFastMergePair *entry;

    if (tokenizer == 0 || tokenizer->fast_merge_pairs == 0 ||
        tokenizer->fast_merge_buckets == 0 || tokenizer->fast_merge_bucket_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkTokenizerFindMergePairEntry(tokenizer, left_token_id, right_token_id) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    entry = &tokenizer->fast_merge_pairs[tokenizer->fast_merge_pair_count];
    entry->left_token_id = left_token_id;
    entry->right_token_id = right_token_id;
    entry->merged_token_id = merged_token_id;
    entry->rank = rank;
    bucket_index = SparkTokenizerHashTokenPair(left_token_id, right_token_id) &
        (tokenizer->fast_merge_bucket_count - 1u);
    entry->next_index = tokenizer->fast_merge_buckets[bucket_index];
    tokenizer->fast_merge_buckets[bucket_index] = tokenizer->fast_merge_pair_count;
    tokenizer->fast_merge_pair_count += 1u;
    return SPARK_STATUS_OK;
}

static int32_t SparkTokenizerJsonFindNextDirectChild(
    const SparkJsonDocument *document,
    int32_t parent_token_index,
    int32_t previous_token_index)
{
    int32_t token_index;

    if (document == 0 || parent_token_index < 0)
    {
        return -1;
    }
    token_index = previous_token_index + 1;
    while (token_index >= 0 && (uint32_t)token_index < document->token_count)
    {
        if (document->tokens[token_index].parent == parent_token_index)
        {
            return token_index;
        }
        token_index += 1;
    }
    return -1;
}

static uint32_t SparkTokenizerJsonGetObjectMemberCount(
    const SparkJsonDocument *document,
    int32_t object_token_index)
{
    if (!SparkJsonTokenIsType(document, object_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        return 0u;
    }
    return document->tokens[object_token_index].child_count / 2u;
}

static SparkStatus SparkTokenizerParseVocabulary(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t vocabulary_object_token_index)
{
    int32_t key_token_index;
    uint32_t entry_index;
    SparkStatus status;

    if (tokenizer == 0 ||
        !SparkJsonTokenIsType(document, vocabulary_object_token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    tokenizer->vocabulary_count = SparkTokenizerJsonGetObjectMemberCount(
        document,
        vocabulary_object_token_index);
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->vocabulary_entries,
        &tokenizer->vocabulary_buckets,
        &tokenizer->vocabulary_bucket_count,
        tokenizer->vocabulary_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    entry_index = 0u;
    key_token_index = SparkTokenizerJsonFindNextDirectChild(
        document,
        vocabulary_object_token_index,
        vocabulary_object_token_index);
    while (key_token_index >= 0)
    {
        int32_t value_token_index;
        char *token_text;
        uint32_t token_id;

        value_token_index = SparkTokenizerJsonFindNextDirectChild(
            document,
            vocabulary_object_token_index,
            key_token_index);
        if (value_token_index < 0 ||
            SparkJsonCopyString(document, key_token_index, &token_text) != SPARK_STATUS_OK ||
            SparkJsonGetUInt32(document, value_token_index, &token_id) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        status = SparkTokenizerInsertStringEntry(
            tokenizer->vocabulary_entries,
            tokenizer->vocabulary_buckets,
            tokenizer->vocabulary_bucket_count,
            entry_index,
            token_text,
            (uint32_t)strlen(token_text),
            token_id);
        if (status != SPARK_STATUS_OK)
        {
            free(token_text);
            return status;
        }
        if (token_id > tokenizer->maximum_token_id)
        {
            tokenizer->maximum_token_id = token_id;
        }
        free(token_text);
        entry_index += 1u;
        key_token_index = SparkTokenizerJsonFindNextDirectChild(
            document,
            vocabulary_object_token_index,
            value_token_index);
    }
    return entry_index == tokenizer->vocabulary_count ? SPARK_STATUS_OK : SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkTokenizerBuildReverseVocabulary(
    SparkTokenizer *tokenizer)
{
    uint32_t entry_index;

    if (tokenizer == 0 || tokenizer->vocabulary_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->token_text_by_id = (char **)calloc(
        (uint64_t)tokenizer->maximum_token_id + 1u,
        sizeof(*tokenizer->token_text_by_id));
    tokenizer->token_text_bytes_by_id = (uint32_t *)calloc(
        (uint64_t)tokenizer->maximum_token_id + 1u,
        sizeof(*tokenizer->token_text_bytes_by_id));
    if (tokenizer->token_text_by_id == 0 || tokenizer->token_text_bytes_by_id == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    for (entry_index = 0u; entry_index < tokenizer->vocabulary_count; ++entry_index)
    {
        SparkTokenizerStringEntry *entry;

        entry = &tokenizer->vocabulary_entries[entry_index];
        if (entry->text != 0 && entry->value <= tokenizer->maximum_token_id)
        {
            tokenizer->token_text_by_id[entry->value] = entry->text;
            tokenizer->token_text_bytes_by_id[entry->value] = entry->text_bytes;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerBuildByteTokenTable(
    SparkTokenizer *tokenizer)
{
    uint32_t byte_value;

    if (tokenizer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (byte_value = 0u; byte_value < SPARK_TOKENIZER_BYTE_COUNT; ++byte_value)
    {
        char encoded_text[SPARK_TOKENIZER_INITIAL_BYTE_SYMBOL_BYTES];
        uint32_t encoded_text_bytes;
        uint32_t token_id;

        tokenizer->byte_token_ids[byte_value] = SPARK_TOKENIZER_NO_TOKEN_ID;
        encoded_text_bytes = SparkTokenizerAppendUtf8CodePoint(
            encoded_text,
            SparkTokenizerByteToUnicodeCodePoint(byte_value));
        if (SparkTokenizerFindTokenId(tokenizer, encoded_text, encoded_text_bytes, &token_id) ==
            SPARK_STATUS_OK)
        {
            tokenizer->byte_token_ids[byte_value] = token_id;
        }
        else if (tokenizer->has_unk_token != 0u)
        {
            tokenizer->byte_token_ids[byte_value] = tokenizer->unk_token_id;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerMakeMergeKey(
    const char *left_text,
    uint32_t left_text_bytes,
    const char *right_text,
    uint32_t right_text_bytes,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    char *merge_key;
    uint64_t merge_key_bytes;

    if (merge_key_out == 0 || merge_key_bytes_out == 0 ||
        left_text == 0 || right_text == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *merge_key_out = 0;
    *merge_key_bytes_out = 0u;
    merge_key_bytes = (uint64_t)left_text_bytes + 1u + right_text_bytes;
    if (merge_key_bytes > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    merge_key = (char *)malloc((size_t)merge_key_bytes + 1u);
    if (merge_key == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    memcpy(merge_key, left_text, left_text_bytes);
    merge_key[left_text_bytes] = ' ';
    memcpy(merge_key + left_text_bytes + 1u, right_text, right_text_bytes);
    merge_key[merge_key_bytes] = '\0';
    *merge_key_out = merge_key;
    *merge_key_bytes_out = (uint32_t)merge_key_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerParseMergeString(
    const char *merge_text,
    uint32_t merge_text_bytes,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    uint32_t byte_index;

    if (merge_text == 0 || merge_key_out == 0 || merge_key_bytes_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (byte_index = 0u; byte_index < merge_text_bytes; ++byte_index)
    {
        if (merge_text[byte_index] == ' ')
        {
            return SparkTokenizerMakeMergeKey(
                merge_text,
                byte_index,
                merge_text + byte_index + 1u,
                merge_text_bytes - byte_index - 1u,
                merge_key_out,
                merge_key_bytes_out);
        }
    }
    return SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkTokenizerParseMergeArray(
    const SparkJsonDocument *document,
    int32_t merge_array_token_index,
    char **merge_key_out,
    uint32_t *merge_key_bytes_out)
{
    int32_t left_token_index;
    int32_t right_token_index;
    char *left_text;
    char *right_text;
    SparkStatus status;

    if (SparkJsonGetArrayElementCount(document, merge_array_token_index) != 2u)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    left_token_index = SparkJsonGetArrayElement(document, merge_array_token_index, 0u);
    right_token_index = SparkJsonGetArrayElement(document, merge_array_token_index, 1u);
    if (SparkJsonCopyString(document, left_token_index, &left_text) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    if (SparkJsonCopyString(document, right_token_index, &right_text) != SPARK_STATUS_OK)
    {
        free(left_text);
        return SPARK_STATUS_PARSE_ERROR;
    }
    status = SparkTokenizerMakeMergeKey(
        left_text,
        (uint32_t)strlen(left_text),
        right_text,
        (uint32_t)strlen(right_text),
        merge_key_out,
        merge_key_bytes_out);
    free(left_text);
    free(right_text);
    return status;
}

static SparkStatus SparkTokenizerParseMerges(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t merges_array_token_index)
{
    uint32_t merge_index;
    SparkStatus status;

    if (tokenizer == 0 ||
        !SparkJsonTokenIsType(document, merges_array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->merge_count = SparkJsonGetArrayElementCount(document, merges_array_token_index);
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->merge_entries,
        &tokenizer->merge_buckets,
        &tokenizer->merge_bucket_count,
        tokenizer->merge_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    merge_index = 0u;
    for (int32_t merge_token_index = SparkTokenizerJsonFindNextDirectChild(
             document,
             merges_array_token_index,
             merges_array_token_index);
         merge_token_index >= 0 && merge_index < tokenizer->merge_count;
         merge_token_index = SparkTokenizerJsonFindNextDirectChild(
             document,
             merges_array_token_index,
             merge_token_index))
    {
        char *merge_text;
        char *merge_key;
        uint32_t merge_key_bytes;

        merge_key = 0;
        if (SparkJsonTokenIsType(document, merge_token_index, SPARK_JSON_TOKEN_STRING))
        {
            if (SparkJsonCopyString(document, merge_token_index, &merge_text) != SPARK_STATUS_OK)
            {
                return SPARK_STATUS_PARSE_ERROR;
            }
            status = SparkTokenizerParseMergeString(
                merge_text,
                (uint32_t)strlen(merge_text),
                &merge_key,
                &merge_key_bytes);
            free(merge_text);
        }
        else
        {
            status = SparkTokenizerParseMergeArray(
                document,
                merge_token_index,
                &merge_key,
                &merge_key_bytes);
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkTokenizerInsertStringEntry(
            tokenizer->merge_entries,
            tokenizer->merge_buckets,
            tokenizer->merge_bucket_count,
            merge_index,
            merge_key,
            merge_key_bytes,
            merge_index);
        free(merge_key);
        merge_index += 1u;
        if (status == SPARK_STATUS_DUPLICATE)
        {
            continue;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return merge_index == tokenizer->merge_count ? SPARK_STATUS_OK : SPARK_STATUS_PARSE_ERROR;
}

static SparkStatus SparkTokenizerBuildMergePairTableFromMergeKeys(
    SparkTokenizer *tokenizer)
{
    uint32_t merge_index;
    SparkStatus status;

    if (tokenizer == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkTokenizerAllocateMergePairTable(tokenizer, tokenizer->merge_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (merge_index = 0u; merge_index < tokenizer->merge_count; ++merge_index)
    {
        SparkTokenizerStringEntry *merge_entry;
        uint32_t separator_index;
        uint32_t left_token_id;
        uint32_t right_token_id;
        uint32_t merged_token_id;
        char *merged_text;
        uint32_t merged_text_bytes;

        merge_entry = &tokenizer->merge_entries[merge_index];
        if (merge_entry->text == 0)
        {
            continue;
        }
        separator_index = 0u;
        while (separator_index < merge_entry->text_bytes &&
               merge_entry->text[separator_index] != ' ')
        {
            separator_index += 1u;
        }
        if (separator_index == 0u || separator_index >= merge_entry->text_bytes)
        {
            continue;
        }
        if (SparkTokenizerFindTokenId(
                tokenizer,
                merge_entry->text,
                separator_index,
                &left_token_id) != SPARK_STATUS_OK)
        {
            continue;
        }
        if (SparkTokenizerFindTokenId(
                tokenizer,
                merge_entry->text + separator_index + 1u,
                merge_entry->text_bytes - separator_index - 1u,
                &right_token_id) != SPARK_STATUS_OK)
        {
            continue;
        }
        merged_text_bytes = merge_entry->text_bytes - 1u;
        merged_text = (char *)malloc((size_t)merged_text_bytes + 1u);
        if (merged_text == 0)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        memcpy(merged_text, merge_entry->text, separator_index);
        memcpy(
            merged_text + separator_index,
            merge_entry->text + separator_index + 1u,
            merge_entry->text_bytes - separator_index - 1u);
        merged_text[merged_text_bytes] = '\0';
        status = SparkTokenizerFindTokenId(tokenizer, merged_text, merged_text_bytes, &merged_token_id);
        free(merged_text);
        if (status != SPARK_STATUS_OK)
        {
            continue;
        }
        status = SparkTokenizerInsertMergePairEntry(
            tokenizer,
            left_token_id,
            right_token_id,
            merged_token_id,
            merge_entry->value);
        if (status == SPARK_STATUS_DUPLICATE)
        {
            continue;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerParseAddedTokens(
    SparkTokenizer *tokenizer,
    const SparkJsonDocument *document,
    int32_t added_tokens_array_token_index)
{
    uint32_t array_count;
    uint32_t array_index;
    uint32_t special_count;

    if (added_tokens_array_token_index < 0 ||
        !SparkJsonTokenIsType(document, added_tokens_array_token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        return SPARK_STATUS_OK;
    }

    array_count = SparkJsonGetArrayElementCount(document, added_tokens_array_token_index);
    tokenizer->special_tokens = (SparkTokenizerSpecialToken *)calloc(
        array_count,
        sizeof(*tokenizer->special_tokens));
    if (array_count != 0u && tokenizer->special_tokens == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    special_count = 0u;
    array_index = 0u;
    for (int32_t token_object_index = SparkTokenizerJsonFindNextDirectChild(
             document,
             added_tokens_array_token_index,
             added_tokens_array_token_index);
         token_object_index >= 0 && array_index < array_count;
         token_object_index = SparkTokenizerJsonFindNextDirectChild(
             document,
             added_tokens_array_token_index,
             token_object_index), ++array_index)
    {
        int32_t id_token_index;
        int32_t content_token_index;
        int32_t special_token_index;
        bool is_special;
        uint32_t token_id;
        uint32_t content_bytes;
        char *content;

        if (!SparkJsonTokenIsType(document, token_object_index, SPARK_JSON_TOKEN_OBJECT))
        {
            continue;
        }
        id_token_index = SparkJsonFindObjectMember(document, token_object_index, "id");
        content_token_index = SparkJsonFindObjectMember(document, token_object_index, "content");
        special_token_index = SparkJsonFindObjectMember(document, token_object_index, "special");
        is_special = false;
        if (special_token_index >= 0 &&
            SparkJsonGetBoolean(document, special_token_index, &is_special) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (id_token_index < 0 || content_token_index < 0)
        {
            continue;
        }
        if (SparkJsonGetUInt32(document, id_token_index, &token_id) != SPARK_STATUS_OK ||
            SparkJsonCopyString(document, content_token_index, &content) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        content_bytes = (uint32_t)strlen(content);
        if (content_bytes == 0u)
        {
            free(content);
            return SPARK_STATUS_SCHEMA_ERROR;
        }
        tokenizer->special_tokens[special_count].text = content;
        tokenizer->special_tokens[special_count].text_bytes = content_bytes;
        tokenizer->special_tokens[special_count].token_id = token_id;
        tokenizer->special_tokens[special_count].is_special = is_special ? 1u : 0u;
        special_count += 1u;
        tokenizer->special_token_count = special_count;
        if (token_id > tokenizer->maximum_token_id)
        {
            tokenizer->maximum_token_id = token_id;
        }
    }
    return SPARK_STATUS_OK;
}

static int SparkTokenizerCompareSpecialTokenLengthDescending(
    const void *left,
    const void *right)
{
    const SparkTokenizerSpecialToken *left_token;
    const SparkTokenizerSpecialToken *right_token;

    left_token = (const SparkTokenizerSpecialToken *)left;
    right_token = (const SparkTokenizerSpecialToken *)right;
    if (left_token->text_bytes != right_token->text_bytes)
    {
        return left_token->text_bytes > right_token->text_bytes ? -1 : 1;
    }
    if (left_token->token_id == right_token->token_id)
    {
        return 0;
    }
    return left_token->token_id < right_token->token_id ? -1 : 1;
}

static void SparkTokenizerSortSpecialTokens(
    SparkTokenizer *tokenizer)
{
    if (tokenizer == 0 || tokenizer->special_token_count <= 1u)
    {
        return;
    }
    qsort(
        tokenizer->special_tokens,
        tokenizer->special_token_count,
        sizeof(*tokenizer->special_tokens),
        SparkTokenizerCompareSpecialTokenLengthDescending);
}

static uint32_t SparkTokenizerJsonSearchBooleanMemberRecursive(
    const SparkJsonDocument *document,
    int32_t token_index,
    const char *member_name,
    bool *value_out)
{
    if (SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_OBJECT))
    {
        int32_t member_token_index;
        int32_t child_token_index;

        member_token_index = SparkJsonFindObjectMember(document, token_index, member_name);
        if (member_token_index >= 0 &&
            SparkJsonGetBoolean(document, member_token_index, value_out) == SPARK_STATUS_OK)
        {
            return 1u;
        }
        child_token_index = SparkTokenizerJsonFindNextDirectChild(document, token_index, token_index);
        while (child_token_index >= 0)
        {
            int32_t value_token_index;

            value_token_index = SparkTokenizerJsonFindNextDirectChild(
                document,
                token_index,
                child_token_index);
            if (value_token_index < 0)
            {
                break;
            }
            if (SparkTokenizerJsonSearchBooleanMemberRecursive(
                    document,
                    value_token_index,
                    member_name,
                    value_out))
            {
                return 1u;
            }
            child_token_index = SparkTokenizerJsonFindNextDirectChild(
                document,
                token_index,
                value_token_index);
        }
    }
    else if (SparkJsonTokenIsType(document, token_index, SPARK_JSON_TOKEN_ARRAY))
    {
        uint32_t element_count;
        uint32_t element_index;

        element_count = SparkJsonGetArrayElementCount(document, token_index);
        for (element_index = 0u; element_index < element_count; ++element_index)
        {
            if (SparkTokenizerJsonSearchBooleanMemberRecursive(
                    document,
                    SparkJsonGetArrayElement(document, token_index, element_index),
                    member_name,
                    value_out))
            {
                return 1u;
            }
        }
    }
    return 0u;
}

void SparkTokenizerReset(
    SparkTokenizer *tokenizer)
{
    uint32_t byte_index;

    if (tokenizer == 0)
    {
        return;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
    tokenizer->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    tokenizer->descriptor_bytes = SPARK_TOKENIZER_DESCRIPTOR_BYTES;
    tokenizer->model_kind = SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL;
    tokenizer->byte_level_use_regex = 1u;
    for (byte_index = 0u; byte_index < SPARK_TOKENIZER_BYTE_COUNT; ++byte_index)
    {
        tokenizer->byte_token_ids[byte_index] = SPARK_TOKENIZER_NO_TOKEN_ID;
    }
}

void SparkTokenizerDestroy(
    SparkTokenizer *tokenizer)
{
    uint32_t entry_index;

    if (tokenizer == 0)
    {
        return;
    }
    for (entry_index = 0u; entry_index < tokenizer->vocabulary_count; ++entry_index)
    {
        free(tokenizer->vocabulary_entries[entry_index].text);
    }
    for (entry_index = 0u; entry_index < tokenizer->merge_count; ++entry_index)
    {
        free(tokenizer->merge_entries[entry_index].text);
    }
    for (entry_index = 0u; entry_index < tokenizer->special_token_count; ++entry_index)
    {
        free(tokenizer->special_tokens[entry_index].text);
    }
    free(tokenizer->vocabulary_entries);
    free(tokenizer->vocabulary_buckets);
    free(tokenizer->merge_entries);
    free(tokenizer->merge_buckets);
    free(tokenizer->fast_merge_pairs);
    free(tokenizer->fast_merge_buckets);
    free(tokenizer->special_tokens);
    free(tokenizer->token_text_by_id);
    free(tokenizer->token_text_bytes_by_id);
    SparkTokenizerReset(tokenizer);
}

void SparkTokenizerEncodingReset(
    SparkTokenizerEncoding *encoding)
{
    uint32_t token_capacity;
    uint32_t *token_ids;

    if (encoding == 0)
    {
        return;
    }
    token_capacity = encoding->token_capacity;
    token_ids = encoding->token_ids;
    memset(encoding, 0, sizeof(*encoding));
    encoding->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    encoding->descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
    encoding->token_capacity = token_capacity;
    encoding->token_ids = token_ids;
}

void SparkTokenizerWorkspaceReset(
    SparkTokenizerWorkspace *workspace)
{
    if (workspace == 0)
    {
        return;
    }
    memset(workspace, 0, sizeof(*workspace));
    workspace->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    workspace->descriptor_bytes = SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES;
}

void SparkTokenizerWorkspaceDestroy(
    SparkTokenizerWorkspace *workspace)
{
    if (workspace == 0)
    {
        return;
    }
    free(workspace->symbol_token_ids);
    free(workspace->previous_symbol_indices);
    free(workspace->next_symbol_indices);
    free(workspace->symbol_generations);
    free(workspace->merge_heap);
    free(workspace->piece_cache_entries);
    free(workspace->piece_cache_token_pool);
    SparkTokenizerWorkspaceReset(workspace);
}

static SparkStatus SparkTokenizerWorkspaceEnsurePieceCache(SparkTokenizerWorkspace *workspace)
{
    if (workspace->piece_cache_entries != 0 && workspace->piece_cache_token_pool != 0)
    {
        return SPARK_STATUS_OK;
    }
    workspace->piece_cache_slot_count = SPARK_TOKENIZER_PIECE_CACHE_SLOT_COUNT;
    workspace->piece_cache_token_capacity = SPARK_TOKENIZER_PIECE_CACHE_TOKEN_CAPACITY;
    workspace->piece_cache_token_used = 0u;
    workspace->piece_cache_entries = (SparkTokenizerPieceCacheEntry *)calloc(
        (uint64_t)workspace->piece_cache_slot_count,sizeof(*workspace->piece_cache_entries));
    workspace->piece_cache_token_pool = (uint32_t *)malloc(
        (uint64_t)workspace->piece_cache_token_capacity * sizeof(*workspace->piece_cache_token_pool));
    if (workspace->piece_cache_entries == 0 || workspace->piece_cache_token_pool == 0)
    {
        free(workspace->piece_cache_entries);
        free(workspace->piece_cache_token_pool);
        workspace->piece_cache_entries = 0;
        workspace->piece_cache_token_pool = 0;
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerWorkspaceEnsureSymbolBuffers(SparkTokenizerWorkspace *workspace,uint32_t maximum_symbol_count)
{
    if (workspace->symbol_token_ids != 0 &&
        workspace->maximum_symbol_count >= maximum_symbol_count)
    {
        return SPARK_STATUS_OK;
    }
    free(workspace->symbol_token_ids);
    free(workspace->previous_symbol_indices);
    free(workspace->next_symbol_indices);
    free(workspace->symbol_generations);
    free(workspace->merge_heap);
    workspace->symbol_token_ids = 0;
    workspace->previous_symbol_indices = 0;
    workspace->next_symbol_indices = 0;
    workspace->symbol_generations = 0;
    workspace->merge_heap = 0;
    workspace->maximum_symbol_count = maximum_symbol_count;
    if (maximum_symbol_count >
        (UINT32_MAX - SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_SLACK) /
        SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_FACTOR)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    workspace->heap_capacity =
        maximum_symbol_count * SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_FACTOR +
        SPARK_TOKENIZER_MERGE_HEAP_CAPACITY_SLACK;
    workspace->symbol_token_ids = (uint32_t *)malloc(
        (uint64_t)maximum_symbol_count * sizeof(*workspace->symbol_token_ids));
    workspace->previous_symbol_indices = (uint32_t *)malloc(
        (uint64_t)maximum_symbol_count * sizeof(*workspace->previous_symbol_indices));
    workspace->next_symbol_indices = (uint32_t *)malloc(
        (uint64_t)maximum_symbol_count * sizeof(*workspace->next_symbol_indices));
    workspace->symbol_generations = (uint32_t *)malloc(
        (uint64_t)maximum_symbol_count * sizeof(*workspace->symbol_generations));
    workspace->merge_heap = (SparkTokenizerMergeCandidate *)malloc(
        (uint64_t)workspace->heap_capacity * sizeof(*workspace->merge_heap));
    if (workspace->symbol_token_ids == 0 || workspace->previous_symbol_indices == 0 ||
        workspace->next_symbol_indices == 0 || workspace->symbol_generations == 0 ||
        workspace->merge_heap == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerWorkspaceInitialize(
    SparkTokenizerWorkspace *workspace,
    uint32_t maximum_symbol_count)
{
    SparkStatus status;
    if (workspace == 0 || maximum_symbol_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkTokenizerWorkspaceDestroy(workspace);
    workspace->abi_version = SPARK_TOKENIZER_ABI_VERSION;
    workspace->descriptor_bytes = SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES;
    status = SparkTokenizerWorkspaceEnsureSymbolBuffers(workspace,maximum_symbol_count);
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerWorkspaceDestroy(workspace);
        return status;
    }
    status = SparkTokenizerWorkspaceEnsurePieceCache(workspace);
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerWorkspaceDestroy(workspace);
        return status;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerFindTokenId(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t *token_id_out)
{
    SparkTokenizerStringEntry *entry;
    uint32_t special_index;

    if (tokenizer == 0 || text == 0 || token_id_out == 0 ||
        tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *token_id_out = 0u;
    entry = SparkTokenizerFindStringEntryInTable(
        tokenizer->vocabulary_entries,
        tokenizer->vocabulary_buckets,
        tokenizer->vocabulary_bucket_count,
        text,
        text_bytes);
    if (entry == 0)
    {
        for (special_index = 0u;
             special_index < tokenizer->special_token_count;
             ++special_index)
        {
            const SparkTokenizerSpecialToken *special_token;

            special_token = &tokenizer->special_tokens[special_index];
            if (special_token->text_bytes == text_bytes &&
                memcmp(special_token->text, text, text_bytes) == 0)
            {
                *token_id_out = special_token->token_id;
                return SPARK_STATUS_OK;
            }
        }
        return SPARK_STATUS_NOT_FOUND;
    }
    *token_id_out = entry->value;
    return SPARK_STATUS_OK;
}

static const char SPARK_TOKENIZER_EXTENDED_SPLIT_PATTERN[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

static const char SPARK_TOKENIZER_EXTENDED_SPLIT_PATTERN_DIGIT_RUNS[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

#define SPARK_TOKENIZER_SPLIT_VARIANT_NONE 0u
#define SPARK_TOKENIZER_SPLIT_VARIANT_EXTENDED 1u
#define SPARK_TOKENIZER_SPLIT_VARIANT_EXTENDED_DIGIT_RUNS 2u

static uint32_t SparkTokenizerJsonSplitElementVariant(
    const SparkJsonDocument *document,
    int32_t element_token_index)
{
    int32_t type_token_index;
    int32_t pattern_token_index;
    if (!SparkJsonTokenIsType(document,element_token_index,SPARK_JSON_TOKEN_OBJECT))
    {
        return SPARK_TOKENIZER_SPLIT_VARIANT_NONE;
    }
    type_token_index = SparkJsonFindObjectMember(document,element_token_index,"type");
    if (type_token_index < 0 ||
        !SparkJsonStringEquals(document,type_token_index,"Split"))
    {
        return SPARK_TOKENIZER_SPLIT_VARIANT_NONE;
    }
    pattern_token_index = SparkJsonFindObjectMember(document,element_token_index,"pattern");
    if (pattern_token_index < 0)
    {
        return SPARK_TOKENIZER_SPLIT_VARIANT_NONE;
    }
    if (SparkJsonTokenIsType(document,pattern_token_index,SPARK_JSON_TOKEN_OBJECT))
    {
        pattern_token_index = SparkJsonFindObjectMember(document,pattern_token_index,"Regex");
    }
    if (pattern_token_index >= 0 &&
        SparkJsonStringEquals(document,pattern_token_index,SPARK_TOKENIZER_EXTENDED_SPLIT_PATTERN))
    {
        return SPARK_TOKENIZER_SPLIT_VARIANT_EXTENDED;
    }
    if (pattern_token_index >= 0 &&
        SparkJsonStringEquals(document,pattern_token_index,SPARK_TOKENIZER_EXTENDED_SPLIT_PATTERN_DIGIT_RUNS))
    {
        return SPARK_TOKENIZER_SPLIT_VARIANT_EXTENDED_DIGIT_RUNS;
    }
    return SPARK_TOKENIZER_SPLIT_VARIANT_NONE;
}

static uint32_t SparkTokenizerJsonHasExtendedSplitPattern(
    const SparkJsonDocument *document,
    int32_t pre_tokenizer_token_index,
    uint32_t *variant_out)
{
    int32_t elements_token_index;
    uint32_t element_count;
    uint32_t element_index;
    uint32_t variant;
    if (pre_tokenizer_token_index < 0)
    {
        return 0u;
    }
    variant = SparkTokenizerJsonSplitElementVariant(document,pre_tokenizer_token_index);
    if (variant != SPARK_TOKENIZER_SPLIT_VARIANT_NONE)
    {
        if (variant_out != 0)
        {
            *variant_out = variant;
        }
        return 1u;
    }
    elements_token_index = SparkJsonFindObjectMember(document,pre_tokenizer_token_index,"pretokenizers");
    if (elements_token_index < 0 ||
        !SparkJsonTokenIsType(document,elements_token_index,SPARK_JSON_TOKEN_ARRAY))
    {
        return 0u;
    }
    element_count = SparkJsonGetArrayElementCount(document,elements_token_index);
    for (element_index = 0u; element_index < element_count; element_index++)
    {
        variant = SparkTokenizerJsonSplitElementVariant(
            document,
            SparkJsonGetArrayElement(document,elements_token_index,element_index));
        if (variant != SPARK_TOKENIZER_SPLIT_VARIANT_NONE)
        {
            if (variant_out != 0)
            {
                *variant_out = variant;
            }
            return 1u;
        }
    }
    return 0u;
}

SparkStatus SparkTokenizerLoadHuggingFaceJson(
    SparkTokenizer *tokenizer,
    const SparkTokenizerHuggingFaceJsonConfiguration *configuration)
{
    SparkJsonDocument document;
    int32_t root_token_index;
    int32_t model_token_index;
    int32_t vocabulary_token_index;
    int32_t merges_token_index;
    int32_t added_tokens_token_index;
    int32_t unknown_token_index;
    int32_t byte_fallback_token_index;
    int32_t pre_tokenizer_token_index;
    bool add_prefix_space;
    bool byte_fallback;
    bool use_regex;
    SparkStatus status;

    if (tokenizer == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->tokenizer_json_path == 0 ||
        configuration->reserved0 != 0u || configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkTokenizerReset(tokenizer);
    SparkJsonDocumentReset(&document);
    status = SparkJsonLoadFile(configuration->tokenizer_json_path, &document);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    root_token_index = SparkJsonGetRootToken(&document);
    model_token_index = SparkJsonFindObjectMember(&document, root_token_index, "model");
    if (model_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return SPARK_STATUS_PARSE_ERROR;
    }
    vocabulary_token_index = SparkJsonFindObjectMember(&document, model_token_index, "vocab");
    merges_token_index = SparkJsonFindObjectMember(&document, model_token_index, "merges");
    if (vocabulary_token_index < 0 || merges_token_index < 0)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return SPARK_STATUS_PARSE_ERROR;
    }

    status = SparkTokenizerParseVocabulary(tokenizer, &document, vocabulary_token_index);

    tokenizer->has_unk_token = 0u;
    tokenizer->unk_token_id = 0u;
    unknown_token_index = SparkJsonFindObjectMember(&document, model_token_index, "unk_token");
    if (unknown_token_index >= 0)
    {
        char *unknown_token;
        uint32_t unknown_token_id;

        if (SparkJsonCopyString(&document, unknown_token_index, &unknown_token) == SPARK_STATUS_OK)
        {
            if (SparkTokenizerFindTokenId(
                    tokenizer,
                    unknown_token,
                    (uint32_t)strlen(unknown_token),
                    &unknown_token_id) == SPARK_STATUS_OK)
            {
                tokenizer->has_unk_token = 1u;
                tokenizer->unk_token_id = unknown_token_id;
            }
            free(unknown_token);
        }
    }

    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerBuildByteTokenTable(tokenizer);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerParseMerges(tokenizer, &document, merges_token_index);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerBuildMergePairTableFromMergeKeys(tokenizer);
    }
    added_tokens_token_index = SparkJsonFindObjectMember(&document, root_token_index, "added_tokens");
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerParseAddedTokens(tokenizer, &document, added_tokens_token_index);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerBuildReverseVocabulary(tokenizer);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkJsonDocumentDestroy(&document);
        SparkTokenizerDestroy(tokenizer);
        return status;
    }

    byte_fallback = false;
    byte_fallback_token_index = SparkJsonFindObjectMember(&document, model_token_index, "byte_fallback");
    if (byte_fallback_token_index >= 0)
    {
        (void)SparkJsonGetBoolean(&document, byte_fallback_token_index, &byte_fallback);
    }
    tokenizer->byte_fallback = byte_fallback ? 1u : 0u;

    add_prefix_space = false;
    use_regex = true;
    pre_tokenizer_token_index = SparkJsonFindObjectMember(&document, root_token_index, "pre_tokenizer");
    if (pre_tokenizer_token_index >= 0)
    {
        (void)SparkTokenizerJsonSearchBooleanMemberRecursive(
            &document,
            pre_tokenizer_token_index,
            "add_prefix_space",
            &add_prefix_space);
        (void)SparkTokenizerJsonSearchBooleanMemberRecursive(
            &document,
            pre_tokenizer_token_index,
            "use_regex",
            &use_regex);
    }
    tokenizer->add_prefix_space = add_prefix_space ? 1u : 0u;
    tokenizer->byte_level_use_regex = use_regex ? 1u : 0u;
    tokenizer->ignore_merges = 0u;
    tokenizer->rank_ordered_merges = 0u;
    {
        int32_t ignore_merges_token_index;
        bool ignore_merges = false;
        ignore_merges_token_index = SparkJsonFindObjectMember(
            &document,
            model_token_index,
            "ignore_merges");
        if (ignore_merges_token_index >= 0)
        {
            (void)SparkJsonGetBoolean(&document,ignore_merges_token_index,&ignore_merges);
        }
        tokenizer->ignore_merges = ignore_merges ? 1u : 0u;
    }
    {
        uint32_t split_variant = SPARK_TOKENIZER_SPLIT_VARIANT_NONE;
        if (SparkTokenizerJsonHasExtendedSplitPattern(
                &document,
                pre_tokenizer_token_index,
                &split_variant) != 0u)
        {
            tokenizer->byte_level_use_regex =
                split_variant == SPARK_TOKENIZER_SPLIT_VARIANT_EXTENDED_DIGIT_RUNS ? 3u : 2u;
        }
    }
    SparkTokenizerSortSpecialTokens(tokenizer);
    SparkJsonDocumentDestroy(&document);
    return SPARK_STATUS_OK;
}


static int SparkTokenizerBase64Value(uint8_t character)
{
    if (character >= 'A' && character <= 'Z')
    {
        return (int)(character - 'A');
    }
    if (character >= 'a' && character <= 'z')
    {
        return 26 + (int)(character - 'a');
    }
    if (character >= '0' && character <= '9')
    {
        return 52 + (int)(character - '0');
    }
    if (character == '+')
    {
        return 62;
    }
    if (character == '/')
    {
        return 63;
    }
    return -1;
}

static SparkStatus SparkTokenizerBase64Decode(
    const char *text,
    uint32_t text_bytes,
    uint8_t *decoded,
    uint32_t decoded_capacity,
    uint32_t *decoded_bytes_out)
{
    uint32_t index;
    uint32_t output_index;
    uint32_t buffer;
    uint32_t buffer_bits;
    uint32_t body_bytes;

    if (text == 0 || decoded_bytes_out == 0 || (text_bytes % 4u) != 0u || text_bytes == 0u)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    body_bytes = text_bytes;
    while (body_bytes > text_bytes - 2u && text[body_bytes - 1u] == '=')
    {
        body_bytes -= 1u;
    }
    output_index = 0u;
    buffer = 0u;
    buffer_bits = 0u;
    for (index = 0u; index < body_bytes; ++index)
    {
        int value = SparkTokenizerBase64Value((uint8_t)text[index]);
        if (value < 0)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        buffer = (buffer << 6) | (uint32_t)value;
        buffer_bits += 6u;
        if (buffer_bits >= 8u)
        {
            if (output_index >= decoded_capacity)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            buffer_bits -= 8u;
            decoded[output_index++] = (uint8_t)((buffer >> buffer_bits) & 0xffu);
        }
    }
    if (output_index == 0u)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    *decoded_bytes_out = output_index;
    return SPARK_STATUS_OK;
}

static char *SparkTokenizerReadWholeFile(
    const char *path,
    uint32_t *bytes_out)
{
    FILE *file;
    long file_bytes;
    char *buffer;

    if (path == 0 || bytes_out == 0)
    {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == 0)
    {
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_bytes = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0)
    {
        fclose(file);
        return 0;
    }
    buffer = (char *)malloc((size_t)file_bytes + 1u);
    if (buffer == 0)
    {
        fclose(file);
        return 0;
    }
    if (fread(buffer, 1u, (size_t)file_bytes, file) != (size_t)file_bytes)
    {
        free(buffer);
        fclose(file);
        return 0;
    }
    fclose(file);
    buffer[file_bytes] = '\0';
    *bytes_out = (uint32_t)file_bytes;
    return buffer;
}

SparkStatus SparkTokenizerLoadTiktokenRanks(
    SparkTokenizer *tokenizer,
    const SparkTokenizerTiktokenRanksConfiguration *configuration)
{
    char *file_text;
    uint32_t file_bytes;
    uint32_t line_count;
    uint32_t cursor;
    SparkStatus status;

    if (tokenizer == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_TIKTOKEN_RANKS_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->ranks_path == 0 ||
        configuration->reserved0 != 0u || configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkTokenizerReset(tokenizer);
    file_text = SparkTokenizerReadWholeFile(configuration->ranks_path, &file_bytes);
    if (file_text == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    line_count = 0u;
    for (cursor = 0u; cursor < file_bytes;)
    {
        uint32_t line_end = cursor;
        while (line_end < file_bytes && file_text[line_end] != '\n')
        {
            line_end += 1u;
        }
        if (line_end > cursor)
        {
            line_count += 1u;
        }
        cursor = line_end + 1u;
    }
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->vocabulary_entries,
        &tokenizer->vocabulary_buckets,
        &tokenizer->vocabulary_bucket_count,
        line_count);
    if (status != SPARK_STATUS_OK)
    {
        free(file_text);
        SparkTokenizerDestroy(tokenizer);
        return status;
    }

    for (cursor = 0u; cursor < file_bytes;)
    {
        uint32_t line_end = cursor;
        uint32_t piece_cursor;
        uint32_t piece_end;
        uint8_t piece_bytes[SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES];
        uint32_t piece_byte_count = 0u;
        unsigned long rank;
        char *end_pointer;
        char glyph_text[SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES * 4u];
        uint32_t glyph_bytes;
        uint32_t byte_index;
        uint32_t token_id;
        char rank_text[16u];
        uint32_t rank_bytes;

        while (line_end < file_bytes && file_text[line_end] != '\n')
        {
            line_end += 1u;
        }
        if (line_end == cursor)
        {
            cursor = line_end + 1u;
            continue;
        }
        piece_end = cursor;
        while (piece_end < line_end && file_text[piece_end] != ' ')
        {
            piece_end += 1u;
        }
        if (piece_end == cursor || piece_end == line_end ||
            (piece_end - cursor) % 4u != 0u)
        {
            free(file_text);
            SparkTokenizerDestroy(tokenizer);
            return SPARK_STATUS_PARSE_ERROR;
        }
        {
            SparkStatus decode_status = SparkTokenizerBase64Decode(
                file_text + cursor,
                piece_end - cursor,
                piece_bytes,
                (uint32_t)sizeof(piece_bytes),
                &piece_byte_count);
            if (decode_status != SPARK_STATUS_OK)
            {
                free(file_text);
                SparkTokenizerDestroy(tokenizer);
                return decode_status;
            }
        }
        rank_bytes = 0u;
        for (piece_cursor = piece_end + 1u; piece_cursor < line_end; ++piece_cursor)
        {
            if (file_text[piece_cursor] == ' ' ||
                file_text[piece_cursor] == '\r' ||
                rank_bytes + 1u >= sizeof(rank_text))
            {
                break;
            }
            rank_text[rank_bytes++] = file_text[piece_cursor];
        }
        rank_text[rank_bytes] = '\0';
        rank = strtoul(rank_text, &end_pointer, 10);
        if (end_pointer == rank_text || *end_pointer != '\0' || rank > 0xfffffffeul)
        {
            free(file_text);
            SparkTokenizerDestroy(tokenizer);
            return SPARK_STATUS_PARSE_ERROR;
        }
        glyph_bytes = 0u;
        for (byte_index = 0u; byte_index < piece_byte_count; ++byte_index)
        {
            glyph_bytes += SparkTokenizerAppendUtf8CodePoint(
                glyph_text + glyph_bytes,
                SparkTokenizerByteToUnicodeCodePoint(piece_bytes[byte_index]));
        }
        token_id = (uint32_t)rank;
        status = SparkTokenizerInsertStringEntry(
            tokenizer->vocabulary_entries,
            tokenizer->vocabulary_buckets,
            tokenizer->vocabulary_bucket_count,
            tokenizer->vocabulary_count,
            glyph_text,
            glyph_bytes,
            token_id);
        if (status == SPARK_STATUS_DUPLICATE)
        {
            free(file_text);
            SparkTokenizerDestroy(tokenizer);
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (status != SPARK_STATUS_OK)
        {
            free(file_text);
            SparkTokenizerDestroy(tokenizer);
            return status;
        }
        tokenizer->vocabulary_count += 1u;
        if (token_id > tokenizer->maximum_token_id)
        {
            tokenizer->maximum_token_id = token_id;
        }
        cursor = line_end + 1u;
    }
    free(file_text);
    if (tokenizer->vocabulary_count != line_count || tokenizer->vocabulary_count == 0u)
    {
        SparkTokenizerDestroy(tokenizer);
        return SPARK_STATUS_PARSE_ERROR;
    }

    tokenizer->rank_ordered_merges = 1u;
    tokenizer->ignore_merges = 0u;
    status = SparkTokenizerBuildByteTokenTable(tokenizer);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerBuildReverseVocabulary(tokenizer);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerDestroy(tokenizer);
        return status;
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkTokenizerBinaryWriteUInt64(
    FILE *file,
    uint64_t value)
{
    return fwrite(&value, sizeof(value), 1u, file) == 1u ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkTokenizerBinaryReadUInt64(
    FILE *file,
    uint64_t *value_out)
{
    if (value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return fread(value_out, sizeof(*value_out), 1u, file) == 1u ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkTokenizerBinaryWriteUInt32(
    FILE *file,
    uint32_t value)
{
    return fwrite(&value, sizeof(value), 1u, file) == 1u ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkTokenizerBinaryReadUInt32(
    FILE *file,
    uint32_t *value_out)
{
    if (value_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return fread(value_out, sizeof(*value_out), 1u, file) == 1u ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkTokenizerBinaryWriteBytes(
    FILE *file,
    const char *text,
    uint32_t text_bytes)
{
    if (text_bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    return fwrite(text, 1u, text_bytes, file) == text_bytes ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkTokenizerBinaryReadAllocatedBytes(
    FILE *file,
    uint32_t text_bytes,
    char **text_out)
{
    char *text;

    if (text_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *text_out = 0;
    text = (char *)malloc((size_t)text_bytes + 1u);
    if (text == 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (text_bytes != 0u && fread(text, 1u, text_bytes, file) != text_bytes)
    {
        free(text);
        return SPARK_STATUS_IO_ERROR;
    }
    text[text_bytes] = '\0';
    *text_out = text;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerSaveCompiledFile(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerCompiledFileConfiguration *configuration)
{
    FILE *file;
    uint32_t index;
    SparkStatus status;

    if (tokenizer == 0 || configuration == 0 ||
        tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->compiled_tokenizer_path == 0 ||
        configuration->reserved0 != 0u || configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    file = fopen(configuration->compiled_tokenizer_path, "wb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    status = SparkTokenizerBinaryWriteUInt64(file, SPARK_TOKENIZER_COMPILED_FILE_MAGIC);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, SPARK_TOKENIZER_COMPILED_FILE_VERSION);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->model_kind);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->add_prefix_space);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->byte_fallback);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->has_unk_token);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->unk_token_id);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->maximum_token_id);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->byte_level_use_regex);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->vocabulary_count);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->fast_merge_pair_count);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, tokenizer->special_token_count);

    for (index = 0u; status == SPARK_STATUS_OK && index < tokenizer->vocabulary_count; ++index)
    {
        const SparkTokenizerStringEntry *entry;

        entry = &tokenizer->vocabulary_entries[index];
        status = SparkTokenizerBinaryWriteUInt32(file, entry->value);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, entry->text_bytes);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteBytes(file, entry->text, entry->text_bytes);
    }
    for (index = 0u; status == SPARK_STATUS_OK && index < tokenizer->fast_merge_pair_count; ++index)
    {
        const SparkTokenizerFastMergePair *entry;

        entry = &tokenizer->fast_merge_pairs[index];
        status = SparkTokenizerBinaryWriteUInt32(file, entry->left_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, entry->right_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, entry->merged_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, entry->rank);
    }
    for (index = 0u; status == SPARK_STATUS_OK && index < tokenizer->special_token_count; ++index)
    {
        const SparkTokenizerSpecialToken *special_token;

        special_token = &tokenizer->special_tokens[index];
        status = SparkTokenizerBinaryWriteUInt32(file, special_token->token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, special_token->is_special);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteUInt32(file, special_token->text_bytes);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryWriteBytes(file, special_token->text, special_token->text_bytes);
    }

    if (fclose(file) != 0 && status == SPARK_STATUS_OK)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    return status;
}

SparkStatus SparkTokenizerLoadCompiledFile(
    SparkTokenizer *tokenizer,
    const SparkTokenizerCompiledFileConfiguration *configuration)
{
    FILE *file;
    uint64_t magic;
    uint32_t version;
    uint32_t vocabulary_count;
    uint32_t fast_merge_pair_count;
    uint32_t special_token_count;
    uint32_t index;
    SparkStatus status;

    if (tokenizer == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->compiled_tokenizer_path == 0 ||
        configuration->reserved0 != 0u || configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkTokenizerReset(tokenizer);
    file = fopen(configuration->compiled_tokenizer_path, "rb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    status = SparkTokenizerBinaryReadUInt64(file, &magic);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &version);
    if (status != SPARK_STATUS_OK || magic != SPARK_TOKENIZER_COMPILED_FILE_MAGIC ||
        (version != SPARK_TOKENIZER_COMPILED_FILE_VERSION &&
         version != SPARK_TOKENIZER_COMPILED_FILE_LEGACY_VERSION))
    {
        fclose(file);
        SparkTokenizerDestroy(tokenizer);
        return status == SPARK_STATUS_OK ? SPARK_STATUS_SCHEMA_ERROR : status;
    }
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->model_kind);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->add_prefix_space);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->byte_fallback);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->has_unk_token);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->unk_token_id);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->maximum_token_id);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &tokenizer->byte_level_use_regex);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &vocabulary_count);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &fast_merge_pair_count);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &special_token_count);
    if (status != SPARK_STATUS_OK)
    {
        fclose(file);
        SparkTokenizerDestroy(tokenizer);
        return status;
    }

    tokenizer->vocabulary_count = vocabulary_count;
    status = SparkTokenizerAllocateStringTable(
        &tokenizer->vocabulary_entries,
        &tokenizer->vocabulary_buckets,
        &tokenizer->vocabulary_bucket_count,
        vocabulary_count);
    for (index = 0u; status == SPARK_STATUS_OK && index < vocabulary_count; ++index)
    {
        uint32_t token_id;
        uint32_t text_bytes;
        char *text;

        status = SparkTokenizerBinaryReadUInt32(file, &token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &text_bytes);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadAllocatedBytes(file, text_bytes, &text);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkTokenizerInsertStringEntry(
                tokenizer->vocabulary_entries,
                tokenizer->vocabulary_buckets,
                tokenizer->vocabulary_bucket_count,
                index,
                text,
                text_bytes,
                token_id);
            free(text);
        }
    }
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBuildByteTokenTable(tokenizer);
    if (status == SPARK_STATUS_OK) status = SparkTokenizerAllocateMergePairTable(tokenizer, fast_merge_pair_count);
    for (index = 0u; status == SPARK_STATUS_OK && index < fast_merge_pair_count; ++index)
    {
        uint32_t left_token_id;
        uint32_t right_token_id;
        uint32_t merged_token_id;
        uint32_t rank;

        status = SparkTokenizerBinaryReadUInt32(file, &left_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &right_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &merged_token_id);
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &rank);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkTokenizerInsertMergePairEntry(
                tokenizer,
                left_token_id,
                right_token_id,
                merged_token_id,
                rank);
        }
    }
    tokenizer->special_tokens = (SparkTokenizerSpecialToken *)calloc(
        special_token_count,
        sizeof(*tokenizer->special_tokens));
    if (status == SPARK_STATUS_OK && special_token_count != 0u && tokenizer->special_tokens == 0)
    {
        status = SPARK_STATUS_INTERNAL_ERROR;
    }
    tokenizer->special_token_count = special_token_count;
    for (index = 0u; status == SPARK_STATUS_OK && index < special_token_count; ++index)
    {
        uint32_t token_id;
        uint32_t is_special;
        uint32_t text_bytes;
        char *text;

        status = SparkTokenizerBinaryReadUInt32(file, &token_id);
        is_special = 1u;
        if (status == SPARK_STATUS_OK &&
            version >= SPARK_TOKENIZER_COMPILED_FILE_VERSION)
        {
            status = SparkTokenizerBinaryReadUInt32(file, &is_special);
            if (status == SPARK_STATUS_OK && is_special > 1u)
            {
                status = SPARK_STATUS_SCHEMA_ERROR;
            }
        }
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadUInt32(file, &text_bytes);
        if (status == SPARK_STATUS_OK && text_bytes == 0u) status = SPARK_STATUS_SCHEMA_ERROR;
        if (status == SPARK_STATUS_OK) status = SparkTokenizerBinaryReadAllocatedBytes(file, text_bytes, &text);
        if (status == SPARK_STATUS_OK)
        {
            tokenizer->special_tokens[index].token_id = token_id;
            tokenizer->special_tokens[index].is_special = is_special;
            tokenizer->special_tokens[index].text_bytes = text_bytes;
            tokenizer->special_tokens[index].text = text;
        }
    }
    if (status == SPARK_STATUS_OK) status = SparkTokenizerBuildReverseVocabulary(tokenizer);

    if (fclose(file) != 0 && status == SPARK_STATUS_OK)
    {
        status = SPARK_STATUS_IO_ERROR;
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerDestroy(tokenizer);
    }
    else
    {
        SparkTokenizerSortSpecialTokens(tokenizer);
    }
    return status;
}

static SparkStatus SparkTokenizerAppendTokenToEncoding(
    SparkTokenizerEncoding *encoding,
    uint32_t token_id)
{
    if (encoding == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (encoding->token_count < encoding->token_capacity)
    {
        encoding->token_ids[encoding->token_count] = token_id;
        encoding->token_count += 1u;
    }
    else
    {
        encoding->overflow_token_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkTokenizerFindSpecialTokenAt(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t remaining_text_bytes,
    uint32_t match_special_tokens,
    uint32_t *token_id_out,
    uint32_t *matched_text_bytes_out)
{
    uint32_t special_token_index;

    if (tokenizer == 0 || text == 0 || token_id_out == 0 || matched_text_bytes_out == 0)
    {
        return 0u;
    }
    for (special_token_index = 0u;
         special_token_index < tokenizer->special_token_count;
         ++special_token_index)
    {
        const SparkTokenizerSpecialToken *special_token;

        special_token = &tokenizer->special_tokens[special_token_index];
        if ((special_token->is_special == 0u || match_special_tokens != 0u) &&
            special_token->text_bytes <= remaining_text_bytes &&
            memcmp(text, special_token->text, special_token->text_bytes) == 0)
        {
            *token_id_out = special_token->token_id;
            *matched_text_bytes_out = special_token->text_bytes;
            return 1u;
        }
    }
    return 0u;
}

static int SparkTokenizerCompareMergeCandidates(
    const SparkTokenizerMergeCandidate *left,
    const SparkTokenizerMergeCandidate *right)
{
    if (left->rank != right->rank)
    {
        return left->rank < right->rank ? -1 : 1;
    }
    if (left->left_symbol_index != right->left_symbol_index)
    {
        return left->left_symbol_index < right->left_symbol_index ? -1 : 1;
    }
    return 0;
}

static void SparkTokenizerHeapSwap(
    SparkTokenizerMergeCandidate *left,
    SparkTokenizerMergeCandidate *right)
{
    SparkTokenizerMergeCandidate temporary;

    temporary = *left;
    *left = *right;
    *right = temporary;
}

static SparkStatus SparkTokenizerHeapPush(
    SparkTokenizerWorkspace *workspace,
    const SparkTokenizerMergeCandidate *candidate)
{
    uint32_t heap_index;

    if (workspace == 0 || candidate == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->merge_heap_count >= workspace->heap_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    heap_index = workspace->merge_heap_count;
    workspace->merge_heap[heap_index] = *candidate;
    workspace->merge_heap_count += 1u;
    while (heap_index > 0u)
    {
        uint32_t parent_index;

        parent_index = (heap_index - 1u) / 2u;
        if (SparkTokenizerCompareMergeCandidates(
                &workspace->merge_heap[heap_index],
                &workspace->merge_heap[parent_index]) >= 0)
        {
            break;
        }
        SparkTokenizerHeapSwap(&workspace->merge_heap[heap_index], &workspace->merge_heap[parent_index]);
        heap_index = parent_index;
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkTokenizerHeapPop(
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerMergeCandidate *candidate_out)
{
    uint32_t heap_index;

    if (workspace == 0 || candidate_out == 0 || workspace->merge_heap_count == 0u)
    {
        return 0u;
    }
    *candidate_out = workspace->merge_heap[0u];
    workspace->merge_heap_count -= 1u;
    if (workspace->merge_heap_count == 0u)
    {
        return 1u;
    }
    workspace->merge_heap[0u] = workspace->merge_heap[workspace->merge_heap_count];
    heap_index = 0u;
    while (1)
    {
        uint32_t left_child_index;
        uint32_t right_child_index;
        uint32_t best_child_index;

        left_child_index = heap_index * 2u + 1u;
        right_child_index = left_child_index + 1u;
        if (left_child_index >= workspace->merge_heap_count)
        {
            break;
        }
        best_child_index = left_child_index;
        if (right_child_index < workspace->merge_heap_count &&
            SparkTokenizerCompareMergeCandidates(
                &workspace->merge_heap[right_child_index],
                &workspace->merge_heap[left_child_index]) < 0)
        {
            best_child_index = right_child_index;
        }
        if (SparkTokenizerCompareMergeCandidates(
                &workspace->merge_heap[heap_index],
                &workspace->merge_heap[best_child_index]) <= 0)
        {
            break;
        }
        SparkTokenizerHeapSwap(&workspace->merge_heap[heap_index], &workspace->merge_heap[best_child_index]);
        heap_index = best_child_index;
    }
    return 1u;
}

static SparkStatus SparkTokenizerFindRankOrderedMerge(
    const SparkTokenizer *tokenizer,
    uint32_t left_token_id,
    uint32_t right_token_id,
    uint32_t *merged_token_id_out)
{
    char merge_key[SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES];
    uint32_t left_bytes;
    uint32_t right_bytes;
    uint32_t merged_token_id;

    if (tokenizer == 0 || merged_token_id_out == 0 ||
        tokenizer->token_text_by_id == 0 || tokenizer->token_text_bytes_by_id == 0 ||
        left_token_id > tokenizer->maximum_token_id ||
        right_token_id > tokenizer->maximum_token_id)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    left_bytes = tokenizer->token_text_bytes_by_id[left_token_id];
    right_bytes = tokenizer->token_text_bytes_by_id[right_token_id];
    if (tokenizer->token_text_by_id[left_token_id] == 0 ||
        tokenizer->token_text_by_id[right_token_id] == 0 ||
        (uint64_t)left_bytes + (uint64_t)right_bytes > sizeof(merge_key))
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    memcpy(merge_key,tokenizer->token_text_by_id[left_token_id],left_bytes);
    memcpy(merge_key + left_bytes,tokenizer->token_text_by_id[right_token_id],right_bytes);
    if (SparkTokenizerFindTokenId(tokenizer,merge_key,left_bytes + right_bytes,&merged_token_id) !=
        SPARK_STATUS_OK)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    *merged_token_id_out = merged_token_id;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerPushPairCandidate(
    const SparkTokenizer *tokenizer,
    SparkTokenizerWorkspace *workspace,
    uint32_t left_symbol_index)
{
    uint32_t right_symbol_index;
    SparkTokenizerFastMergePair *merge_pair;
    SparkTokenizerMergeCandidate candidate;

    if (left_symbol_index == SPARK_TOKENIZER_SYMBOL_NONE)
    {
        return SPARK_STATUS_OK;
    }
    right_symbol_index = workspace->next_symbol_indices[left_symbol_index];
    if (right_symbol_index == SPARK_TOKENIZER_SYMBOL_NONE)
    {
        return SPARK_STATUS_OK;
    }
    if (tokenizer->rank_ordered_merges != 0u)
    {
        uint32_t merged_token_id = 0u;
        if (SparkTokenizerFindRankOrderedMerge(
                tokenizer,
                workspace->symbol_token_ids[left_symbol_index],
                workspace->symbol_token_ids[right_symbol_index],
                &merged_token_id) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_OK;
        }
        candidate.rank = merged_token_id;
        candidate.left_symbol_index = left_symbol_index;
        candidate.left_generation = workspace->symbol_generations[left_symbol_index];
        candidate.right_generation = workspace->symbol_generations[right_symbol_index];
        return SparkTokenizerHeapPush(workspace, &candidate);
    }
    merge_pair = SparkTokenizerFindMergePairEntry(
        tokenizer,
        workspace->symbol_token_ids[left_symbol_index],
        workspace->symbol_token_ids[right_symbol_index]);
    if (merge_pair == 0)
    {
        return SPARK_STATUS_OK;
    }
    candidate.rank = merge_pair->rank;
    candidate.left_symbol_index = left_symbol_index;
    candidate.left_generation = workspace->symbol_generations[left_symbol_index];
    candidate.right_generation = workspace->symbol_generations[right_symbol_index];
    return SparkTokenizerHeapPush(workspace, &candidate);
}

static uint64_t SparkTokenizerPieceHash(const char *text,uint32_t text_bytes)
{
    uint64_t hash = 1469598103934665603u;
    uint32_t byte_index;
    for (byte_index = 0u; byte_index < text_bytes; ++byte_index)
    {
        hash ^= (uint8_t)text[byte_index];
        hash *= 1099511628211u;
    }
    if (hash == SPARK_TOKENIZER_PIECE_CACHE_EMPTY_HASH)
        hash = 1u;
    return hash;
}

static SparkTokenizerPieceCacheEntry *SparkTokenizerPieceCacheLookup(SparkTokenizerWorkspace *workspace,const char *text,uint32_t text_bytes,uint64_t hash)
{
    uint32_t slot = (uint32_t)(hash & (workspace->piece_cache_slot_count - 1u)),probes = 0u;
    while (probes < workspace->piece_cache_slot_count)
    {
        SparkTokenizerPieceCacheEntry *entry = &workspace->piece_cache_entries[slot];
        if (entry->hash == SPARK_TOKENIZER_PIECE_CACHE_EMPTY_HASH)
            return entry;
        if (entry->hash == hash && entry->piece_bytes == text_bytes &&
            memcmp(entry->piece,text,text_bytes) == 0)
            return entry;
        slot = ((slot + 1u) & (workspace->piece_cache_slot_count - 1u));
        probes += 1u;
    }
    return 0;
}

static SparkStatus SparkTokenizerEncodeByteLevelPieceUncached(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    SparkTokenizerWorkspace *workspace,
    uint32_t *token_ids_out,
    uint32_t token_ids_capacity,
    uint32_t *token_count_out,
    uint32_t *invalid_out)
{
    uint32_t symbol_index;
    uint32_t head_symbol_index;
    uint32_t live_symbol_count;
    uint32_t emitted_count;
    SparkStatus status;

    *token_count_out = 0u;
    *invalid_out = 0u;
    if (text_bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (tokenizer->ignore_merges != 0u && text_bytes <= SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES)
    {
        char glyph_text[SPARK_TOKENIZER_MAX_MERGE_KEY_INLINE_BYTES * 4u];
        uint32_t glyph_bytes = 0u;
        uint32_t whole_token_id = 0u;
        uint32_t byte_index;
        for (byte_index = 0u; byte_index < text_bytes; ++byte_index)
        {
            glyph_bytes += SparkTokenizerAppendUtf8CodePoint(
                glyph_text + glyph_bytes,
                SparkTokenizerByteToUnicodeCodePoint((uint8_t)text[byte_index]));
        }
        if (SparkTokenizerFindTokenId(tokenizer,glyph_text,glyph_bytes,&whole_token_id) == SPARK_STATUS_OK)
        {
            if (token_ids_capacity == 0u)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            token_ids_out[0u] = whole_token_id;
            *token_count_out = 1u;
            return SPARK_STATUS_OK;
        }
    }
    if (text_bytes > workspace->maximum_symbol_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    workspace->merge_heap_count = 0u;
    for (symbol_index = 0u; symbol_index < text_bytes; ++symbol_index)
    {
        uint32_t token_id;

        token_id = tokenizer->byte_token_ids[(uint8_t)text[symbol_index]];
        if (token_id == SPARK_TOKENIZER_NO_TOKEN_ID)
        {
            *invalid_out = 1u;
            return SPARK_STATUS_NOT_FOUND;
        }
        workspace->symbol_token_ids[symbol_index] = token_id;
        workspace->previous_symbol_indices[symbol_index] =
            symbol_index == 0u ? SPARK_TOKENIZER_SYMBOL_NONE : symbol_index - 1u;
        workspace->next_symbol_indices[symbol_index] =
            symbol_index + 1u < text_bytes ? symbol_index + 1u : SPARK_TOKENIZER_SYMBOL_NONE;
        workspace->symbol_generations[symbol_index] = 1u;
    }

    for (symbol_index = 0u; symbol_index + 1u < text_bytes; ++symbol_index)
    {
        status = SparkTokenizerPushPairCandidate(tokenizer, workspace, symbol_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    head_symbol_index = 0u;
    live_symbol_count = text_bytes;
    while (live_symbol_count > 1u)
    {
        SparkTokenizerMergeCandidate candidate;
        uint32_t left_symbol_index;
        uint32_t right_symbol_index;
        uint32_t previous_symbol_index;
        uint32_t next_symbol_index;
        uint32_t merged_token_id;
        SparkTokenizerFastMergePair *merge_pair;

        if (!SparkTokenizerHeapPop(workspace, &candidate))
        {
            break;
        }
        left_symbol_index = candidate.left_symbol_index;
        if (left_symbol_index >= text_bytes ||
            workspace->symbol_generations[left_symbol_index] != candidate.left_generation)
        {
            continue;
        }
        right_symbol_index = workspace->next_symbol_indices[left_symbol_index];
        if (right_symbol_index == SPARK_TOKENIZER_SYMBOL_NONE ||
            workspace->symbol_generations[right_symbol_index] != candidate.right_generation)
        {
            continue;
        }
        merged_token_id = 0u;
        if (tokenizer->rank_ordered_merges != 0u)
        {
            if (SparkTokenizerFindRankOrderedMerge(
                    tokenizer,
                    workspace->symbol_token_ids[left_symbol_index],
                    workspace->symbol_token_ids[right_symbol_index],
                    &merged_token_id) != SPARK_STATUS_OK ||
                merged_token_id != candidate.rank)
            {
                continue;
            }
        }
        else
        {
            merge_pair = SparkTokenizerFindMergePairEntry(
                tokenizer,
                workspace->symbol_token_ids[left_symbol_index],
                workspace->symbol_token_ids[right_symbol_index]);
            if (merge_pair == 0 || merge_pair->rank != candidate.rank)
            {
                continue;
            }
            merged_token_id = merge_pair->merged_token_id;
        }

        previous_symbol_index = workspace->previous_symbol_indices[left_symbol_index];
        next_symbol_index = workspace->next_symbol_indices[right_symbol_index];
        workspace->symbol_token_ids[left_symbol_index] = merged_token_id;
        workspace->next_symbol_indices[left_symbol_index] = next_symbol_index;
        if (next_symbol_index != SPARK_TOKENIZER_SYMBOL_NONE)
        {
            workspace->previous_symbol_indices[next_symbol_index] = left_symbol_index;
        }
        workspace->symbol_generations[left_symbol_index] += 1u;
        workspace->symbol_generations[right_symbol_index] += 1u;
        live_symbol_count -= 1u;

        status = SparkTokenizerPushPairCandidate(tokenizer, workspace, previous_symbol_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkTokenizerPushPairCandidate(tokenizer, workspace, left_symbol_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    symbol_index = head_symbol_index;
    emitted_count = 0u;
    while (symbol_index != SPARK_TOKENIZER_SYMBOL_NONE)
    {
        if (emitted_count >= token_ids_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        token_ids_out[emitted_count] = workspace->symbol_token_ids[symbol_index];
        emitted_count += 1u;
        symbol_index = workspace->next_symbol_indices[symbol_index];
    }
    *token_count_out = emitted_count;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerEncodeByteLevelPiece(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    SparkTokenizerPieceCacheEntry *entry;
    uint64_t hash;
    uint32_t token_index;
    SparkStatus status;

    if (text_bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (text_bytes > SPARK_TOKENIZER_PIECE_CACHE_INLINE_BYTES ||
        (encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE) != 0u ||
        workspace->piece_cache_entries == 0 || workspace->piece_cache_token_pool == 0)
    {
        uint32_t direct_count = 0u,invalid = 0u,emit_index;
        status = SparkTokenizerEncodeByteLevelPieceUncached(
            tokenizer,text,text_bytes,workspace,
            workspace->symbol_token_ids,workspace->maximum_symbol_count,
            &direct_count,&invalid);
        if (invalid != 0u)
        {
            encoding->invalid_segment_count += 1u;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (emit_index = 0u; emit_index < direct_count; ++emit_index)
        {
            status = SparkTokenizerAppendTokenToEncoding(encoding,workspace->symbol_token_ids[emit_index]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    hash = SparkTokenizerPieceHash(text,text_bytes);
    entry = SparkTokenizerPieceCacheLookup(workspace,text,text_bytes,hash);
    if (entry != 0 && entry->hash != SPARK_TOKENIZER_PIECE_CACHE_EMPTY_HASH)
    {
        const uint32_t *cached = &workspace->piece_cache_token_pool[entry->token_offset];
        for (token_index = 0u; token_index < entry->token_count; ++token_index)
        {
            status = SparkTokenizerAppendTokenToEncoding(encoding,cached[token_index]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        return SPARK_STATUS_OK;
    }
    {
        uint32_t produced = 0u,invalid = 0u,pool_room,token_index_local;
        uint32_t pool_start = workspace->piece_cache_token_used;
        pool_room = workspace->piece_cache_token_capacity - pool_start;
        status = SparkTokenizerEncodeByteLevelPieceUncached(
            tokenizer,text,text_bytes,workspace,
            workspace->symbol_token_ids,workspace->maximum_symbol_count,
            &produced,&invalid);
        if (invalid != 0u)
        {
            encoding->invalid_segment_count += 1u;
        }
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        for (token_index_local = 0u; token_index_local < produced; ++token_index_local)
        {
            status = SparkTokenizerAppendTokenToEncoding(encoding,workspace->symbol_token_ids[token_index_local]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
        if (entry != 0 && produced <= pool_room)
        {
            for (token_index_local = 0u; token_index_local < produced; ++token_index_local)
                workspace->piece_cache_token_pool[pool_start + token_index_local] =
                    workspace->symbol_token_ids[token_index_local];
            entry->hash = hash;
            entry->piece_bytes = text_bytes;
            entry->token_offset = pool_start;
            entry->token_count = produced;
            memcpy(entry->piece,text,text_bytes);
            workspace->piece_cache_token_used = pool_start + produced;
        }
        return SPARK_STATUS_OK;
    }
}

static uint32_t SparkTokenizerIsAsciiWhitespace(
    uint8_t value)
{
    return value == ' ' || value == '\t' || value == '\n' ||
        value == '\r' || value == '\f' || value == '\v';
}

static const uint8_t g_spark_tokenizer_byte_class[256] =
{
    4,4,4,4,4,4,4,4,4,1,1,1,1,1,4,4, 4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    1,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, 3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,
    4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,
    4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
};

static uint32_t SparkTokenizerScanClassRun(const char *text,uint32_t text_bytes,uint32_t start,uint8_t target_class)
{
    uint32_t scan_position = start;
    while (scan_position < text_bytes &&
        g_spark_tokenizer_byte_class[(uint8_t)text[scan_position]] == target_class)
    {
        scan_position += 1u;
    }
    return scan_position;
}

static uint32_t SparkTokenizerMatchesContraction(
    const char *text,
    uint32_t text_bytes,
    uint32_t position,
    uint32_t *piece_bytes_out)
{
    uint8_t first;
    uint8_t second;
    uint8_t third;

    if (piece_bytes_out == 0 || position + 1u >= text_bytes || text[position] != '\'')
    {
        return 0u;
    }
    first = (uint8_t)text[position + 1u];
    if (first >= 'A' && first <= 'Z')
    {
        first = (uint8_t)(first - 'A' + 'a');
    }
    if (first == 's' || first == 't' || first == 'm' || first == 'd')
    {
        *piece_bytes_out = 2u;
        return 1u;
    }
    if (position + 2u >= text_bytes)
    {
        return 0u;
    }
    second = (uint8_t)text[position + 2u];
    if (second >= 'A' && second <= 'Z')
    {
        second = (uint8_t)(second - 'A' + 'a');
    }
    third = first;
    if ((third == 'r' && second == 'e') ||
        (third == 'v' && second == 'e') ||
        (third == 'l' && second == 'l'))
    {
        *piece_bytes_out = 3u;
        return 1u;
    }
    return 0u;
}

static uint32_t SparkTokenizerFindNextRegexPiece(
    const char *text,
    uint32_t text_bytes,
    uint32_t position,
    uint32_t *piece_start_out,
    uint32_t *piece_bytes_out)
{
    uint32_t class_id;
    uint32_t scan_position;
    uint32_t piece_bytes;

    if (text == 0 || piece_start_out == 0 || piece_bytes_out == 0 || position >= text_bytes)
    {
        return 0u;
    }
    *piece_start_out = position;
    *piece_bytes_out = 0u;

    if (SparkTokenizerMatchesContraction(text, text_bytes, position, &piece_bytes))
    {
        *piece_bytes_out = piece_bytes;
        return 1u;
    }

    if (text[position] == ' ' && position + 1u < text_bytes &&
        !SparkTokenizerIsAsciiWhitespace((uint8_t)text[position + 1u]))
    {
        scan_position = position + 1u;
        class_id = g_spark_tokenizer_byte_class[(uint8_t)text[scan_position]];
        if (class_id == 2u || class_id == 3u)
        {
            scan_position = SparkTokenizerScanClassRun(text, text_bytes, scan_position, (uint8_t)class_id);
        }
        else
        {
            while (scan_position < text_bytes &&
                g_spark_tokenizer_byte_class[(uint8_t)text[scan_position]] == class_id)
            {
                if (SparkTokenizerMatchesContraction(text, text_bytes, scan_position, &piece_bytes))
                {
                    break;
                }
                scan_position += 1u;
            }
        }
        *piece_bytes_out = scan_position - position;
        return 1u;
    }

    class_id = g_spark_tokenizer_byte_class[(uint8_t)text[position]];
    if (class_id == 2u || class_id == 3u || class_id == 1u)
    {
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, position, (uint8_t)class_id);
    }
    else
    {
        scan_position = position;
        while (scan_position < text_bytes &&
            g_spark_tokenizer_byte_class[(uint8_t)text[scan_position]] == class_id)
        {
            if (SparkTokenizerMatchesContraction(text, text_bytes, scan_position, &piece_bytes))
            {
                break;
            }
            scan_position += 1u;
        }
    }
    *piece_bytes_out = scan_position - position;
    return *piece_bytes_out != 0u;
}

static uint32_t SparkTokenizerFindNextExtendedPiece(
    const char *text,
    uint32_t text_bytes,
    uint32_t position,
    uint32_t digit_piece_maximum,
    uint32_t *piece_start_out,
    uint32_t *piece_bytes_out)
{
    uint8_t class_id;
    uint32_t scan_position;
    uint32_t piece_bytes;
    uint32_t last_newline;

    if (text == 0 || piece_start_out == 0 || piece_bytes_out == 0 || position >= text_bytes)
    {
        return 0u;
    }
    *piece_start_out = position;
    *piece_bytes_out = 0u;

    if (SparkTokenizerMatchesContraction(text, text_bytes, position, &piece_bytes))
    {
        *piece_bytes_out = piece_bytes;
        return 1u;
    }

    class_id = g_spark_tokenizer_byte_class[(uint8_t)text[position]];

    if (class_id == 2u)
    {
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, position, 2u);
        *piece_bytes_out = scan_position - position;
        return 1u;
    }
    if (text[position] != '\r' && text[position] != '\n' && class_id != 3u &&
        position + 1u < text_bytes &&
        g_spark_tokenizer_byte_class[(uint8_t)text[position + 1u]] == 2u)
    {
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, position + 1u, 2u);
        *piece_bytes_out = scan_position - position;
        return 1u;
    }

    if (class_id == 3u)
    {
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, position, 3u);
        piece_bytes = scan_position - position;
        if (piece_bytes > digit_piece_maximum)
        {
            piece_bytes = digit_piece_maximum;
        }
        *piece_bytes_out = piece_bytes;
        return 1u;
    }

    if (class_id == 4u ||
        (text[position] == ' ' && position + 1u < text_bytes &&
         g_spark_tokenizer_byte_class[(uint8_t)text[position + 1u]] == 4u))
    {
        scan_position = text[position] == ' ' ? position + 1u : position;
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, scan_position, 4u);
        while (scan_position < text_bytes &&
            (text[scan_position] == '\r' || text[scan_position] == '\n'))
        {
            scan_position += 1u;
        }
        *piece_bytes_out = scan_position - position;
        return 1u;
    }

    if (class_id == 1u)
    {
        scan_position = SparkTokenizerScanClassRun(text, text_bytes, position, 1u);
        last_newline = scan_position;
        while (last_newline > position &&
            text[last_newline - 1u] != '\r' && text[last_newline - 1u] != '\n')
        {
            last_newline -= 1u;
        }
        if (last_newline > position)
        {
            *piece_bytes_out = last_newline - position;
            return 1u;
        }
        if (scan_position < text_bytes && scan_position - position > 1u)
        {
            *piece_bytes_out = scan_position - position - 1u;
            return 1u;
        }
        *piece_bytes_out = scan_position - position;
        return 1u;
    }

    return 0u;
}

static SparkStatus SparkTokenizerEncodeRegularSegmentWithWorkspace(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    uint32_t position;
    SparkStatus status;

    if (text_bytes == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (tokenizer->byte_level_use_regex == 0u ||
        (encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_REGEX_PRETOKENIZATION) != 0u)
    {
        return SparkTokenizerEncodeByteLevelPiece(tokenizer, text, text_bytes, encode_flags, workspace, encoding);
    }

    position = 0u;
    while (position < text_bytes)
    {
        uint32_t piece_start;
        uint32_t piece_bytes;

        if (tokenizer->byte_level_use_regex == 2u
            ? !SparkTokenizerFindNextExtendedPiece(text, text_bytes, position, 1u, &piece_start, &piece_bytes)
            : tokenizer->byte_level_use_regex == 3u
            ? !SparkTokenizerFindNextExtendedPiece(text, text_bytes, position, 3u, &piece_start, &piece_bytes)
            : !SparkTokenizerFindNextRegexPiece(text, text_bytes, position, &piece_start, &piece_bytes))
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        status = SparkTokenizerEncodeByteLevelPiece(
            tokenizer,
            text + piece_start,
            piece_bytes,
            encode_flags,
            workspace,
            encoding);
        if (status != SPARK_STATUS_OK)
        {
            encoding->invalid_segment_count += 1u;
            return status;
        }
        position = piece_start + piece_bytes;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerEncodeUtf8WithWorkspace(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    uint32_t position;
    uint32_t segment_start;
    uint32_t add_prefix_space;
    SparkStatus status;

    if (tokenizer == 0 || tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES ||
        (text == 0 && text_bytes != 0u) ||
        (encode_flags & ~SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS) != 0u ||
        workspace == 0 || workspace->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        workspace->descriptor_bytes != SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES ||
        encoding == 0 || encoding->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        encoding->descriptor_bytes != SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES ||
        (encoding->token_ids == 0 && encoding->token_capacity != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (workspace->maximum_symbol_count < text_bytes + 1u ||
        workspace->symbol_token_ids == 0 ||
        workspace->previous_symbol_indices == 0 ||
        workspace->next_symbol_indices == 0 ||
        workspace->symbol_generations == 0 ||
        workspace->merge_heap == 0)
    {
        status = SparkTokenizerWorkspaceEnsureSymbolBuffers(workspace, text_bytes + 1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    status = SparkTokenizerWorkspaceEnsurePieceCache(workspace);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    encoding->token_count = 0u;
    encoding->overflow_token_count = 0u;
    encoding->invalid_segment_count = 0u;
    add_prefix_space = ((encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE) != 0u ||
        tokenizer->add_prefix_space != 0u) &&
        (text_bytes == 0u || text[0u] != ' ');
    if (add_prefix_space != 0u)
    {
        status = SparkTokenizerEncodeRegularSegmentWithWorkspace(
            tokenizer,
            " ",
            1u,
            encode_flags,
            workspace,
            encoding);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    position = 0u;
    segment_start = 0u;
    while (position < text_bytes)
    {
        uint32_t special_token_id;
        uint32_t matched_text_bytes;

        if (SparkTokenizerFindSpecialTokenAt(
                tokenizer,
                text + position,
                text_bytes - position,
                (encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH) == 0u,
                &special_token_id,
                &matched_text_bytes))
        {
            status = SparkTokenizerEncodeRegularSegmentWithWorkspace(
                tokenizer,
                text + segment_start,
                position - segment_start,
                encode_flags,
                workspace,
                encoding);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            status = SparkTokenizerAppendTokenToEncoding(encoding, special_token_id);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            position += matched_text_bytes;
            segment_start = position;
        }
        else
        {
            position += 1u;
        }
    }
    status = SparkTokenizerEncodeRegularSegmentWithWorkspace(
        tokenizer,
        text + segment_start,
        text_bytes - segment_start,
        encode_flags,
        workspace,
        encoding);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return encoding->overflow_token_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
}

SparkStatus SparkTokenizerEncodeUtf8(
    const SparkTokenizer *tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerEncoding *encoding)
{
    SparkTokenizerWorkspace workspace;
    SparkStatus status;
    uint32_t maximum_symbol_count;

    if (encoding == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkTokenizerWorkspaceReset(&workspace);
    maximum_symbol_count = text_bytes + 1u;
    if (maximum_symbol_count == 0u)
    {
        maximum_symbol_count = 1u;
    }
    status = SparkTokenizerWorkspaceInitialize(&workspace, maximum_symbol_count);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkTokenizerEncodeUtf8WithWorkspace(
            tokenizer,
            text,
            text_bytes,
            encode_flags,
            &workspace,
            encoding);
    }
    SparkTokenizerWorkspaceDestroy(&workspace);
    return status;
}

typedef struct SparkTokenizerBatchWorker
{
    const SparkTokenizer *tokenizer;
    const char *const *texts;
    const uint32_t *text_bytes;
    uint32_t first_text_index;
    uint32_t text_count;
    uint32_t encode_flags;
    uint32_t *token_ids;
    uint32_t token_stride;
    uint32_t *token_counts;
    uint32_t *overflow_token_counts;
    uint32_t *invalid_segment_counts;
    SparkStatus status;
} SparkTokenizerBatchWorker;

static void *SparkTokenizerBatchWorkerMain(
    void *argument)
{
    SparkTokenizerBatchWorker *worker;
    SparkTokenizerWorkspace workspace;
    uint32_t local_index;
    uint32_t maximum_text_bytes;
    SparkStatus final_status;

    worker = (SparkTokenizerBatchWorker *)argument;
    SparkTokenizerWorkspaceReset(&workspace);
    maximum_text_bytes = 1u;
    for (local_index = 0u; local_index < worker->text_count; ++local_index)
    {
        uint32_t text_index;

        text_index = worker->first_text_index + local_index;
        if (worker->text_bytes[text_index] + 1u > maximum_text_bytes)
        {
            maximum_text_bytes = worker->text_bytes[text_index] + 1u;
        }
    }
    final_status = SparkTokenizerWorkspaceInitialize(&workspace, maximum_text_bytes);
    for (local_index = 0u; final_status == SPARK_STATUS_OK && local_index < worker->text_count; ++local_index)
    {
        uint32_t text_index;
        SparkTokenizerEncoding encoding;
        SparkStatus status;

        text_index = worker->first_text_index + local_index;
        memset(&encoding, 0, sizeof(encoding));
        encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        encoding.descriptor_bytes = SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES;
        encoding.token_capacity = worker->token_stride;
        encoding.token_ids = &worker->token_ids[(uint64_t)text_index * worker->token_stride];
        status = SparkTokenizerEncodeUtf8WithWorkspace(
            worker->tokenizer,
            worker->texts[text_index],
            worker->text_bytes[text_index],
            worker->encode_flags,
            &workspace,
            &encoding);
        worker->token_counts[text_index] = encoding.token_count;
        worker->overflow_token_counts[text_index] = encoding.overflow_token_count;
        if (worker->invalid_segment_counts != 0)
        {
            worker->invalid_segment_counts[text_index] = encoding.invalid_segment_count;
        }
        if (status != SPARK_STATUS_OK && final_status == SPARK_STATUS_OK)
        {
            final_status = status;
        }
    }
    SparkTokenizerWorkspaceDestroy(&workspace);
    worker->status = final_status;
    return 0;
}

SparkStatus SparkTokenizerEncodeBatchUtf8ConfiguredInternal(
    const SparkTokenizer *tokenizer,
    const char *const *texts,
    const uint32_t *text_bytes,
    uint32_t text_count,
    uint32_t encode_flags,
    uint32_t *token_ids,
    uint32_t token_stride,
    uint32_t *token_counts,
    uint32_t *overflow_token_counts,
    uint32_t *invalid_segment_counts,
    uint32_t worker_count)
{
    SparkTokenizerBatchWorker *workers;
    pthread_t *threads;
    uint32_t worker_index;
    uint32_t launched_worker_count;
    uint32_t base_count;
    uint32_t remainder_count;
    uint32_t next_text_index;
    SparkStatus final_status;

    if (texts == 0 || text_bytes == 0 ||
        (token_ids == 0 && token_stride != 0u) ||
        token_counts == 0 || overflow_token_counts == 0 ||
        (text_count != 0u && token_stride == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (text_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (worker_count == 0u)
    {
        worker_count = SPARK_TOKENIZER_MAX_PARALLEL_WORKER_COUNT;
    }
    if (worker_count > text_count)
    {
        worker_count = text_count;
    }
    if (worker_count <= 1u)
    {
        SparkTokenizerBatchWorker worker;

        memset(&worker, 0, sizeof(worker));
        worker.tokenizer = tokenizer;
        worker.texts = texts;
        worker.text_bytes = text_bytes;
        worker.first_text_index = 0u;
        worker.text_count = text_count;
        worker.encode_flags = encode_flags;
        worker.token_ids = token_ids;
        worker.token_stride = token_stride;
        worker.token_counts = token_counts;
        worker.overflow_token_counts = overflow_token_counts;
        worker.invalid_segment_counts = invalid_segment_counts;
        (void)SparkTokenizerBatchWorkerMain(&worker);
        return worker.status;
    }

    workers = (SparkTokenizerBatchWorker *)calloc(worker_count, sizeof(*workers));
    threads = (pthread_t *)calloc(worker_count, sizeof(*threads));
    if (workers == 0 || threads == 0)
    {
        free(workers);
        free(threads);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    base_count = text_count / worker_count;
    remainder_count = text_count % worker_count;
    next_text_index = 0u;
    launched_worker_count = 0u;
    final_status = SPARK_STATUS_OK;
    for (worker_index = 0u; worker_index < worker_count; ++worker_index)
    {
        uint32_t assigned_count;

        assigned_count = base_count + (worker_index < remainder_count ? 1u : 0u);
        workers[worker_index].tokenizer = tokenizer;
        workers[worker_index].texts = texts;
        workers[worker_index].text_bytes = text_bytes;
        workers[worker_index].first_text_index = next_text_index;
        workers[worker_index].text_count = assigned_count;
        workers[worker_index].encode_flags = encode_flags;
        workers[worker_index].token_ids = token_ids;
        workers[worker_index].token_stride = token_stride;
        workers[worker_index].token_counts = token_counts;
        workers[worker_index].overflow_token_counts = overflow_token_counts;
        workers[worker_index].invalid_segment_counts = invalid_segment_counts;
        workers[worker_index].status = SPARK_STATUS_OK;
        if (pthread_create(&threads[worker_index], 0, SparkTokenizerBatchWorkerMain, &workers[worker_index]) != 0)
        {
            final_status = SPARK_STATUS_INTERNAL_ERROR;
            break;
        }
        launched_worker_count += 1u;
        next_text_index += assigned_count;
    }

    for (worker_index = 0u; worker_index < launched_worker_count; ++worker_index)
    {
        if (pthread_join(threads[worker_index], 0) != 0 && final_status == SPARK_STATUS_OK)
        {
            final_status = SPARK_STATUS_INTERNAL_ERROR;
        }
        if (workers[worker_index].status != SPARK_STATUS_OK && final_status == SPARK_STATUS_OK)
        {
            final_status = workers[worker_index].status;
        }
    }
    free(workers);
    free(threads);
    return final_status;
}

SparkStatus SparkTokenizerEncodeBatchUtf8(
    const SparkTokenizer *tokenizer,
    const char *const *texts,
    const uint32_t *text_bytes,
    uint32_t text_count,
    uint32_t encode_flags,
    uint32_t *token_ids,
    uint32_t token_stride,
    uint32_t *token_counts,
    uint32_t *overflow_token_counts)
{
    return SparkTokenizerEncodeBatchUtf8ConfiguredInternal(
        tokenizer,
        texts,
        text_bytes,
        text_count,
        encode_flags,
        token_ids,
        token_stride,
        token_counts,
        overflow_token_counts,
        0,
        text_count >= 2u ? SPARK_TOKENIZER_MAX_PARALLEL_WORKER_COUNT : 1u);
}


SparkStatus SparkTokenizerEncodeBatchUtf8Configured(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerBatchEncodeConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_BATCH_ENCODE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->reserved0 != 0u ||
        configuration->reserved1 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkTokenizerEncodeBatchUtf8ConfiguredInternal(
        tokenizer,
        configuration->texts,
        configuration->text_bytes,
        configuration->text_count,
        configuration->encode_flags,
        configuration->token_ids,
        configuration->token_stride,
        configuration->token_counts,
        configuration->overflow_token_counts,
        configuration->invalid_segment_counts,
        configuration->worker_count);
}

static const SparkTokenizerSpecialToken *SparkTokenizerFindAddedTokenById(
    const SparkTokenizer *tokenizer,
    uint32_t token_id)
{
    uint32_t special_index;

    if (tokenizer == 0)
    {
        return 0;
    }
    for (special_index = 0u;
         special_index < tokenizer->special_token_count;
         ++special_index)
    {
        if (tokenizer->special_tokens[special_index].token_id == token_id)
        {
            return &tokenizer->special_tokens[special_index];
        }
    }
    return 0;
}

static SparkStatus SparkTokenizerAppendRawTokenText(
    const char *token_text,
    uint32_t token_text_bytes,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_inout)
{
    if (token_text == 0 || text == 0 || text_bytes_inout == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((uint64_t)*text_bytes_inout + token_text_bytes + 1u > text_capacity)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    memcpy(text + *text_bytes_inout, token_text, token_text_bytes);
    *text_bytes_inout += token_text_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerDecodeOneTokenText(
    const char *token_text,
    uint32_t token_text_bytes,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_inout)
{
    uint32_t position;
    uint32_t code_point;
    uint8_t decoded_byte;

    if (token_text == 0 || text == 0 || text_bytes_inout == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    position = 0u;
    while (position < token_text_bytes)
    {
        if (SparkTokenizerReadUtf8CodePoint(
                token_text,
                token_text_bytes,
                &position,
                &code_point) == 0u)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (SparkTokenizerUnicodeCodePointToByte(
                code_point,
                &decoded_byte) == 0u)
        {
            return SPARK_STATUS_PARSE_ERROR;
        }
        if (*text_bytes_inout + 1u >= text_capacity)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        text[*text_bytes_inout] = (char)decoded_byte;
        *text_bytes_inout += 1u;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerDecodeTokenIds(
    const SparkTokenizer *tokenizer,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t decode_flags,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_out)
{
    uint32_t token_index;
    uint32_t text_bytes;
    SparkStatus status;

    if (text_bytes_out != 0)
    {
        *text_bytes_out = 0u;
    }
    if (tokenizer == 0 ||
        tokenizer->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        tokenizer->descriptor_bytes != SPARK_TOKENIZER_DESCRIPTOR_BYTES ||
        token_ids == 0 ||
        text == 0 ||
        text_capacity == 0u ||
        (decode_flags & ~SPARK_TOKENIZER_DECODE_KNOWN_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    text_bytes = 0u;
    text[0u] = '\0';
    for (token_index = 0u; token_index < token_count; ++token_index)
    {
        uint32_t token_id;
        const char *token_text;
        const SparkTokenizerSpecialToken *added_token;
        uint32_t token_text_bytes;

        token_id = token_ids[token_index];
        added_token = SparkTokenizerFindAddedTokenById(tokenizer, token_id);
        if ((decode_flags & SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS) != 0u &&
            added_token != 0 && added_token->is_special != 0u)
        {
            continue;
        }
        if (added_token != 0)
        {
            status = SparkTokenizerAppendRawTokenText(
                added_token->text,
                added_token->text_bytes,
                text,
                text_capacity,
                &text_bytes);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            continue;
        }
        if (token_id > tokenizer->maximum_token_id ||
            tokenizer->token_text_by_id == 0 ||
            tokenizer->token_text_bytes_by_id == 0 ||
            tokenizer->token_text_by_id[token_id] == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        token_text = tokenizer->token_text_by_id[token_id];
        token_text_bytes = tokenizer->token_text_bytes_by_id[token_id];
        status = SparkTokenizerDecodeOneTokenText(
            token_text,
            token_text_bytes,
            text,
            text_capacity,
            &text_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    text[text_bytes] = '\0';
    if (text_bytes_out != 0)
    {
        *text_bytes_out = text_bytes;
    }
    return SPARK_STATUS_OK;
}
