#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tokenizer.h"

#define SPARK_TEST_TOKEN_A 1u
#define SPARK_TEST_TOKEN_B 2u
#define SPARK_TEST_TOKEN_C 3u
#define SPARK_TEST_TOKEN_AB 4u
#define SPARK_TEST_TOKEN_ABC 5u
#define SPARK_TEST_TOKEN_UNKNOWN 6u
#define SPARK_TEST_TOKEN_STOP 7u
#define SPARK_TEST_TOKEN_SPACE 8u
#define SPARK_TEST_TOKEN_X 9u
#define SPARK_TEST_TOKEN_Y 10u
#define SPARK_TEST_TOKEN_Z 11u
#define SPARK_TEST_TOKEN_XY 12u
#define SPARK_TEST_TOKEN_XYZ 13u

static const char *SparkTestTokenizerJsonPath(void)
{
    return "build/test_tokenizer_hf_byte_bpe.json";
}

static void SparkTestTokenizerWriteFixtureJson(void)
{
    FILE *file;

    file = fopen(SparkTestTokenizerJsonPath(), "wb");
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": \"<unk>\",\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n"
        "      \"a\": %u,\n"
        "      \"b\": %u,\n"
        "      \"c\": %u,\n"
        "      \"ab\": %u,\n"
        "      \"abc\": %u,\n"
        "      \"<unk>\": %u,\n"
        "      \"<|stop|>\": %u,\n"
        "      \"\304\240\": %u,\n"
        "      \"x\": %u,\n"
        "      \"y\": %u,\n"
        "      \"z\": %u,\n"
        "      \"xy\": %u,\n"
        "      \"xyz\": %u\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"a b\",\n"
        "      \"ab c\",\n"
        "      \"x y\",\n"
        "      \"xy z\"\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"ByteLevel\",\n"
        "    \"add_prefix_space\": false\n"
        "  },\n"
        "  \"added_tokens\": [\n"
        "    {\"id\": %u, \"content\": \"<|stop|>\", \"special\": true}\n"
        "  ]\n"
        "}\n",
        SPARK_TEST_TOKEN_A,
        SPARK_TEST_TOKEN_B,
        SPARK_TEST_TOKEN_C,
        SPARK_TEST_TOKEN_AB,
        SPARK_TEST_TOKEN_ABC,
        SPARK_TEST_TOKEN_UNKNOWN,
        SPARK_TEST_TOKEN_STOP,
        SPARK_TEST_TOKEN_SPACE,
        SPARK_TEST_TOKEN_X,
        SPARK_TEST_TOKEN_Y,
        SPARK_TEST_TOKEN_Z,
        SPARK_TEST_TOKEN_XY,
        SPARK_TEST_TOKEN_XYZ,
        SPARK_TEST_TOKEN_STOP);
    assert(fclose(file) == 0);
}

static void SparkTestTokenizerLoadFixture(
    SparkTokenizer *tokenizer)
{
    SparkTokenizerHuggingFaceJsonConfiguration configuration;

    SparkTokenizerReset(tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = SparkTestTokenizerJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(tokenizer, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestTokenizerEncodesByteBpeAndSpecialTokens(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerEncoding encoding;
    uint32_t token_ids[16u];

    SparkTestTokenizerWriteFixtureJson();
    SparkTestTokenizerLoadFixture(&tokenizer);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "abc",
        3u,
        0u,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 1u);
    assert(encoding.overflow_token_count == 0u);
    assert(encoding.invalid_segment_count == 0u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ABC);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "a<|stop|>bc",
        11u,
        0u,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 4u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_STOP);
    assert(token_ids[2u] == SPARK_TEST_TOKEN_B);
    assert(token_ids[3u] == SPARK_TEST_TOKEN_C);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "abc",
        3u,
        SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 2u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_SPACE);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_ABC);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 2u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "a<|stop|>bc",
        11u,
        0u,
        &encoding) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(encoding.token_count == 2u);
    assert(encoding.overflow_token_count == 2u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_STOP);

    SparkTokenizerDestroy(&tokenizer);
}


static void SparkTestTokenizerDecodesByteLevelTokens(void)
{
    SparkTokenizer tokenizer;
    uint32_t token_ids[3u];
    char text[64u];
    uint32_t text_bytes;

    SparkTestTokenizerLoadFixture(&tokenizer);
    token_ids[0u] = SPARK_TEST_TOKEN_SPACE;
    token_ids[1u] = SPARK_TEST_TOKEN_ABC;
    token_ids[2u] = SPARK_TEST_TOKEN_STOP;
    assert(SparkTokenizerDecodeTokenIds(
        &tokenizer,
        token_ids,
        3u,
        SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        text,
        sizeof(text),
        &text_bytes) == SPARK_STATUS_OK);
    assert(text_bytes == 4u);
    assert(strcmp(text, " abc") == 0);

    assert(SparkTokenizerDecodeTokenIds(
        &tokenizer,
        token_ids,
        2u,
        0u,
        text,
        3u,
        &text_bytes) == SPARK_STATUS_CAPACITY_EXCEEDED);

    SparkTokenizerDestroy(&tokenizer);
}

static void SparkTestTokenizerEncodesBatch(void)
{
    SparkTokenizer tokenizer;
    const char *texts[2u];
    uint32_t text_bytes[2u];
    uint32_t token_ids[8u];
    uint32_t token_counts[2u];
    uint32_t overflow_counts[2u];

    SparkTestTokenizerLoadFixture(&tokenizer);
    memset(token_ids, 0, sizeof(token_ids));
    memset(token_counts, 0, sizeof(token_counts));
    memset(overflow_counts, 0, sizeof(overflow_counts));

    texts[0u] = "abc";
    text_bytes[0u] = 3u;
    texts[1u] = "xyz";
    text_bytes[1u] = 3u;
    assert(SparkTokenizerEncodeBatchUtf8(
        &tokenizer,
        texts,
        text_bytes,
        2u,
        0u,
        token_ids,
        4u,
        token_counts,
        overflow_counts) == SPARK_STATUS_OK);
    assert(token_counts[0u] == 1u);
    assert(token_counts[1u] == 1u);
    assert(overflow_counts[0u] == 0u);
    assert(overflow_counts[1u] == 0u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ABC);
    assert(token_ids[4u] == SPARK_TEST_TOKEN_XYZ);

    SparkTokenizerDestroy(&tokenizer);
}


static void SparkTestTokenizerCompiledFileAndConfiguredBatch(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizer loaded_tokenizer;
    SparkTokenizerCompiledFileConfiguration compiled_configuration;
    SparkTokenizerBatchEncodeConfiguration batch_configuration;
    const char *texts[3u];
    uint32_t text_bytes[3u];
    uint32_t token_ids[12u];
    uint32_t token_counts[3u];
    uint32_t overflow_counts[3u];
    uint32_t invalid_counts[3u];

    SparkTestTokenizerLoadFixture(&tokenizer);
    SparkTokenizerReset(&loaded_tokenizer);
    memset(&compiled_configuration, 0, sizeof(compiled_configuration));
    compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    compiled_configuration.descriptor_bytes =
        SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
    compiled_configuration.compiled_tokenizer_path = "build/test_tokenizer.compiled";
    assert(SparkTokenizerSaveCompiledFile(&tokenizer, &compiled_configuration) ==
        SPARK_STATUS_OK);
    assert(SparkTokenizerLoadCompiledFile(&loaded_tokenizer, &compiled_configuration) ==
        SPARK_STATUS_OK);

    texts[0u] = "abc";
    text_bytes[0u] = 3u;
    texts[1u] = "xyz";
    text_bytes[1u] = 3u;
    texts[2u] = "a<|stop|>bc";
    text_bytes[2u] = 11u;
    memset(token_ids, 0, sizeof(token_ids));
    memset(token_counts, 0, sizeof(token_counts));
    memset(overflow_counts, 0, sizeof(overflow_counts));
    memset(invalid_counts, 0, sizeof(invalid_counts));
    memset(&batch_configuration, 0, sizeof(batch_configuration));
    batch_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    batch_configuration.descriptor_bytes =
        SPARK_TOKENIZER_BATCH_ENCODE_CONFIGURATION_DESCRIPTOR_BYTES;
    batch_configuration.texts = texts;
    batch_configuration.text_bytes = text_bytes;
    batch_configuration.text_count = 3u;
    batch_configuration.token_ids = token_ids;
    batch_configuration.token_stride = 4u;
    batch_configuration.token_counts = token_counts;
    batch_configuration.overflow_token_counts = overflow_counts;
    batch_configuration.invalid_segment_counts = invalid_counts;
    batch_configuration.worker_count = 2u;
    assert(SparkTokenizerEncodeBatchUtf8Configured(
        &loaded_tokenizer,
        &batch_configuration) == SPARK_STATUS_OK);
    assert(token_counts[0u] == 1u);
    assert(token_counts[1u] == 1u);
    assert(token_counts[2u] == 4u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ABC);
    assert(token_ids[4u] == SPARK_TEST_TOKEN_XYZ);
    assert(token_ids[8u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[9u] == SPARK_TEST_TOKEN_STOP);
    assert(token_ids[10u] == SPARK_TEST_TOKEN_B);
    assert(token_ids[11u] == SPARK_TEST_TOKEN_C);
    assert(invalid_counts[0u] == 0u);
    assert(invalid_counts[1u] == 0u);
    assert(invalid_counts[2u] == 0u);

    SparkTokenizerDestroy(&loaded_tokenizer);
    SparkTokenizerDestroy(&tokenizer);
}


static const char *SparkTestTokenizerLargeMergeJsonPath(void)
{
    return "build/test_tokenizer_large_merge_hf_byte_bpe.json";
}

static void SparkTestTokenizerWriteLargeMergeFixtureJson(
    uint32_t merge_count)
{
    FILE *file;
    uint32_t merge_index;

    file = fopen(SparkTestTokenizerLargeMergeJsonPath(), "wb");
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": \"<unk>\",\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n"
        "      \"a\": %u,\n"
        "      \"b\": %u,\n"
        "      \"ab\": %u,\n"
        "      \"<unk>\": %u\n"
        "    },\n"
        "    \"merges\": [\n",
        SPARK_TEST_TOKEN_A,
        SPARK_TEST_TOKEN_B,
        SPARK_TEST_TOKEN_AB,
        SPARK_TEST_TOKEN_UNKNOWN);
    fprintf(file, "      \"a b\"");
    for (merge_index = 1u; merge_index < merge_count; ++merge_index)
    {
        fprintf(file,
            ",\n      \"unused_left_%06u unused_right_%06u\"",
            merge_index,
            merge_index);
    }
    fprintf(file,
        "\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"ByteLevel\",\n"
        "    \"add_prefix_space\": false\n"
        "  }\n"
        "}\n");
    assert(fclose(file) == 0);
}

static void SparkTestTokenizerLoadsLargeMergeArrayWithoutIndexedArrayWalk(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizerEncoding encoding;
    uint32_t token_ids[4u];
    uint32_t token_id;

    SparkTestTokenizerWriteLargeMergeFixtureJson(8192u);
    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = SparkTestTokenizerLargeMergeJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration) ==
        SPARK_STATUS_OK);
    assert(tokenizer.merge_count == 8192u);
    assert(SparkTokenizerFindTokenId(&tokenizer, "ab", 2u, &token_id) ==
        SPARK_STATUS_OK);
    assert(token_id == SPARK_TEST_TOKEN_AB);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 4u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "ab",
        2u,
        0u,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 1u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_AB);

    SparkTokenizerDestroy(&tokenizer);
}

int main(void)
{
    SparkTestTokenizerEncodesByteBpeAndSpecialTokens();
    SparkTestTokenizerDecodesByteLevelTokens();
    SparkTestTokenizerEncodesBatch();
    SparkTestTokenizerCompiledFileAndConfiguredBatch();
    SparkTestTokenizerLoadsLargeMergeArrayWithoutIndexedArrayWalk();
    return 0;
}
