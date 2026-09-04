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
#define SPARK_TEST_TOKEN_ROLE 14u
#define SPARK_TEST_TOKEN_ROLE_TEXT "<\357\275\234role\357\275\234>"

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
        "    {\"id\": %u, \"content\": \"<|stop|>\", \"special\": true},\n"
        "    {\"id\": %u, \"content\": \"" SPARK_TEST_TOKEN_ROLE_TEXT "\", \"special\": false}\n"
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
        SPARK_TEST_TOKEN_STOP,
        SPARK_TEST_TOKEN_ROLE);
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
    uint32_t token_index;

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
        "a<|stop|>bc",
        11u,
        SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH,
        &encoding) == SPARK_STATUS_OK);
    for (token_index = 0u; token_index < encoding.token_count; ++token_index)
    {
        assert(token_ids[token_index] != SPARK_TEST_TOKEN_STOP);
    }

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(
        &tokenizer,
        "a" SPARK_TEST_TOKEN_ROLE_TEXT "bc",
        (uint32_t)sizeof("a" SPARK_TEST_TOKEN_ROLE_TEXT "bc") - 1u,
        SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH,
        &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 4u);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_A);
    assert(token_ids[1u] == SPARK_TEST_TOKEN_ROLE);
    assert(token_ids[2u] == SPARK_TEST_TOKEN_B);
    assert(token_ids[3u] == SPARK_TEST_TOKEN_C);
    assert(SparkTokenizerFindTokenId(
        &tokenizer,
        SPARK_TEST_TOKEN_ROLE_TEXT,
        (uint32_t)sizeof(SPARK_TEST_TOKEN_ROLE_TEXT) - 1u,
        &token_ids[0u]) == SPARK_STATUS_OK);
    assert(token_ids[0u] == SPARK_TEST_TOKEN_ROLE);

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
    uint32_t token_ids[4u];
    char text[64u];
    uint32_t text_bytes;

    SparkTestTokenizerLoadFixture(&tokenizer);
    token_ids[0u] = SPARK_TEST_TOKEN_SPACE;
    token_ids[1u] = SPARK_TEST_TOKEN_ABC;
    token_ids[2u] = SPARK_TEST_TOKEN_STOP;
    token_ids[3u] = SPARK_TEST_TOKEN_ROLE;
    assert(SparkTokenizerDecodeTokenIds(
        &tokenizer,
        token_ids,
        4u,
        SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        text,
        sizeof(text),
        &text_bytes) == SPARK_STATUS_OK);
    assert(text_bytes == 4u + sizeof(SPARK_TEST_TOKEN_ROLE_TEXT) - 1u);
    assert(strcmp(text, " abc" SPARK_TEST_TOKEN_ROLE_TEXT) == 0);

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
    uint32_t decode_ids[2u];
    uint32_t decoded_bytes;
    char decoded[32u];

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
    texts[2u] = "a" SPARK_TEST_TOKEN_ROLE_TEXT "bc";
    text_bytes[2u] = (uint32_t)sizeof("a" SPARK_TEST_TOKEN_ROLE_TEXT "bc") - 1u;
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
    assert(token_ids[9u] == SPARK_TEST_TOKEN_ROLE);
    assert(token_ids[10u] == SPARK_TEST_TOKEN_B);
    assert(token_ids[11u] == SPARK_TEST_TOKEN_C);
    assert(invalid_counts[0u] == 0u);
    assert(invalid_counts[1u] == 0u);
    assert(invalid_counts[2u] == 0u);
    decode_ids[0u] = SPARK_TEST_TOKEN_STOP;
    decode_ids[1u] = SPARK_TEST_TOKEN_ROLE;
    assert(SparkTokenizerDecodeTokenIds(
        &loaded_tokenizer,
        decode_ids,
        2u,
        SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        decoded,
        sizeof(decoded),
        &decoded_bytes) == SPARK_STATUS_OK);
    assert(decoded_bytes == sizeof(SPARK_TEST_TOKEN_ROLE_TEXT) - 1u);
    assert(strcmp(decoded, SPARK_TEST_TOKEN_ROLE_TEXT) == 0);

    SparkTokenizerDestroy(&loaded_tokenizer);
    SparkTokenizerDestroy(&tokenizer);
}


static const char *SparkTestTokenizerLegacyCompiledPath(void)
{
    return "build/test_tokenizer_legacy_v1.compiled";
}

static void SparkTestTokenizerWriteLegacyCompiledFile(
    uint32_t added_token_text_bytes)
{
    static const char special_text[] = "<|stop|>";
    const uint64_t magic = SPARK_TOKENIZER_COMPILED_FILE_MAGIC;
    const uint32_t version = SPARK_TOKENIZER_COMPILED_FILE_LEGACY_VERSION;
    const uint32_t header[] = {
        SPARK_TOKENIZER_BPE_MODEL_KIND_BYTE_LEVEL,
        0u,
        0u,
        0u,
        0u,
        SPARK_TEST_TOKEN_STOP,
        1u,
        1u,
        0u,
        1u};
    const uint32_t token_id = SPARK_TEST_TOKEN_STOP;
    const uint32_t text_bytes = (uint32_t)sizeof(special_text) - 1u;
    FILE *file;

    file = fopen(SparkTestTokenizerLegacyCompiledPath(), "wb");
    assert(file != 0);
    assert(fwrite(&magic, sizeof(magic), 1u, file) == 1u);
    assert(fwrite(&version, sizeof(version), 1u, file) == 1u);
    assert(fwrite(header, sizeof(header), 1u, file) == 1u);
    assert(fwrite(&token_id, sizeof(token_id), 1u, file) == 1u);
    assert(fwrite(&text_bytes, sizeof(text_bytes), 1u, file) == 1u);
    assert(fwrite(special_text, text_bytes, 1u, file) == 1u);
    assert(fwrite(&token_id, sizeof(token_id), 1u, file) == 1u);
    assert(fwrite(&added_token_text_bytes, sizeof(added_token_text_bytes), 1u, file) == 1u);
    if (added_token_text_bytes != 0u)
    {
        assert(fwrite(special_text, added_token_text_bytes, 1u, file) == 1u);
    }
    assert(fclose(file) == 0);
}

static void SparkTestTokenizerLoadsLegacyCompiledFile(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerCompiledFileConfiguration configuration;
    uint32_t token_id;
    uint32_t text_bytes;
    char text[16u];

    SparkTestTokenizerWriteLegacyCompiledFile(
        (uint32_t)sizeof("<|stop|>") - 1u);
    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.compiled_tokenizer_path = SparkTestTokenizerLegacyCompiledPath();
    assert(SparkTokenizerLoadCompiledFile(&tokenizer, &configuration) ==
        SPARK_STATUS_OK);
    assert(tokenizer.special_token_count == 1u);
    assert(tokenizer.special_tokens[0u].is_special == 1u);
    token_id = SPARK_TEST_TOKEN_STOP;
    assert(SparkTokenizerDecodeTokenIds(
        &tokenizer,
        &token_id,
        1u,
        SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        text,
        sizeof(text),
        &text_bytes) == SPARK_STATUS_OK);
    assert(text_bytes == 0u);
    assert(strcmp(text, "") == 0);
    SparkTokenizerDestroy(&tokenizer);
}


static void SparkTestTokenizerRejectsEmptyAddedTokens(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerHuggingFaceJsonConfiguration json_configuration;
    SparkTokenizerCompiledFileConfiguration compiled_configuration;
    FILE *file;

    file = fopen(SparkTestTokenizerJsonPath(), "wb");
    assert(file != 0);
    assert(fputs(
        "{\"model\":{\"type\":\"BPE\",\"unk_token\":\"a\","
        "\"byte_fallback\":false,\"vocab\":{\"a\":1},\"merges\":[]},"
        "\"pre_tokenizer\":{\"type\":\"ByteLevel\"},\"added_tokens\":["
        "{\"id\":2,\"content\":\"\",\"special\":false}]}",
        file) >= 0);
    assert(fclose(file) == 0);
    SparkTokenizerReset(&tokenizer);
    memset(&json_configuration, 0, sizeof(json_configuration));
    json_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    json_configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    json_configuration.tokenizer_json_path = SparkTestTokenizerJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(&tokenizer, &json_configuration) ==
        SPARK_STATUS_SCHEMA_ERROR);
    SparkTestTokenizerWriteLegacyCompiledFile(0u);
    memset(&compiled_configuration, 0, sizeof(compiled_configuration));
    compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    compiled_configuration.descriptor_bytes =
        SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
    compiled_configuration.compiled_tokenizer_path =
        SparkTestTokenizerLegacyCompiledPath();
    assert(SparkTokenizerLoadCompiledFile(&tokenizer, &compiled_configuration) ==
        SPARK_STATUS_SCHEMA_ERROR);
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

static void SparkTestTokenizerWarmCacheSurvivesGrowthAndMatches(void)
{
    static const char base[] =
        "the quick brown fox jumps over the lazy dog and the cat sat on the mat "
        "she said don't and he said can't while they walked to the market today "
        "numbers 123 456 and symbols !!! ... mixed with words words words again";
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizer tokenizer;
    SparkTokenizerWorkspace workspace;
    uint32_t sizes[6];
    uint32_t base_bytes = (uint32_t)(sizeof(base) - 1u);
    uint32_t request_index;
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.tokenizer_json_path = "build/test_tokenizer_large_merge_hf_byte_bpe.json";
    if (SparkTokenizerLoadHuggingFaceJson(&tokenizer,&configuration) != SPARK_STATUS_OK)
        return;
    SparkTokenizerWorkspaceReset(&workspace);
    assert(SparkTokenizerWorkspaceInitialize(&workspace,16u) == SPARK_STATUS_OK);
    sizes[0] = 20u;
    sizes[1] = 20u;
    sizes[2] = base_bytes;
    sizes[3] = 20u;
    sizes[4] = base_bytes;
    sizes[5] = 30u;
    for (request_index = 0u; request_index < 6u; ++request_index)
    {
        SparkTokenizerEncoding warm_encoding;
        SparkTokenizerEncoding cold_encoding;
        uint32_t warm_ids[512];
        uint32_t cold_ids[512];
        uint32_t request_bytes = sizes[request_index];
        uint32_t token_index;
        if (request_bytes > base_bytes)
            request_bytes = base_bytes;
        memset(&warm_encoding,0,sizeof(warm_encoding));
        memset(&cold_encoding,0,sizeof(cold_encoding));
        warm_encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        warm_encoding.descriptor_bytes = sizeof(warm_encoding);
        warm_encoding.token_ids = warm_ids;
        warm_encoding.token_capacity = 512u;
        cold_encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        cold_encoding.descriptor_bytes = sizeof(cold_encoding);
        cold_encoding.token_ids = cold_ids;
        cold_encoding.token_capacity = 512u;
        assert(SparkTokenizerEncodeUtf8WithWorkspace(&tokenizer,base,request_bytes,0u,&workspace,&warm_encoding) == SPARK_STATUS_OK);
        assert(SparkTokenizerEncodeUtf8(&tokenizer,base,request_bytes,0u,&cold_encoding) == SPARK_STATUS_OK);
        assert(warm_encoding.token_count == cold_encoding.token_count);
        for (token_index = 0u; token_index < warm_encoding.token_count; ++token_index)
            assert(warm_ids[token_index] == cold_ids[token_index]);
    }
    SparkTokenizerWorkspaceDestroy(&workspace);
    SparkTokenizerDestroy(&tokenizer);
}

static void SparkTestTokenizerPieceCacheMatchesUncached(void)
{
    static const char corpus[] =
        "the quick brown fox the quick brown fox jumps over the lazy dog "
        "the the the quick quick brown fox jumps jumps over over the lazy lazy "
        "hello world hello world foo bar foo bar baz qux the quick brown fox";
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    SparkTokenizer tokenizer;
    SparkTokenizerEncoding cached_encoding;
    SparkTokenizerEncoding uncached_encoding;
    uint32_t cached_ids[512];
    uint32_t uncached_ids[512];
    uint32_t corpus_bytes = (uint32_t)(sizeof(corpus) - 1u);
    uint32_t token_index;
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.tokenizer_json_path = "build/test_tokenizer_large_merge_hf_byte_bpe.json";
    if (SparkTokenizerLoadHuggingFaceJson(&tokenizer,&configuration) != SPARK_STATUS_OK)
        return;
    memset(&cached_encoding,0,sizeof(cached_encoding));
    memset(&uncached_encoding,0,sizeof(uncached_encoding));
    cached_encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    cached_encoding.descriptor_bytes = sizeof(cached_encoding);
    cached_encoding.token_ids = cached_ids;
    cached_encoding.token_capacity = 512u;
    uncached_encoding.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    uncached_encoding.descriptor_bytes = sizeof(uncached_encoding);
    uncached_encoding.token_ids = uncached_ids;
    uncached_encoding.token_capacity = 512u;
    assert(SparkTokenizerEncodeUtf8(&tokenizer,corpus,corpus_bytes,0u,&cached_encoding) == SPARK_STATUS_OK);
    assert(SparkTokenizerEncodeUtf8(&tokenizer,corpus,corpus_bytes,SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_PIECE_CACHE,&uncached_encoding) == SPARK_STATUS_OK);
    assert(cached_encoding.token_count == uncached_encoding.token_count);
    for (token_index = 0u; token_index < cached_encoding.token_count; ++token_index)
        assert(cached_ids[token_index] == uncached_ids[token_index]);
    SparkTokenizerDestroy(&tokenizer);
}

#define SPARK_TEST_QWEN_TOKEN_UNKNOWN 0u
#define SPARK_TEST_QWEN_TOKEN_1 10u
#define SPARK_TEST_QWEN_TOKEN_2 11u
#define SPARK_TEST_QWEN_TOKEN_3 12u
#define SPARK_TEST_QWEN_TOKEN_12 13u
#define SPARK_TEST_QWEN_TOKEN_123 14u
#define SPARK_TEST_QWEN_TOKEN_LBRACKET 20u
#define SPARK_TEST_QWEN_TOKEN_A 21u
#define SPARK_TEST_QWEN_TOKEN_B 22u
#define SPARK_TEST_QWEN_TOKEN_AB 23u
#define SPARK_TEST_QWEN_TOKEN_LBRACKET_AB 24u

static const char *SparkTestQwenTokenizerJsonPath(void)
{
    return "build/test_tokenizer_qwen_byte_bpe.json";
}

static void SparkTestQwenTokenizerWriteFixtureJson(void)
{
    FILE *file;

    file = fopen(SparkTestQwenTokenizerJsonPath(), "wb");
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"unk_token\": \"<unk>\",\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n"
        "      \"<unk>\": %u,\n"
        "      \"1\": %u,\n"
        "      \"2\": %u,\n"
        "      \"3\": %u,\n"
        "      \"12\": %u,\n"
        "      \"123\": %u,\n"
        "      \"[\": %u,\n"
        "      \"a\": %u,\n"
        "      \"b\": %u,\n"
        "      \"ab\": %u,\n"
        "      \"[ab\": %u\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"1 2\",\n"
        "      \"12 3\",\n"
        "      \"a b\",\n"
        "      \"[ ab\"\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"Sequence\",\n"
        "    \"pretokenizers\": [\n"
        "      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"}, \"behavior\": \"Isolated\", \"invert\": false},\n"
        "      {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": false, \"use_regex\": false}\n"
        "    ]\n"
        "  },\n"
        "  \"added_tokens\": [\n"
        "  ]\n"
        "}\n",
        SPARK_TEST_QWEN_TOKEN_UNKNOWN,
        SPARK_TEST_QWEN_TOKEN_1,
        SPARK_TEST_QWEN_TOKEN_2,
        SPARK_TEST_QWEN_TOKEN_3,
        SPARK_TEST_QWEN_TOKEN_12,
        SPARK_TEST_QWEN_TOKEN_123,
        SPARK_TEST_QWEN_TOKEN_LBRACKET,
        SPARK_TEST_QWEN_TOKEN_A,
        SPARK_TEST_QWEN_TOKEN_B,
        SPARK_TEST_QWEN_TOKEN_AB,
        SPARK_TEST_QWEN_TOKEN_LBRACKET_AB);
    assert(fclose(file) == 0);
}

static void SparkTestQwenTokenizerPretokenizesWithQwenSemantics(void)
{
    SparkTokenizer tokenizer;
    SparkTokenizerEncoding encoding;
    SparkTokenizerHuggingFaceJsonConfiguration configuration;
    uint32_t token_ids[16u];

    SparkTestQwenTokenizerWriteFixtureJson();
    SparkTokenizerReset(&tokenizer);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.tokenizer_json_path = SparkTestQwenTokenizerJsonPath();
    assert(SparkTokenizerLoadHuggingFaceJson(&tokenizer, &configuration) ==
        SPARK_STATUS_OK);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(&tokenizer, "123", 3u, 0u, &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 3u);
    assert(token_ids[0u] == SPARK_TEST_QWEN_TOKEN_1);
    assert(token_ids[1u] == SPARK_TEST_QWEN_TOKEN_2);
    assert(token_ids[2u] == SPARK_TEST_QWEN_TOKEN_3);

    memset(token_ids, 0, sizeof(token_ids));
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = 16u;
    encoding.token_ids = token_ids;
    assert(SparkTokenizerEncodeUtf8(&tokenizer, "[ab", 3u, 0u, &encoding) == SPARK_STATUS_OK);
    assert(encoding.token_count == 1u);
    assert(token_ids[0u] == SPARK_TEST_QWEN_TOKEN_LBRACKET_AB);

    SparkTokenizerDestroy(&tokenizer);
}

int main(void)
{
    SparkTestTokenizerPieceCacheMatchesUncached();
    SparkTestTokenizerWarmCacheSurvivesGrowthAndMatches();
    SparkTestTokenizerEncodesByteBpeAndSpecialTokens();
    SparkTestTokenizerDecodesByteLevelTokens();
    SparkTestTokenizerEncodesBatch();
    SparkTestTokenizerCompiledFileAndConfiguredBatch();
    SparkTestTokenizerLoadsLegacyCompiledFile();
    SparkTestTokenizerRejectsEmptyAddedTokens();
    SparkTestTokenizerLoadsLargeMergeArrayWithoutIndexedArrayWalk();
    SparkTestQwenTokenizerPretokenizesWithQwenSemantics();
    return 0;
}
