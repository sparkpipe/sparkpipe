/* Host proof for the tokenizer sidecar (Phase 4, text-in/text-out).
 *
 * Coverage:
 *   1. Asset-format AUTO detection: HuggingFace tokenizer.json, the compiled
 *      format, tiktoken ranks.
 *   2. Encode/decode round trips: empty, ASCII, whitespace runs, digits,
 *      contractions, unicode (CJK, emoji, combining marks), special tokens.
 *   3. The digit-runs split variant (\p{N}{1,3}) vs the single-digit variant.
 *   4. Stop-token-aware decode: per-request stops survive the text edge.
 *   5. THE GROUND TRUTH: the 92 ds4_eval cases carry text+ids pairs
 *      (rendered prompts joined with the pre-tokenized fixture ids, generated
 *      with real tiktoken). Encode(text)==ids and decode(ids)==text for all
 *      92 against the real committed tokenizer asset. Skips with a notice
 *      when the asset is absent (same notice contract as hardware gates).
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_tokenizer_sidecar.h"

#define HF_DIGIT_RUNS_PATH "build/test_tokenizer_sidecar_hf_digit_runs.json"
#define HF_SINGLE_DIGIT_PATH "build/test_tokenizer_sidecar_hf_single_digit.json"
#define RANKS_PATH "build/test_tokenizer_sidecar_ranks.model"
#define COMPILED_PATH "build/test_tokenizer_sidecar_compiled.tok"

/* The real family assets: committed beside the ds4_eval fixtures they
 * ground-truth. */
#define GT_TOKENIZER_PATH "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"
#define GT_FIXTURE_PATH "qualification/ds4_eval/quality-fixtures-glm5.3-flash.json"
#define GT_CASES_PATH "qualification/ds4_eval/runs/kimi-k3-api-20260728/cases.json"
#define K3_RANKS_PATH "qualification/ds4_eval/tokenizer/kimi-k3-tiktoken.model"

static void WriteFileOrDie(const char *path, const char *content)
{
    FILE *file = fopen(path, "wb");
    assert(file != 0);
    assert(fputs(content, file) != EOF);
    assert(fclose(file) == 0);
}

/* A tiny byte-level BPE in the real vocab convention: every byte's glyph is
 * a vocabulary entry (id = byte value), then multi-byte pieces at 300+:
 *  300 "Ġa" 301 "ab" 302 "Ġab" 303 "12" 304 "<|stop|>"(special) 305 "[role]".
 * Merges: "Ġ a"->300, "a b"->301, "1 2"->303. ignore_merges is selectable.
 * The digit pattern is a parameter so one fixture exercises each extended
 * splitter variant. */

/* The GPT-2 byte<->glyph mapping: printable bytes map to themselves, every
 * other byte takes the next code point from 256 upward. */
static uint32_t ByteGlyphCodePoint(uint8_t byte)
{
    uint32_t shifted;
    uint8_t candidate;
    if ((byte >= '!' && byte <= '~') ||
        (byte >= 0xa1 && byte <= 0xac) ||
        (byte >= 0xae))
    {
        return (uint32_t)byte;
    }
    shifted = 0u;
    for (candidate = 0u; ; candidate++)
    {
        if (!((candidate >= '!' && candidate <= '~') ||
                (candidate >= 0xa1 && candidate <= 0xac) ||
                (candidate >= 0xae)))
        {
            if (candidate == byte)
                return 256u + shifted;
            shifted++;
        }
        if (candidate == 255u)
            break;
    }
    return 0u;
}

static void AppendUtf8(FILE *file, uint32_t code_point)
{
    if (code_point < 0x80u)
        fputc((int)code_point, file);
    else if (code_point < 0x800u)
    {
        fputc((int)(0xc0u | (code_point >> 6)), file);
        fputc((int)(0x80u | (code_point & 0x3fu)), file);
    }
    else
    {
        fputc((int)(0xe0u | (code_point >> 12)), file);
        fputc((int)(0x80u | ((code_point >> 6) & 0x3fu)), file);
        fputc((int)(0x80u | (code_point & 0x3fu)), file);
    }
}

static void WriteHuggingFaceFixture(const char *path, uint32_t ignore_merges,
    const char *digit_pattern)
{
    FILE *file = fopen(path, "wb");
    uint32_t byte_value;
    assert(file != 0);
    fprintf(file,
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"ignore_merges\": %s,\n"
        "    \"byte_fallback\": false,\n"
        "    \"vocab\": {\n",
        ignore_merges ? "true" : "false");
    for (byte_value = 0u; byte_value < 256u; byte_value++)
    {
        uint32_t code_point = ByteGlyphCodePoint((uint8_t)byte_value);
        fprintf(file, "      ");
        fputc('"', file);
        if (code_point == (uint32_t)'"')
            fputs("\\\"", file);
        else if (code_point == (uint32_t)'\\')
            fputs("\\\\", file);
        else
            AppendUtf8(file, code_point);
        fprintf(file, "\": %u,\n", byte_value);
    }
    fprintf(file,
        "      \"\304\240a\": 300,\n"
        "      \"ab\": 301,\n"
        "      \"\304\240ab\": 302,\n"
        "      \"12\": 303\n"
        "    },\n"
        "    \"merges\": [\n"
        "      \"\304\240 a\",\n"
        "      \"a b\",\n"
        "      \"1 2\"\n"
        "    ]\n"
        "  },\n"
        "  \"pre_tokenizer\": {\n"
        "    \"type\": \"Sequence\",\n"
        "    \"pretokenizers\": [\n"
        "      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"%s\"}, \"behavior\": \"Isolated\", \"invert\": false},\n"
        "      {\"type\": \"ByteLevel\", \"add_prefix_space\": false}\n"
        "    ]\n"
        "  },\n"
        "  \"added_tokens\": [\n"
        "    {\"id\": 304, \"content\": \"<|stop|>\", \"special\": true},\n"
        "    {\"id\": 305, \"content\": \"[role]\", \"special\": false}\n"
        "  ]\n"
        "}\n",
        digit_pattern);
    assert(fclose(file) == 0);
}

/* Exact GLM family pattern variant strings (the loader recognizes these two
 * and only these two as extended splitters), JSON-escaped: each regex
 * backslash is a doubled backslash in the file so the decoded member value
 * carries the literal backslash sequences the loader compares. */
#define DIGIT_RUNS_PATTERN \
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"
#define SINGLE_DIGIT_PATTERN \
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"

/* Ranks fixture: pieces are raw bytes, base64; merge priority = rank id.
 *   "!"=0x21 ->5, " "->0x20 ->4, "a"->6, "b"->7, "h"->8, "ab"->1, " ab"->2,
 *   "ah"->3, " bah"->9
 * Encoding " ab!": bytes ' '(4) 'a'(6) 'b'(7) '!'(5): pair (a,b)->"ab"=1 is
 * the lowest, merge; then (" ",ab)->" ab"=2, merge; "!" stays. Ids [2,5]. */
static void WriteRanksFixture(void)
{
    WriteFileOrDie(RANKS_PATH,
        "IQ== 5\n"
        "IA== 4\n"
        "YQ== 6\n"
        "Yg== 7\n"
        "aA== 8\n"
        "YWI= 1\n"
        "IGFi 2\n"
        "YWg= 3\n"
        "IGJhaA== 9\n");
}

static void LoadSidecarAuto(SparkTokenizerSidecar *sidecar, const char *path)
{
    SparkTokenizerSidecarConfiguration configuration;
    SparkTokenizerSidecarReset(sidecar);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.asset_path = path;
    configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
    assert(SparkTokenizerSidecarLoad(sidecar, &configuration) == SPARK_STATUS_OK);
}

static void EncodeText(const SparkTokenizerSidecar *sidecar, const char *text,
    uint32_t *ids, uint32_t capacity, uint32_t *count_out)
{
    SparkTokenizerWorkspace workspace;
    SparkTokenizerEncoding encoding;
    SparkTokenizerWorkspaceReset(&workspace);
    assert(SparkTokenizerWorkspaceInitialize(&workspace,
        (uint32_t)strlen(text) + 2u) == SPARK_STATUS_OK);
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = capacity;
    encoding.token_ids = ids;
    assert(SparkTokenizerSidecarEncodeText(sidecar, text,
        (uint32_t)strlen(text), 0u, &workspace, &encoding) == SPARK_STATUS_OK);
    assert(encoding.overflow_token_count == 0u);
    *count_out = encoding.token_count;
    SparkTokenizerWorkspaceDestroy(&workspace);
}

static void DecodeIds(const SparkTokenizerSidecar *sidecar, const uint32_t *ids,
    uint32_t count, const uint32_t *stops, uint32_t stop_count,
    uint32_t decode_flags, char *text, uint32_t capacity, uint32_t *bytes_out)
{
    assert(SparkTokenizerSidecarDecodeText(sidecar, ids, count, stops,
        stop_count, decode_flags, text, capacity, bytes_out) == SPARK_STATUS_OK);
}

static void AssertRoundTrip(const SparkTokenizerSidecar *sidecar, const char *text)
{
    uint32_t ids[512];
    uint32_t count = 0;
    char decoded[4096];
    uint32_t decoded_bytes = 0;
    EncodeText(sidecar, text, ids, 512u, &count);
    DecodeIds(sidecar, ids, count, 0, 0, 0u, decoded, sizeof(decoded), &decoded_bytes);
    assert(decoded_bytes == strlen(text));
    assert(memcmp(decoded, text, decoded_bytes) == 0);
}

static void TestFormatDetectionAndRoundTrips(void)
{
    SparkTokenizerSidecar sidecar;
    uint32_t ids[64];
    uint32_t count = 0;
    char decoded[1024];
    uint32_t decoded_bytes = 0;
    uint32_t stop_tokens[1];

    /* HF json via AUTO ('{' prefix), digit-runs variant, ignore_merges on. */
    WriteHuggingFaceFixture(HF_DIGIT_RUNS_PATH, 1u, DIGIT_RUNS_PATTERN);
    LoadSidecarAuto(&sidecar, HF_DIGIT_RUNS_PATH);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON);
    assert(sidecar.tokenizer.ignore_merges == 1u);

    /* ignore_merges: a piece wholly in the vocabulary is one token. */
    EncodeText(&sidecar, " ab", ids, 64u, &count);
    assert(count == 1u && ids[0] == 302u);

    /* Digit runs: "123" is ONE pretoken piece under \p{N}{1,3}; the piece
     * BPEs to the "12" merge plus the trailing digit. */
    EncodeText(&sidecar, "x123", ids, 64u, &count);
    assert(count == 3u);
    assert(ids[0] == (uint32_t)'x' && ids[1] == 303u && ids[2] == (uint32_t)'3');

    /* Special tokens ride the encode side as single ids. */
    EncodeText(&sidecar, "<|stop|>", ids, 64u, &count);
    assert(count == 1u && ids[0] == 304u);

    /* Round trips incl. the added non-special token and unicode. */
    AssertRoundTrip(&sidecar, " ab!");
    AssertRoundTrip(&sidecar, "[role]x");
    AssertRoundTrip(&sidecar, "caf\xc3\xa9 na\xc3\xafve \xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x8c\x8d");
    AssertRoundTrip(&sidecar, "tabs\tand\nnewlines\r\n\r\n");
    AssertRoundTrip(&sidecar, "contractions don't we'll I've");
    SparkTokenizerSidecarUnload(&sidecar);

    /* Single-digit variant: "123" pretokenizes as three pieces, and pieces
     * never merge across boundaries: one more token than the digit-runs
     * variant on the same text. */
    WriteHuggingFaceFixture(HF_SINGLE_DIGIT_PATH, 0u, SINGLE_DIGIT_PATTERN);
    LoadSidecarAuto(&sidecar, HF_SINGLE_DIGIT_PATH);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON);
    EncodeText(&sidecar, "x123", ids, 64u, &count);
    assert(count == 4u);
    assert(ids[0] == (uint32_t)'x' && ids[1] == (uint32_t)'1' &&
        ids[2] == (uint32_t)'2' && ids[3] == (uint32_t)'3');
    SparkTokenizerSidecarUnload(&sidecar);

    /* The compiled format: v2 save/load keeps plain-BPE behavior; AUTO
     * detects it by magic. */
    {
        SparkTokenizer tokenizer;
        SparkTokenizerHuggingFaceJsonConfiguration load_configuration;
        SparkTokenizerCompiledFileConfiguration compiled_configuration;
        SparkTokenizerReset(&tokenizer);
        memset(&load_configuration, 0, sizeof(load_configuration));
        load_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        load_configuration.descriptor_bytes =
            SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
        load_configuration.tokenizer_json_path = HF_SINGLE_DIGIT_PATH;
        assert(SparkTokenizerLoadHuggingFaceJson(&tokenizer, &load_configuration) ==
            SPARK_STATUS_OK);
        memset(&compiled_configuration, 0, sizeof(compiled_configuration));
        compiled_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
        compiled_configuration.descriptor_bytes =
            SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
        compiled_configuration.compiled_tokenizer_path = COMPILED_PATH;
        assert(SparkTokenizerSaveCompiledFile(&tokenizer, &compiled_configuration) ==
            SPARK_STATUS_OK);
        SparkTokenizerDestroy(&tokenizer);
    }
    LoadSidecarAuto(&sidecar, COMPILED_PATH);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED);
    AssertRoundTrip(&sidecar, " ab!");
    SparkTokenizerSidecarUnload(&sidecar);

    /* tiktoken ranks: merge priority is the rank id of the concatenation. */
    WriteRanksFixture();
    LoadSidecarAuto(&sidecar, RANKS_PATH);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS);
    assert(sidecar.tokenizer.rank_ordered_merges == 1u);
    EncodeText(&sidecar, " ab!", ids, 64u, &count);
    assert(count == 2u);
    assert(ids[0] == 2u && ids[1] == 5u);
    /* " bah": the only mergeable pair is (a,h)->"ah"=3; " bah"=9 exists in
     * the vocabulary but is unreachable (its " b" prefix never forms), and
     * rank-ordered BPE must NOT invent it - pins the exact tiktoken rule. */
    EncodeText(&sidecar, " bah", ids, 64u, &count);
    assert(count == 3u);
    assert(ids[0] == 4u && ids[1] == 7u && ids[2] == 3u);
    AssertRoundTrip(&sidecar, " ab! bah hhh aha");
    SparkTokenizerSidecarUnload(&sidecar);

    /* Stop-token awareness on the REAL decode edge: the first stop id cuts
     * the text and is not rendered. */
    LoadSidecarAuto(&sidecar, HF_DIGIT_RUNS_PATH);
    {
        uint32_t stream[3];
        EncodeText(&sidecar, " ab", stream, 64u, &count);
        assert(count == 1u);
        stream[1] = 304u; /* <|stop|> */
        stream[2] = 302u; /* " ab" - must NOT be rendered */
        stop_tokens[0] = 304u;
        DecodeIds(&sidecar, stream, 3u, stop_tokens, 1u, 0u,
            decoded, sizeof(decoded), &decoded_bytes);
        assert(decoded_bytes == 3u);
        assert(memcmp(decoded, " ab", 3u) == 0);
        /* Without the stop the whole stream decodes: text + stop text + tail. */
        DecodeIds(&sidecar, stream, 3u, 0, 0, 0u,
            decoded, sizeof(decoded), &decoded_bytes);
        assert(decoded_bytes == 3u + 8u + 3u);
        /* skip_special_tokens: the stop text vanishes, the tail stays. */
        DecodeIds(&sidecar, stream, 3u, 0, 0,
            SPARK_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
            decoded, sizeof(decoded), &decoded_bytes);
        assert(decoded_bytes == 6u);
    }
    SparkTokenizerSidecarUnload(&sidecar);

    /* A missing asset is an IO error, loud, not a silent empty tokenizer. */
    {
        SparkTokenizerSidecarConfiguration configuration;
        SparkTokenizerSidecarReset(&sidecar);
        memset(&configuration, 0, sizeof(configuration));
        configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
        configuration.descriptor_bytes =
            SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
        configuration.asset_path = "build/definitely_missing_tokenizer.json";
        configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
        assert(SparkTokenizerSidecarLoad(&sidecar, &configuration) ==
            SPARK_STATUS_IO_ERROR);
    }
    printf("test_tokenizer_sidecar: format detection + round trips OK\n");
}

/* ===================== the ds4_eval ground truth ===================== */

static uint32_t GroundTruthCasesReady(uint32_t *case_count_out,
    SparkJsonDocument *fixture_document, SparkJsonDocument *cases_document)
{
    SparkJsonDocument fixture;
    SparkJsonDocument cases;
    int32_t fixture_root, cases_root, fixture_cases_array, cases_cases_array;
    memset(&fixture, 0, sizeof(fixture));
    memset(&cases, 0, sizeof(cases));
    if (SparkJsonLoadFile(GT_FIXTURE_PATH, &fixture) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SKIP ground truth: %s unavailable\n", GT_FIXTURE_PATH);
        return 0u;
    }
    if (SparkJsonLoadFile(GT_CASES_PATH, &cases) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SKIP ground truth: %s unavailable\n", GT_CASES_PATH);
        SparkJsonDocumentDestroy(&fixture);
        return 0u;
    }
    fixture_root = SparkJsonGetRootToken(&fixture);
    cases_root = SparkJsonGetRootToken(&cases);
    fixture_cases_array = SparkJsonFindObjectMember(&fixture, fixture_root, "cases");
    cases_cases_array = SparkJsonFindObjectMember(&cases, cases_root, "cases");
    assert(SparkJsonTokenIsType(&fixture, fixture_cases_array, SPARK_JSON_TOKEN_ARRAY));
    assert(SparkJsonTokenIsType(&cases, cases_cases_array, SPARK_JSON_TOKEN_ARRAY));
    *case_count_out = SparkJsonGetArrayElementCount(&fixture, fixture_cases_array);
    assert(*case_count_out == SparkJsonGetArrayElementCount(&cases, cases_cases_array));
    *fixture_document = fixture;
    *cases_document = cases;
    return 1u;
}

/* Verify encode(text)==ids and decode(ids)==text for one case. Cases are
 * ordinal-aligned; the id fields prove the alignment case by case. */
static void GroundTruthCase(SparkTokenizerSidecar *sidecar,
    const SparkJsonDocument *fixture_document,
    const SparkJsonDocument *cases_document,
    SparkTokenizerWorkspace *workspace,
    uint32_t case_index)
{
    int32_t fixture_root = SparkJsonGetRootToken((SparkJsonDocument *)fixture_document);
    int32_t cases_root = SparkJsonGetRootToken((SparkJsonDocument *)cases_document);
    int32_t fixture_cases = SparkJsonFindObjectMember((SparkJsonDocument *)fixture_document, fixture_root, "cases");
    int32_t cases_cases = SparkJsonFindObjectMember((SparkJsonDocument *)cases_document, cases_root, "cases");
    int32_t fixture_case = SparkJsonGetArrayElement((SparkJsonDocument *)fixture_document, fixture_cases, case_index);
    int32_t cases_case = SparkJsonGetArrayElement((SparkJsonDocument *)cases_document, cases_cases, case_index);
    int32_t ids_member, prompt_member, fixture_id_member, cases_id_member;
    int32_t ids_element;
    uint32_t ids[8192];
    uint32_t id_count;
    uint32_t encoded[65536];
    char *prompt_text = 0;
    char *fixture_id = 0;
    char *cases_id = 0;
    char decoded_text[1u << 20];
    uint32_t decoded_bytes = 0;
    SparkTokenizerEncoding encoding;
    uint32_t index;

    fixture_id_member = SparkJsonFindObjectMember((SparkJsonDocument *)fixture_document, fixture_case, "id");
    cases_id_member = SparkJsonFindObjectMember((SparkJsonDocument *)cases_document, cases_case, "id");
    assert(SparkJsonCopyString((SparkJsonDocument *)fixture_document, fixture_id_member, &fixture_id) == SPARK_STATUS_OK);
    assert(SparkJsonCopyString((SparkJsonDocument *)cases_document, cases_id_member, &cases_id) == SPARK_STATUS_OK);
    assert(strcmp(fixture_id, cases_id) == 0);

    ids_member = SparkJsonFindObjectMember((SparkJsonDocument *)fixture_document, fixture_case, "prompt_token_ids");
    assert(SparkJsonTokenIsType((SparkJsonDocument *)fixture_document, ids_member, SPARK_JSON_TOKEN_ARRAY));
    id_count = SparkJsonGetArrayElementCount((SparkJsonDocument *)fixture_document, ids_member);
    assert(id_count <= sizeof(ids) / sizeof(ids[0]));
    ids_element = SparkJsonGetArrayElementFirst((SparkJsonDocument *)fixture_document, ids_member);
    for (index = 0u; index < id_count; index++)
    {
        assert(ids_element >= 0);
        assert(SparkJsonGetUInt32((SparkJsonDocument *)fixture_document, ids_element, &ids[index]) == SPARK_STATUS_OK);
        ids_element = SparkJsonGetArrayElementNext((SparkJsonDocument *)fixture_document, ids_member, ids_element);
    }

    prompt_member = SparkJsonFindObjectMember((SparkJsonDocument *)cases_document, cases_case, "rendered_prompt");
    assert(SparkJsonCopyString((SparkJsonDocument *)cases_document, prompt_member, &prompt_text) == SPARK_STATUS_OK);

    /* decode(ids) == text */
    assert(SparkTokenizerSidecarDecodeText(sidecar, ids, id_count, 0, 0, 0u,
        decoded_text, sizeof(decoded_text), &decoded_bytes) == SPARK_STATUS_OK);
    assert(decoded_bytes == strlen(prompt_text));
    assert(memcmp(decoded_text, prompt_text, decoded_bytes) == 0);

    /* encode(text) == ids */
    SparkTokenizerEncodingReset(&encoding);
    encoding.token_capacity = (uint32_t)sizeof(encoded) / sizeof(encoded[0]);
    encoding.token_ids = encoded;
    assert(SparkTokenizerSidecarEncodeText(sidecar, prompt_text,
        (uint32_t)strlen(prompt_text), 0u, workspace, &encoding) == SPARK_STATUS_OK);
    assert(encoding.overflow_token_count == 0u);
    assert(encoding.token_count == id_count);
    for (index = 0u; index < id_count; index++)
        assert(encoded[index] == ids[index]);

    free(prompt_text);
    free(fixture_id);
    free(cases_id);
}

static void TestGroundTruth(void)
{
    SparkTokenizerSidecar sidecar;
    SparkTokenizerSidecarConfiguration configuration;
    SparkJsonDocument fixture_document;
    SparkJsonDocument cases_document;
    SparkTokenizerWorkspace workspace;
    uint32_t case_count = 0;
    uint32_t case_index;
    memset(&fixture_document, 0, sizeof(fixture_document));
    memset(&cases_document, 0, sizeof(cases_document));
    if (GroundTruthCasesReady(&case_count, &fixture_document, &cases_document) == 0u)
        return;
    {
        FILE *probe = fopen(GT_TOKENIZER_PATH, "rb");
        if (probe == 0)
        {
            fprintf(stderr, "SKIP ground truth: %s unavailable\n", GT_TOKENIZER_PATH);
            SparkJsonDocumentDestroy(&fixture_document);
            SparkJsonDocumentDestroy(&cases_document);
            return;
        }
        fclose(probe);
    }
    SparkTokenizerSidecarReset(&sidecar);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.asset_path = GT_TOKENIZER_PATH;
    configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
    assert(SparkTokenizerSidecarLoad(&sidecar, &configuration) == SPARK_STATUS_OK);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON);
    SparkTokenizerWorkspaceReset(&workspace);
    /* One workspace for the whole sweep: it grows to the longest prompt and
     * its piece cache warms across cases, which is exactly how the serving
     * edge uses it. */
    assert(SparkTokenizerWorkspaceInitialize(&workspace, 1u << 16) == SPARK_STATUS_OK);
    for (case_index = 0u; case_index < case_count; case_index++)
        GroundTruthCase(&sidecar, &fixture_document, &cases_document,
            &workspace, case_index);
    SparkTokenizerWorkspaceDestroy(&workspace);
    SparkTokenizerSidecarUnload(&sidecar);
    SparkJsonDocumentDestroy(&fixture_document);
    SparkJsonDocumentDestroy(&cases_document);
    printf("test_tokenizer_sidecar: ds4_eval ground truth %u/%u cases "
        "encode+decode identity OK\n", case_count, case_count);
}

/* The other named family: the tiktoken ranks asset loads through the same
 * sidecar and round trips; its byte order pins single-byte ids. */
static void TestKimiRanksAsset(void)
{
    SparkTokenizerSidecar sidecar;
    SparkTokenizerSidecarConfiguration configuration;
    uint32_t ids[256];
    uint32_t count = 0;
    FILE *probe = fopen(K3_RANKS_PATH, "rb");
    if (probe == 0)
    {
        fprintf(stderr, "SKIP kimi ranks: %s unavailable\n", K3_RANKS_PATH);
        return;
    }
    fclose(probe);
    SparkTokenizerSidecarReset(&sidecar);
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
    configuration.descriptor_bytes = SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.asset_path = K3_RANKS_PATH;
    configuration.format = SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO;
    assert(SparkTokenizerSidecarLoad(&sidecar, &configuration) == SPARK_STATUS_OK);
    assert(sidecar.format == SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS);
    /* "IQ== 0" is the first line: byte '!' is id 0. */
    EncodeText(&sidecar, "!", ids, 256u, &count);
    assert(count == 1u && ids[0] == 0u);
    AssertRoundTrip(&sidecar,
        "Serving ourselves: messages in, text out. \xe6\x9c\x8d\xe5\x8a\xa1\xe8\x87\xaa\xe5\xb7\xb1");
    AssertRoundTrip(&sidecar, "def gate():\n    return 0xdeadbeef  # 12345\n");
    SparkTokenizerSidecarUnload(&sidecar);
    printf("test_tokenizer_sidecar: tiktoken ranks asset round trip OK\n");
}

int main(void)
{
    TestFormatDetectionAndRoundTrips();
    TestKimiRanksAsset();
    TestGroundTruth();
    printf("test_tokenizer_sidecar: ALL OK\n");
    return 0;
}
