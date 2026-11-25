#include "sparkpipe/spark_tokenizer_sidecar.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_TOKENIZER_SIDECAR_DETECT_BYTES 8u
#define SPARK_TOKENIZER_SIDECAR_PAIR_BITS 1024u
#define SPARK_TOKENIZER_SIDECAR_PAIR_WORDS (SPARK_TOKENIZER_SIDECAR_PAIR_BITS / 64u)

typedef struct SparkTokenizerSpecialFilter
{
    uint64_t first_mask[4];
    uint64_t single_mask[4];
    uint64_t pair_bloom[SPARK_TOKENIZER_SIDECAR_PAIR_WORDS];
    uint32_t match_special;
    uint32_t active_count;
} SparkTokenizerSpecialFilter;

static uint32_t SparkTokenizerSidecarBytesLookLikeCompiledMagic(
    const uint8_t *bytes)
{
    uint64_t magic = 0u;
    uint32_t index;
    for (index = 0u; index < SPARK_TOKENIZER_SIDECAR_DETECT_BYTES; ++index)
    {
        magic |= (uint64_t)bytes[index] << (8u * index);
    }
    return magic == SPARK_TOKENIZER_COMPILED_FILE_MAGIC ? 1u : 0u;
}

void SparkTokenizerSidecarReset(
    SparkTokenizerSidecar *sidecar)
{
    if (sidecar == 0)
    {
        return;
    }
    memset(sidecar, 0, sizeof(*sidecar));
    sidecar->abi_version = SPARK_TOKENIZER_SIDECAR_ABI_VERSION;
    sidecar->descriptor_bytes = SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES;
}

void SparkTokenizerSidecarUnload(
    SparkTokenizerSidecar *sidecar)
{
    if (sidecar == 0)
    {
        return;
    }
    SparkTokenizerDestroy(&sidecar->tokenizer);
    SparkTokenizerSidecarReset(sidecar);
}

static SparkStatus SparkTokenizerSidecarDetectFormat(
    const char *asset_path,
    uint32_t *format_out)
{
    FILE *file;
    uint8_t prefix[SPARK_TOKENIZER_SIDECAR_DETECT_BYTES];
    size_t read_bytes;

    file = fopen(asset_path, "rb");
    if (file == 0)
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    read_bytes = fread(prefix, 1u, sizeof(prefix), file);
    fclose(file);
    if (read_bytes == 0u)
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    if (prefix[0] == (uint8_t)'{')
    {
        *format_out = SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON;
        return SPARK_STATUS_OK;
    }
    if (read_bytes >= sizeof(prefix) &&
        SparkTokenizerSidecarBytesLookLikeCompiledMagic(prefix) != 0u)
    {
        *format_out = SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED;
        return SPARK_STATUS_OK;
    }
    *format_out = SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkTokenizerSidecarLoadByFormat(
    SparkTokenizerSidecar *sidecar,
    const SparkTokenizerSidecarConfiguration *configuration)
{
    switch (configuration->format)
    {
        case SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON:
        {
            SparkTokenizerHuggingFaceJsonConfiguration tokenizer_configuration;
            memset(&tokenizer_configuration, 0, sizeof(tokenizer_configuration));
            tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
            tokenizer_configuration.descriptor_bytes =
                SPARK_TOKENIZER_HF_JSON_CONFIGURATION_DESCRIPTOR_BYTES;
            tokenizer_configuration.tokenizer_json_path = configuration->asset_path;
            return SparkTokenizerLoadHuggingFaceJson(
                &sidecar->tokenizer,
                &tokenizer_configuration);
        }
        case SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS:
        {
            SparkTokenizerTiktokenRanksConfiguration tokenizer_configuration;
            memset(&tokenizer_configuration, 0, sizeof(tokenizer_configuration));
            tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
            tokenizer_configuration.descriptor_bytes =
                SPARK_TOKENIZER_TIKTOKEN_RANKS_CONFIGURATION_DESCRIPTOR_BYTES;
            tokenizer_configuration.ranks_path = configuration->asset_path;
            return SparkTokenizerLoadTiktokenRanks(
                &sidecar->tokenizer,
                &tokenizer_configuration);
        }
        case SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED:
        {
            SparkTokenizerCompiledFileConfiguration tokenizer_configuration;
            memset(&tokenizer_configuration, 0, sizeof(tokenizer_configuration));
            tokenizer_configuration.abi_version = SPARK_TOKENIZER_ABI_VERSION;
            tokenizer_configuration.descriptor_bytes =
                SPARK_TOKENIZER_COMPILED_FILE_CONFIGURATION_DESCRIPTOR_BYTES;
            tokenizer_configuration.compiled_tokenizer_path = configuration->asset_path;
            return SparkTokenizerLoadCompiledFile(
                &sidecar->tokenizer,
                &tokenizer_configuration);
        }
        default:
            SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
}

SparkStatus SparkTokenizerSidecarLoad(
    SparkTokenizerSidecar *sidecar,
    const SparkTokenizerSidecarConfiguration *configuration)
{
    uint32_t format;
    uint32_t token_id;
    uint32_t maximum_token_text_bytes;
    SparkStatus status;

    if (sidecar == 0 || configuration == 0 ||
        configuration->abi_version != SPARK_TOKENIZER_SIDECAR_ABI_VERSION ||
        configuration->descriptor_bytes != SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->asset_path == 0 ||
        configuration->asset_path[0] == '\0' ||
        configuration->format > SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    SparkTokenizerSidecarReset(sidecar);
    format = configuration->format;
    if (format == SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO)
    {
        status = SparkTokenizerSidecarDetectFormat(configuration->asset_path, &format);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    {
        SparkTokenizerSidecarConfiguration loaded_configuration;
        loaded_configuration = *configuration;
        loaded_configuration.format = format;
        status = SparkTokenizerSidecarLoadByFormat(sidecar, &loaded_configuration);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkTokenizerSidecarReset(sidecar);
        return status;
    }
    maximum_token_text_bytes = 0u;
    for (token_id = 0u; token_id <= sidecar->tokenizer.maximum_token_id; ++token_id)
    {
        uint32_t token_text_bytes;
        token_text_bytes = sidecar->tokenizer.token_text_bytes_by_id != 0 &&
            sidecar->tokenizer.token_text_by_id != 0 &&
            sidecar->tokenizer.token_text_by_id[token_id] != 0
            ? sidecar->tokenizer.token_text_bytes_by_id[token_id]
            : 0u;
        if (token_text_bytes > maximum_token_text_bytes)
        {
            maximum_token_text_bytes = token_text_bytes;
        }
    }
    if (sidecar->tokenizer.special_tokens != 0)
    {
        uint32_t index;
        for (index = 0u; index < sidecar->tokenizer.special_token_count; ++index)
        {
            if (sidecar->tokenizer.special_tokens[index].text_bytes > maximum_token_text_bytes)
            {
                maximum_token_text_bytes = sidecar->tokenizer.special_tokens[index].text_bytes;
            }
        }
    }
    sidecar->format = format;
    sidecar->maximum_token_text_bytes = maximum_token_text_bytes;
    return SPARK_STATUS_OK;
}

static uint32_t SparkTokenizerSidecarIsWhiteSpace(uint32_t value)
{
    return (value >= 0x09u && value <= 0x0Du) || value == 0x20u || value == 0x85u || value == 0xA0u ||
        value == 0x1680u || (value >= 0x2000u && value <= 0x200Au) || value == 0x2028u || value == 0x2029u ||
        value == 0x202Fu || value == 0x205Fu || value == 0x3000u ? 1u : 0u;
}

static uint32_t SparkTokenizerSidecarStripLeft(const char *text, uint32_t floor, uint32_t position)
{
    while (position > floor)
    {
        uint32_t start = position - 1u;
        uint32_t len, code_point = 0u;
        uint8_t b0;
        while (start > floor && ((uint8_t)text[start] & 0xC0u) == 0x80u)
            start -= 1u;
        len = position - start;
        b0 = (uint8_t)text[start];
        if (len == 1u && b0 < 0x80u)
            code_point = b0;
        else if (len == 2u && b0 >= 0xC2u && b0 <= 0xDFu)
            code_point = (((uint32_t)b0 & 0x1Fu) << 6) | ((uint8_t)text[start + 1u] & 0x3Fu);
        else if (len == 3u && b0 >= 0xE0u && b0 <= 0xEFu)
        {
            uint8_t b1 = (uint8_t)text[start + 1u];
            if ((b0 == 0xE0u && b1 < 0xA0u) || (b0 == 0xEDu && b1 >= 0xA0u))
                break;
            code_point = (((uint32_t)b0 & 0x0Fu) << 12) | (((uint32_t)b1 & 0x3Fu) << 6) | ((uint8_t)text[start + 2u] & 0x3Fu);
        }
        else
            break;
        if (!SparkTokenizerSidecarIsWhiteSpace(code_point))
            break;
        position = start;
    }
    return position;
}

static void SparkTokenizerSidecarBuildFilter(
    const SparkTokenizer *tokenizer,
    uint32_t match_special,
    SparkTokenizerSpecialFilter *filter)
{
    uint32_t index;
    memset(filter, 0, sizeof(*filter));
    filter->match_special = match_special;
    for (index = 0u; index < tokenizer->special_token_count; ++index)
    {
        const SparkTokenizerSpecialToken *token = &tokenizer->special_tokens[index];
        uint8_t b0;
        if ((token->is_special != 0u && match_special == 0u) || token->text == 0 || token->text_bytes == 0u)
            continue;
        filter->active_count += 1u;
        b0 = (uint8_t)token->text[0];
        filter->first_mask[b0 >> 6] |= 1ull << (b0 & 63u);
        if (token->text_bytes == 1u)
            filter->single_mask[b0 >> 6] |= 1ull << (b0 & 63u);
        else
        {
            uint32_t slot = ((uint32_t)b0 * 31u + (uint8_t)token->text[1]) & (SPARK_TOKENIZER_SIDECAR_PAIR_BITS - 1u);
            filter->pair_bloom[slot >> 6] |= 1ull << (slot & 63u);
        }
    }
}

static uint32_t SparkTokenizerSidecarMatchSpecialAt(
    const SparkTokenizer *tokenizer,
    const SparkTokenizerSpecialFilter *filter,
    const char *text,
    uint32_t text_bytes,
    uint32_t pos,
    uint32_t *token_id_out,
    uint32_t *matched_bytes_out,
    uint32_t *strip_left_out)
{
    uint8_t b0 = (uint8_t)text[pos];
    uint32_t rem, index;
    if ((filter->first_mask[b0 >> 6] & (1ull << (b0 & 63u))) == 0u)
        return 0u;
    if ((filter->single_mask[b0 >> 6] & (1ull << (b0 & 63u))) == 0u)
    {
        uint32_t slot;
        if (pos + 1u >= text_bytes)
            return 0u;
        slot = ((uint32_t)b0 * 31u + (uint8_t)text[pos + 1u]) & (SPARK_TOKENIZER_SIDECAR_PAIR_BITS - 1u);
        if ((filter->pair_bloom[slot >> 6] & (1ull << (slot & 63u))) == 0u)
            return 0u;
    }
    rem = text_bytes - pos;
    for (index = 0u; index < tokenizer->special_token_count; ++index)
    {
        const SparkTokenizerSpecialToken *token = &tokenizer->special_tokens[index];
        if ((token->is_special == 0u || filter->match_special != 0u) &&
            token->text != 0 && token->text_bytes > 0u && token->text_bytes <= rem &&
            (uint8_t)token->text[0] == b0 &&
            memcmp(text + pos, token->text, token->text_bytes) == 0)
        {
            *token_id_out = token->token_id;
            *matched_bytes_out = token->text_bytes;
            *strip_left_out = token->strip_left;
            return 1u;
        }
    }
    return 0u;
}

static SparkStatus SparkTokenizerSidecarAppendSegment(
    const SparkTokenizer *fast_tokenizer,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    SparkTokenizerEncoding seg;
    SparkStatus status;
    uint32_t remaining;
    if (text_bytes == 0u)
        return SPARK_STATUS_OK;
    SparkTokenizerEncodingReset(&seg);
    remaining = encoding->token_count < encoding->token_capacity ? encoding->token_capacity - encoding->token_count : 0u;
    seg.token_capacity = remaining;
    seg.token_ids = remaining != 0u ? encoding->token_ids + encoding->token_count : 0;
    status = SparkTokenizerEncodeUtf8WithWorkspace(fast_tokenizer, text, text_bytes, encode_flags, workspace, &seg);
    if (status != SPARK_STATUS_OK && status != SPARK_STATUS_CAPACITY_EXCEEDED)
        return status;
    encoding->token_count += seg.token_count;
    encoding->overflow_token_count += seg.overflow_token_count;
    encoding->invalid_segment_count += seg.invalid_segment_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerSidecarEncodeText(
    const SparkTokenizerSidecar *sidecar,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    SparkTokenizerSpecialFilter filter;
    SparkTokenizer fast_tok;
    uint32_t pos = 0u, segment_start = 0u, started = 0u, seg_flags;
    SparkStatus status;
    if (sidecar == 0 || sidecar->abi_version != SPARK_TOKENIZER_SIDECAR_ABI_VERSION ||
        sidecar->descriptor_bytes != SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES ||
        sidecar->format == SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    if (text == 0 || text_bytes == 0u || sidecar->tokenizer.special_token_count == 0u ||
        (encode_flags & ~SPARK_TOKENIZER_ENCODE_KNOWN_FLAGS) != 0u ||
        workspace == 0 || workspace->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        workspace->descriptor_bytes != SPARK_TOKENIZER_WORKSPACE_DESCRIPTOR_BYTES ||
        encoding == 0 || encoding->abi_version != SPARK_TOKENIZER_ABI_VERSION ||
        encoding->descriptor_bytes != SPARK_TOKENIZER_ENCODING_DESCRIPTOR_BYTES ||
        (encoding->token_ids == 0 && encoding->token_capacity != 0u))
        return SparkTokenizerEncodeUtf8WithWorkspace(&sidecar->tokenizer, text, text_bytes, encode_flags, workspace, encoding);
    SparkTokenizerSidecarBuildFilter(&sidecar->tokenizer,
        (encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_DISABLE_SPECIAL_TOKEN_MATCH) == 0u ? 1u : 0u, &filter);
    fast_tok = sidecar->tokenizer;
    fast_tok.special_token_count = 0u;
    if (filter.active_count == 0u)
        return SparkTokenizerEncodeUtf8WithWorkspace(&fast_tok, text, text_bytes, encode_flags, workspace, encoding);
    seg_flags = encode_flags & ~SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE;
    while (pos < text_bytes)
    {
        uint32_t token_id = 0u, matched_bytes = 0u, strip_left = 0u;
        if (!SparkTokenizerSidecarMatchSpecialAt(&sidecar->tokenizer, &filter, text, text_bytes, pos, &token_id, &matched_bytes, &strip_left))
        {
            pos += 1u;
            continue;
        }
        if (started == 0u)
        {
            uint32_t add_prefix = ((encode_flags & SPARK_TOKENIZER_ENCODE_FLAG_ADD_PREFIX_SPACE) != 0u ||
                sidecar->tokenizer.add_prefix_space != 0u) && text[0] != ' ';
            started = 1u;
            fast_tok.add_prefix_space = 0u;
            encoding->token_count = 0u;
            encoding->overflow_token_count = 0u;
            encoding->invalid_segment_count = 0u;
            if (add_prefix != 0u)
            {
                status = SparkTokenizerSidecarAppendSegment(&fast_tok, " ", 1u, seg_flags, workspace, encoding);
                if (status != SPARK_STATUS_OK)
                    return status;
            }
        }
        {
            uint32_t seg_end = strip_left != 0u ? SparkTokenizerSidecarStripLeft(text, segment_start, pos) : pos;
            status = SparkTokenizerSidecarAppendSegment(&fast_tok, text + segment_start, seg_end - segment_start, seg_flags, workspace, encoding);
            if (status != SPARK_STATUS_OK)
                return status;
        }
        if (encoding->token_count < encoding->token_capacity)
            encoding->token_ids[encoding->token_count++] = token_id;
        else
            encoding->overflow_token_count += 1u;
        pos += matched_bytes;
        segment_start = pos;
    }
    if (started == 0u)
        return SparkTokenizerEncodeUtf8WithWorkspace(&fast_tok, text, text_bytes, encode_flags, workspace, encoding);
    status = SparkTokenizerSidecarAppendSegment(&fast_tok, text + segment_start, text_bytes - segment_start, seg_flags, workspace, encoding);
    if (status != SPARK_STATUS_OK)
        return status;
    if (encoding->overflow_token_count != 0u)
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    return SPARK_STATUS_OK;
}

SparkStatus SparkTokenizerSidecarDecodeText(
    const SparkTokenizerSidecar *sidecar,
    const uint32_t *token_ids,
    uint32_t token_count,
    const uint32_t *stop_token_ids,
    uint32_t stop_token_count,
    uint32_t decode_flags,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_out)
{
    uint32_t effective_count;
    uint32_t index;
    uint32_t stop_index;

    if (text_bytes_out != 0)
        *text_bytes_out = 0u;
    if (sidecar == 0 || sidecar->abi_version != SPARK_TOKENIZER_SIDECAR_ABI_VERSION ||
        sidecar->descriptor_bytes != SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES ||
        sidecar->format == SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    effective_count = token_count;
    for (index = 0u; index < token_count && stop_token_count != 0u; ++index)
    {
        for (stop_index = 0u; stop_index < stop_token_count; ++stop_index)
        {
            if (stop_token_ids[stop_index] == token_ids[index])
            {
                effective_count = index;
                break;
            }
        }
        if (effective_count != token_count)
        {
            break;
        }
    }
    return SparkTokenizerDecodeTokenIds(
        &sidecar->tokenizer,
        token_ids,
        effective_count,
        decode_flags,
        text,
        text_capacity,
        text_bytes_out);
}
