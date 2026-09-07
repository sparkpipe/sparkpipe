#include "sparkpipe/spark_tokenizer_sidecar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_TOKENIZER_SIDECAR_DETECT_BYTES 8u

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
    {
        return SPARK_STATUS_IO_ERROR;
    }
    read_bytes = fread(prefix, 1u, sizeof(prefix), file);
    fclose(file);
    if (read_bytes == 0u)
    {
        return SPARK_STATUS_IO_ERROR;
    }
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
            return SPARK_STATUS_INVALID_ARGUMENT;
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
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
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

SparkStatus SparkTokenizerSidecarEncodeText(
    const SparkTokenizerSidecar *sidecar,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding)
{
    if (sidecar == 0 || sidecar->abi_version != SPARK_TOKENIZER_SIDECAR_ABI_VERSION ||
        sidecar->descriptor_bytes != SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES ||
        sidecar->format == SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkTokenizerEncodeUtf8WithWorkspace(
        &sidecar->tokenizer,
        text,
        text_bytes,
        encode_flags,
        workspace,
        encoding);
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
    {
        *text_bytes_out = 0u;
    }
    if (sidecar == 0 || sidecar->abi_version != SPARK_TOKENIZER_SIDECAR_ABI_VERSION ||
        sidecar->descriptor_bytes != SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES ||
        sidecar->format == SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
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
