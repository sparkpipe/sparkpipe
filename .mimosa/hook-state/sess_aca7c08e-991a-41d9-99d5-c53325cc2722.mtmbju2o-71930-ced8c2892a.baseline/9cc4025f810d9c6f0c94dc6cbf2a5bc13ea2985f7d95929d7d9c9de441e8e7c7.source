#ifndef SPARKPIPE_SPARK_TOKENIZER_SIDECAR_H
#define SPARKPIPE_SPARK_TOKENIZER_SIDECAR_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_TOKENIZER_SIDECAR_ABI_VERSION 1u
#define SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerSidecar))
#define SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerSidecarConfiguration))

#define SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO 0u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON 1u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS 2u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED 3u

typedef struct SparkTokenizerSidecarConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *asset_path;
    uint32_t format;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerSidecarConfiguration;

typedef struct SparkTokenizerSidecar
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t format;
    uint32_t maximum_token_text_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkTokenizer tokenizer;
} SparkTokenizerSidecar;

void SparkTokenizerSidecarReset(
    SparkTokenizerSidecar *sidecar);

void SparkTokenizerSidecarUnload(
    SparkTokenizerSidecar *sidecar);

SparkStatus SparkTokenizerSidecarLoad(
    SparkTokenizerSidecar *sidecar,
    const SparkTokenizerSidecarConfiguration *configuration);

SparkStatus SparkTokenizerSidecarEncodeText(
    const SparkTokenizerSidecar *sidecar,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding);

SparkStatus SparkTokenizerSidecarDecodeText(
    const SparkTokenizerSidecar *sidecar,
    const uint32_t *token_ids,
    uint32_t token_count,
    const uint32_t *stop_token_ids,
    uint32_t stop_token_count,
    uint32_t decode_flags,
    char *text,
    uint32_t text_capacity,
    uint32_t *text_bytes_out);

#ifdef __cplusplus
}
#endif

#endif
