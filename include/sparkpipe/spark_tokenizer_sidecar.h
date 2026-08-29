#ifndef SPARKPIPE_SPARK_TOKENIZER_SIDECAR_H
#define SPARKPIPE_SPARK_TOKENIZER_SIDECAR_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The tokenizer sidecar: per-deployment text-in/text-out infrastructure.
 *
 * A deployment references a tokenizer ASSET (a file shipped beside the pack,
 * e.g. under the runtime root). The sidecar loads it, detects the asset
 * format when asked to, and exposes the bounded encode/decode edge the API
 * front door uses: prompt text -> token ids before the engine sees the
 * request, generated token ids -> response text after the engine events.
 * The engine and every internal path stay token-id-only; no model-specific
 * detail lives here (the dry-law gate enforces this).
 *
 * Encode/decode calls are read-only on the sidecar apart from the workspace,
 * which carries the per-call scratch state; concurrent requests each pass
 * their own workspace. */

#define SPARK_TOKENIZER_SIDECAR_ABI_VERSION 1u
#define SPARK_TOKENIZER_SIDECAR_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerSidecar))
#define SPARK_TOKENIZER_SIDECAR_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkTokenizerSidecarConfiguration))

/* Asset formats. AUTO detects by content: a leading '{' is a HuggingFace
 * tokenizer.json, the compiled-file magic is the compiled format, anything
 * else parses as tiktoken ranks ("base64(piece) rank" lines). */
#define SPARK_TOKENIZER_SIDECAR_FORMAT_AUTO 0u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_HUGGINGFACE_JSON 1u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_TIKTOKEN_RANKS 2u
#define SPARK_TOKENIZER_SIDECAR_FORMAT_COMPILED 3u

typedef struct SparkTokenizerSidecarConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    /* Resolved filesystem path of the tokenizer asset. */
    const char *asset_path;
    /* SPARK_TOKENIZER_SIDECAR_FORMAT_*; AUTO detects by content. */
    uint32_t format;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkTokenizerSidecarConfiguration;

typedef struct SparkTokenizerSidecar
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    /* The resolved format (never AUTO after a successful load). */
    uint32_t format;
    /* Upper bound on the decoded text bytes of ONE token id; a caller sizes
     * a decode buffer as token_count * maximum_token_text_bytes + 1. */
    uint32_t maximum_token_text_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkTokenizer tokenizer;
} SparkTokenizerSidecar;

void SparkTokenizerSidecarReset(
    SparkTokenizerSidecar *sidecar);

void SparkTokenizerSidecarUnload(
    SparkTokenizerSidecar *sidecar);

/* Loads the asset and computes the per-token decode bound. A failed load
 * leaves the sidecar reset; the caller decides whether that is fatal
 * (the API treats a configured-but-unloadable asset as fatal at startup:
 * loud, never a silent fall-back to token-id-only serving). */
SparkStatus SparkTokenizerSidecarLoad(
    SparkTokenizerSidecar *sidecar,
    const SparkTokenizerSidecarConfiguration *configuration);

/* Text -> ids. Wraps SparkTokenizerEncodeUtf8WithWorkspace; the workspace is
 * the caller-owned scratch (one per requesting thread). */
SparkStatus SparkTokenizerSidecarEncodeText(
    const SparkTokenizerSidecar *sidecar,
    const char *text,
    uint32_t text_bytes,
    uint32_t encode_flags,
    SparkTokenizerWorkspace *workspace,
    SparkTokenizerEncoding *encoding);

/* Ids -> text, stop-token aware. The FIRST id that appears in
 * stop_token_ids terminates the text: ids from that position on (including
 * the stop id itself) are not rendered, so a per-request stop_token_ids set
 * survives the text round trip exactly as it behaves in the token stream.
 * text_capacity should be token_count * sidecar->maximum_token_text_bytes + 1. */
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
