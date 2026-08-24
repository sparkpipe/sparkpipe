#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GLM5.2 DRAFT-PACK ingestion (report section 26 / doctrine close-out):
 * the drafter artifacts {manifest.json, config.json, model.safetensors}
 * travel inside ONE glm52 stage-pack-format container instead of three
 * loose env paths. ADDITIVE beside the raw SPARK_GLM52_DSPARK_* triple -
 * the working raw path is NOT retired until GB10 proves the pack path
 * end-to-end (KVcache law). Receipt discipline is identical to the raw
 * path: bytes+sha256 pinned at mint time inside the pack's receipt entry,
 * re-verified here during extraction before any path reaches the backend
 * (which independently re-validates config/safetensors against the
 * manifest at Initialize). */

/* Artifact ids live OUTSIDE SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT so a
 * model-weight reader can never confuse a draft-pack payload entry with a
 * tensor entry. */
#define SPARK_GLM52_DSPARK_PACK_KIND_RECEIPT UINT32_C(0x444431)
#define SPARK_GLM52_DSPARK_PACK_KIND_MANIFEST UINT32_C(0x444432)
#define SPARK_GLM52_DSPARK_PACK_KIND_CONFIG UINT32_C(0x444433)
#define SPARK_GLM52_DSPARK_PACK_KIND_SAFETENSORS UINT32_C(0x444434)

/* Fixed-size mint receipt carried as a KIND_RECEIPT payload: three
 * (sha256, byte_count) pairs in artifact order manifest, config,
 * safetensors. Binary by design - the resolver runs where no JSON parser
 * is linked; the JSON-facing receipt lives in the minter tool. */
typedef struct SparkGlm52DsparkPackReceiptEntry
{
	uint8_t sha256[32];
	uint64_t byte_count;
} SparkGlm52DsparkPackReceiptEntry;

typedef struct SparkGlm52DsparkPackReceipt
{
	SparkGlm52DsparkPackReceiptEntry entries[3];
} SparkGlm52DsparkPackReceipt;

/* Resolves one drafter PACK into three materialized artifact files under
 * scratch_directory/dspark_pack_<first8 of pack sha256>/ . Returns
 * SPARK_STATUS_OK and fills the three output path buffers on success.
 * Loud refusals: bad magic/version/header geometry, missing or duplicated
 * artifact entries, placement outside file bounds or off 256-B alignment,
 * byte-count mismatch against the receipt, sha256 mismatch against the
 * receipt, unreadable pack. */
SparkStatus SparkGlm52DsparkPackResolve(
    const char *pack_path,
    const char *scratch_directory,
    char *manifest_path_out,
    size_t manifest_path_bytes,
    char *config_path_out,
    size_t config_path_bytes,
    char *safetensors_path_out,
    size_t safetensors_path_bytes);

#ifdef __cplusplus
}
#endif
