#ifndef SPARKPIPE_SPARK_K3_DSPARK_PACK_H
#define SPARKPIPE_SPARK_K3_DSPARK_PACK_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * K3DS drafter-pack bind: the load-and-prove half of the k3 family's
 * speculation-provider slot (docs/SPECULATION_PROVIDER_DESIGN.md). Opens a
 * tools/k3_dspark_stagepack.py pack, validates the wire header AND the pinned
 * source geometry (every mismatch names its field in the refusal buffer - the
 * capability query's supports() -> WHY rule), and hands back a read-only
 * file-backed mapping plus the entry table. CUDA-free: host gates and tests
 * run it beside the serving tier.
 *
 * The DRAFT FORWARD is not here and not anywhere in this family yet - the
 * provider's draft ops fail closed until the drafter kernels land. What this
 * buys now is the wire path: packs can be built, bound, and proven before a
 * single draft kernel exists.
 */

#define SPARK_K3_DSPARK_MAX_REFUSAL_BYTES 160u

typedef struct SparkK3DsparkPack
{
	int fd;
	uint8_t *mapping;          /* file-backed pack mmap, PROT_READ */
	uint64_t file_bytes;
	uint32_t tensor_count;
	uint8_t *entries;          /* entry array base, inside the mapping */
	/* geometry, read from the pack header and checked against the pinned
	 * redhatai constants (spark_k3_dspark_format.h) */
	uint32_t hidden;
	uint32_t layer_count;
	uint32_t query_heads;
	uint32_t kv_heads;
	uint32_t head_dim;
	uint32_t ffn_dimension;
	uint32_t vocab;
	uint32_t block_size;
	uint32_t draft_token_count;
	uint32_t target_tap_layers[5];
	uint32_t markov_rank;
	uint32_t mask_token_id;
	uint32_t sliding_window;
	uint32_t flags;
	uint32_t confidence_input_dimension;
} SparkK3DsparkPack;

/* Bind: SPARK_STATUS_OK with the pack filled, or VALIDATION_FAILED /
 * IO_ERROR with `refusal` naming the exact field that failed. */
SparkStatus SparkK3DsparkPackBind(const char *path, SparkK3DsparkPack *pack,
	char *refusal, uint32_t refusal_bytes);

void SparkK3DsparkPackRelease(SparkK3DsparkPack *pack);

/* Payload for a (kind, layer) slot: SPARK_STATUS_OK, NOT_FOUND when the pack
 * carries no such slot, VALIDATION_FAILED when the entry's shape disagrees
 * with the format's kind table. */
SparkStatus SparkK3DsparkPackPayload(const SparkK3DsparkPack *pack,
	uint32_t kind, uint32_t layer, const void **payload, uint64_t *bytes);

#ifdef __cplusplus
}
#endif

#endif
