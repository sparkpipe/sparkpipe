/*
 * K3DS drafter pack: format + bind gate (CPU-only, no CUDA).
 *
 * Builds a full redhatai-geometry pack IN A SPARSE FILE (correct header,
 * entries, and payload bounds; the payloads themselves are holes - the bind
 * never reads payload bytes, it proves layout), then:
 *   - a good pack binds, and the geometry fields round-trip;
 *   - payloads resolve by (kind, layer) with the right byte counts;
 *   - a truncated pack, a wrong magic, a radixark-class block 7, a foreign
 *     tap layer, and a missing-flags pack each FAIL naming the field;
 *   - kind 14 (the reserved slot) never resolves.
 * Run: cc -std=c11 -Wall -Wextra -Werror -O3 -D_GNU_SOURCE -Iinclude -Isrc
 *        -Imodel-families/k3/include -Imodules/k3_resident_decode_stage/include
 *        -Imodules/k3_resident_decode_stage/source
 *        tests/test_k3_dspark_pack.c modules/k3_resident_decode_stage/source/spark_k3_pack_load.c
 *        runtime/json.c src/spark_status.c -o build/test_k3_dspark_pack && ./build/test_k3_dspark_pack
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_k3_dspark_pack.h"

#include "spark_k3_dspark_format.h"

#define ENTRY_U32(p, i) ((uint32_t)(p)[4u*(i)] | ((uint32_t)(p)[4u*(i)+1u] << 8u) | \
	((uint32_t)(p)[4u*(i)+2u] << 16u) | ((uint32_t)(p)[4u*(i)+3u] << 24u))
static void put_u32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)(v & 0xFFu);
	p[1] = (uint8_t)((v >> 8u) & 0xFFu);
	p[2] = (uint8_t)((v >> 16u) & 0xFFu);
	p[3] = (uint8_t)((v >> 24u) & 0xFFu);
}
static void put_u64(uint8_t *p, uint64_t v)
{
	put_u32(p, (uint32_t)(v & 0xFFFFFFFFull));
	put_u32(p + 4u, (uint32_t)(v >> 32u));
}

static const uint32_t test_taps[5] =
{
	SPARK_K3_DSPARK_TARGET_TAP_LAYER_0, SPARK_K3_DSPARK_TARGET_TAP_LAYER_1,
	SPARK_K3_DSPARK_TARGET_TAP_LAYER_2, SPARK_K3_DSPARK_TARGET_TAP_LAYER_3,
	SPARK_K3_DSPARK_TARGET_TAP_LAYER_4
};

/* the packer's inventory: per-layer kinds 0..10 layer-major, then globals */
static uint32_t entry_kind(uint32_t index)
{
	if ( index < SPARK_K3_DSPARK_LAYER_COUNT * SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT )
		return(index % SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT);
	switch ( index )
	{
	case 55: return(SPARK_K3_DSPARK_TENSOR_PROJECTOR);
	case 56: return(SPARK_K3_DSPARK_TENSOR_MARKOV_W1);
	case 57: return(SPARK_K3_DSPARK_TENSOR_MARKOV_W2);
	case 58: return(SPARK_K3_DSPARK_TENSOR_FINAL_NORM);
	case 59: return(SPARK_K3_DSPARK_TENSOR_HIDDEN_NORM);
	case 60: return(SPARK_K3_DSPARK_TENSOR_EMBED);
	case 61: return(SPARK_K3_DSPARK_TENSOR_LM_HEAD);
	case 62: return(SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT);
	default: return(SPARK_K3_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS);
	}
}
static uint32_t entry_layer(uint32_t index)
{
	if ( index < SPARK_K3_DSPARK_LAYER_COUNT * SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT )
		return(index / SPARK_K3_DSPARK_PER_LAYER_KIND_COUNT);
	return(0xFFFFFFFFu);
}

struct TestPack { uint8_t *bytes; uint64_t file_bytes; };

static void build_pack(struct TestPack *out, uint32_t block_size,
	uint32_t tap_0, uint32_t flags, uint32_t magic)
{
	uint32_t index, rows, columns;
	uint64_t cursor, payload_base;
	uint8_t *bytes;
	cursor = SPARK_K3_DSPARK_HEADER_BYTES +
		(uint64_t)SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT * SPARK_K3_DSPARK_ENTRY_BYTES;
	payload_base = (cursor + SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u) &
		~(uint64_t)(SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u);
	for ( index = 0u; index < SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT; index++ )
	{
		SparkK3DsparkKindShape(entry_kind(index), &rows, &columns);
		cursor = payload_base +
			((cursor - payload_base + SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u) &
			 ~(uint64_t)(SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u));
		cursor += (uint64_t)rows * columns * 2u;
	}
	cursor = (cursor + SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u) &
		~(uint64_t)(SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u);
	out->file_bytes = cursor;
	bytes = (uint8_t *)calloc(1u, (size_t)SPARK_K3_DSPARK_HEADER_BYTES +
		(size_t)SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT * SPARK_K3_DSPARK_ENTRY_BYTES);
	out->bytes = bytes;
	put_u32(bytes + 0u, magic);
	put_u32(bytes + 4u, SPARK_K3_DSPARK_FORMAT_VERSION);
	put_u32(bytes + 8u, SPARK_K3_DSPARK_HEADER_BYTES);
	put_u32(bytes + 12u, SPARK_K3_DSPARK_ENTRY_BYTES);
	put_u32(bytes + 16u, SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT);
	put_u32(bytes + 20u, 7168u);                       /* hidden */
	put_u32(bytes + 24u, SPARK_K3_DSPARK_LAYER_COUNT);
	put_u32(bytes + 32u, SPARK_K3_DSPARK_LAYER_COUNT);
	put_u32(bytes + 64u, SPARK_K3_DSPARK_ATTN_QUERY_HEADS);
	put_u32(bytes + 68u, SPARK_K3_DSPARK_ATTN_KV_HEADS);
	put_u32(bytes + 72u, SPARK_K3_DSPARK_ATTN_HEAD_DIMENSION);
	put_u32(bytes + 76u, SPARK_K3_DSPARK_ATTN_ROPE_DIMENSION);
	put_u32(bytes + 80u, SPARK_K3_DSPARK_FFN_INTERMEDIATE);
	put_u32(bytes + 84u, SPARK_K3_DSPARK_VOCAB);
	put_u32(bytes + 88u, block_size);
	put_u32(bytes + 92u, SPARK_K3_DSPARK_TARGET_TAP_COUNT);
	put_u32(bytes + 96u, 1u);                          /* tp_degree */
	put_u32(bytes + 100u, 0u);                         /* tp_rank */
	put_u64(bytes + 104u, SPARK_K3_DSPARK_HEADER_BYTES);
	put_u64(bytes + 112u, out->file_bytes);
	for ( index = 0u; index < 5u; index++ )
		put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * index,
			index == 0u ? tap_0 : test_taps[index]);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 5u, SPARK_K3_DSPARK_MARKOV_RANK);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 6u, SPARK_K3_DSPARK_MASK_TOKEN_ID);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 7u, SPARK_K3_DSPARK_SLIDING_WINDOW);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 8u, flags);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 9u, 10000000u);
	put_u32(bytes + SPARK_K3_DSPARK_CORE_HEADER_BYTES + 4u * 10u,
		SPARK_K3_DSPARK_CONFIDENCE_INPUT_DIMENSION);
	cursor = payload_base;
	for ( index = 0u; index < SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT; index++ )
	{
		uint8_t *entry = bytes + SPARK_K3_DSPARK_HEADER_BYTES +
			(size_t)index * SPARK_K3_DSPARK_ENTRY_BYTES;
		SparkK3DsparkKindShape(entry_kind(index), &rows, &columns);
		cursor = payload_base +
			((cursor - payload_base + SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u) &
			 ~(uint64_t)(SPARK_K3_DSPARK_PAYLOAD_ALIGNMENT - 1u));
		put_u32(entry + 0u, entry_kind(index));
		put_u32(entry + 4u, entry_layer(index));
		put_u32(entry + 8u, SPARK_K3_DSPARK_WEIGHT_BF16);
		put_u32(entry + 12u, rows);
		put_u32(entry + 16u, columns);
		put_u64(entry + 24u, cursor);
		put_u64(entry + 32u, (uint64_t)rows * columns * 2u);
		cursor += (uint64_t)rows * columns * 2u;
	}
}

static int write_sparse(const char *path, const struct TestPack *pack)
{
	int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
	if ( fd < 0 )
		return(-1);
	if ( ftruncate(fd, (off_t)pack->file_bytes) != 0 ||
		pwrite(fd, pack->bytes, (size_t)SPARK_K3_DSPARK_HEADER_BYTES +
			(size_t)SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT * SPARK_K3_DSPARK_ENTRY_BYTES,
			0) < 0 )
	{
		(void)close(fd);
		return(-1);
	}
	return(close(fd));
}

static int expect_refusal(const struct TestPack *pack, const char *needle)
{
	char refusal[SPARK_K3_DSPARK_MAX_REFUSAL_BYTES];
	SparkK3DsparkPack bound;
	SparkStatus status;
	write_sparse("/tmp/k3dsp-test.k3dsp", pack);
	status = SparkK3DsparkPackBind("/tmp/k3dsp-test.k3dsp", &bound, refusal,
		sizeof(refusal));
	if ( status == SPARK_STATUS_OK )
	{
		printf("FAIL: expected refusal containing '%s', bind succeeded\n", needle);
		return(1);
	}
	if ( strstr(refusal, needle) == 0 )
	{
		printf("FAIL: refusal '%s' does not name '%s'\n", refusal, needle);
		return(1);
	}
	return(0);
}

int main(void)
{
	struct TestPack pack;
	char refusal[SPARK_K3_DSPARK_MAX_REFUSAL_BYTES];
	SparkK3DsparkPack bound;
	const void *payload;
	uint64_t payload_bytes;
	uint32_t rows, columns;
	int failures = 0;

	build_pack(&pack, SPARK_K3_DSPARK_BLOCK_SIZE, test_taps[0],
		SPARK_K3_DSPARK_FLAGS_REQUIRED, SPARK_K3_DSPARK_MAGIC);
	if ( write_sparse("/tmp/k3dsp-test.k3dsp", &pack) != 0 )
		return(2);
	if ( SparkK3DsparkPackBind("/tmp/k3dsp-test.k3dsp", &bound, refusal,
		sizeof(refusal)) != SPARK_STATUS_OK )
	{
		printf("FAIL: good pack refused: %s\n", refusal);
		return(1);
	}
	if ( bound.hidden != 7168u || bound.query_heads != 96u ||
		bound.kv_heads != 16u || bound.markov_rank != 256u ||
		bound.block_size != 8u || bound.draft_token_count != 7u ||
		bound.confidence_input_dimension != 7424u ||
		bound.target_tap_layers[0] != 24u || bound.target_tap_layers[4] != 92u ||
		bound.tensor_count != SPARK_K3_DSPARK_REDHATAI_TENSOR_COUNT )
	{
		printf("FAIL: bound geometry fields do not round-trip\n");
		failures++;
	}
	/* payload resolution: the projector (global) and one per-layer kind */
	if ( SparkK3DsparkPackPayload(&bound, SPARK_K3_DSPARK_TENSOR_PROJECTOR,
		0xFFFFFFFFu, &payload, &payload_bytes) != SPARK_STATUS_OK ||
		payload_bytes != (uint64_t)7168u * 5u * 7168u * 2u )
	{
		printf("FAIL: projector payload wrong\n");
		failures++;
	}
	if ( SparkK3DsparkPackPayload(&bound, SPARK_K3_DSPARK_TENSOR_ATTN_QUERY,
		3u, &payload, &payload_bytes) != SPARK_STATUS_OK ||
		payload_bytes != (uint64_t)96u * 64u * 7168u * 2u )
	{
		printf("FAIL: per-layer payload wrong\n");
		failures++;
	}
	/* the reserved slot never resolves, in either scope */
	SparkK3DsparkKindShape(SPARK_K3_DSPARK_TENSOR_RESERVED_14, &rows, &columns);
	if ( rows != 0u || columns != 0u ||
		SparkK3DsparkPackPayload(&bound, SPARK_K3_DSPARK_TENSOR_RESERVED_14,
			0xFFFFFFFFu, &payload, &payload_bytes) != SPARK_STATUS_NOT_FOUND )
	{
		printf("FAIL: reserved slot resolved or has a shape\n");
		failures++;
	}
	SparkK3DsparkPackRelease(&bound);

	/* radixark-class block 7: refused, naming the field */
	{
		struct TestPack radix;
		build_pack(&radix, 7u, test_taps[0], SPARK_K3_DSPARK_FLAGS_REQUIRED,
			SPARK_K3_DSPARK_MAGIC);
		failures += expect_refusal(&radix, "block_size");
		free(radix.bytes);
	}
	/* a foreign first tap: refused, naming the field */
	{
		struct TestPack foreign;
		build_pack(&foreign, SPARK_K3_DSPARK_BLOCK_SIZE, 7u,
			SPARK_K3_DSPARK_FLAGS_REQUIRED, SPARK_K3_DSPARK_MAGIC);
		failures += expect_refusal(&foreign, "tap_layer_0");
		free(foreign.bytes);
	}
	/* a pack missing the confidence+markov flag (a DFlash2-class pack): refused */
	{
		struct TestPack noflag;
		build_pack(&noflag, SPARK_K3_DSPARK_BLOCK_SIZE, test_taps[0],
			SPARK_K3_DSPARK_FLAG_EMBED | SPARK_K3_DSPARK_FLAG_LM_HEAD,
			SPARK_K3_DSPARK_MAGIC);
		failures += expect_refusal(&noflag, "flags");
		free(noflag.bytes);
	}
	/* wrong magic: refused */
	{
		struct TestPack badmagic;
		build_pack(&badmagic, SPARK_K3_DSPARK_BLOCK_SIZE, test_taps[0],
			SPARK_K3_DSPARK_FLAGS_REQUIRED, 0x50533651u);
		failures += expect_refusal(&badmagic, "magic");
		free(badmagic.bytes);
	}
	/* truncated file: the header's file_bytes disagrees with the actual size */
	{
		int fd;
		if ( write_sparse("/tmp/k3dsp-test.k3dsp", &pack) != 0 )
			return(2);
		fd = open("/tmp/k3dsp-test.k3dsp", O_RDWR);
		if ( fd < 0 || ftruncate(fd, (off_t)pack.file_bytes - 4096u) != 0 )
			return(2);
		(void)close(fd);
		if ( SparkK3DsparkPackBind("/tmp/k3dsp-test.k3dsp", &bound, refusal,
			sizeof(refusal)) == SPARK_STATUS_OK ||
			strstr(refusal, "file_bytes") == 0 )
		{
			printf("FAIL: truncated pack not refused naming file_bytes\n");
			failures++;
		}
	}
	free(pack.bytes);
	(void)unlink("/tmp/k3dsp-test.k3dsp");
	if ( failures == 0 )
		printf("k3 dspark drafter pack: bind + refusal paths all pass\n");
	return(failures == 0 ? 0 : 1);
}
