#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "runtime/launch.h"
#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/cell_pack.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "runtime/gemm.cuh"
#include "spark_dsv41_flash_stagepack_format.h"
#include "spark_dsv41_flash_stage_kernels.cuh"

#define D41_TP SPARK_DSV41_FLASH_STAGE_TP_DEGREE
#define D41_HIDDEN SPARK_DSV41_FLASH_STAGE_HIDDEN
#define D41_LAYERS 40u
#define D41_VOCAB 129280u
#define D41_HEADS 64u
#define D41_HEAD_DIM 512u
#define D41_ROPE 64u
#define D41_QLORA 1280u
#define D41_OLORA 1024u
#define D41_OGROUPS 8u
#define D41_WINDOW 128u
#define D41_INDEX_HEADS 32u
#define D41_INDEX_DIM 128u
#define D41_INDEX_TOPK 512u
#define D41_EXPERTS 384u
#define D41_TOPK 6u
#define D41_MOE_INTER 2304u
#define D41_ROUTE_SCALE 1.5f
#define D41_SWIGLU_LIMIT 10.0f
#define D41_EPS 1e-20f
#define D41_HC_ITERATIONS 20u
#define D41_HC_EPS 1e-6f
#define D41_ROPE_THETA 10000.0
#define D41_COMPRESS_THETA 160000.0
#define D41_YARN_FACTOR 16.0
#define D41_YARN_ORIGINAL 65536.0
#define D41_BETA_FAST 32.0
#define D41_BETA_SLOW 1.0
#define D41_CANDIDATE_SOURCE 20u
#define D41_CANDIDATE_BLOCKS 2048u
#define D41_CANDIDATE_ROWS (D41_CANDIDATE_BLOCKS * D41_GRAIN)
#define D41_ENGRAM_HEAD_DIM 256u
#define D41_ENGRAM_COLUMNS 24u
#define D41_ENGRAM_MULTIPLIERS 4u
#define D41_ENGRAM_ORDERS 3u
#define D41_ENGRAM_HEADS 8u
#define D41_ENGRAM_KV_ROWS ((D41K_HC + 1u) * D41_HIDDEN)
#define D41_ENGRAM_EMBED (D41_ENGRAM_COLUMNS * D41_ENGRAM_HEAD_DIM)
#define D41_ENGRAM_KV_LOCAL (D41_ENGRAM_KV_ROWS / D41_TP)
#define D41_FP8_BLOCK 32u
#define D41_SHARED_GRANULE 128u
#define D41_THREADS 256u
#define D41_SLOT_BYTES SPARK_DSV41_FLASH_STAGE_SLOT_BYTES
#define D41_PAGE_TOKENS SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS
#define D41_GRAIN SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN
static_assert(D41_GRAIN == 8u, "a DSV4.1 candidate block is one context shard grain, so each block lives on one rank");
#define D41_A2A_HEADS SPARK_DSV41_FLASH_STAGE_A2A_HEADS
#define D41_LOCAL_HEADS (D41_HEADS / D41_TP)
#define D41_LOCAL_INDEX_HEADS (D41_INDEX_HEADS / D41_TP)
#define D41_LOCAL_EXPERTS (D41_EXPERTS / D41_TP)
#define D41_WO_A_ROWS (D41_OGROUPS * D41_OLORA / D41_TP)
#define D41_RING_ROWS (D41_WINDOW / D41_TP)
#define D41_VERIFY_ROWS SPARK_DSV41_FLASH_STAGE_VERIFY_ROWS
#define D41_PAIR_CACHES 3u
#define D41_SELECTION_CHUNK 64u
#define D41_INDEX_PACK 264u
#define D41_ENGRAM_TABLES_MAGIC 0x45313444u
#define D41_GEMM_TILE_N 128u
#define D41_SKINNY_EXPERT_ROWS 16u
#define D41_TAIL_MIN_ROWS (LM_SKINNY_ROWS_WIDE > D41_SKINNY_EXPERT_ROWS ? LM_SKINNY_ROWS_WIDE : D41_SKINNY_EXPERT_ROWS)
#define D41_RECORD_ROW_ELEMENTS (D41_LOCAL_HEADS * D41K_RECORD * (uint32_t)(sizeof(float) / sizeof(uint16_t)))
#define D41_ENGRAM_SPLIT_ROWS 16u
#define D41_ENGRAM_HALF (D41_ENGRAM_EMBED / 2u)

static_assert(D41_RECORD_ROW_ELEMENTS <= D41_HIDDEN && D41_RECORD_ROW_ELEMENTS % 8u == 0u,
	"one row of attention records fits a collective row and keeps 16-byte copies");
static_assert(D41_ENGRAM_EMBED % 2u == 0u && D41_ENGRAM_HALF <= D41_HIDDEN && D41_ENGRAM_HALF % 8u == 0u,
	"an engram embedding row splits into two collective rows");
static_assert(D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK <= D41_HIDDEN && D41_INDEX_PACK % 8u == 0u,
	"the index query pack rides in the query all-gather row and keeps 16-byte copies");
static_assert(D41_ENGRAM_KV_ROWS % (D41_TP * D41_FP8_BLOCK) == 0u, "the engram key/value rows split into whole fp8 blocks per rank");
static_assert(D41_MOE_INTER % D41_SHARED_GRANULE == 0u && D41_SHARED_GRANULE % D41_FP8_BLOCK == 0u,
	"the shared expert width splits into whole fp8 blocks and whole dense GEMM k tiles");

using D41Ratio2Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS / 2u, true>;
using D41Ratio1Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS, true>;

typedef SparkDsv41FlashStagePackHeader D41PackHeader;
typedef SparkDsv41FlashStagePackEntry D41PackEntry;

typedef struct D41Fp8Linear
{
	const uint8_t *weight, *scale;
	const uint16_t *wide;
} D41Fp8Linear;

typedef struct D41Layer
{
	const uint16_t *attn_norm, *ffn_norm, *q_norm, *kv_norm;
	D41Fp8Linear q_a, q_b, kv_a, o_a, o_b, idx_q_b;
	const uint16_t *idx_wk, *idx_wp, *idx_kn, *c_wkv, *c_wgate, *c_norm;
	const float *sink, *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
	const uint16_t *router;
	const float *router_bias;
	const uint8_t *e_w1, *e_w1_scale, *e_w2, *e_w2_scale, *e_w3, *e_w3_scale;
	const uint8_t *e_w1_cells, *e_w2_cells, *e_w3_cells;
	D41Fp8Linear sh_w1, sh_w2, sh_w3;
} D41Layer;

typedef struct D41Engram
{
	const uint8_t *payload, *scale;
	D41Fp8Linear wkv;
	const uint16_t *q_weight, *k_weight;
	uint64_t first_row, local_rows;
	int64_t *multipliers, *primes, *offsets;
} D41Engram;

typedef struct D41ModelState
{
	const SparkStageRunnerServices *services;
	uint32_t tp_rank, max_rows, slots, context_slots, max_context;
	FILE *pack_file;
	D41PackHeader header;
	D41PackEntry *entries;
	SparkWeightdLazyPack *engram_pack;
	char engram_path[1024], engram_sha256[72];
	uint64_t engram_bytes;
	D41PackEntry engram_entries[8];
	int32_t engram_pad;
	int32_t *token_map;
	D41Layer layers[D41_LAYERS];
	D41Engram engram[2];
	double *full_frequency, *compress_frequency;
	uint8_t *fp8_side;
	uint64_t fp8_side_bytes;
	uint8_t *cells;
	uint64_t cells_bytes;
	void *arena;
	uint64_t arena_bytes;
	uint64_t record_chunk_floats, a2a_units;
	uint32_t local_score_stride, global_score_stride, step_score_stride, step_blocks;
	uint32_t candidate_blocks, local_block_stride, global_block_stride, local_block_keep, global_block_keep, block_round_parts;
	uint16_t *streams, *x, *xq, *q_latent, *qr, *qrq, *q_local, *q_gathered, *kv, *window_kv;
	float *mixes, *pre_attn, *post_attn, *comb_attn, *post_ffn, *comb_ffn, *pre_head;
	float *c_kv, *c_score;
	uint16_t *c_latent, *c_latent_store, *idx_key, *idx_pack_local, *idx_query, *idx_w_full;
	uint32_t *emit_row, *bound_ratio2, *bound_ratio1, *length_ratio2, *length_ratio1, *row_lane, *row_ordinal, *tail_row_begin;
	uint32_t tail_target, tail_first_position, compress_done;
	float *local_scores, *topk_values;
	uint32_t *local_selected, *selected, *topk_positions;
	uint64_t topk_entries;
	uint2 *candidates, *candidates_gathered;
	float *block_scores;
	uint32_t *block_local_selected, *block_selected;
	uint2 *block_candidates, *block_gathered;
	uint8_t *block_mask;
	uint16_t *verify_window;
	float *verify_c_kv, *verify_c_score;
	float *records_send, *records_received;
	uint16_t *attn_merged, *attn_gathered, *group_input, *wo_a_out, *partial;
	float *pre_init;
	uint32_t max_positions;
	float *router_logits, *route_weight, *packed_weight;
	uint32_t *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	uint16_t *expert_gate, *expert_up, *expert_act, *expert_out, *moe_out, *shared_gate, *shared_up, *shared_act, *shared_out;
	int64_t *engram_ids;
	uint16_t *engram_embed, *engram_kv, *engram_kv_local, *engram_kv_gathered;
	uint32_t shared_first, shared_width;
	uint8_t *recurrent;
	uint64_t ring_bytes, compressor_bytes, history_bytes, recurrent_bytes;
	LmKvAccessError *kv_error;
	LmKvShardView latent_views[4], index_views[4];
	FILE *dump;
	uint16_t *dump_streams;
	uint32_t *dump_routes;
	float *dump_weights;
	SparkKvShard shard;
	uint32_t kv_attached;
} D41ModelState;

static uint32_t D41Ratio(uint32_t layer)
{
	return layer < 2u ? 0u : layer < 20u ? 2u : 1u;
}

static uint32_t D41KvSource(uint32_t layer)
{
	return layer == 2u || layer == 8u || layer == 14u || layer == 20u;
}

static uint32_t D41IndexSource(uint32_t layer)
{
	return D41KvSource(layer) || layer == 24u || layer == 28u || layer == 32u || layer == 36u;
}

static uint32_t D41Owner(uint32_t layer)
{
	return layer < 8u ? 2u : layer < 14u ? 8u : layer < 20u ? 14u : 20u;
}

static uint32_t D41CacheIndex(uint32_t layer)
{
	const uint32_t owner = D41Owner(layer);
	return owner == 2u ? 0u : owner == 8u ? 1u : owner == 14u ? 2u : 3u;
}

static int32_t D41EngramIndex(uint32_t layer)
{
	return layer == 1u ? 0 : layer == 14u ? 1 : -1;
}

static uint32_t D41Blocks(uint64_t elements)
{
	const uint64_t blocks = (elements + D41_THREADS - 1u) / D41_THREADS;
	return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
}

static SparkStatus D41LaunchedAt(uint32_t line)
{
	const cudaError_t error = cudaPeekAtLastError();
	if ( error == cudaSuccess )
		return SPARK_STATUS_OK;
	fprintf(stderr, "sparkpipe_dsv41_flash: CUDA error seen at line %u: %s\n", line, cudaGetErrorString(error));
	return SPARK_STATUS_INTERNAL_ERROR;
}

#define D41Launched() D41LaunchedAt(__LINE__)

static int32_t D41Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32,
	uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	if ( rows <= LM_SKINNY_ROWS_WIDE )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = source;
		args.output_bf16 = destination;
		args.output_f32 = destination_f32;
		args.scale = LmScaleTensorNone();
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		const int32_t status = LmSkinnyLaunch<LmBf16Format>(&args, stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	const int32_t status = LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows,
		input_dimension, output_dimension, 0u, 0u, multiprocessors, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status;
	D41NaiveLinearKernel<<<dim3((output_dimension + 7u) / 8u, rows), 256u, 0, stream>>>(weight, source, destination, destination_f32,
		input_dimension, output_dimension);
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t D41Linear(const D41Fp8Linear *linear, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
{
	const LmScaleTensor scale = LmScaleTensorBlockUe8m0(linear->scale, 1u, output_dimension, input_dimension, 1u, LM_MXFP8_GROUP);
	if ( rows <= LM_SKINNY_ROWS_WIDE )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = linear->weight;
		args.activation = source;
		args.output_bf16 = destination;
		args.scale = scale;
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		const int32_t status = LmSkinnyLaunch<LmMxfp8>(&args, stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	return LmStreamGemmDense<LmBf16Block32>(linear->wide, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension,
		output_dimension, 0u, 0u, multiprocessors, stream);
}

static LmSkinnyDenseTarget D41Target(const D41Fp8Linear *linear, uint16_t *destination, uint32_t input_dimension, uint32_t output_dimension)
{
	LmSkinnyDenseTarget target;
	memset(&target, 0, sizeof(target));
	target.weight = linear->weight;
	target.output_bf16 = destination;
	target.output_dimension = output_dimension;
	target.scale = LmScaleTensorBlockUe8m0(linear->scale, 1u, output_dimension, input_dimension, 1u, LM_MXFP8_GROUP);
	return target;
}

static int32_t D41LinearPair(const D41Fp8Linear *first, const D41Fp8Linear *second, const uint16_t *source, uint16_t *first_destination,
	uint16_t *second_destination, uint32_t rows, uint32_t input_dimension, uint32_t first_dimension, uint32_t second_dimension,
	uint32_t multiprocessors, cudaStream_t stream)
{
	if ( rows <= LM_SKINNY_ROWS_WIDE )
	{
		const LmSkinnyDenseTarget targets[2] = { D41Target(first, first_destination, input_dimension, first_dimension),
			D41Target(second, second_destination, input_dimension, second_dimension) };
		const int32_t status = LmSkinnyDenseMulti<LmMxfp8>(targets, 2u, source, rows, input_dimension, stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	const int32_t status = D41Linear(first, source, first_destination, rows, input_dimension, first_dimension, multiprocessors, stream);
	return status != LM_LAUNCH_OK ? status :
		D41Linear(second, source, second_destination, rows, input_dimension, second_dimension, multiprocessors, stream);
}

static SparkStatus D41Round(D41ModelState *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL,
		rows, row_elements, local, full);
}

static void D41RmsNorm(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, uint32_t width, cudaStream_t stream)
{
	LmFusedResidualRmsNormKernel<D41_THREADS,uint16_t><<<rows, D41_THREADS, (width + 8u) * sizeof(float), stream>>>(
		(uint16_t *)input, (const uint16_t *)0, weight, (uint16_t *)0, output, width, width, D41_EPS);
}

static void D41Fp8Qdq(uint16_t *data, uint64_t elements, cudaStream_t stream)
{
	(void)SparkDsv41FlashLaunchFp8Qdq(stream, data, elements);
}

static void D41Fp8QdqCopy(const uint16_t *source, uint16_t *data, uint64_t elements, cudaStream_t stream)
{
	(void)SparkDsv41FlashLaunchFp8QdqCopy(stream, source, data, elements);
}

static void D41Rope(uint16_t *data, uint32_t rows, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *positions, int32_t shift, const double *frequency, uint32_t inverse, const uint32_t *row_valid, cudaStream_t stream)
{
	D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS, 0, stream>>>(data, heads, head_stride, rope_offset, positions, shift,
		frequency, inverse, row_valid);
}

static SparkStatus D41HcMixes(D41ModelState *state, const uint16_t *streams, const float *fn, const float *base, const float *scale,
	float *pre, float *post, float *comb, uint32_t rows, cudaStream_t stream)
{
	D41HcMixPartialKernel<<<dim3(D41K_HC_TASKS, rows), D41K_THREADS, 0, stream>>>(streams, fn, state->mixes, D41_HIDDEN);
	D41HcSinkhornRowsKernel<<<rows, 32u, 0, stream>>>(state->mixes, scale, base, pre, post, comb, D41_HIDDEN, D41_EPS, D41_HC_ITERATIONS,
		D41_HC_EPS);
	return D41Launched();
}

static SparkStatus D41EngramApply(D41ModelState *state, uint32_t which, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows;
	const D41Engram *engram = &state->engram[which];
	const uint32_t split = rows <= D41_ENGRAM_SPLIT_ROWS ? 1u : 0u;
	const uint32_t ar_rows = split != 0u ? 2u * rows : (rows * D41_ENGRAM_EMBED + D41_HIDDEN - 1u) / D41_HIDDEN;
	const uint32_t ar_width = split != 0u ? D41_ENGRAM_HALF : D41_HIDDEN;
	SparkStatus status;
	LmNgramHashKernel<<<(rows + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->positions,
		state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index,
		(const int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		engram->multipliers, engram->primes, engram->offsets, state->engram_ids, rows, D41_ENGRAM_ORDERS, D41_ENGRAM_HEADS, state->engram_pad,
		-1, 0u);
	cudaMemsetAsync(state->engram_embed, 0, (uint64_t)ar_rows * ar_width * sizeof(uint16_t), stream);
	LmNgramGatherE8m0Kernel<<<dim3(rows, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(state->engram_ids, engram->payload, engram->scale,
		state->engram_embed, engram->first_row, engram->local_rows, D41_ENGRAM_COLUMNS, D41_ENGRAM_HEAD_DIM);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, ar_rows, ar_width, state->engram_embed,
			state->engram_embed);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41Fp8Qdq(state->engram_embed, (uint64_t)rows * D41_ENGRAM_EMBED, stream);
	if ( D41Linear(&engram->wkv, state->engram_embed, state->engram_kv_local, rows, D41_ENGRAM_EMBED, D41_ENGRAM_KV_LOCAL,
		step->multiprocessors, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_ENGRAM_KV_LOCAL, state->engram_kv_local,
		state->engram_kv_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41GroupInputKernel<<<dim3(D41Blocks(D41_ENGRAM_KV_ROWS), rows), D41_THREADS, 0, stream>>>(state->engram_kv_gathered, state->engram_kv,
		rows, D41_ENGRAM_KV_LOCAL, 0u, D41_TP);
	if ( SparkDsv41FlashLaunchEngramGate(stream, state->streams, state->engram_kv, engram->q_weight, engram->k_weight, rows, D41K_HC,
		D41_HIDDEN, D41_EPS) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

template<class Geometry>
static void D41StoreCompressed(D41ModelState *state, uint32_t cache, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	D41StoreCompressedKernel<Geometry><<<step->rows, D41_THREADS, 0, stream>>>(state->latent_views[cache], state->c_latent_store,
		state->idx_key, state->emit_row, step->sequence_of_row, D41_INDEX_DIM);
}

static SparkStatus D41Compress(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, ratio = D41Ratio(index), cache = D41CacheIndex(index), sms = step->multiprocessors;
	float *compressor_state = (float *)(state->recurrent + state->ring_bytes) + (uint64_t)cache * 2u * D41_HEAD_DIM;
	const uint32_t state_stride = (uint32_t)(state->recurrent_bytes / sizeof(float));
	if ( D41Project(layer->c_wkv, state->x, 0, state->c_kv, rows, D41_HIDDEN, D41_HEAD_DIM, sms, stream) != LM_LAUNCH_OK ||
		(ratio > 1u && D41Project(layer->c_wgate, state->x, 0, state->c_score, rows, D41_HIDDEN, D41_HEAD_DIM, sms, stream) != LM_LAUNCH_OK) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41CompressorRowsKernel<<<rows, D41_THREADS, 0, stream>>>(state->c_kv, state->c_score, state->c_latent, state->emit_row,
		compressor_state, step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices,
		step->recurrent_index, ratio, state_stride, D41_HEAD_DIM);
	if ( ratio > 1u && step->commit == 0u )
	{
		if ( rows > D41_VERIFY_ROWS || cache >= D41_PAIR_CACHES )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		cudaMemcpyAsync(state->verify_c_kv + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM, state->c_kv,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
		cudaMemcpyAsync(state->verify_c_score + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM, state->c_score,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
	}
	if ( ratio > 1u )
		D41CompressorSaveKernel<<<rows, D41_THREADS, 0, stream>>>(state->c_kv, state->c_score, compressor_state, step->positions,
			state->row_lane, state->row_ordinal, step->sequence_row_begin, step->recurrent_index, rows, ratio, state_stride, D41_HEAD_DIM,
			step->commit, (const uint32_t *)0);
	D41RmsNorm(state->c_latent, layer->c_norm, state->c_latent, rows, D41_HEAD_DIM, stream);
	if ( D41Project(layer->idx_wk, state->c_latent, state->idx_key, 0, rows, D41_HEAD_DIM, D41_INDEX_DIM, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNorm(state->idx_key, layer->idx_kn, state->idx_key, rows, D41_INDEX_DIM, stream);
	D41Rope(state->idx_key, rows, 1u, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions, 1 - (int32_t)ratio,
		state->compress_frequency, 0u, state->emit_row, stream);
	SparkDsv41FlashFp4Pow2QdqKernel<<<(rows * (D41_INDEX_DIM / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_key,
		rows * (D41_INDEX_DIM / 32u), 32u);
	cudaMemcpyAsync(state->c_latent_store, state->c_latent, (uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	D41Rope(state->c_latent_store, rows, 1u, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 1 - (int32_t)ratio,
		state->compress_frequency, 0u, state->emit_row, stream);
	(void)SparkDsv41FlashLaunchKvFp4Qdq(stream, state->c_latent_store, D41_HEAD_DIM, rows, D41_HEAD_DIM);
	if ( cache < 3u )
		D41StoreCompressed<D41Ratio2Kv>(state, cache, step, stream);
	else
		D41StoreCompressed<D41Ratio1Kv>(state, cache, step, stream);
	return D41Launched();
}

static SparkStatus D41CandidateBlocks(D41ModelState *state, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, keep = state->local_block_keep;
	const SparkKvShard blocks = { D41_TP, state->shard.rank, 1u };
	SparkStatus status;
	if ( SparkDsv41FlashLaunchShardBlockMax(stream, state->local_scores, state->step_score_stride, state->shard, step->sequence_of_row,
			length, bound, state->block_scores, state->local_block_stride, rows) != cudaSuccess ||
		LmTopkExactLaunch<D41_THREADS>(state->block_scores, rows, state->local_block_stride, keep, LM_TOPK_EXACT_CHUNK,
			LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->block_local_selected, stream) != cudaSuccess ||
		LmIndexShardCandidatePackLaunch<D41_THREADS>(blocks, state->block_scores, state->local_block_stride, state->block_local_selected,
			keep, rows, state->block_candidates, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( uint32_t first = 0u; first < rows; first += D41_SELECTION_CHUNK )
	{
		const uint32_t count = rows - first < D41_SELECTION_CHUNK ? rows - first : D41_SELECTION_CHUNK;
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, count * state->block_round_parts,
			keep * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)) / state->block_round_parts, state->block_candidates + (uint64_t)first * keep,
			state->block_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_CANDIDATE_BLOCKS>(state->block_gathered, (uint64_t)count * keep, D41_TP, keep, count,
				state->global_block_stride, state->global_block_keep, 0u, state->block_selected, stream) != cudaSuccess ||
			SparkDsv41FlashLaunchShardBlockMask(stream, state->global_block_stride, state->block_selected, state->global_block_keep,
				blocks, state->block_mask + (uint64_t)first * state->local_block_stride, state->local_block_stride, count) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return D41Launched();
}

static SparkStatus D41IndexQuery(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	uint16_t *packed, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors;
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	if ( state->candidate_blocks != 0u && index >= D41_CANDIDATE_SOURCE && D41Ratio(index) != 1u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( D41Project(layer->idx_wp, state->x, state->idx_w_full, 0, rows, D41_HIDDEN, D41_LOCAL_INDEX_HEADS, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->idx_pack_local, rows, D41_LOCAL_INDEX_HEADS, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions, 0,
		state->compress_frequency, 0u, 0, stream);
	SparkDsv41FlashFp4Pow2QdqKernel<<<(rows * (local_width / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_pack_local,
		rows * (local_width / 32u), 32u);
	D41ScaleRowsKernel<<<D41Blocks((uint64_t)rows * D41_LOCAL_INDEX_HEADS), D41_THREADS, 0, stream>>>(state->idx_w_full,
		(uint64_t)rows * D41_LOCAL_INDEX_HEADS, (float)(1.0 / sqrt((double)D41_INDEX_DIM) / sqrt((double)D41_INDEX_HEADS)));
	D41IndexPackKernel<<<rows, D41_THREADS, 0, stream>>>(state->idx_pack_local, state->idx_w_full, packed, local_width,
		D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
	return D41Launched();
}

template<class Geometry>
static SparkStatus D41Select(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, const uint16_t *packed, uint64_t rank_stride, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index);
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	SparkStatus status;
	D41IndexUnpackKernel<<<dim3(rows, D41_TP), D41_THREADS, 0, stream>>>(packed, rank_stride, state->idx_query, state->idx_w_full,
		local_width, D41_LOCAL_INDEX_HEADS, D41_INDEX_PACK);
	if ( LmWeightedSparseScoreLaunch<Geometry,D41_INDEX_HEADS,D41_INDEX_DIM,LmKvShardView,true>(state->idx_query, state->idx_w_full,
		state->index_views[cache], step->sequence_of_row, length, bound, rows, state->step_score_stride, 1.0f, state->local_scores,
		stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->step_blocks != 0u && index == D41_CANDIDATE_SOURCE )
	{
		status = D41CandidateBlocks(state, step, bound, length, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	else if ( state->step_blocks != 0u && index > D41_CANDIDATE_SOURCE &&
		SparkDsv41FlashLaunchShardBlockApply(stream, state->local_scores, state->step_score_stride, D41_GRAIN, state->block_mask,
			state->local_block_stride, rows) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( LmTopkExactLaunch<D41_THREADS>(state->local_scores, rows, state->step_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK,
			LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->local_selected, stream) != cudaSuccess ||
		LmIndexShardCandidatePackLaunch<D41_THREADS>(state->index_views[cache].shard, state->local_scores, state->step_score_stride,
			state->local_selected, D41_INDEX_TOPK, rows, state->candidates, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
		D41_INDEX_TOPK * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->candidates, state->candidates_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LmIndexShardCandidateTopkLaunch<D41_THREADS,D41_INDEX_TOPK>(state->candidates_gathered, (uint64_t)rows * D41_INDEX_TOPK, D41_TP,
			D41_INDEX_TOPK, rows, state->global_score_stride, D41_INDEX_TOPK, 1u, state->selected, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static uint64_t D41DestinationStride(uint32_t rows)
{
	return (uint64_t)rows * D41_LOCAL_HEADS * D41K_RECORD;
}

template<class Geometry>
static SparkStatus D41Partials(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, uint64_t query_rank_stride, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index);
	if ( LmLatentShardPartialHeadsLaunch<Geometry,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_gathered,
		query_rank_stride, D41_LOCAL_HEADS, D41_HEADS, step->sequence_of_row, length, bound, state->selected,
		D41_INDEX_TOPK, D41_INDEX_TOPK, 1.0f / sqrtf((float)D41_HEAD_DIM), state->records_send, D41DestinationStride(rows), rows, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41Attention(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, ratio = D41Ratio(index);
	const double *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint64_t destination_stride = D41DestinationStride(rows);
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t local_width = D41_LOCAL_HEADS * D41_HEAD_DIM;
	const uint32_t indexed = ratio > 0u && D41IndexSource(index) ? 1u : 0u;
	const uint32_t gather_width = local_width + (indexed != 0u ? D41_INDEX_PACK : 0u);
	const uint64_t query_rank_stride = (uint64_t)rows * gather_width;
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	SparkStatus status;
	D41Fp8QdqCopy(state->x, state->xq, (uint64_t)rows * D41_HIDDEN, stream);
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, state->xq, state->q_latent, state->kv, rows, D41_HIDDEN, D41_QLORA, D41_HEAD_DIM, sms, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNorm(state->q_latent, layer->q_norm, state->qr, rows, D41_QLORA, stream);
	D41Fp8QdqCopy(state->qr, state->qrq, (uint64_t)rows * D41_QLORA, stream);
	if ( (indexed != 0u ? D41LinearPair(&layer->q_b, &layer->idx_q_b, state->qrq, state->q_local, state->idx_pack_local, rows, D41_QLORA,
		local_width, D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, sms, stream) :
		D41Linear(&layer->q_b, state->qrq, state->q_local, rows, D41_QLORA, local_width, sms, stream)) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	D41RmsNorm(state->kv, layer->kv_norm, state->kv, rows, D41_HEAD_DIM, stream);
	D41Rope(state->kv, rows, 1u, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	D41Fp8QdqCopy(state->kv, state->window_kv, (uint64_t)rows * D41_HEAD_DIM, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) && state->compress_done == 0u )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK && indexed != 0u )
		status = D41IndexQuery(state, layer, index, step, state->q_local + (uint64_t)rows * local_width, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, gather_width, state->q_local, state->q_gathered);
	if ( status == SPARK_STATUS_OK )
		status = D41Launched();
	if ( status == SPARK_STATUS_OK && indexed != 0u )
	{
		const uint16_t *packed = state->q_gathered + (uint64_t)rows * local_width;
		status = D41CacheIndex(index) < 3u ? D41Select<D41Ratio2Kv>(state, index, step, bound, length, packed, query_rank_stride, stream) :
			D41Select<D41Ratio1Kv>(state, index, step, bound, length, packed, query_rank_stride, stream);
	}
	if ( status == SPARK_STATUS_OK && ratio > 0u )
		status = D41CacheIndex(index) < 3u ? D41Partials<D41Ratio2Kv>(state, index, step, bound, length, query_rank_stride, stream) :
			D41Partials<D41Ratio1Kv>(state, index, step, bound, length, query_rank_stride, stream);
	else if ( status == SPARK_STATUS_OK )
	{
		D41InitRecordsKernel<<<D41Blocks((uint64_t)rows * D41_HEADS), D41_THREADS, 0, stream>>>(state->records_send, destination_stride,
			rows, D41_LOCAL_HEADS, D41_TP);
		status = D41Launched();
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	D41WindowFoldKernel<<<dim3(rows, D41_HEADS), D41K_THREADS, 0, stream>>>(state->records_send, destination_stride, D41_LOCAL_HEADS,
		state->q_gathered, query_rank_stride, state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS,
		D41_WINDOW, D41_TP, state->tp_rank,
		1.0f / sqrtf((float)D41_HEAD_DIM));
	status = D41Launched();
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( step->commit == 0u )
	{
		if ( rows > D41_VERIFY_ROWS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		cudaMemcpyAsync(state->verify_window + (uint64_t)index * D41_VERIFY_ROWS * D41_HEAD_DIM, state->window_kv,
			(uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	}
	D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_TP,
		state->tp_rank, rows, step->commit, (const uint32_t *)0);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, rows, D41_RECORD_ROW_ELEMENTS, state->records_send,
			state->records_received);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41MergeSinkKernel<<<dim3(rows, D41_LOCAL_HEADS), D41_THREADS, 0, stream>>>(state->records_received, destination_stride, D41_TP,
		D41_LOCAL_HEADS, layer->sink, state->attn_merged);
	D41Rope(state->attn_merged, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 1u, 0, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_width, state->attn_merged,
			state->attn_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41GroupInputKernel<<<dim3(D41Blocks(2u * local_width), rows), D41_THREADS, 0, stream>>>(state->attn_gathered, state->group_input, rows,
		local_width, (state->tp_rank / 2u) * 2u, 2u);
	D41Fp8Qdq(state->group_input, (uint64_t)rows * 2u * local_width, stream);
	if ( D41Linear(&layer->o_a, state->group_input, state->wo_a_out, rows, 2u * local_width, D41_WO_A_ROWS, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Fp8Qdq(state->wo_a_out, (uint64_t)rows * D41_WO_A_ROWS, stream);
	if ( D41Linear(&layer->o_b, state->wo_a_out, state->partial, rows, D41_WO_A_ROWS, D41_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41DumpRows(D41ModelState *state, uint32_t layer, uint32_t kind, const uint16_t *rows_bf16, uint32_t rows,
	uint32_t width, cudaStream_t stream);

static int32_t D41Experts(const uint8_t *cells, const uint16_t *activation, uint16_t *output, const D41ModelState *state, uint32_t rows,
	uint32_t routes, uint32_t packed, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	const uint32_t *source = packed != 0u ? (const uint32_t *)0 : state->route_source_token;
	if ( rows <= D41_SKINNY_EXPERT_ROWS )
		return LmSkinnyCellExperts(cells, activation, output, state->group_row_offset, source, D41_LOCAL_EXPERTS, routes, rows, packed,
			input_dimension, output_dimension, LM_SKINNY_CELL_TILE_K, stream);
	return LmCellMmaExperts(cells, activation, output, state->group_row_offset, source, D41_LOCAL_EXPERTS, routes, packed, input_dimension,
		output_dimension, LM_SKINNY_CELL_TILE_K, stream);
}

static SparkStatus D41Moe(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * D41_TOPK;
	if ( D41Project(layer->router, state->x, 0, state->router_logits, rows, D41_HIDDEN, D41_EXPERTS, sms, stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, state->router_logits, D41_EXPERTS, state->route_expert,
			state->route_weight, layer->router_bias, 0, D41_ROUTE_SCALE, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->dump != 0 &&
		(cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_routes, state->route_expert, (size_t)routes * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(state->dump_weights, state->route_weight, (size_t)routes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
		routes, state->tp_rank * D41_LOCAL_EXPERTS, D41_LOCAL_EXPERTS);
	if ( LmRouteBuild<D41_THREADS,D41_LOCAL_EXPERTS + 1u>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset,
		state->route_packed_row, state->route_source_token, D41_MOE_INTER, D41_HIDDEN, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down,
		stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
	D41Fp8QdqCopy(state->x, state->xq, (uint64_t)rows * D41_HIDDEN, stream);
	if ( D41Experts(layer->e_w1_cells, state->xq, state->expert_gate, state, rows, routes, 0u, D41_HIDDEN, D41_MOE_INTER, stream) != LM_LAUNCH_OK ||
		D41Experts(layer->e_w3_cells, state->xq, state->expert_up, state, rows, routes, 0u, D41_HIDDEN, D41_MOE_INTER, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	LmSwigluPackedKernel<<<dim3(D41Blocks(D41_MOE_INTER), routes), D41_THREADS, 0, stream>>>(state->expert_gate, state->expert_up,
		state->packed_weight, state->expert_act, D41_MOE_INTER, D41_SWIGLU_LIMIT);
	D41Fp8Qdq(state->expert_act, (uint64_t)routes * D41_MOE_INTER, stream);
	if ( D41Experts(layer->e_w2_cells, state->expert_act, state->expert_out, state, rows, routes, 1u, D41_MOE_INTER, D41_HIDDEN, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( D41LinearPair(&layer->sh_w1, &layer->sh_w3, state->xq, state->shared_gate, state->shared_up, rows, D41_HIDDEN, state->shared_width,
		state->shared_width, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41SwigluRowsKernel<<<dim3(D41Blocks(state->shared_width), rows), D41_THREADS, 0, stream>>>(state->shared_gate, state->shared_up,
		state->shared_act, state->shared_width, D41_SWIGLU_LIMIT);
	D41Fp8Qdq(state->shared_act, (uint64_t)rows * state->shared_width, stream);
	if ( D41Linear(&layer->sh_w2, state->shared_act, state->shared_out, rows, state->shared_width, D41_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->dump != 0 && step->rows < 64u )
	{
		SparkStatus dumped = D41DumpRows(state, 99u, 8u, state->expert_gate, routes, D41_MOE_INTER, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 9u, state->expert_up, routes, D41_MOE_INTER, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 10u, state->expert_act, routes, D41_MOE_INTER, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 6u, state->expert_out, routes, D41_HIDDEN, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 11u, (const uint16_t *)state->route_packed_row, 1u, 2u * routes, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 12u, (const uint16_t *)state->route_expert, 1u, 2u * routes, stream);
		if ( dumped == SPARK_STATUS_OK )
			dumped = D41DumpRows(state, 99u, 13u, (const uint16_t *)state->packed_weight, 1u, 2u * routes, stream);
		if ( dumped != SPARK_STATUS_OK )
			return dumped;
	}
	LmMoeLocalFinalizeKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
		state->route_expert, state->shared_out, state->partial, D41_TOPK, D41_LOCAL_EXPERTS, D41_HIDDEN);
	return D41Launched();
}

static SparkStatus D41DumpRows(D41ModelState *state, uint32_t layer, uint32_t kind, const uint16_t *rows_bf16, uint32_t rows,
	uint32_t width, cudaStream_t stream)
{
	const uint32_t header[5] = { 0x58344c44u, layer, rows, kind, width };
	if ( state->dump == 0 )
		return SPARK_STATUS_OK;
	if ( cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_streams, rows_bf16, (size_t)rows * width * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	fwrite(header, sizeof(header), 1u, state->dump);
	fwrite(state->dump_streams, sizeof(uint16_t), (size_t)rows * width, state->dump);
	fflush(state->dump);
	return SPARK_STATUS_OK;
}

static SparkStatus D41LayerStep(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const float *pre_previous,
	cudaStream_t stream)
{
	const D41Layer *layer = &state->layers[index];
	const uint32_t rows = step->rows;
	const uint64_t stream_bytes = (uint64_t)rows * D41K_HC * D41_HIDDEN * sizeof(uint16_t);
	const int32_t engram = D41EngramIndex(index);
	SparkStatus status = SPARK_STATUS_OK;
	if ( engram >= 0 )
		status = D41EngramApply(state, (uint32_t)engram, step, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41HcMixes(state, state->streams, layer->hc_attn_fn, layer->hc_attn_base, layer->hc_attn_scale, state->pre_attn,
			state->post_attn, state->comb_attn, rows, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->streams, pre_previous, state->x, D41_HIDDEN);
	D41RmsNorm(state->x, layer->attn_norm, state->x, rows, D41_HIDDEN, stream);
	if ( index < 2u && (status = D41DumpRows(state, index, 3u, state->x, rows, D41_HIDDEN, stream)) != SPARK_STATUS_OK )
		return status;
	status = D41Attention(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 1u, state->partial, rows, D41_HIDDEN, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPostRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->partial, state->post_attn,
		state->comb_attn, state->streams, D41_HIDDEN);
	status = D41HcMixes(state, state->streams, layer->hc_ffn_fn, layer->hc_ffn_base, layer->hc_ffn_scale, state->pre_head,
		state->post_ffn, state->comb_ffn, rows, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_attn, state->x, D41_HIDDEN);
	D41RmsNorm(state->x, layer->ffn_norm, state->x, rows, D41_HIDDEN, stream);
	if ( index < 2u && (status = D41DumpRows(state, index, 4u, state->x, rows, D41_HIDDEN, stream)) != SPARK_STATUS_OK )
		return status;
	status = D41Moe(state, layer, step, stream);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 7u, state->partial, rows, D41_HIDDEN, stream);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 5u, state->shared_out, rows, D41_HIDDEN, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 2u, state->partial, rows, D41_HIDDEN, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPostRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->partial, state->post_ffn,
		state->comb_ffn, state->streams, D41_HIDDEN);
	if ( state->dump != 0 )
	{
		const uint32_t header[4] = { 0x44344c44u, index, rows, D41_TOPK };
		if ( cudaStreamSynchronize(stream) != cudaSuccess ||
			cudaMemcpy(state->dump_streams, state->streams, stream_bytes, cudaMemcpyDeviceToHost) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		fwrite(header, sizeof(header), 1u, state->dump);
		fwrite(state->dump_streams, 1u, (size_t)stream_bytes, state->dump);
		fwrite(state->dump_routes, sizeof(uint32_t), (size_t)rows * D41_TOPK, state->dump);
		fwrite(state->dump_weights, sizeof(float), (size_t)rows * D41_TOPK, state->dump);
		fflush(state->dump);
	}
	return D41Launched();
}

static SparkStatus D41DumpStep(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t header[3] = { 0x53344c44u, step->rows, step->sequences };
	const uint64_t words = 3u * (uint64_t)step->rows + step->sequences + 1u + step->sequences;
	uint32_t *host = (uint32_t *)malloc((size_t)words * sizeof(uint32_t));
	uint32_t *cursor = host;
	cudaError_t error = host == 0 ? cudaErrorMemoryAllocation : cudaStreamSynchronize(stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->positions, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_of_row, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_row_indices, (size_t)step->rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->rows;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->sequence_row_begin, (size_t)(step->sequences + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cursor += step->sequences + 1u;
	if ( error == cudaSuccess )
		error = cudaMemcpy(cursor, step->recurrent_index, (size_t)step->sequences * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess )
	{
		fwrite(header, sizeof(header), 1u, state->dump);
		fwrite(host, sizeof(uint32_t), (size_t)words, state->dump);
		fflush(state->dump);
	}
	free(host);
	return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus D41StepReach(D41ModelState *state, const SparkStageRunnerStep *step)
{
	const uint32_t uncaptured = step->commit != 0u && step->rows > step->sequences && step->host_positions != 0 ? 1u : 0u;
	uint32_t reach = state->max_positions, row, contiguous = 1u;
	state->tail_target = 0u;
	if ( uncaptured != 0u )
		for ( row = 0u, reach = 0u; row < step->rows; ++row )
		{
			reach = step->host_positions[row] + 1u > reach ? step->host_positions[row] + 1u : reach;
			contiguous &= step->host_positions[row] == step->host_positions[0] + row ? 1u : 0u;
		}
	if ( reach > state->max_positions )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->step_score_stride = ((SparkKvShardGatherKeys(state->shard, reach) + 63u) / 64u) * 64u;
	state->step_score_stride = state->step_score_stride < state->local_score_stride ? state->step_score_stride : state->local_score_stride;
	state->step_blocks = state->candidate_blocks != 0u && reach > D41_CANDIDATE_ROWS ? 1u : 0u;
	if ( uncaptured != 0u && contiguous != 0u && step->sequences == 1u && step->rows > D41_TAIL_MIN_ROWS && step->last_rows_only != 0u &&
		step->every_row_hidden == 0u && step->host_state_target != 0 && step->host_state_target[0] >= reach )
	{
		state->tail_target = step->host_state_target[0];
		state->tail_first_position = step->host_positions[0];
	}
	return SPARK_STATUS_OK;
}

static uint32_t D41TailFirstRow(const D41ModelState *state, uint32_t index, uint32_t rows)
{
	const uint32_t reach = D41_WINDOW + (D41_LAYERS - 1u - index) * (D41_WINDOW - 1u);
	const uint32_t first = state->tail_target > reach ? state->tail_target - reach : 0u;
	uint32_t skipped;
	if ( state->tail_target == 0u || index < D41_CANDIDATE_SOURCE || first <= state->tail_first_position )
		return 0u;
	skipped = first - state->tail_first_position < rows ? first - state->tail_first_position : rows;
	return skipped == rows || rows - skipped > D41_TAIL_MIN_ROWS ? skipped : rows - D41_TAIL_MIN_ROWS - 1u;
}

static void D41TailRebase(D41ModelState *state, uint32_t rows)
{
	state->streams += (uint64_t)rows * D41K_HC * D41_HIDDEN;
	state->pre_head += (uint64_t)rows * D41K_HC;
	state->selected += (uint64_t)rows * D41_INDEX_TOPK;
	if ( state->block_mask != 0 )
		state->block_mask += (uint64_t)rows * state->local_block_stride;
}

static SparkStatus D41TailStep(D41ModelState *state, const SparkStageRunnerStep *step, uint32_t first, SparkStageRunnerStep *tail,
	cudaStream_t stream)
{
	*tail = *step;
	tail->rows = step->rows - first;
	tail->hidden_bf16 = step->hidden_bf16 + (uint64_t)first * D41_HIDDEN;
	tail->positions = step->positions + first;
	tail->sequence_of_row = step->sequence_of_row + first;
	tail->sequence_row_begin = state->tail_row_begin;
	tail->token_ids = step->token_ids + first;
	tail->host_positions = step->host_positions + first;
	tail->host_sequence_of_row = step->host_sequence_of_row != 0 ? step->host_sequence_of_row + first : 0;
	if ( tail->rows == 0u )
		return SPARK_STATUS_OK;
	D41RowBeginKernel<<<1u, 1u, 0, stream>>>(state->tail_row_begin, tail->rows);
	D41CompressedEmitPositionKernel<<<(tail->rows + 255u) / 256u, 256u, 0, stream>>>(tail->positions, state->bound_ratio2, tail->rows, 2u);
	D41CompressedEmitPositionKernel<<<(tail->rows + 255u) / 256u, 256u, 0, stream>>>(tail->positions, state->bound_ratio1, tail->rows, 1u);
	LmRowLanesKernel<<<1u, 256u, 0, stream>>>(tail->sequence_row_begin, tail->sequence_row_indices, state->row_lane, state->row_ordinal);
	return D41Launched();
}

static SparkStatus D41TailCompress(D41ModelState *state, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const D41Layer *layer = &state->layers[D41_CANDIDATE_SOURCE];
	D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), step->rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_head, state->x, D41_HIDDEN);
	D41RmsNorm(state->x, layer->attn_norm, state->x, step->rows, D41_HIDDEN, stream);
	return D41Compress(state, layer, D41_CANDIDATE_SOURCE, step, stream);
}

static SparkStatus D41Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
{
	D41ModelState *state = (D41ModelState *)model;
	cudaStream_t stream = (cudaStream_t)stream_void;
	const uint32_t rows = step->rows;
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots || state->kv_attached == 0u || step->token_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->dump != 0 )
	{
		cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
		if ( cudaStreamIsCapturing(stream, &capturing) != cudaSuccess || capturing != cudaStreamCaptureStatusNone )
			return SPARK_STATUS_UNSUPPORTED;
		status = D41DumpStep(state, step, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	status = D41StepReach(state, step);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio2, rows, 2u);
	D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio1, rows, 1u);
	LmRowLanesKernel<<<step->sequences, 256u, 0, stream>>>(step->sequence_row_begin, step->sequence_row_indices, state->row_lane,
		state->row_ordinal);
	D41CompressedLengthKernel<<<(state->context_slots + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio2,
		state->context_slots, 2u);
	D41CompressedLengthKernel<<<(state->context_slots + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio1,
		state->context_slots, 1u);
	D41ReplicateStreamsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(step->hidden_bf16, state->streams, D41_HIDDEN);
	{
		uint16_t *const streams = state->streams;
		float *const pre_head = state->pre_head;
		uint32_t *const selected = state->selected;
		uint8_t *const block_mask = state->block_mask;
		SparkStageRunnerStep tail = *step;
		const SparkStageRunnerStep *current = step;
		uint32_t shifted = 0u, first;
		for ( uint32_t index = 0u; index < D41_LAYERS && status == SPARK_STATUS_OK; ++index )
		{
			first = D41TailFirstRow(state, index, rows);
			if ( index == D41_CANDIDATE_SOURCE && first != 0u )
			{
				status = D41TailCompress(state, step, stream);
				state->compress_done = 1u;
			}
			if ( status == SPARK_STATUS_OK && first != shifted )
			{
				D41TailRebase(state, first - shifted);
				shifted = first;
				status = D41TailStep(state, step, first, &tail, stream);
				current = &tail;
			}
			if ( status == SPARK_STATUS_OK && current->rows != 0u )
				status = D41LayerStep(state, index, current, index == 0u ? state->pre_init : state->pre_head, stream);
			state->compress_done = 0u;
			if ( status != SPARK_STATUS_OK )
				fprintf(stderr, "sparkpipe_dsv41_flash: layer %u failed status=%d rows=%u sequences=%u tail_rows=%u\n", index, (int)status, rows,
					step->sequences, current->rows);
			else if ( state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status == SPARK_STATUS_OK && current->rows != 0u )
			D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), current->rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_head,
				current->hidden_bf16, D41_HIDDEN);
		state->streams = streams;
		state->pre_head = pre_head;
		state->selected = selected;
		state->block_mask = block_mask;
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	LmNgramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->sequence_row_begin,
		step->sequence_row_indices, step->recurrent_index,
		(int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		step->sequences, D41_ENGRAM_ORDERS, step->commit, (const uint32_t *)0);
	return D41Launched();
}

static SparkStatus D41KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_pointer)
{
	D41ModelState *state = (D41ModelState *)model;
	cudaStream_t stream = (cudaStream_t)stream_pointer;
	const uint32_t rows = step->rows, state_stride = (uint32_t)(state->recurrent_bytes / sizeof(float));
	if ( state == 0 || kept == 0 || rows > D41_VERIFY_ROWS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
		D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->verify_window + (uint64_t)index * D41_VERIFY_ROWS * D41_HEAD_DIM,
			(uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM, step->positions, state->row_lane, state->row_ordinal,
			step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_TP,
			state->tp_rank, rows, 1u, kept);
	for ( uint32_t cache = 0u; cache < D41_PAIR_CACHES; ++cache )
		D41CompressorSaveKernel<<<rows, D41_THREADS, 0, stream>>>(state->verify_c_kv + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM,
			state->verify_c_score + (uint64_t)cache * D41_VERIFY_ROWS * D41_HEAD_DIM,
			(float *)(state->recurrent + state->ring_bytes) + (uint64_t)cache * 2u * D41_HEAD_DIM, step->positions, state->row_lane,
			state->row_ordinal, step->sequence_row_begin, step->recurrent_index, rows, 2u, state_stride, D41_HEAD_DIM, 1u, kept);
	LmNgramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->sequence_row_begin,
		step->sequence_row_indices, step->recurrent_index,
		(int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		step->sequences, D41_ENGRAM_ORDERS, 1u, kept);
	return D41Launched();
}

static void D41Frequencies(double *full, double *compress)
{
	for ( uint32_t pair = 0u; pair < D41K_ROPE_PAIRS; ++pair )
	{
		const double exponent = (double)(2u * pair) / (double)D41_ROPE;
		full[pair] = pow(D41_ROPE_THETA, -exponent);
		compress[pair] = pow(D41_COMPRESS_THETA, -exponent);
	}
	const double low_raw = D41_ROPE * log(D41_YARN_ORIGINAL / (D41_BETA_FAST * 2.0 * M_PI)) / (2.0 * log(D41_COMPRESS_THETA));
	const double high_raw = D41_ROPE * log(D41_YARN_ORIGINAL / (D41_BETA_SLOW * 2.0 * M_PI)) / (2.0 * log(D41_COMPRESS_THETA));
	const double low = floor(low_raw) > 0.0 ? floor(low_raw) : 0.0;
	const double high = ceil(high_raw) < (double)(D41_ROPE - 1u) ? ceil(high_raw) : (double)(D41_ROPE - 1u);
	for ( uint32_t pair = 0u; pair < D41K_ROPE_PAIRS; ++pair )
	{
		double ramp = ((double)pair - low) / (high - low > 1e-3 ? high - low : 1e-3);
		ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
		const double smooth = 1.0 - ramp;
		compress[pair] = compress[pair] / D41_YARN_FACTOR * (1.0 - smooth) + compress[pair] * smooth;
	}
}

static const D41PackEntry *D41Find(const D41ModelState *state, uint32_t kind, uint32_t layer)
{
	for ( uint32_t index = 0u; index < state->header.tensor_count; ++index )
		if ( state->entries[index].tensor_kind == kind && state->entries[index].layer_index == layer )
			return &state->entries[index];
	return 0;
}

static SparkStatus D41PlanArena(D41ModelState *state, uint64_t *bytes, uint8_t *base)
{
	const uint64_t r = state->max_rows, s = state->slots, routes = r * D41_TOPK;
	const uint64_t destination = D41DestinationStride(state->max_rows) * sizeof(float);
	const uint64_t engram_rows = (r * D41_ENGRAM_EMBED + D41_HIDDEN - 1u) / D41_HIDDEN;
	struct { void **target; uint64_t bytes; } plan[] =
	{
		{ (void **)&state->streams, r * D41K_HC * D41_HIDDEN * 2u },
		{ (void **)&state->x, r * D41_HIDDEN * 2u }, { (void **)&state->xq, r * D41_HIDDEN * 2u },
		{ (void **)&state->q_latent, r * D41_QLORA * 2u }, { (void **)&state->qr, r * D41_QLORA * 2u }, { (void **)&state->qrq, r * D41_QLORA * 2u },
		{ (void **)&state->q_local, r * (D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) * 2u },
		{ (void **)&state->q_gathered, D41_TP * r * (D41_LOCAL_HEADS * D41_HEAD_DIM + D41_INDEX_PACK) * 2u },
		{ (void **)&state->kv, r * D41_HEAD_DIM * 2u }, { (void **)&state->window_kv, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->mixes, r * D41K_HC_TASKS * D41K_HC_WARPS * 4u },
		{ (void **)&state->pre_attn, r * D41K_HC * 4u }, { (void **)&state->post_attn, r * D41K_HC * 4u }, { (void **)&state->comb_attn, r * 16u * 4u },
		{ (void **)&state->post_ffn, r * D41K_HC * 4u }, { (void **)&state->comb_ffn, r * 16u * 4u },
		{ (void **)&state->pre_head, r * D41K_HC * 4u }, { (void **)&state->pre_init, r * D41K_HC * 4u },
		{ (void **)&state->c_kv, r * D41_HEAD_DIM * 4u }, { (void **)&state->c_score, r * D41_HEAD_DIM * 4u },
		{ (void **)&state->c_latent, r * D41_HEAD_DIM * 2u }, { (void **)&state->c_latent_store, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->idx_key, r * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_pack_local, r * D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_query, r * D41_INDEX_HEADS * D41_INDEX_DIM * 2u }, { (void **)&state->idx_w_full, r * D41_INDEX_HEADS * 2u },
		{ (void **)&state->emit_row, r * 4u }, { (void **)&state->bound_ratio2, r * 4u }, { (void **)&state->bound_ratio1, r * 4u },
		{ (void **)&state->length_ratio2, s * 4u }, { (void **)&state->length_ratio1, s * 4u },
		{ (void **)&state->row_lane, r * 4u }, { (void **)&state->row_ordinal, r * 4u }, { (void **)&state->tail_row_begin, 2u * 4u },
		{ (void **)&state->local_scores, r * state->local_score_stride * 4u },
		{ (void **)&state->topk_values, state->topk_entries * 4u }, { (void **)&state->topk_positions, state->topk_entries * 4u },
		{ (void **)&state->local_selected, r * D41_INDEX_TOPK * 4u }, { (void **)&state->selected, r * D41_INDEX_TOPK * 4u },
		{ (void **)&state->candidates, r * D41_INDEX_TOPK * 8u }, { (void **)&state->candidates_gathered, D41_TP * r * D41_INDEX_TOPK * 8u },
		{ (void **)&state->block_scores, r * state->local_block_stride * 4u },
		{ (void **)&state->block_local_selected, r * state->local_block_keep * 4u },
		{ (void **)&state->block_candidates, r * state->local_block_keep * 8u },
		{ (void **)&state->block_gathered, (uint64_t)D41_TP * D41_SELECTION_CHUNK * state->local_block_keep * 8u },
		{ (void **)&state->block_selected, (uint64_t)D41_SELECTION_CHUNK * state->global_block_keep * 4u },
		{ (void **)&state->block_mask, r * state->local_block_stride },
		{ (void **)&state->verify_window, (uint64_t)D41_LAYERS * D41_VERIFY_ROWS * D41_HEAD_DIM * 2u },
		{ (void **)&state->verify_c_kv, (uint64_t)D41_PAIR_CACHES * D41_VERIFY_ROWS * D41_HEAD_DIM * 4u },
		{ (void **)&state->verify_c_score, (uint64_t)D41_PAIR_CACHES * D41_VERIFY_ROWS * D41_HEAD_DIM * 4u },
		{ (void **)&state->records_send, D41_TP * destination }, { (void **)&state->records_received, D41_TP * destination },
		{ (void **)&state->attn_merged, r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->attn_gathered, D41_TP * r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->group_input, r * 2u * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->wo_a_out, r * D41_WO_A_ROWS * 2u }, { (void **)&state->partial, r * D41_HIDDEN * 2u },
		{ (void **)&state->router_logits, r * D41_EXPERTS * 4u }, { (void **)&state->route_weight, routes * 4u },
		{ (void **)&state->packed_weight, routes * 4u }, { (void **)&state->route_expert, routes * 4u }, { (void **)&state->route_global, routes * 4u },
		{ (void **)&state->group_row_offset, 64u * 4u }, { (void **)&state->route_packed_row, routes * 4u },
		{ (void **)&state->route_source_token, routes * 4u }, { (void **)&state->tile_prefix_up, 64u * 4u }, { (void **)&state->tile_prefix_down, 64u * 4u },
		{ (void **)&state->expert_gate, routes * D41_MOE_INTER * 2u }, { (void **)&state->expert_up, routes * D41_MOE_INTER * 2u },
		{ (void **)&state->expert_act, routes * D41_MOE_INTER * 2u }, { (void **)&state->expert_out, routes * D41_HIDDEN * 2u },
		{ (void **)&state->shared_gate, r * D41_MOE_INTER * 2u }, { (void **)&state->shared_up, r * D41_MOE_INTER * 2u },
		{ (void **)&state->shared_act, r * D41_MOE_INTER * 2u }, { (void **)&state->shared_out, r * D41_HIDDEN * 2u },
		{ (void **)&state->engram_ids, r * D41_ENGRAM_COLUMNS * 8u }, { (void **)&state->engram_embed, engram_rows * D41_HIDDEN * 2u },
		{ (void **)&state->engram_kv, r * D41_ENGRAM_KV_ROWS * 2u }, { (void **)&state->engram_kv_local, r * D41_ENGRAM_KV_LOCAL * 2u },
		{ (void **)&state->engram_kv_gathered, r * D41_ENGRAM_KV_ROWS * 2u },
		{ (void **)&state->recurrent, s * state->recurrent_bytes }, { (void **)&state->kv_error, sizeof(LmKvAccessError) },
	};
	uint64_t cursor = 0u;
	for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
	{
		cursor = (cursor + 255u) & ~255ull;
		if ( base != 0 )
			*plan[index].target = base + cursor;
		cursor += plan[index].bytes;
	}
	*bytes = cursor;
	return SPARK_STATUS_OK;
}

static SparkStatus D41ReadFile(const char *path, uint64_t offset, void *target, uint64_t bytes)
{
	FILE *file = fopen(path, "rb");
	SparkStatus status = SPARK_STATUS_OK;
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( fseeko(file, (off_t)offset, SEEK_SET) != 0 || fread(target, 1u, (size_t)bytes, file) != bytes )
		status = SPARK_STATUS_IO_ERROR;
	fclose(file);
	return status;
}

static uint64_t D41FileBytes(const char *path)
{
	FILE *file = fopen(path, "rb");
	uint64_t bytes = 0u;
	if ( file == 0 )
		return 0u;
	if ( fseeko(file, 0, SEEK_END) == 0 )
		bytes = (uint64_t)ftello(file);
	fclose(file);
	return bytes;
}

static void D41Sibling(const char *pack_path, const char *name, char *out, size_t capacity)
{
	const char *slash = strrchr(pack_path, '/');
	const int prefix = slash != 0 ? (int)(slash - pack_path) : 1;
	snprintf(out, capacity, "%.*s/%s", prefix, slash != 0 ? pack_path : ".", name);
}

static SparkStatus D41OpenEngramTables(D41ModelState *state, const char *pack_path)
{
	char path[1024];
	uint32_t head[5];
	uint64_t rows[2];
	int64_t numbers[8 + 48 + 48];
	D41Sibling(pack_path, "engram_tables.bin", path, sizeof(path));
	if ( D41ReadFile(path, 0u, head, sizeof(head)) != SPARK_STATUS_OK || head[0] != D41_ENGRAM_TABLES_MAGIC || head[1] != 1u ||
		head[2] != D41_VOCAB || head[3] != D41_ENGRAM_COLUMNS ||
		D41ReadFile(path, sizeof(head), rows, sizeof(rows)) != SPARK_STATUS_OK ||
		D41ReadFile(path, sizeof(head) + sizeof(rows), numbers, sizeof(numbers)) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s is not the engram hash table file (tools/dsv41_flash_engram_tables.py)\n", path);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	state->engram_pad = (int32_t)head[4];
	int32_t *map = (int32_t *)malloc(D41_VOCAB * sizeof(int32_t));
	if ( map == 0 || D41ReadFile(path, sizeof(head) + sizeof(rows) + sizeof(numbers), map, D41_VOCAB * sizeof(int32_t)) != SPARK_STATUS_OK )
	{
		free(map);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( cudaMalloc(&state->token_map, D41_VOCAB * sizeof(int32_t)) != cudaSuccess ||
		cudaMemcpy(state->token_map, map, D41_VOCAB * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess )
	{
		free(map);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	free(map);
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		D41Engram *engram = &state->engram[which];
		const uint64_t part = (rows[which] + D41_TP - 1u) / D41_TP;
		const uint64_t first = (uint64_t)state->tp_rank * part;
		engram->first_row = first;
		engram->local_rows = first >= rows[which] ? 0u : (rows[which] - first < part ? rows[which] - first : part);
		if ( cudaMalloc(&engram->multipliers, D41_ENGRAM_MULTIPLIERS * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->primes, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->offsets, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMemcpy(engram->multipliers, numbers + which * D41_ENGRAM_MULTIPLIERS, D41_ENGRAM_MULTIPLIERS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(engram->primes, numbers + 8u + which * D41_ENGRAM_COLUMNS, D41_ENGRAM_COLUMNS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(engram->offsets, numbers + 56u + which * D41_ENGRAM_COLUMNS, D41_ENGRAM_COLUMNS * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41OpenEngramShard(D41ModelState *state, const char *pack_path)
{
	char sha_path[1100];
	FILE *file;
	uint32_t magic = 0u;
	D41Sibling(pack_path, "engram.spengram", state->engram_path, sizeof(state->engram_path));
	snprintf(sha_path, sizeof(sha_path), "%s.digest", state->engram_path);
	state->engram_bytes = D41FileBytes(state->engram_path);
	file = fopen(sha_path, "rb");
	memset(state->engram_sha256, 0, sizeof(state->engram_sha256));
	if ( file != 0 )
	{
		if ( fread(state->engram_sha256, 1u, SPARK_SHA256_HEX_BYTES - 1u, file) != SPARK_SHA256_HEX_BYTES - 1u )
			state->engram_sha256[0] = 0;
		fclose(file);
	}
	if ( state->engram_bytes == 0u || strlen(state->engram_sha256) != 64u ||
		D41ReadFile(state->engram_path, 0u, &magic, sizeof(magic)) != SPARK_STATUS_OK || magic != 0x31474544u ||
		D41ReadFile(state->engram_path, 512u, state->engram_entries, sizeof(state->engram_entries)) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the engram shard %s (with its .digest) is missing or not an engram shard\n", state->engram_path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	for ( uint32_t index = 0u; index < 8u; ++index )
		if ( state->engram_entries[index].payload_offset > state->engram_bytes ||
			state->engram_entries[index].payload_bytes > state->engram_bytes - state->engram_entries[index].payload_offset )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}

static void D41Close(void *model)
{
	D41ModelState *state = (D41ModelState *)model;
	if ( state == 0 )
		return;
	if ( state->engram_pack != 0 )
		SparkWeightdLazyPackDestroy(state->engram_pack);
	if ( state->arena != 0 )
		(void)cudaFree(state->arena);
	if ( state->fp8_side != 0 )
		(void)cudaFree(state->fp8_side);
	if ( state->cells != 0 )
		(void)cudaFree(state->cells);
	(void)cudaFree(state->token_map);
	(void)cudaFree(state->full_frequency);
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		(void)cudaFree(state->engram[which].multipliers);
		(void)cudaFree(state->engram[which].primes);
		(void)cudaFree(state->engram[which].offsets);
	}
	if ( state->dump != 0 )
		fclose(state->dump);
	free(state->dump_streams);
	free(state->dump_routes);
	free(state->dump_weights);
	free(state->entries);
	free(state);
}

static SparkStatus D41Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
{
	const SparkStageRunnerConfiguration *configuration = request->configuration;
	D41ModelState *state;
	double frequencies[2u * D41K_ROPE_PAIRS];
	uint64_t pack_bytes;
	SparkStatus status;
	if ( configuration->stage_count != 1u || configuration->tp_degree != D41_TP || configuration->max_input_row_count == 0u ||
		configuration->state_budget_bytes == 0u || configuration->kv_pages_per_sequence == 0u ||
		configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (D41ModelState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*model = state;
	state->services = request->services;
	state->tp_rank = configuration->tp_rank;
	state->max_rows = configuration->max_input_row_count;
	state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
		configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
	state->context_slots = configuration->max_active_sequence_count;
	state->max_positions = configuration->kv_pages_per_sequence * D41_PAGE_TOKENS;
	if ( getenv("SPARK_DSV41_FLASH_LAYER_DUMP") != 0 )
	{
		state->dump = fopen(getenv("SPARK_DSV41_FLASH_LAYER_DUMP"), "wb");
		state->dump_streams = (uint16_t *)malloc((size_t)state->max_rows * D41K_HC * D41_HIDDEN * sizeof(uint16_t));
		state->dump_routes = (uint32_t *)malloc((size_t)state->max_rows * D41_TOPK * sizeof(uint32_t));
		state->dump_weights = (float *)malloc((size_t)state->max_rows * D41_TOPK * sizeof(float));
		if ( state->dump == 0 || state->dump_streams == 0 || state->dump_routes == 0 || state->dump_weights == 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		fprintf(stderr, "sparkpipe_dsv41_flash: this rank dumps every layer's streams and routes to %s (debug; graphs off)\n",
			getenv("SPARK_DSV41_FLASH_LAYER_DUMP"));
	}
	pack_bytes = D41FileBytes(configuration->rank_pack_path);
	status = D41ReadFile(configuration->rank_pack_path, 0u, &state->header, sizeof(state->header));
	if ( status != SPARK_STATUS_OK || state->header.magic != SPARK_DSV41_FLASH_STAGEPACK_MAGIC || state->header.tp_degree != D41_TP ||
		state->header.tp_rank != configuration->tp_rank || state->header.hidden_dimension != D41_HIDDEN || state->header.layer_count != D41_LAYERS ||
		state->header.vocab_count != D41_VOCAB || state->header.file_bytes != pack_bytes || state->header.tensor_count == 0u )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %s is not this rank's TP16 DSV4.1 stage pack\n", configuration->rank_pack_path);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	state->entries = (D41PackEntry *)malloc((size_t)state->header.tensor_count * sizeof(D41PackEntry));
	if ( state->entries == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = D41ReadFile(configuration->rank_pack_path, state->header.directory_offset, state->entries,
		(uint64_t)state->header.tensor_count * sizeof(D41PackEntry));
	if ( status != SPARK_STATUS_OK )
		return status;
	status = D41OpenEngramTables(state, configuration->rank_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = D41OpenEngramShard(state, configuration->rank_pack_path);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41Frequencies(frequencies, frequencies + D41K_ROPE_PAIRS);
	if ( cudaMalloc(&state->full_frequency, sizeof(frequencies)) != cudaSuccess ||
		cudaMemcpy(state->full_frequency, frequencies, sizeof(frequencies), cudaMemcpyHostToDevice) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->compress_frequency = state->full_frequency + D41K_ROPE_PAIRS;
	{
		const uint32_t blocks = D41_MOE_INTER / D41_SHARED_GRANULE, base = blocks / D41_TP, extra = blocks % D41_TP;
		state->shared_first = (state->tp_rank * base + (state->tp_rank < extra ? state->tp_rank : extra)) * D41_SHARED_GRANULE;
		state->shared_width = (base + (state->tp_rank < extra ? 1u : 0u)) * D41_SHARED_GRANULE;
	}
	state->shard.degree = D41_TP;
	state->shard.rank = state->tp_rank;
	state->shard.grain = D41_GRAIN;
	state->local_score_stride = ((SparkKvShardGatherKeys(state->shard, state->max_positions) + 63u) / 64u) * 64u;
	state->global_score_stride = ((state->max_positions + 63u) / 64u) * 64u;
	state->candidate_blocks = state->max_positions > D41_CANDIDATE_ROWS ? 1u : 0u;
	if ( state->candidate_blocks != 0u )
	{
		state->local_block_stride = ((state->local_score_stride / D41_GRAIN + 63u) / 64u) * 64u;
		state->global_block_stride = (((state->max_positions + D41_GRAIN - 1u) / D41_GRAIN + 63u) / 64u) * 64u;
		state->local_block_keep = state->local_block_stride < D41_CANDIDATE_BLOCKS ? state->local_block_stride : D41_CANDIDATE_BLOCKS;
		state->global_block_keep = state->global_block_stride < D41_CANDIDATE_BLOCKS ? state->global_block_stride : D41_CANDIDATE_BLOCKS;
		state->block_round_parts = 1u;
		while ( state->local_block_keep * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)) / state->block_round_parts > D41_HIDDEN )
			state->block_round_parts *= 2u;
	}
	{
		const uint64_t local = (uint64_t)state->max_rows * LmTopkExactCandidateEntries(state->local_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
		const uint64_t local_blocks = state->candidate_blocks != 0u ?
			(uint64_t)state->max_rows * LmTopkExactCandidateEntries(state->local_block_stride, state->local_block_keep, LM_TOPK_EXACT_CHUNK) : 0u;
		state->topk_entries = 2u * (local > local_blocks ? local : local_blocks);
	}
	state->ring_bytes = (uint64_t)D41_LAYERS * D41_RING_ROWS * D41_HEAD_DIM * sizeof(uint16_t);
	state->compressor_bytes = 3u * 2u * D41_HEAD_DIM * sizeof(float);
	state->history_bytes = 4u * sizeof(int32_t);
	state->recurrent_bytes = (state->ring_bytes + state->compressor_bytes + state->history_bytes + 255u) & ~255ull;
	status = D41PlanArena(state, &state->arena_bytes, 0);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->arena_bytes > configuration->state_budget_bytes )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: rank state %llu bytes is over the %llu-byte budget (rows %u, slots %u, positions %u)\n",
			(unsigned long long)state->arena_bytes, (unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots,
			state->max_positions);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	{
		const D41PackEntry *embed = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EMBEDDING, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		const D41PackEntry *norm = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_FINAL_NORM, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		const D41PackEntry *head = D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_LM_HEAD, SPARK_DSV41_FLASH_STAGEPACK_GLOBAL_LAYER);
		if ( embed == 0 || norm == 0 || head == 0 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = D41_HIDDEN;
		geometry->vocab = D41_VOCAB;
		geometry->total_layers = D41_LAYERS;
		geometry->layer_count = D41_LAYERS;
		geometry->rms_epsilon = D41_EPS;
		geometry->pack_bytes = pack_bytes;
		geometry->embed_offset = embed->payload_offset;
		geometry->embed_bytes = embed->payload_bytes;
		geometry->embed_rows = embed->rows;
		geometry->head_norm_offset = norm->payload_offset;
		geometry->head_norm_bytes = norm->payload_bytes;
		geometry->head_offset = head->payload_offset;
		geometry->head_bytes = head->payload_bytes;
		geometry->head_rows = head->rows;
		geometry->kv_layer_count = 3u;
		geometry->kv_layer_page_bytes = D41Ratio2Kv::kPageBytes;
		geometry->kv_second_layer_count = 1u;
		geometry->kv_second_layer_page_bytes = D41Ratio1Kv::kPageBytes;
		geometry->kv_shard = state->shard;
		geometry->recurrent_bytes = state->recurrent_bytes;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41DenseManifest(const struct SparkWeightdManifest *manifest, void *context)
{
	(void)context;
	if ( manifest == 0 || manifest->range_count != 0u )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: the stage runner attaches the DSV4.1 packs whole (a zero-range .experts manifest)\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus D41ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
{
	return D41DenseManifest(manifest, model);
}

static SparkStatus D41SliceEntry(SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint32_t scale, const void **out)
{
	const uint64_t offset = scale != 0u ? entry->scale_offset : entry->payload_offset;
	const uint64_t bytes = scale != 0u ? entry->scale_bytes : entry->payload_bytes;
	if ( bytes == 0u || SparkWeightdLazyPackSlice(pack, offset, bytes, out) != SPARK_STATUS_OK || *out == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus D41Fp8Slice(D41ModelState *state, SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint32_t row_first,
	uint32_t rows, uint32_t column_first, uint32_t columns, uint64_t *cursor, D41Fp8Linear *out)
{
	const void *payload = 0, *scale = 0;
	const uint32_t whole_rows = column_first == 0u && entry != 0 && columns == entry->columns ? 1u : 0u;
	if ( entry == 0 || entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 || entry->rows % D41_FP8_BLOCK != 0u ||
		entry->columns % D41_FP8_BLOCK != 0u || rows == 0u || columns == 0u || row_first % D41_FP8_BLOCK != 0u ||
		column_first % D41_FP8_BLOCK != 0u || columns % D41_FP8_BLOCK != 0u || row_first > entry->rows || rows > entry->rows - row_first ||
		column_first > entry->columns || columns > entry->columns - column_first ||
		D41SliceEntry(pack, entry, 0u, &payload) != SPARK_STATUS_OK || D41SliceEntry(pack, entry, 1u, &scale) != SPARK_STATUS_OK ||
		((uintptr_t)payload % 16u) != 0u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	const uint64_t scale_bytes = (((uint64_t)rows * (columns / D41_FP8_BLOCK)) + 255u) & ~255ull;
	const uint64_t weight_bytes = whole_rows != 0u ? 0u : (((uint64_t)rows * columns) + 255u) & ~255ull;
	const uint64_t wide_bytes = (((uint64_t)rows * columns * sizeof(uint16_t)) + 255u) & ~255ull;
	if ( state->fp8_side != 0 )
	{
		uint8_t *row_scale = state->fp8_side + *cursor;
		uint8_t *weight = whole_rows != 0u ? (uint8_t *)0 : row_scale + scale_bytes;
		uint16_t *wide = (uint16_t *)(row_scale + scale_bytes + weight_bytes);
		D41Fp8SliceKernel<<<rows, D41_THREADS>>>((const uint8_t *)payload, (const uint8_t *)scale, weight, row_scale, wide, entry->columns,
			row_first, column_first, columns);
		out->scale = row_scale;
		out->weight = whole_rows != 0u ? (const uint8_t *)payload + (uint64_t)row_first * entry->columns : weight;
		out->wide = wide;
	}
	*cursor += scale_bytes + weight_bytes + wide_bytes;
	return SPARK_STATUS_OK;
}

static SparkStatus D41Fp8(D41ModelState *state, SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint64_t *cursor, D41Fp8Linear *out)
{
	if ( entry == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return D41Fp8Slice(state, pack, entry, 0u, entry->rows, 0u, entry->columns, cursor, out);
}

#define D41_ENTRY(kind, layer) D41Find(state, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_##kind, layer)
#define D41_PAYLOAD(kind, layer, field) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); const void *bound = 0; \
		if ( found == 0 || D41SliceEntry(lazy_pack, found, 0u, &bound) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u has no " #kind "\n", layer); return SPARK_STATUS_PARSE_ERROR; } \
		*(const void **)&(field) = bound; } while ( 0 )
#define D41_SCALE(kind, layer, field) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); const void *bound = 0; \
		if ( found == 0 || D41SliceEntry(lazy_pack, found, 1u, &bound) != SPARK_STATUS_OK ) return SPARK_STATUS_PARSE_ERROR; \
		*(const void **)&(field) = bound; } while ( 0 )
#define D41_FP8(kind, layer, field) \
	do { if ( D41Fp8(state, lazy_pack, D41_ENTRY(kind, layer), &cursor, &(field)) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u " #kind " is not a 32x32-block fp8 tensor\n", layer); return SPARK_STATUS_PARSE_ERROR; } } while ( 0 )
#define D41_FP8_SLICE(kind, layer, field, first_row, row_count, first_column, column_count) \
	do { const D41PackEntry *found = D41_ENTRY(kind, layer); \
		if ( found == 0 || D41Fp8Slice(state, lazy_pack, found, first_row, row_count == 0u ? found->rows : row_count, first_column, \
			column_count == 0u ? found->columns : column_count, &cursor, &(field)) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u " #kind " has no 32x32-block fp8 slice for this rank\n", layer); \
			return SPARK_STATUS_PARSE_ERROR; } } while ( 0 )

static SparkStatus D41BindLayers(D41ModelState *state, SparkWeightdLazyPack *lazy_pack, uint64_t *side_bytes)
{
	uint64_t cursor = 0u;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
	{
		D41Layer *layer = &state->layers[index];
		D41_PAYLOAD(ATTN_NORM, index, layer->attn_norm);
		D41_PAYLOAD(FFN_NORM, index, layer->ffn_norm);
		D41_PAYLOAD(Q_NORM, index, layer->q_norm);
		D41_PAYLOAD(KV_NORM, index, layer->kv_norm);
		D41_PAYLOAD(ATTN_SINK, index, layer->sink);
		D41_FP8(Q_A, index, layer->q_a);
		D41_FP8(Q_B, index, layer->q_b);
		D41_FP8(KV_A, index, layer->kv_a);
		D41_FP8(O_A, index, layer->o_a);
		D41_FP8(O_B, index, layer->o_b);
		D41_PAYLOAD(HC_ATTN_FN, index, layer->hc_attn_fn);
		D41_PAYLOAD(HC_ATTN_BASE, index, layer->hc_attn_base);
		D41_PAYLOAD(HC_ATTN_SCALE, index, layer->hc_attn_scale);
		D41_PAYLOAD(HC_FFN_FN, index, layer->hc_ffn_fn);
		D41_PAYLOAD(HC_FFN_BASE, index, layer->hc_ffn_base);
		D41_PAYLOAD(HC_FFN_SCALE, index, layer->hc_ffn_scale);
		D41_PAYLOAD(ROUTER, index, layer->router);
		D41_PAYLOAD(ROUTER_BIAS, index, layer->router_bias);
		D41_PAYLOAD(EXPERT_W1, index, layer->e_w1);
		D41_SCALE(EXPERT_W1, index, layer->e_w1_scale);
		D41_PAYLOAD(EXPERT_W2, index, layer->e_w2);
		D41_SCALE(EXPERT_W2, index, layer->e_w2_scale);
		D41_PAYLOAD(EXPERT_W3, index, layer->e_w3);
		D41_SCALE(EXPERT_W3, index, layer->e_w3_scale);
		D41_FP8_SLICE(SHARED_W1, index, layer->sh_w1, state->shared_first, state->shared_width, 0u, 0u);
		D41_FP8_SLICE(SHARED_W2, index, layer->sh_w2, 0u, 0u, state->shared_first, state->shared_width);
		D41_FP8_SLICE(SHARED_W3, index, layer->sh_w3, state->shared_first, state->shared_width, 0u, 0u);
		if ( D41KvSource(index) )
		{
			D41_PAYLOAD(COMPRESSOR_WKV, index, layer->c_wkv);
			D41_PAYLOAD(COMPRESSOR_NORM, index, layer->c_norm);
			D41_PAYLOAD(INDEXER_WK, index, layer->idx_wk);
			D41_PAYLOAD(INDEXER_K_NORM, index, layer->idx_kn);
			if ( D41Ratio(index) > 1u )
				D41_PAYLOAD(COMPRESSOR_WGATE, index, layer->c_wgate);
		}
		if ( D41Ratio(index) > 0u && D41IndexSource(index) )
		{
			D41_FP8(INDEXER_Q_B, index, layer->idx_q_b);
			D41_PAYLOAD(INDEXER_WEIGHTS_PROJ, index, layer->idx_wp);
		}
	}
	*side_bytes = cursor;
	return SPARK_STATUS_OK;
}

static SparkStatus D41BindEngram(D41ModelState *state, uint64_t *cursor, const SparkStageRunnerConfiguration *configuration)
{
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		const D41PackEntry *embed = &state->engram_entries[which * 4u], *wkv = &state->engram_entries[which * 4u + 1u];
		const D41PackEntry *q = &state->engram_entries[which * 4u + 2u], *k = &state->engram_entries[which * 4u + 3u];
		D41Engram *engram = &state->engram[which];
		const void *bound = 0;
		if ( embed->rows != engram->local_rows || embed->columns != D41_ENGRAM_HEAD_DIM || wkv->rows != D41_ENGRAM_KV_ROWS ||
			wkv->columns != D41_ENGRAM_EMBED || q->rows != D41K_HC || q->columns != D41_HIDDEN )
		{
			fprintf(stderr, "sparkpipe_dsv41_flash: engram shard layer %u holds %u rows of %u, expected %llu of %u\n", which == 0u ? 1u : 14u,
				embed->rows, embed->columns, (unsigned long long)engram->local_rows, D41_ENGRAM_HEAD_DIM);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( D41SliceEntry(state->engram_pack, embed, 0u, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->payload = (const uint8_t *)bound;
		if ( D41SliceEntry(state->engram_pack, embed, 1u, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->scale = (const uint8_t *)bound;
		if ( D41SliceEntry(state->engram_pack, q, 0u, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->q_weight = (const uint16_t *)bound;
		if ( D41SliceEntry(state->engram_pack, k, 0u, &bound) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		engram->k_weight = (const uint16_t *)bound;
		if ( D41Fp8Slice(state, state->engram_pack, wkv, state->tp_rank * D41_ENGRAM_KV_LOCAL, D41_ENGRAM_KV_LOCAL, 0u, D41_ENGRAM_EMBED,
			cursor, &engram->wkv) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	(void)configuration;
	return SPARK_STATUS_OK;
}

static SparkStatus D41AttachEngram(D41ModelState *state, const SparkStageRunnerConfiguration *configuration)
{
	SparkWeightdLazyAttachRequest request;
	memset(&request, 0, sizeof(request));
	memcpy(request.identity.pack_sha256, state->engram_sha256, SPARK_SHA256_HEX_BYTES);
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", "deepseek-v4.1-flash-engram");
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", "official");
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = state->engram_bytes;
	request.identity.topology = D41_TP;
	if ( strlen(state->engram_path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.pack_path, state->engram_path, strlen(state->engram_path) + 1u);
	request.expert_pool_bytes = configuration->weights.expert_pool_bytes;
	return SparkWeightdLazyPackCreateChecked(configuration->weights.socket_path, &request, state->engram_bytes + 255u,
		SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, D41DenseManifest, state, &state->engram_pack);
}

static SparkStatus D41PackCells(D41ModelState *state, SparkWeightdLazyPack *lazy_pack)
{
	const uint32_t kinds[3] = {SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W1, SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W3,
		SPARK_DSV41_FLASH_STAGEPACK_TENSOR_EXPERT_W2};
	const uint64_t up = LmCellPackBytes(D41_LOCAL_EXPERTS, D41_MOE_INTER, D41_HIDDEN), down = LmCellPackBytes(D41_LOCAL_EXPERTS, D41_HIDDEN, D41_MOE_INTER);
	uint8_t *place[D41_LAYERS][3];
	uint64_t separate = 0u, in_place = 0u;
	uint8_t *scratch = 0;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
		for ( uint32_t m = 0u; m < 3u; ++m )
		{
			const D41PackEntry *entry = D41Find(state, kinds[m], index);
			const uint64_t bytes = m < 2u ? up : down;
			void *region = 0;
			place[index][m] = 0;
			if ( entry != 0 && entry->scale_offset == entry->payload_offset + entry->payload_bytes &&
				entry->payload_bytes + entry->scale_bytes == bytes &&
				SparkWeightdLazyPackPrivateSlice(lazy_pack, entry->payload_offset, bytes, &region) == SPARK_STATUS_OK )
			{
				place[index][m] = (uint8_t *)region;
				in_place += bytes;
			}
			else
				separate += bytes;
		}
	state->cells_bytes = separate;
	if ( (separate != 0u && cudaMalloc((void **)&state->cells, separate) != cudaSuccess) ||
		(in_place != 0u && cudaMalloc((void **)&scratch, up > down ? up : down) != cudaSuccess) )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the expert cells could not be allocated\n", (unsigned long long)separate);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	separate = 0u;
	for ( uint32_t index = 0u; index < D41_LAYERS; ++index )
	{
		D41Layer *layer = &state->layers[index];
		const uint8_t *payload[3] = {layer->e_w1, layer->e_w3, layer->e_w2}, *scale[3] = {layer->e_w1_scale, layer->e_w3_scale, layer->e_w2_scale};
		const uint8_t **cells[3] = {&layer->e_w1_cells, &layer->e_w3_cells, &layer->e_w2_cells};
		for ( uint32_t m = 0u; m < 3u; ++m )
		{
			const uint64_t bytes = m < 2u ? up : down;
			const uint32_t rows = m < 2u ? D41_MOE_INTER : D41_HIDDEN, columns = m < 2u ? D41_HIDDEN : D41_MOE_INTER;
			uint8_t *target = place[index][m];
			if ( target != 0 )
			{
				const uint64_t payload_bytes = (uint64_t)D41_LOCAL_EXPERTS * rows * columns / 2u;
				if ( cudaMemcpy(scratch, target, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess ||
					LmCellPackMxfp4(scratch, scratch + payload_bytes, target, D41_LOCAL_EXPERTS, rows, columns, 0) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				target = state->cells + separate;
				separate += bytes;
				if ( LmCellPackMxfp4(payload[m], scale[m], target, D41_LOCAL_EXPERTS, rows, columns, 0) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			*cells[m] = target;
		}
		layer->e_w1 = layer->e_w1_scale = layer->e_w2 = layer->e_w2_scale = layer->e_w3 = layer->e_w3_scale = 0;
	}
	if ( cudaDeviceSynchronize() != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( scratch != 0 )
		(void)cudaFree(scratch);
	fprintf(stderr, "DSV41-CELLS rank=%u in_place_bytes=%llu separate_bytes=%llu\n", state->tp_rank, (unsigned long long)in_place,
		(unsigned long long)state->cells_bytes);
	return SPARK_STATUS_OK;
}

static const SparkStageRunnerConfiguration *D41BindConfiguration;

static SparkStatus D41Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	D41ModelState *state = (D41ModelState *)model;
	uint64_t bytes = 0u;
	SparkStatus status;
	(void)geometry;
	status = D41AttachEngram(state, D41BindConfiguration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: weightd refused the engram shard %s status=%d\n", state->engram_path, (int)status);
		return status;
	}
	status = D41BindLayers(state, lazy_pack, &bytes);
	if ( status == SPARK_STATUS_OK )
		status = D41BindEngram(state, &bytes, D41BindConfiguration);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->fp8_side_bytes = bytes;
	if ( cudaMalloc((void **)&state->fp8_side, bytes) != cudaSuccess )
	{
		state->fp8_side = 0;
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the fp8 row scales, slices and bf16 widenings could not be allocated\n", (unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	bytes = 0u;
	status = D41BindLayers(state, lazy_pack, &bytes);
	if ( status == SPARK_STATUS_OK )
		status = D41BindEngram(state, &bytes, D41BindConfiguration);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = D41PackCells(state, lazy_pack);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( cudaDeviceSynchronize() != cudaSuccess || cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
	{
		state->arena = 0;
		fprintf(stderr, "sparkpipe_dsv41_flash: the %llu-byte rank state could not be allocated\n", (unsigned long long)state->arena_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = D41PlanArena(state, &state->arena_bytes, (uint8_t *)state->arena);
	if ( status != SPARK_STATUS_OK )
		return status;
	{
		float *init = (float *)calloc((size_t)state->max_rows * D41K_HC, sizeof(float));
		if ( init == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		for ( uint32_t row = 0u; row < state->max_rows; ++row )
			init[(uint64_t)row * D41K_HC] = 1.0f;
		const cudaError_t error = cudaMemcpy(state->pre_init, init, (size_t)state->max_rows * D41K_HC * sizeof(float), cudaMemcpyHostToDevice);
		free(init);
		if ( error != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

#undef D41_FP8_SLICE
#undef D41_FP8
#undef D41_SCALE
#undef D41_PAYLOAD
#undef D41_ENTRY

static SparkStatus D41OpenRecorded(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
{
	D41BindConfiguration = request->configuration;
	return D41Open(request, model, geometry);
}

static SparkStatus D41AttachKv(void *model, const SparkStageRunnerKv *kv)
{
	D41ModelState *state = (D41ModelState *)model;
	if ( kv == 0 || kv->layer_count != 3u || kv->second_layer_count != 1u || kv->second_pool == 0 || kv->sequence_count > state->slots ||
		kv->layer_page_bytes != D41Ratio2Kv::kPageBytes || kv->second_layer_page_bytes != D41Ratio1Kv::kPageBytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t cache = 0u; cache < 4u; ++cache )
	{
		uint8_t *pool = cache < 3u ? kv->pool + (uint64_t)cache * kv->layer_stride_bytes : kv->second_pool;
		const int32_t latent = cache < 3u ?
			LmKvShardViewInitialize<D41Ratio2Kv>(&state->latent_views[cache], pool, kv->page_table, kv->page_table_stride, kv->sequence_count,
				kv->pool_page_count, state->kv_error, kv->context_shard) :
			LmKvShardViewInitialize<D41Ratio1Kv>(&state->latent_views[cache], pool, kv->page_table, kv->page_table_stride, kv->sequence_count,
				kv->pool_page_count, state->kv_error, kv->context_shard);
		const int32_t keys = cache < 3u ?
			LmKvShardViewInitialize<D41Ratio2Kv>(&state->index_views[cache], pool + D41_HEAD_DIM * sizeof(uint16_t), kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) :
			LmKvShardViewInitialize<D41Ratio1Kv>(&state->index_views[cache], pool + D41_HEAD_DIM * sizeof(uint16_t), kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard);
		if ( latent != 0 || keys != 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->kv_attached = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus D41RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	D41ModelState *state = (D41ModelState *)model;
	uint8_t *record = state->recurrent + (uint64_t)slot * state->recurrent_bytes;
	cudaError_t error;
	if ( slot >= state->slots || buffer == 0 || bytes != state->recurrent_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = to_buffer != 0u ? cudaMemcpyAsync(buffer, record, (size_t)bytes, cudaMemcpyDefault, (cudaStream_t)stream) :
		cudaMemcpyAsync(record, buffer, (size_t)bytes, cudaMemcpyDefault, (cudaStream_t)stream);
	if ( error == cudaSuccess && stream == 0 )
		error = cudaStreamSynchronize(0);
	return error == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus D41ResetSlot(void *model, uint32_t slot, void *stream)
{
	D41ModelState *state = (D41ModelState *)model;
	if ( slot >= state->slots )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return cudaMemsetAsync(state->recurrent + (uint64_t)slot * state->recurrent_bytes, 0, (size_t)state->recurrent_bytes,
		(cudaStream_t)stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus D41Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
{
	(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
	SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
}

static const void *D41ProbeBuffers(void *model)
{
	return model;
}

static void D41Report(void *model, uint32_t rank)
{
	const D41ModelState *state = (const D41ModelState *)model;
	fprintf(stderr, "DSV41-STATE rank=%u arena_bytes=%llu fp8_side_bytes=%llu slots=%u rows=%u positions=%u recurrent_bytes=%llu\n", rank,
		(unsigned long long)state->arena_bytes, (unsigned long long)state->fp8_side_bytes, state->slots, state->max_rows, state->max_positions,
		(unsigned long long)state->recurrent_bytes);
}

static uint32_t D41GraphKey(void *model, const SparkStageRunnerStep *step)
{
	(void)model;
	return step->commit == 0u && step->last_rows_only == 0u && step->sequences == 1u && step->rows > 1u && step->rows <= D41_VERIFY_ROWS ? 2u : 1u;
}

static const SparkStageRunnerModelInterface spark_dsv41_flash_stage_model =
{
	SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
	"sparkpipe_dsv41_flash",
	"deepseek-v4.1-flash",
	"official",
	D41OpenRecorded,
	D41ManifestCheck,
	D41Bind,
	D41Close,
	D41Step,
	D41Sideband,
	D41AttachKv,
	D41RecurrentCopy,
	D41ResetSlot,
	D41ProbeBuffers,
	D41Report,
	D41GraphKey,
	0,
	D41KeepRows,
	0
};

const SparkStageRunnerModelInterface *SparkDsv41FlashStageModel(void)
{
	return &spark_dsv41_flash_stage_model;
}
