#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "runtime/launch.h"
#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn_shard.cuh"
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
#define D41_CANDIDATE_ROWS (2048u * 8u)
#define D41_ENGRAM_HEAD_DIM 256u
#define D41_ENGRAM_COLUMNS 24u
#define D41_ENGRAM_ORDERS 3u
#define D41_ENGRAM_HEADS 8u
#define D41_ENGRAM_KV_ROWS ((D41K_HC + 1u) * D41_HIDDEN)
#define D41_ENGRAM_EMBED (D41_ENGRAM_COLUMNS * D41_ENGRAM_HEAD_DIM)
#define D41_THREADS 256u
#define D41_SLOT_BYTES SPARK_DSV41_FLASH_STAGE_SLOT_BYTES
#define D41_PAGE_TOKENS SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS
#define D41_GRAIN SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN
#define D41_A2A_HEADS SPARK_DSV41_FLASH_STAGE_A2A_HEADS
#define D41_LOCAL_HEADS (D41_HEADS / D41_TP)
#define D41_LOCAL_INDEX_HEADS (D41_INDEX_HEADS / D41_TP)
#define D41_LOCAL_EXPERTS (D41_EXPERTS / D41_TP)
#define D41_WO_A_ROWS (D41_OGROUPS * D41_OLORA / D41_TP)
#define D41_RING_ROWS (D41_WINDOW / D41_TP)
#define D41_SELECTION_CHUNK 64u
#define D41_INDEX_PACK 264u
#define D41_ENGRAM_TABLES_MAGIC 0x45313444u
#define D41_GEMM_TILE_N 128u
#define D41_GEMM_STAGES 2u
#define D41_GEMM_WARPS 8u

using D41Ratio2Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS / 2u, true>;
using D41Ratio1Kv = LmKvGeometry<D41_SLOT_BYTES, D41_PAGE_TOKENS, true>;

typedef SparkDsv41FlashStagePackHeader D41PackHeader;
typedef SparkDsv41FlashStagePackEntry D41PackEntry;

typedef struct D41Layer
{
	const uint16_t *attn_norm, *ffn_norm, *q_norm, *kv_norm;
	const uint16_t *q_a, *q_b, *kv_a, *o_a, *o_b, *idx_q_b;
	const uint16_t *idx_wk, *idx_wp, *idx_kn, *c_wkv, *c_wgate, *c_norm;
	const float *sink, *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
	const uint16_t *router;
	const float *router_bias;
	const uint8_t *e_w1, *e_w1_scale, *e_w2, *e_w2_scale, *e_w3, *e_w3_scale;
	const uint16_t *sh_w1, *sh_w2, *sh_w3;
} D41Layer;

typedef struct D41Engram
{
	const uint8_t *payload, *scale;
	const uint16_t *wkv, *q_weight, *k_weight;
	uint64_t first_row, local_rows;
	int64_t *multipliers, *primes, *offsets;
} D41Engram;

typedef struct D41ModelState
{
	const SparkStageRunnerServices *services;
	uint32_t tp_rank, max_rows, slots, max_context;
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
	uint16_t *dequant;
	uint64_t dequant_bytes;
	void *arena;
	uint64_t arena_bytes;
	uint64_t record_chunk_floats, a2a_units;
	uint32_t local_score_stride, global_score_stride;
	uint16_t *streams, *residual, *x, *xq, *q_latent, *qr, *qrq, *q_local, *q_gathered, *kv, *window_kv;
	float *mixes, *pre_attn, *post_attn, *comb_attn, *pre_ffn, *post_ffn, *comb_ffn, *pre_head;
	float *c_kv, *c_score;
	uint16_t *c_latent, *c_latent_store, *idx_key, *idx_pack_local, *idx_pack_gathered, *idx_q_full, *idx_query, *idx_w_full;
	uint32_t *emit_row, *bound_ratio2, *bound_ratio1, *length_ratio2, *length_ratio1;
	float *local_scores, *full_scores, *topk_values;
	uint32_t *local_selected, *selected, *topk_positions;
	uint64_t topk_entries;
	uint2 *candidates, *candidates_gathered;
	float *records_send, *records_received;
	uint16_t *attn_merged, *attn_gathered, *group_input, *wo_a_out, *partial;
	float *pre_init;
	uint32_t max_positions;
	float *router_logits, *route_weight, *packed_weight;
	uint32_t *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	uint16_t *expert_gate, *expert_up, *expert_act, *expert_out, *moe_out, *shared_gate, *shared_up, *shared_act, *shared_out;
	int64_t *engram_ids;
	uint16_t *engram_embed, *engram_kv;
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

static SparkStatus D41Launched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

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

static void D41Rope(uint16_t *data, uint32_t rows, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *positions, int32_t shift, const double *frequency, uint32_t inverse, const uint32_t *row_valid, cudaStream_t stream)
{
	D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS, 0, stream>>>(data, heads, head_stride, rope_offset, positions, shift,
		frequency, inverse, row_valid);
}

static SparkStatus D41HcMixes(D41ModelState *state, const uint16_t *streams, const float *fn, const float *base, const float *scale,
	float *pre, float *post, float *comb, uint32_t rows, cudaStream_t stream)
{
	D41HcMixRowsKernel<<<rows, D41K_THREADS, 0, stream>>>(streams, fn, state->mixes, D41_HIDDEN, D41_EPS);
	D41HcSinkhornRowsKernel<<<(rows + 31u) / 32u, 32u, 0, stream>>>(state->mixes, scale, base, pre, post, comb, rows,
		D41_HC_ITERATIONS, D41_HC_EPS);
	return D41Launched();
}

static SparkStatus D41EngramApply(D41ModelState *state, uint32_t which, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows;
	const D41Engram *engram = &state->engram[which];
	const uint32_t ar_rows = (rows * D41_ENGRAM_EMBED + D41_HIDDEN - 1u) / D41_HIDDEN;
	SparkStatus status;
	D41EngramHashKernel<<<(rows + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->positions,
		step->sequence_of_row, step->sequence_row_begin, step->recurrent_index,
		(const int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		engram->multipliers, engram->primes, engram->offsets, state->engram_ids, rows, D41_ENGRAM_ORDERS, D41_ENGRAM_HEADS, state->engram_pad);
	cudaMemsetAsync(state->engram_embed, 0, (uint64_t)ar_rows * D41_HIDDEN * sizeof(uint16_t), stream);
	D41EngramGatherKernel<<<dim3(rows, D41_ENGRAM_COLUMNS), D41_THREADS, 0, stream>>>(state->engram_ids, engram->payload, engram->scale,
		state->engram_embed, engram->first_row, engram->local_rows, D41_ENGRAM_COLUMNS, D41_ENGRAM_HEAD_DIM);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, ar_rows, 0u, state->engram_embed, state->engram_embed);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41Fp8Qdq(state->engram_embed, (uint64_t)rows * D41_ENGRAM_EMBED, stream);
	if ( D41Project(engram->wkv, state->engram_embed, state->engram_kv, 0, rows, D41_ENGRAM_EMBED, D41_ENGRAM_KV_ROWS,
		step->multiprocessors, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
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
		compressor_state, step->positions, step->sequence_of_row, step->recurrent_index, ratio, state_stride, D41_HEAD_DIM);
	if ( ratio > 1u )
		D41CompressorSaveKernel<<<rows, D41_THREADS, 0, stream>>>(state->c_kv, state->c_score, compressor_state, step->positions,
			step->sequence_of_row, step->recurrent_index, rows, ratio, state_stride, D41_HEAD_DIM, step->commit);
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

template<class Geometry>
static SparkStatus D41Select(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step,
	const uint32_t *bound, const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index), sms = step->multiprocessors;
	const uint32_t local_width = D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM;
	SparkStatus status;
	if ( index > D41_CANDIDATE_SOURCE && state->max_positions / D41Ratio(index) > D41_CANDIDATE_ROWS )
	{
		fprintf(stderr, "sparkpipe_dsv41_flash: contexts past %u compressed rows need the candidate-block mask, which is not built yet\n",
			D41_CANDIDATE_ROWS);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( D41Project(layer->idx_q_b, state->qrq, state->idx_pack_local, 0, rows, D41_QLORA, local_width, sms, stream) != LM_LAUNCH_OK ||
		D41Project(layer->idx_wp, state->x, state->idx_w_full, 0, rows, D41_HIDDEN, D41_LOCAL_INDEX_HEADS, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->idx_pack_local, rows, D41_LOCAL_INDEX_HEADS, D41_INDEX_DIM, D41_INDEX_DIM - D41_ROPE, step->positions, 0,
		state->compress_frequency, 0u, 0, stream);
	SparkDsv41FlashFp4Pow2QdqKernel<<<(rows * (local_width / 32u) + 255u) / 256u, 256u, 0, stream>>>(state->idx_pack_local,
		rows * (local_width / 32u), 32u);
	D41ScaleRowsKernel<<<D41Blocks((uint64_t)rows * D41_LOCAL_INDEX_HEADS), D41_THREADS, 0, stream>>>(state->idx_w_full,
		(uint64_t)rows * D41_LOCAL_INDEX_HEADS, (float)(1.0 / sqrt((double)D41_INDEX_DIM) / sqrt((double)D41_INDEX_HEADS)));
	cudaMemsetAsync(state->idx_pack_gathered, 0, (uint64_t)rows * D41_INDEX_PACK * sizeof(uint16_t), stream);
	cudaMemcpy2DAsync(state->idx_pack_gathered, D41_INDEX_PACK * sizeof(uint16_t), state->idx_pack_local, local_width * sizeof(uint16_t),
		local_width * sizeof(uint16_t), rows, cudaMemcpyDeviceToDevice, stream);
	cudaMemcpy2DAsync(state->idx_pack_gathered + local_width, D41_INDEX_PACK * sizeof(uint16_t), state->idx_w_full,
		D41_LOCAL_INDEX_HEADS * sizeof(uint16_t), D41_LOCAL_INDEX_HEADS * sizeof(uint16_t), rows, cudaMemcpyDeviceToDevice, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, D41_INDEX_PACK,
			state->idx_pack_gathered, state->idx_q_full);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t rank = 0u; rank < D41_TP; ++rank )
	{
		const uint16_t *source = state->idx_q_full + (uint64_t)rank * rows * D41_INDEX_PACK;
		cudaMemcpy2DAsync(state->idx_query + (uint64_t)rank * local_width, D41_INDEX_HEADS * D41_INDEX_DIM * sizeof(uint16_t), source,
			D41_INDEX_PACK * sizeof(uint16_t), local_width * sizeof(uint16_t), rows, cudaMemcpyDeviceToDevice, stream);
		cudaMemcpy2DAsync(state->idx_w_full + (uint64_t)rank * D41_LOCAL_INDEX_HEADS, D41_INDEX_HEADS * sizeof(uint16_t), source + local_width,
			D41_INDEX_PACK * sizeof(uint16_t), D41_LOCAL_INDEX_HEADS * sizeof(uint16_t), rows, cudaMemcpyDeviceToDevice, stream);
	}
	if ( LmWeightedSparseScoreLaunch<Geometry,D41_INDEX_HEADS,D41_INDEX_DIM,LmKvShardView,true>(state->idx_query, state->idx_w_full,
		state->index_views[cache], step->sequence_of_row, length, bound, rows, state->local_score_stride, 1.0f, state->local_scores,
		stream) != cudaSuccess ||
		LmTopkExactLaunch<D41_THREADS>(state->local_scores, rows, state->local_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK,
			LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->local_selected, stream) != cudaSuccess ||
		LmIndexShardCandidatePackLaunch<D41_THREADS>(state->index_views[cache].shard, state->local_scores, state->local_score_stride,
			state->local_selected, D41_INDEX_TOPK, rows, state->candidates, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
		D41_INDEX_TOPK * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->candidates, state->candidates_gathered);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t first = 0u; first < rows; first += D41_SELECTION_CHUNK )
	{
		const uint32_t count = rows - first < D41_SELECTION_CHUNK ? rows - first : D41_SELECTION_CHUNK;
		if ( LmIndexShardCandidateScatterLaunch<D41_THREADS>(state->candidates_gathered + (uint64_t)first * D41_INDEX_TOPK,
				(uint64_t)rows * D41_INDEX_TOPK, D41_TP, D41_INDEX_TOPK, count, state->global_score_stride, state->full_scores, stream) != cudaSuccess ||
			LmTopkExactLaunch<D41_THREADS>(state->full_scores, count, state->global_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK,
				LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries,
				state->selected + (uint64_t)first * D41_INDEX_TOPK, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return D41Launched();
}

static uint32_t D41A2aUnits(uint32_t rows)
{
	const uint64_t bytes = (uint64_t)rows * D41_LOCAL_HEADS * D41K_RECORD * sizeof(float);
	return (uint32_t)((bytes + D41_HIDDEN * sizeof(uint16_t) - 1u) / (D41_HIDDEN * sizeof(uint16_t)));
}

static uint64_t D41DestinationStride(uint32_t rows)
{
	return (uint64_t)D41A2aUnits(rows) * D41_HIDDEN * sizeof(uint16_t) / sizeof(float);
}

template<class Geometry>
static SparkStatus D41Partials(D41ModelState *state, uint32_t index, const SparkStageRunnerStep *step, const uint32_t *bound,
	const uint32_t *length, cudaStream_t stream)
{
	const uint32_t rows = step->rows, cache = D41CacheIndex(index);
	if ( LmLatentShardPartialHeadsLaunch<Geometry,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_gathered,
		(uint64_t)rows * D41_LOCAL_HEADS * D41_HEAD_DIM, D41_LOCAL_HEADS, D41_HEADS, step->sequence_of_row, length, bound, state->selected,
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
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	SparkStatus status;
	cudaMemcpyAsync(state->xq, state->x, (uint64_t)rows * D41_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	D41Fp8Qdq(state->xq, (uint64_t)rows * D41_HIDDEN, stream);
	if ( D41Project(layer->q_a, state->xq, state->q_latent, 0, rows, D41_HIDDEN, D41_QLORA, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNorm(state->q_latent, layer->q_norm, state->qr, rows, D41_QLORA, stream);
	cudaMemcpyAsync(state->qrq, state->qr, (uint64_t)rows * D41_QLORA * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	D41Fp8Qdq(state->qrq, (uint64_t)rows * D41_QLORA, stream);
	if ( D41Project(layer->q_b, state->qrq, state->q_local, 0, rows, D41_QLORA, local_width, sms, stream) != LM_LAUNCH_OK ||
		D41Project(layer->kv_a, state->xq, state->kv, 0, rows, D41_HIDDEN, D41_HEAD_DIM, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	D41RmsNorm(state->kv, layer->kv_norm, state->kv, rows, D41_HEAD_DIM, stream);
	D41Rope(state->kv, rows, 1u, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	cudaMemcpyAsync(state->window_kv, state->kv, (uint64_t)rows * D41_HEAD_DIM * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	D41Fp8Qdq(state->window_kv, (uint64_t)rows * D41_HEAD_DIM, stream);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK && D41KvSource(index) )
		status = D41Compress(state, layer, index, step, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_width, state->q_local, state->q_gathered);
	if ( status == SPARK_STATUS_OK && ratio > 0u && D41IndexSource(index) )
		status = D41CacheIndex(index) < 3u ? D41Select<D41Ratio2Kv>(state, layer, index, step, bound, length, stream) :
			D41Select<D41Ratio1Kv>(state, layer, index, step, bound, length, stream);
	if ( status == SPARK_STATUS_OK && ratio > 0u )
		status = D41CacheIndex(index) < 3u ? D41Partials<D41Ratio2Kv>(state, index, step, bound, length, stream) :
			D41Partials<D41Ratio1Kv>(state, index, step, bound, length, stream);
	else if ( status == SPARK_STATUS_OK )
		D41InitRecordsKernel<<<D41Blocks((uint64_t)rows * D41_HEADS), D41_THREADS, 0, stream>>>(state->records_send, destination_stride,
			rows, D41_LOCAL_HEADS, D41_TP);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41WindowFoldKernel<<<dim3(rows, D41_HEADS), D41K_THREADS, 0, stream>>>(state->records_send, destination_stride, D41_LOCAL_HEADS,
		state->q_gathered, (uint64_t)rows * local_width, state->window_kv, ring, step->positions, step->sequence_of_row, step->sequence_row_begin,
		step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_TP, state->tp_rank,
		1.0f / sqrtf((float)D41_HEAD_DIM));
	D41WindowRingStoreKernel<<<rows, D41_THREADS, 0, stream>>>(state->window_kv, ring, step->positions, step->sequence_of_row,
		step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, D41_TP, state->tp_rank, rows, step->commit);
	status = D41Launched();
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, D41A2aUnits(rows), 0u, state->records_send,
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
	if ( D41Project(layer->o_a, state->group_input, state->wo_a_out, 0, rows, 2u * local_width, D41_WO_A_ROWS, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Fp8Qdq(state->wo_a_out, (uint64_t)rows * D41_WO_A_ROWS, stream);
	if ( D41Project(layer->o_b, state->wo_a_out, state->partial, 0, rows, D41_WO_A_ROWS, D41_HIDDEN, sms, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return D41Launched();
}

static SparkStatus D41DumpRows(D41ModelState *state, uint32_t layer, uint32_t kind, const uint16_t *rows_bf16, uint32_t rows,
	uint32_t width, cudaStream_t stream);

static int32_t D41ExpertsUp(D41ModelState *state, const uint8_t *weight, LmScaleTensor scale, uint16_t *output, uint32_t rows,
	uint32_t routes, uint32_t sms, cudaStream_t stream)
{
	int32_t launch = LmSkinnyGroupedExperts<LmMxfp4>(weight, scale, state->xq, output, state->group_row_offset, state->route_source_token,
		D41_LOCAL_EXPERTS, routes, 0u, D41_HIDDEN, D41_MOE_INTER, stream);
	if ( launch == LM_LAUNCH_ERR_SHAPE )
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = state->group_row_offset;
		gemm.group_tile_prefix = state->tile_prefix_up;
		gemm.source_row_map = state->route_source_token;
		gemm.source_row_count = rows;
		gemm.output_bf16 = output;
		launch = LmGemmWeightOnlyIndirectLaunch<LmMxfp4,D41_GEMM_TILE_N,D41_GEMM_STAGES,D41_GEMM_WARPS>(&gemm, state->xq, weight, routes, rows,
			D41_TOPK, D41_LOCAL_EXPERTS, D41_HIDDEN, D41_MOE_INTER, sms, stream);
	}
	return launch;
}

static SparkStatus D41Moe(D41ModelState *state, const D41Layer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * D41_TOPK;
	const LmScaleTensor w1_scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(layer->e_w1_scale, D41_LOCAL_EXPERTS, D41_MOE_INTER, D41_HIDDEN);
	const LmScaleTensor w3_scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(layer->e_w3_scale, D41_LOCAL_EXPERTS, D41_MOE_INTER, D41_HIDDEN);
	const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(layer->e_w2_scale, D41_LOCAL_EXPERTS, D41_HIDDEN, D41_MOE_INTER);
	int32_t launch;
	if ( D41Project(layer->router, state->x, 0, state->router_logits, rows, D41_HIDDEN, D41_EXPERTS, sms, stream) != LM_LAUNCH_OK ||
		LmTopkRouteLaunch<D41_THREADS,D41_TOPK,true,LM_TOPK_SCORE_SQRT_SOFTPLUS>(rows, state->router_logits, D41_EXPERTS, state->route_expert,
			state->route_weight, layer->router_bias, 0, D41_ROUTE_SCALE, stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->dump != 0 &&
		(cudaStreamSynchronize(stream) != cudaSuccess ||
		cudaMemcpy(state->dump_routes, state->route_expert, (size_t)routes * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(state->dump_weights, state->route_weight, (size_t)routes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	D41RouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
		routes, state->tp_rank * D41_LOCAL_EXPERTS, D41_LOCAL_EXPERTS);
	if ( LmRouteBuild<D41_THREADS,D41_LOCAL_EXPERTS + 1u>(state->route_expert, rows, routes, D41_TOPK, state->group_row_offset,
		state->route_packed_row, state->route_source_token, D41_MOE_INTER, D41_HIDDEN, D41_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down,
		stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41PackedWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
	D41Fp8Qdq(state->xq, (uint64_t)rows * D41_HIDDEN, stream);
	launch = D41ExpertsUp(state, layer->e_w1, w1_scale, state->expert_gate, rows, routes, sms, stream);
	if ( launch == LM_LAUNCH_OK )
		launch = D41ExpertsUp(state, layer->e_w3, w3_scale, state->expert_up, rows, routes, sms, stream);
	if ( launch != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41SwigluPackedKernel<<<dim3(D41Blocks(D41_MOE_INTER), routes), D41_THREADS, 0, stream>>>(state->expert_gate, state->expert_up,
		state->packed_weight, state->expert_act, D41_MOE_INTER, D41_SWIGLU_LIMIT);
	D41Fp8Qdq(state->expert_act, (uint64_t)routes * D41_MOE_INTER, stream);
	launch = LmSkinnyGroupedExperts<LmMxfp4>(layer->e_w2, w2_scale, state->expert_act, state->expert_out, state->group_row_offset,
		state->route_source_token, D41_LOCAL_EXPERTS, routes, 1u, D41_MOE_INTER, D41_HIDDEN, stream);
	if ( launch == LM_LAUNCH_ERR_SHAPE )
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = w2_scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = state->group_row_offset;
		gemm.group_tile_prefix = state->tile_prefix_down;
		gemm.output_bf16 = state->expert_out;
		launch = LmGemmWeightOnlyLaunch<LmMxfp4,D41_GEMM_TILE_N,D41_GEMM_STAGES,D41_GEMM_WARPS>(&gemm, state->expert_act, layer->e_w2, routes, rows,
			D41_TOPK, D41_LOCAL_EXPERTS, D41_MOE_INTER, D41_HIDDEN, sms, true, stream);
	}
	if ( launch != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->tp_rank == 0u )
	{
		if ( D41Project(layer->sh_w1, state->xq, state->shared_gate, 0, rows, D41_HIDDEN, D41_MOE_INTER, sms, stream) != LM_LAUNCH_OK ||
			D41Project(layer->sh_w3, state->xq, state->shared_up, 0, rows, D41_HIDDEN, D41_MOE_INTER, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		D41SwigluRowsKernel<<<dim3(D41Blocks(D41_MOE_INTER), rows), D41_THREADS, 0, stream>>>(state->shared_gate, state->shared_up,
			state->shared_act, D41_MOE_INTER, D41_SWIGLU_LIMIT);
		D41Fp8Qdq(state->shared_act, (uint64_t)rows * D41_MOE_INTER, stream);
		if ( D41Project(layer->sh_w2, state->shared_act, state->shared_out, 0, rows, D41_MOE_INTER, D41_HIDDEN, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
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
	D41MoeLocalFinalizeKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
		state->route_expert, state->tp_rank == 0u ? state->shared_out : (const uint16_t *)0, state->partial, D41_TOPK, D41_LOCAL_EXPERTS, D41_HIDDEN);
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
	cudaMemcpyAsync(state->residual, state->streams, stream_bytes, cudaMemcpyDeviceToDevice, stream);
	D41HcPostRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->partial, state->residual, state->post_attn,
		state->comb_attn, state->streams, D41_HIDDEN);
	status = D41HcMixes(state, state->streams, layer->hc_ffn_fn, layer->hc_ffn_base, layer->hc_ffn_scale, state->pre_ffn,
		state->post_ffn, state->comb_ffn, rows, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_attn, state->x, D41_HIDDEN);
	D41RmsNorm(state->x, layer->ffn_norm, state->x, rows, D41_HIDDEN, stream);
	cudaMemcpyAsync(state->xq, state->x, (uint64_t)rows * D41_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
	if ( index < 2u && (status = D41DumpRows(state, index, 4u, state->x, rows, D41_HIDDEN, stream)) != SPARK_STATUS_OK )
		return status;
	status = D41Moe(state, layer, step, stream);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 7u, state->partial, rows, D41_HIDDEN, stream);
	if ( status == SPARK_STATUS_OK && index < 2u && state->tp_rank == 0u )
		status = D41DumpRows(state, index, 5u, state->shared_out, rows, D41_HIDDEN, stream);
	if ( status == SPARK_STATUS_OK )
		status = D41Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
	if ( status == SPARK_STATUS_OK && index < 2u )
		status = D41DumpRows(state, index, 2u, state->partial, rows, D41_HIDDEN, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	cudaMemcpyAsync(state->residual, state->streams, stream_bytes, cudaMemcpyDeviceToDevice, stream);
	D41HcPostRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->partial, state->residual, state->post_ffn,
		state->comb_ffn, state->streams, D41_HIDDEN);
	cudaMemcpyAsync(state->pre_head, state->pre_ffn, (uint64_t)rows * D41K_HC * sizeof(float), cudaMemcpyDeviceToDevice, stream);
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
	}
	D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio2, rows, 2u);
	D41CompressedEmitPositionKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(step->positions, state->bound_ratio1, rows, 1u);
	D41CompressedLengthKernel<<<(step->sequences + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio2, step->sequences, 2u);
	D41CompressedLengthKernel<<<(step->sequences + 255u) / 256u, 256u, 0, stream>>>(step->context_length, state->length_ratio1, step->sequences, 1u);
	D41ReplicateStreamsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(step->hidden_bf16, state->streams, D41_HIDDEN);
	for ( uint32_t index = 0u; index < D41_LAYERS && status == SPARK_STATUS_OK; ++index )
	{
		status = D41LayerStep(state, index, step, index == 0u ? state->pre_init : state->pre_head, stream);
		if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
	}
	if ( status != SPARK_STATUS_OK )
		return status;
	D41HcPreRowsKernel<<<dim3(D41Blocks(D41_HIDDEN), rows), D41_THREADS, 0, stream>>>(state->streams, state->pre_head, step->hidden_bf16, D41_HIDDEN);
	D41EngramHistoryKernel<<<(step->sequences + 63u) / 64u, 64u, 0, stream>>>(step->token_ids, state->token_map, step->positions,
		step->sequence_of_row, step->sequence_row_begin, step->recurrent_index,
		(int32_t *)(state->recurrent + state->ring_bytes + state->compressor_bytes), (uint32_t)(state->recurrent_bytes / sizeof(int32_t)),
		step->sequences, rows, D41_ENGRAM_ORDERS, step->commit);
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
		{ (void **)&state->streams, r * D41K_HC * D41_HIDDEN * 2u }, { (void **)&state->residual, r * D41K_HC * D41_HIDDEN * 2u },
		{ (void **)&state->x, r * D41_HIDDEN * 2u }, { (void **)&state->xq, r * D41_HIDDEN * 2u },
		{ (void **)&state->q_latent, r * D41_QLORA * 2u }, { (void **)&state->qr, r * D41_QLORA * 2u }, { (void **)&state->qrq, r * D41_QLORA * 2u },
		{ (void **)&state->q_local, r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->q_gathered, D41_TP * r * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u },
		{ (void **)&state->kv, r * D41_HEAD_DIM * 2u }, { (void **)&state->window_kv, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->mixes, r * D41K_HC_MIX * 4u },
		{ (void **)&state->pre_attn, r * D41K_HC * 4u }, { (void **)&state->post_attn, r * D41K_HC * 4u }, { (void **)&state->comb_attn, r * 16u * 4u },
		{ (void **)&state->pre_ffn, r * D41K_HC * 4u }, { (void **)&state->post_ffn, r * D41K_HC * 4u }, { (void **)&state->comb_ffn, r * 16u * 4u },
		{ (void **)&state->pre_head, r * D41K_HC * 4u }, { (void **)&state->pre_init, r * D41K_HC * 4u },
		{ (void **)&state->c_kv, r * D41_HEAD_DIM * 4u }, { (void **)&state->c_score, r * D41_HEAD_DIM * 4u },
		{ (void **)&state->c_latent, r * D41_HEAD_DIM * 2u }, { (void **)&state->c_latent_store, r * D41_HEAD_DIM * 2u },
		{ (void **)&state->idx_key, r * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_pack_local, r * D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM * 2u },
		{ (void **)&state->idx_pack_gathered, r * D41_INDEX_PACK * 2u }, { (void **)&state->idx_q_full, D41_TP * r * D41_INDEX_PACK * 2u },
		{ (void **)&state->idx_query, r * D41_INDEX_HEADS * D41_INDEX_DIM * 2u }, { (void **)&state->idx_w_full, r * D41_INDEX_HEADS * 2u },
		{ (void **)&state->emit_row, r * 4u }, { (void **)&state->bound_ratio2, r * 4u }, { (void **)&state->bound_ratio1, r * 4u },
		{ (void **)&state->length_ratio2, s * 4u }, { (void **)&state->length_ratio1, s * 4u },
		{ (void **)&state->local_scores, r * state->local_score_stride * 4u },
		{ (void **)&state->full_scores, (uint64_t)D41_SELECTION_CHUNK * state->global_score_stride * 4u },
		{ (void **)&state->topk_values, state->topk_entries * 4u }, { (void **)&state->topk_positions, state->topk_entries * 4u },
		{ (void **)&state->local_selected, r * D41_INDEX_TOPK * 4u }, { (void **)&state->selected, r * D41_INDEX_TOPK * 4u },
		{ (void **)&state->candidates, r * D41_INDEX_TOPK * 8u }, { (void **)&state->candidates_gathered, D41_TP * r * D41_INDEX_TOPK * 8u },
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
		{ (void **)&state->engram_kv, r * D41_ENGRAM_KV_ROWS * 2u },
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
		if ( cudaMalloc(&engram->multipliers, 4u * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->primes, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc(&engram->offsets, D41_ENGRAM_COLUMNS * sizeof(int64_t)) != cudaSuccess ||
			cudaMemcpy(engram->multipliers, numbers + which * 4u, 4u * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess ||
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
		if ( fread(state->engram_sha256, 1u, 64u, file) != 64u )
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
		if ( state->engram_entries[index].payload_offset + state->engram_entries[index].payload_bytes > state->engram_bytes )
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
	if ( state->dequant != 0 )
		(void)cudaFree(state->dequant);
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
	state->shard.degree = D41_TP;
	state->shard.rank = state->tp_rank;
	state->shard.grain = D41_GRAIN;
	state->local_score_stride = ((SparkKvShardGatherKeys(state->shard, state->max_positions) + 63u) / 64u) * 64u;
	state->global_score_stride = ((state->max_positions + 63u) / 64u) * 64u;
	{
		const uint64_t local = (uint64_t)state->max_rows * LmTopkExactCandidateEntries(state->local_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
		const uint64_t global = (uint64_t)D41_SELECTION_CHUNK * LmTopkExactCandidateEntries(state->global_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
		state->topk_entries = 2u * (local > global ? local : global);
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

static SparkStatus D41Dequant(D41ModelState *state, SparkWeightdLazyPack *pack, const D41PackEntry *entry, uint64_t *cursor,
	const uint16_t **out)
{
	const void *payload = 0, *scale = 0;
	if ( entry == 0 || entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 || entry->rows % 32u != 0u || entry->columns % 32u != 0u ||
		D41SliceEntry(pack, entry, 0u, &payload) != SPARK_STATUS_OK || D41SliceEntry(pack, entry, 1u, &scale) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( state->dequant != 0 )
	{
		uint16_t *target = (uint16_t *)((uint8_t *)state->dequant + *cursor);
		D41DequantFp8BlockKernel<<<dim3(D41Blocks(entry->columns), entry->rows), D41_THREADS>>>((const uint8_t *)payload,
			(const uint8_t *)scale, target, entry->rows, entry->columns);
		*out = target;
	}
	*cursor += (((uint64_t)entry->rows * entry->columns * sizeof(uint16_t)) + 255u) & ~255ull;
	return SPARK_STATUS_OK;
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
#define D41_DEQUANT(kind, layer, field) \
	do { if ( D41Dequant(state, lazy_pack, D41_ENTRY(kind, layer), &cursor, &(field)) != SPARK_STATUS_OK ) { \
			fprintf(stderr, "sparkpipe_dsv41_flash: layer %u " #kind " is not a 32x32-block fp8 tensor\n", layer); return SPARK_STATUS_PARSE_ERROR; } } while ( 0 )

static SparkStatus D41BindLayers(D41ModelState *state, SparkWeightdLazyPack *lazy_pack, uint64_t *dequant_bytes)
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
		D41_DEQUANT(Q_A, index, layer->q_a);
		D41_DEQUANT(Q_B, index, layer->q_b);
		D41_DEQUANT(KV_A, index, layer->kv_a);
		D41_DEQUANT(O_A, index, layer->o_a);
		D41_DEQUANT(O_B, index, layer->o_b);
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
		if ( state->tp_rank == 0u )
		{
			D41_DEQUANT(SHARED_W1, index, layer->sh_w1);
			D41_DEQUANT(SHARED_W2, index, layer->sh_w2);
			D41_DEQUANT(SHARED_W3, index, layer->sh_w3);
		}
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
			D41_DEQUANT(INDEXER_Q_B, index, layer->idx_q_b);
			D41_PAYLOAD(INDEXER_WEIGHTS_PROJ, index, layer->idx_wp);
		}
	}
	*dequant_bytes = cursor;
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
		if ( D41Dequant(state, state->engram_pack, wkv, cursor, &engram->wkv) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	(void)configuration;
	return SPARK_STATUS_OK;
}

static SparkStatus D41AttachEngram(D41ModelState *state, const SparkStageRunnerConfiguration *configuration)
{
	SparkWeightdLazyAttachRequest request;
	memset(&request, 0, sizeof(request));
	memcpy(request.identity.pack_sha256, state->engram_sha256, 65u);
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
	state->dequant_bytes = bytes;
	if ( cudaMalloc((void **)&state->dequant, bytes) != cudaSuccess )
	{
		state->dequant = 0;
		fprintf(stderr, "sparkpipe_dsv41_flash: %llu bytes for the dequantized fp8 projections could not be allocated\n", (unsigned long long)bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	bytes = 0u;
	status = D41BindLayers(state, lazy_pack, &bytes);
	if ( status == SPARK_STATUS_OK )
		status = D41BindEngram(state, &bytes, D41BindConfiguration);
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

#undef D41_DEQUANT
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
	fprintf(stderr, "DSV41-STATE rank=%u arena_bytes=%llu dequant_bytes=%llu slots=%u rows=%u positions=%u recurrent_bytes=%llu\n", rank,
		(unsigned long long)state->arena_bytes, (unsigned long long)state->dequant_bytes, state->slots, state->max_rows, state->max_positions,
		(unsigned long long)state->recurrent_bytes);
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
	D41Report
};

const SparkStageRunnerModelInterface *SparkDsv41FlashStageModel(void)
{
	return &spark_dsv41_flash_stage_model;
}
