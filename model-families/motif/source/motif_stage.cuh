#pragma once

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/diff_attn.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/hc_site.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/poly_norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "sparkpipe/spark_motif_family.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_named_pack_bind.h"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_weightd.h"

#define MO_HIDDEN SPARK_MOTIF_MODEL_HIDDEN_DIMENSION
#define MO_LAYERS SPARK_MOTIF_MODEL_LAYER_COUNT
#define MO_VOCAB SPARK_MOTIF_MODEL_VOCABULARY_SIZE
#define MO_EPS SPARK_MOTIF_MODEL_RMS_NORM_EPSILON
#define MO_GROUPS SPARK_MOTIF_MODEL_NOISE_HEAD_COUNT
#define MO_GROUP SPARK_MOTIF_MODEL_HEAD_GROUP_SIZE
#define MO_SIGNAL (MO_GROUP - 1u)
#define MO_HEAD SPARK_MOTIF_MODEL_HEAD_DIMENSION
#define MO_ROPE SPARK_MOTIF_MODEL_ROPE_DIMENSION
#define MO_NOPE SPARK_MOTIF_MODEL_NOPE_DIMENSION
#define MO_VALUE SPARK_MOTIF_MODEL_VALUE_DIMENSION
#define MO_QA SPARK_MOTIF_MODEL_QUERY_LORA_RANK
#define MO_LATENT SPARK_MOTIF_MODEL_KV_LORA_RANK
#define MO_SLOT SPARK_MOTIF_MODEL_LATENT_ROW_DIMENSION
#define MO_WINDOW SPARK_MOTIF_MODEL_WINDOW_TOKENS
#define MO_EXPERTS SPARK_MOTIF_MODEL_MOE_EXPERT_COUNT
#define MO_TOP_K SPARK_MOTIF_MODEL_MOE_TOP_K
#define MO_EXPERT_INTER SPARK_MOTIF_MODEL_MOE_EXPERT_INTERMEDIATE
#define MO_DENSE_INTER SPARK_MOTIF_MODEL_DENSE_INTERMEDIATE
#define MO_THREADS 256u
#define MO_RECORD LM_LATENT_SHARD_RECORD_FLOATS(MO_LATENT)

static_assert(MO_HEAD == MO_NOPE + MO_ROPE && MO_SLOT == MO_LATENT + MO_ROPE, "a query head is nope | rope and a latent slot is latent | rope");
static_assert(SPARK_MOTIF_MODEL_HC_STREAMS == LM_HC_SITE_STREAMS && SPARK_MOTIF_MODEL_HC_MIX_DIMENSION == LM_HC_SITE_MIX, "four mHC streams");
static_assert(MO_THREADS == LM_HC_SITE_THREADS, "the HC site runs the stage block size");

struct MotifKv
{
	static constexpr uint32_t kSlotBytes = SPARK_MOTIF_MODEL_LATENT_ROW_BYTES;
	static constexpr uint32_t kPageSlots = SPARK_MOTIF_MODEL_KV_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = kSlotBytes * kPageSlots;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / kPageSlots; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % kPageSlots; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + kPageSlots - 1u) / kPageSlots; }
};

struct MotifHcPolicy
{
	static constexpr float kNormEpsilon = SPARK_MOTIF_MODEL_HC_NORM_EPSILON;

	static __device__ void Gates(const float *mixes, const float *scale3, const float *base, float *pre, float *post)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES;
		const float clamp = SPARK_MOTIF_MODEL_HC_GATE_CLAMP;
		if ( lane < LM_HC_SITE_STREAMS )
		{
			pre[lane] = 1.0f / (1.0f + expf(-fminf(fmaxf(mixes[lane] * scale3[0] + base[lane], -clamp), clamp)));
			post[lane] = SPARK_MOTIF_MODEL_HC_POST_COEFFICIENT / (1.0f + expf(-fminf(fmaxf(mixes[LM_HC_SITE_STREAMS + lane] * scale3[1] +
				base[LM_HC_SITE_STREAMS + lane], -clamp), clamp)));
		}
	}

	static __device__ void Comb(const float *mixes, const float *scale3, const float *base, float *comb_out)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES, cell = lane & 15u;
		const float clamp = SPARK_MOTIF_MODEL_HC_RESIDUAL_CLAMP, floor = SPARK_MOTIF_MODEL_HC_SUM_FLOOR;
		float value = mixes[2u * LM_HC_SITE_STREAMS + cell] * scale3[2] + base[2u * LM_HC_SITE_STREAMS + cell];
		value = expf(fminf(fmaxf(value, -clamp), clamp));
		for ( uint32_t iteration = 0u; iteration < SPARK_MOTIF_MODEL_HC_SINKHORN_ITERATIONS; ++iteration )
		{
			value /= fmaxf(LmHcRowSum(value, lane), floor);
			value /= fmaxf(LmHcColumnSum(value, lane), floor);
		}
		if ( lane < LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS )
			comb_out[cell] = value;
	}
};

__global__ static void MotifQueryPackKernel(const uint16_t *__restrict__ latent, const uint16_t *__restrict__ query, uint16_t *__restrict__ packed,
	uint32_t heads)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t pair = (uint64_t)row * heads + head;
	for ( uint32_t element = threadIdx.x; element < MO_SLOT; element += blockDim.x )
		packed[pair * MO_SLOT + element] = element < MO_LATENT ? latent[pair * MO_LATENT + element] : query[pair * MO_HEAD + MO_NOPE + element - MO_LATENT];
}

__global__ static void MotifWindowListKernel(const uint32_t *__restrict__ positions, uint32_t *__restrict__ list, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint32_t position = positions[row], first = position + 1u >= MO_WINDOW ? position + 1u - MO_WINDOW : 0u;
	for ( uint32_t index = threadIdx.x; index < MO_WINDOW; index += blockDim.x )
		list[(uint64_t)row * MO_WINDOW + index] = first + index <= position ? first + index : position;
}

__global__ static void MotifRouteParamKernel(const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ param_of_row, uint32_t routes, uint32_t local_experts)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route < routes && route_expert[route] < local_experts )
		param_of_row[route_packed_row[route]] = route_expert[route];
}

struct MotifLayer
{
	const float *attn_norm, *ffn_norm;
	const float *hc_attn_fn, *hc_attn_scale, *hc_attn_base, *hc_ffn_fn, *hc_ffn_scale, *hc_ffn_base;
	const uint16_t *q_a, *kv_a, *lambda, *q_b, *gate, *out, *uk, *uv;
	const float *q_norm, *kv_norm;
	const uint16_t *ffn_gate_up, *ffn_down, *router, *moe_w13, *moe_w2, *shared_gate_up, *shared_down;
	const float *ffn_poly, *expert_bias, *moe_poly, *shared_poly;
	uint32_t window, owner;
};

template<class G>
struct MotifState
{
	static constexpr uint32_t kGroups = MO_GROUPS / G::kTpDegree;
	static constexpr uint32_t kHeads = kGroups * MO_GROUP;
	static constexpr uint32_t kSignal = kGroups * MO_SIGNAL;
	static constexpr uint32_t kLocalExperts = MO_EXPERTS / G::kTpDegree;
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open, tp_degree, tp_rank, max_rows, slots, expert_first, kv_attached, collective_capacity;
	uint64_t arena_bytes, lane_bytes, steps, prefill_rows;
	MotifLayer layers[MO_LAYERS];
	void *arena, *lane_arena;
	MotifState *lane_states[SPARK_STAGE_RUNNER_LANES];
	uint16_t *streams, *snapshot, *collapsed, *normed, *q_c, *q, *q_latent, *query, *gathered, *slot, *merged, *values, *lambda, *gate, *diff, *partial;
	float *hc_mixes, *hc_pre, *hc_post, *hc_comb, *partials, *received, *router_logits, *route_weight, *full_inv_freq;
	uint32_t *windows, *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	uint32_t *param_of_row;
	uint16_t *expert_gate_up, *expert_act, *expert_out, *mlp_gate_up, *mlp_act, *shared_out;
	LmKvAccessError *kv_error;
	LmKvShardView views[MO_LAYERS];
};

template<class G>
struct MotifStage
{
	using State = MotifState<G>;
	static constexpr uint32_t kGroups = State::kGroups, kHeads = State::kHeads, kSignal = State::kSignal, kLocalExperts = State::kLocalExperts;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		cudaStream_t stream;
		const MotifLayer *layer;
		uint32_t index;
	};

	static uint32_t QuerySequences(uint32_t rows)
	{
		return (uint32_t)(((uint64_t)rows * kHeads * MO_SLOT + MO_HIDDEN - 1u) / MO_HIDDEN);
	}

	static uint32_t RecordSequences(uint32_t rows)
	{
		return (uint32_t)(((uint64_t)rows * kHeads * MO_RECORD * 2u + MO_HIDDEN - 1u) / MO_HIDDEN);
	}

	static SparkStatus Launched(void)
	{
		return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus Check(int32_t status, const char *site, uint32_t layer)
	{
		if ( status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess )
			return SPARK_STATUS_OK;
		fprintf(stderr, "%s: %s failed at layer %u status=%d cuda=%s\n", G::Tag(), site, layer, (int)status, cudaGetErrorString(cudaPeekAtLastError()));
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	static SparkStatus Round(State *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *local, void *full)
	{
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, row_elements, local, full);
	}

	static void RowState(const State *state, uint32_t layer, uint32_t phase)
	{
		if ( state->services->row_state != 0 )
			state->services->row_state(state->services->context, layer, phase);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.scale = LmScaleTensorNone();
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			const int32_t status = LmSkinnyLaunch<LmBf16Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
	}

	static int32_t LinearFloat(const uint16_t *weight, const uint16_t *source, float *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_f32 = destination;
			args.scale = LmScaleTensorNone();
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			const int32_t status = LmSkinnyLaunch<LmBf16Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, (uint16_t *)0, destination, rows, input_dimension, output_dimension,
			0u, 0u, multiprocessors, stream);
	}

	static int32_t Norm(const uint16_t *input, const float *weight, uint16_t *output, uint32_t rows, uint32_t dimension, uint32_t stride,
		cudaStream_t stream)
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<MO_THREADS, float>), rows, MO_THREADS, (dimension + 8u) * sizeof(float), stream, input,
			(const uint16_t *)0, weight, (uint16_t *)0, output, dimension, stride, MO_EPS);
		return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( state->kv_attached == 0u || step->rows == 0u || step->rows > state->max_rows || step->sequences == 0u || step->sequences > state->slots )
		{
			fprintf(stderr, "%s: step refused rows=%u sequences=%u kv_attached=%u\n", G::Tag(), step->rows, step->sequences, state->kv_attached);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		frame->step = step;
		frame->stream = (cudaStream_t)stream;
		frame->layer = 0;
		frame->index = 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus HyperBegin(State *state, Frame &frame)
	{
		LmHcExpandStreamsKernel<MO_HIDDEN><<<dim3((MO_HIDDEN + MO_THREADS - 1u) / MO_THREADS, frame.step->rows), MO_THREADS, 0, frame.stream>>>(
			frame.step->hidden_bf16, state->streams, frame.step->rows);
		return Launched();
	}

	static void BindLayer(State *state, Frame &frame, uint32_t layer)
	{
		frame.layer = &state->layers[layer];
		frame.index = layer;
	}

	static SparkStatus HyperNorm(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		const MotifLayer *w = frame.layer;
		const uint32_t rows = frame.step->rows;
		const bool attention = part == MOTIF_ATTENTION_PART;
		if ( LmHcSite<MO_HIDDEN, MotifHcPolicy>(state->streams, attention ? w->hc_attn_fn : w->hc_ffn_fn, attention ? w->hc_attn_scale : w->hc_ffn_scale,
				attention ? w->hc_attn_base : w->hc_ffn_base, state->hc_mixes, state->hc_pre, state->hc_post, state->hc_comb, state->collapsed,
				state->snapshot, rows, frame.stream) != cudaSuccess )
			return Check(LM_LAUNCH_ERR_LAUNCH, "hc site", layer);
		return Check(Norm(state->collapsed, attention ? w->attn_norm : w->ffn_norm, state->normed, rows, MO_HIDDEN, MO_HIDDEN, frame.stream), "norm", layer);
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer, bool window)
	{
		const MotifLayer *w = frame.layer;
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		cudaStream_t stream = frame.stream;
		const float *inv_freq = window ? (const float *)0 : state->full_inv_freq;
		const float theta = window ? SPARK_MOTIF_MODEL_WINDOW_ROPE_THETA : SPARK_MOTIF_MODEL_ROPE_THETA;
		const uint64_t query_stride = (uint64_t)QuerySequences(rows) * MO_HIDDEN, record_stride = (uint64_t)RecordSequences(rows) * MO_HIDDEN / 2u;
		SparkStatus status;
		RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		status = Check(Linear(w->q_a, state->normed, state->q_c, rows, MO_HIDDEN, MO_QA, sms, stream), "q_a", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Norm(state->q_c, w->q_norm, state->q_c, rows, MO_QA, MO_QA, stream), "q norm", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(w->q_b, state->q_c, state->q, rows, MO_QA, kHeads * MO_HEAD, sms, stream), "q_b", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(w->gate, state->q_c, state->gate, rows, MO_QA, kSignal * MO_VALUE, sms, stream), "gate", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(w->lambda, state->normed, state->lambda, rows, MO_HIDDEN, MO_GROUPS * MO_SIGNAL, sms, stream), "lambda", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(w->kv_a, state->normed, state->slot, rows, MO_HIDDEN, MO_SLOT, sms, stream), "kv_a", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Norm(state->slot, w->kv_norm, state->slot, rows, MO_LATENT, MO_SLOT, stream), "kv norm", layer);
		if ( status != SPARK_STATUS_OK )
			return status;
		LM_LAUNCH((LmRopePerHeadKernel<MO_THREADS>), dim3(rows, kHeads), MO_THREADS, 0, stream, state->q, step->positions, kHeads, MO_HEAD, MO_ROPE, theta,
			inv_freq, 1.0f, MO_NOPE);
		LM_LAUNCH((LmRopePerHeadKernel<MO_THREADS>), dim3(rows, 1u), MO_THREADS, 0, stream, state->slot, step->positions, 1u, MO_SLOT, MO_ROPE, theta,
			inv_freq, 1.0f, MO_LATENT);
		if ( LmPerHeadProjectRowsLaunch<MO_THREADS, MO_NOPE, MO_LATENT, MO_HEAD, 0u>(state->q, w->uk, state->q_latent, kHeads, rows, stream) != cudaSuccess )
			return Check(LM_LAUNCH_ERR_LAUNCH, "query absorb", layer);
		MotifQueryPackKernel<<<dim3(rows, kHeads), MO_THREADS, 0, stream>>>(state->q_latent, state->q, state->query, kHeads);
		LM_LAUNCH((LmKvShardStoreKernel<MotifKv, MO_THREADS>), rows, MO_THREADS, 0, stream, state->views[layer], state->slot, step->sequence_of_row,
			step->positions, rows, MO_SLOT);
		if ( window )
			MotifWindowListKernel<<<rows, MO_THREADS, 0, stream>>>(step->positions, state->windows, rows);
		status = Check(LM_LAUNCH_OK, "kv store", layer);
		RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status == SPARK_STATUS_OK )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, QuerySequences(rows), 0u, state->query, state->gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardPartialLaunch<MotifKv, LmKvShardView, MO_LATENT, MO_ROPE>(state->views[layer], state->gathered, query_stride, kHeads,
				step->sequence_of_row, step->context_length, step->positions, window ? state->windows : (const uint32_t *)0, window ? MO_WINDOW : 0u,
				window ? MO_WINDOW : 0u, window ? SPARK_MOTIF_MODEL_WINDOW_SOFTMAX_SCALE : SPARK_MOTIF_MODEL_FULL_SOFTMAX_SCALE, state->partials, record_stride,
				rows, stream) != cudaSuccess )
			return Check(LM_LAUNCH_ERR_LAUNCH, "attention partial", layer);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, RecordSequences(rows), 0u, state->partials, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardMergeLaunch<MO_LATENT>(state->received, record_stride, degree, kHeads, state->merged, rows, stream) != cudaSuccess ||
			LmPerHeadProjectRowsLaunch<MO_THREADS, MO_LATENT, MO_VALUE>(state->merged, w->uv, state->values, kHeads, rows, stream) != cudaSuccess ||
			LmDifferentialGateLaunch(state->values, state->lambda + (uint64_t)state->tp_rank * kSignal, MO_GROUPS * MO_SIGNAL, state->gate, state->diff,
				rows, kGroups, MO_SIGNAL, MO_VALUE, stream) != cudaSuccess )
			return Check(LM_LAUNCH_ERR_LAUNCH, "attention merge", layer);
		return Check(Linear(w->out, state->diff, state->partial, rows, kSignal * MO_VALUE, MO_HIDDEN, sms, stream), "attention out", layer);
	}

	static SparkStatus FullAttention(State *state, Frame &frame, uint32_t layer)
	{
		return Attention(state, frame, layer, false);
	}

	static SparkStatus WindowAttention(State *state, Frame &frame, uint32_t layer)
	{
		return Attention(state, frame, layer, true);
	}

	static SparkStatus AllReduce(State *state, Frame &frame)
	{
		return Round(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus HyperPost(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		(void)part;
		LmHcPostF32Kernel<<<dim3(frame.step->rows, MO_HIDDEN / MO_THREADS), MO_THREADS, 0, frame.stream>>>(state->partial, state->snapshot, state->hc_post,
			state->hc_comb, state->streams, frame.step->rows, LM_HC_SITE_STREAMS, MO_HIDDEN);
		return Check(LM_LAUNCH_OK, "hc post", layer);
	}

	static SparkStatus OwnedMlp(State *state, Frame &frame, uint32_t layer, const uint16_t *gate_up, const uint16_t *down, const float *poly,
		uint32_t inter, uint16_t *output)
	{
		const uint32_t rows = frame.step->rows, sms = frame.step->multiprocessors;
		SparkStatus status = Check(Linear(gate_up, state->normed, state->mlp_gate_up, rows, MO_HIDDEN, 2u * inter, sms, frame.stream), "mlp gate_up", layer);
		if ( status == SPARK_STATUS_OK && LmPolyNormMulLaunch<false>(state->mlp_gate_up, poly, (const uint32_t *)0, state->mlp_act, rows, inter,
				SPARK_MOTIF_MODEL_HIDDEN_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE,
				SPARK_MOTIF_MODEL_POLYNORM_EPSILON, frame.stream) != cudaSuccess )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "mlp polynorm", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(down, state->mlp_act, output, rows, inter, MO_HIDDEN, sms, frame.stream), "mlp down", layer);
		return status;
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		const MotifLayer *w = frame.layer;
		SparkStatus status = SPARK_STATUS_OK;
		if ( w->owner != 0u )
			status = OwnedMlp(state, frame, layer, w->ffn_gate_up, w->ffn_down, w->ffn_poly, MO_DENSE_INTER, state->partial);
		else if ( cudaMemsetAsync(state->partial, 0, (uint64_t)frame.step->rows * MO_HIDDEN * sizeof(uint16_t), frame.stream) != cudaSuccess )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "dense mlp clear", layer);
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return status;
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const MotifLayer *w = frame.layer;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * MO_TOP_K;
		LmLocalExpertRoutes expert_routes;
		SparkStatus status = Check(LinearFloat(w->router, state->normed, state->router_logits, rows, MO_HIDDEN, MO_EXPERTS, sms, stream), "router", layer);
		if ( status == SPARK_STATUS_OK && LmTopkGroupLimitedLaunch<MO_THREADS, MO_EXPERTS, 1u, 1u, MO_TOP_K>(rows, state->router_logits, w->expert_bias,
				state->route_expert, state->route_weight, SPARK_MOTIF_MODEL_MOE_ROUTED_SCALE, stream) != cudaSuccess )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "router top-k", layer);
		if ( status != SPARK_STATUS_OK )
			return status;
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global, routes,
			state->expert_first, kLocalExperts);
		if ( LmRouteBuild<MO_THREADS, kLocalExperts + 1u>(state->route_expert, rows, routes, MO_TOP_K, state->group_row_offset, state->route_packed_row,
				state->route_source_token, 2u * MO_EXPERT_INTER, MO_HIDDEN, LM_MOE_GEMM_TILE_N, state->tile_prefix_up, state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			return Check(LM_LAUNCH_ERR_LAUNCH, "route build", layer);
		if ( cudaMemsetAsync(state->param_of_row, 0, (uint64_t)routes * sizeof(uint32_t), stream) != cudaSuccess )
			return Check(LM_LAUNCH_ERR_LAUNCH, "route params", layer);
		MotifRouteParamKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_packed_row, state->param_of_row, routes, kLocalExperts);
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = MO_TOP_K;
		expert_routes.local_experts = kLocalExperts;
		expert_routes.multiprocessors = sms;
		status = Check(LmLocalExpertsUp<LmBf16Format>(expert_routes, w->moe_w13, LmScaleTensorNone(), state->normed, state->expert_gate_up, MO_HIDDEN,
			2u * MO_EXPERT_INTER, stream), "experts up", layer);
		if ( status == SPARK_STATUS_OK && LmPolyNormMulLaunch<true>(state->expert_gate_up, w->moe_poly, state->param_of_row, state->expert_act, routes,
				MO_EXPERT_INTER, SPARK_MOTIF_MODEL_HIDDEN_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE,
				SPARK_MOTIF_MODEL_POLYNORM_EPSILON, stream) != cudaSuccess )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "experts polynorm", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(LmLocalExpertsDown<LmBf16Format>(expert_routes, w->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out,
				MO_EXPERT_INTER, MO_HIDDEN, stream), "experts down", layer);
		if ( status == SPARK_STATUS_OK && w->owner != 0u )
			status = OwnedMlp(state, frame, layer, w->shared_gate_up, w->shared_down, w->shared_poly, MO_EXPERT_INTER, state->shared_out);
		if ( status == SPARK_STATUS_OK )
		{
			LmMoeLocalWeightedFinalizeKernel<<<dim3((MO_HIDDEN + MO_THREADS - 1u) / MO_THREADS, rows), MO_THREADS, 0, stream>>>(state->expert_out,
				state->route_packed_row, state->route_expert, state->route_weight, w->owner != 0u ? state->shared_out : (const uint16_t *)0, state->partial,
				MO_TOP_K, kLocalExperts, MO_HIDDEN);
			status = Check(LM_LAUNCH_OK, "experts finalize", layer);
		}
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return status;
	}

	static SparkStatus HyperEnd(State *state, Frame &frame)
	{
		const uint32_t rows = frame.step->rows;
		LmHcStreamsMeanKernel<MO_HIDDEN><<<rows, MO_THREADS, 0, frame.stream>>>(state->streams, frame.step->hidden_bf16, rows);
		if ( cudaPeekAtLastError() != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		state->steps++;
		if ( rows > frame.step->sequences )
			state->prefill_rows += rows;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream)
	{
		return MotifSchedule<G, MotifStage>::Step((State *)model, step, stream);
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, routes = rows * MO_TOP_K, groups = kLocalExperts + 2u, degree = state->tp_degree, b = sizeof(uint16_t);
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->streams, rows * LM_HC_SITE_STREAMS * MO_HIDDEN * b },
			{ (void **)&state->snapshot, rows * LM_HC_SITE_STREAMS * MO_HIDDEN * b },
			{ (void **)&state->collapsed, rows * MO_HIDDEN * b },
			{ (void **)&state->normed, rows * MO_HIDDEN * b },
			{ (void **)&state->hc_mixes, rows * LM_HC_SITE_MIX * sizeof(float) },
			{ (void **)&state->hc_pre, rows * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->hc_post, rows * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->hc_comb, rows * LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->q_c, rows * MO_QA * b },
			{ (void **)&state->q, rows * kHeads * MO_HEAD * b },
			{ (void **)&state->q_latent, rows * kHeads * MO_LATENT * b },
			{ (void **)&state->query, (uint64_t)QuerySequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->gathered, degree * QuerySequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->slot, rows * MO_SLOT * b },
			{ (void **)&state->windows, rows * MO_WINDOW * sizeof(uint32_t) },
			{ (void **)&state->partials, degree * RecordSequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->received, degree * RecordSequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->merged, rows * kHeads * MO_LATENT * b },
			{ (void **)&state->values, rows * kHeads * MO_VALUE * b },
			{ (void **)&state->lambda, rows * MO_GROUPS * MO_SIGNAL * b },
			{ (void **)&state->gate, rows * kSignal * MO_VALUE * b },
			{ (void **)&state->diff, rows * kSignal * MO_VALUE * b },
			{ (void **)&state->partial, rows * MO_HIDDEN * b },
			{ (void **)&state->router_logits, rows * MO_EXPERTS * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->param_of_row, routes * sizeof(uint32_t) },
			{ (void **)&state->expert_gate_up, routes * 2u * MO_EXPERT_INTER * b },
			{ (void **)&state->expert_act, routes * MO_EXPERT_INTER * b },
			{ (void **)&state->expert_out, routes * MO_HIDDEN * b },
			{ (void **)&state->mlp_gate_up, rows * 2u * MO_DENSE_INTER * b },
			{ (void **)&state->mlp_act, rows * MO_DENSE_INTER * b },
			{ (void **)&state->shared_out, rows * MO_HIDDEN * b },
			{ (void **)&state->full_inv_freq, lane == 0u ? (MO_ROPE / 2u) * sizeof(float) : 0u },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		uint64_t cursor = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
		{
			cursor = (cursor + 255u) & ~255ull;
			if ( base != 0 )
				*plan[index].target = plan[index].bytes != 0u ? base + cursor : 0;
			cursor += plan[index].bytes;
		}
		*bytes = cursor;
		return SPARK_STATUS_OK;
	}

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->lane_arena != 0 )
			(void)cudaFree(state->lane_arena);
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
			free(state->lane_states[lane]);
		if ( state->pack_open != 0u )
			SparkNamedPackClose(&state->pack);
		free(state);
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, groups = 0u, group_first = 0u, experts = 0u, local = 0u, embed_rows = 0u;
		State *state;
		SparkStatus status;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->max_input_row_count == 0u ||
			configuration->state_budget_bytes == 0u || configuration->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, G::kPackAlignment, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "head_groups", &groups) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "head_group_first", &group_first) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != MO_HIDDEN || layers != MO_LAYERS ||
			vocab_rows * degree != MO_VOCAB || groups != kGroups || group_first != rank * kGroups || experts != MO_EXPERTS || local != kLocalExperts ||
			state->expert_first != rank * kLocalExperts )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u groups %u+%u experts %u local %u first %u; the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, group_first, groups, experts, local, state->expert_first, configuration->tp_degree,
				configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		state->collective_capacity = configuration->device_collective != 0 ? configuration->device_collective->max_active_sequence_count : 0u;
		if ( state->max_rows > state->collective_capacity || QuerySequences(state->max_rows) > state->collective_capacity ||
			RecordSequences(state->max_rows) > state->collective_capacity )
		{
			fprintf(stderr, "%s: %u rows need collectives of %u query and %u record sequences; the collective carries %u\n", G::Tag(), state->max_rows,
				QuerySequences(state->max_rows), RecordSequences(state->max_rows), state->collective_capacity);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		status = Plan(state, &state->arena_bytes, 0, 0u);
		if ( status == SPARK_STATUS_OK )
			status = Plan(state, &state->lane_bytes, 0, 1u);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes + state->lane_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes (%llu for the second prefill lane) is over the %llu-byte budget\n", G::Tag(),
				(unsigned long long)(state->arena_bytes + state->lane_bytes), (unsigned long long)state->lane_bytes,
				(unsigned long long)configuration->state_budget_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = MO_HIDDEN;
		geometry->vocab = MO_VOCAB;
		geometry->total_layers = MO_LAYERS;
		geometry->layer_count = MO_LAYERS;
		geometry->rms_epsilon = MO_EPS;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = MO_LAYERS;
		geometry->kv_layer_page_bytes = MotifKv::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->head_norm_f32 = 1u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		geometry->embed_rows = embed_rows;
		if ( geometry->embed_bytes != (uint64_t)embed_rows * MO_HIDDEN * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * MO_HIDDEN * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)MO_HIDDEN * sizeof(float) )
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: the rank's experts are resident tensors; its weightd manifest carries no expert ranges\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, T **field, uint64_t bytes)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		const void *bound = 0;
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		const SparkStatus status = SparkNamedPackBindSized(&state->pack, lazy_pack, G::Tag(), key, kind, bytes, &bound);
		*field = (T *)bound;
		return status;
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		const uint64_t h = MO_HIDDEN, b = sizeof(uint16_t), f = sizeof(float), mix = LM_HC_SITE_MIX * LM_HC_SITE_STREAMS * h;
		float inv_freq[MO_ROPE / 2u];
		(void)geometry;
		for ( uint32_t index = 0u; index < MO_LAYERS; ++index )
		{
			MotifLayer *w = &state->layers[index];
			SparkStatus s = SPARK_STATUS_OK;
			w->window = MotifLayout::Window(index) ? 1u : 0u;
			w->owner = index % state->tp_degree == state->tp_rank ? 1u : 0u;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_norm", "f32", &w->attn_norm, h * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "ffn_norm", "f32", &w->ffn_norm, h * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_attn_fn", "f32", &w->hc_attn_fn, mix * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_attn_scale", "f32", &w->hc_attn_scale, 3u * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_attn_base", "f32", &w->hc_attn_base, LM_HC_SITE_MIX * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_ffn_fn", "f32", &w->hc_ffn_fn, mix * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_ffn_scale", "f32", &w->hc_ffn_scale, 3u * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "hc_ffn_base", "f32", &w->hc_ffn_base, LM_HC_SITE_MIX * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_q_a", "bf16", &w->q_a, (uint64_t)MO_QA * h * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_q_norm", "f32", &w->q_norm, (uint64_t)MO_QA * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_kv_a", "bf16", &w->kv_a, (uint64_t)MO_SLOT * h * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_kv_norm", "f32", &w->kv_norm, (uint64_t)MO_LATENT * f) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_lambda", "bf16", &w->lambda, (uint64_t)MO_GROUPS * MO_SIGNAL * h * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_q_b", "bf16", &w->q_b, (uint64_t)kHeads * MO_HEAD * MO_QA * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_gate", "bf16", &w->gate, (uint64_t)kSignal * MO_VALUE * MO_QA * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_out", "bf16", &w->out, h * kSignal * MO_VALUE * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_uk", "bf16", &w->uk, (uint64_t)kHeads * MO_LATENT * MO_NOPE * b) : s;
			s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "attn_uv", "bf16", &w->uv, (uint64_t)kHeads * MO_VALUE * MO_LATENT * b) : s;
			if ( !MotifLayout::Routed(index) && w->owner != 0u )
			{
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "ffn_gate_up", "bf16", &w->ffn_gate_up, 2ull * MO_DENSE_INTER * h * b) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "ffn_down", "bf16", &w->ffn_down, h * MO_DENSE_INTER * b) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "ffn_poly", "f32", &w->ffn_poly, LM_POLY_NORM_PARAMS * f) : s;
			}
			if ( MotifLayout::Routed(index) )
			{
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "router_proj", "bf16", &w->router, (uint64_t)MO_EXPERTS * h * b) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "expert_bias", "f32", &w->expert_bias, (uint64_t)MO_EXPERTS * f) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "moe_w13", "bf16", &w->moe_w13, (uint64_t)kLocalExperts * 2u * MO_EXPERT_INTER * h * b) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "moe_w2", "bf16", &w->moe_w2, (uint64_t)kLocalExperts * h * MO_EXPERT_INTER * b) : s;
				s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "moe_poly", "f32", &w->moe_poly, (uint64_t)kLocalExperts * LM_POLY_NORM_PARAMS * f) : s;
				if ( w->owner != 0u )
				{
					s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "shared_gate_up", "bf16", &w->shared_gate_up, 2ull * MO_EXPERT_INTER * h * b) : s;
					s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "shared_down", "bf16", &w->shared_down, h * MO_EXPERT_INTER * b) : s;
					s = s == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, "shared_poly", "f32", &w->shared_poly, LM_POLY_NORM_PARAMS * f) : s;
				}
			}
			if ( s != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena, 0u) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( cudaMalloc(&state->lane_arena, state->lane_bytes) != cudaSuccess )
		{
			state->lane_arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->lane_arena, 0, state->lane_bytes) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
			if ( (state->lane_states[lane] = (State *)calloc(1u, sizeof(State))) == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		SparkRopeYarnRampTable(inv_freq, MO_ROPE, SPARK_MOTIF_MODEL_ROPE_THETA, SPARK_MOTIF_MODEL_YARN_FACTOR, SPARK_MOTIF_MODEL_YARN_BETA_FAST,
			SPARK_MOTIF_MODEL_YARN_BETA_SLOW, SPARK_MOTIF_MODEL_YARN_ORIGINAL_CONTEXT);
		if ( cudaMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), cudaMemcpyHostToDevice) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != MO_LAYERS || kv->layer_page_bytes != MotifKv::kPageBytes || kv->sequence_count > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < MO_LAYERS; ++layer )
			if ( LmKvShardViewInitialize<MotifKv>(&state->views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes, kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		(void)model; (void)to_buffer; (void)slot; (void)buffer; (void)bytes; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		(void)model; (void)slot; (void)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		const State *state = (const State *)model;
		fprintf(stderr, "%s rank=%u arena_bytes=%llu lane_bytes=%llu slots=%u rows=%u expert_first=%u steps=%llu prefill_rows=%llu\n", G::ReportTag(),
			rank, (unsigned long long)state->arena_bytes, (unsigned long long)state->lane_bytes, state->slots, state->max_rows, state->expert_first,
			(unsigned long long)state->steps, (unsigned long long)state->prefill_rows);
	}

	static SparkStatus Lane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
	{
		State *state = (State *)model;
		uint64_t bytes = 0u;
		if ( lane >= SPARK_STAGE_RUNNER_LANES || step == 0 || services == 0 || lane_model == 0 || state->lane_states[lane] == 0 || state->lane_arena == 0 )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		if ( step->commit == 0u )
			return SPARK_STATUS_UNSUPPORTED;
		State *copy = state->lane_states[lane];
		memcpy(copy, state, sizeof(*copy));
		copy->services = services;
		if ( lane != 0u )
		{
			if ( Plan(copy, &bytes, (uint8_t *)state->lane_arena, lane) != SPARK_STATUS_OK || bytes != state->lane_bytes )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			copy->full_inv_freq = state->full_inv_freq;
			copy->kv_error = state->kv_error;
		}
		*lane_model = copy;
		return SPARK_STATUS_OK;
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::WeightdModel(),
			G::WeightdRevision(),
			Open,
			ManifestCheck,
			BindModel,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report,
			0,
			Lane
		};
		return &interface_table;
	}
};
