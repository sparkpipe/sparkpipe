#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_tp_sequences.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_k3_stage_model.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_error_site.h"
#include "inference/llms/kimi_k3/layer.cuh"

#define K3_RUNNER_STAGE_ALL SPARK_STAGE_RUNNER_ROUND_ALL
#define K3_RUNNER_STAGE_BEGIN SPARK_STAGE_RUNNER_ROUND_BEGIN
#define K3_RUNNER_STAGE_FINISH SPARK_STAGE_RUNNER_ROUND_FINISH
#define K3_RUNNER_PAIR_MIN_ROWS 256u

typedef struct SparkK3ModelState SparkK3ModelState;

typedef struct SparkK3RunnerLane
{
	SparkK3ModelState *state;
	uint32_t index;
} SparkK3RunnerLane;

#define K3_RUNNER_PAIR_MIN_ROWS 256u

__global__ static void K3RunnerDenseOffsetsKernel(uint32_t *offsets, uint32_t rows)
{
	if ( threadIdx.x == 0u )
	{
		offsets[0] = 0u;
		offsets[1] = rows;
	}
}

typedef struct SparkK3RunnerTpContext SparkK3RunnerTpContext;


#define K3_RUNNER_TP_CONTEXT_POOL_DEPTH (2u * K3_LAYERS)

static_assert(K3_RUNNER_TP_CONTEXT_POOL_DEPTH >= 2u * K3_LAYERS,
	"tp context pool must cover both per-layer collectives");

#define K3_RUNNER_GATE_UP_WIDTH (K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u))

typedef struct SparkK3RunnerTpContext
{
	SparkK3RunnerTpContext *pool_next;
	SparkK3ModelState *owner;
	K3LayerBuffers *buffers;
	uint16_t *fused;
	cudaStream_t stream;
	uint32_t rows;
	uint32_t boundary;
	uint32_t phase;
	uint32_t gate_up_elements;
} SparkK3RunnerTpContext;

static SparkK3RunnerTpContext *K3RunnerTpContextAcquire(
	SparkK3ModelState *state);
static void K3RunnerTpContextRelease(SparkK3RunnerTpContext *context);
static void K3RunnerTpApply(SparkK3RunnerTpContext *tp);


__global__ static void K3RunnerFusedPackKernel(const uint16_t *attention,
	const uint16_t *hidden,const uint16_t *shared,const uint16_t *gate_up,
	uint16_t *fused,uint32_t rows,uint32_t phase,uint32_t segments,
	uint32_t gate_up_elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * K3_HIDDEN;
	if ( phase == 2u )
	{
		if ( i >= gate_up_elements )
			return;
		fused[i] = gate_up[i];
		return;
	}
	if ( i >= elements )
		return;
	if ( phase == 0u )
		fused[i] = attention[i];
	else
		fused[i] = segments == 2u ? LmFloatToBf16(LmBf16ToFloat(hidden[i]) + LmBf16ToFloat(shared[i])) : hidden[i];
}

__global__ static void K3RunnerLatentRowsKernel(const uint16_t *gathered, uint16_t *latent,
	uint32_t rows, uint32_t slice, uint32_t ranks)
{
	const uint32_t row = blockIdx.y, column = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( column >= slice * ranks )
		return;
	latent[(uint64_t)row * slice * ranks + column] =
		gathered[((uint64_t)(column / slice) * rows + row) * slice + (column % slice)];
}

__global__ static void K3RunnerMlaDownRowsKernel(const uint16_t *gathered, uint16_t *query,
	uint16_t *key, uint32_t rows, uint32_t query_slice, uint32_t key_slice)
{
	const uint32_t row = blockIdx.y, column = (blockIdx.x * blockDim.x) + threadIdx.x;
	const uint32_t stride = query_slice + key_slice;
	uint32_t rank;
	if ( column < K3_Q_LORA_RANK )
	{
		rank = column / query_slice;
		query[(uint64_t)row * K3_Q_LORA_RANK + column] =
			gathered[((uint64_t)rank * rows + row) * stride + column % query_slice];
	}
	else if ( column < K3_Q_LORA_RANK + K3_MLA_KV_A_DIM )
	{
		rank = (column - K3_Q_LORA_RANK) / key_slice;
		key[(uint64_t)row * K3_MLA_KV_A_DIM + column - K3_Q_LORA_RANK] =
			gathered[((uint64_t)rank * rows + row) * stride + query_slice + (column - K3_Q_LORA_RANK) % key_slice];
	}
}

__global__ static void K3RunnerLatentLogitsKernel(const uint16_t *gathered, uint16_t *latent,
	float *logits, uint32_t rows, uint32_t slice, uint32_t experts, uint32_t ranks)
{
	const uint32_t row = blockIdx.y, column = (blockIdx.x * blockDim.x) + threadIdx.x;
	const uint64_t stride = (uint64_t)rows * (slice + 2u * experts);
	uint32_t expert;
	if ( column < slice * ranks )
	{
		latent[(uint64_t)row * slice * ranks + column] =
			gathered[(column / slice) * stride + (uint64_t)row * slice + (column % slice)];
		return;
	}
	expert = column - slice * ranks;
	if ( expert >= experts * ranks )
		return;
	logits[(uint64_t)row * experts * ranks + expert] =
		((const float *)(gathered + (expert / experts) * stride + (uint64_t)rows * slice))[(uint64_t)row * experts + (expert % experts)];
}

#define K3_RUNNER_GATHER_SOURCES_MAX 16u


struct SparkK3ModelState
{
	SparkK3ModuleState module;
	SparkK3Dispatch dispatch;
	const SparkStageRunnerServices *services;
	uint32_t has_collective;
	uint32_t deferred;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint64_t kv_page_bytes;
	uint64_t layer_w1_offset[K3_LAYERS];
	uint64_t layer_w2_offset[K3_LAYERS];
	uint16_t *fused_device;
	uint32_t fused_rows;
	SparkK3RunnerTpContext *tp_context_free_head;
	SparkK3RunnerTpContext tp_context_pool[K3_RUNNER_TP_CONTEXT_POOL_DEPTH];
	uint32_t tp_context_overflow;
	SparkK3RunnerTpContext *pending_tp;
	cudaStream_t load_stream;
	cudaStream_t pair_stream[2];
	cudaEvent_t pair_fence;
	cudaEvent_t load_fork;
	cudaEvent_t load_join;
	uint32_t rows;
	uint32_t head_last_rows;
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	SparkK3RunnerLane lanes[2];
	void (*layer_collective_override)(void *context, void *stream, uint32_t layer, uint32_t phase);
	void *layer_collective_context;
};

static void K3ModelFail(SparkK3ModelState *state)
{
	state->services->fail(state->services->context, SPARK_STATUS_INTERNAL_ERROR);
}

static cudaError_t K3RunnerCopy(void *destination, const void *source, uint64_t bytes, cudaStream_t stream)
{
	cudaError_t error = cudaMemcpyAsync(destination, source, (size_t)bytes, cudaMemcpyDefault, stream);
	if ( error != cudaSuccess )
		return error;
	return cudaStreamSynchronize(stream);
}

static uint64_t K3ModelRecurrentBytes(const SparkK3ModelState *state)
{
	SparkK3KdaRankLayout layout;
	if ( state->dispatch.kda_count == 0u || SparkK3KdaRankLayoutFor(state->dispatch.tp_degree, &layout) == 0u )
		return 0u;
	return (uint64_t)state->dispatch.kda_count * (layout.state_slot_bytes + 2u * layout.qk_window_slot_bytes + layout.v_window_slot_bytes);
}

static SparkK3RunnerTpContext *K3RunnerTpContextAcquire(
	SparkK3ModelState *state)
{
	SparkK3RunnerTpContext *context = state->tp_context_free_head;
	if ( context == 0 )
	{
		state->tp_context_overflow = 1u;
		return 0;
	}
	state->tp_context_free_head = context->pool_next;
	return context;
}

static void K3RunnerTpContextRelease(SparkK3RunnerTpContext *context)
{
	SparkK3ModelState *state = context->owner;
	context->pool_next = state->tp_context_free_head;
	state->tp_context_free_head = context;
}

static void K3RunnerTpApply(SparkK3RunnerTpContext *tp)
{
	K3LayerBuffers *b = tp->buffers;
	uint32_t rows = tp->rows;
	uint16_t *fused = tp->fused;
	if ( tp->phase == 2u )
	{
		if ( cudaMemcpyAsync(b->gate_up_bf16, fused,
			(uint64_t)tp->gate_up_elements * sizeof(*b->gate_up_bf16),
			cudaMemcpyDeviceToDevice, tp->stream) != cudaSuccess )
			tp->owner->services->fail(tp->owner->services->context, SPARK_STATUS_IO_ERROR);
		K3RunnerTpContextRelease(tp);
		return;
	}
	if ( tp->phase == 0u )
	{
		if ( tp->boundary != 0u )
			K3PartialSet(b, fused, rows, tp->stream);
		else
			K3PartialAdd(b, fused, rows, tp->stream);
	}
	else
		K3PartialAdd(b, fused, rows, tp->stream);
	K3RunnerTpContextRelease(tp);
}


#define K3_RUNNER_PP_STAGE_COUNT 4u
#define K3_RUNNER_PP_STAGE_BASE_LAYERS (K3_LAYERS / K3_RUNNER_PP_STAGE_COUNT)
#define K3_RUNNER_PP_STAGE_REMAINDER (K3_LAYERS % K3_RUNNER_PP_STAGE_COUNT)
#define K3_RUNNER_PP_STAGE_LAYERS(stage_index) \
	(K3_RUNNER_PP_STAGE_BASE_LAYERS + \
	((stage_index) < K3_RUNNER_PP_STAGE_REMAINDER ? 1u : 0u))
#define K3_RUNNER_PP_STAGE_FIRST(stage_index) \
	((stage_index) * K3_RUNNER_PP_STAGE_BASE_LAYERS + \
	((stage_index) < K3_RUNNER_PP_STAGE_REMAINDER ? \
	(stage_index) : K3_RUNNER_PP_STAGE_REMAINDER))

static_assert(K3_RUNNER_PP_STAGE_FIRST(K3_RUNNER_PP_STAGE_COUNT - 1u) +
	K3_RUNNER_PP_STAGE_LAYERS(K3_RUNNER_PP_STAGE_COUNT - 1u) ==
	K3_LAYERS,
	"pp stage bounds must tile the k3 layer stack");

static_assert(SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW == K3_ATTNRES_BANK_BYTES,
	"the pipeline residual bank sideband carries every attention-residual slot");

static_assert(K3_LAYERS == SPARK_K3_MODULE_TOTAL_LAYERS,
	"k3 kernel layer count must equal the module layer total");
static uint32_t K3RunnerFirstLayer(uint32_t stage_index)
{
	return(SPARK_K3_PP_STAGE_FIRST(stage_index % SPARK_K3_PP_STAGE_COUNT));
}

static uint32_t K3RunnerLayerCount(uint32_t stage_index)
{
	return(SPARK_K3_PP_STAGE_LAYERS(stage_index % SPARK_K3_PP_STAGE_COUNT));
}


static void K3RunnerRound(SparkK3ModelState *state, cudaStream_t stream, uint32_t rows,
	const void *local, void *full, uint32_t row_elements, uint32_t operation, uint32_t stage)
{
	if ( state->has_collective == 0 )
	{
		K3ModelFail(state);
		return;
	}
	(void)state->services->round(state->services->context, (void *)stream, operation, stage, rows, row_elements, local, full);
}

static void K3RunnerShardExchange(SparkK3ModelState *state, K3LayerBuffers *b,
	cudaStream_t stream, uint32_t rows, uint32_t phase)
{
	uint32_t degree = b->kv_shard.degree;
	uint32_t partials = (phase & ~K3_COLLECTIVE_FINISH) == K3_COLLECTIVE_MLA_PARTIALS ? 1u : 0u;
	if ( degree < 2u )
	{
		K3ModelFail(state);
		return;
	}
	if ( (phase & ~K3_COLLECTIVE_FINISH) == K3_COLLECTIVE_MLA_KEYS )
	{
		if ( b->shard_gather == 0u || b->shard_gather_stride % ((uint64_t)K3_HIDDEN * sizeof(uint16_t)) != 0u )
		{
			K3ModelFail(state);
			return;
		}
		K3RunnerRound(state, stream, (uint32_t)(b->shard_gather_stride / ((uint64_t)K3_HIDDEN * sizeof(uint16_t))),
			b->shard_gather_send, (void *)b->shard_gather_keys, 0u, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
			(phase & K3_COLLECTIVE_FINISH) != 0u ? K3_RUNNER_STAGE_FINISH : K3_RUNNER_STAGE_BEGIN);
		return;
	}
	K3RunnerRound(state, stream,
		partials != 0u ? SparkK3KvShardPartialSequences(rows, degree)
			: SparkK3KvShardQuerySequences(rows, degree),
		partials != 0u ? (const void *)b->shard_partials_f32 : (const void *)b->query_bf16,
		partials != 0u ? (void *)b->shard_partials_received_f32 : (void *)b->shard_query_gathered_bf16, 0u,
		partials != 0u ? SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL : SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
		partials != 0u ? K3_RUNNER_STAGE_ALL : (phase & K3_COLLECTIVE_FINISH) != 0u ? K3_RUNNER_STAGE_FINISH : K3_RUNNER_STAGE_BEGIN);
}

static void K3RunnerGatherLatent(SparkK3ModelState *state, K3LayerBuffers *b, cudaStream_t stream, uint32_t rows,
	uint32_t with_logits, uint32_t stage)
{
	const uint32_t slice = K3_RANK_DIM(b, routed_down_rows, K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t ranks = K3_ROUTED_EXPERT_HIDDEN / slice;
	const uint32_t experts = with_logits != 0u ? K3_EXPERTS / ranks : 0u;
	uint16_t *gathered = rows == 1u && with_logits == 0u ? b->latent_full_bf16 : state->fused_device;
	if ( with_logits != 0u && experts * ranks != K3_EXPERTS )
	{
		K3ModelFail(state);
		return;
	}
	K3RunnerRound(state, stream, rows, b->latent_bf16, gathered, slice + 2u * experts,
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, stage);
	if ( stage == K3_RUNNER_STAGE_BEGIN )
		return;
	if ( with_logits != 0u )
		K3RunnerLatentLogitsKernel<<<dim3((K3_ROUTED_EXPERT_HIDDEN + K3_EXPERTS + 255u) / 256u, rows), 256u, 0, stream>>>(
			gathered, b->latent_full_bf16, b->router_logits, rows, slice, experts, ranks);
	else if ( rows > 1u )
		K3RunnerLatentRowsKernel<<<dim3((K3_ROUTED_EXPERT_HIDDEN + 255u) / 256u, rows), 256u, 0, stream>>>(
			gathered, b->latent_full_bf16, rows, slice, ranks);
}

static void K3RunnerLayerCollective(void *context, void *stream_void,
	uint32_t layer, uint32_t phase)
{
	const SparkK3RunnerLane *lane = (const SparkK3RunnerLane *)context;
	SparkK3ModelState *state = lane->state;
	K3LayerBuffers *b = lane->index != 0u ? state->dispatch.pair_buffers : state->dispatch.buffers;
	cudaStream_t stream = (cudaStream_t)stream_void;
	uint32_t rows = state->dispatch.pair_active != 0u ? state->dispatch.pair_rows[lane->index] : state->rows;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u;
	uint32_t elements = rows * K3_HIDDEN;
	const uint32_t stage = (phase & K3_COLLECTIVE_FINISH) != 0u ? K3_RUNNER_STAGE_FINISH :
		(phase & K3_COLLECTIVE_BEGIN) != 0u ? K3_RUNNER_STAGE_BEGIN : K3_RUNNER_STAGE_ALL;
	const uint32_t base = phase & ~(K3_COLLECTIVE_FINISH | K3_COLLECTIVE_BEGIN);
	uint32_t segments = base == 0u ? 1u : (layer == 0u ? 1u : 2u);
	uint16_t *phase0_source =
		(K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT)
			? b->hidden_bf16 : b->attention_out_bf16;
	if ( b->tp_sharded == 0u )
		return;
	if ( base == K3_COLLECTIVE_MLA_QUERY || base == K3_COLLECTIVE_MLA_PARTIALS || base == K3_COLLECTIVE_MLA_KEYS )
	{
		K3RunnerShardExchange(state, b, stream, rows, phase);
		return;
	}
	if ( base == K3_COLLECTIVE_MLA_DOWN )
	{
		const uint32_t degree = b->kv_shard.degree;
		K3RunnerRound(state, stream, rows, b->latent_bf16, state->fused_device,
			(K3_Q_LORA_RANK + K3_MLA_KV_A_DIM) / degree, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, K3_RUNNER_STAGE_ALL);
		K3RunnerMlaDownRowsKernel<<<dim3((K3_Q_LORA_RANK + K3_MLA_KV_A_DIM + 255u) / 256u, rows), 256u, 0, stream>>>(
			state->fused_device, b->latent_bf16, b->kv_slot_bf16, rows, K3_Q_LORA_RANK / degree, K3_MLA_KV_A_DIM / degree);
		if ( cudaPeekAtLastError() != cudaSuccess )
			K3ModelFail(state);
		return;
	}
	if ( K3_EXPERT_CELLS(b) && base == 2u )
	{
		K3RunnerGatherLatent(state, b, stream, rows, 1u,
			stage == K3_RUNNER_STAGE_FINISH ? K3_RUNNER_STAGE_FINISH : K3_RUNNER_STAGE_BEGIN);
		return;
	}
	if ( K3_EXPERT_CELLS(b) && base == 3u )
	{
		K3RunnerRound(state, stream, rows, b->latent_bf16, b->latent_full_bf16, K3_ROUTED_EXPERT_HIDDEN,
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
			stage == K3_RUNNER_STAGE_FINISH ? K3_RUNNER_STAGE_FINISH : K3_RUNNER_STAGE_BEGIN);
		return;
	}
	if ( phase == 3u )
	{
		K3RunnerGatherLatent(state, b, stream, rows, 0u, K3_RUNNER_STAGE_ALL);
		return;
	}
	if ( phase == 2u )
	{
		const uint32_t gate_up_elements = rows * K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u);
		uint16_t *reduce_values = b->gate_up_bf16;
		if ( state->has_collective != 0 )
		{
			K3RunnerFusedPackKernel<<<(gate_up_elements + 255u) / 256u,
				256u, 0, stream>>>(
				0, 0, reduce_values, reduce_values, state->fused_device,
				rows, phase, 1u, gate_up_elements);
			SparkK3RunnerTpContext *completion_context =
				K3RunnerTpContextAcquire(state);
			if ( completion_context == 0 )
				return;
			completion_context->owner = state;
			completion_context->fused = state->fused_device;
			completion_context->buffers = b;
			completion_context->stream = stream;
			completion_context->rows = rows;
			completion_context->boundary = 0u;
			completion_context->phase = phase;
			completion_context->gate_up_elements = gate_up_elements;
			(void)state->services->round(state->services->context, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
				K3_RUNNER_STAGE_ALL, SparkK3TpSequences(gate_up_elements), 0u, state->fused_device, state->fused_device);
			K3RunnerTpApply(completion_context);
			return;
		}
		K3ModelFail(state);
		return;
	}
	if ( state->has_collective == 0 )
	{
		K3ModelFail(state);
		return;
	}
	if ( stage == K3_RUNNER_STAGE_FINISH )
	{
		SparkK3RunnerTpContext *pending = state->pending_tp;
		state->pending_tp = 0;
		if ( pending == 0 )
			K3ModelFail(state);
		else
			(void)state->services->round(state->services->context, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
				K3_RUNNER_STAGE_FINISH, rows, 0u, state->fused_device, state->fused_device);
		if ( pending != 0 )
			K3RunnerTpApply(pending);
		return;
	}
	K3RunnerFusedPackKernel<<<(elements + 255u) / 256u, 256u, 0, stream>>>(
		phase0_source,b->hidden_bf16,b->shared_out_bf16,b->gate_up_bf16,
		state->fused_device,rows,base,segments,0u);
	{
		SparkK3RunnerTpContext *completion_context =
			K3RunnerTpContextAcquire(state);
		if ( completion_context == 0 )
			return;
		completion_context->owner = state;
		completion_context->fused = state->fused_device;
		completion_context->buffers = b;
		completion_context->stream = stream;
		completion_context->rows = rows;
		completion_context->boundary = boundary;
		completion_context->phase = base;
		completion_context->gate_up_elements = 0u;
		(void)state->services->round(state->services->context, (void *)stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16,
			stage, rows, 0u, state->fused_device, state->fused_device);
		if ( stage == K3_RUNNER_STAGE_BEGIN )
		{
			state->pending_tp = completion_context;
			return;
		}
		K3RunnerTpApply(completion_context);
	}
}


typedef struct SparkK3ManifestCheckContext
{
	SparkK3Pack *pack;
} SparkK3ManifestCheckContext;

static SparkStatus SparkK3ManifestCheck(const SparkWeightdManifest *manifest,
	void *context)
{
	SparkK3ManifestCheckContext *check = (SparkK3ManifestCheckContext *)context;
	SparkK3Pack *pack = check->pack;
	uint32_t group_index = 0u;
	uint32_t layer;
	if ( manifest->group_count == 0u ||
		manifest->range_count < manifest->group_count )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	for ( layer = pack->config.first_layer;
		layer < pack->config.first_layer + pack->config.layers;
		++layer )
	{
		char name[SPARK_K3_PACK_MAX_NAME_BYTES];
		SparkK3PackEntry w1;
		SparkK3PackEntry w2;
		int have_w1;
		int have_w2;
		uint32_t expert;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w1_weight",
			layer);
		have_w1 = SparkK3PackLoadEntry(pack, name, &w1) == SPARK_STATUS_OK;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w2_weight",
			layer);
		have_w2 = SparkK3PackLoadEntry(pack, name, &w2) == SPARK_STATUS_OK;
		if ( have_w1 != have_w2 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( !have_w1 )
			continue;
		if ( w1.bytes % pack->config.experts != 0u ||
			w2.bytes % pack->config.experts != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		for ( expert = 0u; expert < pack->config.experts; ++expert )
		{
			const SparkWeightdRangeGroup *group =
				&manifest->groups[group_index + expert];
			const SparkWeightdRange *range;
			uint32_t r;
			if ( group->layer != layer || group->expert != expert ||
				group->range_count != 2u )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			for ( r = 0u; r < 2u; ++r )
			{
				const SparkK3PackEntry *tensor =
					r == 0u ? &w1 : &w2;
				uint64_t expert_bytes =
					tensor->bytes / pack->config.experts;
				uint64_t expected = pack->payload_base +
					tensor->payload_offset +
					(uint64_t)expert * expert_bytes;
				range = &manifest->ranges[group->first_range + r];
				if ( range->offset != expected ||
					range->bytes != expert_bytes ||
					range->kind != r ||
					range->layer != layer ||
					range->expert != expert )
					SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			}
		}
		group_index += pack->config.experts;
	}
	if ( group_index != manifest->group_count )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}


static SparkStatus K3RunnerCreateDispatch(SparkK3ModelState *state,
	const SparkStageRunnerConfiguration *configuration)
{
	uint64_t budget = configuration->state_budget_bytes, planned;
	SparkK3RankStateBytes state_plan;
	SparkK3ScratchWidths widths;
	memset(&state_plan, 0, sizeof(state_plan));
	if ( budget == 0u )
	{
		fprintf(stderr, "sparkpipe_k3: the rank state budget is required"
			" (per-rank KDA state + windows + MLA KV + dispatch scratch)\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( SparkK3RankStateBytesFor(state->module.sizing.kda_layer_count,
			state->module.sizing.mla_layer_count,
			configuration->max_active_sequence_count, configuration->tp_degree,
			configuration->kv_pages_per_sequence, state->kv_page_bytes,
			&state_plan) == 0u || state_plan.total > budget )
	{
		fprintf(stderr, "sparkpipe_k3: rank state %llu exceeds"
			" SPARK_K3_STATE_BUDGET_BYTES %llu or tp_degree %u is invalid"
			" (refused before allocation)\n",
			(unsigned long long)state_plan.total, (unsigned long long)budget,
			configuration->tp_degree);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( SparkK3DispatchScratchWidths(&state->module.pack, state->module.sizing.first_layer,
		state->module.sizing.layer_count, configuration->tp_degree, &widths) != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: the pack does not give the per-rank scratch widths\n");
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( SparkK3DispatchCreate(&state->dispatch, &state->module.sizing,
		configuration->max_active_sequence_count,
		configuration->max_input_row_count,
		configuration->kv_pages_per_sequence,
		state->kv_page_bytes, configuration->tp_degree, configuration->tp_rank, &widths, 0) != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: dispatch create failed tp_degree=%u\n",
			configuration->tp_degree);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	planned = state->dispatch.state_bytes.total + state->dispatch.scratch_bytes;
	fprintf(stderr, "sparkpipe_k3: rank state planned=%llu budget=%llu"
		" kda_state=%llu kda_windows=%llu mla_kv=%llu scratch=%llu"
		" sequences=%u kda_heads_per_rank=%u\n",
		(unsigned long long)planned, (unsigned long long)budget,
		(unsigned long long)state->dispatch.state_bytes.kda_state,
		(unsigned long long)state->dispatch.state_bytes.kda_windows,
		(unsigned long long)state->dispatch.state_bytes.mla_kv,
		(unsigned long long)state->dispatch.scratch_bytes,
		configuration->max_active_sequence_count,
		state->dispatch.kda_rank_heads);
	if ( planned > budget )
	{
		fprintf(stderr, "sparkpipe_k3: rank state %llu exceeds"
			" SPARK_K3_STATE_BUDGET_BYTES %llu (fail-closed)\n",
			(unsigned long long)planned, (unsigned long long)budget);
		SparkK3DispatchDestroy(&state->dispatch);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return SPARK_STATUS_OK;
}

static uint32_t K3ModelPublished(void *context)
{
	const SparkK3ModelState *state = ((const SparkK3RunnerLane *)context)->state;
	return state->services->published(state->services->context);
}

static int32_t K3ModelLazyAcquire(void *context, uint32_t layer, void *buffers_void)
{
	SparkK3ModelState *state = (SparkK3ModelState *)context;
	K3LayerBuffers *buffers = (K3LayerBuffers *)buffers_void;
	const void *w1 = 0, *w2 = 0;
	SparkStatus status;
	if ( state == 0 || buffers == 0 || layer >= K3_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = state->services->expert_weights(state->services->context, layer, buffers->group_row_offset,
		state->rows * K3_TOP_K, &w1, &w2);
	if ( status != SPARK_STATUS_OK )
		return status;
	buffers->expert_w1_weight = w1;
	buffers->expert_w2_weight = w2;
	return LM_LAUNCH_OK;
}

static void K3ModelLazyRelease(void *context, uint32_t layer)
{
	SparkK3ModelState *state = (SparkK3ModelState *)context;
	state->services->layer_done(state->services->context, layer);
}

static SparkStatus K3ModelEntry(SparkK3ModelState *state, const char *name, uint64_t *offset, uint64_t *bytes, uint32_t *rows)
{
	SparkK3PackEntry entry;
	if ( SparkK3PackLoadEntry(&state->module.pack, name, &entry) != SPARK_STATUS_OK || entry.shape_count < 1u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	*offset = state->module.pack.payload_base + entry.payload_offset;
	*bytes = entry.bytes;
	if ( rows != 0 )
		*rows = (uint32_t)entry.shape[0];
	return SPARK_STATUS_OK;
}

static SparkStatus K3ModelExpertOffsets(SparkK3ModelState *state)
{
	uint32_t routed, routed_layers = 0u;
	for ( routed = state->module.pack.config.first_layer;
		routed < state->module.pack.config.first_layer + state->module.pack.config.layers; ++routed )
	{
		SparkK3PackEntry w1, w2;
		char name[SPARK_K3_PACK_MAX_NAME_BYTES];
		int have_w1, have_w2;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w1_weight", routed);
		have_w1 = SparkK3PackLoadEntry(&state->module.pack, name, &w1) == SPARK_STATUS_OK;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w2_weight", routed);
		have_w2 = SparkK3PackLoadEntry(&state->module.pack, name, &w2) == SPARK_STATUS_OK;
		if ( have_w1 != have_w2 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( !have_w1 )
			continue;
		state->layer_w1_offset[routed] = state->module.pack.payload_base + w1.payload_offset;
		state->layer_w2_offset[routed] = state->module.pack.payload_base + w2.payload_offset;
		routed_layers++;
	}
	return routed_layers != 0u ? SPARK_STATUS_OK : SPARK_STATUS_PARSE_ERROR;
}

static void K3ModelClose(void *model)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	if ( state == 0 )
		return;
	cudaFree(state->fused_device);
	if ( state->load_join != 0 )
		(void)cudaEventDestroy(state->load_join);
	if ( state->load_fork != 0 )
		(void)cudaEventDestroy(state->load_fork);
	if ( state->load_stream != 0 )
		(void)cudaStreamDestroy(state->load_stream);
	if ( state->pair_fence != 0 )
		(void)cudaEventDestroy(state->pair_fence);
	for ( uint32_t lane = 0u; lane < 2u; ++lane )
		if ( state->pair_stream[lane] != 0 )
			(void)cudaStreamDestroy(state->pair_stream[lane]);
	cudaFree(state->route_expert);
	cudaFree(state->route_packed_row);
	cudaFree(state->route_source_token);
	cudaFree(state->route_weight);
	cudaFree(state->group_row_offset);
	cudaFree(state->group_tile_prefix_w1);
	cudaFree(state->group_tile_prefix_w2);
	cudaFree(state->dense_row_offset);
	cudaFree(state->dense_tile_prefix);
	if ( state->dispatch.buffers != 0 )
		SparkK3DispatchDestroy(&state->dispatch);
	SparkK3ModuleDestroy(&state->module);
	delete state;
}

static SparkStatus K3ModelOpen(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
{
	const SparkStageRunnerConfiguration *configuration = request->configuration;
	SparkK3ModelState *state;
	SparkStatus status;
	uint32_t first_layer, layer_count, pool_index;
	if ( configuration->stage_count != 1u && configuration->stage_count != 4u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = new SparkK3ModelState;
	memset(state, 0, sizeof(*state));
	*model = state;
	state->services = request->services;
	state->tp_degree = configuration->tp_degree;
	state->tp_rank = configuration->tp_rank;
	state->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = configuration->stage_index + 1u == configuration->stage_count ? 1u : 0u;
	state->has_collective = configuration->device_collective != 0 ? 1u : 0u;
	state->deferred = configuration->device_collective != 0 &&
		configuration->device_collective->wait_mode == SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE ? 1u : 0u;
	state->layer_collective_override = configuration->layer_collective_override;
	state->layer_collective_context = configuration->layer_collective_context;
	state->kv_page_bytes = configuration->kv_page_bytes != 0u ? configuration->kv_page_bytes :
		K3GlobalKv::kPageBytes / (configuration->tp_degree > 1u ? configuration->tp_degree : 1u);
	for ( pool_index = 0u; pool_index < K3_RUNNER_TP_CONTEXT_POOL_DEPTH; ++pool_index )
	{
		state->tp_context_pool[pool_index].owner = state;
		state->tp_context_pool[pool_index].pool_next = state->tp_context_free_head;
		state->tp_context_free_head = &state->tp_context_pool[pool_index];
	}
	first_layer = configuration->stage_count == 4u ? K3RunnerFirstLayer(configuration->stage_index) : SPARK_K3_MODULE_DERIVE_SLICE;
	layer_count = configuration->stage_count == 4u ? K3RunnerLayerCount(configuration->stage_index) : SPARK_K3_MODULE_DERIVE_SLICE;
	status = SparkK3ModuleInitialize(&state->module, configuration->rank_pack_path, first_layer, layer_count);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = K3RunnerCreateDispatch(state, configuration);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = K3ModelExpertOffsets(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	memset(geometry, 0, sizeof(*geometry));
	geometry->hidden = K3_HIDDEN;
	geometry->vocab = state->module.pack.config.vocab;
	geometry->total_layers = K3_LAYERS;
	geometry->first_layer = state->module.sizing.first_layer;
	geometry->layer_count = state->module.sizing.layer_count;
	geometry->rms_epsilon = K3_RMS_EPSILON;
	geometry->sideband_bytes_per_row = SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW;
	geometry->pack_bytes = state->module.pack.file_bytes;
	if ( state->owns_embedding != 0u &&
		K3ModelEntry(state, "model.embed_tokens.weight", &geometry->embed_offset, &geometry->embed_bytes, &geometry->embed_rows) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( state->owns_final_head != 0u &&
		(K3ModelEntry(state, "model.norm.weight", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
		K3ModelEntry(state, "lm_head.weight", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK) )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	geometry->kv_layer_count = state->dispatch.mla_count;
	geometry->kv_layer_page_bytes = K3GlobalKv::kPageBytes;
	geometry->kv_shard = state->dispatch.buffers->kv_shard;
	geometry->recurrent_bytes = K3ModelRecurrentBytes(state);
	geometry->experts = K3_EXPERTS;
	geometry->top_k = K3_TOP_K;
	geometry->expert_w1_offset = state->layer_w1_offset;
	geometry->expert_w2_offset = state->layer_w2_offset;
	return SPARK_STATUS_OK;
}

static SparkStatus K3ModelManifestCheck(void *model, const SparkWeightdManifest *manifest)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	SparkK3ManifestCheckContext check;
	check.pack = &state->module.pack;
	return SparkK3ManifestCheck(manifest, &check);
}

static SparkStatus K3ModelBind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	const uint32_t max_rows = state->dispatch.max_rows;
	const uint64_t routes = (uint64_t)max_rows * K3_TOP_K;
	if ( SparkK3DispatchBindWeights(&state->dispatch, &state->module.pack, state->module.bound, state->module.bound_count,
		lazy_pack) != SPARK_K3_DISPATCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->dispatch.buffers->tp_sharded = state->tp_degree > 1u ? 1u : 0u;
	state->dispatch.buffers->tp_rank = state->tp_rank;
	geometry->lease_tensor_base = (uint32_t)(state->dispatch.buffers->routed_down_rows % K3_LAYER_TILE_N != 0u);
	state->dispatch.slice_state->layer_collective = K3RunnerLayerCollective;
	state->dispatch.slice_state->collective_published = K3ModelPublished;
	state->lanes[0].state = state;
	state->lanes[0].index = 0u;
	state->lanes[1].state = state;
	state->lanes[1].index = 1u;
	state->dispatch.slice_state->collective_context = &state->lanes[0];
	state->dispatch.slice_state->collective_pair_context = &state->lanes[1];
	if ( state->tp_degree > 1u &&
		(cudaStreamCreateWithFlags(&state->load_stream, cudaStreamNonBlocking) != cudaSuccess ||
		 cudaEventCreateWithFlags(&state->load_fork, cudaEventDisableTiming) != cudaSuccess ||
		 cudaEventCreateWithFlags(&state->load_join, cudaEventDisableTiming) != cudaSuccess ||
		 cudaStreamCreateWithFlags(&state->pair_stream[0], cudaStreamNonBlocking) != cudaSuccess ||
		 cudaStreamCreateWithFlags(&state->pair_stream[1], cudaStreamNonBlocking) != cudaSuccess ||
		 cudaEventCreateWithFlags(&state->pair_fence, cudaEventDisableTiming) != cudaSuccess) )
	{
		fprintf(stderr, "sparkpipe_k3: the weight load stream could not be created\n");
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	state->dispatch.slice_state->load_stream = state->load_stream;
	state->dispatch.slice_state->pair_stream_a = state->pair_stream[0];
	state->dispatch.slice_state->pair_stream_b = state->pair_stream[1];
	state->dispatch.slice_state->pair_fence = state->pair_fence;
	state->dispatch.slice_state->load_fork = state->load_fork;
	state->dispatch.slice_state->load_join = state->load_join;
	state->dispatch.slice_state->lazy_context = state;
	state->dispatch.slice_state->lazy_acquire = K3ModelLazyAcquire;
	state->dispatch.slice_state->lazy_release = K3ModelLazyRelease;
	if ( state->layer_collective_override != 0 )
	{
		state->dispatch.slice_state->layer_collective = state->layer_collective_override;
		state->dispatch.slice_state->collective_context = state->layer_collective_context;
		state->dispatch.slice_state->collective_pair_context = 0;
		state->dispatch.slice_state->collective_published = 0;
	}
	if ( state->has_collective != 0u )
	{
		const uint64_t fused_bytes = (uint64_t)SparkK3TpSequences((uint64_t)max_rows * state->dispatch.widths.fused) *
			K3_HIDDEN * sizeof(uint16_t);
		state->fused_rows = max_rows;
		if ( cudaMalloc(&state->fused_device, fused_bytes) != cudaSuccess ||
			cudaMemset(state->fused_device, 0, fused_bytes) != cudaSuccess )
			{ state->fused_device = 0; SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); }
	}
	if ( cudaMalloc(&state->route_expert, routes * 4u) != cudaSuccess ||
		cudaMalloc(&state->route_packed_row, routes * 4u) != cudaSuccess ||
		cudaMalloc(&state->route_source_token, routes * 4u) != cudaSuccess ||
		cudaMalloc(&state->route_weight, routes * 4u) != cudaSuccess ||
		cudaMalloc(&state->group_row_offset, (uint64_t)(K3_EXPERTS + 1u) * 4u) != cudaSuccess ||
		cudaMalloc(&state->group_tile_prefix_w1, (uint64_t)(K3_EXPERTS + 1u) * 4u) != cudaSuccess ||
		cudaMalloc(&state->group_tile_prefix_w2, (uint64_t)(K3_EXPERTS + 1u) * 4u) != cudaSuccess ||
		cudaMalloc(&state->dense_row_offset, 8u) != cudaSuccess ||
		cudaMalloc(&state->dense_tile_prefix, 8u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return SPARK_STATUS_OK;
}

static SparkStatus K3ModelStep(void *model, const SparkStageRunnerStep *step, void *stream_void)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	cudaStream_t stream = (cudaStream_t)stream_void;
	SparkK3StepInput in;
	int32_t status;
	memset(&in, 0, sizeof(in));
	in.hidden_in = step->hidden_bf16;
	in.positions = step->positions;
	in.context_length = step->context_length;
	in.sequence_of_row = step->sequence_of_row;
	in.sequence_row_begin = step->sequence_row_begin;
	in.sequence_row_indices = step->sequence_row_indices;
	in.kda_state_index = step->recurrent_index;
	in.gather_sequence = step->gather_sequence;
	in.gather_context = step->gather_context;
	in.route_expert = state->route_expert;
	in.route_packed_row = state->route_packed_row;
	in.route_source_token = state->route_source_token;
	in.route_weight = state->route_weight;
	in.group_row_offset = state->group_row_offset;
	in.group_tile_prefix_w1 = state->group_tile_prefix_w1;
	in.group_tile_prefix_w2 = state->group_tile_prefix_w2;
	in.dense_row_offset = state->dense_row_offset;
	in.dense_tile_prefix = state->dense_tile_prefix;
	state->rows = step->rows;
	state->head_last_rows = step->last_rows_only;
	K3RunnerDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, step->rows);
	in.pair_rows = state->head_last_rows != 0u && step->sequences == 1u && step->rows >= 2u * K3_RUNNER_PAIR_MIN_ROWS &&
		state->has_collective != 0u && state->deferred != 0u ? step->rows / 2u : 0u;
	status = SparkK3DispatchStep(&state->dispatch, &in, step->rows, step->sequences, step->commit,
		step->rows * K3_TOP_K, step->context, step->multiprocessors, stream);
	if ( status != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: slice dispatch failed %d\n", status);
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	return state->tp_context_overflow != 0u ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_OK;
}

static SparkStatus K3ModelSideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream_void)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	K3LayerBuffers *b = state->dispatch.buffers;
	cudaStream_t stream = (cudaStream_t)stream_void;
	if ( to_buffer != 0u )
		return buffer == 0 || K3RunnerCopy(buffer, b->attnres_bank_bf16, bytes, stream) != cudaSuccess ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
	if ( buffer != 0 && K3RunnerCopy(b->attnres_bank_bf16, buffer, bytes, stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	return K3RunnerCopy(b->attnres_partial_bf16, hidden, (uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess ?
		SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
}

static SparkStatus K3ModelAttachKv(void *model, const SparkStageRunnerKv *kv)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	if ( SparkK3DispatchAttachKv(&state->dispatch, kv->pool, kv->layer_stride_bytes, kv->page_table, kv->page_table_stride,
		kv->pool_page_count, kv->sequence_count) != SPARK_K3_DISPATCH_OK )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return SPARK_STATUS_OK;
}

static SparkStatus K3ModelRecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	SparkK3KdaRankLayout layout;
	uint8_t *pools[SPARK_K3_SLOT_POOLS], *packed = (uint8_t *)buffer;
	uint64_t widths[SPARK_K3_SLOT_POOLS];
	cudaError_t error = cudaSuccess;
	uint32_t part;
	if ( slot >= state->dispatch.sequences || bytes != K3ModelRecurrentBytes(state) ||
		SparkK3KdaRankLayoutFor(state->dispatch.tp_degree, &layout) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	pools[SPARK_K3_SLOT_POOL_STATE] = state->dispatch.kda_state_pool;
	pools[SPARK_K3_SLOT_POOL_Q_WINDOW] = (uint8_t *)state->dispatch.kda_q_window_pool;
	pools[SPARK_K3_SLOT_POOL_K_WINDOW] = (uint8_t *)state->dispatch.kda_k_window_pool;
	pools[SPARK_K3_SLOT_POOL_V_WINDOW] = (uint8_t *)state->dispatch.kda_v_window_pool;
	widths[SPARK_K3_SLOT_POOL_STATE] = layout.state_slot_bytes;
	widths[SPARK_K3_SLOT_POOL_Q_WINDOW] = layout.qk_window_slot_bytes;
	widths[SPARK_K3_SLOT_POOL_K_WINDOW] = layout.qk_window_slot_bytes;
	widths[SPARK_K3_SLOT_POOL_V_WINDOW] = layout.v_window_slot_bytes;
	for ( part = 0u; part < SPARK_K3_SLOT_POOLS && error == cudaSuccess; part++ )
	{
		uint8_t *pool = pools[part] + (uint64_t)slot * widths[part];
		size_t pitch = (size_t)(widths[part] * state->dispatch.sequences);
		error = to_buffer != 0u ?
			cudaMemcpy2DAsync(packed, (size_t)widths[part], pool, pitch, (size_t)widths[part], state->dispatch.kda_count, cudaMemcpyDefault, (cudaStream_t)stream) :
			cudaMemcpy2DAsync(pool, pitch, packed, (size_t)widths[part], (size_t)widths[part], state->dispatch.kda_count, cudaMemcpyDefault, (cudaStream_t)stream);
		packed += widths[part] * state->dispatch.kda_count;
	}
	if ( error == cudaSuccess && stream == 0 )
		error = cudaStreamSynchronize(0);
	if ( error != cudaSuccess )
	{
		fprintf(stderr, "sparkpipe_k3: KDA record copy failed slot=%u cuda=%s\n", slot, cudaGetErrorString(error));
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K3ModelResetSlot(void *model, uint32_t slot, void *stream)
{
	SparkK3ModelState *state = (SparkK3ModelState *)model;
	const int32_t status = SparkK3DispatchResetSlot(&state->dispatch, slot, state->tp_degree, (cudaStream_t)stream);
	if ( status == SPARK_K3_DISPATCH_ERR_ARGUMENT )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return status == SPARK_K3_DISPATCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static const void *K3ModelProbeBuffers(void *model)
{
	return ((SparkK3ModelState *)model)->dispatch.buffers;
}

static void K3ModelReport(void *model, uint32_t rank)
{
	const SparkK3ModelState *state = (const SparkK3ModelState *)model;
	fprintf(stderr, "K3-STEP-TIMING rank=%u mla_gather_waves=%llu mla_scatter_waves=%llu pair_waves=%llu\n", rank,
		(unsigned long long)state->dispatch.mla_gather_waves, (unsigned long long)state->dispatch.mla_scatter_waves,
		(unsigned long long)state->dispatch.pair_waves);
}

static const SparkStageRunnerModelInterface spark_k3_stage_model =
{
	SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
	"sparkpipe_k3",
	"kimi-k3",
	"mxfp4",
	K3ModelOpen,
	K3ModelManifestCheck,
	K3ModelBind,
	K3ModelClose,
	K3ModelStep,
	K3ModelSideband,
	K3ModelAttachKv,
	K3ModelRecurrentCopy,
	K3ModelResetSlot,
	K3ModelProbeBuffers,
	K3ModelReport
};

const SparkStageRunnerModelInterface *SparkK3StageModel(void)
{
	return &spark_k3_stage_model;
}
