#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_named_pack_bind.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "inference/kernels/kv_shard.cuh"

#define CUDA_STAGE_SLICE_ALIGNMENT 256u
#define CUDA_STAGE_MAX_TENSORS 48u
#define CUDA_STAGE_MAX_RECURRENT 8u
#define CUDA_STAGE_MAX_REPORT 12u
#define CUDA_STAGE_MAX_COUNTERS 4u
#define CUDA_STAGE_REPORT_BYTES 512u

struct CudaStageState
{
	const SparkStageRunnerServices *services;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	void *arena;
	uint64_t arena_bytes;
	void *lane_arena;
	uint64_t lane_bytes;
	void *lane_states[SPARK_STAGE_RUNNER_LANES];
	uint32_t lane_bound[SPARK_STAGE_RUNNER_LANES];
	uint64_t lane_base[SPARK_STAGE_RUNNER_LANES][CUDA_STAGE_MAX_COUNTERS];
	LmKvAccessError *kv_error;
	uint32_t kv_attached;
};

struct CudaStagePackState : CudaStageState
{
	SparkNamedPack pack;
	uint32_t pack_open;
};

typedef struct CudaStageSlice
{
	void **target;
	uint64_t bytes;
} CudaStageSlice;

typedef struct CudaStageRecurrent
{
	uint8_t *pool;
	uint64_t width;
	uint32_t layers;
} CudaStageRecurrent;

typedef struct CudaStageReportField
{
	const char *name;
	uint64_t value;
} CudaStageReportField;

typedef struct CudaStageTensor
{
	char name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
	const char *kind;
	uint64_t bytes;
	const void **field;
	uint32_t rows;
} CudaStageTensor;

struct CudaStageTensors
{
	CudaStageTensor entries[CUDA_STAGE_MAX_TENSORS];
	uint32_t count;
	uint32_t overflow;

	template<class T>
	void Add(const char *name, const char *kind, uint64_t bytes, T **field, uint32_t rows = 0u)
	{
		if ( count == CUDA_STAGE_MAX_TENSORS )
		{
			overflow = 1u;
			return;
		}
		snprintf(entries[count].name, sizeof(entries[count].name), "%s", name);
		entries[count].kind = kind;
		entries[count].bytes = bytes;
		entries[count].field = (const void **)field;
		entries[count].rows = rows;
		count++;
	}

	template<class T>
	void AddScaled(const char *name, const char *suffix, const char *kind, uint64_t bytes, T **field, uint32_t rows = 0u)
	{
		char scaled[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(scaled, sizeof(scaled), "%s%s", name, suffix);
		Add(scaled, kind, bytes, field, rows);
	}
};

typedef struct CudaStagePackValue
{
	const char *key;
	uint32_t *value;
} CudaStagePackValue;

template<size_t N>
static inline SparkStatus CudaStagePackValues(const SparkNamedPack *pack, const CudaStagePackValue (&values)[N])
{
	for ( size_t index = 0u; index < N; ++index )
		if ( SparkNamedPackConfigU32(pack, values[index].key, values[index].value) != SPARK_STATUS_OK )
			return SPARK_STATUS_PARSE_ERROR;
	return SPARK_STATUS_OK;
}

template<size_t N>
static inline uint32_t CudaStageReportOut(CudaStageReportField *out, const CudaStageReportField (&fields)[N])
{
	for ( size_t index = 0u; index < N; ++index )
		out[index] = fields[index];
	return (uint32_t)N;
}

template<size_t N>
static inline uint64_t CudaStagePlace(const CudaStageSlice (&slices)[N], uint8_t *base)
{
	uint64_t cursor = 0u;
	for ( size_t index = 0u; index < N; ++index )
	{
		cursor = (cursor + CUDA_STAGE_SLICE_ALIGNMENT - 1u) & ~(uint64_t)(CUDA_STAGE_SLICE_ALIGNMENT - 1u);
		if ( base != 0 )
			*slices[index].target = slices[index].bytes != 0u ? base + cursor : 0;
		cursor += slices[index].bytes;
	}
	return cursor;
}

static inline SparkStatus CudaStageRound(const CudaStageState *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, row_elements, local, full);
}

static inline SparkStatus CudaStageLaunched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static inline void CudaStageRowState(const CudaStageState *state, uint32_t layer, uint32_t phase)
{
	if ( state->services->row_state != 0 )
		state->services->row_state(state->services->context, layer, phase);
}

static inline void CudaStageLayerDone(const CudaStageState *state, uint32_t layer)
{
	if ( state->services->layer_done != 0 )
		state->services->layer_done(state->services->context, layer);
}

static inline uint64_t CudaStageRecurrentBytes(const CudaStageRecurrent *parts, uint32_t count)
{
	uint64_t bytes = 0u;
	for ( uint32_t part = 0u; part < count; ++part )
		bytes += parts[part].width * parts[part].layers;
	return bytes;
}

static inline SparkStatus CudaStageRecurrentCopy(const char *tag, const CudaStageRecurrent *parts, uint32_t count, uint32_t slots, uint32_t to_buffer,
	uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	uint8_t *packed = (uint8_t *)buffer;
	cudaError_t error = cudaSuccess;
	if ( count == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( slot >= slots || buffer == 0 || bytes != CudaStageRecurrentBytes(parts, count) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t part = 0u; part < count && error == cudaSuccess; ++part )
	{
		uint8_t *pool = parts[part].pool + (uint64_t)slot * parts[part].width;
		const size_t width = (size_t)parts[part].width, pitch = (size_t)(parts[part].width * slots);
		error = to_buffer != 0u ?
			cudaMemcpy2DAsync(packed, width, pool, pitch, width, parts[part].layers, cudaMemcpyDefault, (cudaStream_t)stream) :
			cudaMemcpy2DAsync(pool, pitch, packed, width, width, parts[part].layers, cudaMemcpyDefault, (cudaStream_t)stream);
		packed += parts[part].width * parts[part].layers;
	}
	if ( error == cudaSuccess && stream == 0 )
		error = cudaStreamSynchronize(0);
	if ( error != cudaSuccess )
	{
		fprintf(stderr, "%s: recurrent copy failed slot=%u cuda=%s\n", tag, slot, cudaGetErrorString(error));
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

static inline SparkStatus CudaStageRecurrentReset(const CudaStageRecurrent *parts, uint32_t count, uint32_t slots, uint32_t slot, void *stream)
{
	cudaError_t error = cudaSuccess;
	if ( count == 0u )
		return SPARK_STATUS_OK;
	if ( slot >= slots )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t part = 0u; part < count && error == cudaSuccess; ++part )
		error = cudaMemset2DAsync(parts[part].pool + (uint64_t)slot * parts[part].width, (size_t)(parts[part].width * slots), 0,
			(size_t)parts[part].width, parts[part].layers, (cudaStream_t)stream);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static inline void CudaStageReport(const char *tag, uint32_t rank, const CudaStageReportField *fields, uint32_t count)
{
	char line[CUDA_STAGE_REPORT_BYTES];
	int used = snprintf(line, sizeof(line), "%s rank=%u", tag, rank);
	for ( uint32_t index = 0u; index < count && used > 0 && (size_t)used < sizeof(line); ++index )
		used += snprintf(line + used, sizeof(line) - (size_t)used, " %s=%llu", fields[index].name, (unsigned long long)fields[index].value);
	fprintf(stderr, "%s\n", line);
}

template<auto... M>
struct CudaStageFields
{
	static constexpr uint32_t kCount = (uint32_t)sizeof...(M);

	template<class S>
	static void Copy(S *to, const S *from)
	{
		(void)to;
		(void)from;
		((to->*M = from->*M), ...);
	}

	template<class S>
	static void Mark(const S *from, uint64_t *base)
	{
		static_assert(sizeof...(M) <= CUDA_STAGE_MAX_COUNTERS, "a lane folds at most four counters");
		uint32_t index = 0u;
		(void)from;
		(void)base;
		(void)index;
		((base[index++] = from->*M), ...);
	}

	template<class S>
	static void Fold(S *to, const S *from, uint64_t *base)
	{
		static_assert(sizeof...(M) <= CUDA_STAGE_MAX_COUNTERS, "a lane folds at most four counters");
		uint32_t index = 0u;
		(void)to;
		(void)from;
		(void)base;
		(void)index;
		((to->*M += from->*M - base[index], base[index] = from->*M, ++index), ...);
	}
};

struct CudaStageContextShard
{
	template<class S>
	static SparkStatus Of(const S *state, const SparkStageRunnerKv *kv, SparkKvShard *shard)
	{
		(void)state;
		*shard = kv->context_shard;
		return SPARK_STATUS_OK;
	}
};

template<class Geometry, auto Views, uint32_t POOL, uint32_t LAYERS, uint32_t FIRST = 0u, uint64_t OFFSET = 0u, class Shard = CudaStageContextShard>
struct CudaStageKvViews
{
	static_assert(POOL < 2u, "a stage binds the first or the second KV pool");
	static constexpr uint32_t kPool = POOL;
	static constexpr uint32_t kLayers = LAYERS;
	static constexpr uint64_t kPageBytes = Geometry::kPageBytes;

	template<class S>
	static SparkStatus Attach(S *state, const SparkStageRunnerKv *kv)
	{
		uint8_t *pool = POOL == 0u ? kv->pool : kv->second_pool;
		const uint64_t stride = POOL == 0u ? kv->layer_stride_bytes : kv->second_layer_stride_bytes;
		SparkKvShard shard;
		if ( Shard::Of(state, kv, &shard) != SPARK_STATUS_OK )
			return SPARK_STATUS_INVALID_ARGUMENT;
		for ( uint32_t layer = 0u; layer < LAYERS; ++layer )
			if ( LmKvShardViewInitialize<Geometry>(&(state->*Views)[FIRST + layer], pool + (uint64_t)layer * stride + OFFSET, kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, shard) != 0 )
				return SPARK_STATUS_INVALID_ARGUMENT;
		return SPARK_STATUS_OK;
	}
};

template<class... V>
struct CudaStageKv
{
	static constexpr bool Uses(uint32_t pool)
	{
		return ((V::kPool == pool) || ...);
	}

	static constexpr uint32_t Layers(uint32_t pool)
	{
		uint32_t layers = 0u;
		((layers = V::kPool == pool && V::kLayers > layers ? V::kLayers : layers), ...);
		return layers;
	}

	static constexpr uint64_t PageBytes(uint32_t pool)
	{
		uint64_t bytes = 0u;
		((bytes = V::kPool == pool && bytes == 0u ? V::kPageBytes : bytes), ...);
		return bytes;
	}

	static constexpr bool Consistent(void)
	{
		return (((V::kPool != 0u || V::kPageBytes == PageBytes(0u)) && (V::kPool != 1u || V::kPageBytes == PageBytes(1u))) && ...);
	}

	static_assert(sizeof...(V) == 0u || (Uses(0u) && Consistent()), "a stage's KV views fill the first pool and agree on each pool's page");

	static void Geometry(SparkStageRunnerModelGeometry *geometry)
	{
		geometry->kv_layer_count = Layers(0u);
		geometry->kv_layer_page_bytes = PageBytes(0u);
		geometry->kv_second_layer_count = Layers(1u);
		geometry->kv_second_layer_page_bytes = PageBytes(1u);
	}

	template<class S>
	static SparkStatus Attach(const char *tag, S *state, const SparkStageRunnerKv *kv)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( kv == 0 || kv->layer_count != Layers(0u) || kv->layer_page_bytes != PageBytes(0u) || kv->sequence_count > state->slots ||
			(Uses(1u) && (kv->second_pool == 0 || kv->second_layer_count != Layers(1u) || kv->second_layer_page_bytes != PageBytes(1u))) )
		{
			if ( kv != 0 )
				fprintf(stderr, "%s: KV attach refused: %u+%u layers of %llu+%llu-byte pages for %u sequences; the stage holds %u+%u of %llu+%llu for %u\n",
					tag, kv->layer_count, kv->second_layer_count, (unsigned long long)kv->layer_page_bytes,
					(unsigned long long)kv->second_layer_page_bytes, kv->sequence_count, Layers(0u), Layers(1u), (unsigned long long)PageBytes(0u),
					(unsigned long long)PageBytes(1u), state->slots);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		((status = status == SPARK_STATUS_OK ? V::Attach(state, kv) : status), ...);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}
};

template<class F>
struct CudaStage
{
	static constexpr bool kLanes = false;
	static constexpr bool kKeepRows = false;
	static constexpr bool kTapPlan = false;
	static constexpr bool kGraphKey = false;
	static constexpr bool kCheck = false;
	static constexpr bool kWaves = false;
	static constexpr uint32_t kChunkRows = 0u;
	static constexpr SparkStatus kLaneRefusal = SPARK_STATUS_UNSUPPORTED;
	using Kv = CudaStageKv<>;
	using LaneShared = CudaStageFields<>;
	using LaneCounters = CudaStageFields<>;

	static const char *WeightdRevision(void)
	{
		return "official";
	}

	template<class S>
	static const auto &LayerOf(S *state, uint32_t index)
	{
		return state->layers[index];
	}

	template<class S, class Frame>
	static SparkStatus Begin(S *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		frame->step = step;
		frame->stream = (cudaStream_t)stream;
		return SPARK_STATUS_OK;
	}

	template<class S, class Frame>
	static SparkStatus Check(S *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return CudaStageLaunched();
	}

	template<class S, class Frame>
	static SparkStatus LayerDone(S *state, const Frame &frame, uint32_t index)
	{
		(void)frame;
		CudaStageLayerDone(state, index);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration)
	{
		(void)configuration;
		return 1u;
	}

	template<class S>
	static SparkStatus Prepare(S *state, const SparkStageRunnerConfiguration *configuration)
	{
		(void)state;
		(void)configuration;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus Shape(S *state)
	{
		(void)state;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus Size(S *state)
	{
		const SparkStatus status = F::Shape(state);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->arena_bytes = F::Plan(state, 0, 0u);
		state->lane_bytes = F::kLanes ? F::Plan(state, 0, 1u) : 0u;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus GlobalTensors(S *state, CudaStageTensors *tensors)
	{
		(void)state;
		(void)tensors;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus LayerBound(S *state, uint32_t index)
	{
		(void)state;
		(void)index;
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindRefusal(SparkStatus status)
	{
		(void)status;
		return SPARK_STATUS_PARSE_ERROR;
	}

	template<class S>
	static SparkStatus Arrange(S *state)
	{
		(void)state;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus Upload(S *state)
	{
		(void)state;
		return SPARK_STATUS_OK;
	}

	template<class S>
	static void Release(S *state)
	{
		(void)state;
	}

	template<class S>
	static uint32_t RecurrentParts(const S *state, CudaStageRecurrent *parts)
	{
		(void)state;
		(void)parts;
		return 0u;
	}

	template<class S>
	static uint64_t RecurrentBytes(const S *state)
	{
		CudaStageRecurrent parts[CUDA_STAGE_MAX_RECURRENT];
		return CudaStageRecurrentBytes(parts, F::RecurrentParts(state, parts));
	}

	template<class S>
	static uint32_t LaneReady(const S *state)
	{
		return state->lane_arena != 0 ? 1u : 0u;
	}

	template<class S>
	static SparkStatus LaneAdmit(S *state, const SparkStageRunnerStep *step)
	{
		(void)state;
		return step->commit == 0u ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus LaneScratch(S *copy, const S *state, uint32_t lane)
	{
		if ( F::Plan(copy, (uint8_t *)state->lane_arena, lane) != state->lane_bytes )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		F::LaneShared::Copy(copy, state);
		return SPARK_STATUS_OK;
	}

	template<class S>
	static void LaneFold(S *state)
	{
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		{
			const S *copy = (const S *)state->lane_states[lane];
			if ( state->lane_bound[lane] == 0u || copy == 0 )
				continue;
			F::LaneCounters::Fold(state, copy, state->lane_base[lane]);
		}
	}

	template<class S>
	static SparkStatus BindTensors(S *state, SparkWeightdLazyPack *lazy_pack, const char *prefix, const CudaStageTensors *tensors)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		uint64_t offset, bytes;
		uint32_t rows;
		if ( tensors->overflow != 0u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		for ( uint32_t index = 0u; index < tensors->count; ++index )
		{
			const CudaStageTensor *tensor = &tensors->entries[index];
			const void *bound = 0;
			SparkStatus status;
			snprintf(key, sizeof(key), "%s%s", prefix, tensor->name);
			if ( tensor->rows != 0u )
			{
				rows = 0u;
				if ( SparkNamedPackTensorRange(&state->pack, F::Tag(), key, tensor->kind, &offset, &bytes, &rows) != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
				if ( rows != tensor->rows )
				{
					fprintf(stderr, "%s: pack tensor %s has %u rows, the model needs %u\n", F::Tag(), key, rows, tensor->rows);
					SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
				}
			}
			status = SparkNamedPackBindSized(&state->pack, lazy_pack, F::Tag(), key, tensor->kind, tensor->bytes, &bound);
			*tensor->field = bound;
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus HeadRanges(S *state, SparkStageRunnerModelGeometry *geometry, bool tied, bool prenormed)
	{
		if ( SparkNamedPackTensorRange(&state->pack, F::Tag(), "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &geometry->embed_rows) !=
			SPARK_STATUS_OK || (!prenormed && SparkNamedPackTensorRange(&state->pack, F::Tag(), "head_norm", "f32", &geometry->head_norm_offset,
			&geometry->head_norm_bytes, 0) != SPARK_STATUS_OK) )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( tied )
		{
			geometry->head_offset = geometry->embed_offset;
			geometry->head_bytes = geometry->embed_bytes;
			geometry->head_rows = geometry->embed_rows;
		}
		else if ( SparkNamedPackTensorRange(&state->pack, F::Tag(), "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) !=
			SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		return SPARK_STATUS_OK;
	}

	template<class S>
	static SparkStatus Configure(S *state, const SparkStageRunnerConfiguration *configuration, SparkStageRunnerModelGeometry *geometry)
	{
		SparkStatus status = F::Prepare(state, configuration);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, F::kPackAlignment, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		status = F::Config(state, configuration);
		if ( status == SPARK_STATUS_OK )
			status = F::Size(state);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes + state->lane_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes (%llu for the second prefill lane) is over the %llu-byte budget (rows %u, slots %u)\n", F::Tag(),
				(unsigned long long)(state->arena_bytes + state->lane_bytes), (unsigned long long)state->lane_bytes,
				(unsigned long long)configuration->state_budget_bytes, state->max_rows, state->slots);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		geometry->hidden = F::kHidden;
		geometry->vocab = F::kVocab;
		geometry->total_layers = F::kLayers;
		geometry->layer_count = F::kLayers;
		geometry->rms_epsilon = F::kEpsilon;
		geometry->pack_bytes = state->pack.file_bytes;
		F::Kv::Geometry(geometry);
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->recurrent_bytes = F::RecurrentBytes(state);
		geometry->head_norm_f32 = 1u;
		return F::Head(state, geometry);
	}

	template<class S>
	static SparkStatus Allocate(S *state)
	{
		SparkStatus status = F::Arrange(state);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			fprintf(stderr, "%s: the %llu-byte rank state could not be allocated\n", F::Tag(), (unsigned long long)state->arena_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || F::Plan(state, (uint8_t *)state->arena, 0u) != state->arena_bytes )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if constexpr ( F::kLanes )
		{
			if ( cudaMalloc(&state->lane_arena, state->lane_bytes) != cudaSuccess )
			{
				state->lane_arena = 0;
				fprintf(stderr, "%s: the %llu-byte second lane state could not be allocated\n", F::Tag(), (unsigned long long)state->lane_bytes);
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			}
			if ( cudaMemset(state->lane_arena, 0, state->lane_bytes) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
				if ( (state->lane_states[lane] = calloc(1u, sizeof(S))) == 0 )
					SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		return F::Upload(state);
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		using State = typename F::State;
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		State *state;
		if ( F::Accepts(configuration) == 0u || configuration->stage_count != 1u || configuration->tp_degree != F::kTpDegree ||
			configuration->max_input_row_count == 0u || configuration->state_budget_bytes == 0u )
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
		memset(geometry, 0, sizeof(*geometry));
		return F::Configure(state, configuration, geometry);
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: a dense pack's weightd manifest carries no expert ranges\n", F::Tag());
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		using State = typename F::State;
		State *state = (State *)model;
		CudaStageTensors tensors;
		char prefix[32];
		SparkStatus status;
		(void)geometry;
		memset(&tensors, 0, sizeof(tensors));
		status = F::GlobalTensors(state, &tensors);
		if ( status == SPARK_STATUS_OK )
			status = F::BindTensors(state, lazy_pack, "", &tensors);
		if ( status != SPARK_STATUS_OK )
			return F::BindRefusal(status);
		for ( uint32_t index = 0u; index < F::kLayers; ++index )
		{
			memset(&tensors, 0, sizeof(tensors));
			snprintf(prefix, sizeof(prefix), "layers.%u.", index);
			status = F::LayerTensors(state, index, &tensors);
			if ( status == SPARK_STATUS_OK )
				status = F::BindTensors(state, lazy_pack, prefix, &tensors);
			if ( status != SPARK_STATUS_OK )
				return F::BindRefusal(status);
			status = F::LayerBound(state, index);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		return F::Allocate(state);
	}

	static void Close(void *model)
	{
		using State = typename F::State;
		State *state = (State *)model;
		if ( state == 0 )
			return;
		F::Release(state);
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->lane_arena != 0 )
			(void)cudaFree(state->lane_arena);
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
			free(state->lane_states[lane]);
		if constexpr ( std::is_base_of<CudaStagePackState, State>::value )
			if ( state->pack_open != 0u )
				SparkNamedPackClose(&state->pack);
		free(state);
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream)
	{
		return F::Schedule::Step((typename F::State *)model, step, stream);
	}

	static SparkStatus StepBegin(void *model, const SparkStageRunnerStep *step, void *frame, void *queue)
	{
		return F::Schedule::Begin((typename F::State *)model, step, queue, (typename F::Frame *)frame);
	}

	static SparkStatus StepLayer(void *model, void *frame, uint32_t index, void *queue)
	{
		(void)queue;
		return F::Schedule::Layer((typename F::State *)model, *(typename F::Frame *)frame, index);
	}

	static SparkStatus StepEnd(void *model, void *frame, SparkStatus status, void *queue)
	{
		(void)queue;
		return F::Schedule::End((typename F::State *)model, *(typename F::Frame *)frame, status);
	}

	static SparkStatus StepWave(void *model, void *frame, void *queue, uint32_t *more)
	{
		(void)queue;
		return F::Schedule::Wave((typename F::State *)model, *(typename F::Frame *)frame, more);
	}

	static SparkStatus StepWaveEnd(void *model, void *frame, void *queue)
	{
		(void)queue;
		return F::Schedule::WaveEnd((typename F::State *)model, *(typename F::Frame *)frame);
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		return F::Kv::Attach(F::Tag(), (typename F::State *)model, kv);
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		using State = typename F::State;
		const State *state = (const State *)model;
		CudaStageRecurrent parts[CUDA_STAGE_MAX_RECURRENT];
		const uint32_t count = F::RecurrentParts(state, parts);
		return CudaStageRecurrentCopy(F::Tag(), parts, count, state->slots, to_buffer, slot, buffer, bytes, stream);
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		using State = typename F::State;
		const State *state = (const State *)model;
		CudaStageRecurrent parts[CUDA_STAGE_MAX_RECURRENT];
		const uint32_t count = F::RecurrentParts(state, parts);
		return CudaStageRecurrentReset(parts, count, state->slots, slot, stream);
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		using State = typename F::State;
		State *state = (State *)model;
		CudaStageReportField fields[CUDA_STAGE_MAX_REPORT];
		F::LaneFold(state);
		CudaStageReport(F::ReportTag(), rank, fields, F::ReportFields(state, fields));
	}

	static SparkStatus Lane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
	{
		using State = typename F::State;
		State *state = (State *)model;
		State *copy;
		SparkStatus status;
		if ( lane >= SPARK_STAGE_RUNNER_LANES || step == 0 || services == 0 || lane_model == 0 || state->lane_states[lane] == 0 ||
			F::LaneReady(state) == 0u )
			SPARK_FAIL(F::kLaneRefusal);
		status = F::LaneAdmit(state, step);
		if ( status != SPARK_STATUS_OK )
			return status;
		F::LaneFold(state);
		copy = (State *)state->lane_states[lane];
		memcpy(copy, state, sizeof(*copy));
		copy->services = services;
		if ( lane != 0u )
		{
			status = F::LaneScratch(copy, state, lane);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		state->lane_bound[lane] = 1u;
		F::LaneCounters::Mark(copy, state->lane_base[lane]);
		*lane_model = copy;
		return SPARK_STATUS_OK;
	}

	static uint32_t (*GraphKeyEntry(void))(void *, const SparkStageRunnerStep *)
	{
		if constexpr ( F::kGraphKey )
			return F::GraphKey;
		else
			return 0;
	}

	static SparkStatus (*LaneEntry(void))(void *, uint32_t, const SparkStageRunnerStep *, const SparkStageRunnerServices *, void **)
	{
		if constexpr ( F::kLanes )
			return F::Lane;
		else
			return 0;
	}

	static SparkStatus (*KeepRowsEntry(void))(void *, const SparkStageRunnerStep *, const uint32_t *, void *)
	{
		if constexpr ( F::kKeepRows )
			return F::KeepRows;
		else
			return 0;
	}

	static SparkStatus (*TapPlanEntry(void))(void *, const SparkStageRunnerTapPlan *)
	{
		if constexpr ( F::kTapPlan )
			return F::TapPlan;
		else
			return 0;
	}

	static SparkStatus (*CheckEntry(void))(void *)
	{
		if constexpr ( F::kCheck )
			return F::CheckErrors;
		else
			return 0;
	}

	static SparkStatus (*WaveEntry(void))(void *, void *, void *, uint32_t *)
	{
		if constexpr ( F::kWaves )
			return StepWave;
		else
			return 0;
	}

	static SparkStatus (*WaveEndEntry(void))(void *, void *, void *)
	{
		if constexpr ( F::kWaves )
			return StepWaveEnd;
		else
			return 0;
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			F::Tag(),
			F::WeightdModel(),
			F::WeightdRevision(),
			F::Open,
			F::ManifestCheck,
			F::Bind,
			F::Close,
			F::Step,
			F::Sideband,
			F::AttachKv,
			F::RecurrentCopy,
			F::ResetSlot,
			F::ProbeBuffers,
			F::Report,
			GraphKeyEntry(),
			LaneEntry(),
			KeepRowsEntry(),
			TapPlanEntry(),
			0,
			CheckEntry(),
			0,
			F::StepBegin,
			WaveEntry(),
			F::StepLayer,
			WaveEndEntry(),
			F::StepEnd,
			(uint32_t)sizeof(typename F::Frame),
			F::kChunkRows
		};
		return &interface_table;
	}
};
