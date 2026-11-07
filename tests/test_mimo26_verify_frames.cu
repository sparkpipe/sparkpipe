#include "modules/mimo26_flash_tp16_resident_decode_stage/source/spark_mimo26_flash_tp16_stage_model.cu"
#include "modules/mimo26_pro_tp16_resident_decode_stage/source/spark_mimo26_pro_tp16_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define VF_MAX_FRAME SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE
#define VF_SLOTS 2u
#define VF_PAGES 8u
#define VF_PREFIX 160u
#define VF_OTHER_PREFIX 100u
#define VF_MAX_ALLOCATIONS 4096u
#define VF_FILL_BF16 0u
#define VF_FILL_F32 1u
#define VF_FILL_FP8 2u
#define VF_FILL_E2M1 3u
#define VF_FILL_E8M0 4u

struct SmallFlash : Mimo26FlashGeometry<SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kLayers = 6u;
	static constexpr uint32_t kFullLayers = 2u;
	static constexpr uint32_t kMoeInter = 128u;
	static bool FullLayer(uint32_t layer) { return layer == 0u || layer == 5u; }
	static const char *Tag(void) { return "mimo-v2.6-flash"; }
};

struct SmallPro : Mimo26ProGeometry<SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kFullLayers = 2u;
	static constexpr uint32_t kMoeInter = 128u;
	static constexpr uint32_t kWindow = 24u;
	static bool FullLayer(uint32_t layer) { return layer == 0u || layer == 3u; }
	static const char *Tag(void) { return "mimo-v2.6-pro"; }
};

__global__ static void VfFillKernel(void *target, uint64_t count, uint32_t kind, uint32_t seed, float scale, float offset)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint32_t bits = Mix(seed * 0x9e3779b9u + (uint32_t)index);
		const float value = offset + scale * Uniform(seed, (uint32_t)index);
		if ( kind == VF_FILL_BF16 )
			((uint16_t *)target)[index] = Bf16(value);
		else if ( kind == VF_FILL_F32 )
			((float *)target)[index] = value;
		else if ( kind == VF_FILL_FP8 )
			((uint8_t *)target)[index] = (uint8_t)((bits & 0x80u) | ((5u + ((bits >> 3u) & 3u)) << 3u) | (bits & 7u));
		else if ( kind == VF_FILL_E2M1 )
			((uint8_t *)target)[index] = (uint8_t)(bits >> 8u);
		else
			((uint8_t *)target)[index] = (uint8_t)((uint32_t)offset + (bits >> 8u) % 3u);
	}
}

__global__ static void VfSumKernel(const uint8_t *staging, uint64_t pitch, uint32_t ranks, uint16_t *output, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t rank = 0u; rank < ranks; ++rank )
			total += LmBf16ToFloat(((const uint16_t *)(staging + (uint64_t)rank * pitch))[index]);
		output[index] = LmFloatToBf16(total);
	}
}

typedef struct VfBarrier
{
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	uint32_t parties;
	uint32_t waiting;
	uint32_t generation;
	uint32_t aborted;
} VfBarrier;

static int VfWait(VfBarrier *barrier)
{
	int ok;
	pthread_mutex_lock(&barrier->mutex);
	const uint32_t generation = barrier->generation;
	if ( barrier->aborted == 0u && ++barrier->waiting == barrier->parties )
	{
		barrier->waiting = 0u;
		barrier->generation++;
		pthread_cond_broadcast(&barrier->cond);
	}
	else
		while ( barrier->aborted == 0u && generation == barrier->generation )
			pthread_cond_wait(&barrier->cond, &barrier->mutex);
	ok = barrier->aborted == 0u;
	pthread_mutex_unlock(&barrier->mutex);
	return ok;
}

static void VfAbort(VfBarrier *barrier)
{
	pthread_mutex_lock(&barrier->mutex);
	barrier->aborted = 1u;
	pthread_cond_broadcast(&barrier->cond);
	pthread_mutex_unlock(&barrier->mutex);
}

template<class G>
struct VfRig
{
	using M = MimoModel<G>;
	using S = MimoShape<G>;
	using State = MimoState<G>;
	static constexpr uint32_t kRanks = G::kTpDegree;
	static constexpr uint64_t kSlidingStride = (uint64_t)VF_SLOTS * VF_PAGES * (S::SlidingKv::kPageBytes / kRanks);
	static constexpr uint64_t kFullStride = (uint64_t)VF_SLOTS * VF_PAGES * (S::FullKv::kPageBytes / kRanks);
	static constexpr uint64_t kSlidingRegion = (uint64_t)S::kSlidingLayers * kSlidingStride;
	static constexpr uint64_t kRankKv = kSlidingRegion + (uint64_t)G::kFullLayers * kFullStride;
	static_assert(S::SlidingKv::kPageBytes % kRanks == 0u && S::FullKv::kPageBytes % kRanks == 0u, "a rank's page holds whole slots");

	struct Rank
	{
		VfRig *rig;
		uint32_t index;
		State *state;
		SparkStageRunnerServices services;
		cudaStream_t stream;
		uint32_t *positions, *context, *sequence_of_row;
		uint16_t *hidden;
		SparkStageRunnerStep step;
		SparkStatus status;
	};

	Rank ranks[kRanks];
	VfBarrier barrier;
	uint8_t *kv, *staging;
	uint64_t staging_bytes;
	uint32_t *page_table;
	uint32_t multiprocessors;
	uint32_t host_context[VF_SLOTS];
	void *allocations[VF_MAX_ALLOCATIONS];
	uint32_t allocated;

	void *Fill(uint64_t count, uint32_t kind, float scale, float offset)
	{
		const uint64_t bytes = count * (kind == VF_FILL_BF16 ? sizeof(uint16_t) : kind == VF_FILL_F32 ? sizeof(float) : 1u);
		void *device = 0;
		if ( allocated == VF_MAX_ALLOCATIONS || cudaMalloc(&device, bytes) != cudaSuccess )
		{
			printf("FAIL weight allocation of %llu bytes\n", (unsigned long long)bytes);
			exit(1);
		}
		allocations[allocated++] = device;
		VfFillKernel<<<1024u, 256u>>>(device, count, kind, vf_seed++, scale, offset);
		return device;
	}

	const uint16_t *Matrix(uint64_t rows, uint64_t columns)
	{
		return (const uint16_t *)Fill(rows * columns, VF_FILL_BF16, 1.7f / sqrtf((float)columns), 0.0f);
	}

	MimoScaled Fp8(uint32_t rows, uint32_t columns)
	{
		const float unit = 1.0f / (1.6f * sqrtf((float)columns));
		MimoScaled weight;
		weight.weight = (const uint8_t *)Fill((uint64_t)rows * columns, VF_FILL_FP8, 0.0f, 0.0f);
		weight.scale = Fill((uint64_t)rows * ((columns + MM_FP8_GROUP - 1u) / MM_FP8_GROUP), VF_FILL_F32, 0.25f * unit, 0.75f * unit);
		return weight;
	}

	MimoScaled Mxfp4(uint32_t rows, uint32_t columns)
	{
		const int exponent = 127 + (int)floorf(log2f(1.0f / (2.93f * sqrtf((float)columns))));
		MimoScaled weight;
		weight.weight = (const uint8_t *)Fill((uint64_t)G::kLocalExperts * rows * columns / 2u, VF_FILL_E2M1, 0.0f, 0.0f);
		weight.scale = Fill((uint64_t)G::kLocalExperts * rows * (columns / 32u), VF_FILL_E8M0, 0.0f, (float)(exponent - 1));
		return weight;
	}

	void Shared(MimoLayer *layer, uint32_t index)
	{
		memset(layer, 0, sizeof(*layer));
		layer->input_norm = (const uint16_t *)Fill(G::kHidden, VF_FILL_BF16, 0.2f, 1.0f);
		layer->post_attn_norm = (const uint16_t *)Fill(G::kHidden, VF_FILL_BF16, 0.2f, 1.0f);
		if ( !G::FullLayer(index) )
			layer->sinks = (const float *)Fill(G::kHeads, VF_FILL_F32, 1.0f, 0.5f);
		if ( index >= G::kDenseLayers )
		{
			layer->router = Matrix(G::kExperts, G::kHidden);
			layer->router_bias = (const float *)Fill(G::kExperts, VF_FILL_F32, 0.05f, 0.0f);
		}
	}

	void Local(MimoLayer *layer, uint32_t index)
	{
		layer->qkv = Fp8((G::FullLayer(index) ? S::kFullFused : S::kSlidingFused) / kRanks, G::kHidden);
		layer->out = Matrix(G::kHidden, S::kRankWidth);
		if ( index < G::kDenseLayers )
		{
			layer->ffn_gate_up = Fp8(2u * S::kDenseRows, G::kHidden);
			layer->ffn_down = Fp8(G::kHidden, S::kDenseRows);
			return;
		}
		layer->moe_w1 = Mxfp4(G::kMoeInter, G::kHidden);
		layer->moe_w3 = Mxfp4(G::kMoeInter, G::kHidden);
		layer->moe_w2 = Mxfp4(G::kHidden, G::kMoeInter);
	}

	static SparkStatus Round(void *context, void *stream_pointer, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		Rank *rank = (Rank *)context;
		VfRig *rig = rank->rig;
		cudaStream_t stream = (cudaStream_t)stream_pointer;
		const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G::kHidden) * sizeof(uint16_t);
		const uint64_t bytes = operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? kRanks * block : block;
		cudaError_t error = cudaSuccess;
		if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL || bytes > rig->staging_bytes || (operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER &&
			operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL && operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) )
			return SPARK_STATUS_UNSUPPORTED;
		if ( cudaMemcpyAsync(rig->staging + rank->index * rig->staging_bytes, local, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaStreamSynchronize(stream) != cudaSuccess || VfWait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
			error = cudaMemcpy2DAsync(full, block, rig->staging, rig->staging_bytes, block, kRanks, cudaMemcpyDeviceToDevice, stream);
		else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
			error = cudaMemcpy2DAsync(full, block, rig->staging + rank->index * block, rig->staging_bytes, block, kRanks, cudaMemcpyDeviceToDevice,
				stream);
		else
		{
			VfSumKernel<<<256u, 256u, 0, stream>>>(rig->staging, rig->staging_bytes, kRanks, (uint16_t *)full, block / sizeof(uint16_t));
			error = cudaPeekAtLastError();
		}
		if ( error != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess || VfWait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}

	static void *Thread(void *argument)
	{
		Rank *rank = (Rank *)argument;
		rank->status = M::Step(rank->state, &rank->step, rank->stream);
		if ( rank->status == SPARK_STATUS_OK && cudaStreamSynchronize(rank->stream) != cudaSuccess )
			rank->status = SPARK_STATUS_IO_ERROR;
		if ( rank->status != SPARK_STATUS_OK )
			VfAbort(&rank->rig->barrier);
		return 0;
	}

	int OpenRank(uint32_t index, const MimoLayer *shared)
	{
		Rank *rank = &ranks[index];
		SparkStageRunnerKv attach;
		State *state = (State *)calloc(1u, sizeof(State));
		rank->rig = this;
		rank->index = index;
		rank->state = state;
		rank->services.context = rank;
		rank->services.round = Round;
		if ( state == 0 )
			return 0;
		state->services = &rank->services;
		state->tp_degree = kRanks;
		state->tp_rank = index;
		state->max_rows = VF_MAX_FRAME;
		state->slots = VF_SLOTS;
		state->expert_first = index * G::kLocalExperts;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
		{
			state->layers[layer] = shared[layer];
			Local(&state->layers[layer], layer);
		}
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&rank->stream) != cudaSuccess ||
			cudaMalloc((void **)&rank->positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&rank->sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&rank->context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&rank->hidden, (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv + index * kRankKv;
		attach.layer_stride_bytes = kSlidingStride;
		attach.layer_page_bytes = S::SlidingKv::kPageBytes;
		attach.layer_count = S::kSlidingLayers;
		attach.page_table = page_table;
		attach.page_table_stride = VF_PAGES;
		attach.pool_page_count = VF_SLOTS * VF_PAGES;
		attach.sequence_count = VF_SLOTS;
		attach.context_shard.degree = kRanks;
		attach.context_shard.rank = index;
		attach.context_shard.grain = 1u;
		attach.second_pool = kv + index * kRankKv + kSlidingRegion;
		attach.second_layer_stride_bytes = kFullStride;
		attach.second_layer_page_bytes = S::FullKv::kPageBytes;
		attach.second_layer_count = G::kFullLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		MimoLayer shared[G::kLayers];
		uint64_t gather = 0u;
		memset(this, 0, sizeof(*this));
		barrier.parties = kRanks;
		if ( pthread_mutex_init(&barrier.mutex, 0) != 0 || pthread_cond_init(&barrier.cond, 0) != 0 ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kRanks * kRankKv) != cudaSuccess || cudaMemset(kv, 0, kRanks * kRankKv) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess )
			return 0;
		for ( uint32_t index = 0u; index < table.size(); ++index )
			table[index] = index;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
			Shared(&shared[layer], layer);
		for ( uint32_t index = 0u; index < kRanks; ++index )
			if ( OpenRank(index, shared) == 0 )
				return 0;
		gather = (uint64_t)VF_MAX_FRAME * (S::kMaxFused / kRanks) * sizeof(uint16_t);
		staging_bytes = (uint64_t)kRanks * ranks[0].state->record_stride * sizeof(float);
		staging_bytes = staging_bytes > gather ? staging_bytes : gather;
		staging_bytes = staging_bytes > (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t) ? staging_bytes :
			(uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t);
		return cudaMalloc((void **)&staging, kRanks * staging_bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
	}

	void Input(const VfRow &row, uint16_t *target) const
	{
		const uint32_t seed = Mix(row.sequence * 7919u + row.variant * 104729u) ^ row.position;
		for ( uint32_t index = 0u; index < G::kHidden; ++index )
			target[index] = Bf16(Uniform(seed, index));
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden), result(input.size()), other(input.size());
		std::vector<uint32_t> position(count), sequence(count);
		uint32_t seen[VF_SLOTS] = { 0u, 0u }, sequences = 0u;
		pthread_t threads[kRanks];
		int ok = 1;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden);
			position[row] = rows[row].position;
			sequence[row] = rows[row].sequence;
			if ( seen[rows[row].sequence] == 0u )
				sequences++;
			seen[rows[row].sequence] = 1u;
			host_context[rows[row].sequence] = rows[row].position + 1u;
		}
		for ( uint32_t index = 0u; index < kRanks; ++index )
		{
			Rank *rank = &ranks[index];
			memset(&rank->step, 0, sizeof(rank->step));
			rank->step.hidden_bf16 = rank->hidden;
			rank->step.positions = rank->positions;
			rank->step.context_length = rank->context;
			rank->step.sequence_of_row = rank->sequence_of_row;
			rank->step.rows = count;
			rank->step.sequences = sequences;
			rank->step.commit = commit;
			rank->step.context = VF_PAGES * G::kPageSlots;
			rank->step.multiprocessors = multiprocessors;
			rank->step.route_fault = VfRouteFault();
			if ( cudaMemcpyAsync(rank->hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, rank->stream) != cudaSuccess ||
				cudaMemcpyAsync(rank->positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rank->stream) != cudaSuccess ||
				cudaMemcpyAsync(rank->sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rank->stream) != cudaSuccess ||
				cudaMemcpyAsync(rank->context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, rank->stream) != cudaSuccess )
				return 0;
		}
		barrier.waiting = 0u;
		barrier.aborted = 0u;
		for ( uint32_t index = 0u; index < kRanks; ++index )
			if ( pthread_create(&threads[index], 0, Thread, &ranks[index]) != 0 )
			{
				VfAbort(&barrier);
				for ( uint32_t started = 0u; started < index; ++started )
					pthread_join(threads[started], 0);
				return 0;
			}
		for ( uint32_t index = 0u; index < kRanks; ++index )
		{
			pthread_join(threads[index], 0);
			ok = ok && ranks[index].status == SPARK_STATUS_OK;
		}
		if ( ok == 0 || cudaMemcpy(result.data(), ranks[0].hidden, result.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
			return 0;
		for ( uint32_t index = 1u; index < kRanks; ++index )
		{
			if ( cudaMemcpy(other.data(), ranks[index].hidden, other.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
			Require(memcmp(other.data(), result.data(), other.size() * sizeof(uint16_t)) == 0, G::Tag(), "every rank holds the same stage output (rank)",
				index, 0u, 0u);
		}
		if ( output != 0 )
			output->insert(output->end(), result.begin(), result.end());
		return 1;
	}

	int Decode(uint32_t sequence, uint32_t position, uint32_t variant, std::vector<uint16_t> *output)
	{
		const VfRow row = { sequence, position, variant };
		return Run(&row, 1u, 1u, output);
	}

	std::vector<uint8_t> Save(void) const
	{
		std::vector<uint8_t> copy(kRanks * kRankKv);
		if ( cudaMemcpy(copy.data(), kv, copy.size(), cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		return copy.size() == kRanks * kRankKv && cudaMemcpy(kv, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess;
	}

	template<class Kv>
	static uint32_t RegionMatches(const uint8_t *a, const uint8_t *b, uint32_t rank, uint32_t layers, uint64_t layer_stride, uint32_t limit)
	{
		SparkKvShard shard;
		shard.degree = kRanks;
		shard.rank = rank;
		shard.grain = 1u;
		for ( uint32_t layer = 0u; layer < layers; ++layer )
			for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
				for ( uint32_t position = rank; position < VF_PAGES * Kv::kPageSlots; position += kRanks )
				{
					const uint64_t offset = (uint64_t)layer * layer_stride + ((uint64_t)sequence * VF_PAGES + position / Kv::kPageSlots) *
						(Kv::kPageBytes / kRanks) + (uint64_t)SparkKvShardSlotInPage(shard, Kv::kPageSlots, position) * Kv::kSlotBytes;
					if ( sequence == 0u && position >= limit )
						continue;
					if ( memcmp(a + offset, b + offset, Kv::kSlotBytes) != 0 )
						return 0u;
				}
		return 1u;
	}

	static uint32_t KvMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t limit)
	{
		if ( a.size() != b.size() || a.size() != kRanks * kRankKv )
			return 0u;
		for ( uint32_t rank = 0u; rank < kRanks; ++rank )
		{
			const uint8_t *left = a.data() + rank * kRankKv, *right = b.data() + rank * kRankKv;
			if ( RegionMatches<typename S::SlidingKv>(left, right, rank, S::kSlidingLayers, kSlidingStride, limit) == 0u ||
				RegionMatches<typename S::FullKv>(left + kSlidingRegion, right + kSlidingRegion, rank, G::kFullLayers, kFullStride, limit) == 0u )
				return 0u;
		}
		return 1u;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError error, zero;
		memset(&zero, 0, sizeof(zero));
		for ( uint32_t index = 0u; index < kRanks; ++index )
			if ( cudaMemcpy(&error, ranks[index].state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) != cudaSuccess ||
				memcmp(&error, &zero, sizeof(error)) != 0 )
				return 0u;
		return 1u;
	}

	void Close(void)
	{
		for ( uint32_t index = 0u; index < kRanks; ++index )
		{
			Rank *rank = &ranks[index];
			M::Close(rank->state);
			cudaFree(rank->positions);
			cudaFree(rank->sequence_of_row);
			cudaFree(rank->context);
			cudaFree(rank->hidden);
			if ( rank->stream != 0 )
				cudaStreamDestroy(rank->stream);
		}
		for ( uint32_t index = 0u; index < allocated; ++index )
			cudaFree(allocations[index]);
		cudaFree(kv);
		cudaFree(staging);
		cudaFree(page_table);
		pthread_cond_destroy(&barrier.cond);
		pthread_mutex_destroy(&barrier.mutex);
	}
};

template<class G>
static void Check(void)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u, VF_MAX_FRAME };
	static VfRig<G> rig;
	std::vector<VfRow> rows;
	std::vector<uint8_t> prefix;
	const char *model = G::Tag();
	uint32_t checked = 0u;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "the synthetic stage opens its ranks, binds and attaches its KV pools", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
	{
		rows.clear();
		for ( uint32_t position = 0u; position < (sequence == 0u ? VF_PREFIX : VF_OTHER_PREFIX); ++position )
			rows.push_back({ sequence, position, 0u });
		Require(rig.Run(rows.data(), (uint32_t)rows.size(), 1u, 0) != 0, model, "the prefix prefill runs", sequence, 0u, 0u);
	}
	prefix = rig.Save();
	for ( uint32_t frame : frames )
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<std::vector<uint8_t> > before;
		std::vector<uint32_t> kept;
		std::vector<uint8_t> after_decode, after_verify;
		Kept(frame, &kept);
		rig.Load(prefix);
		for ( uint32_t row = 0u; row < frame; ++row )
		{
			for ( uint32_t k : kept )
				if ( k == row )
					before.push_back(rig.Save());
			Require(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode runs", frame, row, 0u);
		}
		before.push_back(rig.Save());
		after_decode = before.back();
		rig.Load(prefix);
		rows.clear();
		for ( uint32_t row = 0u; row < frame; ++row )
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
		Require(rig.Run(rows.data(), frame, 0u, &verified) != 0, model, "the verify frame runs", frame, 0u, 0u);
		after_verify = rig.Save();
		Require(Finite(verified) != 0u, model, "verify rows are finite and not constant", frame, 0u, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every verify row equals its one-row decode bit for bit (frame rows, first differing row)", frame, FirstDifference(verified, decoded, G::kHidden), 0u);
		Require(rig.KvMatches(after_verify, after_decode, UINT32_MAX) != 0u, model,
			"the KV the verify frame leaves on every rank equals the KV of the one-row decodes at every position", frame, 0u, 0u);
		for ( uint32_t index = 0u; index < kept.size(); ++index )
		{
			const uint32_t k = kept[index], limit = VF_PREFIX + k + 2u;
			std::vector<uint16_t> speculated, plain;
			rig.Load(after_verify);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &speculated) != 0, model, "the decode after a rejected tail runs", frame, k, 0u);
			Require(rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &speculated) != 0, model, "the second decode after a rejected tail runs", frame, k, 0u);
			const std::vector<uint8_t> speculated_kv = rig.Save();
			rig.Load(before[index]);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &plain) != 0, model,
				"the decodes of the run that never saw the draft run", frame, k, 0u);
			Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"after keeping k rows the next two decodes equal a run that never saw the draft (frame rows, k, first differing step)", frame, k,
				FirstDifference(speculated, plain, G::kHidden));
			Require(rig.KvMatches(speculated_kv, rig.Save(), limit) != 0u, model,
				"after keeping k rows the KV up to the next decode equals a run that never saw the draft (frame rows, k)", frame, k, 0u);
			checked++;
		}
	}
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<uint8_t> after_decode;
		const uint32_t first = 7u, second = 16u;
		rig.Load(prefix);
		rows.clear();
		for ( uint32_t row = 0u; row < first; ++row )
		{
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
			Require(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the first sequence runs", row, 0u, 0u);
		}
		for ( uint32_t row = 0u; row < second; ++row )
		{
			rows.push_back({ 1u, VF_OTHER_PREFIX + row, 0u });
			Require(rig.Decode(1u, VF_OTHER_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the second sequence runs", row, 0u, 0u);
		}
		after_decode = rig.Save();
		rig.Load(prefix);
		Require(rig.Run(rows.data(), first + second, 0u, &verified) != 0, model, "a verify frame over two sequences runs", first, second, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every row of a two-sequence verify frame equals its one-row decode (first differing row)", FirstDifference(verified, decoded, G::kHidden), 0u, 0u);
		Require(rig.KvMatches(rig.Save(), after_decode, UINT32_MAX) != 0u, model, "a two-sequence verify frame leaves the decodes' KV", 0u, 0u, 0u);
	}
	Require(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("%s %s: %u ranks, frames of 1-%u rows, %u rejection points\n", vf_failures == 0 ? "ok" : "checked", model, VfRig<G>::kRanks, VF_MAX_FRAME,
		checked);
	rig.Close();
}

int main(void)
{
	Check<SmallFlash>();
	Check<SmallPro>();
	if ( vf_failures != 0 )
	{
		printf("FAIL mimo26 verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS mimo26 verify frames: every row of every verify frame equals one-row decode bit for bit, and a rejected tail leaves no trace\n");
	return 0;
}
