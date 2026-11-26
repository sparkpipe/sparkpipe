#include "model-families/hy4/source/hy4_stage.cuh"
#include "model-families/mimo26/source/mimo26_stage.cuh"
#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define GB_MAX_ROWS 512u
#define GB_SLOTS 2u
#define GB_PAGES 96u
#define GB_SHORT_CTX 128u
#define GB_LONG_CTX 2560u
#define GB_WARM_CTX 2048u
#define GB_SPARSE_CTX 4096u
#define GB_FILL_BF16 0u
#define GB_FILL_F32 1u
#define GB_FILL_FP8 2u
#define GB_FILL_E2M1 3u
#define GB_FILL_E8M0 4u

static int gb_failures = 0;

static void GbCheck(int ok, const char *model, const char *detail, uint32_t a, uint32_t b)
{
	if ( ok == 0 )
	{
		fprintf(stderr, "FAIL %s: %s (%u, %u)\n", model, detail, a, b);
		gb_failures++;
	}
}

__global__ static void GbFillKernel(void *target, uint64_t count, uint32_t kind, uint32_t seed, float scale, float offset)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint32_t bits = Mix(seed * 0x9e3779b9u + (uint32_t)index);
		const float value = offset + scale * Uniform(seed, (uint32_t)index);
		if ( kind == GB_FILL_BF16 )
			((uint16_t *)target)[index] = Bf16(value);
		else if ( kind == GB_FILL_F32 )
			((float *)target)[index] = value;
		else if ( kind == GB_FILL_FP8 )
			((uint8_t *)target)[index] = (uint8_t)((bits & 0x80u) | ((5u + ((bits >> 3u) & 3u)) << 3u) | (bits & 7u));
		else if ( kind == GB_FILL_E2M1 )
			((uint8_t *)target)[index] = (uint8_t)(bits >> 8u);
		else
			((uint8_t *)target)[index] = (uint8_t)((uint32_t)offset + (bits >> 8u) % 3u);
	}
}

__global__ static void GbAllGatherKernel(const uint8_t *src, uint8_t *dst, uint64_t slice_bytes, uint32_t degree)
{
	const uint64_t words = slice_bytes >> 2u;
	const uint32_t *s32 = (const uint32_t *)src;
	uint32_t *d32 = (uint32_t *)dst;
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint32_t v = s32[i];
		for ( uint32_t r = 0u; r < degree; ++r )
			d32[(uint64_t)r * words + i] = v;
	}
}

__global__ static void GbCopyBytesKernel(const uint8_t *src, uint8_t *dst, uint64_t bytes)
{
	const uint64_t words = bytes >> 2u;
	const uint32_t *s32 = (const uint32_t *)src;
	uint32_t *d32 = (uint32_t *)dst;
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (uint64_t)gridDim.x * blockDim.x )
		d32[i] = s32[i];
}

struct GbPool
{
	std::vector<void *> ptrs;
	uint32_t seed;
	cudaStream_t stream;

	GbPool(void) : seed(101u), stream(0) {}

	void *Alloc(uint64_t bytes)
	{
		void *p = 0;
		const uint64_t aligned = ((bytes + 127u) / 128u) * 128u;
		if ( cudaMalloc(&p, aligned > 0u ? aligned : 128u) != cudaSuccess )
			return 0;
		ptrs.push_back(p);
		return p;
	}

	void *Fill(uint64_t count, uint32_t elem_bytes, uint32_t kind, float scale, float offset)
	{
		void *p = Alloc(count * elem_bytes);
		if ( p == 0 )
			return 0;
		const uint32_t blocks = (uint32_t)((count + 255u) / 256u);
		const uint32_t capped = blocks > 4096u ? 4096u : (blocks == 0u ? 1u : blocks);
		GbFillKernel<<<capped, 256u, 0, stream>>>(p, count, kind, seed++, scale, offset);
		return p;
	}

	void Release(void)
	{
		for ( void *p : ptrs )
			cudaFree(p);
		ptrs.clear();
	}
};

struct GbRow
{
	uint32_t seq;
	uint32_t pos;
	uint32_t variant;
};

static std::vector<uint16_t> GbHiddenHost(const GbRow *rows, uint32_t count, uint32_t width)
{
	std::vector<uint16_t> out((uint64_t)count * width);
	for ( uint32_t r = 0u; r < count; ++r )
	{
		const uint32_t seed = 0x517cc1b7u ^ (rows[r].seq * 4099u) ^ (rows[r].pos * 131u) ^ (rows[r].variant * 17u);
		for ( uint32_t c = 0u; c < width; ++c )
			out[(uint64_t)r * width + c] = Bf16(0.5f * Uniform(seed, c));
	}
	return out;
}

static uint32_t GbFirstDiff(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, uint32_t width)
{
	const uint64_t rows = a.size() / width;
	if ( a.size() != b.size() )
		return 0u;
	for ( uint64_t r = 0u; r < rows; ++r )
		if ( memcmp(a.data() + r * width, b.data() + r * width, width * sizeof(uint16_t)) != 0 )
			return (uint32_t)r;
	return UINT32_MAX;
}

static uint32_t GbFinite(const std::vector<uint16_t> &v)
{
	if ( v.empty() )
		return 0u;
	uint16_t first = v[0];
	uint32_t same = 1u;
	for ( uint16_t bits : v )
	{
		const uint32_t u = (uint32_t)bits << 16u;
		float f = 0.0f;
		memcpy(&f, &u, sizeof(float));
		if ( !isfinite(f) )
			return 0u;
		if ( bits != first )
			same = 0u;
	}
	return same == 0u ? 1u : 0u;
}

template<class G>
struct GbHyGeom : Hy4Geometry<SPARK_HY4_TP16_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kLayers = 3u;
	static constexpr uint32_t kIndexLayers = 2u;
	static constexpr bool kMixedFp8 = G::kMixedFp8;
	static const char *Tag(void) { return G::Tag(); }
};

struct GbHyPreviewCfg
{
	static constexpr bool kMixedFp8 = false;
	static const char *Tag(void) { return "hy4-preview"; }
};

struct GbHyFp8Cfg
{
	static constexpr bool kMixedFp8 = true;
	static const char *Tag(void) { return "hy4-fp8"; }
};

template<class Cfg>
struct GbHyRig
{
	using G = GbHyGeom<Cfg>;
	using M = HyModel<G>;
	using S = HyShape<G>;
	using State = HyState<G>;
	static constexpr bool kSingleRankPrefixExact = true;
	static constexpr uint64_t kPool = (uint64_t)GB_SLOTS * GB_PAGES;
	static constexpr uint64_t kLatentRegion = (uint64_t)G::kLayers * kPool * S::Kv::kPageBytes;
	static constexpr uint64_t kIndexRegion = (uint64_t)G::kIndexLayers * kPool * S::IndexKv::kPageBytes;

	GbPool pool;
	VfBus bus;
	SparkStageRunnerServices services;
	State *state;
	cudaStream_t stream;
	uint8_t *kv;
	uint16_t *hidden;
	uint32_t *seq_of_row;
	uint32_t *pos_dev;
	uint32_t *ctx_dev;
	uint32_t *pages_dev;
	uint32_t *fault_dev;
	uint32_t sms;

	GbHyRig(void) : bus{}, services{}, state(0), stream(0), kv(0), hidden(0), seq_of_row(0), pos_dev(0), ctx_dev(0), pages_dev(0), fault_dev(0), sms(48u) {}

	HyScaled MakeScaled(uint64_t rows, uint64_t cols, uint8_t codec)
	{
		HyScaled w = {};
		w.codec = codec;
		if ( codec == HY_CODEC_BF16 )
		{
			w.weight = pool.Fill(rows * cols, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)cols), 0.0f);
			w.scale = 0;
			return w;
		}
		if ( codec == HY_CODEC_FP8_BLOCK128 )
		{
			w.weight = pool.Fill(rows * cols, 1u, GB_FILL_FP8, 1.0f, 0.0f);
			w.scale = pool.Fill(rows * (cols / HY_FP8_BLOCK), 4u, GB_FILL_F32, 0.25f / sqrtf((float)cols), 1.0f / sqrtf((float)cols));
			return w;
		}
		const int exp = 127 - (int)ceilf(log2f(sqrtf((float)cols)));
		w.weight = pool.Fill(rows * cols, 1u, GB_FILL_FP8, 1.0f, 0.0f);
		w.scale = pool.Fill(rows * (cols / HY_SCALE_GROUP), 1u, GB_FILL_E8M0, 1.0f, (float)(exp - 1));
		return w;
	}

	int Open(void)
	{
		cudaDeviceProp prop = {};
		if ( cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0 )
			sms = (uint32_t)prop.multiProcessorCount;
		if ( cudaStreamCreate(&stream) != cudaSuccess )
			return 0;
		pool.stream = stream;
		bus.hidden = G::kHidden;
		services.context = &bus;
		services.round = VfRound;
		state = (State *)calloc(1u, sizeof(State));
		if ( state == 0 )
			return 0;
		state->services = &services;
		state->tp_degree = G::kTpDegree;
		state->tp_rank = 0u;
		state->slots = GB_SLOTS;
		state->max_rows = GB_MAX_ROWS;
		state->ffn_rows = 1024u;
		state->shared_rows = 128u;
		state->expert_first = 0u;
		state->context_capacity = GB_PAGES * G::kPageSlots;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		state->head_hc_fn = (const float *)pool.Fill((uint64_t)G::kStreams * S::kResidual, 4u, GB_FILL_F32, 0.01f, 0.0f);
		state->head_hc_base = (const float *)pool.Fill(G::kStreams, 4u, GB_FILL_F32, 0.1f, 0.25f);
		state->head_hc_scale = (const float *)pool.Fill(1u, 4u, GB_FILL_F32, 0.05f, 0.1f);
		for ( uint32_t i = 0u; i < G::kLayers; ++i )
		{
			HyLayer &l = state->layers[i];
			const uint8_t lin = Cfg::kMixedFp8 ? (i == 0u ? HY_CODEC_BF16 : HY_CODEC_FP8_BLOCK128) : HY_CODEC_MXFP8;
			const uint8_t moe = Cfg::kMixedFp8 ? HY_CODEC_FP8_BLOCK128 : HY_CODEC_MXFP8;
			const uint8_t shd = Cfg::kMixedFp8 ? HY_CODEC_BF16 : HY_CODEC_MXFP8;
			l.attn_hc_fn = (const float *)pool.Fill((uint64_t)2u * G::kStreams * S::kResidual, 4u, GB_FILL_F32, 0.01f, 0.0f);
			l.attn_hc_base = (const float *)pool.Fill(2u * G::kStreams, 4u, GB_FILL_F32, 0.1f, 0.25f);
			l.attn_hc_scale = (const float *)pool.Fill(2u, 4u, GB_FILL_F32, 0.05f, 0.1f);
			l.mlp_hc_fn = (const float *)pool.Fill((uint64_t)2u * G::kStreams * S::kResidual, 4u, GB_FILL_F32, 0.01f, 0.0f);
			l.mlp_hc_base = (const float *)pool.Fill(2u * G::kStreams, 4u, GB_FILL_F32, 0.1f, 0.25f);
			l.mlp_hc_scale = (const float *)pool.Fill(2u, 4u, GB_FILL_F32, 0.05f, 0.1f);
			l.input_norm = (const uint16_t *)pool.Fill(G::kHidden, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.post_norm = (const uint16_t *)pool.Fill(G::kHidden, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.q_a = MakeScaled(G::kQLora, G::kHidden, lin);
			l.q_norm = (const uint16_t *)pool.Fill(G::kQLora, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.q_b = MakeScaled((uint64_t)S::kRankHeads * S::kQkDim, G::kQLora, lin);
			l.kv_a = MakeScaled(S::kSlot, G::kHidden, lin);
			l.kv_norm = (const uint16_t *)pool.Fill(G::kLatent, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.uk = (const uint16_t *)pool.Fill((uint64_t)S::kRankHeads * G::kLatent * G::kNope, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)G::kNope), 0.0f);
			l.uv = (const uint16_t *)pool.Fill((uint64_t)S::kRankHeads * G::kValueDim * G::kLatent, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)G::kLatent), 0.0f);
			l.gate = MakeScaled((uint64_t)S::kRankHeads * G::kValueDim, G::kHidden, lin);
			l.out = MakeScaled(G::kHidden, (uint64_t)S::kRankHeads * G::kValueDim, lin);
			l.sinks = (const float *)pool.Fill(G::kHeads, 4u, GB_FILL_F32, 0.5f, 0.0f);
			if ( G::IndexerLayer(i) )
			{
				l.index_q = MakeScaled(S::kIndexWidth, G::kQLora, lin);
				l.index_k = MakeScaled(G::kIndexDim, G::kHidden, lin);
				l.index_k_norm = (const float *)pool.Fill(G::kIndexDim, 4u, GB_FILL_F32, 0.1f, 1.0f);
				l.index_k_bias = (const float *)pool.Fill(G::kIndexDim, 4u, GB_FILL_F32, 0.05f, 0.0f);
				l.index_weights = (const uint16_t *)pool.Fill((uint64_t)G::kIndexHeads * G::kHidden, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)G::kHidden), 0.0f);
			}
			if ( i < G::kDenseLayers )
			{
				l.ffn_gate_up = MakeScaled(2u * (uint64_t)state->ffn_rows, G::kHidden, lin);
				l.ffn_down = MakeScaled(G::kHidden, state->ffn_rows, lin);
			}
			else
			{
				l.router = (const uint16_t *)pool.Fill((uint64_t)G::kExperts * G::kHidden, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)G::kHidden), 0.0f);
				l.router_bias = (const float *)pool.Fill(G::kExperts, 4u, GB_FILL_F32, 0.05f, 0.0f);
				l.moe_w1 = MakeScaled((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden, moe);
				l.moe_w3 = MakeScaled((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden, moe);
				l.moe_w2 = MakeScaled((uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter, moe);
				l.shared_gate_up = MakeScaled(2u * (uint64_t)state->shared_rows, G::kHidden, shd);
				l.shared_down = MakeScaled(G::kHidden, state->shared_rows, shd);
			}
		}
		if ( M::Allocate(state) != SPARK_STATUS_OK )
			return 0;
		kv = (uint8_t *)pool.Alloc(kLatentRegion + kIndexRegion);
		hidden = (uint16_t *)pool.Alloc((uint64_t)GB_MAX_ROWS * G::kHidden * 2u);
		seq_of_row = (uint32_t *)pool.Alloc(GB_MAX_ROWS * 4u);
		pos_dev = (uint32_t *)pool.Alloc(GB_MAX_ROWS * 4u);
		ctx_dev = (uint32_t *)pool.Alloc(GB_SLOTS * 4u);
		pages_dev = (uint32_t *)pool.Alloc(kPool * 4u);
		fault_dev = (uint32_t *)pool.Alloc(4u);
		if ( kv == 0 || hidden == 0 || pages_dev == 0 )
			return 0;
		std::vector<uint32_t> pages(kPool);
		for ( uint64_t p = 0u; p < kPool; ++p )
			pages[p] = (uint32_t)p;
		cudaMemcpyAsync(pages_dev, pages.data(), kPool * 4u, cudaMemcpyHostToDevice, stream);
		SparkStageRunnerKv attach = {};
		attach.pool = kv;
		attach.layer_stride_bytes = kPool * S::Kv::kPageBytes;
		attach.layer_page_bytes = S::Kv::kPageBytes;
		attach.layer_count = G::kLayers;
		attach.page_table = pages_dev;
		attach.page_table_stride = GB_PAGES;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = GB_SLOTS;
		attach.context_shard.degree = 1u;
		attach.context_shard.rank = 0u;
		attach.context_shard.grain = 1u;
		attach.second_pool = kv + kLatentRegion;
		attach.second_layer_stride_bytes = kPool * S::IndexKv::kPageBytes;
		attach.second_layer_page_bytes = S::IndexKv::kPageBytes;
		attach.second_layer_count = G::kIndexLayers;
		if ( M::AttachKv(state, &attach) != SPARK_STATUS_OK )
			return 0;
		ResetKv(true);
		return cudaStreamSynchronize(stream) == cudaSuccess ? 1 : 0;
	}

	void ResetKv(bool seeded)
	{
		if ( seeded )
			GbFillKernel<<<4096u, 256u, 0, stream>>>(kv, (kLatentRegion + kIndexRegion) / 2u, GB_FILL_BF16, 777u, 0.25f, 0.0f);
		else
			cudaMemsetAsync(kv, 0, kLatentRegion + kIndexRegion, stream);
	}

	int Run(const GbRow *rows, uint32_t count, uint32_t ctx0, uint32_t ctx1, uint32_t commit, uint32_t last_rows_only, uint32_t gather_ctx,
		std::vector<uint16_t> *out, float *median_ms = 0)
	{
		std::vector<uint16_t> h_in = GbHiddenHost(rows, count, G::kHidden);
		std::vector<uint32_t> h_seq(count), h_pos(count);
		uint32_t h_ctx[GB_SLOTS] = { ctx0, ctx1 };
		uint32_t seq_mask = 0u;
		for ( uint32_t r = 0u; r < count; ++r )
		{
			h_seq[r] = rows[r].seq;
			h_pos[r] = rows[r].pos;
			seq_mask |= 1u << rows[r].seq;
		}
		const uint32_t seqs = (seq_mask == 3u) ? 2u : 1u;
		cudaMemcpyAsync(seq_of_row, h_seq.data(), count * 4u, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(pos_dev, h_pos.data(), count * 4u, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(ctx_dev, h_ctx, sizeof(h_ctx), cudaMemcpyHostToDevice, stream);
		cudaMemsetAsync(fault_dev, 0, 4u, stream);
		SparkStageRunnerStep step = {};
		step.rows = count;
		step.sequences = seqs;
		step.hidden_bf16 = hidden;
		step.sequence_of_row = seq_of_row;
		step.positions = pos_dev;
		step.context_length = ctx_dev;
		step.route_fault = fault_dev;
		step.commit = commit;
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_ctx;
		step.context = GB_PAGES * G::kPageSlots;
		step.multiprocessors = sms;
		if ( median_ms != 0 )
		{
			cudaEvent_t t0, t1;
			cudaEventCreate(&t0);
			cudaEventCreate(&t1);
			for ( int w = 0; w < 3; ++w )
			{
				cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
				if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
					return 0;
			}
			cudaStreamSynchronize(stream);
			std::vector<float> times;
			for ( int it = 0; it < 9; ++it )
			{
				cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
				cudaEventRecord(t0, stream);
				if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
					return 0;
				cudaEventRecord(t1, stream);
				cudaEventSynchronize(t1);
				float ms = 0.0f;
				cudaEventElapsedTime(&ms, t0, t1);
				times.push_back(ms);
			}
			cudaEventDestroy(t0);
			cudaEventDestroy(t1);
			std::sort(times.begin(), times.end());
			*median_ms = times[times.size() / 2u];
		}
		else
		{
			cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
			if ( M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
				return 0;
		}
		if ( out != 0 )
		{
			out->resize((uint64_t)count * G::kHidden);
			if ( cudaMemcpy(out->data(), hidden, out->size() * 2u, cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError error = {}, zero = {};
		return cudaMemcpy(&error, state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(&error, &zero, sizeof(error)) == 0 ? 1u : 0u;
	}

	void Close(void)
	{
		if ( state != 0 )
			M::Close(state);
		state = 0;
		pool.Release();
		if ( stream != 0 )
			cudaStreamDestroy(stream);
	}
};

template<class Base>
struct GbMimoGeom : Base
{
	static constexpr uint32_t kLayers = 3u;
	static constexpr uint32_t kFullLayers = 2u;
	static bool FullLayer(uint32_t layer) { return layer == 0u || layer == 2u; }
	static const char *Tag(void) { return Base::Tag(); }
};

template<class Base>
struct GbMimoRig
{
	using G = GbMimoGeom<Base>;
	using M = MimoModel<G>;
	using S = MimoShape<G>;
	using State = MimoState<G>;
	static constexpr bool kSingleRankPrefixExact = false;
	static constexpr uint64_t kPool = (uint64_t)GB_SLOTS * GB_PAGES;
	static constexpr uint64_t kSlidingStride = kPool * (S::SlidingKv::kPageBytes / G::kTpDegree);
	static constexpr uint64_t kFullStride = kPool * (S::FullKv::kPageBytes / G::kTpDegree);
	static constexpr uint64_t kSlidingRegion = (uint64_t)S::kSlidingLayers * kSlidingStride;
	static constexpr uint64_t kFullRegion = (uint64_t)G::kFullLayers * kFullStride;

	GbPool pool;
	SparkStageRunnerServices services;
	State *state;
	cudaStream_t stream;
	uint8_t *kv;
	uint16_t *hidden;
	uint32_t *seq_of_row;
	uint32_t *pos_dev;
	uint32_t *ctx_dev;
	uint32_t *pages_dev;
	uint32_t *fault_dev;
	uint32_t sms;

	GbMimoRig(void) : services{}, state(0), stream(0), kv(0), hidden(0), seq_of_row(0), pos_dev(0), ctx_dev(0), pages_dev(0), fault_dev(0), sms(48u) {}

	static SparkStatus MockRound(void *context, void *stream_void, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		(void)context;
		if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
			return SPARK_STATUS_UNSUPPORTED;
		const uint64_t slice = (uint64_t)rows * (row_elements != 0u ? row_elements : G::kHidden) * sizeof(uint16_t);
		if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		{
			if ( slice != 0u )
				GbAllGatherKernel<<<256u, 256u, 0, (cudaStream_t)stream_void>>>((const uint8_t *)local, (uint8_t *)full, slice, G::kTpDegree);
		}
		else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		{
			if ( slice != 0u && local != full )
				GbCopyBytesKernel<<<256u, 256u, 0, (cudaStream_t)stream_void>>>((const uint8_t *)local, (uint8_t *)full, slice * G::kTpDegree);
		}
		else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			if ( slice != 0u && local != full )
				GbCopyBytesKernel<<<256u, 256u, 0, (cudaStream_t)stream_void>>>((const uint8_t *)local, (uint8_t *)full, slice);
		}
		return SPARK_STATUS_OK;
	}

	MimoScaled MakeFp8(uint64_t rows, uint64_t cols)
	{
		MimoScaled w = {};
		w.weight = (const uint8_t *)pool.Fill(rows * cols, 1u, GB_FILL_FP8, 1.0f, 0.0f);
		w.scale = pool.Fill(rows * (cols / MM_FP8_GROUP), 4u, GB_FILL_F32, 0.25f / sqrtf((float)cols), 1.0f / sqrtf((float)cols));
		return w;
	}

	MimoScaled MakeMxfp4(uint64_t rows, uint64_t cols)
	{
		MimoScaled w = {};
		const int exp = 127 - (int)ceilf(log2f(sqrtf((float)cols)));
		w.weight = (const uint8_t *)pool.Fill((uint64_t)G::kLocalExperts * rows * (cols / 2u), 1u, GB_FILL_E2M1, 1.0f, 0.0f);
		w.scale = pool.Fill((uint64_t)G::kLocalExperts * rows * (cols / 32u), 1u, GB_FILL_E8M0, 1.0f, (float)(exp - 1));
		return w;
	}

	int Open(void)
	{
		cudaDeviceProp prop = {};
		if ( cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0 )
			sms = (uint32_t)prop.multiProcessorCount;
		if ( cudaStreamCreate(&stream) != cudaSuccess )
			return 0;
		pool.stream = stream;
		services.context = 0;
		services.round = MockRound;
		state = (State *)calloc(1u, sizeof(State));
		if ( state == 0 )
			return 0;
		state->services = &services;
		state->tp_degree = G::kTpDegree;
		state->tp_rank = 0u;
		state->slots = GB_SLOTS;
		state->max_rows = GB_MAX_ROWS;
		state->expert_first = 0u;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t i = 0u; i < G::kLayers; ++i )
		{
			MimoLayer &l = state->layers[i];
			const bool is_full = G::FullLayer(i);
			const uint32_t qkv_rows = (is_full ? S::kFullFused : S::kSlidingFused) / G::kTpDegree;
			l.input_norm = (const uint16_t *)pool.Fill(G::kHidden, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.post_attn_norm = (const uint16_t *)pool.Fill(G::kHidden, 2u, GB_FILL_BF16, 0.1f, 1.0f);
			l.qkv = MakeFp8(qkv_rows, G::kHidden);
			l.out = (const uint16_t *)pool.Fill((uint64_t)G::kHidden * S::kRankWidth, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)S::kRankWidth), 0.0f);
			l.sinks = !is_full ? (const float *)pool.Fill(G::kHeads, 4u, GB_FILL_F32, 0.5f, 0.0f) : 0;
			if ( i < G::kDenseLayers )
			{
				l.ffn_gate_up = MakeFp8(2u * (uint64_t)S::kDenseRows, G::kHidden);
				l.ffn_down = MakeFp8(G::kHidden, S::kDenseRows);
			}
			else
			{
				l.router = (const uint16_t *)pool.Fill((uint64_t)G::kExperts * G::kHidden, 2u, GB_FILL_BF16, 1.0f / sqrtf((float)G::kHidden), 0.0f);
				l.router_bias = (const float *)pool.Fill(G::kExperts, 4u, GB_FILL_F32, 0.05f, 0.0f);
				l.moe_w1 = MakeMxfp4(G::kMoeInter, G::kHidden);
				l.moe_w3 = MakeMxfp4(G::kMoeInter, G::kHidden);
				l.moe_w2 = MakeMxfp4(G::kHidden, G::kMoeInter);
			}
		}
		if ( M::Allocate(state) != SPARK_STATUS_OK )
			return 0;
		kv = (uint8_t *)pool.Alloc(kSlidingRegion + kFullRegion);
		hidden = (uint16_t *)pool.Alloc((uint64_t)GB_MAX_ROWS * G::kHidden * 2u);
		seq_of_row = (uint32_t *)pool.Alloc(GB_MAX_ROWS * 4u);
		pos_dev = (uint32_t *)pool.Alloc(GB_MAX_ROWS * 4u);
		ctx_dev = (uint32_t *)pool.Alloc(GB_SLOTS * 4u);
		pages_dev = (uint32_t *)pool.Alloc(kPool * 4u);
		fault_dev = (uint32_t *)pool.Alloc(4u);
		if ( kv == 0 || hidden == 0 || pages_dev == 0 )
			return 0;
		std::vector<uint32_t> pages(kPool);
		for ( uint64_t p = 0u; p < kPool; ++p )
			pages[p] = (uint32_t)p;
		cudaMemcpyAsync(pages_dev, pages.data(), kPool * 4u, cudaMemcpyHostToDevice, stream);
		SparkStageRunnerKv attach = {};
		attach.pool = kv;
		attach.layer_stride_bytes = kSlidingStride;
		attach.layer_page_bytes = S::SlidingKv::kPageBytes;
		attach.layer_count = S::kSlidingLayers;
		attach.page_table = pages_dev;
		attach.page_table_stride = GB_PAGES;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = GB_SLOTS;
		attach.context_shard.degree = G::kTpDegree;
		attach.context_shard.rank = 0u;
		attach.context_shard.grain = 1u;
		attach.second_pool = kv + kSlidingRegion;
		attach.second_layer_stride_bytes = kFullStride;
		attach.second_layer_page_bytes = S::FullKv::kPageBytes;
		attach.second_layer_count = G::kFullLayers;
		if ( M::AttachKv(state, &attach) != SPARK_STATUS_OK )
			return 0;
		ResetKv(true);
		return cudaStreamSynchronize(stream) == cudaSuccess ? 1 : 0;
	}

	void ResetKv(bool seeded)
	{
		if ( seeded )
			GbFillKernel<<<4096u, 256u, 0, stream>>>(kv, (kSlidingRegion + kFullRegion) / 2u, GB_FILL_BF16, 991u, 0.25f, 0.0f);
		else
			cudaMemsetAsync(kv, 0, kSlidingRegion + kFullRegion, stream);
	}

	int Run(const GbRow *rows, uint32_t count, uint32_t ctx0, uint32_t ctx1, uint32_t commit, uint32_t last_rows_only, uint32_t gather_ctx,
		std::vector<uint16_t> *out, float *median_ms = 0)
	{
		std::vector<uint16_t> h_in = GbHiddenHost(rows, count, G::kHidden);
		std::vector<uint32_t> h_seq(count), h_pos(count);
		uint32_t h_ctx[GB_SLOTS] = { ctx0, ctx1 };
		uint32_t seq_mask = 0u;
		for ( uint32_t r = 0u; r < count; ++r )
		{
			h_seq[r] = rows[r].seq;
			h_pos[r] = rows[r].pos;
			seq_mask |= 1u << rows[r].seq;
		}
		const uint32_t seqs = (seq_mask == 3u) ? 2u : 1u;
		cudaMemcpyAsync(seq_of_row, h_seq.data(), count * 4u, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(pos_dev, h_pos.data(), count * 4u, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(ctx_dev, h_ctx, sizeof(h_ctx), cudaMemcpyHostToDevice, stream);
		cudaMemsetAsync(fault_dev, 0, 4u, stream);
		SparkStageRunnerStep step = {};
		step.rows = count;
		step.sequences = seqs;
		step.hidden_bf16 = hidden;
		step.sequence_of_row = seq_of_row;
		step.positions = pos_dev;
		step.context_length = ctx_dev;
		step.route_fault = fault_dev;
		step.commit = commit;
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_ctx;
		step.context = GB_PAGES * G::kPageSlots;
		step.multiprocessors = sms;
		if ( median_ms != 0 )
		{
			cudaEvent_t t0, t1;
			cudaEventCreate(&t0);
			cudaEventCreate(&t1);
			for ( int w = 0; w < 3; ++w )
			{
				cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
				if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
					return 0;
			}
			cudaStreamSynchronize(stream);
			std::vector<float> times;
			for ( int it = 0; it < 9; ++it )
			{
				cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
				cudaEventRecord(t0, stream);
				if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
					return 0;
				cudaEventRecord(t1, stream);
				cudaEventSynchronize(t1);
				float ms = 0.0f;
				cudaEventElapsedTime(&ms, t0, t1);
				times.push_back(ms);
			}
			cudaEventDestroy(t0);
			cudaEventDestroy(t1);
			std::sort(times.begin(), times.end());
			*median_ms = times[times.size() / 2u];
		}
		else
		{
			cudaMemcpyAsync(hidden, h_in.data(), h_in.size() * 2u, cudaMemcpyHostToDevice, stream);
			if ( M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
				return 0;
		}
		if ( out != 0 )
		{
			out->resize((uint64_t)count * G::kHidden);
			if ( cudaMemcpy(out->data(), hidden, out->size() * 2u, cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError error = {}, zero = {};
		return cudaMemcpy(&error, state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(&error, &zero, sizeof(error)) == 0 ? 1u : 0u;
	}

	void Close(void)
	{
		if ( state != 0 )
			M::Close(state);
		state = 0;
		pool.Release();
		if ( stream != 0 )
			cudaStreamDestroy(stream);
	}
};

template<class Rig>
static void VerifyAndBench(const char *name, uint32_t prefix_cached, uint32_t prod_layers)
{
	using G = typename Rig::G;
	Rig rig;
	GbCheck(rig.Open() != 0, name, "rig opens", 0u, 0u);
	if ( rig.state == 0 )
		return;
	const uint32_t ctxs[2] = { GB_SHORT_CTX, GB_LONG_CTX };
	for ( uint32_t ci = 0u; ci < 2u; ++ci )
	{
		const uint32_t ctx = ctxs[ci];
		GbRow b1_row = { 0u, ctx, 5u };
		std::vector<uint16_t> b1_out, b4_out, b8_out;
		GbCheck(rig.Run(&b1_row, 1u, ctx + 1u, GB_SHORT_CTX, 1u, 0u, 0u, &b1_out) != 0, name, "B1 runs", ctx, 0u);
		GbCheck(GbFinite(b1_out) != 0u, name, "B1 output finite", ctx, 0u);
		std::vector<GbRow> batch(8u);
		batch[0] = b1_row;
		for ( uint32_t r = 1u; r < 8u; ++r )
			batch[r] = { 1u, r, 20u + r };
		GbCheck(rig.Run(batch.data(), 4u, ctx + 1u, GB_SHORT_CTX, 1u, 0u, 0u, &b4_out) != 0, name, "B4 runs", ctx, 4u);
		GbCheck(memcmp(b1_out.data(), b4_out.data(), G::kHidden * 2u) == 0, name, "0-ULP B1==B4", ctx, 4u);
		GbCheck(rig.Run(batch.data(), 8u, ctx + 1u, GB_SHORT_CTX, 1u, 0u, 0u, &b8_out) != 0, name, "B8 runs", ctx, 8u);
		GbCheck(memcmp(b1_out.data(), b8_out.data(), G::kHidden * 2u) == 0, name, "0-ULP B1==B8", ctx, 8u);
		const uint32_t vframes[] = { 1u, 4u, 8u, 16u, 40u, 64u };
		for ( uint32_t vf : vframes )
		{
			std::vector<GbRow> vr(vf);
			vr[0] = b1_row;
			for ( uint32_t r = 1u; r < vf; ++r )
				vr[r] = { 1u, r, 50u + r };
			std::vector<uint16_t> vout;
			GbCheck(rig.Run(vr.data(), vf, ctx + 1u, GB_SHORT_CTX, 0u, 0u, 0u, &vout) != 0, name, "verify runs", ctx, vf);
			GbCheck(memcmp(b1_out.data(), vout.data(), G::kHidden * 2u) == 0, name, "0-ULP B1==verify lane0", ctx, vf);
		}
	}
	{
		const uint32_t deltas[] = { 1u, 8u, 16u, 64u, 128u };
		std::vector<GbRow> all_rows(prefix_cached + 128u);
		for ( uint32_t p = 0u; p < prefix_cached + 128u; ++p )
			all_rows[p] = { 0u, p, 0u };
		for ( uint32_t delta : deltas )
		{
			const uint32_t total = prefix_cached + delta;
			std::vector<uint16_t> cold_out, warm_out;
			rig.ResetKv(false);
			GbCheck(rig.Run(all_rows.data(), total, total, 0u, 1u, 1u, total, &cold_out) != 0, name, "cold prefill runs", total, delta);
			GbCheck(GbFinite(cold_out) != 0u, name, "cold prefill finite", total, delta);
			std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)prefix_cached * G::kHidden), cold_out.end());
			rig.ResetKv(false);
			GbCheck(rig.Run(all_rows.data(), prefix_cached, prefix_cached, 0u, 1u, 1u, prefix_cached, 0) != 0, name, "prefix warm-up runs",
				prefix_cached, delta);
			GbCheck(rig.Run(all_rows.data() + prefix_cached, delta, total, 0u, 1u, 1u, total, &warm_out) != 0, name, "warm delta runs",
				prefix_cached, delta);
			GbCheck(GbFinite(warm_out) != 0u, name, "warm delta finite", total, delta);
			if constexpr ( Rig::kSingleRankPrefixExact )
				GbCheck(GbFirstDiff(warm_out, cold_tail, G::kHidden) == UINT32_MAX, name, "0-ULP warm==cold prefill", delta,
					GbFirstDiff(warm_out, cold_tail, G::kHidden));
		}
	}
	rig.ResetKv(true);
	float ms_b1s = 0.0f, ms_b1l = 0.0f, ms_v8 = 0.0f, ms_v64 = 0.0f, ms_b8 = 0.0f, ms_b64 = 0.0f, ms_pf_c = 0.0f, ms_pf_w = 0.0f, ms_pf_s = 0.0f;
	std::vector<GbRow> r64(64u), r512_c(GB_MAX_ROWS), r512_w(GB_MAX_ROWS), r512_s(GB_MAX_ROWS);
	for ( uint32_t i = 0u; i < 64u; ++i )
		r64[i] = { i & 1u, (i & 1u) == 0u ? (GB_LONG_CTX + (i >> 1u)) : (GB_SHORT_CTX + (i >> 1u)), i };
	for ( uint32_t i = 0u; i < GB_MAX_ROWS; ++i )
	{
		r512_c[i] = { 0u, i, 0u };
		r512_w[i] = { 0u, GB_WARM_CTX + i, 0u };
		r512_s[i] = { 0u, GB_SPARSE_CTX + i, 0u };
	}
	GbRow b1s = { 0u, GB_SHORT_CTX, 1u }, b1l = { 0u, GB_LONG_CTX, 2u };
	GbCheck(rig.Run(&b1s, 1u, GB_SHORT_CTX + 1u, GB_SHORT_CTX, 1u, 0u, 0u, 0, &ms_b1s) != 0, name, "bench B1s", 0u, 0u);
	GbCheck(rig.Run(&b1l, 1u, GB_LONG_CTX + 1u, GB_SHORT_CTX, 1u, 0u, 0u, 0, &ms_b1l) != 0, name, "bench B1l", 0u, 0u);
	GbCheck(rig.Run(r64.data(), 8u, GB_LONG_CTX + 32u, GB_SHORT_CTX + 32u, 0u, 0u, 0u, 0, &ms_v8) != 0, name, "bench ver8", 0u, 0u);
	GbCheck(rig.Run(r64.data(), 64u, GB_LONG_CTX + 32u, GB_SHORT_CTX + 32u, 0u, 0u, 0u, 0, &ms_v64) != 0, name, "bench ver64", 0u, 0u);
	GbCheck(rig.Run(r64.data(), 8u, GB_LONG_CTX + 32u, GB_SHORT_CTX + 32u, 1u, 0u, 0u, 0, &ms_b8) != 0, name, "bench B8", 0u, 0u);
	GbCheck(rig.Run(r64.data(), 64u, GB_LONG_CTX + 32u, GB_SHORT_CTX + 32u, 1u, 0u, 0u, 0, &ms_b64) != 0, name, "bench B64", 0u, 0u);
	GbCheck(rig.Run(r512_c.data(), GB_MAX_ROWS, GB_MAX_ROWS, 0u, 1u, 1u, GB_MAX_ROWS, 0, &ms_pf_c) != 0, name, "bench PF512 cold", 0u, 0u);
	GbCheck(rig.Run(r512_w.data(), GB_MAX_ROWS, GB_WARM_CTX + GB_MAX_ROWS, 0u, 1u, 1u, GB_WARM_CTX + GB_MAX_ROWS, 0, &ms_pf_w) != 0, name,
		"bench PF512 warm", 0u, 0u);
	GbCheck(rig.Run(r512_s.data(), GB_MAX_ROWS, GB_SPARSE_CTX + GB_MAX_ROWS, 0u, 1u, 1u, GB_SPARSE_CTX + GB_MAX_ROWS, 0, &ms_pf_s) != 0, name,
		"bench PF512 sparse", 0u, 0u);
	GbCheck(rig.KvClean() != 0u, name, "no KV access errors", 0u, 0u);
	const float scale = (float)prod_layers / (float)G::kLayers;
	printf("BENCH %-16s (3L->%uL scaled): B1s=%.2fms (%.1f tok/s) B1l=%.2fms (%.1f tok/s) ver8=%.2fms ver64=%.2fms B8=%.2fms (%.1f tok/s) B64=%.2fms (%.1f tok/s) PF512c=%.2fms (%.0f tok/s) PF512w=%.2fms (%.0f tok/s) PF512s=%.2fms (%.0f tok/s)\n",
		name, prod_layers,
		ms_b1s * scale, 1000.0f / (ms_b1s * scale),
		ms_b1l * scale, 1000.0f / (ms_b1l * scale),
		ms_v8 * scale, ms_v64 * scale,
		ms_b8 * scale, 8000.0f / (ms_b8 * scale),
		ms_b64 * scale, 64000.0f / (ms_b64 * scale),
		ms_pf_c * scale, (float)GB_MAX_ROWS * 1000.0f / (ms_pf_c * scale),
		ms_pf_w * scale, (float)GB_MAX_ROWS * 1000.0f / (ms_pf_w * scale),
		ms_pf_s * scale, (float)GB_MAX_ROWS * 1000.0f / (ms_pf_s * scale));
	rig.Close();
}

int main(void)
{
	VerifyAndBench<GbHyRig<GbHyPreviewCfg>>("hy4-preview", 128u, 78u);
	VerifyAndBench<GbHyRig<GbHyFp8Cfg>>("hy4-fp8", 128u, 78u);
	VerifyAndBench<GbMimoRig<Mimo26ProGeometry<SPARK_MIMO26_PRO_TP16_STAGE_TP_DEGREE>>>("mimo-v2.6-pro", 64u, 70u);
	VerifyAndBench<GbMimoRig<Mimo26FlashGeometry<SPARK_MIMO26_FLASH_TP16_STAGE_TP_DEGREE>>>("mimo-v2.6-flash", 64u, 48u);
	if ( gb_failures != 0 )
	{
		fprintf(stderr, "FAIL mimo26+hy4 gb10 bench+invariance: %d failures\n", gb_failures);
		return 1;
	}
	printf("PASS mimo26+hy4 gb10 bench+invariance: all 4 geometries 0-ULP invariant and benchmarked (max_rows=%u)\n", GB_MAX_ROWS);
	return 0;
}
