
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <vector>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

struct LmHostRecorderFormat
{
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 15));
	}
};
#include "inference/llms/kimi_k3/layer.cuh"

#define ROWS 2u
#define ROUTES (ROWS * K3_TOP_K)

static uint16_t hidden[ROWS * K3_HIDDEN];
static uint16_t normed[ROWS * K3_HIDDEN], attention_out[ROWS * K3_HIDDEN];
static uint16_t shared_out[ROWS * K3_HIDDEN];
static uint16_t latent[ROUTES * K3_ROUTED_EXPERT_HIDDEN];
static uint16_t gate_up[ROUTES * K3_SHARED_INTERMEDIATE * 2u];
static uint16_t intermediate[ROUTES * K3_SHARED_INTERMEDIATE];
static uint16_t norm_weight[K3_HIDDEN];
static float router_logits[ROWS * K3_EXPERTS], router_bias[K3_EXPERTS];
static float route_weight[ROUTES];
static uint32_t route_expert[ROUTES], route_packed[ROUTES], route_source[ROUTES];
static uint32_t group_offsets[K3_EXPERTS + 1u], group_tiles[K3_EXPERTS + 1u];
static uint32_t group_tiles_down[K3_EXPERTS + 1u];
static uint32_t dense_offsets[2], dense_tiles[2];

int main(void)
{
	static K3LayerBuffers b;
	uint32_t index;
	memset(&b, 0, sizeof(b));
	for (index = 0u; index < K3_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < ROWS * K3_HIDDEN; ++index)
		hidden[index] = LmFloatToBf16(0.01f * (float)(index % 17));
	for (index = 0u; index < ROUTES; ++index)
	{
		route_weight[index] = 1.0f / (float)K3_TOP_K;
	}
	b.hidden_bf16 = hidden; b.normed_bf16 = normed;
	b.attention_out_bf16 = attention_out; b.shared_out_bf16 = shared_out;
	b.latent_bf16 = latent; b.gate_up_bf16 = gate_up;
	b.intermediate_bf16 = intermediate;
	b.mlp_norm_weight = norm_weight; b.routed_norm_weight = norm_weight;
	b.router_logits = router_logits; b.router_bias = router_bias;
	b.route_expert = route_expert; b.route_weight = route_weight;
	b.route_packed_row = route_packed; b.route_source_token = route_source;
	b.group_row_offset = group_offsets; b.group_tile_prefix_w1 = group_tiles;
	b.group_tile_prefix_w2 = group_tiles_down;
	dense_offsets[0] = 0u; dense_offsets[1] = ROWS;
	b.dense_row_offset = dense_offsets; b.dense_tile_prefix = dense_tiles;

	b.expert_interleave = 0u;

	printf("start\n"); fflush(stdout);
	K3LayerLatentMoe<LmHostRecorderFormat>(&b, ROWS, ROUTES, 1u, 0, 0u);
	K3LayerLatentMoe<LmHostRecorderFormat>(&b, ROWS, ROUTES, 1u, 0, 1u);
	printf("survived\n"); fflush(stdout);

	printf("gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (size_t i = 0u; i < lm_recorded_gemms.size(); ++i)
	{
		const LmRecordedGemm &g = lm_recorded_gemms[i];
		const char *name = g.output == (void *)hidden ? "hidden"
			: g.output == (void *)shared_out ? "shared_out"
			: g.output == (void *)latent ? "latent"
			: g.output == (void *)gate_up ? "gate_up"
			: g.output == (void *)router_logits ? "router_logits" : "other";
		printf("gemm %zu dest %s in %u out %u rows %u grouped %d ind %d\n",
			i, name, g.input_dimension, g.output_dimension,
			g.packed_rows, g.grouped ? 1 : 0, g.indirect ? 1 : 0);
	}
	printf("hidden[0] %.6f\n", (double)LmBf16ToFloat(hidden[0]));
	printf("shared_out[0] %.6f\n", (double)LmBf16ToFloat(shared_out[0]));
	return 0;
}
