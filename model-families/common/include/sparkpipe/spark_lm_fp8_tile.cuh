#ifndef SPARK_LM_FP8_TILE_CUH
#define SPARK_LM_FP8_TILE_CUH

// GB10 FP8 tensor tile for the shared linear/expert GEMM. This path is
// opt-in and remains NOT_MEASURED until its fragment mapping and numerical
// output are qualified on the exact sm_121a package. The BF16 tile remains
// the independently testable reference implementation, not an implicit
// production fallback.
//
// WHY. The bf16 tile decodes the stored FP8 E4M3 weights UP into bf16, then
// runs 16x16x16 bf16 wmma. That wastes the tensor cores twice: bf16 tensor
// ops run at half the FP8 rate, and the shipped weights were already FP8, so
// the up-decode adds no precision. glm52 measured that path at 6.5 TFLOP/s,
// ~2.6% of the FP8 peak (docs/GLM52_B256_PER_TOKEN_KERNELS_20260704,
// docs/GB10_CUDA_COST_MODEL_CALIBRATION.md). This tile keeps the weights in
// FP8, quantizes the activation tile to E4M3 with a per-row absmax scale,
// multiplies on the FP8 tensor path (mma.sync.m16n8k32), and accumulates in
// FP32.
//
// ACCURACY. No WEIGHT precision is lost - the weights are already FP8, and
// FP32 accumulate is preserved. The only new approximation is activation
// E4M3 quantization with a per-row scale, exactly what glm52's proven FP8
// scaled-GEMM path already does (SiluMulFp8E4m3Quantize). This is NOT the FP4
// WEIGHT quantization that loses accuracy. If a driver's activation
// quantization is unacceptable on the ring, its caller stays on the bf16
// tile - this path is opt-in.
//
// VALIDATION STATE - READ BEFORE ENABLING. The source targets the
// mma.sync.m16n8k32.f32.e4m3.e4m3.f32 instruction for sm_121a. This source
// snapshot has not been compiled with CUDA 13 in the current environment. The
// per-thread register-to-matrix-element mapping the instruction requires
// (SparkLmFp8LoadFragA / SparkLmFp8LoadFragB) is defined below from the PTX
// ISA m16n8k32 layout and is NOT silicon-validated. The fragment mapping is
// the one part of an mma.sync kernel that a wrong implementation renders
// silently incorrect while still assembling. This file is excluded unless
// SPARK_LM_FP8_TILE is explicitly defined, and one ring run must still confirm
// its output matches the bf16 tile on identical weights before production use.
//
// KNOWN PERFORMANCE DEFECT - READ BEFORE ENABLING. The per-row activation
// absmax is computed inside the tile body, which runs once per (M-tile,
// N-tile) CTA, so every N-tile rescans the FULL K of the same rows. The
// redundancy is the N-tile count: 212x for mimo25 qkv (K4096 N13568), 272x
// for qwen36 ffn (K5120 N17408), 64x for mimo25 o_proj, 9x for dsv4 kv. That
// is not a micro-inefficiency - it can make this path SLOWER than the bf16
// tile it is meant to replace, so enabling the flag without fixing it may
// regress rather than improve.
//
// The correct fix is to compute the activation quantization ONCE: either a
// prequantized activation workspace shared by all N-tiles, or a per-K-tile
// scale computed during staging (which also removes the full-K prescan and
// gives finer, more accurate scale granularity, at the cost of accumulating
// each K-tile's partial product into a float total outside the mma). Neither
// is done here. This is left for the FP8 validation work, where the change
// can actually be measured on silicon rather than reasoned about - changing
// an unvalidated kernel's numerics with no way to check them only stacks
// unverifiable risk.
//
// GEOMETRY IS PARAMETRIC. No model dimension is hard-coded; the tile SHAPE
// (16x8x32 FP8 fragments, K accumulated deep, per-row activation scale) is
// the durable artifact and carries unchanged to MiMo 3.1, dsv4 GA, Qwen 3.8.

#include "sparkpipe/spark_lm_async_copy.cuh"
#include <cuda_fp8.h>

// -- FP8 tensor fragment shape (PTX m16n8k32 E4M3) --
#define SPARK_LM_FP8_MMA_M 16u
#define SPARK_LM_FP8_MMA_N 8u
#define SPARK_LM_FP8_MMA_K 32u
#define SPARK_LM_FP8_TILE_N 64u                  // 8 warps x MMA_N per CTA
#define SPARK_LM_FP8_TILE_K 64u                  // 2 K-steps staged per pass
#define SPARK_LM_FP8_E4M3_MAX 448.0f

static_assert(
    128u % SPARK_LM_FP8_TILE_N == 0u,
    "FP8 output tiles must not cross an F32B128 neuron-scale boundary");
static_assert(
    128u % SPARK_LM_FP8_TILE_K == 0u,
    "FP8 K tiles must divide an F32B128 input-scale boundary");
static_assert(
    SPARK_LM_TILE_N % SPARK_LM_FP8_TILE_N == 0u,
    "FP8 subtiles must exactly cover the shared output-tile geometry");

// One mma.sync.m16n8k32, E4M3 operands, F32 accumulate. The ISA contract
// lives in exactly this one inline for review.
static __device__ __forceinline__ void SparkLmFp8Mma16x8x32(float acc[4], const unsigned a[4], const unsigned b[2])
{
	asm volatile(
		"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -- FRAGMENT MAPPING: PTX ISA contract, pending silicon verification --
// PTX m16n8k32 E4M3 thread layout (ISA .target sm_121a family): lane L holds,
// for A, rows {L/4, L/4+8} and K groups {(L%4)*4, (L%4)*4+16}; for B,
// columns L/4 and K groups {(L%4)*4, (L%4)*4+16}. Each uint32 packs four
// E4M3 elements. Register order follows the PTX element-index formulas.
// These loaders read the staged shared tiles into the packed registers per
// that layout. They are isolated so the ring fix touches ONLY these two
// functions, never the tile body or the caller.
static __device__ __forceinline__ void SparkLmFp8LoadFragA(unsigned a[4], const __nv_fp8_storage_t *tile_input, uint32_t k_step, uint32_t lane)
{
	// A tile is [MMA_M=16][TILE_K=64] row-major E4M3. Thread holds rows
	// {L/4, L/4+8}; for the 32-wide K each row contributes two uint32 at
	// K base k_step*32 + (lane%4)*4 and its +16 pair. PTX element indices
	// a0..a15 require register order {low/K0, high/K0, low/K16, high/K16}.
	const uint32_t *base = (const uint32_t *)tile_input;
	uint32_t row_lo = lane >> 2u,stride_words = SPARK_LM_FP8_TILE_K / 4u;
	uint32_t k_word0 = (k_step * (SPARK_LM_FP8_MMA_K / 4u)) + (lane & 3u);
	uint32_t k_word1 = k_word0 + 4u;
	a[0] = base[(row_lo * stride_words) + k_word0];
	a[1] = base[((row_lo + 8u) * stride_words) + k_word0];
	a[2] = base[(row_lo * stride_words) + k_word1];
	a[3] = base[((row_lo + 8u) * stride_words) + k_word1];
}

static __device__ __forceinline__ void SparkLmFp8LoadFragB(unsigned b[2], const __nv_fp8_storage_t *tile_weight, uint32_t warp, uint32_t k_step, uint32_t lane)
{
	// B tile is [TILE_N=64][TILE_K=64] row-major E4M3 (neuron-major). This
	// warp owns 8 columns from warp*MMA_N; column = warp*8 + lane/4. For the
	// 32-wide K the column contributes two uint32, b[0] at K base and b[1]
	// at +16, matching PTX element indices b0..b7.
	const uint32_t *base = (const uint32_t *)tile_weight;
	uint32_t col = (warp * SPARK_LM_FP8_MMA_N) + (lane >> 2u),stride_words = SPARK_LM_FP8_TILE_K / 4u;
	uint32_t k_word0 = (k_step * (SPARK_LM_FP8_MMA_K / 4u)) + (lane & 3u);
	b[0] = base[(col * stride_words) + k_word0];
	b[1] = base[(col * stride_words) + k_word0 + 4u];
}

// -- activation quantize to E4M3 against a per-row scale --
static __device__ __forceinline__ float SparkLmFp8RowAbsmax(const void *input_bf16, const uint32_t *input_row_map, uint32_t slot_base, uint32_t slot_count, uint32_t row_in_tile, uint32_t input_dimension, uint32_t lane)
{
	uint32_t slot = slot_base + row_in_tile,source_row,element;
	float local = 0.0f;
	float2 pair_value;
	if ( slot >= slot_count )
		return(1.0f);
	source_row = input_row_map != 0 ? input_row_map[slot] : slot;
	for (element = lane; element < (input_dimension >> 1u); element += SPARK_LM_WARP_LANES)
	{
		pair_value = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)source_row * input_dimension) >> 1u) + element);
		local = fmaxf(local,fmaxf(fabsf(pair_value.x),fabsf(pair_value.y)));
	}
	#pragma unroll
	for (element = SPARK_LM_WARP_LANES >> 1u; element > 0u; element >>= 1u)
		local = fmaxf(local,__shfl_xor_sync(0xffffffffu,local,element));
	return(local > 0.0f ? local : 1.0f);
}

// Stage MMA_M rows x TILE_K activation columns, quantized to E4M3 per row.
static __device__ void SparkLmFp8StageInput(const void *input_bf16, const uint32_t *input_row_map, const float *row_inverse_scale, uint32_t slot_base, uint32_t slot_count, uint32_t k_base, uint32_t input_dimension, __nv_fp8_storage_t *tile_input)
{
	uint32_t entry,row_in_tile,k_local,slot,source_row;
	float value,inv_scale;
	float2 pair_value;
	for (entry = threadIdx.x; entry < (SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_K) >> 1u; entry += blockDim.x)
	{
		row_in_tile = entry / (SPARK_LM_FP8_TILE_K >> 1u);
		k_local = (entry % (SPARK_LM_FP8_TILE_K >> 1u)) << 1u;
		slot = slot_base + row_in_tile;
		inv_scale = row_inverse_scale[row_in_tile];
		if ( slot < slot_count && (k_base + k_local) < input_dimension )
		{
			source_row = input_row_map != 0 ? input_row_map[slot] : slot;
			pair_value = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)source_row * input_dimension) + k_base + k_local) >> 1u);
			value = pair_value.x;
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] = __nv_cvt_float_to_fp8(value * inv_scale,__NV_SATFINITE,__NV_E4M3);
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] = __nv_cvt_float_to_fp8(pair_value.y * inv_scale,__NV_SATFINITE,__NV_E4M3);
		}
		else
		{
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] = 0;
			tile_input[(row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] = 0;
		}
	}
}

// Stage TILE_N neurons x TILE_K weight columns, native E4M3 (no decode). The
// weight is already E4M3 in weight_payload; weight_scale is the per-block
// dequant applied at accumulate, not here.
// Stage one weight tile asynchronously.
//
// The linear contract guarantees input_dimension is a multiple of
// SPARK_LM_TILE_K, and fp8 storage is one byte per element, so a row segment is
// SPARK_LM_FP8_TILE_K contiguous bytes at a 16-byte aligned offset: four
// 16-byte transfers. That also makes the old (k_base + k_local) < input_dimension
// test unconditionally true, so only the neuron bound survives, and a row past
// the end is issued with a zero source size which the hardware zero-fills. The
// per-element branch is gone.
//
// The source neuron is clamped for the address computation so a padding row
// never forms an out-of-bounds pointer, even though a zero-length transfer
// would not read it.
//
// The caller must SparkLmAsyncCommit and wait before reading tile_weight.
static __device__ void SparkLmFp8StageWeight(const void *weight_payload_fp8, uint32_t neuron_base, uint32_t k_base, uint32_t input_dimension, uint32_t output_dimension, __nv_fp8_storage_t *tile_weight)
{
	const __nv_fp8_storage_t *payload = (const __nv_fp8_storage_t *)weight_payload_fp8;
	uint32_t chunk,chunks_per_row,neuron_local,chunk_offset,neuron,source_bytes;
	chunks_per_row = SPARK_LM_FP8_TILE_K / SPARK_LM_ASYNC_COPY_WIDEST_BYTES;
	for (chunk = threadIdx.x; chunk < SPARK_LM_FP8_TILE_N * chunks_per_row; chunk += blockDim.x)
	{
		neuron_local = chunk / chunks_per_row;
		chunk_offset = (chunk % chunks_per_row) * SPARK_LM_ASYNC_COPY_WIDEST_BYTES;
		neuron = neuron_base + neuron_local;
		source_bytes = neuron < output_dimension ? SPARK_LM_ASYNC_COPY_WIDEST_BYTES : 0u;
		neuron = neuron < output_dimension ? neuron : 0u;
		SparkLmAsyncCopyBounded<SPARK_LM_ASYNC_COPY_WIDEST_BYTES, SPARK_LM_ASYNC_COPY_CACHE_GLOBAL>(
			tile_weight + (neuron_local * SPARK_LM_FP8_TILE_K) + chunk_offset,
			payload + ((uint64_t)neuron * input_dimension) + k_base + chunk_offset,
			source_bytes);
	}
}

static __device__ void SparkLmFp8StageInputProducerGroup(
    const void *input_bf16,
    const uint32_t *input_row_map,
    const float *row_inverse_scale,
    uint32_t slot_base,
    uint32_t slot_count,
    uint32_t k_base,
    uint32_t input_dimension,
    uint32_t producer_warp_base,
    __nv_fp8_storage_t *tile_input)
{
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t producer_thread_index;
    uint32_t entry;

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    if (warp_index < producer_warp_base ||
        warp_index >= producer_warp_base + 4u)
    {
        return;
    }
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    producer_thread_index =
        ((warp_index - producer_warp_base) * SPARK_LM_WARP_LANES) +
        lane_index;
    for (entry = producer_thread_index;
         entry < (SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_K) >> 1u;
         entry += 4u * SPARK_LM_WARP_LANES)
    {
        uint32_t row_in_tile;
        uint32_t k_local;
        uint32_t slot;
        float inverse_scale;

        row_in_tile = entry / (SPARK_LM_FP8_TILE_K >> 1u);
        k_local =
            (entry % (SPARK_LM_FP8_TILE_K >> 1u)) << 1u;
        slot = slot_base + row_in_tile;
        inverse_scale = row_inverse_scale[row_in_tile];
        if (slot < slot_count && k_base + k_local < input_dimension)
        {
            uint32_t source_row;
            float2 pair_value;

            source_row = input_row_map != 0 ? input_row_map[slot] : slot;
            pair_value = SparkLmLoadBf16Pair(
                input_bf16,
                (((uint64_t)source_row * input_dimension) +
                 k_base + k_local) >> 1u);
            tile_input[
                (row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] =
                __nv_cvt_float_to_fp8(
                    pair_value.x * inverse_scale,
                    __NV_SATFINITE,
                    __NV_E4M3);
            tile_input[
                (row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] =
                __nv_cvt_float_to_fp8(
                    pair_value.y * inverse_scale,
                    __NV_SATFINITE,
                    __NV_E4M3);
        }
        else
        {
            tile_input[
                (row_in_tile * SPARK_LM_FP8_TILE_K) + k_local] = 0;
            tile_input[
                (row_in_tile * SPARK_LM_FP8_TILE_K) + k_local + 1u] = 0;
        }
    }
}

// Accumulator write: mma.sync m16n8 lane L holds acc[0..3] for rows
// {L/4, L/4+8} x cols {(L%4)*2, (L%4)*2+1}. F32B128 weight scales are folded
// into the accumulator at every 128-wide K boundary. This final store applies
// only the per-row activation scale.
static __device__ __forceinline__ void SparkLmFp8StoreAccumulator(
    float tile_output[
        SPARK_LM_FP8_MMA_M][SPARK_LM_FP8_TILE_N + 8u],
    const float accumulator[4],
    const float *row_scale,
    uint32_t warp_index,
    uint32_t lane_index)
{
    uint32_t accumulator_row;
    uint32_t accumulator_column;

    accumulator_row = lane_index >> 2u;
    accumulator_column =
        (warp_index * SPARK_LM_FP8_MMA_N) +
        ((lane_index & 3u) << 1u);
    if (accumulator_column >= SPARK_LM_FP8_TILE_N)
    {
        return;
    }
    tile_output[accumulator_row][accumulator_column] =
        accumulator[0] * row_scale[accumulator_row];
    tile_output[accumulator_row][accumulator_column + 1u] =
        accumulator[1] * row_scale[accumulator_row];
    tile_output[accumulator_row + 8u][accumulator_column] =
        accumulator[2] * row_scale[accumulator_row + 8u];
    tile_output[accumulator_row + 8u][accumulator_column + 1u] =
        accumulator[3] * row_scale[accumulator_row + 8u];
}

// FP8 expert/linear tile body. Same interface contract as the bf16
// SparkLmExpertTileBody. weight_payload_fp8 is E4M3; weight_scale is the
// F32B128 dequant scales, accumulated independently for each 128-wide K
// block before the row activation scale is applied. Output is bf16.
static __device__ void SparkLmExpertTileBodyFp8(
    const void *weight_payload_fp8,
    const void *weight_scale,
    const void *input_bf16,
    const uint32_t *input_row_map,
    void *output_bf16,
    uint32_t slot_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t slot_base,
    uint32_t neuron_base)
{
    __shared__ __align__(SPARK_LM_ASYNC_COPY_WIDEST_BYTES) __nv_fp8_storage_t tile_input[2u][
        SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_K];
    __shared__ __align__(SPARK_LM_ASYNC_COPY_WIDEST_BYTES) __nv_fp8_storage_t tile_weight[3u][
        SPARK_LM_FP8_TILE_N * SPARK_LM_FP8_TILE_K];
    __shared__ float row_scale[SPARK_LM_FP8_MMA_M];
    __shared__ float row_inverse_scale[SPARK_LM_FP8_MMA_M];
    __shared__ float tile_output[
        SPARK_LM_FP8_MMA_M][SPARK_LM_FP8_TILE_N + 8u];
    float accumulator[4];
    float block_accumulator[4];
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t neuron_tile_offset;
    uint32_t neuron_tile_base;
    uint32_t current_buffer;
    uint32_t weight_stage;
    uint32_t prefetch_k_base;
    uint32_t next_buffer;
    uint32_t k_base;
    uint32_t next_k_base;
    uint32_t k_step;
    uint32_t row_in_tile;
    uint32_t neuron;
    uint32_t entry;
    uint32_t accumulator_index;
    unsigned fragment_a[4];
    unsigned fragment_b[2];

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    for (row_in_tile = warp_index;
         row_in_tile < SPARK_LM_FP8_MMA_M;
         row_in_tile += blockDim.x / SPARK_LM_WARP_LANES)
    {
        float row_absmax;

        row_absmax = SparkLmFp8RowAbsmax(
            input_bf16,
            input_row_map,
            slot_base,
            slot_count,
            row_in_tile,
            input_dimension,
            lane_index);
        if (lane_index == 0u)
        {
            row_scale[row_in_tile] =
                row_absmax / SPARK_LM_FP8_E4M3_MAX;
            row_inverse_scale[row_in_tile] =
                row_scale[row_in_tile] > 0.0f
                ? 1.0f / row_scale[row_in_tile]
                : 0.0f;
        }
    }
    __syncthreads();

    for (neuron_tile_offset = 0u;
         neuron_tile_offset < SPARK_LM_TILE_N;
         neuron_tile_offset += SPARK_LM_FP8_TILE_N)
    {
        neuron_tile_base = neuron_base + neuron_tile_offset;
        #pragma unroll
        for (accumulator_index = 0u;
             accumulator_index < 4u;
             ++accumulator_index)
        {
            accumulator[accumulator_index] = 0.0f;
            block_accumulator[accumulator_index] = 0.0f;
        }
        current_buffer = 0u;
        weight_stage = 0u;
        SparkLmFp8StageInput(
            input_bf16,
            input_row_map,
            row_inverse_scale,
            slot_base,
            slot_count,
            0u,
            input_dimension,
            tile_input[current_buffer]);
        // Prime two weight tiles. Retiring only the first leaves the second in
        // flight, so the memory system is never idle waiting to be asked - which
        // is the whole point when the weight stream dominates bandwidth.
        SparkLmFp8StageWeight(
            weight_payload_fp8,
            neuron_tile_base,
            0u,
            input_dimension,
            output_dimension,
            tile_weight[0u]);
        SparkLmAsyncCommit();
        if (SPARK_LM_FP8_TILE_K < input_dimension)
        {
            SparkLmFp8StageWeight(
                weight_payload_fp8,
                neuron_tile_base,
                SPARK_LM_FP8_TILE_K,
                input_dimension,
                output_dimension,
                tile_weight[1u]);
        }
        SparkLmAsyncCommit();
        // At most one group outstanding: stage 0 has landed, stage 1 still flies.
        SparkLmAsyncWait<1u>();
        __syncthreads();

        for (k_base = 0u;
             k_base < input_dimension;
             k_base += SPARK_LM_FP8_TILE_K)
        {
            next_k_base = k_base + SPARK_LM_FP8_TILE_K;
            next_buffer = current_buffer ^ 1u;
            // Issue the entire next weight tile here, with every thread, before
            // any mma runs. cp.async issue does not block, so the reason the
            // synchronous version split this across the two warp phases - to
            // share out the blocking loads - no longer applies. Splitting it now
            // only delays bytes: the half that used to be issued in phase two
            // had no mma left to overlap and was retired against an idle bus.
            // Issued here, the whole tile is in flight across both phases.
            // Issue two tiles ahead, into the stage neither the current mma nor
            // the already-in-flight prefetch is using.
            prefetch_k_base = next_k_base + SPARK_LM_FP8_TILE_K;
            if (prefetch_k_base < input_dimension)
            {
                SparkLmFp8StageWeight(
                    weight_payload_fp8,
                    neuron_tile_base,
                    prefetch_k_base,
                    input_dimension,
                    output_dimension,
                    tile_weight[(weight_stage + 2u) % 3u]);
            }
            SparkLmAsyncCommit();
            if (warp_index < 4u)
            {
                for (k_step = 0u;
                     k_step < SPARK_LM_FP8_TILE_K / SPARK_LM_FP8_MMA_K;
                     ++k_step)
                {
                    SparkLmFp8LoadFragA(
                        fragment_a,
                        tile_input[current_buffer],
                        k_step,
                        lane_index);
                    SparkLmFp8LoadFragB(
                        fragment_b,
                        tile_weight[weight_stage],
                        warp_index,
                        k_step,
                        lane_index);
                    SparkLmFp8Mma16x8x32(
                        block_accumulator,
                        fragment_a,
                        fragment_b);
                }
            }
            else if (next_k_base < input_dimension)
            {
                SparkLmFp8StageInputProducerGroup(
                    input_bf16,
                    input_row_map,
                    row_inverse_scale,
                    slot_base,
                    slot_count,
                    next_k_base,
                    input_dimension,
                    4u,
                    tile_input[next_buffer]);
            }
            __syncthreads();

            if (warp_index >= 4u)
            {
                for (k_step = 0u;
                     k_step < SPARK_LM_FP8_TILE_K / SPARK_LM_FP8_MMA_K;
                     ++k_step)
                {
                    SparkLmFp8LoadFragA(
                        fragment_a,
                        tile_input[current_buffer],
                        k_step,
                        lane_index);
                    SparkLmFp8LoadFragB(
                        fragment_b,
                        tile_weight[weight_stage],
                        warp_index,
                        k_step,
                        lane_index);
                    SparkLmFp8Mma16x8x32(
                        block_accumulator,
                        fragment_a,
                        fragment_b);
                }
            }
            // Retire exactly one group rather than draining. That leaves the
            // tile issued this iteration still in flight while the next
            // iteration computes on the one that just landed, so the weight
            // stream never stops. Draining here - which cp.async.wait_all does -
            // idles the bus at every iteration boundary.
            // __syncthreads does not wait on cp.async, hence the explicit retire.
            SparkLmAsyncWait<1u>();
            __syncthreads();
            if ((next_k_base % 128u) == 0u ||
                next_k_base >= input_dimension)
            {
                uint64_t scale_index;
                float block_scale;

                scale_index =
                    ((uint64_t)(neuron_tile_base / 128u) *
                     (uint64_t)(input_dimension / 128u)) +
                    (uint64_t)(k_base / 128u);
                block_scale = __ldg(
                    reinterpret_cast<const float *>(weight_scale) +
                    scale_index);
                #pragma unroll
                for (accumulator_index = 0u;
                     accumulator_index < 4u;
                     ++accumulator_index)
                {
                    accumulator[accumulator_index] = fmaf(
                        block_accumulator[accumulator_index],
                        block_scale,
                        accumulator[accumulator_index]);
                    block_accumulator[accumulator_index] = 0.0f;
                }
            }
            current_buffer = next_buffer;
            weight_stage = (weight_stage + 1u) % 3u;
        }

        SparkLmFp8StoreAccumulator(
            tile_output,
            accumulator,
            row_scale,
            warp_index,
            lane_index);
        __syncthreads();
        for (entry = threadIdx.x;
             entry < SPARK_LM_FP8_MMA_M * SPARK_LM_FP8_TILE_N;
             entry += blockDim.x)
        {
            row_in_tile = entry / SPARK_LM_FP8_TILE_N;
            neuron =
                neuron_tile_base + (entry % SPARK_LM_FP8_TILE_N);
            if (slot_base + row_in_tile < slot_count &&
                neuron < output_dimension)
            {
                SparkLmFloatToBf16(
                    output_bf16,
                    ((uint64_t)(slot_base + row_in_tile) *
                     output_dimension) +
                        neuron,
                    tile_output[
                        row_in_tile][entry % SPARK_LM_FP8_TILE_N]);
            }
        }
        __syncthreads();
    }
}

#endif
