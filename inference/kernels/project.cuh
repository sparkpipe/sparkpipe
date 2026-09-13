#pragma once

// Projection primitives. Low-rank with an intermediate norm, and its absorbed
// form.
//
// This is a separate module from attention because the pattern outlives any one
// model's attention. A projection that goes down to a low rank, normalises
// there, and comes back up is how every MLA-family model compresses its query
// and KV paths - the V2-through-V4 lineage, the GLM class, the KDA class - and the only thing
// that differs between them is the ranks.
//
// TWO FORMS OF THE SAME PROJECTION, AND WHY BOTH EXIST.
//
// RAW: hidden -> down -> norm -> up -> per-head keys and values. This is what
// the checkpoint contains and what the arithmetic says.
//
// ABSORBED: the up-projection is folded into the query and output weights at
// pack time, so attention happens directly in the compressed space and per-head
// K and V are never materialised. Four plain linears replace the two-stage path.
//
// The absorbed form is strictly better at decode and strictly worse at prefill,
// which is why a model ships both sets of weights rather than choosing. At
// decode one row attends over a whole cache, so not materialising per-head K and
// V saves the dominant read. At prefill many rows share the cache, the
// materialisation amortises, and the raw form's smaller GEMMs win.
//
// A model that ships only raw weights uses LmLowRankProject. One that ships
// absorbed weights uses four LmGemmLaunch calls and needs nothing from this
// file. GLM 5.2 ships both and selects at bind time, which is a property of the
// checkpoint rather than a runtime mode.

#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include <stdint.h>

// One side of a low-rank projection: the two weights and the norm between them.
//
// Quantisation is per-weight rather than per-projection because the down and up
// matrices have very different shapes - hidden-by-rank against rank-by-output -
// and a rank of 2048 against a hidden of 6144 means the down matrix is three
// times the size. They earn different formats.
struct LmLowRankWeights
{
	const void *down_weight;
	const void *down_scale;
	const void *norm_weight;
	const void *up_weight;
	const void *up_scale;
	uint32_t input_dimension;
	uint32_t rank;
	uint32_t output_dimension;
	float norm_epsilon;
};

// Scratch the projection needs: the compressed rows, and their quantised form.
//
// Sized by rank rather than by output, which is the point of the compression -
// at GLM 5.2's 6144 hidden and 2048 rank this is a third of what a fused
// projection would need, and the second GEMM reads it instead of the hidden.
struct LmLowRankScratch
{
	uint8_t *input_codes;
	uint8_t *input_scales;
	uint16_t *compressed_bf16;
	uint8_t *compressed_codes;
	uint8_t *compressed_scales;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
};

// hidden -> down -> norm -> up.
//
// The norm is INSIDE this primitive rather than the caller's business, because
// it operates on the compressed representation and nothing outside sees that.
// A caller that had to sequence it would need the rank, the scratch and the
// epsilon, which is the whole argument list back again.
//
// The norm is a plain RMS norm with no residual: there is nothing to add at this
// point, the compressed row is not a hidden state. Passing a residual here would
// be adding a 2048-wide vector to something that is not the same tensor.
template<class Format>
static LmScaleTensor LmProjectionWeightScale(
    const void *scale_data,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    if constexpr (Format::kScaleGroup == 0u)
    {
        return scale_data == 0
            ? LmScaleTensorNone()
            : LmScaleTensorInvalid(LM_SCALE_ENCODING_NONE);
    }
    else
    {
        return LmScaleTensorBlockF32(
            scale_data,
            1u,
            output_dimension,
            input_dimension,
            Format::kScaleGroup,
            Format::kScaleGroup);
    }
}

template<class Format>
static const void *LmProjectionPrepareInput(
    const uint16_t *input_bf16,
    const uint32_t *source_row_map,
    uint8_t *input_codes,
    uint8_t *input_scales,
    uint32_t rows,
    uint32_t input_dimension,
    uint32_t threads,
    LmScaleTensor *scale_out,
    cudaStream_t stream)
{
    if constexpr (Format::kScaleGroup == 0u)
    {
        *scale_out = LmScaleTensorNone();
        return input_bf16;
    }
    else
    {
        if (input_codes == 0 || input_scales == 0 ||
            (input_dimension % Format::kScaleGroup) != 0u)
        {
            *scale_out = LmScaleTensorInvalid(LM_SCALE_ENCODING_UE4M3);
            return 0;
        }
        LM_LAUNCH(
            (LmQuantiseRowsKernel<Format,256u>),
            dim3(rows,input_dimension / Format::kScaleGroup),
            threads,
            (Format::kScaleGroup + 8u) * sizeof(float),
            stream,
            input_bf16,
            source_row_map,
            input_codes,
            input_scales,
            rows,
            input_dimension);
        *scale_out = LmScaleTensorRowsUe4m3(
            input_scales,
            rows,
            input_dimension,
            Format::kScaleGroup);
        return input_codes;
    }
}

// hidden -> down -> norm -> up.
template<class Format>
static int32_t LmLowRankProject(
    const LmLowRankWeights *weights,
    const LmLowRankScratch *scratch,
    const uint16_t *input_bf16,
    uint16_t *output_bf16,
    uint32_t rows,
    uint32_t threads,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;
    const void *activation;
    int32_t status;

    if (weights == 0 || scratch == 0 || input_bf16 == 0 ||
        output_bf16 == 0 || rows == 0u || weights->input_dimension == 0u ||
        weights->rank == 0u || weights->output_dimension == 0u ||
        scratch->compressed_bf16 == 0 ||
        scratch->dense_row_offset == 0 || scratch->dense_tile_prefix == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    memset(&gemm,0,sizeof(gemm));
    activation = LmProjectionPrepareInput<Format>(
        input_bf16,
        0,
        scratch->input_codes,
        scratch->input_scales,
        rows,
        weights->input_dimension,
        threads,
        &gemm.scale_a,
        stream);
    if (activation == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    gemm.scale_b = LmProjectionWeightScale<Format>(
        weights->down_scale,
        weights->rank,
        weights->input_dimension);
    gemm.group_row_offset = scratch->dense_row_offset;
    gemm.group_tile_prefix = scratch->dense_tile_prefix;
    gemm.output_bf16 = scratch->compressed_bf16;
    status = LmGemmLaunch<
        Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
            &gemm,
            activation,
            weights->down_weight,
            rows,
            rows,
            1u,
            1u,
            weights->input_dimension,
            weights->rank,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<256u,uint16_t>),
        rows,
        threads,
        (weights->rank + 8u) * sizeof(float),
        stream,
        scratch->compressed_bf16,
        0,
        (const uint16_t *)weights->norm_weight,
        0,
        scratch->compressed_bf16,
        weights->rank,
        weights->rank,
        weights->norm_epsilon);

    memset(&gemm,0,sizeof(gemm));
    activation = LmProjectionPrepareInput<Format>(
        scratch->compressed_bf16,
        0,
        scratch->compressed_codes,
        scratch->compressed_scales,
        rows,
        weights->rank,
        threads,
        &gemm.scale_a,
        stream);
    if (activation == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    gemm.scale_b = LmProjectionWeightScale<Format>(
        weights->up_scale,
        weights->output_dimension,
        weights->rank);
    gemm.group_row_offset = scratch->dense_row_offset;
    gemm.group_tile_prefix = scratch->dense_tile_prefix;
    gemm.output_bf16 = output_bf16;
    return LmGemmLaunch<
        Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
            &gemm,
            activation,
            weights->up_weight,
            rows,
            rows,
            1u,
            1u,
            weights->rank,
            weights->output_dimension,
            multiprocessors,
            false,
            stream);
}

// The absorbed form: four plain projections from the hidden state.
//
// query-latent and kv-latent go into the compressed space directly; the two rope
// projections produce the positional halves. There is no norm and no
// intermediate, because the folding happened at pack time.
//
// Grouped as one call rather than four at the call site because the four share
// an input and a row count, and because getting one of the four pointed at the
// wrong weight is the kind of mistake a four-field struct prevents and four
// separate calls invite.
struct LmAbsorbedWeights
{
	const void *query_latent_weight;
	const void *query_latent_scale;
	const void *query_rope_weight;
	const void *query_rope_scale;
	const void *key_rope_weight;
	const void *key_rope_scale;
	const void *kv_latent_weight;
	const void *kv_latent_scale;
	uint32_t input_dimension;
	uint32_t query_latent_dimension;
	uint32_t query_rope_dimension;
	uint32_t key_rope_dimension;
	uint32_t kv_latent_dimension;
};

struct LmAbsorbedOutputs
{
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *key_rope_bf16;
	uint16_t *kv_latent_bf16;
};


template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmJoinRowsKernel(
    const uint16_t *__restrict__ left_bf16,
    uint32_t left_dimension,
    const uint16_t *__restrict__ right_bf16,
    uint32_t right_dimension,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count)
{
    uint32_t row = blockIdx.x;
    uint32_t output_dimension = left_dimension + right_dimension;
    uint32_t column;

    if (row >= row_count)
    {
        return;
    }
    for (column = threadIdx.x; column < output_dimension; column += THREADS)
    {
        if (column < left_dimension)
        {
            output_bf16[((uint64_t)row * output_dimension) + column] =
                left_bf16[((uint64_t)row * left_dimension) + column];
        }
        else
        {
            uint32_t right_column = column - left_dimension;
            output_bf16[((uint64_t)row * output_dimension) + column] =
                right_bf16[((uint64_t)row * right_dimension) + right_column];
        }
    }
}

template<class Format>
static int32_t LmAbsorbedProject(
    const LmAbsorbedWeights *weights,
    const LmAbsorbedOutputs *out,
    const uint16_t *input_bf16,
    const uint8_t *input_codes,
    const LmScaleTensor *input_scale,
    const uint32_t *dense_row_offset,
    uint32_t *dense_tile_prefix,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    struct LmAbsorbedPass
    {
        const void *weight;
        const void *scale;
        uint16_t *output;
        uint32_t width;
    };
    const LmAbsorbedPass pass[4] =
    {
        { weights->query_latent_weight, weights->query_latent_scale,
            out->query_latent_bf16, weights->query_latent_dimension },
        { weights->query_rope_weight, weights->query_rope_scale,
            out->query_rope_bf16, weights->query_rope_dimension },
        { weights->key_rope_weight, weights->key_rope_scale,
            out->key_rope_bf16, weights->key_rope_dimension },
        { weights->kv_latent_weight, weights->kv_latent_scale,
            out->kv_latent_bf16, weights->kv_latent_dimension }
    };
    const void *activation;
    LmGemmArguments gemm;
    int32_t status;
    uint32_t index;

    if (weights == 0 || out == 0 || input_bf16 == 0 ||
        dense_row_offset == 0 || dense_tile_prefix == 0 || rows == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if constexpr (Format::kScaleGroup == 0u)
    {
        activation = input_bf16;
    }
    else
    {
        if (input_codes == 0 || input_scale == 0 ||
            LmScaleTensorIsValid(input_scale) == 0u)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        activation = input_codes;
    }

    for (index = 0u; index < 4u; ++index)
    {
        if (pass[index].weight == 0 || pass[index].output == 0 ||
            pass[index].width == 0u)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        memset(&gemm,0,sizeof(gemm));
        gemm.scale_a = Format::kScaleGroup == 0u
            ? LmScaleTensorNone()
            : *input_scale;
        gemm.scale_b = LmProjectionWeightScale<Format>(
            pass[index].scale,
            pass[index].width,
            weights->input_dimension);
        gemm.group_row_offset = dense_row_offset;
        gemm.group_tile_prefix = dense_tile_prefix;
        gemm.output_bf16 = pass[index].output;
        status = LmGemmLaunch<
            Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
                &gemm,
                activation,
                pass[index].weight,
                rows,
                rows,
                1u,
                1u,
                weights->input_dimension,
                pass[index].width,
                multiprocessors,
                false,
                stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    return LM_LAUNCH_OK;
}

// -- fused QKV ------------------------------------------------------------------
//
// The other shape a model's attention projection takes: one GEMM producing
// query, key and value concatenated per row, split afterwards.
//
// MiMo 2.5 does this where GLM 5.2 does four separate projections, and neither
// is a variant of the other. A fused projection is one large GEMM with better
// arithmetic intensity; four separate ones let each output have its own
// quantisation and let a latent-absorbed model skip materialising K and V at
// all. Which a model uses is in its checkpoint, not a choice at run time.
//
// The split is a copy rather than a view because the three parts go to different
// places - query to RoPE and then attention, key and value into a cache slot -
// and a view would make every consumer carry the row stride and the offset. One
// copy of a decode row is 27 KB at MiMo 2.5's widths, which is nothing against
// the projection that produced it.
struct LmQkvLayout
{
	uint32_t query_dimension;      /* heads * qk_head_dim */
	uint32_t key_dimension;        /* kv_heads * qk_head_dim */
	uint32_t value_dimension;      /* kv_heads * v_head_dim */
	uint32_t rope_dimension;       /* rotated suffix of each qk head */
	uint32_t head_dimension;       /* qk dim per head, for locating the rope part */
};

// Query, key and value out of a fused row.
//
// The value scale is applied here rather than after attention because it belongs
// to the value tensor: MiMo 2.5 carries a 0.707 factor on V, and folding it into
// the attention output instead is the same number only when the softmax weights
// sum to one - which they do, but the equality stops holding the moment anything
// masks a position after the softmax. Scaling the tensor it belongs to survives
// that.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSplitQkvKernel(const uint16_t *__restrict__ fused_bf16, LmQkvLayout layout, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint32_t rows, float value_scale)
{
	uint32_t row = blockIdx.x,index;
	uint32_t total = layout.query_dimension + layout.key_dimension + layout.value_dimension;
	uint64_t base = (uint64_t)row * total;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < layout.query_dimension; index += THREADS)
		query_bf16[((uint64_t)row * layout.query_dimension) + index] = fused_bf16[base + index];
	for (index = threadIdx.x; index < layout.key_dimension; index += THREADS)
		key_bf16[((uint64_t)row * layout.key_dimension) + index] =
			fused_bf16[base + layout.query_dimension + index];
	for (index = threadIdx.x; index < layout.value_dimension; index += THREADS)
		value_bf16[((uint64_t)row * layout.value_dimension) + index] =
			LmFloatToBf16(LmBf16ToFloat(
				fused_bf16[base + layout.query_dimension + layout.key_dimension + index])
				* value_scale);
}

// Split a fused per-head query|gate row into its two halves.
//
// The GDN model's attention query projection carries the output gate INSIDE the
// query section: each head's 2 * head_dimension rows are head_dimension of
// query then head_dimension of gate (config.h, attn_output_gate). The query
// half feeds RoPE and attention, the gate half LmOutputGateKernel after
// attention, and both consumers want contiguous heads - which is why this is
// a copy and not a stride, the same argument LmSplitQkvKernel carries.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSplitQueryGateKernel(const uint16_t *__restrict__ fused_bf16, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension, uint32_t rows)
{
	uint32_t row = blockIdx.x,index,head,element;
	uint64_t fused_base = (uint64_t)row * heads * 2u * head_dimension;
	uint64_t half_base = (uint64_t)row * heads * head_dimension;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < heads * head_dimension; index += THREADS)
	{
		head = index / head_dimension;
		element = index % head_dimension;
		query_bf16[half_base + index] =
			fused_bf16[fused_base + (head * 2u * head_dimension) + element];
		gate_bf16[half_base + index] =
			fused_bf16[fused_base + (head * 2u * head_dimension) + head_dimension + element];
	}
}

// RoPE over the rotated suffix of every head in a packed multi-head row.
//
// A fused query row is heads x head_dimension with the rope part at the end of
// each head, not at the end of the row. Rotating the row's tail would rotate the
// last head only and leave the other sixty-three unrotated - which produces
// fluent text whose attention ignores position for all but one head.
template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t base = (((uint64_t)row * heads) + head) * head_dimension
		+ (head_dimension - rope_dimension);
	float position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dimension));
}

// Apply a per-head block-diagonal projection: out[h] = W[h] @ in[h].
//
// MLA's value absorption folds kv_b_value into the output projection. That is
// algebraically clean and wrong twice over here.
//
// Wrong for correctness: the output gate is elementwise in v-space, and
// elementwise gating does not commute with a per-head fold - Diag(g) W is not
// W Diag(g'). The reference gates the attention output before o_proj, and no
// checkpoint tensor exists for a latent-space gate: g_proj emits
// heads * v_head_dim.
//
// Wrong for speed on GB10, which is what makes the choice easy rather than a
// trade. docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md: 273 GB/s LPDDR5x unified at
// eta_bw 0.80, and a calibrated 6.5 TFLOP/s on the linear path - 30 FLOP per
// byte before compute can bind. Absorbing the value half inflates the MLA
// output projection from heads*v_head to heads*kv_lora, 8.30 GB to 19.55 GB
// across 24 layers, which costs 55 ms per token in weight reads to save the
// 302 MFLOP this kernel performs, 46 us. Three orders of magnitude the wrong
// way, on the one machine this runs on.
//
// So attention stays in the latent, this brings it back to v-space, and the
// gate and output projection use the checkpoint's tensors unchanged.
template<
	uint32_t THREADS,
	uint32_t IN_DIM,
	uint32_t OUT_DIM,
	uint32_t INPUT_HEAD_DIM = IN_DIM,
	uint32_t INPUT_OFFSET = 0u>
__global__ __launch_bounds__(THREADS, 1)
void LmPerHeadProjectKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	__shared__ float shared_input[IN_DIM];
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t input_base,weight_base,output_base;
	static_assert(INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM,
		"per-head input slice exceeds its source head");
	if ( row >= rows || head >= heads )
		return;
	input_base = ((((uint64_t)row * heads) + head) * INPUT_HEAD_DIM)
		+ INPUT_OFFSET;
	weight_base = (uint64_t)head * OUT_DIM * IN_DIM;
	output_base = (((uint64_t)row * heads) + head) * OUT_DIM;
	for (index = threadIdx.x; index < IN_DIM; index += THREADS)
		shared_input[index] = LmBf16ToFloat(input_bf16[input_base + index]);
	__syncthreads();
	// The input is staged once and read OUT_DIM times from shared rather than
	// IN_DIM * OUT_DIM times from memory. The weight read is the whole cost, as
	// everything is on this machine.
	for (index = threadIdx.x; index < OUT_DIM; index += THREADS)
	{
		float total = 0.0f;
		uint32_t element;
		for (element = 0u; element < IN_DIM; ++element)
			total += shared_input[element]
				* LmBf16ToFloat(weight_bf16[weight_base + (index * IN_DIM) + element]);
		output_bf16[output_base + index] = LmFloatToBf16(total);
	}
}

// Extract and rotate one positional slice from every packed query head.
// Keeping this as one launch avoids materialising the no-PE slice: the key
// projection above reads it in place through INPUT_HEAD_DIM/INPUT_OFFSET.
template<
	uint32_t THREADS,
	LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmExtractRopePerHeadKernel(
	const uint16_t *__restrict__ input_bf16,
	uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ positions,
	uint32_t heads,
	uint32_t input_head_dimension,
	uint32_t input_offset,
	uint32_t rope_dimension,
	float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t input_base,output_base;
	float position;
	if ( head >= heads || input_offset > input_head_dimension ||
		rope_dimension > input_head_dimension - input_offset )
		return;
	input_base = ((((uint64_t)row * heads) + head) * input_head_dimension)
		+ input_offset;
	output_base = (((uint64_t)row * heads) + head) * rope_dimension;
	position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
	{
		uint32_t low_offset,high_offset;
		float low,high,angle;
		low_offset = PAIRING == LM_ROPE_INTERLEAVED ? index * 2u : index;
		high_offset = PAIRING == LM_ROPE_INTERLEAVED
			? (index * 2u) + 1u : half + index;
		low = LmBf16ToFloat(input_bf16[input_base + low_offset]);
		high = LmBf16ToFloat(input_bf16[input_base + high_offset]);
		angle = position * __powf(
			theta,-2.0f * (float)index / (float)rope_dimension);
		LmRopePair(&low,&high,angle);
		output_bf16[output_base + low_offset] = LmFloatToBf16(low);
		output_bf16[output_base + high_offset] = LmFloatToBf16(high);
	}
}
