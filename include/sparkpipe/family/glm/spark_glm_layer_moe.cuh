#pragma once

template<uint32_t ExpertCodec>
static int32_t SPARK_FAMILY_BARE(LayerMoeValidate)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    uint32_t rows,
    uint32_t packed_rows)
{
    using ExpertFormat = typename LmWeightCodec<ExpertCodec>::Format;

    static_assert(ExpertCodec != SPARK_WEIGHT_CODEC_BF16,
        "GLM 5.2 routed experts require an explicit compressed codec");
    static_assert(SPARK_FAMILY_BARE_CONST(HIDDEN) % ExpertFormat::kScaleGroup == 0u &&
        SPARK_FAMILY_BARE_CONST(EXPERT_INTERMEDIATE) % ExpertFormat::kScaleGroup == 0u,
        "GLM 5.2 expert dimensions must contain complete codec scale groups");

    if (buffers == 0 || rows == 0u ||
        packed_rows != rows * SPARK_FAMILY_BARE_CONST(TOP_K) ||
        buffers->attention_out_bf16 == 0 || buffers->residual_bf16 == 0 ||
        buffers->mlp_norm_weight == 0 || buffers->normed_bf16 == 0 ||
        buffers->router_weight == 0 || buffers->router_logits == 0 ||
        buffers->router_correction_bias == 0 ||
        buffers->route_expert == 0 || buffers->route_weight == 0 ||
        buffers->route_source_token == 0 || buffers->route_packed_row == 0 ||
        buffers->group_row_offset == 0 ||
        buffers->group_tile_prefix_w1 == 0 ||
        buffers->group_tile_prefix_w2 == 0 ||
        buffers->expert_out_bf16 == 0 || buffers->gate_up_bf16 == 0 ||
        buffers->intermediate_bf16 == 0 || buffers->hidden_bf16 == 0 ||
        buffers->shared_gate_up_weight == 0 ||
        buffers->shared_down_weight == 0 || buffers->shared_out_bf16 == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec>
static int32_t SPARK_FAMILY_BARE(LayerMoe)(
    const SPARK_FAMILY_BARE(LayerBuffers) *buffers,
    uint32_t rows,
    uint32_t packed_rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    int32_t status = SPARK_FAMILY_BARE(LayerMoeRoute)<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
    if (status != LM_LAUNCH_OK)
        return status;
    return SPARK_FAMILY_BARE(LayerMoeExperts)<ExpertCodec>(buffers,rows,packed_rows,multiprocessors,stream);
}
