#pragma once

static inline uint32_t Glm5NextTileShift(uint32_t tile_m)
{
	return tile_m == 128u ? 7u : (tile_m == 64u ? 6u : (tile_m == 32u ? 5u : 4u));
}

static inline uint32_t Glm5NextMoePersistentBlocks(uint32_t sms, uint32_t sh)
{
	return sh <= 49152u ? sms * 2u : sms;
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeMmaW1ActLaunch(const Glm5NextLayerBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	const uint32_t mean = (packed_rows + G5N_EXPERTS - 1u) / G5N_EXPERTS;
	const uint32_t tile_m = mean > 12u ? 64u : mean > 6u ? 32u : 16u;
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_build_tile_prefix_32, 1u, 32u, 0u, stream,
		b->group_row_offset, b->group_tile_prefix_w1, Glm5NextTileShift(tile_m), (uint32_t)2u) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		const uint32_t sh = 3u * (tile_m * 256u + 16384u);
		const uint32_t id = tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w1_act_fp8_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w1_act_fp8_m32 : GLM5NEXTSKL2_g5n_moe_mma_w1_act_fp8_m16);
		if ( Glm5NextSkl2Launch(id, Glm5NextMoePersistentBlocks(sms, sh), 256u, sh, stream,
			(const uint8_t *)b->expert_w1_weight, (const float *)b->expert_w1_scale, b->normed_bf16, b->group_row_offset,
			b->group_tile_prefix_w1, b->route_source_token, b->intermediate_bf16) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
	{
		const uint32_t sh = 3u * (tile_m * 128u + 16384u);
		const uint32_t threads = tile_m == 16u ? 128u : 256u;
		const uint32_t id = tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w1_act_bf16_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w1_act_bf16_m32 : GLM5NEXTSKL2_g5n_moe_mma_w1_act_bf16_m16);
		if ( Glm5NextSkl2Launch(id, Glm5NextMoePersistentBlocks(sms, sh), threads, sh, stream,
			(const uint16_t *)b->expert_w1_weight, b->normed_bf16, b->group_row_offset, b->group_tile_prefix_w1,
			b->route_source_token, b->intermediate_bf16) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec>
static int32_t Glm5NextMoeMmaW2K128Launch(const Glm5NextLayerBuffers *b, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	const uint32_t mean = (packed_rows + G5N_EXPERTS - 1u) / G5N_EXPERTS;
	const uint32_t tile_m = mean > 16u ? 128u : mean > 8u ? 64u : mean > 4u ? 32u : 16u;
	constexpr uint32_t n_tiles = G5N_HIDDEN / 128u;
	if constexpr ( ExpertCodec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
		if ( tile_m == 16u && b->prefill_unified == 0u )
			return LmStreamGemmGrouped<LmBf16Format>(b->expert_w2_weight, LmScaleTensorNone(), b->intermediate_bf16, b->expert_out_bf16,
				b->group_row_offset, 0, G5N_EXPERTS, packed_rows, b->expert_intermediate, G5N_HIDDEN, sms, stream);
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_build_tile_prefix_32, 1u, 32u, 0u, stream,
		b->group_row_offset, b->group_tile_prefix_w2, Glm5NextTileShift(tile_m), n_tiles) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
	{
		const uint32_t sh = 2u * (tile_m * 256u + 16384u);
		const uint32_t id = tile_m == 128u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m128 :
			(tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m32 : GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m16));
		if ( Glm5NextSkl2Launch(id, Glm5NextMoePersistentBlocks(sms, sh), 256u, sh, stream,
			(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
			b->group_tile_prefix_w2, b->expert_out_bf16, (uint32_t)0u, n_tiles, (uint32_t)1u) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	else
	{
		const uint32_t sh = 2u * (tile_m * 128u + 16384u);
		const uint32_t id = tile_m == 128u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m128 :
			(tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m32 : GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m16));
		if ( Glm5NextSkl2Launch(id, Glm5NextMoePersistentBlocks(sms, sh), 256u, sh, stream,
			(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2,
			b->expert_out_bf16, (uint32_t)0u, n_tiles, (uint32_t)1u) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextSharedMmaW1ActLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream,
	uint16_t *inter_dst = 0)
{
	(void)sms;
	uint16_t *dst = inter_dst != 0 ? inter_dst : b->intermediate_bf16;
	const uint32_t tile_m = rows >= 512u ? 32u : 16u;
	const uint32_t threads = tile_m == 16u ? 128u : 256u;
	const uint32_t sh = 3u * (tile_m * 128u + 16384u);
	const uint32_t id = tile_m == 32u ? GLM5NEXTSKL2_g5n_shared_mma_w1_act_m32 : GLM5NEXTSKL2_g5n_shared_mma_w1_act_m16;
	if ( Glm5NextSkl2Launch(id, dim3(2u, (rows + tile_m - 1u) / tile_m), threads, sh, stream,
		(const uint16_t *)b->w->shared_gate_up, b->normed_bf16, dst, rows) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

static int32_t Glm5NextSharedDownFinalizeAddLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t sms, SparkGpuStream stream,
	const uint16_t *shared_inter = 0)
{
	(void)sms;
	constexpr uint32_t sh = 2u * (64u * 128u + 16384u);
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_shared_down_finalize_add_mma_m64, dim3(G5N_HIDDEN / 128u, (rows + 63u) / 64u), 256u, sh, stream,
		(const uint16_t *)b->w->shared_down, shared_inter != 0 ? shared_inter : b->intermediate_bf16, b->expert_out_bf16,
		b->route_packed_row, b->route_weight, b->attention_out_bf16, rows, (uint32_t)0u, (uint32_t)G5N_HIDDEN) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return LM_LAUNCH_OK;
}

template<uint32_t ExpertCodec, uint32_t N_TILES_SLICE>
static int32_t Glm5NextMoeMmaW2SharedSlicedLaunch(const Glm5NextLayerBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, SparkGpuStream stream)
{
	constexpr uint32_t N_SLICE = N_TILES_SLICE * 128u;
	const uint32_t mean = (packed_rows + G5N_EXPERTS - 1u) / G5N_EXPERTS;
	const uint32_t tile_m = mean > 24u ? 64u : mean > 8u ? 32u : 16u;
	constexpr uint32_t FIN_TM = N_TILES_SLICE <= 2u ? 32u : 64u;
	constexpr uint32_t fin_sh = 2u * (FIN_TM * 128u + 16384u);
	const uint32_t fin_id = FIN_TM == 32u ? GLM5NEXTSKL2_g5n_shared_down_finalize_add_mma_m32 : GLM5NEXTSKL2_g5n_shared_down_finalize_add_mma_m64;
	if ( Glm5NextSkl2Launch(GLM5NEXTSKL2_g5n_build_tile_prefix_32, 1u, 32u, 0u, stream,
		b->group_row_offset, b->group_tile_prefix_w2, Glm5NextTileShift(tile_m), N_TILES_SLICE) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w2_sh = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 ? 2u * (tile_m * 256u + 16384u) : 2u * (tile_m * 128u + 16384u);
	const uint32_t w2_blocks = Glm5NextMoePersistentBlocks(sms, w2_sh);
	const uint32_t w2_id = ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 ?
		(tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m32 : GLM5NEXTSKL2_g5n_moe_mma_w2_k128_fp8_m16)) :
		(tile_m == 64u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m64 :
			(tile_m == 32u ? GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m32 : GLM5NEXTSKL2_g5n_moe_mma_w2_k128_bf16_m16));
	for ( uint32_t col_off = 0u; col_off < G5N_HIDDEN; col_off += N_SLICE )
	{
		if constexpr ( ExpertCodec == SPARK_WEIGHT_CODEC_FP8_E4M3 )
		{
			if ( Glm5NextSkl2Launch(w2_id, w2_blocks, 256u, w2_sh, stream,
				(const uint8_t *)b->expert_w2_weight, (const float *)b->expert_w2_scale, b->intermediate_bf16, b->group_row_offset,
				b->group_tile_prefix_w2, b->expert_out_bf16, col_off, N_TILES_SLICE, (uint32_t)0u) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		else
		{
			if ( Glm5NextSkl2Launch(w2_id, w2_blocks, 256u, w2_sh, stream,
				(const uint16_t *)b->expert_w2_weight, b->intermediate_bf16, b->group_row_offset, b->group_tile_prefix_w2,
				b->expert_out_bf16, col_off, N_TILES_SLICE, (uint32_t)0u) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		}
		if ( Glm5NextSkl2Launch(fin_id, dim3(N_TILES_SLICE, (rows + FIN_TM - 1u) / FIN_TM), 256u, fin_sh, stream,
			(const uint16_t *)b->w->shared_down, b->gate_up_bf16, b->expert_out_bf16, b->route_packed_row, b->route_weight,
			b->attention_out_bf16, rows, col_off, N_SLICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return LM_LAUNCH_OK;
}
