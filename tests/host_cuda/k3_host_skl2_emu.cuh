#pragma once

#include <cmath>
#include <stdint.h>
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "sparkpipe/spark_gpu.h"

enum K3HostSkl2Ids
{
	K3SKL2_k3_split_fused_proj_256 = 1u,
	K3SKL2_k3_kda_prep_finish_128 = 2u,
	K3SKL2_k3_kda_norm_gate_128 = 3u,
	K3SKL2_k3_rms_norm_slice_256 = 4u,
	K3SKL2_k3_fold_l2_norm_128 = 5u,
	K3SKL2_k3_rms_norm_kv_shard_store_64 = 6u,
	K3SKL2_k3_layer_norm_kv_shard_store_32 = 7u
};

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	const uint16_t *qkvb, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *beta,
	uint16_t *rq, uint16_t *rk, uint16_t *rv,
	uint32_t rows, uint32_t qk_dim, uint32_t v_dim, uint32_t heads, uint32_t fused_rows)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < rows; ++row)
	{
		const uint16_t *src = qkvb + (uint64_t)row * fused_rows;
		for (uint32_t i = 0u; i < qk_dim; ++i)
		{
			q[(uint64_t)row * qk_dim + i] = src[i];
			if ( rq != 0 ) rq[(uint64_t)row * qk_dim + i] = src[i];
			k[(uint64_t)row * qk_dim + i] = src[qk_dim + i];
			if ( rk != 0 ) rk[(uint64_t)row * qk_dim + i] = src[qk_dim + i];
		}
		for (uint32_t i = 0u; i < v_dim; ++i)
		{
			v[(uint64_t)row * v_dim + i] = src[2u * qk_dim + i];
			if ( rv != 0 ) rv[(uint64_t)row * v_dim + i] = src[2u * qk_dim + i];
		}
		for (uint32_t i = 0u; i < heads; ++i)
			beta[(uint64_t)row * heads + i] = src[2u * qk_dim + v_dim + i];
	}
	return SPARK_GPU_SUCCESS;
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	uint16_t *query_bf16, uint16_t *key_bf16, const uint16_t *decay_logit_bf16,
	const float *channel_bias, const float *head_log_scale, const uint16_t *beta_logit_bf16,
	float *retention, float *write_gate,
	uint32_t heads, float min_log_decay, float epsilon, uint32_t rows)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < rows; ++row)
		for (uint32_t head = 0u; head < heads; ++head)
		{
			const uint64_t base = ((uint64_t)row * heads + head) * 128u;
			float qs = 0.0f, ks = 0.0f;
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				const float qv = LmBf16ToFloat(query_bf16[base + i]);
				const float kv = LmBf16ToFloat(key_bf16[base + i]);
				qs += qv * qv;
				ks += kv * kv;
			}
			const float qi = 1.0f / sqrtf(qs + epsilon), ki = 1.0f / sqrtf(ks + epsilon);
			const float hls = head_log_scale[head];
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				query_bf16[base + i] = LmFloatToBf16(LmBf16ToFloat(query_bf16[base + i]) * qi);
				key_bf16[base + i] = LmFloatToBf16(LmBf16ToFloat(key_bf16[base + i]) * ki);
				const float dl = LmBf16ToFloat(decay_logit_bf16[base + i]);
				const float scaled = expf(hls) * (dl + channel_bias[head * 128u + i]);
				retention[base + i] = expf(min_log_decay / (1.0f + expf(-scaled)));
			}
			const float bl = LmBf16ToFloat(beta_logit_bf16[(uint64_t)row * heads + head]);
			write_gate[(uint64_t)row * heads + head] = 1.0f / (1.0f + expf(-bl));
		}
	return SPARK_GPU_SUCCESS;
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	uint16_t *output_bf16, const uint16_t *gate_bf16, const float *norm_weight,
	uint32_t heads, float epsilon, uint32_t rows)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < rows; ++row)
		for (uint32_t head = 0u; head < heads; ++head)
		{
			const uint64_t base = ((uint64_t)row * heads + head) * 128u;
			float sum = 0.0f;
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				const float v = LmBf16ToFloat(output_bf16[base + i]);
				sum += v * v;
			}
			const float sc = 1.0f / sqrtf(sum / 128.0f + epsilon);
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				const float n = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(output_bf16[base + i]) * sc * norm_weight[i]));
				const float g = LmBf16ToFloat(gate_bf16[base + i]);
				output_bf16[base + i] = LmFloatToBf16(n / (1.0f + expf(-g)));
			}
		}
	return SPARK_GPU_SUCCESS;
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16,
	uint32_t dim, uint32_t col_offset, uint32_t slice_width, float epsilon)
{
	(void)id; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < grid.x; ++row)
	{
		const uint64_t base = (uint64_t)row * dim;
		float sum = 0.0f;
		for (uint32_t i = 0u; i < dim; ++i)
		{
			const float v = LmBf16ToFloat(input_bf16[base + i]);
			sum += v * v;
		}
		const float sc = 1.0f / sqrtf(sum / (float)dim + epsilon);
		for (uint32_t i = 0u; i < slice_width; ++i)
		{
			const uint32_t col = col_offset + i;
			const float n = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(input_bf16[base + col]) * sc));
			output_bf16[(uint64_t)row * slice_width + i] = LmFloatToBf16(n * LmBf16ToFloat(weight_bf16[col]));
		}
	}
	return SPARK_GPU_SUCCESS;
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	uint16_t *query_bf16, uint16_t *key_bf16, uint32_t heads, uint32_t rows, float epsilon)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < rows; ++row)
		for (uint32_t head = 0u; head < heads; ++head)
		{
			const uint64_t base = ((uint64_t)row * heads + head) * 128u;
			float qs = 0.0f, ks = 0.0f;
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				const float qv = LmBf16ToFloat(query_bf16[base + i]);
				const float kv = LmBf16ToFloat(key_bf16[base + i]);
				qs += qv * qv;
				ks += kv * kv;
			}
			const float qi = 1.0f / sqrtf(qs + epsilon), ki = 1.0f / sqrtf(ks + epsilon);
			for (uint32_t i = 0u; i < 128u; ++i)
			{
				query_bf16[base + i] = LmFloatToBf16(LmBf16ToFloat(query_bf16[base + i]) * qi);
				key_bf16[base + i] = LmFloatToBf16(LmBf16ToFloat(key_bf16[base + i]) * ki);
			}
		}
	return SPARK_GPU_SUCCESS;
}

static inline uint16_t *K3HostShardSlot(const LmKvShardView &view, uint32_t seq, uint32_t pos, uint32_t page_slots, uint32_t slot_bytes)
{
	if ( SparkKvShardOwns(view.shard, pos) == 0u || seq >= view.pages.sequence_count )
		return 0;
	const uint32_t page = pos / page_slots;
	if ( page >= view.pages.page_table_stride )
		return 0;
	const uint32_t phys = view.pages.page_table[(uint64_t)seq * view.pages.page_table_stride + page];
	if ( phys >= view.pages.pool_page_count )
		return 0;
	const uint32_t sh_ps = page_slots / view.shard.degree;
	const uint32_t sip = view.shard.degree > 1u ? SparkKvShardSlotInPage(view.shard, page_slots, pos) : (pos % page_slots);
	return (uint16_t *)(view.pages.pool + ((uint64_t)phys * sh_ps + sip) * slot_bytes);
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	LmKvShardView view, const uint16_t *rows_bf16, const uint16_t *norm_weight,
	const uint32_t *sequence_of_row, const uint32_t *position_of_row,
	uint32_t row_count, float epsilon, uint16_t *kv_all, uint32_t total_ctx, uint32_t page_slots, uint32_t slot_bytes)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < row_count; ++row)
	{
		const uint32_t pos = position_of_row[row];
		uint16_t *dst_all = kv_all != 0 && pos < total_ctx ? kv_all + (uint64_t)pos * 576u : 0;
		uint16_t *slot = K3HostShardSlot(view, sequence_of_row[row], pos, page_slots, slot_bytes);
		if ( slot == 0 && dst_all == 0 )
			continue;
		const uint16_t *src = rows_bf16 + (uint64_t)row * 576u;
		float sum = 0.0f;
		for (uint32_t i = 0u; i < 512u; ++i) { const float v = LmBf16ToFloat(src[i]); sum += v * v; }
		const float sc = 1.0f / sqrtf(sum / 512.0f + epsilon);
		for (uint32_t i = 0u; i < 512u; ++i)
		{
			const uint16_t out = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(src[i]) * sc)) * LmBf16ToFloat(norm_weight[i]));
			if ( slot != 0 ) slot[i] = out;
			if ( dst_all != 0 ) dst_all[i] = out;
		}
		for (uint32_t i = 0u; i < 64u; ++i)
		{
			if ( slot != 0 ) slot[512u + i] = src[512u + i];
			if ( dst_all != 0 ) dst_all[512u + i] = src[512u + i];
		}
	}
	return SPARK_GPU_SUCCESS;
}

static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream,
	LmKvShardView view, const uint16_t *rows_bf16, const uint16_t *weight, const uint16_t *bias,
	const uint32_t *sequence_of_row, const uint32_t *position_of_row,
	uint32_t row_count, float epsilon, uint32_t page_slots, uint32_t slot_bytes)
{
	(void)id; (void)grid; (void)block; (void)shared; (void)stream;
	for (uint32_t row = 0u; row < row_count; ++row)
	{
		uint16_t *slot = K3HostShardSlot(view, sequence_of_row[row], position_of_row[row], page_slots, slot_bytes);
		if ( slot == 0 )
			continue;
		const uint16_t *src = rows_bf16 + (uint64_t)row * 128u;
		float sum = 0.0f, sum_sq = 0.0f;
		for (uint32_t i = 0u; i < 128u; ++i) { const float v = LmBf16ToFloat(src[i]); sum += v; sum_sq += v * v; }
		const float mean = sum / 128.0f;
		const float inv = 1.0f / sqrtf(fmaxf(sum_sq / 128.0f - mean * mean, 0.0f) + epsilon);
		for (uint32_t i = 0u; i < 128u; ++i)
			slot[i] = LmFloatToBf16(((LmBf16ToFloat(src[i]) - mean) * inv * LmBf16ToFloat(weight[i])) + LmBf16ToFloat(bias[i]));
	}
	return SPARK_GPU_SUCCESS;
}
