#pragma once


#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

static __device__ __forceinline__ void LmLinearCopyAsync(void *destination, const void *source, uint32_t bytes)
{
#if defined(__CUDA_ARCH__)
	const uint32_t shared = (uint32_t)__cvta_generic_to_shared(destination);
	if ( bytes == 16u )
		asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared), "l"(source) : "memory");
	else
		asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared), "l"(source) : "memory");
#else
	memcpy(destination,source,bytes);
#endif
}

static __device__ __forceinline__ float LmLinearWarpSum4(float a, float b, float c, float d)
{
	float sum = (a + b) + (c + d);
	#pragma unroll
	for ( uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu,sum,offset);
	return(sum);
}

static __device__ __forceinline__ float LmBoundedDecay(float logit, float bias, float head_log_scale, float minimum_log_decay)
{
	return(SparkMathBoundedDecay(logit,bias,head_log_scale,minimum_log_decay));
}

template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmBoundedDecayKernel(const uint16_t *__restrict__ logit_bf16, const float *__restrict__ channel_bias, const float *__restrict__ head_log_scale, float *__restrict__ retention, uint32_t heads, float minimum_log_decay, uint32_t rows)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t base;
	if ( row >= rows || head >= heads )
		return;
	base = (((uint64_t)row * heads) + head) * KEY_DIM;
	for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		retention[base + index] = LmBoundedDecay(LmBf16ToFloat(logit_bf16[base + index]),
			channel_bias[(head * KEY_DIM) + index],head_log_scale[head],minimum_log_decay);
}

template<uint32_t THREADS, uint32_t KEY_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGdnGateKernel(const uint16_t *__restrict__ decay_logit_bf16, const uint16_t *__restrict__ beta_logit_bf16, const float *__restrict__ head_log_scale, const float *__restrict__ head_bias, float *__restrict__ retention, float *__restrict__ write_gate, uint32_t heads, uint32_t rows, uint32_t logit_stride = 0u)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t scalar,logit;
	float shifted,factor;
	if ( row >= rows || head >= heads )
		return;
	scalar = ((uint64_t)row * heads) + head;
	logit = ((uint64_t)row * (logit_stride != 0u ? logit_stride : heads)) + head;
	shifted = LmBf16ToFloat(decay_logit_bf16[logit]) + head_bias[head];
	factor = SparkMathGatedDeltaDecay(head_log_scale[head],shifted);
	for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		retention[(scalar * KEY_DIM) + index] = factor;
	if ( threadIdx.x == 0u )
	{
		float beta_logit = LmBf16ToFloat(beta_logit_bf16[logit]);
		write_gate[scalar] = SparkMathSigmoidFast(beta_logit);
	}
}

template<uint32_t THREADS, uint32_t HEAD_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmL2NormalisePerHeadKernel(uint16_t *__restrict__ rows_bf16, uint32_t heads, uint32_t rows, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t base;
	float total = 0.0f,inverse;
	if ( row >= rows || head >= heads )
		return;
	base = (((uint64_t)row * heads) + head) * HEAD_DIM;
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
	{
		float value = LmBf16ToFloat(rows_bf16[base + index]);
		total += value * value;
	}
	inverse = rsqrtf(LmBlockSum<THREADS>(total,reduction) + epsilon);
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
		rows_bf16[base + index] =
			LmFloatToBf16(LmBf16ToFloat(rows_bf16[base + index]) * inverse);
}

struct LmReplayStep
{
	const uint16_t *key_bf16;
	const uint16_t *value_bf16;
	const float *retention;
	const float *write_gate;
};

static __device__ __forceinline__ void LmStoreState(float *slot, float value)
{
	*slot = value;
}

static __device__ __forceinline__ void LmStoreState(uint16_t *slot, float value)
{
	*slot = LmFloatToBf16(value);
}

template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
__global__ __launch_bounds__(THREADS, 1)
void LmReplayFoldKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const LmReplayStep *__restrict__ steps, const uint32_t *__restrict__ accepted_length, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t rows)
{
	__shared__ float shared_key[KEY_DIM];
	__shared__ float fold_reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_predicted[VALUE_DIM];
	uint32_t row = blockIdx.x,head = blockIdx.y,step,index,flat;
	State *state;
	float key_inverse;
	if ( row >= rows || head >= key_heads )
		return;
	state = (State *)(state_pool + ((uint64_t)state_index[row]
		* slot_bytes)) + ((uint64_t)head * KEY_DIM * VALUE_DIM);
	for (step = 0u; step < accepted_length[row]; ++step)
	{
		const LmReplayStep *input = &steps[step];
		uint64_t head_key = (((uint64_t)row * key_heads) + head) * KEY_DIM;
		uint64_t head_value = (((uint64_t)row * key_heads * value_heads_per_key)
			+ (head * value_heads_per_key)) * VALUE_DIM;
		float beta = input->write_gate[(row * key_heads) + head];
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
			shared_key[index] = LmBf16ToFloat(input->key_bf16[head_key + index]);
		__syncthreads();
		{
			float kk = 0.0f;
			for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
				kk += shared_key[index] * shared_key[index];
			key_inverse = rsqrtf(LmBlockSum<THREADS>(kk,fold_reduction) + 1e-6f);
		}
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
			shared_key[index] *= key_inverse;
		__syncthreads();
		for (index = threadIdx.x; index < VALUE_DIM; index += THREADS)
		{
			float total = 0.0f;
			uint32_t channel;
			for (channel = 0u; channel < KEY_DIM; ++channel)
				total += LmScalarToFloat(state[(channel * VALUE_DIM) + index])
					* shared_key[channel] * input->retention[head_key + channel];
			shared_predicted[index] = total;
		}
		__syncthreads();
		for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		{
			uint32_t channel = flat / VALUE_DIM,element = flat % VALUE_DIM;
			float v = LmBf16ToFloat(input->value_bf16[head_value + element]);
			LmStoreState(&state[flat],(
				(input->retention[head_key + channel] * LmScalarToFloat(state[flat]))
				+ (beta * (v - shared_predicted[element]) * shared_key[channel])));
		}
		__syncthreads();
	}
}

template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float, uint32_t COLUMNS = VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmDeltaRuleKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0)
{
	static_assert(COLUMNS != 0u && (VALUE_DIM % COLUMNS) == 0u, "value columns split into equal slices");
	extern __shared__ float state_s[];
	__shared__ float shared_key[KEY_DIM];
	__shared__ float shared_query[KEY_DIM];
	__shared__ float norm_reduction[2u * (THREADS / LM_WARP_LANES)];
	__shared__ float shared_predicted[COLUMNS];
	uint32_t sequence = blockIdx.x,head = blockIdx.y,column = blockIdx.z * COLUMNS,index,element,row,begin,end,flat,ordinal;
	State *state;
	float beta,key_inverse,query_inverse;
	if ( sequence >= sequences || head >= key_heads || column >= VALUE_DIM )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	state = (State *)(state_pool
		+ ((uint64_t)state_index[sequence] * slot_bytes)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(State)));
	for (flat = threadIdx.x; flat < KEY_DIM * COLUMNS; flat += THREADS)
		state_s[flat] = LmScalarToFloat(state[(flat / COLUMNS) * VALUE_DIM + column + flat % COLUMNS]);
	for (ordinal = begin; ordinal < end; ++ordinal)
	{
		row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
		const float *forget = forget_gate + ((((uint64_t)row * key_heads) + head) * KEY_DIM);
		beta = write_gate[(row * key_heads) + head];
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		{
			shared_key[index] = LmBf16ToFloat(key_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
			shared_query[index] = LmBf16ToFloat(query_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
		}
		for (index = threadIdx.x; index < COLUMNS; index += THREADS)
			shared_predicted[index] = 0.0f;
		__syncthreads();
		{
			float kk = 0.0f,qq = 0.0f;
			for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
			{
				kk += shared_key[index] * shared_key[index];
				qq += shared_query[index] * shared_query[index];
			}
			key_inverse = rsqrtf(LmBlockSum<THREADS>(kk,norm_reduction) + 1e-6f);
			query_inverse = rsqrtf(LmBlockSum<THREADS>(qq,
				norm_reduction + (THREADS / LM_WARP_LANES)) + 1e-6f);
		}
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		{
			shared_key[index] *= key_inverse;
			shared_query[index] *= query_inverse * rsqrtf((float)KEY_DIM);
		}
		__syncthreads();
		for (element = threadIdx.x; element < COLUMNS; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * COLUMNS) + element]
					* shared_key[index] * forget[index];
			shared_predicted[element] = total;
		}
		__syncthreads();
		{
			uint32_t value_head = head * value_heads_per_key;
			uint64_t value_base =
				(((uint64_t)row * key_heads * value_heads_per_key) + value_head) * VALUE_DIM + column;
			for (flat = threadIdx.x; flat < KEY_DIM * COLUMNS; flat += THREADS)
			{
				uint32_t channel = flat / COLUMNS,element_index = flat % COLUMNS;
				float v = LmBf16ToFloat(value_bf16[value_base + element_index]);
				float previous = state_s[flat];
				state_s[flat] =
					(forget[channel] * previous)
					+ (beta * (v - shared_predicted[element_index]) * shared_key[channel]);
			}
		}
		__syncthreads();
		for (element = threadIdx.x; element < COLUMNS; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * COLUMNS) + element]
					* shared_query[index];
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + column + element] =
				LmFloatToBf16(total);
		}
		__syncthreads();
	}
	if ( commit == 0u )
		return;
	for (flat = threadIdx.x; flat < KEY_DIM * COLUMNS; flat += THREADS)
		LmStoreState(&state[(flat / COLUMNS) * VALUE_DIM + column + flat % COLUMNS],state_s[flat]);
}

template<uint32_t WARPS, uint32_t PARTS>
static __device__ __forceinline__ float LmDeltaRuleWarpTotal(const float (&parts)[PARTS], uint32_t lane)
{
	float sums[PARTS], value = 0.0f;
	uint32_t part, offset;
	#pragma unroll
	for ( part = 0u; part < PARTS; part++ )
	{
		float partial = parts[part];
		for ( offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
			partial += __shfl_down_sync(0xffffffffu,partial,offset);
		sums[part] = __shfl_sync(0xffffffffu,partial,0);
	}
	#pragma unroll
	for ( part = 0u; part < PARTS; part++ )
		if ( lane == part )
			value = sums[part];
	if ( lane >= WARPS )
		value = 0.0f;
	for ( offset = WARPS / 2u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(__shfl_sync(0xffffffffu,value,0));
}

#define LM_DELTA_COLUMN_WARPS 4u
#define LM_DELTA_COLUMN_THREADS (LM_DELTA_COLUMN_WARPS * LM_WARP_LANES)
#define LM_DELTA_COLUMN_RING 3u

template<uint32_t WARPS, uint32_t KEY_DIM, uint32_t VALUE_DIM>
static __device__ __forceinline__ void LmDeltaRuleColumnPrepare(uint32_t query_side, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint32_t row, uint32_t head, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t column, uint32_t lane, float *key_s, float *query_s, float *forget_s, float *value_s, float *beta_s)
{
	constexpr uint32_t PARTS = KEY_DIM / LM_WARP_LANES;
	const uint64_t base = (((uint64_t)row * key_heads) + head) * KEY_DIM;
	const uint16_t *source = query_side != 0u ? query_bf16 : key_bf16;
	float parts[PARTS], squares[PARTS], inverse;
	uint32_t part;
	#pragma unroll
	for ( part = 0u; part < PARTS; part++ )
	{
		parts[part] = LmBf16ToFloat(__ldg(source + base + part * LM_WARP_LANES + lane));
		squares[part] = 0.0f;
		squares[part] += parts[part] * parts[part];
	}
	if ( query_side != 0u )
		value_s[lane] = LmBf16ToFloat(__ldg(value_bf16 + (((uint64_t)row * key_heads * value_heads_per_key) + (uint64_t)head * value_heads_per_key) * VALUE_DIM + column + lane));
	else
	{
		#pragma unroll
		for ( part = 0u; part < PARTS; part++ )
			forget_s[part * LM_WARP_LANES + lane] = __ldg(forget_gate + base + part * LM_WARP_LANES + lane);
		if ( lane == 0u )
			*beta_s = __ldg(write_gate + (row * key_heads) + head);
	}
	inverse = rsqrtf((LmDeltaRuleWarpTotal<WARPS,PARTS>(squares,lane) + 1e-6f));
	#pragma unroll
	for ( part = 0u; part < PARTS; part++ )
		if ( query_side != 0u )
			query_s[part * LM_WARP_LANES + lane] = parts[part] * (inverse * rsqrtf((float)KEY_DIM));
		else
			key_s[part * LM_WARP_LANES + lane] = parts[part] * inverse;
}

template<uint32_t KEY_DIM>
static __device__ __forceinline__ float LmDeltaRulePredict4(const float (&state_s)[KEY_DIM][LM_WARP_LANES], const float *key_s, const float *forget_s, uint32_t lane)
{
	float predicted = 0.0f;
	#pragma unroll 4
	for ( uint32_t index = 0u; index < KEY_DIM; index += 4u )
	{
		const float4 key4 = *(const float4 *)&key_s[index];
		const float4 forget4 = *(const float4 *)&forget_s[index];
		predicted = __fmaf_rn(__fmul_rn(state_s[index][lane],key4.x),forget4.x,predicted);
		predicted = __fmaf_rn(__fmul_rn(state_s[index + 1u][lane],key4.y),forget4.y,predicted);
		predicted = __fmaf_rn(__fmul_rn(state_s[index + 2u][lane],key4.z),forget4.z,predicted);
		predicted = __fmaf_rn(__fmul_rn(state_s[index + 3u][lane],key4.w),forget4.w,predicted);
	}
	return(predicted);
}

#define LM_DELTA_SNAPSHOT_MAX 8u

template<uint32_t REDUCE_THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
__global__ __launch_bounds__(LM_DELTA_COLUMN_THREADS)
void LmDeltaRuleColumnKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0, const uint32_t *__restrict__ state_loads = 0, const uint32_t *__restrict__ state_saves = 0, float *__restrict__ snapshots = 0)
{
	static_assert((KEY_DIM % LM_WARP_LANES) == 0u && (VALUE_DIM % LM_WARP_LANES) == 0u && KEY_DIM <= REDUCE_THREADS &&
		(REDUCE_THREADS % LM_WARP_LANES) == 0u && REDUCE_THREADS / LM_WARP_LANES <= LM_WARP_LANES &&
		(KEY_DIM % (4u * LM_DELTA_COLUMN_WARPS)) == 0u,
		"a lane owns one value column; the norms reduce as a REDUCE_THREADS block sum would");
	constexpr uint32_t WARPS = REDUCE_THREADS / LM_WARP_LANES;
	__shared__ __align__(16) float state_s[KEY_DIM][LM_WARP_LANES];
	__shared__ __align__(16) float key_s[LM_DELTA_COLUMN_RING][KEY_DIM];
	__shared__ __align__(16) float query_s[LM_DELTA_COLUMN_RING][KEY_DIM];
	__shared__ __align__(16) float forget_s[LM_DELTA_COLUMN_RING][KEY_DIM];
	__shared__ float value_s[LM_DELTA_COLUMN_RING][LM_WARP_LANES];
	__shared__ float beta_s[LM_DELTA_COLUMN_RING];
	__shared__ float coefficient_s[LM_WARP_LANES];
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t sequence = blockIdx.x, head = blockIdx.y, column_base = blockIdx.z * LM_WARP_LANES;
	uint32_t index, begin, end, ordinal, count, step;
	State *state;
	if ( sequence >= sequences || head >= key_heads || column_base >= VALUE_DIM )
		return;
	state = (State *)(state_pool
		+ ((uint64_t)state_index[sequence] * slot_bytes)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(State)));
	if constexpr ( sizeof(State) == sizeof(float) )
	{
		const uint32_t vec_col = (threadIdx.x & 7u) * 4u;
		#pragma unroll
		for ( index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
			LmLinearCopyAsync(&state_s[index][vec_col],state + (index * VALUE_DIM) + column_base + vec_col,16u);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	else
	{
		for ( index = threadIdx.x; index < KEY_DIM * LM_WARP_LANES; index += LM_DELTA_COLUMN_THREADS )
			state_s[index / LM_WARP_LANES][index % LM_WARP_LANES] = LmScalarToFloat(state[((index / LM_WARP_LANES) * VALUE_DIM) + column_base + (index % LM_WARP_LANES)]);
	}
	LmDependentRelease();
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	count = end > begin ? end - begin : 0u;
	for ( step = 0u; step < 2u && step < count; step++ )
		if ( warp >= 2u )
		{
			ordinal = begin + step;
			LmDeltaRuleColumnPrepare<WARPS,KEY_DIM,VALUE_DIM>(warp - 2u,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,
				sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal,head,key_heads,value_heads_per_key,column_base,lane,
				key_s[step],query_s[step],forget_s[step],value_s[step],&beta_s[step]);
		}
	if constexpr ( sizeof(State) == sizeof(float) )
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	if ( count == 1u && state_loads == 0 && state_saves == 0 )
	{
		if ( warp == 0u )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin] : begin;
			const float coefficient = beta_s[0] * (value_s[0][lane] - LmDeltaRulePredict4<KEY_DIM>(state_s,key_s[0],forget_s[0],lane));
			float total = 0.0f;
			#pragma unroll 4
			for ( index = 0u; index < KEY_DIM; index += 4u )
			{
				const float4 forget4 = *(const float4 *)&forget_s[0][index], key4 = *(const float4 *)&key_s[0][index], query4 = *(const float4 *)&query_s[0][index];
				const float s0 = __fmaf_rn(forget4.x,state_s[index][lane],__fmul_rn(coefficient,key4.x));
				const float s1 = __fmaf_rn(forget4.y,state_s[index + 1u][lane],__fmul_rn(coefficient,key4.y));
				const float s2 = __fmaf_rn(forget4.z,state_s[index + 2u][lane],__fmul_rn(coefficient,key4.z));
				const float s3 = __fmaf_rn(forget4.w,state_s[index + 3u][lane],__fmul_rn(coefficient,key4.w));
				total = __fmaf_rn(s0,query4.x,total);
				total = __fmaf_rn(s1,query4.y,total);
				total = __fmaf_rn(s2,query4.z,total);
				total = __fmaf_rn(s3,query4.w,total);
				if ( commit != 0u )
				{
					state_s[index][lane] = s0;
					state_s[index + 1u][lane] = s1;
					state_s[index + 2u][lane] = s2;
					state_s[index + 3u][lane] = s3;
				}
			}
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + column_base + lane] = LmFloatToBf16(total);
		}
		if ( commit == 0u )
			return;
		__syncthreads();
	}
	else
	{
		if ( warp == 1u && count != 0u )
			coefficient_s[lane] = beta_s[0] * (value_s[0][lane] - LmDeltaRulePredict4<KEY_DIM>(state_s,key_s[0],forget_s[0],lane));
		__syncthreads();
		for ( step = 0u; step < count; step++ )
		{
			const uint32_t buffer = step % LM_DELTA_COLUMN_RING, next = (step + 1u) % LM_DELTA_COLUMN_RING, ahead = (step + 2u) % LM_DELTA_COLUMN_RING;
			const uint32_t current = sequence_row_indices != 0 ? sequence_row_indices[begin + step] : begin + step;
			if ( state_loads != 0 && state_loads[current] != 0u )
			{
				const float *snapshot = snapshots + (((uint64_t)(state_loads[current] - 1u) * key_heads + head) * KEY_DIM) * VALUE_DIM + column_base;
				const uint32_t vec_col = (threadIdx.x & 7u) * 4u;
				#pragma unroll
				for ( index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
					LmLinearCopyAsync(&state_s[index][vec_col],snapshot + (index * VALUE_DIM) + vec_col,16u);
				asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
				__syncthreads();
				if ( warp == 1u )
					coefficient_s[lane] = beta_s[buffer] * (value_s[buffer][lane] - LmDeltaRulePredict4<KEY_DIM>(state_s,key_s[buffer],forget_s[buffer],lane));
				__syncthreads();
			}
			{
				const float coefficient = coefficient_s[lane];
				#pragma unroll
				for ( index = warp * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index < (warp + 1u) * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index += 4u )
				{
					const float4 forget4 = *(const float4 *)&forget_s[buffer][index], key4 = *(const float4 *)&key_s[buffer][index];
					state_s[index][lane] = __fmaf_rn(forget4.x,state_s[index][lane],__fmul_rn(coefficient,key4.x));
					state_s[index + 1u][lane] = __fmaf_rn(forget4.y,state_s[index + 1u][lane],__fmul_rn(coefficient,key4.y));
					state_s[index + 2u][lane] = __fmaf_rn(forget4.z,state_s[index + 2u][lane],__fmul_rn(coefficient,key4.z));
					state_s[index + 3u][lane] = __fmaf_rn(forget4.w,state_s[index + 3u][lane],__fmul_rn(coefficient,key4.w));
				}
			}
			__syncthreads();
			if ( warp == 0u )
			{
				float total = 0.0f;
				#pragma unroll 4
				for ( index = 0u; index < KEY_DIM; index += 4u )
				{
					const float4 query4 = *(const float4 *)&query_s[buffer][index];
					total = __fmaf_rn(state_s[index][lane],query4.x,total);
					total = __fmaf_rn(state_s[index + 1u][lane],query4.y,total);
					total = __fmaf_rn(state_s[index + 2u][lane],query4.z,total);
					total = __fmaf_rn(state_s[index + 3u][lane],query4.w,total);
				}
				output_bf16[(((uint64_t)current * key_heads) + head) * VALUE_DIM + column_base + lane] = LmFloatToBf16(total);
			}
			else if ( warp == 1u && step + 1u < count )
				coefficient_s[lane] = beta_s[next] * (value_s[next][lane] - LmDeltaRulePredict4<KEY_DIM>(state_s,key_s[next],forget_s[next],lane));
			else if ( warp >= 2u && step + 2u < count )
			{
				ordinal = begin + step + 2u;
				LmDeltaRuleColumnPrepare<WARPS,KEY_DIM,VALUE_DIM>(warp - 2u,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,
					sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal,head,key_heads,value_heads_per_key,column_base,lane,
					key_s[ahead],query_s[ahead],forget_s[ahead],value_s[ahead],&beta_s[ahead]);
			}
			__syncthreads();
			if ( state_saves != 0 && state_saves[current] != 0u )
			{
				const uint32_t vec_col = (threadIdx.x & 7u) * 4u;
				for ( uint32_t branch = 0u; branch < LM_DELTA_SNAPSHOT_MAX; branch++ )
					if ( ((state_saves[current] >> branch) & 1u) != 0u )
					{
						float *snapshot = snapshots + (((uint64_t)branch * key_heads + head) * KEY_DIM) * VALUE_DIM + column_base;
						#pragma unroll
						for ( index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
							__stcs((float4 *)(snapshot + (index * VALUE_DIM) + vec_col),*(const float4 *)&state_s[index][vec_col]);
					}
				__syncthreads();
			}
		}
		if ( commit == 0u )
			return;
	}
	if constexpr ( sizeof(State) == sizeof(float) )
	{
		const uint32_t vec_col = (threadIdx.x & 7u) * 4u;
		#pragma unroll
		for ( index = threadIdx.x >> 3u; index < KEY_DIM; index += LM_DELTA_COLUMN_THREADS >> 3u )
			__stcs((float4 *)(state + (index * VALUE_DIM) + column_base + vec_col),*(const float4 *)&state_s[index][vec_col]);
	}
	else
	{
		for ( index = threadIdx.x; index < KEY_DIM * LM_WARP_LANES; index += LM_DELTA_COLUMN_THREADS )
			LmStoreState(&state[((index / LM_WARP_LANES) * VALUE_DIM) + column_base + (index % LM_WARP_LANES)],state_s[index / LM_WARP_LANES][index % LM_WARP_LANES]);
	}
}

#if defined(__CUDACC__)
#define LM_DELTA_CHUNK 16u
#define LM_DELTA_CHUNK_THREADS 512u
#define LM_DELTA_CHUNK_SPLIT 4u

template<uint32_t KEY_DIM, uint32_t COL_SLICE, uint32_t SPLIT = LM_DELTA_CHUNK_SPLIT>
struct LmDeltaRuleChunkShared
{
	static constexpr uint32_t PART_STRIDE = COL_SLICE < 32u ? COL_SLICE + 1u : COL_SLICE;
	__align__(16) float key[LM_DELTA_CHUNK][KEY_DIM + 4u];
	__align__(16) float query[LM_DELTA_CHUNK][KEY_DIM + 4u];
	__align__(16) float carried[LM_DELTA_CHUNK][KEY_DIM + 4u];
	__align__(16) float value[LM_DELTA_CHUNK][COL_SLICE];
	float update[LM_DELTA_CHUNK][COL_SLICE];
	float part[SPLIT][LM_DELTA_CHUNK][PART_STRIDE];
	float tri_a[LM_DELTA_CHUNK][LM_DELTA_CHUNK + 1u];
	float tri_m[LM_DELTA_CHUNK][LM_DELTA_CHUNK + 1u];
	float tri_x[LM_DELTA_CHUNK][LM_DELTA_CHUNK + 1u];
	__align__(16) float tail[KEY_DIM];
	float beta[LM_DELTA_CHUNK];
	__align__(16) uint16_t raw_key[LM_DELTA_CHUNK][KEY_DIM];
	__align__(16) uint16_t raw_query[LM_DELTA_CHUNK][KEY_DIM];
	__align__(16) uint16_t raw_value[LM_DELTA_CHUNK][COL_SLICE];
	__align__(16) float raw_forget[LM_DELTA_CHUNK][KEY_DIM];
	__align__(16) float raw_beta[LM_DELTA_CHUNK];
};

template<uint32_t KEY_DIM, uint32_t VALUE_DIM, uint32_t COL_SLICE, uint32_t SPLIT, uint32_t THREADS>
static __device__ __forceinline__ void LmDeltaRuleChunkFetch(LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT> &sh, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, const uint32_t *__restrict__ sequence_row_indices, uint32_t first_row, uint32_t rows, uint32_t head, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t col_base)
{
	constexpr uint32_t KV = KEY_DIM / 8u, FV = KEY_DIM / 4u, VV = COL_SLICE / 8u, PER = 2u * KV + FV + VV + 1u;
	for ( uint32_t index = threadIdx.x; index < rows * PER; index += THREADS )
	{
		const uint32_t t = index / PER, piece = index % PER;
		const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[first_row + t] : first_row + t;
		const uint64_t base = (((uint64_t)row * key_heads) + head) * KEY_DIM;
		if ( piece < KV )
			LmLinearCopyAsync(&sh.raw_key[t][piece * 8u],key_bf16 + base + piece * 8u,16u);
		else if ( piece < 2u * KV )
			LmLinearCopyAsync(&sh.raw_query[t][(piece - KV) * 8u],query_bf16 + base + (piece - KV) * 8u,16u);
		else if ( piece < 2u * KV + FV )
			LmLinearCopyAsync(&sh.raw_forget[t][(piece - 2u * KV) * 4u],forget_gate + base + (piece - 2u * KV) * 4u,16u);
		else if ( piece < PER - 1u )
			LmLinearCopyAsync(&sh.raw_value[t][(piece - 2u * KV - FV) * 8u],
				value_bf16 + (((uint64_t)row * key_heads * value_heads_per_key) + (uint64_t)head * value_heads_per_key) * VALUE_DIM + col_base + (piece - 2u * KV - FV) * 8u,16u);
		else
			LmLinearCopyAsync(&sh.raw_beta[t],write_gate + (row * key_heads) + head,4u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<uint32_t KEY_DIM, uint32_t PART_STRIDE, uint32_t SPLIT, uint32_t HALF>
static __device__ __forceinline__ void LmDeltaRuleChunkDot2(const float (&vectors)[LM_DELTA_CHUNK][KEY_DIM + 4u], const float (&columns)[HALF], float (&part)[SPLIT][LM_DELTA_CHUNK][PART_STRIDE], uint32_t half, uint32_t key_first, uint32_t col)
{
	#pragma unroll
	for ( uint32_t r = 0u; r < LM_DELTA_CHUNK; r += 2u )
	{
		float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
		#pragma unroll
		for ( uint32_t index = 0u; index < HALF; index += 4u )
		{
			const float4 v0 = *(const float4 *)&vectors[r][key_first + index], v1 = *(const float4 *)&vectors[r + 1u][key_first + index];
			a0 = fmaf(v0.x,columns[index],a0); b0 = fmaf(v1.x,columns[index],b0);
			a1 = fmaf(v0.y,columns[index + 1u],a1); b1 = fmaf(v1.y,columns[index + 1u],b1);
			a2 = fmaf(v0.z,columns[index + 2u],a2); b2 = fmaf(v1.z,columns[index + 2u],b2);
			a3 = fmaf(v0.w,columns[index + 3u],a3); b3 = fmaf(v1.w,columns[index + 3u],b3);
		}
		part[half][r][col] = (a0 + a1) + (a2 + a3);
		part[half][r + 1u][col] = (b0 + b1) + (b2 + b3);
	}
}

template<uint32_t KEY_DIM, uint32_t COL_SLICE, uint32_t SPLIT, uint32_t THREADS, uint32_t SIZE, uint32_t LANES>
static __device__ __forceinline__ void LmDeltaRuleChunkMerge(LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT> &sh, uint32_t rows)
{
	constexpr uint32_t C = LM_DELTA_CHUNK, TWICE = 2u * SIZE, PAIRS = C * SIZE / 2u, VECS = (KEY_DIM / 4u) / LANES, KV4 = KEY_DIM / 4u;
	static_assert(((KEY_DIM / LANES) % 4u) == 0u && PAIRS * LANES <= THREADS,"a lane of a pair team holds whole float4 spans");
	const uint32_t pair = threadIdx.x / LANES, lane = threadIdx.x % LANES;
	uint32_t index;
	if ( pair < PAIRS )
	{
		const uint32_t block = pair / (SIZE * SIZE), at = pair % (SIZE * SIZE);
		const uint32_t j = block * TWICE + at / SIZE, t = block * TWICE + SIZE + at % SIZE;
		float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f, sum_a, sum_m;
		if ( t < rows )
		{
			#pragma unroll
			for ( index = 0u; index < VECS; index++ )
			{
				const uint32_t c = (index * LANES + lane) * 4u;
				const float4 w4 = *(const float4 *)&sh.carried[j][c];
				const float4 k4 = *(const float4 *)&sh.key[t][c];
				const float4 q4 = *(const float4 *)&sh.query[t][c];
				a0 = fmaf(k4.x,w4.x,a0);
				a1 = fmaf(k4.y,w4.y,a1);
				a2 = fmaf(k4.z,w4.z,a2);
				a3 = fmaf(k4.w,w4.w,a3);
				m0 = fmaf(q4.x,w4.x,m0);
				m1 = fmaf(q4.y,w4.y,m1);
				m2 = fmaf(q4.z,w4.z,m2);
				m3 = fmaf(q4.w,w4.w,m3);
			}
		}
		sum_a = (a0 + a1) + (a2 + a3);
		sum_m = (m0 + m1) + (m2 + m3);
		#pragma unroll
		for ( index = 1u; index < LANES; index *= 2u )
		{
			sum_a += __shfl_xor_sync(0xffffffffu,sum_a,index);
			sum_m += __shfl_xor_sync(0xffffffffu,sum_m,index);
		}
		if ( lane == 0u && t < rows )
		{
			sh.tri_a[t][j] = sh.beta[t] * sum_a;
			sh.tri_m[t][j] = sum_m;
		}
	}
	__syncthreads();
	for ( index = threadIdx.x; index < (C / 2u) * KV4; index += THREADS )
	{
		const uint32_t half = index / KV4, dim = (index % KV4) * 4u, first = (half / SIZE) * TWICE, left = first + half % SIZE, right = left + SIZE;
		const float4 fr = *(const float4 *)&sh.raw_forget[first + SIZE][dim];
		const float4 fl = *(const float4 *)&sh.raw_forget[first][dim];
		float4 c4 = *(const float4 *)&sh.carried[left][dim];
		float4 k4 = *(const float4 *)&sh.key[right][dim];
		float4 q4 = *(const float4 *)&sh.query[right][dim];
		c4.x *= fr.x; c4.y *= fr.y; c4.z *= fr.z; c4.w *= fr.w;
		k4.x *= fl.x; k4.y *= fl.y; k4.z *= fl.z; k4.w *= fl.w;
		q4.x *= fl.x; q4.y *= fl.y; q4.z *= fl.z; q4.w *= fl.w;
		*(float4 *)&sh.carried[left][dim] = c4;
		*(float4 *)&sh.key[right][dim] = k4;
		*(float4 *)&sh.query[right][dim] = q4;
	}
	if constexpr ( TWICE == C )
	{
		if ( threadIdx.x < KV4 )
		{
			const uint32_t dim = threadIdx.x * 4u;
			const float4 f0 = *(const float4 *)&sh.raw_forget[0][dim];
			const float4 f1 = *(const float4 *)&sh.raw_forget[SIZE][dim];
			*(float4 *)&sh.tail[dim] = {f0.x * f1.x,f0.y * f1.y,f0.z * f1.z,f0.w * f1.w};
		}
		__syncthreads();
	}
	else
	{
		__syncthreads();
		for ( index = threadIdx.x; index < (C / TWICE) * KV4; index += THREADS )
		{
			const uint32_t row = (index / KV4) * TWICE, dim = (index % KV4) * 4u;
			const float4 f0 = *(const float4 *)&sh.raw_forget[row][dim];
			const float4 f1 = *(const float4 *)&sh.raw_forget[row + SIZE][dim];
			*(float4 *)&sh.raw_forget[row][dim] = {f0.x * f1.x,f0.y * f1.y,f0.z * f1.z,f0.w * f1.w};
		}
		__syncthreads();
	}
}

template<uint32_t SPLIT, uint32_t PART_STRIDE>
static __device__ __forceinline__ float LmDeltaRuleChunkPartSum(const float (&part)[SPLIT][LM_DELTA_CHUNK][PART_STRIDE], uint32_t r, uint32_t col)
{
	if constexpr ( SPLIT == 4u )
		return((part[0][r][col] + part[1][r][col]) + (part[2][r][col] + part[3][r][col]));
	else if constexpr ( SPLIT == 8u )
		return(((part[0][r][col] + part[1][r][col]) + (part[2][r][col] + part[3][r][col])) +
			((part[4][r][col] + part[5][r][col]) + (part[6][r][col] + part[7][r][col])));
	else
	{
		float s0 = 0.0f, s1 = 0.0f;
		#pragma unroll
		for ( uint32_t s = 0u; s < SPLIT; s += 2u )
		{
			s0 += part[s][r][col];
			s1 += part[s + 1u][r][col];
		}
		return(s0 + s1);
	}
}

template<uint32_t REDUCE_THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float, uint32_t COL_SLICE = VALUE_DIM, uint32_t THREADS = LM_DELTA_CHUNK_THREADS>
__global__ __launch_bounds__(THREADS)
void LmDeltaRuleChunkKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0)
{
	constexpr uint32_t C = LM_DELTA_CHUNK, TEAM = THREADS / LM_WARP_LANES;
	constexpr uint32_t SPLIT = THREADS / COL_SLICE, HALF = KEY_DIM / SPLIT, ROWS_PER_HALF = C / SPLIT;
	static_assert(COL_SLICE * SPLIT == THREADS && (VALUE_DIM % COL_SLICE) == 0u && (KEY_DIM % (4u * SPLIT)) == 0u && (COL_SLICE % 8u) == 0u && (C % SPLIT) == 0u && (C % TEAM) == 0u,
		"each thread holds a value column of a key slice");
	extern __shared__ __align__(16) uint8_t lm_delta_chunk_shared[];
	LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT> &sh = *(LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT> *)lm_delta_chunk_shared;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	const uint32_t col = threadIdx.x % COL_SLICE, half = threadIdx.x / COL_SLICE, key_first = half * HALF;
	const uint32_t sequence = blockIdx.x, head = blockIdx.y, col_base = blockIdx.z * COL_SLICE, gcol = col_base + col;
	float columns[HALF];
	uint32_t index, begin, end, count, first, t, j, r;
	State *state;
	if ( sequence >= sequences || head >= key_heads || gcol >= VALUE_DIM )
		return;
	LmDependentRelease();
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	count = end > begin ? end - begin : 0u;
	state = (State *)(state_pool + ((uint64_t)state_index[sequence] * slot_bytes) + ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(State)));
	if ( count != 0u )
		LmDeltaRuleChunkFetch<KEY_DIM,VALUE_DIM,COL_SLICE,SPLIT,THREADS>(sh,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,sequence_row_indices,begin,count < C ? count : C,head,key_heads,value_heads_per_key,col_base);
	#pragma unroll
	for ( index = 0u; index < HALF; index++ )
	{
		if constexpr ( sizeof(State) == sizeof(float) )
			columns[index] = __ldcs((const float *)state + ((key_first + index) * VALUE_DIM) + gcol);
		else
			columns[index] = LmScalarToFloat(state[((key_first + index) * VALUE_DIM) + gcol]);
	}
	for ( first = 0u; first < count; first += C )
	{
		const uint32_t rows = count - first < C ? count - first : C;
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			const uint32_t c0 = lane * 4u;
			#pragma unroll
			for ( uint32_t row_w = warp; row_w < C; row_w += TEAM )
			{
				if ( row_w < rows )
				{
					const uint2 rk = *(const uint2 *)&sh.raw_key[row_w][c0];
					const uint2 rq = *(const uint2 *)&sh.raw_query[row_w][c0];
					const float4 fg = *(const float4 *)&sh.raw_forget[row_w][c0];
					const float ka = __uint_as_float(rk.x << 16u), kb = __uint_as_float(rk.x & 0xffff0000u);
					const float kc = __uint_as_float(rk.y << 16u), kd = __uint_as_float(rk.y & 0xffff0000u);
					const float qa = __uint_as_float(rq.x << 16u), qb = __uint_as_float(rq.x & 0xffff0000u);
					const float qc = __uint_as_float(rq.y << 16u), qd = __uint_as_float(rq.y & 0xffff0000u);
					const float kinv = rsqrtf(LmLinearWarpSum4(ka * ka,kb * kb,kc * kc,kd * kd) + 1e-6f);
					const float qinv = rsqrtf(LmLinearWarpSum4(qa * qa,qb * qb,qc * qc,qd * qd) + 1e-6f) * rsqrtf((float)KEY_DIM);
					const float nka = ka * kinv, nkb = kb * kinv, nkc = kc * kinv, nkd = kd * kinv;
					const float nqa = qa * qinv, nqb = qb * qinv, nqc = qc * qinv, nqd = qd * qinv;
					const float diag = LmLinearWarpSum4(nka * nqa,nkb * nqb,nkc * nqc,nkd * nqd);
					*(float4 *)&sh.carried[row_w][c0] = {nka,nkb,nkc,nkd};
					*(float4 *)&sh.key[row_w][c0] = {nka * fg.x,nkb * fg.y,nkc * fg.z,nkd * fg.w};
					*(float4 *)&sh.query[row_w][c0] = {nqa * fg.x,nqb * fg.y,nqc * fg.z,nqd * fg.w};
					if ( lane == 0u )
						sh.tri_m[row_w][row_w] = diag;
					if constexpr ( COL_SLICE == KEY_DIM )
					{
						const uint2 rv = *(const uint2 *)&sh.raw_value[row_w][c0];
						*(float4 *)&sh.value[row_w][c0] = {__uint_as_float(rv.x << 16u),__uint_as_float(rv.x & 0xffff0000u),__uint_as_float(rv.y << 16u),__uint_as_float(rv.y & 0xffff0000u)};
					}
				}
				else
				{
					const float4 z4 = {0.0f,0.0f,0.0f,0.0f};
					*(float4 *)&sh.raw_forget[row_w][c0] = {1.0f,1.0f,1.0f,1.0f};
					*(float4 *)&sh.carried[row_w][c0] = z4;
					*(float4 *)&sh.key[row_w][c0] = z4;
					*(float4 *)&sh.query[row_w][c0] = z4;
				}
			}
		}
		if constexpr ( COL_SLICE != KEY_DIM )
			for ( index = threadIdx.x; index < rows * COL_SLICE; index += THREADS )
				sh.value[index / COL_SLICE][index % COL_SLICE] = LmBf16ToFloat(sh.raw_value[index / COL_SLICE][index % COL_SLICE]);
		if ( threadIdx.x < C )
			sh.beta[threadIdx.x] = threadIdx.x < rows ? sh.raw_beta[threadIdx.x] : 0.0f;
		__syncthreads();
		if constexpr ( THREADS >= 512u )
		{
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,1u,32u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,2u,32u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,4u,16u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,8u,8u>(sh,rows);
		}
		else
		{
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,1u,32u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,2u,16u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,4u,8u>(sh,rows);
			LmDeltaRuleChunkMerge<KEY_DIM,COL_SLICE,SPLIT,THREADS,8u,4u>(sh,rows);
		}
		if ( first + C < count )
			LmDeltaRuleChunkFetch<KEY_DIM,VALUE_DIM,COL_SLICE,SPLIT,THREADS>(sh,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,sequence_row_indices,begin + first + C,
				count - first - C < C ? count - first - C : C,head,key_heads,value_heads_per_key,col_base);
		if ( warp == 0u && lane < C )
		{
			float x[C];
			#pragma unroll
			for ( t = 0u; t < C; t++ )
			{
				float total = t == lane ? sh.beta[t] : 0.0f;
				#pragma unroll
				for ( j = 0u; j < t; j++ )
					total = fmaf(-sh.tri_a[t][j],x[j],total);
				x[t] = t < rows ? total : 0.0f;
			}
			#pragma unroll
			for ( t = 0u; t < C; t++ )
				sh.tri_x[t][lane] = x[t];
		}
		LmDeltaRuleChunkDot2<KEY_DIM,LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT>::PART_STRIDE,SPLIT,HALF>(sh.key,columns,sh.part,half,key_first,col);
		__syncthreads();
		for ( r = half * ROWS_PER_HALF; r < (half + 1u) * ROWS_PER_HALF; r++ )
			sh.value[r][col] = r < rows ? sh.value[r][col] - LmDeltaRuleChunkPartSum<SPLIT,LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT>::PART_STRIDE>(sh.part,r,col) : 0.0f;
		__syncthreads();
		for ( r = half * ROWS_PER_HALF; r < (half + 1u) * ROWS_PER_HALF; r++ )
		{
			float u = 0.0f;
			for ( j = 0u; j <= r && j < rows; j++ )
				u = fmaf(sh.tri_x[r][j],sh.value[j][col],u);
			sh.update[r][col] = r < rows ? u : 0.0f;
		}
		LmDeltaRuleChunkDot2<KEY_DIM,LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT>::PART_STRIDE,SPLIT,HALF>(sh.query,columns,sh.part,half,key_first,col);
		__syncthreads();
		for ( r = half * ROWS_PER_HALF; r < (half + 1u) * ROWS_PER_HALF && r < rows; r++ )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + first + r] : begin + first + r;
			float o = LmDeltaRuleChunkPartSum<SPLIT,LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT>::PART_STRIDE>(sh.part,r,col);
			for ( j = 0u; j <= r; j++ )
				o = fmaf(sh.tri_m[r][j],sh.update[j][col],o);
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + gcol] = LmFloatToBf16(o);
		}
		if ( first + C >= count && commit == 0u )
			break;
		#pragma unroll
		for ( index = 0u; index < HALF; index += 4u )
		{
			const float4 t4 = *(const float4 *)&sh.tail[key_first + index];
			columns[index] *= t4.x;
			columns[index + 1u] *= t4.y;
			columns[index + 2u] *= t4.z;
			columns[index + 3u] *= t4.w;
		}
		#pragma unroll
		for ( r = 0u; r < C; r += 2u )
		{
			const float u0 = sh.update[r][col], u1 = sh.update[r + 1u][col];
			#pragma unroll
			for ( index = 0u; index < HALF; index += 4u )
			{
				const float4 e0 = *(const float4 *)&sh.carried[r][key_first + index];
				const float4 e1 = *(const float4 *)&sh.carried[r + 1u][key_first + index];
				columns[index] = fmaf(e1.x,u1,fmaf(e0.x,u0,columns[index]));
				columns[index + 1u] = fmaf(e1.y,u1,fmaf(e0.y,u0,columns[index + 1u]));
				columns[index + 2u] = fmaf(e1.z,u1,fmaf(e0.z,u0,columns[index + 2u]));
				columns[index + 3u] = fmaf(e1.w,u1,fmaf(e0.w,u0,columns[index + 3u]));
			}
		}
	}
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	if ( commit == 0u )
		return;
	#pragma unroll
	for ( index = 0u; index < HALF; index++ )
	{
		if constexpr ( sizeof(State) == sizeof(float) )
			__stcs((float *)state + ((key_first + index) * VALUE_DIM) + gcol,columns[index]);
		else
			LmStoreState(&state[((key_first + index) * VALUE_DIM) + gcol],columns[index]);
	}
}

template<uint32_t REDUCE_THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float, uint32_t COL_SLICE = VALUE_DIM, uint32_t THREADS = LM_DELTA_CHUNK_THREADS>
static inline SparkGpuError LmDeltaRuleChunkDispatch(uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint16_t *query_bf16, const uint16_t *key_bf16, const uint16_t *value_bf16, const float *forget_gate, const float *write_gate, uint16_t *output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream)
{
	constexpr uint32_t SPLIT = THREADS / COL_SLICE;
	const uint32_t shared = (uint32_t)sizeof(LmDeltaRuleChunkShared<KEY_DIM,COL_SLICE,SPLIT>);
	static bool granted = false;
	if ( !granted )
	{
		const SparkGpuError error = SparkGpuFuncSetAttribute(LmDeltaRuleChunkKernel<REDUCE_THREADS,KEY_DIM,VALUE_DIM,State,COL_SLICE,THREADS>,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared);
		if ( error != SPARK_GPU_SUCCESS )
			return(error);
		granted = true;
	}
	LM_LAUNCH((LmDeltaRuleChunkKernel<REDUCE_THREADS,KEY_DIM,VALUE_DIM,State,COL_SLICE,THREADS>),dim3(sequences,key_heads,VALUE_DIM / COL_SLICE),THREADS,shared,stream,state_pool,slot_bytes,state_index,
		sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,forget_gate,write_gate,output_bf16,key_heads,value_heads_per_key,sequences,commit,sequence_row_indices);
	return(SparkGpuPeekAtLastError());
}

template<uint32_t REDUCE_THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
static inline SparkGpuError LmDeltaRuleChunkLaunch(uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint16_t *query_bf16, const uint16_t *key_bf16, const uint16_t *value_bf16, const float *forget_gate, const float *write_gate, uint16_t *output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream)
{
	if ( sequences == 0u || key_heads == 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	if constexpr ( (VALUE_DIM % 16u) == 0u && VALUE_DIM > 16u && (KEY_DIM % 64u) == 0u )
	{
		if ( (uint64_t)sequences * key_heads <= 12u )
			return(LmDeltaRuleChunkDispatch<REDUCE_THREADS,KEY_DIM,VALUE_DIM,State,16u,256u>(state_pool,slot_bytes,state_index,sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,
				forget_gate,write_gate,output_bf16,key_heads,value_heads_per_key,sequences,commit,sequence_row_indices,stream));
	}
	if constexpr ( (VALUE_DIM % 32u) == 0u && VALUE_DIM > 32u && (KEY_DIM % 32u) == 0u )
	{
		if ( (uint64_t)sequences * key_heads <= 24u )
			return(LmDeltaRuleChunkDispatch<REDUCE_THREADS,KEY_DIM,VALUE_DIM,State,32u,256u>(state_pool,slot_bytes,state_index,sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,
				forget_gate,write_gate,output_bf16,key_heads,value_heads_per_key,sequences,commit,sequence_row_indices,stream));
	}
	return(LmDeltaRuleChunkDispatch<REDUCE_THREADS,KEY_DIM,VALUE_DIM,State,VALUE_DIM,LM_DELTA_CHUNK_THREADS>(state_pool,slot_bytes,state_index,sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,
		forget_gate,write_gate,output_bf16,key_heads,value_heads_per_key,sequences,commit,sequence_row_indices,stream));
}
#endif

static inline uint64_t LmDeltaRuleSnapshotBytes(uint32_t key_heads, uint32_t key_dimension, uint32_t value_dimension)
{
	return((uint64_t)LM_DELTA_SNAPSHOT_MAX * key_heads * key_dimension * value_dimension * sizeof(float));
}

template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
static inline SparkGpuError LmDeltaRuleLaunch(uint32_t rows, uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint16_t *query_bf16, const uint16_t *key_bf16, const uint16_t *value_bf16, const float *forget_gate, const float *write_gate, uint16_t *output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream, const uint32_t *state_loads = 0, const uint32_t *state_saves = 0, float *snapshots = 0)
{
#if defined(__CUDACC__)
	if ( commit != 0u && sequence_row_count == 0 && rows >= sequences * LM_DELTA_CHUNK && state_loads == 0 && state_saves == 0 )
		return(LmDeltaRuleChunkLaunch<THREADS,KEY_DIM,VALUE_DIM,State>(state_pool,slot_bytes,state_index,sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,
			forget_gate,write_gate,output_bf16,key_heads,value_heads_per_key,sequences,commit,sequence_row_indices,stream));
#endif
	LM_LAUNCH((LmDeltaRuleColumnKernel<THREADS,KEY_DIM,VALUE_DIM,State>), dim3(sequences,key_heads,VALUE_DIM / LM_WARP_LANES), LM_DELTA_COLUMN_THREADS, 0, stream,
		state_pool,slot_bytes,state_index,sequence_row_begin,sequence_row_count,query_bf16,key_bf16,value_bf16,forget_gate,write_gate,output_bf16,key_heads,
		value_heads_per_key,sequences,commit,sequence_row_indices,state_loads,state_saves,snapshots);
	return(SparkGpuPeekAtLastError());
}

enum LmConvActivation
{
	LM_CONV_NONE = 0,
	LM_CONV_SWISH = 1,
	LM_CONV_SWISH_ONCE = 2
};

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight, uint32_t DILATION = 1u>
static __device__ __forceinline__ void LmCausalConvBody(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ parent_rows = 0)
{
	uint32_t sequence = blockIdx.x,channel = (blockIdx.y * THREADS) + threadIdx.x;
	uint32_t begin,end,row,tap,ordinal;
	constexpr uint32_t SPAN = (KERNEL - 1u) * DILATION + 1u;
	uint16_t taps[SPAN];
	uint16_t *slot;
	if ( sequence >= sequences || channel >= channels )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	slot = window + ((uint64_t)state_index[sequence] * channels * SPAN);
	for (tap = 0u; tap < SPAN; ++tap)
		taps[tap] = slot[(channel * SPAN) + tap];
	for (ordinal = begin; ordinal < end; ++ordinal)
	{
		row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
		float total = 0.0f;
		if ( parent_rows != 0 )
		{
			uint32_t ancestor = row, missing = SPAN;
			for (tap = 0u; tap < SPAN; ++tap)
			{
				if ( ancestor != 0xffffffffu )
				{
					taps[SPAN - 1u - tap] = input_bf16[((uint64_t)ancestor * channels) + channel];
					ancestor = parent_rows[ancestor];
				}
				else
				{
					missing = missing == SPAN ? tap : missing;
					taps[SPAN - 1u - tap] = slot[(channel * SPAN) + SPAN - tap + missing - 1u];
				}
			}
		}
		else
		{
			for (tap = 0u; tap + 1u < SPAN; ++tap)
				taps[tap] = taps[tap + 1u];
			taps[SPAN - 1u] = input_bf16[((uint64_t)row * channels) + channel];
		}
		for (tap = 0u; tap < KERNEL; ++tap)
			total += LmBf16ToFloat(taps[tap * DILATION])
				* LmScalarToFloat(weight[(channel * KERNEL) + tap]);
		if ( ACTIVATION == LM_CONV_SWISH )
		{
			if ( sizeof(Weight) == sizeof(uint16_t) )
				total = LmBf16ToFloat(LmFloatToBf16(total));
			total = SparkMathSwishFast(total);
		}
		if ( ACTIVATION == LM_CONV_SWISH_ONCE )
			total = SparkMathSwishFast(total);
		output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
	}
	if ( commit == 0u || parent_rows != 0 )
		return;
	for (tap = 0u; tap < SPAN; ++tap)
		slot[(channel * SPAN) + tap] = taps[tap];
}

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight, uint32_t DILATION = 1u>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0, const uint32_t *__restrict__ parent_rows = 0)
{
	LmCausalConvBody<THREADS,KERNEL,ACTIVATION,Weight,DILATION>(window,state_index,sequence_row_begin,sequence_row_count,input_bf16,weight,output_bf16,channels,sequences,commit,sequence_row_indices,parent_rows);
}

#define LM_CAUSAL_CONV_STREAMS 3u

template<class Weight>
struct LmCausalConvStreams
{
	uint16_t *window[LM_CAUSAL_CONV_STREAMS];
	const uint16_t *input_bf16[LM_CAUSAL_CONV_STREAMS];
	const Weight *weight[LM_CAUSAL_CONV_STREAMS];
	uint16_t *output_bf16[LM_CAUSAL_CONV_STREAMS];
	uint32_t channels[LM_CAUSAL_CONV_STREAMS];
};

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvStreamsKernel(const __grid_constant__ LmCausalConvStreams<Weight> streams, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ parent_rows = 0)
{
	const uint32_t stream = blockIdx.z;
	LmCausalConvBody<THREADS,KERNEL,ACTIVATION,Weight>(streams.window[stream],state_index,sequence_row_begin,sequence_row_count,streams.input_bf16[stream],streams.weight[stream],streams.output_bf16[stream],streams.channels[stream],sequences,commit,sequence_row_indices,parent_rows);
}

#define LM_CAUSAL_CONV_TILE_THREADS 256u
#define LM_CAUSAL_CONV_TILE_CHANNELS 32u
#define LM_CAUSAL_CONV_TILE_ROWS 64u
#define LM_CAUSAL_CONV_SERIAL_ROWS 4u

template<uint32_t KERNEL, uint32_t ACTIVATION, class Weight, uint32_t DILATION>
static __device__ __forceinline__ void LmCausalConvTiledBody(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *input_bf16, const Weight *__restrict__ weight, uint16_t *output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices)
{
	constexpr uint32_t SPAN = (KERNEL - 1u) * DILATION + 1u, HALO = SPAN - 1u;
	constexpr uint32_t GROUPS = LM_CAUSAL_CONV_TILE_THREADS / LM_CAUSAL_CONV_TILE_CHANNELS;
	__shared__ uint16_t tile[HALO + LM_CAUSAL_CONV_TILE_ROWS][LM_CAUSAL_CONV_TILE_CHANNELS];
	const uint32_t sequence = blockIdx.x, lane = threadIdx.x % LM_CAUSAL_CONV_TILE_CHANNELS, group = threadIdx.x / LM_CAUSAL_CONV_TILE_CHANNELS;
	const uint32_t channel = (blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS) + lane;
	const bool live = channel < channels;
	uint32_t begin, end, base, count, offset, tap, row;
	float taps_weight[KERNEL];
	uint16_t carried[HALO > 0u ? HALO : 1u];
	uint16_t *slot;
	if ( sequence >= sequences )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	slot = window + ((uint64_t)state_index[sequence] * channels * SPAN);
	for (tap = 0u; tap < KERNEL; ++tap)
		taps_weight[tap] = live ? LmScalarToFloat(weight[(channel * KERNEL) + tap]) : 0.0f;
	if ( group == 0u && live )
		for (tap = 0u; tap < HALO; ++tap)
			tile[tap][lane] = slot[(channel * SPAN) + 1u + tap];
	for (base = begin; base < end; base += LM_CAUSAL_CONV_TILE_ROWS)
	{
		count = end - base < LM_CAUSAL_CONV_TILE_ROWS ? end - base : LM_CAUSAL_CONV_TILE_ROWS;
		__syncthreads();
		for (offset = group; offset < count && live; offset += GROUPS)
		{
			row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			tile[HALO + offset][lane] = input_bf16[((uint64_t)row * channels) + channel];
		}
		__syncthreads();
		for (offset = group; offset < count && live; offset += GROUPS)
		{
			float total = 0.0f;
			row = sequence_row_indices != 0 ? sequence_row_indices[base + offset] : base + offset;
			for (tap = 0u; tap < KERNEL; ++tap)
				total += LmBf16ToFloat(tile[offset + tap * DILATION][lane]) * taps_weight[tap];
			if ( ACTIVATION == LM_CONV_SWISH )
			{
				if ( sizeof(Weight) == sizeof(uint16_t) )
					total = LmBf16ToFloat(LmFloatToBf16(total));
				total = SparkMathSwishFast(total);
			}
			if ( ACTIVATION == LM_CONV_SWISH_ONCE )
				total = SparkMathSwishFast(total);
			output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
		}
		__syncthreads();
		if ( group == 0u && live )
		{
			if ( commit != 0u && base + count == end )
				for (tap = 0u; tap < SPAN; ++tap)
					slot[(channel * SPAN) + tap] = tile[count - 1u + tap][lane];
			for (tap = 0u; tap < HALO; ++tap)
				carried[tap] = tile[count + tap][lane];
			for (tap = 0u; tap < HALO; ++tap)
				tile[tap][lane] = carried[tap];
		}
	}
}

template<uint32_t KERNEL, uint32_t ACTIVATION, class Weight, uint32_t DILATION = 1u>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 1)
void LmCausalConvTiledKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *input_bf16, const Weight *__restrict__ weight, uint16_t *output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0)
{
	LmCausalConvTiledBody<KERNEL,ACTIVATION,Weight,DILATION>(window,state_index,sequence_row_begin,sequence_row_count,input_bf16,weight,output_bf16,channels,sequences,commit,sequence_row_indices);
}

template<uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
__global__ __launch_bounds__(LM_CAUSAL_CONV_TILE_THREADS, 1)
void LmCausalConvStreamsTiledKernel(const __grid_constant__ LmCausalConvStreams<Weight> streams, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices)
{
	const uint32_t stream = blockIdx.z;
	if ( blockIdx.y * LM_CAUSAL_CONV_TILE_CHANNELS >= streams.channels[stream] )
		return;
	LmCausalConvTiledBody<KERNEL,ACTIVATION,Weight,1u>(streams.window[stream],state_index,sequence_row_begin,sequence_row_count,streams.input_bf16[stream],streams.weight[stream],streams.output_bf16[stream],streams.channels[stream],sequences,commit,sequence_row_indices);
}

template<uint32_t KERNEL, uint32_t ACTIVATION, class Weight, uint32_t DILATION = 1u>
static SparkGpuError LmCausalConvLaunch(uint16_t *window, const uint32_t *state_index, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint16_t *input_bf16, const Weight *weight, uint16_t *output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, uint32_t rows, SparkGpuStream stream, const uint32_t *parent_rows = 0)
{
	if ( parent_rows != 0 || rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
		LM_LAUNCH((LmCausalConvKernel<LM_CAUSAL_CONV_TILE_THREADS,KERNEL,ACTIVATION,Weight,DILATION>), dim3(sequences, (channels + LM_CAUSAL_CONV_TILE_THREADS - 1u) / LM_CAUSAL_CONV_TILE_THREADS),
			LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, state_index, sequence_row_begin, sequence_row_count, input_bf16, weight, output_bf16, channels, sequences,
			commit, sequence_row_indices, parent_rows);
	else
		LM_LAUNCH((LmCausalConvTiledKernel<KERNEL,ACTIVATION,Weight,DILATION>), dim3(sequences, (channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
		LM_CAUSAL_CONV_TILE_THREADS, 0, stream, window, state_index, sequence_row_begin, sequence_row_count, input_bf16, weight, output_bf16, channels, sequences,
		commit, sequence_row_indices);
	return SparkGpuPeekAtLastError();
}

template<uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
static SparkGpuError LmCausalConvStreamsLaunch(const LmCausalConvStreams<Weight> &streams, const uint32_t *state_index, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, uint32_t rows, SparkGpuStream stream, const uint32_t *parent_rows = 0)
{
	uint32_t widest = 0u;
	for (uint32_t index = 0u; index < LM_CAUSAL_CONV_STREAMS; ++index)
		widest = streams.channels[index] > widest ? streams.channels[index] : widest;
	if ( parent_rows != 0 || rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
		LM_LAUNCH((LmCausalConvStreamsKernel<LM_CAUSAL_CONV_TILE_THREADS,KERNEL,ACTIVATION,Weight>), dim3(sequences, (widest + LM_CAUSAL_CONV_TILE_THREADS - 1u) / LM_CAUSAL_CONV_TILE_THREADS, LM_CAUSAL_CONV_STREAMS),
			LM_CAUSAL_CONV_TILE_THREADS, 0, stream, streams, state_index, sequence_row_begin, sequence_row_count, sequences, commit, sequence_row_indices, parent_rows);
	else
		LM_LAUNCH((LmCausalConvStreamsTiledKernel<KERNEL,ACTIVATION,Weight>), dim3(sequences, (widest + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS, LM_CAUSAL_CONV_STREAMS),
		LM_CAUSAL_CONV_TILE_THREADS, 0, stream, streams, state_index, sequence_row_begin, sequence_row_count, sequences, commit, sequence_row_indices);
	return SparkGpuPeekAtLastError();
}
