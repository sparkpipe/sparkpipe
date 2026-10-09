#pragma once


#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

static __device__ __forceinline__ float LmBoundedDecay(float logit, float bias, float head_log_scale, float minimum_log_decay)
{
	float scaled = __expf(head_log_scale) * (logit + bias);
	float log_decay = minimum_log_decay * (1.0f / (1.0f + __expf(-scaled)));
	return(__expf(log_decay));
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
void LmGdnGateKernel(const uint16_t *__restrict__ decay_logit_bf16, const uint16_t *__restrict__ beta_logit_bf16, const float *__restrict__ head_log_scale, const float *__restrict__ head_bias, float *__restrict__ retention, float *__restrict__ write_gate, uint32_t heads, uint32_t rows)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t scalar;
	float shifted,softplus,factor;
	if ( row >= rows || head >= heads )
		return;
	scalar = ((uint64_t)row * heads) + head;
	shifted = LmBf16ToFloat(decay_logit_bf16[scalar]) + head_bias[head];
	softplus = shifted > 20.0f ? shifted : __logf(1.0f + __expf(shifted));
	factor = __expf(-__expf(head_log_scale[head]) * softplus);
	for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		retention[(scalar * KEY_DIM) + index] = factor;
	if ( threadIdx.x == 0u )
	{
		float beta_logit = LmBf16ToFloat(beta_logit_bf16[scalar]);
		write_gate[scalar] = 1.0f / (1.0f + __expf(-beta_logit));
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
		parts[part] = LmBf16ToFloat(source[base + part * LM_WARP_LANES + lane]);
		squares[part] = 0.0f;
		squares[part] += parts[part] * parts[part];
	}
	if ( query_side != 0u )
		value_s[lane] = LmBf16ToFloat(value_bf16[(((uint64_t)row * key_heads * value_heads_per_key) + (uint64_t)head * value_heads_per_key) * VALUE_DIM + column + lane]);
	else
	{
		#pragma unroll
		for ( part = 0u; part < PARTS; part++ )
			forget_s[part * LM_WARP_LANES + lane] = forget_gate[base + part * LM_WARP_LANES + lane];
		if ( lane == 0u )
			*beta_s = write_gate[(row * key_heads) + head];
	}
	inverse = rsqrtf(LmDeltaRuleWarpTotal<WARPS,PARTS>(squares,lane) + 1e-6f);
	#pragma unroll
	for ( part = 0u; part < PARTS; part++ )
		if ( query_side != 0u )
			query_s[part * LM_WARP_LANES + lane] = parts[part] * (inverse * rsqrtf((float)KEY_DIM));
		else
			key_s[part * LM_WARP_LANES + lane] = parts[part] * inverse;
}

template<uint32_t REDUCE_THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
__global__ __launch_bounds__(LM_DELTA_COLUMN_THREADS)
void LmDeltaRuleColumnKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0)
{
	static_assert((KEY_DIM % LM_WARP_LANES) == 0u && (VALUE_DIM % LM_WARP_LANES) == 0u && KEY_DIM <= REDUCE_THREADS &&
		(REDUCE_THREADS % LM_WARP_LANES) == 0u && REDUCE_THREADS / LM_WARP_LANES <= LM_WARP_LANES &&
		(KEY_DIM % (4u * LM_DELTA_COLUMN_WARPS)) == 0u,
		"a lane owns one value column; the norms reduce as a REDUCE_THREADS block sum would");
	constexpr uint32_t WARPS = REDUCE_THREADS / LM_WARP_LANES;
	__shared__ float state_s[KEY_DIM][LM_WARP_LANES];
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
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	count = end > begin ? end - begin : 0u;
	state = (State *)(state_pool
		+ ((uint64_t)state_index[sequence] * slot_bytes)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(State)));
	for ( index = threadIdx.x; index < KEY_DIM * LM_WARP_LANES; index += LM_DELTA_COLUMN_THREADS )
		state_s[index / LM_WARP_LANES][index % LM_WARP_LANES] = LmScalarToFloat(state[((index / LM_WARP_LANES) * VALUE_DIM) + column_base + (index % LM_WARP_LANES)]);
	for ( step = 0u; step < 2u && step < count; step++ )
		if ( warp >= 2u )
		{
			ordinal = begin + step;
			LmDeltaRuleColumnPrepare<WARPS,KEY_DIM,VALUE_DIM>(warp - 2u,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,
				sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal,head,key_heads,value_heads_per_key,column_base,lane,
				key_s[step],query_s[step],forget_s[step],value_s[step],&beta_s[step]);
		}
	__syncthreads();
	if ( warp == 1u && count != 0u )
	{
		float predicted = 0.0f;
		#pragma unroll 16
		for ( index = 0u; index < KEY_DIM; index++ )
			predicted = __fmaf_rn(__fmul_rn(state_s[index][lane],key_s[0][index]),forget_s[0][index],predicted);
		coefficient_s[lane] = beta_s[0] * (value_s[0][lane] - predicted);
	}
	__syncthreads();
	for ( step = 0u; step < count; step++ )
	{
		const uint32_t buffer = step % LM_DELTA_COLUMN_RING, next = (step + 1u) % LM_DELTA_COLUMN_RING, ahead = (step + 2u) % LM_DELTA_COLUMN_RING;
		{
			const float coefficient = coefficient_s[lane];
			#pragma unroll
			for ( index = warp * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index < (warp + 1u) * (KEY_DIM / LM_DELTA_COLUMN_WARPS); index += 4u )
			{
				const float4 forget4 = *(const float4 *)&forget_s[buffer][index];
				const float4 key4 = *(const float4 *)&key_s[buffer][index];
				state_s[index][lane] = __fmaf_rn(forget4.x,state_s[index][lane],__fmul_rn(coefficient,key4.x));
				state_s[index + 1u][lane] = __fmaf_rn(forget4.y,state_s[index + 1u][lane],__fmul_rn(coefficient,key4.y));
				state_s[index + 2u][lane] = __fmaf_rn(forget4.z,state_s[index + 2u][lane],__fmul_rn(coefficient,key4.z));
				state_s[index + 3u][lane] = __fmaf_rn(forget4.w,state_s[index + 3u][lane],__fmul_rn(coefficient,key4.w));
			}
		}
		__syncthreads();
		if ( warp == 0u )
		{
			const uint32_t row = sequence_row_indices != 0 ? sequence_row_indices[begin + step] : begin + step;
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
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + column_base + lane] = LmFloatToBf16(total);
		}
		else if ( warp == 1u && step + 1u < count )
		{
			float predicted = 0.0f;
			#pragma unroll 4
			for ( index = 0u; index < KEY_DIM; index += 4u )
			{
				const float4 key4 = *(const float4 *)&key_s[next][index];
				const float4 forget4 = *(const float4 *)&forget_s[next][index];
				predicted = __fmaf_rn(__fmul_rn(state_s[index][lane],key4.x),forget4.x,predicted);
				predicted = __fmaf_rn(__fmul_rn(state_s[index + 1u][lane],key4.y),forget4.y,predicted);
				predicted = __fmaf_rn(__fmul_rn(state_s[index + 2u][lane],key4.z),forget4.z,predicted);
				predicted = __fmaf_rn(__fmul_rn(state_s[index + 3u][lane],key4.w),forget4.w,predicted);
			}
			coefficient_s[lane] = beta_s[next] * (value_s[next][lane] - predicted);
		}
		else if ( warp >= 2u && step + 2u < count )
		{
			ordinal = begin + step + 2u;
			LmDeltaRuleColumnPrepare<WARPS,KEY_DIM,VALUE_DIM>(warp - 2u,key_bf16,query_bf16,value_bf16,forget_gate,write_gate,
				sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal,head,key_heads,value_heads_per_key,column_base,lane,
				key_s[ahead],query_s[ahead],forget_s[ahead],value_s[ahead],&beta_s[ahead]);
		}
		__syncthreads();
	}
	if ( commit == 0u )
		return;
	for ( index = threadIdx.x; index < KEY_DIM * LM_WARP_LANES; index += LM_DELTA_COLUMN_THREADS )
		LmStoreState(&state[((index / LM_WARP_LANES) * VALUE_DIM) + column_base + (index % LM_WARP_LANES)],state_s[index / LM_WARP_LANES][index % LM_WARP_LANES]);
}

enum LmConvActivation
{
	LM_CONV_NONE = 0,
	LM_CONV_SWISH = 1
};

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
static __device__ __forceinline__ void LmCausalConvBody(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices)
{
	uint32_t sequence = blockIdx.x,channel = (blockIdx.y * THREADS) + threadIdx.x;
	uint32_t begin,end,row,tap,ordinal;
	uint16_t taps[KERNEL];
	uint16_t *slot;
	if ( sequence >= sequences || channel >= channels )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	slot = window + ((uint64_t)state_index[sequence] * channels * KERNEL);
	for (tap = 0u; tap < KERNEL; ++tap)
		taps[tap] = slot[(channel * KERNEL) + tap];
	for (ordinal = begin; ordinal < end; ++ordinal)
	{
		row = sequence_row_indices != 0 ? sequence_row_indices[ordinal] : ordinal;
		float total = 0.0f;
		for (tap = 0u; tap + 1u < KERNEL; ++tap)
			taps[tap] = taps[tap + 1u];
		taps[KERNEL - 1u] = input_bf16[((uint64_t)row * channels) + channel];
		for (tap = 0u; tap < KERNEL; ++tap)
			total += LmBf16ToFloat(taps[tap])
				* LmScalarToFloat(weight[(channel * KERNEL) + tap]);
		if ( ACTIVATION == LM_CONV_SWISH )
		{
			if ( sizeof(Weight) == sizeof(uint16_t) )
				total = LmBf16ToFloat(LmFloatToBf16(total));
			total = total * (1.0f / (1.0f + __expf(-total)));
		}
		output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
	}
	if ( commit == 0u )
		return;
	for (tap = 0u; tap < KERNEL; ++tap)
		slot[(channel * KERNEL) + tap] = taps[tap];
}

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices = 0)
{
	LmCausalConvBody<THREADS,KERNEL,ACTIVATION,Weight>(window,state_index,sequence_row_begin,sequence_row_count,input_bf16,weight,output_bf16,channels,sequences,commit,sequence_row_indices);
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
void LmCausalConvStreamsKernel(const __grid_constant__ LmCausalConvStreams<Weight> streams, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, uint32_t sequences, uint32_t commit, const uint32_t *__restrict__ sequence_row_indices)
{
	const uint32_t stream = blockIdx.z;
	LmCausalConvBody<THREADS,KERNEL,ACTIVATION,Weight>(streams.window[stream],state_index,sequence_row_begin,sequence_row_count,streams.input_bf16[stream],streams.weight[stream],streams.output_bf16[stream],streams.channels[stream],sequences,commit,sequence_row_indices);
}
