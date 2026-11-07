#include "inference/kernels/metal/common.metal"

#define SPARK_SAMPLE_DIGIT_BITS 8u
#define SPARK_SAMPLE_DIGITS (1u << SPARK_SAMPLE_DIGIT_BITS)
#define SPARK_SAMPLE_PASSES (64u / SPARK_SAMPLE_DIGIT_BITS)

struct SparkSampleScratch
{
	threadgroup float *values;
	threadgroup ulong *keys;
	threadgroup atomic_uint *low;
	threadgroup atomic_uint *high;
	threadgroup ulong *choice;
	threadgroup ulong *picked;
	threadgroup ulong *unique_key;
	threadgroup atomic_uint *unique_count;
	threadgroup uint *unique_weighted;
	threadgroup atomic_uint *picked_count;
	uint tid;
};

static inline float SparkSampleMax(thread const SparkSampleScratch &s, float value)
{
	s.values[s.tid] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint stride = SPARK_SAMPLE_THREADS / 2u; stride > 0u; stride >>= 1u )
	{
		if ( s.tid < stride )
			s.values[s.tid] = fmax(s.values[s.tid],s.values[s.tid + stride]);
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const float result = s.values[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return result;
}

static inline float SparkSampleSum(thread const SparkSampleScratch &s, float value)
{
	s.values[s.tid] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint stride = SPARK_SAMPLE_THREADS / 2u; stride > 0u; stride >>= 1u )
	{
		if ( s.tid < stride )
			s.values[s.tid] += s.values[s.tid + stride];
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const float result = s.values[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return result;
}

static inline ulong SparkSampleMaxKey(thread const SparkSampleScratch &s, ulong key)
{
	s.keys[s.tid] = key;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint stride = SPARK_SAMPLE_THREADS / 2u; stride > 0u; stride >>= 1u )
	{
		if ( s.tid < stride && s.keys[s.tid + stride] > s.keys[s.tid] )
			s.keys[s.tid] = s.keys[s.tid + stride];
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const ulong result = s.keys[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return result;
}

static inline ulong SparkSampleSumKey(thread const SparkSampleScratch &s, ulong value)
{
	s.keys[s.tid] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint stride = SPARK_SAMPLE_THREADS / 2u; stride > 0u; stride >>= 1u )
	{
		if ( s.tid < stride )
			s.keys[s.tid] += s.keys[s.tid + stride];
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const ulong result = s.keys[0];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return result;
}

static inline void SparkSampleCount(thread const SparkSampleScratch &s, uint digit, ulong weight)
{
	const uint low = (uint)weight;
	const uint before = atomic_fetch_add_explicit(&s.low[digit],low,memory_order_relaxed);
	const uint high = (uint)(weight >> 32u) + (before + low < before ? 1u : 0u);
	if ( high != 0u )
		atomic_fetch_add_explicit(&s.high[digit],high,memory_order_relaxed);
}

static inline ulong SparkSampleCounted(thread const SparkSampleScratch &s, uint digit)
{
	return(((ulong)atomic_load_explicit(&s.high[digit],memory_order_relaxed) << 32u) |
		(ulong)atomic_load_explicit(&s.low[digit],memory_order_relaxed));
}

static inline ulong SparkSampleSelect(thread const SparkSampleScratch &s, device const float *logits, uint vocabulary, ulong floor_key,
	ulong target, bool by_mass, float maximum, float inverse_temperature)
{
	ulong prefix = 0ul, above = 0ul;
	for ( uint pass = 0u; pass < SPARK_SAMPLE_PASSES; pass++ )
	{
		const uint shift = 64u - SPARK_SAMPLE_DIGIT_BITS * (pass + 1u);
		uint matches = 0u, match_weighted = 0u;
		ulong match_key = 0ul;
		for ( uint digit = s.tid; digit < SPARK_SAMPLE_DIGITS; digit += SPARK_SAMPLE_THREADS )
		{
			atomic_store_explicit(&s.low[digit],0u,memory_order_relaxed);
			atomic_store_explicit(&s.high[digit],0u,memory_order_relaxed);
		}
		if ( s.tid == 0u )
			atomic_store_explicit(s.unique_count,0u,memory_order_relaxed);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint token = s.tid; token < vocabulary; token += SPARK_SAMPLE_THREADS )
		{
			const float logit = logits[token];
			const ulong key = SparkMathSampleKey(logit,token);
			if ( key < floor_key || (pass != 0u && (key >> (shift + SPARK_SAMPLE_DIGIT_BITS)) != (prefix >> (shift + SPARK_SAMPLE_DIGIT_BITS))) )
				continue;
			const ulong weight = by_mass ? SparkMathSampleMass(logit,maximum,inverse_temperature) : 1ul;
			matches++;
			match_key = key;
			match_weighted = weight != 0ul ? 1u : 0u;
			if ( weight != 0ul )
				SparkSampleCount(s,(uint)((key >> shift) & (SPARK_SAMPLE_DIGITS - 1u)),weight);
		}
		if ( matches != 0u )
			atomic_fetch_add_explicit(s.unique_count,matches,memory_order_relaxed);
		if ( matches == 1u )
		{
			*s.unique_key = match_key;
			*s.unique_weighted = match_weighted;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( atomic_load_explicit(s.unique_count,memory_order_relaxed) == 1u && *s.unique_weighted != 0u )
		{
			const ulong key = *s.unique_key;
			threadgroup_barrier(mem_flags::mem_threadgroup);
			return key;
		}
		if ( s.tid == 0u )
		{
			ulong running = above;
			uint chosen = 0u;
			bool found = false;
			for ( uint digit = SPARK_SAMPLE_DIGITS; digit-- > 0u; )
			{
				const ulong count = SparkSampleCounted(s,digit);
				if ( count == 0ul )
					continue;
				if ( running + count >= target )
				{
					chosen = digit;
					found = true;
					break;
				}
				running += count;
			}
			for ( uint digit = 0u; !found && digit < SPARK_SAMPLE_DIGITS; digit++ )
				if ( SparkSampleCounted(s,digit) != 0ul )
				{
					chosen = digit;
					found = true;
				}
			s.choice[0] = prefix | ((ulong)chosen << shift);
			s.choice[1] = running;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		prefix = s.choice[0];
		above = s.choice[1];
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	return prefix;
}

static inline ulong SparkSampleNucleusTarget(ulong total, float nucleus)
{
	const uint bits = as_type<uint>(nucleus), biased = (bits >> 23u) & 0xffu;
	const ulong mantissa = biased != 0u ? (ulong)((bits & 0x7fffffu) | 0x800000u) : (ulong)(bits & 0x7fffffu);
	const uint shift = biased != 0u ? 150u - biased : 149u;
	ulong high = mulhi(total,mantissa), low = total * mantissa;
	const uint length = high != 0ul ? 128u - (uint)clz(high) : 64u - (uint)clz(low);
	if ( length > 53u )
	{
		const uint drop = length - 53u;
		const ulong unit = 1ul << drop, rest = low & (unit - 1ul), midpoint = unit >> 1u;
		low -= rest;
		if ( rest > midpoint || (rest == midpoint && (low & unit) != 0ul) )
		{
			low += unit;
			high += low < unit ? 1ul : 0ul;
		}
	}
	return(shift >= 128u ? 0ul : shift >= 64u ? high >> (shift - 64u) : (low >> shift) | (high << (64u - shift)));
}

static inline void SparkSampleTopLogprobs(thread const SparkSampleScratch &s, device const float *logits, uint vocabulary, uint count, float lse,
	device SparkSampleLogprob *out)
{
	const ulong threshold = SparkSampleSelect(s,logits,vocabulary,0ul,count,false,0.0f,0.0f);
	if ( s.tid == 0u )
		atomic_store_explicit(s.picked_count,0u,memory_order_relaxed);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint token = s.tid; token < vocabulary; token += SPARK_SAMPLE_THREADS )
	{
		const ulong key = SparkMathSampleKey(logits[token],token);
		if ( key >= threshold )
		{
			const uint slot = atomic_fetch_add_explicit(s.picked_count,1u,memory_order_relaxed);
			if ( slot < count )
				s.picked[slot] = key;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( s.tid == 0u )
	{
		const uint total = atomic_load_explicit(s.picked_count,memory_order_relaxed);
		const uint picked = total < count ? total : count;
		for ( uint index = 1u; index < picked; index++ )
			for ( uint other = index; other > 0u && s.picked[other] > s.picked[other - 1u]; other-- )
			{
				const ulong key = s.picked[other];
				s.picked[other] = s.picked[other - 1u];
				s.picked[other - 1u] = key;
			}
		for ( uint index = 0u; index < count; index++ )
		{
			const uint token = index < picked ? SparkMathSampleKeyToken(s.picked[index]) : SPARK_SAMPLE_NO_TOKEN;
			out[index].token = token;
			out[index].logprob = index < picked ? logits[token] - lse : 0.0f;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void spark_sample_rows(constant SparkSampleParams &p [[buffer(0)]], device const float *logits [[buffer(1)]],
	device const SparkSampleRow *rows [[buffer(2)]], device uint *tokens [[buffer(3)]], device float *chosen_logits [[buffer(4)]],
	device SparkSampleLogprob *logprobs [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
	threadgroup float values[SPARK_SAMPLE_THREADS];
	threadgroup ulong keys[SPARK_SAMPLE_THREADS];
	threadgroup atomic_uint low[SPARK_SAMPLE_DIGITS];
	threadgroup atomic_uint high[SPARK_SAMPLE_DIGITS];
	threadgroup ulong choice[2];
	threadgroup ulong picked[SPARK_SAMPLE_MAX_LOGPROBS];
	threadgroup ulong unique_key[1];
	threadgroup atomic_uint unique_count[1];
	threadgroup uint unique_weighted[1];
	threadgroup atomic_uint picked_count[1];
	const SparkSampleScratch s = { values, keys, low, high, choice, picked, unique_key, unique_count, unique_weighted, picked_count, tid };
	if ( row >= p.rows )
		return;
	const SparkSampleRow rule = rows[row];
	device const float *v = logits + (ulong)rule.head_row * p.logit_stride;
	device SparkSampleLogprob *out = logprobs + (ulong)row * SPARK_SAMPLE_MAX_LOGPROBS;
	float maximum = -INFINITY, lse;
	ulong best = 0ul;
	for ( uint token = tid; token < p.vocabulary; token += SPARK_SAMPLE_THREADS )
		maximum = fmax(maximum,v[token]);
	maximum = SparkSampleMax(s,maximum);
	lse = maximum;
	if ( rule.logprobs != 0u )
	{
		float sum = 0.0f;
		for ( uint token = tid; token < p.vocabulary; token += SPARK_SAMPLE_THREADS )
			sum += SPARK_MATH_EXP(v[token] - maximum);
		lse = maximum + SPARK_MATH_LOG(SparkSampleSum(s,sum));
	}
	if ( rule.inverse_temperature == 0.0f )
		for ( uint token = tid; token < p.vocabulary; token += SPARK_SAMPLE_THREADS )
		{
			const ulong key = SparkMathSampleKey(v[token],token);
			best = key > best ? key : best;
		}
	else
	{
		ulong floor_key = 0ul;
		if ( rule.top_k != 0u && rule.top_k < p.vocabulary )
			floor_key = SparkSampleSelect(s,v,p.vocabulary,0ul,rule.top_k,false,0.0f,0.0f);
		if ( rule.nucleus != 0.0f )
		{
			ulong total = 0ul;
			for ( uint token = tid; token < p.vocabulary; token += SPARK_SAMPLE_THREADS )
				if ( SparkMathSampleKey(v[token],token) >= floor_key )
					total += SparkMathSampleMass(v[token],maximum,rule.inverse_temperature);
			total = SparkSampleSumKey(s,total);
			ulong target = SparkSampleNucleusTarget(total,rule.nucleus);
			target = target != 0ul ? target : 1ul;
			const ulong key = SparkSampleSelect(s,v,p.vocabulary,floor_key,target,true,maximum,rule.inverse_temperature);
			floor_key = key > floor_key ? key : floor_key;
		}
		for ( uint token = tid; token < p.vocabulary; token += SPARK_SAMPLE_THREADS )
		{
			const float logit = v[token];
			if ( SparkMathSampleKey(logit,token) < floor_key )
				continue;
			const ulong key = SparkMathSampleKey(fma(logit,rule.inverse_temperature,SparkMathGumbelNoise(rule.seed,rule.position,token)),token);
			best = key > best ? key : best;
		}
	}
	best = SparkSampleMaxKey(s,best);
	const uint chosen = SparkMathSampleKeyToken(best);
	if ( tid == 0u )
	{
		tokens[row] = chosen;
		chosen_logits[row] = v[chosen];
		if ( rule.logprobs != 0u )
		{
			out[0].token = chosen;
			out[0].logprob = v[chosen] - lse;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( rule.logprobs > 1u )
		SparkSampleTopLogprobs(s,v,p.vocabulary,rule.logprobs - 1u,lse,out + 1u);
}
