#pragma once

// Speculative decode. Verify and accept.
//
// This tree has two speculative systems - MTP and DSpark - and they are the same
// algorithm. MTP is 6 kernels in the GLM decode stage; the DSpark backend is
// 2,887 lines. What differs between them is where the draft comes from: MTP uses
// an extra model layer, DSpark uses a separate small model. What is identical is
// everything after: run the target over the draft, compare, take the longest
// prefix that matches, and roll the KV cache back to where it diverged.
//
// So the drafter is a policy and the verifier is a kernel. That split is the
// whole extraction, and it is why this file is short.
//
// WHY ACCEPTANCE IS EXACT, NOT PROBABILISTIC. Greedy decoding accepts a draft
// token if it equals the target's argmax, which is a comparison. Sampled
// decoding needs the modified rejection rule that preserves the target
// distribution, and getting that wrong biases output in a way no test catches
// because the output is still fluent. Both are here; the greedy path is not a
// special case of the sampled one and pretending otherwise is how the bias gets
// introduced.
//
// THE CACHE ROLLBACK IS THE PART THAT BITES. A rejected draft token has already
// had its KV written. Leaving it there poisons every subsequent step for that
// sequence with a key that was never in the accepted sequence - fluent, wrong,
// and attributable only by diffing against a non-speculative run. The context
// length is the rollback, and it must be set from the accepted count and nothing
// else.

#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

// Longest prefix of the draft that the target agrees with, under greedy
// decoding. One block per sequence, one thread doing the walk - the draft is a
// handful of tokens and a parallel scan would cost more than it saves.
//
// Writes the accepted count AND the first token the target would have produced
// past the accepted prefix. That extra token is free - the target already
// computed it - and taking it is what makes speculative decode a win at
// acceptance rate zero rather than a break-even.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSpeculativeVerifyGreedyKernel(const uint32_t *__restrict__ draft_tokens, const uint32_t *__restrict__ target_argmax, uint32_t draft_length, uint32_t *__restrict__ accepted_count, uint32_t *__restrict__ committed_tokens, uint32_t *__restrict__ context_length)
{
	uint32_t sequence = blockIdx.x,index,accepted = 0u;
	uint64_t base = (uint64_t)sequence * draft_length;
	if ( threadIdx.x != 0u )
		return;
	while ( accepted < draft_length
		&& draft_tokens[base + accepted] == target_argmax[base + accepted] )
		++accepted;
	for (index = 0u; index < accepted; ++index)
		committed_tokens[base + index] = draft_tokens[base + index];
	// The bonus token: the target's own prediction at the divergence point. It
	// is correct by construction whether or not the draft matched, so a fully
	// rejected draft still advances by one and speculation never loses.
	committed_tokens[base + accepted] = target_argmax[base + accepted];
	accepted_count[sequence] = accepted + 1u;
	// Roll the cache back. Every draft position past the accepted prefix had its
	// KV written speculatively and must not be visible: a stale key is a key that
	// was never in the accepted sequence, and it poisons every later step.
	context_length[sequence] = context_length[sequence] - (draft_length - accepted);
}

// Sampled acceptance, preserving the target distribution.
//
// Accept draft token t with probability min(1, p_target(t) / p_draft(t)). On
// rejection, sample from the normalised positive part of (p_target - p_draft),
// which is what makes the overall output distribution identical to sampling from
// the target directly.
//
// That correction is the entire correctness argument for sampled speculation and
// it is easy to omit, because omitting it produces output that is still fluent
// and merely biased toward whatever the drafter prefers.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSpeculativeVerifySampledKernel(const uint32_t *__restrict__ draft_tokens, const float *__restrict__ draft_probability, const float *__restrict__ target_probability, const float *__restrict__ uniform, uint32_t draft_length, uint32_t vocabulary, uint32_t *__restrict__ accepted_count, uint32_t *__restrict__ committed_tokens, uint32_t *__restrict__ context_length)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ uint32_t shared_accepted;
	uint32_t sequence = blockIdx.x,step,token,index;
	uint64_t draft_base = (uint64_t)sequence * draft_length;
	float residual_total,pick,running;
	if ( threadIdx.x == 0u )
		shared_accepted = 0u;
	__syncthreads();
	for (step = 0u; step < draft_length; ++step)
	{
		token = draft_tokens[draft_base + step];
		if ( threadIdx.x == 0u )
		{
			float pd = draft_probability[((draft_base + step) * vocabulary) + token];
			float pt = target_probability[((draft_base + step) * vocabulary) + token];
			shared_accepted = (uniform[draft_base + step] < fminf(1.0f,pt / fmaxf(pd,1e-20f)))
				? 1u : 0u;
		}
		__syncthreads();
		if ( shared_accepted == 0u )
			break;
		if ( threadIdx.x == 0u )
			committed_tokens[draft_base + step] = token;
		__syncthreads();
	}
	// Rejection: resample from the positive part of (target - draft). Doing this
	// with the raw target instead would bias toward tokens the drafter already
	// over-proposed.
	residual_total = 0.0f;
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		residual_total += fmaxf(0.0f,
			target_probability[((draft_base + step) * vocabulary) + index]
			- draft_probability[((draft_base + step) * vocabulary) + index]);
	residual_total = LmBlockSum<THREADS>(residual_total,reduction);
	if ( threadIdx.x == 0u )
	{
		pick = uniform[draft_base + draft_length] * fmaxf(residual_total,1e-20f);
		running = 0.0f;
		for (index = 0u; index < vocabulary; ++index)
		{
			running += fmaxf(0.0f,
				target_probability[((draft_base + step) * vocabulary) + index]
				- draft_probability[((draft_base + step) * vocabulary) + index]);
			if ( running >= pick )
				break;
		}
		committed_tokens[draft_base + step] = index < vocabulary ? index : 0u;
		accepted_count[sequence] = step + 1u;
		context_length[sequence] = context_length[sequence] - (draft_length - step);
	}
}
