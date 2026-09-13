#pragma once

// The sampling head. Final norm, logits, token selection.
//
// The last thing a decode step does and the only part that produces a token.
// Every model needs it and none had it; the exemption list in
// tests/test_config_coverage.py named it for five models at once, which is what
// that list is for.
//
// TWO SHAPES, AND THE CHOICE IS NOT A PREFERENCE. Full-vocabulary decoding
// projects the hidden state against the whole embedding matrix: at GLM 5.2's
// 6144 hidden and 154,880 vocabulary that is 952 M parameters, 1.9 GB in BF16,
// read once per step. That is a third of a routed layer's weight stream for one
// token, which is why it dominates at small batch and why the restricted form
// exists.
//
// Restricted decoding projects against a subset - a grammar, a tool schema, a
// set of legal continuations - and costs the subset's size. A 256-token
// restriction is 1.6 MB against 1.9 GB, three orders of magnitude, and it is
// exact rather than approximate: the tokens outside the set were going to be
// rejected anyway.
//
// So the head is not one kernel with a flag. It is two, and a caller that knows
// its continuation set is constrained should say so.
//
// WHY THE ARGMAX IS SEPARATE FROM THE PROJECTION. Fusing them looks free - the
// logits are in registers, take the max there - but the projection is one block
// per (row, vocabulary tile) and the max is over the whole row. Fusing forces
// either one block per row, which leaves the machine idle at small batch, or a
// grid-wide reduction inside the kernel, which needs cooperative launch. Two
// kernels and a small intermediate is the cheaper arrangement, and the
// intermediate is candidates rather than full logits: each projection block
// emits its own best, and the second pass picks among those.

#include "inference/kernels/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include <stdint.h>

// One block per (row, vocabulary tile). Each emits the best token in its tile,
// so the second pass reduces over tiles rather than over the vocabulary - 152
// candidates instead of 154,880 at a 1024-wide tile.
template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,stride;
	float best = -INFINITY;
	uint32_t best_token = 0u;
	for (index = tile * TILE + threadIdx.x; index < (tile + 1u) * TILE && index < vocabulary; index += THREADS)
	{
		// token_ids null means the full vocabulary and the row index IS the
		// token; non-null means a restricted set and the row is an index into it.
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		if ( total > best )
		{
			best = total;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride && shared_score[threadIdx.x + stride] > shared_score[threadIdx.x] )
		{
			shared_score[threadIdx.x] = shared_score[threadIdx.x + stride];
			shared_token[threadIdx.x] = shared_token[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		candidate_score[(row * gridDim.x) + tile] = shared_score[0];
		candidate_token[(row * gridDim.x) + tile] = shared_token[0];
	}
}

// Reduce the per-tile candidates to one token per row.
//
// Ties go to the lower token id, deterministically. An argmax that breaks ties
// by whichever thread arrived first is not reproducible across launches, and a
// decoder that is not reproducible cannot be compared against a reference - which
// is the whole reason a reference exists.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	uint32_t row = blockIdx.x,index,stride;
	float best = -INFINITY;
	uint32_t best_token = 0xffffffffu;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < tiles; index += THREADS)
	{
		float score = candidate_score[(row * tiles) + index];
		uint32_t token = candidate_token[(row * tiles) + index];
		if ( score > best || (score == best && token < best_token) )
		{
			best = score;
			best_token = token;
		}
	}
	shared_score[threadIdx.x] = best;
	shared_token[threadIdx.x] = best_token;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
		{
			float other = shared_score[threadIdx.x + stride];
			uint32_t other_token = shared_token[threadIdx.x + stride];
			if ( other > shared_score[threadIdx.x]
				|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
			{
				shared_score[threadIdx.x] = other;
				shared_token[threadIdx.x] = other_token;
			}
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		token_out[row] = shared_token[0];
		if ( score_out != 0 )
			score_out[row] = shared_score[0];
	}
}

// Softmax over a row of logits, in place, for sampled decoding.
//
// Subtracts the maximum before exponentiating. Without that a logit of 90 -
// ordinary for a confident model at this vocabulary size - overflows a float
// exponential and the whole row becomes NaN, which propagates into every
// subsequent token rather than producing one bad one.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSoftmaxKernel(float *__restrict__ logits, uint32_t rows, uint32_t vocabulary, float temperature)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint64_t base = (uint64_t)blockIdx.x * vocabulary;
	uint32_t index;
	float local = -INFINITY,maximum,total = 0.0f,inverse_temperature;
	if ( blockIdx.x >= rows )
		return;
	inverse_temperature = 1.0f / fmaxf(temperature,1.0e-4f);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		local = fmaxf(local,logits[base + index] * inverse_temperature);
	maximum = LmBlockMax<THREADS>(local,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
	{
		float value = __expf((logits[base + index] * inverse_temperature) - maximum);
		logits[base + index] = value;
		total += value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		logits[base + index] /= fmaxf(total,1.0e-20f);
}

// TOP-K SELECTION WITHOUT THE LOGITS ROUND TRIP.
//
// Sampled decoding needs the k best tokens and their scores. The naive
// arrangement - project to a full logits buffer, sort on the host - moves
// rows x 154,880 floats across the bus every step to extract at most a few
// hundred bytes, and the projection itself already touched exactly the data
// the selection reads. This is the chunked form of the same algorithm the
// argmax pair above uses: each (row, tile) block emits ITS top-k as
// (score, token) candidates, a second pass selects the row's top-k among
// tiles*k candidates, and the only bytes that leave the device for the
// sampler are rows * k * 8. At k = 64 and B1024 that is 512 KB against the
// 635 MB a full f32 logits buffer would stream.
//
// WHAT THE SAMPLER CONSUMES. token_out and score_out are [rows][K],
// descending by score, ties to the lower token id - the same tie rule as the
// argmax commit, so greedy is topk at K = 1 bit-for-bit. Temperature and
// top-p are operations on those K scores; applying them needs no vocabulary,
// only the selected scores, so they belong to the sampler and stay out of
// this file. A row with fewer than K live tokens pads with token
// 0xffffffff and score -INF, which a sampler must skip.
//
// WHEN NOT TO USE THIS. K is a compile-time template parameter and the
// per-tile pass carries K candidates per block, so the regime is K <= 64:
// nucleus-style sampling over a shortlist. A caller that needs the full
// distribution - logprobs, perplexity, beam scores over the vocabulary -
// takes the full-logits variant instead: project with output_f32 through
// the ordinary GEMM path and run LmHeadSoftmaxKernel on the buffer. That
// variant stays exactly as supported as this one; it is simply not a decode
// hot path.
//
// WHY THE CANDIDATE KERNEL KEEPS ITS LOGITS IN REGISTERS. A per-round
// rescan would recompute the dot products K times and stream the head
// weight K times - at 1.9 GB that is the one cost this file exists to
// avoid. Each thread's strided share is TILE / THREADS elements, a
// compile-time bound, so the tile's logits are computed once and the K
// selection rounds only compare.

// Candidates per row for buffer sizing: tiles * K (score, token) pairs.
// Constexpr-shaped so a caller sizes the workspace without a device query.
template<uint32_t TILE, uint32_t K>
static inline uint32_t LmHeadTopkCandidatePairs(uint32_t rows, uint32_t vocabulary)
{
	return(rows * ((vocabulary + TILE - 1u) / TILE) * K);
}

// One block per (row, vocabulary tile), K rounds of block-best with the
// winners suppressed. Emits the tile's K best as (score, token) pairs at
// candidate[(row * tiles + tile) * K + round].
template<uint32_t THREADS, uint32_t TILE, uint32_t K>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadTopkCandidateKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ head_weight_bf16, const uint32_t *__restrict__ token_ids, float *__restrict__ candidate_score, uint32_t *__restrict__ candidate_token, uint32_t rows, uint32_t hidden, uint32_t vocabulary)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	__shared__ uint32_t taken[K];
	// A thread's strided share of the tile, computed once so the head weight
	// streams exactly once; a per-round rescan would recompute the dot
	// products K times and re-read 1.9 GB per round. The bound guards the
	// ragged final tile, where a thread's share can be empty.
	#define LM_HEAD_TOPK_LOCAL ((TILE + THREADS - 1u) / THREADS)
	float local_score[LM_HEAD_TOPK_LOCAL];
	uint32_t local_token[LM_HEAD_TOPK_LOCAL];
	uint32_t row = blockIdx.y,tile = blockIdx.x,index,element,round,pick,stride;
	uint32_t tile_begin = tile * TILE,tile_end = tile_begin + TILE,count = 0u;
	if ( tile_end > vocabulary )
		tile_end = vocabulary;
	for (index = tile_begin + threadIdx.x; index < tile_end; index += THREADS)
	{
		// token_ids null means the full vocabulary and the row index IS the
		// token; non-null means a restricted set and the row is an index
		// into it. Same rule as the argmax candidate kernel.
		uint32_t token = token_ids != 0 ? token_ids[index] : index;
		float total = 0.0f;
		for (element = 0u; element < hidden; ++element)
			total += LmBf16ToFloat(normed_bf16[((uint64_t)row * hidden) + element])
				* LmBf16ToFloat(head_weight_bf16[((uint64_t)token * hidden) + element]);
		local_score[count] = total;
		local_token[count] = token;
		++count;
	}
	for (round = 0u; round < K; ++round)
	{
		float best = -INFINITY;
		uint32_t best_token = 0xffffffffu;
		for (pick = 0u; pick < count; ++pick)
		{
			uint32_t held = 0u;
			for (index = 0u; index < round; ++index)
				if ( taken[index] == local_token[pick] )
					held = 1u;
			if ( held != 0u )
				continue;
			if ( local_score[pick] > best ||
				(local_score[pick] == best && local_token[pick] < best_token) )
			{
				best = local_score[pick];
				best_token = local_token[pick];
			}
		}
		shared_score[threadIdx.x] = best;
		shared_token[threadIdx.x] = best_token;
		__syncthreads();
		for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
		{
			if ( threadIdx.x < stride )
			{
				float other = shared_score[threadIdx.x + stride];
				uint32_t other_token = shared_token[threadIdx.x + stride];
				if ( other > shared_score[threadIdx.x]
					|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
				{
					shared_score[threadIdx.x] = other;
					shared_token[threadIdx.x] = other_token;
				}
			}
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			candidate_score[((row * gridDim.x) + tile) * K + round] = shared_score[0];
			candidate_token[((row * gridDim.x) + tile) * K + round] = shared_token[0];
			// Suppression is by TOKEN, not index: the token is what the
			// commit pass sees, and a restricted set that repeats an id is a
			// caller error whose cost is a skipped duplicate, not wrong
			// output.
			taken[round] = shared_token[0];
		}
		__syncthreads();
	}
	#undef LM_HEAD_TOPK_LOCAL
}

// Reduce the per-tile shortlists to the row's top-k. tiles * K candidates,
// K rounds of block-best with the winners suppressed - the candidate data
// is a few kilobytes per row, so the rescan costs nothing and the register
// staging the projection half needs would buy nothing here.
template<uint32_t THREADS, uint32_t K>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadTopkCommitKernel(const float *__restrict__ candidate_score, const uint32_t *__restrict__ candidate_token, uint32_t tiles, uint32_t *__restrict__ token_out, float *__restrict__ score_out, uint32_t rows)
{
	__shared__ float shared_score[THREADS];
	__shared__ uint32_t shared_token[THREADS];
	__shared__ uint32_t taken[K];
	uint32_t row = blockIdx.x,candidates = tiles * K,index,round,stride,prior;
	if ( row >= rows )
		return;
	for (round = 0u; round < K; ++round)
	{
		float best = -INFINITY;
		uint32_t best_token = 0xffffffffu;
		for (index = threadIdx.x; index < candidates; index += THREADS)
		{
			float score = candidate_score[(row * candidates) + index];
			uint32_t token = candidate_token[(row * candidates) + index];
			uint32_t held = 0u;
			for (prior = 0u; prior < round; ++prior)
				if ( taken[prior] == token )
					held = 1u;
			if ( held != 0u )
				continue;
			if ( score > best || (score == best && token < best_token) )
			{
				best = score;
				best_token = token;
			}
		}
		shared_score[threadIdx.x] = best;
		shared_token[threadIdx.x] = best_token;
		__syncthreads();
		for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
		{
			if ( threadIdx.x < stride )
			{
				float other = shared_score[threadIdx.x + stride];
				uint32_t other_token = shared_token[threadIdx.x + stride];
				if ( other > shared_score[threadIdx.x]
					|| (other == shared_score[threadIdx.x] && other_token < shared_token[threadIdx.x]) )
				{
					shared_score[threadIdx.x] = other;
					shared_token[threadIdx.x] = other_token;
				}
			}
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			token_out[(row * K) + round] = shared_token[0];
			if ( score_out != 0 )
				score_out[(row * K) + round] = shared_score[0];
			taken[round] = shared_token[0];
		}
		__syncthreads();
	}
}

// Both launches and their validation. candidate_score/candidate_token hold
// LmHeadTopkCandidatePairs<TILE,K>(rows,vocabulary) elements each; token_ids
// is null for the full vocabulary or a restricted set, as in the argmax
// pair. The norm that feeds normed_bf16 is the caller's - the head selects,
// it does not normalise.
template<uint32_t THREADS, uint32_t TILE, uint32_t K>
static int32_t LmHeadTopk(
	const uint16_t *normed_bf16,
	const uint16_t *head_weight_bf16,
	const uint32_t *token_ids,
	float *candidate_score,
	uint32_t *candidate_token,
	uint32_t *token_out,
	float *score_out,
	uint32_t rows,
	uint32_t hidden,
	uint32_t vocabulary,
	cudaStream_t stream)
{
	uint32_t tiles;

	if ( normed_bf16 == 0 || head_weight_bf16 == 0 ||
		candidate_score == 0 || candidate_token == 0 || token_out == 0 ||
		rows == 0u || hidden == 0u || vocabulary == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	tiles = (vocabulary + TILE - 1u) / TILE;
	LM_LAUNCH((LmHeadTopkCandidateKernel<THREADS,TILE,K>), dim3(tiles,rows), THREADS, 0, stream,
		normed_bf16,head_weight_bf16,token_ids,candidate_score,candidate_token,rows,hidden,vocabulary);
	LM_LAUNCH((LmHeadTopkCommitKernel<THREADS,K>), rows, THREADS, 0, stream,
		candidate_score,candidate_token,tiles,token_out,score_out,rows);
	return(cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH);
}
