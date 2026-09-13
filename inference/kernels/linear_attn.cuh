#pragma once

// Linear attention with a recurrent state. The delta rule at decode.
//
// The other half of this tree's architecture space. The GDN model runs Gated DeltaNet
// on 48 of 64 layers and the KDA model runs Delta Attention on 3 of every 4, and
// both are the same recurrence with different gate parameterisations.
//
// WHAT MAKES IT DIFFERENT FROM EVERYTHING ELSE HERE. Softmax attention keeps
// every past key and value and re-reads them each step, so its cost grows with
// context. This keeps a fixed matrix per head - key_dim by value_dim - and
// updates it in place, so its cost per token is constant no matter how long the
// conversation is. At the GDN model's widths that is 512 KB per sequence against a KV
// cache that passes 512 KB at about 170 tokens and keeps growing.
//
// THE DELTA RULE, and why it is not just an accumulation. A naive linear
// attention sets S += v k^T, which writes every value into the state and never
// removes anything - keys that recur just pile up. The delta rule instead
// computes what the state ALREADY predicts for this key, and writes only the
// difference:
//
//     predicted = S^T k
//     S = alpha * S + beta * (v - predicted) k^T
//
// So a key the state already handles correctly changes nothing, and a key it
// gets wrong is corrected in proportion to the error. That is what makes the
// fixed-size state usable for long context rather than saturating.
//
// alpha is the forget gate, beta the write strength. Both are per-token and
// per-head, produced by the layer's projections. Setting alpha to 1 and beta to
// 1 recovers the ungated rule; setting beta to 1 and dropping the prediction
// recovers naive linear attention. Neither is what these models do.
//
// STATE LAYOUT. [head][key_dim][value_dim], key-major, so the k^T outer product
// writes contiguous value_dim runs and the S^T q read is a strided gather that
// every thread does once. The alternative layout makes the update contiguous and
// the read strided, and the update happens once per token where the read happens
// once per token too - so the tie is broken by the outer product being the
// larger of the two.

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

// The decay logit to a per-channel retention factor. The KDA technical report
// equation 5:
//
//     g = g_min * Sigmoid(exp(A_h) * z)      in (g_min, 0)
//     alpha = exp(g)                          in (exp(g_min), 1)
//
// A_h is a learnable per-head log-scale initialised to 0; g_min is fixed at -5.
//
// THE LOWER BOUND IS THE POINT, and it is a numerical argument rather than a
// modelling one. Gated DeltaNet and Mamba-2 use an UNBOUNDED negative-softplus
// mapping, g = -exp(A) * Softplus(z), which lets the cumulative decay over a
// chunk go to zero and its reciprocal - which the chunkwise form divides by -
// grow without bound. Bounding g below at -5 keeps every retention factor above
// exp(-5), so the cumulative log-decay over a 16-token tile stays within
// (-80, 0) and the reciprocal stays inside BF16 range. That is what lets the
// diagonal tiles use dense Tensor Core matrix multiplications instead of an
// explicit position-pair path.
//
// So a plausible-looking substitution here - softplus for sigmoid, or dropping
// the bound - does not merely change the model. It reintroduces an overflow the
// architecture was designed to remove.
// THE BIAS IS ADDED IN FP32, BEFORE THE SCALE. FlashKDA's reference keeps
// dt_bias as a separate fp32 tensor and adds it after upcasting the bf16 logit:
// folding a bf16 bias into a bf16 logit and upcasting afterwards is a different
// number. The report writes z = W_up W_down x + b_alpha and hides that ordering.
//
// FlashKDA also works in LOG2: it computes g_min * log2(e) * sigmoid(...) and
// exponentiates with ex2, because the chunkwise form takes a cumulative sum of
// these logs and ex2 is the hardware instruction. exp(g_min * s) and
// exp2(g_min * log2(e) * s) are the same number; anyone writing the chunkwise
// path should carry log2 rather than convert per element.
// IF SPECULATION EVER REPLAYS THIS RECURRENCE, IT MUST NOT RECOMPUTE THE GATE.
//
// SGLang's ReplaySSM handles rejected drafts by storing each draft step's raw
// inputs - v, k, the gate value and beta, about 1 KB - instead of snapshotting
// the whole state after every step, which at 96 heads by 128 by 128 would be
// 64 KB per request per layer per head. After the sampler fixes the accepted
// length, one fold kernel replays only the accepted prefix.
//
// Their note on getting it wrong is the part to keep: an early version
// recomputed the gate during the fold with a subtly different formula, "which
// left every output looking correct while the state quietly drifted
// underneath". The fold consumes the gate values the verify kernel stored, not
// a second call to this function.
//
// So LmBoundedDecay is the wrong thing to call twice. A replay path stores what
// this returns and reads it back; it does not evaluate the mapping again, even
// with identical inputs, because __expf is an approximation and two call sites
// that agree today are two call sites that can be edited apart.
static __device__ __forceinline__ float LmBoundedDecay(float logit, float bias, float head_log_scale, float minimum_log_decay)
{
	float scaled = __expf(head_log_scale) * (logit + bias);
	float log_decay = minimum_log_decay * (1.0f / (1.0f + __expf(-scaled)));
	return(__expf(log_decay));
}

// Map a row of decay logits to retention factors, one block per (row, head).
// The logits arrive from a low-rank projection - W_up(W_down(x)) plus a
// per-head bias - so they are PER HEAD PER CHANNEL, not one scalar per head.
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

// The GDN model's gate producer, one block per (row, head). The reference form is
// the UNBOUNDED negative-softplus mapping the bounded kernel above argues
// against for the chunkwise path - g = -exp(A_h) * softplus(z + dt_bias),
// retention = exp(g) - with the write gate beta = sigmoid(W_beta x). The
// per-token delta rule never inverts a cumulative decay, which is where the
// bound matters, so the decode recurrence takes the reference's mapping as
// it stands.
//
// Both logits are PER HEAD SCALARS - one per value head from two 48-row
// projections of the hidden state, not a per-channel vector - so the
// retention written here is one value repeated across the head's KEY_DIM
// channels: the "per-head caller passes a vector with every channel equal"
// case the delta rule's gate layout names. A_log and dt_bias are the
// checkpoint's own per-head fp32 tensors, and the bias is added in fp32
// after the bf16 upcast, the same ordering argument LmBoundedDecay carries.
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

// L2-normalise each head's vector. KDA normalises q and k after the convolution
// and before the recurrence; without it the delta rule's k k^T term is not a
// projection and the state grows without bound.
//
// FlashKDA reduces with a warp-shuffle tree and FMA, and its torch reference
// reproduces that exact order rather than calling torch.norm - the accumulation
// order is part of the contract when the result feeds a recurrence. This is a
// plain block reduction, so it will not be bit-identical; that is a difference
// worth knowing about before comparing outputs element-wise.
template<uint32_t THREADS, uint32_t HEAD_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmL2NormalisePerHeadKernel(uint16_t *__restrict__ rows_bf16, uint32_t heads, uint32_t rows, float epsilon)
{
	// LmBlockSum, not a hand-rolled tree. The first version of this kernel wrote
	// its own shared-memory reduction with a barrier per step - log2(THREADS)
	// barriers where the shuffle path needs one - and duplicated a reduction the
	// tree already had. Same defect twice over: slower and a second copy.
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

// -- ReplaySSM ------------------------------------------------------------------
//
// Speculation over a recurrent state is the reason the KDA model's decode can go from ~113
// to ~423 tok/s, and the reason it is hard. Verification accepts a prefix of the
// draft, so the state must be rewindable - and a KDA state overwrites itself
// every token. The obvious fix, snapshotting after each draft step, costs
// heads * KEY_DIM * VALUE_DIM per step per layer: at the KDA shape 96 * 128 * 128
// bf16 is 3 MB, times gamma+1 steps, times 69 layers, per running request. It
// outgrows the state pool it competes with and caps concurrency.
//
// ReplaySSM stores the INPUTS instead. Verify reads the committed checkpoint and
// never writes it, recording each step's (v, k, gate, beta) - about a kilobyte
// against three megabytes. Once the sampler fixes the accepted length one fold
// replays only the accepted prefix. Rejected drafts are never replayed, so
// rollback is free.
//
// THE FOLD MUST NOT RECOMPUTE THE GATE. SGLang's account of getting this wrong
// is the whole reason this comment is long: an early version recomputed it with
// a subtly different formula, "which left every output looking correct while the
// state quietly drifted underneath". The retention factor is stored by verify
// and read back here. LmBoundedDecay is not called on this path.
struct LmReplayStep
{
	const uint16_t *key_bf16;
	const uint16_t *value_bf16;
	const float *retention;
	const float *write_gate;
};

// The commit store is the ONLY place the recurrent state leaves fp32, in both
// kernels below, and this is its whole implementation: the identity for an
// fp32 pool, dtype.cuh's round-to-nearest-even for a bf16 one. The load
// direction needs no new code - LmScalarToFloat already upcasts both element
// types. Two overloads rather than a runtime branch, so the rounding rule
// lives in exactly one place and the fp32 path compiles to exactly what it
// was. A plain truncation here would bias every committed element toward
// zero and compound inside a recurrence that never renormalises.
static __device__ __forceinline__ void LmStoreState(float *slot, float value)
{
	*slot = value;
}

static __device__ __forceinline__ void LmStoreState(uint16_t *slot, float value)
{
	*slot = LmFloatToBf16(value);
}

// Advance the committed state through the accepted prefix, in place. One block
// per (row, head); the recurrence over steps is serial by nature and the work
// inside a step is the same flat pass the decode kernel uses.
//
// THE POOL ELEMENT TYPE IS THE DECODE KERNEL'S State. Under State = uint16_t
// every step's read upcasts and every step's write re-rounds - the same two
// conversion points a serial decode commit crosses, once per step, which is
// exactly what keeps the fold byte-exact against the decode path it stands
// in for. A fold that deferred the rounding to the end of the prefix would
// drift from serial decode by one bf16 rounding per accepted token - the
// "outputs look correct while the state quietly drifts" failure above, one
// ulp at a time.
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
	// THE POOL STRIDE IS A PARAMETER, NOT AN ASSUMPTION.
	//
	// This computed its slot from key_heads * KEY_DIM * VALUE_DIM * 2, hard-wiring
	// both the element size and the absence of anything else in the slot. The KDA model's
	// config declares the state fp32 and bundles the three convolution windows
	// into the same per-request block - sized against SGLang's measured 54 MB at
	// TP=8 - so a pool allocated at that stride and addressed at this one aliases
	// sequence 1 into sequence 0 for any batch above one.
	//
	// Neither side was wrong on its own. They were two expressions of the same
	// quantity that nothing forced to agree, which is the third time this branch
	// has found that shape: the MLA latent standing in for the KDA head dim, and
	// the GDN producer's KV heads before it.
	//
	// slot_bytes now comes from the caller, which reads it from the model's
	// config, and the element size follows from the same place: State is
	// float for the contract slot, uint16_t for the admission-time bf16
	// option, and the two are never mixed in one launch.
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
		// the decode path L2-normalizes k before the update; the fold must
		// replay the SAME arithmetic or the states diverge byte by byte -
		// the same strided partials and the same LmBlockSum the decode
		// kernel runs, at the same THREADS, never a hand-copied loop whose
		// order can be edited apart from it. This was thread 0 walking 128
		// elements serially while the block waited (audit PERF-002).
		// One buffer suffices here: one reduction per step, and the fill
		// loop's barrier stands between this step's final read and the next
		// step's first write.
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

// One decode step of the gated delta rule, one block per (row, head).
//
// GROUPED VALUE HEADS. The GDN model has 16 key heads and 48 value heads, so three
// value heads share each key head's state slice. The state is indexed by key
// head and the value offset selects within it, which is why value_heads_per_key
// is a parameter rather than assumed to be one.
template<uint32_t THREADS, uint32_t KEY_DIM, uint32_t VALUE_DIM, class State = float>
__global__ __launch_bounds__(THREADS, 1)
void LmDeltaRuleKernel(uint8_t *__restrict__ state_pool, uint32_t slot_bytes, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const float *__restrict__ forget_gate, const float *__restrict__ write_gate, uint16_t *__restrict__ output_bf16, uint32_t key_heads, uint32_t value_heads_per_key, uint32_t sequences, uint32_t commit)
{
	// ONE KERNEL FOR DECODE, PREFILL AND VERIFY. The recurrence is serial in
	// the token and parallel in nothing else a batch offers, so the run is the
	// unit: this block owns one sequence's rows for one head, loads the state
	// into shared memory once, streams the run through it, and touches the
	// global slot again only if commit says the run really happened. Decode is
	// a run of one - a null sequence_row_begin means row i IS sequence i - and
	// a run of T is bit-identical to T decode calls, because the state lives in
	// shared memory at exactly the width the slot stores between calls.
	// That equivalence is a gate, not a comment - and it is stated for
	// State = float; what the bf16-state option keeps and signs away is
	// written at the commit store below.
	//
	// Verify is the third caller: DSpark scores a drafted block by running it
	// forward, and a draft must not advance what the sequence remembers.
	// commit == 0 computes every output and abandons the state.
	// THE SHARED TILE IS FP32 WHATEVER THE POOL HOLDS. The contract
	// (KDA_STATE_ELEMENT_BYTES = 4) makes float the default State; the
	// admission-time half-width option (config.h's
	// KDA_STATE_SLOT_BYTES_BF16) instantiates State = uint16_t and halves
	// the pool stream. Either way the slot is upcast into this same fp32
	// tile on the way in, the recurrence below runs in fp32 exactly as
	// today - bit-identical per-step math given the same fp32 input state -
	// and the ONLY rounding is LmStoreState's round-to-nearest-even on the
	// commit store. Verify (commit == 0) never stores, so it stays
	// dtype-neutral.
	//
	// What the bf16 option signs away: a committed multi-row RUN rounds
	// once, at its end, so under bf16 a run is no longer bit-identical to
	// its decode steps. Serial decode, verify and the replay fold still
	// agree byte for byte under bf16, because they convert at the same two
	// points once per committed step: the load and the commit store.
	//
	// 64 KB per head-block exceeds the static shared limit, so the tile is
	// dynamic - the launch passes KEY_DIM * VALUE_DIM * 4 bytes at either
	// State, because the tile stays fp32.
	extern __shared__ float state_s[];
	__shared__ float shared_key[KEY_DIM];
	__shared__ float shared_query[KEY_DIM];
	// TWO reduction buffers, not one. LmBlockSum's last act is a read of
	// shared[0] AFTER its final barrier, and a second call on the same buffer
	// writes shared[warp] BEFORE its first barrier - warp 0 can overwrite the
	// key total before another warp has read it.
	__shared__ float norm_reduction[2u * (THREADS / LM_WARP_LANES)];
	__shared__ float shared_predicted[VALUE_DIM];
	uint32_t sequence = blockIdx.x,head = blockIdx.y,index,element,row,begin,end,flat;
	State *state;
	float beta,key_inverse,query_inverse;
	if ( sequence >= sequences || head >= key_heads )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	// One slot per sequence, never paged: the state does not grow, so its
	// address is the sequence's slot base plus this head's slice, at the
	// pool's own element width - slot_bytes and State come from the same
	// config, as in the fold.
	state = (State *)(state_pool
		+ ((uint64_t)state_index[sequence] * slot_bytes)
		+ ((uint64_t)head * KEY_DIM * VALUE_DIM * sizeof(State)));
	for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		state_s[flat] = LmScalarToFloat(state[flat]);
	for (row = begin; row < end; ++row)
	{
		// PER HEAD PER CHANNEL. This read was one scalar per head, which cannot
		// express the KDA model's channel-wise decay - its retention factor is a d_k
		// vector from a low-rank projection. The GDN model's gated DeltaNet is the
		// same shape, so the width serves both; a per-head caller passes a
		// vector with every channel equal.
		const float *forget = forget_gate + ((((uint64_t)row * key_heads) + head) * KEY_DIM);
		beta = write_gate[(row * key_heads) + head];
		for (index = threadIdx.x; index < KEY_DIM; index += THREADS)
		{
			shared_key[index] = LmBf16ToFloat(key_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
			shared_query[index] = LmBf16ToFloat(query_bf16[(((uint64_t)row * key_heads) + head) * KEY_DIM + index]);
		}
		for (index = threadIdx.x; index < VALUE_DIM; index += THREADS)
			shared_predicted[index] = 0.0f;
		__syncthreads();
		// QK L2-NORM, IN KERNEL, per the reference's
		// use_qk_l2norm_in_kernel=True: q and k are unit vectors before the
		// delta rule sees them. Reduced by the WHOLE BLOCK: this ran as 128
		// serial FMAs on thread 0 while the other THREADS-1 threads sat at
		// the barrier, once per row per head (audit PERF-002) - the
		// longest dependency chain in the kernel, in the middle of its
		// serial token loop. Strided partials plus LmBlockSum's shuffle tree
		// is a handful of steps.
		//
		// THE SUMMATION ORDER CHANGES ON DEVICE. A tree over strided
		// partials is not the old 0..127 serial walk, so the norms move in
		// the last ulp. The gate that matters survives: decode, verify and
		// the replay fold all reduce through LmBlockSum at the same THREADS,
		// so they still agree with EACH OTHER bit for bit, and at
		// THREADS == 1 - every host harness - the strided partial loop IS
		// the serial walk, so the harness numbers do not move at all.
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
			shared_query[index] *= query_inverse;
		}
		__syncthreads();
		// predicted = S^T k, against the DECAYED state. Report eq. 1 expands to
		// S = Diag(a) S + b k (v^T - k^T Diag(a) S): the retention factor
		// applies BEFORE the prediction, not only to the state that survives
		// it. FlashKDA folds it into the key instead - the same product,
		// written the other way round.
		for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * VALUE_DIM) + element]
					* shared_key[index] * forget[index];
			shared_predicted[element] = total;
		}
		__syncthreads();
		// S = alpha*S + beta*(v - predicted) k^T, and o = S^T q with the
		// UPDATED state. Updating first is not an optimisation detail: the
		// token attends to its own value, exactly as a softmax token attends
		// to its own key. Flat over the whole outer product - every
		// (channel, element) pair independent, consecutive threads on the
		// contiguous axis.
		{
			uint32_t value_head = head * value_heads_per_key;
			uint64_t value_base =
				(((uint64_t)row * key_heads * value_heads_per_key) + value_head) * VALUE_DIM;
			for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
			{
				uint32_t channel = flat / VALUE_DIM,element_index = flat % VALUE_DIM;
				float v = LmBf16ToFloat(value_bf16[value_base + element_index]);
				float previous = state_s[flat];
				state_s[flat] =
					(forget[channel] * previous)
					+ (beta * (v - shared_predicted[element_index]) * shared_key[channel]);
			}
		}
		__syncthreads();
		for (element = threadIdx.x; element < VALUE_DIM; element += THREADS)
		{
			float total = 0.0f;
			for (index = 0u; index < KEY_DIM; ++index)
				total += state_s[(index * VALUE_DIM) + element]
					* shared_query[index];
			output_bf16[(((uint64_t)row * key_heads) + head) * VALUE_DIM + element] =
				LmFloatToBf16(total);
		}
		__syncthreads();
	}
	if ( commit == 0u )
		return;
	// The one rounding the bf16-state option adds: round-to-nearest-even, on
	// the commit store, and nowhere else in the kernel. An uncommitted run
	// never reaches this line, which is why verify is dtype-neutral.
	for (flat = threadIdx.x; flat < KEY_DIM * VALUE_DIM; flat += THREADS)
		LmStoreState(&state[flat],state_s[flat]);
}

// Short causal convolution over the last KERNEL tokens, per channel.
//
// Both GDN and KDA put one of these before the delta rule. It gives the
// recurrence a few tokens of exact local memory, which the state alone provides
// only approximately - the delta rule compresses everything into a fixed matrix
// and a 4-tap convolution keeps the immediate past verbatim.
//
// The window lives in the same non-growing slot as the state, after it, because
// both are per-sequence and neither grows. That is why kernels/kv.cuh sizes the
// slot from the sum rather than from the state alone.
enum LmConvActivation
{
	LM_CONV_NONE = 0,
	LM_CONV_SWISH = 1
};

template<uint32_t THREADS, uint32_t KERNEL, uint32_t ACTIVATION, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmCausalConvKernel(uint16_t *__restrict__ window, const uint32_t *__restrict__ state_index, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_count, const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t channels, uint32_t sequences, uint32_t commit)
{
	// ONE KERNEL FOR DECODE, PREFILL AND VERIFY. A sequence's rows are a run -
	// contiguous, positions ascending - and the window walks the run with the
	// taps held in registers, touching the slot once on the way in and, when
	// commit says so, once on the way out. Decode is a run of one: a null
	// sequence_row_begin means row i IS sequence i, which keeps every decode
	// call site's data exactly as it was. Verify is a run that must not
	// advance what the sequence remembers: commit zero computes every output
	// and abandons the window, which is the whole difference between
	// speculating about tokens and having accepted them.
	uint32_t sequence = blockIdx.x,channel = (blockIdx.y * THREADS) + threadIdx.x;
	uint32_t begin,end,row,tap;
	uint16_t taps[KERNEL];
	uint16_t *slot;
	if ( sequence >= sequences || channel >= channels )
		return;
	begin = sequence_row_begin != 0 ? sequence_row_begin[sequence] : sequence;
	end = sequence_row_begin != 0 ? sequence_row_begin[sequence + 1u] : sequence + 1u;
	// A fold replays only the ACCEPTED prefix of a verify run: the slab keeps
	// its stride, the run keeps its begin, and the count says how much of it
	// really happened. Null means the whole run, which is every other caller.
	if ( sequence_row_count != 0 )
		end = begin + sequence_row_count[sequence];
	slot = window + ((uint64_t)state_index[sequence] * channels * KERNEL);
	for (tap = 0u; tap < KERNEL; ++tap)
		taps[tap] = slot[(channel * KERNEL) + tap];
	for (row = begin; row < end; ++row)
	{
		float total = 0.0f;
		// Shift the window and admit the new token. A ring buffer would avoid
		// the shift, but KERNEL is 4 and a ring needs a head index every reader
		// agrees on - the shift is three register moves and no state.
		for (tap = 0u; tap + 1u < KERNEL; ++tap)
			taps[tap] = taps[tap + 1u];
		taps[KERNEL - 1u] = input_bf16[((uint64_t)row * channels) + channel];
		for (tap = 0u; tap < KERNEL; ++tap)
			total += LmBf16ToFloat(taps[tap])
				* LmScalarToFloat(weight[(channel * KERNEL) + tap]);
		// SWISH, NOT NOTHING. FlashKDA's projections are
		// L2Norm(Swish(ShortConv(Wx))) for q and k and Swish(ShortConv(Wx)) for
		// v; the reference builds ShortConvolution with activation='silu'. A
		// convolution that returns its raw sum runs and is a different model, so
		// the activation is a template parameter with no default.
		if ( ACTIVATION == LM_CONV_SWISH )
			total = total * (1.0f / (1.0f + __expf(-total)));
		output_bf16[((uint64_t)row * channels) + channel] = LmFloatToBf16(total);
	}
	if ( commit == 0u )
		return;
	for (tap = 0u; tap < KERNEL; ++tap)
		slot[(channel * KERNEL) + tap] = taps[tap];
}
