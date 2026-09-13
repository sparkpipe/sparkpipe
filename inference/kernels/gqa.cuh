#pragma once

// Per-head KV decode attention, for the models without latent compression.
//
// attn.cuh's decode kernels are MLA-shaped: a slot is one shared latent row and
// the first LATENT elements double as the value, which is the whole trick of
// latent absorption. The GDN model and MiMo 2.5 store per-head keys AND values, and
// neither kernel can express that - the value is a different tensor at a
// different offset, not a prefix of the key. Both drivers were launched through
// LmAttentionDecodeKernel anyway: the GDN model attended every query head over the first
// 256 elements of a 2048-element slot and wrote 192-wide outputs into a
// 256-wide pipeline, mimo read 64 elements past each key head and wrote 12288
// outputs into an 8192-wide pipeline. Fluent and wrong, with every shape check
// passing, because the slot is opaque bytes and nothing knew the layout.
//
// THE SLOT LAYOUT IS A CONTRACT, stated here once: per (sequence, position),
//
//     [ key:   KV_HEADS x HEAD_DIM   bf16 ]
//     [ value: KV_HEADS x VALUE_DIM  bf16 ]
//
// The store kernel below is the only writer and the decode kernel the only
// reader, and the geometry's slot size is static_asserted against this sum at
// both, so a driver that sizes its pool for another layout fails to compile
// rather than reading past the key region. HEAD_DIM and VALUE_DIM differ on
// MiMo 2.5 (192 key, 128 value), which is why they are separate parameters and
// why LmKvHeads - which prices the value at the key's width - is not the
// geometry such a model can use.
//
// The store takes the split kernel's two dense outputs rather than a pre-packed
// image because LmKvStoreKernel's contract is stride-equals-slot, and the
// split's K and V live at different row strides. Packing here is one launch,
// the same count the broken call sites already paid.

#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <math.h>
#include <stdint.h>

#define LM_KV_POSITION_UNUSED 0xffffffffu

// Build the explicit position list consumed by the sliding-window path. The
// list is produced on the device for every launch, from the exact sequence
// lengths and row positions, so it cannot go stale when batches interleave.
// Rows shorter than the configured window are padded with a sentinel that the
// attention kernel skips without touching the page table.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmBuildSlidingWindowPositionsKernel(
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ position_of_row,
	uint32_t row_count,
	uint32_t window,
	uint32_t *__restrict__ positions_out)
{
	uint32_t row = blockIdx.x;
	uint32_t sequence;
	uint32_t available;
	uint32_t selected;
	uint32_t start;
	uint32_t index;

	if ( row >= row_count || window == 0u )
		return;
	sequence = sequence_of_row[row];
	available = context_length[sequence];
	if ( position_of_row != 0 && position_of_row[row] != 0xffffffffu
		&& available > position_of_row[row] + 1u )
		available = position_of_row[row] + 1u;
	selected = available < window ? available : window;
	start = available - selected;
	for ( index = threadIdx.x; index < window; index += THREADS )
		positions_out[((uint64_t)row * window) + index] = index < selected
			? start + index
			: LM_KV_POSITION_UNUSED;
}

// Repeat every head GROUP times: out[head] = in[head / GROUP]. This is GQA
// expansion materialised - the attention decode above gets the same sharing
// for free through its head-to-KV-head mapping, but a consumer that indexes
// heads densely, like the delta rule, needs the repeated tensor in memory.
// The GDN model's GDN is the case: 16 key heads and 48 value heads, and the
// recurrence holds one state per VALUE head with q and k repeated three ways,
// which is the only form the reference (the next-generation modeling file, FLA's
// gated delta rule) defines - a state per key head shared three ways is not
// GDN, and it silently drops two of every three value heads.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmExpandHeadsKernel(const uint16_t *__restrict__ input_bf16, uint16_t *__restrict__ output_bf16, uint32_t source_heads, uint32_t head_dim, uint32_t group, uint32_t rows)
{
	uint32_t row = blockIdx.x,head,index;
	for (head = 0u; head < source_heads * group; ++head)
		for (index = threadIdx.x; index < head_dim; index += THREADS)
			output_bf16[(((uint64_t)row * source_heads * group) + head) * head_dim + index] =
				input_bf16[(((uint64_t)row * source_heads) + (head / group)) * head_dim + index];
}

// Pack one row's K and V into its slot, in the layout above. One block per
// row; the two sources are the split kernel's dense [row][heads * dim] tensors.
template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaKvStoreKernel(LmKvView view, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count)
{
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"the slot is [K: heads x head_dim][V: heads x value_dim] bf16 and nothing else");
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count )
		return;
	slot = LmKvSlotMutableRequired<Geometry>(
		view,sequence_of_row[row],position_of_row[row],row);
	// Same rule as LmKvStoreKernel: an unmapped page is a scheduler failure, and
	// skipping loses the token's key silently. There is nothing better to do
	// here; the decode side treats the hole as absent.
	if ( slot == 0 )
		return;
	for (index = threadIdx.x; index < KV_HEADS * (HEAD_DIM + VALUE_DIM); index += THREADS)
		((uint16_t *)slot)[index] = index < KV_HEADS * HEAD_DIM
			? key_bf16[((uint64_t)row * KV_HEADS * HEAD_DIM) + index]
			: value_bf16[((uint64_t)row * KV_HEADS * VALUE_DIM)
				+ (index - (KV_HEADS * HEAD_DIM))];
}

// One decode step of grouped-query attention, one block per (row, query head).
//
// The structure is LmAttentionDecodeKernel's - online softmax in a single pass
// over the cache, so the cache is read once - with the two things a per-head
// model adds: the query head maps to its KV head through the group ratio, and
// the value comes from its own region of the slot at its own width.
//
// Sparse selection and the causal bound are the same arguments the latent
// kernel takes, with the same meaning: a position list makes this the
// sliding-window or sparse path, and row_position caps a prefill row at itself.
template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaAttentionDecodeKernel(const uint16_t *__restrict__ query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
{
	// THE ACCUMULATOR IS EXACTLY SIZED. The latent kernel this is shaped from
	// holds eight slots and guards element < LATENT, which silently drops the
	// tail of any latent wider than 8 * THREADS - its own comment tells that
	// story. VALUE_DIM is a template argument here, so the array is the exact
	// stride count: one slot per thread on device (256 values at 256 threads),
	// and the whole head on the host harness's one-thread schedule, which is
	// the only reason the full layer can be emulated at model width.
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[HEAD_DIM];
	float accumulator[(VALUE_DIM + THREADS - 1u) / THREADS];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"store and decode must agree on the slot layout");
	uint32_t sequence = sequence_of_row[row];
	uint32_t kv_head;
	uint64_t query_base;
	float running_max = -INFINITY,running_sum = 0.0f;

	if ( !LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			!LmKvViewIsConfigured(cache)
				? LM_KV_ACCESS_ERROR_INVALID_VIEW
				: LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			0xffffffffu,
			0xffffffffu);
		return;
	}
	if ( heads < KV_HEADS || ( heads % KV_HEADS ) != 0u || head >= heads )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			head,
			heads);
		return;
	}
	// GQA: heads / KV_HEADS query heads share each KV head. The ratio is a
	// runtime value because the query head count is, and it divides the block
	// index once per block - nothing on the cache-read path.
	kv_head = head / (heads / KV_HEADS);
	query_base = ((uint64_t)row * heads + head) * HEAD_DIM;
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
		shared_query[index] = LmBf16ToFloat(query_bf16[query_base + index]);
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : context_length[sequence];
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? selected_positions[(row * selected_count) + step] : step;
		const uint8_t *slot;
		if ( position == LM_KV_POSITION_UNUSED )
			continue;
		const uint16_t *key,*value;
		float score = 0.0f,scaled,previous;
		// Causal: a prefill row must not see past itself. Skipping keeps the
		// online softmax's running maximum honest, as in the latent kernel.
		if ( row_position != 0 && position > row_position[row] )
			continue;
		slot = LmKvSlotRequired<Geometry>(
			cache,sequence,position,row,LM_KV_ACCESS_READ);
		if ( slot == 0 )
			return;
		key = (const uint16_t *)slot + (kv_head * HEAD_DIM);
		value = (const uint16_t *)slot + (KV_HEADS * HEAD_DIM) + (kv_head * VALUE_DIM);
		for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(key[index]);
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		// Online softmax: rescale what is already accumulated rather than
		// revisit the cache once the maximum is known.
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		running_sum = (running_sum * scaled) + __expf(score - running_max);
		for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < VALUE_DIM )
				accumulator[index] = (accumulator[index] * scaled)
					+ (__expf(score - running_max) * LmBf16ToFloat(value[element]));
		}
	}
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < VALUE_DIM )
			output_bf16[(((uint64_t)row * heads) + head) * VALUE_DIM + element] =
				LmFloatToBf16(accumulator[index] / fmaxf(running_sum,1.0e-20f));
	}
}
