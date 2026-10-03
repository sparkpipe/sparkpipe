#pragma once

#include <stdint.h>

#include "inference/kernels/kv.cuh"
#include "sparkpipe/spark_kv_shard.h"

#define LM_KV_ACCESS_ERROR_SHARD_NOT_OWNED LM_FRAME_ERROR_SHARD_NOT_OWNED
#define LM_KV_SHARD_UNPACK_BLOCKS 4096u

struct LmKvShardView
{
	LmKvView pages;
	SparkKvShard shard;
};

struct LmKvShardReplicaView
{
	LmKvView pages;
	SparkKvShard shard;
};

struct LmKvShardGatherView
{
	LmKvView pages;
	SparkKvShard shard;
	const uint8_t *keys;
	const uint32_t *key_offset;
	const uint32_t *key_context;
};

template<class Geometry>
static __host__ __forceinline__ int32_t LmKvShardViewInitialize(
	LmKvShardView *view,
	uint8_t *pool,
	const uint32_t *page_table,
	uint32_t page_table_stride,
	uint32_t sequence_count,
	uint32_t pool_page_count,
	LmKvAccessError *access_error,
	SparkKvShard shard)
{
	if ( view == 0 || !Geometry::kGrows || SparkKvShardValid(shard,Geometry::kPageSlots) == 0u )
		return(-1);
	view->shard = shard;
	return(LmKvViewInitialize(&view->pages,pool,page_table,page_table_stride,sequence_count,pool_page_count,access_error));
}

template<class Geometry>
static __host__ __forceinline__ int32_t LmKvShardReplicaViewInitialize(
	LmKvShardReplicaView *view,
	const LmKvView &pages,
	SparkKvShard shard)
{
	if ( view == 0 || !Geometry::kGrows || SparkKvShardValid(shard,Geometry::kPageSlots) == 0u || !LmKvViewIsConfigured(pages) )
		return(-1);
	view->pages = pages;
	view->shard = shard;
	return(0);
}

template<class Geometry>
static __host__ __forceinline__ int32_t LmKvShardGatherViewInitialize(
	LmKvShardGatherView *view,
	const LmKvView &pages,
	SparkKvShard shard,
	const uint8_t *keys,
	const uint32_t *key_offset,
	const uint32_t *key_context)
{
	if ( view == 0 || !Geometry::kGrows || SparkKvShardValid(shard,Geometry::kPageSlots) == 0u || !LmKvViewIsConfigured(pages) ||
		keys == 0 || key_offset == 0 || key_context == 0 )
		return(-1);
	view->pages = pages;
	view->shard = shard;
	view->keys = keys;
	view->key_offset = key_offset;
	view->key_context = key_context;
	return(0);
}

template<class Geometry>
static __host__ __device__ __forceinline__ uint64_t LmKvShardPageStride(SparkKvShard shard)
{
	return((uint64_t)Geometry::kPageBytes / shard.degree);
}

template<class Geometry>
static __device__ __forceinline__ uint32_t LmKvShardReportForeign(
	const LmKvView &pages,
	SparkKvShard shard,
	uint32_t sequence,
	uint32_t position,
	uint32_t row,
	LmKvAccessKind access_kind)
{
	if ( SparkKvShardOwns(shard,position) != 0u )
		return(0u);
	LmKvReportRequiredAccessFailure(
		pages,
		LM_KV_ACCESS_ERROR_SHARD_NOT_OWNED,
		access_kind,
		row,
		sequence,
		position,
		Geometry::PageOf(position));
	return(1u);
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvShardSlotRequired(
	const LmKvShardView &view,
	uint32_t sequence,
	uint32_t position,
	uint32_t row,
	LmKvAccessKind access_kind)
{
	uint32_t physical_page;

	if ( LmKvShardReportForeign<Geometry>(view.pages,view.shard,sequence,position,row,access_kind) != 0u )
		return(0);
	physical_page = LmKvPhysicalPageRequired<Geometry>(
		view.pages,sequence,position,row,access_kind);
	if ( physical_page == LM_KV_PAGE_UNMAPPED )
		return(0);
	return(view.pages.pool
		+ (uint64_t)physical_page * LmKvShardPageStride<Geometry>(view.shard)
		+ (uint64_t)SparkKvShardSlotInPage(view.shard,Geometry::kPageSlots,position) * Geometry::kSlotBytes);
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvShardSlotRequired(
	const LmKvShardReplicaView &view,
	uint32_t sequence,
	uint32_t position,
	uint32_t row,
	LmKvAccessKind access_kind)
{
	if ( LmKvShardReportForeign<Geometry>(view.pages,view.shard,sequence,position,row,access_kind) != 0u )
		return(0);
	return(LmKvSlotRequired<Geometry>(view.pages,sequence,position,row,access_kind));
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvShardSlotRequired(
	const LmKvShardGatherView &view,
	uint32_t sequence,
	uint32_t position,
	uint32_t row,
	LmKvAccessKind access_kind)
{
	if ( LmKvShardReportForeign<Geometry>(view.pages,view.shard,sequence,position,row,access_kind) != 0u )
		return(0);
	if ( sequence >= view.pages.sequence_count || position >= view.key_context[sequence] )
	{
		LmKvReportRequiredAccessFailure(
			view.pages,
			sequence >= view.pages.sequence_count ? LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE : LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE,
			access_kind,
			row,
			sequence,
			position,
			Geometry::PageOf(position));
		return(0);
	}
	return(view.keys + ((uint64_t)view.key_offset[sequence] + SparkKvShardLocalIndex(view.shard,position)) * Geometry::kSlotBytes);
}

#ifdef __CUDACC__
template<class Geometry, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvShardStoreKernel(LmKvShardView view, const uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count, uint32_t elements)
{
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count || SparkKvShardOwns(view.shard,position_of_row[row]) == 0u )
		return;
	slot = (uint8_t *)LmKvShardSlotRequired<Geometry>(
		view,sequence_of_row[row],position_of_row[row],row,LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
		return;
	for (index = threadIdx.x; index < elements; index += THREADS)
		((uint16_t *)slot)[index] = rows_bf16[((uint64_t)row * elements) + index];
}

template<class Geometry, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvShardGatherPackKernel(LmKvShardView view, const uint32_t *__restrict__ sequences, const uint32_t *__restrict__ key_offset, const uint32_t *__restrict__ key_context, uint8_t *__restrict__ keys)
{
	static_assert(Geometry::kSlotBytes % 16u == 0u, "a gathered key moves as whole 16-byte words");
	uint32_t entry = blockIdx.y, sequence = sequences[entry], context = key_context[sequence], count, local, position, index;
	const uint8_t *slot;
	uint4 *target, zero = {0u, 0u, 0u, 0u};
	count = SparkKvShardGatherKeys(view.shard, context);
	for (local = blockIdx.x; local < count; local += gridDim.x)
	{
		position = SparkKvShardLocalPosition(view.shard, local);
		slot = position < context ? LmKvShardSlotRequired<Geometry>(view, sequence, position, entry, LM_KV_ACCESS_READ) : 0;
		target = (uint4 *)(keys + ((uint64_t)key_offset[sequence] + local) * Geometry::kSlotBytes);
		for (index = threadIdx.x; index < Geometry::kSlotBytes / 16u; index += THREADS)
			target[index] = slot != 0 ? ((const uint4 *)slot)[index] : zero;
	}
}

template<uint32_t SLOT_BYTES, uint32_t PAGE_SLOTS, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvShardGatherUnpackKernel(SparkKvShard shard, const uint8_t *__restrict__ gathered, uint64_t chunk_bytes, uint64_t section_offset, const uint32_t *__restrict__ sequences, const uint32_t *__restrict__ key_offset, const uint32_t *__restrict__ key_context, const uint32_t *__restrict__ page_base, uint8_t *__restrict__ pages)
{
	static_assert(SLOT_BYTES % 16u == 0u, "a gathered key moves as whole 16-byte words");
	uint32_t entry = blockIdx.y, sequence = sequences[entry], context = key_context[sequence], position, index;
	uint64_t byte;
	const uint4 *source;
	uint4 *target;
	for (position = blockIdx.x; position < context; position += gridDim.x)
	{
		byte = section_offset + ((uint64_t)key_offset[sequence] + SparkKvShardLocalIndex(shard, position)) * SLOT_BYTES;
		source = (const uint4 *)(gathered + ((byte / chunk_bytes) * shard.degree + SparkKvShardOwner(shard, position)) * chunk_bytes + byte % chunk_bytes);
		target = (uint4 *)(pages + (((uint64_t)page_base[sequence] + position / PAGE_SLOTS) * PAGE_SLOTS + position % PAGE_SLOTS) * SLOT_BYTES);
		for (index = threadIdx.x; index < SLOT_BYTES / 16u; index += THREADS)
			target[index] = source[index];
	}
}

template<uint32_t PAGE_SLOTS, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvShardGatherTableKernel(const uint32_t *__restrict__ sequences, const uint32_t *__restrict__ key_context, const uint32_t *__restrict__ page_base, uint32_t page_table_stride, uint32_t *__restrict__ page_table)
{
	uint32_t entry = blockIdx.x, sequence = sequences[entry], pages = (key_context[sequence] + PAGE_SLOTS - 1u) / PAGE_SLOTS, page;
	for (page = threadIdx.x; page < pages && page < page_table_stride; page += THREADS)
		page_table[(uint64_t)sequence * page_table_stride + page] = page_base[sequence] + page;
}

template<uint32_t SLOT_BYTES, uint32_t PAGE_SLOTS, uint32_t THREADS>
static inline cudaError_t LmKvShardGatherUnpackLaunch(
	SparkKvShard shard,
	const uint8_t *gathered,
	uint64_t chunk_bytes,
	uint64_t section_offset,
	const uint32_t *sequences,
	const uint32_t *key_offset,
	const uint32_t *key_context,
	const uint32_t *page_base,
	uint32_t sequence_count,
	uint32_t most_context,
	uint32_t page_table_stride,
	uint32_t *page_table,
	uint8_t *pages,
	cudaStream_t stream)
{
	if ( gathered == 0 || sequences == 0 || key_offset == 0 || key_context == 0 || page_base == 0 || pages == 0 ||
		sequence_count == 0u || most_context == 0u || page_table_stride == 0u || SparkKvShardValid(shard,PAGE_SLOTS) == 0u ||
		chunk_bytes == 0u || chunk_bytes % SLOT_BYTES != 0u || section_offset % SLOT_BYTES != 0u )
		return(cudaErrorInvalidValue);
	if ( page_table != 0 )
		LM_LAUNCH((LmKvShardGatherTableKernel<PAGE_SLOTS, THREADS>), sequence_count, THREADS, 0, stream, sequences, key_context, page_base, page_table_stride, page_table);
	LM_LAUNCH((LmKvShardGatherUnpackKernel<SLOT_BYTES, PAGE_SLOTS, THREADS>), dim3(most_context < LM_KV_SHARD_UNPACK_BLOCKS ? most_context : LM_KV_SHARD_UNPACK_BLOCKS, sequence_count), THREADS, 0, stream, shard, gathered, chunk_bytes, section_offset, sequences, key_offset, key_context, page_base, pages);
	return(cudaPeekAtLastError());
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvShardRemapKernel(SparkKvShard shard, SparkKvShardSectionLayout layout, const uint32_t *__restrict__ old_bound, uint32_t capacity, uint32_t *__restrict__ table)
{
	uint32_t position, old = *old_bound;
	for (position = blockIdx.x * THREADS + threadIdx.x; position < capacity; position += gridDim.x * THREADS)
		table[position] = position < old ? (uint32_t)SparkKvShardSectionSlot(shard, layout, position) : LM_KV_PAGE_UNMAPPED;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvRowsRemapKernel(const uint32_t *__restrict__ position_of_row, uint32_t row_count, uint32_t tail_slot, uint32_t capacity, uint32_t *__restrict__ table, LmKvAccessError *error)
{
	uint32_t row, position;
	for (row = blockIdx.x * THREADS + threadIdx.x; row < row_count; row += gridDim.x * THREADS)
	{
		position = position_of_row[row];
		if ( position >= capacity )
		{
			LmFrameErrorReport(error, LM_FRAME_ERROR_PAGE_TABLE_OUT_OF_RANGE, LM_KV_ACCESS_WRITE, row, 0u, position, capacity);
			continue;
		}
		if ( row == 0u || position_of_row[row - 1u] != position )
			table[position] = tail_slot + row;
	}
}

template<uint32_t THREADS>
static inline cudaError_t LmKvShardRemapLaunch(
	SparkKvShard shard,
	SparkKvShardSectionLayout layout,
	const uint32_t *old_bound,
	uint32_t capacity,
	uint32_t *table,
	cudaStream_t stream)
{
	uint32_t blocks = (capacity + THREADS - 1u) / THREADS;
	if ( old_bound == 0 || table == 0 || capacity == 0u || shard.degree == 0u || shard.grain == 0u ||
		(layout.keys != 0u && (layout.chunk_bytes == 0u || layout.slot_bytes == 0u || layout.chunk_bytes % layout.slot_bytes != 0u)) )
		return(cudaErrorInvalidValue);
	if ( layout.keys == 0u )
		layout.chunk_bytes = layout.slot_bytes = 1u;
	LM_LAUNCH((LmKvShardRemapKernel<THREADS>), blocks < LM_KV_SHARD_UNPACK_BLOCKS ? blocks : LM_KV_SHARD_UNPACK_BLOCKS, THREADS, 0, stream, shard, layout, old_bound, capacity, table);
	return(cudaPeekAtLastError());
}

template<uint32_t THREADS>
static inline cudaError_t LmKvRowsRemapLaunch(
	const uint32_t *position_of_row,
	uint32_t row_count,
	uint32_t tail_slot,
	uint32_t capacity,
	uint32_t *table,
	LmKvAccessError *error,
	cudaStream_t stream)
{
	if ( position_of_row == 0 || table == 0 || row_count == 0u || capacity == 0u )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmKvRowsRemapKernel<THREADS>), (row_count + THREADS - 1u) / THREADS, THREADS, 0, stream, position_of_row, row_count, tail_slot, capacity, table, error);
	return(cudaPeekAtLastError());
}

template<class Geometry, uint32_t THREADS>
static inline cudaError_t LmKvShardGatherPackLaunch(
	LmKvShardView view,
	const uint32_t *sequences,
	const uint32_t *key_offset,
	const uint32_t *key_context,
	uint32_t sequence_count,
	uint32_t most_keys,
	uint8_t *keys,
	cudaStream_t stream)
{
	if ( sequences == 0 || key_offset == 0 || key_context == 0 || keys == 0 || sequence_count == 0u || most_keys == 0u ||
		SparkKvShardValid(view.shard,Geometry::kPageSlots) == 0u || !LmKvViewIsConfigured(view.pages) )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmKvShardGatherPackKernel<Geometry, THREADS>), dim3(most_keys, sequence_count), THREADS, 0, stream, view, sequences, key_offset, key_context, keys);
	return(cudaPeekAtLastError());
}
#endif
