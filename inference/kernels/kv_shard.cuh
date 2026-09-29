#pragma once

#include <stdint.h>

#include "inference/kernels/kv.cuh"
#include "sparkpipe/spark_kv_shard.h"

#define LM_KV_ACCESS_ERROR_SHARD_NOT_OWNED LM_FRAME_ERROR_SHARD_NOT_OWNED

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
		+ (uint64_t)physical_page * SparkKvShardPageBytes(view.shard,Geometry::kPageSlots,Geometry::kSlotBytes)
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
#endif
