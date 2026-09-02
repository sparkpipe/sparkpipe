#pragma once


#include <stdint.h>

#define LM_KV_PAGE_UNMAPPED 0xffffffffu

template<uint32_t SLOT_BYTES, uint32_t PAGE_SLOTS, bool GROWS>
struct LmKvGeometry
{
	static constexpr uint32_t kSlotBytes = SLOT_BYTES;
	static constexpr uint32_t kPageSlots = PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = SLOT_BYTES * PAGE_SLOTS;
	static constexpr bool kGrows = GROWS;

	static_assert(PAGE_SLOTS != 0u && (PAGE_SLOTS & (PAGE_SLOTS - 1u)) == 0u,
		"page size must be a power of two so position/page is a shift");
	static_assert(SLOT_BYTES % 16u == 0u,
		"slot must be 16-byte aligned so a cache read is a vector load");
	static_assert(GROWS || PAGE_SLOTS == 1u,
		"a non-growing pool holds one slot per sequence; pages have no meaning");

	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
	{
		return(position / PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
	{
		return(position % PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
	{
		return((tokens + PAGE_SLOTS - 1u) / PAGE_SLOTS);
	}

	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
	{
		return(pages * (uint64_t)kPageBytes);
	}
};

typedef enum LmKvAccessKind
{
	LM_KV_ACCESS_READ = 1u,
	LM_KV_ACCESS_WRITE = 2u
}
LmKvAccessKind;

#include "inference/kernels/frame_error.cuh"

typedef LmFrameError LmKvAccessError;
typedef LmFrameErrorCode LmKvAccessErrorCode;

#define LM_KV_ACCESS_ERROR_NONE LM_FRAME_ERROR_NONE
#define LM_KV_ACCESS_ERROR_INVALID_VIEW LM_FRAME_ERROR_INVALID_VIEW
#define LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE LM_FRAME_ERROR_SEQUENCE_OUT_OF_RANGE
#define LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE LM_FRAME_ERROR_PAGE_TABLE_OUT_OF_RANGE
#define LM_KV_ACCESS_ERROR_PAGE_UNMAPPED LM_FRAME_ERROR_PAGE_UNMAPPED
#define LM_KV_ACCESS_ERROR_POOL_PAGE_OUT_OF_RANGE LM_FRAME_ERROR_POOL_PAGE_OUT_OF_RANGE
#define LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY LM_FRAME_ERROR_INVALID_GQA_GEOMETRY
#define LM_KV_ACCESS_ERROR_RECORDING LM_FRAME_ERROR_RECORDING

struct LmKvView
{
	uint8_t *pool;
	const uint32_t *page_table;
	uint32_t page_table_stride;
	uint32_t sequence_count;
	uint32_t pool_page_count;
	LmKvAccessError *access_error;
};

static __host__ __device__ __forceinline__ void LmKvAccessErrorReset(
	LmKvAccessError *error)
{
	LmFrameErrorReset(error);
}

static __host__ __forceinline__ int32_t LmKvViewInitialize(
	LmKvView *view,
	uint8_t *pool,
	const uint32_t *page_table,
	uint32_t page_table_stride,
	uint32_t sequence_count,
	uint32_t pool_page_count,
	LmKvAccessError *access_error)
{
	if ( view == 0 )
		return(-1);
	view->pool = pool;
	view->page_table = page_table;
	view->page_table_stride = page_table_stride;
	view->sequence_count = sequence_count;
	view->pool_page_count = pool_page_count;
	view->access_error = access_error;
	if ( pool == 0 || page_table == 0 || page_table_stride == 0u
		|| sequence_count == 0u || pool_page_count == 0u
		|| access_error == 0 )
		return(-1);
	LmKvAccessErrorReset(access_error);
	return(0);
}

static __device__ __forceinline__ void LmKvReportRequiredAccessFailure(
	const LmKvView &view,
	LmKvAccessErrorCode error_code,
	LmKvAccessKind access_kind,
	uint32_t row,
	uint32_t sequence,
	uint32_t position,
	uint32_t page)
{
	LmFrameErrorReport(
		view.access_error,
		(uint32_t)error_code,
		(uint32_t)access_kind,
		row,
		sequence,
		position,
		page);
}

static __host__ __device__ __forceinline__ int32_t LmKvViewIsConfigured(
	const LmKvView &view)
{
	return(view.pool != 0
		&& view.page_table != 0
		&& view.page_table_stride != 0u
		&& view.sequence_count != 0u
		&& view.pool_page_count != 0u
		&& view.access_error != 0);
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvSlotRequired(
	const LmKvView &view,
	uint32_t sequence,
	uint32_t position,
	uint32_t row,
	LmKvAccessKind access_kind)
{
	uint32_t logical_page;
	uint32_t physical_page;

	if ( !LmKvViewIsConfigured(view) )
	{
		LmKvReportRequiredAccessFailure(
			view,
			LM_KV_ACCESS_ERROR_INVALID_VIEW,
			access_kind,
			row,
			sequence,
			position,
			0xffffffffu);
		return(0);
	}
	if ( sequence >= view.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			view,
			LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			access_kind,
			row,
			sequence,
			position,
			0xffffffffu);
		return(0);
	}
	logical_page = Geometry::PageOf(position);
	if ( logical_page >= view.page_table_stride )
	{
		LmKvReportRequiredAccessFailure(
			view,
			LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE,
			access_kind,
			row,
			sequence,
			position,
			logical_page);
		return(0);
	}
	physical_page = view.page_table[
		((uint64_t)sequence * view.page_table_stride) + logical_page];
	if ( physical_page == LM_KV_PAGE_UNMAPPED )
	{
		LmKvReportRequiredAccessFailure(
			view,
			LM_KV_ACCESS_ERROR_PAGE_UNMAPPED,
			access_kind,
			row,
			sequence,
			position,
			logical_page);
		return(0);
	}
	if ( physical_page >= view.pool_page_count )
	{
		LmKvReportRequiredAccessFailure(
			view,
			LM_KV_ACCESS_ERROR_POOL_PAGE_OUT_OF_RANGE,
			access_kind,
			row,
			sequence,
			position,
			physical_page);
		return(0);
	}
	return(view.pool + ((uint64_t)physical_page * Geometry::kPageBytes)
		+ ((uint64_t)Geometry::SlotInPage(position) * Geometry::kSlotBytes));
}

template<class Geometry>
static __device__ __forceinline__ const uint8_t *LmKvSlot(
	const LmKvView &view,
	uint32_t sequence,
	uint32_t position)
{
	return(LmKvSlotRequired<Geometry>(
		view,sequence,position,0xffffffffu,LM_KV_ACCESS_READ));
}

template<class Geometry>
static __device__ __forceinline__ uint8_t *LmKvSlotMutableRequired(
	const LmKvView &view,
	uint32_t sequence,
	uint32_t position,
	uint32_t row)
{
	return((uint8_t *)LmKvSlotRequired<Geometry>(
		view,sequence,position,row,LM_KV_ACCESS_WRITE));
}

template<class Geometry>
static __device__ __forceinline__ uint8_t *LmKvSlotMutable(
	const LmKvView &view,
	uint32_t sequence,
	uint32_t position)
{
	return(LmKvSlotMutableRequired<Geometry>(
		view,sequence,position,0xffffffffu));
}

template<class Geometry>
static __host__ __device__ __forceinline__ uint64_t LmKvBytesForSequence(uint32_t length)
{
	return((uint64_t)Geometry::PagesForTokens(length) * (uint64_t)Geometry::kPageBytes);
}


template<uint32_t ELEMENT_BITS, uint32_t LATENT_ELEMENTS, uint32_t ROPE_ELEMENTS, uint32_t PAGE_SLOTS>
using LmKvLatent = LmKvGeometry<(((LATENT_ELEMENTS + ROPE_ELEMENTS) * ELEMENT_BITS) / 8u), PAGE_SLOTS, true>;

template<uint32_t ELEMENT_BITS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t PAGE_SLOTS>
using LmKvHeads = LmKvGeometry<((KV_HEADS * HEAD_DIM * 2u * ELEMENT_BITS) / 8u), PAGE_SLOTS, true>;

template<uint32_t STATE_BYTES>
using LmKvState = LmKvGeometry<STATE_BYTES, 1u, false>;

#ifdef __CUDACC__
template<class Geometry, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmKvStoreKernel(LmKvView view, const uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count, uint32_t elements)
{
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count )
		return;
	slot = LmKvSlotMutableRequired<Geometry>(
		view,sequence_of_row[row],position_of_row[row],row);
	if ( slot == 0 )
		return;
	for (index = threadIdx.x; index < elements; index += THREADS)
		((uint16_t *)slot)[index] = rows_bf16[((uint64_t)row * elements) + index];
}
#endif
