#pragma once

#include "inference/kernels/kv.cuh"
#include "sparkpipe/spark_glm5_next_model.h"

// BindLayer already advances the pool to this layer's slab. Physical pages
// therefore stride within one layer, not across the model's DSA layer count.
// Each scalar-index slot is 257 BF16 values (514 bytes), so the vector-aligned
// LmKvGeometry contract does not apply. Shared KV access still handles mapping.
struct Glm5NextIndexKv
{
	static constexpr uint32_t kSlotBytes = SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	static constexpr uint32_t kPageSlots = SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = (kSlotBytes * kPageSlots);
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position)
	{ return(position / kPageSlots); }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position)
	{ return(position % kPageSlots); }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens)
	{ return((tokens + kPageSlots - 1u) / kPageSlots); }
	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages)
	{ return(pages * (uint64_t)kPageBytes); }
};
