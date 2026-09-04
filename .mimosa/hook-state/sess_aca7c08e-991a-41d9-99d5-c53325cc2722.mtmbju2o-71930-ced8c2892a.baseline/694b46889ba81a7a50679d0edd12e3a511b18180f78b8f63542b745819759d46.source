// KV geometry: one allocator, three models whose slots mean different things.
// Compile-only - every check below is a static_assert, so a wrong geometry is a
// build failure on the sparkdev's laptop rather than a launch failure on the ring.
#include <stdint.h>
// kv.cuh needs nothing but stdint, which is the point: a host tool can reason
// about cache capacity without a CUDA toolchain.
#define __host__
#define __device__
#define __forceinline__ inline
#include "inference/kernels/kv.cuh"
#include "modules/glm52_resident_decode_stage/source/cuda/config.h"

using Glm52Kv    = LmKvLatent<GLM52_KV_BITS, GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;
using Mimo25Full = LmKvHeads<16u, 4u, 128u, 64u>;
using Qwen36Gdn  = LmKvState<131072u>;

static_assert(Glm52Kv::kSlotBytes == 1152u, "MLA latent slot: (512+64) x bf16");
static_assert(Glm52Kv::kSlotBytes == GLM52_KV_SLOT_BYTES, "config.h agrees with kv.cuh");
static_assert(Glm52Kv::kPageBytes == 73728u, "64-slot page is 72 KB");
static_assert(Glm52Kv::PageOf(129u) == 2u, "position 129 is in page 2");
static_assert(Glm52Kv::SlotInPage(129u) == 1u, "position 129 is slot 1 of it");
static_assert(Glm52Kv::PagesForTokens(1048576u) == 16384u, "1M context is 16384 pages");
static_assert(Mimo25Full::kSlotBytes == 2048u, "4 KV heads x 128 dim x (k+v) x bf16");
static_assert(Qwen36Gdn::kGrows == false, "recurrent state does not grow");
static_assert(Qwen36Gdn::kPageSlots == 1u, "state is one slot per sequence");

// The cache format is independent of the weight format: the same model with an
// FP8 cache halves its slot and its whole pool, and nothing else changes.
using Glm52KvFp8 = LmKvLatent<8u, GLM52_LATENT, GLM52_ROPE_DIM, GLM52_KV_PAGE_SLOTS>;
static_assert(Glm52KvFp8::kSlotBytes == 576u, "FP8 cache halves the slot");
static_assert(Glm52KvFp8::kSlotBytes * 2u == Glm52Kv::kSlotBytes, "exactly half of BF16");
static_assert(Glm52KvFp8::PagesForTokens(1048576u) == Glm52Kv::PagesForTokens(1048576u),
	"page count is a function of slots, not bytes, so it is unchanged");

// The same address helper serves a paged cache and a non-paged state pool.
void probe(void)
{
	LmKvView view = {};
	view.pool = 0;
	view.page_table = 0;
	view.page_table_stride = 0;
	view.sequence_count = 0;
	(void)LmKvSlot<Glm52Kv>(view, 0u, 0u);
	(void)LmKvSlot<Qwen36Gdn>(view, 0u, 0u);
	(void)LmKvBytesForSequence<Glm52Kv>(1024u);
}
