/* Integration tests for the SHARED paged-KV host core
 * (runtime/paged_kv_common): the lane-table engine over the real
 * prefix-cache core, plus the header-only page-geometry primitives.
 *
 * These are INTEGRATION tests, not mocks: every scenario drives the
 * production Initialize/Admit/Cover/checkpoint path over the real
 * content-addressed pool, and pins the exact observable contract the
 * three ports inherit - the numbers below are the contract:
 *
 *   S2  a cold walk publishes ceil(tokens/block) blocks and moves the
 *       free count by exactly that much;
 *   S3  an UNWITNESSED match clamps to a private restart at
 *       block_tokens-1 tokens (never attaches donor blocks);
 *   S4  a WITNESSED boundary reuses exactly the donor blocks - the
 *       lane's reused ordinals ARE the donor's physical blocks;
 *   S5  checkpoint offers are single-flight and block-aligned, and a
 *       commit past the committed frontier is refused (the open-block
 *       witness bug class);
 *   S6  coverage past the committed frontier borrows PRIVATE scratch;
 *       under pool exhaustion the borrow fails leaving the partial row
 *       counted (F1: counts move WITH each borrow so reset returns all),
 *       and LaneReset restores the free list exactly;
 *   S7  a cross-lane LRU steal shortens later matches but never turns
 *       them wrong;
 *   S8  published blocks outlive their lane; scratch does not; a dead
 *       witness downgrades reuse to a private walk;
 *   S9  checkpoint_slot_count == 0 disables reuse entirely.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "runtime/paged_kv_common.h"

#define TK_BLOCK 64u
#define TK_LANES 4u
#define TK_BLOCKS_PER_LANE 8u
#define TK_PHYSICAL 16u
#define TK_LOGICAL 32u
#define TK_SLOTS 2u
#define TK_STRIDE 4096ull

static uint32_t TestToken(uint32_t salt, uint32_t index)
{
	return salt * 2654435761u + index * 40503u + 1u;
}

static void TestFill(uint32_t *tokens, uint32_t salt, uint32_t count)
{
	uint32_t i;
	for (i = 0u; i < count; ++i)
		tokens[i] = TestToken(salt, i);
}

static int g_failures;

static void Expect(int condition, const char *what)
{
	if (!condition)
	{
		printf("FAIL %s\n", what);
		g_failures++;
	}
}

static void ExpectU32(uint32_t got, uint32_t want, const char *what)
{
	if (got != want)
	{
		printf("FAIL %s: got %u want %u\n", what, got, want);
		g_failures++;
	}
}

static void ExpectU64(uint64_t got, uint64_t want, const char *what)
{
	if (got != want)
	{
		printf("FAIL %s: got %llu want %llu\n", what,
			(unsigned long long)got, (unsigned long long)want);
		g_failures++;
	}
}

static SparkStatus RefusingValidate(
	const struct SparkPagedKvConfiguration *configuration)
{
	(void)configuration;
	return SPARK_STATUS_SCHEMA_ERROR;
}

static const SparkPagedKvGeometryCallbacks TestGeometry =
{
	9000ull, TK_BLOCKS_PER_LANE, 0
};

static const SparkPagedKvGeometryCallbacks RefusingGeometry =
{
	9100ull, TK_BLOCKS_PER_LANE, RefusingValidate
};

static uint32_t s_blocks[TK_LANES * TK_BLOCKS_PER_LANE];
static uint32_t s_counts[TK_LANES];

static void TestFillConfig(SparkPagedKvConfiguration *c, uint32_t slots,
	uint32_t lanes, uint32_t physical, uint32_t logical)
{
	memset(c, 0, sizeof(*c));
	c->block_token_count = TK_BLOCK;
	c->lane_count = lanes;
	c->blocks_per_lane = TK_BLOCKS_PER_LANE;
	c->physical_page_capacity = physical;
	c->logical_page_capacity = logical;
	c->checkpoint_slot_count = slots;
	c->block_stride_bytes = TK_STRIDE;
}

/* ---- S0: the header-only primitives -------------------------------------- */

static void TestPrimitives(void)
{
	void *p;
	ExpectU32(SparkPagedKvBlocksForPositions(0ull, 64u), 0u, "blocks(0)=0");
	ExpectU32(SparkPagedKvBlocksForPositions(1ull, 64u), 1u, "blocks(1)=1");
	ExpectU32(SparkPagedKvBlocksForPositions(64ull, 64u), 1u, "blocks(64)=1");
	ExpectU32(SparkPagedKvBlocksForPositions(65ull, 64u), 2u, "blocks(65)=2");
	ExpectU32(SparkPagedKvBlocksForPositions(320ull, 64u), 5u, "blocks(320)=5");
	Expect(SparkPagedKvCheckedCalloc(0ull, 8ull) == 0, "calloc zero-count fails");
	Expect(SparkPagedKvCheckedCalloc(8ull, 0ull) == 0, "calloc zero-bytes fails");
	Expect(SparkPagedKvCheckedCalloc((uint64_t)-1 / 2ull, 3ull) == 0,
		"calloc overflow fails");
	p = SparkPagedKvCheckedCalloc(4ull, 8ull);
	Expect(p != 0, "calloc sane request succeeds");
	free(p);
	ExpectU32(SparkPagedKvPoolGeometryIsValid(0u, 4u, 256ull, 64u), 0u,
		"geometry: zero logical refused");
	ExpectU32(SparkPagedKvPoolGeometryIsValid(4u, 0u, 256ull, 64u), 0u,
		"geometry: zero physical refused");
	ExpectU32(SparkPagedKvPoolGeometryIsValid(4u, 4u, 256ull, 0u), 0u,
		"geometry: zero block tokens refused");
	ExpectU32(SparkPagedKvPoolGeometryIsValid(4u, 8u, 256ull, 64u), 0u,
		"geometry: logical below physical refused");
	ExpectU32(SparkPagedKvPoolGeometryIsValid(8u, 4u, 1024ull, 64u), 0u,
		"geometry: directory below widest lane refused");
	ExpectU32(SparkPagedKvPoolGeometryIsValid(8u, 4u, 512ull, 64u), 1u,
		"geometry: covering directory accepted");
}

/* ---- S1: initialize validation -------------------------------------------- */

static void TestInitializeValidation(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, 0, s_counts), (uint32_t)SPARK_STATUS_INVALID_ARGUMENT,
		"init: null table refused");
	config.block_token_count = 0u;
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts),
		(uint32_t)SPARK_STATUS_INVALID_ARGUMENT, "init: zero block refused");
	config.block_token_count = TK_BLOCK;
	config.blocks_per_lane = TK_BLOCKS_PER_LANE + 1u;
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts),
		(uint32_t)SPARK_STATUS_INVALID_ARGUMENT,
		"init: row wider than geometry refused");
	config.blocks_per_lane = TK_BLOCKS_PER_LANE;
	config.logical_page_capacity = TK_PHYSICAL - 1u;
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts),
		(uint32_t)SPARK_STATUS_INVALID_ARGUMENT,
		"init: directory below pool refused");
	config.logical_page_capacity = TK_LOGICAL;
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&RefusingGeometry, s_blocks, s_counts),
		(uint32_t)SPARK_STATUS_SCHEMA_ERROR,
		"init: geometry hook refusal propagated");
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts), (uint32_t)SPARK_STATUS_OK,
		"init: valid configuration accepted");
	ExpectU32(cache.reuse_enabled, 1u, "init: slots arm reuse");
	ExpectU32(s_counts[0], 0u, "init: counts zeroed");
	ExpectU32(s_blocks[0], SPARK_PAGED_KV_NO_BLOCK,
		"init: rows prefilled NO_BLOCK");
	SparkPagedKvDestroy(&cache);
	TestFillConfig(&config, 0u, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts), (uint32_t)SPARK_STATUS_OK,
		"init: zero-slot configuration accepted");
	ExpectU32(cache.reuse_enabled, 0u, "init: no slots disarms reuse");
	SparkPagedKvDestroy(&cache);
}
/* ---- S2: cold walk publishing and accounting ------------------------------ */

static void TestColdWalk(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t tokens[320];
	SparkPagedKvMatch match;
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	ExpectU32((uint32_t)SparkPagedKvInitialize(&cache, &config,
		&TestGeometry, s_blocks, s_counts), (uint32_t)SPARK_STATUS_OK,
		"S2 init");
	TestFill(tokens, 1u, 192u);
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 0u, tokens, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S2 admit lane0");
	ExpectU32(match.block_count, 0u, "S2 cold match depth 0");
	ExpectU32(match.checkpoint_slot, SPARK_PAGED_KV_NO_SLOT,
		"S2 cold match slot none");
	ExpectU32(s_counts[0], 3u, "S2 three blocks published");
	Expect(s_blocks[0] != SPARK_PAGED_KV_NO_BLOCK &&
		s_blocks[1] != SPARK_PAGED_KV_NO_BLOCK &&
		s_blocks[2] != SPARK_PAGED_KV_NO_BLOCK &&
		s_blocks[3] == SPARK_PAGED_KV_NO_BLOCK,
		"S2 row holds exactly three");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 0u), 192ull,
		"S2 committed frontier 192");
	ExpectU32(SparkPagedKvFreeBlocks(&cache), 13u,
		"S2 free moved by exactly three");
	/* Re-admitting the SAME lane resets it and walks a different prompt:
	 * released full blocks stay cached (never freed), so the pool pays
	 * only the fresh publication. */
	TestFill(tokens, 2u, 192u);
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 0u, tokens, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S2 re-admit lane0");
	ExpectU32(s_counts[0], 3u, "S2 re-admit republishes three");
	ExpectU32(SparkPagedKvFreeBlocks(&cache), 10u,
		"S2 released full blocks stay cached, not freed");
	SparkPagedKvDestroy(&cache);
}

/* ---- S3: an unwitnessed match restarts privately -------------------------- */

static void TestUnwitnessedClamp(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t a[192];
	SparkPagedKvMatch match;
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	TestFill(a, 7u, 192u);
	SparkPagedKvAdmit(&cache, 0u, a, 192u, &match);
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 1u, a, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S3 admit identical prefix lane1");
	ExpectU32(match.block_count, 0u,
		"S3 no checkpoint bound -> zero reuse despite full LCP");
	ExpectU32(match.checkpoint_slot, SPARK_PAGED_KV_NO_SLOT, "S3 no slot");
	/* The clamp restarts the walk at block_tokens-1 tokens: the lane
	 * opens ONE private partial block, never attaching donor blocks. */
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 1u), 63ull,
		"S3 private restart at 63 tokens");
	ExpectU32(s_counts[1], 1u, "S3 one open core block on the row");
	/* Growing with the true continuation appends normally. */
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 1u, 192ull, a + 63u,
		129u), (uint32_t)SPARK_STATUS_OK, "S3 cover continuation");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 1u), 192ull,
		"S3 covered to 192");
	ExpectU32(s_counts[1], 3u, "S3 row grown to three core blocks");
	SparkPagedKvDestroy(&cache);
}

/* ---- S4: witnessed reuse shares the donor blocks --------------------------- */

static void TestWitnessedReuse(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t ext[320], cont[192];
	uint32_t donor0, donor1, slot;
	SparkPagedKvMatch match;
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	TestFill(ext, 11u, 320u);
	memcpy(cont, ext + 128u, sizeof(cont));
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 0u, ext, 320u, &match),
		(uint32_t)SPARK_STATUS_OK, "S4 donor walk");
	donor0 = s_blocks[0];
	donor1 = s_blocks[1];
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot), 1u,
		"S4 offer at block-aligned 128");
	SparkPagedKvCheckpointCommit(&cache, 0u, slot, 128ull);
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 1u, ext, 320u, &match),
		(uint32_t)SPARK_STATUS_OK, "S4 matcher walk");
	ExpectU32(match.block_count, 2u, "S4 witnessed depth exactly two");
	ExpectU32(match.checkpoint_slot, slot, "S4 donor slot reported");
	ExpectU32(s_counts[1], 2u, "S4 row clamped to donor depth");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 1u), 128ull,
		"S4 sequence holds exactly the donor prefix");
	Expect(s_blocks[0] == donor0 && s_blocks[1] == donor1,
		"S4 reused ordinals ARE the donor physical blocks");
	/* Growth past the clamp appends the continuation normally. */
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 1u, 320ull, cont, 192u),
		(uint32_t)SPARK_STATUS_OK, "S4 cover past clamp");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 1u), 320ull,
		"S4 covered to full length");
	ExpectU32(s_counts[1], 5u, "S4 five blocks on the row");
	Expect(s_blocks[0] == donor0 && s_blocks[1] == donor1,
		"S4 donor prefix untouched by growth");
	SparkPagedKvDestroy(&cache);
}
/* ---- S5: checkpoint lifecycle guards --------------------------------------- */

static void TestCheckpointLifecycle(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t c[130], ext[192];
	uint32_t slot, slot2;
	SparkPagedKvMatch match;
	TestFillConfig(&config, 1u, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	/* One C prompt: two full blocks plus two tokens into the third. */
	TestFill(c, 21u, 130u);
	memcpy(ext, c, sizeof(c));
	{
		uint32_t i;
		for (i = 130u; i < 192u; ++i)
			ext[i] = TestToken(22u, i);
	}
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 0u, c, 130u, &match),
		(uint32_t)SPARK_STATUS_OK, "S5 walk 130 tokens");
	/* Offers refuse misaligned boundaries and dead lanes. */
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 100ull, &slot), 0u,
		"S5 unaligned end_position refused");
	ExpectU32(slot, SPARK_PAGED_KV_NO_SLOT,
		"S5 refused offer leaves NO_SLOT");
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 1u, 128ull, &slot), 0u,
		"S5 offer on a dead lane refused");
	/* Single-flight: while one reservation is outstanding no other offer
	 * succeeds. */
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot), 1u,
		"S5 aligned offer granted");
	ExpectU32(slot, 0u, "S5 single slot handed out");
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 192ull, &slot2), 0u,
		"S5 second offer while reserved refused");
	/* A commit without a live reservation is a no-op. */
	SparkPagedKvCheckpointAbort(&cache, 0u, slot);
	SparkPagedKvCheckpointCommit(&cache, 0u, slot, 128ull);
	/* F4 guard class: the boundary sits inside counted core blocks but
	 * PAST the committed frontier (130 < 192): binding would witness a
	 * still-open mutable block - refused, reservation released. */
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 192ull, &slot), 1u,
		"S5 past-frontier offer granted");
	SparkPagedKvCheckpointCommit(&cache, 0u, slot, 192ull);
	/* The refusal freed the single flight AND left the slot dead: the
	 * next offer hands out the same slot and a proper commit sticks. */
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot), 1u,
		"S5 slot reusable after refused commit");
	ExpectU32(slot, 0u, "S5 same physical slot handed back");
	SparkPagedKvCheckpointCommit(&cache, 0u, slot, 128ull);
	/* The bound witness carries lane1's identical-prefix admit at exactly
	 * the donor depth - including when the prompt is LONGER. */
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 1u, ext, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S5 matcher admit");
	ExpectU32(match.block_count, 2u, "S5 witnessed depth two");
	ExpectU32(match.checkpoint_slot, slot, "S5 witnessed via that slot");
	SparkPagedKvDestroy(&cache);
}

/* ---- S6: borrowed scratch, exhaustion, exact reset accounting -------------- */

static void TestCoverAndExhaustion(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t a[128];
	uint32_t blocks1[TK_BLOCKS_PER_LANE];
	uint32_t counts1[1];
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	TestFill(a, 31u, 128u);
	SparkPagedKvAdmit(&cache, 0u, a, 128u, 0);
	/* Borrow-only growth past the committed frontier: PRIVATE scratch,
	 * never published, invisible to the committed frontier. */
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 0u, 300ull, 0, 0u),
		(uint32_t)SPARK_STATUS_OK, "S6 borrow-only cover");
	ExpectU32(s_counts[0], 5u, "S6 row covered to five");
	ExpectU32(cache.lane_core_blocks[0], 2u,
		"S6 three ordinals are borrowed scratch");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 0u), 128ull,
		"S6 scratch does not advance the frontier");
	/* Once scratch is outstanding, token appends are IGNORED (the device
	 * truth lives in private blocks): coverage stays borrow-only and the
	 * frontier must not move. */
	{
		uint32_t more[64];
		TestFill(more, 32u, 64u);
		ExpectU32((uint32_t)SparkPagedKvCover(&cache, 0u, 320ull, more,
			64u), (uint32_t)SPARK_STATUS_OK,
			"S6 token cover over scratch accepted");
		ExpectU64(SparkPagedKvCommittedTokens(&cache, 0u), 128ull,
			"S6 frontier frozen while scratch outstanding");
		ExpectU32(s_counts[0], 5u, "S6 row unchanged");
	}
	/* Reset returns the borrowed ordinal to the free list; the published
	 * pair goes back to cached-not-free. */
	{
		uint32_t free_before = SparkPagedKvFreeBlocks(&cache);
		SparkPagedKvLaneReset(&cache, 0u);
		ExpectU32(SparkPagedKvFreeBlocks(&cache), free_before + 3u,
			"S6 reset returns exactly the borrowed blocks");
		ExpectU32(s_counts[0], 0u, "S6 row cleared");
	}
	SparkPagedKvDestroy(&cache);

	/* DETERMINISTIC EXHAUSTION: four-block pool, a two-published-block
	 * walk, then borrow-only growth demanding five. The third borrow
	 * fails (the published pair is referenced by the live sequence and
	 * unevictable), the partial row STAYS COUNTED, and reset returns the
	 * borrows - the F1 orphaned-borrow bug class, pinned. */
	TestFillConfig(&config, 1u, 1u, 4u, 8u);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, blocks1,
		counts1);
	TestFill(a, 41u, 128u);
	SparkPagedKvAdmit(&cache, 0u, a, 128u, 0);
	ExpectU32(SparkPagedKvFreeBlocks(&cache), 2u, "S6x pool after walk");
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 0u, 320ull, 0, 0u),
		(uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED,
		"S6x exhaustion refused");
	ExpectU32(counts1[0], 4u, "S6x partial growth stays counted (F1)");
	ExpectU32(SparkPagedKvFreeBlocks(&cache), 0u, "S6x pool empty");
	SparkPagedKvLaneReset(&cache, 0u);
	ExpectU32(SparkPagedKvFreeBlocks(&cache), 2u,
		"S6x reset returns scratch; published stay cached");
	ExpectU32(counts1[0], 0u, "S6x row cleared on reset");
	SparkPagedKvDestroy(&cache);
}
/* ---- S7: a cross-lane LRU steal shortens, never corrupts -------------------- */

static void TestLruSteal(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t a[192], b[192];
	uint32_t slot;
	SparkPagedKvMatch match;
	TestFillConfig(&config, 1u, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	TestFill(a, 51u, 192u);
	TestFill(b, 52u, 192u);
	/* Lane0 walks A and binds the only slot at boundary two. */
	SparkPagedKvAdmit(&cache, 0u, a, 192u, &match);
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot), 1u,
		"S7 lane0 offer");
	SparkPagedKvCheckpointCommit(&cache, 0u, slot, 128ull);
	/* Lane1 walks DIFFERENT content B: no dead slot, no same-lane
	 * shallower victim - the global-LRU steal takes lane0's checkpoint
	 * (its witness leaves with it). */
	SparkPagedKvAdmit(&cache, 1u, b, 192u, &match);
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 1u, 128ull, &slot), 1u,
		"S7 steal offer granted");
	SparkPagedKvCheckpointCommit(&cache, 1u, slot, 128ull);
	/* The stolen witness must NEVER turn A's match wrong: lane2 sees no
	 * witnessed boundary and restarts privately instead. */
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 2u, a, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S7 A after steal");
	ExpectU32(match.block_count, 0u,
		"S7 stolen witness shortens A to a private walk");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 2u), 63ull,
		"S7 A restarted at the clamp offset");
	/* And the SURVIVOR still reuses exactly: B matches at donor depth
	 * through the same physical block lane1 attached. */
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 3u, b, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S7 B survivor admit");
	ExpectU32(match.block_count, 2u, "S7 survivor witnessed depth two");
	Expect(s_blocks[3 * TK_BLOCKS_PER_LANE + 0] ==
		s_blocks[1 * TK_BLOCKS_PER_LANE + 0],
		"S7 survivor shares the physical block");
	SparkPagedKvDestroy(&cache);
}

/* ---- S8: published blocks outlive their lane; scratch does not ------------- */

static void TestResetSemantics(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t a[128], b[128];
	uint32_t slot0, slot1;
	SparkPagedKvMatch match;
	TestFillConfig(&config, TK_SLOTS, TK_LANES, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, s_blocks,
		s_counts);
	TestFill(a, 61u, 128u);
	TestFill(b, 62u, 128u);
	/* Lane0: walk, borrow scratch over it, bind a checkpoint. */
	SparkPagedKvAdmit(&cache, 0u, a, 128u, &match);
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 0u, 192ull, 0, 0u),
		(uint32_t)SPARK_STATUS_OK, "S8 lane0 scratch cover");
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot0), 1u,
		"S8 lane0 offer");
	SparkPagedKvCheckpointCommit(&cache, 0u, slot0, 128ull);
	/* Lane1: different content, second slot. */
	SparkPagedKvAdmit(&cache, 1u, b, 128u, &match);
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 1u, 128ull, &slot1), 1u,
		"S8 lane1 offer");
	Expect(slot1 != slot0, "S8 distinct slots handed out");
	SparkPagedKvCheckpointCommit(&cache, 1u, slot1, 128ull);
	/* Resetting lane0 returns its scratch, clears its row and ITS
	 * checkpoint; lane1's residency and checkpoint are untouched. */
	{
		uint32_t free_before = SparkPagedKvFreeBlocks(&cache);
		SparkPagedKvLaneReset(&cache, 0u);
		ExpectU32(SparkPagedKvFreeBlocks(&cache), free_before + 1u,
			"S8 lane0 scratch returned");
		ExpectU32(s_counts[0], 0u, "S8 lane0 row cleared");
	}
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 3u, b, 128u, &match),
		(uint32_t)SPARK_STATUS_OK, "S8 B matcher after lane0 reset");
	ExpectU32(match.block_count, 2u, "S8 lane1 checkpoint survived reset");
	/* A's published blocks persist for CONTENT matching, but with the
	 * lane0 witness gone the reuse is unwitnessed: private restart,
	 * never unsound attachment. */
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 2u, a, 128u, &match),
		(uint32_t)SPARK_STATUS_OK, "S8 A after witness loss");
	ExpectU32(match.block_count, 0u,
		"S8 dead witness -> zero reuse, content alone insufficient");
	SparkPagedKvDestroy(&cache);
}

/* ---- S9: checkpoint_slot_count == 0 disables reuse entirely ----------------- */

static void TestReuseDisabled(void)
{
	SparkPagedKv cache;
	SparkPagedKvConfiguration config;
	uint32_t blocks1[TK_BLOCKS_PER_LANE];
	uint32_t counts1[1];
	uint32_t a[192];
	uint32_t slot;
	SparkPagedKvMatch match;
	TestFillConfig(&config, 0u, 1u, TK_PHYSICAL, TK_LOGICAL);
	SparkPagedKvInitialize(&cache, &config, &TestGeometry, blocks1,
		counts1);
	TestFill(a, 71u, 192u);
	ExpectU32((uint32_t)SparkPagedKvAdmit(&cache, 0u, a, 192u, &match),
		(uint32_t)SPARK_STATUS_OK, "S9 admit without reuse");
	ExpectU32(match.block_count, 0u, "S9 nothing reused");
	ExpectU32(counts1[0], 0u, "S9 nothing published onto the row");
	ExpectU64(SparkPagedKvCommittedTokens(&cache, 0u), 0ull,
		"S9 no sequence bound");
	ExpectU32(SparkPagedKvFreeBlocks(&cache), TK_PHYSICAL,
		"S9 pool untouched by admit");
	ExpectU32(SparkPagedKvCheckpointOffer(&cache, 0u, 128ull, &slot), 0u,
		"S9 offers always refused");
	/* All coverage is borrowed scratch, returned on reset. */
	ExpectU32((uint32_t)SparkPagedKvCover(&cache, 0u, 192ull, 0, 0u),
		(uint32_t)SPARK_STATUS_OK, "S9 scratch-only cover");
	ExpectU32(counts1[0], 3u, "S9 row all borrowed");
	ExpectU32(SparkPagedKvFreeBlocks(&cache), TK_PHYSICAL - 3u,
		"S9 borrows move the free count");
	SparkPagedKvLaneReset(&cache, 0u);
	ExpectU32(SparkPagedKvFreeBlocks(&cache), TK_PHYSICAL,
		"S9 reset restores the pool exactly");
	SparkPagedKvDestroy(&cache);
}

/* ---- S10/S11: id->slot index churn oracle + integrity audit --------------- */

/* Drives the raw prefix-cache core's PUBLIC api through thousands of
 * randomized admit/release/query rounds over FOUR slots and THOUSANDS of
 * candidate ids, asserting every status against a shadow linear-scan
 * oracle (S10), then auditing the free list for duplicates, overruns,
 * count agreement, and FREE-state truth (S11). This pins the M2
 * id->slot index contract - including backward-shift deletion - at the
 * same gate that pins every other engine scenario. Deterministic seed.
 *
 * Allocation determinism trick: the corpus holds exactly 32 distinct
 * token blocks; once all are published, every admit is pure attachment,
 * so CAPACITY_EXCEEDED can only mean 'no free slot' - never a mid-run
 * pool drain - keeping the oracle exact. */

static uint32_t s10_rng = 0x12345678u;
static uint32_t S10Rand(void)
{
	s10_rng ^= s10_rng << 13u;
	s10_rng ^= s10_rng >> 17u;
	s10_rng ^= s10_rng << 5u;
	return s10_rng;
}

#define S10_SLOTS 8u
#define S10_TOKENS 512u          /* 8 blocks of 64 */
#define S10_SALTS 64u            /* distinct token blocks total */
#define S10_IDS 4096u            /* candidate id space */
#define S10_ROUNDS 4000u

/* Sweep-query EVERY shadow-live id: a broken probe chain (e.g. a bad
 * backward-shift deletion leaving a hole) loses exactly the ids behind
 * the hole, and this sweep detects the loss on the very next op. */
static void S10SweepLive(SparkPrefixCacheCore *core,
	const uint64_t *live_id)
{
	uint32_t slot;
	for (slot = 0u; slot < S10_SLOTS; slot++)
		if (live_id[slot] != UINT64_MAX)
			ExpectU64(SparkPrefixCacheCoreSequenceTokenCount(
				core, live_id[slot]), S10_TOKENS,
				"S10 sweep: live id resolves");
}

static void TestIdIndexChurn(void)
{
	SparkPrefixCacheCoreConfiguration cfg;
	SparkPrefixCacheCore *core;
	uint32_t *corpus[S10_SALTS];
	uint64_t live_id[S10_SLOTS];
	uint32_t salt, i, round;
	core = (SparkPrefixCacheCore *)calloc(1, sizeof(*core));
	memset(&cfg, 0, sizeof(cfg));
	cfg.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	cfg.descriptor_bytes =
		SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	cfg.block_token_count = 64u;
	cfg.block_stride_bytes = TK_STRIDE;
	cfg.block_count = 256u;
	cfg.max_sequence_count = S10_SLOTS;
	cfg.sequence_block_capacity = 10u;
	cfg.hash_bucket_count = 64u;
	ExpectU32((uint32_t)SparkPrefixCacheCoreInitialize(core, &cfg),
		(uint32_t)SPARK_STATUS_OK, "S10 init");
	for (salt = 0u; salt < S10_SALTS; salt++)
	{
		corpus[salt] =
			(uint32_t *)malloc(S10_TOKENS * sizeof(uint32_t));
		TestFill(corpus[salt], 500u + salt, S10_TOKENS);
	}
	for (i = 0u; i < S10_SLOTS; i++)
		live_id[i] = UINT64_MAX; /* empty marker */
	for (round = 0u; round < S10_ROUNDS; round++)
	{
		uint64_t id = UINT64_C(0x7000000000000000) +
			(uint64_t)(S10Rand() % S10_IDS);
		uint32_t op = S10Rand() % 5u;
		uint32_t slot, found = 0u, free_slot = S10_SLOTS;
		/* Shadow oracle: linear scan mirrors CoreFindSequence. */
		for (slot = 0u; slot < S10_SLOTS; slot++)
		{
			if (live_id[slot] != UINT64_MAX &&
				live_id[slot] == id)
				found = 1u;
			if (live_id[slot] == UINT64_MAX &&
				free_slot == S10_SLOTS)
				free_slot = slot;
		}
		if (op <= 1u) /* admit attempt */
		{
			uint32_t m = 0u;
			SparkStatus st = SparkPrefixCacheCoreAdmitSequence(
				core, id, corpus[S10Rand() % S10_SALTS],
				S10_TOKENS, &m);
			if (found != 0u)
				ExpectU32((uint32_t)st,
					(uint32_t)SPARK_STATUS_DUPLICATE,
					"S10 live id -> DUPLICATE");
			else if (free_slot != S10_SLOTS)
			{
				ExpectU32((uint32_t)st,
					(uint32_t)SPARK_STATUS_OK,
					"S10 cold id -> OK");
				live_id[free_slot] = id;
			}
			else
				ExpectU32((uint32_t)st,
					(uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED,
					"S10 full table -> CAPACITY_EXCEEDED");
		}
		else if (op == 2u) /* release attempt */
		{
			SparkStatus st = SparkPrefixCacheCoreReleaseSequence(
				core, id);
			if (found != 0u)
			{
				ExpectU32((uint32_t)st, (uint32_t)SPARK_STATUS_OK,
					"S10 live id -> release OK");
				for (slot = 0u; slot < S10_SLOTS; slot++)
					if (live_id[slot] == id)
						live_id[slot] = UINT64_MAX;
			}
			else
				ExpectU32((uint32_t)st,
					(uint32_t)SPARK_STATUS_NOT_FOUND,
					"S10 unknown id -> NOT_FOUND");
		}
		else /* query: committed tokens iff live, else 0 */
		{
			uint64_t got = SparkPrefixCacheCoreSequenceTokenCount(
				core, id);
			ExpectU64(got, found != 0u ? S10_TOKENS : 0ull,
				"S10 query matches shadow liveness");
		}
		if (op != 3u || (round & 7u) == 0u)
			S10SweepLive(core, live_id);
	}
	/* Drain: every still-live id must release cleanly (index unbinds). */
	for (i = 0u; i < S10_SLOTS; i++)
		if (live_id[i] != UINT64_MAX)
			ExpectU32((uint32_t)SparkPrefixCacheCoreReleaseSequence(
				core, live_id[i]), (uint32_t)SPARK_STATUS_OK,
				"S10 drain release");
	ExpectU32(core->live_sequence_count, 0u, "S10 drained");

	/* ---- S11: free-list + index integrity audit ----------------------- */
	{
		SparkPrefixCacheCoreStats stats;
		uint8_t *seen = (uint8_t *)calloc(core->block_count, 1);
		uint32_t node, walked = 0u, bidx;
		Expect(seen != 0, "S11 audit alloc");
		SparkPrefixCacheCoreQueryStats(core, &stats);
		node = core->free_block_head;
		while (node != SPARK_PREFIX_CACHE_CORE_NO_BLOCK)
		{
			Expect(node < core->block_count && seen[node] == 0u,
				"S11 free list duplicate-free and in range");
			if (node >= core->block_count || seen[node] != 0u)
				break;
			seen[node] = 1u;
			walked++;
			node = core->blocks[node].free_next;
		}
		ExpectU32(walked, stats.free_block_count,
			"S11 free-list length equals stats");
		for (bidx = 0u; bidx < core->block_count; bidx++)
			Expect(seen[bidx] != 0u ||
				core->blocks[bidx].state !=
					SPARK_PREFIX_CACHE_CORE_BLOCK_FREE,
				"S11 every FREE block sits on the list");
		free(seen);
	}
	for (salt = 0u; salt < S10_SALTS; salt++)
		free(corpus[salt]);
	SparkPrefixCacheCoreDestroy(core);
	free(core);
}

int main(void)
{
	TestPrimitives();
	TestInitializeValidation();
	TestColdWalk();
	TestUnwitnessedClamp();
	TestWitnessedReuse();
	TestCheckpointLifecycle();
	TestCoverAndExhaustion();
	TestLruSteal();
	TestResetSemantics();
	TestReuseDisabled();
	TestIdIndexChurn();
	printf("test_paged_kv_core: %d failures\n", g_failures);
	return g_failures != 0 ? 1 : 0;
}