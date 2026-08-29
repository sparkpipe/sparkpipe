// The topology switch: quiesce bounds, checkpoint pins, the swap poll loop,
// warm versus recompute resume, and the budget arithmetic.
//
// Everything the machine decides is schedule arithmetic over two vtables -
// the swap device and the checkpoint write path - so the whole protocol is
// checkable on a host: the swap mock completes after a programmed number of
// polls and remembers which recipes it was asked to move between, and the
// write mock remembers every manifest offset, which is how the assertions
// can say "the pinned blocks survived a full tier" and "the drain cost no
// extra step" rather than hoping.
#include "sparkpipe/spark_topology_switch.h"

#include "sparkpipe/spark_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if ( !condition )
		++failures;
}

#define MOCK_MAX_READS 64u
#define MOCK_BLOCK_BYTES 4096u
#define SWITCH_MAX_SEQS 4u
#define SWITCH_MAX_BLOCKS 8u
#define TEST_NAMESPACE 0x5150u

// -- the mock tier drive -----------------------------------------------------

typedef struct MockRead
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint8_t active;
}
MockRead;

typedef struct MockDevice
{
	MockRead reads[MOCK_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;
	uint32_t submits;
}
MockDevice;

static SparkStatus MockSubmitRead(
	void *context, uint64_t device_offset, void *destination,
	uint32_t bytes, uint64_t *ticket_out)
{
	MockDevice *device = (MockDevice *)context;
	uint32_t index;
	if ( bytes != MOCK_BLOCK_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < MOCK_MAX_READS; ++index )
	{
		if ( device->reads[index].active )
			continue;
		device->reads[index].active = 1u;
		device->reads[index].ticket = device->next_ticket++;
		device->reads[index].offset = device_offset;
		device->reads[index].destination = (uint8_t *)destination;
		device->reads[index].polls_left = device->polls_per_read;
		device->submits++;
		*ticket_out = device->reads[index].ticket;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus MockPollRead(void *context, uint64_t ticket)
{
	MockDevice *device = (MockDevice *)context;
	uint32_t index;
	for ( index = 0u; index < MOCK_MAX_READS; ++index )
	{
		MockRead *read = &device->reads[index];
		if ( !read->active || read->ticket != ticket )
			continue;
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
		}
		read->active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

// -- the mock swap device ------------------------------------------------------

typedef struct MockSwap
{
	uint64_t from_recipe_id;
	uint64_t target_recipe_id;
	uint32_t polls_per_swap;         /* programmed swap latency, in polls */
	uint32_t polls_left;
	uint32_t begins;
}
MockSwap;

static SparkStatus MockBeginSwap(
	void *context,
	const SparkTopologyRecipe *from,
	const SparkTopologyRecipe *target)
{
	MockSwap *swap = (MockSwap *)context;
	swap->from_recipe_id = from->recipe_id;
	swap->target_recipe_id = target->recipe_id;
	swap->polls_left = swap->polls_per_swap;
	swap->begins++;
	return(SPARK_STATUS_OK);
}

static SparkStatus MockPollSwap(void *context)
{
	MockSwap *swap = (MockSwap *)context;
	if ( swap->polls_left != 0u )
	{
		swap->polls_left--;
		return(SPARK_STATUS_BUSY);
	}
	return(SPARK_STATUS_OK);
}

// -- the mock checkpoint write path --------------------------------------------

typedef struct MockWrites
{
	uint64_t offsets[64];
	uint32_t bytes[64];
	uint32_t count;
}
MockWrites;

static SparkStatus MockWriteBlock(
	void *context, uint64_t device_offset, const void *payload, uint32_t bytes)
{
	MockWrites *writes = (MockWrites *)context;
	(void)payload;
	if ( writes->count >= 64u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	writes->offsets[writes->count] = device_offset;
	writes->bytes[writes->count] = bytes;
	writes->count++;
	return(SPARK_STATUS_OK);
}

// -- the fixture -----------------------------------------------------------------

#define RECIPE_TP_ID 11u
#define RECIPE_PP_ID 22u

typedef struct Fixture
{
	MockDevice device;
	SparkNvmeTierDevice tier_vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration tier_config;
	MockSwap swap;
	SparkTopologySwitchSwapDevice swap_vtable;
	MockWrites writes;
	SparkTopologySwitch sw;
	SparkTopologySwitchConfiguration sw_config;
	SparkTopologyRecipe recipe_tp;
	SparkTopologyRecipe recipe_pp;
	_Alignas(64) uint8_t tier_tables[128u * 1024u];
	_Alignas(MOCK_BLOCK_BYTES) uint8_t staging[4u * MOCK_BLOCK_BYTES];
	_Alignas(64) uint8_t sw_tables[64u * 1024u];
}
Fixture;

static SparkStatus FixtureOpen(
	Fixture *fixture,
	uint32_t tier_budget_blocks,
	uint32_t swap_polls)
{
	SparkStatus status;
	uint64_t table_bytes;
	memset(fixture,0,sizeof(*fixture));
	fixture->device.polls_per_read = 2u;
	fixture->device.next_ticket = 1u;
	fixture->tier_vtable.context = &fixture->device;
	fixture->tier_vtable.submit_read = MockSubmitRead;
	fixture->tier_vtable.poll_read = MockPollRead;
	fixture->tier_vtable.cancel_read = 0;
	fixture->tier_config.abi_version = SPARK_NVME_TIER_ABI_VERSION;
	fixture->tier_config.descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	fixture->tier_config.budget_bytes = (uint64_t)tier_budget_blocks * MOCK_BLOCK_BYTES;
	fixture->tier_config.base_offset = 1u << 20;
	fixture->tier_config.block_bytes = MOCK_BLOCK_BYTES;
	fixture->tier_config.hash_bucket_count = 32u;
	fixture->tier_config.staging_buffer_count = 4u;
	fixture->tier_config.demand_reserve_buffers = 1u;
	fixture->tier_config.pending_capacity = 32u;
	fixture->tier_config.device_bytes_per_second = 2048u;
	fixture->tier_config.step_time_microseconds = 1000000u;
	table_bytes = SparkNvmeTierTableBytes(&fixture->tier_config);
	if ( table_bytes == 0u || table_bytes > sizeof(fixture->tier_tables) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkNvmeTierInitialize(&fixture->tier,&fixture->tier_config,
		&fixture->tier_vtable,fixture->tier_tables,sizeof(fixture->tier_tables),
		fixture->staging,sizeof(fixture->staging));
	if ( status != SPARK_STATUS_OK )
		return(status);
	fixture->recipe_tp.recipe_id = RECIPE_TP_ID;
	fixture->recipe_tp.weight_pack_bytes = 100000000000ULL;   /* 100 GB */
	fixture->recipe_tp.strategy = SPARK_TOPOLOGY_STRATEGY_TENSOR_PARALLEL;
	fixture->recipe_pp.recipe_id = RECIPE_PP_ID;
	fixture->recipe_pp.weight_pack_bytes = 80000000000ULL;    /* 80 GB */
	fixture->recipe_pp.strategy = SPARK_TOPOLOGY_STRATEGY_PIPELINE_PARALLEL;
	fixture->swap.polls_per_swap = swap_polls;
	fixture->swap_vtable.context = &fixture->swap;
	fixture->swap_vtable.begin_swap = MockBeginSwap;
	fixture->swap_vtable.poll_swap = MockPollSwap;
	fixture->sw_config.abi_version = SPARK_TOPOLOGY_SWITCH_ABI_VERSION;
	fixture->sw_config.descriptor_bytes = SPARK_TOPOLOGY_SWITCH_CONFIGURATION_BYTES;
	fixture->sw_config.kv_namespace = TEST_NAMESPACE;
	fixture->sw_config.max_sequences = SWITCH_MAX_SEQS;
	fixture->sw_config.max_blocks_per_sequence = SWITCH_MAX_BLOCKS;
	fixture->sw_config.step_time_microseconds = 20000u;       /* a 20 ms step */
	fixture->sw_config.manifest_block_bytes = MOCK_BLOCK_BYTES;
	fixture->sw_config.nvme_read_bytes_per_second = 5000000000ULL;   /* 5 GB/s */
	fixture->sw_config.nvme_write_bytes_per_second = 2000000000ULL;  /* 2 GB/s */
	fixture->sw_config.swap_fixed_microseconds = 500000u;            /* 0.5 s */
	fixture->sw_config.tier = &fixture->tier;
	fixture->sw_config.initial_recipe = fixture->recipe_tp;
	table_bytes = SparkTopologySwitchTableBytes(&fixture->sw_config);
	if ( table_bytes == 0u || table_bytes > sizeof(fixture->sw_tables) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SparkTopologySwitchInitialize(&fixture->sw,&fixture->sw_config,
		&fixture->swap_vtable,MockWriteBlock,&fixture->writes,fixture->sw_tables));
}

static SparkStatus FixturePublishCommitted(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	uint64_t *device_offset_out)
{
	SparkNvmeTierWriteReservation reservation;
	SparkStatus status;

	/* Serving write-back always presents the payload's SHA-256 (B3): the
	 * fixture's stand-in content is the key itself, digested for real. */
	{
		uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES];
		SparkSha256Context digest_context;
		SparkSha256Initialize(&digest_context);
		SparkSha256Update(&digest_context,&content_hash,sizeof(content_hash));
		SparkSha256Finalize(&digest_context,content_digest);
		status = SparkNvmeTierReserveWrite(tier,content_hash,content_digest,
			&reservation);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	*device_offset_out = reservation.device_offset;
	return(SparkNvmeTierCommitWrite(tier,&reservation));
}

// Publish a sequence's blocks the way serving write-back would: under the
// same namespaced key the switch computes, which is the whole reason resume
// can find them.
static void FixturePublishBlocks(
	Fixture *fixture,
	const uint64_t *content_hashes,
	uint32_t hash_count)
{
	uint32_t index;
	uint64_t offset;
	for ( index = 0u; index < hash_count; ++index )
		(void)FixturePublishCommitted(&fixture->tier,
			SparkTopologySwitchKvKey(TEST_NAMESPACE,content_hashes[index]),&offset);
}

static void FixtureTrackWithKv(
	Fixture *fixture,
	uint64_t sequence_id,
	uint32_t position_tokens,
	const uint64_t *content_hashes,
	uint32_t hash_count)
{
	expect(SparkTopologySwitchTrackSequence(&fixture->sw,sequence_id,RECIPE_TP_ID)
		== SPARK_STATUS_OK, "sequence tracks under the running recipe");
	expect(SparkTopologySwitchSetSequenceKv(&fixture->sw,sequence_id,
		position_tokens,content_hashes,hash_count) == SPARK_STATUS_OK,
		"its tier keys register");
}

// Drive Advance until STEADY, bounded so a wedged machine fails the test
// rather than hanging it. Returns the number of Advance calls used.
static uint32_t FixtureAdvanceToSteady(Fixture *fixture, uint32_t first_step)
{
	uint32_t step;
	for ( step = first_step; step < first_step + 64u; ++step )
		if ( SparkTopologySwitchAdvance(&fixture->sw,step)
			== SPARK_TOPOLOGY_SWITCH_STEADY )
			return(step - first_step + 1u);
	return(0u);
}

int main(void)
{
	printf("topology switch\n");

	printf("\nkeying: the strategy is not in the name\n");
	{
		uint64_t key_a = SparkTopologySwitchKvKey(TEST_NAMESPACE,777u);
		expect(key_a != 0u, "a key is never zero (zero means unhashed)");
		expect(SparkTopologySwitchKvKey(TEST_NAMESPACE,777u) == key_a,
			"keying is deterministic");
		expect(SparkTopologySwitchKvKey(TEST_NAMESPACE,778u) != key_a,
			"different content hashes differently");
		expect(SparkTopologySwitchKvKey(0x9999u,777u) != key_a,
			"a different model namespace hashes differently");
		/* The namespace carries model + neutral geometry and NOT the
		   strategy, so the same function names a block before and after a
		   TP<->PP switch. What would make it differ - a strategy field -
		   is absent from the signature by construction. */
	}

	printf("\ninitialisation validates the shape\n");
	{
		Fixture fixture;
		expect(FixtureOpen(&fixture,32u,0u) == SPARK_STATUS_OK,
			"a 32-record tier and the machine initialise");
		expect(SparkTopologySwitchAdmissionsOpen(&fixture.sw) == 1u,
			"admissions start open");
		expect(SparkTopologySwitchStateOf(&fixture.sw)
			== SPARK_TOPOLOGY_SWITCH_STEADY, "the machine starts STEADY");
		expect(SparkTopologySwitchCurrentRecipe(&fixture.sw)->recipe_id
			== RECIPE_TP_ID, "the running recipe is the initial one");
		{
			SparkTopologySwitchConfiguration broken = fixture.sw_config;
			broken.manifest_block_bytes = 16u;   /* cannot hold one key list */
			expect(SparkTopologySwitchInitialize(&fixture.sw,&broken,
				&fixture.swap_vtable,MockWriteBlock,&fixture.writes,
				fixture.sw_tables) == SPARK_STATUS_INVALID_ARGUMENT,
				"a manifest too small for its own key list is refused");
			broken = fixture.sw_config;
			broken.nvme_read_bytes_per_second = 0u;
			expect(SparkTopologySwitchInitialize(&fixture.sw,&broken,
				&fixture.swap_vtable,MockWriteBlock,&fixture.writes,
				fixture.sw_tables) == SPARK_STATUS_INVALID_ARGUMENT,
				"a zero bandwidth is refused: the budget would lie");
		}
	}

	printf("\nthe admission gate closes synchronously at Begin\n");
	{
		Fixture fixture;
		(void)FixtureOpen(&fixture,32u,0u);
		expect(SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_tp)
			== SPARK_STATUS_INVALID_ARGUMENT,
			"switching to the running recipe is refused, not a fast path");
		expect(SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp)
			== SPARK_STATUS_OK, "a real target starts the switch");
		expect(SparkTopologySwitchAdmissionsOpen(&fixture.sw) == 0u,
			"admissions are closed the moment Begin returns");
		expect(SparkTopologySwitchTrackSequence(&fixture.sw,99u,RECIPE_TP_ID)
			== SPARK_STATUS_BUSY, "the tracking backstop agrees");
		expect(SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_tp)
			== SPARK_STATUS_BUSY, "a second Begin mid-switch is BUSY");
		(void)FixtureAdvanceToSteady(&fixture,1u);
		expect(SparkTopologySwitchAdmissionsOpen(&fixture.sw) == 1u,
			"admissions reopen when the new recipe is resident");
		expect(SparkTopologySwitchCurrentRecipe(&fixture.sw)->recipe_id
			== RECIPE_PP_ID, "the machine now serves the target recipe");
	}

	printf("\nquiesce waits for the last boundary and costs no extra step\n");
	{
		Fixture fixture;
		uint64_t blocks_a[2] = { 1000u,1001u };
		uint64_t blocks_b[2] = { 1002u,1003u };
		SparkTopologySwitchStatistics statistics;
		(void)FixtureOpen(&fixture,32u,0u);
		FixturePublishBlocks(&fixture,blocks_a,2u);
		FixturePublishBlocks(&fixture,blocks_b,2u);
		FixtureTrackWithKv(&fixture,1u,128u,blocks_a,2u);
		FixtureTrackWithKv(&fixture,2u,96u,blocks_b,2u);
		FixtureTrackWithKv(&fixture,3u,64u,blocks_a,2u);
		expect(SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp)
			== SPARK_STATUS_OK, "the switch begins with three in flight");
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,1u);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,2u);
		expect(SparkTopologySwitchAdvance(&fixture.sw,1u)
			== SPARK_TOPOLOGY_SWITCH_QUIESCE,
			"one sequence still decoding holds the machine in QUIESCE");
		/* Sequence 3 does not reach a boundary; it finishes. Completing
		   mid-quiesce is a drain, and checkpointing finished work is waste. */
		expect(SparkTopologySwitchSequenceComplete(&fixture.sw,3u)
			== SPARK_STATUS_OK, "a sequence completing mid-quiesce drains it");
		expect(FixtureAdvanceToSteady(&fixture,2u) == 1u,
			"drained, the whole switch bar the swap is ONE advance: "
			"checkpoint and resume cost no step of their own");
		SparkTopologySwitchGetStatistics(&fixture.sw,&statistics);
		expect(statistics.sequences_completed_mid_switch == 1u,
			"the mid-switch completion is counted");
		expect(statistics.sequences_checkpointed == 2u,
			"only live sequences were checkpointed");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,1u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_WARM, "sequence 1 resumes warm");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,2u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_WARM, "sequence 2 resumes warm");
		expect(fixture.writes.count == 2u,
			"one manifest write per checkpointed sequence");
		expect(fixture.swap.from_recipe_id == RECIPE_TP_ID
			&& fixture.swap.target_recipe_id == RECIPE_PP_ID,
			"the swap device saw TP out, PP in");
		expect(statistics.switches_completed == 1u, "the switch completed");
	}

	printf("\nthe swap poll loop holds until the recipe is resident\n");
	{
		Fixture fixture;
		uint64_t blocks[2] = { 2000u,2001u };
		(void)FixtureOpen(&fixture,32u,3u);
		FixturePublishBlocks(&fixture,blocks,2u);
		FixtureTrackWithKv(&fixture,7u,128u,blocks,2u);
		(void)SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,7u);
		expect(SparkTopologySwitchAdvance(&fixture.sw,1u)
			== SPARK_TOPOLOGY_SWITCH_SWAP,
			"checkpoint and swap issue complete in the drain step");
		expect(fixture.swap.begins == 1u, "begin_swap fired exactly once");
		expect(SparkTopologySwitchAdvance(&fixture.sw,2u)
			== SPARK_TOPOLOGY_SWITCH_SWAP, "first poll: still streaming");
		expect(SparkTopologySwitchAdvance(&fixture.sw,3u)
			== SPARK_TOPOLOGY_SWITCH_SWAP, "second poll: still streaming");
		expect(SparkTopologySwitchAdvance(&fixture.sw,4u)
			== SPARK_TOPOLOGY_SWITCH_STEADY,
			"when the stream lands, resume and STEADY are the same advance");
	}

	printf("\ncheckpoint pins survive a full tier's worth of churn\n");
	{
		Fixture fixture;
		uint64_t blocks[4] = { 3000u,3001u,3002u,3003u };
		uint64_t offset;
		uint32_t index,alive = 0u;
		SparkNvmeTierStatistics tier_statistics;
		(void)FixtureOpen(&fixture,32u,2u);
		FixturePublishBlocks(&fixture,blocks,4u);
		FixtureTrackWithKv(&fixture,8u,256u,blocks,4u);
		(void)SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,8u);
		expect(SparkTopologySwitchAdvance(&fixture.sw,1u)
			== SPARK_TOPOLOGY_SWITCH_SWAP, "checkpointed, swap in flight");
		/* Mid-swap, ordinary serving write-back continues and floods the
		   32-record tier with fresh records. */
		for ( index = 0u; index < 64u; ++index )
			(void)FixturePublishCommitted(&fixture.tier,9000u + index,&offset);
		SparkNvmeTierGetStatistics(&fixture.tier,&tier_statistics);
		expect(tier_statistics.evictions != 0u, "the churn really evicted");
		expect(tier_statistics.pinned_eviction_skips != 0u,
			"the clock met the pins and stepped over them");
		for ( index = 0u; index < 4u; ++index )
			if ( SparkNvmeTierOffsetOf(&fixture.tier,
				SparkTopologySwitchKvKey(TEST_NAMESPACE,blocks[index]),0,
				&offset) == SPARK_STATUS_OK )
				alive++;
		expect(alive == 4u, "every pinned block survived the flood");
		expect(FixtureAdvanceToSteady(&fixture,2u) != 0u, "the switch lands");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,8u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_WARM,
			"and the sequence resumes warm because of it");
	}

	printf("\nresume: warm hits reload, the never-written-back recompute\n");
	{
		Fixture fixture;
		uint64_t blocks_warm[3] = { 4000u,4001u,4002u };
		uint64_t blocks_cold[2] = { 4100u,4101u };
		SparkTopologySwitchStatistics statistics;
		uint32_t submits_before;
		(void)FixtureOpen(&fixture,32u,0u);
		FixturePublishBlocks(&fixture,blocks_warm,3u);
		/* blocks_cold are never published: the tier never saw them. */
		FixtureTrackWithKv(&fixture,11u,192u,blocks_warm,3u);
		FixtureTrackWithKv(&fixture,12u,128u,blocks_cold,2u);
		(void)SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,11u);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,12u);
		submits_before = fixture.device.submits;
		expect(FixtureAdvanceToSteady(&fixture,1u) != 0u, "the switch lands");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,11u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_WARM, "published blocks resume warm");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,12u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE,
			"absent blocks go back through prefill");
		SparkTopologySwitchGetStatistics(&fixture.sw,&statistics);
		expect(statistics.tier_blocks_absent == 2u,
			"the checkpoint counted the absent blocks it could not pin");
		expect(statistics.sequences_resumed_warm == 1u
			&& statistics.sequences_resumed_recompute == 1u,
			"both classes are counted");
		(void)SparkNvmeTierPump(&fixture.tier,10u);
		expect(fixture.device.submits > submits_before,
			"the warm sequence's blocks went into the tier's lookahead: "
			"resume is a planned fetch, not a demand stall");
	}

	printf("\neviction between serving and switch sends the sequence to prefill\n");
	{
		Fixture fixture;
		uint64_t blocks[4] = { 5000u,5001u,5002u,5003u };
		uint64_t offset;
		uint32_t index;
		(void)FixtureOpen(&fixture,8u,0u);   /* an 8-record tier, on purpose */
		FixturePublishBlocks(&fixture,blocks,4u);
		FixtureTrackWithKv(&fixture,21u,256u,blocks,4u);
		/* Long after the write-back, a busy serving mix floods the tier and
		   the sequence's records age out - before any switch is requested. */
		for ( index = 0u; index < 32u; ++index )
			(void)FixturePublishCommitted(&fixture.tier,8000u + index,&offset);
		(void)SparkTopologySwitchBegin(&fixture.sw,&fixture.recipe_pp);
		(void)SparkTopologySwitchSequenceAtBoundary(&fixture.sw,21u);
		expect(FixtureAdvanceToSteady(&fixture,1u) != 0u, "the switch lands");
		expect(SparkTopologySwitchResumeClassOf(&fixture.sw,21u)
			== SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE,
			"evicted from the 1TB means recompute, by the protocol's own rule");
	}

	printf("\nthe budget arithmetic is the contract, in numbers\n");
	{
		Fixture fixture;
		SparkTopologySwitchBudget budget;
		(void)FixtureOpen(&fixture,32u,0u);
		expect(SparkTopologySwitchEstimateBudget(&fixture.sw_config,
			&fixture.recipe_pp,4u,10000000000ULL,&budget) == SPARK_STATUS_OK,
			"the estimate computes");
		expect(budget.quiesce_us == 20000u, "quiesce is one decode step");
		/* 4 manifests x 4096 B at 2 GB/s = 8 us. */
		expect(budget.checkpoint_us == 8u, "checkpoint is manifest writes");
		expect(budget.swap_fixed_us == 500000u, "the fixed cost passes through");
		/* 80 GB at 5 GB/s = 16 s. */
		expect(budget.swap_stream_us == 16000000u,
			"the stream is pack bytes over read bandwidth");
		expect(budget.resume_warm_us == 2000000u,
			"the warm reload is priced separately");
		expect(budget.total_us == 20000u + 8u + 500000u + 16000000u,
			"and the total EXCLUDES it: resume overlaps serving");
	}

	printf("\n%s (%d failure%s)\n",failures == 0 ? "PASS" : "FAIL",
		failures,failures == 1 ? "" : "s");
	return(failures == 0 ? 0 : 1);
}
