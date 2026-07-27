// Shared-prefix admission against the KV state production actually runs.
//
// SparkGlm52RingWorkControlKvState owns block identity, residency, backing and
// the share refcount in one place. A B8 chat batch presenting one prefix must
// therefore produce ONE physical block per shared logical block and eight
// directory entries pointing at it, not eight physical blocks.
//
// The invariant that makes this safe is that only committed blocks may carry a
// content key. Speculative and MTP draft tokens sit above the committed
// frontier, so the block a draft lands in stays private and a rejected draft can
// never reach another sequence. Both properties are asserted here.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_ring_work_control.h"

#define SPARK_TEST_SHARED_ROWS 8u
#define SPARK_TEST_BLOCK_TOKENS 64u
#define SPARK_TEST_SHARED_BLOCKS 4u
#define SPARK_TEST_LANE_STRIDE 8u
#define SPARK_TEST_PHYSICAL_BLOCKS 128u
#define SPARK_TEST_INDEX_CAPACITY 1024u

typedef struct SparkTestSharedStorage
{
	uint32_t physical_blocks[SPARK_TEST_SHARED_ROWS * SPARK_TEST_LANE_STRIDE];
	uint32_t lane_counts[SPARK_TEST_SHARED_ROWS];
	uint8_t block_states[SPARK_TEST_PHYSICAL_BLOCKS];
	SparkGlm52RingKvKey block_keys[SPARK_TEST_PHYSICAL_BLOCKS];
	uint64_t block_last_used_epochs[SPARK_TEST_PHYSICAL_BLOCKS];
	uint32_t block_pin_counts[SPARK_TEST_PHYSICAL_BLOCKS];
	SparkGlm52RingWorkControlKvDirectoryEntry
		directory_entries[SPARK_TEST_INDEX_CAPACITY];
	SparkGlm52RingWorkControlKvBlockEntry
		block_entries[SPARK_TEST_INDEX_CAPACITY];
	SparkGlm52RingKvKey lane_block_keys[
		SPARK_TEST_SHARED_ROWS * SPARK_TEST_LANE_STRIDE];
} SparkTestSharedStorage;

static SparkTestSharedStorage storage;

static void SparkTestSharedInitializeState(SparkGlm52RingWorkControlKvState *state,uint32_t lane_count)
{
	memset(&storage,0,sizeof(storage));
	assert(SparkGlm52RingWorkControlInitializeKvState(state,lane_count,SPARK_TEST_LANE_STRIDE,SPARK_TEST_BLOCK_TOKENS,SPARK_TEST_PHYSICAL_BLOCKS,SPARK_TEST_INDEX_CAPACITY,SPARK_TEST_INDEX_CAPACITY,storage.physical_blocks,storage.lane_counts,storage.block_states,storage.block_keys,storage.block_last_used_epochs,storage.directory_entries,storage.block_entries) == SPARK_STATUS_OK);
	assert(SparkGlm52RingWorkControlConfigureKvPins(state,storage.block_pin_counts) == SPARK_STATUS_OK);
}

// Every lane presents the same prefix digest for the same logical block, which
// is what a shared system prompt looks like once the caller has rolled a hash
// over the token ids.
static void SparkTestSharedPublishPrefixKeys(uint32_t lane_count,uint32_t shared_block_count)
{
	uint32_t lane_index,block_index;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		for (block_index = 0u; block_index < shared_block_count; ++block_index)
			storage.lane_block_keys[(lane_index * SPARK_TEST_LANE_STRIDE) + block_index] = SparkGlm52RingWorkControlContentKey(0xA5A5000000000000ull + block_index,0x5A5A000000000000ull + block_index);
	}
}

static void SparkTestSharedInitializePacket(SparkGlm52RingWorkControlPacket *packet,uint32_t lane_count,uint32_t context_token_count)
{
	uint32_t lane_index;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
	packet->control_generation = SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
	packet->active_sequence_count = lane_count;
	packet->lane_count = lane_count;
	packet->rows_per_lane = 1u;
	packet->execution_row_count = lane_count;
	packet->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
	packet->descriptor_bytes = SparkGlm52RingWorkControlCalculatePacketBytes(lane_count);
	packet->request_id = 1u;
	packet->sequence_id = 1000u;
	packet->new_token_count = 1u;
	packet->block_token_count = SPARK_TEST_BLOCK_TOKENS;
	packet->kv_block_table_token_count = context_token_count;
	packet->max_blocks_per_sequence = SPARK_TEST_LANE_STRIDE;
	packet->sequence_position = context_token_count - 1u;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		packet->lanes[lane_index].request_id = 1u + lane_index;
		packet->lanes[lane_index].sequence_id = 1000u + lane_index;
		packet->lanes[lane_index].sequence_position = context_token_count - 1u;
		packet->lanes[lane_index].request_slot_index = lane_index;
		packet->lanes[lane_index].context_token_count = context_token_count;
	}
}

static uint32_t SparkTestSharedDistinctPhysicalBlocks(const SparkGlm52RingWorkControlKvState *state,uint32_t lane_count,uint32_t block_index)
{
	uint32_t lane_index,other_index,distinct_count;
	distinct_count = 0u;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		for (other_index = 0u; other_index < lane_index; ++other_index)
		{
			if (state->physical_block_indices[(other_index * state->lane_stride) + block_index] == state->physical_block_indices[(lane_index * state->lane_stride) + block_index])
				break;
		}
		if (other_index == lane_index)
			distinct_count += 1u;
	}
	return distinct_count;
}

// Eight rows, one prefix: the shared blocks collapse to one physical block each
// and the divergent tail stays private to every row.
static void SparkTestSharedPrefixCollapses(void)
{
	SparkGlm52RingWorkControlPacket packet;
	SparkGlm52RingWorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t block_index,context_token_count,expected_block_count;
	context_token_count = (SPARK_TEST_SHARED_BLOCKS * SPARK_TEST_BLOCK_TOKENS) + 1u;
	expected_block_count = SPARK_TEST_SHARED_BLOCKS + 1u;
	SparkTestSharedInitializeState(&state,SPARK_TEST_SHARED_ROWS);
	SparkTestSharedPublishPrefixKeys(SPARK_TEST_SHARED_ROWS,SPARK_TEST_SHARED_BLOCKS);
	assert(SparkGlm52RingWorkControlConfigureKvSharing(&state,storage.lane_block_keys,SPARK_TEST_LANE_STRIDE) == SPARK_STATUS_OK);
	SparkTestSharedInitializePacket(&packet,SPARK_TEST_SHARED_ROWS,context_token_count);
	packet.flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL;
	assert(SparkGlm52RingWorkControlBuildHostKvBlockTable(&packet,&state,&view) == SPARK_STATUS_OK);
	for (block_index = 0u; block_index < SPARK_TEST_SHARED_BLOCKS; ++block_index)
		assert(SparkTestSharedDistinctPhysicalBlocks(&state,SPARK_TEST_SHARED_ROWS,block_index) == 1u);
	assert(SparkTestSharedDistinctPhysicalBlocks(&state,SPARK_TEST_SHARED_ROWS,SPARK_TEST_SHARED_BLOCKS) == SPARK_TEST_SHARED_ROWS);
	assert(state.directory_entry_count == SPARK_TEST_SHARED_ROWS * expected_block_count);
	assert(state.block_entry_count == SPARK_TEST_SHARED_BLOCKS + SPARK_TEST_SHARED_ROWS);
	assert(state.share_admit_count == SPARK_TEST_SHARED_BLOCKS + SPARK_TEST_SHARED_ROWS);
	assert(state.share_hit_count == (SPARK_TEST_SHARED_ROWS - 1u) * SPARK_TEST_SHARED_BLOCKS);
	printf("B%u shared prefix: %u physical blocks for %u sequence slots\n",SPARK_TEST_SHARED_ROWS,state.block_entry_count,state.directory_entry_count);
	printf("  admissions=%llu shares=%llu (unshared would be %u)\n",(unsigned long long)state.share_admit_count,(unsigned long long)state.share_hit_count,SPARK_TEST_SHARED_ROWS * expected_block_count);
}

// Releasing a sequence must not free a block another sequence still names.
static void SparkTestSharedReleaseIsRefcounted(void)
{
	SparkGlm52RingWorkControlPacket packet;
	SparkGlm52RingWorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t lane_index,context_token_count,expected_block_count,shared_physical_block;
	context_token_count = (SPARK_TEST_SHARED_BLOCKS * SPARK_TEST_BLOCK_TOKENS) + 1u;
	expected_block_count = SPARK_TEST_SHARED_BLOCKS + 1u;
	SparkTestSharedInitializeState(&state,SPARK_TEST_SHARED_ROWS);
	SparkTestSharedPublishPrefixKeys(SPARK_TEST_SHARED_ROWS,SPARK_TEST_SHARED_BLOCKS);
	assert(SparkGlm52RingWorkControlConfigureKvSharing(&state,storage.lane_block_keys,SPARK_TEST_LANE_STRIDE) == SPARK_STATUS_OK);
	SparkTestSharedInitializePacket(&packet,SPARK_TEST_SHARED_ROWS,context_token_count);
	packet.flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL;
	assert(SparkGlm52RingWorkControlBuildHostKvBlockTable(&packet,&state,&view) == SPARK_STATUS_OK);
	shared_physical_block = state.physical_block_indices[0u];
	for (lane_index = 0u; lane_index + 1u < SPARK_TEST_SHARED_ROWS; ++lane_index)
	{
		assert(SparkGlm52RingWorkControlReleaseSequence(&state,1000u + lane_index,expected_block_count) == SPARK_STATUS_OK);
		assert(SparkGlm52RingWorkControlKvKeyEqual(state.physical_block_keys[shared_physical_block],SparkGlm52RingWorkControlContentKey(0xA5A5000000000000ull,0x5A5A000000000000ull)) != 0u);
	}
	assert(state.block_entry_count == SPARK_TEST_SHARED_BLOCKS + 1u);
	assert(SparkGlm52RingWorkControlReleaseSequence(&state,1000u + lane_index,expected_block_count) == SPARK_STATUS_OK);
	assert(state.block_entry_count == 0u);
	assert(state.directory_entry_count == 0u);
	assert(state.allocated_physical_block_count == 0u);
	printf("  %u releases keep the block, the last frees it\n",SPARK_TEST_SHARED_ROWS - 1u);
}

// The committed frontier is what keeps a rejected draft from reaching another
// sequence: any block that is not entirely below it stays private, so no block
// a draft can still be written into is ever shareable. Asserted directly across
// the whole legal draft range because it is the load-bearing invariant.
static void SparkTestSharedFrontierExcludesDrafts(void)
{
	SparkGlm52RingWorkControlPacket packet;
	uint64_t frontier,block_end_token;
	uint32_t draft_count,block_index,last_committed_block;
	SparkTestSharedInitializePacket(&packet,1u,SPARK_TEST_SHARED_BLOCKS * SPARK_TEST_BLOCK_TOKENS);
	for (draft_count = 0u; draft_count <= SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT; ++draft_count)
	{
		packet.lanes[0u].mtp_draft_token_count = draft_count;
		packet.lanes[0u].speculative_token_count = draft_count;
		frontier = SparkGlm52RingWorkControlKvCommittedFrontier(&packet.lanes[0u]);
		assert(frontier == (uint64_t)packet.lanes[0u].context_token_count - (2u * draft_count));
		last_committed_block = draft_count == 0u ? SPARK_TEST_SHARED_BLOCKS : SPARK_TEST_SHARED_BLOCKS - 1u;
		for (block_index = 0u; block_index < SPARK_TEST_SHARED_BLOCKS; ++block_index)
		{
			block_end_token = ((uint64_t)block_index + 1u) * SPARK_TEST_BLOCK_TOKENS;
			assert((block_end_token <= frontier ? 1u : 0u) == (block_index < last_committed_block ? 1u : 0u));
		}
	}
	// An outstanding count above the context length must yield an empty
	// frontier, so nothing is shareable, rather than wrapping and sharing all.
	packet.lanes[0u].mtp_draft_token_count = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
	packet.lanes[0u].speculative_token_count = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
	packet.lanes[0u].context_token_count = 1u;
	assert(SparkGlm52RingWorkControlKvCommittedFrontier(&packet.lanes[0u]) == 0u);
	printf("  committed frontier excludes every draft-bearing block\n");
}

// A block first computed privately keeps its physical block when it later
// publishes content, so the bytes are never recomputed.
static void SparkTestSharedPromotionKeepsBytes(void)
{
	SparkGlm52RingWorkControlPacket packet;
	SparkGlm52RingWorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	uint32_t private_physical_block,shared_physical_block,context_token_count;
	context_token_count = (SPARK_TEST_SHARED_BLOCKS * SPARK_TEST_BLOCK_TOKENS) + 1u;
	SparkTestSharedInitializeState(&state,SPARK_TEST_SHARED_ROWS);
	SparkTestSharedInitializePacket(&packet,1u,context_token_count);
	packet.flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL;
	assert(SparkGlm52RingWorkControlBuildHostKvBlockTable(&packet,&state,&view) == SPARK_STATUS_OK);
	private_physical_block = state.physical_block_indices[0u];
	assert(state.block_entry_count == SPARK_TEST_SHARED_BLOCKS + 1u);
	SparkTestSharedPublishPrefixKeys(1u,SPARK_TEST_SHARED_BLOCKS);
	assert(SparkGlm52RingWorkControlConfigureKvSharing(&state,storage.lane_block_keys,SPARK_TEST_LANE_STRIDE) == SPARK_STATUS_OK);
	assert(SparkGlm52RingWorkControlBuildHostKvBlockTable(&packet,&state,&view) == SPARK_STATUS_OK);
	shared_physical_block = state.physical_block_indices[0u];
	assert(shared_physical_block == private_physical_block);
	assert(state.block_entry_count == SPARK_TEST_SHARED_BLOCKS + 1u);
	assert(SparkGlm52RingWorkControlKvKeyEqual(state.physical_block_keys[shared_physical_block],SparkGlm52RingWorkControlContentKey(0xA5A5000000000000ull,0x5A5A000000000000ull)) != 0u);
	printf("  promotion rekeys block %u in place, no recompute\n",shared_physical_block);
}

// A private key and a content key can never collide, whatever the caller feeds
// in, because they occupy disjoint halves of the key space.
static void SparkTestSharedKeyDomainsAreDisjoint(void)
{
	SparkGlm52RingKvKey private_key,content_key;
	uint32_t index;
	for (index = 0u; index < 4096u; ++index)
	{
		private_key = SparkGlm52RingWorkControlPrivateKey(1u + index,index);
		content_key = SparkGlm52RingWorkControlContentKey(private_key.low,private_key.high);
		assert(SparkGlm52RingWorkControlKvKeyEqual(private_key,content_key) == 0u);
		assert((private_key.low != 0u) || (private_key.high != 0u));
		assert((content_key.low != 0u) || (content_key.high != 0u));
	}
	printf("  key domains disjoint over %u samples\n",4096u);
}

int main(void)
{
	SparkTestSharedPrefixCollapses();
	SparkTestSharedReleaseIsRefcounted();
	SparkTestSharedFrontierExcludesDrafts();
	SparkTestSharedPromotionKeepsBytes();
	SparkTestSharedKeyDomainsAreDisjoint();
	printf("\nPASS - shared prefix collapses on the production KV state\n");
	return 0;
}
