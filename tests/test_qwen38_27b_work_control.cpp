#include "sparkpipe/spark_qwen38_27b_work_control.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

extern "C" const SparkKvStoreInterface *SparkKvStoreGetInterface(void);

static SparkStatus SparkTestWaitReady(SparkStageKvClient *client, SparkQwen38_27bWorkControlKvState *kv, SparkQwen38_27bWorkControlKvBatchState *batch)
{
	uint32_t poll;
	for (poll = 0; poll < 20000u; poll++)
	{
		assert(SparkQwen38_27bWorkControlProgress(client,kv) == SPARK_STATUS_OK);
		if ( batch->state == SPARK_QWEN38_27B_WORK_CONTROL_BATCH_READY )
			return(batch->status);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return(SPARK_STATUS_BUSY);
}

static void SparkTestPlanMath(void)
{
	SparkQwen38_27bWorkControlKvPlanConfig config;
	uint32_t blocks_a[3] = {0u,1u,2u},blocks_b[1] = {7u},blocks_c[2] = {3u,4u};
	SparkQwen38_27bWorkControlPendingLane pending[4];
	uint32_t packet_lane_counts[3] = {2u,1u,1u},cumulative[3],equivalents,selected;
	memset(&config,0,sizeof(config));
	memset(pending,0,sizeof(pending));
	equivalents = SparkQwen38_27bWorkControlGdnBlockEquivalents(200u,96u);
	assert(equivalents == 3u);
	assert(SparkQwen38_27bWorkControlGdnBlockEquivalents(0u,96u) == 0u);
	// Packet 0: lane 0 needs 3 blocks + GDN, lane 1 fully resident.
	// Packet 1: lane 2 needs 1 block. Packet 2: lane 3 needs 2 + GDN.
	pending[0].sequence_id = 101u;
	pending[0].nonresident_block_count = 3u;
	pending[0].nonresident_blocks = blocks_a;
	pending[0].gdn_nonresident = 1u;
	pending[1].sequence_id = 102u;
	pending[2].sequence_id = 103u;
	pending[2].nonresident_block_count = 1u;
	pending[2].nonresident_blocks = blocks_b;
	pending[3].sequence_id = 104u;
	pending[3].nonresident_block_count = 2u;
	pending[3].nonresident_blocks = blocks_c;
	pending[3].gdn_nonresident = 1u;
	assert(SparkQwen38_27bWorkControlCumulativeNonresident(pending,4u,packet_lane_counts,3u,equivalents,cumulative) == SPARK_STATUS_OK);
	assert(cumulative[0] == 6u && cumulative[1] == 7u && cumulative[2] == 12u);
	// A miscounted queue is refused, never silently truncated.
	assert(SparkQwen38_27bWorkControlCumulativeNonresident(pending,3u,packet_lane_counts,3u,equivalents,cumulative) == SPARK_STATUS_INVALID_ARGUMENT);
	config.lookahead_packet_count = 8u;
	config.physical_block_capacity = 64u;
	config.allocated_physical_block_count = 0u;
	config.staging_block_capacity = 64u;
	selected = SparkQwen38_27bWorkControlSelectRestorePackets(&config,3u,cumulative);
	assert(selected == 3u);
	// Staging pressure caps the horizon at two packets.
	config.staging_block_capacity = 7u;
	selected = SparkQwen38_27bWorkControlSelectRestorePackets(&config,3u,cumulative);
	assert(selected == 2u);
	// Physical pressure caps at packet zero alone.
	config.staging_block_capacity = 64u;
	config.allocated_physical_block_count = 60u;
	selected = SparkQwen38_27bWorkControlSelectRestorePackets(&config,3u,cumulative);
	assert(selected == 1u);
}

static void SparkTestLaneAtomicBuild(void)
{
	SparkQwen38_27bWorkControlKvPlanConfig config;
	uint32_t blocks_a[2] = {0u,1u},blocks_b[2] = {0u,1u};
	SparkQwen38_27bWorkControlPendingLane pending[2];
	uint32_t packet_lane_counts[1] = {2u},block_count,lanes_built;
	SparkKvStoreBlock blocks[SPARK_KV_STORE_MAX_BATCH_BLOCKS];
	uint8_t block_staging[4u * 96u],gdn_staging[2u * 200u];
	memset(&config,0,sizeof(config));
	memset(pending,0,sizeof(pending));
	config.model_fingerprint = 0x1111u;
	config.cache_layout_fingerprint = 0x2222u;
	config.rank_index = 4u;
	config.block_record_bytes = 96u;
	config.gdn_record_bytes = 200u;
	pending[0].sequence_id = 55u;
	pending[0].nonresident_block_count = 2u;
	pending[0].nonresident_blocks = blocks_a;
	pending[0].gdn_nonresident = 1u;
	pending[1].sequence_id = 56u;
	pending[1].nonresident_block_count = 2u;
	pending[1].nonresident_blocks = blocks_b;
	// Capacity four holds lane zero's three records but not lane one's two:
	// the batch stops on the lane boundary with lane one untouched.
	assert(SparkQwen38_27bWorkControlBuildRestoreBatch(&config,pending,2u,packet_lane_counts,1u,block_staging,4u,gdn_staging,2u,blocks,4u,&block_count,&lanes_built) == SPARK_STATUS_OK);
	assert(block_count == 3u && lanes_built == 1u);
	assert(strcmp(blocks[0].key,"kv/0000000000001111/0000000000002222/r4/s55/b4294967295") == 0);
	assert(strcmp(blocks[1].key,"kv/0000000000001111/0000000000002222/r4/s55/b0") == 0);
	assert(blocks[0].payload == gdn_staging && blocks[0].payload_bytes == 200u);
	assert(blocks[1].payload == block_staging && blocks[2].payload == block_staging + 96u);
	// Full capacity carries both lanes.
	assert(SparkQwen38_27bWorkControlBuildRestoreBatch(&config,pending,2u,packet_lane_counts,1u,block_staging,4u,gdn_staging,2u,blocks,8u,&block_count,&lanes_built) == SPARK_STATUS_OK);
	assert(block_count == 5u && lanes_built == 2u);
	assert(strcmp(blocks[3].key,"kv/0000000000001111/0000000000002222/r4/s56/b0") == 0);
}

static void SparkTestTierRoundtrip(void)
{
	const SparkKvStoreInterface *store_interface = SparkKvStoreGetInterface();
	SparkKvStoreConfiguration store_configuration;
	SparkQwen38_27bWorkControlKvPlanConfig config;
	SparkQwen38_27bWorkControlKvState kv;
	SparkStageKvClient client;
	SparkKvStoreBlock blocks[SPARK_KV_STORE_MAX_BATCH_BLOCKS];
	SparkQwen38_27bWorkControlPendingLane pending[1];
	uint32_t resident_blocks[2] = {0u,1u},packet_lane_counts[1] = {1u};
	uint8_t evict_block_staging[2u * 96u],evict_gdn_staging[200u];
	uint8_t restore_block_staging[2u * 96u],restore_gdn_staging[200u];
	uint32_t block_count,lanes_built,index;
	uint64_t busy_probe_id;
	SparkStatus status;
	assert(SparkKvStoreValidateInterface(store_interface,SPARK_KV_STORE_REQUIRED_CAPS) == SPARK_STATUS_OK);
	memset(&store_configuration,0,sizeof(store_configuration));
	store_configuration.abi_version = SPARK_KV_STORE_ABI_VERSION;
	store_configuration.descriptor_bytes = SPARK_KV_STORE_CONFIGURATION_BYTES;
	store_configuration.layer_count = 6u;
	store_configuration.worker_count = 1u;
	store_configuration.maximum_inflight_batch_count = 1u;
	store_configuration.maximum_batch_block_count = SPARK_KV_STORE_MAX_BATCH_BLOCKS;
	store_configuration.model_fingerprint = 0xAAAAu;
	store_configuration.cache_layout_fingerprint = 0xBBBBu;
	store_configuration.client_memory_pool_bytes = 1u << 20u;
	store_configuration.service_address = "dummy:1";
	store_configuration.ipc_socket_path = "/tmp/spark-test-kv.sock";
	memset(&client,0,sizeof(client));
	client.module_tag = "test_qwen38_27b_wc";
	client.next_batch_id = 1u;
	client.library.store_interface = *store_interface;
	assert(store_interface->initialize(&store_configuration,&client.store_state) == SPARK_STATUS_OK);
	client.enabled = 1u;
	memset(&kv,0,sizeof(kv));
	memset(&config,0,sizeof(config));
	config.model_fingerprint = 0xAAAAu;
	config.cache_layout_fingerprint = 0xBBBBu;
	config.rank_index = 2u;
	config.block_record_bytes = 96u;
	config.gdn_record_bytes = 200u;
	for (index = 0; index < sizeof(evict_block_staging); index++)
		evict_block_staging[index] = (uint8_t)(index * 7u);
	for (index = 0; index < sizeof(evict_gdn_staging); index++)
		evict_gdn_staging[index] = (uint8_t)(index * 3u + 1u);
	// Evict: the dummy injects one capacity failure; the provider retries
	// it internally, so the very first PUT still completes clean.
	assert(SparkQwen38_27bWorkControlBuildEvictBatch(&config,777u,resident_blocks,2u,1u,evict_block_staging,evict_gdn_staging,blocks,8u,&block_count) == SPARK_STATUS_OK);
	assert(block_count == 3u && blocks[0].operation == SPARK_KV_STORE_OPERATION_PUT);
	assert(SparkQwen38_27bWorkControlSubmit(&client,&kv.evict,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,SPARK_QWEN38_27B_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE) == SPARK_STATUS_OK);
	// Single inflight: a second submit on the same direction is refused,
	// and the provider window of one bounces a probe batch with BUSY -
	// nothing consumed, the frame stays queued.
	assert(SparkQwen38_27bWorkControlSubmit(&client,&kv.evict,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,1u) == SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvClientSubmit(&client,SPARK_KV_STORE_OPERATION_PUT,blocks,1u,1u,&busy_probe_id);
	assert(status == SPARK_STATUS_BUSY || status == SPARK_STATUS_OK);
	assert(SparkTestWaitReady(&client,&kv,&kv.evict) == SPARK_STATUS_OK);
	assert(SparkQwen38_27bWorkControlAcknowledge(&kv.evict) == SPARK_STATUS_OK);
	// Restore the same lane through the pending-queue path at packet-zero
	// priority and verify the byte-identical roundtrip of every record.
	memset(pending,0,sizeof(pending));
	pending[0].sequence_id = 777u;
	pending[0].nonresident_block_count = 2u;
	pending[0].nonresident_blocks = resident_blocks;
	pending[0].gdn_nonresident = 1u;
	assert(SparkQwen38_27bWorkControlBuildRestoreBatch(&config,pending,1u,packet_lane_counts,1u,restore_block_staging,2u,restore_gdn_staging,1u,blocks,8u,&block_count,&lanes_built) == SPARK_STATUS_OK);
	assert(block_count == 3u && lanes_built == 1u);
	assert(SparkQwen38_27bWorkControlSubmit(&client,&kv.restore,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_QWEN38_27B_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE) == SPARK_STATUS_OK);
	assert(SparkTestWaitReady(&client,&kv,&kv.restore) == SPARK_STATUS_OK);
	assert(SparkQwen38_27bWorkControlAcknowledge(&kv.restore) == SPARK_STATUS_OK);
	assert(memcmp(restore_gdn_staging,evict_gdn_staging,sizeof(evict_gdn_staging)) == 0);
	assert(memcmp(restore_block_staging,evict_block_staging,sizeof(evict_block_staging)) == 0);
	// A lane that was never stored surfaces as a failed completion, not a
	// fabricated payload.
	pending[0].sequence_id = 888u;
	assert(SparkQwen38_27bWorkControlBuildRestoreBatch(&config,pending,1u,packet_lane_counts,1u,restore_block_staging,2u,restore_gdn_staging,1u,blocks,8u,&block_count,&lanes_built) == SPARK_STATUS_OK);
	assert(SparkQwen38_27bWorkControlSubmit(&client,&kv.restore,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,0u) == SPARK_STATUS_OK);
	assert(SparkTestWaitReady(&client,&kv,&kv.restore) != SPARK_STATUS_OK);
	assert(SparkQwen38_27bWorkControlAcknowledge(&kv.restore) == SPARK_STATUS_OK);
	store_interface->destroy(client.store_state);
}

int main()
{
	SparkTestPlanMath();
	SparkTestLaneAtomicBuild();
	SparkTestTierRoundtrip();
	std::printf("test_qwen38_27b_work_control PASS\n");
	return(0);
}
