// The NVMe tier: budget enforcement, JIT lookahead, eviction under churn,
// and the priority contract between demand loads and prefetch.
//
// Everything the tier decides is arithmetic over a schedule, so all of it is
// checkable on a host: the device is a mock that completes a read after a
// programmed number of polls and remembers every offset it was asked for,
// which is how the assertions can say "arrived before its need-by step" and
// "read from the record the hash actually owns" rather than hoping.
#include "sparkpipe/spark_nvme_tier.h"

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

typedef struct MockRead
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint32_t cancel_polls_left;
	uint8_t active;
	uint8_t cancel_requested;
}
MockRead;

typedef struct MockDevice
{
	MockRead reads[MOCK_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;         /* device latency, in poll calls */
	uint32_t cancel_polls_per_read;  /* zero means cancellation is synchronous */
	uint32_t submits;
	uint32_t cancels;
	uint32_t data_fills;
}
MockDevice;

static void MockDeviceReset(MockDevice *device, uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

// What the "drive" holds at an offset is a function of the offset, so a
// landing proves two things at once: the tier asked for the right record,
// and the bytes reached the right staging buffer.
static void MockDeviceFill(MockDevice *device, MockRead *read)
{
	uint32_t index;
	uint8_t pattern = (uint8_t)( read->offset / MOCK_BLOCK_BYTES );
	for ( index = 0u; index < MOCK_BLOCK_BYTES; ++index )
		read->destination[index] = (uint8_t)( pattern ^ (uint8_t)index );
	device->data_fills++;
}

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
		if ( read->cancel_requested != 0u )
		{
			if ( read->cancel_polls_left != 0u )
			{
				read->cancel_polls_left--;
				return(SPARK_STATUS_PENDING);
			}
			read->active = 0u;
			return(SPARK_STATUS_NOT_FOUND);
		}
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
			MockDeviceFill(device,read);
		}
		read->active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

static SparkStatus MockCancelRead(void *context, uint64_t ticket)
{
	MockDevice *device = (MockDevice *)context;
	uint32_t index;
	for ( index = 0u; index < MOCK_MAX_READS; ++index )
	{
		if ( !device->reads[index].active || device->reads[index].ticket != ticket )
			continue;
		device->cancels++;
		if ( device->cancel_polls_per_read != 0u )
		{
			device->reads[index].cancel_requested = 1u;
			device->reads[index].cancel_polls_left =
				device->cancel_polls_per_read;
			return(SPARK_STATUS_PENDING);
		}
		device->reads[index].active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

#define TIER_TEST_MAX_STAGING 4u
#define TIER_TEST_MAX_SLOTS 64u

typedef struct TierFixture
{
	MockDevice device;
	SparkNvmeTierDevice vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration configuration;
	_Alignas(64) uint8_t tables[128u * 1024u];
	_Alignas(MOCK_BLOCK_BYTES) uint8_t staging[TIER_TEST_MAX_STAGING * MOCK_BLOCK_BYTES];
}
TierFixture;

static SparkStatus TierFixtureOpen(
	TierFixture *fixture,
	uint32_t budget_blocks,
	uint32_t staging_buffers,
	uint32_t demand_reserve,
	uint32_t polls_per_read)
{
	SparkStatus status;
	uint64_t table_bytes;
	memset(fixture,0,sizeof(*fixture));
	MockDeviceReset(&fixture->device,polls_per_read);
	fixture->vtable.context = &fixture->device;
	fixture->vtable.submit_read = MockSubmitRead;
	fixture->vtable.poll_read = MockPollRead;
	fixture->vtable.cancel_read = MockCancelRead;
	fixture->configuration.abi_version = SPARK_NVME_TIER_ABI_VERSION;
	fixture->configuration.descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	fixture->configuration.budget_bytes = (uint64_t)budget_blocks * MOCK_BLOCK_BYTES;
	fixture->configuration.base_offset = 1u << 20;
	fixture->configuration.block_bytes = MOCK_BLOCK_BYTES;
	fixture->configuration.hash_bucket_count = 32u;
	fixture->configuration.staging_buffer_count = staging_buffers;
	fixture->configuration.demand_reserve_buffers = demand_reserve;
	fixture->configuration.pending_capacity = 32u;
	/* One block per two steps: the ETA the planner computes is 2, matching
	   the mock's two-poll latency, so "before its need-by step" is a
	   statement about the same clock on both sides. */
	fixture->configuration.device_bytes_per_second = 2048u;
	fixture->configuration.step_time_microseconds = 1000000u;
	table_bytes = SparkNvmeTierTableBytes(&fixture->configuration);
	if ( table_bytes == 0u || table_bytes > sizeof(fixture->tables) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkNvmeTierInitialize(&fixture->tier,&fixture->configuration,
		&fixture->vtable,fixture->tables,sizeof(fixture->tables),
		fixture->staging,sizeof(fixture->staging));
	return(status);
}

static SparkStatus TierFixturePublish(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	uint64_t *device_offset_out)
{
	SparkNvmeTierWriteReservation reservation;
	SparkStatus status;

	status = SparkNvmeTierReserveWrite(tier,content_hash,&reservation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*device_offset_out = reservation.device_offset;
	return(SparkNvmeTierCommitWrite(tier,&reservation));
}

// Pump until no mock read remains active, bounded so a wedged device fails
// the test rather than hanging it.
static void TierFixtureDrain(TierFixture *fixture, uint32_t first_step)
{
	uint32_t step;
	for ( step = first_step; step < first_step + 64u; ++step )
	{
		uint32_t index,active = 0u;
		SparkNvmeTierPump(&fixture->tier,step);
		for ( index = 0u; index < MOCK_MAX_READS; ++index )
			if ( fixture->device.reads[index].active )
				active = 1u;
		if ( !active )
			return;
	}
}

int main(void)
{
	printf("nvme tier\n\npartitioning: the budget is a real ceiling\n");
	{
		TierFixture fixture;
		SparkNvmeTierStatistics statistics;
		uint64_t offset[8],extra;
		uint32_t index;
		expect(TierFixtureOpen(&fixture,8u,2u,1u,2u) == SPARK_STATUS_OK,
			"a 8-block, double-buffered tier initialises");
		expect(fixture.tier.slot_count == 8u, "the budget buys exactly 8 records");
		expect(fixture.tier.transfer_steps == 2u,
			"the transfer ETA is precomputed: one block is two steps");
		for ( index = 0u; index < 8u; ++index )
			expect(TierFixturePublish(&fixture.tier,1000u + index,&offset[index])
				== SPARK_STATUS_OK, "publish within budget");
		{
			int32_t distinct = 1;
			uint32_t other;
			for ( index = 0u; index < 8u && distinct; ++index )
				for ( other = index + 1u; other < 8u; ++other )
					if ( offset[index] == offset[other] )
						distinct = 0;
			expect(distinct, "every record owns a distinct device offset");
		}
		expect(TierFixturePublish(&fixture.tier,9999u,&extra) == SPARK_STATUS_OK,
			"the ninth publish still succeeds: eviction is cheap, not fatal");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.slots_in_use == 8u && statistics.evictions == 1u,
			"capacity never moves: one in, one out");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,9999u,&extra) == SPARK_STATUS_OK
			&& extra >= fixture.configuration.base_offset
			&& extra < fixture.configuration.base_offset + 8u * MOCK_BLOCK_BYTES,
			"and the recycled record stays inside the budget region");
	}

	printf("\nconfiguration rejects what the drive would reject at runtime\n");
	{
		TierFixture fixture;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		fixture.configuration.block_bytes = 1000u;
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
                    &fixture.vtable,fixture.tables,sizeof(fixture.tables),
                    fixture.staging,sizeof(fixture.staging))
			== SPARK_STATUS_INVALID_ARGUMENT,
			"a non-4096-multiple block cannot do O_DIRECT");
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
                    &fixture.vtable,fixture.tables,sizeof(fixture.tables),
                    fixture.staging + 16u,sizeof(fixture.staging) - 16u)
			== SPARK_STATUS_INVALID_ARGUMENT,
			"misaligned staging is an init error, not a DMA error");
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		fixture.configuration.staging_buffer_count = 1u;
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
                    &fixture.vtable,fixture.tables,sizeof(fixture.tables),
                    fixture.staging,sizeof(fixture.staging))
			== SPARK_STATUS_INVALID_ARGUMENT,
			"single buffering is not double buffering");
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		fixture.configuration.device_bytes_per_second = UINT64_MAX;
		fixture.configuration.step_time_microseconds = UINT32_MAX;
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
                    &fixture.vtable,fixture.tables,sizeof(fixture.tables),
                    fixture.staging,sizeof(fixture.staging))
			== SPARK_STATUS_OK
			&& fixture.tier.bytes_per_step == UINT32_MAX
			&& fixture.tier.transfer_steps == 1u,
			"bandwidth-times-step arithmetic saturates instead of wrapping");
	}

	printf("\nwrite reservation visibility and ownership\n");
	{
		TierFixture fixture;
		SparkNvmeTierWriteReservation first;
		SparkNvmeTierWriteReservation second;
		SparkNvmeTierDemandResult demand;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;

		expect(TierFixtureOpen(&fixture,4u,2u,1u,1u) == SPARK_STATUS_OK,
			"reservation fixture initialises");
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3500u,&first)
			== SPARK_STATUS_OK && first.already_present == 0u,
			"a new hash reserves one private device slot");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,3500u,&offset)
			== SPARK_STATUS_NOT_FOUND,
			"an uncommitted write is invisible to OffsetOf");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,3500u,0u,&demand)
			== SPARK_STATUS_OK && demand.state == SPARK_NVME_TIER_DEMAND_MISS,
			"an uncommitted write is invisible to readers");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&first) == SPARK_STATUS_OK,
			"commit publishes the completed write");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,3500u,&offset)
			== SPARK_STATUS_OK && offset == first.device_offset,
			"the committed hash resolves to its reserved offset");
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3500u,&second)
			== SPARK_STATUS_OK && second.already_present != 0u
			&& second.device_offset == first.device_offset,
			"reserving an existing hash returns the committed record");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&second)
			== SPARK_STATUS_OK,
			"commit of an already-present reservation is idempotent");

		expect(SparkNvmeTierReserveWrite(&fixture.tier,3501u,&first)
			== SPARK_STATUS_OK && first.already_present == 0u,
			"another hash reserves a fresh slot");
		expect(SparkNvmeTierAbortWrite(&fixture.tier,&first) == SPARK_STATUS_OK,
			"abort releases an unpublished slot");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&first)
			== SPARK_STATUS_VALIDATION_FAILED,
			"a stale aborted reservation cannot publish later");
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3502u,&second)
			== SPARK_STATUS_OK && second.already_present == 0u,
			"the aborted capacity can be reused with a new generation");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&first)
			== SPARK_STATUS_VALIDATION_FAILED,
			"the old generation cannot publish the reused slot");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&second) == SPARK_STATUS_OK,
			"the current generation can publish the reused slot");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.write_reservations == 3u
			&& statistics.publishes == 2u
			&& statistics.write_aborts == 1u,
			"write lifecycle accounting distinguishes reserve, commit, and abort");
	}

	printf("\ninitialisation rejects undersized ownership regions\n");
	{
		TierFixture fixture;
		uint64_t required_tables;
		uint64_t required_staging;

		expect(TierFixtureOpen(&fixture,8u,2u,1u,2u) == SPARK_STATUS_OK,
			"ownership-region fixture initialises");
		required_tables = SparkNvmeTierTableBytes(&fixture.configuration);
		required_staging = (uint64_t)fixture.configuration.staging_buffer_count
			* fixture.configuration.block_bytes;
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
			&fixture.vtable,fixture.tables,required_tables - 1u,
			fixture.staging,required_staging) == SPARK_STATUS_CAPACITY_EXCEEDED,
			"undersized table ownership is rejected");
		expect(SparkNvmeTierInitialize(&fixture.tier,&fixture.configuration,
			&fixture.vtable,fixture.tables,required_tables,
			fixture.staging,required_staging - 1u) == SPARK_STATUS_CAPACITY_EXCEEDED,
			"undersized staging ownership is rejected");
	}

	printf("\nhit, miss, and the read that joins\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset,hash = 4242u;
		uint32_t index;
		int32_t intact;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"an absent block is a miss, not a wait");
		TierFixturePublish(&fixture.tier,hash,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"a present block starts a demand read immediately");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_IN_FLIGHT,
			"a second requester joins the same read instead of issuing another");
		TierFixtureDrain(&fixture,1u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,3u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_READY
			&& result.staging_pointer != 0,
			"once landed, the hit is immediate");
		intact = 1;
		for ( index = 0u; index < MOCK_BLOCK_BYTES; ++index )
			if ( ((const uint8_t *)result.staging_pointer)[index]
				!= (uint8_t)( (uint8_t)( offset / MOCK_BLOCK_BYTES ) ^ (uint8_t)index ) )
				intact = 0;
		expect(intact, "the bytes are the record the hash owns, read at its offset");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.demand_misses == 1u && statistics.demand_loads == 1u
			&& statistics.demand_joins == 1u && statistics.demand_hits == 1u,
			"and the accounting says exactly what happened");
		expect(SparkNvmeTierConsume(&fixture.tier,hash) == SPARK_STATUS_OK,
			"consumption releases the staging buffer");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,hash,&offset) == SPARK_STATUS_OK,
			"while the on-drive record stays: consumption is a copy, not a move");
	}

	printf("\nthe lookahead: blocks land before their need-by step\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed needs[4];
		SparkNvmeTierPlanReport report;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		uint32_t index,step;
		TierFixtureOpen(&fixture,16u,4u,1u,2u);
		for ( index = 0u; index < 4u; ++index )
		{
			TierFixturePublish(&fixture.tier,5000u + index,&offset);
			needs[index].content_hash = 5000u + index;
			needs[index].need_by_step = 4u + index;
			needs[index].reserved0 = 0u;
		}
		expect(SparkNvmeTierPlanLookahead(&fixture.tier,needs,4u,0u,&report)
			== SPARK_STATUS_OK && report.queued_count == 4u
			&& report.absent_count == 0u && report.late_risk_count == 0u,
			"four needs inside the window all queue, none late");
		/* the simulated schedule: pump, then the layer asks for what it needs,
		   then the consumed buffer goes back into circulation */
		for ( step = 0u; step <= 8u; ++step )
		{
			SparkNvmeTierPump(&fixture.tier,step);
			for ( index = 0u; index < 4u; ++index )
			{
				if ( needs[index].need_by_step != step )
					continue;
				expect(SparkNvmeTierRequestDemand(
						&fixture.tier,needs[index].content_hash,step,&result)
					== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_READY,
					"at its need-by step the block is already there");
				SparkNvmeTierConsume(&fixture.tier,needs[index].content_hash);
			}
		}
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.demand_misses == 0u && statistics.demand_loads == 0u
			&& statistics.prefetch_late_landings == 0u,
			"the decode path never touched the drive: no misses, no demand reads, nothing late");
		expect(statistics.prefetch_issues == 4u && statistics.prefetch_landings == 4u,
			"four prefetches issued, four landed");
		expect(statistics.read_bytes == 4u * MOCK_BLOCK_BYTES,
			"and the byte counter the acceptance run wants says 4 blocks");
		/* a need closer than the transfer time is flagged, not hidden */
		needs[0].content_hash = 5000u;
		needs[0].need_by_step = 1u;
		expect(SparkNvmeTierPlanLookahead(&fixture.tier,needs,1u,0u,&report)
			== SPARK_STATUS_OK && report.late_risk_count == 1u,
			"a deadline inside the transfer time is reported as late risk");
	}

	printf("\npriority: demand preempts prefetch, prefetch never starves demand\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed needs[2];
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		uint32_t index;
		/* reserve 0: prefetch may fill every buffer, so demand must take one */
		TierFixtureOpen(&fixture,16u,2u,0u,8u);
		for ( index = 0u; index < 3u; ++index )
			TierFixturePublish(&fixture.tier,7000u + index,&offset);
		for ( index = 0u; index < 2u; ++index )
		{
			needs[index].content_hash = 7000u + index;
			needs[index].need_by_step = 20u + index;
			needs[index].reserved0 = 0u;
		}
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u,
			"with no reserve, both buffers fill with prefetch reads");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,7002u,1u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"demand does not wait behind prefetch: it preempts");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.prefetch_preemptions == 1u && statistics.demand_stalls == 0u,
			"one preemption, zero stalls");
		expect(fixture.device.cancels == 1u,
			"the in-flight prefetch was cancelled, furthest deadline first");
		expect(statistics.demand_loads == 1u,
			"and the demand read went to the drive at once");
		/* the displaced prefetch is re-queued, not forgotten */
		TierFixtureDrain(&fixture,2u);
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.prefetch_landings >= 1u,
			"the displaced prefetch is re-fetched after preemption, not lost");
	}
	{
		TierFixture fixture;
		SparkNvmeTierNeed needs[2];
		SparkNvmeTierDemandResult result;
		uint64_t offset;
		uint32_t index;
		/* reserve 1: the last buffer is not prefetch's to take */
		TierFixtureOpen(&fixture,16u,2u,1u,8u);
		for ( index = 0u; index < 3u; ++index )
			TierFixturePublish(&fixture.tier,7100u + index,&offset);
		for ( index = 0u; index < 2u; ++index )
		{
			needs[index].content_hash = 7100u + index;
			needs[index].need_by_step = 20u + index;
			needs[index].reserved0 = 0u;
		}
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 1u,
			"the demand reserve holds one buffer back from prefetch");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,7102u,1u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"so demand finds a buffer without preempting anything");
	}

	printf("\nthe stall that must stay loud: every buffer demand-held\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		TierFixtureOpen(&fixture,16u,2u,0u,1u);
		TierFixturePublish(&fixture.tier,8000u,&offset);
		TierFixturePublish(&fixture.tier,8001u,&offset);
		TierFixturePublish(&fixture.tier,8002u,&offset);
		SparkNvmeTierRequestDemand(&fixture.tier,8000u,0u,&result);
		SparkNvmeTierRequestDemand(&fixture.tier,8001u,0u,&result);
		TierFixtureDrain(&fixture,1u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,8002u,2u,&result)
			== SPARK_STATUS_BUSY,
			"unconsumed demand data is a sizing bug: BUSY, never a silent wait");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.demand_stalls == 1u, "and it is counted as one");
		expect(SparkNvmeTierConsume(&fixture.tier,8000u) == SPARK_STATUS_OK
			&& SparkNvmeTierRequestDemand(&fixture.tier,8002u,3u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"consuming upstream unblocks it");
	}

	printf("\neviction under churn never fails a pinned block\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		uint32_t index;
		TierFixtureOpen(&fixture,8u,2u,1u,1u);
		TierFixturePublish(&fixture.tier,9000u,&offset);
		expect(SparkNvmeTierPin(&fixture.tier,9000u,1) == SPARK_STATUS_OK,
			"pin the block an admitted sequence needs");
		for ( index = 0u; index < 60u; ++index )
			TierFixturePublish(&fixture.tier,10000u + index,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,9000u,0u,&result)
			== SPARK_STATUS_OK && result.state != SPARK_NVME_TIER_DEMAND_MISS,
			"60 publishes through 8 records, and the pinned one is still there");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.pinned_eviction_skips != 0u,
			"the clock walked past it rather than through it");
		/* settle the block back to a plain on-drive record: a slot that stays
		   demand-held is unevictable by design, which is not what this churn
		   is measuring */
		TierFixtureDrain(&fixture,1u);
		expect(SparkNvmeTierConsume(&fixture.tier,9000u) == SPARK_STATUS_OK,
			"drain and consume it first");
		expect(SparkNvmeTierPin(&fixture.tier,9000u,0) == SPARK_STATUS_OK, "unpin");
		for ( index = 0u; index < 60u; ++index )
			TierFixturePublish(&fixture.tier,20000u + index,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,9000u,1u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"unpinned, the same churn reclaims it - eviction is cheap in both directions");
	}

	printf("\na fully pinned tier refuses loudly\n");
	{
		TierFixture fixture;
		uint64_t offset;
		TierFixtureOpen(&fixture,2u,2u,1u,1u);
		TierFixturePublish(&fixture.tier,30000u,&offset);
		TierFixturePublish(&fixture.tier,30001u,&offset);
		SparkNvmeTierPin(&fixture.tier,30000u,1);
		SparkNvmeTierPin(&fixture.tier,30001u,1);
		expect(TierFixturePublish(&fixture.tier,30002u,&offset) == SPARK_STATUS_BUSY,
			"nothing evictable is BUSY, not a silently evicted promise");
	}

	printf("\neviction is safe against DMA already in flight\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed need;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		TierFixtureOpen(&fixture,2u,2u,0u,8u);
		TierFixturePublish(&fixture.tier,40000u,&offset);
		TierFixturePublish(&fixture.tier,40001u,&offset);
		need.content_hash = 40000u;
		need.need_by_step = 30u;
		need.reserved0 = 0u;
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		need.content_hash = 40001u;
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u, "both records mid-flight");
		expect(TierFixturePublish(&fixture.tier,40002u,&offset) == SPARK_STATUS_OK,
			"evicting a filling record cancels its read and moves on");
		expect(fixture.device.cancels == 1u,
			"the cancelled read will never touch the buffer again");
		SparkNvmeTierPump(&fixture.tier,1u);
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.stale_completions == 0u && statistics.slots_in_use == 2u,
			"no late completion is believed, and the budget holds");
	}

	printf("\npending cancellation retains device ownership until terminal poll\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed needs[2];
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		uint32_t submits_before;

		TierFixtureOpen(&fixture,4u,2u,0u,32u);
		fixture.device.cancel_polls_per_read = 2u;
		TierFixturePublish(&fixture.tier,45000u,&offset);
		TierFixturePublish(&fixture.tier,45001u,&offset);
		TierFixturePublish(&fixture.tier,45002u,&offset);
		needs[0].content_hash = 45000u;
		needs[0].need_by_step = 40u;
		needs[0].reserved0 = 0u;
		needs[1].content_hash = 45001u;
		needs[1].need_by_step = 41u;
		needs[1].reserved0 = 0u;
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u,
			"both staging buffers are owned by in-flight prefetch reads");
		submits_before = fixture.device.submits;
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,1u,&result)
			== SPARK_STATUS_BUSY,
			"demand does not reuse a buffer whose cancellation is pending");
		expect(fixture.device.submits == submits_before
			&& fixture.device.cancels == 2u,
			"no replacement DMA starts while either old DMA still owns memory");
		SparkNvmeTierPump(&fixture.tier,2u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,2u,&result)
			== SPARK_STATUS_BUSY,
			"pending cancellation remains unavailable after a nonterminal poll");
		SparkNvmeTierPump(&fixture.tier,3u);
		SparkNvmeTierPump(&fixture.tier,4u);
		expect(fixture.device.submits > submits_before,
			"terminal cancellation proof releases the buffer for queued work");
		fixture.device.cancel_polls_per_read = 0u;
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,5u,&result)
			== SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"demand can reclaim the released buffer without a late DMA owner");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.cancel_pending_count == 2u
			&& statistics.demand_stalls >= 2u,
			"pending ownership and its backpressure remain visible in statistics");
	}

	printf("\na queued prefetch whose record was evicted is dropped, not issued\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed need;
		SparkNvmeTierDemandResult result;
		uint64_t offset;
		uint32_t index;
		TierFixtureOpen(&fixture,8u,2u,1u,1u);
		TierFixturePublish(&fixture.tier,50000u,&offset);
		need.content_hash = 50000u;
		need.need_by_step = 40u;
		need.reserved0 = 0u;
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		for ( index = 0u; index < 40u; ++index )
			TierFixturePublish(&fixture.tier,60000u + index,&offset);
		SparkNvmeTierPump(&fixture.tier,1u);
		expect(fixture.device.submits == 0u,
			"the queue entry died with its record's generation");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,50000u,2u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"and the block reports honestly as gone");
	}

	printf("\nadmission asks: will these blocks be resident by step N?\n");
	{
		TierFixture fixture;
		SparkNvmeTierResidencyAssessment assessment;
		uint64_t offset;
		uint64_t warm[2] = { 70000u, 70001u };
		uint64_t mixed[3] = { 70000u, 70001u, 70002u };
		TierFixtureOpen(&fixture,16u,4u,1u,1u);
		TierFixturePublish(&fixture.tier,70000u,&offset);
		TierFixturePublish(&fixture.tier,70001u,&offset);
		{
			SparkNvmeTierDemandResult result;
			SparkNvmeTierRequestDemand(&fixture.tier,70000u,0u,&result);
			TierFixtureDrain(&fixture,1u);
		}
		expect(SparkNvmeTierWillBeResidentBy(&fixture.tier,warm,2u,3u,20u,&assessment)
			== SPARK_STATUS_OK
			&& assessment.confidence == SPARK_NVME_TIER_CONFIDENCE_ALL
			&& assessment.ready_count == 1u
			&& assessment.planned_confident_count == 1u,
			"one upstairs, one fetchable in time: ALL, and admission prefers it");
		expect(SparkNvmeTierWillBeResidentBy(&fixture.tier,mixed,3u,3u,20u,&assessment)
			== SPARK_STATUS_OK
			&& assessment.confidence == SPARK_NVME_TIER_CONFIDENCE_PARTIAL
			&& assessment.absent_count == 1u,
			"one block absent: PARTIAL, and the flag says why");
		expect(SparkNvmeTierWillBeResidentBy(&fixture.tier,&warm[1],1u,3u,3u,&assessment)
			== SPARK_STATUS_OK
			&& assessment.confidence == SPARK_NVME_TIER_CONFIDENCE_NONE
			&& assessment.late_count == 1u,
			"a deadline no transfer can meet: NONE, counted as late");
	}

	printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
	return(failures ? 1 : 0);
}
