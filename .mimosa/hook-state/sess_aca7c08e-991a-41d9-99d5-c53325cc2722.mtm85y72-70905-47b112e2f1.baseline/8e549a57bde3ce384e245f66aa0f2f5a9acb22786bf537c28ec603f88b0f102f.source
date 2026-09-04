#include "sparkpipe/spark_nvme_tier.h"
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
	uint32_t polls_per_read;
	uint32_t cancel_polls_per_read;
	uint32_t submits;
	uint32_t cancels;
	uint32_t data_fills;
	const uint8_t *drive;
	uint64_t drive_base;
	uint64_t drive_bytes;
}
MockDevice;

static void MockDeviceReset(MockDevice *device, uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

static void MockDeviceFill(MockDevice *device, MockRead *read)
{
	uint64_t index;
	if ( device->drive != 0 && read->offset >= device->drive_base &&
		read->offset + MOCK_BLOCK_BYTES <=
			device->drive_base + device->drive_bytes )
	{
		memcpy(read->destination,
			device->drive + (read->offset - device->drive_base),
			MOCK_BLOCK_BYTES);
	}
	else
		memset(read->destination,0xAAu,MOCK_BLOCK_BYTES);
	for ( index = 0u; index < MOCK_MAX_READS; ++index )
		(void)index;
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
	_Alignas(MOCK_BLOCK_BYTES) uint8_t drive[TIER_TEST_MAX_SLOTS * MOCK_BLOCK_BYTES];
}
TierFixture;

static void TierContentFill(uint64_t content_seed, uint8_t *block)
{
	uint32_t index;
	for ( index = 0u; index < MOCK_BLOCK_BYTES; ++index )
		block[index] = (uint8_t)( content_seed ^ (uint8_t)(index * 7u +
			(uint8_t)(content_seed >> 8) + 1u) );
}

static void TierContentDigest(uint64_t content_seed, uint8_t *digest)
{
	uint8_t block[MOCK_BLOCK_BYTES];
	SparkSha256Context context;
	TierContentFill(content_seed,block);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,block,MOCK_BLOCK_BYTES);
	SparkSha256Finalize(&context,digest);
}

static SparkStatus TierFixturePublishSeed(
	TierFixture *fixture,
	uint64_t content_hash,
	uint64_t content_seed,
	uint64_t *device_offset_out)
{
	uint8_t content[MOCK_BLOCK_BYTES];
	uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
	SparkNvmeTierWriteReservation reservation;
	SparkStatus status;
	uint64_t drive_index;

	TierContentFill(content_seed,content);
	TierContentDigest(content_seed,digest);
	status = SparkNvmeTierReserveWrite(&fixture->tier,content_hash,digest,
		&reservation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( reservation.already_present == 0u )
	{
		if ( reservation.device_offset < fixture->configuration.base_offset ||
			reservation.device_offset + MOCK_BLOCK_BYTES >
				fixture->configuration.base_offset + sizeof(fixture->drive) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		drive_index = reservation.device_offset -
			fixture->configuration.base_offset;
		memcpy(fixture->drive + drive_index,content,MOCK_BLOCK_BYTES);
	}
	*device_offset_out = reservation.device_offset;
	return(SparkNvmeTierCommitWrite(&fixture->tier,&reservation));
}

static SparkStatus TierFixturePublish(
	TierFixture *fixture,
	uint64_t content_hash,
	uint64_t *device_offset_out)
{
	return(TierFixturePublishSeed(fixture,content_hash,content_hash,
		device_offset_out));
}

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
	fixture->device.drive = fixture->drive;
	fixture->device.drive_base = 1u << 20;
	fixture->device.drive_bytes = sizeof(fixture->drive);
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
			expect(TierFixturePublish(&fixture,1000u + index,&offset[index])
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
		expect(TierFixturePublish(&fixture,9999u,&extra) == SPARK_STATUS_OK,
			"the ninth publish still succeeds: eviction is cheap, not fatal");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.slots_in_use == 8u && statistics.evictions == 1u,
			"capacity never moves: one in, one out");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,9999u,0,&extra) == SPARK_STATUS_OK
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
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset;

		expect(TierFixtureOpen(&fixture,4u,2u,1u,1u) == SPARK_STATUS_OK,
			"reservation fixture initialises");
		TierContentDigest(3500u,digest);
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3500u,digest,&first)
			== SPARK_STATUS_OK && first.already_present == 0u,
			"a new hash reserves one private device slot");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,3500u,digest,&offset)
			== SPARK_STATUS_NOT_FOUND,
			"an uncommitted write is invisible to OffsetOf");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,3500u,digest,0u,&demand)
			== SPARK_STATUS_OK && demand.state == SPARK_NVME_TIER_DEMAND_MISS,
			"an uncommitted write is invisible to readers");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&first) == SPARK_STATUS_OK,
			"commit publishes the completed write");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,3500u,digest,&offset)
			== SPARK_STATUS_OK && offset == first.device_offset,
			"the committed hash resolves to its reserved offset");
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3500u,digest,&second)
			== SPARK_STATUS_OK && second.already_present != 0u
			&& second.device_offset == first.device_offset,
			"reserving an existing hash with the same digest returns the record");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&second)
			== SPARK_STATUS_OK,
			"commit of an already-present reservation is idempotent");

		TierContentDigest(3501u,digest);
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3501u,digest,&first)
			== SPARK_STATUS_OK && first.already_present == 0u,
			"another hash reserves a fresh slot");
		expect(SparkNvmeTierAbortWrite(&fixture.tier,&first) == SPARK_STATUS_OK,
			"abort releases an unpublished slot");
		expect(SparkNvmeTierCommitWrite(&fixture.tier,&first)
			== SPARK_STATUS_VALIDATION_FAILED,
			"a stale aborted reservation cannot publish later");
		TierContentDigest(3502u,digest);
		expect(SparkNvmeTierReserveWrite(&fixture.tier,3502u,digest,&second)
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
		{
			SparkNvmeTierWriteReservation no_digest;
			expect(SparkNvmeTierReserveWrite(&fixture.tier,3503u,0,&no_digest)
				== SPARK_STATUS_INVALID_ARGUMENT,
				"a write reservation without a digest is refused");
		}
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
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint8_t expected[MOCK_BLOCK_BYTES];
		uint64_t offset,hash = 4242u;
		int32_t intact;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		TierContentDigest(hash,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"an absent block is a miss, not a wait");
		TierFixturePublish(&fixture,hash,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"a present block starts a demand read immediately");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,0u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_IN_FLIGHT,
			"a second requester joins the same read instead of issuing another");
		TierFixtureDrain(&fixture,1u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,3u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_READY
			&& result.staging_pointer != 0,
			"once landed, the hit is immediate");
		TierContentFill(hash,expected);
		intact = memcmp(result.staging_pointer,expected,MOCK_BLOCK_BYTES) == 0;
		expect(intact, "the bytes are the record the hash owns, verified against its digest");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.digest_verifications >= 1u && statistics.digest_mismatches == 0u,
			"the landing was digest-verified before it was served");
		expect(statistics.demand_misses == 1u && statistics.demand_loads == 1u
			&& statistics.demand_joins == 1u && statistics.demand_hits == 1u,
			"and the accounting says exactly what happened");
		expect(SparkNvmeTierConsume(&fixture.tier,hash,digest) == SPARK_STATUS_OK,
			"consumption releases the staging buffer");
		expect(SparkNvmeTierOffsetOf(&fixture.tier,hash,digest,&offset) == SPARK_STATUS_OK,
			"while the on-drive record stays: consumption is a copy, not a move");
	}

	printf("\nB3: corrupted bytes are quarantined, never served\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset,hash = 4243u;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		TierContentDigest(hash,digest);
		expect(TierFixturePublish(&fixture,hash,&offset) == SPARK_STATUS_OK,
			"the record publishes");
		fixture.drive[offset - fixture.configuration.base_offset] ^= 0x40u;
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,0u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"the demand read is issued (the tier cannot pre-read the drive)");
		{
			uint32_t step;
			SparkStatus pump_status = SPARK_STATUS_OK;
			for ( step = 1u; step < 8u; ++step )
			{
				pump_status = SparkNvmeTierPump(&fixture.tier,step);
				if ( pump_status == SPARK_STATUS_HASH_MISMATCH )
					break;
			}
			expect(pump_status == SPARK_STATUS_HASH_MISMATCH,
				"the landing fails its digest and the tier answers HASH_MISMATCH");
		}
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.digest_mismatches == 1u,
			"the mismatch is counted");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,2u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"the quarantined record is gone: the caller recomputes, never consumes bad bytes");
		expect(TierFixturePublish(&fixture,hash,&offset) == SPARK_STATUS_OK,
			"and the slot is recyclable by an honest write-back");
	}

	printf("\nB3: a 64-bit hash collision fails loud, never aliases\n");
	{
		TierFixture fixture;
		SparkNvmeTierWriteReservation reservation;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint8_t tenant_a[SPARK_NVME_TIER_DIGEST_BYTES];
		uint8_t tenant_b[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset,shared_hash = 5150u;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		TierContentDigest(shared_hash,tenant_a);
		TierContentDigest(shared_hash + 1u,tenant_b);
		expect(TierFixturePublish(&fixture,shared_hash,&offset) == SPARK_STATUS_OK,
			"tenant A commits the shared hash with its digest");
		expect(SparkNvmeTierReserveWrite(&fixture.tier,shared_hash,tenant_b,
				&reservation) == SPARK_STATUS_HASH_MISMATCH,
			"tenant B's write-back under the same hash is refused, not aliased");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,shared_hash,tenant_b,0u,
				&result) == SPARK_STATUS_HASH_MISMATCH,
			"tenant B's demand under the same hash is refused, not served A's bytes");
		expect(SparkNvmeTierConsume(&fixture.tier,shared_hash,tenant_b)
			== SPARK_STATUS_HASH_MISMATCH,
			"consumption refuses a digest that is not the record's");
		expect(SparkNvmeTierPin(&fixture.tier,shared_hash,tenant_b,1)
			== SPARK_STATUS_HASH_MISMATCH,
			"even pinning refuses a contradicting digest");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,shared_hash,tenant_a,0u,
				&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"tenant A is unaffected: its own digest still resolves");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.digest_mismatches >= 3u,
			"every refused access is counted");
	}

	printf("\nB3: a wrong digest is an argument error even off-drive\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset,hash = 5151u;
		TierFixtureOpen(&fixture,8u,2u,1u,2u);
		TierFixturePublish(&fixture,hash,&offset);
		memset(digest,0,sizeof(digest));
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,digest,0u,&result)
			== SPARK_STATUS_INVALID_ARGUMENT,
			"an all-zero digest is not an identity: refused");
		expect(SparkNvmeTierRequestDemand(&fixture.tier,hash,0,0u,&result)
			== SPARK_STATUS_INVALID_ARGUMENT,
			"the decode path cannot request bytes without an identity");
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
			TierFixturePublish(&fixture,5000u + index,&offset);
			needs[index].content_hash = 5000u + index;
			needs[index].need_by_step = 4u + index;
			needs[index].reserved0 = 0u;
			TierContentDigest(5000u + index,needs[index].content_digest);
		}
		expect(SparkNvmeTierPlanLookahead(&fixture.tier,needs,4u,0u,&report)
			== SPARK_STATUS_OK && report.queued_count == 4u
			&& report.absent_count == 0u && report.late_risk_count == 0u,
			"four needs inside the window all queue, none late");
		for ( step = 0u; step <= 8u; ++step )
		{
			SparkNvmeTierPump(&fixture.tier,step);
			for ( index = 0u; index < 4u; ++index )
			{
				if ( needs[index].need_by_step != step )
					continue;
				expect(SparkNvmeTierRequestDemand(
						&fixture.tier,needs[index].content_hash,
						needs[index].content_digest,step,&result)
					== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_READY,
					"at its need-by step the block is already there");
				SparkNvmeTierConsume(&fixture.tier,needs[index].content_hash,
					needs[index].content_digest);
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
		expect(statistics.digest_verifications >= 4u
			&& statistics.digest_mismatches == 0u,
			"every landing was digest-verified on the way in");
		needs[0].content_hash = 5000u;
		needs[0].need_by_step = 1u;
		TierContentDigest(5000u,needs[0].content_digest);
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
		TierFixtureOpen(&fixture,16u,2u,0u,8u);
		for ( index = 0u; index < 3u; ++index )
			TierFixturePublish(&fixture,7000u + index,&offset);
		for ( index = 0u; index < 2u; ++index )
		{
			needs[index].content_hash = 7000u + index;
			needs[index].need_by_step = 20u + index;
			needs[index].reserved0 = 0u;
			TierContentDigest(7000u + index,needs[index].content_digest);
		}
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u,
			"with no reserve, both buffers fill with prefetch reads");
		{
			uint8_t demand_digest[SPARK_NVME_TIER_DIGEST_BYTES];
			TierContentDigest(7002u,demand_digest);
			expect(SparkNvmeTierRequestDemand(&fixture.tier,7002u,demand_digest,1u,&result)
				== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_STARTED,
				"demand does not wait behind prefetch: it preempts");
		}
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.prefetch_preemptions == 1u && statistics.demand_stalls == 0u,
			"one preemption, zero stalls");
		expect(fixture.device.cancels == 1u,
			"the in-flight prefetch was cancelled, furthest deadline first");
		expect(statistics.demand_loads == 1u,
			"and the demand read went to the drive at once");
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
		uint8_t demand_digest[SPARK_NVME_TIER_DIGEST_BYTES];
		TierFixtureOpen(&fixture,16u,2u,1u,8u);
		for ( index = 0u; index < 3u; ++index )
			TierFixturePublish(&fixture,7100u + index,&offset);
		for ( index = 0u; index < 2u; ++index )
		{
			needs[index].content_hash = 7100u + index;
			needs[index].need_by_step = 20u + index;
			needs[index].reserved0 = 0u;
			TierContentDigest(7100u + index,needs[index].content_digest);
		}
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 1u,
			"the demand reserve holds one buffer back from prefetch");
		TierContentDigest(7102u,demand_digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,7102u,demand_digest,1u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"so demand finds a buffer without preempting anything");
	}

	printf("\nthe stall that must stay loud: every buffer demand-held\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset;
		TierFixtureOpen(&fixture,16u,2u,0u,1u);
		TierFixturePublish(&fixture,8000u,&offset);
		TierFixturePublish(&fixture,8001u,&offset);
		TierFixturePublish(&fixture,8002u,&offset);
		TierContentDigest(8000u,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,8000u,digest,0u,&result)
			== SPARK_STATUS_OK, "first demand load");
		TierContentDigest(8001u,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,8001u,digest,0u,&result)
			== SPARK_STATUS_OK, "second demand load");
		TierFixtureDrain(&fixture,1u);
		TierContentDigest(8002u,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,8002u,digest,2u,&result)
			== SPARK_STATUS_BUSY,
			"unconsumed demand data is a sizing bug: BUSY, never a silent wait");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.demand_stalls == 1u, "and it is counted as one");
		TierContentDigest(8000u,digest);
		expect(SparkNvmeTierConsume(&fixture.tier,8000u,digest) == SPARK_STATUS_OK,
			"consume the first holder");
		TierContentDigest(8002u,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,8002u,digest,3u,&result) == SPARK_STATUS_OK
			&& result.state == SPARK_NVME_TIER_DEMAND_STARTED,
			"consuming upstream unblocks it");
	}

	printf("\neviction under churn never fails a pinned block\n");
	{
		TierFixture fixture;
		SparkNvmeTierDemandResult result;
		SparkNvmeTierStatistics statistics;
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset;
		uint32_t index;
		TierFixtureOpen(&fixture,8u,2u,1u,1u);
		TierFixturePublish(&fixture,9000u,&offset);
		TierContentDigest(9000u,digest);
		expect(SparkNvmeTierPin(&fixture.tier,9000u,digest,1) == SPARK_STATUS_OK,
			"pin the block an admitted sequence needs");
		for ( index = 0u; index < 60u; ++index )
			TierFixturePublish(&fixture,10000u + index,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,9000u,digest,0u,&result)
			== SPARK_STATUS_OK && result.state != SPARK_NVME_TIER_DEMAND_MISS,
			"60 publishes through 8 records, and the pinned one is still there");
		SparkNvmeTierGetStatistics(&fixture.tier,&statistics);
		expect(statistics.pinned_eviction_skips != 0u,
			"the clock walked past it rather than through it");
		TierFixtureDrain(&fixture,1u);
		expect(SparkNvmeTierConsume(&fixture.tier,9000u,digest) == SPARK_STATUS_OK,
			"drain and consume it first");
		expect(SparkNvmeTierPin(&fixture.tier,9000u,digest,0) == SPARK_STATUS_OK, "unpin");
		for ( index = 0u; index < 60u; ++index )
			TierFixturePublish(&fixture,20000u + index,&offset);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,9000u,digest,1u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"unpinned, the same churn reclaims it - eviction is cheap in both directions");
	}

	printf("\na fully pinned tier refuses loudly\n");
	{
		TierFixture fixture;
		uint64_t offset;
		TierFixtureOpen(&fixture,2u,2u,1u,1u);
		TierFixturePublish(&fixture,30000u,&offset);
		TierFixturePublish(&fixture,30001u,&offset);
		{
			uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
			TierContentDigest(30000u,digest);
			SparkNvmeTierPin(&fixture.tier,30000u,digest,1);
			TierContentDigest(30001u,digest);
			SparkNvmeTierPin(&fixture.tier,30001u,digest,1);
		}
		expect(TierFixturePublish(&fixture,30002u,&offset) == SPARK_STATUS_BUSY,
			"nothing evictable is BUSY, not a silently evicted promise");
	}

	printf("\neviction is safe against DMA already in flight\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed need;
		SparkNvmeTierStatistics statistics;
		uint64_t offset;
		TierFixtureOpen(&fixture,2u,2u,0u,8u);
		TierFixturePublish(&fixture,40000u,&offset);
		TierFixturePublish(&fixture,40001u,&offset);
		need.content_hash = 40000u;
		need.need_by_step = 30u;
		need.reserved0 = 0u;
		TierContentDigest(40000u,need.content_digest);
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		need.content_hash = 40001u;
		TierContentDigest(40001u,need.content_digest);
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u, "both records mid-flight");
		expect(TierFixturePublish(&fixture,40002u,&offset) == SPARK_STATUS_OK,
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
		uint8_t demand_digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset;
		uint32_t submits_before;

		TierFixtureOpen(&fixture,4u,2u,0u,32u);
		fixture.device.cancel_polls_per_read = 2u;
		TierFixturePublish(&fixture,45000u,&offset);
		TierFixturePublish(&fixture,45001u,&offset);
		TierFixturePublish(&fixture,45002u,&offset);
		needs[0].content_hash = 45000u;
		needs[0].need_by_step = 40u;
		needs[0].reserved0 = 0u;
		TierContentDigest(45000u,needs[0].content_digest);
		needs[1].content_hash = 45001u;
		needs[1].need_by_step = 41u;
		needs[1].reserved0 = 0u;
		TierContentDigest(45001u,needs[1].content_digest);
		SparkNvmeTierPlanLookahead(&fixture.tier,needs,2u,0u,0);
		SparkNvmeTierPump(&fixture.tier,0u);
		expect(fixture.device.submits == 2u,
			"both staging buffers are owned by in-flight prefetch reads");
		submits_before = fixture.device.submits;
		TierContentDigest(45002u,demand_digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,demand_digest,1u,&result)
			== SPARK_STATUS_BUSY,
			"demand does not reuse a buffer whose cancellation is pending");
		expect(fixture.device.submits == submits_before
			&& fixture.device.cancels == 2u,
			"no replacement DMA starts while either old DMA still owns memory");
		SparkNvmeTierPump(&fixture.tier,2u);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,demand_digest,2u,&result)
			== SPARK_STATUS_BUSY,
			"pending cancellation remains unavailable after a nonterminal poll");
		SparkNvmeTierPump(&fixture.tier,3u);
		SparkNvmeTierPump(&fixture.tier,4u);
		expect(fixture.device.submits > submits_before,
			"terminal cancellation proof releases the buffer for queued work");
		fixture.device.cancel_polls_per_read = 0u;
		expect(SparkNvmeTierRequestDemand(&fixture.tier,45002u,demand_digest,5u,&result)
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
		uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
		uint64_t offset;
		uint32_t index;
		TierFixtureOpen(&fixture,8u,2u,1u,1u);
		TierFixturePublish(&fixture,50000u,&offset);
		need.content_hash = 50000u;
		need.need_by_step = 40u;
		need.reserved0 = 0u;
		TierContentDigest(50000u,need.content_digest);
		SparkNvmeTierPlanLookahead(&fixture.tier,&need,1u,0u,0);
		for ( index = 0u; index < 40u; ++index )
			TierFixturePublish(&fixture,60000u + index,&offset);
		SparkNvmeTierPump(&fixture.tier,1u);
		expect(fixture.device.submits == 0u,
			"the queue entry died with its record's generation");
		TierContentDigest(50000u,digest);
		expect(SparkNvmeTierRequestDemand(&fixture.tier,50000u,digest,2u,&result)
			== SPARK_STATUS_OK && result.state == SPARK_NVME_TIER_DEMAND_MISS,
			"and the block reports honestly as gone");
	}

	printf("\nadmission asks: will these blocks be resident by step N?\n");
	{
		TierFixture fixture;
		SparkNvmeTierNeed warm[2],mixed[3];
		SparkNvmeTierResidencyAssessment assessment;
		uint64_t offset;
		TierFixtureOpen(&fixture,16u,4u,1u,1u);
		TierFixturePublish(&fixture,70000u,&offset);
		TierFixturePublish(&fixture,70001u,&offset);
		{
			SparkNvmeTierDemandResult result;
			uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
			TierContentDigest(70000u,digest);
			SparkNvmeTierRequestDemand(&fixture.tier,70000u,digest,0u,&result);
			TierFixtureDrain(&fixture,1u);
		}
		memset(warm,0,sizeof(warm));
		memset(mixed,0,sizeof(mixed));
		warm[0].content_hash = 70000u;
		TierContentDigest(70000u,warm[0].content_digest);
		warm[1].content_hash = 70001u;
		TierContentDigest(70001u,warm[1].content_digest);
		mixed[0] = warm[0];
		mixed[1] = warm[1];
		mixed[2].content_hash = 70002u;
		TierContentDigest(70002u,mixed[2].content_digest);
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
