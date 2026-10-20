#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_kv_shared_index.h"

#define TEST_SLOTS 8u
#define TEST_PAGE_BYTES 4096u
#define TEST_STRESS_ROUNDS 200000u

static uint8_t g_layout[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES] = {7u};

static void TestIdentity(uint8_t identity[32],uint32_t seed)
{
	uint32_t index;
	for (index=0u; index<32u; index++)
		identity[index] = (uint8_t)(seed * 31u + index);
}

static void *TestFormatted(uint32_t slots)
{
	uint64_t bytes = SparkKvSharedIndexBytes(slots);
	void *memory = malloc((size_t)bytes);
	assert(memory != 0);
	assert(SparkKvSharedIndexFormat(memory,bytes,slots,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	return(memory);
}

static void TestAttachValidates(void)
{
	uint8_t other[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES] = {8u};
	uint64_t bytes = SparkKvSharedIndexBytes(TEST_SLOTS);
	void *memory = TestFormatted(TEST_SLOTS);
	SparkKvSharedIndex index;
	assert(SparkKvSharedIndexAttach(&index,memory,bytes,0u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK && index.slot_count == TEST_SLOTS);
	assert(SparkKvSharedIndexAttach(&index,memory,bytes,0u,TEST_PAGE_BYTES,other) == SPARK_STATUS_VALIDATION_FAILED);
	assert(SparkKvSharedIndexAttach(&index,memory,bytes,0u,TEST_PAGE_BYTES * 2u,g_layout) == SPARK_STATUS_VALIDATION_FAILED);
	assert(SparkKvSharedIndexAttach(&index,memory,bytes,SPARK_KV_SHARED_INDEX_HOLDERS_MAX,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkKvSharedIndexAttach(&index,memory,bytes - 1u,0u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_SCHEMA_ERROR);
	((SparkKvSharedIndexHeader *)memory)->magic = 0u;
	assert(SparkKvSharedIndexAttach(&index,memory,bytes,0u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_SCHEMA_ERROR);
	free(memory);
}

static void TestPublishFindAcquireReclaim(void)
{
	uint64_t bytes = SparkKvSharedIndexBytes(2u),generation,stale;
	void *memory = TestFormatted(2u);
	SparkKvSharedIndex writer,reader;
	SparkKvSharedIndexView view;
	uint8_t identity[32],second[32],third[32];
	uint32_t slot,root,child;
	assert(SparkKvSharedIndexAttach(&writer,memory,bytes,3u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexAttach(&reader,memory,bytes,9u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	TestIdentity(identity,1u);
	TestIdentity(second,2u);
	TestIdentity(third,3u);
	assert(SparkKvSharedIndexReserve(&writer,&root,&generation) == SPARK_STATUS_OK && SparkKvSharedIndexWriting(&writer,root) == 1u);
	assert(SparkKvSharedIndexWriting(&reader,root) == 0u);
	assert(SparkKvSharedIndexFind(&reader,identity,64u,&view) == SPARK_KV_SHARED_NO_SLOT);
	assert(SparkKvSharedIndexPublish(&reader,root,identity,64u,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkKvSharedIndexPublish(&writer,root,identity,64u,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexFind(&reader,identity,63u,&view) == SPARK_KV_SHARED_NO_SLOT);
	slot = SparkKvSharedIndexFind(&reader,identity,64u,&view);
	assert(slot == root && view.generation == generation && view.parent_slot == SPARK_KV_SHARED_NO_SLOT);
	assert(SparkKvSharedIndexAcquire(&reader,root,view.generation) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexAcquire(&reader,root,view.generation) == SPARK_STATUS_VALIDATION_FAILED);
	assert(SparkKvSharedIndexReserve(&writer,&child,&generation) == SPARK_STATUS_OK && child != root);
	assert(SparkKvSharedIndexPublish(&writer,child,second,128u,root,view.generation) == SPARK_STATUS_OK);
	slot = SparkKvSharedIndexFind(&reader,second,128u,&view);
	assert(slot == child && view.parent_slot == root);
	SparkKvSharedIndexRelease(&writer,root);
	SparkKvSharedIndexRelease(&writer,child);
	assert(SparkKvSharedIndexReserve(&writer,&slot,&generation) == SPARK_STATUS_OK && slot == child);
	assert(SparkKvSharedIndexFind(&reader,second,128u,&view) == SPARK_KV_SHARED_NO_SLOT);
	stale = generation - 1u;
	assert(SparkKvSharedIndexAcquire(&reader,child,stale) == SPARK_STATUS_NOT_FOUND);
	assert(SparkKvSharedIndexReserve(&writer,&slot,&generation) == SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkKvSharedIndexAbandon(&writer,child);
	assert(atomic_load(&writer.slots[child].state) == SPARK_KV_SHARED_SLOT_FREE);
	assert(SparkKvSharedIndexReserve(&writer,&slot,&generation) == SPARK_STATUS_OK && slot == child);
	assert(SparkKvSharedIndexPublish(&writer,child,third,64u,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_OK);
	SparkKvSharedIndexRelease(&reader,root);
	SparkKvSharedIndexRelease(&writer,child);
	assert(SparkKvSharedIndexReserve(&writer,&slot,&generation) == SPARK_STATUS_OK && slot == root);
	free(memory);
	printf("PASS kv shared index publishes, finds, acquires and reclaims only unheld slots, oldest first\n");
}

static void TestClearHolder(void)
{
	uint64_t bytes = SparkKvSharedIndexBytes(3u),generation;
	void *memory = TestFormatted(3u);
	SparkKvSharedIndex dead,live;
	SparkKvSharedIndexView view;
	uint8_t identity[32];
	uint32_t ready,writing;
	assert(SparkKvSharedIndexAttach(&dead,memory,bytes,5u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexAttach(&live,memory,bytes,6u,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	TestIdentity(identity,4u);
	assert(SparkKvSharedIndexReserve(&dead,&ready,&generation) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexPublish(&dead,ready,identity,64u,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexFind(&live,identity,64u,&view) == ready && SparkKvSharedIndexAcquire(&live,ready,view.generation) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexReserve(&dead,&writing,&generation) == SPARK_STATUS_OK);
	assert(SparkKvSharedIndexClearHolder(memory,bytes,5u) == 1u);
	assert(atomic_load(&live.slots[writing].state) == SPARK_KV_SHARED_SLOT_FREE);
	assert(atomic_load(&live.slots[ready].holders) == SparkKvSharedIndexBit(6u));
	assert(atomic_load(&live.slots[ready].state) == SPARK_KV_SHARED_SLOT_READY);
	free(memory);
	printf("PASS kv shared index clears a gone holder: its bits drop and its writing slots free\n");
}

typedef struct TestStress
{
	void *memory;
	uint64_t bytes;
	uint32_t holder;
	uint32_t acquired;
	uint32_t published;
} TestStress;

static void *TestStressMain(void *context)
{
	TestStress *stress = (TestStress *)context;
	SparkKvSharedIndex index;
	SparkKvSharedIndexView view,again;
	uint8_t identity[32];
	uint64_t generation;
	uint32_t round,slot,key;
	assert(SparkKvSharedIndexAttach(&index,stress->memory,stress->bytes,stress->holder,TEST_PAGE_BYTES,g_layout) == SPARK_STATUS_OK);
	for (round=0u; round<TEST_STRESS_ROUNDS; round++)
	{
		key = (round * 2654435761u + stress->holder) % 24u;
		TestIdentity(identity,key);
		slot = SparkKvSharedIndexFind(&index,identity,64u + key,&view);
		if ( slot != SPARK_KV_SHARED_NO_SLOT && SparkKvSharedIndexAcquire(&index,slot,view.generation) == SPARK_STATUS_OK )
		{
			assert(SparkKvSharedIndexRead(&index,slot,&again) == 1u && again.generation == view.generation && again.token_count == 64u + key &&
				memcmp(again.identity,identity,sizeof(identity)) == 0);
			stress->acquired++;
			SparkKvSharedIndexRelease(&index,slot);
			continue;
		}
		if ( SparkKvSharedIndexReserve(&index,&slot,&generation) != SPARK_STATUS_OK )
			continue;
		assert(SparkKvSharedIndexPublish(&index,slot,identity,64u + key,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_OK);
		stress->published++;
		SparkKvSharedIndexRelease(&index,slot);
	}
	return(0);
}

static void TestConcurrentHoldersNeverLoseAPage(void)
{
	uint64_t bytes = SparkKvSharedIndexBytes(TEST_SLOTS);
	void *memory = TestFormatted(TEST_SLOTS);
	TestStress stress[4];
	pthread_t threads[4];
	uint32_t index;
	for (index=0u; index<4u; index++)
	{
		memset(&stress[index],0,sizeof(stress[index]));
		stress[index].memory = memory;
		stress[index].bytes = bytes;
		stress[index].holder = index * 7u;
		assert(pthread_create(&threads[index],0,TestStressMain,&stress[index]) == 0);
	}
	for (index=0u; index<4u; index++)
	{
		assert(pthread_join(threads[index],0) == 0);
		assert(stress[index].acquired != 0u && stress[index].published != 0u);
	}
	for (index=0u; index<TEST_SLOTS; index++)
		assert(atomic_load(&((SparkKvSharedIndexSlot *)((SparkKvSharedIndexHeader *)memory + 1))[index].holders) == 0u);
	free(memory);
	printf("PASS kv shared index under four concurrent holders: every acquired slot keeps its identity until release\n");
}

int main(void)
{
	TestAttachValidates();
	TestPublishFindAcquireReclaim();
	TestClearHolder();
	TestConcurrentHoldersNeverLoseAPage();
	return(0);
}
