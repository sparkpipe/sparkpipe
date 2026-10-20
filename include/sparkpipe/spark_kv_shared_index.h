#ifndef SPARKPIPE_SPARK_KV_SHARED_INDEX_H
#define SPARKPIPE_SPARK_KV_SHARED_INDEX_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

#define SPARK_KV_SHARED_INDEX_MAGIC UINT64_C(0x3158444e4950534b)
#define SPARK_KV_SHARED_INDEX_VERSION 1u
#define SPARK_KV_SHARED_INDEX_HOLDERS_MAX 63u
#define SPARK_KV_SHARED_INDEX_SLOTS_MAX 65536u
#define SPARK_KV_SHARED_INDEX_RECLAIM_LOCK (UINT64_C(1) << 63)
#define SPARK_KV_SHARED_INDEX_IDENTITY_WORDS 4u
#define SPARK_KV_SHARED_INDEX_LAYOUT_BYTES 32u
#define SPARK_KV_SHARED_NO_SLOT UINT32_MAX
#define SPARK_KV_SHARED_NO_HOLDER UINT32_MAX
#define SPARK_KV_SHARED_SLOT_FREE 0u
#define SPARK_KV_SHARED_SLOT_WRITING 1u
#define SPARK_KV_SHARED_SLOT_READY 2u

typedef struct SparkKvSharedIndexHeader
{
	uint64_t magic;
	uint32_t version;
	uint32_t slot_count;
	uint64_t page_bytes;
	uint8_t layout_sha256[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES];
	_Atomic uint64_t clock;
	uint64_t reserved[3];
} SparkKvSharedIndexHeader;

typedef struct SparkKvSharedIndexSlot
{
	_Atomic uint32_t state;
	_Atomic uint32_t writer;
	_Atomic uint64_t holders;
	_Atomic uint64_t generation;
	_Atomic uint64_t last_use;
	_Atomic uint32_t token_count;
	_Atomic uint32_t parent_slot;
	_Atomic uint64_t parent_generation;
	_Atomic uint64_t identity[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS];
} SparkKvSharedIndexSlot;

typedef struct SparkKvSharedIndex
{
	SparkKvSharedIndexHeader *header;
	SparkKvSharedIndexSlot *slots;
	uint32_t slot_count;
	uint32_t holder;
} SparkKvSharedIndex;

typedef struct SparkKvSharedIndexView
{
	uint64_t generation;
	uint64_t parent_generation;
	uint32_t token_count;
	uint32_t parent_slot;
	uint64_t identity[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS];
} SparkKvSharedIndexView;

static inline uint64_t SparkKvSharedIndexBytes(uint32_t slot_count)
{
	return((uint64_t)sizeof(SparkKvSharedIndexHeader) + (uint64_t)slot_count * sizeof(SparkKvSharedIndexSlot));
}

static inline SparkStatus SparkKvSharedIndexFormat(void *memory,uint64_t bytes,uint32_t slot_count,uint64_t page_bytes,const uint8_t layout_sha256[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES])
{
	SparkKvSharedIndexHeader *header = (SparkKvSharedIndexHeader *)memory;
	SparkKvSharedIndexSlot *slots;
	uint32_t slot;
	if ( memory == 0 || layout_sha256 == 0 || slot_count == 0u || page_bytes == 0u || bytes < SparkKvSharedIndexBytes(slot_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(memory,0,(size_t)SparkKvSharedIndexBytes(slot_count));
	header->version = SPARK_KV_SHARED_INDEX_VERSION;
	header->slot_count = slot_count;
	header->page_bytes = page_bytes;
	memcpy(header->layout_sha256,layout_sha256,SPARK_KV_SHARED_INDEX_LAYOUT_BYTES);
	atomic_init(&header->clock,1u);
	slots = (SparkKvSharedIndexSlot *)(header + 1);
	for (slot=0u; slot<slot_count; slot++)
	{
		atomic_init(&slots[slot].state,SPARK_KV_SHARED_SLOT_FREE);
		atomic_init(&slots[slot].writer,SPARK_KV_SHARED_NO_HOLDER);
		atomic_init(&slots[slot].parent_slot,SPARK_KV_SHARED_NO_SLOT);
	}
	atomic_thread_fence(memory_order_seq_cst);
	header->magic = SPARK_KV_SHARED_INDEX_MAGIC;
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkKvSharedIndexAttach(SparkKvSharedIndex *index,void *memory,uint64_t bytes,uint32_t holder,uint64_t page_bytes,const uint8_t layout_sha256[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES])
{
	SparkKvSharedIndexHeader *header = (SparkKvSharedIndexHeader *)memory;
	if ( index == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(index,0,sizeof(*index));
	if ( memory == 0 || layout_sha256 == 0 || bytes < sizeof(*header) || holder >= SPARK_KV_SHARED_INDEX_HOLDERS_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_KV_SHARED_INDEX_MAGIC || header->version != SPARK_KV_SHARED_INDEX_VERSION || header->slot_count == 0u ||
		bytes < SparkKvSharedIndexBytes(header->slot_count) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->page_bytes != page_bytes || memcmp(header->layout_sha256,layout_sha256,SPARK_KV_SHARED_INDEX_LAYOUT_BYTES) != 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	index->header = header;
	index->slots = (SparkKvSharedIndexSlot *)(header + 1);
	index->slot_count = header->slot_count;
	index->holder = holder;
	return(SPARK_STATUS_OK);
}

static inline uint64_t SparkKvSharedIndexBit(uint32_t holder)
{
	return(UINT64_C(1) << holder);
}

static inline uint32_t SparkKvSharedIndexClaim(SparkKvSharedIndex *index,uint32_t slot,uint64_t *generation_out)
{
	SparkKvSharedIndexSlot *entry = &index->slots[slot];
	*generation_out = atomic_fetch_add(&entry->generation,1u) + 1u;
	atomic_store(&entry->writer,index->holder);
	atomic_store(&entry->holders,SparkKvSharedIndexBit(index->holder));
	atomic_store(&entry->parent_slot,SPARK_KV_SHARED_NO_SLOT);
	return(slot);
}

static inline SparkStatus SparkKvSharedIndexReserve(SparkKvSharedIndex *index,uint32_t *slot_out,uint64_t *generation_out)
{
	uint32_t slot,expected,victim;
	uint64_t oldest;
	if ( index == 0 || index->slots == 0 || slot_out == 0 || generation_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*slot_out = SPARK_KV_SHARED_NO_SLOT;
	*generation_out = 0u;
	for (slot=0u; slot<index->slot_count; slot++)
	{
		expected = SPARK_KV_SHARED_SLOT_FREE;
		if ( atomic_load(&index->slots[slot].state) == SPARK_KV_SHARED_SLOT_FREE &&
			atomic_compare_exchange_strong(&index->slots[slot].state,&expected,SPARK_KV_SHARED_SLOT_WRITING) )
		{
			*slot_out = SparkKvSharedIndexClaim(index,slot,generation_out);
			return(SPARK_STATUS_OK);
		}
	}
	for (;;)
	{
		victim = SPARK_KV_SHARED_NO_SLOT;
		oldest = UINT64_MAX;
		for (slot=0u; slot<index->slot_count; slot++)
			if ( atomic_load(&index->slots[slot].state) == SPARK_KV_SHARED_SLOT_READY && atomic_load(&index->slots[slot].holders) == 0u &&
				atomic_load(&index->slots[slot].last_use) < oldest )
			{
				oldest = atomic_load(&index->slots[slot].last_use);
				victim = slot;
			}
		if ( victim == SPARK_KV_SHARED_NO_SLOT )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		{
			uint64_t unheld = 0u;
			if ( !atomic_compare_exchange_strong(&index->slots[victim].holders,&unheld,SPARK_KV_SHARED_INDEX_RECLAIM_LOCK) )
				continue;
		}
		expected = SPARK_KV_SHARED_SLOT_READY;
		if ( !atomic_compare_exchange_strong(&index->slots[victim].state,&expected,SPARK_KV_SHARED_SLOT_WRITING) )
		{
			atomic_store(&index->slots[victim].holders,0u);
			continue;
		}
		*slot_out = SparkKvSharedIndexClaim(index,victim,generation_out);
		return(SPARK_STATUS_OK);
	}
}

static inline uint32_t SparkKvSharedIndexWriting(const SparkKvSharedIndex *index,uint32_t slot)
{
	return(index != 0 && index->slots != 0 && slot < index->slot_count &&
		atomic_load(&index->slots[slot].state) == SPARK_KV_SHARED_SLOT_WRITING &&
		atomic_load(&index->slots[slot].writer) == index->holder ? 1u : 0u);
}

static inline SparkStatus SparkKvSharedIndexPublish(SparkKvSharedIndex *index,uint32_t slot,const uint8_t identity[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS * sizeof(uint64_t)],uint32_t token_count,uint32_t parent_slot,uint64_t parent_generation)
{
	SparkKvSharedIndexSlot *entry;
	uint64_t words[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS];
	uint32_t word;
	if ( SparkKvSharedIndexWriting(index,slot) == 0u || identity == 0 || token_count == 0u ||
		(parent_slot != SPARK_KV_SHARED_NO_SLOT && parent_slot >= index->slot_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	entry = &index->slots[slot];
	memcpy(words,identity,sizeof(words));
	for (word=0u; word<SPARK_KV_SHARED_INDEX_IDENTITY_WORDS; word++)
		atomic_store_explicit(&entry->identity[word],words[word],memory_order_relaxed);
	atomic_store_explicit(&entry->token_count,token_count,memory_order_relaxed);
	atomic_store_explicit(&entry->parent_slot,parent_slot,memory_order_relaxed);
	atomic_store_explicit(&entry->parent_generation,parent_generation,memory_order_relaxed);
	atomic_store(&entry->last_use,atomic_fetch_add(&index->header->clock,1u));
	atomic_store(&entry->writer,SPARK_KV_SHARED_NO_HOLDER);
	atomic_store(&entry->state,SPARK_KV_SHARED_SLOT_READY);
	return(SPARK_STATUS_OK);
}

static inline void SparkKvSharedIndexAbandon(SparkKvSharedIndex *index,uint32_t slot)
{
	if ( SparkKvSharedIndexWriting(index,slot) == 0u )
		return;
	atomic_store(&index->slots[slot].holders,0u);
	atomic_store(&index->slots[slot].writer,SPARK_KV_SHARED_NO_HOLDER);
	atomic_store(&index->slots[slot].state,SPARK_KV_SHARED_SLOT_FREE);
}

static inline uint32_t SparkKvSharedIndexRead(const SparkKvSharedIndex *index,uint32_t slot,SparkKvSharedIndexView *view)
{
	const SparkKvSharedIndexSlot *entry;
	uint64_t before;
	uint32_t word;
	if ( index == 0 || index->slots == 0 || slot >= index->slot_count || view == 0 )
		return(0u);
	entry = &index->slots[slot];
	before = atomic_load(&entry->generation);
	if ( atomic_load(&entry->state) != SPARK_KV_SHARED_SLOT_READY )
		return(0u);
	for (word=0u; word<SPARK_KV_SHARED_INDEX_IDENTITY_WORDS; word++)
		view->identity[word] = atomic_load_explicit(&entry->identity[word],memory_order_relaxed);
	view->token_count = atomic_load_explicit(&entry->token_count,memory_order_relaxed);
	view->parent_slot = atomic_load_explicit(&entry->parent_slot,memory_order_relaxed);
	view->parent_generation = atomic_load_explicit(&entry->parent_generation,memory_order_relaxed);
	atomic_thread_fence(memory_order_acquire);
	if ( atomic_load(&entry->state) != SPARK_KV_SHARED_SLOT_READY || atomic_load(&entry->generation) != before )
		return(0u);
	view->generation = before;
	return(1u);
}

static inline uint32_t SparkKvSharedIndexFind(const SparkKvSharedIndex *index,const uint8_t identity[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS * sizeof(uint64_t)],uint32_t token_count,SparkKvSharedIndexView *view)
{
	uint64_t words[SPARK_KV_SHARED_INDEX_IDENTITY_WORDS];
	uint32_t slot;
	if ( index == 0 || index->slots == 0 || identity == 0 || view == 0 )
		return(SPARK_KV_SHARED_NO_SLOT);
	memcpy(words,identity,sizeof(words));
	for (slot=0u; slot<index->slot_count; slot++)
		if ( atomic_load_explicit(&index->slots[slot].identity[0],memory_order_relaxed) == words[0] && SparkKvSharedIndexRead(index,slot,view) != 0u &&
			view->token_count == token_count && memcmp(view->identity,words,sizeof(words)) == 0 )
			return(slot);
	return(SPARK_KV_SHARED_NO_SLOT);
}

static inline SparkStatus SparkKvSharedIndexAcquire(SparkKvSharedIndex *index,uint32_t slot,uint64_t generation)
{
	SparkKvSharedIndexSlot *entry;
	uint64_t bit,held;
	if ( index == 0 || index->slots == 0 || slot >= index->slot_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	entry = &index->slots[slot];
	bit = SparkKvSharedIndexBit(index->holder);
	held = atomic_fetch_or(&entry->holders,bit);
	if ( (held & bit) != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( (held & SPARK_KV_SHARED_INDEX_RECLAIM_LOCK) != 0u || atomic_load(&entry->state) != SPARK_KV_SHARED_SLOT_READY || atomic_load(&entry->generation) != generation )
	{
		atomic_fetch_and(&entry->holders,~bit);
		return(SPARK_STATUS_NOT_FOUND);
	}
	atomic_store(&entry->last_use,atomic_fetch_add(&index->header->clock,1u));
	return(SPARK_STATUS_OK);
}

static inline void SparkKvSharedIndexRelease(SparkKvSharedIndex *index,uint32_t slot)
{
	if ( index == 0 || index->slots == 0 || slot >= index->slot_count )
		return;
	atomic_store(&index->slots[slot].last_use,atomic_fetch_add(&index->header->clock,1u));
	atomic_fetch_and(&index->slots[slot].holders,~SparkKvSharedIndexBit(index->holder));
}

static inline uint32_t SparkKvSharedIndexClearHolder(void *memory,uint64_t bytes,uint32_t holder)
{
	SparkKvSharedIndexHeader *header = (SparkKvSharedIndexHeader *)memory;
	SparkKvSharedIndexSlot *slots;
	uint32_t slot,freed = 0u;
	if ( memory == 0 || bytes < sizeof(*header) || holder >= SPARK_KV_SHARED_INDEX_HOLDERS_MAX || header->magic != SPARK_KV_SHARED_INDEX_MAGIC ||
		bytes < SparkKvSharedIndexBytes(header->slot_count) )
		return(0u);
	slots = (SparkKvSharedIndexSlot *)(header + 1);
	for (slot=0u; slot<header->slot_count; slot++)
	{
		atomic_fetch_and(&slots[slot].holders,~SparkKvSharedIndexBit(holder));
		if ( atomic_load(&slots[slot].state) == SPARK_KV_SHARED_SLOT_WRITING && atomic_load(&slots[slot].writer) == holder )
		{
			atomic_store(&slots[slot].holders,0u);
			atomic_store(&slots[slot].writer,SPARK_KV_SHARED_NO_HOLDER);
			atomic_store(&slots[slot].state,SPARK_KV_SHARED_SLOT_FREE);
			freed++;
		}
	}
	return(freed);
}

#endif
