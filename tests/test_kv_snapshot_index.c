#include "../cache/kv_snapshot.c"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INDEX_SCALE_ROOTS 1000000u
#define INDEX_SCALE_SECONDS 60.0
#define INDEX_MODEL_KEYS 63u
#define INDEX_MODEL_STEPS 20000u
#define INDEX_FILE_BYTES 114688u

static uint32_t index_checks,index_failures;

#define CHECK(condition) do { index_checks++; if ( !(condition) ) { index_failures++; fprintf(stderr,"FAIL line=%d %s\n",__LINE__,#condition); } } while (0)

static uint64_t IndexMix(uint64_t value)
{
	value += UINT64_C(0x9e3779b97f4a7c15);
	value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
	value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
	return(value ^ (value >> 31u));
}

static void IndexKey(SparkKvSnapshotKey *key,uint8_t layout,uint64_t identity,uint32_t token_count,uint32_t file_kind)
{
	uint64_t mixed = IndexMix(identity);
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,layout,sizeof(key->layout_sha256));
	memcpy(key->identity_sha256,&mixed,sizeof(mixed));
	memcpy(key->identity_sha256 + sizeof(mixed),&identity,sizeof(identity));
	key->token_count = token_count;
	key->file_kind = file_kind;
}

static uint32_t IndexOpen(SparkKvSnapshotStore *store,char *directory)
{
	strcpy(directory,"/tmp/sparkpipe-kv-snapshot-index-XXXXXX");
	if ( mkdtemp(directory) == 0 )
		return(0u);
	return(SparkKvSnapshotStoreOpen(store,directory,UINT64_C(1) << 46u,UINT64_C(1) << 20u) == SPARK_STATUS_OK ? 1u : 0u);
}

static void IndexClose(SparkKvSnapshotStore *store,const char *directory)
{
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	SparkKvSnapshotStoreClose(store);
	(void)snprintf(path,sizeof(path),"%s/%s",directory,SPARK_KV_SNAPSHOT_LOCK_NAME);
	(void)unlink(path);
	(void)rmdir(directory);
}

static SparkStatus IndexInsert(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	SparkStatus status;
	pthread_mutex_lock(&runtime->mutex);
	status = SparkKvSnapshotIndexInsert(store,key,parent,INDEX_FILE_BYTES,++runtime->next_sequence,1u,0);
	pthread_mutex_unlock(&runtime->mutex);
	return(status);
}

static void IndexForget(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	uint32_t index;
	pthread_mutex_lock(&runtime->mutex);
	index = SparkKvSnapshotIndexFind(runtime,key);
	if ( index != SPARK_KV_SNAPSHOT_NO_ENTRY )
		SparkKvSnapshotIndexRemoveAt(store,index);
	pthread_mutex_unlock(&runtime->mutex);
}

static uint32_t IndexFlags(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key)
{
	SparkKvSnapshotRuntime *runtime = SparkKvSnapshotRuntimeOf(store);
	uint32_t index = SparkKvSnapshotIndexFind(runtime,key);
	return(index != SPARK_KV_SNAPSHOT_NO_ENTRY ? SparkKvSnapshotEntryAt(runtime,index)->flags : 0u);
}

static double IndexSeconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((double)now.tv_sec + (double)now.tv_nsec * 1e-9);
}

static void IndexTestEntrySize(void)
{
	CHECK(sizeof(SparkKvSnapshotIndexEntry) <= 72u);
	printf("kv snapshot index: one file costs %u bytes of index entry\n",(unsigned)sizeof(SparkKvSnapshotIndexEntry));
}

static void IndexTestScaleWithOrphan(void)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotKey key,parent,child,found;
	char directory[64];
	double start;
	uint32_t index,missing = 0u,high;
	if ( IndexOpen(&store,directory) == 0u )
	{
		CHECK(0);
		return;
	}
	runtime = SparkKvSnapshotRuntimeOf(&store);
	start = IndexSeconds();
	IndexKey(&parent,1u,UINT64_C(1) << 40u,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	IndexKey(&child,1u,(UINT64_C(1) << 40u) + 1u,128u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	CHECK(IndexInsert(&store,&child,&parent) == SPARK_STATUS_OK && store.orphan_count == 1u);
	for (index=0u; index<INDEX_SCALE_ROOTS; index++)
	{
		IndexKey(&key,1u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		if ( IndexInsert(&store,&key,0) != SPARK_STATUS_OK )
			missing++;
	}
	CHECK(missing == 0u && store.orphan_count == 1u && store.file_count == INDEX_SCALE_ROOTS + 1u);
	for (index=0u; index<INDEX_SCALE_ROOTS; index+=997u)
	{
		IndexKey(&key,1u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		missing += SparkKvSnapshotContains(&store,&key) != 0u ? 0u : 1u;
	}
	CHECK(missing == 0u);
	CHECK(IndexInsert(&store,&parent,0) == SPARK_STATUS_OK && store.orphan_count == 0u);
	CHECK((IndexFlags(&store,&parent) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) == 0u && (IndexFlags(&store,&child) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u);
	IndexForget(&store,&parent);
	CHECK(store.orphan_count == 1u && store.file_count == INDEX_SCALE_ROOTS + 1u && SparkKvSnapshotContains(&store,&parent) == 0u);
	CHECK(SparkKvSnapshotParent(&store,&child,&found) == SPARK_STATUS_OK && memcmp(&found,&parent,sizeof(found)) == 0);
	IndexForget(&store,&child);
	CHECK(store.orphan_count == 0u && runtime->entry_count == INDEX_SCALE_ROOTS);
	for (index=0u; index<INDEX_SCALE_ROOTS; index++)
	{
		IndexKey(&key,1u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		IndexForget(&store,&key);
	}
	CHECK(runtime->entry_count == 0u && store.file_count == 0u && store.used_bytes == 0u && runtime->lru_head == SPARK_KV_SNAPSHOT_NO_ENTRY);
	high = runtime->entry_high;
	for (index=0u; index<1000u; index++)
	{
		IndexKey(&key,1u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		CHECK(IndexInsert(&store,&key,0) == SPARK_STATUS_OK);
	}
	CHECK(runtime->entry_high == high && store.file_count == 1000u);
	CHECK(IndexSeconds() - start < INDEX_SCALE_SECONDS);
	printf("kv snapshot index: %u inserts and removes with an orphan present took %.2f s\n",2u * INDEX_SCALE_ROOTS,IndexSeconds() - start);
	IndexClose(&store,directory);
}

static void IndexTestLayoutCounts(void)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey key;
	uint8_t layout[SPARK_SHA256_DIGEST_BYTES];
	uint64_t matching = 0u,foreign = 0u;
	char directory[64];
	uint32_t index;
	if ( IndexOpen(&store,directory) == 0u )
	{
		CHECK(0);
		return;
	}
	for (index=0u; index<3u; index++)
	{
		IndexKey(&key,0xa0u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		CHECK(IndexInsert(&store,&key,0) == SPARK_STATUS_OK);
	}
	for (index=0u; index<2u; index++)
	{
		IndexKey(&key,0xb0u,index,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
		CHECK(IndexInsert(&store,&key,0) == SPARK_STATUS_OK);
	}
	IndexKey(&key,0xc0u,7u,4096u,SPARK_KV_SNAPSHOT_FILE_STATE);
	CHECK(IndexInsert(&store,&key,0) == SPARK_STATUS_OK);
	memset(layout,0xa0u,sizeof(layout));
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 3u && foreign == 3u);
	memset(layout,0xb0u,sizeof(layout));
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 2u && foreign == 4u);
	memset(layout,0xd0u,sizeof(layout));
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 0u && foreign == 6u);
	IndexKey(&key,0xa0u,1u,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	IndexForget(&store,&key);
	memset(layout,0xa0u,sizeof(layout));
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 2u && foreign == 3u);
	CHECK(SparkKvSnapshotPrune(&store,layout) == SPARK_STATUS_OK && store.file_count == 2u && store.pruned_count == 3u);
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 2u && foreign == 0u);
	IndexKey(&key,0xd0u,9u,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	CHECK(IndexInsert(&store,&key,0) == SPARK_STATUS_OK && SparkKvSnapshotRuntimeOf(&store)->layout_count == 3u);
	memset(layout,0xd0u,sizeof(layout));
	CHECK(SparkKvSnapshotCountLayout(&store,layout,&matching,&foreign) == SPARK_STATUS_OK && matching == 1u && foreign == 2u);
	printf("kv snapshot index: per-layout counts stay exact through removal and prune, and freed layout ids are reused\n");
	IndexClose(&store,directory);
}

static void IndexTestParentKeys(void)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey root,middle,leaf,found;
	char directory[64],expected[SPARK_KV_SNAPSHOT_PATH_BYTES],actual[SPARK_KV_SNAPSHOT_PATH_BYTES];
	if ( IndexOpen(&store,directory) == 0u )
	{
		CHECK(0);
		return;
	}
	IndexKey(&root,0x31u,1u,64u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	IndexKey(&middle,0x31u,2u,128u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	IndexKey(&leaf,0x31u,3u,192u,SPARK_KV_SNAPSHOT_FILE_PAGES);
	CHECK(IndexInsert(&store,&root,0) == SPARK_STATUS_OK && IndexInsert(&store,&middle,&root) == SPARK_STATUS_OK && IndexInsert(&store,&leaf,&middle) == SPARK_STATUS_OK);
	CHECK(store.orphan_count == 0u && SparkKvSnapshotParent(&store,&leaf,&found) == SPARK_STATUS_OK && memcmp(&found,&middle,sizeof(found)) == 0);
	CHECK((IndexFlags(&store,&root) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) == 0u && (IndexFlags(&store,&middle) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) == 0u && (IndexFlags(&store,&leaf) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u);
	IndexForget(&store,&middle);
	CHECK(store.orphan_count == 1u && SparkKvSnapshotParent(&store,&leaf,&found) == SPARK_STATUS_OK && memcmp(&found,&middle,sizeof(found)) == 0);
	CHECK((IndexFlags(&store,&root) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u && SparkKvSnapshotRuntimeOf(&store)->lru_head == SparkKvSnapshotIndexFind(SparkKvSnapshotRuntimeOf(&store),&leaf));
	CHECK(SparkKvSnapshotBuildPath(&store,&leaf,&middle,SparkKvSnapshotEntryAt(SparkKvSnapshotRuntimeOf(&store),SparkKvSnapshotIndexFind(SparkKvSnapshotRuntimeOf(&store),&leaf))->sequence,INDEX_FILE_BYTES / SPARK_KV_SNAPSHOT_ALIGNMENT,expected,sizeof(expected)) == SPARK_STATUS_OK &&
		SparkKvSnapshotFilePath(&store,&leaf,actual,sizeof(actual)) == SPARK_STATUS_OK && strcmp(expected,actual) == 0);
	IndexForget(&store,&root);
	CHECK(SparkKvSnapshotRuntimeOf(&store)->entry_count == 2u && store.file_count == 1u);
	CHECK(IndexInsert(&store,&middle,&root) == SPARK_STATUS_OK && store.orphan_count == 1u && store.file_count == 2u && SparkKvSnapshotRuntimeOf(&store)->entry_count == 3u);
	CHECK(SparkKvSnapshotParent(&store,&middle,&found) == SPARK_STATUS_OK && memcmp(&found,&root,sizeof(found)) == 0);
	CHECK(SparkKvSnapshotParent(&store,&leaf,&found) == SPARK_STATUS_OK && memcmp(&found,&middle,sizeof(found)) == 0);
	CHECK((IndexFlags(&store,&middle) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) == 0u);
	CHECK(IndexInsert(&store,&root,0) == SPARK_STATUS_OK && store.orphan_count == 0u && SparkKvSnapshotRuntimeOf(&store)->entry_count == 3u);
	IndexForget(&store,&leaf);
	CHECK((IndexFlags(&store,&middle) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u && SparkKvSnapshotRuntimeOf(&store)->lru_head == SparkKvSnapshotIndexFind(SparkKvSnapshotRuntimeOf(&store),&middle));
	printf("kv snapshot index: orphaned and adopted pages keep their full parent keys and leaf-first order\n");
	IndexClose(&store,directory);
}

static uint32_t IndexModelDepth(uint32_t node)
{
	uint32_t depth = 0u;
	for (node++; node > 1u; node>>=1u)
		depth++;
	return(depth);
}

static void IndexModelKey(SparkKvSnapshotKey *key,uint32_t node)
{
	IndexKey(key,0x55u,UINT64_C(1000) + node,64u * (IndexModelDepth(node) + 1u),SPARK_KV_SNAPSHOT_FILE_PAGES);
}

static void IndexTestAgainstModel(void)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotRuntime *runtime;
	SparkKvSnapshotKey key,parent,found;
	uint8_t present[INDEX_MODEL_KEYS];
	char directory[64];
	uint64_t state = 7u;
	uint32_t step,node,other,files,orphans,phantoms,has_child,wrong_lru = 0u,wrong_parent = 0u,wrong_count = 0u;
	if ( IndexOpen(&store,directory) == 0u )
	{
		CHECK(0);
		return;
	}
	runtime = SparkKvSnapshotRuntimeOf(&store);
	memset(present,0,sizeof(present));
	for (step=0u; step<INDEX_MODEL_STEPS; step++)
	{
		state = IndexMix(state);
		node = (uint32_t)(state % INDEX_MODEL_KEYS);
		IndexModelKey(&key,node);
		if ( present[node] == 0u )
		{
			if ( node != 0u )
				IndexModelKey(&parent,(node - 1u) / 2u);
			wrong_count += IndexInsert(&store,&key,node != 0u ? &parent : 0) == SPARK_STATUS_OK ? 0u : 1u;
			present[node] = 1u;
		}
		else
		{
			IndexForget(&store,&key);
			present[node] = 0u;
		}
		files = orphans = phantoms = 0u;
		for (node=0u; node<INDEX_MODEL_KEYS; node++)
		{
			has_child = 0u;
			for (other=2u * node + 1u; other<=2u * node + 2u && other<INDEX_MODEL_KEYS; other++)
				has_child |= present[other];
			if ( present[node] == 0u )
			{
				phantoms += has_child;
				continue;
			}
			files++;
			orphans += node != 0u && present[(node - 1u) / 2u] == 0u ? 1u : 0u;
			IndexModelKey(&key,node);
			wrong_lru += ((IndexFlags(&store,&key) & SPARK_KV_SNAPSHOT_ENTRY_IN_LRU) != 0u) == (has_child == 0u) ? 0u : 1u;
			if ( node != 0u )
			{
				IndexModelKey(&parent,(node - 1u) / 2u);
				wrong_parent += SparkKvSnapshotParent(&store,&key,&found) == SPARK_STATUS_OK && memcmp(&found,&parent,sizeof(found)) == 0 ? 0u : 1u;
			}
		}
		wrong_count += store.file_count == files && store.orphan_count == orphans && runtime->entry_count == files + phantoms && store.used_bytes == (uint64_t)files * INDEX_FILE_BYTES ? 0u : 1u;
	}
	CHECK(wrong_count == 0u && wrong_lru == 0u && wrong_parent == 0u);
	printf("kv snapshot index: %u random inserts and removes match the brute-force model (files, orphans, phantoms, LRU, parent keys)\n",INDEX_MODEL_STEPS);
	IndexClose(&store,directory);
}

int main(void)
{
	IndexTestEntrySize();
	IndexTestLayoutCounts();
	IndexTestParentKeys();
	IndexTestAgainstModel();
	IndexTestScaleWithOrphan();
	printf("test_kv_snapshot_index: %u checks, %u failures\n",index_checks,index_failures);
	return(index_failures == 0u ? 0 : 1);
}
