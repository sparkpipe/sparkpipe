#include "../cache/kv_snapshot.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_CHAIN_FILES 64u
#define BENCH_PAGE_UNITS 28u

static void BenchKey(SparkKvSnapshotKey *key,uint64_t chain,uint32_t depth)
{
	uint64_t mixed = chain * UINT64_C(0x9e3779b97f4a7c15) + depth;
	memset(key,0,sizeof(*key));
	memset(key->layout_sha256,0x5au,sizeof(key->layout_sha256));
	mixed ^= mixed >> 31u;
	mixed *= UINT64_C(0xbf58476d1ce4e5b9);
	mixed ^= mixed >> 29u;
	memcpy(key->identity_sha256,&mixed,sizeof(mixed));
	memcpy(key->identity_sha256 + sizeof(mixed),&chain,sizeof(chain));
	memcpy(key->identity_sha256 + 2u * sizeof(mixed),&depth,sizeof(depth));
	key->token_count = 64u * (depth + 1u);
	key->file_kind = SPARK_KV_SNAPSHOT_FILE_PAGES;
}

static int BenchCreate(const char *directory,uint64_t count)
{
	SparkKvSnapshotStore store;
	SparkKvSnapshotKey key,parent;
	char path[SPARK_KV_SNAPSHOT_PATH_BYTES],leaf[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t file;
	int descriptor;
	memset(&store,0,sizeof(store));
	(void)snprintf(store.directory,sizeof(store.directory),"%s",directory);
	for (file=0u; file<count; file++)
	{
		BenchKey(&key,file / BENCH_CHAIN_FILES,(uint32_t)(file % BENCH_CHAIN_FILES));
		if ( file % BENCH_CHAIN_FILES != 0u )
			BenchKey(&parent,file / BENCH_CHAIN_FILES,(uint32_t)(file % BENCH_CHAIN_FILES) - 1u);
		else
			memset(&parent,0,sizeof(parent));
		if ( SparkKvSnapshotBuildPath(&store,&key,&parent,file + 1u,BENCH_PAGE_UNITS,path,sizeof(path)) != SPARK_STATUS_OK ||
			SparkKvSnapshotMakeLeaf(&store,path,leaf,sizeof(leaf)) != SPARK_STATUS_OK )
			return(1);
		descriptor = open(path,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0600);
		if ( descriptor < 0 || close(descriptor) != 0 )
			return(1);
	}
	printf("KV-OPEN-BENCH created files=%llu directory=%s\n",(unsigned long long)count,directory);
	return(0);
}

static int BenchOpen(const char *directory)
{
	SparkKvSnapshotStore store;
	if ( SparkKvSnapshotStoreOpen(&store,directory,UINT64_C(1) << 50u,UINT64_C(1) << 20u) != SPARK_STATUS_OK )
		return(1);
	printf("KV-OPEN-BENCH open_us=%llu files=%llu used_bytes=%llu orphans=%llu layout_dirs=%llu scan_threads=%llu\n",
		(unsigned long long)(store.open_ns / 1000u),(unsigned long long)store.file_count,(unsigned long long)store.used_bytes,
		(unsigned long long)store.orphan_count,(unsigned long long)store.layout_directory_count,(unsigned long long)store.scan_thread_count);
	SparkKvSnapshotStoreClose(&store);
	return(0);
}

int main(int argc,char **argv)
{
	if ( argc == 4 && strcmp(argv[1],"create") == 0 )
		return(BenchCreate(argv[2],strtoull(argv[3],0,10)));
	if ( argc == 3 && strcmp(argv[1],"open") == 0 )
		return(BenchOpen(argv[2]));
	fprintf(stderr,"usage: kv_store_open_bench create <empty store directory> <files> | open <store directory>\n");
	return(2);
}
