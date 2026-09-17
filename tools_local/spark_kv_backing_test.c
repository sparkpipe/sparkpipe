#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_backing.h"

static int failures = 0;

#define CHECK(cond, name) do { \
	if ( !(cond) ) { printf("FAIL %s\n", name); failures++; } \
	else printf("ok   %s\n", name); } while (0)

static int file_mode_is(const char *path, unsigned int expected_mode)
{
	struct stat status;
	return stat(path,&status) == 0 &&
		(unsigned int)(status.st_mode & 0777u) == expected_mode;
}

int main(void)
{
	SparkKvBackingConfiguration configuration;
	SparkKvBacking backing;
	unsigned char *block = (unsigned char *)malloc(SPARK_KV_BACKING_SLOT_BYTES);
	unsigned char *verify = (unsigned char *)malloc(SPARK_KV_BACKING_SLOT_BYTES);
	int64_t a,b,c;
	uint32_t i;
	const char *path = "/tmp/spark_kv_backing_test.bin";
	unlink(path);
	for ( i = 0u; i < SPARK_KV_BACKING_SLOT_BYTES; i++ )
		block[i] = (unsigned char)(i * 131u + 7u);
	memset(&configuration,0,sizeof(configuration));
	configuration.path = path;
	configuration.maximum_bytes = 3ull * 1024ull * 1024ull * 1024ull;
	CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_OK, "open");
	CHECK(backing.slot_count == 768u, "geometry 768 slots");
	CHECK(file_mode_is(path,0600u), "B4: fresh slot file is 0600, world-blind");
	a = SparkKvBackingAllocate(&backing);
	b = SparkKvBackingAllocate(&backing);
	CHECK(a == 0 && b == 1, "sequential alloc");
	CHECK(SparkKvBackingWriteBlock(&backing,(uint32_t)a,block) == SPARK_STATUS_OK, "write block");
	memset(verify,0,SPARK_KV_BACKING_SLOT_BYTES);
	CHECK(SparkKvBackingReadBlock(&backing,(uint32_t)a,verify) == SPARK_STATUS_OK, "read block");
	CHECK(memcmp(block,verify,SPARK_KV_BACKING_SLOT_BYTES) == 0, "round-trip integrity");
	SparkKvBackingRelease(&backing,(uint32_t)b);
	{
		int64_t d = SparkKvBackingAllocate(&backing);
		CHECK(d == 2, "alloc continues from free_hint");
		SparkKvBackingRelease(&backing,(uint32_t)d);
		SparkKvBackingRelease(&backing,(uint32_t)d);
	}
	c = SparkKvBackingAllocate(&backing);
	CHECK(c >= 0 && backing.slot_live[c] != 0u, "released slot is reusable");
	CHECK(SparkKvBackingWriteBlock(&backing,(uint32_t)b,(void *)0) == SPARK_STATUS_INVALID_ARGUMENT, "null buffer rejected");
	{
		int64_t slot;
		uint32_t allocated = 2u;
		while ( (slot = SparkKvBackingAllocate(&backing)) >= 0 )
			allocated++;
		CHECK(slot == -1 && allocated == backing.slot_count, "horizon exhaustion returns -1");
		CHECK(SparkKvBackingLiveBytes(&backing) == 3ull * 1024ull * 1024ull * 1024ull, "live bytes == budget");
	}
	SparkKvBackingClose(&backing);
	configuration.maximum_bytes = 3ull * 1024ull * 1024ull * 1024ull;
	CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_OK, "same-geometry reopen");
	SparkKvBackingClose(&backing);
	{
		FILE *foreign = fopen(path,"r+b");
		if ( foreign != 0 )
		{
			fwrite("NOTKV01",1,8,foreign);
			fclose(foreign);
		}
		CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_TARGET_MISMATCH, "foreign header refused");
	}
	{
		FILE *legacy = fopen(path,"wb");
		if ( legacy != 0 )
			fclose(legacy);
		(void)chmod(path,0644u);
		CHECK(file_mode_is(path,0644u), "B4: legacy file starts 0644");
		CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_OK, "empty legacy file opens as fresh");
		SparkKvBackingClose(&backing);
		CHECK(file_mode_is(path,0600u), "B4: open migrated 0644 to 0600");
		unlink(path);
	}
	{
		const char *target = "/tmp/spark_kv_backing_symlink_target.bin";
		unlink(target);
		unlink(path);
		{
			FILE *real_file = fopen(target,"wb");
			if ( real_file != 0 )
				fclose(real_file);
		}
		CHECK(symlink(target,path) == 0, "B4: symlink planted");
		CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_IO_ERROR, "B4: symlink at slot path refused (O_NOFOLLOW)");
		unlink(path);
		unlink(target);
	}
	{
		char resolved[1024];
		char expected[1024];
		const char *root = "/tmp/spark_kv_backing_namespaces";
		CHECK(SparkKvBackingResolvePath(resolved,sizeof(resolved),root,
			"deploy1","tenant-a","glm5_next") == SPARK_STATUS_OK,
			"B4: namespaced path resolves");
		snprintf(expected,sizeof(expected),"%s/deploy1/tenant-a/glm5_next.slots",root);
		CHECK(strcmp(resolved,expected) == 0, "B4: namespaced path layout");
		CHECK(SparkKvBackingResolvePath(resolved,sizeof(resolved),root,
			"../escape","tenant-a","glm5_next") == SPARK_STATUS_INVALID_ARGUMENT,
			"B4: traversal deployment id rejected");
		CHECK(SparkKvBackingResolvePath(resolved,sizeof(resolved),root,
			"deploy1","..","glm5_next") == SPARK_STATUS_INVALID_ARGUMENT,
			"B4: traversal tenant id rejected");
		CHECK(SparkKvBackingResolvePath(resolved,sizeof(resolved),root,
			"de ploy","tenant-a","glm5_next") == SPARK_STATUS_INVALID_ARGUMENT,
			"B4: separator/space in id rejected");
		CHECK(SparkKvBackingResolvePath(resolved,sizeof(resolved),root,
			"deploy1","tenant-a","") == SPARK_STATUS_INVALID_ARGUMENT,
			"B4: empty model id rejected");
		CHECK(SparkKvBackingCreateNamespaces(root,"deploy1","tenant-a") ==
			SPARK_STATUS_OK, "B4: namespaces created");
		CHECK(file_mode_is(root,0700u), "B4: root namespace 0700");
		{
			char tenant_dir[1024];
			snprintf(tenant_dir,sizeof(tenant_dir),"%s/deploy1/tenant-a",root);
			CHECK(file_mode_is(tenant_dir,0700u), "B4: tenant namespace 0700");
		}
		{
			char deployment_dir[1024];
			snprintf(deployment_dir,sizeof(deployment_dir),"%s/deploy1",root);
			(void)chmod(deployment_dir,0755u);
			CHECK(SparkKvBackingCreateNamespaces(root,"deploy1","tenant-a") ==
				SPARK_STATUS_OK, "B4: namespace reopen ok");
			CHECK(file_mode_is(deployment_dir,0700u),
				"B4: loose namespace tightened to 0700");
		}
		{
			char cleanup[1024];
			snprintf(cleanup,sizeof(cleanup),"%s/deploy1/tenant-a",root);
			unlink(cleanup);
			snprintf(cleanup,sizeof(cleanup),"%s/deploy1",root);
			rmdir(cleanup);
			rmdir(root);
		}
	}
	free(block);
	free(verify);
	unlink(path);
	printf(failures == 0 ? "ALL PASS\n" : "%d FAILURES\n", failures);
	return(failures == 0 ? 0 : 1);
}
