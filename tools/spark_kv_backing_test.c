/* Unit test for the JIT-KV backing store (docs/JIT_KV_DESIGN.md step 1).
 * Verifies: open/geometry, alloc/free lifecycle, 4 MiB block round-trip
 * integrity, horizon exhaustion (-1 -> backpressure signal, not thrash),
 * release-then-realloc, reopen-with-mismatch refusal.
 * Build: make build/spark_kv_backing_test && ./build/spark_kv_backing_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_backing.h"

static int failures = 0;

#define CHECK(cond, name) do { \
	if ( !(cond) ) { printf("FAIL %s\n", name); failures++; } \
	else printf("ok   %s\n", name); } while (0)

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
	configuration.maximum_bytes = 3ull * 1024ull * 1024ull * 1024ull; /* 3 GB = 768 slots */
	CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_OK, "open");
	CHECK(backing.slot_count == 768u, "geometry 768 slots");
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
	/* horizon exhaustion */
	{
		int64_t slot;
		uint32_t allocated = 2u;
		while ( (slot = SparkKvBackingAllocate(&backing)) >= 0 )
			allocated++;
		CHECK(slot == -1 && allocated == backing.slot_count, "horizon exhaustion returns -1");
		CHECK(SparkKvBackingLiveBytes(&backing) == 3ull * 1024ull * 1024ull * 1024ull, "live bytes == budget");
	}
	SparkKvBackingClose(&backing);
	/* reopen with the same budget succeeds (header persisted) */
	configuration.maximum_bytes = 3ull * 1024ull * 1024ull * 1024ull;
	CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_OK, "same-geometry reopen");
	SparkKvBackingClose(&backing);
	/* a file carrying a foreign non-zero header must be refused */
	{
		FILE *foreign = fopen(path,"r+b");
		if ( foreign != 0 )
		{
			fwrite("NOTKV01",1,8,foreign);
			fclose(foreign);
		}
		CHECK(SparkKvBackingOpen(&configuration,&backing) == SPARK_STATUS_TARGET_MISMATCH, "foreign header refused");
	}
	free(block);
	free(verify);
	unlink(path);
	printf(failures == 0 ? "ALL PASS\n" : "%d FAILURES\n", failures);
	return(failures == 0 ? 0 : 1);
}
