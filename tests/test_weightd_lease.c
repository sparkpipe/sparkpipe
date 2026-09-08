#include "sparkpipe/spark_weightd_lease.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void check_arbitrary_working_sets(void)
{
	SparkWeightdRangeGroup groups[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	uint8_t seen[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkWeightdManifest manifest = {0,groups,SPARK_WEIGHTD_LEASE_GROUPS_MAX,SPARK_WEIGHTD_LEASE_GROUPS_MAX};
	SparkWeightdLeaseTable *table;
	const SparkWeightdLease *lease;
	uint64_t id;
	uint32_t i,n,unique,state = 123u;
	for (i=0u; i<SPARK_WEIGHTD_LEASE_GROUPS_MAX; i++)
		groups[i] = (SparkWeightdRangeGroup){3u,i,i,1u};
	assert(SparkWeightdLeaseTableCreate(&manifest,&table) == SPARK_STATUS_OK);
	for (n=1u; n<=SPARK_WEIGHTD_LEASE_GROUPS_MAX; n++)
	{
		memset(seen,0,sizeof(seen));
		unique = 0u;
		for (i=0u; i<n; i++)
		{
			state = ((state * 1664525u) + 1013904223u);
			keys[i] = (SparkWeightdExpertKey){3u,state % SPARK_WEIGHTD_LEASE_GROUPS_MAX};
			if ( seen[keys[i].expert] == 0u )
				unique++;
			seen[keys[i].expert] = 1u;
		}
		assert(SparkWeightdLeaseAcquire(table,1u,keys,n,&id) == SPARK_STATUS_OK);
		lease = SparkWeightdLeaseFind(table,1u,id);
		assert(lease != 0 && lease->count == unique);
		for (i=1u; i<lease->count; i++)
			assert(lease->groups[i - 1u] < lease->groups[i]);
		for (i=0u; i<SPARK_WEIGHTD_LEASE_GROUPS_MAX; i++)
			assert(table->pins[i] == seen[i]);
		assert(SparkWeightdLeaseRelease(table,1u,id) == SPARK_STATUS_OK);
	}
	assert(SparkWeightdLeaseTableDestroy(table) == SPARK_STATUS_OK);
}

int main(void)
{
	SparkWeightdRangeGroup groups[3] = {{3u,0u,0u,4u},{3u,1u,4u,4u},{4u,0u,8u,4u}};
	SparkWeightdManifest manifest = {0,groups,12u,3u};
	SparkWeightdLeaseTable *table;
	SparkWeightdExpertKey a[3] = {{3u,1u},{3u,0u},{3u,1u}},bad[2] = {{3u,0u},{99u,0u}};
	uint64_t first,second,failed,ids[SPARK_WEIGHTD_LEASE_COUNT_MAX];
	uint32_t i;
	check_arbitrary_working_sets();
	assert(SparkWeightdLeaseTableCreate(&manifest,&table) == SPARK_STATUS_OK);
	assert(SparkWeightdLeaseAcquire(table,1u,a,3u,&first) == SPARK_STATUS_OK);
	assert(table->pins[0] == 1u && table->pins[1] == 1u && table->pins[2] == 0u);
	assert(SparkWeightdLeaseFind(table,1u,first)->count == 2u);
	assert(SparkWeightdLeaseTableDestroy(table) == SPARK_STATUS_BUSY);
	assert(SparkWeightdLeaseAcquire(table,2u,a,1u,&second) == SPARK_STATUS_OK);
	assert(table->pins[1] == 2u);
	assert(SparkWeightdLeaseAcquire(table,3u,bad,2u,&failed) == SPARK_STATUS_NOT_FOUND);
	assert(failed == 0u && table->pins[0] == 1u && table->pins[1] == 2u);
	assert(SparkWeightdLeaseRelease(table,2u,first) == SPARK_STATUS_NOT_FOUND);
	assert(table->pins[0] == 1u && table->pins[1] == 2u);
	assert(SparkWeightdLeaseRelease(table,1u,first) == SPARK_STATUS_OK);
	assert(table->pins[0] == 0u && table->pins[1] == 1u);
	assert(SparkWeightdLeaseRelease(table,1u,first) == SPARK_STATUS_NOT_FOUND);
	assert(SparkWeightdLeaseRelease(table,2u,second) == SPARK_STATUS_OK);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		assert(SparkWeightdLeaseAcquire(table,1u,a,1u,&ids[i]) == SPARK_STATUS_OK);
	assert(ids[0] > second && SparkWeightdLeaseFind(table,1u,first) == 0);
	assert(SparkWeightdLeaseAcquire(table,1u,a,1u,&failed) == SPARK_STATUS_BUSY);
	assert(failed == 0u && table->pins[1] == SPARK_WEIGHTD_LEASE_COUNT_MAX);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		assert(SparkWeightdLeaseRelease(table,1u,ids[i]) == SPARK_STATUS_OK);
	table->pins[1] = UINT32_MAX;
	assert(SparkWeightdLeaseAcquire(table,1u,a,3u,&failed) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(failed == 0u && table->pins[0] == 0u && table->pins[1] == UINT32_MAX);
	table->pins[1] = 0u;
	table->next_identifier = UINT64_MAX;
	assert(SparkWeightdLeaseAcquire(table,1u,a,1u,&failed) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(table->pins[1] == 0u);
	assert(SparkWeightdLeaseTableDestroy(table) == SPARK_STATUS_OK);
	puts("PASS weightd leases: shared pins, atomic failure, ownership, stale IDs and bounds");
	return(0);
}
