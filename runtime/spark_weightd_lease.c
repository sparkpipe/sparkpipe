#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_error_site.h"
#include <stdlib.h>

SparkStatus SparkWeightdLeaseTableCreate(const SparkWeightdManifest *manifest,SparkWeightdLeaseTable **out)
{
	SparkWeightdLeaseTable *table;
	if ( out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*out = 0;
	if ( manifest == 0 || manifest->groups == 0 || manifest->group_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	table = calloc(1u,sizeof(*table));
	if ( table == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	table->pins = calloc(manifest->group_count,sizeof(*table->pins));
	if ( table->pins == 0 )
	{
		free(table);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	table->manifest = manifest;
	table->next_identifier = 1u;
	*out = table;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLeaseTableDestroy(SparkWeightdLeaseTable *table)
{
	uint32_t i;
	if ( table == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( table->leases[i].count != 0u )
			SPARK_FAIL(SPARK_STATUS_BUSY);
	free(table->pins);
	free(table);
	return(SPARK_STATUS_OK);
}

static void sift_groups(uint32_t *groups,uint32_t root,uint32_t count)
{
	uint32_t child,value;
	for (;;)
	{
		child = ((root * 2u) + 1u);
		if ( child >= count )
			return;
		if ( (child + 1u) < count && groups[child] < groups[child + 1u] )
			child++;
		if ( groups[root] >= groups[child] )
			return;
		value = groups[root];
		groups[root] = groups[child];
		groups[child] = value;
		root = child;
	}
}

static void sort_groups(uint32_t *groups,uint32_t count)
{
	uint32_t i,value;
	for (i=(count / 2u); i>0u; i--)
		sift_groups(groups,(i - 1u),count);
	for (i=count; i>1u; i--)
	{
		value = groups[0];
		groups[0] = groups[i - 1u];
		groups[i - 1u] = value;
		sift_groups(groups,0u,(i - 1u));
	}
}

static SparkStatus prepare_groups(SparkWeightdLeaseTable *table,SparkWeightdLease *lease,const SparkWeightdExpertKey *keys,uint32_t count,uint32_t *unique)
{
	const SparkWeightdRangeGroup *group;
	uint32_t i;
	*unique = 0u;
	for (i=0u; i<count; i++)
	{
		group = SparkWeightdManifestFind(table->manifest,keys[i].layer,keys[i].expert);
		if ( group == 0 )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
		lease->groups[i] = (uint32_t)(group - table->manifest->groups);
	}
	sort_groups(lease->groups,count);
	for (i=0u; i<count; i++)
		if ( *unique == 0u || lease->groups[i] != lease->groups[*unique - 1u] )
			lease->groups[(*unique)++] = lease->groups[i];
	for (i=0u; i<*unique; i++)
		if ( table->pins[lease->groups[i]] == UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLeaseAcquire(SparkWeightdLeaseTable *table,uint64_t owner,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier)
{
	SparkWeightdLease *lease = 0;
	SparkStatus status;
	uint32_t i,unique;
	if ( identifier == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*identifier = 0u;
	if ( table == 0 || owner == 0u || keys == 0 || count == 0u || count > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( table->next_identifier == UINT64_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( table->leases[i].count == 0u )
		{
			lease = &table->leases[i];
			break;
		}
	if ( lease == 0 )
		SPARK_FAIL(SPARK_STATUS_BUSY);
	status = prepare_groups(table,lease,keys,count,&unique);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (i=0u; i<unique; i++)
		table->pins[lease->groups[i]]++;
	lease->count = unique;
	lease->owner = owner;
	lease->identifier = table->next_identifier++;
	*identifier = lease->identifier;
	return(SPARK_STATUS_OK);
}

const SparkWeightdLease *SparkWeightdLeaseFind(const SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier)
{
	uint32_t i;
	if ( table == 0 || owner == 0u || identifier == 0u )
		return(0);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( table->leases[i].count != 0u && table->leases[i].owner == owner && table->leases[i].identifier == identifier )
			return(&table->leases[i]);
	return(0);
}

SparkStatus SparkWeightdLeaseRelease(SparkWeightdLeaseTable *table,uint64_t owner,uint64_t identifier)
{
	const SparkWeightdLease *found;
	SparkWeightdLease *lease;
	uint32_t i;
	found = SparkWeightdLeaseFind(table,owner,identifier);
	if ( found == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	lease = &table->leases[found - table->leases];
	for (i=0u; i<lease->count; i++)
		if ( table->pins[lease->groups[i]] == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (i=0u; i<lease->count; i++)
		table->pins[lease->groups[i]]--;
	lease->count = 0u;
	lease->owner = 0u;
	lease->identifier = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *offsets,uint32_t expert_count,uint32_t packed_rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count)
{
	uint32_t i,needed = 0u;
	if ( count == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*count = 0u;
	if ( offsets == 0 || keys == 0 || expert_count == 0u || expert_count > SPARK_WEIGHTD_RANGE_COUNT_MAX || packed_rows == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( offsets[0] != 0u || offsets[expert_count] != packed_rows )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (i=0u; i<expert_count; i++)
	{
		if ( offsets[i] > offsets[i + 1u] || offsets[i + 1u] > packed_rows )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( offsets[i] != offsets[i + 1u] )
			needed++;
	}
	if ( needed > capacity || needed > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (i=0u; i<expert_count; i++)
		if ( offsets[i] != offsets[i + 1u] )
		{
			keys[*count].layer = layer;
			keys[*count].expert = i;
			(*count)++;
		}
	return(SPARK_STATUS_OK);
}
