#pragma once

#include "sparkpipe/spark_expert_planes.h"

static SparkStatus SPARK_FAMILY(ManifestCheck)(const SparkWeightdManifest *manifest,void *opaque)
{
	const SPARK_FAMILY(ManifestContext) *context = (const SPARK_FAMILY(ManifestContext) *)opaque;
	const SPARK_FAMILY(StagePackEntry) *entry;
	SparkExpertPlanes planes;
	SparkStatus status;
	uint64_t expected = 0u;
	uint32_t index,plane;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( entry->tensor_kind != SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_UP_GATE) && entry->tensor_kind != SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_DOWN) )
			continue;
		status = SparkExpertPlanesDescribe(entry->tensor_kind,entry->weight_codec,entry->scale_encoding,entry->group_count,entry->payload_offset,entry->payload_bytes,entry->scale_offset,entry->scale_bytes,&planes);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		for (plane=0u; plane<planes.count; plane++)
		{
			status = SparkExpertPlaneCovered(manifest,entry->layer_index,entry->group_count,&planes.planes[plane]);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}
