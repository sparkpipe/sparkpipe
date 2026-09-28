#pragma once

static SparkStatus SPARK_FAMILY(ManifestCheck)(const SparkWeightdManifest *manifest,void *opaque)
{
	const SPARK_FAMILY(ManifestContext) *context = (const SPARK_FAMILY(ManifestContext) *)opaque;
	const SPARK_FAMILY(StagePackEntry) *entry;
	SparkStatus status;
	uint64_t expected = 0u;
	uint32_t index,plane;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( entry->tensor_kind != SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_UP_GATE) && entry->tensor_kind != SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_DOWN) )
			continue;
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		for (plane=0u; plane<2u; plane++)
		{
			status = SPARK_FAMILY(ManifestPlane)(manifest,entry,plane);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}
