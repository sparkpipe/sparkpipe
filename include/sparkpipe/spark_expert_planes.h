#pragma once

#include <stdint.h>
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define SPARK_EXPERT_PLANE_PAYLOAD 0u
#define SPARK_EXPERT_PLANE_SCALE 1u
#define SPARK_EXPERT_PLANE_GLOBAL 2u
#define SPARK_EXPERT_PLANE_COUNT_MAX 3u
#define SPARK_EXPERT_PLANE_GLOBAL_KIND_BASE UINT32_C(0x10000)
#define SPARK_EXPERT_PLANE_TENSOR_KIND_MAX UINT32_C(0x7fff)
#define SPARK_EXPERT_PLANE_GLOBAL_BYTES 4u

typedef struct SparkExpertPlane
{
	uint64_t base;
	uint64_t stride;
	uint64_t bytes;
	uint32_t kind;
	uint32_t plane;
} SparkExpertPlane;

typedef struct SparkExpertPlanes
{
	SparkExpertPlane planes[SPARK_EXPERT_PLANE_COUNT_MAX];
	uint32_t count;
} SparkExpertPlanes;

static inline uint32_t SparkExpertPlaneKind(uint32_t tensor_kind,uint32_t plane)
{
	if ( plane == SPARK_EXPERT_PLANE_GLOBAL )
		return(SPARK_EXPERT_PLANE_GLOBAL_KIND_BASE + tensor_kind);
	return((tensor_kind * 2u) + plane);
}

static inline void SparkExpertPlaneSet(SparkExpertPlanes *out,uint32_t tensor_kind,uint32_t plane,uint64_t base,uint64_t stride,uint64_t bytes)
{
	SparkExpertPlane *target = &out->planes[out->count++];
	target->base = base;
	target->stride = stride;
	target->bytes = bytes;
	target->plane = plane;
	target->kind = SparkExpertPlaneKind(tensor_kind,plane);
}

static inline SparkStatus SparkExpertPlanesDescribe(
	uint32_t tensor_kind,
	uint32_t codec,
	uint32_t scale_encoding,
	uint32_t group_count,
	uint64_t payload_offset,
	uint64_t payload_bytes,
	uint64_t scale_offset,
	uint64_t scale_bytes,
	SparkExpertPlanes *out)
{
	uint64_t globals,block;
	if ( out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	out->count = 0u;
	if ( SparkWeightCodecIsKnown(codec) == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( tensor_kind > SPARK_EXPERT_PLANE_TENSOR_KIND_MAX || scale_encoding != SparkWeightCodecScaleEncoding(codec) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( group_count == 0u || payload_bytes == 0u || (payload_bytes % group_count) != 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	SparkExpertPlaneSet(out,tensor_kind,SPARK_EXPERT_PLANE_PAYLOAD,payload_offset,payload_bytes / group_count,payload_bytes / group_count);
	if ( scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_NONE )
		return(scale_bytes == 0u && scale_offset == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL )
	{
		globals = (uint64_t)group_count * SPARK_EXPERT_PLANE_GLOBAL_BYTES;
		if ( scale_bytes <= globals || ((scale_bytes - globals) % group_count) != 0u || scale_offset > UINT64_MAX - scale_bytes )
			return(SPARK_STATUS_SCHEMA_ERROR);
		block = (scale_bytes - globals) / group_count;
		SparkExpertPlaneSet(out,tensor_kind,SPARK_EXPERT_PLANE_SCALE,scale_offset + globals,block,block);
		SparkExpertPlaneSet(out,tensor_kind,SPARK_EXPERT_PLANE_GLOBAL,scale_offset,SPARK_EXPERT_PLANE_GLOBAL_BYTES,SPARK_EXPERT_PLANE_GLOBAL_BYTES);
		return(SPARK_STATUS_OK);
	}
	if ( scale_bytes == 0u || (scale_bytes % group_count) != 0u || scale_offset > UINT64_MAX - scale_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	SparkExpertPlaneSet(out,tensor_kind,SPARK_EXPERT_PLANE_SCALE,scale_offset,scale_bytes / group_count,scale_bytes / group_count);
	return(SPARK_STATUS_OK);
}

static inline uint64_t SparkExpertPlaneOffset(const SparkExpertPlane *plane,uint32_t expert)
{
	return(plane->base + ((uint64_t)expert * plane->stride));
}

static inline SparkStatus SparkExpertPlaneCovered(const SparkWeightdManifest *manifest,uint32_t layer,uint32_t group_count,const SparkExpertPlane *plane)
{
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	uint32_t expert,index;
	for (expert=0u; expert<group_count; expert++)
	{
		group = SparkWeightdManifestFind(manifest,layer,expert);
		if ( group == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		range = 0;
		for (index=0u; index<group->range_count; index++)
			if ( manifest->ranges[group->first_range + index].kind == plane->kind )
			{
				range = &manifest->ranges[group->first_range + index];
				break;
			}
		if ( range == 0 || range->layer != layer || range->expert != expert || range->offset != SparkExpertPlaneOffset(plane,expert) || range->bytes != plane->bytes )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}
