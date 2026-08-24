/*
 * Stage-pack parity harness, glm52 dialect. The glm52 header is the family
 * that already rides the shared reader; this pin holds its live revision
 * against the frozen reference so the shared reader's evolution can never
 * drift glm52's accepted, received or rejected streams either. Built twice
 * by tests/test_stagepack_parity.py; outputs must be byte-identical.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h> /* the folded contract sits beside the CUDA launch surface; stub path comes from the harness */
#include "spark_glm52_resident_decode_stage_internal.h"

static void emit(const char *tag,long value)
{
	printf("%s %ld\n",tag,value);
}

/* The module's mask resolver: ExpectedShape with its context pinned. */
static int32_t resolve_shape(void *context,uint32_t tensor_kind,uint32_t layer_index,SparkStagePackShape *shape)
{
	return(SparkGlm52StagePackExpectedShape(tensor_kind,layer_index,*(const uint32_t *)context,1u,shape));
}

int main(void)
{
	static const uint32_t layers[] = {SPARK_GLM52_STAGEPACK_GLOBAL_LAYER,0u,1u,2u,3u,4u,5u,8u,20u,40u,60u,SPARK_GLM52_MODEL_LAYER_COUNT,SPARK_GLM52_MODEL_LAYER_COUNT + 1u,UINT32_MAX};
	static const uint32_t codecs[] = {SPARK_WEIGHT_CODEC_BF16,SPARK_WEIGHT_CODEC_INT6,SPARK_WEIGHT_CODEC_INT7,SPARK_WEIGHT_CODEC_INT8,SPARK_WEIGHT_CODEC_FP8_E4M3,SPARK_WEIGHT_CODEC_NVFP4_E2M1,SPARK_WEIGHT_CODEC_MXFP4_E2M1,SPARK_WEIGHT_CODEC_NONE,UINT32_MAX};
	static const uint32_t degrees[] = {0u,1u,2u,4u,7u,8u,16u,64u,256u};
	char tag[96];
	uint32_t kind,i,f,r,c;

	/* A: expected-shape refusal codes over the hostile grid. */
	for (kind = 0; kind < SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
			for (f = 0; f < (uint32_t)(sizeof(codecs)/sizeof(codecs[0])); f++)
				for (r = 0; r < (uint32_t)(sizeof(degrees)/sizeof(degrees[0])); r += 4u)
				{
					SparkStagePackShape shape;
					int32_t code;
					memset(&shape,0,sizeof(shape));
					code = SparkGlm52StagePackExpectedShape(kind,layers[i],codecs[f],degrees[r],&shape);
					snprintf(tag,sizeof(tag),"A kind=%u layer=%u codec=%u degree=%u",kind,layers[i],codecs[f],degrees[r]);
					emit(tag,(long)code);
					emit("Ashape",(long)(((uint64_t)shape.payload_type << 56) | ((uint64_t)shape.weight_codec << 40) | ((uint64_t)shape.group_count << 20) | shape.rows));
					emit("Acols",(long)shape.columns);
					emit("Ascale",(long)shape.scale_encoding);
				}

	/* B: packed-shape byte accounting over codecs and the dimension grid. */
	{
		static const uint32_t dims[] = {0u,1u,15u,16u,31u,32u,33u,63u,64u,127u,128u,129u,1024u,4096u,2147483647u,UINT32_MAX};
		for (f = 1; f < (uint32_t)(sizeof(codecs)/sizeof(codecs[0])); f++)
			for (i = 0; i < (uint32_t)(sizeof(dims)/sizeof(dims[0])); i++)
				for (c = 0; c < (uint32_t)(sizeof(dims)/sizeof(dims[0])); c++)
				{
					SparkStagePackShape shape;
					uint32_t groups = 1u + (i & 7u);
					memset(&shape,0,sizeof(shape));
					if ( SparkStagePackShapePacked(&shape,codecs[f],groups,dims[i],dims[c]) == 0 )
					{
						snprintf(tag,sizeof(tag),"B codec=%u groups=%u r=%u c=%u payload",codecs[f],groups,dims[i],dims[c]);
						emit(tag,(long)SparkStagePackPayloadBytes(&shape));
						snprintf(tag,sizeof(tag),"B codec=%u groups=%u r=%u c=%u scale",codecs[f],groups,dims[i],dims[c]);
						emit(tag,(long)SparkStagePackScaleBytes(&shape));
					}
					SparkStagePackShapeBf16(&shape,groups,dims[i],dims[c]);
					snprintf(tag,sizeof(tag),"B bf16 groups=%u r=%u c=%u payload",groups,dims[i],dims[c]);
					emit(tag,(long)SparkStagePackPayloadBytes(&shape));
					emit("B bf16 scale",(long)SparkStagePackScaleBytes(&shape));
				}
	}

	/* C: entry-bounds refusals over offset/byte mutation classes. */
	{
		static const uint64_t offsets[] = {0u,1u,255u,256u,512u,4095u,4096u,4097u,8192u,UINT64_MAX};
		SparkStagePackShape shape;
		uint32_t o1,o2;
		memset(&shape,0,sizeof(shape));
		SparkStagePackShapeBf16(&shape,1u,64u,128u);
		for (o1 = 0; o1 < (uint32_t)(sizeof(offsets)/sizeof(offsets[0])); o1++)
			for (o2 = 0; o2 < (uint32_t)(sizeof(offsets)/sizeof(offsets[0])); o2++)
			{
				int32_t code = SparkStagePackCheckEntryBounds(256u,&shape,256u,8192u,offsets[o1],SparkStagePackPayloadBytes(&shape),offsets[o2],SparkStagePackScaleBytes(&shape));
				snprintf(tag,sizeof(tag),"C po=%llu so=%llu",(unsigned long long)offsets[o1],(unsigned long long)offsets[o2]);
				emit(tag,(long)code);
				code = SparkStagePackCheckEntryBounds(256u,&shape,256u,8192u,offsets[o1],SparkStagePackPayloadBytes(&shape) + 1u,offsets[o2],SparkStagePackScaleBytes(&shape));
				emit("C badpayload",(long)code);
				code = SparkStagePackCheckEntryBounds(256u,&shape,256u,8192u,offsets[o1],SparkStagePackPayloadBytes(&shape),offsets[o2],SparkStagePackScaleBytes(&shape) + 1u);
				emit("C badscale",(long)code);
			}
	}

	/* D: seen-bitmask and range-overlap mechanics. */
	{
		uint64_t seen = 0u;
		uint32_t k;
		for (k = 0; k < 70u; k++)
		{
			uint32_t had = SparkStagePackSeenHas(seen,k);
			SparkStagePackSeenMark(&seen,k);
			snprintf(tag,sizeof(tag),"D kind=%u had=%u has=%u",k,had,SparkStagePackSeenHas(seen,k));
			emit(tag,(long)0);
		}
		for (i = 0; i < 6u; i++)
			for (r = 0; r < 6u; r++)
			{
				uint64_t a = i * 300u,b = r * 300u;
				snprintf(tag,sizeof(tag),"D overlap a=%llu b=%llu",(unsigned long long)a,(unsigned long long)b);
				emit(tag,(long)SparkStagePackRangesOverlap(a,256u,b,256u));
			}
	}

	/* E: shard application over policies and degrees. */
	{
		static const uint32_t policies[] = {SPARK_STAGE_PACK_SHARD_NONE,SPARK_STAGE_PACK_SHARD_ROWS,SPARK_STAGE_PACK_SHARD_COLUMNS,9u};
		for (kind = 0; kind < SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
			for (i = 0; i < (uint32_t)(sizeof(degrees)/sizeof(degrees[0])); i++)
			{
				SparkStagePackShape shape;
				memset(&shape,0,sizeof(shape));
				SparkStagePackShapeBf16(&shape,1u,4096u,1024u);
				snprintf(tag,sizeof(tag),"E kind=%u degree=%u",kind,degrees[i]);
				emit(tag,(long)SparkStagePackApplyShard(&shape,SparkGlm52StagePackTpShardPolicy(kind),degrees[i]));
				emit("E rows",(long)shape.rows);
				emit("E cols",(long)shape.columns);
			}
		for (i = 0; i < (uint32_t)(sizeof(policies)/sizeof(policies[0])); i++)
			for (r = 0; r < 4u; r++)
			{
				SparkStagePackShape shape;
				memset(&shape,0,sizeof(shape));
				SparkStagePackShapeBf16(&shape,1u,r == 0u ? 0u : 4096u,r == 1u ? 0u : 1024u);
				snprintf(tag,sizeof(tag),"E policy=%u dim=%u",policies[i],r);
				emit(tag,(long)SparkStagePackApplyShard(&shape,policies[i],degrees[3]));
			}
	}

	/* F: expected-layer masks across the stack (module's resolver context
	 * pinned to its representative codec/degree values). */
	{
		static const uint32_t mask_codecs[] = {SPARK_WEIGHT_CODEC_INT8,SPARK_WEIGHT_CODEC_MXFP4_E2M1};
		uint32_t mc;
		for (mc = 0; mc < (uint32_t)(sizeof(mask_codecs)/sizeof(mask_codecs[0])); mc++)
			for (i = 0; i <= SPARK_GLM52_MODEL_LAYER_COUNT + 2u; i++)
			{
				snprintf(tag,sizeof(tag),"F layer=%u codec=%u mask",i,mask_codecs[mc]);
				emit(tag,(long)SparkStagePackExpectedLayerMask(&mask_codecs[mc],resolve_shape,SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM,SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT,i));
			}
	}

	/* G: header TP identity getters. */
	{
		SparkGlm52StagePackHeader header;
		memset(&header,0,sizeof(header));
		header.reserved0 = 8u;
		header.reserved1 = 3u;
		emit("G degree",(long)SparkGlm52StagePackHeaderTpDegree(&header));
		emit("G rank",(long)SparkGlm52StagePackHeaderTpRank(&header));
		emit("G null degree",(long)SparkGlm52StagePackHeaderTpDegree(0));
	}

	printf("DONE glm52\n");
	return(0);
}
