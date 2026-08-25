/*
 * Stage-pack parity harness, qwen38 dialect. Built twice by
 * tests/test_stagepack_parity.py - frozen reference revision vs live tree -
 * and the outputs must be byte-identical. Walks SparkQwen38ModuleValidateEntry's
 * decision order over the header surface with module-local state pinned.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "spark_qwen38_max_stagepack_format.h"

static void emit(const char *tag,long value)
{
	printf("%s %ld\n",tag,value);
}

static int validate_like_module(uint32_t tensor_kind,uint32_t layer_index,uint32_t weight_format,
	uint32_t rows,uint32_t columns,uint32_t scale_group_size,uint64_t payload_bytes,uint64_t scale_bytes,
	uint64_t payload_offset,uint64_t scale_offset,uint64_t file_bytes)
{
	SparkQwen38StagePackTensorShape shape;
	uint32_t global = layer_index == SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	memset(&shape,0,sizeof(shape));
	if ( SparkQwen38StagePackResolvedShape(tensor_kind,global != 0u ? 0u : layer_index,global,&shape) != 0 || rows != shape.rows || columns != shape.columns )
		return(1);
	/* Strict natural format, except the three routed-expert tensors may also arrive BF16. */
	if ( weight_format != shape.natural_format )
	{
		if ( (shape.natural_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && shape.natural_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128) || weight_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
			return(1);
	}
	if ( SparkHybridStagePackScaleGroupSizeOk(SparkQwen38StagePackWeightClass(weight_format),scale_group_size) == 0u )
		return(1);
	if ( payload_bytes != SparkQwen38StagePackPayloadBytes(weight_format,rows,columns) || scale_bytes != SparkQwen38StagePackScaleBytes(weight_format,rows,columns) )
		return(1);
	if ( payload_offset > file_bytes || payload_bytes > file_bytes - payload_offset )
		return(1);
	if ( scale_bytes != 0u && (scale_offset > file_bytes || scale_bytes > file_bytes - scale_offset) )
		return(1);
	if ( layer_index == SPARK_QWEN38_STAGEPACK_MTP_LAYER || (global != 0u && (tensor_kind >= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC && tensor_kind <= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( 0u == 1u ) /* owns_final_head pinned true */
			return(1);
	}
	else if ( global == 0u && (layer_index < 0u || layer_index >= 0u + SPARK_QWEN38_MODEL_LAYER_COUNT) )
		return(1);
	return(0);
}

int main(void)
{
	static const uint32_t layers[] = {SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER,SPARK_QWEN38_STAGEPACK_MTP_LAYER,0u,1u,2u,3u,7u,30u,31u,32u,100u,UINT32_MAX};
	static const uint32_t formats[] = {SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32,SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32,5u,6u,UINT32_MAX};
	static const uint32_t dims[] = {0u,1u,31u,32u,33u,63u,64u,127u,128u,129u,255u,256u,1024u,4096u,16384u,2147483647u,UINT32_MAX};
	uint32_t fmt_count = (uint32_t)(sizeof(formats)/sizeof(formats[0]));
	uint32_t dim_count = (uint32_t)(sizeof(dims)/sizeof(dims[0]));
	char tag[96];
	uint32_t kind,i,f,r,c;

	/* A: resolved-shape refusal codes over the hostile grid. */
	for (kind = 0; kind < SPARK_QWEN38_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
		{
			SparkQwen38StagePackTensorShape shape;
			uint32_t global = layers[i] == SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
			int32_t code;
			memset(&shape,0,sizeof(shape));
			code = SparkQwen38StagePackResolvedShape(kind,global != 0u ? 0u : layers[i],global,&shape);
			snprintf(tag,sizeof(tag),"A kind=%u layer=%u",kind,layers[i]);
			emit(tag,(long)code);
			emit("Ashape",(long)((uint64_t)shape.rows << 32 | shape.columns));
			emit("Afmt",(long)((uint64_t)shape.natural_format << 32 | shape.layer_class));
		}

	/* B: the full ValidateEntry-replica verdict grid. */
	for (kind = 0; kind < SPARK_QWEN38_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
			for (f = 0; f < fmt_count; f++)
			{
				uint32_t bad;
				SparkQwen38StagePackTensorShape shape;
				uint32_t global = layers[i] == SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
				uint64_t pb,sb;
				memset(&shape,0,sizeof(shape));
				if ( SparkQwen38StagePackResolvedShape(kind,global != 0u ? 0u : layers[i],global,&shape) != 0 )
					continue;
				pb = SparkQwen38StagePackPayloadBytes(formats[f],shape.rows,shape.columns);
				sb = SparkQwen38StagePackScaleBytes(formats[f],shape.rows,shape.columns);
				snprintf(tag,sizeof(tag),"B base kind=%u layer=%u fmt=%u",kind,layers[i],formats[f]);
				emit(tag,(long)validate_like_module(kind,layers[i],formats[f],shape.rows,shape.columns,0u,pb,sb,0u,0u,(uint64_t)pb + sb + 16u));
				for (bad = 0; bad < 13u; bad++)
				{
					uint32_t mrows = shape.rows,mcols = shape.columns,msgs = 0u;
					uint64_t mpb = pb,msb = sb,mpo = 0u,mso = sb != 0u ? pb : 0u;
					int verdict;
					switch ( bad )
					{
					case 0u: mrows = shape.rows + 1u; break;
					case 1u: mcols = shape.columns + 1u; break;
					case 2u: mrows = 0u; break;
					case 3u: mcols = 0u; break;
					case 4u: mpb += 1u; break;
					case 5u: msb += 1u; break;
					case 6u: msgs = 32u; break;
					case 7u: msgs = 128u; break;
					case 8u: mpo = (uint64_t)pb + sb + 16u; break;
					case 9u: mso = (uint64_t)pb + sb + 17u; break;
					case 10u: mso = 0u; break;
					case 11u: mpo = UINT64_MAX; break;
					case 12u: mrows = UINT32_MAX; break;
					default: break;
					}
					verdict = validate_like_module(kind,layers[i],formats[f],mrows,mcols,msgs,mpb,msb,mpo,mso,(uint64_t)pb + sb + 16u);
					snprintf(tag,sizeof(tag),"B mut kind=%u layer=%u fmt=%u bad=%u",kind,layers[i],formats[f],bad);
					emit(tag,(long)verdict);
				}
			}

	/* C: raw byte accounting over the dimension grid, all formats. */
	for (f = 0; f < fmt_count; f++)
		for (r = 0; r < dim_count; r++)
			for (c = 0; c < dim_count; c++)
			{
				snprintf(tag,sizeof(tag),"C fmt=%u r=%u c=%u payload",f,dims[r],dims[c]);
				emit(tag,(long)SparkQwen38StagePackPayloadBytes(formats[f],dims[r],dims[c]));
				snprintf(tag,sizeof(tag),"C fmt=%u r=%u c=%u scale",f,dims[r],dims[c]);
				emit(tag,(long)SparkQwen38StagePackScaleBytes(formats[f],dims[r],dims[c]));
			}

	/* D: computed slice inventories across the split space. */
	{
		uint32_t first,count;
		static const uint32_t firsts[] = {0u,1u,2u,15u,16u,31u,32u,45u,46u,47u,48u,49u,50u,51u,52u};
		for (i = 0; i < (uint32_t)(sizeof(firsts)/sizeof(firsts[0])); i++)
			for (count = 0; count <= SPARK_QWEN38_MODEL_LAYER_COUNT + 1u; count++)
			{
				first = firsts[i];
				if ( first > 60u )
					continue;
				snprintf(tag,sizeof(tag),"D first=%u count=%u",first,count);
				emit(tag,(long)SparkQwen38StagePackExpectedTensorCount(first,count));
			}
	}

	/* E: header-compare codes and names under single-field hostility. */
	{
		SparkQwen38StagePackHeader expected,file_header;
		uint32_t field,u32_count = (uint32_t)SPARK_QWEN38_STAGEPACK_COMPARE_U32_FIELDS;
		static const uint32_t hostiles[] = {0u,1u,2u,UINT32_MAX};
		memset(&expected,0,sizeof(expected));
		SparkQwen38StagePackExpectedGeometry(&expected,0u,SPARK_QWEN38_MODEL_LAYER_COUNT);
		expected.directory_offset = SPARK_QWEN38_STAGEPACK_HEADER_BYTES;
		expected.file_bytes = 4096u;
		for (field = 0; field < u32_count; field++)
		{
			uint32_t h;
			for (h = 0; h < (uint32_t)(sizeof(hostiles)/sizeof(hostiles[0])); h++)
			{
				int32_t code;
				file_header = expected;
				((uint32_t *)&file_header)[field] = hostiles[h];
				code = SparkQwen38StagePackHeaderMatches(&file_header,&expected);
				snprintf(tag,sizeof(tag),"E field=%u hostile=%u",field,hostiles[h]);
				emit(tag,(long)code);
			}
		}
		emit("E clean",(long)SparkQwen38StagePackHeaderMatches(&expected,&expected));
	}

	/* G: period arithmetic behind the inventories. */
	for (i = 0; i <= 70u; i++)
	{
		snprintf(tag,sizeof(tag),"G below=%u",i);
		emit(tag,(long)SparkHybridStagePackFullAttentionLayersBelow(SPARK_QWEN38_MODEL_ATTENTION_PERIOD,i));
	}

	printf("DONE qwen38\n");
	return(0);
}
