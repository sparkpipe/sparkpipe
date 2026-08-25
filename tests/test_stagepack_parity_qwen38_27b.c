/*
 * Stage-pack parity harness, qwen36 dialect: exercises every header-exported
 * parse decision the loader chain consumes and prints one deterministic line
 * per case. Built twice by tests/test_stagepack_parity.py - once against the
 * frozen reference revision of spark_qwen38_27b_stagepack_format.h and once
 * against the live tree - and the two outputs must be byte-identical, which
 * is the differential proof that a collapse changed no accepted, received or
 * rejected stream and no error code.
 *
 * The validation replica below walks SparkQwen38_27bModuleValidateEntry's exact
 * decision order over the header surface (resolved shape, weight-format
 * classes, scale-group rule, declared byte counts, bounds, admission); the
 * module-local state it reads is pinned to representative values.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "spark_qwen38_27b_stagepack_format.h"

static void emit(const char *tag,long value)
{
	printf("%s %ld\n",tag,value);
}

/* The module's ValidateEntry sequence over the header surface. */
static uint32_t required_group_size(uint32_t weight_format)
{
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 )
		return(32u);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 || weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return(128u);
	return(0u);
}

static int validate_like_module(uint32_t tensor_kind,uint32_t layer_index,uint32_t weight_format,
	uint32_t rows,uint32_t columns,uint32_t scale_group_size,uint64_t payload_bytes,uint64_t scale_bytes,
	uint64_t payload_offset,uint64_t scale_offset,uint64_t file_bytes,uint32_t tp_degree)
{
	SparkQwen38_27bStagePackTensorShape shape;
	uint32_t global = layer_index == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	memset(&shape,0,sizeof(shape));
	if ( SparkQwen38_27bStagePackResolvedShape(tensor_kind,global != 0u ? 0u : layer_index,global,tp_degree,&shape) != 0 || rows != shape.rows || columns != shape.columns )
		return(1);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS )
	{
		if ( shape.quantizable == 0u )
			return(1);
		if ( (rows % 64u) != 0u || (columns % 128u) != 0u || scale_bytes != 0u || scale_group_size != 0u )
			return(1);
	}
	else if ( shape.quantizable != 0u )
	{
		if ( weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 && weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
			return(1);
	}
	else if ( weight_format != shape.natural_format )
		return(1);
	if ( weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? scale_group_size != 32u : ((weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 || weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128) ? scale_group_size != 128u : scale_group_size != 0u) )
		return(1);
	if ( weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS && (payload_bytes != SparkQwen38_27bStagePackPayloadBytes(weight_format,rows,columns) || scale_bytes != SparkQwen38_27bStagePackScaleBytes(weight_format,rows,columns)) )
		return(1);
	if ( payload_offset > file_bytes || payload_bytes > file_bytes - payload_offset )
		return(1);
	if ( scale_bytes != 0u && (scale_offset > file_bytes || scale_bytes > file_bytes - scale_offset) )
		return(1);
	if ( layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER || (global != 0u && (tensor_kind >= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC && tensor_kind <= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( 0u == 1u ) /* owns_final_head pinned true */
			return(1);
	}
	else if ( global == 0u && (layer_index < 0u || layer_index >= 0u + SPARK_QWEN38_27B_MODEL_LAYER_COUNT) )
		return(1);
	return(0);
}

int main(void)
{
	static const uint32_t layers[] = {SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER,SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER,0u,1u,2u,3u,7u,62u,63u,64u,100u,UINT32_MAX};
	static const uint32_t formats[] = {SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32,6u,7u,UINT32_MAX};
	static const uint32_t dims[] = {0u,1u,31u,32u,33u,47u,48u,63u,64u,127u,128u,129u,255u,256u,1023u,1024u,4095u,4096u,16384u,2147483647u,UINT32_MAX};
	uint32_t fmt_count = (uint32_t)(sizeof(formats)/sizeof(formats[0]));
	uint32_t dim_count = (uint32_t)(sizeof(dims)/sizeof(dims[0]));
	char tag[96];
	uint32_t kind,i,f,r,c,tp;

	/* A: resolved-shape refusal codes over the hostile grid. */
	for (kind = 0; kind < SPARK_QWEN38_27B_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
			for (tp = 1; tp <= 2; tp++)
			{
				SparkQwen38_27bStagePackTensorShape shape;
				uint32_t global = layers[i] == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
				int32_t code;
				memset(&shape,0,sizeof(shape));
				code = SparkQwen38_27bStagePackResolvedShape(kind,global != 0u ? 0u : layers[i],global,tp,&shape);
				snprintf(tag,sizeof(tag),"A kind=%u layer=%u tp=%u",kind,layers[i],tp);
				emit(tag,(long)code);
				/* rows/columns are read by the loader even on refusal (its
				 * debug print); the class/format fields are only consumed
				 * on success, so they ride behind the code gate. */
				emit("Ashape",(long)((uint64_t)shape.rows << 32 | shape.columns));
				if ( code == 0 )
				{
					emit("Afmt",(long)((uint64_t)shape.natural_format << 32 | shape.quantizable));
					emit("Aclass",(long)shape.layer_class);
				}
			}

	/* B: the full ValidateEntry-replica verdict grid. */
	for (kind = 0; kind < SPARK_QWEN38_27B_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
			for (f = 0; f < fmt_count; f++)
			{
				uint32_t bad;
				SparkQwen38_27bStagePackTensorShape shape;
				uint32_t global = layers[i] == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
				uint64_t pb,sb;
				memset(&shape,0,sizeof(shape));
				if ( SparkQwen38_27bStagePackResolvedShape(kind,global != 0u ? 0u : layers[i],global,1u,&shape) != 0 )
					continue;
				pb = SparkQwen38_27bStagePackPayloadBytes(formats[f],shape.rows,shape.columns);
				sb = SparkQwen38_27bStagePackScaleBytes(formats[f],shape.rows,shape.columns);
				snprintf(tag,sizeof(tag),"B base kind=%u layer=%u fmt=%u",kind,layers[i],formats[f]);
				emit(tag,(long)validate_like_module(kind,layers[i],formats[f],shape.rows,shape.columns,required_group_size(formats[f]),pb,sb,0u,0u,(uint64_t)pb + sb + 16u,1u));
				/* Field-class mutations off the valid baseline. */
				for (bad = 0; bad < 14u; bad++)
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
					case 4u: mrows = UINT32_MAX; break;
					case 5u: mcols = UINT32_MAX; break;
					case 6u: mpb += 1u; break;
					case 7u: msb += 1u; break;
					case 8u: msgs = 32u; break;
					case 9u: msgs = 128u; break;
					case 10u: mpo = (uint64_t)pb + sb + 16u; break; /* payload past EOF */
					case 11u: mso = (uint64_t)pb + sb + 17u; break; /* scale past EOF */
					case 12u: mso = 0u; break;                      /* zero-scale offset on scaled class */
					case 13u: mpo = UINT64_MAX; break;
					default: break;
					}
					verdict = validate_like_module(kind,layers[i],formats[f],mrows,mcols,msgs,mpb,msb,mpo,mso,(uint64_t)pb + sb + 16u,1u);
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
				emit(tag,(long)SparkQwen38_27bStagePackPayloadBytes(formats[f],dims[r],dims[c]));
				snprintf(tag,sizeof(tag),"C fmt=%u r=%u c=%u scale",f,dims[r],dims[c]);
				emit(tag,(long)SparkQwen38_27bStagePackScaleBytes(formats[f],dims[r],dims[c]));
			}

	/* D: computed slice inventories across the split space. */
	{
		uint32_t first,count;
		static const uint32_t firsts[] = {0u,1u,2u,15u,16u,31u,32u,33u,47u,48u,49u,61u,62u,63u,64u};
		for (i = 0; i < (uint32_t)(sizeof(firsts)/sizeof(firsts[0])); i++)
			for (count = 0; count <= SPARK_QWEN38_27B_MODEL_LAYER_COUNT + 1u; count++)
			{
				first = firsts[i];
				if ( first + count > SPARK_QWEN38_27B_MODEL_LAYER_COUNT + 1u )
					continue;
				snprintf(tag,sizeof(tag),"D first=%u count=%u",first,count);
				emit(tag,(long)SparkQwen38_27bStagePackExpectedTensorCount(first,count));
			}
	}

	/* E: geometry-compare codes and names under single-field hostility. */
	{
		SparkQwen38_27bStagePackHeader expected,file_header;
		uint32_t field,u32_count = (uint32_t)SPARK_QWEN38_27B_STAGEPACK_COMPARE_U32_FIELDS;
		static const uint32_t hostiles[] = {0u,1u,2u,UINT32_MAX};
		memset(&expected,0,sizeof(expected));
		SparkQwen38_27bStagePackExpectedGeometry(&expected,0u,SPARK_QWEN38_27B_MODEL_LAYER_COUNT);
		expected.directory_offset = SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES;
		expected.file_bytes = 4096u;
		for (field = 0; field < u32_count; field++)
		{
			uint32_t h;
			for (h = 0; h < (uint32_t)(sizeof(hostiles)/sizeof(hostiles[0])); h++)
			{
				int32_t code;
				file_header = expected;
				((uint32_t *)&file_header)[field] = hostiles[h];
				code = SparkQwen38_27bStagePackCompareGeometry(&file_header,&expected);
				snprintf(tag,sizeof(tag),"E field=%u hostile=%u",field,hostiles[h]);
				emit(tag,(long)code);
				printf("%s name=%s\n",tag,SparkQwen38_27bStagePackGeometryFieldName(code));
			}
		}
		emit("E clean",(long)SparkQwen38_27bStagePackCompareGeometry(&expected,&expected));
	}

	/* F: the TP shard policy over every kind and degree. */
	{
		static const uint32_t degrees[] = {1u,2u,3u,4u,7u,8u,16u,64u,256u};
		SparkQwen38_27bStagePackTensorShape shape;
		for (kind = 0; kind < SPARK_QWEN38_27B_STAGEPACK_TENSOR_KIND_COUNT; kind++)
			for (i = 0; i < (uint32_t)(sizeof(degrees)/sizeof(degrees[0])); i++)
			{
				uint32_t d;
				memset(&shape,0,sizeof(shape));
				if ( SparkQwen38_27bStagePackTensorShapeOf(kind,&shape) != 0 )
					continue;
				for (d = 0; d < 3u; d++)
				{
					shape.rows = 4096u >> d;
					shape.columns = 1024u << d;
					SparkQwen38_27bStagePackApplyTpShard(kind,degrees[i],&shape);
					snprintf(tag,sizeof(tag),"F kind=%u degree=%u rot=%u",kind,degrees[i],d);
					emit(tag,(long)((uint64_t)shape.rows << 32 | shape.columns));
				}
			}
	}

	/* G: period arithmetic behind the inventories. */
	for (i = 0; i <= 70u; i++)
	{
		snprintf(tag,sizeof(tag),"G below=%u",i);
		emit(tag,(long)SparkHybridStagePackFullAttentionLayersBelow(SPARK_QWEN38_27B_MODEL_ATTENTION_PERIOD,i));
	}

	printf("DONE qwen36\n");
	return(0);
}
