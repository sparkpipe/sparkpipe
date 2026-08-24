/*
 * Stage-pack parity harness, dsv4 dialect (Flash and Pro variants - the
 * variant rides SPARK_PARITY_DSV4_MODEL). Built twice per variant by
 * tests/test_stagepack_parity.py - frozen reference revision vs live tree -
 * and the outputs must be byte-identical. Walks the loader's decision order
 * (SparkDsv4ModuleValidateEntry consumes the header surface exactly as
 * replicated here; its TP>1 shard arithmetic is module-local and untouched
 * by the collapse, so this pin is taken at degree 1).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef SPARK_PARITY_DSV4_MODEL
#define SPARK_PARITY_DSV4_MODEL "sparkpipe/spark_dsv4_model.h"
#endif

#include SPARK_PARITY_DSV4_MODEL
#ifdef SPARK_PARITY_DSV4_PRO
/* The Pro variant renames onto the shared SPARK_DSV4_MODEL_* namespace via
 * the generated alias header, exactly like the generated contract TUs. */
#include "sparkpipe/spark_dsv4_pro_model_aliases.h"
#endif
#include "spark_dsv4_stagepack_format.h"

static void emit(const char *tag,long value)
{
	printf("%s %ld\n",tag,value);
}

/* The module's ValidateEntry sequence over the header surface, slice [0,N). */
static int validate_like_module(uint32_t tensor_kind,uint32_t layer_index,uint32_t weight_format,
	uint32_t rows,uint32_t columns,uint64_t payload_offset,uint64_t scale_offset,uint64_t file_bytes,
	uint32_t first_layer_index,uint32_t slice_count)
{
	SparkDsv4StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = (SparkDsv4StagePackLayerIsMtp(layer_index) != 0u && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u) || (layer_index >= first_layer_index && layer_index < first_layer_index + slice_count) ? 1u : 0u;
	memset(&shape,0,sizeof(shape));
	if ( tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(1);
	if ( SparkDsv4StagePackResolvedShape(tensor_kind,layer_index,global,&shape) < 0 )
		return(1);
	if ( shape.rows != rows || shape.columns != columns || shape.weight_format != weight_format )
		return(1);
	payload_bytes = SparkDsv4StagePackPayloadBytes(weight_format,rows,columns);
	scale_bytes = SparkDsv4StagePackScaleBytes(weight_format,rows,columns);
	if ( payload_offset + payload_bytes > file_bytes || (scale_bytes != 0u && (scale_offset != payload_offset + payload_bytes || scale_offset + scale_bytes > file_bytes)) )
		return(1);
	return(0);
}

int main(void)
{
	static const uint32_t dims[] = {0u,1u,31u,32u,33u,63u,64u,127u,128u,129u,255u,256u,1024u,4096u,16384u,2147483647u,UINT32_MAX};
	static const uint32_t formats[] = {SPARK_DSV4_STAGEPACK_WEIGHT_BF16,SPARK_DSV4_STAGEPACK_WEIGHT_F32,SPARK_DSV4_STAGEPACK_WEIGHT_U32,SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1,SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3,5u,6u,UINT32_MAX};
	uint32_t dim_count = (uint32_t)(sizeof(dims)/sizeof(dims[0]));
	uint32_t fmt_count = (uint32_t)(sizeof(formats)/sizeof(formats[0]));
	char tag[96];
	uint32_t kind,i,f,r,c;

	/* A: resolved-shape refusal codes over the hostile grid. */
	for (kind = 0; kind < SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
	{
		static const uint32_t layers[] = {SPARK_DSV4_STAGEPACK_GLOBAL_LAYER,0u,1u,2u,20u,40u,42u,43u,60u,61u,62u,UINT32_MAX};
		for (i = 0; i < (uint32_t)(sizeof(layers)/sizeof(layers[0])); i++)
		{
			SparkDsv4StagePackTensorShape shape;
			uint32_t global = layers[i] == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
			int32_t code;
			memset(&shape,0,sizeof(shape));
			code = SparkDsv4StagePackResolvedShape(kind,layers[i],global,&shape);
			snprintf(tag,sizeof(tag),"A kind=%u layer=%u",kind,layers[i]);
			emit(tag,(long)code);
			/* rows/columns stay zero through a refusal on both revisions;
			 * format/class are only consumed on success. */
			emit("Ashape",(long)((uint64_t)shape.rows << 32 | shape.columns));
			if ( code >= 0 )
				emit("Afmt",(long)((uint64_t)shape.weight_format << 32 | shape.layer_class));
		}
	}

	/* B: the ValidateEntry-replica verdict grid over every kind and layer. */
	for (kind = 0; kind < SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT + 2u; kind++)
	{
		uint32_t layer;
		for (layer = 0; layer <= SPARK_DSV4_MODEL_LAYER_COUNT; layer++)
		{
			SparkDsv4StagePackTensorShape shape;
			uint32_t global = layer == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
			uint32_t bad;
			memset(&shape,0,sizeof(shape));
			if ( SparkDsv4StagePackResolvedShape(kind,layer,global,&shape) < 0 )
				continue;
			for (f = 0; f < fmt_count; f++)
			{
				/* Strict natural-format family: only the natural format can validate. */
				uint64_t pb,sb;
				int verdict;
				if ( formats[f] != shape.weight_format )
					continue;
				pb = SparkDsv4StagePackPayloadBytes(formats[f],shape.rows,shape.columns);
				sb = SparkDsv4StagePackScaleBytes(formats[f],shape.rows,shape.columns);
				verdict = validate_like_module(kind,layer,formats[f],shape.rows,shape.columns,0u,pb,(uint64_t)pb + sb + 16u,0u,SPARK_DSV4_MODEL_LAYER_COUNT);
				snprintf(tag,sizeof(tag),"B base kind=%u layer=%u fmt=%u",kind,layer,formats[f]);
				emit(tag,(long)verdict);
				for (bad = 0; bad < 10u; bad++)
				{
					uint32_t mrows = shape.rows,mcols = shape.columns,mwf = formats[f];
					uint64_t mpo = 0u,mso = pb;
					switch ( bad )
					{
					case 0u: mrows += 1u; break;
					case 1u: mcols += 1u; break;
					case 2u: mrows = 0u; break;
					case 3u: mcols = 0u; break;
					case 4u: mso = pb + 1u; break; /* gap between payload and scale */
					case 5u: mso = 0u; break;      /* overlap */
					case 6u: mpo = UINT64_MAX; break;
					case 7u: mrows = UINT32_MAX; break;
					case 8u: mwf = UINT32_MAX; break;
					case 9u: mwf = SPARK_DSV4_STAGEPACK_WEIGHT_BF16; break;
					default: break;
					}
					verdict = validate_like_module(kind,layer,mwf,mrows,mcols,mpo,mso,(uint64_t)pb + sb + 16u,0u,SPARK_DSV4_MODEL_LAYER_COUNT);
					snprintf(tag,sizeof(tag),"B mut kind=%u layer=%u fmt=%u bad=%u",kind,layer,formats[f],bad);
					emit(tag,(long)verdict);
				}
			}
		}
	}

	/* C: raw byte accounting over the dimension grid, all formats. */
	for (f = 0; f < fmt_count; f++)
		for (r = 0; r < dim_count; r++)
			for (c = 0; c < dim_count; c++)
			{
				snprintf(tag,sizeof(tag),"C fmt=%u r=%u c=%u payload",f,dims[r],dims[c]);
				emit(tag,(long)SparkDsv4StagePackPayloadBytes(formats[f],dims[r],dims[c]));
				snprintf(tag,sizeof(tag),"C fmt=%u r=%u c=%u scale",f,dims[r],dims[c]);
				emit(tag,(long)SparkDsv4StagePackScaleBytes(formats[f],dims[r],dims[c]));
			}

	/* D: computed inventories across the split and ownership space. */
	{
		uint32_t first,count,own_e,own_f;
		static const uint32_t firsts[] = {0u,1u,2u,10u,20u,39u,40u,41u,42u,43u,60u,61u,62u};
		for (i = 0; i < (uint32_t)(sizeof(firsts)/sizeof(firsts[0])); i++)
			for (count = 0; count <= SPARK_DSV4_MODEL_LAYER_COUNT; count++)
			{
				first = firsts[i];
				if ( first + count > SPARK_DSV4_MODEL_LAYER_COUNT + 1u || first >= SPARK_DSV4_MODEL_LAYER_COUNT + 1u )
					continue;
				for (own_e = 0; own_e <= 1u; own_e++)
					for (own_f = 0; own_f <= 1u; own_f++)
					{
						snprintf(tag,sizeof(tag),"D first=%u count=%u emb=%u fin=%u",first,count,own_e,own_f);
						emit(tag,(long)SparkDsv4StagePackExpectedTensorCountForOwnership(first,count,own_e,own_f));
					}
				snprintf(tag,sizeof(tag),"D plain first=%u count=%u",first,count);
				emit(tag,(long)SparkDsv4StagePackExpectedTensorCount(first,count));
			}
	}

	/* E: geometry-compare codes and names under single-field hostility. */
	{
		SparkDsv4StagePackHeader expected,file_header;
		uint32_t field;
		static const uint32_t hostiles[] = {0u,1u,2u,UINT32_MAX};
		SparkDsv4StagePackExpectedGeometry(&expected,0u,SPARK_DSV4_MODEL_LAYER_COUNT);
		expected.directory_offset = SPARK_DSV4_STAGEPACK_HEADER_BYTES;
		expected.file_bytes = 4096u;
		expected.tensor_count = SparkDsv4StagePackExpectedTensorCount(0u,SPARK_DSV4_MODEL_LAYER_COUNT);
		for (field = 0; field < 16u; field++)
		{
			uint32_t h;
			for (h = 0; h < (uint32_t)(sizeof(hostiles)/sizeof(hostiles[0])); h++)
			{
				int32_t code;
				file_header = expected;
				((uint32_t *)&file_header)[field] = hostiles[h];
				code = SparkDsv4StagePackCompareGeometry(&file_header,&expected);
				snprintf(tag,sizeof(tag),"E field=%u hostile=%u",field,hostiles[h]);
				emit(tag,(long)code);
				printf("%s name=%s\n",tag,SparkDsv4StagePackGeometryFieldName(code));
			}
		}
		emit("E clean",(long)SparkDsv4StagePackCompareGeometry(&expected,&expected));
	}

	/* F: layer-kind helpers across the whole stack plus the MTP range. */
	for (i = 0; i <= SPARK_DSV4_MODEL_LAYER_COUNT + 8u; i++)
	{
		snprintf(tag,sizeof(tag),"F layer=%u kind",i);
		emit(tag,(long)SparkDsv4StagePackLayerKind(i));
		emit("F hashrouted",(long)SparkDsv4StagePackLayerIsHashRouted(i));
		emit("F ismtp",(long)SparkDsv4StagePackLayerIsMtp(i));
	}

	printf("DONE dsv4\n");
	return(0);
}
