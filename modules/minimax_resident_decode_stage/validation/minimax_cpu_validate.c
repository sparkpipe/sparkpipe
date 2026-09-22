#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_minimax_model.h"
#include "spark_minimax_stagepack_format.h"

typedef struct SparkMinimaxModuleState SparkMinimaxModuleState;

#define TOOL_HIDDEN SPARK_MINIMAX_TEXT_HIDDEN_DIMENSION
#define TOOL_HEAD_DIM SPARK_MINIMAX_TEXT_HEAD_DIMENSION
#define TOOL_LOCAL_Q_HEADS 16u
#define TOOL_LOCAL_KV_HEADS 2u
#define TOOL_LOCAL_Q_ROWS (TOOL_LOCAL_Q_HEADS * TOOL_HEAD_DIM)
#define TOOL_LOCAL_KV_ROWS (TOOL_LOCAL_KV_HEADS * TOOL_HEAD_DIM)
#define TOOL_MAX_POSITIONS 64u
#define TOOL_FFN_SHARD (SPARK_MINIMAX_TEXT_DENSE_INTERMEDIATE_DIMENSION / 4u)
#define TOOL_ATTN_GROUP (SPARK_MINIMAX_TEXT_ATTENTION_HEAD_COUNT / SPARK_MINIMAX_TEXT_KV_HEAD_COUNT)

SparkStatus SparkMinimaxModuleLoadMappedPack(SparkMinimaxModuleState *state,const char *path,uint32_t resident,uint32_t *mapped_flag);
SparkStatus SparkMinimaxModuleValidationView(const SparkMinimaxModuleState *state,uint32_t kind,uint32_t layer,const void **payload,uint32_t *rows,uint32_t *columns);
void *SparkMinimaxModuleStateAllocate(void);
void SparkMinimaxModuleStateFree(void *state);
void SparkMinimaxModuleStateConfigureTp(void *state,uint32_t tp_degree,uint32_t tp_rank);

static float ToolBf16(uint16_t value)
{
	uint32_t expanded = (uint32_t)value << 16;
	float result;
	memcpy(&result,&expanded,sizeof(result));
	return(result);
}

static uint16_t ToolRoundBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits = bits + 0x7fffu + ((bits >> 16) & 1u);
	return((uint16_t)(bits >> 16));
}

static float ToolHeadVariance(const uint16_t *values,uint32_t count)
{
	float summed = 0.0f;
	for (uint32_t index = 0u; index < count; index++)
	{
		float value = ToolBf16(values[index]);
		summed += value * value;
	}
	return(summed / (float)count);
}

static void ToolRope(float *head,float position)
{
	for (uint32_t half = 0u; half < TOOL_HEAD_DIM / 2u; half++)
	{
		float real = head[half];
		float imag = head[half + TOOL_HEAD_DIM / 2u];
		float angle = position * powf(5000000.0f,-2.0f * (float)half / (float)TOOL_HEAD_DIM);
		float cosine = cosf(angle);
		float sine = sinf(angle);
		head[half] = real * cosine - imag * sine;
		head[half + TOOL_HEAD_DIM / 2u] = imag * cosine + real * sine;
	}
}

static void ToolMatvec(float *output,const uint16_t *weight,uint32_t rows,uint32_t columns,const float *input)
{
	for (uint32_t row = 0u; row < rows; row++)
	{
		float summed = 0.0f;
		const uint16_t *row_values = weight + (uint64_t)row * columns;
		for (uint32_t column = 0u; column < columns; column++)
			summed += ToolBf16(row_values[column]) * input[column];
		output[row] = summed;
	}
}

static uint64_t ToolPeakRssBytes(void)
{
	struct rusage usage;
	if ( getrusage(RUSAGE_SELF,&usage) != 0 )
		return(0u);
	return((uint64_t)usage.ru_maxrss * 1024u);
}

static SparkStatus ToolLoadRankPack(SparkMinimaxModuleState **state,const char *path,uint32_t tp_rank)
{
	uint32_t mapped = 0u;
	SparkStatus status;
	*state = SparkMinimaxModuleStateAllocate();
	if ( *state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	SparkMinimaxModuleStateConfigureTp(*state,4u,tp_rank);
	status = SparkMinimaxModuleLoadMappedPack(*state,path,0u,&mapped);
	if ( status != SPARK_STATUS_OK )
	{
		SparkMinimaxModuleStateFree(*state);
		*state = 0;
	}
	return(status);
}

static uint16_t *ToolReadTokenFile(const char *path,uint32_t *token_count)
{
	FILE *file = fopen(path,"rb");
	uint32_t count = 0u;
	uint16_t *streams = 0;
	if ( file == 0 || fread(&count,sizeof(count),1u,file) != 1u ||
		count == 0u || count > TOOL_MAX_POSITIONS )
	{
		fprintf(stderr,"minimax_cpu_validate: token file invalid: %s\n",path);
		if ( file != 0 )
			fclose(file);
		return(0);
	}
	*token_count = count;
	streams = (uint16_t *)malloc((size_t)count * TOOL_HIDDEN * sizeof(uint16_t));
	if ( streams == 0 || fread(streams,sizeof(uint16_t),(size_t)count * TOOL_HIDDEN,file) != (size_t)count * TOOL_HIDDEN )
	{
		fprintf(stderr,"minimax_cpu_validate: token file truncated: %s\n",path);
		free(streams);
		fclose(file);
		return(0);
	}
	fclose(file);
	return(streams);
}

static int ToolModePackLoad(const char *pack_path,uint32_t tp_rank)
{
	SparkMinimaxModuleState *state = 0;
	SparkStatus status = ToolLoadRankPack(&state,pack_path,tp_rank);
	fprintf(stderr,"minimax_cpu_validate: pack-load %s result=%d peak_rss=%llu\n",
		pack_path,(int)status,(unsigned long long)ToolPeakRssBytes());
	if ( state != 0 )
		SparkMinimaxModuleStateFree(state);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static uint16_t *ToolNormalizedEmbedding(SparkMinimaxModuleState *state,const uint32_t *token_ids,uint32_t position)
{
	const uint16_t *embedding = 0;
	const uint16_t *input_norm = 0;
	uint16_t *normalized;
	uint32_t columns = 0u;
	float variance;
	if ( SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING,0,(const void **)&embedding,0,&columns) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM,0,(const void **)&input_norm,0,&columns) != SPARK_STATUS_OK )
		return(0);
	normalized = (uint16_t *)malloc(TOOL_HIDDEN * sizeof(uint16_t));
	if ( normalized == 0 )
		return(0);
	variance = ToolHeadVariance(embedding + (uint64_t)token_ids[position] * TOOL_HIDDEN,TOOL_HIDDEN);
	for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
	{
		float value = ToolBf16(embedding[(uint64_t)token_ids[position] * TOOL_HIDDEN + column]);
		normalized[column] = ToolRoundBf16(value / sqrtf(variance + SPARK_MINIMAX_TEXT_RMS_NORM_EPSILON) * ToolBf16(input_norm[column]));
	}
	return(normalized);
}

static uint16_t *ToolRawEmbedding(SparkMinimaxModuleState *state,const uint32_t *token_ids,uint32_t position)
{
	const uint16_t *embedding = 0;
	uint16_t *raw;
	uint32_t columns = 0u;
	if ( SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING,0,(const void **)&embedding,0,&columns) != SPARK_STATUS_OK )
		return(0);
	raw = (uint16_t *)malloc(TOOL_HIDDEN * sizeof(uint16_t));
	if ( raw == 0 )
		return(0);
	memcpy(raw,embedding + (uint64_t)token_ids[position] * TOOL_HIDDEN,TOOL_HIDDEN * sizeof(uint16_t));
	return(raw);
}

static int ToolModeAttentionRank(const char *pack_path,uint32_t tp_rank,const char *positions_path,const char *output_path)
{
	SparkMinimaxModuleState *state = 0;
	const uint16_t *q_norm = 0;
	const uint16_t *k_norm = 0;
	const uint16_t *query_weight = 0;
	const uint16_t *key_weight = 0;
	const uint16_t *value_weight = 0;
	const uint16_t *output_weight = 0;
	uint32_t query_rows = 0u,query_columns = 0u;
	uint32_t token_count = 0u;
	uint32_t *token_ids = 0;
	float (*attn_partials)[TOOL_HIDDEN];
	FILE *positions;
	FILE *output;
	float cache_keys[TOOL_MAX_POSITIONS][TOOL_LOCAL_KV_ROWS];
	float cache_values[TOOL_MAX_POSITIONS][TOOL_LOCAL_KV_ROWS];
	if ( ToolLoadRankPack(&state,pack_path,tp_rank) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: pack load failed: %s\n",pack_path);
		return(1);
	}
	if ( SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_Q_NORM,0,(const void **)&q_norm,0,0) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_K_NORM,0,(const void **)&k_norm,0,0) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY,0,(const void **)&query_weight,&query_rows,&query_columns) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_KEY,0,(const void **)&key_weight,0,0) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE,0,(const void **)&value_weight,0,0) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT,0,(const void **)&output_weight,0,0) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: rank %u attention views incomplete\n",tp_rank);
		return(1);
	}
	positions = fopen(positions_path,"rb");
	if ( positions == 0 || fread(&token_count,sizeof(token_count),1u,positions) != 1u ||
		token_count == 0u || token_count > TOOL_MAX_POSITIONS )
	{
		fprintf(stderr,"minimax_cpu_validate: positions file invalid: %s\n",positions_path);
		return(1);
	}
	token_ids = (uint32_t *)malloc((size_t)token_count * sizeof(uint32_t));
	attn_partials = (float (*)[TOOL_HIDDEN])calloc((size_t)token_count,sizeof(*attn_partials));
	if ( token_ids == 0 || attn_partials == 0 || fread(token_ids,sizeof(uint32_t),(size_t)token_count,positions) != (size_t)token_count )
	{
		fprintf(stderr,"minimax_cpu_validate: positions file truncated: %s\n",positions_path);
		return(1);
	}
	fclose(positions);
	for (uint32_t position = 0u; position < token_count; position++)
	{
		uint16_t *x_rounded = ToolNormalizedEmbedding(state,token_ids,position);
		float x[TOOL_HIDDEN];
		float raw[TOOL_LOCAL_Q_ROWS];
		uint16_t rounded[TOOL_LOCAL_Q_ROWS];
		float heads[TOOL_LOCAL_Q_HEADS][TOOL_HEAD_DIM];
		float k_raw[TOOL_LOCAL_KV_ROWS];
		float v_raw[TOOL_LOCAL_KV_ROWS];
		uint16_t k_rounded[TOOL_LOCAL_KV_ROWS];
		uint16_t v_rounded[TOOL_LOCAL_KV_ROWS];
		float o_partial[TOOL_HIDDEN];
		if ( x_rounded == 0 )
			return(1);
		for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
			x[column] = ToolBf16(x_rounded[column]);
		free(x_rounded);
		ToolMatvec(raw,query_weight,query_rows,query_columns,x);
		for (uint32_t element = 0u; element < query_rows; element++)
			rounded[element] = ToolRoundBf16(raw[element]);
		for (uint32_t head = 0u; head < TOOL_LOCAL_Q_HEADS; head++)
		{
			float head_variance = ToolHeadVariance(rounded + (uint64_t)head * TOOL_HEAD_DIM,TOOL_HEAD_DIM);
			for (uint32_t d = 0u; d < TOOL_HEAD_DIM; d++)
				heads[head][d] = ToolBf16(rounded[(uint64_t)head * TOOL_HEAD_DIM + d]) / sqrtf(head_variance + SPARK_MINIMAX_TEXT_RMS_NORM_EPSILON) * ToolBf16(q_norm[d]);
			ToolRope(heads[head],(float)position);
		}
		ToolMatvec(k_raw,key_weight,TOOL_LOCAL_KV_ROWS,query_columns,x);
		ToolMatvec(v_raw,value_weight,TOOL_LOCAL_KV_ROWS,query_columns,x);
		for (uint32_t element = 0u; element < TOOL_LOCAL_KV_ROWS; element++)
		{
			k_rounded[element] = ToolRoundBf16(k_raw[element]);
			v_rounded[element] = ToolRoundBf16(v_raw[element]);
		}
		for (uint32_t kv_head = 0u; kv_head < TOOL_LOCAL_KV_HEADS; kv_head++)
		{
			float head_variance = ToolHeadVariance(k_rounded + (uint64_t)kv_head * TOOL_HEAD_DIM,TOOL_HEAD_DIM);
			float k_head[TOOL_HEAD_DIM];
			for (uint32_t d = 0u; d < TOOL_HEAD_DIM; d++)
				k_head[d] = ToolBf16(k_rounded[(uint64_t)kv_head * TOOL_HEAD_DIM + d]) / sqrtf(head_variance + SPARK_MINIMAX_TEXT_RMS_NORM_EPSILON) * ToolBf16(k_norm[d]);
			ToolRope(k_head,(float)position);
			for (uint32_t d = 0u; d < TOOL_HEAD_DIM; d++)
			{
				cache_keys[position][(uint64_t)kv_head * TOOL_HEAD_DIM + d] = ToolBf16(ToolRoundBf16(k_head[d]));
				cache_values[position][(uint64_t)kv_head * TOOL_HEAD_DIM + d] = ToolBf16(v_rounded[(uint64_t)kv_head * TOOL_HEAD_DIM + d]);
			}
		}
		for (uint32_t head = 0u; head < TOOL_LOCAL_Q_HEADS; head++)
		{
			uint32_t kv_head = head / TOOL_ATTN_GROUP;
			float scores[TOOL_MAX_POSITIONS];
			float maximum = -3.402823466e38f;
			float denominator = 0.0f;
			for (uint32_t past = 0u; past <= position; past++)
			{
				float score = 0.0f;
				for (uint32_t d = 0u; d < TOOL_HEAD_DIM; d++)
					score += cache_keys[past][(uint64_t)kv_head * TOOL_HEAD_DIM + d] * heads[head][d];
				scores[past] = score / sqrtf((float)TOOL_HEAD_DIM);
				if ( scores[past] > maximum )
					maximum = scores[past];
			}
			for (uint32_t past = 0u; past <= position; past++)
			{
				scores[past] = expf(scores[past] - maximum);
				denominator += scores[past];
			}
			for (uint32_t d = 0u; d < TOOL_HEAD_DIM; d++)
			{
				float accumulated = 0.0f;
				for (uint32_t past = 0u; past <= position; past++)
					accumulated += scores[past] / denominator * cache_values[past][(uint64_t)kv_head * TOOL_HEAD_DIM + d];
				heads[head][d] = ToolBf16(ToolRoundBf16(accumulated));
			}
		}
		ToolMatvec(o_partial,output_weight,TOOL_HIDDEN,query_rows,heads[0]);
		for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
			attn_partials[position][column] = o_partial[column];
	}
	output = fopen(output_path,"wb");
	if ( output == 0 || fwrite(attn_partials,sizeof(*attn_partials),(size_t)token_count,output) != (size_t)token_count )
	{
		fprintf(stderr,"minimax_cpu_validate: attention partial write failed: %s\n",output_path);
		return(1);
	}
	fclose(output);
	fprintf(stderr,"minimax_cpu_validate: att-rank rank=%u positions=%u peak_rss=%llu\n",
		tp_rank,token_count,(unsigned long long)ToolPeakRssBytes());
	free(token_ids);
	free(attn_partials);
	SparkMinimaxModuleStateFree(state);
	return(0);
}

static int ToolModeFfnRank(const char *pack_path,uint32_t tp_rank,const char *streams_path,const char *output_path)
{
	SparkMinimaxModuleState *state = 0;
	const uint16_t *post_norm = 0;
	const uint16_t *gate_weight = 0;
	const uint16_t *up_weight = 0;
	const uint16_t *down_weight = 0;
	uint32_t ffn_rows = 0u,ffn_columns = 0u,columns = 0u;
	uint32_t token_count = 0u;
	uint16_t *streams = 0;
	float (*down_partials)[TOOL_HIDDEN];
	FILE *output;
	streams = ToolReadTokenFile(streams_path,&token_count);
	if ( streams == 0 )
		return(1);
	if ( ToolLoadRankPack(&state,pack_path,tp_rank) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: pack load failed: %s\n",pack_path);
		return(1);
	}
	if ( SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_POST_ATTENTION_NORM,0,(const void **)&post_norm,0,&columns) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE,0,(const void **)&gate_weight,&ffn_rows,&ffn_columns) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP,0,(const void **)&up_weight,0,0) != SPARK_STATUS_OK ||
		SparkMinimaxModuleValidationView(state,SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN,0,(const void **)&down_weight,0,0) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: rank %u mlp views incomplete\n",tp_rank);
		return(1);
	}
	down_partials = (float (*)[TOOL_HIDDEN])calloc((size_t)token_count,sizeof(*down_partials));
	if ( down_partials == 0 )
		return(1);
	for (uint32_t position = 0u; position < token_count; position++)
	{
		float streams_f32[TOOL_HIDDEN];
		float normalized[TOOL_HIDDEN];
		for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
			streams_f32[column] = ToolBf16(streams[(uint64_t)position * TOOL_HIDDEN + column]);
		{
			float post_variance = 0.0f;
			for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
				post_variance += streams_f32[column] * streams_f32[column];
			post_variance /= (float)TOOL_HIDDEN;
			for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
				normalized[column] = ToolBf16(ToolRoundBf16(streams_f32[column] / sqrtf(post_variance + SPARK_MINIMAX_TEXT_RMS_NORM_EPSILON) * ToolBf16(post_norm[column])));
		}
		{
			float gate_raw[TOOL_FFN_SHARD];
			float up_raw[TOOL_FFN_SHARD];
			float activated[TOOL_FFN_SHARD];
			float down_partial[TOOL_HIDDEN];
			ToolMatvec(gate_raw,gate_weight,ffn_rows,ffn_columns,normalized);
			ToolMatvec(up_raw,up_weight,ffn_rows,ffn_columns,normalized);
			for (uint32_t element = 0u; element < ffn_rows; element++)
			{
				float gate = ToolBf16(ToolRoundBf16(gate_raw[element]));
				float up = ToolBf16(ToolRoundBf16(up_raw[element]));
				activated[element] = ToolBf16(ToolRoundBf16(gate / (1.0f + expf(-gate)) * up));
			}
			ToolMatvec(down_partial,down_weight,TOOL_HIDDEN,ffn_rows,activated);
			for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
				down_partials[position][column] = down_partial[column];
		}
	}
	output = fopen(output_path,"wb");
	if ( output == 0 || fwrite(down_partials,sizeof(*down_partials),(size_t)token_count,output) != (size_t)token_count )
	{
		fprintf(stderr,"minimax_cpu_validate: ffn partial write failed: %s\n",output_path);
		return(1);
	}
	fclose(output);
	fprintf(stderr,"minimax_cpu_validate: ffn-rank rank=%u positions=%u peak_rss=%llu\n",
		tp_rank,token_count,(unsigned long long)ToolPeakRssBytes());
	free(streams);
	free(down_partials);
	SparkMinimaxModuleStateFree(state);
	return(0);
}

static float *ToolReadPartials(const char *path,uint32_t token_count)
{
	FILE *file = fopen(path,"rb");
	float *partials;
	if ( file == 0 )
	{
		fprintf(stderr,"minimax_cpu_validate: partial file missing: %s\n",path);
		return(0);
	}
	partials = (float *)malloc((size_t)token_count * TOOL_HIDDEN * sizeof(float));
	if ( partials == 0 || fread(partials,sizeof(float),(size_t)token_count * TOOL_HIDDEN,file) != (size_t)token_count * TOOL_HIDDEN )
	{
		fprintf(stderr,"minimax_cpu_validate: partial file short: %s\n",path);
		free(partials);
		fclose(file);
		return(0);
	}
	fclose(file);
	return(partials);
}

static uint32_t *ToolReadPositions(const char *path,uint32_t *token_count)
{
	FILE *positions = fopen(path,"rb");
	uint32_t count = 0u;
	uint32_t *token_ids;
	if ( positions == 0 || fread(&count,sizeof(count),1u,positions) != 1u ||
		count == 0u || count > TOOL_MAX_POSITIONS )
	{
		fprintf(stderr,"minimax_cpu_validate: positions file invalid: %s\n",path);
		if ( positions != 0 )
			fclose(positions);
		return(0);
	}
	token_ids = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	if ( token_ids == 0 || fread(token_ids,sizeof(uint32_t),(size_t)count,positions) != (size_t)count )
	{
		fprintf(stderr,"minimax_cpu_validate: positions file truncated: %s\n",path);
		free(token_ids);
		fclose(positions);
		return(0);
	}
	fclose(positions);
	*token_count = count;
	return(token_ids);
}

static int ToolModeCombineAttn(const char *pack_path,const char *positions_path,const char *const *attn_paths,const char *streams_out_path)
{
	SparkMinimaxModuleState *state = 0;
	uint32_t token_count = 0u;
	uint32_t *token_ids = 0;
	uint16_t *streams_out;
	float *attn[4];
	FILE *output;
	token_ids = ToolReadPositions(positions_path,&token_count);
	if ( token_ids == 0 )
		return(1);
	if ( ToolLoadRankPack(&state,pack_path,0u) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: combine pack load failed: %s\n",pack_path);
		return(1);
	}
	streams_out = (uint16_t *)malloc((size_t)token_count * TOOL_HIDDEN * sizeof(uint16_t));
	if ( streams_out == 0 )
		return(1);
	for (uint32_t rank = 0u; rank < 4u; rank++)
	{
		attn[rank] = ToolReadPartials(attn_paths[rank],token_count);
		if ( attn[rank] == 0 )
			return(1);
	}
	for (uint32_t position = 0u; position < token_count; position++)
	{
		uint16_t *raw_embedding = ToolRawEmbedding(state,token_ids,position);
		if ( raw_embedding == 0 )
			return(1);
		for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
		{
			float attention = 0.0f;
			for (uint32_t rank = 0u; rank < 4u; rank++)
				attention += attn[rank][(uint64_t)position * TOOL_HIDDEN + column];
			streams_out[(uint64_t)position * TOOL_HIDDEN + column] = ToolRoundBf16(ToolBf16(raw_embedding[column]) + ToolBf16(ToolRoundBf16(attention)));
		}
		free(raw_embedding);
	}
	output = fopen(streams_out_path,"wb");
	if ( output == 0 || fwrite(&token_count,sizeof(token_count),1u,output) != 1u ||
		fwrite(streams_out,sizeof(uint16_t),(size_t)token_count * TOOL_HIDDEN,output) != (size_t)token_count * TOOL_HIDDEN )
	{
		fprintf(stderr,"minimax_cpu_validate: streams write failed: %s\n",streams_out_path);
		return(1);
	}
	fclose(output);
	fprintf(stderr,"minimax_cpu_validate: combine-attn positions=%u wrote %s peak_rss=%llu\n",
		token_count,streams_out_path,(unsigned long long)ToolPeakRssBytes());
	for (uint32_t rank = 0u; rank < 4u; rank++)
		free(attn[rank]);
	free(token_ids);
	free(streams_out);
	SparkMinimaxModuleStateFree(state);
	return(0);
}

static int ToolModeCombineFinal(const char *pack_path,const char *positions_path,const char *expected_path,const char *streams_path,const char *const *ffn_paths)
{
	SparkMinimaxModuleState *state = 0;
	uint32_t token_count = 0u;
	uint32_t *token_ids = 0;
	uint16_t *expected = 0;
	uint16_t *streams = 0;
	float *ffn[4];
	uint64_t exact = 0u,one_ulp = 0u,two_ulp = 0u,failures = 0u;
	float maximum_delta = 0.0f;
	int status = 0;
	token_ids = ToolReadPositions(positions_path,&token_count);
	if ( token_ids == 0 )
		return(1);
	if ( ToolLoadRankPack(&state,pack_path,0u) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"minimax_cpu_validate: combine pack load failed: %s\n",pack_path);
		return(1);
	}
	expected = ToolReadTokenFile(expected_path,&token_count);
	streams = ToolReadTokenFile(streams_path,&token_count);
	if ( expected == 0 || streams == 0 )
		return(1);
	for (uint32_t rank = 0u; rank < 4u; rank++)
	{
		ffn[rank] = ToolReadPartials(ffn_paths[rank],token_count);
		if ( ffn[rank] == 0 )
			return(1);
	}
	for (uint32_t position = 0u; position < token_count; position++)
	{
		for (uint32_t column = 0u; column < TOOL_HIDDEN; column++)
		{
			float down = 0.0f;
			float combined;
			uint16_t produced_bits,expected_bits;
			float delta;
			for (uint32_t rank = 0u; rank < 4u; rank++)
				down += ffn[rank][(uint64_t)position * TOOL_HIDDEN + column];
			combined = ToolBf16(ToolRoundBf16(ToolBf16(streams[(uint64_t)position * TOOL_HIDDEN + column]) + ToolBf16(ToolRoundBf16(down))));
			produced_bits = ToolRoundBf16(combined);
			expected_bits = expected[(uint64_t)position * TOOL_HIDDEN + column];
			delta = fabsf(combined - ToolBf16(expected_bits));
			if ( delta > maximum_delta )
				maximum_delta = delta;
			if ( produced_bits == expected_bits )
				exact++;
			else
			{
				int32_t difference = (int32_t)produced_bits - (int32_t)expected_bits;
				if ( difference < 0 )
					difference = -difference;
				if ( difference == 1 )
					one_ulp++;
				else if ( difference == 2 )
					two_ulp++;
				else
					failures++;
			}
		}
	}
	status = (maximum_delta <= 0.03125f) &&
		((double)exact >= 0.80 * (double)(token_count * TOOL_HIDDEN)) ? 0 : 2;
	fprintf(stderr,"minimax_cpu_validate: combine-final positions=%u exact=%llu/%u one_ulp=%llu two_ulp=%llu max_delta=%.6g over_two_ulp=%llu peak_rss=%llu verdict=%s\n",
		token_count,(unsigned long long)exact,(uint32_t)(token_count * TOOL_HIDDEN),
		(unsigned long long)one_ulp,(unsigned long long)two_ulp,(double)maximum_delta,
		(unsigned long long)failures,(unsigned long long)ToolPeakRssBytes(),
		status == 0 ? "PASS" : "FAIL");
	for (uint32_t rank = 0u; rank < 4u; rank++)
		free(ffn[rank]);
	free(token_ids);
	free(expected);
	free(streams);
	SparkMinimaxModuleStateFree(state);
	return(status);
}

int main(int argument_count,char **arguments)
{
	if ( argument_count >= 3 && strcmp(arguments[1],"pack-load") == 0 )
		return(ToolModePackLoad(arguments[2],argument_count >= 4 ? (uint32_t)atoi(arguments[3]) : 0u));
	if ( argument_count >= 6 && strcmp(arguments[1],"att-rank") == 0 )
		return(ToolModeAttentionRank(arguments[2],(uint32_t)atoi(arguments[3]),arguments[4],arguments[5]));
	if ( argument_count >= 6 && strcmp(arguments[1],"ffn-rank") == 0 )
		return(ToolModeFfnRank(arguments[2],(uint32_t)atoi(arguments[3]),arguments[4],arguments[5]));
	if ( argument_count >= 9 && strcmp(arguments[1],"combine-attn") == 0 )
	{
		const char *const attn_paths[4] = {arguments[4],arguments[5],arguments[6],arguments[7]};
		return(ToolModeCombineAttn(arguments[2],arguments[3],attn_paths,arguments[8]));
	}
	if ( argument_count >= 10 && strcmp(arguments[1],"combine-final") == 0 )
	{
		const char *const ffn_paths[4] = {arguments[6],arguments[7],arguments[8],arguments[9]};
		return(ToolModeCombineFinal(arguments[2],arguments[3],arguments[4],arguments[5],ffn_paths));
	}
	fprintf(stderr,"usage: %s pack-load <pack> | att-rank <pack> <rank> <positions.bin> <out> | ffn-rank <pack> <rank> <streams.bin> <out> | combine-attn <pack> <positions> <a0> <a1> <a2> <a3> <streams-out> | combine-final <pack> <positions> <expected> <streams> <f0> <f1> <f2> <f3>\n",arguments[0]);
	return(1);
}
