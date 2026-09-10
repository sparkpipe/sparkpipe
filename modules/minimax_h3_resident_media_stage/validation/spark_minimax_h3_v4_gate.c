#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_MINIMAX_H3_V4_VIDEO_BLOCKS 36u
#define SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN 2048u
#define SPARK_MINIMAX_H3_V4_VIDEO_HEADS 32u
#define SPARK_MINIMAX_H3_V4_VIDEO_HEAD_DIM 64u
#define SPARK_MINIMAX_H3_V4_VIDEO_FFN 16384u
#define SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT 3072u
#define SPARK_MINIMAX_H3_V4_VIDEO_EPS 1e-05f
#define SPARK_MINIMAX_H3_V4_VIDEO_TOKENS 33u
#define SPARK_MINIMAX_H3_V4_VIDEO_PATCHES 28u
#define SPARK_MINIMAX_H3_V4_VIDEO_FRAMES 22u
#define SPARK_MINIMAX_H3_V4_VIDEO_SIDE 32u
#define SPARK_MINIMAX_H3_V4_VIDEO_MAX_ABS (2.0 / 255.0)
#define SPARK_MINIMAX_H3_V4_VIDEO_MAX_REL 5e-4
#define SPARK_MINIMAX_H3_V4_AUDIO_BATCH 2u
#define SPARK_MINIMAX_H3_V4_AUDIO_CHANNELS 32u
#define SPARK_MINIMAX_H3_V4_AUDIO_LATENTS 4u
#define SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES 3200u
#define SPARK_MINIMAX_H3_V4_AUDIO_MAX_REL 5e-4
#define SPARK_MINIMAX_H3_V4_AUDIO_STAGES 7u

static void SparkMinimaxH3V4Stats(const char *label, const float *buffer,
	uint64_t count)
{
	double maximum = 0.0;
	uint64_t nonfinite = 0;
	uint64_t index;
	for (index=0u; index<count; index++)
	{
		double value = (double)buffer[index];
		if ( isnan(value) || isinf(value) )
			nonfinite++;
		else if ( fabs(value) > maximum )
			maximum = fabs(value);
	}
	printf("%-16s max=%.6g nonfinite=%llu\n",label,maximum,
		(unsigned long long)nonfinite);
}

static uint32_t SparkMinimaxH3V4Failures;

static const char *SparkMinimaxH3V4StageDirectory;
static const char *SparkMinimaxH3V4PreSnakeCompare;
static uint32_t SparkMinimaxH3V4VideoQInstrument;

static void SparkMinimaxH3V4CompareStageF32(const char *name, const float *actual,
	uint64_t count)
{
	char path[1024];
	float *reference;
	FILE *file;
	uint64_t index;
	double sum_squared = 0.0;
	double sum_squared_reference = 0.0;
	double relative;
	if ( SparkMinimaxH3V4StageDirectory == 0 )
		return;
	snprintf(path,sizeof(path),"%s/%s",SparkMinimaxH3V4StageDirectory,name);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		printf("missing stage fixture %s\n",path);
		SparkMinimaxH3V4Failures++;
		return;
	}
	reference = (float *)malloc((size_t)(count * 4u));
	if ( reference == 0 )
		exit(2);
	if ( fread(reference,1,count * 4u,file) != count * 4u )
	{
		printf("short read on %s\n",path);
		exit(2);
	}
	fclose(file);
	for (index=0u; index<count; index++)
	{
		double difference = (double)actual[index] - (double)reference[index];
		sum_squared += difference * difference;
		sum_squared_reference += (double)reference[index] *
			(double)reference[index];
	}
	relative = sqrt(sum_squared / (sum_squared_reference + 1e-30));
	printf("%-16s rel=%.3e %s\n",name,relative,
		relative <= 5e-3 ? "OK" : "FAIL");
	if ( relative > 5e-3 )
		SparkMinimaxH3V4Failures++;
	free(reference);
}

static void *SparkMinimaxH3V4ReadBin(const char *directory, const char *name,
	uint64_t bytes)
{
	char path[1024];
	FILE *file;
	void *buffer;
	snprintf(path,sizeof(path),"%s/%s",directory,name);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		printf("missing weight file %s\n",path);
		exit(2);
	}
	buffer = malloc((size_t)bytes);
	if ( buffer == 0 )
	{
		printf("host alloc failure %s\n",name);
		exit(2);
	}
	if ( fread(buffer,1,(size_t)bytes,file) != bytes )
	{
		printf("short read %s\n",path);
		exit(2);
	}
	fclose(file);
	return(buffer);
}

static void SparkMinimaxH3V4LoadWeight(float **target, const char *dir,
	const char *prefix, const char *name, uint64_t elements)
{
	char full[512];
	snprintf(full,sizeof(full),"%s%s.bin",prefix,name);
	*target = (float *)SparkMinimaxH3V4ReadBin(dir,full,elements * 4u);
}

static void SparkMinimaxH3V4Gem(float *out, uint32_t rows, uint32_t width,
	uint32_t depth, const float *activations, const float *weights, const float *bias)
{
	uint32_t row;
	#pragma omp parallel for schedule(static)
	for (row=0u; row<rows; row++)
	{
		uint32_t column;
		for (column=0u; column<width; column++)
		{
			const float *weight_row = weights + (uint64_t)column * depth;
			const float *activation_row = activations + (uint64_t)row * depth;
			float total = bias != 0 ? bias[column] : 0.0f;
			uint32_t index;
			for (index=0u; index<depth; index++)
				total += activation_row[index] * weight_row[index];
			out[(uint64_t)row * width + column] = total;
		}
	}
}

static void SparkMinimaxH3V4RmsNormWeighted(float *out, const float *input,
	const float *weight, uint32_t rows, uint32_t width, float epsilon)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		const float *input_row = input + (uint64_t)row * width;
		float sum = 0.0f;
		float inverse;
		float *out_row = out + (uint64_t)row * width;
		uint32_t column;
		for (column=0u; column<width; column++)
			sum += input_row[column] * input_row[column];
		inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
		for (column=0u; column<width; column++)
			out_row[column] = input_row[column] * inverse * weight[column];
	}
}

static void SparkMinimaxH3V4RmsNormBare(float *out, const float *input, uint32_t rows,
	uint32_t width, float epsilon)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		const float *input_row = input + (uint64_t)row * width;
		float sum = 0.0f;
		float inverse;
		float *out_row = out + (uint64_t)row * width;
		uint32_t column;
		for (column=0u; column<width; column++)
			sum += input_row[column] * input_row[column];
		inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
		for (column=0u; column<width; column++)
			out_row[column] = input_row[column] * inverse;
	}
}

static void SparkMinimaxH3V4LayerNorm(float *out, const float *input, const float *weight,
	const float *bias, uint32_t rows, uint32_t width, float epsilon)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		const float *input_row = input + (uint64_t)row * width;
		float mean = 0.0f;
		float variance = 0.0f;
		float *out_row = out + (uint64_t)row * width;
		uint32_t column;
		for (column=0u; column<width; column++)
			mean += input_row[column];
		mean /= (float)width;
		for (column=0u; column<width; column++)
		{
			float centered = input_row[column] - mean;
			variance += centered * centered;
		}
		variance /= (float)width;
		for (column=0u; column<width; column++)
			out_row[column] = (input_row[column] - mean) /
				sqrtf(variance + epsilon) * weight[column] + bias[column];
	}
}

static void SparkMinimaxH3V4SelfAttention(float *attention_out, const float *input,
	const float *query_weight, const float *query_bias, const float *key_weight,
	const float *key_bias, const float *value_weight, const float *value_bias,
	const float *output_weight, const float *output_bias, const float *cos_angles,
	const float *sin_angles, uint32_t tokens, uint32_t hidden, uint32_t heads,
	uint32_t head_dim, uint32_t rope_dim, float epsilon, float *query_buffer,
	float *key_buffer, float *value_buffer)
{
	uint32_t token,head,key_dim;
	SparkMinimaxH3V4Gem(query_buffer,tokens,hidden,hidden,input,query_weight,query_bias);
	if ( SparkMinimaxH3V4VideoQInstrument != 0u )
		SparkMinimaxH3V4CompareStageF32("refv_b0_q__33x2048.f32",query_buffer,
			(uint64_t)tokens * hidden);
	SparkMinimaxH3V4Gem(key_buffer,tokens,hidden,hidden,input,key_weight,key_bias);
	SparkMinimaxH3V4Gem(value_buffer,tokens,hidden,hidden,input,value_weight,value_bias);
	SparkMinimaxH3V4RmsNormBare(query_buffer,query_buffer,tokens * heads,head_dim,
		epsilon);
	if ( SparkMinimaxH3V4VideoQInstrument != 0u )
		SparkMinimaxH3V4CompareStageF32("refv_b0_qn__33x2048.f32",query_buffer,
			(uint64_t)tokens * hidden);
	SparkMinimaxH3V4RmsNormBare(key_buffer,key_buffer,tokens * heads,head_dim,epsilon);
	if ( SparkMinimaxH3V4VideoQInstrument != 0u )
		SparkMinimaxH3V4CompareStageF32("refv_b0_kn__33x2048.f32",key_buffer,
			(uint64_t)tokens * hidden);
	for (token=0u; token<tokens; token++)
	{
		for (head=0u; head<heads; head++)
		{
			float *query_row =
				query_buffer + ((uint64_t)token * heads + head) * head_dim;
			float *key_row = key_buffer + ((uint64_t)token * heads + head) * head_dim;
			for (key_dim=0u; key_dim<rope_dim / 2u; key_dim++)
			{
				uint32_t half = rope_dim / 2u;
				float cosine_a = cos_angles[token * rope_dim + key_dim];
				float sine_a = sin_angles[token * rope_dim + key_dim];
				float cosine_b = cos_angles[token * rope_dim + half + key_dim];
				float sine_b = sin_angles[token * rope_dim + half + key_dim];
				float query_a = query_row[key_dim];
				float query_b = query_row[half + key_dim];
				query_row[key_dim] = query_a * cosine_a - query_b * sine_a;
				query_row[half + key_dim] = query_b * cosine_b + query_a * sine_b;
				float key_a = key_row[key_dim];
				float key_b = key_row[half + key_dim];
				key_row[key_dim] = key_a * cosine_a - key_b * sine_a;
				key_row[half + key_dim] = key_b * cosine_b + key_a * sine_b;
			}
		}
	}
	{
		float scale = 1.0f / sqrtf((float)head_dim);
		uint32_t query_token;
		for (query_token=0u; query_token<tokens; query_token++)
		{
			uint32_t head_index;
			for (head_index=0u; head_index<heads; head_index++)
			{
				const float *query_row = query_buffer +
					((uint64_t)query_token * heads + head_index) * head_dim;
				float *out_row = attention_out +
					((uint64_t)query_token * heads + head_index) * head_dim;
				float scores[SPARK_MINIMAX_H3_V4_VIDEO_TOKENS];
				float maximum = -INFINITY;
				float total = 0.0f;
				uint32_t key_token,element;
				for (key_token=0u; key_token<tokens; key_token++)
				{
					const float *key_row = key_buffer +
						((uint64_t)key_token * heads + head_index) * head_dim;
					float dot = 0.0f;
					for (element=0u; element<head_dim; element++)
						dot += query_row[element] * key_row[element];
					scores[key_token] = dot * scale;
					if ( scores[key_token] > maximum )
						maximum = scores[key_token];
				}
				for (key_token=0u; key_token<tokens; key_token++)
				{
					scores[key_token] = expf(scores[key_token] - maximum);
					total += scores[key_token];
				}
				for (element=0u; element<head_dim; element++)
					out_row[element] = 0.0f;
				for (key_token=0u; key_token<tokens; key_token++)
				{
					const float *value_row = value_buffer +
						((uint64_t)key_token * heads + head_index) * head_dim;
					float weight = scores[key_token] / total;
					for (element=0u; element<head_dim; element++)
						out_row[element] += weight * value_row[element];
				}
			}
		}
	}
	SparkMinimaxH3V4Gem(value_buffer,tokens,hidden,hidden,attention_out,
		output_weight,output_bias);
	memcpy(attention_out,value_buffer,(uint64_t)tokens * hidden * 4u);
}

static void SparkMinimaxH3V4Swiglu(float *out, const float *input, const float *gate_up,
	const float *gate_bias, const float *down, const float *down_bias, uint32_t tokens,
	uint32_t hidden, uint32_t ffn, float *scratch)
{
	uint32_t token,column;
	SparkMinimaxH3V4Gem(scratch,tokens,ffn * 2u,hidden,input,gate_up,gate_bias);
	for (token=0u; token<tokens; token++)
	{
		for (column=0u; column<ffn; column++)
		{
			float up = scratch[(uint64_t)token * (2u * ffn) + column];
			float gate = scratch[(uint64_t)token * (2u * ffn) + ffn + column];
			scratch[(uint64_t)token * (2u * ffn) + column] =
				up * (gate / (1.0f + expf(-gate)));
		}
		for (column=0u; column<ffn; column++)
			scratch[(uint64_t)token * ffn + column] =
				scratch[(uint64_t)token * (2u * ffn) + column];
	}
	SparkMinimaxH3V4Gem(out,tokens,hidden,ffn,scratch,down,down_bias);
}

static void SparkMinimaxH3V4VideoRope(float *cos_angles, float *sin_angles)
{
	float inverse_frequency[8];
	uint32_t axis,frequency,token;
	for (frequency=0u; frequency<8u; frequency++)
		inverse_frequency[frequency] = powf(100.0f, -(2.0f * 3.0f / 48.0f) *
			(float)frequency);
	for (token=0u; token<SPARK_MINIMAX_H3_V4_VIDEO_TOKENS; token++)
	{
		float position[3];
		uint32_t coordinate;
		for (coordinate=0u; coordinate<3u; coordinate++)
		{
			float raw = 0.0f;
			if ( token < SPARK_MINIMAX_H3_V4_VIDEO_PATCHES )
			{
				uint32_t frame = token / 4u;
				uint32_t remainder = token - frame * 4u;
				uint32_t height = remainder / 2u;
				uint32_t width = remainder - height * 2u;
				float grids[3] = {
					2.0f * ((float)frame + 0.5f) / 7.0f - 1.0f,
					2.0f * ((float)height + 0.5f) / 2.0f - 1.0f,
					2.0f * ((float)width + 0.5f) / 2.0f - 1.0f
				};
				raw = grids[coordinate];
			}
			position[coordinate] = raw;
		}
		for (axis=0u; axis<3u; axis++)
		{
			for (frequency=0u; frequency<8u; frequency++)
			{
				float angle = 2.0f * (float)M_PI * position[axis] *
					inverse_frequency[frequency];
				cos_angles[token * 48u + axis * 8u + frequency] = cosf(angle);
				cos_angles[token * 48u + 24u + axis * 8u + frequency] = cosf(angle);
				sin_angles[token * 48u + axis * 8u + frequency] = sinf(angle);
				sin_angles[token * 48u + 24u + axis * 8u + frequency] = sinf(angle);
			}
		}
	}
}

static void SparkMinimaxH3V4VideoGate(const char *fixture_dir, const char *weight_dir)
{
	uint64_t z_elements = 24u * 7u * 2u * 2u;
	uint64_t decoded_elements = 3u * 22u * 32u * 32u;
	uint64_t hidden_elements = (uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS *
		SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN;
	float *z = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,"z__1x24x7x2x2.f32",
		z_elements * 4u);
	float *expected = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,
		"decoded__1x3x22x32x32.f32",decoded_elements * 4u);
	float *mean = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,"latents_mean__24.f32",96u);
	float *std = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,"latents_std__24.f32",96u);
	float *post_quant_weight,*post_quant_bias,*proj_in_weight,*proj_in_bias,*registers;
	float *latent_rows,*hidden,*tokens,*normed,*query_buffer,*key_buffer,*value_buffer;
	float *attention_out,*ffn_scratch,*cos_angles,*sin_angles,*proj_out,*pixels;
	float *decoded = (float *)malloc(decoded_elements * 4u);
	uint32_t block,frame,channel,height,width;
	uint64_t index;
	SparkMinimaxH3V4LoadWeight(&post_quant_weight,weight_dir,"","post_quant_conv_weight",
		24u * 24u);
	SparkMinimaxH3V4LoadWeight(&post_quant_bias,weight_dir,"","post_quant_conv_bias",24u);
	SparkMinimaxH3V4LoadWeight(&proj_in_weight,weight_dir,"decoder_","proj_in_weight",
		(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * 24u);
	SparkMinimaxH3V4LoadWeight(&proj_in_bias,weight_dir,"decoder_","proj_in_bias",
		SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
	SparkMinimaxH3V4LoadWeight(&registers,weight_dir,"decoder_","register_tokens",
		4u * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
	latent_rows = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_PATCHES * 24u * 4u);
	for (index=0u; index<z_elements / 24u; index++)
	{
		for (channel=0u; channel<24u; channel++)
		{
			float total = post_quant_bias[channel];
			uint32_t inner;
			for (inner=0u; inner<24u; inner++)
				total += z[inner * (z_elements / 24u) + index] *
					post_quant_weight[channel * 24u + inner];
			latent_rows[index * 24u + channel] = total;
		}
	}
	free(z);
	SparkMinimaxH3V4Stats("post_quant",latent_rows,(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_PATCHES * 24u);
	tokens = (float *)malloc(hidden_elements * 4u);
	SparkMinimaxH3V4Gem(tokens,SPARK_MINIMAX_H3_V4_VIDEO_PATCHES,
		SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,24u,latent_rows,proj_in_weight,proj_in_bias);
	free(latent_rows);
	SparkMinimaxH3V4Stats("proj_in_weight",proj_in_weight,
		(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * 24u);
	SparkMinimaxH3V4Stats("registers",registers,
		4u * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
	memcpy(tokens + (uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_PATCHES *
		SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,registers,
		4u * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * 4u);
	memset(tokens + (uint64_t)(SPARK_MINIMAX_H3_V4_VIDEO_PATCHES + 4u) *
		SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,0,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * 4u);
	SparkMinimaxH3V4StageDirectory = fixture_dir;
	SparkMinimaxH3V4CompareStageF32("refv_in33__33x2048.f32",tokens,
		hidden_elements);
	normed = (float *)malloc(hidden_elements * 4u);
	query_buffer = (float *)malloc(hidden_elements * 4u);
	key_buffer = (float *)malloc(hidden_elements * 4u);
	value_buffer = (float *)malloc(hidden_elements * 4u);
	attention_out = (float *)malloc(hidden_elements * 4u);
	ffn_scratch = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS *
		SPARK_MINIMAX_H3_V4_VIDEO_FFN * 4u);
	cos_angles = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * 48u * 4u);
	sin_angles = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * 48u * 4u);
	proj_out = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS *
		SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT * 4u);
	pixels = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS *
		SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT * 4u);
	hidden = (float *)malloc(hidden_elements * 4u);
	SparkMinimaxH3V4VideoRope(cos_angles,sin_angles);
	SparkMinimaxH3V4Stats("tokens_in",tokens,hidden_elements);
	SparkMinimaxH3V4CompareStageF32("refv_rope_cos__33x48.f32",cos_angles,
		(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * 48u);
	SparkMinimaxH3V4CompareStageF32("refv_rope_sin__33x48.f32",sin_angles,
		(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * 48u);
	SparkMinimaxH3V4Stats("weights_postquant_w",post_quant_weight,576u);
	memcpy(hidden,tokens,hidden_elements * 4u);
	for (block=0u; block<SPARK_MINIMAX_H3_V4_VIDEO_BLOCKS; block++)
	{
		char prefix[96];
		float *query_weight,*query_bias,*key_weight,*key_bias,*value_weight,*value_bias;
		float *output_weight,*output_bias,*gate_up,*gate_bias,*down,*down_bias;
		float *norm1,*norm2,*scale1,*scale2;
		snprintf(prefix,sizeof(prefix),"decoder_transformer_blocks_%u_",block);
		SparkMinimaxH3V4LoadWeight(&query_weight,weight_dir,prefix,"attn_to_q_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&query_bias,weight_dir,prefix,"attn_to_q_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&key_weight,weight_dir,prefix,"attn_to_k_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&key_bias,weight_dir,prefix,"attn_to_k_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&value_weight,weight_dir,prefix,"attn_to_v_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&value_bias,weight_dir,prefix,"attn_to_v_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&output_weight,weight_dir,prefix,"attn_to_out_0_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&output_bias,weight_dir,prefix,"attn_to_out_0_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&gate_up,weight_dir,prefix,"ff_net_0_proj_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_FFN * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&gate_bias,weight_dir,prefix,"ff_net_0_proj_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_FFN);
		SparkMinimaxH3V4LoadWeight(&down,weight_dir,prefix,"ff_net_2_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN * 8192u);
		SparkMinimaxH3V4LoadWeight(&down_bias,weight_dir,prefix,"ff_net_2_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&norm1,weight_dir,prefix,"norm1_weight",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&norm2,weight_dir,prefix,"norm2_weight",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&scale1,weight_dir,prefix,"scale1",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&scale2,weight_dir,prefix,"scale2",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4RmsNormWeighted(normed,hidden,norm1,
			SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,
			SPARK_MINIMAX_H3_V4_VIDEO_EPS);
		if ( block == 0u )
			SparkMinimaxH3V4CompareStageF32("refv_b0_n1__33x2048.f32",normed,
				hidden_elements);
		SparkMinimaxH3V4VideoQInstrument = block == 0u;
		SparkMinimaxH3V4SelfAttention(attention_out,normed,query_weight,query_bias,
			key_weight,key_bias,value_weight,value_bias,output_weight,output_bias,
			cos_angles,sin_angles,SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,SPARK_MINIMAX_H3_V4_VIDEO_HEADS,
			SPARK_MINIMAX_H3_V4_VIDEO_HEAD_DIM,48u,SPARK_MINIMAX_H3_V4_VIDEO_EPS,
			query_buffer,key_buffer,value_buffer);
		SparkMinimaxH3V4VideoQInstrument = 0u;
		if ( block == 0u )
		{
			SparkMinimaxH3V4CompareStageF32("refv_b0_qr__33x2048.f32",
				query_buffer,hidden_elements);
			SparkMinimaxH3V4CompareStageF32("refv_b0_kr__33x2048.f32",
				key_buffer,hidden_elements);
			SparkMinimaxH3V4CompareStageF32("refv_b0_ao__33x2048.f32",
				attention_out,hidden_elements);
		}
		for (index=0u; index<hidden_elements; index++)
			hidden[index] += attention_out[index] * scale1[index %
				SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN];
		if ( block == 0u )
			SparkMinimaxH3V4CompareStageF32("refv_b0_attn__33x2048.f32",hidden,
				hidden_elements);
		SparkMinimaxH3V4RmsNormWeighted(normed,hidden,norm2,
			SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,
			SPARK_MINIMAX_H3_V4_VIDEO_EPS);
		if ( block == 0u )
			SparkMinimaxH3V4CompareStageF32("refv_b0_n2__33x2048.f32",normed,
				hidden_elements);
		SparkMinimaxH3V4Swiglu(attention_out,normed,gate_up,gate_bias,down,down_bias,
			SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,
			SPARK_MINIMAX_H3_V4_VIDEO_FFN / 2u,ffn_scratch);
		if ( block == 0u )
			SparkMinimaxH3V4CompareStageF32("refv_b0_ffn__33x2048.f32",
				attention_out,hidden_elements);
		for (index=0u; index<hidden_elements; index++)
			hidden[index] += attention_out[index] * scale2[index %
				SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN];
		if ( block == 0u )
		{
			SparkMinimaxH3V4CompareStageF32("refv_b0_attn__33x2048.f32",hidden,
				hidden_elements);
			SparkMinimaxH3V4CompareStageF32("refv_b0__33x2048.f32",hidden,
				hidden_elements);
		}
		if ( block == 1u )
			SparkMinimaxH3V4CompareStageF32("refv_b1__33x2048.f32",hidden,
				hidden_elements);
		free(query_weight); free(query_bias); free(key_weight); free(key_bias);
		free(value_weight); free(value_bias); free(output_weight); free(output_bias);
		free(gate_up); free(gate_bias); free(down); free(down_bias);
		free(norm1); free(norm2); free(scale1); free(scale2);
	}
	{
		float *norm_weight,*norm_bias,*proj_weight,*proj_bias;
		SparkMinimaxH3V4LoadWeight(&norm_weight,weight_dir,"decoder_","norm_out_weight",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&norm_bias,weight_dir,"decoder_","norm_out_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&proj_weight,weight_dir,"decoder_","proj_out_weight",
			(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LoadWeight(&proj_bias,weight_dir,"decoder_","proj_out_bias",
			SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT);
		SparkMinimaxH3V4Stats("norm_out_w",norm_weight,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4Stats("proj_out_w",proj_weight,(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4LayerNorm(proj_out,hidden,norm_weight,norm_bias,
			SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,
			SPARK_MINIMAX_H3_V4_VIDEO_EPS);
		SparkMinimaxH3V4Stats("post_layernorm",proj_out,(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN);
		SparkMinimaxH3V4Gem(pixels,SPARK_MINIMAX_H3_V4_VIDEO_TOKENS,
			SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT,SPARK_MINIMAX_H3_V4_VIDEO_HIDDEN,
			proj_out,proj_weight,proj_bias);
		SparkMinimaxH3V4Stats("pixels",pixels,(uint64_t)SPARK_MINIMAX_H3_V4_VIDEO_TOKENS * SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT);
		free(norm_weight); free(norm_bias); free(proj_weight); free(proj_bias);
	}
	for (frame=0u; frame<SPARK_MINIMAX_H3_V4_VIDEO_FRAMES; frame++)
	{
		uint32_t source_frame = frame < 17u ? frame + 3u : frame + 6u;
		for (channel=0u; channel<3u; channel++)
		{
			for (height=0u; height<SPARK_MINIMAX_H3_V4_VIDEO_SIDE; height++)
			{
				for (width=0u; width<SPARK_MINIMAX_H3_V4_VIDEO_SIDE; width++)
				{
					uint32_t destination =
						((channel * 22u + frame) * 32u + height) * 32u + width;
					uint32_t token_grid_frame = source_frame / 4u;
					uint32_t intra = source_frame - token_grid_frame * 4u;
					uint32_t token = (token_grid_frame * 2u + height / 16u) * 2u +
						width / 16u;
					uint32_t element = channel * 1024u + intra * 256u +
						(height % 16u) * 16u + (width % 16u);
					decoded[destination] =
						pixels[(uint64_t)token * SPARK_MINIMAX_H3_V4_VIDEO_PROJ_OUT +
							element];
				}
			}
		}
	}
		SparkMinimaxH3V4Stats("decoded",decoded,decoded_elements);
		SparkMinimaxH3V4Stats("expected",expected,decoded_elements);
	{
		double sum_squared = 0.0;
		double max_abs = 0.0;
		double sum_squared_reference = 0.0;
		uint32_t within;
		double relative;
		uint64_t expected_nonfinite = 0;
		uint32_t expected_usable = 1;
		for (index=0u; index<decoded_elements; index++)
		{
			if ( !isfinite(expected[index]) )
				expected_nonfinite++;
		}
		if ( expected_nonfinite != 0u )
		{
			printf("%-16s expected fixture carries %llu/%llu nonfinite values - "
				"the anchor generator's diffusers decode produced NaN; the gate "
				"cannot bind until that lane regenerates the fixture\n","video_vae",
				(unsigned long long)expected_nonfinite,
				(unsigned long long)decoded_elements);
			SparkMinimaxH3V4Failures++;
			expected_usable = 0u;
		}
		if ( expected_usable != 0u )
		{
			for (index=0u; index<decoded_elements; index++)
			{
				double difference = (double)decoded[index] - (double)expected[index];
				sum_squared += difference * difference;
				sum_squared_reference += (double)expected[index] * (double)expected[index];
				if ( fabs(difference) > max_abs )
					max_abs = fabs(difference);
			}
			relative = sqrt(sum_squared / (sum_squared_reference + 1e-30));
			within = relative <= SPARK_MINIMAX_H3_V4_VIDEO_MAX_REL &&
				max_abs <= SPARK_MINIMAX_H3_V4_VIDEO_MAX_ABS;
			printf("%-16s rel=%.3e max_abs=%.6f %s\n","video_vae",relative,max_abs,
				within != 0u ? "OK" : "FAIL");
			if ( within == 0u )
				SparkMinimaxH3V4Failures++;
		}
	}
	free(expected); free(mean); free(std); free(post_quant_weight);
	free(post_quant_bias); free(proj_in_weight); free(proj_in_bias); free(registers);
	free(tokens); free(normed); free(query_buffer); free(key_buffer);
	free(value_buffer); free(attention_out); free(ffn_scratch); free(cos_angles);
	free(sin_angles); free(proj_out); free(pixels); free(hidden); free(decoded);
}

static float *SparkMinimaxH3V4WnCombine(uint32_t out_channels, uint32_t inner,
	uint32_t kernel, const float *weight_g, const float *weight_v)
{
	float *weight = (float *)malloc((uint64_t)out_channels * inner * kernel * 4u);
	uint32_t channel;
	#pragma omp parallel for schedule(static)
	for (channel=0u; channel<out_channels; channel++)
	{
		double sum = 0.0;
		float norm;
		uint32_t index;
		for (index=0u; index<(uint64_t)inner * kernel; index++)
		{
			double value = (double)weight_v[(uint64_t)channel * inner * kernel + index];
			sum += value * value;
		}
		norm = (float)sqrt(sum);
		for (index=0u; index<(uint64_t)inner * kernel; index++)
			weight[(uint64_t)channel * inner * kernel + index] =
				weight_g[channel] * weight_v[(uint64_t)channel * inner * kernel + index] /
				norm;
	}
	return(weight);
}

static void SparkMinimaxH3V4Conv1d(float *out, const float *input, uint32_t out_channels,
	uint32_t in_channels, uint32_t length, const float *weight, const float *bias,
	uint32_t kernel, uint32_t dilation, uint32_t padding)
{
	uint32_t channel;
	#pragma omp parallel for schedule(static)
	for (channel=0u; channel<out_channels; channel++)
	{
		uint32_t position;
		for (position=0u; position<length; position++)
		{
			float total = bias != 0 ? bias[channel] : 0.0f;
			uint32_t inner,tap;
			for (inner=0u; inner<in_channels; inner++)
			{
				const float *input_row = input + (uint64_t)inner * length;
				const float *weight_row = weight +
					(((uint64_t)channel * in_channels) + inner) * kernel;
			for (tap=0u; tap<kernel; tap++)
			{
				int32_t source = (int32_t)position +
					(int32_t)(tap * dilation) - (int32_t)padding;
				if ( source >= 0 && source < (int32_t)length )
					total += input_row[source] * weight_row[tap];
			}
			}
			out[(uint64_t)channel * length + position] = total;
		}
	}
}

static void SparkMinimaxH3V4ConvTranspose1d(float *out, const float *input,
	uint32_t in_channels, uint32_t out_channels, uint32_t input_length,
	uint32_t output_length, const float *weight, const float *bias, uint32_t kernel,
	uint32_t stride, uint32_t padding)
{
	uint32_t in_channel;
	for (in_channel=0u; in_channel<in_channels; in_channel++)
	{
		uint32_t position;
		for (position=0u; position<input_length; position++)
		{
			float value = input[(uint64_t)in_channel * input_length + position];
			uint32_t out_channel,tap;
			for (out_channel=0u; out_channel<out_channels; out_channel++)
			{
				const float *weight_row = weight +
					((uint64_t)in_channel * out_channels + out_channel) * kernel;
				for (tap=0u; tap<kernel; tap++)
				{
					int32_t target = (int32_t)(position * stride + tap) -
						(int32_t)padding;
					if ( target >= 0 && target < (int32_t)output_length )
						out[(uint64_t)out_channel * output_length + target] +=
							value * weight_row[tap];
				}
			}
		}
	}
	if ( bias != 0 )
	{
		uint32_t channel;
		for (channel=0u; channel<out_channels; channel++)
		{
			uint32_t position;
			for (position=0u; position<output_length; position++)
				out[(uint64_t)channel * output_length + position] += bias[channel];
		}
	}
}

static void SparkMinimaxH3V4ReplicatePadAsymmetric(float *out, const float *input,
	uint32_t channels, uint32_t length, uint32_t pad_left, uint32_t pad_right)
{
	uint32_t channel;
	for (channel=0u; channel<channels; channel++)
	{
		const float *input_row = input + (uint64_t)channel * length;
		float *out_row = out + (uint64_t)channel * (length + pad_left + pad_right);
		uint32_t position;
		for (position=0u; position<pad_left; position++)
			out_row[position] = input_row[0];
		memcpy(out_row + pad_left,input_row,length * 4u);
		for (position=0u; position<pad_right; position++)
			out_row[pad_left + length + position] = input_row[length - 1u];
	}
}

static void SparkMinimaxH3V4Upsample1d(float *out, uint32_t output_length,
	const float *input, uint32_t channels, uint32_t input_length, const float *filter,
	uint32_t kernel, uint32_t ratio, float *padded, float *expanded)
{
	uint32_t pad = kernel / ratio - 1u;
	uint32_t pad_left = pad * ratio + (kernel - ratio) / 2u;
	uint32_t padded_length = input_length + 2u * pad;
	uint32_t stride_length = (padded_length - 1u) * ratio + kernel;
	uint32_t channel,position;
	SparkMinimaxH3V4ReplicatePadAsymmetric(padded,input,channels,input_length,
		pad,pad);
	memset(expanded,0,(uint64_t)channels * stride_length * 4u);
	for (position=0u; position<padded_length; position++)
	{
		for (channel=0u; channel<channels; channel++)
		{
			float value = padded[(uint64_t)channel * padded_length + position];
			uint32_t tap;
			for (tap=0u; tap<kernel; tap++)
			{
				uint32_t target = position * ratio + tap;
				expanded[(uint64_t)channel * stride_length + target] +=
					value * filter[tap];
			}
		}
	}
	for (channel=0u; channel<channels; channel++)
	{
		for (position=0u; position<output_length; position++)
			out[(uint64_t)channel * output_length + position] = ratio *
				expanded[(uint64_t)channel * stride_length + pad_left + position];
	}
}

static void SparkMinimaxH3V4Lowpass(float *out, uint32_t output_length,
	const float *input, uint32_t channels, uint32_t input_length, const float *filter,
	uint32_t kernel, uint32_t stride, float *padded)
{
	uint32_t pad_left = kernel / 2u - 1u;
	uint32_t pad_right = kernel / 2u;
	uint32_t padded_length = input_length + pad_left + pad_right;
	uint32_t channel;
	SparkMinimaxH3V4ReplicatePadAsymmetric(padded,input,channels,input_length,
		pad_left,pad_right);
	#pragma omp parallel for schedule(static)
	for (channel=0u; channel<channels; channel++)
	{
		uint32_t position;
		for (position=0u; position<output_length; position++)
		{
			float total = 0.0f;
			uint32_t tap;
			for (tap=0u; tap<kernel; tap++)
			{
				uint32_t source = position * stride + tap;
				if ( source < padded_length )
					total += padded[(uint64_t)channel * padded_length + source] *
						filter[tap];
			}
			out[(uint64_t)channel * output_length + position] = total;
		}
	}
}

static void SparkMinimaxH3V4Activation1d(float *out, uint32_t output_length,
	const float *input, uint32_t channels, uint32_t input_length, const float *alpha,
	const float *beta, const float *up_filter, const float *down_filter,
	float *upsampled, float *padded, float *expanded)
{
	uint32_t mid_length = input_length * 2u;
	uint32_t channel,position;
	SparkMinimaxH3V4Upsample1d(upsampled,mid_length,input,channels,input_length,
		up_filter,12u,2u,padded,expanded);
	if ( SparkMinimaxH3V4PreSnakeCompare != 0 )
	{
		SparkMinimaxH3V4CompareStageF32(SparkMinimaxH3V4PreSnakeCompare,
			upsampled,(uint64_t)channels * mid_length);
		SparkMinimaxH3V4PreSnakeCompare = 0;
	}
	for (channel=0u; channel<channels; channel++)
	{
		for (position=0u; position<mid_length; position++)
		{
			float value = upsampled[(uint64_t)channel * mid_length + position];
			float a = expf(alpha[channel]);
			float b = expf(beta[channel]);
			float sine = sinf(a * value);
			upsampled[(uint64_t)channel * mid_length + position] =
				value + sine * sine / (b + 1e-9f);
		}
	}
	SparkMinimaxH3V4Lowpass(out,output_length,upsampled,channels,mid_length,down_filter,
		12u,2u,padded);
}

static void SparkMinimaxH3V4AudioGate(const char *fixture_dir, const char *weight_dir)
{
	uint64_t latent_elements = (uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
		SPARK_MINIMAX_H3_V4_AUDIO_CHANNELS * SPARK_MINIMAX_H3_V4_AUDIO_LATENTS;
	uint64_t wave_elements = (uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES;
	static const uint32_t rates[7] = {5u,5u,2u,2u,2u,2u,2u};
	static const uint32_t kernels[7] = {9u,9u,4u,4u,4u,4u,4u};
	static const uint32_t block_kernels[3] = {3u,7u,11u};
	static const uint32_t dilations[3] = {1u,3u,5u};
	float *latents = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,
		"latents__2x32x4.f32",latent_elements * 4u);
	float *expected = (float *)SparkMinimaxH3V4ReadBin(fixture_dir,
		"waveform__2x1x3200.f32",wave_elements * 4u);
	float *x = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *next = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *residual = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *activation_buffer = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
		1024u * SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *conv_out = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *upsampled = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *padded = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *expanded = (float *)malloc((uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES * 4u);
	float *waveform = (float *)malloc(wave_elements * 4u);
	uint32_t stage,block,dilation_index,batch;
	uint64_t index;
	{
		float *weight,*bias;
		SparkMinimaxH3V4LoadWeight(&weight,weight_dir,"","dec_in_proj_weight",
			2048u * SPARK_MINIMAX_H3_V4_AUDIO_CHANNELS);
		SparkMinimaxH3V4LoadWeight(&bias,weight_dir,"","dec_in_proj_bias",2048u);
		for (batch=0u; batch<SPARK_MINIMAX_H3_V4_AUDIO_BATCH; batch++)
			SparkMinimaxH3V4Conv1d(x + (uint64_t)batch * 2048u *
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,latents + (uint64_t)batch *
				SPARK_MINIMAX_H3_V4_AUDIO_CHANNELS *
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,2048u,
				SPARK_MINIMAX_H3_V4_AUDIO_CHANNELS,
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,weight,bias,1u,1u,0u);
		free(weight); free(bias);
		SparkMinimaxH3V4StageDirectory = fixture_dir;
		SparkMinimaxH3V4CompareStageF32("refa_dec_in__2x2048x4.f32",x,
			(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 2048u *
			SPARK_MINIMAX_H3_V4_AUDIO_LATENTS);
	}
	{
		float *weight_g,*weight_v,*bias,*weight;
		SparkMinimaxH3V4LoadWeight(&weight_g,weight_dir,"","decoder_conv_pre_weight_g",
			1024u);
		SparkMinimaxH3V4LoadWeight(&weight_v,weight_dir,"","decoder_conv_pre_weight_v",
			1024u * 2048u * 7u);
		SparkMinimaxH3V4LoadWeight(&bias,weight_dir,"","decoder_conv_pre_bias",1024u);
		weight = SparkMinimaxH3V4WnCombine(1024u,2048u,7u,weight_g,weight_v);
		for (batch=0u; batch<SPARK_MINIMAX_H3_V4_AUDIO_BATCH; batch++)
			SparkMinimaxH3V4Conv1d(next + (uint64_t)batch * 1024u *
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,x + (uint64_t)batch * 2048u *
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,1024u,2048u,
				SPARK_MINIMAX_H3_V4_AUDIO_LATENTS,weight,bias,7u,1u,3u);
		free(bias); free(weight); free(weight_g); free(weight_v);
	}
	{
		float *swap = x;
		x = next;
		next = swap;
	}
	SparkMinimaxH3V4CompareStageF32("refa_conv_pre__2x1024x4.f32",x,
		(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * 1024u *
		SPARK_MINIMAX_H3_V4_AUDIO_LATENTS);
	for (stage=0u; stage<SPARK_MINIMAX_H3_V4_AUDIO_STAGES; stage++)
	{
		char stage_tag[32];
		uint32_t in_channels = 1024u >> stage;
		uint32_t out_channels = 1024u >> (stage + 1u);
		uint32_t input_length = SPARK_MINIMAX_H3_V4_AUDIO_LATENTS;
		uint32_t output_length;
		uint32_t step;
		char prefix[64];
		float *weight_g,*weight_v,*bias,*weight;
		for (step=0u; step<stage; step++)
			input_length *= rates[step];
		output_length = (input_length - 1u) * rates[stage] - 2u *
			((kernels[stage] - rates[stage]) / 2u) + kernels[stage];
		snprintf(prefix,sizeof(prefix),"decoder_ups_%u_0_",stage);
		SparkMinimaxH3V4LoadWeight(&weight_g,weight_dir,prefix,"weight_g",in_channels);
		SparkMinimaxH3V4LoadWeight(&weight_v,weight_dir,prefix,"weight_v",
			(uint64_t)in_channels * out_channels * kernels[stage]);
		SparkMinimaxH3V4LoadWeight(&bias,weight_dir,prefix,"bias",out_channels);
		weight = SparkMinimaxH3V4WnCombine(in_channels,out_channels,kernels[stage],
			weight_g,weight_v);
		memset(next,0,(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
			output_length * 4u);
		for (batch=0u; batch<SPARK_MINIMAX_H3_V4_AUDIO_BATCH; batch++)
			SparkMinimaxH3V4ConvTranspose1d(next + (uint64_t)batch * out_channels *
				output_length,x + (uint64_t)batch * in_channels * input_length,
				in_channels,out_channels,input_length,output_length,weight,bias,
				kernels[stage],rates[stage],(kernels[stage] - rates[stage]) / 2u);
		free(weight_g); free(weight_v); free(bias); free(weight);
		{
			float *swap = x;
			x = next;
			next = swap;
		}
		if ( stage == 0u )
			SparkMinimaxH3V4CompareStageF32("refa_ups0__2x512x20.f32",x,
				(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
				output_length);
		memset(residual,0,(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
			output_length * 4u);
		for (block=0u; block<3u; block++)
		{
			snprintf(prefix,sizeof(prefix),"decoder_resblocks_%u_",stage * 3u + block);
			memcpy(next,x,(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
				output_length * 4u);
			for (dilation_index=0u; dilation_index<3u; dilation_index++)
			{
				uint32_t dilation = dilations[dilation_index];
				uint32_t pass;
				for (pass=0u; pass<2u; pass++)
				{
					char act_prefix[96];
					uint32_t conv_dilation = pass == 0u ? dilation : 1u;
					uint32_t conv_padding = pass == 0u ?
						(block_kernels[block] * dilation - dilation) / 2u :
						(block_kernels[block] - 1u) / 2u;
					float *alpha,*beta,*up_filter,*down_filter;
					float *weight_bg,*weight_bv,*bias_conv,*weight_conv;
					snprintf(act_prefix,sizeof(act_prefix),"%sactivations_%u_",prefix,
						2u * dilation_index + pass);
					SparkMinimaxH3V4LoadWeight(&alpha,weight_dir,act_prefix,"act_alpha",
						out_channels);
					SparkMinimaxH3V4LoadWeight(&beta,weight_dir,act_prefix,"act_beta",
						out_channels);
					SparkMinimaxH3V4LoadWeight(&up_filter,weight_dir,act_prefix,
						"upsample_filter",12u);
					SparkMinimaxH3V4LoadWeight(&down_filter,weight_dir,act_prefix,
						"downsample_lowpass_filter",12u);
					if ( stage == 0u && block == 0u && dilation_index == 0u &&
						pass == 0u )
						SparkMinimaxH3V4PreSnakeCompare =
							"refa_s0_u0__2x512x40.f32";
					SparkMinimaxH3V4Activation1d(activation_buffer,output_length,next,
						out_channels,output_length,alpha,beta,up_filter,down_filter,
						upsampled,padded,expanded);
					free(alpha); free(beta); free(up_filter); free(down_filter);
					if ( stage == 0u && block == 0u && dilation_index == 0u )
					{
						char ref_tag[48];
						snprintf(ref_tag,sizeof(ref_tag),
							"refa_s0_m%u__2x512x40.f32",
							2u * dilation_index + pass);
						SparkMinimaxH3V4CompareStageF32(ref_tag,upsampled,
							(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
							out_channels * output_length * 2u);
						if ( pass == 0u )
							printf("drv_m0[0..7]=%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
								upsampled[0],upsampled[1],upsampled[2],upsampled[3],
								upsampled[4],upsampled[5],upsampled[6],upsampled[7]);
					}
					snprintf(act_prefix,sizeof(act_prefix),"%sconvs%u_%u_",prefix,
						pass + 1u,dilation_index);
					SparkMinimaxH3V4LoadWeight(&weight_bg,weight_dir,act_prefix,
						"weight_g",out_channels);
					SparkMinimaxH3V4LoadWeight(&weight_bv,weight_dir,act_prefix,
						"weight_v",(uint64_t)out_channels * out_channels *
						block_kernels[block]);
					SparkMinimaxH3V4LoadWeight(&bias_conv,weight_dir,act_prefix,"bias",
						out_channels);
					weight_conv = SparkMinimaxH3V4WnCombine(out_channels,out_channels,
						block_kernels[block],weight_bg,weight_bv);
					SparkMinimaxH3V4Conv1d(conv_out,activation_buffer,out_channels,
						out_channels,output_length,weight_conv,bias_conv,
						block_kernels[block],conv_dilation,conv_padding);
					free(weight_bg); free(weight_bv); free(bias_conv); free(weight_conv);
					{
						float *swap = activation_buffer;
						activation_buffer = conv_out;
						conv_out = swap;
					}
				}
				for (index=0u; index<(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
					out_channels * output_length; index++)
					next[index] += activation_buffer[index];
				if ( stage == 0u && block == 0u )
				{
					char ref_tag[48];
					snprintf(ref_tag,sizeof(ref_tag),"refa_s0_d%u__2x512x20.f32",
						dilation_index + 1u);
					SparkMinimaxH3V4CompareStageF32(ref_tag,next,
						(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
						output_length);
				}
			}
			snprintf(stage_tag,sizeof(stage_tag),"audio_s%u_b%u",stage,block);
			SparkMinimaxH3V4Stats(stage_tag,next,
				(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
				output_length);
			if ( stage < 2u )
			{
				char ref_tag[48];
				snprintf(ref_tag,sizeof(ref_tag),"refa_s%u_b%u__2x%ux%u.f32",
					stage,block,out_channels,output_length);
				SparkMinimaxH3V4CompareStageF32(ref_tag,next,
					(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
					output_length);
			}
			for (index=0u; index<(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
				out_channels * output_length; index++)
				residual[index] += next[index];
		}
		for (index=0u; index<(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH *
			out_channels * output_length; index++)
			x[index] = residual[index] / 3.0f;
		snprintf(stage_tag,sizeof(stage_tag),"audio_after_stage_%u",stage);
		SparkMinimaxH3V4Stats(stage_tag,x,
			(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
			output_length);
		if ( stage == 0u )
			SparkMinimaxH3V4CompareStageF32("refa_s0_mean__2x512x20.f32",x,
				(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * out_channels *
				output_length);
	}
	{
		float *alpha,*beta,*up_filter,*down_filter;
		uint32_t channels = 8u;
		uint32_t length = SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES;
		SparkMinimaxH3V4LoadWeight(&alpha,weight_dir,"",
			"decoder_activation_post_act_alpha",channels);
		SparkMinimaxH3V4LoadWeight(&beta,weight_dir,"","decoder_activation_post_act_beta",
			channels);
		SparkMinimaxH3V4LoadWeight(&up_filter,weight_dir,"",
			"decoder_activation_post_upsample_filter",12u);
		SparkMinimaxH3V4LoadWeight(&down_filter,weight_dir,"",
			"decoder_activation_post_downsample_lowpass_filter",12u);
		SparkMinimaxH3V4Activation1d(activation_buffer,length,x,channels,length,alpha,
			beta,up_filter,down_filter,upsampled,padded,expanded);
		SparkMinimaxH3V4Stats("audio_after_activation_post",activation_buffer,
			(uint64_t)SPARK_MINIMAX_H3_V4_AUDIO_BATCH * channels * length);
		free(alpha); free(beta); free(up_filter); free(down_filter);
	}
	{
		float *weight_g,*weight_v,*weight;
		SparkMinimaxH3V4LoadWeight(&weight_g,weight_dir,"","decoder_conv_post_weight_g",1u);
		SparkMinimaxH3V4LoadWeight(&weight_v,weight_dir,"","decoder_conv_post_weight_v",
			8u * 7u);
		weight = SparkMinimaxH3V4WnCombine(1u,8u,7u,weight_g,weight_v);
		SparkMinimaxH3V4Stats("conv_post_weight",weight,56u);
		for (batch=0u; batch<SPARK_MINIMAX_H3_V4_AUDIO_BATCH; batch++)
			SparkMinimaxH3V4Conv1d(waveform + (uint64_t)batch *
				SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES,activation_buffer + (uint64_t)batch *
				8u * SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES,1u,8u,
				SPARK_MINIMAX_H3_V4_AUDIO_SAMPLES,weight,0,7u,1u,3u);
		free(weight_g); free(weight_v); free(weight);
	}
	SparkMinimaxH3V4Stats("audio_before_clamp",waveform,wave_elements);
	SparkMinimaxH3V4Stats("audio_expected_wave",expected,wave_elements);
	for (index=0u; index<wave_elements; index++)
	{
		if ( waveform[index] > 1.0f )
			waveform[index] = 1.0f;
		if ( waveform[index] < -1.0f )
			waveform[index] = -1.0f;
	}
	{
		double sum_squared = 0.0;
		double max_abs = 0.0;
		double sum_squared_reference = 0.0;
		uint32_t within;
		double relative;
		for (index=0u; index<wave_elements; index++)
		{
			double difference = (double)waveform[index] - (double)expected[index];
			sum_squared += difference * difference;
			sum_squared_reference += (double)expected[index] * (double)expected[index];
			if ( fabs(difference) > max_abs )
				max_abs = fabs(difference);
		}
		relative = sqrt(sum_squared / (sum_squared_reference + 1e-30));
		within = relative <= SPARK_MINIMAX_H3_V4_AUDIO_MAX_REL;
		printf("%-16s rel=%.3e max_abs=%.6f %s\n","audio_vae",relative,max_abs,
			within != 0u ? "OK" : "FAIL");
		if ( within == 0u )
			SparkMinimaxH3V4Failures++;
	}
	free(latents); free(expected); free(x); free(next); free(residual);
	free(activation_buffer); free(conv_out); free(upsampled); free(padded);
	free(expanded); free(waveform);
}

int main(int argc, char **argv)
{
	if ( argc != 4 )
	{
		printf("usage: %s video FIXTURE_DIR WEIGHTS_DIR | %s audio FIXTURE_DIR WEIGHTS_DIR\n",
			argv[0],argv[0]);
		return(2);
	}
	printf("minimax_h3 V4 gate: real-weight VAE decode vs anchor fixture (%s)\n",argv[1]);
	if ( strcmp(argv[1],"video") == 0 )
		SparkMinimaxH3V4VideoGate(argv[2],argv[3]);
	else if ( strcmp(argv[1],"audio") == 0 )
		SparkMinimaxH3V4AudioGate(argv[2],argv[3]);
	else
	{
		printf("unknown mode %s FAIL\n",argv[1]);
		return(2);
	}
	if ( SparkMinimaxH3V4Failures != 0u )
	{
		printf("V4 gate FAIL (%u failures)\n",SparkMinimaxH3V4Failures);
		return(1);
	}
	printf("V4 gate OK\n");
	return(0);
}
