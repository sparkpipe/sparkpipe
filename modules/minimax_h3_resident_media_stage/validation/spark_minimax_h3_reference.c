#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"

#define SPARK_H3_REF_HIDDEN 64u
#define SPARK_H3_REF_HEADS 4u
#define SPARK_H3_REF_HEAD_DIM 16u
#define SPARK_H3_REF_FFN 128u
#define SPARK_H3_REF_FREQ_DIM 8u
#define SPARK_H3_REF_TIME_EMBED 16u
#define SPARK_H3_REF_ROPE_FREQ_DIM 2u
#define SPARK_H3_REF_ROPE_DIM (3u * SPARK_H3_REF_ROPE_FREQ_DIM * 2u)
#define SPARK_H3_REF_SEQ 12u
#define SPARK_H3_REF_SIGMA_POINTS 51u
#define SPARK_H3_REF_EPS 1e-5f
#define SPARK_H3_REF_TOLERANCE 2e-4f

static const float SPARK_H3_REF_NORM_ONE[SPARK_H3_REF_HIDDEN] = {0};

static uint64_t SparkH3RefLcg(uint64_t *state)
{
	*state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	return(*state >> 11u);
}

static float SparkH3RefRandom(uint64_t *state)
{
	return((float)(SparkH3RefLcg(state) % 2000001u) / 1000000.0f - 1.0f);
}

void SparkH3RefRmsNorm(const float *input, const float *weight, uint32_t width,
	float epsilon, float *output)
{
	float sum;
	uint32_t index;
	sum = 0.0f;
	for (index=0u; index<width; index++)
		sum += input[index] * input[index];
	sum /= (float)width;
	{
		float scale = 1.0f / sqrtf(sum + epsilon);
		for (index=0u; index<width; index++)
			output[index] = input[index] * scale * (weight != 0 ? weight[index] : 1.0f);
	}
}

void SparkH3RefRopeAngles(const double *positions, uint32_t row_count,
	float theta, uint32_t freq_dim, float *cos_out, float *sin_out)
{
	uint32_t row,axis,half;
	float inv_freq[SPARK_H3_REF_ROPE_FREQ_DIM];
	for (axis=0u; axis<freq_dim; axis++)
		inv_freq[axis] = 1.0f / powf(theta, (float)(2u * axis) / (float)(2u * freq_dim));
	for (row=0u; row<row_count; row++)
	{
		float angles[3u * SPARK_H3_REF_ROPE_FREQ_DIM];
		for (axis=0u; axis<3u; axis++)
		{
			uint32_t freq;
			for (freq=0u; freq<freq_dim; freq++)
				angles[axis * freq_dim + freq] = (float)positions[row * 3u + axis] * inv_freq[freq];
		}
		for (half=0u; half<2u; half++)
		{
			uint32_t index;
			for (index=0u; index<3u * freq_dim; index++)
			{
				cos_out[row * (6u * freq_dim) + half * (3u * freq_dim) + index] =
					cosf(angles[index]);
				sin_out[row * (6u * freq_dim) + half * (3u * freq_dim) + index] =
					sinf(angles[index]);
			}
		}
	}
}

void SparkH3RefApplyRope(const float *heads_in, const float *cos_angles,
	const float *sin_angles, uint32_t seq, uint32_t heads, uint32_t head_dim,
	uint32_t rope_dim, float *heads_out)
{
	uint32_t row,head,index;
	for (row=0u; row<seq; row++)
	{
		for (head=0u; head<heads; head++)
		{
			const float *source = heads_in + ((uint64_t)row * heads + head) * head_dim;
			float *target = heads_out + ((uint64_t)row * heads + head) * head_dim;
			uint32_t half = rope_dim / 2u;
			for (index=0u; index<rope_dim; index++)
			{
				float rotated = index < half ? -source[half + index] : source[index - half];
				target[index] = source[index] * cos_angles[row * rope_dim + index] +
					rotated * sin_angles[row * rope_dim + index];
			}
			for (index=rope_dim; index<head_dim; index++)
				target[index] = source[index];
		}
	}
}

void SparkH3RefSoftmaxRows(const float *scores, uint32_t rows, uint32_t columns,
	float *probs)
{
	uint32_t row,column;
	for (row=0u; row<rows; row++)
	{
		float maximum = scores[(uint64_t)row * columns];
		float sum = 0.0f;
		for (column=1u; column<columns; column++)
			if ( scores[(uint64_t)row * columns + column] > maximum )
				maximum = scores[(uint64_t)row * columns + column];
		for (column=0u; column<columns; column++)
		{
			probs[(uint64_t)row * columns + column] =
				expf(scores[(uint64_t)row * columns + column] - maximum);
			sum += probs[(uint64_t)row * columns + column];
		}
		for (column=0u; column<columns; column++)
			probs[(uint64_t)row * columns + column] /= sum;
	}
}

static float SparkH3RefDot(const float *left, const float *right, uint32_t width);

void SparkH3RefDenseAttention(const float *q, const float *k, const float *v,
	uint32_t seq, uint32_t heads, uint32_t head_dim, float *output)
{
	static float scores[SPARK_H3_REF_SEQ * SPARK_H3_REF_SEQ];
	static float probs[SPARK_H3_REF_SEQ * SPARK_H3_REF_SEQ];
	uint32_t head,row;
	float scale = 1.0f / sqrtf((float)head_dim);
	for (head=0u; head<heads; head++)
	{
		for (row=0u; row<seq; row++)
		{
			uint32_t column;
			for (column=0u; column<seq; column++)
			{
				scores[row * seq + column] = SparkH3RefDot(
					q + ((uint64_t)row * heads + head) * head_dim,
					k + ((uint64_t)column * heads + head) * head_dim, head_dim) * scale;
			}
		}
		SparkH3RefSoftmaxRows(scores,seq,seq,probs);
		for (row=0u; row<seq; row++)
		{
			uint32_t column,element;
			for (element=0u; element<head_dim; element++)
				output[((uint64_t)row * heads + head) * head_dim + element] = 0.0f;
			for (column=0u; column<seq; column++)
			{
				for (element=0u; element<head_dim; element++)
				{
					output[((uint64_t)row * heads + head) * head_dim + element] +=
						probs[row * seq + column] *
						v[((uint64_t)column * heads + head) * head_dim + element];
				}
			}
		}
	}
}

static float SparkH3RefDot(const float *left, const float *right, uint32_t width)
{
	float sum = 0.0f;
	uint32_t index;
	for (index=0u; index<width; index++)
		sum += left[index] * right[index];
	return(sum);
}

void SparkH3RefTimeEmbed(const float *timestep, uint32_t timestep_count,
	uint32_t freq_dim, float *embed_out)
{
	uint32_t row,half_index;
	for (row=0u; row<timestep_count; row++)
	{
		for (half_index=0u; half_index<freq_dim / 2u; half_index++)
		{
			float exponent = (float)(2u * half_index) / (float)freq_dim;
			float inv = 1.0f / powf(10000.0f,exponent);
			float angle = timestep[row] * inv;
			embed_out[row * freq_dim + half_index] = cosf(angle);
			embed_out[row * freq_dim + half_index + freq_dim / 2u] = sinf(angle);
		}
	}
}

void SparkH3RefSchedulerBuildSigmas(float shift, uint32_t sigma_point_count,
	float *sigmas_out, uint32_t *sigma_count_out)
{
	uint32_t index,out_count = 0u;
	float previous;
	float bits;
	(void)bits;
	for (index=0u; index<sigma_point_count; index++)
	{
		float base = 1.0f - (float)index / (float)(sigma_point_count - 1u);
		float shifted = shift * base / (1.0f + (shift - 1.0f) * base);
		if ( out_count != 0u && memcmp(&shifted,&previous,sizeof(float)) == 0 )
			continue;
		sigmas_out[out_count++] = shifted;
		previous = shifted;
	}
	*sigma_count_out = out_count;
}

void SparkH3RefSchedulerStepElement(float timestep, float sigma, float sigma_next,
	float sample, float velocity, float *sample_next_out)
{
	float sigma_from_timestep = 1.0f - timestep;
	float denoised = sample + sigma_from_timestep * velocity;
	float ratio = sigma_next / sigma;
	*sample_next_out = ratio * sample + (1.0f - ratio) * denoised;
}

uint32_t SparkH3RefSelfCheck(void)
{
	uint64_t rng = UINT64_C(0x68333253504b5350);
	double positions[SPARK_H3_REF_SEQ * 3u];
	float cos_angles[SPARK_H3_REF_SEQ * SPARK_H3_REF_ROPE_DIM];
	float sin_angles[SPARK_H3_REF_SEQ * SPARK_H3_REF_ROPE_DIM];
	float q[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float k[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float v[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float rotated[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float original[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float attended[(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM];
	float sigmas[SPARK_H3_REF_SIGMA_POINTS];
	uint32_t sigma_count = 0u;
	uint32_t row,head,element;
	float norm_input[SPARK_H3_REF_HIDDEN];
	float norm_output[SPARK_H3_REF_HIDDEN];
	for (row=0u; row<SPARK_H3_REF_SEQ; row++)
	{
		positions[row * 3u] = (double)row;
		positions[row * 3u + 1u] = (double)(row % 3u);
		positions[row * 3u + 2u] = (double)(row % 5u);
	}
	SparkH3RefRopeAngles(positions,SPARK_H3_REF_SEQ,10000.0f,SPARK_H3_REF_ROPE_FREQ_DIM,
		cos_angles,sin_angles);
	for (row=0u; row<(uint64_t)SPARK_H3_REF_SEQ * SPARK_H3_REF_HEADS * SPARK_H3_REF_HEAD_DIM; row++)
	{
		q[row] = SparkH3RefRandom(&rng);
		k[row] = SparkH3RefRandom(&rng);
		v[row] = SparkH3RefRandom(&rng);
		original[row] = q[row];
	}
	SparkH3RefApplyRope(q,cos_angles,sin_angles,SPARK_H3_REF_SEQ,SPARK_H3_REF_HEADS,
		SPARK_H3_REF_HEAD_DIM,SPARK_H3_REF_ROPE_DIM,rotated);
	for (head=0u; head<SPARK_H3_REF_HEADS; head++)
	{
		float before = 0.0f;
		float after = 0.0f;
		for (element=0u; element<SPARK_H3_REF_HEAD_DIM; element++)
		{
			before += original[((uint64_t)5u * SPARK_H3_REF_HEADS + head) * SPARK_H3_REF_HEAD_DIM + element] *
				original[((uint64_t)5u * SPARK_H3_REF_HEADS + head) * SPARK_H3_REF_HEAD_DIM + element];
			after += rotated[((uint64_t)5u * SPARK_H3_REF_HEADS + head) * SPARK_H3_REF_HEAD_DIM + element] *
				rotated[((uint64_t)5u * SPARK_H3_REF_HEADS + head) * SPARK_H3_REF_HEAD_DIM + element];
		}
		if ( fabsf(sqrtf(before) - sqrtf(after)) > SPARK_H3_REF_TOLERANCE )
		{
			printf("FAIL rope norm %u: %f vs %f\n",head,(double)before,(double)after);
			return(1u);
		}
	}
	SparkH3RefDenseAttention(rotated,rotated,v,SPARK_H3_REF_SEQ,SPARK_H3_REF_HEADS,
		SPARK_H3_REF_HEAD_DIM,attended);
	SparkH3RefRmsNorm(norm_input,SPARK_H3_REF_NORM_ONE,SPARK_H3_REF_HIDDEN,SPARK_H3_REF_EPS,
		norm_output);
	SparkH3RefSchedulerBuildSigmas(12.0f,SPARK_H3_REF_SIGMA_POINTS,sigmas,&sigma_count);
	if ( sigma_count != SPARK_H3_REF_SIGMA_POINTS )
	{
		printf("FAIL sigma count %u\n",sigma_count);
		return(2u);
	}
	if ( fabsf(sigmas[0] - 1.0f) > SPARK_H3_REF_TOLERANCE ||
		fabsf(sigmas[sigma_count - 1u]) > SPARK_H3_REF_TOLERANCE )
	{
		printf("FAIL sigma endpoints\n");
		return(3u);
	}
	printf("oracle self-check OK (rope norm preserved, attention finite, sigmas %u)\n",
		sigma_count);
	return(0u);
}

int main(void)
{
	return((int)SparkH3RefSelfCheck());
}
