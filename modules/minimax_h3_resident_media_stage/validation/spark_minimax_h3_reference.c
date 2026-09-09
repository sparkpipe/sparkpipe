#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
#define SPARK_H3_REF_CONV_C 4u
#define SPARK_H3_REF_CONV_H 8u
#define SPARK_H3_REF_CONV_W 8u

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

void SparkH3RefAdaLNAffine(const float *input, const float *scale, const float *shift,
	uint32_t rows, uint32_t width, float *output)
{
	uint32_t row,column;
	for (row=0u; row<rows; row++)
	{
		for (column=0u; column<width; column++)
		{
			output[(uint64_t)row * width + column] =
				input[(uint64_t)row * width + column] * (1.0f + scale[column]) +
				shift[column];
		}
	}
}

void SparkH3RefGateResidual(const float *value, const float *gate, const float *residual,
	uint32_t rows, uint32_t width, float *output)
{
	uint32_t row,column;
	for (row=0u; row<rows; row++)
	{
		for (column=0u; column<width; column++)
		{
			output[(uint64_t)row * width + column] =
				residual[(uint64_t)row * width + column] +
				gate[column] * value[(uint64_t)row * width + column];
		}
	}
}

void SparkH3RefSiluMul(const float *gate_up, uint32_t rows, uint32_t ffn, float *output)
{
	uint32_t row,column;
	for (row=0u; row<rows; row++)
	{
		for (column=0u; column<ffn; column++)
		{
			float gate = gate_up[(uint64_t)row * (2u * ffn) + column];
			float up = gate_up[(uint64_t)row * (2u * ffn) + ffn + column];
			output[(uint64_t)row * ffn + column] = gate / (1.0f + expf(-gate)) * up;
		}
	}
}

void SparkH3RefConv2d3x3Reflect(const float *input, const float *weight, const float *bias,
	uint32_t in_channels, uint32_t out_channels, uint32_t height, uint32_t width,
	float *output)
{
	uint32_t out_channel,in_channel,y,x,ky,kx;
	for (out_channel=0u; out_channel<out_channels; out_channel++)
	{
		for (y=0u; y<height; y++)
		{
			for (x=0u; x<width; x++)
			{
				float accumulator = bias != 0 ? bias[out_channel] : 0.0f;
				for (in_channel=0u; in_channel<in_channels; in_channel++)
				{
					for (ky=0u; ky<3u; ky++)
					{
						int32_t sy = (int32_t)y + (int32_t)ky - 1;
						if ( sy < 0 )
							sy = -sy;
						if ( sy >= (int32_t)height )
							sy = 2 * (int32_t)height - 2 - sy;
						for (kx=0u; kx<3u; kx++)
						{
							int32_t sx = (int32_t)x + (int32_t)kx - 1;
							if ( sx < 0 )
								sx = -sx;
							if ( sx >= (int32_t)width )
								sx = 2 * (int32_t)width - 2 - sx;
							accumulator += input[((uint64_t)in_channel * height +
								(uint64_t)sy) * width + (uint64_t)sx] *
								weight[(((uint64_t)out_channel * in_channels +
								in_channel) * 3u + ky) * 3u + kx];
						}
					}
				}
				output[((uint64_t)out_channel * height + y) * width + x] = accumulator;
			}
		}
	}
}

void SparkH3RefConv1dDilated(const float *input, const float *weight, const float *bias,
	uint32_t channels, uint32_t length, uint32_t dilation, float *output)
{
	uint32_t channel,position,tap;
	for (channel=0u; channel<channels; channel++)
	{
		for (position=0u; position<length; position++)
		{
			float accumulator = bias != 0 ? bias[channel] : 0.0f;
			for (tap=0u; tap<3u; tap++)
			{
				int32_t source = (int32_t)position + (int32_t)(tap * dilation) - (int32_t)dilation;
				if ( source >= 0 && source < (int32_t)length )
					accumulator += input[(uint64_t)channel * length + (uint32_t)source] *
						weight[(uint64_t)channel * 3u + tap];
			}
			output[(uint64_t)channel * length + position] = accumulator;
		}
	}
}

void SparkH3RefSnake(const float *input, const float *alpha, uint32_t channels,
	uint32_t length, float *output)
{
	uint32_t channel,index;
	for (channel=0u; channel<channels; channel++)
	{
		float alpha_value = alpha[channel];
		for (index=0u; index<length; index++)
		{
			float value = input[(uint64_t)channel * length + index];
			float sine = sinf(alpha_value * value);
			output[(uint64_t)channel * length + index] =
				value + sine * sine / alpha_value;
		}
	}
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
	{
		float conv_in[SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_H * SPARK_H3_REF_CONV_W];
		float conv_weight[SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_C * 9u];
		float conv_out[SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_H * SPARK_H3_REF_CONV_W];
		float conv_bias[SPARK_H3_REF_CONV_C];
		uint32_t index;
		for (index=0u; index<SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_H * SPARK_H3_REF_CONV_W; index++)
		{
			conv_in[index] = SparkH3RefRandom(&rng);
			conv_out[index] = 0.0f;
		}
		for (index=0u; index<SPARK_H3_REF_CONV_C; index++)
			conv_bias[index] = 0.0f;
		for (index=0u; index<SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_C * 9u; index++)
			conv_weight[index] = 0.0f;
		for (index=0u; index<SPARK_H3_REF_CONV_C; index++)
			conv_weight[((uint64_t)index * SPARK_H3_REF_CONV_C + index) * 9u + 4u] = 1.0f;
		SparkH3RefConv2d3x3Reflect(conv_in,conv_weight,conv_bias,SPARK_H3_REF_CONV_C,
			SPARK_H3_REF_CONV_C,SPARK_H3_REF_CONV_H,SPARK_H3_REF_CONV_W,conv_out);
		for (index=0u; index<SPARK_H3_REF_CONV_C * SPARK_H3_REF_CONV_H * SPARK_H3_REF_CONV_W; index++)
		{
			if ( fabsf(conv_out[index] - conv_in[index]) > SPARK_H3_REF_TOLERANCE )
			{
				printf("FAIL conv2d identity %u\n",index);
				return(4u);
			}
		}
	}
	{
		float snake_in[SPARK_H3_REF_SEQ];
		float snake_alpha[1u];
		float snake_out[SPARK_H3_REF_SEQ];
		uint32_t index;
		snake_alpha[0] = 1.0f;
		for (index=0u; index<SPARK_H3_REF_SEQ; index++)
			snake_in[index] = 0.0f;
		SparkH3RefSnake(snake_in,snake_alpha,1u,SPARK_H3_REF_SEQ,snake_out);
		for (index=0u; index<SPARK_H3_REF_SEQ; index++)
		{
			if ( fabsf(snake_out[index]) > SPARK_H3_REF_TOLERANCE )
			{
				printf("FAIL snake zero %u\n",index);
				return(5u);
			}
		}
	}
	printf("oracle self-check OK (rope norm preserved, attention finite, sigmas %u, "
		"conv2d identity, snake zero)\n",sigma_count);
	return(0u);
}

#ifndef SPARK_H3_REF_EMBED
int main(void)
{
	return((int)SparkH3RefSelfCheck());
}
#endif
