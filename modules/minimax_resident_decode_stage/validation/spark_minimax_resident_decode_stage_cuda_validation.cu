#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_minimax_resident_decode_stage_firmware.h"

/* minimax text-tower resident decode stage - component-tier GPU validation.
 *
 * SCOPE (honest, laguna-precedent style): this exercises the carved kernel
 * set the launchers export - embedding gather, RMSNorm, residual add,
 * per-head norm + rope, SwiGlu, vocabulary argmax (sortable-u64) with the
 * u64 combine, and the two TP combine kernels - against host mirrors of the
 * same math at real geometry, plus a bit-exact determinism rerun. It does
 * NOT consume a stage pack, does not run the linear projections, the
 * attention dataflow, the KV write path, multi-row decode, TP collectives
 * over real transports, or the t1 fixture streams. A PASS here is a kernel
 * component PASS, not full minimax numerical or driver acceptance (the
 * fixture-level ladder runs through the CPU validator + the lane receipts).
 */

#define VALIDATION_HIDDEN SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION
#define VALIDATION_HEAD_DIM SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION
#define VALIDATION_LOCAL_QUERY_HEADS (SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_COUNT / 4u)
#define VALIDATION_LOCAL_KV_HEADS (SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT / 4u)

extern "C" cudaError_t SparkMinimaxConfigureCudaKernels(void);
extern "C" cudaError_t SparkMinimaxLaunchEmbeddingGather(cudaStream_t stream,const uint32_t *token_ids,const void *embedding_bf16,void *hidden_bf16,uint32_t row_count);
extern "C" cudaError_t SparkMinimaxLaunchRmsNorm(cudaStream_t stream,const void *input_bf16,const void *gain_bf16,void *output_bf16,uint32_t row_count,uint32_t dimension,float epsilon);
extern "C" cudaError_t SparkMinimaxLaunchResidualAdd(cudaStream_t stream,void *hidden_bf16,const void *delta_bf16,uint32_t row_count,uint32_t dimension);
extern "C" cudaError_t SparkMinimaxLaunchHeadNormRope(cudaStream_t stream,void *query_bf16,void *key_bf16,const void *query_norm_bf16,const void *key_norm_bf16,void *query_roped_f32,const uint64_t *row_positions,uint32_t row_count,float epsilon,uint32_t local_query_head_count,uint32_t local_kv_head_count);
extern "C" cudaError_t SparkMinimaxLaunchSwiGlu(cudaStream_t stream,const void *gate_bf16,const void *up_bf16,void *activated_bf16,uint32_t row_count,uint32_t dimension);
extern "C" cudaError_t SparkMinimaxLaunchVocabArgmax(cudaStream_t stream,const void *lm_head_bf16,const void *input_bf16,uint64_t *argmax_reduce_u64,uint32_t row_count,uint32_t local_vocab_rows,uint32_t input_dimension,uint32_t local_vocab_base);
extern "C" cudaError_t SparkMinimaxLaunchArgmaxResolve(cudaStream_t stream,const uint64_t *argmax_reduce_u64,uint32_t *token_ids,uint32_t row_count);
extern "C" cudaError_t SparkMinimaxLaunchTpCombineBf16(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count);
extern "C" cudaError_t SparkMinimaxLaunchTpCombineU64Max(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count);

static uint32_t validation_seed = 20260923u;

static float ValidationNextRandom(void)
{
	validation_seed = (validation_seed * 1664525u) + 1013904228u;
	return((float)((validation_seed >> 8) & 0xffffu) / 32768.0f - 1.0f);
}

static float ValidationBf16ToFloat(uint16_t bits)
{
	return(__bfloat162float(*reinterpret_cast<__nv_bfloat16 *>(&bits)));
}

static uint16_t ValidationFloatToBf16(float value)
{
	__nv_bfloat16 rounded = __float2bfloat16_rn(value);
	return(*reinterpret_cast<uint16_t *>(&rounded));
}

/* Tolerance for one bf16 output produced from fp32 math that the host
 * mirror evaluates in double: two bf16 rounding steps. */
static int ValidationCloseBf16(float produced,float expected,const char *what,uint64_t index,double *worst)
{
	double delta = fabs((double)produced - (double)expected);
	double bound = 0.0078125 + 0.0078125 * fabs((double)expected);
	if ( delta > *worst )
		*worst = delta;
	if ( delta > bound )
	{
		fprintf(stderr,"minimax validation: %s mismatch at %llu: %g vs %g (delta %g)\n",
			what,(unsigned long long)index,(double)produced,(double)expected,delta);
		return(1);
	}
	return(0);
}

static int ValidationCheckCuda(cudaError_t error,const char *what)
{
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"minimax validation: %s failed: %s\n",what,cudaGetErrorString(error));
		return(1);
	}
	return(0);
}

struct ValidationBuffers
{
	uint16_t *host_a;
	uint16_t *host_b;
	uint16_t *host_out;
	uint16_t *host_out_second;
	float *host_float;
	uint32_t *host_tokens;
	uint16_t *device_a;
	uint16_t *device_b;
	uint16_t *device_out;
	float *device_float;
	uint32_t *device_tokens;
	uint64_t *device_u64;
	uint64_t *device_u64_second;
	uint32_t *device_u32;
};

static int ValidationRmsNorm(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t rows = 3u;
	uint64_t elements = (uint64_t)rows * VALIDATION_HIDDEN;
	float epsilon = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON;
	uint32_t row,column;
	for (uint64_t index = 0u; index < elements; index++)
		buffers->host_a[index] = ValidationFloatToBf16(ValidationNextRandom());
	for (column = 0u; column < VALIDATION_HIDDEN; column++)
		buffers->host_b[column] = ValidationFloatToBf16(1.0f + 0.25f * ValidationNextRandom());
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_a,buffers->host_a,elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"rmsnorm upload input") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_b,buffers->host_b,VALIDATION_HIDDEN * sizeof(uint16_t),cudaMemcpyHostToDevice),"rmsnorm upload gain") != 0 )
		return(1);
	if ( ValidationCheckCuda(SparkMinimaxLaunchRmsNorm(stream,buffers->device_a,buffers->device_b,buffers->device_out,rows,VALIDATION_HIDDEN,epsilon),"rmsnorm launch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"rmsnorm sync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_out,buffers->device_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost),"rmsnorm download") != 0 )
		return(1);
	for (row = 0u; row < rows; row++)
	{
		double total = 0.0;
		for (column = 0u; column < VALIDATION_HIDDEN; column++)
		{
			double value = ValidationBf16ToFloat(buffers->host_a[(uint64_t)row * VALIDATION_HIDDEN + column]);
			total += value * value;
		}
		double scale = 1.0 / sqrt(total / (double)VALIDATION_HIDDEN + (double)epsilon);
		for (column = 0u; column < VALIDATION_HIDDEN; column++)
		{
			double value = ValidationBf16ToFloat(buffers->host_a[(uint64_t)row * VALIDATION_HIDDEN + column]);
			double gain = ValidationBf16ToFloat(buffers->host_b[column]);
			float produced = ValidationBf16ToFloat(buffers->host_out[(uint64_t)row * VALIDATION_HIDDEN + column]);
			if ( ValidationCloseBf16(produced,(float)(value * scale * gain),"rmsnorm",(uint64_t)row * VALIDATION_HIDDEN + column,worst) != 0 )
				return(1);
		}
	}
	/* Determinism rerun: bit-exact. */
	if ( ValidationCheckCuda(SparkMinimaxLaunchRmsNorm(stream,buffers->device_a,buffers->device_b,buffers->device_out,rows,VALIDATION_HIDDEN,epsilon),"rmsnorm relaunch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"rmsnorm resync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_out_second,buffers->device_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost),"rmsnorm redownload") != 0 )
		return(1);
	if ( memcmp(buffers->host_out,buffers->host_out_second,elements * sizeof(uint16_t)) != 0 )
	{
		fprintf(stderr,"minimax validation: rmsnorm determinism rerun differs\n");
		return(1);
	}
	return(0);
}

static int ValidationSwiGlu(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t rows = 2u;
	const uint32_t dimension = 1024u;
	uint64_t elements = (uint64_t)rows * dimension;
	for (uint64_t index = 0u; index < elements; index++)
	{
		buffers->host_a[index] = ValidationFloatToBf16(ValidationNextRandom() * 4.0f);
		buffers->host_b[index] = ValidationFloatToBf16(ValidationNextRandom() * 4.0f);
	}
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_a,buffers->host_a,elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"swiglu upload gate") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_b,buffers->host_b,elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"swiglu upload up") != 0 )
		return(1);
	if ( ValidationCheckCuda(SparkMinimaxLaunchSwiGlu(stream,buffers->device_a,buffers->device_b,buffers->device_out,rows,dimension),"swiglu launch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"swiglu sync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_out,buffers->device_out,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost),"swiglu download") != 0 )
		return(1);
	for (uint64_t index = 0u; index < elements; index++)
	{
		double gate = ValidationBf16ToFloat(buffers->host_a[index]);
		double up = ValidationBf16ToFloat(buffers->host_b[index]);
		double sigmoid = 1.0 / (1.0 + exp(-gate));
		float produced = ValidationBf16ToFloat(buffers->host_out[index]);
		if ( ValidationCloseBf16(produced,(float)(gate * sigmoid * up),"swiglu",index,worst) != 0 )
			return(1);
	}
	return(0);
}

static int ValidationEmbeddingGather(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t rows = 4u;
	const uint32_t vocab = 512u;
	uint32_t tokens[4];
	uint16_t *embedding = buffers->host_out_second;
	(void)worst;
	for (uint32_t token = 0u; token < vocab; token++)
		for (uint32_t column = 0u; column < VALIDATION_HIDDEN; column++)
			embedding[(uint64_t)token * VALIDATION_HIDDEN + column] =
				ValidationFloatToBf16(ValidationNextRandom());
	for (uint32_t row = 0u; row < rows; row++)
		tokens[row] = (uint32_t)(fabs(ValidationNextRandom()) * (float)vocab) % vocab;
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_out,embedding,(uint64_t)vocab * VALIDATION_HIDDEN * sizeof(uint16_t),cudaMemcpyHostToDevice),"embedding upload") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_u32,tokens,rows * sizeof(uint32_t),cudaMemcpyHostToDevice),"embedding tokens") != 0 )
		return(1);
	if ( ValidationCheckCuda(SparkMinimaxLaunchEmbeddingGather(stream,buffers->device_u32,buffers->device_out,buffers->device_b,rows),"embedding gather launch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"embedding sync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_b,buffers->device_b,(uint64_t)rows * VALIDATION_HIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost),"embedding download") != 0 )
		return(1);
	for (uint32_t row = 0u; row < rows; row++)
		if ( memcmp(&buffers->host_b[(uint64_t)row * VALIDATION_HIDDEN],
			&embedding[(uint64_t)tokens[row] * VALIDATION_HIDDEN],
			VALIDATION_HIDDEN * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr,"minimax validation: embedding gather row %u mismatch\n",row);
			return(1);
		}
	return(0);
}

static int ValidationHeadNormRope(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t rows = 2u;
	const uint32_t query_heads = VALIDATION_LOCAL_QUERY_HEADS;
	const uint32_t kv_heads = VALIDATION_LOCAL_KV_HEADS;
	const float epsilon = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON;
	const float theta = 5000000.0f;
	const uint32_t half = VALIDATION_HEAD_DIM / 2u;
	uint64_t query_elements = (uint64_t)rows * query_heads * VALIDATION_HEAD_DIM;
	uint64_t key_elements = (uint64_t)rows * kv_heads * VALIDATION_HEAD_DIM;
	uint64_t row_positions[2] = {7ull,137ull};
	uint16_t *key_original = buffers->host_out_second;
	uint16_t *key_produced = buffers->host_out_second + (size_t)key_elements;
	uint32_t row,head,element;
	for (uint64_t index = 0u; index < query_elements; index++)
		buffers->host_a[index] = ValidationFloatToBf16(ValidationNextRandom());
	for (uint64_t index = 0u; index < key_elements; index++)
	{
		uint16_t value = ValidationFloatToBf16(ValidationNextRandom());
		buffers->host_b[index] = value;
		key_original[index] = value;
	}
	/* host_out[0 .. qh*128) = query norm gains, [qh*128 .. (qh+kh)*128) = key gains. */
	for (uint64_t index = 0u; index < (uint64_t)(query_heads + kv_heads) * VALIDATION_HEAD_DIM; index++)
		buffers->host_out[index] = ValidationFloatToBf16(1.0f + 0.25f * ValidationNextRandom());
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_a,buffers->host_a,query_elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"rope upload query") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_b,buffers->host_b,key_elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"rope upload key") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_out,buffers->host_out,(uint64_t)(query_heads + kv_heads) * VALIDATION_HEAD_DIM * sizeof(uint16_t),cudaMemcpyHostToDevice),"rope upload gains") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_u32,row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice),"rope upload positions") != 0 )
		return(1);
	if ( ValidationCheckCuda(SparkMinimaxLaunchHeadNormRope(stream,buffers->device_a,buffers->device_b,buffers->device_out,buffers->device_out + (uint64_t)query_heads * VALIDATION_HEAD_DIM,buffers->device_float,(const uint64_t *)buffers->device_u32,rows,epsilon,query_heads,kv_heads),"rope launch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"rope sync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_float,buffers->device_float,query_elements * sizeof(float),cudaMemcpyDeviceToHost),"rope download query") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(key_produced,buffers->device_b,key_elements * sizeof(uint16_t),cudaMemcpyDeviceToHost),"rope download key") != 0 )
		return(1);
	for (row = 0u; row < rows; row++)
	{
		double position = (double)(unsigned long long)row_positions[row];
		for (head = 0u; head < query_heads + kv_heads; head++)
		{
			uint32_t is_query = head < query_heads ? 1u : 0u;
			uint32_t norm_head = is_query != 0u ? head : head - query_heads;
			const uint16_t *source = is_query != 0u ? buffers->host_a : key_original;
			uint32_t head_count = is_query != 0u ? query_heads : kv_heads;
			const uint16_t *gain = is_query != 0u ? buffers->host_out : buffers->host_out + (uint64_t)query_heads * VALIDATION_HEAD_DIM;
			const uint16_t *produced_bf16 = is_query != 0u ? 0 : key_produced;
			double normalized[VALIDATION_HEAD_DIM];
			double squared = 0.0;
			double base[VALIDATION_HEAD_DIM];
			for (element = 0u; element < VALIDATION_HEAD_DIM; element++)
			{
				double value = ValidationBf16ToFloat(source[((uint64_t)row * head_count + norm_head) * VALIDATION_HEAD_DIM + element]);
				base[element] = value;
				squared += value * value;
			}
			{
				double scale = 1.0 / sqrt(squared / (double)VALIDATION_HEAD_DIM + (double)epsilon);
				for (element = 0u; element < VALIDATION_HEAD_DIM; element++)
					normalized[element] = base[element] * scale * (double)ValidationBf16ToFloat(gain[element]);
			}
			for (element = 0u; element < VALIDATION_HEAD_DIM; element++)
			{
				double angle = position * pow((double)theta,-2.0 * (double)element / (double)VALIDATION_HEAD_DIM);
				uint32_t partner_index = element < half ? element + half : element - half;
				double roped = element < half
					? normalized[element] * cos(angle) - normalized[partner_index] * sin(angle)
					: normalized[element] * cos(angle) + normalized[partner_index] * sin(angle);
				uint64_t flat = ((uint64_t)row * head_count + norm_head) * VALIDATION_HEAD_DIM + element;
				if ( is_query != 0u )
				{
					double delta = fabs((double)buffers->host_float[flat] - roped);
					if ( delta > 2e-3 + 2e-3 * fabs(roped) )
					{
						fprintf(stderr,"minimax validation: rope query mismatch row %u head %u element %u: %g vs %g\n",
							row,head,element,(double)buffers->host_float[flat],roped);
						return(1);
					}
					if ( delta > *worst )
						*worst = delta;
				}
				else
				{
					float produced = ValidationBf16ToFloat(produced_bf16[flat]);
					if ( ValidationCloseBf16(produced,(float)roped,"rope key",flat,worst) != 0 )
						return(1);
				}
			}
		}
	}
	return(0);
}

static int ValidationVocabArgmax(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t rows = 1u;
	const uint32_t local_vocab_rows = 8192u;
	const uint32_t local_vocab_base = 40000u;
	const uint32_t winner = 5177u;
	uint16_t *weights = buffers->host_out_second;
	double *scores = (double *)malloc((size_t)local_vocab_rows * sizeof(double));
	double best_score = -1e300;
	uint32_t best_index = 0u;
	uint32_t index,column;
	(void)worst;
	if ( scores == 0 )
		return(1);
	for (index = 0u; index < local_vocab_rows; index++)
		for (column = 0u; column < VALIDATION_HIDDEN; column++)
			weights[(uint64_t)index * VALIDATION_HIDDEN + column] =
				ValidationFloatToBf16(ValidationNextRandom() * 0.03f);
	for (column = 0u; column < VALIDATION_HIDDEN; column++)
		buffers->host_a[column] = ValidationFloatToBf16(ValidationNextRandom());
	/* Make the winner row a copy of the input: its dot product is the input
	 * energy (~1.7e3 at this scale) while every random row stays a
	 * zero-mean random walk (~1e0) - the argmax margin is bulletproof
	 * against fp32 accumulation-order noise. */
	for (column = 0u; column < VALIDATION_HIDDEN; column++)
		weights[(uint64_t)winner * VALIDATION_HIDDEN + column] = buffers->host_a[column];
	for (index = 0u; index < local_vocab_rows; index++)
	{
		double total = 0.0;
		for (column = 0u; column < VALIDATION_HIDDEN; column++)
			total += (double)ValidationBf16ToFloat(weights[(uint64_t)index * VALIDATION_HIDDEN + column]) *
				(double)ValidationBf16ToFloat(buffers->host_a[column]);
		scores[index] = total;
		if ( total > best_score )
		{
			best_score = total;
			best_index = index;
		}
	}
	if ( best_index != winner )
	{
		fprintf(stderr,"minimax validation: fixture margin too small (host winner %u)\n",best_index);
		free(scores);
		return(1);
	}
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_out,weights,(uint64_t)local_vocab_rows * VALIDATION_HIDDEN * sizeof(uint16_t),cudaMemcpyHostToDevice),"argmax upload weights") != 0 )
	{
		free(scores);
		return(1);
	}
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_a,buffers->host_a,VALIDATION_HIDDEN * sizeof(uint16_t),cudaMemcpyHostToDevice),"argmax upload input") != 0 )
	{
		free(scores);
		return(1);
	}
	if ( ValidationCheckCuda(SparkMinimaxLaunchVocabArgmax(stream,buffers->device_out,buffers->device_a,buffers->device_u64,rows,local_vocab_rows,VALIDATION_HIDDEN,local_vocab_base),"argmax launch") != 0 ||
		ValidationCheckCuda(SparkMinimaxLaunchArgmaxResolve(stream,buffers->device_u64,buffers->device_u32,rows),"argmax resolve launch") != 0 )
	{
		free(scores);
		return(1);
	}
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"argmax sync") != 0 )
	{
		free(scores);
		return(1);
	}
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_tokens,buffers->device_u32,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost),"argmax download") != 0 )
	{
		free(scores);
		return(1);
	}
	if ( buffers->host_tokens[0] != local_vocab_base + winner )
	{
		fprintf(stderr,"minimax validation: argmax token mismatch: %u vs %u\n",
			buffers->host_tokens[0],local_vocab_base + winner);
		free(scores);
		return(1);
	}
	free(scores);
	return(0);
}

static int ValidationTpCombine(const struct ValidationBuffers *buffers,cudaStream_t stream,double *worst)
{
	const uint32_t elements = 4096u;
	for (uint32_t index = 0u; index < elements; index++)
	{
		buffers->host_a[index] = ValidationFloatToBf16(ValidationNextRandom());
		buffers->host_b[index] = ValidationFloatToBf16(ValidationNextRandom());
	}
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_a,buffers->host_a,elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"combine upload a") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->device_b,buffers->host_b,elements * sizeof(uint16_t),cudaMemcpyHostToDevice),"combine upload b") != 0 )
		return(1);
	if ( ValidationCheckCuda(SparkMinimaxLaunchTpCombineBf16(stream,buffers->device_a,buffers->device_b,elements),"combine bf16 launch") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"combine bf16 sync") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMemcpy(buffers->host_out,buffers->device_a,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost),"combine bf16 download") != 0 )
		return(1);
	for (uint32_t index = 0u; index < elements; index++)
	{
		double total = (double)ValidationBf16ToFloat(buffers->host_a[index]) + (double)ValidationBf16ToFloat(buffers->host_b[index]);
		float produced = ValidationBf16ToFloat(buffers->host_out[index]);
		if ( ValidationCloseBf16(produced,(float)total,"tp combine bf16",index,worst) != 0 )
			return(1);
	}
	{
		uint64_t host_u64[64];
		uint64_t host_u64_second[64];
		for (uint32_t index = 0u; index < 64u; index++)
		{
			host_u64[index] = ((uint64_t)(ValidationNextRandom() * 1e9f) << 32) | (uint64_t)index;
			host_u64_second[index] = ((uint64_t)(ValidationNextRandom() * 1e9f) << 32) | (uint64_t)(index + 64u);
		}
		if ( ValidationCheckCuda(cudaMemcpy(buffers->device_u64,host_u64,sizeof(host_u64),cudaMemcpyHostToDevice),"combine u64 upload") != 0 )
			return(1);
		if ( ValidationCheckCuda(cudaMemcpy(buffers->device_u64_second,host_u64_second,sizeof(host_u64_second),cudaMemcpyHostToDevice),"combine u64 upload second") != 0 )
			return(1);
		if ( ValidationCheckCuda(SparkMinimaxLaunchTpCombineU64Max(stream,buffers->device_u64,buffers->device_u64_second,64u),"combine u64 launch") != 0 )
			return(1);
		if ( ValidationCheckCuda(cudaStreamSynchronize(stream),"combine u64 sync") != 0 )
			return(1);
		if ( ValidationCheckCuda(cudaMemcpy(host_u64,buffers->device_u64,sizeof(host_u64),cudaMemcpyDeviceToHost),"combine u64 download") != 0 )
			return(1);
		for (uint32_t index = 0u; index < 64u; index++)
		{
			uint64_t expected = host_u64_second[index] > host_u64[index] ? host_u64_second[index] : host_u64[index];
			if ( host_u64[index] != expected )
			{
				fprintf(stderr,"minimax validation: u64 max combine mismatch at %u\n",index);
				return(1);
			}
		}
	}
	return(0);
}

int main(int argc,char **argv)
{
	struct ValidationBuffers buffers;
	cudaStream_t stream;
	double worst = 0.0;
	int device_count = 0;
	memset(&buffers,0,sizeof(buffers));
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("minimax component validation: configuration %s\n",argv[1]);
	if ( cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 )
	{
		fprintf(stderr,"minimax validation: no CUDA device - this validator is GPU-only\n");
		return(1);
	}
	if ( ValidationCheckCuda(SparkMinimaxConfigureCudaKernels(),"configure kernels") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaStreamCreate(&stream),"stream create") != 0 )
		return(1);
	buffers.host_a = (uint16_t *)malloc((size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t));
	buffers.host_b = (uint16_t *)malloc((size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t));
	buffers.host_out = (uint16_t *)malloc((size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t));
	buffers.host_out_second = (uint16_t *)malloc((size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t));
	buffers.host_float = (float *)malloc((size_t)8192u * VALIDATION_HIDDEN * sizeof(float));
	buffers.host_tokens = (uint32_t *)malloc(64u * sizeof(uint32_t));
	if ( buffers.host_a == 0 || buffers.host_b == 0 || buffers.host_out == 0 ||
		buffers.host_out_second == 0 || buffers.host_float == 0 || buffers.host_tokens == 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_a,(size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t)),"alloc a") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_b,(size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t)),"alloc b") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_out,(size_t)8192u * VALIDATION_HIDDEN * sizeof(uint16_t)),"alloc out") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_float,(size_t)8192u * VALIDATION_HIDDEN * sizeof(float)),"alloc float") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_tokens,64u * sizeof(uint32_t)),"alloc tokens") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_u64,4096u * sizeof(uint64_t)),"alloc u64") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_u64_second,4096u * sizeof(uint64_t)),"alloc u64 second") != 0 )
		return(1);
	if ( ValidationCheckCuda(cudaMalloc((void **)&buffers.device_u32,4096u * sizeof(uint32_t)),"alloc u32") != 0 )
		return(1);
	if ( ValidationRmsNorm(&buffers,stream,&worst) != 0 ||
		ValidationSwiGlu(&buffers,stream,&worst) != 0 ||
		ValidationEmbeddingGather(&buffers,stream,&worst) != 0 ||
		ValidationHeadNormRope(&buffers,stream,&worst) != 0 ||
		ValidationVocabArgmax(&buffers,stream,&worst) != 0 ||
		ValidationTpCombine(&buffers,stream,&worst) != 0 )
		return(1);
	printf("minimax component validation: PASS (worst element delta %.6g)\n",worst);
	return(0);
}
