// NVFP4 dense-linear smoke gate for the qwen38_27b module paths.
//
// Loads the FFN gate entry from a wire-8 TP4 rank pack and runs the
// shared SparkLmHostLaunchBatchedLinear path at two batch sizes (the
// small GEMV kernel at rows<16 and the tile pipeline at rows>=16),
// comparing both against a host dequant of the SAME pack bytes:
//   W = e2m1(nibble) * e4m3(plane_byte) * weight_global(segment tail - 4)
// Verdict is the exit code. Build on an sm_121 node:
//   nvcc -std=c++17 -O3 --expt-relaxed-constexpr -gencode arch=compute_121a,code=sm_121a \
//     -I<repo> -I<repo>/model-families/common/include \
//     tools/qwen38_27b_nvfp4_smoke.cu -o /tmp/q27b_nvfp4_smoke

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define KIND_FFN_GATE 5u

static uint8_t *ReadFileBytes(const char *path, uint64_t offset, uint64_t bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *buffer;
	if ( file == 0 )
		return(0);
	fseek(file,(long)offset,SEEK_SET);
	buffer = (uint8_t *)malloc(bytes);
	if ( buffer == 0 || fread(buffer,1,bytes,file) != bytes )
	{
		free(buffer);
		fclose(file);
		return(0);
	}
	fclose(file);
	return(buffer);
}

static float DecodeE2m1Host(uint32_t code)
{
	static const float table[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
		-0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
	return(table[code & 15u]);
}

static float DecodeE4m3Host(uint32_t code)
{
	int32_t sign = (code & 0x80u) ? -1 : 1;
	uint32_t exponent = (code >> 3) & 15u;
	uint32_t mantissa = code & 7u;
	if ( exponent == 0u )
		return((float)sign * (float)mantissa * powf(2.0f,-9.0f));
	if ( exponent == 15u && mantissa == 7u )
		return(nanf(""));
	return((float)sign * (1.0f + (float)mantissa / 8.0f) * powf(2.0f,(int32_t)exponent - 7));
}

static int CompareBf16(const char *leg, const void *device_bf16, const float *reference, uint32_t count)
{
	__nv_bfloat16 *got = (__nv_bfloat16 *)malloc((uint64_t)count * 2u);
	uint32_t index;
	int failures = 0;
	double worst = 0.0;
	cudaMemcpy(got,device_bf16,(uint64_t)count * 2u,cudaMemcpyDeviceToHost);
	for (index = 0; index < count; index++)
	{
		float value = __bfloat162float(got[index]);
		float want = reference[index];
		double absolute = fabs((double)value - (double)want);
		double relative = absolute / fmax(fabs((double)want),1e-4);
		if ( relative > 0.05 && absolute > 1e-2 )
		{
			if ( failures < 4u )
				fprintf(stderr,"SMOKE %s idx %u: got %.5f want %.5f (rel %.3f)\\n",
					leg,index,value,want,relative);
			failures++;
		}
		if ( relative > worst )
			worst = relative;
	}
	printf("SMOKE %s: %u values, %u outside tolerance, worst rel %.4f\\n",
		leg,count,failures,worst);
	free(got);
	return(failures != 0);
}

int main(int argc, char **argv)
{
	const char *pack_path;
	uint8_t *header_bytes,*directory_bytes;
	uint32_t tensor_count,i;
	uint64_t directory_offset,file_bytes;
	uint32_t gate_entry = 0xffffffffu;
	uint32_t gate_rows,gate_cols;
	uint64_t gate_payload_off,gate_payload_bytes,gate_scale_off,gate_scale_bytes;
	float *gate_matrix,*input_host,*reference;
	void *input_dev,*output_dev,*payload_dev,*scale_dev;
	uint8_t *payload_host,*scale_host;
	FILE *file;
	cudaError_t error;
	int failures = 0;
	uint32_t rows;
	if ( argc < 2 )
	{
		fprintf(stderr,"usage: %s <pack.q38sp> [layer]\n",argv[0]);
		return(2);
	}
	pack_path = argv[1];
	header_bytes = ReadFileBytes(pack_path,0,120u);
	if ( header_bytes == 0 )
	{
		fprintf(stderr,"SMOKE header unreadable\\n");
		return(2);
	}
	memcpy(&tensor_count,header_bytes + 16u,4u);
	memcpy(&directory_offset,header_bytes + 104u,8u);
	memcpy(&file_bytes,header_bytes + 112u,8u);
	(void)file_bytes;
	directory_bytes = ReadFileBytes(pack_path,directory_offset,(uint64_t)tensor_count * 56u);
	if ( directory_bytes == 0 )
	{
		fprintf(stderr,"SMOKE directory unreadable\\n");
		return(2);
	}
	for (i = 0; i < tensor_count; i++)
	{
		const uint8_t *block = directory_bytes + (uint64_t)i * 56u;
		uint32_t kind,layer,format;
		memcpy(&kind,block,4u);
		memcpy(&layer,block + 4u,4u);
		memcpy(&format,block + 8u,4u);
		if ( kind == KIND_FFN_GATE && layer == 1u && format == 8u )
		{
			memcpy(&gate_rows,block + 12u,4u);
			memcpy(&gate_cols,block + 16u,4u);
			memcpy(&gate_payload_off,block + 24u,8u);
			memcpy(&gate_payload_bytes,block + 32u,8u);
			memcpy(&gate_scale_off,block + 40u,8u);
			memcpy(&gate_scale_bytes,block + 48u,8u);
			gate_entry = i;
			break;
		}
	}
	if ( gate_entry == 0xffffffffu )
	{
		fprintf(stderr,"SMOKE no wire-8 FFN gate entry at layer 1\\n");
		return(2);
	}
	printf("SMOKE gate entry: [%u,%u] payload %llu scale %llu\\n",
		gate_rows,gate_cols,
		(unsigned long long)gate_payload_bytes,(unsigned long long)gate_scale_bytes);
	payload_host = ReadFileBytes(pack_path,gate_payload_off,gate_payload_bytes);
	scale_host = ReadFileBytes(pack_path,gate_scale_off,gate_scale_bytes);
	if ( payload_host == 0 || scale_host == 0 )
	{
		fprintf(stderr,"SMOKE segment read failed\\n");
		return(2);
	}
	// Host dequant of the first neurons' rows + the reference dots.
	{
		uint64_t plane_bytes = (uint64_t)gate_rows * (gate_cols / 16u);
		float weight_global;
		/* the dense segment = [plane][global F32]: the global is the
		 * LAST 4 bytes (no input scale in the a16 layout). */
		memcpy(&weight_global,scale_host + plane_bytes,4u);
		printf("SMOKE weight_global = %.6g\\n",weight_global);
		gate_matrix = (float *)malloc((uint64_t)gate_rows * gate_cols * sizeof(float));
		for (uint32_t row = 0; row < gate_rows; row++)
			for (uint32_t col = 0; col < gate_cols; col++)
			{
				float scale = DecodeE4m3Host(scale_host[(uint64_t)row * (gate_cols / 16u) + (col / 16u)]) * weight_global;
				uint8_t byte = payload_host[((uint64_t)row * gate_cols + col) >> 1u];
				uint32_t code = (col & 1u) ? (byte >> 4) : (byte & 15u);
				gate_matrix[(uint64_t)row * gate_cols + col] = DecodeE2m1Host(code) * scale;
			}
		rows = 20u;
		input_host = (float *)malloc((uint64_t)rows * gate_cols * sizeof(float));
		for (i = 0; i < rows * gate_cols; i++)
			input_host[i] = ((float)((i * 2654435761u) % 2001u) - 1000.0f) / 1000.0f;
		reference = (float *)malloc((uint64_t)rows * gate_rows * sizeof(float));
		for (uint32_t r = 0; r < rows; r++)
			for (uint32_t n = 0; n < gate_rows; n++)
			{
				double total = 0.0;
				for (uint32_t k = 0; k < gate_cols; k++)
					total += (double)input_host[(uint64_t)r * gate_cols + k] * (double)gate_matrix[(uint64_t)n * gate_cols + k];
				reference[(uint64_t)r * gate_rows + n] = (float)total;
			}
	}
	cudaMalloc(&payload_dev,gate_payload_bytes);
	cudaMalloc(&scale_dev,gate_scale_bytes);
	cudaMalloc(&input_dev,(uint64_t)rows * gate_cols * 2u);
	cudaMalloc(&output_dev,(uint64_t)rows * gate_rows * 2u);
	cudaMemcpy(payload_dev,payload_host,gate_payload_bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(scale_dev,scale_host,gate_scale_bytes,cudaMemcpyHostToDevice);
	{
		__nv_bfloat16 *input_bf16 = (__nv_bfloat16 *)malloc((uint64_t)rows * gate_cols * 2u);
		for (i = 0; i < rows * gate_cols; i++)
			input_bf16[i] = __float2bfloat16(input_host[i]);
		cudaMemcpy(input_dev,input_bf16,(uint64_t)rows * gate_cols * 2u,cudaMemcpyHostToDevice);
		free(input_bf16);
	}
	// LEG 1: the small GEMV kernel (rows < SPARK_LM_TILE).
	{
		__nv_bfloat16 *saved = (__nv_bfloat16 *)malloc((uint64_t)rows * gate_cols * 2u);
		cudaMemcpy(saved,input_dev,(uint64_t)rows * gate_cols * 2u,cudaMemcpyDeviceToHost);
		__nv_bfloat16 *four = (__nv_bfloat16 *)malloc(4u * gate_cols * 2u);
		memcpy(four,saved,4u * gate_cols * 2u);
		cudaMemcpy(input_dev,four,4u * gate_cols * 2u,cudaMemcpyHostToDevice);
		error = SparkLmHostLaunchBatchedLinear<32u>(0,SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1,
			payload_dev,scale_dev,input_dev,output_dev,4u,gate_cols,gate_rows);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"SMOKE gemv launch failed: %s\\n",cudaGetErrorString(error));
			failures++;
		}
		else
		{
			cudaDeviceSynchronize();
			failures += CompareBf16("gemv rows4",output_dev,reference,4u * gate_rows);
		}
		cudaMemcpy(input_dev,saved,(uint64_t)rows * gate_cols * 2u,cudaMemcpyHostToDevice);
		free(saved);
		free(four);
	}
	// LEG 2: the tile pipeline (rows >= SPARK_LM_TILE).
	{
		error = SparkLmHostLaunchBatchedLinear<32u>(0,SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1,
			payload_dev,scale_dev,input_dev,output_dev,rows,gate_cols,gate_rows);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"SMOKE tile launch failed: %s\\n",cudaGetErrorString(error));
			failures++;
		}
		else
		{
			cudaDeviceSynchronize();
			failures += CompareBf16("tile rows20",output_dev,reference,rows * gate_rows);
		}
	}
	printf("SMOKE %s\\n",failures == 0 ? "PASS" : "FAIL");
	return(failures == 0 ? 0 : 1);
}
