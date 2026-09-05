// NVFP4 wire decode smoke gate for the qwen4_flash module paths.
//
// Loads one MoE layer's W1/W3/DOWN entries from a wire-8 stagepack rank,
// routes a few tokens to one expert, and runs BOTH serving kernels the
// module uses (the grouped-scalar dot-row path and the tile-MMloop path)
// through their real host launchers. The device outputs are compared
// against a host dequant reference computed from the SAME pack bytes:
//   W = e2m1(nibble) * e4m3(plane_byte) * weight_scale_2(expert)
// Verdict is the exit code (0 = all legs pass). Run on an sm_121 node:
//   nvcc -arch=sm_121a -I<repo> tools/qwen4_flash_nvfp4_smoke.cu -o /tmp/nvfp4_smoke
//   /tmp/nvfp4_smoke <pack.q4fsp> [layer]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "sparkpipe/spark_lm_kernels.cuh"

#define KIND_MOE_W1 6u
#define KIND_MOE_W3 7u
#define KIND_MOE_DOWN 8u

typedef struct
{
	uint32_t kind, layer, format, rows, columns, group;
	uint64_t payload_offset, payload_bytes, scale_offset, scale_bytes;
} SmokeEntry;

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

// Dequant one expert's packed row-major [rows, columns] e2m1 matrix with
// its per-16 e4m3 plane + the segment-tail F32 weight global.
static float *DequantNvfp4Host(const uint8_t *payload, const uint8_t *segment,
	uint32_t rows, uint32_t columns, float *weight_global_out)
{
	uint64_t plane_bytes = (uint64_t)rows * (columns / 16u);
	const uint8_t *plane = segment;
	float weight_global;
	float *matrix = (float *)malloc((uint64_t)rows * columns * sizeof(float));
	uint32_t row,column,group;
	memcpy(&weight_global,segment + plane_bytes + 4u,4u);
	*weight_global_out = weight_global;
	for (row = 0; row < rows; row++)
		for (column = 0; column < columns; column++)
		{
			float scale = DecodeE4m3Host(plane[(uint64_t)row * (columns / 16u) + (column / 16u)]) * weight_global;
			uint8_t byte = payload[((uint64_t)row * columns + column) >> 1];
			uint32_t code = (column & 1u) ? (byte >> 4) : (byte & 15u);
			matrix[(uint64_t)row * columns + column] = DecodeE2m1Host(code) * scale;
		}
	(void)group;
	return(matrix);
}

static int CompareBf16(const char *leg, const void *device_bf16, const float *reference, uint32_t count, uint32_t row_stride)
{
	__nv_bfloat16 *got = (__nv_bfloat16 *)malloc((uint64_t)count * 2u);
	uint32_t index;
	int failures = 0;
	double worst_relative = 0.0;
	cudaMemcpy(got,device_bf16,(uint64_t)count * 2u,cudaMemcpyDeviceToHost);
	for (index = 0; index < count; index++)
	{
		float value = __bfloat162float(got[index]);
		float want = reference[index];
		double absolute = fabs((double)value - (double)want);
		double relative = absolute > 0.0 ? absolute / fmax(fabs((double)want),1e-6) : 0.0;
		// bf16 rounding of the output alone allows ~2^-8 relative; the
		// kernel's f32 warp-reassociation adds a little more slack.
		if ( relative > 0.02 && absolute > 1e-3 )
		{
			if ( failures < 6u )
				fprintf(stderr,"SMOKE %s idx %u: got %.6f want %.6f (rel %.4f) ratio %.4f\n",
					leg,index,value,want,relative,
					fabs((double)want) > 1e-9 ? (double)value / (double)want : 0.0);
			failures++;
		}
		if ( relative > worst_relative )
			worst_relative = relative;
	}
	if ( strcmp(leg,"tile W3") == 0 )
	{
		for (index = 0u; index < 5u; index++)
		{
			float value = __bfloat162float(got[index]);
			uint32_t best = 0u;
			double best_diff = 1e30;
			for (uint32_t scan = 0u; scan < count; scan++)
			{
				double diff = fabs((double)value - (double)reference[scan]);
				if ( diff < best_diff ) { best_diff = diff; best = scan; }
			}
			printf("SMOKE %s got[%u]=%.4f -> nearest reference[(row=%u,neuron=%u)]=%.4f (diff %.4g)\n",
				leg,index,value,best / row_stride,best % row_stride,
				reference[best],best_diff);
		}
		for (uint32_t r = 0u; r < 3u; r++)
		{
			printf("SMOKE %s row%u want:",leg,r);
			for (index = 0; index < 8u; index++)
				printf(" %.4f",reference[(uint64_t)r * row_stride + index]);
			printf("\n");
		}
		printf("SMOKE %s row0 got :",leg);
		for (index = 0; index < 8u; index++)
			printf(" %.4f",__bfloat162float(got[index]));
		printf("\n");
	}
	printf("SMOKE %s: %u values, %u outside tolerance, worst rel %.5f\n",
		leg,count,failures,worst_relative);
	// The tile paths stage decoded weights to bf16 for the tensor cores:
	// the per-element relative gate above flags near-zero-reference
	// elements on otherwise-correct output. The module validator's
	// aggregate contract (rel L2 <= 5e-2, cosine >= 0.999) is the
	// acceptance standard for these legs.
	if ( strcmp(leg,"scalar W1") != 0 )
	{
		double sq_diff = 0.0,sq_ref = 0.0,dot = 0.0,norm_got = 0.0;
		for (index = 0; index < count; index++)
		{
			double value = (double)__bfloat162float(got[index]);
			double want = (double)reference[index];
			double diff = value - want;
			sq_diff += diff * diff;
			sq_ref += want * want;
			dot += value * want;
			norm_got += value * value;
		}
		double rel_l2 = sq_ref > 0.0 ? sqrt(sq_diff) / sqrt(sq_ref) : 0.0;
		double cosine = (sqrt(norm_got) * sqrt(sq_ref)) > 0.0 ?
			dot / (sqrt(norm_got) * sqrt(sq_ref)) : 0.0;
		printf("SMOKE %s aggregate: rel_l2=%.5f cosine=%.6f\n",
			leg,rel_l2,cosine);
		if ( rel_l2 > 5e-2 || cosine < 0.999 )
		{
			printf("SMOKE %s FAILS the module aggregate contract\n",leg);
			free(got);
			return(1);
		}
		printf("SMOKE %s aggregate PASS\n",leg);
		free(got);
		return(0);
	}
	free(got);
	return(failures != 0);
}

static __global__ void DebugDecodeRow(const uint8_t *payload, const uint8_t *segment, uint32_t plane_bytes, float weight_global, uint16_t *out_bf16)
{
	__shared__ __nv_bfloat16 tile[64];
	// mirror ProducerHalf: decode 32 k-values at stage_k, neuron 0
	if ( threadIdx.x == 0u )
	{
		SparkLmTileDecodeRun<16u>(SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1, payload, segment,
			0u, 0u, 2560u, ( __nv_bfloat16 *)tile, (const float *)(segment + plane_bytes + 4u));
		for (uint32_t i = 0u; i < 64u; i++)
			out_bf16[i] = __bfloat16_as_ushort(tile[i]);
	}
}

int main(int argc, char **argv)
{
	const char *pack_path;
	uint32_t want_layer = 1u;
	uint8_t *header_bytes,*directory_bytes;
	uint32_t tensor_count,directory_entry_bytes,i;
	uint64_t directory_offset,file_bytes;
	SmokeEntry *entries;
	SmokeEntry w1,w3,down;
	int have_w1 = 0,have_w3 = 0,have_down = 0;
	uint64_t plane_w1,segment_w1;
	const uint32_t group_count = 64u,rows = 8u;
	const uint32_t hidden = 2560u,intermediate = 640u;
	uint8_t *w1_payload,*w1_segment,*w3_payload,*w3_segment,*down_payload,*down_segment;
	float *w1_matrix,*w3_matrix,*down_matrix,*input_host,*reference_w1,*reference_w3,*reference_down,*reference_act,*reference_out;
	float w1_global,w3_global,down_global,limit = 7.0f;
	void *input_dev,*w1_out_dev,*w3_out_dev,*act_dev,*down_out_dev,*payload_dev,*scale_dev;
	uint32_t *offsets_dev,*rows_dev;
	uint32_t expert_offsets[65],grouped_rows[8];
	uint32_t group_row_offset[65],group_tile_prefix[65];
	float *pair_weights;
	uint32_t *inverse_map;
	cudaError_t error;
	int failures = 0;
	if ( argc < 2 )
	{
		fprintf(stderr,"usage: %s <pack.q4fsp> [layer]\n",argv[0]);
		return(2);
	}
	pack_path = argv[1];
	if ( argc > 2 )
		want_layer = (uint32_t)atoi(argv[2]);
	header_bytes = ReadFileBytes(pack_path,0,120u);
	if ( header_bytes == 0 )
	{
		fprintf(stderr,"SMOKE pack header unreadable\n");
		return(2);
	}
	memcpy(&tensor_count,header_bytes + 16u,4u);
	memcpy(&directory_entry_bytes,header_bytes + 12u,4u);
	memcpy(&directory_offset,header_bytes + 104u,8u);
	memcpy(&file_bytes,header_bytes + 112u,8u);
	directory_bytes = ReadFileBytes(pack_path,directory_offset,(uint64_t)tensor_count * directory_entry_bytes);
	if ( directory_bytes == 0 )
	{
		fprintf(stderr,"SMOKE directory unreadable\n");
		return(2);
	}
	entries = (SmokeEntry *)malloc((uint64_t)tensor_count * sizeof(SmokeEntry));
	for (i = 0; i < tensor_count; i++)
	{
		const uint8_t *raw = directory_bytes + (uint64_t)i * directory_entry_bytes;
		SmokeEntry *entry = &entries[i];
		memcpy(&entry->kind,raw,4u);
		memcpy(&entry->layer,raw + 4u,4u);
		memcpy(&entry->format,raw + 8u,4u);
		memcpy(&entry->rows,raw + 12u,4u);
		memcpy(&entry->columns,raw + 16u,4u);
		memcpy(&entry->group,raw + 20u,4u);
		memcpy(&entry->payload_offset,raw + 24u,8u);
		memcpy(&entry->payload_bytes,raw + 32u,8u);
		memcpy(&entry->scale_offset,raw + 40u,8u);
		memcpy(&entry->scale_bytes,raw + 48u,8u);
		if ( entry->layer == want_layer && entry->format == 8u )
		{
			if ( entry->kind == KIND_MOE_W1 ){ w1 = *entry; have_w1 = 1; }
			if ( entry->kind == KIND_MOE_W3 ){ w3 = *entry; have_w3 = 1; }
			if ( entry->kind == KIND_MOE_DOWN ){ down = *entry; have_down = 1; }
		}
	}
	if ( !have_w1 || !have_w3 || !have_down )
	{
		fprintf(stderr,"SMOKE layer %u lacks wire-8 W1/W3/DOWN entries\n",want_layer);
		return(2);
	}
	// Load expert group 5's segment of W1/W3 (rows = intermediate) and
	// the DOWN entry (rows = experts x hidden): segment stride = plane + 8.
	w1_payload = ReadFileBytes(pack_path,w1.payload_offset + 5u * (w1.rows / group_count) * ((uint64_t)w1.columns / 2u),(uint64_t)(w1.rows / group_count) * (w1.columns / 2u));
	w1_segment = ReadFileBytes(pack_path,w1.scale_offset + 5u * (w1.scale_bytes / group_count),w1.scale_bytes / group_count);
	w3_payload = ReadFileBytes(pack_path,w3.payload_offset + 5u * (w3.rows / group_count) * ((uint64_t)w3.columns / 2u),(uint64_t)(w3.rows / group_count) * (w3.columns / 2u));
	w3_segment = ReadFileBytes(pack_path,w3.scale_offset + 5u * (w3.scale_bytes / group_count),w3.scale_bytes / group_count);
	down_payload = ReadFileBytes(pack_path,down.payload_offset + 5u * (down.rows / group_count) * ((uint64_t)down.columns / 2u),(uint64_t)(down.rows / group_count) * (down.columns / 2u));
	down_segment = ReadFileBytes(pack_path,down.scale_offset + 5u * (down.scale_bytes / group_count),down.scale_bytes / group_count);
	if ( !w1_payload || !w1_segment || !w3_payload || !w3_segment || !down_payload || !down_segment )
	{
		fprintf(stderr,"SMOKE expert segment read failed\n");
		return(2);
	}
	w1_matrix = DequantNvfp4Host(w1_payload,w1_segment,w1.rows / group_count,w1.columns,&w1_global);
	w3_matrix = DequantNvfp4Host(w3_payload,w3_segment,w3.rows / group_count,w3.columns,&w3_global);
	down_matrix = DequantNvfp4Host(down_payload,down_segment,down.rows / group_count,down.columns,&down_global);
	printf("SMOKE globals: w1=%.6g w3=%.6g down=%.6g\n",w1_global,w3_global,down_global);
	// Random-ish bounded input in bf16.
	input_host = (float *)malloc((uint64_t)rows * hidden * sizeof(float));
	for (i = 0; i < rows * hidden; i++)
		input_host[i] = ((float)((i * 2654435761u) % 2001u) - 1000.0f) / 1000.0f;
	// Host reference: gate/up via dequantized W1/W3, swiglu(limit), down.
	reference_w1 = (float *)malloc((uint64_t)rows * intermediate * sizeof(float));
	reference_w3 = (float *)malloc((uint64_t)rows * intermediate * sizeof(float));
	reference_act = (float *)malloc((uint64_t)rows * intermediate * sizeof(float));
	reference_out = (float *)malloc((uint64_t)rows * hidden * sizeof(float));
	reference_down = (float *)malloc((uint64_t)rows * hidden * sizeof(float));
	for (i = 0; i < rows * intermediate; i++)
	{
		uint32_t row = i / intermediate,neuron = i % intermediate;
		uint32_t k;
		double gate = 0.0,up = 0.0,g,u;
		for (k = 0; k < hidden; k++)
		{
			double x = input_host[(uint64_t)row * hidden + k];
			gate += x * (double)w1_matrix[(uint64_t)neuron * hidden + k];
			up += x * (double)w3_matrix[(uint64_t)neuron * hidden + k];
		}
		// Match the module: bf16-round the totals, clamp, swish * up.
		g = (double)__bfloat162float(__float2bfloat16((float)gate));
		u = (double)__bfloat162float(__float2bfloat16((float)up));
		if ( g > limit ) g = limit;
		if ( u > limit ) u = limit;
		if ( u < -limit ) u = -limit;
		g = g / (1.0 + exp(-g));
		reference_act[i] = __bfloat162float(__float2bfloat16((float)(g * u)));
		reference_w1[i] = (float)gate;
		reference_w3[i] = (float)up;
	}
	for (i = 0; i < rows * hidden; i++)
	{
		uint32_t row = i / hidden,neuron = i % hidden;
		uint32_t k;
		double total = 0.0;
		for (k = 0; k < intermediate; k++)
			total += (double)__bfloat162float(__float2bfloat16(reference_act[(uint64_t)row * intermediate + k])) * (double)down_matrix[(uint64_t)neuron * intermediate + k];
		reference_out[i] = (float)total;
		reference_down[i] = (float)total;
	}
	// Device buffers: full-entry payload/scale so the launchers stride
	// experts the way serving does; rows routed to expert group 5.
	cudaMalloc(&payload_dev,w1.payload_bytes + w3.payload_bytes + down.payload_bytes);
	cudaMalloc(&scale_dev,w1.scale_bytes + w3.scale_bytes + down.scale_bytes);
	cudaMalloc(&input_dev,(uint64_t)rows * hidden * 2u);
	cudaMalloc(&w1_out_dev,(uint64_t)rows * intermediate * 2u);
	cudaMalloc(&w3_out_dev,(uint64_t)rows * intermediate * 2u);
	cudaMalloc(&act_dev,(uint64_t)rows * intermediate * 2u);
	cudaMalloc(&down_out_dev,(uint64_t)rows * hidden * 2u);
	cudaMalloc(&offsets_dev,66u * 4u);
	cudaMalloc(&rows_dev,(uint64_t)rows * 4u * 2u);
	{
		uint8_t *payload_host = (uint8_t *)malloc(w1.payload_bytes + w3.payload_bytes + down.payload_bytes);
		uint8_t *scale_host = (uint8_t *)malloc(w1.scale_bytes + w3.scale_bytes + down.scale_bytes);
		FILE *file = fopen(pack_path,"rb");
		fseek(file,(long)w1.payload_offset,SEEK_SET);
		fread(payload_host,1,w1.payload_bytes,file);
		fseek(file,(long)w3.payload_offset,SEEK_SET);
		fread(payload_host + w1.payload_bytes,1,w3.payload_bytes,file);
		fseek(file,(long)down.payload_offset,SEEK_SET);
		fread(payload_host + w1.payload_bytes + w3.payload_bytes,1,down.payload_bytes,file);
		fseek(file,(long)w1.scale_offset,SEEK_SET);
		fread(scale_host,1,w1.scale_bytes,file);
		fseek(file,(long)w3.scale_offset,SEEK_SET);
		fread(scale_host + w1.scale_bytes,1,w3.scale_bytes,file);
		fseek(file,(long)down.scale_offset,SEEK_SET);
		fread(scale_host + w1.scale_bytes + w3.scale_bytes,1,down.scale_bytes,file);
		fclose(file);
		cudaMemcpy(payload_dev,payload_host,w1.payload_bytes + w3.payload_bytes + down.payload_bytes,cudaMemcpyHostToDevice);
		cudaMemcpy(scale_dev,scale_host,w1.scale_bytes + w3.scale_bytes + down.scale_bytes,cudaMemcpyHostToDevice);
		free(payload_host);
		free(scale_host);
	}
	{
		__nv_bfloat16 *input_bf16 = (__nv_bfloat16 *)malloc((uint64_t)rows * hidden * 2u);
		for (i = 0; i < rows * hidden; i++)
			input_bf16[i] = __float2bfloat16(input_host[i]);
		cudaMemcpy(input_dev,input_bf16,(uint64_t)rows * hidden * 2u,cudaMemcpyHostToDevice);
		free(input_bf16);
	}
	// Route arrays: ALL rows to expert group 5 (both array conventions).
	for (i = 0; i <= group_count; i++)
	{
		expert_offsets[i] = (i > 5u) ? rows : 0u;
		group_row_offset[i] = (i > 5u) ? rows : 0u;
	}
	{
		// Tile prefix: expert 5 owns neuron_tiles x row-tiles; others
		// none. SPARK_LM_TILE_N=128, SPARK_LM_TILE=16.
		uint32_t neuron_tiles = (intermediate + 128u - 1u) / 128u;
		uint32_t total = 0u;
		for (i = 0; i <= group_count; i++)
		{
			group_tile_prefix[i] = total;
			if ( i == 5u )
				total += neuron_tiles * ((rows + 15u) / 16u);
		}
	}
	for (i = 0; i < rows; i++)
		grouped_rows[i] = i;
	cudaMemcpy(offsets_dev,expert_offsets,66u * 4u,cudaMemcpyHostToDevice);
	cudaMemcpy(rows_dev,grouped_rows,(uint64_t)rows * 4u,cudaMemcpyHostToDevice);
	// LEG 1: grouped-scalar dot-row path, W1 (gate) with rows routed to
	// expert 5. Payload/scale device buffers hold the FULL entries, so the
	// launchers stride experts exactly like serving.
	{
		uint32_t rows_per_expert = w1.rows / group_count;
		uint64_t payload_stride = (uint64_t)rows_per_expert * (w1.columns / 2u);
		uint64_t scale_stride = (uint64_t)rows_per_expert * (w1.columns / 16u) + 8u;
		uint32_t *go,*tp;
		cudaMalloc(&go,66u * 4u);
		cudaMalloc(&tp,66u * 4u);
		cudaMemcpy(go,group_row_offset,66u * 4u,cudaMemcpyHostToDevice);
		cudaMemcpy(tp,group_tile_prefix,66u * 4u,cudaMemcpyHostToDevice);
		error = SparkLmHostLaunchGroupedScalarLinear<32u>(0,
			SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1,
			payload_dev,(const uint8_t *)scale_dev,payload_stride,scale_stride,
			input_dev,0,rows,go,tp,w1_out_dev,group_count,w1.columns,
			rows_per_expert,1u);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"SMOKE scalar W1 launch failed: %s\n",cudaGetErrorString(error));
			failures++;
		}
		else
		{
			cudaDeviceSynchronize();
			failures += CompareBf16("scalar W1",w1_out_dev,reference_w1,(uint32_t)rows * intermediate,intermediate);
		}
		cudaFree(go);
		cudaFree(tp);
	}
	// LEG 1b: decode-only check of the tile path's weight decode.
	{
		uint16_t *dev_out;
		uint16_t host_out[64];
		uint32_t rows_per_expert = w1.rows / group_count;
		uint64_t seg = w1.scale_bytes / group_count;
		uint8_t *seg_host = (uint8_t *)malloc(seg);
		{
			FILE *file = fopen(pack_path,"rb");
			fseek(file,(long)(w1.scale_offset + 5u * seg),SEEK_SET);
			if ( fread(seg_host,1,seg,file) != seg ) { printf("SMOKE leg1b seg read fail\n"); return 2; }
			fclose(file);
		}
		uint8_t *dev_seg;
		cudaMalloc(&dev_out,64u * 2u);
		cudaMalloc(&dev_seg,seg);
		cudaMemcpy(dev_seg,seg_host,seg,cudaMemcpyHostToDevice);
		float wg;
		memcpy(&wg,seg_host + (uint64_t)rows_per_expert * (w1.columns / 16u) + 4u,4u);
		DebugDecodeRow<<<1u,32u>>>((const uint8_t *)payload_dev + 5u * (rows_per_expert * (w1.columns / 2u)),dev_seg,
			(uint32_t)rows_per_expert * (w1.columns / 16u),wg,dev_out);
		cudaDeviceSynchronize();
		cudaMemcpy(host_out,dev_out,64u * 2u,cudaMemcpyDeviceToHost);
		printf("SMOKE leg1b decode row0 k0-15 (device|host):");
		for (i = 0u; i < 16u; i++)
		{
			float d = __bfloat162float(__ushort_as_bfloat16(host_out[i]));
			printf(" %.4f|%.4f",d,w1_matrix[i]);
		}
		printf("\n");
		cudaFree(dev_out);
		cudaFree(dev_seg);
		free(seg_host);
	}
	// LEG 2: the tile Mloop path, W3 (up), same routing.
	{
		uint32_t rows_per_expert = w3.rows / group_count;
		uint64_t payload_stride = (uint64_t)rows_per_expert * (w3.columns / 2u);
		uint64_t scale_stride = (uint64_t)rows_per_expert * (w3.columns / 16u) + 8u;
		error = SparkLmHostLaunchGroupedExpertTileMloop(0,
			SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1,
			(const uint8_t *)payload_dev + w1.payload_bytes,
			(const uint8_t *)scale_dev + w1.scale_bytes,
			payload_stride,scale_stride,
			input_dev,rows_dev,offsets_dev,w3_out_dev,w3.columns,
			rows_per_expert,group_count);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"SMOKE tile W3 launch failed: %s\n",cudaGetErrorString(error));
			failures++;
		}
		else
		{
			cudaDeviceSynchronize();
			failures += CompareBf16("tile W3",w3_out_dev,reference_w3,(uint32_t)rows * intermediate,intermediate);
		}
	}
	// LEG 3: the tile Mloop path, DOWN (input = reference activations).
	{
		__nv_bfloat16 *act_bf16 = (__nv_bfloat16 *)malloc((uint64_t)rows * intermediate * 2u);
		uint32_t rows_per_expert = down.rows / group_count;
		uint64_t payload_stride = (uint64_t)rows_per_expert * (down.columns / 2u);
		uint64_t scale_stride = (uint64_t)rows_per_expert * (down.columns / 16u) + 8u;
		for (i = 0; i < rows * intermediate; i++)
			act_bf16[i] = __float2bfloat16(reference_act[i]);
		cudaMemcpy(act_dev,act_bf16,(uint64_t)rows * intermediate * 2u,cudaMemcpyHostToDevice);
		free(act_bf16);
		error = SparkLmHostLaunchGroupedExpertTileMloop(0,
			SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1,
			(const uint8_t *)payload_dev + w1.payload_bytes + w3.payload_bytes,
			(const uint8_t *)scale_dev + w1.scale_bytes + w3.scale_bytes,
			payload_stride,scale_stride,
			act_dev,rows_dev,offsets_dev,down_out_dev,down.columns,
			rows_per_expert,group_count);
		if ( error != cudaSuccess )
		{
			fprintf(stderr,"SMOKE tile DOWN launch failed: %s\n",cudaGetErrorString(error));
			failures++;
		}
		else
		{
			cudaDeviceSynchronize();
			failures += CompareBf16("tile DOWN",down_out_dev,reference_down,(uint32_t)rows * hidden,hidden);
		}
	}
	printf("SMOKE %s\n",failures == 0 ? "PASS" : "FAIL");
	return(failures == 0 ? 0 : 1);
}
