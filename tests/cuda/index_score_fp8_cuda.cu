#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

#include "runtime/launch.h"
#include "inference/kernels/index_score.cuh"

#define PROBE_HEADS 32u
#define PROBE_DIM 128u
#define PROBE_WEIGHT_SCALE 0.015625f

using ProbeIndexKv = LmKvLatent<16u, PROBE_DIM, 0u, 64u>;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static int probe_failures;

static float ProbeBf16Value(uint16_t bits)
{
	const uint32_t widened = (uint32_t)bits << 16u;
	float value;
	memcpy(&value,&widened,sizeof(value));
	return value;
}

static uint16_t ProbeBf16Bits(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)(bits >> 16u);
}

static float ProbeScale(float amax, float floor)
{
	const float ratio = std::max(amax,floor) / 448.0f;
	int exponent;
	const float fraction = frexpf(ratio,&exponent);
	return ldexpf(1.0f,fraction == 0.5f ? exponent - 1 : exponent);
}

static uint8_t ProbeE4m3(float value)
{
	const uint8_t sign = signbit(value) ? 0x80u : 0u;
	const float magnitude = fabsf(value);
	int exponent,step;
	uint32_t code;
	if ( !(magnitude < 448.0f) )
		return (uint8_t)(sign | 0x7eu);
	if ( magnitude < 0.015625f )
		return (uint8_t)(sign | (uint8_t)nearbyintf(magnitude * 512.0f));
	step = (int)nearbyintf(frexpf(magnitude,&exponent) * 16.0f);
	exponent += 6;
	if ( step == 16 )
	{
		step = 8;
		exponent++;
	}
	code = ((uint32_t)exponent << 3u) | (uint32_t)(step - 8);
	return (uint8_t)(sign | (code > 0x7eu ? 0x7eu : code));
}

static double ProbeE4m3Value(uint8_t code)
{
	const uint32_t exponent = (code >> 3u) & 15u,mantissa = code & 7u;
	const double magnitude = exponent == 0u ? ldexp((double)mantissa,-9) : ldexp(1.0 + mantissa / 8.0,(int)exponent - 7);
	return (code & 0x80u) != 0u ? -magnitude : magnitude;
}

static uint32_t ProbeNext(uint32_t *state)
{
	*state = *state * 1664525u + 1013904223u;
	return *state >> 8u;
}

static void ProbeVector(uint16_t *values, uint32_t count, uint32_t *state, float spread)
{
	for (uint32_t index = 0u; index < count; ++index)
	{
		const float unit = (float)(ProbeNext(state) & 0xffffu) / 32768.0f - 1.0f;
		values[index] = ProbeBf16Bits(unit * spread);
	}
}

static void ProbeQuantizeRow(const uint16_t *source, float floor, bool clamp, uint8_t *codes, float *scale)
{
	float amax = 0.0f;
	for (uint32_t index = 0u; index < PROBE_DIM; ++index)
		amax = std::max(amax,fabsf(ProbeBf16Value(source[index])));
	*scale = ProbeScale(amax,floor);
	for (uint32_t index = 0u; index < PROBE_DIM; ++index)
	{
		float value = ProbeBf16Value(source[index]) / *scale;
		if ( clamp )
			value = std::min(std::max(value,-448.0f),448.0f);
		codes[index] = ProbeE4m3(value);
	}
}

struct ProbeShape
{
	const char *name;
	uint32_t rows;
	uint32_t keys;
	uint32_t check;
};

static void ProbeRun(const ProbeShape &shape)
{
	const uint32_t rows = shape.rows,keys = shape.keys,pages = (keys + ProbeIndexKv::kPageSlots - 1u) / ProbeIndexKv::kPageSlots;
	const uint64_t score_count = (uint64_t)rows * keys;
	std::vector<uint16_t> query((size_t)rows * PROBE_HEADS * PROBE_DIM),heads((size_t)rows * PROBE_HEADS),key_values((size_t)pages * ProbeIndexKv::kPageSlots * PROBE_DIM);
	std::vector<uint32_t> positions(rows),page_table(pages);
	std::vector<uint8_t> key_codes,query_codes;
	std::vector<float> key_scales,head_weights,scores;
	uint32_t state = keys * 2654435761u + rows,index,head;
	uint16_t *device_query,*device_heads,*device_keys;
	uint8_t *device_key_codes,*device_query_codes;
	uint32_t *device_sequence,*device_context,*device_position,*device_page_table;
	float *device_key_scales,*device_weights,*device_scores,*device_bf16_scores,best_fp8 = 1e30f,best_bf16 = 1e30f,best_keys = 1e30f,best_query = 1e30f,elapsed;
	LmKvAccessError *device_error;
	cudaEvent_t start,stop;
	double flops = 0.0;
	for (index = 0u; index < rows; ++index)
		ProbeVector(&query[(size_t)index * PROBE_HEADS * PROBE_DIM],PROBE_HEADS * PROBE_DIM,&state,index % 7u == 3u ? 40.0f : 1.5f);
	ProbeVector(heads.data(),(uint32_t)heads.size(),&state,1.0f);
	for (index = 0u; index < keys; ++index)
		ProbeVector(&key_values[(size_t)index * PROBE_DIM],PROBE_DIM,&state,index % 13u == 5u ? 300.0f : (index % 17u == 2u ? 0.002f : 2.0f));
	std::fill(key_values.begin(),key_values.begin() + PROBE_DIM,(uint16_t)0u);
	if ( keys > 1u )
		key_values[PROBE_DIM + 9u] = ProbeBf16Bits(1.75f);
	std::fill(query.begin() + PROBE_DIM,query.begin() + 2u * PROBE_DIM,(uint16_t)0x8000u);
	for (index = 0u; index < rows; ++index)
	{
		positions[index] = index % 5u == 0u ? keys - 1u : (uint32_t)(((uint64_t)index * 2654435761u) % keys);
		if ( index % 11u == 3u )
			positions[index] = index % 300u;
		flops += 2.0 * PROBE_HEADS * PROBE_DIM * (double)(positions[index] + 1u < keys ? positions[index] + 1u : keys);
	}
	for (index = 0u; index < pages; ++index)
		page_table[index] = index;
	PROBE_CUDA(cudaMalloc((void **)&device_query,query.size() * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_heads,heads.size() * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_keys,key_values.size() * sizeof(uint16_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_key_codes,(size_t)keys * PROBE_DIM));
	PROBE_CUDA(cudaMalloc((void **)&device_key_scales,(size_t)keys * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_query_codes,query.size()));
	PROBE_CUDA(cudaMalloc((void **)&device_weights,heads.size() * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_sequence,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_context,sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_position,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_page_table,pages * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_scores,score_count * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_bf16_scores,score_count * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&device_error,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemcpy(device_query,query.data(),query.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_heads,heads.data(),heads.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_keys,key_values.data(),key_values.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_sequence,0,rows * sizeof(uint32_t)));
	PROBE_CUDA(cudaMemcpy(device_context,&keys,sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_position,positions.data(),rows * sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_page_table,page_table.data(),pages * sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_error,0,sizeof(LmKvAccessError)));
	PROBE_CUDA(cudaMemset(device_scores,0x3c,score_count * sizeof(float)));
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	LmKvView pages_view;
	memset(&pages_view,0,sizeof(pages_view));
	pages_view.pool = (uint8_t *)device_keys;
	pages_view.page_table = device_page_table;
	pages_view.page_table_stride = pages;
	pages_view.sequence_count = 1u;
	pages_view.pool_page_count = pages;
	pages_view.access_error = device_error;
	LmIndexScorePackedView bf16_view;
	bf16_view.pages = pages_view;
	bf16_view.shard.degree = 1u;
	bf16_view.shard.rank = 0u;
	bf16_view.shard.grain = 1u;
	bf16_view.keys = (const uint8_t *)device_keys;
	LmIndexScoreFp8Keys fp8_keys;
	fp8_keys.pages = pages_view;
	fp8_keys.values = device_key_codes;
	fp8_keys.scales = device_key_scales;
	for (uint32_t pass = 0u; pass < 3u; ++pass)
	{
		PROBE_CUDA(cudaEventRecord(start));
		PROBE_CUDA(LmIndexKeysFp8Launch<PROBE_DIM>(device_keys,keys,device_key_codes,device_key_scales,0));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_keys = std::min(best_keys,elapsed);
		PROBE_CUDA(cudaEventRecord(start));
		PROBE_CUDA((LmIndexQueryFp8Launch<PROBE_HEADS,PROBE_DIM>(device_query,device_heads,rows,PROBE_WEIGHT_SCALE,device_query_codes,device_weights,0)));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_query = std::min(best_query,elapsed);
		PROBE_CUDA(cudaEventRecord(start));
		PROBE_CUDA((LmWeightedSparseScoreFp8Launch<PROBE_HEADS,PROBE_DIM>(device_query_codes,device_weights,fp8_keys,device_sequence,device_context,device_position,
			rows,keys,device_scores,0)));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_fp8 = std::min(best_fp8,elapsed);
		PROBE_CUDA(cudaEventRecord(start));
		PROBE_CUDA((LmWeightedSparseScoreLaunch<ProbeIndexKv,PROBE_HEADS,PROBE_DIM,LmIndexScorePackedView>(device_query,device_heads,bf16_view,device_sequence,
			device_context,device_position,rows,keys,PROBE_WEIGHT_SCALE,device_bf16_scores,0)));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed,start,stop));
		best_bf16 = std::min(best_bf16,elapsed);
	}
	PROBE_CUDA(cudaDeviceSynchronize());
	printf("%s rows=%u keys=%u: FP8 score %.3f ms (%.1f TFLOPS), BF16 tile kernel %.3f ms (%.1f TFLOPS), %.2fx; key quantize %.3f ms, query quantize %.3f ms\n",
		shape.name,rows,keys,best_fp8,flops / best_fp8 / 1e9,best_bf16,flops / best_bf16 / 1e9,best_bf16 / best_fp8,best_keys,best_query);
	if ( shape.check != 0u )
	{
		std::vector<uint8_t> device_key_bytes((size_t)keys * PROBE_DIM),device_query_bytes(query.size());
		std::vector<float> device_key_scale_values(keys),device_weight_values(heads.size()),device_score_values(score_count);
		uint64_t key_mismatches = 0u,query_mismatches = 0u,score_mismatches = 0u,bound_mismatches = 0u;
		double worst = 0.0;
		key_codes.resize((size_t)keys * PROBE_DIM);
		key_scales.resize(keys);
		query_codes.resize(query.size());
		head_weights.resize(heads.size());
		for (index = 0u; index < keys; ++index)
			ProbeQuantizeRow(&key_values[(size_t)index * PROBE_DIM],1e-4f,false,&key_codes[(size_t)index * PROBE_DIM],&key_scales[index]);
		for (index = 0u; index < rows * PROBE_HEADS; ++index)
		{
			float scale;
			ProbeQuantizeRow(&query[(size_t)index * PROBE_DIM],1e-10f,true,&query_codes[(size_t)index * PROBE_DIM],&scale);
			head_weights[index] = ProbeBf16Value(heads[index]) * scale * PROBE_WEIGHT_SCALE;
		}
		PROBE_CUDA(cudaMemcpy(device_key_bytes.data(),device_key_codes,device_key_bytes.size(),cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(device_key_scale_values.data(),device_key_scales,keys * sizeof(float),cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(device_query_bytes.data(),device_query_codes,device_query_bytes.size(),cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(device_weight_values.data(),device_weights,heads.size() * sizeof(float),cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(device_score_values.data(),device_scores,score_count * sizeof(float),cudaMemcpyDeviceToHost));
		for (index = 0u; index < keys * PROBE_DIM; ++index)
			key_mismatches += device_key_bytes[index] != key_codes[index] ? 1u : 0u;
		for (index = 0u; index < keys; ++index)
			key_mismatches += memcmp(&device_key_scale_values[index],&key_scales[index],sizeof(float)) != 0 ? 1u : 0u;
		for (index = 0u; index < (uint32_t)query.size(); ++index)
			query_mismatches += device_query_bytes[index] != query_codes[index] ? 1u : 0u;
		for (index = 0u; index < (uint32_t)heads.size(); ++index)
			query_mismatches += memcmp(&device_weight_values[index],&head_weights[index],sizeof(float)) != 0 ? 1u : 0u;
		for (index = 0u; index < rows; ++index)
		{
			const uint32_t bound = positions[index] + 1u < keys ? positions[index] + 1u : keys;
			for (uint32_t key = 0u; key < keys; ++key)
			{
				const float got = device_score_values[(size_t)index * keys + key];
				double reference = 0.0,magnitude = 0.0;
				if ( key >= bound )
				{
					bound_mismatches += got == -INFINITY ? 0u : 1u;
					continue;
				}
				for (head = 0u; head < PROBE_HEADS; ++head)
				{
					const uint8_t *q = &query_codes[((size_t)index * PROBE_HEADS + head) * PROBE_DIM],*k = &key_codes[(size_t)key * PROBE_DIM];
					double dot = 0.0,absolute = 0.0;
					for (uint32_t element = 0u; element < PROBE_DIM; ++element)
					{
						dot += ProbeE4m3Value(q[element]) * ProbeE4m3Value(k[element]);
						absolute += fabs(ProbeE4m3Value(q[element]) * ProbeE4m3Value(k[element]));
					}
					reference += (double)head_weights[(size_t)index * PROBE_HEADS + head] * (dot > 0.0 ? dot : 0.0);
					magnitude += fabs((double)head_weights[(size_t)index * PROBE_HEADS + head]) * absolute;
				}
				reference *= key_scales[key];
				magnitude *= key_scales[key];
				const double error = fabs((double)got - reference);
				if ( magnitude > 0.0 )
					worst = std::max(worst,error / magnitude);
				if ( error > 2e-6 * magnitude + 1e-30 )
					score_mismatches++;
			}
		}
		if ( key_mismatches != 0u || query_mismatches != 0u || score_mismatches != 0u || bound_mismatches != 0u )
		{
			printf("FAIL %s: key codes or scales %llu, query codes or weights %llu, scores %llu, positions past the bound %llu\n",shape.name,
				(unsigned long long)key_mismatches,(unsigned long long)query_mismatches,(unsigned long long)score_mismatches,(unsigned long long)bound_mismatches);
			probe_failures++;
		}
		else
			printf("PASS %s: FP8 keys, scales, queries and folded weights equal the publisher's formulas bit for bit; every score within %.2e of FP64 over the FP8 values\n",
				shape.name,worst);
	}
	PROBE_CUDA(cudaEventDestroy(start));
	PROBE_CUDA(cudaEventDestroy(stop));
	PROBE_CUDA(cudaFree(device_query));
	PROBE_CUDA(cudaFree(device_heads));
	PROBE_CUDA(cudaFree(device_keys));
	PROBE_CUDA(cudaFree(device_key_codes));
	PROBE_CUDA(cudaFree(device_key_scales));
	PROBE_CUDA(cudaFree(device_query_codes));
	PROBE_CUDA(cudaFree(device_weights));
	PROBE_CUDA(cudaFree(device_sequence));
	PROBE_CUDA(cudaFree(device_context));
	PROBE_CUDA(cudaFree(device_position));
	PROBE_CUDA(cudaFree(device_page_table));
	PROBE_CUDA(cudaFree(device_scores));
	PROBE_CUDA(cudaFree(device_bf16_scores));
	PROBE_CUDA(cudaFree(device_error));
}

int main(void)
{
	static const ProbeShape shapes[] = {
		{"checked chunk", 128u, 3000u, 1u},
		{"checked short rows", 40u, 777u, 1u},
		{"pass chunk", 128u, 65536u, 0u},
		{"pass chunk, long", 128u, 262144u, 0u},
		{"pass chunk, 1M", 128u, 1048576u, 0u},
	};
	for (size_t index = 0; index < sizeof(shapes) / sizeof(shapes[0]); ++index)
		ProbeRun(shapes[index]);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d FP8 index cases\n",probe_failures);
		return 1;
	}
	printf("PASS the FP8 index quantizes as the publisher's engine and scores within FP32 rounding\n");
	return 0;
}
