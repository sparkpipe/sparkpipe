#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/attention_fp8.cuh"

#define FP8_HEAD 192u
#define FP8_VALUE 128u
#define FP8_HEADS 2u
#define FP8_PITCH (FP8_HEADS * (FP8_HEAD + FP8_VALUE))
#define FP8_ROWS 300u
#define FP8_KEYS 700u
#define FP8_SPLIT 512u
#define FP8_FIRST_POSITION 400u
#define FP8_SCALE 0.0721687836f

#define FP8_CUDA(call) do { cudaError_t fp8_error = (call); if ( fp8_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(fp8_error)); exit(1); } } while (0)

static int fp8_failures;

static void Fp8Check(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	fp8_failures++;
}

static uint64_t Fp8Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return(z ^ (z >> 31u));
}

static float Fp8Uniform(uint64_t seed)
{
	return((float)((Fp8Mix(seed) >> 11u) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f);
}

static uint16_t Fp8ToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,4u);
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float Fp8FromBf16(uint16_t value)
{
	const uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,4u);
	return(out);
}

static float Fp8RoundE4m3(float value)
{
	const float magnitude = fabsf(value);
	float step, rounded;
	int exponent;
	if ( magnitude == 0.0f )
		return(0.0f);
	if ( magnitude >= 448.0f )
		return(value < 0.0f ? -448.0f : 448.0f);
	frexpf(magnitude,&exponent);
	step = magnitude < 0.015625f ? 0.001953125f : ldexpf(1.0f,exponent - 4);
	rounded = nearbyintf(magnitude / step) * step;
	rounded = rounded > 448.0f ? 448.0f : rounded;
	return(value < 0.0f ? -rounded : rounded);
}

static float Fp8ScaleOf(float amax)
{
	return(amax > 0.0f ? amax / 448.0f : 1.0f);
}

struct Fp8Data
{
	std::vector<uint16_t> query;
	std::vector<uint16_t> slots;
	std::vector<uint32_t> positions;
};

static Fp8Data Fp8Make(void)
{
	Fp8Data data;
	data.query.resize((size_t)FP8_ROWS * FP8_HEADS * FP8_HEAD);
	data.slots.resize((size_t)FP8_KEYS * FP8_PITCH);
	data.positions.resize(FP8_ROWS);
	for (uint32_t row = 0u; row < FP8_ROWS; ++row)
	{
		data.positions[row] = FP8_FIRST_POSITION + row;
		for (uint32_t head = 0u; head < FP8_HEADS; ++head)
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				data.query[((size_t)row * FP8_HEADS + head) * FP8_HEAD + d] =
					Fp8ToBf16(1.5f * Fp8Uniform(1000003ull * row + 7919ull * head + d) + (d < 8u ? 2.0f : 0.0f));
	}
	for (uint32_t key = 0u; key < FP8_KEYS; ++key)
		for (uint32_t head = 0u; head < FP8_HEADS; ++head)
		{
			const float sink = key % 97u == 0u ? 3.0f : 0.0f;
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				data.slots[(size_t)key * FP8_PITCH + head * FP8_HEAD + d] =
					Fp8ToBf16(1.5f * Fp8Uniform(77ull + 1000033ull * key + 104729ull * head + d) + (d < 8u ? sink : 0.0f));
			for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				data.slots[(size_t)key * FP8_PITCH + FP8_HEADS * FP8_HEAD + head * FP8_VALUE + d] =
					Fp8ToBf16(Fp8Uniform(991ull + 1000037ull * key + 15485863ull * head + d) * (d % 37u == 5u ? 24.0f : 1.0f));
		}
	return(data);
}

static void Fp8EmulateBlock(const Fp8Data &data, uint32_t key_first, uint32_t key_count, std::vector<float> &records)
{
	const uint32_t width = FP8_HEADS * FP8_VALUE, record = LmGqaShardRecordFloats(FP8_HEADS,FP8_VALUE,1u);
	records.assign((size_t)FP8_ROWS * record,0.0f);
	for (uint32_t head = 0u; head < FP8_HEADS; ++head)
	{
		std::vector<float> key8((size_t)key_count * FP8_HEAD), key_scale(key_count), value8((size_t)key_count * FP8_VALUE), value_scale(FP8_VALUE,0.0f);
		for (uint32_t key = 0u; key < key_count; ++key)
		{
			const uint16_t *row = &data.slots[(size_t)(key_first + key) * FP8_PITCH + head * FP8_HEAD];
			float amax = 0.0f;
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				amax = fmaxf(amax,fabsf(Fp8FromBf16(row[d])));
			key_scale[key] = Fp8ScaleOf(amax);
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				key8[(size_t)key * FP8_HEAD + d] = Fp8RoundE4m3(Fp8FromBf16(row[d]) / key_scale[key]);
			for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				value_scale[d] = fmaxf(value_scale[d],fabsf(Fp8FromBf16(data.slots[(size_t)(key_first + key) * FP8_PITCH + FP8_HEADS * FP8_HEAD + head * FP8_VALUE + d])));
		}
		for (uint32_t d = 0u; d < FP8_VALUE; ++d)
			value_scale[d] = Fp8ScaleOf(value_scale[d]);
		for (uint32_t key = 0u; key < key_count; ++key)
			for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				value8[(size_t)key * FP8_VALUE + d] = Fp8RoundE4m3(
					Fp8FromBf16(data.slots[(size_t)(key_first + key) * FP8_PITCH + FP8_HEADS * FP8_HEAD + head * FP8_VALUE + d]) / value_scale[d]);
		for (uint32_t row = 0u; row < FP8_ROWS; ++row)
		{
			const uint16_t *source = &data.query[((size_t)row * FP8_HEADS + head) * FP8_HEAD];
			const uint32_t position = data.positions[row];
			const uint32_t visible = position < key_first ? 0u : (position - key_first + 1u < key_count ? position - key_first + 1u : key_count);
			std::vector<double> output(FP8_VALUE,0.0);
			float query8[FP8_HEAD], amax = 0.0f, query_scale, running_max = -INFINITY, sum = 0.0f;
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				amax = fmaxf(amax,fabsf(Fp8FromBf16(source[d])));
			query_scale = Fp8ScaleOf(amax);
			for (uint32_t d = 0u; d < FP8_HEAD; ++d)
				query8[d] = Fp8RoundE4m3(Fp8FromBf16(source[d]) / query_scale);
			for (uint32_t tile = 0u; tile * LM_FP8_ATTENTION_TILE < visible; ++tile)
			{
				const uint32_t end = (tile + 1u) * LM_FP8_ATTENTION_TILE < visible ? (tile + 1u) * LM_FP8_ATTENTION_TILE : visible;
				float score[LM_FP8_ATTENTION_TILE], peak = -INFINITY, next, rescale;
				for (uint32_t key = tile * LM_FP8_ATTENTION_TILE; key < end; ++key)
				{
					double dot = 0.0;
					for (uint32_t d = 0u; d < FP8_HEAD; ++d)
						dot += (double)query8[d] * key8[(size_t)key * FP8_HEAD + d];
					score[key - tile * LM_FP8_ATTENTION_TILE] = (float)dot * query_scale * key_scale[key];
					peak = fmaxf(peak,score[key - tile * LM_FP8_ATTENTION_TILE]);
				}
				next = fmaxf(running_max,peak);
				rescale = running_max == -INFINITY ? 0.0f : exp2f((running_max - next) * FP8_SCALE * 1.4426950408889634f);
				running_max = next;
				sum *= rescale;
				for (uint32_t d = 0u; d < FP8_VALUE; ++d)
					output[d] *= rescale;
				for (uint32_t key = tile * LM_FP8_ATTENTION_TILE; key < end; ++key)
				{
					const float weight = Fp8RoundE4m3(exp2f((score[key - tile * LM_FP8_ATTENTION_TILE] - next) * FP8_SCALE * 1.4426950408889634f) * 448.0f);
					sum += weight;
					for (uint32_t d = 0u; d < FP8_VALUE; ++d)
						output[d] += (double)weight * value8[(size_t)key * FP8_VALUE + d];
				}
			}
			float *base = &records[(size_t)row * record];
			base[width + 2u * head] = running_max == -INFINITY ? -INFINITY : running_max * FP8_SCALE;
			base[width + 2u * head + 1u] = sum / 448.0f;
			for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				base[head * FP8_VALUE + d] = (float)output[d] * value_scale[d] / 448.0f;
		}
	}
}

static void Fp8Exact(const Fp8Data &data, std::vector<float> &out)
{
	out.assign((size_t)FP8_ROWS * FP8_HEADS * FP8_VALUE,0.0f);
	for (uint32_t row = 0u; row < FP8_ROWS; ++row)
		for (uint32_t head = 0u; head < FP8_HEADS; ++head)
		{
			const uint32_t visible = data.positions[row] + 1u < FP8_KEYS ? data.positions[row] + 1u : FP8_KEYS;
			std::vector<double> score(visible), output(FP8_VALUE,0.0);
			double peak = -INFINITY, sum = 0.0;
			for (uint32_t key = 0u; key < visible; ++key)
			{
				double dot = 0.0;
				for (uint32_t d = 0u; d < FP8_HEAD; ++d)
					dot += (double)Fp8FromBf16(data.query[((size_t)row * FP8_HEADS + head) * FP8_HEAD + d]) * Fp8FromBf16(data.slots[(size_t)key * FP8_PITCH + head * FP8_HEAD + d]);
				score[key] = dot * FP8_SCALE;
				peak = score[key] > peak ? score[key] : peak;
			}
			for (uint32_t key = 0u; key < visible; ++key)
			{
				const double weight = exp(score[key] - peak);
				sum += weight;
				for (uint32_t d = 0u; d < FP8_VALUE; ++d)
					output[d] += weight * Fp8FromBf16(data.slots[(size_t)key * FP8_PITCH + FP8_HEADS * FP8_HEAD + head * FP8_VALUE + d]);
			}
			for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				out[((size_t)row * FP8_HEADS + head) * FP8_VALUE + d] = (float)(output[d] / sum);
		}
}

static void Fp8RunBlock(const uint16_t *query, const uint16_t *slots, const uint32_t *positions, uint32_t key_first, uint32_t key_count,
	float *send, std::vector<float> &records)
{
	const uint32_t record = LmGqaShardRecordFloats(FP8_HEADS,FP8_VALUE,1u), capacity = key_count + 3u;
	uint8_t *keys8, *values8;
	float *key_scale;
	uint32_t *value_amax;
	FP8_CUDA(cudaMalloc(&keys8,(size_t)FP8_HEADS * capacity * FP8_HEAD));
	FP8_CUDA(cudaMalloc(&key_scale,(size_t)FP8_HEADS * capacity * sizeof(float)));
	FP8_CUDA(cudaMalloc(&values8,(size_t)FP8_HEADS * LmFp8AttentionTiles(capacity) * FP8_VALUE * LM_FP8_ATTENTION_TILE));
	FP8_CUDA(cudaMalloc(&value_amax,(size_t)FP8_HEADS * FP8_VALUE * sizeof(uint32_t)));
	FP8_CUDA(cudaMemset(send,0x7f,(size_t)FP8_ROWS * record * sizeof(float)));
	FP8_CUDA((LmFp8AttentionQuantizeLaunch<FP8_HEAD,FP8_VALUE>(slots + (size_t)key_first * FP8_PITCH,FP8_PITCH,FP8_HEADS,key_count,capacity,keys8,key_scale,values8,value_amax,0)));
	FP8_CUDA((LmFp8AttentionPartialLaunch<FP8_HEAD,FP8_VALUE>(query,FP8_HEADS,positions,FP8_ROWS,keys8,key_scale,values8,value_amax,key_first,key_count,capacity,FP8_SCALE,
		send,(uint64_t)FP8_ROWS * record,0)));
	FP8_CUDA(cudaDeviceSynchronize());
	records.resize((size_t)FP8_ROWS * record);
	FP8_CUDA(cudaMemcpy(records.data(),send,records.size() * sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(keys8);
	cudaFree(key_scale);
	cudaFree(values8);
	cudaFree(value_amax);
}

static double Fp8CompareRecords(const std::vector<float> &got, const std::vector<float> &want, const char *label)
{
	const uint32_t width = FP8_HEADS * FP8_VALUE, record = LmGqaShardRecordFloats(FP8_HEADS,FP8_VALUE,1u);
	double numerator = 0.0, denominator = 0.0;
	uint32_t slot_mismatch = 0u;
	for (uint32_t row = 0u; row < FP8_ROWS; ++row)
	{
		const float *g = &got[(size_t)row * record], *w = &want[(size_t)row * record];
		for (uint32_t head = 0u; head < FP8_HEADS; ++head)
		{
			const float gm = g[width + 2u * head], wm = w[width + 2u * head];
			if ( (isinf(wm) != 0) != (isinf(gm) != 0) || (isinf(wm) == 0 && fabsf(gm - wm) > 1e-3f * fmaxf(1.0f,fabsf(wm))) ||
				fabsf(g[width + 2u * head + 1u] - w[width + 2u * head + 1u]) > 2e-3f * fmaxf(1.0f,w[width + 2u * head + 1u]) )
				slot_mismatch++;
		}
		for (uint32_t index = 0u; index < width; ++index)
		{
			numerator += ((double)g[index] - w[index]) * ((double)g[index] - w[index]);
			denominator += (double)w[index] * w[index];
		}
	}
	printf("%s: emulated recipe rel rms %.3e, max/sum slot mismatches %u\n",label,sqrt(numerator / fmax(denominator,1e-30)),slot_mismatch);
	Fp8Check(slot_mismatch == 0u,"every row's max and sum match the emulated online softmax");
	return(sqrt(numerator / fmax(denominator,1e-30)));
}

int main(void)
{
	const Fp8Data data = Fp8Make();
	const uint32_t record = LmGqaShardRecordFloats(FP8_HEADS,FP8_VALUE,1u);
	std::vector<float> got_whole, want_whole, got_first, want_first, got_second, want_second, exact;
	uint16_t *query, *slots, *merged_device;
	uint32_t *positions;
	float *whole, *blocks;
	double numerator = 0.0, denominator = 0.0;
	uint32_t empty_rows = 0u;
	FP8_CUDA(cudaMalloc(&query,data.query.size() * sizeof(uint16_t)));
	FP8_CUDA(cudaMalloc(&slots,data.slots.size() * sizeof(uint16_t)));
	FP8_CUDA(cudaMalloc(&positions,data.positions.size() * sizeof(uint32_t)));
	FP8_CUDA(cudaMalloc(&whole,(size_t)FP8_ROWS * record * sizeof(float)));
	FP8_CUDA(cudaMalloc(&blocks,2u * (size_t)FP8_ROWS * record * sizeof(float)));
	FP8_CUDA(cudaMalloc(&merged_device,(size_t)FP8_ROWS * FP8_HEADS * FP8_VALUE * sizeof(uint16_t)));
	FP8_CUDA(cudaMemcpy(query,data.query.data(),data.query.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	FP8_CUDA(cudaMemcpy(slots,data.slots.data(),data.slots.size() * sizeof(uint16_t),cudaMemcpyHostToDevice));
	FP8_CUDA(cudaMemcpy(positions,data.positions.data(),data.positions.size() * sizeof(uint32_t),cudaMemcpyHostToDevice));

	Fp8RunBlock(query,slots,positions,0u,FP8_KEYS,whole,got_whole);
	Fp8EmulateBlock(data,0u,FP8_KEYS,want_whole);
	Fp8Check(Fp8CompareRecords(got_whole,want_whole,"one block of 700 keys") < 2e-3,"the kernel computes the fp8 recipe over one block");

	Fp8RunBlock(query,slots,positions,0u,FP8_SPLIT,blocks,got_first);
	Fp8EmulateBlock(data,0u,FP8_SPLIT,want_first);
	Fp8Check(Fp8CompareRecords(got_first,want_first,"keys 0..512") < 2e-3,"the kernel computes the fp8 recipe over the first block");
	Fp8RunBlock(query,slots,positions,FP8_SPLIT,FP8_KEYS - FP8_SPLIT,blocks + (size_t)FP8_ROWS * record,got_second);
	Fp8EmulateBlock(data,FP8_SPLIT,FP8_KEYS - FP8_SPLIT,want_second);
	Fp8Check(Fp8CompareRecords(got_second,want_second,"keys 512..700") < 2e-3,"the kernel computes the fp8 recipe over a later block");
	for (uint32_t row = 0u; row < FP8_ROWS; ++row)
		if ( data.positions[row] < FP8_SPLIT )
		{
			const float *base = &got_second[(size_t)row * record];
			int empty = 1;
			for (uint32_t head = 0u; head < FP8_HEADS; ++head)
				empty = empty && isinf(base[FP8_HEADS * FP8_VALUE + 2u * head]) && base[FP8_HEADS * FP8_VALUE + 2u * head] < 0.0f &&
					base[FP8_HEADS * FP8_VALUE + 2u * head + 1u] == 0.0f;
			empty_rows += empty ? 1u : 0u;
		}
	Fp8Check(empty_rows == FP8_SPLIT - FP8_FIRST_POSITION,"a row that sees no key of a block records an empty softmax for it");

	Fp8Exact(data,exact);
	{
		const uint32_t width = FP8_HEADS * FP8_VALUE;
		for (uint32_t row = 0u; row < FP8_ROWS; ++row)
			for (uint32_t head = 0u; head < FP8_HEADS; ++head)
			{
				const float *a = &got_first[(size_t)row * record], *b = &got_second[(size_t)row * record];
				const float am = a[width + 2u * head], bm = b[width + 2u * head], m = fmaxf(am,bm);
				const float sa = isinf(am) ? 0.0f : expf(am - m), sb = isinf(bm) ? 0.0f : expf(bm - m);
				const float denominator_row = a[width + 2u * head + 1u] * sa + b[width + 2u * head + 1u] * sb;
				for (uint32_t d = 0u; d < FP8_VALUE; ++d)
				{
					const double value = (a[head * FP8_VALUE + d] * sa + b[head * FP8_VALUE + d] * sb) / denominator_row;
					const double reference = exact[((size_t)row * FP8_HEADS + head) * FP8_VALUE + d];
					numerator += (value - reference) * (value - reference);
					denominator += reference * reference;
				}
			}
	}
	printf("two blocks merged vs exact fp32 attention: rel rms %.3e\n",sqrt(numerator / denominator));
	Fp8Check(sqrt(numerator / denominator) < 1e-1,"two merged fp8 blocks stay within fp8 precision of exact attention");
	{
		std::vector<uint16_t> one((size_t)FP8_ROWS * FP8_HEADS * FP8_VALUE);
		double n1 = 0.0, d1 = 0.0;
		FP8_CUDA((LmGqaShardMergeLaunch<FP8_VALUE>(whole,(uint64_t)FP8_ROWS * record,1u,FP8_HEADS,0u,merged_device,FP8_ROWS,0)));
		FP8_CUDA(cudaDeviceSynchronize());
		FP8_CUDA(cudaMemcpy(one.data(),merged_device,one.size() * sizeof(uint16_t),cudaMemcpyDeviceToHost));
		for (size_t index = 0u; index < one.size(); ++index)
		{
			n1 += ((double)Fp8FromBf16(one[index]) - exact[index]) * ((double)Fp8FromBf16(one[index]) - exact[index]);
			d1 += (double)exact[index] * exact[index];
		}
		printf("one block normalized by the shared merge vs exact fp32 attention: rel rms %.3e\n",sqrt(n1 / d1));
		Fp8Check(sqrt(n1 / d1) < 1e-1,"the shared merge normalizes fp8 records into attention within fp8 precision");
	}
	cudaFree(query);
	cudaFree(slots);
	cudaFree(positions);
	cudaFree(whole);
	cudaFree(blocks);
	cudaFree(merged_device);
	if ( fp8_failures != 0 )
	{
		printf("attention_fp8_cuda: %d failures\n",fp8_failures);
		return(1);
	}
	printf("attention_fp8_cuda: PASS\n");
	return(0);
}
