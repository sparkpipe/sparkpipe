#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/gemm.cuh"
#include "inference/kernels/cell_mx.cuh"
#include "inference/kernels/attn_expanded_prefill.cuh"

#define BENCH_WARMUP 5u
#define BENCH_ITERS 20u

#define CHECK_CUDA(...) do { cudaError_t err_ = (__VA_ARGS__); if ( err_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #__VA_ARGS__, cudaGetErrorString(err_)); exit(1); } } while (0)
#define CHECK_GPU(...) do { SparkGpuError err_ = (__VA_ARGS__); if ( err_ != SPARK_GPU_SUCCESS ) { printf("FAIL gpu %s: %d\n", #__VA_ARGS__, (int)err_); exit(1); } } while (0)
#define CHECK_OK(...) do { int32_t st_ = (__VA_ARGS__); if ( st_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #__VA_ARGS__, st_); exit(1); } } while (0)

static int g_failures;
static uint32_t g_sms;

static __host__ __device__ __forceinline__ uint64_t Hash64(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static __global__ void FillFp8Kernel(uint8_t *dst, uint64_t count, uint64_t seed)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
	{
		uint8_t code = (uint8_t)Hash64(idx * 3u + seed);
		dst[idx] = (code & 0x7fu) == 0x7fu ? (uint8_t)(code ^ 1u) : code;
	}
}

static __global__ void FillScaleKernel(uint8_t *dst, uint64_t count, uint64_t seed, uint32_t base_exp)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
		dst[idx] = (uint8_t)(base_exp + (uint32_t)(Hash64(idx + seed) % 9u));
}

static __global__ void FillBf16Kernel(uint16_t *dst, uint64_t count, uint64_t seed, float scale)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
	{
		const float v = (float)((int64_t)(Hash64(idx + seed) % 2001u) - 1000) * scale;
		dst[idx] = LmFloatToBf16(v);
	}
}

static __global__ void FillCellWeightKernel(uint8_t *dst, uint64_t cells_total, uint64_t seed)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t total_bytes = cells_total * 272u;
	if ( idx < total_bytes )
	{
		const uint32_t off = (uint32_t)(idx % 272u);
		if ( off >= 256u )
			dst[idx] = (uint8_t)(122u + (uint32_t)(Hash64(idx + seed) % 9u));
		else
			dst[idx] = (uint8_t)Hash64(idx * 7u + seed);
	}
}

static __global__ void CountDiffU16Kernel(const uint16_t *a, const uint16_t *b, uint64_t count, uint64_t *diffs)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count && a[idx] != b[idx] )
		atomicAdd((unsigned long long *)diffs, 1ull);
}

static __global__ void CountDiffF32Kernel(const float *a, const float *b, uint64_t count, uint64_t *diffs)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count && __float_as_uint(a[idx]) != __float_as_uint(b[idx]) )
		atomicAdd((unsigned long long *)diffs, 1ull);
}

static uint64_t DiffCountU16(const uint16_t *a, const uint16_t *b, uint64_t count, cudaStream_t stream)
{
	uint64_t *d_diffs = 0, host_diffs = 0u;
	CHECK_CUDA(cudaMalloc(&d_diffs, sizeof(uint64_t)));
	CHECK_CUDA(cudaMemsetAsync(d_diffs, 0, sizeof(uint64_t), stream));
	CountDiffU16Kernel<<<(uint32_t)((count + 255u) / 256u), 256u, 0u, stream>>>(a, b, count, d_diffs);
	CHECK_CUDA(cudaMemcpyAsync(&host_diffs, d_diffs, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CHECK_CUDA(cudaFree(d_diffs));
	return host_diffs;
}

static uint64_t DiffCountF32(const float *a, const float *b, uint64_t count, cudaStream_t stream)
{
	uint64_t *d_diffs = 0, host_diffs = 0u;
	CHECK_CUDA(cudaMalloc(&d_diffs, sizeof(uint64_t)));
	CHECK_CUDA(cudaMemsetAsync(d_diffs, 0, sizeof(uint64_t), stream));
	CountDiffF32Kernel<<<(uint32_t)((count + 255u) / 256u), 256u, 0u, stream>>>(a, b, count, d_diffs);
	CHECK_CUDA(cudaMemcpyAsync(&host_diffs, d_diffs, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CHECK_CUDA(cudaFree(d_diffs));
	return host_diffs;
}

static double E2m1Host(uint32_t nibble)
{
	static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
	return (nibble & 8u) != 0u ? -mag[nibble & 7u] : mag[nibble & 7u];
}

static double E4m3Host(uint32_t code)
{
	const uint32_t exp = (code >> 3u) & 15u, man = code & 7u;
	const double mag = exp == 0u ? ldexp((double)man, -9) : ldexp(1.0 + man / 8.0, (int)exp - 7);
	return (code & 0x80u) != 0u ? -mag : mag;
}

static uint8_t E4m3FromValue(double value)
{
	for ( uint32_t code = 0u; code < 256u; code++ )
		if ( (code & 0x7fu) != 0x7fu && E4m3Host(code) == value && (value != 0.0 || code == (signbit(value) ? 0x80u : 0u)) )
			return (uint8_t)code;
	return 0x7fu;
}

static uint64_t CellByteOffset(uint32_t neuron, uint32_t k)
{
	const uint32_t half = neuron / 8u, g = neuron % 8u, c = k / 8u;
	return (uint64_t)half * 128u + (uint64_t)(g * 4u + c) * 4u + (k % 8u) / 2u;
}

static __global__ void PublisherFp4RefKernel(const uint8_t *codes, const uint8_t *code_scales, const uint8_t *weight_codes, const uint8_t *weight_scales,
	const uint32_t *offset, const uint32_t *source, uint32_t input, uint32_t output, uint16_t *out)
{
	const uint32_t group = blockIdx.z, lane = threadIdx.x, g = lane / 4u, t = lane % 4u, blocks = input / 32u;
	const uint32_t first = offset[group] + blockIdx.y * 16u, end = offset[group + 1u], column = blockIdx.x * 8u;
	float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	if ( first >= end )
		return;
	for ( uint32_t block = 0u; block < blocks; block++ )
	{
		uint32_t a[4], b[2];
		float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		for ( uint32_t r = 0u; r < 4u; r++ )
		{
			const uint32_t row = first + g + 8u * (r & 1u), k = block * 32u + t * 4u + 16u * (r >> 1u);
			a[r] = row < end ? *(const uint32_t *)(codes + (uint64_t)(source != 0 ? source[row] : row) * input + k) : 0u;
		}
		for ( uint32_t r = 0u; r < 2u; r++ )
			b[r] = *(const uint32_t *)(weight_codes + ((uint64_t)group * output + column + g) * input + block * 32u + t * 4u + 16u * r);
		LmMmaE4m3(part, a, b);
		for ( uint32_t e = 0u; e < 4u; e++ )
		{
			const uint32_t row = first + g + 8u * (e >> 1u), neuron = column + t * 2u + (e & 1u);
			const float sa = row < end ? __uint_as_float((uint32_t)code_scales[(uint64_t)(source != 0 ? source[row] : row) * blocks + block] << 23u) : 0.0f;
			const float sb = __uint_as_float((uint32_t)weight_scales[((uint64_t)group * output + neuron) * blocks + block] << 23u);
			total[e] += part[e] * sa * sb;
		}
	}
	for ( uint32_t e = 0u; e < 4u; e++ )
	{
		const uint32_t row = first + g + 8u * (e >> 1u), neuron = column + t * 2u + (e & 1u);
		if ( row < end )
			out[(uint64_t)row * output + neuron] = LmFloatToBf16(total[e]);
	}
}

static __global__ void Bf16MmaRefKernel(const uint16_t *act, const uint16_t *weight, const uint32_t *offset, uint32_t input, uint32_t output, uint16_t *out)
{
	const uint32_t group = blockIdx.z, lane = threadIdx.x, g = lane / 4u, t = lane % 4u;
	const uint32_t first = offset[group] + blockIdx.y * 16u, end = offset[group + 1u], column = blockIdx.x * 8u;
	float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	if ( first >= end || column >= output )
		return;
	const uint32_t r0 = first + g, r1 = r0 + 8u;
	const uint16_t *w_row = weight + ((uint64_t)group * output + column + g) * input;
	for ( uint32_t k = 0u; k < input; k += 16u )
	{
		uint32_t a[4], b[2];
		a[0] = r0 < end ? *(const uint32_t *)(act + (uint64_t)r0 * input + k + 2u * t) : 0u;
		a[1] = r1 < end ? *(const uint32_t *)(act + (uint64_t)r1 * input + k + 2u * t) : 0u;
		a[2] = r0 < end ? *(const uint32_t *)(act + (uint64_t)r0 * input + k + 8u + 2u * t) : 0u;
		a[3] = r1 < end ? *(const uint32_t *)(act + (uint64_t)r1 * input + k + 8u + 2u * t) : 0u;
		b[0] = *(const uint32_t *)(w_row + k + 2u * t);
		b[1] = *(const uint32_t *)(w_row + k + 8u + 2u * t);
		LmMmaBf16(acc, a, b);
	}
	if ( r0 < end )
		*(uint32_t *)(out + (uint64_t)r0 * output + column + 2u * t) = LmPackBf16Pair(acc[0], acc[1]);
	if ( r1 < end )
		*(uint32_t *)(out + (uint64_t)r1 * output + column + 2u * t) = LmPackBf16Pair(acc[2], acc[3]);
}

static void BuildRoutes(uint32_t groups, uint32_t tokens, uint32_t top_k, bool skewed, std::vector<uint32_t> &offset, std::vector<uint32_t> &source)
{
	const uint32_t packed = tokens * top_k;
	std::vector<std::vector<uint32_t>> lists(groups);
	for ( uint32_t t = 0u; t < tokens; t++ )
		for ( uint32_t r = 0u; r < top_k; r++ )
		{
			uint32_t g;
			if ( skewed )
			{
				const uint64_t h = Hash64((uint64_t)t * 131u + r * 17u + 99u);
				g = (h % 10u) < 4u ? (uint32_t)(h % (groups / 8u > 0u ? groups / 8u : 1u)) : (uint32_t)(h % groups);
			}
			else
				g = (uint32_t)((Hash64((uint64_t)t * 31u + r) % (groups / top_k > 0u ? groups / top_k : 1u)) * top_k + r) % groups;
			lists[g].push_back(t);
		}
	offset.assign(groups + 1u, 0u);
	source.resize(packed);
	uint32_t at = 0u;
	for ( uint32_t g = 0u; g < groups; g++ )
	{
		offset[g] = at;
		for ( uint32_t tok : lists[g] )
			source[at++] = tok;
		offset[g + 1u] = at;
	}
}

static void VerifyCellMxExact(uint32_t groups, uint32_t tokens, uint32_t top_k, uint32_t input, uint32_t output, bool skewed, cudaStream_t stream)
{
	const uint32_t tiles = input / 32u, cells = output / 16u, packed = tokens * top_k;
	std::vector<uint8_t> weight((uint64_t)groups * tiles * cells * 272u, 0u);
	std::vector<uint8_t> weight_codes((uint64_t)groups * output * input);
	std::vector<uint8_t> weight_scales((uint64_t)groups * output * tiles);
	uint8_t e2m1_e4m3[16];
	for ( uint32_t n = 0u; n < 16u; n++ )
		e2m1_e4m3[n] = E4m3FromValue(E2m1Host(n));
	for ( uint32_t g = 0u; g < groups; g++ )
		for ( uint32_t t = 0u; t < tiles; t++ )
			for ( uint32_t c = 0u; c < cells; c++ )
			{
				uint8_t *base = &weight[(((uint64_t)g * tiles + t) * cells + c) * 272u];
				for ( uint32_t n = 0u; n < 16u; n++ )
				{
					const uint32_t sc = 122u + (uint32_t)(Hash64(((uint64_t)g * 1024u + c) * 64u + t * 17u + n) % 9u);
					base[256u + n] = (uint8_t)sc;
					weight_scales[((uint64_t)g * output + c * 16u + n) * tiles + t] = (uint8_t)sc;
					for ( uint32_t k = 0u; k < 32u; k++ )
					{
						const uint32_t nib = (uint32_t)(Hash64((((uint64_t)g * tiles + t) * output + c * 16u + n) * 32u + k) & 15u);
						base[CellByteOffset(n, k)] |= (uint8_t)(nib << ((k & 1u) * 4u));
						weight_codes[((uint64_t)g * output + c * 16u + n) * input + t * 32u + k] = e2m1_e4m3[nib];
					}
				}
			}
	std::vector<uint32_t> offset, source;
	BuildRoutes(groups, tokens, top_k, skewed, offset, source);
	uint32_t longest = 0u;
	for ( uint32_t g = 0u; g < groups; g++ )
	{
		const uint32_t len = offset[g + 1u] - offset[g];
		if ( len > longest )
			longest = len;
	}
	uint8_t *d_w = 0, *d_wc = 0, *d_ws = 0, *d_c = 0, *d_cs = 0;
	uint32_t *d_off = 0, *d_src = 0;
	uint16_t *d_mx = 0, *d_ref = 0;
	CHECK_CUDA(cudaMalloc(&d_w, weight.size()));
	CHECK_CUDA(cudaMalloc(&d_wc, weight_codes.size()));
	CHECK_CUDA(cudaMalloc(&d_ws, weight_scales.size()));
	CHECK_CUDA(cudaMalloc(&d_c, (uint64_t)tokens * input));
	CHECK_CUDA(cudaMalloc(&d_cs, (uint64_t)tokens * tiles));
	CHECK_CUDA(cudaMalloc(&d_off, offset.size() * 4u));
	CHECK_CUDA(cudaMalloc(&d_src, source.size() * 4u));
	CHECK_CUDA(cudaMalloc(&d_mx, (uint64_t)packed * output * 2u));
	CHECK_CUDA(cudaMalloc(&d_ref, (uint64_t)packed * output * 2u));
	CHECK_CUDA(cudaMemcpyAsync(d_w, weight.data(), weight.size(), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_wc, weight_codes.data(), weight_codes.size(), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_ws, weight_scales.data(), weight_scales.size(), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_off, offset.data(), offset.size() * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_src, source.data(), source.size() * 4u, cudaMemcpyHostToDevice, stream));
	FillFp8Kernel<<<(uint32_t)(((uint64_t)tokens * input + 255u) / 256u), 256u, 0u, stream>>>(d_c, (uint64_t)tokens * input, 701u + tokens);
	FillScaleKernel<<<(uint32_t)(((uint64_t)tokens * tiles + 255u) / 256u), 256u, 0u, stream>>>(d_cs, (uint64_t)tokens * tiles, 809u + tokens, 118u);
	CHECK_CUDA(cudaMemsetAsync(d_mx, 0xff, (uint64_t)packed * output * 2u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_ref, 0xff, (uint64_t)packed * output * 2u, stream));
	CHECK_OK(LmCellMxExperts(d_w, d_c, d_cs, d_mx, d_off, d_src, groups, packed, 0u, input, output, stream));
	PublisherFp4RefKernel<<<dim3(output / 8u, (longest + 15u) / 16u, groups), 32u, 0u, stream>>>(
		d_c, d_cs, d_wc, d_ws, d_off, d_src, input, output, d_ref);
	const uint64_t diff = DiffCountU16(d_ref, d_mx, (uint64_t)packed * output, stream);
	if ( diff != 0u )
		g_failures++;
	printf("%s cell_mx_exact groups=%u M=%u routes=%u %s K=%u N=%u: diff=%llu\n",
		diff == 0u ? "PASS" : "FAIL", groups, tokens, packed, skewed ? "skewed" : "uniform", input, output, (unsigned long long)diff);
	CHECK_CUDA(cudaFree(d_w));
	CHECK_CUDA(cudaFree(d_wc));
	CHECK_CUDA(cudaFree(d_ws));
	CHECK_CUDA(cudaFree(d_c));
	CHECK_CUDA(cudaFree(d_cs));
	CHECK_CUDA(cudaFree(d_off));
	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_mx));
	CHECK_CUDA(cudaFree(d_ref));
}

static void BenchCellMxShapes(cudaStream_t stream)
{
	static const uint32_t ms[] = {64u, 128u, 256u, 512u, 1024u, 2048u};
	const uint32_t groups = 64u, top_k = 8u, input = 5120u, output = 2048u;
	const uint64_t cells_total = (uint64_t)groups * (input / 32u) * (output / 16u);
	const uint32_t max_m = 2048u, max_packed = max_m * top_k;
	uint8_t *d_w = 0, *d_c = 0, *d_cs = 0;
	uint32_t *d_off = 0, *d_src = 0;
	uint16_t *d_out = 0;
	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	CHECK_CUDA(cudaMalloc(&d_w, cells_total * 272u));
	CHECK_CUDA(cudaMalloc(&d_c, (uint64_t)max_m * input));
	CHECK_CUDA(cudaMalloc(&d_cs, (uint64_t)max_m * (input / 32u)));
	CHECK_CUDA(cudaMalloc(&d_off, (groups + 1u) * 4u));
	CHECK_CUDA(cudaMalloc(&d_src, (uint64_t)max_packed * 4u));
	CHECK_CUDA(cudaMalloc(&d_out, (uint64_t)max_packed * output * 2u));
	FillCellWeightKernel<<<(uint32_t)((cells_total * 272u + 255u) / 256u), 256u, 0u, stream>>>(d_w, cells_total, 911u);
	FillFp8Kernel<<<(uint32_t)(((uint64_t)max_m * input + 255u) / 256u), 256u, 0u, stream>>>(d_c, (uint64_t)max_m * input, 919u);
	FillScaleKernel<<<(uint32_t)(((uint64_t)max_m * (input / 32u) + 255u) / 256u), 256u, 0u, stream>>>(d_cs, (uint64_t)max_m * (input / 32u), 929u, 118u);
	for ( uint32_t skewed = 0u; skewed < 2u; skewed++ )
		for ( uint32_t mi = 0u; mi < 6u; mi++ )
		{
			const uint32_t m = ms[mi], packed = m * top_k;
			std::vector<uint32_t> offset, source;
			BuildRoutes(groups, m, top_k, skewed != 0u, offset, source);
			CHECK_CUDA(cudaMemcpyAsync(d_off, offset.data(), offset.size() * 4u, cudaMemcpyHostToDevice, stream));
			CHECK_CUDA(cudaMemcpyAsync(d_src, source.data(), source.size() * 4u, cudaMemcpyHostToDevice, stream));
			for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
				CHECK_OK(LmCellMxExperts(d_w, d_c, d_cs, d_out, d_off, d_src, groups, packed, 0u, input, output, stream));
			CHECK_CUDA(cudaEventRecord(ev0, stream));
			for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
				CHECK_OK(LmCellMxExperts(d_w, d_c, d_cs, d_out, d_off, d_src, groups, packed, 0u, input, output, stream));
			CHECK_CUDA(cudaEventRecord(ev1, stream));
			CHECK_CUDA(cudaEventSynchronize(ev1));
			float total_ms = 0.0f;
			CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev0, ev1));
			const double ms_each = (double)total_ms / (double)BENCH_ITERS;
			const double tflops = (2.0 * (double)packed * (double)input * (double)output) / (ms_each * 1.0e9);
			printf("BENCH cell_mx %s G=%u M=%u routes=%u K=%u N=%u: %.3f ms (%.1f TFLOPS)\n",
				skewed != 0u ? "skewed" : "uniform", groups, m, packed, input, output, ms_each, tflops);
		}
	CHECK_CUDA(cudaFree(d_w));
	CHECK_CUDA(cudaFree(d_c));
	CHECK_CUDA(cudaFree(d_cs));
	CHECK_CUDA(cudaFree(d_off));
	CHECK_CUDA(cudaFree(d_src));
	CHECK_CUDA(cudaFree(d_out));
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
}

static void VerifyBf16GemmExact(uint32_t groups, uint32_t rows, uint32_t input, uint32_t output, cudaStream_t stream)
{
	std::vector<uint32_t> offset(groups + 1u, 0u);
	uint32_t longest = 0u;
	for ( uint32_t g = 0u; g < groups; g++ )
	{
		const uint32_t span = groups == 1u ? rows : (rows / groups + (g < (rows % groups) ? 1u : 0u));
		offset[g + 1u] = offset[g] + span;
		if ( span > longest )
			longest = span;
	}
	const uint64_t a_n = (uint64_t)rows * input, w_n = (uint64_t)groups * output * input, o_n = (uint64_t)rows * output;
	uint16_t *d_a = 0, *d_w = 0, *d_out = 0, *d_ref = 0;
	uint32_t *d_off = 0, *d_prefix = 0;
	CHECK_CUDA(cudaMalloc(&d_a, a_n * 2u));
	CHECK_CUDA(cudaMalloc(&d_w, w_n * 2u));
	CHECK_CUDA(cudaMalloc(&d_out, o_n * 2u));
	CHECK_CUDA(cudaMalloc(&d_ref, o_n * 2u));
	CHECK_CUDA(cudaMalloc(&d_off, (groups + 1u) * 4u));
	CHECK_CUDA(cudaMalloc(&d_prefix, (groups + 1u) * 4u));
	FillBf16Kernel<<<(uint32_t)((a_n + 255u) / 256u), 256u, 0u, stream>>>(d_a, a_n, 1511u + rows, 0.002f);
	FillBf16Kernel<<<(uint32_t)((w_n + 255u) / 256u), 256u, 0u, stream>>>(d_w, w_n, 1523u + output, 0.001f);
	CHECK_CUDA(cudaMemcpyAsync(d_off, offset.data(), (groups + 1u) * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemsetAsync(d_prefix, 0, (groups + 1u) * 4u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, o_n * 2u, stream));
	CHECK_CUDA(cudaMemsetAsync(d_ref, 0xff, o_n * 2u, stream));
	LmGemmArguments gemm;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.prefix_built = 0u;
	gemm.group_row_offset = d_off;
	gemm.group_tile_prefix = d_prefix;
	gemm.output_bf16 = d_out;
	gemm.output_row_stride = output;
	CHECK_OK((LmGemmLaunch<LmBf16Format, 128u, LmBf16Format::kTileK, 2u, 8u>(
		&gemm, d_a, d_w, rows, rows, 1u, groups, input, output, g_sms, groups > 1u, stream)));
	Bf16MmaRefKernel<<<dim3(output / 8u, (longest + 15u) / 16u, groups), 32u, 0u, stream>>>(d_a, d_w, d_off, input, output, d_ref);
	const uint64_t diff = DiffCountU16(d_ref, d_out, o_n, stream);
	if ( diff != 0u )
		g_failures++;
	printf("%s bf16_gemm_exact G=%u M=%u K=%u N=%u: diff=%llu\n",
		diff == 0u ? "PASS" : "FAIL", groups, rows, input, output, (unsigned long long)diff);
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_w));
	CHECK_CUDA(cudaFree(d_out));
	CHECK_CUDA(cudaFree(d_ref));
	CHECK_CUDA(cudaFree(d_off));
	CHECK_CUDA(cudaFree(d_prefix));
}

static void BenchBf16GemmShapes(cudaStream_t stream)
{
	static const uint32_t dims[] = {4096u, 5120u, 6144u};
	static const uint32_t ms[] = {64u, 256u, 512u, 1024u, 2048u};
	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	for ( uint32_t di = 0u; di < 3u; di++ )
	{
		const uint32_t h = dims[di], max_m = 2048u;
		uint16_t *d_a = 0, *d_w = 0, *d_out = 0;
		uint32_t *d_off = 0, *d_prefix = 0;
		CHECK_CUDA(cudaMalloc(&d_a, (uint64_t)max_m * h * 2u));
		CHECK_CUDA(cudaMalloc(&d_w, (uint64_t)h * h * 2u));
		CHECK_CUDA(cudaMalloc(&d_out, (uint64_t)max_m * h * 2u));
		CHECK_CUDA(cudaMalloc(&d_off, 8u));
		CHECK_CUDA(cudaMalloc(&d_prefix, 8u));
		FillBf16Kernel<<<(uint32_t)(((uint64_t)max_m * h + 255u) / 256u), 256u, 0u, stream>>>(d_a, (uint64_t)max_m * h, 1601u + h, 0.002f);
		FillBf16Kernel<<<(uint32_t)(((uint64_t)h * h + 255u) / 256u), 256u, 0u, stream>>>(d_w, (uint64_t)h * h, 1607u + h, 0.001f);
		CHECK_CUDA(cudaMemsetAsync(d_prefix, 0, 8u, stream));
		for ( uint32_t mi = 0u; mi < 5u; mi++ )
		{
			const uint32_t m = ms[mi];
			const uint32_t host_off[2] = {0u, m};
			CHECK_CUDA(cudaMemcpyAsync(d_off, host_off, 8u, cudaMemcpyHostToDevice, stream));
			LmGemmArguments gemm;
			memset(&gemm, 0, sizeof(gemm));
			gemm.scale_a = LmScaleTensorNone();
			gemm.scale_b = LmScaleTensorNone();
			gemm.prefix_built = 0u;
			gemm.group_row_offset = d_off;
			gemm.group_tile_prefix = d_prefix;
			gemm.output_bf16 = d_out;
			gemm.output_row_stride = h;
			for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
				CHECK_OK((LmGemmLaunch<LmBf16Format, 128u, LmBf16Format::kTileK, 2u, 8u>(
					&gemm, d_a, d_w, m, m, 1u, 1u, h, h, g_sms, false, stream)));
			CHECK_CUDA(cudaEventRecord(ev0, stream));
			for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
				CHECK_OK((LmGemmLaunch<LmBf16Format, 128u, LmBf16Format::kTileK, 2u, 8u>(
					&gemm, d_a, d_w, m, m, 1u, 1u, h, h, g_sms, false, stream)));
			CHECK_CUDA(cudaEventRecord(ev1, stream));
			CHECK_CUDA(cudaEventSynchronize(ev1));
			float total_ms = 0.0f;
			CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev0, ev1));
			const double ms_each = (double)total_ms / (double)BENCH_ITERS;
			const double tflops = (2.0 * (double)m * (double)h * (double)h) / (ms_each * 1.0e9);
			printf("BENCH bf16_gemm H=%u M=%u: %.3f ms (%.1f TFLOPS)\n", h, m, ms_each, tflops);
		}
		CHECK_CUDA(cudaFree(d_a));
		CHECK_CUDA(cudaFree(d_w));
		CHECK_CUDA(cudaFree(d_out));
		CHECK_CUDA(cudaFree(d_off));
		CHECK_CUDA(cudaFree(d_prefix));
	}
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
}

struct BenchExpandKv
{
	static constexpr uint32_t kSlotBytes = 576u * 2u;
	static constexpr uint32_t kPageSlots = 64u;
	static constexpr uint32_t kPageBytes = kSlotBytes * 64u;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / 64u; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % 64u; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + 63u) / 64u; }
};

static void VerifyAndBenchExpandedPrefill(cudaStream_t stream)
{
	constexpr uint32_t LATENT = 512u, ROPE = 64u, WIDTH = 576u, VALUE = 128u;
	constexpr uint32_t DEGREE = 4u, HEADS = 16u, CONTEXT = 4096u, MAX_ROWS = 1024u;
	const uint32_t pages = (CONTEXT + 63u) / 64u;
	const SparkKvShard probe = {DEGREE, 0u, 1u};
	const uint32_t most = SparkKvShardGatherKeys(probe, CONTEXT);
	const uint64_t key_stride = ((uint64_t)most * BenchExpandKv::kSlotBytes + 4095u) / 4096u * 4096u;
	const uint64_t value_stride = (uint64_t)most * HEADS * VALUE * 2u;
	const uint64_t record_floats = LM_LATENT_SHARD_RECORD_FLOATS(VALUE);
	const uint64_t max_partial_stride = (uint64_t)MAX_ROWS * HEADS * record_floats;
	const float qk_scale = 0.07216878365f;
	std::vector<uint32_t> table(pages, 0u), seqs(MAX_ROWS, 0u), ctxs(1u, CONTEXT), plan(3u, 0u);
	plan[2] = CONTEXT;
	uint8_t *d_pool = 0, *d_keys = 0, *d_values = 0;
	uint32_t *d_table = 0, *d_seqs = 0, *d_ctxs = 0, *d_pos = 0, *d_plan = 0;
	uint16_t *d_query = 0, *d_weight = 0;
	float *d_part_fused = 0, *d_part_single = 0;
	LmKvAccessError *d_err = 0, h_err;
	CHECK_CUDA(cudaMalloc(&d_pool, 4096u));
	CHECK_CUDA(cudaMalloc(&d_keys, (uint64_t)DEGREE * key_stride));
	CHECK_CUDA(cudaMalloc(&d_values, (uint64_t)DEGREE * value_stride));
	CHECK_CUDA(cudaMalloc(&d_table, pages * 4u));
	CHECK_CUDA(cudaMalloc(&d_seqs, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMalloc(&d_ctxs, 4u));
	CHECK_CUDA(cudaMalloc(&d_pos, MAX_ROWS * 4u));
	CHECK_CUDA(cudaMalloc(&d_plan, 12u));
	CHECK_CUDA(cudaMalloc(&d_query, (uint64_t)MAX_ROWS * HEADS * WIDTH * 2u));
	CHECK_CUDA(cudaMalloc(&d_weight, (uint64_t)HEADS * VALUE * LATENT * 2u));
	CHECK_CUDA(cudaMalloc(&d_part_fused, (uint64_t)DEGREE * max_partial_stride * 4u));
	CHECK_CUDA(cudaMalloc(&d_part_single, (uint64_t)DEGREE * max_partial_stride * 4u));
	CHECK_CUDA(cudaMalloc(&d_err, sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&h_err);
	CHECK_CUDA(cudaMemcpyAsync(d_table, table.data(), pages * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_seqs, seqs.data(), MAX_ROWS * 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_ctxs, ctxs.data(), 4u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_plan, plan.data(), 12u, cudaMemcpyHostToDevice, stream));
	CHECK_CUDA(cudaMemcpyAsync(d_err, &h_err, sizeof(LmKvAccessError), cudaMemcpyHostToDevice, stream));
	FillBf16Kernel<<<(uint32_t)(((uint64_t)DEGREE * (key_stride / 2u) + 255u) / 256u), 256u, 0u, stream>>>((uint16_t *)d_keys, (uint64_t)DEGREE * (key_stride / 2u), 1709u, 0.002f);
	FillBf16Kernel<<<(uint32_t)(((uint64_t)MAX_ROWS * HEADS * WIDTH + 255u) / 256u), 256u, 0u, stream>>>(d_query, (uint64_t)MAX_ROWS * HEADS * WIDTH, 1721u, 0.002f);
	FillBf16Kernel<<<(uint32_t)(((uint64_t)HEADS * VALUE * LATENT + 255u) / 256u), 256u, 0u, stream>>>(d_weight, (uint64_t)HEADS * VALUE * LATENT, 1733u, 0.0001f);

	auto make_view = [&](uint32_t rank, LmKvShardGatherView *out_view)
	{
		SparkKvShard shard = {DEGREE, rank, 1u};
		LmKvShardView pv;
		if ( LmKvShardViewInitialize<BenchExpandKv>(&pv, d_pool, d_table, pages, 1u, pages, d_err, shard) != 0 )
		{
			printf("FAIL expand shard view init\n");
			exit(1);
		}
		if ( LmKvShardGatherViewInitialize<BenchExpandKv>(out_view, pv.pages, shard, d_keys + (uint64_t)rank * key_stride, d_plan + 1u, d_plan + 2u) != 0 )
		{
			printf("FAIL expand gather view init\n");
			exit(1);
		}
	};

	CHECK_GPU(LmLatentValueExpandLaunch<LATENT, WIDTH, VALUE>(d_keys, key_stride, DEGREE, most, d_weight, HEADS, d_values, value_stride, stream));
	{
		const uint32_t eq_rows = 128u;
		const uint64_t eq_stride = (uint64_t)eq_rows * HEADS * record_floats;
		std::vector<uint32_t> pos(eq_rows);
		for ( uint32_t r = 0u; r < eq_rows; r++ )
			pos[r] = CONTEXT - eq_rows + r;
		CHECK_CUDA(cudaMemcpyAsync(d_pos, pos.data(), eq_rows * 4u, cudaMemcpyHostToDevice, stream));
		LmKvShardGatherView base_view;
		make_view(0u, &base_view);
		CHECK_GPU(LmExpandedGatherPrefillSourcesLaunch<BenchExpandKv, LATENT, ROPE, VALUE>(
			base_view, DEGREE, key_stride, d_values, value_stride, d_query, HEADS, d_seqs, d_ctxs, d_pos, qk_scale, d_part_fused, eq_stride, eq_rows, stream));
		for ( uint32_t rank = 0u; rank < DEGREE; rank++ )
		{
			LmKvShardGatherView one;
			make_view(rank, &one);
			CHECK_GPU(LmExpandedGatherPrefillLaunch<BenchExpandKv, LATENT, ROPE, VALUE>(
				one, d_values + (uint64_t)rank * value_stride, d_query, HEADS, d_seqs, d_ctxs, d_pos, qk_scale, d_part_single + (uint64_t)rank * eq_stride, eq_rows, stream));
		}
		const uint64_t diff = DiffCountF32(d_part_fused, d_part_single, (uint64_t)DEGREE * eq_stride, stream);
		CHECK_CUDA(cudaMemcpy(&h_err, d_err, sizeof(LmKvAccessError), cudaMemcpyDeviceToHost));
		const int ok = diff == 0u && h_err.error_code == LM_FRAME_ERROR_NONE;
		if ( !ok )
			g_failures++;
		printf("%s expanded_prefill_exact rows=%u ctx=%u degree=%u heads=%u: diff=%llu err=%u\n",
			ok ? "PASS" : "FAIL", eq_rows, CONTEXT, DEGREE, HEADS, (unsigned long long)diff, h_err.error_code);
	}

	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
		CHECK_GPU(LmLatentValueExpandLaunch<LATENT, WIDTH, VALUE>(d_keys, key_stride, DEGREE, most, d_weight, HEADS, d_values, value_stride, stream));
	CHECK_CUDA(cudaEventRecord(ev0, stream));
	for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
		CHECK_GPU(LmLatentValueExpandLaunch<LATENT, WIDTH, VALUE>(d_keys, key_stride, DEGREE, most, d_weight, HEADS, d_values, value_stride, stream));
	CHECK_CUDA(cudaEventRecord(ev1, stream));
	CHECK_CUDA(cudaEventSynchronize(ev1));
	float expand_ms = 0.0f;
	CHECK_CUDA(cudaEventElapsedTime(&expand_ms, ev0, ev1));
	printf("BENCH attn_value_expand ctx=%u degree=%u heads=%u: %.3f ms\n", CONTEXT, DEGREE, HEADS, (double)expand_ms / (double)BENCH_ITERS);

	static const uint32_t row_cases[] = {256u, 1024u};
	LmKvShardGatherView base_view;
	make_view(0u, &base_view);
	for ( uint32_t ri = 0u; ri < 2u; ri++ )
	{
		const uint32_t rows = row_cases[ri];
		const uint64_t p_stride = (uint64_t)rows * HEADS * record_floats;
		std::vector<uint32_t> pos(rows);
		for ( uint32_t r = 0u; r < rows; r++ )
			pos[r] = CONTEXT - rows + r;
		CHECK_CUDA(cudaMemcpyAsync(d_pos, pos.data(), rows * 4u, cudaMemcpyHostToDevice, stream));
		for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
			CHECK_GPU(LmExpandedGatherPrefillSourcesLaunch<BenchExpandKv, LATENT, ROPE, VALUE>(
				base_view, DEGREE, key_stride, d_values, value_stride, d_query, HEADS, d_seqs, d_ctxs, d_pos, qk_scale, d_part_fused, p_stride, rows, stream));
		CHECK_CUDA(cudaEventRecord(ev0, stream));
		for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
			CHECK_GPU(LmExpandedGatherPrefillSourcesLaunch<BenchExpandKv, LATENT, ROPE, VALUE>(
				base_view, DEGREE, key_stride, d_values, value_stride, d_query, HEADS, d_seqs, d_ctxs, d_pos, qk_scale, d_part_fused, p_stride, rows, stream));
		CHECK_CUDA(cudaEventRecord(ev1, stream));
		CHECK_CUDA(cudaEventSynchronize(ev1));
		float attn_ms = 0.0f;
		CHECK_CUDA(cudaEventElapsedTime(&attn_ms, ev0, ev1));
		printf("BENCH attn_expanded_prefill rows=%u ctx=%u degree=%u heads=%u: %.3f ms\n",
			rows, CONTEXT, DEGREE, HEADS, (double)attn_ms / (double)BENCH_ITERS);
	}
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
	CHECK_CUDA(cudaFree(d_pool));
	CHECK_CUDA(cudaFree(d_keys));
	CHECK_CUDA(cudaFree(d_values));
	CHECK_CUDA(cudaFree(d_table));
	CHECK_CUDA(cudaFree(d_seqs));
	CHECK_CUDA(cudaFree(d_ctxs));
	CHECK_CUDA(cudaFree(d_pos));
	CHECK_CUDA(cudaFree(d_plan));
	CHECK_CUDA(cudaFree(d_query));
	CHECK_CUDA(cudaFree(d_weight));
	CHECK_CUDA(cudaFree(d_part_fused));
	CHECK_CUDA(cudaFree(d_part_single));
	CHECK_CUDA(cudaFree(d_err));
}

int main(void)
{
	cudaDeviceProp prop;
	cudaStream_t stream = 0;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
	g_sms = (uint32_t)prop.multiProcessorCount;
	CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	VerifyCellMxExact(64u, 64u, 8u, 512u, 512u, false, stream);
	VerifyCellMxExact(64u, 128u, 8u, 512u, 512u, false, stream);
	VerifyCellMxExact(32u, 160u, 8u, 1024u, 512u, true, stream);
	VerifyCellMxExact(32u, 256u, 8u, 1024u, 512u, false, stream);
	VerifyCellMxExact(16u, 448u, 8u, 1024u, 512u, true, stream);

	VerifyBf16GemmExact(1u, 64u, 1024u, 1024u, stream);
	VerifyBf16GemmExact(1u, 192u, 768u, 1536u, stream);
	VerifyBf16GemmExact(1u, 512u, 1024u, 1024u, stream);
	VerifyBf16GemmExact(8u, 320u, 1024u, 1024u, stream);

	BenchCellMxShapes(stream);
	BenchBf16GemmShapes(stream);
	VerifyAndBenchExpandedPrefill(stream);

	CHECK_CUDA(cudaStreamDestroy(stream));
	printf("%s shared_prefill_gemm_bench failures=%d sms=%u\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_sms);
	return g_failures != 0;
}
