#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/kernels/cell_mx.cuh"
#include "inference/kernels/mx_gemm.cuh"

#define BENCH_WARMUP 5u
#define BENCH_ITERS 20u

#define CHECK_CUDA(call) do { cudaError_t err_ = (call); if ( err_ != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(err_)); exit(1); } } while (0)
#define CHECK_OK(call) do { int32_t st_ = (call); if ( st_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #call, st_); exit(1); } } while (0)

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

static __global__ void FillBytesKernel(uint8_t *dst, uint64_t count, uint64_t seed)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
		dst[idx] = (uint8_t)Hash64(idx * 5u + seed);
}

static __global__ void FillBf16Kernel(uint16_t *dst, uint64_t count, uint64_t seed)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count )
	{
		const float v = (float)((int64_t)(Hash64(idx + seed) % 2001u) - 1000) * 0.0025f;
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

static __global__ void CountDiffKernel(const uint16_t *a, const uint16_t *b, uint64_t count, uint64_t *diffs)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < count && a[idx] != b[idx] )
		atomicAdd((unsigned long long *)diffs, 1ull);
}

static uint64_t DiffCount(const uint16_t *a, const uint16_t *b, uint64_t count, cudaStream_t stream)
{
	uint64_t *d_diffs = 0, host_diffs = 0u;
	CHECK_CUDA(cudaMalloc(&d_diffs, sizeof(uint64_t)));
	CHECK_CUDA(cudaMemsetAsync(d_diffs, 0, sizeof(uint64_t), stream));
	const uint32_t grid = (uint32_t)((count + 255u) / 256u);
	CountDiffKernel<<<grid, 256u, 0u, stream>>>(a, b, count, d_diffs);
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

static void VerifyMxGemmExact(uint32_t rows, uint32_t input, uint32_t output, cudaStream_t stream)
{
	const uint64_t a_n = (uint64_t)rows * input, w_n = (uint64_t)output * input, o_n = (uint64_t)rows * output;
	uint8_t *d_a = 0, *d_as = 0, *d_w = 0, *d_ws = 0;
	uint16_t *d_ref = 0, *d_out = 0;
	CHECK_CUDA(cudaMalloc(&d_a, a_n));
	CHECK_CUDA(cudaMalloc(&d_as, a_n / 32u));
	CHECK_CUDA(cudaMalloc(&d_w, w_n));
	CHECK_CUDA(cudaMalloc(&d_ws, w_n / 32u));
	CHECK_CUDA(cudaMalloc(&d_ref, o_n * 2u));
	CHECK_CUDA(cudaMalloc(&d_out, o_n * 2u));
	FillFp8Kernel<<<(uint32_t)((a_n + 255u) / 256u), 256u, 0u, stream>>>(d_a, a_n, 11u + rows);
	FillScaleKernel<<<(uint32_t)((a_n / 32u + 255u) / 256u), 256u, 0u, stream>>>(d_as, a_n / 32u, 23u + rows, 118u);
	FillFp8Kernel<<<(uint32_t)((w_n + 255u) / 256u), 256u, 0u, stream>>>(d_w, w_n, 37u + output);
	FillScaleKernel<<<(uint32_t)((w_n / 32u + 255u) / 256u), 256u, 0u, stream>>>(d_ws, w_n / 32u, 53u + output, 116u);
	LmMxGemmArguments args;
	memset(&args, 0, sizeof(args));
	args.activation = d_a;
	args.activation_scale = d_as;
	args.weight = d_w;
	args.weight_scale = d_ws;
	args.output_bf16 = d_ref;
	args.rows = rows;
	args.input_dimension = input;
	args.output_dimension = output;
	args.output_row_stride = output;
	CHECK_OK((LmMxGemmLaunchTile<128u, 128u, 2u>(&args, g_sms, stream)));
	uint64_t diff_disp = 0u, diff_64x128 = 0u, diff_64x64 = 0u, diff_32x128 = 0u;
	CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, o_n * 2u, stream));
	CHECK_OK(LmMxGemmDense(d_a, d_as, d_w, d_ws, d_out, rows, input, output, 0u, 0u, g_sms, stream));
	diff_disp = DiffCount(d_ref, d_out, o_n, stream);
	args.output_bf16 = d_out;
	CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, o_n * 2u, stream));
	CHECK_OK((LmMxGemmLaunchTile<64u, 128u, 2u, 4u, 2u>(&args, g_sms, stream)));
	diff_64x128 = DiffCount(d_ref, d_out, o_n, stream);
	CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, o_n * 2u, stream));
	CHECK_OK((LmMxGemmLaunchTile<64u, 64u, 2u, 4u, 2u>(&args, g_sms, stream)));
	diff_64x64 = DiffCount(d_ref, d_out, o_n, stream);
	CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, o_n * 2u, stream));
	CHECK_OK((LmMxGemmLaunchTile<32u, 128u, 2u>(&args, g_sms, stream)));
	diff_32x128 = DiffCount(d_ref, d_out, o_n, stream);
	const uint64_t total = diff_disp + diff_64x128 + diff_64x64 + diff_32x128;
	if ( total != 0u )
		g_failures++;
	printf("%s mx_gemm_exact M=%u K=%u N=%u: diff disp=%llu 64x128=%llu 64x64=%llu 32x128=%llu\n",
		total == 0u ? "PASS" : "FAIL", rows, input, output,
		(unsigned long long)diff_disp, (unsigned long long)diff_64x128, (unsigned long long)diff_64x64, (unsigned long long)diff_32x128);
	CHECK_CUDA(cudaFree(d_a));
	CHECK_CUDA(cudaFree(d_as));
	CHECK_CUDA(cudaFree(d_w));
	CHECK_CUDA(cudaFree(d_ws));
	CHECK_CUDA(cudaFree(d_ref));
	CHECK_CUDA(cudaFree(d_out));
}

static void BenchMxGemmShapes(cudaStream_t stream)
{
	static const uint32_t dims[] = {7168u, 6144u, 5120u, 4096u};
	static const uint32_t ms[] = {64u, 256u, 512u, 1024u, 2048u};
	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	for ( uint32_t di = 0u; di < 4u; di++ )
	{
		const uint32_t h = dims[di];
		const uint64_t max_m = 2048u, a_n = max_m * h, w_n = (uint64_t)h * h, o_n = max_m * h;
		uint8_t *d_a = 0, *d_as = 0, *d_w = 0, *d_ws = 0;
		uint16_t *d_out = 0;
		CHECK_CUDA(cudaMalloc(&d_a, a_n));
		CHECK_CUDA(cudaMalloc(&d_as, a_n / 32u));
		CHECK_CUDA(cudaMalloc(&d_w, w_n));
		CHECK_CUDA(cudaMalloc(&d_ws, w_n / 32u));
		CHECK_CUDA(cudaMalloc(&d_out, o_n * 2u));
		FillFp8Kernel<<<(uint32_t)((a_n + 255u) / 256u), 256u, 0u, stream>>>(d_a, a_n, 101u + h);
		FillScaleKernel<<<(uint32_t)((a_n / 32u + 255u) / 256u), 256u, 0u, stream>>>(d_as, a_n / 32u, 202u + h, 120u);
		FillFp8Kernel<<<(uint32_t)((w_n + 255u) / 256u), 256u, 0u, stream>>>(d_w, w_n, 303u + h);
		FillScaleKernel<<<(uint32_t)((w_n / 32u + 255u) / 256u), 256u, 0u, stream>>>(d_ws, w_n / 32u, 404u + h, 118u);
		for ( uint32_t mi = 0u; mi < 5u; mi++ )
		{
			const uint32_t m = ms[mi];
			for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
				CHECK_OK(LmMxGemmDense(d_a, d_as, d_w, d_ws, d_out, m, h, h, 0u, 0u, g_sms, stream));
			CHECK_CUDA(cudaEventRecord(ev0, stream));
			for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
				CHECK_OK(LmMxGemmDense(d_a, d_as, d_w, d_ws, d_out, m, h, h, 0u, 0u, g_sms, stream));
			CHECK_CUDA(cudaEventRecord(ev1, stream));
			CHECK_CUDA(cudaEventSynchronize(ev1));
			float total_ms = 0.0f;
			CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev0, ev1));
			const double ms_each = (double)total_ms / (double)BENCH_ITERS;
			const double tflops = (2.0 * (double)m * (double)h * (double)h) / (ms_each * 1.0e9);
			printf("BENCH mx_gemm H=%u M=%u: %.3f ms (%.1f TFLOPS)\n", h, m, ms_each, tflops);
		}
		CHECK_CUDA(cudaFree(d_a));
		CHECK_CUDA(cudaFree(d_as));
		CHECK_CUDA(cudaFree(d_w));
		CHECK_CUDA(cudaFree(d_ws));
		CHECK_CUDA(cudaFree(d_out));
	}
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
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
	const uint64_t diff = DiffCount(d_ref, d_mx, (uint64_t)packed * output, stream);
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
	static const uint32_t ms[] = {64u, 256u, 512u, 1024u, 2048u};
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
		for ( uint32_t mi = 0u; mi < 5u; mi++ )
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

template<class Format>
static void VerifyAndBenchCellMma(const char *name, cudaStream_t stream)
{
	constexpr uint32_t BITS = LmCellMmaFormat<Format>::kBits;
	constexpr uint32_t SCALES = LmCellMmaFormat<Format>::kScales;
	static const uint32_t exact_tokens[] = {32u, 80u, 160u};
	const uint32_t eq_groups = 16u, top_k = 8u, eq_k = 1024u, eq_n = 512u;
	const uint64_t eq_w_bytes = (uint64_t)eq_groups * eq_n * eq_k * BITS / 8u;
	const uint64_t eq_s_bytes = (uint64_t)eq_groups * eq_n * (eq_k / Format::kScaleGroup);
	uint8_t *d_w = 0, *d_s = 0;
	CHECK_CUDA(cudaMalloc(&d_w, eq_w_bytes));
	CHECK_CUDA(cudaMalloc(&d_s, eq_s_bytes));
	if constexpr ( BITS == 8u )
		FillFp8Kernel<<<(uint32_t)((eq_w_bytes + 255u) / 256u), 256u, 0u, stream>>>(d_w, eq_w_bytes, 1103u);
	else
		FillBytesKernel<<<(uint32_t)((eq_w_bytes + 255u) / 256u), 256u, 0u, stream>>>(d_w, eq_w_bytes, 1103u);
	FillScaleKernel<<<(uint32_t)((eq_s_bytes + 255u) / 256u), 256u, 0u, stream>>>(d_s, eq_s_bytes, 1117u, 118u);
	const LmScaleTensor scale = LmScaleTensorBuild(d_s, LM_SCALE_ENCODING_UE8M0, eq_groups, eq_n, eq_k, 1u, Format::kScaleGroup);
	for ( uint32_t ti = 0u; ti < 3u; ti++ )
	{
		const uint32_t tokens = exact_tokens[ti], packed = tokens * top_k;
		std::vector<uint32_t> offset, source;
		BuildRoutes(eq_groups, tokens, top_k, true, offset, source);
		uint32_t *d_off = 0, *d_src = 0;
		uint16_t *d_act = 0, *d_out = 0, *d_ref = 0;
		CHECK_CUDA(cudaMalloc(&d_off, offset.size() * 4u));
		CHECK_CUDA(cudaMalloc(&d_src, source.size() * 4u));
		CHECK_CUDA(cudaMalloc(&d_act, (uint64_t)tokens * eq_k * 2u));
		CHECK_CUDA(cudaMalloc(&d_out, (uint64_t)packed * eq_n * 2u));
		CHECK_CUDA(cudaMalloc(&d_ref, (uint64_t)packed * eq_n * 2u));
		CHECK_CUDA(cudaMemcpyAsync(d_off, offset.data(), offset.size() * 4u, cudaMemcpyHostToDevice, stream));
		CHECK_CUDA(cudaMemcpyAsync(d_src, source.data(), source.size() * 4u, cudaMemcpyHostToDevice, stream));
		FillBf16Kernel<<<(uint32_t)(((uint64_t)tokens * eq_k + 255u) / 256u), 256u, 0u, stream>>>(d_act, (uint64_t)tokens * eq_k, 1201u + tokens);
		CHECK_CUDA(cudaMemsetAsync(d_out, 0xff, (uint64_t)packed * eq_n * 2u, stream));
		CHECK_CUDA(cudaMemsetAsync(d_ref, 0xff, (uint64_t)packed * eq_n * 2u, stream));
		CHECK_OK(LmCellMmaStridedExperts<Format>(d_w, scale, d_act, d_out, d_off, d_src, eq_groups, packed, 0u, eq_k, eq_n, stream));
		LmSkinnyCellArguments args;
		LmCellMmaSegments segments;
		LmCellMmaStride stride;
		const void *wptr = d_w;
		CHECK_OK(LmCellMmaLaunchSegments(args, segments, &wptr, 1u, d_act, d_ref, d_off, d_src, eq_groups, packed, 0u, eq_k, eq_n, LM_SKINNY_CELL_TILE_K));
		stride.scale = (const uint8_t *)scale.data;
		stride.global = (const float *)scale.global_data;
		stride.payload_pitch = (uint64_t)eq_k * BITS / 8u;
		stride.scale_group_entries = scale.group_stride_entries;
		stride.scale_row_entries = scale.row_group_stride_entries;
		const uint32_t col_blocks = (args.cells / LM_CELL_MMA_CELLS + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS;
		LmCellMmaKernel<SCALES, BITS, LM_CELL_MMA_CHUNKS><<<eq_groups * col_blocks, LM_CELL_MMA_THREADS, 0u, stream>>>(args, segments, stride);
		const uint64_t diff = DiffCount(d_ref, d_out, (uint64_t)packed * eq_n, stream);
		if ( diff != 0u )
			g_failures++;
		printf("%s cell_mma_exact %s G=%u M=%u mean=%u: diff=%llu\n",
			diff == 0u ? "PASS" : "FAIL", name, eq_groups, tokens, packed / eq_groups, (unsigned long long)diff);
		CHECK_CUDA(cudaFree(d_off));
		CHECK_CUDA(cudaFree(d_src));
		CHECK_CUDA(cudaFree(d_act));
		CHECK_CUDA(cudaFree(d_out));
		CHECK_CUDA(cudaFree(d_ref));
	}
	CHECK_CUDA(cudaFree(d_w));
	CHECK_CUDA(cudaFree(d_s));

	static const uint32_t bench_ms[] = {64u, 256u, 512u, 1024u};
	const uint32_t bg = 64u, bk = 4096u, bn = 2048u, max_m = 1024u, max_packed = max_m * top_k;
	const uint64_t bw_bytes = (uint64_t)bg * bn * bk * BITS / 8u;
	const uint64_t bs_bytes = (uint64_t)bg * bn * (bk / Format::kScaleGroup);
	uint8_t *db_w = 0, *db_s = 0;
	uint32_t *db_off = 0, *db_src = 0;
	uint16_t *db_act = 0, *db_out = 0;
	cudaEvent_t ev0, ev1;
	CHECK_CUDA(cudaEventCreate(&ev0));
	CHECK_CUDA(cudaEventCreate(&ev1));
	CHECK_CUDA(cudaMalloc(&db_w, bw_bytes));
	CHECK_CUDA(cudaMalloc(&db_s, bs_bytes));
	CHECK_CUDA(cudaMalloc(&db_off, (bg + 1u) * 4u));
	CHECK_CUDA(cudaMalloc(&db_src, (uint64_t)max_packed * 4u));
	CHECK_CUDA(cudaMalloc(&db_act, (uint64_t)max_m * bk * 2u));
	CHECK_CUDA(cudaMalloc(&db_out, (uint64_t)max_packed * bn * 2u));
	if constexpr ( BITS == 8u )
		FillFp8Kernel<<<(uint32_t)((bw_bytes + 255u) / 256u), 256u, 0u, stream>>>(db_w, bw_bytes, 1409u);
	else
		FillBytesKernel<<<(uint32_t)((bw_bytes + 255u) / 256u), 256u, 0u, stream>>>(db_w, bw_bytes, 1409u);
	FillScaleKernel<<<(uint32_t)((bs_bytes + 255u) / 256u), 256u, 0u, stream>>>(db_s, bs_bytes, 1423u, 118u);
	FillBf16Kernel<<<(uint32_t)(((uint64_t)max_m * bk + 255u) / 256u), 256u, 0u, stream>>>(db_act, (uint64_t)max_m * bk, 1433u);
	const LmScaleTensor bscale = LmScaleTensorBuild(db_s, LM_SCALE_ENCODING_UE8M0, bg, bn, bk, 1u, Format::kScaleGroup);
	for ( uint32_t mi = 0u; mi < 4u; mi++ )
	{
		const uint32_t m = bench_ms[mi], packed = m * top_k;
		std::vector<uint32_t> offset, source;
		BuildRoutes(bg, m, top_k, false, offset, source);
		CHECK_CUDA(cudaMemcpyAsync(db_off, offset.data(), offset.size() * 4u, cudaMemcpyHostToDevice, stream));
		CHECK_CUDA(cudaMemcpyAsync(db_src, source.data(), source.size() * 4u, cudaMemcpyHostToDevice, stream));
		for ( uint32_t w = 0u; w < BENCH_WARMUP; w++ )
			CHECK_OK(LmCellMmaStridedExperts<Format>(db_w, bscale, db_act, db_out, db_off, db_src, bg, packed, 0u, bk, bn, stream));
		CHECK_CUDA(cudaEventRecord(ev0, stream));
		for ( uint32_t it = 0u; it < BENCH_ITERS; it++ )
			CHECK_OK(LmCellMmaStridedExperts<Format>(db_w, bscale, db_act, db_out, db_off, db_src, bg, packed, 0u, bk, bn, stream));
		CHECK_CUDA(cudaEventRecord(ev1, stream));
		CHECK_CUDA(cudaEventSynchronize(ev1));
		float total_ms = 0.0f;
		CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev0, ev1));
		const double ms_each = (double)total_ms / (double)BENCH_ITERS;
		const double tflops = (2.0 * (double)packed * (double)bk * (double)bn) / (ms_each * 1.0e9);
		printf("BENCH cell_mma %s G=%u M=%u routes=%u K=%u N=%u: %.3f ms (%.1f TFLOPS)\n",
			name, bg, m, packed, bk, bn, ms_each, tflops);
	}
	CHECK_CUDA(cudaFree(db_w));
	CHECK_CUDA(cudaFree(db_s));
	CHECK_CUDA(cudaFree(db_off));
	CHECK_CUDA(cudaFree(db_src));
	CHECK_CUDA(cudaFree(db_act));
	CHECK_CUDA(cudaFree(db_out));
	CHECK_CUDA(cudaEventDestroy(ev0));
	CHECK_CUDA(cudaEventDestroy(ev1));
}

int main(void)
{
	cudaDeviceProp prop;
	cudaStream_t stream = 0;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
	g_sms = (uint32_t)prop.multiProcessorCount;
	CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	VerifyMxGemmExact(64u, 1024u, 1024u, stream);
	VerifyMxGemmExact(192u, 768u, 1536u, stream);
	VerifyMxGemmExact(256u, 1024u, 1024u, stream);
	VerifyMxGemmExact(512u, 1024u, 1024u, stream);

	VerifyCellMxExact(64u, 64u, 8u, 512u, 512u, false, stream);
	VerifyCellMxExact(64u, 160u, 8u, 512u, 512u, false, stream);
	VerifyCellMxExact(32u, 160u, 8u, 1024u, 512u, true, stream);
	VerifyCellMxExact(32u, 256u, 8u, 1024u, 512u, false, stream);

	VerifyAndBenchCellMma<LmMxfp4>("mxfp4", stream);
	VerifyAndBenchCellMma<LmMxfp8>("mxfp8", stream);
	BenchMxGemmShapes(stream);
	BenchCellMxShapes(stream);

	CHECK_CUDA(cudaStreamDestroy(stream));
	printf("%s shared_prefill_gemm_bench failures=%d sms=%u\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_sms);
	return g_failures != 0;
}
