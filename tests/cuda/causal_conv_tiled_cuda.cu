#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "inference/kernels/linear_attn.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)
#define SERIAL_THREADS 256u

static uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static float Uniform(uint64_t seed, uint64_t index)
{
	return (float)(Mix(seed * 0x100000001ull + index) & 0xffffffu) / 8388608.0f - 1.0f;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

template<class T>
static T *Upload(const std::vector<T> &host)
{
	T *device = 0;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template<class T>
static std::vector<T> Download(const T *device, uint64_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

static uint32_t failures;

template<class W> static W WeightValue(uint64_t index);
template<> float WeightValue<float>(uint64_t index) { return 0.5f * Uniform(7u, index); }
template<> uint16_t WeightValue<uint16_t>(uint64_t index) { return Bf16(0.5f * Uniform(7u, index)); }

template<uint32_t KERNEL, class W, uint32_t DILATION>
static void Case(const char *label, const std::vector<uint32_t> &lengths, uint32_t channels, uint32_t commit, bool counted, bool indexed, bool in_place,
	bool timed)
{
	constexpr uint32_t SPAN = (KERNEL - 1u) * DILATION + 1u;
	const uint32_t sequences = (uint32_t)lengths.size();
	std::vector<uint32_t> begin(sequences + 1u, 0u), count(sequences), state(sequences);
	for ( uint32_t s = 0u; s < sequences; s++ )
	{
		begin[s + 1u] = begin[s] + lengths[s] + (counted ? 2u : 0u);
		count[s] = lengths[s];
		state[s] = (s * 5u + 1u) % (sequences + 3u);
	}
	const uint32_t rows = begin[sequences] > 0u ? begin[sequences] : 1u;
	std::vector<uint32_t> indices(rows);
	for ( uint32_t r = 0u; r < rows; r++ )
		indices[r] = indexed ? (uint32_t)((r * 7919ull + 13u) % rows) : r;
	if ( indexed )
	{
		std::vector<uint8_t> used(rows, 0u);
		for ( uint32_t r = 0u; r < rows; r++ )
		{
			while ( used[indices[r]] != 0u )
				indices[r] = (indices[r] + 1u) % rows;
			used[indices[r]] = 1u;
		}
	}
	std::vector<uint16_t> input((uint64_t)rows * channels), window((uint64_t)(sequences + 3u) * channels * SPAN);
	std::vector<W> weight((uint64_t)channels * KERNEL);
	for ( uint64_t i = 0u; i < input.size(); i++ )
		input[i] = Bf16(2.0f * Uniform(1u, i));
	for ( uint64_t i = 0u; i < window.size(); i++ )
		window[i] = Bf16(2.0f * Uniform(2u, i));
	for ( uint64_t i = 0u; i < weight.size(); i++ )
		weight[i] = WeightValue<W>(i);
	uint32_t *d_begin = Upload(begin), *d_count = Upload(count), *d_state = Upload(state), *d_indices = Upload(indices);
	W *d_weight = Upload(weight);
	uint16_t *in_a = Upload(input), *in_b = Upload(input), *win_a = Upload(window), *win_b = Upload(window);
	uint16_t *out_a = in_place ? in_a : Upload(input), *out_b = in_place ? in_b : Upload(input);
	const uint32_t *row_count = counted ? d_count : 0, *row_indices = indexed ? d_indices : 0;
	LmCausalConvKernel<SERIAL_THREADS,KERNEL,LM_CONV_SWISH,W,DILATION><<<dim3(sequences, (channels + SERIAL_THREADS - 1u) / SERIAL_THREADS), SERIAL_THREADS>>>(
		win_a, d_state, d_begin, row_count, in_a, d_weight, out_a, channels, sequences, commit, row_indices);
	CUDA((LmCausalConvLaunch<KERNEL,LM_CONV_SWISH,W,DILATION>(win_b, d_state, d_begin, row_count, in_b, d_weight, out_b, channels, sequences, commit,
		row_indices, rows, 0)));
	CUDA(cudaDeviceSynchronize());
	const bool outputs = Download(out_a, input.size()) == Download(out_b, input.size()), windows = Download(win_a, window.size()) == Download(win_b, window.size());
	printf("%-34s sequences %3u rows %5u channels %4u: outputs %s, windows %s", label, sequences, rows, channels, outputs ? "equal" : "DIFFER",
		windows ? "equal" : "DIFFER");
	failures += !outputs || !windows;
	if ( timed )
	{
		cudaEvent_t start, stop;
		float serial_ms = 0.0f, tiled_ms = 0.0f;
		CUDA(cudaEventCreate(&start));
		CUDA(cudaEventCreate(&stop));
		CUDA(cudaEventRecord(start));
		for ( uint32_t i = 0u; i < 10u; i++ )
			LmCausalConvKernel<SERIAL_THREADS,KERNEL,LM_CONV_SWISH,W,DILATION><<<dim3(sequences, (channels + SERIAL_THREADS - 1u) / SERIAL_THREADS), SERIAL_THREADS>>>(
				win_a, d_state, d_begin, row_count, in_a, d_weight, out_a, channels, sequences, 0u, row_indices);
		CUDA(cudaEventRecord(stop));
		CUDA(cudaEventSynchronize(stop));
		CUDA(cudaEventElapsedTime(&serial_ms, start, stop));
		CUDA(cudaEventRecord(start));
		for ( uint32_t i = 0u; i < 10u; i++ )
			CUDA((LmCausalConvLaunch<KERNEL,LM_CONV_SWISH,W,DILATION>(win_b, d_state, d_begin, row_count, in_b, d_weight, out_b, channels, sequences, 0u,
				row_indices, rows, 0)));
		CUDA(cudaEventRecord(stop));
		CUDA(cudaEventSynchronize(stop));
		CUDA(cudaEventElapsedTime(&tiled_ms, start, stop));
		printf("; serial %.1f us, tiled %.1f us", 100.0f * serial_ms, 100.0f * tiled_ms);
	}
	printf("\n");
	cudaFree(d_begin); cudaFree(d_count); cudaFree(d_state); cudaFree(d_indices); cudaFree(d_weight);
	cudaFree(in_a); cudaFree(in_b); cudaFree(win_a); cudaFree(win_b);
	if ( !in_place )
	{
		cudaFree(out_a);
		cudaFree(out_b);
	}
}

static void Streams(void)
{
	const uint32_t sequences = 3u, channels[LM_CAUSAL_CONV_STREAMS] = {512u, 512u, 1024u};
	std::vector<uint32_t> begin = {0u, 700u, 701u, 2049u}, state = {2u, 0u, 1u};
	uint32_t *d_begin = Upload(begin), *d_state = Upload(state);
	LmCausalConvStreams<uint16_t> serial, tiled;
	for ( uint32_t stream = 0u; stream < LM_CAUSAL_CONV_STREAMS; stream++ )
	{
		std::vector<uint16_t> input((uint64_t)2049u * channels[stream]), window((uint64_t)3u * channels[stream] * 4u), weight((uint64_t)channels[stream] * 4u);
		for ( uint64_t i = 0u; i < input.size(); i++ )
			input[i] = Bf16(Uniform(10u + stream, i));
		for ( uint64_t i = 0u; i < window.size(); i++ )
			window[i] = Bf16(Uniform(20u + stream, i));
		for ( uint64_t i = 0u; i < weight.size(); i++ )
			weight[i] = Bf16(0.5f * Uniform(30u + stream, i));
		serial.input_bf16[stream] = serial.output_bf16[stream] = Upload(input);
		tiled.input_bf16[stream] = tiled.output_bf16[stream] = Upload(input);
		serial.window[stream] = Upload(window);
		tiled.window[stream] = Upload(window);
		serial.weight[stream] = tiled.weight[stream] = Upload(weight);
		serial.channels[stream] = tiled.channels[stream] = channels[stream];
	}
	LmCausalConvStreamsKernel<SERIAL_THREADS,4u,LM_CONV_SWISH,uint16_t><<<dim3(sequences, (1024u + SERIAL_THREADS - 1u) / SERIAL_THREADS, LM_CAUSAL_CONV_STREAMS),
		SERIAL_THREADS>>>(serial, d_state, d_begin, (const uint32_t *)0, sequences, 1u, (const uint32_t *)0);
	CUDA((LmCausalConvStreamsLaunch<4u,LM_CONV_SWISH,uint16_t>(tiled, d_state, d_begin, (const uint32_t *)0, sequences, 1u, (const uint32_t *)0, 2049u, 0)));
	CUDA(cudaDeviceSynchronize());
	uint32_t differ = 0u;
	for ( uint32_t stream = 0u; stream < LM_CAUSAL_CONV_STREAMS; stream++ )
		differ += Download(serial.output_bf16[stream], (uint64_t)2049u * channels[stream]) != Download(tiled.output_bf16[stream], (uint64_t)2049u * channels[stream]) ||
			Download(serial.window[stream], (uint64_t)3u * channels[stream] * 4u) != Download(tiled.window[stream], (uint64_t)3u * channels[stream] * 4u);
	printf("%-34s three streams of 512, 512 and 1024 channels: %u streams differ\n", "streams, in place, commit", differ);
	failures += differ != 0u;
}

int main(void)
{
	Case<4u,uint16_t,1u>("bf16 weights, in place, commit", {2048u}, 1536u, 1u, false, false, true, true);
	Case<4u,float,1u>("f32 weights, in place, commit", {1024u}, 512u, 1u, false, false, true, true);
	Case<4u,uint16_t,1u>("short sequences", {0u, 1u, 2u, 3u, 5u, 63u, 64u, 65u, 200u}, 1000u, 1u, false, false, true, false);
	Case<4u,uint16_t,1u>("no commit", {3u, 130u}, 100u, 0u, false, false, true, false);
	Case<4u,float,1u>("row counts and row indices", {1u, 2u, 67u, 300u}, 512u, 1u, true, true, false, false);
	Case<4u,uint16_t,1u>("row indices, in place", {5u, 129u, 1u}, 256u, 1u, false, true, true, false);
	Case<1u,uint16_t,1u>("one tap", {1u, 70u, 3u}, 4096u, 1u, false, false, false, false);
	Case<3u,uint16_t,2u>("dilation 2", {0u, 4u, 65u, 300u}, 96u, 1u, false, false, true, false);
	Case<4u,uint16_t,1u>("decode rows", {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}, 1536u, 1u, false, false, true, true);
	Streams();
	if ( failures != 0u )
	{
		printf("FAIL causal conv tiled: %u cases differ from the serial kernel\n", failures);
		return 1;
	}
	printf("PASS causal conv tiled: outputs and committed windows equal the serial kernel bit for bit\n");
	return 0;
}
