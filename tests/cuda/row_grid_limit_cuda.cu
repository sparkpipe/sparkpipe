#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"

#define GRID_ROWS 70000u
#define GRID_THREADS 128u
#define GRID_DIM 160u
#define GRID_STRIDE 200u
#define GRID_TOP_K 3u
#define GRID_RANKS 4u
#define GRID_WIDTH 40u

static int grid_failures;

#define GRID_CUDA(call) do { cudaError_t grid_error = (call); if ( grid_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(grid_error)); exit(1); } } while (0)

static void GridCheck(int condition, const char *what)
{
	printf("%s %s\n",condition ? "PASS" : "FAIL",what);
	if ( !condition )
		grid_failures++;
}

static float GridBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out,&bits,sizeof(out));
	return(out);
}

static uint16_t GridValue(uint64_t index, uint64_t salt)
{
	uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z ^= z >> 31u;
	return((uint16_t)(0x3c00u + (z & 0x7fu) + ((z >> 20u) & 1u) * 0x8000u));
}

static std::vector<uint16_t> GridFill(uint64_t count, uint64_t salt)
{
	std::vector<uint16_t> values(count);
	for (uint64_t index = 0u; index < count; index++)
		values[index] = GridValue(index,salt);
	return(values);
}

template<class T>
static T *GridUpload(const std::vector<T> &host)
{
	T *device;
	GRID_CUDA(cudaMalloc(&device,host.size() * sizeof(T)));
	GRID_CUDA(cudaMemcpy(device,host.data(),host.size() * sizeof(T),cudaMemcpyHostToDevice));
	return(device);
}

template<class T>
static std::vector<T> GridDownload(const T *device, uint64_t count)
{
	std::vector<T> host(count);
	GRID_CUDA(cudaDeviceSynchronize());
	GRID_CUDA(cudaMemcpy(host.data(),device,count * sizeof(T),cudaMemcpyDeviceToHost));
	return(host);
}

static uint32_t GridBlocks(uint32_t width)
{
	return((width + GRID_THREADS - 1u) / GRID_THREADS);
}

int main(void)
{
	const uint64_t cells = (uint64_t)GRID_ROWS * GRID_DIM;
	const std::vector<uint16_t> a = GridFill(cells,1u), b = GridFill(cells,2u);
	uint16_t *da = GridUpload(a), *db = GridUpload(b), *dout;
	std::vector<uint16_t> out;
	uint64_t index, bad;
	GridCheck(LmLaunchGridRows(GRID_ROWS) == LM_LAUNCH_GRID_Y_MAX && LmLaunchGridRows(17u) == 17u,"launches clamp rows in y to 65535");
	GRID_CUDA(cudaMalloc(&dout,(uint64_t)GRID_ROWS * GRID_STRIDE * sizeof(uint16_t)));

	GRID_CUDA(cudaMemset(dout,0,cells * sizeof(uint16_t)));
	LM_LAUNCH((LmCopyRowsKernel<GRID_THREADS>),dim3(GridBlocks(GRID_DIM),LmLaunchGridRows(GRID_ROWS)),GRID_THREADS,0,0,da,dout,GRID_ROWS,GRID_DIM);
	GRID_CUDA(cudaPeekAtLastError());
	out = GridDownload(dout,cells);
	GridCheck(memcmp(out.data(),a.data(),cells * sizeof(uint16_t)) == 0,"copy rows covers all 70000 rows");

	LM_LAUNCH((LmAddRowsKernel<GRID_THREADS>),dim3(GridBlocks(GRID_DIM),LmLaunchGridRows(GRID_ROWS)),GRID_THREADS,0,0,dout,db,dout,GRID_ROWS,GRID_DIM);
	GRID_CUDA(cudaPeekAtLastError());
	out = GridDownload(dout,cells);
	for (index = 0u, bad = 0u; index < cells; index++)
	{
		const float want = GridBf16(a[index]) + GridBf16(b[index]);
		bad += fabsf(GridBf16(out[index]) - want) > fabsf(want) * 0.01f + 1e-3f ? 1u : 0u;
	}
	GridCheck(bad == 0u,"add rows in place covers all 70000 rows");

	GRID_CUDA(cudaMemset(dout,0,(uint64_t)GRID_ROWS * GRID_STRIDE * sizeof(uint16_t)));
	LM_LAUNCH((LmCopyRowsStridedKernel<GRID_THREADS>),dim3(GridBlocks(GRID_DIM),LmLaunchGridRows(GRID_ROWS)),GRID_THREADS,0,0,da,dout,GRID_ROWS,GRID_DIM,
		(uint64_t)GRID_STRIDE);
	GRID_CUDA(cudaPeekAtLastError());
	out = GridDownload(dout,(uint64_t)GRID_ROWS * GRID_STRIDE);
	for (index = 0u, bad = 0u; index < (uint64_t)GRID_ROWS * GRID_STRIDE; index++)
	{
		const uint64_t row = index / GRID_STRIDE, column = index % GRID_STRIDE;
		bad += out[index] != (column < GRID_DIM ? a[row * GRID_DIM + column] : (uint16_t)0u) ? 1u : 0u;
	}
	GridCheck(bad == 0u,"strided copy covers all 70000 rows and leaves the padding");

	{
		const uint64_t routes = (uint64_t)GRID_ROWS * GRID_TOP_K;
		std::vector<uint32_t> packed_row(routes);
		std::vector<float> weight(routes);
		const std::vector<uint16_t> packed = GridFill(routes * GRID_DIM,3u);
		for (index = 0u; index < routes; index++)
		{
			packed_row[index] = (uint32_t)((index * 7919u) % routes);
			weight[index] = 0.25f + (float)(index % 5u) * 0.125f;
		}
		uint16_t *dpacked = GridUpload(packed);
		uint32_t *drow = GridUpload(packed_row);
		float *dweight = GridUpload(weight);
		LM_LAUNCH((LmMoeFinalizeKernel<GRID_THREADS>),dim3(GridBlocks(GRID_DIM),LmLaunchGridRows(GRID_ROWS)),GRID_THREADS,0,0,dpacked,drow,dweight,dout,GRID_ROWS,
			GRID_TOP_K,GRID_DIM);
		GRID_CUDA(cudaPeekAtLastError());
		out = GridDownload(dout,cells);
		for (index = 0u, bad = 0u; index < cells; index++)
		{
			const uint64_t token = index / GRID_DIM, element = index % GRID_DIM;
			float want = 0.0f;
			for (uint32_t route = 0u; route < GRID_TOP_K; route++)
				want += GridBf16(packed[(uint64_t)packed_row[token * GRID_TOP_K + route] * GRID_DIM + element]) * weight[token * GRID_TOP_K + route];
			bad += fabsf(GridBf16(out[index]) - want) > fabsf(want) * 0.01f + 1e-3f ? 1u : 0u;
		}
		GridCheck(bad == 0u,"MoE finalize covers all 70000 tokens");
		GRID_CUDA(cudaFree(dpacked));
		GRID_CUDA(cudaFree(drow));
		GRID_CUDA(cudaFree(dweight));
	}

	{
		const uint64_t count = (uint64_t)GRID_ROWS * GRID_WIDTH * GRID_RANKS;
		const std::vector<uint16_t> gathered = GridFill(count,4u);
		uint16_t *dgathered = GridUpload(gathered), *drows;
		GRID_CUDA(cudaMalloc(&drows,count * sizeof(uint16_t)));
		GRID_CUDA(cudaMemset(drows,0,count * sizeof(uint16_t)));
		LM_LAUNCH((LmRankMajorRowsKernel<GRID_THREADS>),dim3(GridBlocks(GRID_WIDTH * GRID_RANKS),LmLaunchGridRows(GRID_ROWS)),GRID_THREADS,0,0,dgathered,drows,
			GRID_ROWS,GRID_WIDTH,GRID_RANKS);
		GRID_CUDA(cudaPeekAtLastError());
		out = GridDownload(drows,count);
		for (index = 0u, bad = 0u; index < count; index++)
		{
			const uint64_t row = index / (GRID_WIDTH * GRID_RANKS), column = index % (GRID_WIDTH * GRID_RANKS);
			bad += out[index] != gathered[(((column / GRID_WIDTH) * GRID_ROWS) + row) * GRID_WIDTH + (column % GRID_WIDTH)] ? 1u : 0u;
		}
		GridCheck(bad == 0u,"rank-major rows cover all 70000 rows");
		GRID_CUDA(cudaFree(dgathered));
		GRID_CUDA(cudaFree(drows));
	}

	GRID_CUDA(cudaFree(da));
	GRID_CUDA(cudaFree(db));
	GRID_CUDA(cudaFree(dout));
	printf("%s row grid limit\n",grid_failures == 0 ? "PASS" : "FAIL");
	return(grid_failures == 0 ? 0 : 1);
}
