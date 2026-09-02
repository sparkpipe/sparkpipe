#pragma once
#define LM_HOST_CUDA_PRELUDE 1


#include <math.h>
#include "cuda_fp16.h"
#include <string.h>
#include <stdint.h>

#ifndef LM_HOST_CUDA_WMMA_STUB
#define LM_HOST_CUDA_WMMA_STUB
namespace nvcuda
{
namespace wmma
{
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum { mem_row_major = 0 };
template <typename Use, int M, int N, int K, typename Element,
	typename Layout = void>
struct fragment { unsigned storage[8]; };
template <typename Fragment, typename Value> static inline void fill_fragment(Fragment &, Value) {}
template <typename Fragment, typename Pointer, typename Stride> static inline void load_matrix_sync(Fragment &, Pointer, Stride) {}
template <typename Pointer, typename Fragment, typename Stride, typename Layout> static inline void store_matrix_sync(Pointer, Fragment &, Stride, Layout) {}
template <typename Accum, typename A, typename B, typename C> static inline void mma_sync(Accum &, A &, B &, C &) {}
}
}
#endif

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__

struct LmHostDim3
{
	unsigned x, y, z;
	LmHostDim3(unsigned a = 1u, unsigned b = 1u, unsigned c = 1u) : x(a), y(b), z(c) {}
};

typedef LmHostDim3 dim3;

extern LmHostDim3 blockIdx;
extern LmHostDim3 threadIdx;
extern LmHostDim3 blockDim;
extern LmHostDim3 gridDim;

#define __shared__
#define LM_HOST_SHARED_BYTES 65536u

#define LM_HOST_WARP_LANES 1u
#define __syncthreads() ((void)0)
#define __syncwarp(...) ((void)0)

#define __expf(x) expf(x)
#define __logf(x) logf(x)
#define __powf(x, y) powf((x), (y))
#define __cosf(x) cosf(x)
#define __sinf(x) sinf(x)
#define __tanf(x) tanf(x)
#define __fdividef(x, y) ((x) / (y))

static inline float rsqrtf(float value) { return 1.0f / sqrtf(value); }

static inline float __uint_as_float(unsigned bits)
{
	float out; memcpy(&out, &bits, sizeof(out)); return out;
}
static inline float __int_as_float(int bits)
{
	float out; memcpy(&out, &bits, sizeof(out)); return out;
}
static inline unsigned __float_as_uint(float value)
{
	unsigned bits; memcpy(&bits, &value, sizeof(bits)); return bits;
}
static inline int __float2int_rn(float value) { return (int)nearbyintf(value); }
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
static inline float2 make_float2(float a, float b) { float2 v; v.x = a; v.y = b; return v; }
static inline __half __ushort_as_half(unsigned short bits) { __half h; h.raw = bits; return h; }
static inline unsigned short __half_as_ushort(__half h) { return h.raw; }
static inline float __shfl_down_sync(unsigned, float, unsigned, int = 32) { return 0.0f; }
static inline float __shfl_sync(unsigned, float value, int, int = 32) { return value; }
static inline unsigned __ballot_sync(unsigned, int) { return 0u; }
static inline void __threadfence_block(void) {}

static inline unsigned long long __cvta_generic_to_shared(const void *p) { return (unsigned long long)p; }
static inline unsigned long long __cvta_generic_to_global(const void *p) { return (unsigned long long)p; }

static inline unsigned atomicAdd(unsigned *address, unsigned value)
{
	unsigned old = *address; *address = old + value; return old;
}
static inline int atomicAdd(int *address, int value)
{
	int old = *address; *address = old + value; return old;
}
static inline float atomicAdd(float *address, float value)
{
	float old = *address; *address = old + value; return old;
}
static inline unsigned atomicMax(unsigned *address, unsigned value)
{
	unsigned old = *address; if (value > old) *address = value; return old;
}

template <typename T> static inline T __ldg(const T *pointer) { return *pointer; }
template <typename T> static inline T __ldcs(const T *pointer) { return *pointer; }
static inline __half2 __floats2half2_rn(float low, float high)
{
	__half2 out;
	out.x = __float2half(low);
	out.y = __float2half(high);
	return out;
}
static inline float2 __half22float2(__half2 value)
{
	float2 out;
	out.x = __half2float(value.x);
	out.y = __half2float(value.y);
	return out;
}

typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorNotSupported 801
static inline cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaMemsetAsync(void *, int, unsigned long long, dim3) { return cudaSuccess; }

#define LM_UNPAREN(...) __VA_ARGS__
#define LM_LAUNCH(kernel, grid, block, shared, stream, ...)                   \
	do {                                                                      \
		dim3 lm_grid = (grid);                                                \
		(void)(block); (void)(shared); (void)(stream);                        \
		blockDim = dim3(1u, 1u, 1u);                                          \
		gridDim = lm_grid;                                                    \
		threadIdx = dim3(0u, 0u, 0u);                                         \
		for (unsigned lm_z = 0u; lm_z < lm_grid.z; ++lm_z)                    \
			for (unsigned lm_y = 0u; lm_y < lm_grid.y; ++lm_y)                \
				for (unsigned lm_x = 0u; lm_x < lm_grid.x; ++lm_x)            \
				{                                                             \
					blockIdx = dim3(lm_x, lm_y, lm_z);                        \
					LM_UNPAREN kernel(__VA_ARGS__);                           \
				}                                                             \
	} while (0)

#define LM_HOST_LAUNCH(grid, kernel_call)                                     \
	do {                                                                      \
		dim3 lm_grid = (grid);                                                \
		blockDim = dim3(1u, 1u, 1u);                                          \
		gridDim = lm_grid;                                                    \
		threadIdx = dim3(0u, 0u, 0u);                                         \
		for (unsigned lm_z = 0u; lm_z < lm_grid.z; ++lm_z)                    \
			for (unsigned lm_y = 0u; lm_y < lm_grid.y; ++lm_y)                \
				for (unsigned lm_x = 0u; lm_x < lm_grid.x; ++lm_x)            \
				{                                                             \
					blockIdx = dim3(lm_x, lm_y, lm_z);                        \
					kernel_call;                                              \
				}                                                             \
	} while (0)
