/*
 * rocm_tu_host_proof_shim.h — LOCAL STATIC-PROOF SHIM ONLY. NEVER COMPILED
 * INTO ANY SERVING OR VALIDATION IMAGE.
 *
 * Companion of rocm_tu_host_proof.sh. Where rocm_syntax_shim.h proves the
 * launch-free common header, this shim lets g++ -fsyntax-only parse FULL
 * island translation units: kernels become plain functions (annotations
 * inert), triple-angle launches are rewritten by the driver script into
 * calls to the variadic no-op below, so EVERY kernel signature and EVERY
 * launch argument list still type-checks against real variables.
 *
 * This is a syntax/type proof only. It is NOT a device compile and never
 * replaces hipcc --offload-arch=gfx950 on ROCm hardware.
 */
#pragma once

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <stdint.h>
#include <math.h>

/* --- annotations inert ------------------------------------------------ */
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
/* Empty: sized arrays become locals, `extern __shared__` stays a legal
 * block-scope declaration. Proof cares about types, not placement. */
#define __shared__

/* --- HIP runtime spellings -------------------------------------------- */
typedef enum {
	hipSuccess = 0,
	hipErrorInvalidValue = 1,
	hipErrorNotReady = 600
} hipError_t;

typedef struct { unsigned x, y, z; } dim3;
typedef void *hipStream_t;
typedef void *hipEvent_t;

hipError_t hipGetLastError(void);

/* Thread-coordinate builtins as plain globals; the proof never executes,
 * it only type-checks the expressions that consume them. */
extern dim3 threadIdx;
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

/* --- device builtins referenced by island bodies ----------------------- */

inline float __uint_as_float(uint32_t bits)
{
	float out;
	__builtin_memcpy(&out, &bits, sizeof(out));
	return(out);
}

inline uint32_t __float_as_uint(float value)
{
	uint32_t bits;
	__builtin_memcpy(&bits, &value, sizeof(bits));
	return(bits);
}

inline void __syncthreads(void) {}

float __shfl_down(float value, unsigned int delta);		/* declared: parses only */
float rsqrtf(float value);
float __expf(float value);
float __fmul_rn(float a, float b);
float __fmaf_rn(float a, float b, float c);

inline uint32_t atomicAdd(uint32_t *address, uint32_t value)
{
	(void)address; (void)value;
	return(0u);
}

inline uint64_t atomicAdd(uint64_t *address, uint64_t value)
{
	(void)address; (void)value;
	return(0u);
}

inline int atomicAdd(int *address, int value)
{
	(void)address; (void)value;
	return(0);
}

inline float atomicAdd(float *address, float value)
{
	(void)address; (void)value;
	return(0.0f);
}

/* Constant-macro belt and braces for C++ translation units. */
#ifndef UINT32_C
#define UINT32_C(c) c##u
#endif
#ifndef UINT64_C
#define UINT64_C(c) c##ull
#endif
#ifndef UINT32_MAX
#define UINT32_MAX 4294967295u
#endif

/* Launch sink: the driver rewrites kernel<<<cfg>>>(args...) into a call to
 * this variadic no-op, keeping every argument expression in the type-check. */
template <typename First, typename... Rest>
inline void SparkRocmHostProofLaunch(First &&, Rest &&...)
{
}
