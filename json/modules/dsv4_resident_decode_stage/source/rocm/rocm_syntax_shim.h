/*
 * rocm_syntax_shim.h — LOCAL SYNTAX-PROOF SHIM ONLY. NEVER COMPILED INTO
 * ANY SERVING OR VALIDATION IMAGE.
 *
 * The authoring workstation has no ROCm toolchain (same box limitation
 * documented in hwiface_evidence_r2c2_rocm_probe.hip). This shim lets
 * g++ -fsyntax-only check the shared helper header's host+device surface
 * by providing minimal HIP spellings. It deliberately does NOT try to make
 * kernel-launch syntax (<<<>>>) parseable — the .hip island TUs get their
 * real proof from hipcc --offload-arch=gfx950 -fsyntax-only on hardware or
 * any ROCm-equipped machine; validate_rocm_syntax.sh prints those commands.
 */
#pragma once

#include <stdint.h>

typedef enum {
	hipSuccess = 0,
	hipErrorNotReady = 600,
	hipErrorInvalidValue = 1
} hipError_t;

typedef struct { unsigned x, y, z; } dim3;
typedef void *hipStream_t;
typedef void *hipEvent_t;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

__attribute__((unused)) static float __uint_as_float(uint32_t bits)
{
	float out;
	__builtin_memcpy(&out, &bits, sizeof(out));
	return(out);
}

/* Builtins referenced inside __global__ bodies. Under the shim the kernels
 * become plain uncalled functions, so these only need to parse. */
__attribute__((unused)) static uint32_t __float_as_uint(float value)
{
	uint32_t bits;
	__builtin_memcpy(&bits, &value, sizeof(bits));
	return(bits);
}

float __shfl_down(float value, unsigned int delta);
void __syncthreads(void);

typedef struct { unsigned x, y, z; } RocmShimThreadCoord;
extern RocmShimThreadCoord threadIdx;
extern RocmShimThreadCoord blockIdx;
