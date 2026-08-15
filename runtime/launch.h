#pragma once

// Host launcher for the grouped GEMM.
//
// This is what stands between kernels/gemm.cuh and deleting 305,005 lines of
// vendored CUTLASS: the two call sites that reach the external GEMM need a
// first-party function with the same shape to call instead.
//
// Three things have to be right and none of them are the kernel:
//
//   1. TILE HEIGHT PER BUCKET. Rows per group is tokens*top_k/experts, and each
//      M tile re-reads its group's weight tile. A tile shorter than the group
//      splits it and doubles the weight stream, which is 96 percent of decode
//      traffic. The selection rounds UP, always: padded mma rows are free on a
//      path at 1.4 percent of BF16 peak, re-read weights are not.
//
//   2. THE DYNAMIC SHARED OPT-IN. ptxas caps a static __shared__ declaration at
//      48 KB. The kernel therefore carves its stages out of dynamic shared, and
//      dynamic shared above 48 KB requires cudaFuncSetAttribute before the first
//      launch. Skipping it fails the launch rather than corrupting anything,
//      which is the good outcome, but it fails every time.
//
//   3. THE DESCRIPTOR SWIZZLE. The tensor map encodes a swizzle span and the
//      kernel applies a matching xor. They are computed from the same row pitch
//      by the same function, which is the only reason they cannot disagree.
//
// Everything here except the four CUDA calls is arithmetic and is checked by
// tests/test_launch.c on a host.

// Geometry only. The launch itself needs the kernel, but the PLAN is arithmetic
// and a host must be able to compute it without a CUDA toolchain - which is why
// kernels/layout.cuh exists separately from kernels/mma.cuh.
#include "inference/kernels/layout.cuh"
#include <stdint.h>

// LM_LAUNCH: one spelling for a kernel launch, on a device or on a host.
//
// <<<grid, block, shared, stream>>> is syntax only nvcc accepts, so a layer
// written with it cannot be compiled for a CPU - and a layer is made of
// launches. That is what kept tests/host_cuda from reaching past individual
// kernels, which is where an external audit found three defects that every
// per-kernel test passed straight through.
//
// It also makes the launch itself checkable. tests/test_kernel_launches.py
// reads launches with a regular expression, because <<< >>> is not something a
// compiler will hand you; that gate exists because one launch was wrong four
// ways and compiled, and a regex can be defeated by reformatting the call.
// Through a macro the grid and the argument list are ordinary C++.
//
// The host expansion lives in tests/host_cuda/lm_host_cuda.cuh and runs one
// thread per block in block order, which is a schedule a correct kernel must
// also be valid under.
//
// THE KERNEL IS PARENTHESISED, AND IT HAS TO BE. LmQuantiseRowsKernel<Format,256u>
// contains a comma, and the preprocessor splits macro arguments on commas
// before it knows anything about templates - so the kernel arrives as two
// arguments and the expansion is nonsense. Wrapping it and unwrapping with a
// variadic pass-through is the standard way out, and requiring the parentheses
// on every call rather than only the ones that need them keeps the form
// uniform: a reader never has to work out whether this particular kernel has a
// comma in it.
#define LM_UNPAREN(...) __VA_ARGS__
#ifdef __CUDACC__
#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \
	LM_UNPAREN kernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__)
#endif

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)
#define LM_LAUNCH_ERR_OUTPUT (-47)

// Tile heights the library instantiates. A bucket outside this range is a
// caller error rather than a case to approximate, because the tile sizes shared
// memory and the accumulator array and both are compile-time.
#define LM_LAUNCH_TILE_MIN 16u
#define LM_LAUNCH_TILE_MAX 64u

typedef struct LmLaunchShape
{
	uint32_t tokens,top_k,expert_count,input_dimension,output_dimension;
	// stored_bits sizes the WEIGHT operand; stored_bits_a the activation, and
	// zero means symmetric. Weight-only formats stream BF16 activations against
	// sub-byte weights, and one width for both was the assumption that priced
	// that path out of the planner.
	uint32_t stored_bits,stored_bits_a,tile_n,tile_k,stages;
	// Nonzero marks the interleaved-B launch (pack V2 mxfp4_ws_interleaved_v1):
	// the weight stage is the 17-row cell grid and a TILE_K=128 BF16 activation
	// row (256 bytes) is staged as two 128-byte sectors, so the activation pitch
	// is a multiple of its swizzle span rather than equal to it.
	uint32_t interleaved_b;
}
LmLaunchShape;

typedef struct LmLaunchPlan
{
	uint32_t tile_m;
	uint32_t shared_bytes;
	uint32_t grid_blocks;
	uint32_t block_threads;
	uint64_t workspace_bytes;
	uint32_t swizzle_span;
}
LmLaunchPlan;

// Rows the busiest group is expected to hold.
//
// The mean understates it because routing is not uniform, and under a grouped
// launch the max-loaded group sets step time. The 2x headroom is a heuristic and
// is the one unmeasured number in this file; understating it costs a weight
// re-read for the overloaded groups only, not for all of them, so the failure is
// graceful. A measured route distribution would replace it.
static uint32_t LmLaunchPeakRowsPerGroup(const LmLaunchShape *shape)
{
	uint64_t mean;
	if ( shape->expert_count == 0u )
		return(0u);
	mean = (((uint64_t)shape->tokens * shape->top_k) + shape->expert_count - 1u)
		/ shape->expert_count;
	return((uint32_t)(mean * 2u));
}

static uint32_t LmLaunchSelectTile(uint32_t peak_rows)
{
	if ( peak_rows <= 16u )
		return(16u);
	if ( peak_rows <= 32u )
		return(32u);
	return(64u);
}

// The grouped tile height as a pure function of the batch shape, so a layer
// can price its expert tile tables in the route build and skip the prefix
// launches. This IS the planner's choice - one truth, exported.
static inline uint32_t LmLaunchGroupedTileM(uint32_t tokens, uint32_t top_k, uint32_t expert_count)
{
	LmLaunchShape shape = {};
	shape.tokens = tokens;
	shape.top_k = top_k;
	shape.expert_count = expert_count;
	return(LmLaunchSelectTile(LmLaunchPeakRowsPerGroup(&shape)));
}

// Shared memory the kernel will carve, matching LmGemmSharedBytes exactly. Kept
// as arithmetic rather than a call into the template so the host can size a
// pool without instantiating a kernel.
static uint32_t LmLaunchSharedBytes(const LmLaunchShape *shape, uint32_t tile_m)
{
	uint32_t bits_a = shape->stored_bits_a != 0u ? shape->stored_bits_a : shape->stored_bits;
	uint32_t a = (tile_m * shape->tile_k * bits_a) / 8u;
	uint32_t b = shape->interleaved_b != 0u
		? (17u * (shape->tile_n / 16u) * 64u)
		: (shape->tile_n * shape->tile_k * shape->stored_bits) / 8u;
	return((shape->stages * (a + b)) + (shape->stages * 8u));
}

static int32_t LmLaunchPlanBuild(const LmLaunchShape *shape, uint32_t multiprocessors, LmLaunchPlan *plan)
{
	uint32_t pitch;
	if ( shape == 0 || plan == 0 || shape->expert_count == 0u || shape->tile_n == 0u
		|| shape->tile_k == 0u || shape->stages < 2u || multiprocessors == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	plan->tile_m = LmLaunchSelectTile(LmLaunchPeakRowsPerGroup(shape));
	if ( plan->tile_m < LM_LAUNCH_TILE_MIN || plan->tile_m > LM_LAUNCH_TILE_MAX )
		return(LM_LAUNCH_ERR_TILE);
	// K MUST BE A WHOLE NUMBER OF TILES. LmGemmKernel computes
	// k_tiles = input_dimension / TILE_K, an integer division, and the stagers
	// bound rows and neurons but never K. A trailing partial tile is therefore
	// dropped from the dot product: wrong output, no crash, nothing to catch it.
	//
	// Nothing else can catch this. The tile geometry static_asserts are
	// compile-time and input_dimension is a runtime argument, so the only place
	// the two meet is here. Every K extent in the three drivers today is a
	// multiple of 256, which is why it has never bitten, but INT7 tiles at 256
	// rather than 128 and the two models without a layer.cuh are unwritten.
	// two models' 192-wide head dims exist in this tree today and would silently
	// compute nothing at all under INT7: 192 / 256 == 0 tiles.
	if ( (shape->input_dimension % shape->tile_k) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	plan->shared_bytes = LmLaunchSharedBytes(shape,plan->tile_m);
	if ( plan->shared_bytes > LM_SMEM_SM_TOTAL )
		return(LM_LAUNCH_ERR_SHARED);
	// The row pitch decides the swizzle span, and the descriptor must be built
	// with the same one the kernel applies. Both come from here. The
	// interleaved launch stages 64-byte cell rows, so its weight pitch equals
	// its span like every other path.
	pitch = (shape->tile_k * shape->stored_bits) / 8u;
	plan->swizzle_span = LmSwizzleSpanFor(pitch);
	if ( plan->swizzle_span == 0u || pitch != plan->swizzle_span )
		return(LM_LAUNCH_ERR_MAP);
	// The activation pitch must be swizzleable in its own right when the widths
	// differ; the weight span above does not vouch for it. A TILE_K=128 BF16
	// row is 256 bytes and no single span covers it, so the interleaved direct
	// path stages it as two 128-byte sectors - the pitch must then be a whole
	// number of sectors, not equal to one span.
	if ( shape->stored_bits_a != 0u )
	{
		uint32_t activation_pitch =
			(shape->tile_k * shape->stored_bits_a) / 8u;
		uint32_t activation_span = LmSwizzleSpanFor(activation_pitch);
		if ( activation_span == 0u ||
			(shape->interleaved_b == 0u
				? activation_pitch != activation_span
				: (activation_pitch % activation_span) != 0u) )
			return(LM_LAUNCH_ERR_MAP);
	}
	// Persistent grid: one CTA per SM, sized to the machine rather than the
	// problem, so a short group never leaves an SM idle behind a long one. The
	// kernel bounds its own loop on the device-side tile prefix, so an
	// over-estimate here costs an idle block and never a phantom tile.
	plan->grid_blocks = multiprocessors;
	plan->block_threads = 8u * 32u;
	plan->workspace_bytes = 0u;
	return(LM_LAUNCH_OK);
}

// THE 48 KiB DYNAMIC SHARED OPT-IN, for any kernel that needs it.
//
// ptxas grants 48 KiB of dynamic shared without asking; anything larger fails
// the launch on device every time unless cudaFuncSetAttribute opts the kernel
// in first. The delta rule carves KEY_DIM * VALUE_DIM floats (64 KiB at both
// drivers' widths), and every model family launches its own instantiation, so
// the opt-in lives here once, keyed on the kernel address, rather than being
// copied per driver. Once per kernel per device, mutex-guarded, and checked,
// because a skipped opt-in fails the launch rather than corrupting anything.
// The host recorders have no launch to attribute, so the guard compiles to a
// no-op there and the call sites keep one spelling.
#define LM_LAUNCH_MAX_OPTIN_KERNELS 16u
#define LM_LAUNCH_MAX_TRACKED_DEVICES 64u

#ifdef __CUDACC__
#include <mutex>

typedef struct LmKernelOptInEntry
{
	const void *kernel;
	uint64_t device_mask;
}
LmKernelOptInEntry;

static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	static std::mutex grant_mutex;
	static LmKernelOptInEntry grants[LM_LAUNCH_MAX_OPTIN_KERNELS];
	static uint32_t grant_count = 0u;
	cudaError_t status;
	uint64_t device_bit;
	uint32_t index;
	int device_index;

	if ( kernel == 0 )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	status = cudaGetDevice(&device_index);
	if ( status != cudaSuccess )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( device_index < 0
		|| (uint32_t)device_index >= LM_LAUNCH_MAX_TRACKED_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	device_bit = UINT64_C(1) << (uint32_t)device_index;
	{
		std::lock_guard<std::mutex> lock(grant_mutex);
		for ( index = 0u; index < grant_count; index++ )
		{
			if ( grants[index].kernel == kernel )
			{
				if ( (grants[index].device_mask & device_bit) != 0u )
					return(LM_LAUNCH_OK);
				break;
			}
		}
		if ( index == grant_count )
		{
			if ( grant_count >= LM_LAUNCH_MAX_OPTIN_KERNELS )
				return(LM_LAUNCH_ERR_ATTRIBUTE);
			grants[grant_count].kernel = kernel;
			grants[grant_count].device_mask = 0u;
			grant_count++;
		}
		status = cudaFuncSetAttribute(kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared_bytes);
		if ( status == cudaSuccess )
			grants[index].device_mask |= device_bit;
	}
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}
#else
static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	(void)kernel;
	(void)shared_bytes;
	return(LM_LAUNCH_OK);
}
#endif
