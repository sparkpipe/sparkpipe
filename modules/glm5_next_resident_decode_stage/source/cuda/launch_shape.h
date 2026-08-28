#pragma once
// The layer's launch geometry, in a header of its own so a host harness can
// shadow it through the shim include path - the same mechanism that swaps the
// GEMM for the recorder. The one-thread-per-block host macro is faithful only
// to kernels instantiated at one thread, so the shim's copy of this file sets
// GLM5_NEXT_LAYER_THREADS to 1 and nothing about the production values changes
// here.
#define GLM5_NEXT_LAYER_THREADS 256u
// The latent attention decode kernel requires latent <= 8 * threads; named
// separately so a one-thread host build can keep every other kernel at one
// thread.
#define GLM5_NEXT_ATTN_THREADS 256u
