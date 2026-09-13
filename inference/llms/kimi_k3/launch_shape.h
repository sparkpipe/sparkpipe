#pragma once
// The slice's launch geometry, in a header of its own so a host harness can
// shadow it through the shim include path - the same mechanism that swaps the
// GEMM for the recorder. The one-thread-per-block host macro is faithful only
// to kernels instantiated at one thread, so the shim's copy of this file sets
// K3_LAYER_THREADS to 1 and nothing about the production values changes here.
#define K3_LAYER_THREADS 256u
// The MLA decode kernel requires latent <= 8 * threads; named separately so a
// one-thread host build can keep every other kernel at one thread.
#define K3_ATTN_THREADS 256u
#define K3_LAYER_TILE_N 128u
#define K3_LAYER_STAGES 2u
#define K3_LAYER_WARPS 8u
#define K3_HEAD_TILE 1024u
