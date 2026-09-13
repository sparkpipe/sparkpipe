#pragma once
// Host copy of the slice launch geometry. One thread per block, because the
// host launch macro runs exactly one, and a kernel instantiated wider strides
// its per-thread loops past everything but element zero. Every kernel in the
// slice is proven correct at one thread by the per-kernel harnesses; the MLA
// attention decode covers only its first eight latent elements here, which the
// recorder GEMM erases before anything reads them.
#define K3_LAYER_THREADS 1u
// 64, the floor the accumulator assert allows at a 512 latent. Only elements
// visited by thread zero are written on the host; the recorder erases them.
#define K3_ATTN_THREADS 64u
#define K3_LAYER_TILE_N 128u
#define K3_LAYER_STAGES 2u
#define K3_LAYER_WARPS 8u
#define K3_HEAD_TILE 1024u
