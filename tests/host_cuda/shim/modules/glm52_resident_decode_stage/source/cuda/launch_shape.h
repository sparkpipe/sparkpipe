#pragma once
// Host copy of the layer launch geometry. One thread per block, because the
// host launch macro runs exactly one, and a kernel instantiated wider strides
// its per-thread loops past everything but element zero. Every kernel in the
// layer is proven correct at one thread by the per-kernel harnesses; the
// latent attention decode covers only every sixty-fourth query element here,
// which the recorder GEMM erases before anything reads it.
#define GLM52_LAYER_THREADS 1u
// 64, the floor the accumulator assert allows at a 512 latent. Only elements
// visited by thread zero are written on the host; the recorder erases them.
#define GLM52_ATTN_THREADS 64u
