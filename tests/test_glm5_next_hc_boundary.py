#!/usr/bin/env python3
"""Execute production boundary kernels on host, or on CUDA with --cuda."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    text = SOURCE.read_text()
    kernels = text[text.index("__global__ static void SparkGlm5NextBoundaryLoadKernel"):
                   text.index("__global__ static void SparkGlm5NextEmbeddingKernel")]
    launches = [line for line in text.splitlines() if "SparkGlm5NextBoundary" in line and "<<<" in line]
    assert len(launches) == 2
    assert all("GLM5_NEXT_HC * GLM5_NEXT_HIDDEN +" in line for line in launches)
    prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#define __global__
#define GLM5_NEXT_HC SPARK_GLM5_NEXT_MODEL_HC_MULT
#define GLM5_NEXT_HIDDEN SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define WIDTH (GLM5_NEXT_HC * GLM5_NEXT_HIDDEN)
static struct { uint32_t x,y; } blockIdx,blockDim,threadIdx;
static uint16_t source[100u * WIDTH],boundary[102u * WIDTH],result[100u * WIDTH];
'''
    probe = r'''
int32_t main(void)
{
	uint32_t counts[5] = {1u,3u,17u,97u,100u},i,test,rows;
	assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT == WIDTH);
	assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION == 2u);
	blockDim.x = 256u;
	for (test=0u; test<5u; test++)
	{
		rows = counts[test];
		for (i=0u; i<(rows * WIDTH); i++)
			source[i] = (uint16_t)((i * 31u) ^ (i / GLM5_NEXT_HIDDEN));
		memset(boundary,0xa5,sizeof(boundary));
		memset(result,0x5a,sizeof(result));
		for (blockIdx.y=0u; blockIdx.y<=rows; blockIdx.y++)
			for (blockIdx.x=0u; blockIdx.x<=(WIDTH / blockDim.x); blockIdx.x++)
				for (threadIdx.x=0u; threadIdx.x<blockDim.x; threadIdx.x++)
					SparkGlm5NextBoundaryStoreKernel(source,boundary,1u,rows);
		assert(memcmp(source,boundary + WIDTH,rows * WIDTH * sizeof(uint16_t)) == 0);
		for (i=0u; i<WIDTH; i++)
			assert(boundary[i] == 0xa5a5u && boundary[((rows + 1u) * WIDTH) + i] == 0xa5a5u);
		for (blockIdx.y=0u; blockIdx.y<=rows; blockIdx.y++)
			for (blockIdx.x=0u; blockIdx.x<=(WIDTH / blockDim.x); blockIdx.x++)
				for (threadIdx.x=0u; threadIdx.x<blockDim.x; threadIdx.x++)
					SparkGlm5NextBoundaryLoadKernel(boundary,result,1u,rows);
		assert(memcmp(source,result,rows * WIDTH * sizeof(uint16_t)) == 0);
	}
	return(0);
}
'''
    compiler = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"]
    if args.cuda:
        prefix = "#include <cuda_runtime.h>\n" + prefix.replace("#define __global__\n", "")
        prefix = prefix.replace("static struct { uint32_t x,y; } blockIdx,blockDim,threadIdx;", "")
        prefix = prefix.replace("static uint16_t source", "__device__ __managed__ uint16_t source")
        probe = probe.replace("blockDim.x = 256u;", "assert(cudaFree(0) == cudaSuccess);")
        loop = "\t\tfor (blockIdx.y=0u; blockIdx.y<=rows; blockIdx.y++)\n\t\t\tfor (blockIdx.x=0u; blockIdx.x<=(WIDTH / blockDim.x); blockIdx.x++)\n\t\t\t\tfor (threadIdx.x=0u; threadIdx.x<blockDim.x; threadIdx.x++)\n\t\t\t\t\t"
        for name, params in (("Store", "source,boundary,1u,rows"), ("Load", "boundary,result,1u,rows")):
            old = loop + f"SparkGlm5NextBoundary{name}Kernel({params});"
            assert probe.count(old) == 1
            probe = probe.replace(old, f"\t\tSparkGlm5NextBoundary{name}Kernel<<<dim3((WIDTH / 256u) + 1u,rows + 1u),256u>>>({params});\n\t\tassert(cudaDeviceSynchronize() == cudaSuccess);")
        compiler = ["nvcc", "-std=c++17", "-arch=sm_121a", "-O2"]
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / ("probe.cu" if args.cuda else "probe.c"), Path(directory) / "probe"
        source.write_text(prefix + kernels + probe)
        subprocess.run([*compiler,
                        '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"', "-Iinclude",
                        "-Imodel-families/glm5_next/include",
                        "-Imodules/glm5_next_resident_decode_stage/include",
                        str(source), "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print(f"PASS {'CUDA' if args.cuda else 'host'} HC boundary bit preservation, row offsets and guards at B1/3/17/97/100")


if __name__ == "__main__":
    main()
