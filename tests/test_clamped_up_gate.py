import argparse
import pathlib
import subprocess
import tempfile

from host_cuda_compiler import host_cuda_cxx

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    text = (ROOT / 'inference/kernels/norm.cuh').read_text()
    start = text.rindex('template<uint32_t THREADS>', 0, text.index('void LmClampedUpGateKernel'))
    end = text.index('template<uint32_t THREADS>', text.index('void LmClampedUpGateKernel'))
    prefix = r'''
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
QUAL float LmBf16ToFloat(uint16_t x)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)x << 16;
    return(v.f);
}
QUAL uint16_t LmFloatToBf16(float x)
{
    union { uint32_t u; float f; } v;
    v.f = x;
    return((uint16_t)((v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16));
}
STORAGE uint16_t input[294],output[149];
'''
    suffix = r'''
int main(void)
{
    float values[7] = {-20,-10,-1,0,1,10,20};
    uint32_t row,i,j;
    float gate,up,expected,actual;
    for (row=0; row<3; row++)
        for (i=0; i<7; i++)
            for (j=0; j<7; j++)
            {
                input[row*98+i*7+j] = LmFloatToBf16(values[j]);
                input[row*98+49+i*7+j] = LmFloatToBf16(values[i]);
            }
    output[0] = output[148] = 0xa5a5u;
    LAUNCH
    for (row=0; row<3; row++)
        for (i=0; i<7; i++)
            for (j=0; j<7; j++)
            {
                gate = fminf(values[i],10.0f);
                up = fmaxf(-10.0f,fminf(values[j],10.0f));
                expected = LmBf16ToFloat(LmFloatToBf16((float)(gate/(1.0+exp(-(double)gate))*up)));
                actual = LmBf16ToFloat(output[1+row*49+i*7+j]);
                if ( fabsf(actual-expected) > fabsf(expected)*0.008f+1e-12f )
                    return(1);
            }
    assert(output[0] == 0xa5a5u && output[148] == 0xa5a5u);
    puts("PASS clamped up-gate: signs, bounds, unclamped negative gate, three rows and output guards");
    return(0);
}
'''
    if args.cuda:
        prefix = '#include <cuda_runtime.h>\n#include "inference/kernels/kernel_math.cuh"\n' + prefix.replace('QUAL','__host__ __device__').replace('STORAGE','__device__ __managed__')
        launch = 'LmClampedUpGateKernel<32><<<3,32>>>(input,output+1,49,10.0f); assert(cudaDeviceSynchronize() == cudaSuccess);'
        compiler = ['/usr/local/cuda/bin/nvcc','-std=c++17','-arch=sm_121a','-I'+str(ROOT)]
    else:
        prefix = '#include "tests/host_cuda/lm_host_cuda.cuh"\n#include "inference/kernels/kernel_math.cuh"\nLmHostDim3 blockIdx,threadIdx,blockDim,gridDim;\n' + prefix.replace('QUAL','static').replace('STORAGE','')
        launch = 'for (blockIdx.x=0; blockIdx.x<3; blockIdx.x++) for (threadIdx.x=0; threadIdx.x<32; threadIdx.x++) LmClampedUpGateKernel<32>(input,output+1,49,10.0f);'
        compiler = [host_cuda_cxx(),'-std=c++17','-I'+str(ROOT),'-I'+str(ROOT / 'tests/host_cuda')]
    with tempfile.TemporaryDirectory() as directory:
        source = pathlib.Path(directory) / ('probe.cu' if args.cuda else 'probe.cpp')
        binary = pathlib.Path(directory) / 'probe'
        source.write_text(prefix+text[start:end]+suffix.replace('LAUNCH',launch))
        subprocess.run(compiler+[str(source),'-o',str(binary)],check=True)
        subprocess.run([str(binary)],check=True,timeout=15)

if __name__ == '__main__':
    main()
