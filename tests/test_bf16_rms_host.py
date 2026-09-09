import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from host_cuda_compiler import host_cuda_cxx
from test_hc_post_host import bf16, fp32

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=Path, default=ROOT/'inference/kernels/norm.cuh')
    args = parser.parse_args()
    source = r'''
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#include "RMS_KERNEL"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
float lm_norm_shared[4104];
static uint16_t input[17*4101],output[17*4101],weight[4096];
int main(void)
{
	uint32_t header[4],rows,width,stride,alias;
	uint16_t *destination;
	if ( fread(header,sizeof(uint32_t),4,stdin) != 4 )
		return(1);
	rows = header[0]; width = header[1]; stride = header[2]; alias = header[3];
	if ( rows == 0 || rows > 17 || width == 0 || width > 4096 || stride != width+5 )
		return(2);
	if ( fread(input,sizeof(uint16_t),rows*stride,stdin) != rows*stride || fread(weight,sizeof(uint16_t),width,stdin) != width )
		return(3);
	memset(output,0xa5,sizeof(output));
	destination = alias != 0 ? input : output;
	threadIdx.x = 0; blockDim.x = 1;
	for (blockIdx.x=0; blockIdx.x<rows; blockIdx.x++)
		LmBf16RmsNormKernel<1>(input,weight,destination,width,stride,1e-6f);
	return(fwrite(destination,sizeof(uint16_t),rows*stride,stdout) == rows*stride ? 0 : 4);
}
'''.replace('RMS_KERNEL', str(args.kernel.resolve()))
    rng = np.random.default_rng(915)
    count = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'probe.cu').write_text(source)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-x', 'c++', str(path/'probe.cu'), '-o', str(path/'probe')], check=True)
        for rows in (1,3,17):
            for width in (7,257,4096):
                inputs = bf16(rng.integers(-16,17,(rows,width+5))/16)
                if rows > 1:
                    inputs[0,:width] = 0
                weight = bf16(rng.integers(-128,129,width)/64)
                values = fp32(inputs[:,:width])
                mean = np.mean(values.astype(np.float64)**2,axis=1,keepdims=True).astype(np.float32)
                scale = np.float32(1)/np.sqrt(mean+np.float32(1e-6))
                normalized = fp32(bf16(values*scale))
                answer = bf16(normalized*fp32(weight))
                for alias in (0,1):
                    expected = inputs.copy() if alias else np.full(inputs.shape,0xa5a5,dtype=np.uint16)
                    expected[:,:width] = answer
                    data = struct.pack('<4I',rows,width,width+5,alias)+inputs.tobytes()+weight.tobytes()
                    actual = np.frombuffer(subprocess.run([str(path/'probe')],input=data,capture_output=True,check=True,timeout=10).stdout,dtype=np.uint16).reshape(expected.shape)
                    if not np.array_equal(actual,expected):
                        raise RuntimeError(f'RMS mismatch rows={rows} width={width} alias={alias}: {np.count_nonzero(actual != expected)}')
                    count += 1
    print(f'PASS {count} actual BF16 RMS kernel cases: weighted rounding, zero rows, stride guards and aliasing')


if __name__ == '__main__':
    main()
