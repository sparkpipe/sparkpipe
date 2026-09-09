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
#include "SIGMOID_KERNEL"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
float lm_norm_shared[4104];
static uint16_t input[17*257];
static float output[17*257+2];
int main(void)
{
	uint32_t header[2],rows,width,index;
	if ( fread(header,4,2,stdin) != 2 )
		return(1);
	rows = header[0]; width = header[1];
	if ( rows == 0 || rows > 17 || width == 0 || width > 257 )
		return(2);
	if ( fread(input,2,rows*width,stdin) != rows*width )
		return(3);
	for (index=0; index<rows*width+2; index++)
		output[index] = -123.0f;
	blockDim.x = 32;
	for (blockIdx.x=0; blockIdx.x<rows; blockIdx.x++)
		for (threadIdx.x=0; threadIdx.x<32; threadIdx.x++)
			LmBf16SigmoidRowsKernel<32>(input,output+1,width);
	return(fwrite(output,4,rows*width+2,stdout) == rows*width+2 ? 0 : 4);
}
'''.replace('SIGMOID_KERNEL', str(args.kernel.resolve()))
    rng = np.random.default_rng(9531)
    cases = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'probe.cu').write_text(source)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-x', 'c++', str(path/'probe.cu'), '-o', str(path/'probe')], check=True)
        for rows in (1, 3, 17):
            for width in (4, 7, 257):
                values = rng.integers(-256, 257, rows*width)/32
                anchors = [-0.376953125, 0.69921875, 1.234375, -0.2412109375, 0, -80, 80]
                values[:min(len(values),len(anchors))] = anchors[:min(len(values),len(anchors))]
                inputs = bf16(values)
                expected = np.concatenate(([-123], fp32(bf16(1/(1+np.exp(-fp32(inputs).astype(np.float64))))), [-123])).astype(np.float32)
                data = struct.pack('<2I',rows,width)+inputs.tobytes()
                actual = np.frombuffer(subprocess.run([str(path/'probe')],input=data,capture_output=True,check=True,timeout=10).stdout,dtype=np.float32)
                if not np.array_equal(actual,expected):
                    raise RuntimeError(f'Sigmoid mismatch rows={rows} width={width}: {np.count_nonzero(actual != expected)}')
                cases += 1
    print(f'PASS {cases} actual BF16 sigmoid kernel cases: captured beta logits, odd batches and widths, saturation, zero and buffer guards')


if __name__ == '__main__':
    main()
