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
    parser.add_argument('--kernel', type=Path, default=ROOT/'inference/kernels/tp_reduce.cuh')
    args = parser.parse_args()
    source = r'''
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "TP_KERNEL"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
static uint16_t input[16][17*4096],output[4][17*4096+8];
static float sums[16][17*4096+8],groups[4][17*4096+8];
static void fold(float *destination,LmTpF32Contributions<16> inputs,uint32_t elements)
{
	for (blockIdx.x=0; blockIdx.x<elements+8; blockIdx.x++)
		LmTpF32SumKernel(destination,inputs,elements);
}
int main(void)
{
	uint32_t header[3],degree,elements,rank,group,other,index;
	LmTpF32Contributions<16> inputs = {};
	if ( fread(header,4,3,stdin) != 3 )
		return(1);
	degree = header[0]; elements = header[1]*header[2];
	if ( (degree != 4 && degree != 16) || elements == 0 || elements > 17*4096 )
		return(2);
	blockDim.x = 1; threadIdx.x = 0;
	for (rank=0; rank<degree; rank++)
	{
		if ( fread(input[rank],2,elements,stdin) != elements )
			return(3);
		for (index=elements; index<elements+8; index++)
			sums[rank][index] = 12345.0f;
		for (blockIdx.x=0; blockIdx.x<elements+8; blockIdx.x++)
			LmTpBf16ToF32Kernel(sums[rank],input[rank],elements);
	}
	for (rank=0; rank<degree; rank+=2)
	{
		inputs = {}; inputs.rank[rank] = sums[rank]; inputs.rank[rank+1] = sums[rank+1];
		fold(sums[rank],inputs,elements);
	}
	for (rank=0; rank<degree; rank+=4)
	{
		inputs = {}; inputs.rank[rank] = sums[rank]; inputs.rank[rank+2] = sums[rank+2];
		fold(sums[rank],inputs,elements);
		memcpy(groups[rank/4],sums[rank],elements*sizeof(float));
	}
	for (group=0; group<degree/4; group++)
	{
		if ( degree == 16 )
		{
			inputs = {};
			for (other=0; other<degree/4; other++)
				inputs.rank[other*4] = other == group ? sums[group*4] : groups[other];
			fold(sums[group*4],inputs,elements);
		}
		memset(output[group],0xa5,sizeof(output[group]));
		for (blockIdx.x=0; blockIdx.x<elements+8; blockIdx.x++)
			LmTpF32ToBf16Kernel(output[group],sums[group*4],elements);
		if ( fwrite(output[group],2,elements+8,stdout) != elements+8 )
			return(4);
	}
	for (rank=0; rank<degree; rank++)
		for (index=elements; index<elements+8; index++)
			if ( sums[rank][index] != 12345.0f )
				return(5);
	return(0);
}
'''.replace('TP_KERNEL', str(args.kernel.resolve()))
    rng = np.random.default_rng(1616)
    cases = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'probe.cu').write_text(source)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-I'+str(ROOT/'inference/kernels'), '-x', 'c++', str(path/'probe.cu'), '-o', str(path/'probe')], check=True)
        for degree in (4, 16):
            for rows in (1, 3, 17):
                for width in (7, 257, 4096):
                    values = bf16(rng.integers(-8192,8193,(degree,rows*width))/64)
                    values[:,0] = 0
                    values[:4,0] = bf16([256,1,-256,0])
                    answer = bf16(np.sum(fp32(values).astype(np.float64),axis=0))
                    expected = np.full((degree//4,rows*width+8),0xa5a5,dtype=np.uint16)
                    expected[:,:rows*width] = answer
                    data = struct.pack('<3I',degree,rows,width)+values.tobytes()
                    actual = np.frombuffer(subprocess.run([str(path/'probe')],input=data,capture_output=True,check=True,timeout=20).stdout,dtype=np.uint16).reshape(expected.shape)
                    if not np.array_equal(actual,expected):
                        raise RuntimeError(f'FP32 tree mismatch TP{degree} B{rows} width={width}: {np.count_nonzero(actual != expected)}')
                    cases += 1
    print(f'PASS {cases} actual FP32 tree arithmetic cases: TP4/TP16, B1/B3/B17, all group roots, aliasing, cancellation and bounds')


if __name__ == '__main__':
    main()
