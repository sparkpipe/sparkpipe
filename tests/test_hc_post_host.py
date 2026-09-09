import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parents[1]


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fp32(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=Path, default=ROOT/'inference/kernels/hc.cuh')
    args = parser.parse_args()
    source = r'''
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "HC_KERNEL"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
static uint16_t out[17*4096],snapshot[17*4*4096],result[17*4*4096];
static float post[17*4],comb[17*16];
int main(void)
{
	uint32_t header[4],rows,hc,width,alias;
	uint16_t *destination;
	if ( fread(header,sizeof(uint32_t),4,stdin) != 4 )
		return(1);
	rows = header[0]; hc = header[1]; width = header[2]; alias = header[3];
	if ( rows == 0 || rows > 17 || hc == 0 || hc > 4 || width == 0 || width > 4096 )
		return(2);
	if ( fread(out,sizeof(uint16_t),rows*width,stdin) != rows*width || fread(snapshot,sizeof(uint16_t),rows*hc*width,stdin) != rows*hc*width || fread(post,sizeof(float),rows*hc,stdin) != rows*hc || fread(comb,sizeof(float),rows*hc*hc,stdin) != rows*hc*hc )
		return(3);
	destination = alias != 0 ? snapshot : result;
	threadIdx.x = 0; blockDim.x = 1;
	for (blockIdx.x=0; blockIdx.x<=rows; blockIdx.x++)
		LmHcPostBf16Kernel(out,snapshot,post,comb,destination,rows,hc,width);
	return(fwrite(destination,sizeof(uint16_t),rows*hc*width,stdout) == rows*hc*width ? 0 : 4);
}
'''.replace('HC_KERNEL', str(args.kernel.resolve()))
    rng = np.random.default_rng(713)
    count = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'probe.cu').write_text(source)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-I'+str(ROOT/'inference/kernels'), '-x', 'c++', str(path/'probe.cu'), '-o', str(path/'probe')], check=True)
        for rows in (1, 3, 17):
            for hc in (1, 2, 4):
                for width in (7, 257, 4096):
                    out = bf16(rng.integers(-128,129,(rows,width))/64)
                    snapshot = bf16(rng.integers(-128,129,(rows,hc,width))/64)
                    post = (rng.integers(-8192,8193,(rows,hc))/4096).astype(np.float32)
                    comb = (rng.integers(-8192,8193,(rows,hc,hc))/4096).astype(np.float32)
                    contribution = fp32(bf16(fp32(bf16(post))[:,:,None]*fp32(out)[:,None,:]))
                    residual = fp32(bf16(np.matmul(fp32(bf16(comb)).transpose(0,2,1).astype(np.float64),fp32(snapshot).astype(np.float64))))
                    expected = bf16(contribution+residual)
                    for alias in (0,1):
                        data = struct.pack('<4I',rows,hc,width,alias)+out.tobytes()+snapshot.tobytes()+post.tobytes()+comb.tobytes()
                        actual = np.frombuffer(subprocess.run([str(path/'probe')],input=data,capture_output=True,check=True,timeout=10).stdout,dtype=np.uint16).reshape(expected.shape)
                        if not np.array_equal(actual,expected):
                            raise RuntimeError(f'HC mismatch rows={rows} hc={hc} width={width} alias={alias}: {np.count_nonzero(actual != expected)}')
                        count += 1
    print(f'PASS {count} actual HC kernel cases: BF16 boundaries, arbitrary rows, widths and aliasing')


if __name__ == '__main__':
    main()
