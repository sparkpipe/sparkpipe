import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from host_cuda_compiler import host_cuda_cxx
from test_hc_post_host import bf16, fp32

ROOT = Path(__file__).resolve().parents[1]


def reference(window, inputs, weights, slots, starts, counts, commit, float_weights):
    result = np.full(inputs.shape, 0xa5a5, dtype=np.uint16)
    updated = window.copy()
    for slot, start, count in zip(slots, starts, counts):
        history = window[slot].copy()
        for row in range(start, start+count):
            history = np.concatenate((history[:, 1:], inputs[row, :, None]), axis=1)
            conv = np.sum(fp32(history).astype(np.float64)*fp32(weights), axis=1)
            if not float_weights:
                conv = fp32(bf16(conv)).astype(np.float64)
            result[row] = bf16(conv/(1+np.exp(-conv)))
        if commit:
            updated[slot] = history
    return result, updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=Path, default=ROOT/'inference/kernels/linear_attn.cuh')
    args = parser.parse_args()
    source = r'''
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#include "CONV_KERNEL"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
float lm_norm_shared[LM_HOST_SHARED_BYTES/sizeof(float)],state_s[LM_HOST_SHARED_BYTES/sizeof(float)];
static uint16_t window[19*257*4],input[85*257],weight[257*4],output[85*257];
static float weight_f32[257*4];
static uint32_t slots[17],starts[18],counts[17];
int main(void)
{
	uint32_t h[5],sequences,channels,rows,commit,i;
	if ( fread(h,sizeof(uint32_t),5,stdin) != 5 )
		return(1);
	sequences = h[0]; channels = h[1]; rows = h[2]; commit = h[3];
	if ( sequences > 17 || channels > 257 || rows > 85 )
		return(2);
	if ( fread(slots,4,sequences,stdin) != sequences || fread(starts,4,sequences+1,stdin) != sequences+1 || fread(counts,4,sequences,stdin) != sequences )
		return(3);
	if ( fread(window,2,19*channels*4,stdin) != 19*channels*4 || fread(input,2,rows*channels,stdin) != rows*channels || fread(weight,2,channels*4,stdin) != channels*4 )
		return(4);
	memset(output,0xa5,sizeof(output));
	for (i=0; i<channels*4; i++)
		weight_f32[i] = LmBf16ToFloat(weight[i]);
	threadIdx.x = 0; blockDim.x = 1;
	for (blockIdx.x=0; blockIdx.x<=sequences; blockIdx.x++)
		for (blockIdx.y=0; blockIdx.y<=channels; blockIdx.y++)
			if ( h[4] == 0 )
				LmCausalConvKernel<1,4,LM_CONV_SWISH,uint16_t>(window,slots,starts,counts,input,weight,output,channels,sequences,commit);
			else
				LmCausalConvKernel<1,4,LM_CONV_SWISH,float>(window,slots,starts,counts,input,weight_f32,output,channels,sequences,commit);
	if ( fwrite(output,2,rows*channels,stdout) != rows*channels )
		return(5);
	return(fwrite(window,2,19*channels*4,stdout) == 19*channels*4 ? 0 : 6);
}
'''.replace('CONV_KERNEL', str(args.kernel.resolve()))
    rng = np.random.default_rng(953)
    cases = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'probe.cu').write_text(source)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'tests/host_cuda'), '-x', 'c++', str(path/'probe.cu'), '-o', str(path/'probe')], check=True)
        for sequences in (1, 3, 17):
            for channels in (7, 257):
                slots = rng.permutation(19)[:sequences].astype(np.uint32)
                starts = np.arange(sequences+1, dtype=np.uint32)*5
                counts = (np.arange(sequences, dtype=np.uint32)%5)+1
                if sequences > 1:
                    counts[0] = 0
                rows = sequences*5
                window = bf16(rng.integers(-128,129,(19,channels,4))/64)
                inputs = bf16(rng.integers(-128,129,(rows,channels))/64)
                weights = bf16(rng.integers(-128,129,(channels,4))/64)
                for variant in range(4):
                    commit, float_weights = variant%2, variant//2
                    expected, state = reference(window, inputs, weights, slots, starts, counts, commit, float_weights)
                    data = struct.pack('<5I',sequences,channels,rows,commit,float_weights)+slots.tobytes()+starts.tobytes()+counts.tobytes()+window.tobytes()+inputs.tobytes()+weights.tobytes()
                    actual = np.frombuffer(subprocess.run([str(path/'probe')],input=data,capture_output=True,check=True,timeout=10).stdout,dtype=np.uint16)
                    wanted = np.concatenate((expected.ravel(),state.ravel()))
                    if not np.array_equal(actual,wanted):
                        raise RuntimeError(f'Convolution mismatch sequences={sequences} channels={channels} commit={commit} float_weights={float_weights}: {np.count_nonzero(actual != wanted)}')
                    cases += 1
    print(f'PASS {cases} actual convolution cases: BF16/FP32 weights, ragged rows, slot permutation, padding, empty sequences and committed/uncommitted history')


if __name__ == '__main__':
    main()
