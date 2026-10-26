import random
import struct
import subprocess
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parents[1]

SOURCE = r'''
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/kernel_math.cuh"
#include <stdio.h>
int main(void)
{
	uint32_t bits;
	while ( fread(&bits,4,1,stdin) == 1 )
	{
		const uint16_t out = SparkMathFloatToBf16(__uint_as_float(bits));
		if ( fwrite(&out,2,1,stdout) != 1 )
			return(1);
	}
	return(0);
}
'''


def expected(bits):
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return (bits >> 16) | 0x0040
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def main():
    nans = [0x7F800001, 0x7F80FFFF, 0x7FBFFFFF, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF, 0xFFC00000]
    edges = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x7F7F7FFF, 0x7F7F8000, 0x3F808000, 0x3F818000, 0x00008000, 0x00018000]
    rng = random.Random(2610)
    finite = [rng.randrange(0, 0x7F800000) for _ in range(4096)]
    finite += [b | 0x80000000 for b in finite[:1024]]
    cases = nans + edges + finite
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / 'probe.cu').write_text(SOURCE)
        subprocess.run([host_cuda_cxx(), '-std=c++17', '-O2', '-I' + str(ROOT), '-I' + str(ROOT / 'tests/host_cuda'), '-x', 'c++',
                        str(path / 'probe.cu'), '-o', str(path / 'probe')], check=True)
        result = subprocess.run([str(path / 'probe')], input=struct.pack(f'<{len(cases)}I', *cases), capture_output=True, check=True)
    got = struct.unpack(f'<{len(cases)}H', result.stdout)
    for bits, value in zip(cases, got):
        if value != expected(bits):
            raise SystemExit(f'FAIL bf16 conversion of 0x{bits:08x}: got 0x{value:04x}, expected 0x{expected(bits):04x}')
    for bits, value in zip(nans, got):
        if (value & 0x7F80) != 0x7F80 or (value & 0x007F) == 0 or (value >> 15) != (bits >> 31):
            raise SystemExit(f'FAIL NaN 0x{bits:08x} became 0x{value:04x}, not a NaN of the same sign')
    print(f'PASS bf16 conversion: {len(nans)} NaN payloads stay NaN with their sign, {len(edges)} edges and {len(finite)} finite values round to nearest even')


if __name__ == '__main__':
    main()
