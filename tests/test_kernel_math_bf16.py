#!/usr/bin/env python3
"""SparkMathFloatToBf16 rounds to nearest even and keeps every NaN a NaN.

The shared conversion (inference/kernels/kernel_math.cuh, used by every CUDA
and Metal kernel through LmFloatToBf16) is built through the host CUDA shim
and run on a set of float32 bit patterns: the NaNs whose payload sits only
in the low 16 bits (which the bare round-to-nearest-even formula carries into
the exponent and turns into infinity), quiet and signalling NaNs of both
signs, the infinities, the largest finites (which round to infinity), every
tie and carry around a power of two, subnormals, and random patterns. Each
result is compared with a reference that picks the nearer of the two
neighbouring bf16 values exactly (ties to the even one) and maps every NaN
to the canonical 0x7fff, as CUDA's __float2bfloat16_rn does. The bare
formula is run on the same patterns and must fail on the NaNs, so the check
can see the bug it guards.
"""
import random
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent

PROBE = r"""
#include "tests/host_cuda/lm_host_cuda.cuh"
#include "inference/kernels/kernel_math.cuh"
#include <stdio.h>
#include <string.h>
int main(void)
{
	uint32_t bits;
	while ( fread(&bits,4,1,stdin) == 1 )
	{
		float value;
		uint16_t shared,bare;
		memcpy(&value,&bits,4);
		shared = SparkMathFloatToBf16(value);
		bare = (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
		if ( fwrite(&shared,2,1,stdout) != 1 || fwrite(&bare,2,1,stdout) != 1 )
			return(2);
	}
	return(0);
}
"""


def exact(bits16):
    exponent = (bits16 >> 7) & 0xff
    mantissa = bits16 & 0x7f
    sign = -1 if bits16 & 0x8000 else 1
    if exponent == 0:
        return sign * Fraction(mantissa, 1 << 7) * Fraction(1, 1 << 126)
    return sign * (1 + Fraction(mantissa, 1 << 7)) * (Fraction(2) ** (exponent - 127))


def reference(bits32):
    if (bits32 & 0x7fffffff) > 0x7f800000:
        return 0x7fff
    if (bits32 & 0x7fffffff) == 0x7f800000:
        return bits32 >> 16
    value = struct.unpack("<f", struct.pack("<I", bits32))[0]
    target = Fraction(value)
    low = bits32 >> 16
    high = low + 1
    low_value = exact(low)
    if (high & 0x7f80) == 0x7f80:
        high_value = None
    else:
        high_value = exact(high)
    if high_value is None:
        limit = (Fraction(2) ** 128) * (1 if value > 0 else -1)
        midpoint = (low_value + limit) / 2
        if abs(target) < abs(midpoint):
            return low
        if abs(target) > abs(midpoint):
            return high
        return low if low % 2 == 0 else high
    if abs(target - low_value) < abs(high_value - target):
        return low
    if abs(target - low_value) > abs(high_value - target):
        return high
    return low if low % 2 == 0 else high


def patterns():
    out = []
    for sign in (0, 0x80000000):
        for payload in (1, 0x7fff, 0x8000, 0xffff, 0x10000, 0x3fffff, 0x400000, 0x7fffff):
            out.append(sign | 0x7f800000 | payload)
        out += [sign | 0x7f800000, sign | 0x7f7fffff, sign | 0x7f7f8000, sign | 0x7f7f7fff, sign | 0x7f7fc000,
                sign | 0x00000001, sign | 0x00007fff, sign | 0x00008000, sign | 0x00018000, sign | 0x007fffff, sign]
        for exponent in (1, 64, 126, 127, 128, 200, 254):
            base = sign | (exponent << 23)
            for low in (0x0000, 0x7fff, 0x8000, 0x8001, 0xffff, 0x17fff, 0x18000, 0x7f8000, 0x7fffff):
                out.append(base | low)
    rng = random.Random(20261027)
    out += [rng.getrandbits(32) for _ in range(200000)]
    return out


def main():
    cases = patterns()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "bf16_probe.cu"
        binary = Path(directory) / "bf16_probe"
        source.write_text(PROBE)
        subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-I" + str(ROOT), "-I" + str(ROOT / "tests/host_cuda"), "-x", "c++",
                        str(source), "-o", str(binary)], check=True)
        output = subprocess.run([str(binary)], input=struct.pack("<%dI" % len(cases), *cases), capture_output=True, check=True, timeout=120).stdout
    results = struct.unpack("<%dH" % (2 * len(cases)), output)
    failures = 0
    bare_nan_failures = 0
    for index, bits in enumerate(cases):
        shared, bare = results[2 * index], results[2 * index + 1]
        want = reference(bits)
        if shared != want:
            failures += 1
            if failures <= 8:
                print(f"  FAIL 0x{bits:08x}: shared 0x{shared:04x}, reference 0x{want:04x}")
        if (bits & 0x7fffffff) > 0x7f800000 and (bare & 0x7f80) == 0x7f80 and (bare & 0x7f) == 0:
            bare_nan_failures += 1
    print(f"patterns {len(cases)}, shared conversion failures {failures}, NaNs the bare formula turns into infinity {bare_nan_failures}")
    if bare_nan_failures == 0:
        print("  FAIL the bare formula keeps every NaN here; the check cannot see the bug it guards")
        failures += 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nSparkMathFloatToBf16 rounds to nearest even, overflows to infinity, and maps every NaN to 0x7fff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
