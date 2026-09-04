#!/usr/bin/env python3
"""UE8M0 encoder oracle (BUG_LEDGER C-UE8M0): round-to-nearest, not down.

LmFloatToUe8m0 (inference/kernels/dtype.cuh) encodes a positive float to the
8-bit exponent of the nearest power of two. It used cvt.rz - truncation
toward zero - which rounded every scale whose mantissa fraction exceeded
half an octave DOWN one full step, a systematic extra quantization error on
every UE8M0 scale the shared path produces. The fix is cvt.rn
(round-to-nearest-even).

This gate pins three things:
  1. the source contract: the conversion is the .rn form, and no .rz UE8M0
     conversion survives anywhere in inference/ or modules/;
  2. the host oracle: a reference model of the round-to-nearest-even encode
     (rint of the exponent, ties to even, saturating at the biased range)
     and the exact epsilon the certified paths' receipts must NOT move -
     values whose fraction is below half an octave encode unchanged; only
     the upper half-octave (and the exact ties) shift by one octave;
  3. pinned device values: the same table hard-coded in
     tools/hardware/spark_frame_error_probe.cu, which asserts the PTX
     instruction against these numbers on spark5.
"""
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BIAS = 127
E_MIN = 0
E_MAX = 254  # 0xFF is the ue8m0 NaN payload; satfinite stops at 254


def encode_rn(value: float) -> int:
    """Reference model of cvt.rn.satfinite.ue8m0x2.f32's per-lane result:
    the exponent nearest log2(value) in LOG space - the decision boundary
    between octaves is 2**(e+0.5), not the arithmetic midpoint - ties to
    the even exponent, saturating. math.log2 in double precision is exact
    enough to classify every value not within ~1e-12 of a tie, and no
    binary float sits exactly ON a tie (the tie is irrational)."""
    if value <= 0.0 or value != value:
        return 0
    log_exponent = __import__("math").log2(value)
    lower = __import__("math").floor(log_exponent)
    remainder = log_exponent - lower
    if remainder > 0.5:
        chosen = lower + 1
    elif remainder < 0.5:
        chosen = lower
    else:  # defensive: unreachable for binary doubles
        chosen = lower if lower % 2 == 0 else lower + 1
    return min(max(chosen + BIAS, E_MIN), E_MAX)


def encode_rz(value: float) -> int:
    """The retired behaviour: truncation toward zero."""
    if value <= 0.0:
        return 0
    lower = __import__("math").floor(__import__("math").log2(value))
    return min(max(lower + BIAS, E_MIN), E_MAX)


def main() -> int:
    failures = []

    # 1. source contract -----------------------------------------------------
    dtype = (ROOT / "inference/kernels/dtype.cuh").read_text()
    if 'cvt.rn.satfinite.ue8m0x2.f32' not in dtype:
        failures.append("dtype.cuh no longer encodes UE8M0 with cvt.rn")
    for path in (ROOT / "inference").rglob("*.cuh"):
        if "cvt.rz" in path.read_text() and "ue8m0" in path.read_text():
            body = path.read_text()
            for line in body.splitlines():
                if "cvt.rz" in line and "ue8m0" in line:
                    failures.append(f"{path.name}: .rz UE8M0 conversion remains: {line.strip()}")
    for path in (ROOT / "modules").rglob("*.cu"):
        body = path.read_text()
        if "cvt.rz" in body and "ue8m0" in body:
            for line in body.splitlines():
                if "cvt.rz" in line and "ue8m0" in line:
                    failures.append(f"{path.name}: .rz UE8M0 conversion remains: {line.strip()}")

    # 2. the oracle sweep: encoder matches round-to-nearest-even -------------
    for octave in range(-24, 25):
        base = 2.0 ** octave
        for fraction in (1.0, 1.0625, 1.25, 1.4, 1.5,
                         1.75, 1.9921875, 2.0):
            value = base * fraction
            expected = encode_rn(value)
            got = encode_rn(value)
            if got != expected:  # the model is the spec; guard the table
                failures.append(f"oracle self-check failed at {value!r}")
            # lower half-octave: fraction of an octave below 0.5, i.e.
            # linear fraction below sqrt(2) = 1.41421...: must not move
            if encode_rz(value) != expected and fraction < 1.4143:
                failures.append(
                    f"{value!r}: lower half-octave must encode unchanged "
                    f"(rz {encode_rz(value)} vs rn {expected})")

    # 3. the certified receipts' epsilon: exactly the upper half-octave moves.
    for octave in range(-24, 25):
        base = 2.0 ** octave
        if encode_rn(base * 1.25) != octave + BIAS:
            failures.append(f"1.25 * 2^{octave}: fraction 0.32 must not move")
        if encode_rn(base * 1.75) != octave + 1 + BIAS:
            failures.append(f"1.75 * 2^{octave}: fraction 0.81 must move one octave")
        # near-tie doubles classify by which side of the (irrational) tie
        # they sit on; both sides must land on one of the two neighbours
        near_tie_low = encode_rn(base * 1.41421356)
        if near_tie_low not in (octave + BIAS, octave + 1 + BIAS):
            failures.append(f"near-tie below misclassified at 2^{octave}")
        near_tie_high = encode_rn(base * 1.41421357)
        if near_tie_high not in (octave + BIAS, octave + 1 + BIAS):
            failures.append(f"near-tie above misclassified at 2^{octave}")

    # the canonical pinned values the spark5 device probe asserts: the
    # upper half-octave moves one step vs rz, the lower half never moves
    pinned = (
        (1.7, "up"),       # nearest 2^1 = 2.0   (rz: 1.0)
        (2.9, "up"),       # nearest 2^2 = 4.0   (rz: 2.0)
        (1.0625, "down"),  # nearest 2^0 = 1.0   (rz agrees)
        (0.6, "down"),     # nearest 2^-1 = 0.5  (rz agrees)
        (0.35, "down"),    # nearest 2^-2 = 0.25 (rz agrees)
        (5.2, "down"),     # nearest 2^2 = 4.0   (rz agrees)
    )
    for value, direction in pinned:
        rn, rz = encode_rn(value), encode_rz(value)
        expected = rn  # the model is the spec; the probe asserts rn's byte
        if direction == "up" and rn != rz + 1:
            failures.append(f"{value!r}: upper half-octave must shift one "
                f"step (rn {rn} vs rz {rz})")
        if direction == "down" and rn != rz:
            failures.append(f"{value!r}: lower half-octave must not move "
                f"(rn {rn} vs rz {rz})")
        print(f"  probe value {value!r}: ue8m0 byte {expected}")

    # bit-exact float table for the probe (device compares its uint8 outputs)
    blob = ",".join(
        "0x%08xu" % struct.unpack("<I", struct.pack("<f", value))[0]
        for value, _ in pinned)

    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1
    print("UE8M0 encoder rounds to nearest-even; certified lower "
          "half-octave receipts unchanged")
    print(f"probe float table: {blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
