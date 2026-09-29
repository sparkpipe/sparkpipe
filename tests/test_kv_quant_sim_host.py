#!/usr/bin/env python3
import itertools
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kv_quant_codec as codec_reference

SOURCE = ROOT / "tests" / "host_cuda" / "kv_quant_sim_host.cu"
CODEC_ID = {codec_reference.BF16: 0, codec_reference.FP8_E4M3: 1, codec_reference.MXFP4: 2}
LAUNCH_OK = 0
LAUNCH_ERR_SHAPE = -41
TOKEN_PROBE = r'''
#include "sparkpipe/spark_kv_quant_sim.h"
#include <stdio.h>
int main(void)
{
	static const uint32_t groups[] = {0u,32u,64u,128u,256u};
	SparkKvQuantSim sim;
	char token[SPARK_KV_QUANT_SIM_TOKEN_BYTES];
	uint32_t latent,latent_group,index,index_group,state,parsed;
	for (latent = 0u; latent < 4u; latent++)
	for (latent_group = 0u; latent_group < 5u; latent_group++)
	for (index = 0u; index < 4u; index++)
	for (index_group = 0u; index_group < 5u; index_group++)
	for (state = 0u; state < 3u; state++)
	{
		sim.latent_codec = latent; sim.latent_group = groups[latent_group];
		sim.index_codec = index; sim.index_group = groups[index_group];
		sim.state_codec = state;
		printf("%u %u %u %u %u %u %d %s\n",latent,groups[latent_group],index,groups[index_group],state,
			SparkKvQuantSimValid(&sim),SparkKvQuantSimToken(&sim,token,sizeof(token)),token);
	}
	for (latent = 0u; latent < 3u; latent++)
		if ( SparkKvQuantSimParseCodec(SparkKvQuantSimCodecName(latent),&parsed) != 0 || parsed != latent )
			return(2);
	for (state = 0u; state < 2u; state++)
		if ( SparkKvStateSimParseCodec(SparkKvStateSimCodecName(state),&parsed) != 0 || parsed != state )
			return(3);
	if ( SparkKvQuantSimParseCodec("fp8",&parsed) == 0 || SparkKvStateSimParseCodec("fp16",&parsed) == 0 )
		return(4);
	sim.latent_codec = 1u; sim.latent_group = 128u; sim.index_codec = 1u; sim.index_group = 128u; sim.state_codec = 1u;
	if ( SparkKvQuantSimToken(&sim,token,8u) != -3 || token[0] != 0 )
		return(5);
	for (parsed = 0u; parsed < 512u; parsed++)
	{
		SparkKvQuantSim unpacked;
		uint32_t repacked;
		int32_t status = SparkKvQuantSimUnpack(parsed,&unpacked);
		if ( status == 0 && (SparkKvQuantSimPack(&unpacked,&repacked) != 0 || repacked != parsed || SparkKvQuantSimToken(&unpacked,token,sizeof(token)) != 0) )
			return(6);
		printf("packed %u %d %s\n",parsed,status,status == 0 ? token : "-");
	}
	return(0);
}
'''


def fail(message):
    print("FAIL", message)
    raise SystemExit(1)


def build(directory):
    binary = Path(directory) / "kv_quant_sim_host"
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O2", f"-I{ROOT}", f"-I{ROOT}/include", f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(binary)], capture_output=True, text=True)
    if result.returncode != 0:
        fail("host build: " + result.stderr[:2000])
    return binary


def run_rows(binary, rows, width, codec, group):
    rows = np.ascontiguousarray(rows, dtype=np.uint16)
    header = struct.pack("<5I", rows.shape[0], width, rows.shape[1], CODEC_ID[codec], group)
    result = subprocess.run([str(binary), "rows"], input=header + rows.tobytes(), capture_output=True, timeout=60)
    if result.returncode != 0:
        fail(f"host run rows exited {result.returncode}")
    status = struct.unpack("<i", result.stdout[:4])[0]
    return status, np.frombuffer(result.stdout[4:], dtype=np.uint16).reshape(rows.shape)


def bf16(values):
    return codec_reference.f32_to_bf16(np.asarray(values, dtype=np.float32))


def random_rows(rng, count, width):
    mantissa = rng.standard_normal((count, width))
    spread = np.exp2(rng.integers(-12, 6, (count, 1)) + rng.integers(-3, 4, (count, width)))
    return bf16(mantissa * spread)


def edge_rows(width, codec, group):
    table, _, format_max = codec_reference.FORMAT[codec]
    rows = []
    rows.append(np.zeros(width, dtype=np.uint16))
    signed_zero = np.zeros(width, dtype=np.uint16)
    signed_zero[1::2] = 0x8000
    rows.append(signed_zero)
    for exponent in (-20, 0, 7):
        grid = np.concatenate(([format_max], (table[:-1] + table[1:]) / 2.0))
        values = np.resize(grid, width) * 2.0 ** exponent
        values[1::3] *= -1.0
        rows.append(bf16(values))
    boundary = np.full(width, 0.5, dtype=np.float64)
    boundary[::group] = format_max
    rows.append(bf16(boundary))
    above = boundary.copy()
    above[::group] = format_max * (1.0 + 2.0 ** -7)
    rows.append(bf16(above))
    tiny = np.full(width, 2.0 ** -130)
    tiny[::5] = 2.0 ** -133
    rows.append(bf16(tiny))
    huge = np.linspace(-3.0e38, 3.0e38, width)
    rows.append(bf16(huge))
    special = np.linspace(-1.0, 1.0, width)
    special[3] = np.nan
    if width >= 3 * group:
        special[group + 1] = np.inf
        special[2 * group + 2] = -np.inf
    rows.append(bf16(special))
    return np.stack(rows)


def check_codec_known_answers():
    if codec_reference.store_bytes_per_row("fp8_e4m3", 128, 512) != 516:
        fail("fp8 group-128 latent row must store 512 + 4 bytes")
    if codec_reference.store_bytes_per_row("mxfp4", 32, 512) != 272:
        fail("mxfp4 group-32 latent row must store 256 + 16 bytes")
    one = bf16([448.0] + [1.0] * 127).reshape(1, 128)
    codes, scales = codec_reference.encode(one, "fp8_e4m3", 128)
    if scales[0, 0] != 127 or codes[0, 0] != 0x7E or codes[0, 1] != 0x38:
        fail(f"fp8 known answer: scale {scales[0, 0]} codes {codes[0, :2]}")
    half = bf16([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4).reshape(1, 32)
    decoded = codec_reference.bf16_to_f32(codec_reference.round_trip(half, "mxfp4", 32))[0, :8]
    if not np.array_equal(decoded, np.array([6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)):
        fail(f"mxfp4 ties must go to the even code: {decoded}")


def check_rows(binary, rng):
    cases = 0
    for codec, group in (("fp8_e4m3", 128), ("fp8_e4m3", 64), ("mxfp4", 32)):
        for width, stride in ((512, 512), (512, 576), (128, 128), (256, 320)):
            rows = random_rows(rng, 33, stride)
            rows = np.concatenate((rows, np.pad(edge_rows(width, codec, group), ((0, 0), (0, stride - width)))))
            status, got = run_rows(binary, rows, width, codec, group)
            if status != LAUNCH_OK:
                fail(f"{codec} g{group} width {width}: launch status {status}")
            want = rows.copy()
            want[:, :width] = codec_reference.round_trip(rows[:, :width], codec, group)
            if not np.array_equal(got, want):
                bad = np.argwhere(got != want)[0]
                fail(f"{codec} g{group} width {width} stride {stride}: {np.count_nonzero(got != want)} elements "
                     f"differ from the real-codec decode, first at {tuple(bad)}: "
                     f"in {rows[tuple(bad)]:#06x} got {got[tuple(bad)]:#06x} want {want[tuple(bad)]:#06x}")
            permutation = rng.permutation(rows.shape[0])
            status, reordered = run_rows(binary, rows[permutation], width, codec, group)
            if status != LAUNCH_OK or not np.array_equal(reordered, got[permutation]):
                fail(f"{codec} g{group}: reordering the rows changed the bits")
            finite = np.all(np.isfinite(codec_reference.bf16_to_f32(got)), axis=1)
            status, again = run_rows(binary, got[finite], width, codec, group)
            if status != LAUNCH_OK or not np.array_equal(again, got[finite]):
                fail(f"{codec} g{group}: a stored row does not re-store to itself")
            cases += 1
    return cases


def check_bf16_and_refusals(binary, rng):
    rows = random_rows(rng, 17, 512)
    rows[3, 7] = 0x7FC1
    status, got = run_rows(binary, rows, 512, "bf16", 0)
    if status != LAUNCH_OK or got.tobytes() != rows.tobytes():
        fail("bf16 must be a byte-identical no-op")
    for codec, group, width in (("bf16", 32, 512), ("fp8_e4m3", 32, 512), ("fp8_e4m3", 256, 512),
                                ("mxfp4", 64, 512), ("mxfp4", 16, 512), ("fp8_e4m3", 128, 448),
                                ("fp8_e4m3", 128, 640)):
        status, got = run_rows(binary, rows, width, codec, group)
        if status != LAUNCH_ERR_SHAPE or got.tobytes() != rows.tobytes():
            fail(f"{codec} group {group} width {width} must be refused without touching the rows (status {status})")
        if width > rows.shape[1]:
            continue
        try:
            codec_reference.round_trip(rows[:, :width], codec, group)
        except ValueError:
            continue
        fail(f"the reference codec accepted {codec} group {group} width {width}")


def check_tokens(directory):
    source = Path(directory) / "token_probe.c"
    binary = Path(directory) / "token_probe"
    source.write_text(TOKEN_PROBE)
    result = subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", f"-I{ROOT}/include",
                             str(source), "-o", str(binary)], capture_output=True, text=True)
    if result.returncode != 0:
        fail("token probe build: " + result.stderr[:2000])
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        fail(f"token probe exited {result.returncode}")
    tokens = {}
    valid_groups = {0: {0}, 1: {64, 128}, 2: {32}}
    packed_tokens = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "packed":
            if int(parts[2]) == 0:
                packed_tokens[parts[3]] = int(parts[1])
            continue
        latent, latent_group, index, index_group, state, valid, status = map(int, parts[:7])
        token = parts[7] if len(parts) > 7 else ""
        expected = (latent in valid_groups and latent_group in valid_groups[latent]
                    and index in valid_groups and index_group in valid_groups[index] and state < 2)
        if bool(valid) != expected or (status == 0) != expected:
            fail(f"validity of {parts[:5]} is {valid}/{status}, expected {expected}")
        if not expected:
            if token:
                fail(f"an invalid configuration produced token {token}")
            continue
        if token in tokens:
            fail(f"{token} names both {tokens[token]} and {parts[:5]}")
        tokens[token] = parts[:5]
    if tokens.get("kv=bf16/bf16/fp32/store") != ["0", "0", "0", "0", "0"]:
        fail("the identity configuration must be kv=bf16/bf16/fp32/store")
    if set(packed_tokens) != set(tokens):
        fail(f"node-context packing covers {len(packed_tokens)} configurations, the token set has {len(tokens)}")
    if packed_tokens["kv=bf16/bf16/fp32/store"] != 0:
        fail("the identity configuration must pack to 0 so an unset flag field means BF16")
    for token in ("kv=fp8_e4m3.g128/bf16/fp32/sim", "kv=fp8_e4m3.g64/bf16/fp32/sim",
                  "kv=mxfp4.g32/bf16/fp32/sim", "kv=bf16/fp8_e4m3.g128/fp32/sim", "kv=bf16/bf16/bf16/sim"):
        if token not in tokens:
            fail(f"missing token {token}")
    return len(tokens)


def check_state(binary):
    result = subprocess.run([str(binary), "state"], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        fail(f"host run state exited {result.returncode}")
    values = {line.split()[0]: int(line.split()[1]) for line in result.stdout.splitlines()}
    for key in ("grid_wave_vs_steps_mismatch", "grid_vs_rounded_reference_mismatch", "grid_state_off_bf16_grid"):
        if values.get(key) != 0:
            fail(f"{key} = {values.get(key)}")
    if values.get("grid_vs_fp32_differs", 0) == 0:
        fail("the BF16-grid state never differed from the FP32 state; the test cannot see the rounding")
    return values


def main():
    rng = np.random.default_rng(20260929)
    check_codec_known_answers()
    with tempfile.TemporaryDirectory() as directory:
        binary = build(directory)
        cases = check_rows(binary, rng)
        check_bf16_and_refusals(binary, rng)
        tokens = check_tokens(directory)
        state = check_state(binary)
    print(f"PASS {cases} row-sim cases equal the real-codec store round trip (fp8_e4m3 g128/g64, mxfp4 g32; "
          f"ties, zeros, -0, subnormals, BF16 extremes, NaN/Inf groups, padded rows); row order and "
          f"re-storing leave the bits unchanged; bf16 is a byte no-op; bad groups and widths are refused")
    print(f"PASS {tokens} distinct kv= tokens, one per valid configuration; each packs into a 7-bit node-context field and back, identity packs to 0, every other field value is refused")
    print(f"PASS BF16-grid KDA state: a {6}-token wave equals 6 decode steps bit for bit, every step equals the "
          f"FP32 update rounded to BF16, the state stays on the BF16 grid "
          f"(write-back-only BF16 state differs wave vs steps in "
          f"{state.get('writeback_bf16_wave_vs_steps_mismatch')} values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
