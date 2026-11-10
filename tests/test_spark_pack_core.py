#!/usr/bin/env python3
import math
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import spark_pack_codecs as codecs  # noqa: E402
import spark_pack_common as common  # noqa: E402
import spark_pack_wire as wire  # noqa: E402

DRIVER = r"""
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_weightd_manifest.h"

static int digest(const char *path)
{
    static const size_t steps[] = {1u, 5u, 16u, 37u};
    static unsigned char buffer[1u << 16];
    size_t bytes, used = 0u, step = 0u;
    FILE *file = fopen(path, "rb");
    SparkCk128Context context;
    uint8_t value[16];
    char hex[SPARK_CK128_HEX_BYTES];
    if (file == 0)
        return 1;
    bytes = fread(buffer, 1u, sizeof(buffer), file);
    fclose(file);
    SparkCk128Initialize(&context);
    while (used < bytes)
    {
        size_t take = steps[step++ % 4u];
        if (take > bytes - used)
            take = bytes - used;
        SparkCk128Update(&context, buffer + used, take);
        used += take;
    }
    SparkCk128Finalize(&context, value);
    SparkCk128DigestToHex(value, hex);
    printf("%s\n", hex);
    return 0;
}

static int manifest(const char *path, const char *pack_bytes)
{
    SparkWeightdManifest loaded;
    uint32_t index, byte;
    SparkStatus status = SparkWeightdManifestLoad(path, strtoull(pack_bytes, 0, 10), &loaded);
    printf("status %d\n", (int)status);
    if (status != SPARK_STATUS_OK)
        return 0;
    for (index = 0u; index < loaded.range_count; index++)
    {
        const SparkWeightdRange *range = &loaded.ranges[index];
        printf("%u %u %u %llu %llu ", range->layer, range->expert, range->kind,
               (unsigned long long)range->offset, (unsigned long long)range->bytes);
        for (byte = 0u; byte < 16u; byte++)
            printf("%02x", range->digest[byte]);
        printf("\n");
    }
    SparkWeightdManifestDestroy(&loaded);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "layout") == 0)
    {
        printf("%zu %zu %zu %zu\n", sizeof(SparkStagePackHeaderCommon),
               offsetof(SparkStagePackHeaderCommon, mtp_layer_count),
               offsetof(SparkStagePackHeaderCommon, directory_offset),
               offsetof(SparkStagePackHeaderCommon, file_bytes));
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "ck128") == 0)
        return digest(argv[2]);
    if (argc == 4 && strcmp(argv[1], "manifest") == 0)
        return manifest(argv[2], argv[3]);
    return 2;
}
"""


def bits_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def refuses(call):
    try:
        call()
    except common.PackFailure:
        return True
    return False


def reference_pack(codes, bits):
    out = bytearray()
    for row in codes.tolist():
        for first in range(0, len(row), 8):
            word = 0
            for index, code in enumerate(row[first:first + 8]):
                word |= (code & ((1 << bits) - 1)) << (index * bits)
            out += word.to_bytes(bits, "little")
    return bytes(out)


def check_scalar_codecs(failures):
    narrow = {0x3F800000: 0x3F80, 0x3F808000: 0x3F80, 0x3F818000: 0x3F82, 0x3F808001: 0x3F81, 0xC0000000: 0xC000}
    for bits, expected in narrow.items():
        if codecs.float_to_bf16_bits(bits_float(bits)) != expected:
            failures.append(f"BF16 narrowing rounds 0x{bits:08x} to nearest even 0x{expected:04x}")
    if codecs.bf16_bytes_to_floats(codecs.floats_to_bf16_bytes([1.0, -2.0, 0.5])) != [1.0, -2.0, 0.5]:
        failures.append("BF16 bytes round-trip exact values")
    if bytes(codecs.widen_bf16_bytes(b"\x80\x3f\x00\xc0")) != b"\x00\x00\x80\x3f\x00\x00\x00\xc0":
        failures.append("BF16 widening places each word in the high half of an F32")
    if codecs.f32_round(1e39) != math.inf or codecs.f32_round(-1e39) != -math.inf:
        failures.append("F32 rounding saturates to signed infinity")
    if codecs.dtype_bytes("BF16") != 2 or codecs.dtype_bytes("F8_E4M3") != 1 or not refuses(lambda: codecs.dtype_bytes("Q3")):
        failures.append("dtype widths come from the table and unknown dtypes are refused")
    if not refuses(lambda: codecs.check_e8m0_scales("plane", b"\x7f\xff")):
        failures.append("an E8M0 NaN scale is refused")


def check_vector_codecs(failures):
    pattern = np.random.default_rng(5).integers(0, 1 << 32, 4096, dtype=np.uint64).astype(np.uint32)
    pattern = pattern[(pattern & 0x7F800000) != 0x7F800000]
    vector = codecs.f32_to_bf16(pattern.view(np.float32))
    if [int(value) for value in vector] != [codecs.float_to_bf16_bits(bits_float(int(bits))) for bits in pattern]:
        failures.append("vector BF16 narrowing matches the scalar rounding for every finite value")
    nan = np.array([0x7F800001, 0xFF800001, 0x7FC00000], dtype=np.uint32).view(np.float32)
    if [int(value) for value in codecs.f32_to_bf16(nan, quiet_nan=True)] != [0x7FC0, 0xFFC0, 0x7FC0]:
        failures.append("quiet-NaN narrowing keeps the sign and sets the quiet bit")
    lut = codecs.E4M3_LUT
    if (lut[0x7E], lut[0x01], lut[0x08], lut[0xBE]) != (448.0, 2.0 ** -9, 2.0 ** -6, -1.75) or not math.isnan(lut[0x7F]):
        failures.append("the E4M3 table decodes normals, subnormals, sign and NaN")
    if codecs.E4M3_FINITE_LUT[0x7F] != 480.0 or codecs.E4M3_FINITE_LUT[0xFF] != -480.0:
        failures.append("the finite E4M3 table decodes 0x7f as 480")
    if list(codecs.E2M1_LUT) != [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]:
        failures.append("the E2M1 table lists the sixteen FP4 values")
    if list(codecs.e2m1_unpack(np.array([[0x7A]], dtype=np.uint8))[0]) != [-1.0, 6.0]:
        failures.append("E2M1 unpacking takes the low nibble first")
    packed = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)
    decoded = codecs.nvfp4_to_bf16(packed, np.array([[3.0]], dtype=np.float32), 0.5)
    expected = codecs.f32_to_bf16(codecs.E2M1_LUT * np.float32(1.5))
    if decoded.tolist() != [expected.tolist()] or codecs.nvfp4_to_bf16(packed, np.array([[3.0]], dtype=np.float32), 0.5, scale_first=True).tolist() != [expected.tolist()]:
        failures.append("NVFP4 decoding applies the group scale and the global scale")
    blocks = codecs.fp8_block_dequant_bf16(np.full((130, 129), 0x38, dtype=np.uint8), np.array([[0.5, 2.0], [4.0, 8.0]], dtype=np.float32), lut)
    if (int(blocks[0, 0]), int(blocks[0, 128]), int(blocks[129, 0]), int(blocks[129, 128])) != (0x3F00, 0x4000, 0x4080, 0x4100):
        failures.append("FP8 block dequantization uses one scale per 128x128 block")
    codes, values = codecs.encode_e4m3_round_up(np.array([0.0, 2.0 ** -9, 1.0, 1.01, 448.0, 1000.0], dtype=np.float32))
    if codes.tolist() != [0, 1, 0x38, 0x39, 0x7E, 0x7E] or values.tolist() != [0.0, 2.0 ** -9, 1.0, 1.125, 448.0, 448.0]:
        failures.append("E4M3 scale encoding rounds up to the next code and saturates at 448")
    generator = np.random.default_rng(9)
    for bits in (6, 7, 8):
        sample = generator.integers(-(1 << (bits - 1)), 1 << (bits - 1), (3, 32), dtype=np.int16)
        if codecs.pack_subbyte(sample, bits) != reference_pack(sample, bits):
            failures.append(f"{bits}-bit packing stores eight two's-complement codes per {bits} bytes, low code first")
    if not refuses(lambda: codecs.pack_subbyte(np.zeros((1, 12), dtype=np.int16), 6)):
        failures.append("sub-byte packing refuses a partial eight-code group")


def check_wire(failures):
    digests = [bytes([1]) * 32, bytes([2]) * 32]
    header = wire.codec_header(list(range(20)), 512, 4096, b"0123", digests)
    fields = wire.CODEC_HEADER.unpack(header)
    if len(header) != 264 or fields[:22] != tuple(range(20)) + (512, 4096) or fields[22] != b"0123" + bytes(61):
        failures.append("the codec header carries twenty fields, the directory offset, the file size and a NUL-padded revision")
    if fields[23:] != (digests[0], digests[1], bytes(32)) or header[-7:] != bytes(7):
        failures.append("the codec header carries three digest slots, unset slots zero, and zero padding")
    if len(wire.codec_header(list(range(20)), 512, 4096, b"r", header_bytes=257)) != 257:
        failures.append("the codec header can be cut to 257 bytes")
    if not refuses(lambda: wire.codec_header(list(range(20)), 512, 4096, b"r" * 66)):
        failures.append("a revision longer than its field is refused")
    entry = wire.Entry(3, 7, 4, 5, 1, 2, 64, 128)
    entry.payload_offset, entry.payload_bytes, entry.scale_offset, entry.scale_bytes = 1024, 16384, 17408, 256
    if wire.CODEC_ENTRY.unpack(wire.serialize_entry(entry)) != (3, 7, 4, 5, 1, 2, 64, 128, 1024, 16384, 17408, 256):
        failures.append("a directory entry serializes its eight fields and four extents")
    sizes = [(10, 3), (0, 0), (5, 2)]
    policies = {
        (): ([(512, 768), (1024, 0), (1024, 1280)], 1282),
        (("scales_last", True),): ([(512, 1024), (768, 0), (768, 1280)], 1282),
        (("align_scales", False),): ([(512, 522), (768, 0), (768, 773)], 775),
        (("align_payloads", False), ("align_scales", False)): ([(300, 310), (313, 0), (313, 318)], 320),
    }
    for policy, expected in policies.items():
        if wire.region_offsets(sizes, 300, **dict(policy)) != expected:
            failures.append(f"region placement follows the {dict(policy) or 'default'} policy")
    if wire.STAGEPACK_HEADER.size != 120 or wire.STAGEPACK_ENTRY.size != 56:
        failures.append("the stage-pack header is 120 bytes and its entries 56")


def check_common(failures):
    if common.tp_shard_range(512, 4, 3) != (384, 128) or not refuses(lambda: common.tp_shard_range(512, 4, 4)):
        failures.append("a tensor-parallel shard is the rank's equal slice and the rank must exist")
    if not refuses(lambda: common.tp_shard_range(384, 4, 0, block=128)):
        failures.append("a shard that splits a quantization block is refused")
    if common.stage_pack_name("m", "sp", 4, 2, 1, 0) != "m.tp4.rank2.sp" or common.stage_pack_name("m", "sp", 4, 2, 3, 1) != "m.tp4.pp3.stage1.rank2.sp":
        failures.append("stage-pack names carry the pipeline stage only when there is more than one")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "pack"
        path.write_bytes(b"old")
        try:
            with common.replaced_output(path) as out:
                out.write(b"partial")
                raise common.PackFailure("stop")
        except common.PackFailure:
            pass
        if path.read_bytes() != b"old" or sorted(item.name for item in Path(scratch).iterdir()) != ["pack"]:
            failures.append("a failed write keeps the old output and leaves no temporary")
        with common.replaced_output(path) as out:
            writer = common.HashingWriter(out)
            writer.write(b"new bytes")
        digest = common.write_sha256_sidecar(path)
        if path.read_bytes() != b"new bytes" or digest != writer.digest.hexdigest():
            failures.append("a finished write replaces the output and the hashing writer digests what it wrote")
        if (Path(scratch) / "pack.sha256").read_text() != f"{digest}  pack\n":
            failures.append("the sha256 sidecar names the file in sha256sum format")


def check_native(failures):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        (scratch / "driver.c").write_text(DRIVER)
        binary = scratch / "driver"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"), str(scratch / "driver.c"),
                        str(ROOT / "src/spark_ck128.c"), str(ROOT / "runtime/spark_weightd_manifest.c"), "-o", str(binary)], check=True)
        layout = subprocess.run([str(binary), "layout"], capture_output=True, text=True, check=True).stdout.split()
        if [int(value) for value in layout] != [wire.STAGEPACK_HEADER.size, 25 * 4, 26 * 4, 26 * 4 + 8]:
            failures.append("the stage-pack header struct matches the C header field for field")
        generator = random.Random(17)
        for length in (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 4099):
            data = generator.randbytes(length)
            (scratch / "data").write_bytes(data)
            native = subprocess.run([str(binary), "ck128", str(scratch / "data")], capture_output=True, text=True, check=True).stdout.strip()
            if native != wire.ck128(data).hex():
                failures.append(f"ck128 of {length} bytes matches the runtime digest")
        pack = generator.randbytes(1 << 15)
        records = []
        for layer in (3, 4):
            for expert in (1, 0):
                for kind in (45, 44):
                    offset = 4096 + ((layer - 3) * 4 + expert * 2 + (kind - 44)) * 1024
                    records.append((layer, expert, kind, offset, 512, wire.ck128(pack[offset:offset + 512])))
        (scratch / "pack.experts").write_bytes(wire.expert_manifest_bytes(records, len(records)))
        loaded = subprocess.run([str(binary), "manifest", str(scratch / "pack.experts"), str(len(pack))],
                                capture_output=True, text=True, check=True).stdout.split("\n")
        expected = ["status 0"] + [f"{layer} {expert} {kind} {offset} {length} {digest.hex()}"
                                   for layer, expert, kind, offset, length, digest in sorted(records)]
        if loaded[:-1] != expected:
            failures.append("the runtime loads the expert manifest with every range, extent and digest")
        overlapping = records[:1] + [(3, 1, 46, records[0][3] + 256, 512, bytes(16))]
        for label, raw in (("overlapping ranges", wire.expert_manifest_bytes(overlapping, 2)),
                           ("a count below the records", wire.expert_manifest_bytes(records, len(records) - 1)),
                           ("a range past the pack", wire.expert_manifest_bytes([(3, 0, 44, len(pack) - 256, 512, bytes(16))], 1))):
            (scratch / "bad.experts").write_bytes(raw)
            status = subprocess.run([str(binary), "manifest", str(scratch / "bad.experts"), str(len(pack))],
                                    capture_output=True, text=True, check=True).stdout.split("\n")[0]
            if status == "status 0":
                failures.append(f"the runtime refuses a manifest with {label}")


def main():
    failures = []
    check_scalar_codecs(failures)
    check_vector_codecs(failures)
    check_wire(failures)
    check_common(failures)
    check_native(failures)
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS spark pack core: codecs, wire layout, tensor-parallel slicing and atomic output; ck128, the "
          "stage-pack header and the expert manifest agree with the runtime")


if __name__ == "__main__":
    main()
