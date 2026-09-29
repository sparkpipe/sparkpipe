import hashlib
import io
import json
import struct
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_expert_dequant as dq
import glm5_next_exl3_graft as adapter
import glm5_next_expert_graft as graft
import glm5_next_resident_stagepack as pack
from test_glm5_next_expert_graft import write_pack

SMALL_HIDDEN, SMALL_INTER, SMALL_EXPERTS, SMALL_LAYERS, TP = 256, 256, 3, 5, 2


def check_lut_known_answers():
    for name, codebook in dq.CODEBOOKS.items():
        got = hashlib.sha256(dq.codebook_lut(codebook).tobytes()).hexdigest()
        assert got == dq.LUT_SHA256[codebook], (name, got)


def reference_inner(trellis, codebook):
    lut = dq.codebook_lut(codebook)
    perm = dq.tile_permutation()
    tiles_k, tiles_n, width = trellis.shape
    bits = width // 16
    out = np.zeros((tiles_k * 16, tiles_n * 16), dtype=np.float16)
    total = 256 * bits
    for i in range(tiles_k):
        for j in range(tiles_n):
            words = trellis[i, j].view(np.uint16)
            stream = 0
            for w in range(0, len(words), 2):
                stream = (stream << 32) | (int(words[w + 1]) << 16) | int(words[w])
            for position in range(256):
                end = ((position + 1) * bits) % total
                start = end - 16
                window = 0
                for bit in range(start, end):
                    window = (window << 1) | ((stream >> (total - 1 - (bit % total))) & 1)
                row, col = divmod(int(perm[position]), 16)
                out[i * 16 + row, j * 16 + col] = lut[window]
    return out


def check_inner_against_bitwise_reference():
    rng = np.random.default_rng(7)
    decoder = dq.Decoder()
    for bits in (2, 3, 4, 6):
        trellis = rng.integers(-32768, 32768, size=(2, 3, 16 * bits)).astype(np.int16)
        for codebook in (1, 2):
            assert np.array_equal(decoder.inner(trellis, codebook).view(np.uint16),
                                  reference_inner(trellis, codebook).view(np.uint16)), (bits, codebook)


KNOWN_TILE_PERMUTATION = "0e4298d72984e59f1dc4e54bc1369f652eff7261aa539292353a7aad2114062e"
KNOWN_INNER = {
    (0, 3): "102b99d67cb19e9dcbd770b7ec5f6138", (0, 4): "9c0665c228876cb0ed984d8ec7a66ca5",
    (1, 3): "53cca9ea1b4220ebb335d47ccda13e62", (1, 4): "d7bc5ada98aaa9b96397d6718452e83f",
    (2, 3): "5288b3e7cdf3633758bdbc56d39ee35a", (2, 4): "09ff61f60e8c729f15c8935c4f68ee87",
}
KNOWN_WEIGHT = {
    0: "9e258aeded3387f63ab8d5921ff203fbec602a7ca6e26d3851103d45209217b2",
    1: "95429c686887ef51fa60573b706dc028f9bc56c8033bf77a99c20fd45821a36c",
    2: "b906a2712a3b0c623fbdfdb76226074246b61cced5764476ee9008a8fd1c1b18",
}


def known_stream(tag, count):
    return np.frombuffer(hashlib.shake_256(tag.encode()).digest(count), dtype=np.uint8)


def known_trellis(tag, tiles_k, tiles_n, bits):
    raw = known_stream(tag, tiles_k * tiles_n * 16 * bits * 2).view("<i2")
    return raw.reshape(tiles_k, tiles_n, 16 * bits).astype(np.int16)


def known_scales(tag, count, low, high):
    raw = known_stream(tag, count * 2).view("<u2").astype(np.float64)
    sign = np.where(raw.astype(np.int64) & 1, -1.0, 1.0)
    return (sign * (low + (high - low) * raw / 65535.0)).astype(np.float16)


def sylvester(order):
    h = np.ones((1, 1))
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def check_known_answers():
    got = hashlib.sha256(dq.tile_permutation().astype("<i8").tobytes()).hexdigest()
    assert got == KNOWN_TILE_PERMUTATION, got
    decoder = dq.Decoder()
    for (codebook, bits), want in KNOWN_INNER.items():
        inner = decoder.inner(known_trellis(f"inner-{codebook}-{bits}", 2, 3, bits), codebook)
        got = hashlib.sha256(inner.view("<u2").tobytes()).hexdigest()[:32]
        assert got == want, (codebook, bits, got)
    for codebook, want in KNOWN_WEIGHT.items():
        codes, _ = decoder.weight_bf16(known_trellis(f"weight-{codebook}", 16, 16, 4),
                                       known_scales(f"suh-{codebook}", 256, 0.5, 2.0),
                                       known_scales(f"svh-{codebook}", 256, 0.002, 0.02), codebook)
        got = hashlib.sha256(codes.astype("<u2").tobytes()).hexdigest()
        assert got == want, (codebook, got)


def check_reconstruction_against_blockwise_reference():
    decoder = dq.Decoder()
    h = sylvester(128)
    for codebook in (0, 1, 2):
        trellis = known_trellis(f"blockwise-{codebook}", 16, 24, 3)
        suh = known_scales(f"blockwise-suh-{codebook}", 256, 0.5, 2.0)
        svh = known_scales(f"blockwise-svh-{codebook}", 384, 0.002, 0.02)
        inner = decoder.inner(trellis, codebook).astype(np.float64)
        want = np.empty_like(inner)
        for i in range(0, 256, 128):
            for j in range(0, 384, 128):
                block = h @ inner[i:i + 128, j:j + 128] @ h
                block = block * suh[i:i + 128].astype(np.float64)[:, None]
                want[i:i + 128, j:j + 128] = block * svh[j:j + 128].astype(np.float64)[None, :] / 128
        got = decoder.weight_f64(trellis, suh, svh, codebook)
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64)), codebook


def exact_bf16(value):
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    value = abs(value)
    exponent = 0
    while value >= 2:
        value /= 2
        exponent += 1
    while value < 1:
        value *= 2
        exponent -= 1
    scaled = value * 128
    floor = int(scaled)
    rest = scaled - floor
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and floor % 2 == 1):
        floor += 1
    if floor == 256:
        floor = 128
        exponent += 1
    as_f32 = struct.unpack("<I", struct.pack("<f", float(Fraction(floor, 128) * Fraction(2) ** exponent)))[0]
    return (as_f32 >> 16) | (sign << 15)


def check_bf16_rounding_is_exact():
    rng = np.random.default_rng(11)
    decoder = dq.Decoder()
    trellis = rng.integers(-32768, 32768, size=(8, 8, 48)).astype(np.int16)
    suh = (rng.choice([-1.0, 1.0], 128) * rng.uniform(0.5, 2.0, 128)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], 128) * rng.uniform(0.002, 0.02, 128)).astype(np.float16)
    codes, _ = decoder.weight_bf16(trellis, suh, svh, 1)
    left, right = decoder.weight_parts(trellis, suh, svh, 1)
    for index in rng.choice(codes.size, 400, replace=False):
        r, c = divmod(int(index), codes.shape[1])
        want = exact_bf16(Fraction(float(left[r, c])) * Fraction(float(right[c])) / 128)
        assert int(codes[r, c]) == want, (r, c, hex(int(codes[r, c])), hex(want))
    tie = np.array([[1.0 + 2.0 ** -8]])
    codes, corrected = dq.round_bf16_exact(tie * 3 / 3, tie, np.array([1.0 * 128]))
    assert int(codes[0, 0]) == 0x3F80 and corrected == 0
    above = np.array([[1.0 + 2.0 ** -8 + 2.0 ** -40]])
    rounded = np.array([[1.0 + 2.0 ** -8]])
    codes, corrected = dq.round_bf16_exact(rounded, above, np.array([128.0]))
    assert int(codes[0, 0]) == 0x3F81 and corrected == 1


def check_slices_are_exact():
    rng = np.random.default_rng(3)
    decoder = dq.Decoder()
    trellis = rng.integers(-32768, 32768, size=(16, 32, 64)).astype(np.int16)
    suh = rng.uniform(-2, 2, 256).astype(np.float16)
    svh = rng.uniform(-0.02, 0.02, 512).astype(np.float16)
    full, _ = decoder.weight_bf16(trellis, suh, svh, 2)
    for axis, start in (("out", 128), ("out", 384), ("in", 0), ("in", 128)):
        part, _ = decoder.weight_bf16(*dq.slice_linear(trellis, suh, svh, axis, start, 128), 2)
        want = full[:, start:start + 128] if axis == "out" else full[start:start + 128, :]
        assert np.array_equal(part, want), (axis, start)
    for bad in ((64, 128), (0, 64)):
        try:
            dq.slice_linear(trellis, suh, svh, "out", *bad)
        except dq.DequantFailure:
            continue
        raise AssertionError(f"misaligned slice {bad} accepted")


def small_model(monkey):
    saved = {name: getattr(adapter, name) for name in ("HIDDEN", "EXPERT_INTER", "EXPERTS", "LAYERS")}
    adapter.HIDDEN, adapter.EXPERT_INTER, adapter.EXPERTS, adapter.LAYERS = \
        SMALL_HIDDEN, SMALL_INTER, SMALL_EXPERTS, SMALL_LAYERS
    monkey.append(saved)


def restore(monkey):
    for saved in monkey:
        for name, value in saved.items():
            setattr(adapter, name, value)


def write_checkpoint(root, rng, marker_shape):
    tensors = []
    for layer in range(pack.FIRST_ROUTED, SMALL_LAYERS):
        for expert in range(SMALL_EXPERTS):
            for projection in ("up", "gate", "down"):
                name = adapter.prefix(layer, expert, projection)
                bits = 3 if (layer + expert) % 2 else 4
                k, n = (SMALL_INTER, SMALL_HIDDEN) if projection == "down" else (SMALL_HIDDEN, SMALL_INTER)
                tensors.append((f"{name}.trellis", rng.integers(-32768, 32768, size=(k // 16, n // 16, 16 * bits))
                                .astype(np.int16)))
                tensors.append((f"{name}.suh", (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 2, k)).astype(np.float16)))
                tensors.append((f"{name}.svh", rng.uniform(-0.02, 0.02, n).astype(np.float16)))
                tensors.append((f"{name}.mcg", np.array([dq.MCG_MULT - (1 << 32)], dtype=np.int32).reshape(marker_shape)))
    dq.write_safetensors(root / "model-00001-of-00001.safetensors", tensors, {})
    fetch = {"repo": "example/exl3", "revision": "a" * 40, "status": "complete",
             "lfs_sha256": {"model-00001-of-00001.safetensors":
                            hashlib.sha256((root / "model-00001-of-00001.safetensors").read_bytes()).hexdigest()}}
    (root / "SPARKPIPE_FETCH.json").write_text(json.dumps(fetch))
    return dict(tensors)


def write_spine(path, rank, rng_seed):
    entries = [
        (pack.K_ATTN_NORM, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE, 1, 1, 64, 128, 0),
        (pack.K_ROUTER, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE, 1, 3, 64, 384, 0),
    ]
    for layer in range(pack.FIRST_ROUTED, SMALL_LAYERS):
        entries.append((pack.K_EXPERT_UP_GATE, layer, pack.PAYLOAD_PACKED_WEIGHT, pack.CODEC_BF16, pack.SCALE_NONE,
                        SMALL_EXPERTS, 256, SMALL_HIDDEN, SMALL_EXPERTS * 256 * SMALL_HIDDEN * 2, 0))
        entries.append((pack.K_EXPERT_DOWN, layer, pack.PAYLOAD_PACKED_WEIGHT, pack.CODEC_BF16, pack.SCALE_NONE,
                        SMALL_EXPERTS, SMALL_HIDDEN, 128, SMALL_EXPERTS * SMALL_HIDDEN * 128 * 2, 0))
        entries.append((pack.K_SHARED_DOWN, layer, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE,
                        1, 2, 64, 256, 0))
    write_pack(path, entries, rng_seed, header_codec=pack.CODEC_BF16, tp=(TP, rank))


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    saved = sys.argv
    sys.argv = ["glm5_next_exl3_graft.py"] + [str(a) for a in argv]
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = adapter.main()
    finally:
        sys.argv = saved
    return code, out.getvalue(), err.getvalue()


def check_end_to_end(marker_shape):
    monkey = []
    small_model(monkey)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source"
            source.mkdir()
            tensors = write_checkpoint(source, np.random.default_rng(5), marker_shape)
            code, out, err = run(["slice", "--source", source, "--arm", "armx", "--out-dir", tmp / "slices",
                                  "--tp-degree", TP])
            assert code == 0, err
            decoder = dq.Decoder()
            for rank in range(TP):
                spine = tmp / f"spine{rank}" / "spine.sp"
                spine.parent.mkdir()
                write_spine(spine, rank, 100 + rank)
                output = tmp / "arm" / f"armx.rank{rank}.sp"
                output.parent.mkdir(exist_ok=True)
                slices = tmp / "slices" / f"armx.exl3slice.rank{rank:x}.safetensors"
                code, out, err = run(["graft", "--slices", slices, "--spine-pack", spine, "--output", output,
                                      "--tp-degree", TP, "--tp-rank", rank, "--model-revision", "f12e0fe1",
                                      "--arm", "armx"])
                assert code == 0, err
                receipt = json.loads(Path(str(output) + ".receipt.json").read_text())
                assert receipt["spine_digest"] == graft.pack_spine_digest(spine)
                assert receipt["expert_source"]["slice_metadata"]["tp_rank"] == rank
                assert receipt["expert_source"]["decode_stats"]["mcg.K3"] > 0
                assert receipt["expert_source"]["decode_stats"]["mcg.K4"] > 0
                result = graft.RankPack(output)
                assert result.header["expert_codec"] == pack.CODEC_BF16
                experts = result.experts()
                data = output.read_bytes()
                start = rank * 128
                for layer in range(pack.FIRST_ROUTED, SMALL_LAYERS):
                    for expert in range(SMALL_EXPERTS):
                        for projection in ("up", "gate", "down"):
                            name = adapter.prefix(layer, expert, projection)
                            full, _ = decoder.weight_bf16(tensors[f"{name}.trellis"], tensors[f"{name}.suh"],
                                                          tensors[f"{name}.svh"], 1)
                            if projection == "down":
                                want = full[start:start + 128, :].T
                                entry = experts[(pack.K_EXPERT_DOWN, layer)]
                                offset = entry["payload_offset"] + expert * SMALL_HIDDEN * 128 * 2
                            else:
                                want = full[:, start:start + 128].T
                                entry = experts[(pack.K_EXPERT_UP_GATE, layer)]
                                offset = entry["payload_offset"] + (expert * 2 + (projection == "gate")) * \
                                    128 * SMALL_HIDDEN * 2
                            got = np.frombuffer(data[offset:offset + want.size * 2], dtype=np.uint16)
                            assert np.array_equal(got.reshape(want.shape), want), (rank, layer, expert, projection)
                spine_regions = {(e["kind"], e["layer"]): e for e in graft.RankPack(spine).entries}
                spine_data = spine.read_bytes()
                for entry in result.entries:
                    if entry["kind"] in graft.EXPERT_KINDS:
                        continue
                    origin = spine_regions[(entry["kind"], entry["layer"])]
                    assert data[entry["payload_offset"]:entry["payload_offset"] + entry["payload_bytes"]] == \
                        spine_data[origin["payload_offset"]:origin["payload_offset"] + origin["payload_bytes"]]
                code, out, err = run(["check", "--source", source, "--pack", output,
                                      "--layers", "3,4", "--experts", "0,2"])
                assert code == 0 and "CHECK-PASS" in out, out + err
                bad = tmp / "arm" / f"armx.bad{rank}.sp"
                blob = bytearray(data)
                blob[experts[(pack.K_EXPERT_UP_GATE, 4)]["payload_offset"] + 2 * 128 * SMALL_HIDDEN * 2 + 1] ^= 0x40
                bad.write_bytes(bytes(blob))
                code, out, err = run(["check", "--source", source, "--pack", bad, "--layers", "4", "--experts", "1"])
                assert code == 1 and "CHECK-FAIL" in out and "2/3 equal" in out, out + err
                code, out, err = run(["graft", "--slices", slices, "--spine-pack", spine, "--output", output,
                                      "--tp-degree", TP, "--tp-rank", rank, "--model-revision", "f12e0fe1"])
                assert code == 1 and "already exist" in err
                other = tmp / "arm" / f"armx.wrong{rank}.sp"
                code, out, err = run(["graft", "--slices", tmp / "slices" / f"armx.exl3slice.rank{1 - rank:x}.safetensors",
                                      "--spine-pack", spine, "--output", other, "--tp-degree", TP,
                                      "--tp-rank", rank, "--model-revision", "f12e0fe1"])
                assert code == 1 and "requested tp2 rank" in err and not other.exists()
            corrupt = tmp / "slices" / "armx.exl3slice.rank0.safetensors"
            blob = bytearray(corrupt.read_bytes())
            blob[-1] ^= 1
            corrupt.write_bytes(bytes(blob))
            code, out, err = run(["graft", "--slices", corrupt, "--spine-pack", tmp / "spine0" / "spine.sp",
                                  "--output", tmp / "arm" / "corrupt.sp", "--tp-degree", TP, "--tp-rank", 0,
                                  "--model-revision", "f12e0fe1"])
            assert code == 1 and "sha256 differs" in err
    finally:
        restore(monkey)


def check_refusals():
    monkey = []
    small_model(monkey)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tensors = [("model.language_model.layers.3.mlp.experts.0.up_proj.trellis",
                        np.zeros((16, 16, 40), dtype=np.int16))]
            dq.write_safetensors(tmp / "model.safetensors", tensors, {})
            code, out, err = run(["slice", "--source", tmp, "--arm", "a", "--out-dir", tmp / "o", "--tp-degree", TP])
            assert code == 1 and "GRAFT-REFUSED" in err
            try:
                dq.Decoder().inner(np.zeros((1, 1, 40), dtype=np.int16), 1)
            except dq.DequantFailure:
                pass
            else:
                raise AssertionError("tile width 40 accepted")
            try:
                adapter.rank_window(32, 1)
            except dq.DequantFailure:
                pass
            else:
                raise AssertionError("64-wide rank window accepted")
    finally:
        restore(monkey)


def main():
    check_lut_known_answers()
    check_known_answers()
    check_reconstruction_against_blockwise_reference()
    check_inner_against_bitwise_reference()
    check_bf16_rounding_is_exact()
    check_slices_are_exact()
    for marker_shape in ((), (1,)):
        check_end_to_end(marker_shape)
    check_refusals()
    print("test_exl3_expert_dequant: PASS")


if __name__ == "__main__":
    main()
