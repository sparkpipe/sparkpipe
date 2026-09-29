import hashlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import tracemalloc
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack
import glm5_next_expert_graft as graft

REVISION = "f12e0fe1" + "0" * 32
GLOBAL = pack.GLOBAL_LAYER


def expert_planes(codec, groups, rows, columns):
    payload, scale, encoding = graft.expected_expert_bytes(codec, groups, rows, columns)
    return payload, scale, encoding


def entries_for(codec, layers, groups=2, rows=4, columns=256, spine_rows=3):
    entries = [
        (pack.K_ATTN_NORM, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE, 1, 1, 64, 128, 0),
        (pack.K_ROUTER, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE, 1, spine_rows, 64,
         spine_rows * 128, 0),
        (pack.K_KDA_OUT_NORM, 3, pack.PAYLOAD_F32, pack.CODEC_NONE, pack.SCALE_NONE, 1, 1, 32, 128, 0),
    ]
    for layer in layers:
        for kind in (pack.K_EXPERT_UP_GATE, pack.K_EXPERT_DOWN):
            payload, scale, encoding = expert_planes(codec, groups, rows, columns)
            entries.append((kind, layer, pack.PAYLOAD_PACKED_WEIGHT, codec, encoding, groups, rows,
                            columns, payload, scale))
        entries.append((pack.K_SHARED_DOWN, layer, pack.PAYLOAD_BF16, pack.CODEC_BF16,
                        pack.SCALE_NONE, 1, 2, 64, 256, 0))
    entries.append((pack.K_EMBEDDING, GLOBAL, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE,
                    1, 2, 32, 128, 0))
    return entries


def write_pack(path, entries, seed, header_codec=pack.CODEC_FP8, tp=(16, 3), span=(1, 0, 3, 42),
               flags=0, hidden=pack.HIDDEN, revision="84c6a6aa", linear=pack.CODEC_BF16):
    rng = random.Random(seed)
    directory_offset = graft.align(pack.HEADER_BYTES)
    cursor = directory_offset + len(entries) * pack.ENTRY_BYTES
    rows = []
    for entry in entries:
        payload_bytes, scale_bytes = entry[8], entry[9]
        payload_offset = graft.align(cursor)
        cursor = payload_offset + payload_bytes
        scale_offset = 0
        if scale_bytes:
            scale_offset = graft.align(cursor)
            cursor = scale_offset + scale_bytes
        rows.append(entry[:8] + (payload_offset, payload_bytes, scale_offset, scale_bytes))
    fields = [pack.MAGIC, pack.FORMAT_VERSION, pack.HEADER_BYTES, pack.ENTRY_BYTES,
              pack.CODEC_ABI_VERSION, flags, len(entries), span[0], span[1], span[2], span[3],
              pack.LAYERS, hidden, pack.VOCAB, pack.EXPERTS, linear, header_codec,
              pack.CODEC_BF16, tp[0], tp[1]]
    header = struct.pack("<20I", *fields) + struct.pack("<QQ", directory_offset, cursor)
    header += revision.encode().ljust(65, b"\0") + bytes([7]) * 32 + bytes([8]) * 32 + bytes(32)
    header += bytes(pack.HEADER_BYTES - len(header))
    data = bytearray(cursor)
    data[:len(header)] = header
    for index, row in enumerate(rows):
        struct.pack_into(graft.ENTRY_FORMAT, data, directory_offset + index * pack.ENTRY_BYTES, *row)
        for offset, count in ((row[8], row[9]), (row[10], row[11])):
            data[offset:offset + count] = bytes(rng.getrandbits(8) for _ in range(count))
    path.write_bytes(bytes(data))
    return rows


def regions(path):
    source = graft.RankPack(path)
    data = path.read_bytes()
    out = {}
    for entry in source.entries:
        key = (entry["kind"], entry["layer"])
        out[key] = (data[entry["payload_offset"]:entry["payload_offset"] + entry["payload_bytes"]],
                    data[entry["scale_offset"]:entry["scale_offset"] + entry["scale_bytes"]]
                    if entry["scale_bytes"] else b"", dict(entry))
    return source, out


def run_graft(spine, expert, output, codec, rank=3, degree=16, chunk=4096):
    return graft.graft(spine, expert, output, degree, rank, codec, REVISION, chunk,
                       arm="test", source_commit="abc")


def expect_refused(directory, spine, expert, codec, needle, rank=3, degree=16):
    output = directory / "refused.sp"
    try:
        run_graft(spine, expert, output, codec, rank, degree)
    except pack.PackFailure as error:
        assert needle in str(error), (needle, str(error))
    else:
        raise AssertionError(f"graft accepted a mismatch that must be refused: {needle}")
    leftovers = [p.name for p in directory.iterdir() if p.name.startswith("refused.sp")]
    assert leftovers == [], leftovers


def check_graft(directory, spine, expert, codec_name, codec):
    output = directory / f"out-{codec_name}-{expert.name}.sp"
    receipt = run_graft(spine, expert, output, codec_name)
    spine_pack, spine_regions = regions(spine)
    _, expert_regions = regions(expert)
    out_pack, out_regions = regions(output)
    assert [(e["kind"], e["layer"]) for e in out_pack.entries] == \
        [(e["kind"], e["layer"]) for e in spine_pack.entries]
    for key, (payload, scale, entry) in out_regions.items():
        if key[0] in graft.EXPERT_KINDS:
            source_payload, source_scale, source_entry = expert_regions[key]
        else:
            source_payload, source_scale, source_entry = spine_regions[key]
        assert payload == source_payload and scale == source_scale, key
        for field in graft.ENTRY_FIELDS[:8]:
            assert entry[field] == source_entry[field], (key, field)
        assert entry["payload_offset"] % pack.ALIGNMENT == 0
        assert entry["scale_offset"] % pack.ALIGNMENT == 0
    header = out_pack.header
    assert header["expert_codec"] == codec
    assert out_pack.revision == REVISION
    for field in graft.SPAN_FIELDS:
        assert header[field] == spine_pack.header[field], field
    raw = output.read_bytes()
    assert raw[161:pack.HEADER_BYTES] == spine.read_bytes()[161:pack.HEADER_BYTES]
    sha = hashlib.sha256(raw).hexdigest()
    assert Path(str(output) + ".sha256").read_text() == f"{sha}  {output.name}\n"
    stored = json.loads(Path(str(output) + ".receipt.json").read_text())
    assert stored == receipt and stored["output"]["sha256"] == sha
    assert stored["spine_digest"] == graft.pack_spine_digest(output) == graft.pack_spine_digest(spine)
    assert stored["expert_codec"] == codec_name and stored["tp_rank"] == 3
    assert stored["spine_source"]["header_expert_codec"] == spine_pack.header["expert_codec"]
    return output, stored


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        spine = directory / "spine-bf16.sp"
        fp8 = directory / "experts-fp8.sp"
        nvfp4 = directory / "experts-nvfp4.sp"
        write_pack(spine, entries_for(pack.CODEC_BF16, (3, 4)), 1)
        write_pack(fp8, entries_for(pack.CODEC_FP8, (3, 4)), 2)
        write_pack(nvfp4, entries_for(pack.CODEC_NVFP4, (3, 4)), 3, header_codec=pack.CODEC_FP8)

        f1, f1_receipt = check_graft(directory, spine, spine, "bf16", pack.CODEC_BF16)
        f2, f2_receipt = check_graft(directory, spine, fp8, "fp8", pack.CODEC_FP8)
        f3, f3_receipt = check_graft(directory, spine, nvfp4, "nvfp4", pack.CODEC_NVFP4)
        assert f1_receipt["spine_digest"] == f2_receipt["spine_digest"] == f3_receipt["spine_digest"]
        assert len({f1_receipt["expert_digest"], f2_receipt["expert_digest"],
                    f3_receipt["expert_digest"]}) == 3
        original, rewritten = spine.read_bytes(), f1.read_bytes()
        assert len(original) == len(rewritten)
        differing = [i for i in range(len(original)) if original[i] != rewritten[i]]
        assert differing and all(64 <= i < 68 or 96 <= i < 161 for i in differing), differing[:8]
        assert graft.pack_spine_digest(fp8) != f2_receipt["spine_digest"]

        other_spine = directory / "spine-other.sp"
        write_pack(other_spine, entries_for(pack.CODEC_BF16, (3, 4), spine_rows=5), 9)
        assert graft.pack_spine_digest(other_spine) != f1_receipt["spine_digest"]
        flipped = bytearray(spine.read_bytes())
        router = next(e for e in graft.RankPack(spine).entries if e["kind"] == pack.K_ROUTER)
        flipped[router["payload_offset"]] ^= 1
        flipped_path = directory / "spine-flipped.sp"
        flipped_path.write_bytes(bytes(flipped))
        assert graft.pack_spine_digest(flipped_path) != f1_receipt["spine_digest"]

        cases = [
            (dict(tp=(8, 3)), "tp8 rank 3 != requested tp16"),
            (dict(tp=(16, 4)), "rank 4 != requested"),
            (dict(span=(1, 0, 4, 41)), "header first_layer"),
            (dict(flags=1), "header flags"),
            (dict(hidden=2048), "model geometry"),
            (dict(linear=pack.CODEC_FP8), "linear/kv codec"),
        ]
        for index, (override, needle) in enumerate(cases):
            bad = directory / f"bad-{index}.sp"
            write_pack(bad, entries_for(pack.CODEC_FP8, (3, 4)), 4, **override)
            expect_refused(directory, spine, bad, "fp8", needle)
        expect_refused(directory, spine, fp8, "fp8", "rank 3 != requested tp16 rank 5", rank=5)
        expect_refused(directory, spine, fp8, "fp8", "requested tp8", degree=8)

        renamed = directory / "bad-names.sp"
        write_pack(renamed, entries_for(pack.CODEC_FP8, (3, 5)), 5)
        expect_refused(directory, spine, renamed, "fp8", "entry names differ")
        fewer = directory / "bad-fewer.sp"
        write_pack(fewer, entries_for(pack.CODEC_FP8, (3,)), 5)
        expect_refused(directory, spine, fewer, "fp8", "header tensor_count")
        reshaped = directory / "bad-shape.sp"
        write_pack(reshaped, entries_for(pack.CODEC_FP8, (3, 4), rows=8), 6)
        expect_refused(directory, spine, reshaped, "fp8", "rows 8 in the expert pack != 4")
        expect_refused(directory, spine, fp8, "nvfp4", "expert codec 5 != requested 6")
        expect_refused(directory, spine, nvfp4, "bf16", "expert codec 6 != requested 1")
        wrong_bytes = entries_for(pack.CODEC_FP8, (3, 4))
        wrong_bytes = [e[:9] + (e[9] + 256,) if e[0] == pack.K_EXPERT_DOWN else e for e in wrong_bytes]
        mis_sized = directory / "bad-bytes.sp"
        write_pack(mis_sized, wrong_bytes, 7)
        expect_refused(directory, spine, mis_sized, "fp8", "payload/scale bytes and encoding")
        truncated = directory / "bad-truncated.sp"
        truncated.write_bytes(fp8.read_bytes()[:-1])
        expect_refused(directory, spine, truncated, "fp8", "header file_bytes")
        no_experts = directory / "bad-no-experts.sp"
        write_pack(no_experts, entries_for(pack.CODEC_BF16, ()), 8)
        expect_refused(directory, no_experts, no_experts, "bf16", "no routed-expert entries")

        before = f2.read_bytes()
        try:
            run_graft(spine, fp8, f2, "fp8")
        except pack.PackFailure as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing output overwritten")
        assert f2.read_bytes() == before

        corrupt_output = directory / "corrupt.sp"
        real_copy = graft.Streamer.copy

        def corrupting_copy(self, source, role, offset, count, out):
            digest = real_copy(self, source, role, offset, count, out)
            if role == "expert":
                position = out.tell()
                out.seek(position - 1)
                last = out.read(1)
                out.seek(position - 1)
                out.write(bytes([last[0] ^ 1]))
            return digest
        with patch.object(graft.Streamer, "copy", corrupting_copy):
            try:
                run_graft(spine, fp8, corrupt_output, "fp8")
            except pack.PackFailure as error:
                assert "differs from its source region" in str(error), str(error)
            else:
                raise AssertionError("corrupted output published")
        assert [p.name for p in directory.iterdir() if p.name.startswith("corrupt.sp")] == []

        peaks = {}
        for rows in (16, 256):
            big_spine = directory / f"big-bf16-{rows}.sp"
            big_fp8 = directory / f"big-fp8-{rows}.sp"
            write_pack(big_spine, entries_for(pack.CODEC_BF16, (3,), groups=4, rows=rows,
                                              columns=1024), 10)
            write_pack(big_fp8, entries_for(pack.CODEC_FP8, (3,), groups=4, rows=rows,
                                            columns=1024), 11)
            tracemalloc.start()
            run_graft(big_spine, big_fp8, directory / f"big-out-{rows}.sp", "fp8", chunk=8192)
            peaks[rows] = (big_spine.stat().st_size, tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
        assert peaks[256][0] > 4 << 20 and peaks[256][0] > 15 * peaks[16][0]
        assert peaks[256][1] < 1 << 20, peaks
        assert abs(peaks[256][1] - peaks[16][1]) < 32 * 1024, peaks
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["graft", "--spine-pack", str(spine), "--expert-pack", str(nvfp4), "--output",
                str(directory / "cli.sp"), "--tp-degree", "16", "--tp-rank", "3",
                "--expert-codec", "fp8", "--model-revision", REVISION]
        with patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            assert graft.main() == 1
        assert "GRAFT-REFUSED" in stderr.getvalue()
        argv[argv.index("fp8")] = "nvfp4"
        with patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
            assert graft.main() == 0
        assert "GRAFT-PASS" in stdout.getvalue()
        with patch.object(sys, "argv", ["graft", "--spine-digest", str(directory / "cli.sp")]), \
                redirect_stdout(stdout):
            assert graft.main() == 0
        assert f3_receipt["spine_digest"] in stdout.getvalue()
        if hasattr(os, "posix_fadvise"):
            receipt = graft.graft(spine, fp8, directory / "dropped.sp", 16, 3, "fp8", REVISION, 4096,
                                  {"spine": True, "expert": True, "output": True})
            assert receipt["output"]["sha256"] == f2_receipt["output"]["sha256"]
        else:
            try:
                graft.graft(spine, fp8, directory / "dropped.sp", 16, 3, "fp8", REVISION, 4096,
                            {"output": True})
            except pack.PackFailure as error:
                assert "posix_fadvise" in str(error)
            else:
                raise AssertionError("cache drop silently skipped")
    print("PASS expert graft: spine bytes and digest preserved, expert planes byte-identical "
          "(bf16/fp8/nvfp4), header codec and revision rewritten, refusals on tp/rank/span/flags/"
          "geometry/names/shape/codec/bytes/existing output/read-back corruption, bounded memory")


if __name__ == "__main__":
    main()
