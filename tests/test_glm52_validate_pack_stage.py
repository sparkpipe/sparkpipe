#!/usr/bin/env python3
"""glm52_validate_pack --stage: per-stage verification keyed on the header.

The full 78-layer inventory mode fails a per-stage tp4pp4 rank pack by
construction (the other stages' layers are absent); the stage mode reads
first_layer/layer_count from the stage header and inventories only that
span, checks the extent, and exits nonzero on any error.
"""

import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "glm52_validate_pack", str(ROOT / "tools" / "glm52_validate_pack.py"))
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

TOOL = ROOT / "tools" / "glm52_validate_pack.py"
MAGIC = 0x32534C47
HEADER_BYTES = 264
ENTRY_BYTES = 64
GLOBAL_LAYER = 0xFFFFFFFF
FIRST_LAYER, LAYER_COUNT = 40, 19
TP_DEGREE = 4


def expected_entries():
    entries = []
    for kind in (V.K_EMBEDDING, V.K_FINAL_NORM, V.K_LM_HEAD):
        es, _err = V.expected_shape(kind, GLOBAL_LAYER, TP_DEGREE, V.BF16)
        pt, wc, se, g, rows, cols = es
        entries.append(dict(kind=kind, layer=GLOBAL_LAYER, pt=pt, wc=wc, se=se,
                            g=g, rows=rows, cols=cols))
    for layer in range(FIRST_LAYER, FIRST_LAYER + LAYER_COUNT):
        for kind in range(3, V.KIND_COUNT):
            es, err = V.expected_shape(kind, layer, TP_DEGREE, V.BF16)
            if es is None:
                continue
            entries.append(dict(kind=kind, layer=layer, pt=es[0], wc=es[1],
                                se=es[2], g=es[3], rows=es[4], cols=es[5]))
    return entries


def build_stage_pack(path: Path, corrupt: bool = False) -> int:
    entries = expected_entries()
    dir_off = 512
    cursor = 0
    directory = b""
    for entry in entries:
        payload_bytes = V.expected_payload_bytes(
            (entry["pt"], entry["wc"], entry["se"], entry["g"], entry["rows"],
             entry["cols"]))
        scale_bytes = V.expected_scale_bytes(
            (entry["pt"], entry["wc"], entry["se"], entry["g"], entry["rows"],
             entry["cols"]))
        cursor = (cursor + 255) & ~255
        poff = cursor + dir_off + len(entries) * ENTRY_BYTES
        cursor += payload_bytes
        soff = 0
        if scale_bytes:
            cursor = (cursor + 255) & ~255
            soff = cursor + dir_off + len(entries) * ENTRY_BYTES
            cursor += scale_bytes
        directory += struct.pack(
            "<8I4Q", entry["kind"],
            entry["layer"] if entry["layer"] != GLOBAL_LAYER else GLOBAL_LAYER,
            entry["pt"], entry["wc"], entry["se"], entry["g"], entry["rows"],
            entry["cols"], poff, payload_bytes, soff, scale_bytes)
    file_bytes = dir_off + len(entries) * ENTRY_BYTES + cursor
    header = struct.pack(
        "<20I2Q", MAGIC, 3, HEADER_BYTES, ENTRY_BYTES, 1, 0, len(entries),
        1, 0, FIRST_LAYER, LAYER_COUNT, 78, 6144, 154880, 256, 1, 1, 1,
        TP_DEGREE, 3, dir_off, file_bytes)
    revision = b"\0" * 65
    if corrupt:
        directory = directory[:40] + struct.pack("<I", 7) + directory[44:]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(file_bytes)
        handle.seek(0)
        handle.write(header + revision
                     + b"\0" * (dir_off - 96 - 65) + directory)
    return file_bytes


def run_tool(pack: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, str(TOOL), str(pack), str(TP_DEGREE), *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_stage_mode_passes_on_per_stage_pack():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.glm52sp"
        build_stage_pack(pack)
        code, output = run_tool(pack, "--stage")
        assert code == 0, output
        assert "errors: 0" in output
        assert "layers=[40,59)" in output


def test_full_mode_fails_the_same_pack():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.glm52sp"
        build_stage_pack(pack)
        code, output = run_tool(pack)
        assert code == 1, output
        assert "layer 0 inventory" in output


def test_corrupt_entry_fails_stage_mode():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.glm52sp"
        build_stage_pack(pack, corrupt=True)
        code, output = run_tool(pack, "--stage")
        assert code == 1, output
        assert "entry 0: payload_bytes 7 != expected" in output and "errors: 1" in output, output


def test_out_of_span_layer_fails_stage_mode():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.glm52sp"
        build_stage_pack(pack)
        code, output = run_tool(pack, "--stage")
        assert code == 0
        with pack.open("r+b") as handle:
            handle.seek(512 + 3 * ENTRY_BYTES + 4)
            handle.write(struct.pack("<I", 5))
        code, output = run_tool(pack, "--stage")
        assert code == 1, output
        assert "entry 3 kind ATTN_NORM layer 5: outside the stage span [40,59)" in output, output


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
