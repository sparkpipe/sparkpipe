"""The TP8xPP2 mesh cutter must reproduce the module's stage semantics from
the placed TP8 rank packs.

These checks drive tools/dsv41_flash_tp8pp2_stagepacks.py against a
synthetic full-model TP8 pack (the placed dsv41flash.mxfp4.tp8 wire form):
the cut is fail-closed on a source sha mismatch, stage 0 owns the embedding
global plus layers [0,20), stage 1 owns the final-norm/lm-head globals plus
layers [20,40), every payload/scale region is byte-identical to its source
region (offsets move, bytes never do), the header declares the (8,2) mesh
with the 20/20 window and inherits revision + digests + tp identity from
the source, and the verify mode re-proves an already-cut pack against its
placed sha sidecar.
"""

import hashlib
import importlib.util
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from dsv41_flash_tp8pp2_stagepacks import (  # noqa: E402
    ENTRY_BYTES, ENTRY_STRUCT, Fail, GLOBAL_LAYER, HEADER_BYTES, HEADER_STRUCT,
    K_EMBEDDING, K_FINAL_NORM, K_LM_HEAD, LAYER_COUNT, cut, read_header,
    stage_entries, verify,
)

MAGIC = 0x31413444
DIRECTORY_OFFSET = 512


def write_source_pack(root: Path, tp_rank: int = 3) -> tuple[Path, str]:
    entries = []
    for kind, rows in ((K_EMBEDDING, 65536), (K_FINAL_NORM, 5120), (K_LM_HEAD, 32320)):
        entries.append(dict(kind=kind, layer=GLOBAL_LAYER, payload_type=1,
                            codec=1, scale_encoding=0, groups=0, rows=rows, cols=0,
                            payload_bytes=rows * 2, scale_bytes=0))
    for layer in range(LAYER_COUNT):
        entries.append(dict(kind=3, layer=layer, payload_type=1, codec=1,
                            scale_encoding=0, groups=0, rows=5120, cols=0,
                            payload_bytes=5120 * 2, scale_bytes=0))
        entries.append(dict(kind=29, layer=layer, payload_type=4, codec=2,
                            scale_encoding=1, groups=48, rows=1152, cols=576,
                            payload_bytes=48 * 4096, scale_bytes=48 * 128))
    payload_base = (DIRECTORY_OFFSET + len(entries) * ENTRY_BYTES + 255) & ~255
    cursor = payload_base
    laid = []
    for entry in entries:
        payload_offset = (cursor + 255) & ~255
        cursor = payload_offset + entry["payload_bytes"]
        scale_offset = 0
        if entry["scale_bytes"]:
            scale_offset = (cursor + 255) & ~255
            cursor = scale_offset + entry["scale_bytes"]
        laid.append(dict(entry, payload_offset=payload_offset, scale_offset=scale_offset))
    file_bytes = cursor
    image = bytearray(file_bytes)
    header = bytearray(HEADER_BYTES)
    HEADER_STRUCT.pack_into(header, 0, MAGIC, 1, HEADER_BYTES, ENTRY_STRUCT.size, 1, 0,
                            len(laid), 1, 0, 0, LAYER_COUNT, LAYER_COUNT, 5120, 129280,
                            384, 1, 2, 8, tp_rank, 0, DIRECTORY_OFFSET, file_bytes)
    header[96:96 + 6] = b"r2rev@"
    header[161:193] = bytes(range(32))
    header[193:225] = bytes(range(32, 64))
    header[225:257] = bytes(range(64, 96))
    image[0:HEADER_BYTES] = header
    for index, entry in enumerate(laid):
        image[DIRECTORY_OFFSET + index * ENTRY_STRUCT.size:DIRECTORY_OFFSET + (index + 1) * ENTRY_STRUCT.size] = \
            ENTRY_STRUCT.pack(
                entry["kind"], entry["layer"], entry["payload_type"], entry["codec"],
                entry["scale_encoding"], entry["groups"], entry["rows"], entry["cols"],
                entry["payload_offset"], entry["payload_bytes"],
                entry["scale_offset"], entry["scale_bytes"])
        for plane in ("payload", "scale"):
            length = entry[f"{plane}_bytes"]
            if length:
                image[entry[f"{plane}_offset"]:entry[f"{plane}_offset"] + length] = \
                    bytes([0x40 + index]) * length
    pack_path = root / "rank3.spstage"
    pack_path.write_bytes(image)
    digest = hashlib.sha256(image).hexdigest()
    (root / "rank3.spstage.sha256").write_text(f"{digest}  rank3.spstage\n")
    return pack_path, digest


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "dsv41_flash_tp8pp2_stagepacks",
        ROOT / "tools" / "dsv41_flash_tp8pp2_stagepacks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_fail(action, description: str) -> None:
    try:
        action()
    except Fail:
        return
    raise SystemExit(f"{description} must fail closed")


def main() -> int:
    load_tool()
    checks = 0
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source_pack, source_sha = write_source_pack(root)
        header, entries = read_header(str(source_pack))
        assert header["tensor_count"] == len(entries) == 83, "synthetic pack census"
        stage0 = stage_entries(entries, 0)
        stage1 = stage_entries(entries, 1)
        assert len(stage0) == 41 and stage0[0]["kind"] == K_EMBEDDING, "stage 0 selector"
        assert all(0 <= e["layer"] < 20 for e in stage0 if e["layer"] != GLOBAL_LAYER), \
            "stage 0 window"
        assert len(stage1) == 42, "stage 1 census"
        assert {e["kind"] for e in stage1 if e["layer"] == GLOBAL_LAYER} == \
            {K_FINAL_NORM, K_LM_HEAD}, "stage 1 globals"
        assert all(20 <= e["layer"] < 40 for e in stage1 if e["layer"] != GLOBAL_LAYER), \
            "stage 1 window"
        checks += 1

        receipt0 = cut(str(source_pack), source_sha, str(root), 0)
        receipt1 = cut(str(source_pack), source_sha, str(root), 1)
        assert receipt0["tensor_count"] == 41 and receipt1["tensor_count"] == 42, \
            "cut census"
        assert receipt0["world_rank"] == 3 and receipt1["world_rank"] == 11, "world rank"
        checks += 1

        source_by_key = {(e["kind"], e["layer"]): e for e in entries}
        raw_source = source_pack.read_bytes()
        for stage_index in (0, 1):
            stage_pack = root / f"stage{stage_index}.rank3.spstage"
            stage_header, stage_entries_list = read_header(str(stage_pack), full_model=False)
            assert stage_header["stage_count"] == 2, "stage count on the wire"
            assert stage_header["revision"] == "r2rev@", "revision inherited"
            assert stage_header["tp_degree"] == 8 and stage_header["tp_rank"] == 3, \
                "tp identity inherited"
            raw_stage = stage_pack.read_bytes()
            for entry in stage_entries_list:
                origin = source_by_key[(entry["kind"], entry["layer"])]
                for plane in ("payload", "scale"):
                    length = entry[f"{plane}_bytes"]
                    if length:
                        assert raw_stage[entry[f"{plane}_offset"]:entry[f"{plane}_offset"] + length] == \
                            raw_source[origin[f"{plane}_offset"]:origin[f"{plane}_offset"] + length], \
                            f"region identity kind={entry['kind']} layer={entry['layer']}"
        checks += 1

        verify(str(source_pack), source_sha, str(root / "stage0.rank3.spstage"))
        verify(str(source_pack), source_sha, str(root / "stage1.rank3.spstage"))
        checks += 1

        expect_fail(lambda: cut(str(source_pack), source_sha, str(root), 0),
                    "re-cut over an existing pack")
        other = root / "other"
        other.mkdir()
        corrupt_pack, _ = write_source_pack(other, tp_rank=5)
        (other / "rank3.spstage.sha256").write_text("0" * 64 + "  rank3.spstage\n")
        expect_fail(lambda: cut(str(corrupt_pack), "0" * 64, str(other), 0),
                    "cut on a sha-mismatched source")
        checks += 1
    print(f"PASS dsv41 tp8pp2 cutter: {checks} check groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
