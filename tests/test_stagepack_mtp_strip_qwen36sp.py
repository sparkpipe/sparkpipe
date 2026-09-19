"""The qwen36sp strip family map must match the qwen38_27b packer header.

The 2026-09-05 16-wide strip of qwen27b.tp4 copied the qwen4_flash header
indices verbatim and zeroed u32[25] - the 27b packer's tp_rank field -
on every rank != 0 pack, while leaving the real mtp_layer_count field at
u32[23] untouched. These checks pin the strip tool's family entry to the
field order the packer actually writes, and exercise a full strip on a
synthetic pack: the MTP tail is removed, tensor_count and mtp_layer_count
are patched, and tp_degree/tp_rank survive byte-identical.
"""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from qwen38_27b_stagepack import (  # noqa: E402
    HEADER_BYTES, HEADER_STRUCT, LAYER_COUNT, MTP_LAYERS,
)

TOOL = ROOT / "tools" / "stagepack_mtp_strip.py"


def load_strip_tool():
    spec = importlib.util.spec_from_file_location(
        "stagepack_mtp_strip", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADER_FIELDS = (
    "magic", "format_version", "header_bytes", "entry_bytes", "tensor_count",
    "hidden_dimension", "layer_count", "first_layer_index", "total_layer_count",
    "attention_period", "full_attention_phase",
    "gdn_key_head_count", "gdn_value_head_count",
    "gdn_head_key_dimension", "gdn_head_value_dimension", "gdn_conv_kernel",
    "attn_query_head_count", "attn_kv_head_count", "attn_head_dimension",
    "attn_rope_dimension", "ffn_intermediate_dimension", "output_vocab_count",
    "mxfp4_group_size", "mtp_layer_count", "tp_degree", "tp_rank",
)
MTP_INDEX = HEADER_FIELDS.index("mtp_layer_count")
TP_DEGREE_INDEX = HEADER_FIELDS.index("tp_degree")
TP_RANK_INDEX = HEADER_FIELDS.index("tp_rank")
COUNT_INDEX = HEADER_FIELDS.index("tensor_count")

ENTRY_BYTES = 56
DIRECTORY_OFFSET = HEADER_BYTES
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE


def synthetic_pack(directory: Path) -> Path:
    header_values = {
        "magic": 0x50533651, "format_version": 3,
        "header_bytes": HEADER_BYTES, "entry_bytes": ENTRY_BYTES,
        "tensor_count": 2, "hidden_dimension": 5120,
        "layer_count": 4, "first_layer_index": 0,
        "total_layer_count": LAYER_COUNT, "attention_period": 4,
        "full_attention_phase": 3, "gdn_key_head_count": 16,
        "gdn_value_head_count": 48, "gdn_head_key_dimension": 128,
        "gdn_head_value_dimension": 128, "gdn_conv_kernel": 4,
        "attn_query_head_count": 24, "attn_kv_head_count": 4,
        "attn_head_dimension": 256, "attn_rope_dimension": 64,
        "ffn_intermediate_dimension": 17408,
        "output_vocab_count": 248320, "mxfp4_group_size": 32,
        "mtp_layer_count": MTP_LAYERS, "tp_degree": 4, "tp_rank": 3,
    }
    assert set(header_values) == set(HEADER_FIELDS)
    payload_offset = 256
    entries = [
        (5, 7, 0, 1, 64, payload_offset, 128, 0, 0),
        (23, MTP_LAYER, 0, 1, 64, payload_offset + 128, 64, 0, 0),
    ]
    body = HEADER_STRUCT.pack(
        *(header_values[name] for name in HEADER_FIELDS),
        DIRECTORY_OFFSET, payload_offset + 128 + 64)
    body += b"".join(struct.pack("<6I", kind, layer, fmt, rows, cols, 0)
                     + struct.pack("<4Q", poff, pbytes, soff, sbytes)
                     for kind, layer, fmt, rows, cols, poff, pbytes, soff, sbytes
                     in entries)
    body += b"\0" * (payload_offset - len(body))
    body += b"\xa5" * 192
    pack = directory / "qwen27b.fp8.tp4.rank3.qwen36sp"
    pack.write_bytes(body)
    return pack


def test_family_map_matches_packer_header():
    strip = load_strip_tool()
    family = strip.FAMILIES["qwen36sp"]
    assert family["magic"] == 0x50533651
    assert family["header_bytes"] == HEADER_BYTES == 120
    assert family["entry_bytes"] == ENTRY_BYTES == 56
    assert family["count_index"] == COUNT_INDEX == 4
    assert family["mtp_index"] == MTP_INDEX == 23
    assert family["mtp_index"] != TP_RANK_INDEX


def test_strip_removes_mtp_tail_and_preserves_tp_fields():
    strip = load_strip_tool()
    with tempfile.TemporaryDirectory() as tmp:
        pack = synthetic_pack(Path(tmp))
        before = HEADER_STRUCT.unpack(pack.read_bytes()[:HEADER_BYTES])
        assert before[MTP_INDEX] == MTP_LAYERS
        assert before[TP_RANK_INDEX] == 3
        run = subprocess.run(
            [sys.executable, str(TOOL), "--pack", str(pack),
             "--family", "qwen36sp"],
            capture_output=True, text=True)
        assert run.returncode in (0, 3), run.stderr
        raw = pack.read_bytes()
        after = HEADER_STRUCT.unpack(raw[:HEADER_BYTES])
        assert after[COUNT_INDEX] == 1
        assert after[MTP_INDEX] == 0
        assert after[TP_DEGREE_INDEX] == 4
        assert after[TP_RANK_INDEX] == 3
        assert after[-1] == len(raw)
        assert len(raw) == 256 + 128
        kept_entry = raw[HEADER_BYTES:HEADER_BYTES + ENTRY_BYTES]
        kind, layer = struct.unpack_from("<2I", kept_entry, 0)
        assert (kind, layer) == (5, 7)
        receipt = json.loads(
            Path(str(pack) + ".receipt.json").read_text())
        assert receipt["mtp"] == "stripped"
        assert receipt["mtp_entries_dropped"] == 1
        assert receipt["file_bytes"] == len(raw)
        assert receipt["output_sha256"] == strip.sha256_chunked(pack)


def main() -> int:
    test_family_map_matches_packer_header()
    test_strip_removes_mtp_tail_and_preserves_tp_fields()
    print("test_stagepack_mtp_strip_qwen36sp: 2 checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
