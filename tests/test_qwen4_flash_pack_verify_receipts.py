"""qwen4_flash_pack_verify: the receipt pair and the structure-only form.

The fleet receipt convention is <pack>.receipt.json + <pack>.sha256; the
verifier emits it on PASS (--emit-receipt) but never overwrites a
packer-written receipt, and --checkpoint is now optional so placed packs
can be verified structure-only on nodes without the warm source.
"""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import qwen4_flash_pack_verify as V

TOOL = ROOT / "tools" / "qwen4_flash_pack_verify.py"


HEADER_FIELDS = ("magic", "format_version", "header_bytes", "directory_entry_bytes",
                 "tensor_count", "hidden_dimension", "layer_count", "first_layer_index",
                 "total_layer_count", "attention_period", "full_attention_phase",
                 "gdn_key_head_count", "gdn_value_head_count", "gdn_head_key_dimension",
                 "gdn_head_value_dimension", "gdn_conv_kernel", "attn_query_head_count",
                 "attn_kv_head_count", "attn_head_dimension", "attn_rope_dimension",
                 "routed_expert_count", "experts_per_token", "expert_intermediate_dimension",
                 "output_vocab_count", "mxfp4_group_size", "mtp_layer_count",
                 "directory_offset", "file_bytes")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_tool(pack: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, str(TOOL), "--pack", str(pack), *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_emit_receipt_writes_the_fleet_pair():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        pack.write_bytes(b"x" * 4096)
        header = {"first_layer_index": 0, "layer_count": 48,
                  "tensor_count": 1246}
        written = V.emit_receipt(pack, header, 8, 3, True)
        require(written is not None, "emit_receipt wrote nothing")
        receipt_path, sidecar = written
        receipt = json.loads(receipt_path.read_text())
        digest = hashlib.sha256(pack.read_bytes()).hexdigest()
        require(receipt["output_sha256"] == digest == receipt["sha256"], receipt)
        require(receipt["verify_mode"].startswith("structure-only"), receipt["verify_mode"])
        require(receipt["tp_degree"] == 8 and receipt["tp_rank"] == 3, receipt)
        require(sidecar.read_text() == f"{digest}  {pack.name}\n", sidecar.read_text())


def test_emit_receipt_never_overwrites_a_packer_receipt():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        pack.write_bytes(b"y" * 128)
        packer_receipt = Path(str(pack) + ".receipt.json")
        packer_receipt.write_text(json.dumps({"kind": "packer", "output_sha256": "0"}))
        header = {"first_layer_index": 0, "layer_count": 48, "tensor_count": 1}
        require(V.emit_receipt(pack, header, 8, 3, True) is None,
                "emit_receipt must not replace a packer receipt")
        require(json.loads(packer_receipt.read_text())["kind"] == "packer",
                packer_receipt.read_text())
        require(not Path(str(pack) + ".sha256").exists(),
                "no digest sidecar is written beside a packer receipt")


def test_structure_only_mode_runs_without_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "short.pack"
        pack.write_bytes(b"\0" * 8)
        code, output = run_tool(pack)
        require(code == 1, output)
        require("pack header truncated: 8 bytes" in output, output)
        require("--checkpoint" not in output and "required" not in output, output)


def test_header_format_version_gate_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        from qwen4_flash_stagepack import FORMAT_VERSION, HEADER_BYTES, HEADER_STRUCT
        fields = V.expected_header_geometry(0, 48, 0)
        fields["format_version"] = FORMAT_VERSION + 1
        fields["directory_offset"] = HEADER_BYTES
        fields["file_bytes"] = HEADER_BYTES
        pack.write_bytes(HEADER_STRUCT.pack(*[fields[name] for name in HEADER_FIELDS]))
        code, output = run_tool(pack)
        require(code == 1, output)
        require(output.splitlines() ==
                [f"FAIL header format_version={FORMAT_VERSION + 1} expected {FORMAT_VERSION}"],
                f"only the format version may fail: {output}")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
