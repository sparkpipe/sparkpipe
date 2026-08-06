#!/usr/bin/env python3
"""Host-side contract tests for the DSV4 checkpoint stage packer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import dsv4_stagepack as pack  # noqa: E402


class MemorySource:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, _name: str) -> bytes:
        return self.data


def main() -> int:
    contract = json.loads(
        (ROOT / "model_contracts" / "dsv4_flash.json").read_text(encoding="utf-8")
    )
    ratios = contract["attention"]["compression_ratios"]

    assert pack.HEADER_STRUCT.size == 80
    assert pack.ENTRY_STRUCT.size == 40
    assert pack.FORMAT_VERSION == 3
    assert pack.WEIGHT_FP4 == 3
    assert pack.WEIGHT_FP8 == 4
    assert contract["dspark"]["layer_count"] == 3
    assert contract["model"]["mtp_layer_count"] == 0
    assert contract["runtime"]["packed_mtp_layer_count"] == 0

    full = pack.build_records(contract, 0, 43)
    expected = sum(
        24 + (4 if ratios[layer] != 0 else 0) + (6 if ratios[layer] == 4 else 0)
        for layer in range(43)
    ) + 1 + 5
    assert len(full) == expected

    final_stage = pack.build_records(contract, 41, 2)
    assert not any(record.kind == pack.KIND_EMBEDDING for record in final_stage)
    assert len(final_stage) == 67
    assert all(record.layer != pack.MTP_LAYER for record in final_stage)
    assert all("mtp." not in name for record in final_stage
               for name in record.source_names + record.scale_names)
    assert sum(record.kind == pack.KIND_GATE_BIAS for record in final_stage) == 2

    fp8_scales = pack.expand_fp8_scale(
        MemorySource(bytes(range(4))), "scale", 256, 256
    )
    assert len(fp8_scales) == 256 * 2
    assert fp8_scales[:2] == bytes((0, 1))
    assert fp8_scales[256:258] == bytes((2, 3))

    expert = next(record for record in full if record.kind == pack.KIND_EXPERTS_W1)
    assert expert.weight_format == pack.WEIGHT_FP4
    assert expert.rows == 256 * 2048
    assert expert.columns == 4096
    assert expert.payload_bytes == 1 << 30
    assert expert.scale_bytes == 256 * 2048 * (4096 // 32)

    entries, file_bytes = pack.make_directory(final_stage)
    assert file_bytes == 8219895692
    assert entries[0].payload_offset == pack.HEADER_STRUCT.size + pack.ENTRY_STRUCT.size * len(entries)
    assert entries[-1].payload_offset + entries[-1].record.payload_bytes + entries[-1].record.scale_bytes == file_bytes
    for previous, current in zip(entries, entries[1:]):
        assert previous.payload_offset + previous.record.payload_bytes + previous.record.scale_bytes == current.payload_offset

    codec_names = (
        contract["precision"]["non_expert_linear_weight_codec"],
        contract["precision"]["routed_expert_weight_codec"],
        contract["precision"]["kv_cache_codec"],
    )
    codecs = tuple(pack.CODEC_IDS[name] for name in codec_names)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "stage.spstage"
        source_root = root / "source"
        source_root.mkdir()
        index_bytes = b'{"weight_map":{}}\n'
        (source_root / pack.SOURCE_INDEX_NAME).write_bytes(index_bytes)
        identity_contract = {
            "source_index_sha256": hashlib.sha256(index_bytes).hexdigest()
        }
        assert pack.validate_source_identity(source_root, identity_contract) == (
            identity_contract["source_index_sha256"])
        try:
            pack.validate_source_identity(
                source_root, {"source_index_sha256": "0" * 64})
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("wrong checkpoint index hash was accepted")
        fragment_bytes = b"stage fragment"
        fragment_name = "model-00001-of-00001.safetensors"
        (source_root / fragment_name).write_bytes(fragment_bytes)
        (source_root / pack.SOURCE_INDEX_NAME).write_bytes(
            b'{"weight_map":{"x":"model-00001-of-00001.safetensors"}}\n')
        (source_root / pack.STAGE_SOURCE_MANIFEST_NAME).write_text(json.dumps({
            "source_index_sha256": identity_contract["source_index_sha256"],
            "fragment": fragment_name,
            "fragment_sha256": hashlib.sha256(fragment_bytes).hexdigest(),
        }), encoding="utf-8")
        assert pack.validate_source_identity(source_root, identity_contract) == (
            identity_contract["source_index_sha256"])
        (source_root / fragment_name).write_bytes(b"corrupt")
        try:
            pack.validate_source_identity(source_root, identity_contract)
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("corrupt stage fragment was accepted")
        with path.open("wb") as file:
            file.write(pack.pack_header(
                final_stage, 41, 2, file_bytes, codecs))
            for entry in entries:
                file.write(pack.pack_entry(entry))
            file.truncate(file_bytes)
        result = pack.verify_pack(path, contract, codecs, False)
        assert result["validated"] is True
        assert result["first_layer"] == 41
        assert result["layer_count"] == 2
        assert pack.HEADER_STRUCT.unpack(path.read_bytes()[:pack.HEADER_STRUCT.size])[15] == 0
        assert result["expert_weight_codec_id"] == pack.CODEC_IDS["mxfp4_e2m1"]
        assert pack.main(["--verify-pack", str(path)]) == 0
        with path.open("r+b") as file:
            file.seek(pack.HEADER_STRUCT.size + 20)
            file.write(b"\1")
        try:
            pack.verify_pack(path, contract, codecs, False)
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("corrupt stage-pack directory was accepted")

    print("PASS DSV4 stagepack source and wire contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
