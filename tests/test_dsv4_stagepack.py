#!/usr/bin/env python3
"""Host-side contract tests for the DSV4 checkpoint stage packer."""

from __future__ import annotations

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
    assert pack.WEIGHT_FP4 == 3
    assert pack.WEIGHT_FP8 == 4
    assert pack.layer_kind(ratios, pack.MTP_LAYER) == 0

    full = pack.build_records(contract, 0, 43)
    expected = sum(
        24 + (4 if ratios[layer] != 0 else 0) + (6 if ratios[layer] == 4 else 0)
        for layer in range(43)
    ) + 1 + 13 + 24
    assert len(full) == expected

    final_stage = pack.build_records(contract, 40, 3)
    assert any(record.kind == pack.KIND_EMBEDDING for record in final_stage)
    assert final_stage[-1].layer == pack.MTP_LAYER
    assert sum(record.kind == pack.KIND_GATE_BIAS for record in final_stage) == 4

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
        path = Path(temporary) / "stage.spstage"
        with path.open("wb") as file:
            file.write(pack.pack_header(
                final_stage, 40, 3, file_bytes, codecs))
            for entry in entries:
                file.write(pack.pack_entry(entry))
            file.truncate(file_bytes)
        result = pack.verify_pack(path, contract, codecs, False)
        assert result["validated"] is True
        assert result["first_layer"] == 40
        assert result["layer_count"] == 3
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
