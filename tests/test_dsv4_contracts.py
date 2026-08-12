#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run(
        ["python3", str(ROOT / "tools" / "generate_dsv4_contracts.py"), "--check"],
        check=True,
        cwd=ROOT,
    )
    flash = json.loads((ROOT / "model_contracts" / "dsv4_flash.json").read_text(encoding="utf-8"))
    pro = json.loads((ROOT / "model_contracts" / "dsv4_pro.json").read_text(encoding="utf-8"))
    assert flash["model"]["layer_count"] == 43
    assert pro["model"]["layer_count"] == 61
    assert len(flash["attention"]["compression_ratios"]) == 46
    assert len(pro["attention"]["compression_ratios"]) == 62
    assert flash["attention"]["compression_ratios"][:2] == [0, 0]
    assert pro["attention"]["compression_ratios"][:2] == [128, 128]
    assert flash["attention"]["compression_ratios"][-3:] == [0, 0, 0]
    assert flash["model"]["mtp_layer_count"] == 0
    assert flash["dspark"]["layer_count"] == 3
    assert flash["runtime"]["packed_mtp_layer_count"] == 0
    assert flash["source_index_sha256"] == (
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b")
    source_files = flash["source_files"]
    shard_names = [
        f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
    ]
    assert sorted(source_files) == sorted(
        ["model.safetensors.index.json"] + shard_names)
    assert len({source_files[name]["sha256"] for name in shard_names}) == 48
    assert all(len(source_files[name]["sha256"]) == 64 for name in source_files)
    assert sum(source_files[name]["bytes"] for name in shard_names) == 166886535336
    assert flash["source_indexed_payload_bytes"] == 166878536440
    assert flash["source_shard_count"] == 48
    header = (ROOT / "model-families" / "dsv4" / "include" / "sparkpipe" /
              "spark_dsv4_model.h").read_text(encoding="utf-8")
    assert "SparkDsv4ModelCompressionRatios[43u]" in header
    description = json.loads(
        (ROOT / "examples" / "model_descriptions" /
         "dsv4_resident_decode_stage_firmware.json").read_text(encoding="utf-8")
    )
    scheduling = description["stages"][0]["programs"][0]["scheduling"]
    assert scheduling["max_active_slots"] == 1024
    assert scheduling["max_resident_sequences"] == 16384
    b1_description = json.loads(
        (ROOT / "examples" / "model_descriptions" /
         "dsv4_resident_decode_stage_firmware_b1.json").read_text(
             encoding="utf-8")
    )
    b1_program = b1_description["stages"][0]["programs"][0]
    assert b1_program["max_inflight"] == 1
    assert b1_program["operations"][0]["module"].endswith(".b1.v3")
    assert b1_program["scheduling"]["max_active_slots"] == 1
    assert b1_program["scheduling"]["max_resident_sequences"] == 1
    deployment = json.loads(
        (ROOT / "examples" / "deployments" /
         "dsv4_flash_pp13_host_rdma.spec.json").read_text(encoding="utf-8")
    )
    limits = deployment["runtime_limits"]
    assert scheduling["max_active_slots"] >= limits["max_active_sequences"]
    assert scheduling["max_resident_sequences"] >= limits["resident_sequence_capacity"]
    assert pro["attention"]["compression_ratios"][-1] == 0
    assert flash["qualification"]["cuda_target"] == "sm_121a"
    assert pro["qualification"]["cuda_target"] == "sm_121a"
    print("PASS DSV4 Flash and Pro generated contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
