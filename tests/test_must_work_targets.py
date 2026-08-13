#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model_contracts" / "must_work_targets.json"
EXPECTED_FAMILIES = {
    "k3",
    "glm52",
    "qwen36",
    "dsv4_flash",
    "dsv4_pro",
}


def main() -> int:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["cuda_target"] == "sm_121a"
    targets = document["targets"]
    assert len(targets) == len(EXPECTED_FAMILIES)
    families = {target["model_family"] for target in targets}
    assert families == EXPECTED_FAMILIES
    identifiers = [target["id"] for target in targets]
    assert len(identifiers) == len(set(identifiers))
    for target in targets:
        assert target["production_ready"] is False
        assert target["required_features"]
        contract = ROOT / target["contract"]
        assert contract.exists(), contract
        assert target["accumulator_format"] == "fp32"

    by_family = {target["model_family"]: target for target in targets}
    assert by_family["k3"]["routed_expert_weight_format"] == "mxfp4_e2m1"
    assert by_family["k3"]["routed_expert_activation_format"] == "bf16"
    assert by_family["glm52"]["routed_expert_weight_formats"] == [
        "int6_block_f32",
        "int7_block_f32",
        "int8_block_f32",
        "fp8_e4m3_block_f32",
        "nvfp4_e2m1_ue4m3_global_f32",
        "mxfp4_e2m1_e8m0",
    ]
    assert by_family["glm52"]["non_expert_weight_format"] == "bf16"
    assert by_family["qwen36"]["non_expert_weight_format"] == "bf16"
    assert by_family["dsv4_flash"]["routed_expert_weight_codec"] == "mxfp4_e2m1"
    assert by_family["dsv4_pro"]["routed_expert_weight_codec"] == "mxfp4_e2m1"
    assert by_family["dsv4_flash"]["non_expert_weight_format"] == "fp8_e4m3_block_128x128"
    assert by_family["dsv4_pro"]["non_expert_weight_format"] == "fp8_e4m3_block_128x128"
    assert by_family["dsv4_flash"]["non_expert_activation_format"] == "bf16"
    print("PASS mandatory model target contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
