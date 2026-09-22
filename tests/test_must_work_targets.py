#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model_contracts" / "must_work_targets.json"
QWEN38_27B_SERVING_CONSTANTS = (
    ROOT / "model-families" / "qwen38_27b" / "include" / "sparkpipe"
    / "spark_qwen38_27b_serving_constants.h"
)
EXPECTED_FAMILIES = {
    "k3",
    "glm52",
    "qwen38_27b",
    "dsv4_flash",
    "dsv4_pro",
    "mimo26_pro",
    "mimo26_flash",
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
    # The Qwen 3.8 27B target pins the exact checkpoint the serving constants
    # compile against (the 3.6 line is deprecated; gate-breaking by directive).
    qwen = by_family["qwen38_27b"]
    assert qwen["non_expert_weight_format"] == "bf16"
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B"
    constants = QWEN38_27B_SERVING_CONSTANTS.read_text(encoding="utf-8")

    def constant(name: str) -> str:
        match = re.search(r'^#define ' + name + r' "([^"]+)"', constants, re.MULTILINE)
        assert match, name
        return match.group(1)

    assert qwen["model_id"] == constant("SPARK_QWEN38_27B_SERVING_MODEL_ID")
    assert qwen["model_revision"] == constant("SPARK_QWEN38_27B_SERVING_MODEL_REVISION")
    assert len(qwen["model_revision"]) == 40
    assert by_family["dsv4_flash"]["routed_expert_weight_codec"] == "mxfp4_e2m1"
    assert by_family["dsv4_pro"]["routed_expert_weight_codec"] == "mxfp4_e2m1"
    assert by_family["dsv4_flash"]["non_expert_weight_format"] == "fp8_e4m3_block_128x128"
    assert by_family["dsv4_pro"]["non_expert_weight_format"] == "fp8_e4m3_block_128x128"
    assert by_family["dsv4_flash"]["non_expert_activation_format"] == "bf16"
    assert by_family["mimo26_pro"]["routed_expert_weight_format"] == "mxfp4_e2m1_e8m0_block32"
    assert by_family["mimo26_flash"]["routed_expert_weight_format"] == "mxfp4_e2m1_e8m0_block32"
    assert by_family["mimo26_pro"]["non_expert_weight_format"] == (
        "fp8_e4m3_block_128x128_with_bf16_o_proj_embed_head"
    )
    assert by_family["mimo26_pro"]["contract"] == "model_contracts/mimo26_pro_authoritative.json"
    assert by_family["mimo26_flash"]["contract"] == "model_contracts/mimo26_flash_authoritative.json"
    print("PASS mandatory model target contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
