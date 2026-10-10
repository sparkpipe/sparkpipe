#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model_contracts" / "must_work_targets.json"
EXPECTED_FAMILIES = {
    "k3",
    "glm52",
    "glm5_next",
    "qwen38_27b",
    "dsv41_flash",
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
    glm53_flash = by_family["glm5_next"]
    glm53_contract = json.loads((ROOT / glm53_flash["contract"]).read_text(encoding="utf-8"))
    assert glm53_flash["model_id"] == glm53_contract["model_id"]
    assert glm53_flash["model_revision"] == glm53_contract["source_revision"]
    assert glm53_contract["precision"]["weight_format"] == "fp8_e4m3"
    assert glm53_contract["precision"]["weight_block_size"] == [128, 128]
    assert glm53_flash["routed_expert_weight_format"] == "fp8_e4m3_block_128x128"
    qwen = by_family["qwen38_27b"]
    qwen_contract = json.loads((ROOT / qwen["contract"]).read_text(encoding="utf-8"))
    assert qwen["non_expert_weight_format"] == "bf16"
    assert qwen["model_id"] == "Qwen/Qwen3.8-27B" == qwen_contract["model_id"]
    assert len(qwen["model_revision"]) == 40
    dsv41 = by_family["dsv41_flash"]
    dsv41_contract = json.loads((ROOT / dsv41["contract"]).read_text(encoding="utf-8"))
    assert dsv41["model_id"] == dsv41_contract["model_id"]
    assert dsv41["model_revision"] == dsv41_contract["source_revision"]
    assert dsv41["routed_expert_weight_codec"] == "mxfp4_e2m1"
    assert dsv41["non_expert_weight_format"] == "fp8_e4m3_block_32x32_ue8m0"
    assert dsv41["non_expert_activation_format"] == "bf16"
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
