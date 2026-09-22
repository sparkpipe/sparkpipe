#!/usr/bin/env python3
"""Bind the mimo26 family facts to the measured census (header == census).

The census JSONs under model-families/mimo26/census/ are the measured ground
truth (shard-header walk of the warm checkpoints); this test fails if the
geometry headers, contracts or must-work registration drift from them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "model-families" / "mimo26"
CENSUS = FAMILY / "census"


def census(tag):
    return json.loads((CENSUS / f"{tag}_census.json").read_text(encoding="utf-8"))


def header_macros(path, prefix):
    text = Path(path).read_text(encoding="utf-8")
    macros = {}
    for match in re.finditer(r"#define\s+(" + prefix + r"[A-Z0-9_]+)\s+([0-9e\-.+]+)", text):
        macros[match.group(1)] = float(match.group(2))
    kind_match = re.search(r"SPARK_MIMO26[A-Z_]*_MODEL_LAYER_KIND\[[^\]]*\]\s*=\s*\{(.*?)\};",
                           text, re.S)
    assert kind_match, path
    kind = [int(x) for x in re.findall(r"\d+", kind_match.group(1))]
    moe_match = re.search(r"SPARK_MIMO26[A-Z_]*_MODEL_LAYER_IS_MOE\[[^\]]*\]\s*=\s*\{(.*?)\};",
                          text, re.S)
    assert moe_match, path
    moe = [int(x) for x in re.findall(r"\d+", moe_match.group(1))]
    return macros, kind, moe


def check_arm(tag, header_name, macro_prefix):
    c = census(tag)
    cfg = c["config"]
    macros, kind, moe = header_macros(FAMILY / "include" / "sparkpipe" / header_name, macro_prefix)
    assert kind == list(cfg["hybrid_layer_pattern"]), f"{tag} layer kind table vs census"
    assert moe == list(cfg["moe_layer_freq"]), f"{tag} moe table vs census"
    assert len(kind) == int(macros[macro_prefix + "LAYER_COUNT"]) == cfg["num_hidden_layers"]
    assert int(macros[macro_prefix + "HIDDEN_DIMENSION"]) == cfg["hidden_size"]
    assert int(macros[macro_prefix + "VOCAB_COUNT"]) == cfg["vocab_size"]
    assert int(macros[macro_prefix + "ROUTED_EXPERT_COUNT"]) == cfg["n_routed_experts"]
    assert int(macros[macro_prefix + "EXPERTS_PER_TOKEN"]) == cfg["num_experts_per_tok"]
    assert macros[macro_prefix + "RMS_NORM_EPSILON"] == cfg["layernorm_epsilon"]
    assert macros[macro_prefix + "ATTN_VALUE_SCALE"] == cfg["attention_value_scale"]

    patterns = c["patterns"]
    assert c["index_cross_check"]["match"], f"{tag} index cross-check"
    assert patterns["layer.expert"]["dtypes"] == {"U8": patterns["layer.expert"]["count"]}
    assert patterns["layer.expert_scale"]["dtypes"] == {"U8": patterns["layer.expert_scale"]["count"]}
    assert patterns["layer.o_proj"]["dtypes"] == {"BF16": patterns["layer.o_proj"]["count"]}
    assert patterns["layer.qkv"]["dtypes"] == {"F8_E4M3": patterns["layer.qkv"]["count"]}

    swa_layers = sum(kind)
    assert patterns["layer.sink_bias"]["count"] == swa_layers, f"{tag} sink bias vs SWA layer count"

    total = c["totals"]["payload_bytes"]
    summed = sum(p["bytes"] for p in patterns.values())
    assert summed == total, f"{tag} pattern bytes vs totals"
    experts = sum(patterns[k]["bytes"] for k in ("layer.expert", "layer.expert_scale"))
    out_of_scope = sum(patterns[k]["bytes"] for k in patterns if k.split(".")[0] in ("mtp", "visual", "audio", "speech"))
    assert experts + out_of_scope < total, f"{tag} served/expert/out-of-scope partition"
    assert c["experts"]["moe_layers"] == sum(moe)
    assert c["experts"]["distinct_expert_ids"] == cfg["n_routed_experts"]
    assert c["experts"]["expert_id_range"] == [0, cfg["n_routed_experts"] - 1]

    ignored = c["quantization_config"]["ignored_layers"]
    o_proj_ignored = sum(1 for n in ignored if re.fullmatch(r"model\.layers\.\d+\.self_attn\.o_proj", n))
    assert o_proj_ignored == cfg["num_hidden_layers"], f"{tag} every layer o_proj stays BF16"
    assert all(n.endswith("self_attn.o_proj") for n in ignored), f"{tag} ignored subset is o_proj-only"
    return c


def check_qkv_geometry():
    pro = census("pro")
    flash = census("flash")
    pro_macros, _, _ = header_macros(FAMILY / "include" / "sparkpipe" / "spark_mimo26_pro_model.h",
                                     "SPARK_MIMO26_PRO_MODEL_")
    flash_macros, _, _ = header_macros(FAMILY / "include" / "sparkpipe" / "spark_mimo26_model.h",
                                       "SPARK_MIMO26_MODEL_")

    def variant_rows(c, tag):
        v = c["patterns"]["layer.qkv"]["shape_variants"]
        return {tuple(json.loads(s.split(":", 1)[1])): n for s, n in v.items()}

    pro_rows = variant_rows(pro, "pro")
    assert list(pro_rows) == [(27136, 6144)], pro_rows
    assert int(pro_macros["SPARK_MIMO26_PRO_MODEL_FULL_QKV_DIMENSION"]) == 27136
    assert int(pro_macros["SPARK_MIMO26_PRO_MODEL_SWA_QKV_DIMENSION"]) == 27136

    flash_rows = variant_rows(flash, "flash")
    assert flash_rows == {(13568, 4096): 9, (14848, 4096): 39}, flash_rows
    assert int(flash_macros["SPARK_MIMO26_MODEL_FULL_QKV_DIMENSION"]) == 13568
    assert int(flash_macros["SPARK_MIMO26_MODEL_SWA_QKV_DIMENSION"]) == 14848

    def scale_rows(c):
        v = c["patterns"]["layer.qkv_scale"]["shape_variants"]
        return {json.loads(s.split(":", 1)[1])[0]: n for s, n in v.items()}

    assert scale_rows(pro) == {216: 70}, scale_rows(pro)
    assert scale_rows(flash) == {108: 9, 116: 39}, scale_rows(flash)
    assert int(pro_macros["SPARK_MIMO26_PRO_MODEL_FULL_QKV_SCALE_ROWS"]) == 216
    assert int(pro_macros["SPARK_MIMO26_PRO_MODEL_SWA_QKV_SCALE_ROWS"]) == 216
    assert int(flash_macros["SPARK_MIMO26_MODEL_FULL_QKV_SCALE_ROWS"]) == 108
    assert int(flash_macros["SPARK_MIMO26_MODEL_SWA_QKV_SCALE_ROWS"]) == 116


def check_topology_registration():
    doc = json.loads((ROOT / "model_contracts" / "must_work_targets.json").read_text(encoding="utf-8"))
    families = {t["model_family"] for t in doc["targets"]}
    assert {"mimo26_pro", "mimo26_flash"} <= families
    by_id = {t["id"]: t for t in doc["targets"]}
    pro = by_id["mimo26_pro_mxfp4_experts_fp8_spine"]
    flash = by_id["mimo26_flash_mxfp4_experts_fp8_spine"]
    for target, contract in ((pro, "mimo26_pro_authoritative.json"), (flash, "mimo26_flash_authoritative.json")):
        assert target["contract"] == f"model_contracts/{contract}"
        assert target["routed_expert_weight_format"] == "mxfp4_e2m1_e8m0_block32"
        assert target["routed_expert_activation_format"] == "bf16"
        assert target["accumulator_format"] == "fp32"
        assert target["production_ready"] is False
        path = ROOT / target["contract"]
        assert path.exists(), path
        authoritative = json.loads(path.read_text(encoding="utf-8"))
        census_tag = "pro" if "pro" in contract else "flash"
        measured = census(census_tag)
        assert authoritative["sources"]["census"] == (
            f"model-families/mimo26/census/{census_tag}_census.json")
        assert authoritative["served"]["expert_payload_bytes"] == (
            measured["patterns"]["layer.expert"]["bytes"]
            + measured["patterns"]["layer.expert_scale"]["bytes"])
    assert pro["required_features"] and flash["required_features"]


def main() -> int:
    check_arm("pro", "spark_mimo26_pro_model.h", "SPARK_MIMO26_PRO_MODEL_")
    check_arm("flash", "spark_mimo26_model.h", "SPARK_MIMO26_MODEL_")
    check_qkv_geometry()
    check_topology_registration()
    print("PASS mimo26 census/facts binding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
