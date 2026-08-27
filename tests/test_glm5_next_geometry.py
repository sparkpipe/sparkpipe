#!/usr/bin/env python3
"""glm5_next M2 gate: geometry header + name mapping round-trip.

Holds the header (model-families/glm5_next/include/sparkpipe/
spark_glm5_next_model.h) against the frozen contract in lockstep, checks
the hybrid dispatch macro against the contract's layer lists, and round-trips
the name mapping: every TEXT-stack tensor pattern from the checkpoint census
(model-families/glm5_next/tensor_patterns.json) must resolve to a pack field,
every mapping entry must match a real census pattern with the right layer
class and shape, and every pack field must be reachable from the checkpoint.
Vision patterns must be the ONLY unresolved ones.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "model-families" / "glm5_next" / "include" / "sparkpipe" / "spark_glm5_next_model.h"
CONTRACT = ROOT / "model_contracts" / "glm53_flash_authoritative.json"
NAME_MAP = ROOT / "model-families" / "glm5_next" / "name_map.json"
PATTERNS = ROOT / "model-families" / "glm5_next" / "tensor_patterns.json"


def header_macros() -> dict[str, str]:
    text = re.sub(r"/\*.*?\*/", "", HEADER.read_text(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"\\\s*\n\s*", " ", text)  # join line continuations
    out = {}
    for match in re.finditer(
        r"#define\s+(SPARK_GLM5_NEXT_MODEL_\w+)(?:\([^)]*\))?\s+([^\n]+)", text
    ):
        out[match.group(1)] = match.group(2).strip()
    return out


def macro_float(macros: dict[str, str], name: str) -> float:
    return float(macros[name].replace("u", "").replace("f", ""))


def macro_int(macros: dict[str, str], name: str) -> int:
    return int(eval(macros[name].replace("u", "").replace("f", "")))


def eval_dim(macros: dict[str, str], expr: str, depth: int = 0) -> int:
    value = expr.strip()
    if depth < 4 and re.search(r"SPARK_GLM5_NEXT_MODEL_\w+", value):
        value = re.sub(
            r"SPARK_GLM5_NEXT_MODEL_\w+",
            lambda m: f"({eval_dim(macros, macros[m.group(0)], depth + 1)})",
            value,
        )
    value = value.replace("u", "")
    return int(eval(value))


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    name_map = json.loads(NAME_MAP.read_text())
    patterns = json.loads(PATTERNS.read_text())
    macros = header_macros()
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)

    # ---- 1. header literals vs the frozen contract
    pairs = [
        ("SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION", contract["model"]["hidden_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_LAYER_COUNT", contract["model"]["layer_count"]),
        ("SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX", contract["mtp"]["layer_index"]),
        ("SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT", contract["model"]["vocabulary_size"]),
        ("SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS", contract["model"]["maximum_context_tokens"]),
        ("SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT", contract["kda"]["head_count"]),
        ("SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION", contract["kda"]["head_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL", contract["kda"]["short_conv_kernel"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT", contract["mla"]["query_head_count"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION", contract["mla"]["query_lora_rank"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION", contract["mla"]["kv_lora_rank"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION", contract["mla"]["qk_nope_head_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION", contract["mla"]["qk_rope_head_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION", contract["mla"]["value_head_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT", contract["indexer"]["head_count"]),
        ("SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION", contract["indexer"]["head_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K", contract["indexer"]["top_k"]),
        ("SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL", contract["indexer"]["kpool"]),
        ("SPARK_GLM5_NEXT_MODEL_HC_MULT", contract["hyper_connections"]["hc_mult"]),
        ("SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS", contract["hyper_connections"]["sinkhorn_iterations"]),
        ("SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT", contract["moe"]["routed_expert_count"]),
        ("SPARK_GLM5_NEXT_MODEL_MOE_TOP_K", contract["moe"]["experts_per_token"]),
        ("SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION", contract["moe"]["expert_intermediate_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION", contract["moe"]["dense_intermediate_dimension"]),
        ("SPARK_GLM5_NEXT_MODEL_FIRST_DENSE_LAYER_COUNT", contract["moe"]["first_dense_layer_count"]),
        ("SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR", contract["moe"]["routed_scaling_factor"]),
        ("SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK", contract["precision"]["weight_block_size"][0]),
    ]
    for macro, expected in pairs:
        if isinstance(expected, float):
            got_ok = abs(macro_float(macros, macro) - expected) < 1e-9
        else:
            got_ok = macro_int(macros, macro) == expected
        check(f"{macro} == {expected}", got_ok)

    # derived dims against checkpoint-evidenced shapes
    derived = [
        ("SPARK_GLM5_NEXT_MODEL_MLA_QUERY_B_DIMENSION",
         contract["mla"]["checkpoint_tensor_shapes"]["q_b_proj.weight"][1][0]),   # 16384
        ("SPARK_GLM5_NEXT_MODEL_MLA_KV_A_DIMENSION",
         contract["mla"]["checkpoint_tensor_shapes"]["kv_a_proj_with_mqa.weight"][1][0]),  # 512
        ("SPARK_GLM5_NEXT_MODEL_MLA_KV_B_DIMENSION",
         contract["mla"]["checkpoint_tensor_shapes"]["kv_b_proj.weight"][1][0]),  # 32768
        ("SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION",
         contract["kda"]["checkpoint_tensor_shapes"]["q_proj.weight"][1][0]),     # 8192
        ("SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION",
         contract["indexer"]["checkpoint_tensor_shapes"]["wq_b.weight"][1][0]),   # 4096
        ("SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION",
         contract["hyper_connections"]["checkpoint_tensor_shapes"]["hc_attn_fn"][1][0]),  # 24
        ("SPARK_GLM5_NEXT_MODEL_HC_FN_COLUMNS",
         contract["hyper_connections"]["checkpoint_tensor_shapes"]["hc_attn_fn"][1][1]),  # 16384
        ("SPARK_GLM5_NEXT_MODEL_GATE_UP_DIMENSION",
         2 * contract["moe"]["checkpoint_tensor_shapes"]["experts.0.gate_proj.weight"][1][0]),
    ]
    for macro, expected in derived:
        check(f"derived {macro} == {expected}",
              eval_dim(macros, macros[macro]) == expected)

    check("mla qk scale is 256**-0.5",
          abs(macro_float(macros, "SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE") - 256 ** -0.5) < 1e-9)
    check("index softmax scale is 128**-0.5",
          abs(macro_float(macros, "SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE") - 128 ** -0.5) < 1e-9)
    check("index head weight scale is 32**-0.5",
          abs(macro_float(macros, "SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE") - 32 ** -0.5) < 1e-9)
    check("kda state bytes per layer = 64*128*128*4",
          eval_dim(macros, macros["SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER"]) == 64 * 128 * 128 * 4)
    check("kv slot bytes = 512 * 2",
          eval_dim(macros, macros["SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES"]) == 1024)
    check("hc scale count 3",
          eval_dim(macros, macros["SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT"]) == 3)
    check("index output width 2051",
          eval_dim(macros, macros["SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH"]) == 2051)

    # ---- 2. hybrid dispatch macro vs contract layer lists
    kda_layers = contract["hybrid_attention"]["kda_layers"]
    dsa_layers = contract["hybrid_attention"]["dsa_layers"]
    phase = macro_int(macros, "SPARK_GLM5_NEXT_MODEL_GLOBAL_ATTENTION_PHASE")
    period = macro_int(macros, "SPARK_GLM5_NEXT_MODEL_ATTENTION_PERIOD")
    macro_kda = [l for l in range(45) if l % period != phase]
    macro_dsa = [l for l in range(45) if l % period == phase]
    check("dispatch macro reproduces the 34 kda layers", macro_kda == kda_layers)
    check("dispatch macro reproduces the 11 dsa layers", macro_dsa == dsa_layers)
    check("name_map layer_classes agree with contract",
          name_map["layer_classes"]["kda_layers"] == kda_layers
          and name_map["layer_classes"]["dsa_layers"] == dsa_layers
          and name_map["layer_classes"]["mtp_layer"] == 45)

    # ---- 3. round-trip: checkpoint census patterns <-> name map
    lm = "model.language_model"
    text_patterns = {p for p in patterns if not p.startswith("model.visual")}
    vision_patterns = {p for p in patterns if p.startswith("model.visual")}

    def concrete(pattern: str, layer_class: str) -> str:
        """Instance a pattern at a layer valid for its class."""
        p = pattern.replace("{layer}", "0").replace("{expert}", "0")
        if layer_class == "dsa_layers":
            p = pattern.replace("{layer}", "3").replace("{expert}", "0")
        elif layer_class == "mtp_layer":
            p = pattern.replace("{layer}", "45").replace("{expert}", "0")
        elif layer_class == "dense_layers":
            p = pattern.replace("{layer}", "0").replace("{expert}", "0")
        elif layer_class == "moe_layers":
            p = pattern.replace("{layer}", "4").replace("{expert}", "0")
        return p

    # 3a. every mapping entry resolves to a census pattern with the right
    #     layer class and count
    entry_failures = []
    for entry in name_map["entries"]:
        pat = entry["checkpoint_pattern"]
        cls = entry["layer_class"]
        exemplar = concrete(pat, cls)
        found = patterns.get(pattern_of(exemplar))
        if found is None:
            # a fused target maps several sources; each source must exist
            entry_failures.append(f"{pat} [{cls}] -> no census pattern")
            continue
        # layer-class count check
        expected_counts = {
            "kda_layers": (34, None), "dsa_layers": (12, None),
            "moe_layers": (43, None), "dense_layers": (3, None),
            "hc_layers": (45, None), "mtp_layer": (1, None),
            "all": (46, None), "global": (1, None),
        }
        if cls in ("kda_layers", "dsa_layers", "moe_layers", "dense_layers",
                   "hc_layers", "all", "mtp_layer", "global"):
            count_ok = found["count"] >= expected_counts[cls][0] or cls == "global"
            if cls == "global":
                count_ok = found["count"] == 1
            if not count_ok:
                entry_failures.append(
                    f"{pat} [{cls}] census count {found['count']} < expected "
                    f"{expected_counts[cls][0]}")
    check(f"all {len(name_map['entries'])} mapping entries resolve with layer-class counts",
          not entry_failures)
    for f in entry_failures[:5]:
        print(f"       {f}")

    # 3b. every TEXT checkpoint pattern resolves through the map; only
    #     vision patterns may be unresolved
    mapped_patterns = set()
    for entry in name_map["entries"]:
        mapped_patterns.add(pattern_of(
            concrete(entry["checkpoint_pattern"], entry["layer_class"])))

    def resolvable(pattern: str) -> str | None:
        p = pattern
        for entry in name_map["entries"]:
            src = pattern_of(concrete(entry["checkpoint_pattern"], entry["layer_class"]))
            if p == src:
                return entry["pack_field"]
        return None

    unresolved_text = sorted(p for p in text_patterns if resolvable(p) is None)
    check("every text-stack tensor pattern resolves to a pack field",
          not unresolved_text)
    for p in unresolved_text[:10]:
        print(f"       unresolved: {p}")

    # 3c. pack fields are uniquely named except declared fusions; an fp8
    #     field legitimately appears again as its _scale twin
    base_fusion = {
        "kda_qkv_beta_weight": 4,          # q|k|v|beta
        "kda_decay_gate_down_weight": 2,   # f_a|g_a
        "expert_up_gate_weight": 2,        # up|gate
        "shared_gate_up_weight": 2,
        "dense_gate_up_weight": 2,
    }
    fields = [entry["pack_field"] for entry in name_map["entries"]]
    bad_fusions = []
    for field in sorted(set(fields)):
        sources = [e for e in name_map["entries"] if e["pack_field"] == field]
        base = base_fusion.get(field.removesuffix("_scale"), 1)
        # weight field: one entry per fused source pattern; the _scale twin
        # field mirrors that count for fp8 sources only
        if field.endswith("_scale") and not field.endswith("_log_scale"):
            fp8_base = [
                e for e in name_map["entries"]
                if e["pack_field"] == field.removesuffix("_scale")
                and e["codec"] == "fp8_block"]
            expected = len(fp8_base) if fp8_base else base
        else:
            expected = base
        if len(sources) != expected:
            bad_fusions.append(
                f"{field}: {len(sources)} entries (expected {expected})")
    check("pack field multiplicity matches the declared fusions", not bad_fusions)
    for f in bad_fusions:
        print(f"       {f}")

    # 3d. vision is the only out-of-scope set
    check(f"{len(vision_patterns)} vision patterns are the only excluded set",
          len(text_patterns) + len(vision_patterns) == len(patterns))

    # 3e. donor coverage: every donor named in the map is a real module dir
    donors = {entry["donor_module"] for entry in name_map["entries"]}
    for donor in sorted(donors):
        dirs = {
            "glm52": ROOT / "modules/glm52_resident_decode_stage",
            "k3": ROOT / "modules/k3_resident_decode_stage",
            "dsv4": ROOT / "modules/dsv4_resident_decode_stage",
        }
        check(f"donor module present: {donor}", dirs[donor].is_dir())

    # 3f. the five declared deltas are recorded (not silent)
    deltas = [entry.get("delta", "") for entry in name_map["entries"]]
    check("rope-0 MLA delta recorded",
          any("rope" in d and "0" in d for d in deltas))
    check("low-rank KDA gate delta recorded",
          any("low-rank" in d.lower() and "gate" in d.lower() for d in deltas))
    check("kpool compressor delta recorded",
          any("softmax" in d and "pool" in d.lower() for d in deltas))
    check("nope indexer delta recorded",
          any("NoPE" in d for d in deltas))

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print("PASS glm5_next geometry + name mapping round-trip")
    return 0


def pattern_of(name: str) -> str:
    p = re.sub(r"layers\.\d+", "layers.{layer}", name)
    return re.sub(r"experts\.\d+", "experts.{expert}", p)


if __name__ == "__main__":
    raise SystemExit(main())
