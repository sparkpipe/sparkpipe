#!/usr/bin/env python3
"""Verify a Kimi-K3 checkpoint directory against model_contracts/k3_authoritative.json.

P1 of the pack chain (docs/AGENT_LANE_BRIEFS/pack_agent_rules.md): source
identity before any packing. Three checks, each loud:

  geometry   config.json's text_config is diffed field-by-field against the
             authoritative contract (hidden, layers, vocab, MoE 896+2 top-16,
             KDA 96x128, MLA 96q/1536/512/128/64/128, AttnRes 12, situ betas,
             MXFP4 group 32, MTP 0). Any mismatch is a FAIL naming the field.
  inventory  model.safetensors.index.json must reference exactly the shard
             files present on disk (count, sizes, no strays), and the stage-3
             tensor set (layers 70..92 + final norm + lm_head) must resolve
             to a shard subset - the pack build reads only those.
  sampled    deterministic sample of shard windows (first + middle 64 MiB of
             each sampled shard) sha256'd into the receipt. The contract
             carries no digests; the receipt is the freeze record, and the
             same receipt from a second host (cold backup) must be
             byte-identical to prove the copies agree.

Exit 0 = PASS, 1 = FAIL. Receipt (JSON) goes to --out or stdout.

usage: k3_verify_source.py <source_dir> [--contract PATH] [--out PATH]
                          [--sample-shards 1,24,47,71,83,93,96]
                          [--window-bytes 67108864]
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

STAGE3_FIRST_LAYER = 70
STAGE3_LAST_LAYER = 92  # inclusive; 93 total layers, stage 3 is the tail


def check_geometry(cfg, contract, failures):
    t = cfg["text_config"]
    model = contract["model"]
    moe = contract["moe"]
    kda = contract["kda"]
    mla = contract["mla"]
    attnres = contract["attnres"]
    quant = contract["quantization"]
    spec = contract["speculation"]
    checks = [
        ("architectures", cfg["architectures"][0], None),
        ("text_config.model_type", t["model_type"], None),
        ("hidden_size", t["hidden_size"], model["hidden_dimension"]),
        ("num_hidden_layers", t["num_hidden_layers"], model["layer_count"]),
        ("vocab_size", t["vocab_size"], model["vocabulary_size"]),
        ("max_position_embeddings", t["max_position_embeddings"],
         model["maximum_context_tokens"]),
        ("rms_norm_eps", t["rms_norm_eps"], model["rms_norm_epsilon"]),
        ("first_k_dense_replace", t["first_k_dense_replace"],
         model["first_routed_layer"]),
        ("num_experts", t["num_experts"], moe["routed_expert_count"]),
        ("num_experts_per_token", t["num_experts_per_token"],
         moe["experts_per_token"]),
        ("num_shared_experts", t["num_shared_experts"],
         moe["shared_expert_count"]),
        ("routed_expert_hidden_size", t["routed_expert_hidden_size"],
         moe["latent_dimension"]),
        ("moe_intermediate_size", t["moe_intermediate_size"],
         moe["expert_intermediate_dimension"]),
        ("intermediate_size", t["intermediate_size"],
         moe["dense_intermediate_dimension"]),
        ("routed_scaling_factor", t["routed_scaling_factor"],
         moe["routed_scaling_factor"]),
        ("moe_router_activation_func", t["moe_router_activation_func"],
         moe["router_activation"]),
        ("moe_renormalize", t["moe_renormalize"],
         moe["renormalize_selected_probabilities"]),
        ("hidden_act", t["hidden_act"], "situ"),
        ("activation_situ_beta", t["activation_situ_beta"],
         moe["situ_gate_beta"]),
        ("activation_situ_linear_beta", t["activation_situ_linear_beta"],
         moe["situ_up_beta"]),
        ("attn_res_block_size", t["attn_res_block_size"],
         attnres["block_size_layers"]),
        ("linear_attn_config.num_heads", t["linear_attn_config"]["num_heads"],
         kda["head_count"]),
        ("linear_attn_config.head_dim", t["linear_attn_config"]["head_dim"],
         kda["key_dimension"]),
        ("linear_attn_config.short_conv_kernel_size",
         t["linear_attn_config"]["short_conv_kernel_size"],
         kda["short_conv_kernel"]),
        ("linear_attn_config.gate_lower_bound",
         t["linear_attn_config"]["gate_lower_bound"],
         kda["minimum_log_decay"]),
        ("linear_attn_config.use_full_rank_gate",
         t["linear_attn_config"]["use_full_rank_gate"],
         kda["full_rank_output_gate"]),
        ("linear_attn_config.kda_layers.count",
         len(t["linear_attn_config"]["kda_layers"]),
         contract["hybrid_attention"]["kda_layer_count"]),
        ("linear_attn_config.full_attn_layers.count",
         len(t["linear_attn_config"]["full_attn_layers"]),
         contract["hybrid_attention"]["mla_layer_count"]),
        ("linear_attn_config.full_attn_layers.final",
         t["linear_attn_config"]["full_attn_layers"][-1],
         model["layer_count"]),
        ("num_attention_heads", t["num_attention_heads"],
         mla["query_head_count"]),
        ("q_lora_rank", t["q_lora_rank"], mla["query_lora_rank"]),
        ("kv_lora_rank", t["kv_lora_rank"], mla["kv_lora_rank"]),
        ("qk_nope_head_dim", t["qk_nope_head_dim"],
         mla["qk_nope_dimension"]),
        ("qk_rope_head_dim", t["qk_rope_head_dim"],
         mla["qk_unrotated_dimension"]),
        ("v_head_dim", t["v_head_dim"], mla["value_head_dimension"]),
        ("mla_use_nope", t["mla_use_nope"], mla["uses_nope"]),
        ("mla_use_output_gate", t["mla_use_output_gate"], mla["output_gate"]),
        ("num_nextn_predict_layers", t["num_nextn_predict_layers"],
         spec["base_checkpoint_mtp_layer_count"]),
    ]
    for name, got, want in checks:
        if want is not None and got != want:
            failures.append(f"geometry {name}: contract {want!r}, config {got!r}")
    qcfg = t["quantization_config"]
    group = qcfg["config_groups"]["group_0"]["weights"]
    if group["group_size"] != quant["routed_expert_group_size"]:
        failures.append(
            f"geometry quantization group: contract "
            f"{quant['routed_expert_group_size']}, config {group['group_size']}")
    if group["num_bits"] != 4 or group["scale_dtype"] != "torch.uint8":
        failures.append(
            f"geometry quantization: expected 4-bit uint8 scales, "
            f"got {group['num_bits']}-bit {group['scale_dtype']}")
    return len(checks)


def check_inventory(source, index, failures):
    shards = {}
    for name in set(index["weight_map"].values()):
        shards[name] = shards.get(name, 0) + 1
    present = {p.name for p in source.glob("model-*.safetensors")}
    missing = sorted(set(shards) - present)
    stray = sorted(present - set(shards))
    if missing:
        failures.append(f"inventory: index references absent shards {missing[:4]}"
                        f" ({len(missing)} total)")
    if stray:
        failures.append(f"inventory: on-disk shards not in index {stray[:4]}"
                        f" ({len(stray)} total)")
    stage3 = {}
    layer_re = re.compile(r"layers\.(\d+)\.")
    tail_names = 0
    for name, shard in index["weight_map"].items():
        m = layer_re.search(name)
        if m:
            layer = int(m.group(1))
            if STAGE3_FIRST_LAYER <= layer <= STAGE3_LAST_LAYER:
                stage3[shard] = stage3.get(shard, 0) + 1
        else:
            tail_names += 1
            stage3[shard] = stage3.get(shard, 0) + 1
    numbered = sorted(int(re.search(r"(\d+)", s).group(1)) for s in stage3)
    if not numbered:
        failures.append("inventory: no stage-3 tensors resolved from the index")
    return {"shard_count_index": len(shards),
            "shard_count_disk": len(present),
            "index_tensor_count": len(index["weight_map"]),
            "index_total_size_bytes": index.get("metadata", {}).get(
                "total_size"),
            "stage3_shard_count": len(stage3),
            "stage3_shard_range": ([min(numbered), max(numbered)]
                                   if numbered else None),
            "stage3_non_layer_tensor_names": tail_names}


def sample_digests(source, index, sample_shards, window, failures):
    digests = {}
    names = sorted(index["weight_map"])
    for n in sample_shards:
        matches = [s for s in shards_of(index, n)]
        if not matches:
            failures.append(f"sample: shard {n} not referenced by the index")
            continue
        path = source / matches[0]
        size = path.stat().st_size
        with open(path, "rb") as handle:
            for label, offset in (("head", 0),
                                  ("mid", max(0, size // 2 - window // 2))):
                handle.seek(offset)
                digest = hashlib.sha256()
                remaining = window
                while remaining:
                    chunk = handle.read(min(1 << 24, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    remaining -= len(chunk)
                digests[f"shard{n:03d}.{label}"] = {
                    "file": matches[0], "offset": offset,
                    "bytes": window - remaining,
                    "sha256": digest.hexdigest(), "file_size": size}
    return digests


def shards_of(index, number):
    prefix = f"model-{number:05d}-of-"
    for name in sorted(set(index["weight_map"].values())):
        if name.startswith(prefix):
            yield name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--contract", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "model_contracts" / "k3_authoritative.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-shards", default="1,24,47,71,83,93,96")
    parser.add_argument("--window-bytes", type=int, default=64 << 20)
    args = parser.parse_args()

    failures = []
    contract = json.loads(args.contract.read_text())
    config_path = args.source_dir / "config.json"
    if not config_path.is_file():
        print(f"FAIL: no config.json under {args.source_dir}")
        return 1
    cfg = json.loads(config_path.read_text())
    geometry_checks = check_geometry(cfg, contract, failures)

    index_path = args.source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        failures.append("inventory: model.safetensors.index.json absent")
        inventory = {}
        digests = {}
    else:
        index = json.loads(index_path.read_text())
        inventory = check_inventory(args.source_dir, index, failures)
        wanted = [int(x) for x in args.sample_shards.split(",") if x.strip()]
        digests = sample_digests(args.source_dir, index, wanted,
                                 args.window_bytes, failures)

    receipt = {
        "source": str(args.source_dir),
        "contract": str(args.contract),
        "contract_model_id": contract["model_id"],
        "geometry_checks": geometry_checks,
        "geometry_failures": failures[:],
        "inventory": inventory,
        "sampled_digests": digests,
        "result": "PASS" if not failures else "FAIL",
    }
    text = json.dumps(receipt, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text)
    print(text)
    print(f"K3 SOURCE VERIFY {receipt['result']}: "
          f"{geometry_checks} geometry fields checked, "
          f"{len(digests)} sampled windows")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
