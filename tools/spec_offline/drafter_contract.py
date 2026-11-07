import hashlib
import json
import re
from pathlib import Path

CHUNK = 16 << 20
SHA256 = re.compile(r"[0-9a-f]{64}")
RECEIPT = "TRAINING-RECEIPT.json"
LAYER_WEIGHTS = ("input_layernorm", "post_attention_layernorm", "self_attn.q_a_proj", "self_attn.q_a_layernorm", "self_attn.q_b_proj",
                 "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm", "self_attn.kv_b_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "attention_conv.kernel_projection", "mlp_conv.kernel_projection")


class DrafterContractError(ValueError):
    pass


def refuse_unsupported(config):
    dflash = config.get("dflash_config") or {}
    checks = (
        (config.get("architectures") == ["DFlash2DraftModel"], "architectures must be [DFlash2DraftModel]"),
        (str(dflash.get("attention_mode", "")).lower() == "mla", "attention_mode must be mla"),
        (dflash.get("aux_hidden_stream", config.get("aux_hidden_stream")) in (None, "prefix"), "aux_hidden_stream must be the prefix stream"),
        (not config.get("fc_norm", False), "fc_norm must be false"),
        (not config.get("mla_use_output_gate", False), "mla_use_output_gate must be false"),
        (float(dflash.get("input_embedding_scale", config.get("input_embedding_scale", 1.0))) == 1.0, "input_embedding_scale must be 1"),
        (float(dflash.get("output_multiplier", 1.0)) == 1.0, "output_multiplier must be 1"),
        (not dflash.get("final_logit_softcapping"), "final_logit_softcapping is unsupported"),
        (not config.get("is_causal", False), "the block must be non-causal"),
        (len(config.get("layer_types", [])) == config.get("num_hidden_layers"), "layer_types must name every layer"),
        (set(config.get("layer_types", [])) <= {"sliding_attention", "full_attention"}, "layer_types must be sliding_attention or full_attention"),
        (config["hidden_size"] % int(dflash.get("conv_group_size", 1)) == 0, "conv_group_size must divide hidden_size"),
        (1 <= int(dflash.get("conv_kernel_size", 0)) <= int(dflash.get("block_size", 0)), "conv_kernel_size must be within the block"),
    )
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise DrafterContractError("latent block drafter refuses this checkpoint: " + "; ".join(failed))


def tap_entries(config):
    return [f"o{layer}" for layer in config["dflash_config"]["target_layer_ids"]]


def drafter_tensors(config):
    refuse_unsupported(config)
    dflash = config["dflash_config"]
    hidden, heads, inter = config["hidden_size"], config["num_attention_heads"], config["intermediate_size"]
    nope, rope, value = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
    q_rank, kv_rank = config["q_lora_rank"], config["kv_lora_rank"]
    taps, groups, rank = dflash["conv_kernel_size"], hidden // dflash["conv_group_size"], dflash["selector_rank"]
    vocab = config.get("draft_vocab_size", config["vocab_size"])
    shapes = {"fc.weight": [hidden, hidden * len(dflash["target_layer_ids"])], "hidden_norm.weight": [hidden], "norm.weight": [hidden],
              "candidate_selector.hidden_projection.weight": [rank, hidden], "candidate_selector.predecessor_codebook": [vocab, rank],
              "candidate_selector.successor_codebook": [vocab, rank]}
    layer = {"input_layernorm": [hidden], "post_attention_layernorm": [hidden], "self_attn.q_a_proj": [q_rank, hidden], "self_attn.q_a_layernorm": [q_rank],
             "self_attn.q_b_proj": [heads * (nope + rope), q_rank], "self_attn.kv_a_proj_with_mqa": [kv_rank + rope, hidden],
             "self_attn.kv_a_layernorm": [kv_rank], "self_attn.kv_b_proj": [heads * (nope + value), kv_rank], "self_attn.o_proj": [hidden, heads * value],
             "mlp.gate_proj": [inter, hidden], "mlp.up_proj": [inter, hidden], "mlp.down_proj": [hidden, inter],
             "attention_conv.kernel_projection": [2 * taps * groups, hidden], "mlp_conv.kernel_projection": [2 * taps * groups, hidden]}
    for index in range(config["num_hidden_layers"]):
        for name in LAYER_WEIGHTS:
            shapes[f"layers.{index}.{name}.weight"] = layer[name]
        shapes[f"layers.{index}.attention_conv.base_kernel"] = [2, taps, hidden]
        shapes[f"layers.{index}.mlp_conv.base_kernel"] = [2, taps, hidden]
    return shapes


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def score_identity(drafter, target_shard, taps, donor_taps):
    return {"drafter_sha256": sha256_file(Path(drafter) / "model.safetensors"), "target_shard": file_identity(target_shard),
            "taps": [file_identity(path) for path in taps], "donor_taps": [file_identity(path) for path in donor_taps]}


def named(value):
    return isinstance(value, str) and value.strip() != ""


def check_licenses(base_license, taps, tap_licenses):
    if not named(base_license):
        raise DrafterContractError("the base checkpoint needs its license (--base-license)")
    if len(tap_licenses) != len(taps) or not all(named(license) for license in tap_licenses):
        raise DrafterContractError(f"each of the {len(taps)} --taps files needs its license: --tap-licenses names {len(tap_licenses)}, in the same order")


def training_sources(base, base_license, taps, tap_licenses):
    check_licenses(base_license, taps, tap_licenses)
    base = Path(base)
    weights = sha256_file(base / "model.safetensors")
    origin = weights
    if (base / RECEIPT).exists():
        try:
            receipt = json.loads((base / RECEIPT).read_text())
        except (OSError, ValueError) as error:
            raise DrafterContractError(f"{base / RECEIPT} is not readable JSON: {error}")
        origin = receipt.get("origin_sha256") if isinstance(receipt, dict) else None
        if not isinstance(origin, str) or SHA256.fullmatch(origin) is None:
            raise DrafterContractError(f"{base / RECEIPT} names no origin_sha256, so the drafter this one descends from is unknown")
    return {"base": {"directory": str(base), "model_sha256": weights, "license": base_license}, "origin_sha256": origin,
            "taps": [dict(file_identity(path), license=license) for path, license in zip(taps, tap_licenses)]}
