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
