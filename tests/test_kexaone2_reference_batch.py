#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kexaone2_reference as reference  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, MOE_INTER, EXPERTS, VOCAB = 192, 6, 2, 32, 320, 96, 8, 97
LAYERS = ["dense", "dense", "sparse", "sparse"]
WINDOWS = [0, 6, 4, 0]
LIMITS = [0.0, 0.0, 0.0, 7.0]


def bf16_bytes(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def write_checkpoint(directory):
    rng = np.random.default_rng(23)
    tensors = {"model.embed_tokens.weight": rng.normal(0, 1.0, (VOCAB, HIDDEN)), "model.norm.weight": rng.normal(1, 0.1, HIDDEN),
               "lm_head.weight": rng.normal(0, 0.08, (VOCAB, HIDDEN))}
    for layer, kind in enumerate(LAYERS):
        p = f"model.layers.{layer}."
        tensors.update({p + "input_layernorm.weight": rng.normal(1, 0.1, HIDDEN), p + "post_attention_layernorm.weight": rng.normal(1, 0.1, HIDDEN),
                        p + "self_attn.q_proj.weight": rng.normal(0, 0.08, (HEADS * HEAD_DIM, HIDDEN)),
                        p + "self_attn.k_proj.weight": rng.normal(0, 0.08, (KV_HEADS * HEAD_DIM, HIDDEN)),
                        p + "self_attn.v_proj.weight": rng.normal(0, 0.08, (KV_HEADS * HEAD_DIM, HIDDEN)),
                        p + "self_attn.o_proj.weight": rng.normal(0, 0.08, (HIDDEN, HEADS * HEAD_DIM)),
                        p + "self_attn.q_norm.weight": rng.normal(1, 0.1, HEAD_DIM), p + "self_attn.k_norm.weight": rng.normal(1, 0.1, HEAD_DIM)})
        blocks = [p + "mlp."] if kind == "dense" else [p + "mlp.shared_experts."] + [f"{p}mlp.experts.{e}." for e in range(EXPERTS)]
        for block in blocks:
            width = INTER if kind == "dense" else MOE_INTER
            tensors.update({block + "gate_proj.weight": rng.normal(0, 0.1, (width, HIDDEN)), block + "up_proj.weight": rng.normal(0, 0.1, (width, HIDDEN)),
                            block + "down_proj.weight": rng.normal(0, 0.1, (HIDDEN, width))})
        if kind == "sparse":
            tensors[p + "mlp.gate.weight"] = rng.normal(0, 0.1, (EXPERTS, HIDDEN))
            tensors[p + "mlp.e_score_correction_bias"] = rng.normal(0, 0.05, EXPERTS)
    header, payload, offset = {}, bytearray(), 0
    for name, values in tensors.items():
        data = bf16_bytes(values)
        header[name] = {"dtype": "BF16", "shape": list(np.shape(values)), "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}))
    (directory / "config.json").write_text(json.dumps({
        "rms_norm_eps": 1e-5, "head_dim": HEAD_DIM, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.5, "rope_parameters": {"rope_theta": 1000000.0}, "num_hidden_layers": len(LAYERS),
        "sliding_windows": WINDOWS, "swiglu_limits": LIMITS, "mlp_layer_types": LAYERS}))


def run(model, sequences):
    finals = model.forward(sequences)
    return finals, [model.logits(final) for final in finals]


def main():
    failures = []
    rng = np.random.default_rng(5)
    first = [int(t) for t in rng.integers(0, VOCAB, 9)]
    others = [[int(t) for t in rng.integers(0, VOCAB, n)] for n in (37, 3, 61)]
    longer = first + [int(t) for t in rng.integers(0, VOCAB, 44)]
    with tempfile.TemporaryDirectory() as scratch:
        write_checkpoint(Path(scratch))
        model = reference.Reference(scratch)
        alone_final, alone_logits = run(model, [first])
        batch_final, batch_logits = run(model, [first] + others)
        long_final, long_logits = run(model, [longer] + others[:1])
    if not np.isfinite(alone_logits[0]).all() or float(np.abs(alone_logits[0]).max()) == 0.0:
        failures.append("the synthetic model produces finite, nonzero logits")
    if not (np.array_equal(alone_final[0], batch_final[0]) and np.array_equal(alone_logits[0], batch_logits[0])):
        failures.append(f"a prompt scored with others has other bits than alone: max |difference| {float(np.abs(alone_logits[0] - batch_logits[0]).max()):.3g}")
    if not (np.array_equal(alone_final[0], long_final[0][:len(first)]) and np.array_equal(alone_logits[0], long_logits[0][:len(first)])):
        failures.append(f"a prompt's positions change when more served tokens follow: max |difference| {float(np.abs(alone_logits[0] - long_logits[0][:len(first)]).max()):.3g}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS K-EXAONE reference batch invariance: a prompt's final states and logits are bit-identical alone, beside three other prompts "
          "and with 44 more served tokens after it (dense and routed layers, global and windowed attention, the SwiGLU clamp)")


if __name__ == "__main__":
    main()
