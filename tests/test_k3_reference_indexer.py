#!/usr/bin/env python3
import json
import pathlib
import struct
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import k3_reference

HIDDEN, Q_LORA, HEADS, DIM, TOPK, ROWS = 64, 1536, 4, 16, 5, 13
LAYER = "language_model.model.layers.3.self_attn."


def bf16_bytes(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def write_checkpoint(directory, tensors, dsa):
    header, blob = {}, b""
    for name, values in tensors.items():
        data = bf16_bytes(values)
        header[name] = {"dtype": "BF16", "shape": list(values.shape), "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    text = {"hidden_size": HIDDEN, "num_hidden_layers": 4, "vocab_size": 8, "rms_norm_eps": 1e-5}
    if dsa is not None:
        text["dsa_config"] = dsa
    (directory / "config.json").write_text(json.dumps({"text_config": text}))


def rounded(values):
    return k3_reference.widen(np.frombuffer(bf16_bytes(values), dtype=np.uint16).reshape(values.shape))


def expected(tensors, normed, compressed, causal):
    query = rounded(compressed @ tensors[LAYER + "indexer.wq_b.weight"].astype(np.float64).T).reshape(ROWS, HEADS, DIM).astype(np.float64)
    raw = rounded(normed @ tensors[LAYER + "indexer.wk.weight"].astype(np.float64).T).astype(np.float64)
    centred = raw - raw.mean(axis=-1, keepdims=True)
    key = rounded(centred / np.sqrt((centred * centred).mean(axis=-1, keepdims=True) + 1e-5) * tensors[LAYER + "indexer.k_norm.weight"]
                  + tensors[LAYER + "indexer.k_norm.bias"]).astype(np.float64)
    weight = rounded(normed @ tensors[LAYER + "indexer.weights_proj.weight"].astype(np.float64).T).astype(np.float64)
    scores = np.einsum("th,ths->ts", weight, np.maximum(np.einsum("thd,sd->ths", query, key), 0.0)) * DIM ** -0.5 * HEADS ** -0.5
    chosen = np.zeros_like(causal)
    for row in range(ROWS):
        allowed = [position for position in range(ROWS) if causal[row, position]]
        allowed.sort(key=lambda position: (-scores[row, position], position))
        chosen[row, allowed[:TOPK]] = True
    return chosen


def refused(dsa, directory):
    write_checkpoint(directory, {}, dsa)
    try:
        k3_reference.Reference(directory)
    except SystemExit as stop:
        return str(stop)
    return None


def main():
    rng = np.random.default_rng(31)
    failures = []
    dsa = {"index_topk": TOPK, "index_num_heads": HEADS, "index_head_dim": DIM, "index_rope_dim": 0, "index_k_norm_type": "layernorm",
           "index_topk_freq": 1, "index_skip_topk_offset": 1}
    tensors = {LAYER + "indexer.wq_b.weight": rng.standard_normal((HEADS * DIM, Q_LORA)).astype(np.float32) * 0.03,
               LAYER + "indexer.wk.weight": rng.standard_normal((DIM, HIDDEN)).astype(np.float32) * 0.2,
               LAYER + "indexer.weights_proj.weight": rng.standard_normal((HEADS, HIDDEN)).astype(np.float32) * 0.2,
               LAYER + "indexer.k_norm.weight": 1.0 + rng.standard_normal(DIM).astype(np.float32) * 0.1,
               LAYER + "indexer.k_norm.bias": rng.standard_normal(DIM).astype(np.float32) * 0.1}
    tensors = {name: rounded(values) for name, values in tensors.items()}
    normed = rounded(rng.standard_normal((ROWS, HIDDEN)).astype(np.float32))
    compressed = rounded(rng.standard_normal((ROWS, Q_LORA)).astype(np.float32))
    positions = np.arange(ROWS)
    causal = positions[None, :] <= positions[:, None]
    with tempfile.TemporaryDirectory() as scratch:
        directory = pathlib.Path(scratch)
        write_checkpoint(directory, tensors, dsa)
        reference = k3_reference.Reference(directory)
        chosen = reference.selected(LAYER, normed, compressed, causal)
        if np.any(chosen & ~causal):
            failures.append("the indexer selected a later position")
        counts = chosen.sum(axis=-1)
        if not np.array_equal(counts, np.minimum(positions + 1, TOPK)):
            failures.append(f"row selections {counts.tolist()}: a row keeps every earlier position up to index_topk and exactly index_topk beyond it")
        if not np.array_equal(chosen, expected(tensors, normed, compressed, causal)):
            failures.append("the selected positions are not the top index_topk of d^-0.5 H^-0.5 sum_h w_h relu(q_h . LayerNorm(wk x)) over earlier positions")
        if not np.array_equal(chosen[:TOPK], causal[:TOPK]):
            failures.append("a row with at most index_topk earlier positions must attend all of them")
        tied = dict(tensors)
        tied[LAYER + "indexer.weights_proj.weight"] = np.zeros((HEADS, HIDDEN), dtype=np.float32)
        write_checkpoint(directory, tied, dsa)
        flat = k3_reference.Reference(directory).selected(LAYER, normed, compressed, causal)
        if not np.array_equal(flat[ROWS - 1], np.arange(ROWS) < TOPK):
            failures.append(f"equal scores must keep the earliest positions, as the exact top-k kernel does: {np.nonzero(flat[ROWS - 1])[0].tolist()}")
        write_checkpoint(directory, {}, None)
        plain = k3_reference.Reference(directory)
        if not np.array_equal(plain.selected(LAYER, normed, compressed, causal), causal):
            failures.append("a checkpoint without indexer weights (K3) must attend every earlier position")
        for change, why in (({"index_topk_freq": 2}, "top-k shared across layers"), ({"index_rope_dim": 64}, "a rotated indexer"),
                            ({"index_k_norm_type": "rmsnorm"}, "an RMS-normed indexer key")):
            if refused(dict(dsa, **change), directory) is None:
                failures.append(f"a dsa_config with {why} ({change}) must be refused, not computed as the NoPE layer-normed indexer")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print("PASS k3 reference indexer: a DSA layer attends the top index_topk earlier positions by d^-0.5 H^-0.5 sum_h w_h relu(q_h . LayerNorm(wk x)), "
          "every earlier position up to index_topk, the earliest on equal scores; K3 without indexer weights is unchanged; "
          "shared top-k, a rotated indexer and an RMS-normed key are refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
