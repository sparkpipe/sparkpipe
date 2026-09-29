import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import Safetensors, read_fixture  # noqa: E402

FP8_BLOCK = 128
INTERLEAVE = 4
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def qkv_row_blocks(rows, grid_rows):
    per = rows // INTERLEAVE
    blocks = -(-per // FP8_BLOCK)
    if per * INTERLEAVE != rows or blocks * INTERLEAVE != grid_rows:
        raise SystemExit(f"qkv {rows} rows / {grid_rows} grid rows is not {INTERLEAVE} padded rank segments")
    index = np.arange(rows)
    return (index // per) * blocks + (index % per) // FP8_BLOCK


def routed_experts(arrays, layers):
    used = [set() for _ in range(layers)]
    for key, value in arrays.items():
        if key.endswith("_route_ids"):
            used[int(key.split("_layer")[1][:4])].update(int(v) for v in value)
    return [sorted(s) for s in used]


def require_route_sets(arrays, moe_layer_freq):
    positions = len(arrays["prompt_token_ids"]) + len(arrays["generated_token_ids"]) - 1
    missing = [f"pos{p:04d}_layer{layer:04d}_route_ids" for p in range(positions)
               for layer, moe in enumerate(moe_layer_freq) if moe
               and f"pos{p:04d}_layer{layer:04d}_route_ids" not in arrays]
    if missing:
        raise SystemExit(f"fixture lacks {len(missing)} route sets the harness compares, first {missing[0]}")
    return positions * sum(1 for moe in moe_layer_freq if moe)


def dense_rows(st, name):
    payload = st.pread(name)
    grid = st.pread(name + "_scale_inv")
    if grid.shape != (-(-payload.shape[0] // FP8_BLOCK), payload.shape[1] // FP8_BLOCK):
        raise SystemExit(f"{name}: grid {grid.shape} does not tile {payload.shape}")
    return payload, np.ascontiguousarray(grid[np.arange(payload.shape[0]) // FP8_BLOCK])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-from", default=None)
    args = parser.parse_args()
    config = json.load(open(os.path.join(args.checkpoint, "config.json")))
    layers = int(config["num_hidden_layers"])
    experts = int(config["n_routed_experts"])
    _, arrays = read_fixture(args.fixture)
    used = routed_experts(arrays, layers)
    route_sets = require_route_sets(arrays, config["moe_layer_freq"])
    st = Safetensors(args.checkpoint)
    os.makedirs(args.out, exist_ok=True)

    def put(name, array):
        np.ascontiguousarray(array).tofile(os.path.join(args.out, name))

    put("prompt.i32", np.asarray(arrays["prompt_token_ids"], dtype=np.int32))
    put("generated.i32", np.asarray(arrays["generated_token_ids"], dtype=np.int32))
    if args.weights_from is None:
        put("embed.bf16", st.pread("model.embed_tokens.weight"))
        put("final_norm.bf16", st.pread("model.norm.weight"))
        put("lm_head.bf16", st.pread("lm_head.weight"))
    for key in sorted(arrays):
        if key.endswith("_streams"):
            put(key + ".bf16", arrays[key])
        if key.endswith("_route_ids"):
            put(key + ".i32", np.asarray(arrays[key], dtype=np.int32))
    summary = {"layers": layers, "experts": [len(u) for u in used], "route_sets": route_sets}
    if args.weights_from is not None:
        source = os.path.abspath(args.weights_from)
        for name in sorted(os.listdir(source)):
            if name.startswith(("l", "embed", "final_norm", "lm_head")) and not name.endswith("expert_used.i32") \
                    and not os.path.exists(os.path.join(args.out, name)):
                os.symlink(os.path.join(source, name), os.path.join(args.out, name))
        for layer in range(layers):
            if config["moe_layer_freq"][layer]:
                put(f"l{layer:02d}_expert_used.i32", np.asarray(used[layer], dtype=np.int32))
        json.dump(summary, open(os.path.join(args.out, "manifest.json"), "w"))
        print(json.dumps({"out": args.out, "resident_experts": sum(summary["experts"]), "weights_from": source}))
        return 0
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        tag = f"l{layer:02d}_"
        put(tag + "attn_norm.bf16", st.pread(prefix + "input_layernorm.weight"))
        put(tag + "mlp_norm.bf16", st.pread(prefix + "post_attention_layernorm.weight"))
        qkv = st.pread(prefix + "self_attn.qkv_proj.weight")
        grid = st.pread(prefix + "self_attn.qkv_proj.weight_scale_inv")
        put(tag + "qkv.fp8", qkv)
        put(tag + "qkv_scale_rows.f32", grid[qkv_row_blocks(qkv.shape[0], grid.shape[0])])
        put(tag + "o_proj.bf16", st.pread(prefix + "self_attn.o_proj.weight"))
        if config["hybrid_layer_pattern"][layer] == 1:
            put(tag + "sink.bf16", st.pread(prefix + "self_attn.attention_sink_bias"))
        if not config["moe_layer_freq"][layer]:
            for proj in PROJECTIONS:
                payload, rows = dense_rows(st, prefix + f"mlp.{proj}.weight")
                put(tag + f"dense_{proj}.fp8", payload)
                put(tag + f"dense_{proj}_scale_rows.f32", rows)
            continue
        put(tag + "router.bf16", st.pread(prefix + "mlp.gate.weight"))
        put(tag + "router_bias.f32", st.pread(prefix + "mlp.gate.e_score_correction_bias").astype(np.float32))
        put(tag + "expert_used.i32", np.asarray(used[layer], dtype=np.int32))
        names = [prefix + f"mlp.experts.{e}.{proj}.{kind}" for proj in PROJECTIONS for e in range(experts)
                 for kind in ("weight", "weight_scale")]
        for name in names:
            st.entry(name)
        with ThreadPoolExecutor(max_workers=16) as pool:
            blobs = dict(zip(names, pool.map(st.pread, names)))
        for proj in PROJECTIONS:
            put(tag + f"{proj}.mxfp4", np.stack([blobs[prefix + f"mlp.experts.{e}.{proj}.weight"] for e in range(experts)]))
            put(tag + f"{proj}.ue8m0", np.stack([blobs[prefix + f"mlp.experts.{e}.{proj}.weight_scale"] for e in range(experts)]))
        print(json.dumps({"layer": layer, "experts": len(used[layer])}), flush=True)
    json.dump(summary, open(os.path.join(args.out, "manifest.json"), "w"))
    print(json.dumps({"out": args.out, "resident_experts": sum(summary["experts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
