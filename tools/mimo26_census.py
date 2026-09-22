#!/usr/bin/env python3
"""MiMo 2.6 tensor census: shard headers only, never payload bytes.

Measures the mimo-v2.6 family checkpoints on the warm mount (pro-rl 535G,
flash-rl 166G, distill-qwen-9b 18G) by ranged reads of each safetensors
header slab (8-byte length + JSON map, read in <=1 MiB slabs) and emits the
aggregate inventory the family facts, geometry header, contract and stagepack
plan are derived from. No tensor payload is ever read, so peak RSS is bounded
by the largest header plus one index.json parse (well under the 1 GiB tooling
law; the queue job wraps MemoryMax at 1536 MiB).

The census records out-of-scope regions (MTP head, vision tower, audio
encoder, dflash draft tree) with their exact byte ranges - weights are
documented, never silently stripped (the qwen27b MTP-strip repair precedent).

Usage: python3 tools/mimo26_census.py --checkpoint /mnt/model-warm/mimo-v2.6-pro-rl \
           [--out census.json] [--index-check on|off] [--include-side-trees on|off]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import struct
import sys
import time
from collections import defaultdict

HEADER_SLAB_BYTES = 1 << 20
HEADER_MAX_BYTES = 64 << 20
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
               "I16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "F8_E4M0": 1, "BOOL": 1}

PATTERN_RULES = [
    ("mtp.eh_proj", r"^model\.mtp\.layers\.(\d+)\.eh_proj\.weight$"),
    ("mtp.norms", r"^model\.mtp\.layers\.(\d+)\.(enorm|hnorm|final_layernorm|input_layernorm|pre_mlp_layernorm)\.weight$"),
    ("mtp.self_attn", r"^model\.mtp\.layers\.(\d+)\.self_attn\.(qkv_proj|o_proj)\.weight(_scale_inv)?$"),
    ("mtp.self_attn.sink", r"^model\.mtp\.layers\.(\d+)\.self_attn\.attention_sink_bias$"),
    ("mtp.mlp", r"^model\.mtp\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight(_scale_inv)?$"),
    ("layer.norm", r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$"),
    ("layer.qkv", r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$"),
    ("layer.qkv_scale", r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight_scale_inv$"),
    ("layer.o_proj", r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
    ("layer.sink_bias", r"^model\.layers\.(\d+)\.self_attn\.attention_sink_bias$"),
    ("layer.router", r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"),
    ("layer.router_bias", r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"),
    ("layer.dense_mlp", r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"),
    ("layer.dense_mlp_scale", r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight_scale_inv$"),
    ("layer.expert", r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"),
    ("layer.expert_scale", r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_scale$"),
    ("embed_tokens", r"^model\.embed_tokens\.weight$"),
    ("final_norm", r"^model\.norm\.weight$"),
    ("lm_head", r"^lm_head\.weight$"),
    ("visual", r"^visual\.(.+)$"),
    ("audio_encoder", r"^audio_encoder\.(.+)$"),
    ("speech_embeddings", r"^speech_embeddings\.(\d+)\.weight$"),
]


def pattern_of(name):
    for tag, rx in PATTERN_RULES:
        m = re.match(rx, name)
        if m:
            return tag, tuple(int(g) for g in m.groups() if g is not None and g.lstrip("-").isdigit())
    return "unclassified:" + name, ()


def read_header_slab(path):
    """Return the parsed header dict without touching payload bytes."""
    with open(path, "rb", buffering=0) as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("short header length read: " + path)
        (n,) = struct.unpack("<Q", raw)
        if not (0 < n < HEADER_MAX_BYTES):
            raise ValueError("implausible header length %d: %s" % (n, path))
        chunks = []
        left = n
        while left > 0:
            slab = f.read(min(HEADER_SLAB_BYTES, left))
            if not slab:
                raise ValueError("truncated header: " + path)
            chunks.append(slab)
            left -= len(slab)
        header_sha = hashlib.sha256(b"".join(chunks)).hexdigest()
        return json.loads(b"".join(chunks).decode("utf-8")), n, header_sha


def element_bytes(dtype, nelts):
    per = DTYPE_BYTES.get(dtype)
    if per is None:
        raise ValueError("unmapped dtype " + dtype)
    return nelts * per


def census_shard(path, state):
    header, header_len, header_sha = read_header_slab(path)
    size = os.path.getsize(path)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    meta = header.get("__metadata__")
    file_rec = {"file": os.path.basename(path), "bytes": size,
                "header_bytes": header_len, "header_sha256": header_sha,
                "tensors": len(tensors)}
    if meta:
        file_rec["metadata"] = {str(k): str(v) for k, v in list(meta.items())[:8]}
    groups = defaultdict(lambda: {"tensors": 0, "bytes": 0, "dtypes": defaultdict(int)})
    for name, rec in tensors.items():
        if name in state["names"]:
            raise ValueError("duplicate tensor name across shards: " + name)
        state["names"].add(name)
        beg, end = rec["data_offsets"]
        nelts = 1
        for d in rec["shape"]:
            nelts *= d
        payload = element_bytes(rec["dtype"], nelts)
        if end - beg != payload:
            raise ValueError("offset span %d != element bytes %d for %s" % (end - beg, payload, name))
        if end > size:
            raise ValueError("data beyond EOF for %s in %s" % (name, path))
        tag, idx = pattern_of(name)
        pat = state["patterns"][tag]
        pat["count"] += 1
        pat["bytes"] += payload
        shape_key = rec["dtype"] + ":" + json.dumps(rec["shape"])
        pat.setdefault("shape_variants", defaultdict(int))[shape_key] += 1
        if pat["sample"] is None:
            pat["sample"] = {"name": name, "dtype": rec["dtype"], "shape": rec["shape"]}
        pat.setdefault("dtypes", defaultdict(int))[rec["dtype"]] += 1
        top = tag.split(".")[0]
        groups[top]["tensors"] += 1
        groups[top]["bytes"] += payload
        groups[top]["dtypes"][rec["dtype"]] += 1
        state["total_bytes"] += payload
        state["total_tensors"] += 1
        if tag == "layer.expert":
            state["experts"]["layers"].add(idx[0])
            state["experts"]["ids"].add(idx[1])
            mat = name.rsplit(".", 1)[1]
            key = (idx[0], idx[1], mat)
            state["expert_tensors"][key] = (rec["dtype"], tuple(rec["shape"]), payload, os.path.basename(path))
        elif tag == "layer.expert_scale":
            mat = name.rsplit(".", 2)[1]
            key = (idx[0], idx[1], mat)
            state["expert_tensors"][key] = (rec["dtype"], tuple(rec["shape"]), payload, os.path.basename(path))
        else:
            state["regions"][tag].append({"name": name, "dtype": rec["dtype"],
                                          "shape": rec["shape"], "file": os.path.basename(path),
                                          "begin": beg + 8 + header_len, "end": end + 8 + header_len,
                                          "bytes": payload})
    state["files"].append(file_rec)
    state["groups"][os.path.basename(path)] = {g: {"tensors": v["tensors"], "bytes": v["bytes"]}
                                               for g, v in groups.items()}


def classify_codec(patterns):
    codecs = {}
    for tag, rec in patterns.items():
        if rec["count"] == 0:
            continue
        dt = rec["sample"]["dtype"]
        if tag == "layer.expert":
            codecs[tag] = {"dtype": dt, "verdict": "mxfp4_e2m1_e8m0_block32" if dt == "U8" else dt}
        elif tag == "layer.expert_scale":
            codecs[tag] = {"dtype": dt, "verdict": "mxfp4_e8m0_scale_u8" if dt == "U8" else dt}
        elif tag in ("layer.qkv", "layer.dense_mlp", "mtp.self_attn", "mtp.mlp"):
            codecs[tag] = {"dtype": dt, "verdict": "fp8_e4m3" if dt == "F8_E4M3" else dt}
        elif tag in ("layer.qkv_scale", "layer.dense_mlp_scale"):
            codecs[tag] = {"dtype": dt, "verdict": "f32_scale_inv" if dt == "F32" else dt}
        else:
            codecs[tag] = {"dtype": dt}
    return codecs


def index_cross_check(checkpoint, state):
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        index = json.load(f)
    wm = index["weight_map"]
    missing = sorted(n for n in wm if n not in state["names"])
    extra = sorted(n for n in state["names"] if n not in wm)
    return {"index_tensors": len(wm), "census_tensors": len(state["names"]),
            "missing_from_census": missing[:20], "extra_in_census": extra[:20],
            "match": not missing and not extra and len(wm) == len(state["names"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out")
    ap.add_argument("--index-check", default="on", choices=["on", "off"])
    ap.add_argument("--include-side-trees", default="off", choices=["on", "off"])
    args = ap.parse_args()

    t0 = time.time()
    checkpoint = args.checkpoint.rstrip("/")
    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)

    state = {"names": set(), "total_bytes": 0, "total_tensors": 0,
             "patterns": defaultdict(lambda: {"count": 0, "bytes": 0, "sample": None, "dtypes": defaultdict(int)}),
             "experts": {"layers": set(), "ids": set()},
             "expert_tensors": {}, "regions": defaultdict(list), "files": [], "groups": {}}

    shard_files = sorted(fn for fn in os.listdir(checkpoint) if fn.endswith(".safetensors"))
    for fn in shard_files:
        census_shard(os.path.join(checkpoint, fn), state)

    side_trees = {}
    if args.include_side_trees == "on":
        for sub in ("dflash", "audio_tokenizer"):
            sub_path = os.path.join(checkpoint, sub)
            if not os.path.isdir(sub_path):
                continue
            sub_state = {"names": set(), "total_bytes": 0, "total_tensors": 0,
                         "patterns": defaultdict(lambda: {"count": 0, "bytes": 0, "sample": None, "dtypes": defaultdict(int)}),
                         "experts": {"layers": set(), "ids": set()},
                         "expert_tensors": {}, "regions": defaultdict(list), "files": [], "groups": {}}
            for fn in sorted(f for f in os.listdir(sub_path) if f.endswith(".safetensors")):
                census_shard(os.path.join(sub_path, fn), sub_state)
            side_trees[sub] = {"tensors": sub_state["total_tensors"], "bytes": sub_state["total_bytes"],
                               "files": sub_state["files"]}

    expert_by_layer = defaultdict(list)
    for (layer, expert, mat), rec in state["expert_tensors"].items():
        expert_by_layer[layer].append((expert, mat, rec))
    expert_summary = {}
    for layer in sorted(expert_by_layer):
        entries = expert_by_layer[layer]
        ids = sorted({e for e, _, _ in entries})
        per_id = defaultdict(dict)
        for expert, mat, rec in entries:
            per_id[expert][mat] = {"dtype": rec[0], "shape": list(rec[1]), "bytes": rec[2]}
        expert_summary[layer] = {"experts": len(ids), "min_id": ids[0], "max_id": ids[-1],
                                 "per_expert": per_id[ids[0]]}

    region_summary = {}
    for tag, items in sorted(state["regions"].items()):
        region_summary[tag] = {"tensors": len(items), "bytes": sum(i["bytes"] for i in items),
                               "byte_ranges": [{"file": i["file"], "begin": i["begin"], "end": i["end"]} for i in items]}

    receipt_path = os.path.join(checkpoint, "DOWNLOAD-RECEIPT.json")
    provenance = {}
    if os.path.exists(receipt_path):
        with open(receipt_path) as f:
            receipt = json.load(f)
        provenance = {"receipt_bytes": receipt.get("bytes"),
                      "receipt_files": len(receipt.get("file_records", [])),
                      "receipt_shas_pinned": sum(1 for r in receipt.get("file_records", []) if r.get("sha256"))}

    group_totals = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    for per_file in state["groups"].values():
        for g, v in per_file.items():
            group_totals[g]["tensors"] += v["tensors"]
            group_totals[g]["bytes"] += v["bytes"]

    out = {
        "schema": "mimo26-census/1",
        "checkpoint": os.path.basename(checkpoint),
        "path": checkpoint,
        "captured_host": platform.node(),
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": hashlib.sha256(open(os.path.join(checkpoint, "config.json"), "rb").read()).hexdigest(),
        "config": {k: config[k] for k in sorted(config)
                   if k in ("architectures", "model_type", "hidden_size", "num_hidden_layers",
                            "num_attention_heads", "num_key_value_heads", "head_dim", "v_head_dim",
                            "swa_head_dim", "swa_num_key_value_heads", "intermediate_size",
                            "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok",
                            "n_shared_experts", "moe_layer_freq", "hybrid_layer_pattern",
                            "sliding_window", "vocab_size", "max_position_embeddings",
                            "partial_rotary_factor", "rope_theta", "swa_rope_theta",
                            "attention_value_scale", "attention_projection_layout",
                            "add_full_attention_sink_bias", "add_swa_attention_sink_bias",
                            "attention_chunk_size", "scoring_func", "topk_method", "norm_topk_prob",
                            "tie_word_embeddings", "moe_router_dtype", "dtype",
                            "num_nextn_predict_layers", "layernorm_epsilon")},
        "quantization_config": config.get("quantization_config"),
        "provenance": provenance,
        "shards": state["files"],
        "totals": {"tensors": state["total_tensors"], "payload_bytes": state["total_bytes"],
                   "shard_count": len(state["files"]),
                   "sum_shard_file_bytes": sum(f["bytes"] for f in state["files"])},
        "group_totals": dict(group_totals),
        "patterns": {t: {"count": r["count"], "bytes": r["bytes"], "sample": r["sample"],
                         "dtypes": dict(r["dtypes"]),
                         "shape_variants": dict(r.get("shape_variants", {}))}
                     for t, r in sorted(state["patterns"].items()) if r["count"]},
        "codecs": classify_codec(state["patterns"]),
        "experts": {"moe_layers": len(state["experts"]["layers"]),
                    "distinct_expert_ids": len(state["experts"]["ids"]),
                    "expert_id_range": [min(state["experts"]["ids"]), max(state["experts"]["ids"])]
                    if state["experts"]["ids"] else None,
                    "per_layer": {str(k): v for k, v in expert_summary.items()}},
        "out_of_scope_regions": region_summary,
        "side_trees": side_trees,
    }
    if args.index_check == "on":
        out["index_cross_check"] = index_cross_check(checkpoint, state)

    text = json.dumps(out, indent=1, sort_keys=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print("wrote %s (%d bytes)" % (args.out, len(text)))
    else:
        sys.stdout.write(text + "\n")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = (rss / 1024.0) if sys.platform != "darwin" else (rss / 2**20)
    print("tensors=%d payload=%.3f GiB shards=%d elapsed=%.1fs peak_rss=%.1f MiB" % (
        state["total_tensors"], state["total_bytes"] / 2**30, len(state["files"]),
        time.time() - t0, rss_mib), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
