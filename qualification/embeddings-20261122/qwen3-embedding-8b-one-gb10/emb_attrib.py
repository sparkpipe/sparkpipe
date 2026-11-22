import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--variants", default="bf16,table,eager,fp32")
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()
variants = args.variants.split(",")
data = json.loads((args.out / "corpus.json").read_text())
ids = data["ids"][:args.limit] if args.limit else data["ids"]
groups = data["groups"]


def run(model, name):
    started = time.monotonic()
    rows = []
    with torch.inference_mode():
        for row in ids:
            state = model(input_ids=torch.tensor([row], device="cuda")).last_hidden_state[0, -1].float()
            rows.append(functional.normalize(state, p=2, dim=0).cpu().numpy().astype(np.float32))
    out = np.stack(rows)
    np.save(args.out / f"local_{name}.npy", out)
    print(json.dumps({"step": "local", "variant": name, "texts": len(rows), "seconds": round(time.monotonic() - started, 2),
                      "gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}), flush=True)
    return out


def theta_of(config):
    value = getattr(config, "rope_theta", None)
    if value is None:
        value = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    return float(value)


def rounded_table(model):
    rotary = model.rotary_emb
    half = rotary.inv_freq.numel()
    theta = theta_of(model.config)
    table = [np.float32(1.0) / np.float32(math.pow(theta, float(np.float32(2 * index) / np.float32(2 * half)))) for index in range(half)]
    table = torch.tensor(np.array(table, dtype=np.float32), device=rotary.inv_freq.device, dtype=rotary.inv_freq.dtype)
    changed = [int(index) for index in torch.nonzero(table != rotary.inv_freq).flatten().tolist()]
    saved = rotary.inv_freq.clone()
    rotary.inv_freq.copy_(table)
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq.copy_(table)
    return saved, changed


def restore_table(model, saved):
    model.rotary_emb.inv_freq.copy_(saved)
    if hasattr(model.rotary_emb, "original_inv_freq"):
        model.rotary_emb.original_inv_freq.copy_(saved)


def compare(a, b):
    count = min(len(a), len(b))
    a, b = a[:count].astype(np.float64), b[:count].astype(np.float64)
    distance = 1.0 - np.sum(a * b, axis=1)
    worst = int(distance.argmax())
    start, per_group = 0, {}
    for name, size in groups.items():
        part = distance[start:start + size]
        if len(part):
            per_group[name] = float(part.mean())
        start += size
    return {"texts": count, "bit_equal_rows": int(np.all(a == b, axis=1).sum()), "distance_max": float(distance.max()), "distance_median": float(np.median(distance)),
            "distance_mean": float(distance.mean()), "max_abs_difference": float(np.abs(a - b).max()), "worst_text": worst, "worst_tokens": len(ids[worst]),
            "mean_by_group": per_group}


def ranking(a, b):
    count = min(len(a), len(b))
    total = sum(groups.values())
    if count < total:
        return None
    queries = groups["queries"]
    documents = total - queries
    score_a, score_b = a[documents:] @ a[:documents].T, b[documents:] @ b[:documents].T
    order_a, order_b = np.argsort(-score_a, axis=1), np.argsort(-score_b, axis=1)
    swaps = sum(1 for row in range(queries) for position in range(9)
                if score_a[row][order_b[row][position]] < score_a[row][order_b[row][position + 1]])
    return {"score_difference_max": float(np.abs(score_a - score_b).max()), "score_difference_mean": float(np.abs(score_a - score_b).mean()),
            "top1_same": float(np.mean(order_a[:, 0] == order_b[:, 0])),
            "top10_overlap": float(np.mean([len(set(order_a[row][:10]) & set(order_b[row][:10])) / 10 for row in range(queries)])),
            "top10_same_order": float(np.mean([np.array_equal(order_a[row][:10], order_b[row][:10]) for row in range(queries)])),
            "adjacent_swaps_in_top10": swaps}


model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
print(json.dumps({"step": "loaded", "torch": torch.__version__, "attention": model.config._attn_implementation, "rope_theta": theta_of(model.config),
                  "table_entries": int(model.rotary_emb.inv_freq.numel())}), flush=True)
results = {}
if "bf16" in variants:
    results["bf16"] = run(model, "bf16")
    results["bf16_again"] = run(model, "bf16_again")
if "table" in variants:
    saved, changed = rounded_table(model)
    print(json.dumps({"step": "table", "entries_changed": changed}), flush=True)
    results["table"] = run(model, "table")
    restore_table(model, saved)
if "eager" in variants:
    model.config._attn_implementation = "eager"
    results["eager"] = run(model, "eager")
    model.config._attn_implementation = "sdpa"
if "fp32" in variants:
    model = model.float()
    results["fp32"] = run(model, "fp32")
for name in ("ours", "reference"):
    path = args.out / f"{name}.npy"
    if path.exists():
        results[name] = np.load(path)
pairs = [("ours", "reference"), ("ours", "bf16"), ("ours", "table"), ("ours", "eager"), ("ours", "fp32"), ("reference", "bf16"), ("bf16", "bf16_again"),
         ("bf16", "table"), ("bf16", "eager"), ("bf16", "fp32"), ("reference", "fp32"), ("table", "fp32")]
for first, second in pairs:
    if first in results and second in results:
        print(json.dumps({"step": "pair", "a": first, "b": second, "vectors": compare(results[first], results[second]),
                          "ranking": ranking(results[first], results[second])}), flush=True)
sys.stdout.flush()
