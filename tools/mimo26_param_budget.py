#!/usr/bin/env python3
"""MiMo 2.6 sizing: where the parameters sit and what each topology costs on GB10.

Census-driven (model-families/mimo26/census/*.json, measured from the shard
headers - not config inference). Answers the lane-7 topology question
(TP4/TP8/TP16) against the per-node budgets TOTAL_MIB 9792 / DEVICE_MIB 6400,
the 8-Spark lane envelope (78,336 MiB summed) and the fleet-shared weightd
arena (28.5 GiB device per node, shared with other lanes - only bounded
working sets attach, never the whole rank pack).

Served set = text spine + routed experts at native codecs (quality law:
mxfp4 experts + fp8 spine + bf16 o_proj/embed, byte-identical, no requant).
The MTP head, vision tower, audio encoder and speech embeddings are censored
out of the v1 budget (documented regions in the census, not stripped weights).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "model-families" / "mimo26" / "census"

TOTAL_MIB = 9792
DEVICE_MIB = 6400
ENVELOPE_SPARKS = 8
WEIGHTD_ARENA_GIB = 28.5
BANDWIDTH = 273e9
EFFICIENCY = 0.80

SPINE_REPLICATED = ("layer.router", "layer.router_bias", "layer.norm",
                    "layer.sink_bias", "final_norm")
EXPERT_CLASSES = ("layer.expert", "layer.expert_scale")
OUT_OF_SCOPE = ("mtp.eh_proj", "mtp.mlp", "mtp.norms", "mtp.self_attn",
                "mtp.self_attn.sink", "visual", "audio_encoder",
                "speech_embeddings")


def pattern_bytes(census):
    return {k: v["bytes"] for k, v in census["patterns"].items()}


def load(tag):
    return json.loads((CENSUS / f"{tag}_census.json").read_text())


def model_rows(tag, census):
    b = pattern_bytes(census)
    served = sum(v for k, v in b.items() if k not in OUT_OF_SCOPE)
    experts = sum(b.get(k, 0) for k in EXPERT_CLASSES)
    spine = served - experts
    replicated = sum(b.get(k, 0) for k in SPINE_REPLICATED)
    sliced = sum(b.get(k, 0) for k in ("layer.qkv", "layer.qkv_scale",
                                       "layer.o_proj", "layer.dense_mlp",
                                       "layer.dense_mlp_scale",
                                       "embed_tokens", "lm_head"))
    oos = sum(b.get(k, 0) for k in OUT_OF_SCOPE)
    cfg = census["config"]
    full_layers = sum(1 for kind in cfg["hybrid_layer_pattern"] if kind == 0)
    return {
        "tag": tag,
        "hidden": cfg["hidden_size"],
        "layers": cfg["num_hidden_layers"],
        "full_layers": full_layers,
        "swa_layers": cfg["num_hidden_layers"] - full_layers,
        "full_kv_heads": cfg["num_key_value_heads"],
        "swa_kv_heads": cfg["swa_num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "v_head_dim": cfg["v_head_dim"],
        "experts": cfg["n_routed_experts"],
        "topk": cfg["num_experts_per_tok"],
        "expert_gib": experts / 2**30,
        "spine_gib": spine / 2**30,
        "replicated_gib": replicated / 2**30,
        "sliced_gib": sliced / 2**30,
        "oos_gib": oos / 2**30,
        "total_gib": census["totals"]["payload_bytes"] / 2**30,
    }


def kv_bytes_per_token_per_rank(row, tp):
    full = row["full_layers"] * row["full_kv_heads"] * (row["head_dim"] + row["v_head_dim"]) * 2
    return full / tp


def kv_swa_window_bytes_per_rank(row, tp, sequences=1):
    swa = row["swa_layers"] * row["swa_kv_heads"] * (row["head_dim"] + row["v_head_dim"]) * 2 * 128
    return swa * sequences / tp


def topology_table(row):
    print(f"\n== {row['tag']} (hidden {row['hidden']}, {row['layers']} layers, "
          f"{row['experts']} experts top-{row['topk']}) ==")
    print(f"   measured bytes: spine {row['spine_gib']:.2f} GiB "
          f"(replicated {row['replicated_gib']:.3f}), experts {row['expert_gib']:.2f} GiB, "
          f"out-of-scope {row['oos_gib']:.2f} GiB, total {row['total_gib']:.2f} GiB")
    for tp in (4, 8, 16):
        spine_rank = row["replicated_gib"] + row["sliced_gib"] / tp
        experts_rank = row["expert_gib"] / tp
        kv_32k = (kv_bytes_per_token_per_rank(row, tp) * 32768
                  + kv_swa_window_bytes_per_rank(row, tp)) / 2**20
        kv_128k = (kv_bytes_per_token_per_rank(row, tp) * 131072
                   + kv_swa_window_bytes_per_rank(row, tp)) / 2**20
        headroom = DEVICE_MIB / 1024 - spine_rank - kv_32k / 1024
        envelope = tp * TOTAL_MIB
        fit = "FITS" if headroom > 1.0 else "TIGHT/OVER"
        envelope_ok = "within" if tp <= ENVELOPE_SPARKS else "EXCEEDS"
        print(f"   TP{tp:<2} rank NVMe {spine_rank + experts_rank:7.2f} GiB "
              f"(spine {spine_rank:5.2f} resid + experts {experts_rank:6.2f}) | "
              f"device spine+KV32k {spine_rank + kv_32k/1024:5.2f} GiB -> "
              f"{headroom:5.2f} GiB left {fit} | envelope {envelope/1024:.1f} GiB {envelope_ok}")
    active_experts = row["topk"] * row["expert_gib"] / row["experts"]
    bytes_token = row["spine_gib"] + active_experts
    tok_s = BANDWIDTH * EFFICIENCY / (bytes_token * 2**30)
    print(f"   decode floor: {bytes_token:.2f} GiB/token (spine all + top-{row['topk']} experts) "
          f"-> {tok_s:.2f} tok/s at B1, {BANDWIDTH*EFFICIENCY/1e9:.0f} GB/s effective")


def main() -> int:
    pro = model_rows("pro", load("pro"))
    flash = model_rows("flash", load("flash"))
    print("MiMo 2.6 lane-7 sizing (GB10: %d GB/s x%.2f; node TOTAL %d MiB / DEVICE %d MiB; "
          "weightd arena %.1f GiB shared fleet-wide)" % (BANDWIDTH / 1e9, EFFICIENCY,
                                                         TOTAL_MIB, DEVICE_MIB, WEIGHTD_ARENA_GIB))
    for row in (pro, flash):
        topology_table(row)
    print("\nDecision:")
    print("  pro   -> TP8 : 8 nodes = the full lane envelope (78,336 MiB), rank pack ~%.1f GiB NVMe,"
          % ((pro["replicated_gib"] + pro["sliced_gib"] / 8 + pro["expert_gib"] / 8)))
    print("           device spine ~%.2f GiB + KV leaves >2 GiB workspace/pool inside 6400 MiB;"
          % (pro["replicated_gib"] + pro["sliced_gib"] / 8))
    print("           TP16 would exceed the 8-Spark envelope; TP4 experts/rank %.1f GiB is fine on"
          % (pro["expert_gib"] / 4))
    print("           NVMe but spine+KV per node crowds DEVICE_MIB.")
    print("  flash -> TP4 : rank pack ~%.1f GiB NVMe, device spine ~%.2f GiB, half the envelope;"
          % ((flash["replicated_gib"] + flash["sliced_gib"] / 4 + flash["expert_gib"] / 4),
             flash["replicated_gib"] + flash["sliced_gib"] / 4))
    print("           TP8 also fits (quarter pack each) if co-residency with the pro arm matters.")
    print("  smoke working set (weightd shared arena %.1f GiB/node, fraction only): "
          "8 experts/layer + spine slice stays under ~2 GiB device per rank on both arms."
          % WEIGHTD_ARENA_GIB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
