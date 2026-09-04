#!/usr/bin/env python3
"""Where K3's 104B active parameters actually sit, and what that costs on GB10.

Written because a proposal to factorise the routed experts turned on the claim
that attention is 42B of the 104B and 83% of the per-token bytes. Enumerating
directly rather than by residual puts attention at 36.2B and 47%, and moves the
target: KDA holds 85% of attention weight, not MLA.

The total lands at 104.62B against Moonshot's published 104.2B, 0.4% - which is
what makes the split trustworthy. A residual method attributes every error to
whatever it did not enumerate.
"""
import sys

H, KV, QL, NOPE, ROPE, V, HEADS = 7168, 512, 1536, 128, 64, 128, 96
EXPERTS, TOP_K, EH, EI, SHARED = 896, 16, 3584, 3072, 2
LAYERS, MLA, KDA, MOE, VOCAB = 93, 24, 69, 92, 163840
QKV = HEADS * 128
KDIM = 128
BW, ETA = 273e9, 0.80          # docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md


def kda_layer():
    return (3 * H * QKV        # q, k, v
            + H * QKV          # full-rank output gate
            + QKV * H          # out projection
            + H * KDIM + KDIM * QKV + H * HEADS)


def mla_layer():
    return (H * QL + QL * HEADS * (NOPE + ROPE)
            + H * (KV + ROPE) + KV * HEADS * (NOPE + V)
            + H * HEADS * V + HEADS * V * H)


def main():
    parts = {
        "routed (active)": TOP_K * 3 * EI * EH * MOE,
        "shared experts": 3 * (EI * SHARED) * H * MOE,
        "latent projections": 2 * H * EH * MOE,
        "router": H * EXPERTS * MOE,
        "embed + head": 2 * VOCAB * H,
        "attention": kda_layer() * KDA + mla_layer() * MLA,
    }
    total = sum(parts.values())
    print("active parameters")
    for name, value in parts.items():
        print(f"  {name:20s} {value/1e9:7.2f} B  {100*value/total:5.1f}%")
    print(f"  {'total':20s} {total/1e9:7.2f} B   (published 104.2B)")

    print("\nattention, by kind")
    print(f"  KDA {kda_layer()*KDA/1e9:6.2f} B over 69 layers")
    print(f"  MLA {mla_layer()*MLA/1e9:6.2f} B over 24 layers")
    print("\n  KDA per layer:")
    for name, value in [("q,k,v", 3*H*QKV), ("output gate", H*QKV),
                        ("out projection", QKV*H),
                        ("decay + beta", H*KDIM + KDIM*QKV + H*HEADS)]:
        print(f"    {name:16s} {value/1e6:7.1f} M  {100*value/kda_layer():5.1f}%")

    effective = BW * ETA
    attention = parts["attention"] * 1.0                     # fp8
    routed = parts["routed (active)"] * 0.5                   # mxfp4, as trained
    rest = (parts["shared experts"] + parts["latent projections"]) * 1.0
    bytes_total = attention + routed + rest
    print(f"\nbytes per token, fp8 attention and mxfp4 routed experts")
    print(f"  attention      {attention/2**30:6.1f} GB  {100*attention/bytes_total:4.1f}%")
    print(f"  routed         {routed/2**30:6.1f} GB  {100*routed/bytes_total:4.1f}%")
    print(f"  shared+latent  {rest/2**30:6.1f} GB  {100*rest/bytes_total:4.1f}%")
    print(f"  total          {bytes_total/2**30:6.1f} GB"
          f"  -> {bytes_total/effective*1e3:.0f} ms  -> {effective/bytes_total:.2f} tok/s at B1")
    print(f"\nat {effective/1e9:.0f} GB/s effective. Batch and speculation amortise this;"
          f"\nnothing about a single token's weight read does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
