#!/usr/bin/env python3
"""Compare OUR engine's FP8-target taps against the BF16 reference's context
hiddens at the same prompt positions.

Ours: /tmp/our_taps.bin = ctxrun taps [positions, 5, H] bf16 (raw 5-layer).
Theirs: /tmp/vllmin_0.pt target_hidden_states = the COMBINED (fc-applied)
rows from the BF16 verify/prefill forward. We apply our fc to our raw taps
and compare per position: relative L2 of fc(cat(taps)) vs their combined."""
import numpy as np
import torch
from safetensors import safe_open

DRAFTER = "/home/spark3/sparkdata/qwen38-dflash2-drafter"
H = 5120

with safe_open(f"{DRAFTER}/model.safetensors", framework="pt") as f:
    fc = f.get_tensor("fc.weight").float()   # [5120, 5*5120]

ours_raw = np.fromfile("/tmp/our_taps.bin", dtype=np.uint16).astype(np.uint32) << 16
ours = torch.from_numpy(ours_raw.view(np.float32).copy()).view(-1, 5, H)
print("our taps rows:", ours.shape[0])

d = torch.load("/tmp/vllmin_0.pt", weights_only=False, map_location="cpu")
theirs = d["target_hidden_states"].float()
print("their combined rows:", theirs.shape[0])

n = min(ours.shape[0], theirs.shape[0])
# per-position: fc @ cat(5 taps) vs their combined row
rel = []
for p in range(n):
    cat = ours[p].reshape(-1)
    comb = cat @ fc.T
    denom = theirs[p].norm().clamp_min(1e-9)
    rel.append(((comb - theirs[p]).norm() / denom).item())
rel = np.array(rel)
print(f"per-position rel-L2 of fc(our fp8 taps) vs their bf16 combined:")
print(f"  mean={rel.mean():.4f} median={np.median(rel):.4f} p90={np.percentile(rel,90):.4f} max={rel.max():.4f}")
print("first 5 positions:", " ".join(f"{r:.3f}" for r in rel[:5]))
