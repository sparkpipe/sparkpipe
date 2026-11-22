#!/usr/bin/env python3
"""Compare the K3.1 Pro DSpark drafter module's context rows with the Python drafter and with float64.

    k31_dspark_context_compare.py --checkpoint DIR --target SHARD --out DIR --rows N [--first P]

`build/k31_dspark_context_probe` writes taps.bin (N rows of five 7168-wide bf16 taps) and rows0..4.bin (the module's
bf16 context rows per layer, kv_lora_rank + qk_rope_head_dim wide, positions P..P+N-1). This recomputes the rows with K3DsparkDrafter.context and in
float64 from the same weights, and prints per layer the worst row's relative L2 error of the module and of the Python
drafter against float64. The module passes when its error is no larger than bf16 rounding (below 0.01) on every layer.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import rms  # noqa: E402
from k3_dspark_drafter import K3DsparkDrafter  # noqa: E402
from latent_block_drafter import rotate_pairs  # noqa: E402

NAMES = {"embed": "language_model.model.embed_tokens.weight", "head": "language_model.lm_head.weight", "norm": "language_model.model.norm.weight"}


def bf16_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16).view(*shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--first", default=100, type=int)
    options = parser.parse_args()
    drafter = K3DsparkDrafter(options.checkpoint, options.target, NAMES, "cuda")
    width = drafter.kv_rank + drafter.rope_dim
    taps = bf16_file(options.out / "taps.bin", (options.rows, len(drafter.layer_ids) * drafter.fc.shape[0])).cuda()
    positions = torch.arange(options.first, options.first + options.rows, device="cuda")
    worst = 0.0
    with torch.no_grad():
        reference, _ = drafter.context(taps, positions)
        features = rms(taps.double() @ drafter.fc.double().t(), drafter.hidden_norm, drafter.eps).double()
        for index, layer in enumerate(drafter.layers):
            latent = features @ layer["self_attn.kv_a_proj_with_mqa"].double().t()
            truth = torch.cat((rms(latent[:, :drafter.kv_rank], layer["self_attn.kv_a_layernorm"], drafter.eps).double(),
                               rotate_pairs(latent[:, drafter.kv_rank:].float(), positions, drafter.inverse, drafter.rotary_gain).double()), -1).cpu()
            module = bf16_file(options.out / f"rows{index}.bin", (options.rows, width)).double()
            python = reference[index].double().cpu()
            module_error = ((module - truth).norm(dim=-1) / truth.norm(dim=-1)).max().item()
            python_error = ((python - truth).norm(dim=-1) / truth.norm(dim=-1)).max().item()
            worst = max(worst, module_error)
            print(f"layer {index}: module {module_error:.5f} python {python_error:.5f} (worst row, relative L2 against float64)")
    print(("PASS" if worst < 0.01 else "FAIL") + f" worst module error {worst:.5f}")
    return 0 if worst < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
