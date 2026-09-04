#!/usr/bin/env python3
"""Build the sixteen physical-rank packs for Qwen38-27B TP4 x PP4.

World rank r maps to (pp_stage = r // 4, tp_rank = r % 4) over a
16/16/16/16 layer split of the 64-layer stack; stage three carries the
head + MTP chain and stage zero the embedding, per the stagepack's own
inventory rules. Each rank pack is re-verified with the packer's own
verify() after the build. Placement map: rank r -> spark{hex r}.

The 27B stagepack module is imported and driven through its convert()
/ verify() entry points (same DRY pattern as the dsv4 TP4PP4 builder);
no shell is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LAYER_COUNT = 64
PP_STAGES = 4
TP_DEGREE = 4
STAGE_LAYERS = LAYER_COUNT // PP_STAGES  # 16/16/16/16


def load_packer(here: Path):
    spec = importlib.util.spec_from_file_location(
        "qwen38_27b_stagepack", here / "qwen38_27b_stagepack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="safetensors checkpoint directory")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only: no packs are written")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    packer = load_packer(here)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    ranks = []
    for world_rank in range(PP_STAGES * TP_DEGREE):
        pp_stage, tp_rank = divmod(world_rank, TP_DEGREE)
        first_layer = pp_stage * STAGE_LAYERS
        layer_count = STAGE_LAYERS
        output = args.output_directory / (
            f"qwen38_27b.tp4_pp4.rank{world_rank:02d}.spstage")
        receipt = {
            "kind": "sparkpipe.qwen38_27b.stagepack-receipt.v1",
            "tool": "tools/qwen38_27b_tp4pp4_stagepacks.py",
            "checkpoint": str(args.checkpoint),
            "world_rank": world_rank,
            "pp_stage": pp_stage,
            "tp_rank": tp_rank,
            "weight_formats": {"projections": "bf16", "gdn_a_log_dt_bias": "f32"},
        }
        try:
            result = packer.convert(args.checkpoint, output, first_layer,
                                    layer_count, receipt, args.dry_run,
                                    TP_DEGREE, tp_rank)
        except packer.PackFailure as error:
            print(f"rank{world_rank:02d} BUILD FAILED: {error}", flush=True)
            return 1
        if not args.dry_run:
            try:
                checked = packer.verify(output)
            except packer.PackFailure as error:
                print(f"rank{world_rank:02d} VERIFY FAILED: {error}", flush=True)
                return 1
            print(f"rank{world_rank:02d} (stage{pp_stage},tp{tp_rank}) ok "
                  f"tensors={checked['tensor_count']} bytes={checked['bytes']}",
                  flush=True)
            packer.write_receipt(result, Path(str(output) + ".receipt.json"),
                                 suffix=None)
        else:
            print(f"rank{world_rank:02d} (stage{pp_stage},tp{tp_rank}) "
                  f"dry-plan ok tensors={result.get('tensor_count')}",
                  flush=True)
        ranks.append({"world_rank": world_rank, "pp_stage": pp_stage,
                      "tp_rank": tp_rank, "first_layer": first_layer,
                      "layer_count": layer_count,
                      "output": str(output) if not args.dry_run else None})
    if not args.dry_run:
        manifest = {
            "schema_version": 1,
            "topology": "TP4xPP4",
            "world_size": PP_STAGES * TP_DEGREE,
            "tp_degree": TP_DEGREE,
            "pp_stage_count": PP_STAGES,
            "layer_slices": [[p * STAGE_LAYERS, STAGE_LAYERS]
                             for p in range(PP_STAGES)],
            "source_checkpoint": str(args.checkpoint),
            "ranks": ranks,
        }
        manifest_path = args.output_directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # surface module-level PackFailure too
        print(f"qwen38_27b_tp4pp4_stagepacks: {error}", file=sys.stderr)
        sys.exit(1)
