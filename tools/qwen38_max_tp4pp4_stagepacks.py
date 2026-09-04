#!/usr/bin/env python3
"""Build the sixteen physical-rank packs for Qwen38-Max TP4 x PP4.

Conforming per-rank sharding (operator design law): every pack file
maps to exactly one rank. World rank r maps to (pp_stage = r // 4,
tp_rank = r % 4) over a 23/23/23/23 layer split of the 92-layer stack;
stage three carries the head + MTP chain and stage zero the embedding.
Packs emit the v2 128-byte header carrying tp_degree/tp_rank; the max
module reader rework (tp-aware ResolvedShape + config guard) is the
max lane's ticket. Placement map: rank r -> spark{hex r}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LAYER_COUNT = 92
PP_STAGES = 4
TP_DEGREE = 4
STAGE_LAYERS = LAYER_COUNT // PP_STAGES  # 23/23/23/23


def load_packer(here: Path):
    spec = importlib.util.spec_from_file_location(
        "qwen38_stagepack", here / "qwen38_stagepack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="safetensors checkpoint directory")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expert-codec", choices=("fp8", "nvfp4"), default="nvfp4")
    parser.add_argument("--only-ranks", type=int, nargs="*", default=None,
                        help="rebuild only these world ranks (default: all 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only: no packs are written")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    packer = load_packer(here)
    packer.EXPERT_CODEC = args.expert_codec
    args.output_directory.mkdir(parents=True, exist_ok=True)
    ranks = []
    for world_rank in range(PP_STAGES * TP_DEGREE):
        if args.only_ranks is not None and world_rank not in args.only_ranks:
            continue
        pp_stage, tp_rank = divmod(world_rank, TP_DEGREE)
        first_layer = pp_stage * STAGE_LAYERS
        layer_count = STAGE_LAYERS
        output = args.output_directory / (
            f"qwen38_max.tp4_pp4.rank{world_rank:02d}.spstage")
        receipt = {
            "kind": "sparkpipe.qwen38.stagepack-receipt.v1",
            "tool": "tools/qwen38_max_tp4pp4_stagepacks.py",
            "checkpoint": str(args.checkpoint),
            "expert_codec": args.expert_codec,
            "world_rank": world_rank,
            "pp_stage": pp_stage,
            "tp_rank": tp_rank,
        }
        try:
            result = packer.convert(args.checkpoint, output, first_layer,
                                    layer_count, receipt, args.dry_run,
                                    TP_DEGREE, tp_rank)
        except packer.PackFailure as error:
            print(f"rank{world_rank:02d} BUILD FAILED: {error}", flush=True)
            return 1
        if not args.dry_run:
            print(f"rank{world_rank:02d} (stage{pp_stage},tp{tp_rank}) ok "
                  f"tensors={result['tensor_count']} bytes={result['bytes']} "
                  f"sha={result.get('output_sha256', '')[:16]}", flush=True)
            packer.write_receipt(result, Path(str(output) + ".receipt.json"),
                                 suffix=None)
        else:
            print(f"rank{world_rank:02d} (stage{pp_stage},tp{tp_rank}) "
                  f"dry-plan ok tensors={result.get('tensor_count')} "
                  f"bytes={result.get('bytes')}", flush=True)
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
            "expert_codec": args.expert_codec,
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
    except Exception as error:
        print(f"qwen38_max_tp4pp4_stagepacks: {error}", file=sys.stderr)
        sys.exit(1)
