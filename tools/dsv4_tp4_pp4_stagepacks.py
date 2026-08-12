#!/usr/bin/env python3
"""Build the sixteen physical-rank packs for DSV4 Flash TP4 x PP4."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence


def load_sharder(repository_root: Path):
    path = repository_root / "tools" / "dsv4_tp16_stagepack.py"
    spec = importlib.util.spec_from_file_location("dsv4_parallel_sharder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sharder = load_sharder(root)
    sharder.TP_DEGREE = 4
    args.output_directory.mkdir(parents=True, exist_ok=True)
    ranks = []
    for pp_stage in range(4):
        for tp_rank in range(4):
            world_rank = pp_stage * 4 + tp_rank
            output = args.output_directory / (
                f"dsv4_flash.tp4_pp4.rank{world_rank:02d}.spstage"
            )
            result = sharder.shard_pack(
                args.input_pack, output, tp_rank, 4, pp_stage
            )
            result["world_rank"] = world_rank
            ranks.append(result)
    manifest = {
        "schema_version": 1,
        "topology": "TP4xPP4",
        "world_size": 16,
        "tp_degree": 4,
        "pp_stage_count": 4,
        "layer_slices": [[0, 11], [11, 11], [22, 11], [33, 10]],
        "microbatch_rows": 1024,
        "pipeline_microbatch_count": 4,
        "full_pipeline_request_count": 4096,
        "source_pack": str(args.input_pack),
        "ranks": ranks,
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
