#!/usr/bin/env python3
"""Build the TP4xPP4 pack set for Qwen3.8-2.4T-A95B: 4 PP-stage packs + the
16-rank deployment manifest.

TOPOLOGY REALITY (checked against the module, not assumed): the qwen38_max
resident-decode-stage loader validates FULL-WIDTH tensor shapes
(SparkQwen38MaxModuleValidateEntry -> SparkQwen38MaxStagePackResolvedShape)
and its TP kernels slice heads/experts at RUN time from the resident
full-width buffers (SparkQwen38MaxLaunchGroupedExpertLinear offsets the
expert payload by tp_rank * experts_per_rank; the attention kernels index
heads and KV pages by tp_rank the same way). So a TP4 group shares ONE pack
file: world_rank = pp_stage * 4 + tp_rank, ranks of a stage load the same
bytes. Rank-local pre-sharded packs would fail pack_entry_invalid at load
and mis-index the expert buffer at execution. The manifest records each
rank against its stage pack so a deployment can lay the files out per rank.

Stage plan: 92 layers as 23/23/23/23. The 3:1 GDN/full period is NOT aligned
to 23, so stages carry 5/6/6/6 full-attention layers (18/17/17/17 GDN); the
packer's inventory arithmetic handles that. The last stage additionally
carries embedding, final norm, LM head and the MTP layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "tools" / "qwen38_stagepack.py"
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen38_authoritative.json"

TP_DEGREE = 4
PP_STAGE_COUNT = 4
WORLD_SIZE = TP_DEGREE * PP_STAGE_COUNT
LAYER_SLICES = [[0, 23], [23, 23], [46, 23], [69, 23]]


def build_one(checkpoint: Path, output_dir: Path, pp_stage: int,
              contract: Path) -> dict:
    first_layer, layer_count = LAYER_SLICES[pp_stage]
    output = output_dir / f"qwen38max.tp4_pp4.stage{pp_stage}.spstage"
    command = [
        sys.executable, str(PACKER),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--first-layer", str(first_layer),
        "--layer-count", str(layer_count),
        "--contract", str(contract),
    ]
    print(f"stage{pp_stage}: {' '.join(command)}", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"stage{pp_stage}: packer exited {result.returncode}")
    receipt = json.loads(Path(str(output) + ".receipt.json").read_text())
    return {
        "pp_stage": pp_stage,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "pack": str(output),
        "bytes": receipt["bytes"],
        "sha256": receipt["output_sha256"],
        "tensor_count": receipt["tensor_count"],
        "source_index_sha256": receipt["source_index_sha256"],
        "source_config_sha256": receipt["source_config_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--jobs", type=int, default=2,
                        help="concurrent pack builds (cluster rule: max 2)")
    parser.add_argument("--hosts", type=str, default="",
                        help="comma-separated spark hosts in world_rank order "
                             "(rank = pp_stage * 4 + tp_rank); optional")
    args = parser.parse_args()

    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()] if args.hosts else []
    if hosts and len(hosts) != WORLD_SIZE:
        parser.error(f"--hosts needs {WORLD_SIZE} entries in world_rank order, got {len(hosts)}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 2))) as pool:
        stages = list(pool.map(
            lambda stage: build_one(args.checkpoint, args.output_directory, stage,
                                    args.contract),
            range(PP_STAGE_COUNT)))

    ranks = []
    for stage in stages:
        for tp_rank in range(TP_DEGREE):
            world_rank = stage["pp_stage"] * TP_DEGREE + tp_rank
            ranks.append({
                "world_rank": world_rank,
                "pp_stage": stage["pp_stage"],
                "tp_rank": tp_rank,
                "host": hosts[world_rank] if hosts else "",
                "pack": stage["pack"],
                "pack_sha256": stage["sha256"],
                "pack_bytes": stage["bytes"],
                "load_note": "stage pack is shared by the TP group; the module "
                             "slices heads/experts at run time (tp_rank env)",
            })

    manifest = {
        "schema_version": 1,
        "model": "Qwen/Qwen3.8-2.4T-A95B",
        "topology": "TP4xPP4",
        "world_size": WORLD_SIZE,
        "tp_degree": TP_DEGREE,
        "pp_stage_count": PP_STAGE_COUNT,
        "layer_slices": LAYER_SLICES,
        "checkpoint": str(args.checkpoint),
        "contract_sha256": stages[0].get("source_config_sha256"),
        "source_index_sha256": stages[0]["source_index_sha256"],
        "tp_sharding_note": "packs are whole PP-stage slices replicated across "
                            "the TP group; qwen38_max module loads full-width "
                            "tensors and slices heads/experts by tp_rank at run "
                            "time (kernel-time TP, see module .c/.cu)",
        "packs": stages,
        "ranks": ranks,
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest {manifest_path}")
    for rank in ranks:
        print(f"rank{rank['world_rank']:02d} pp{rank['pp_stage']} tp{rank['tp_rank']} "
              f"host={rank['host'] or '-'} pack={Path(rank['pack']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
