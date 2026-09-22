#!/usr/bin/env python3
"""Shard specific dsv4_pro TP4xPP4 rank packs from a layer-slice source
pack (lane 5 restore, slice-direct plan).

tools/dsv4_tp16_stagepack.shard_pack() refuses inputs that do not cover
the complete 61-layer model. The approved restore plan reads only the
Ceph generation slices it needs (ranks 1/2/3: stage 0, layers [0,16);
rank 6: stage 1, layers [16,15) - no 865G full-pack intermediate), so
this helper reuses the sharder module's own primitives (plan_entry,
copy_payload, copy_scales, layer_slice, payload_bytes - the identical
record-selection and byte-copy math) with exactly one change: the input
completeness guard becomes a slice-aware check requiring the input's
(first_layer, layer_count) to equal the requested pp stage's backbone
range. Nothing else differs; the replicated MTP/globals path keeps the
module's behavior verbatim. The caller sha256-verifies every produced
pack against the round-10 placement table BEFORE anything is placed.

Usage:
  python3 tools/devcycle/dsv4pro_restore_ranks.py \
      --input slice0.spstage --stage 0 --ranks 1,2,3 \
      --output-root /path/to/out
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]


def load_sharder():
    path = ROOT / "tools" / "dsv4_tp16_stagepack.py"
    spec = importlib.util.spec_from_file_location("dsv4_restore_sharder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.apply_model_geometry("pro")
    module.TP_DEGREE = 4
    return module


def shard_from_slice(module, input_path: Path, output_path: Path,
                     rank: int, pp_stages: int, pp_stage: int) -> dict:
    PackFailure = module.PackFailure
    HEADER, ENTRY = module.HEADER, module.ENTRY
    first_layer, layer_count = module.layer_slice(pp_stages, pp_stage)
    with input_path.open("rb") as source:
        header_raw = source.read(HEADER.size)
        if len(header_raw) != HEADER.size:
            raise PackFailure("short stage-pack header")
        header = list(HEADER.unpack(header_raw))
        if header[0] != module.MAGIC or header[1] != module.VERSION:
            raise PackFailure("input is not a DSV4 stage pack")
        # Slice-aware guard (the only deviation from shard_pack): the input
        # must cover exactly this stage's backbone range. Replicated
        # MTP/global records may carry layers outside it by design.
        if header[9] != first_layer or header[10] != layer_count:
            raise PackFailure(
                f"input covers layers {header[9]}+{header[10]}, stage "
                f"{pp_stage} of {pp_stages} needs {first_layer}+{layer_count}")
        if rank < 0 or rank >= module.TP_DEGREE:
            raise PackFailure("tp rank outside degree")
        source.seek(header[16])
        entries = [ENTRY.unpack(source.read(ENTRY.size))
                   for _ in range(header[8])]
        plans = []
        for entry in entries:
            try:
                plans.append(
                    (module.plan_entry(entry, rank, pp_stages, pp_stage),
                     entry))
            except PackFailure as error:
                if str(error) == "filtered":
                    continue
                raise
        cursor = HEADER.size + ENTRY.size * len(plans)
        output_entries = []
        for (planned, original) in plans:
            new_entry, indices, col_start, new_scale_bytes = planned
            kind, layer, weight, rows, columns, reserved, payload, scale \
                = original
            payload_offset = cursor
            cursor += module.payload_bytes(weight, len(indices),
                                           new_entry[4])
            scale_offset = 0
            if new_scale_bytes:
                scale_offset = cursor
                cursor += new_scale_bytes
            output_entries.append(
                (new_entry[:6] + (payload_offset, scale_offset),
                 original, indices, col_start))
        header[8] = len(output_entries)
        header[9], header[10] = first_layer, layer_count
        header[16] = HEADER.size
        header[17] = cursor
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                prefix=f".{output_path.name}.",
                suffix=".tmp", dir=output_path.parent,
                delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(HEADER.pack(*header))
            for output_entry, _, _, _ in output_entries:
                temporary.write(ENTRY.pack(*output_entry))
            for output_entry, original, indices, col_start in output_entries:
                kind, layer, weight, rows, columns, reserved, payload, scale \
                    = original
                module.copy_payload(source, temporary, payload, weight,
                                    rows, columns, indices, col_start,
                                    output_entry[4])
                if output_entry[7]:
                    module.copy_scales(source, temporary, scale, weight,
                                       rows, columns, indices, col_start,
                                       output_entry[4])
            temporary.flush()
            temporary_path.replace(output_path)
    digest = module.sha256_file(output_path)
    return {"file": str(output_path), "rank": rank,
            "tp_degree": module.TP_DEGREE, "pp_stages": pp_stages,
            "pp_stage": pp_stage, "first_layer": header[9],
            "layer_count": header[10],
            "bytes": output_path.stat().st_size,
            "tensor_count": len(output_entries), "sha256": digest,
            "validated": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="layer-slice source pack (from "
                             "tools/dsv4_pro_stagepack.py --first-layer "
                             "F --layer-count C)")
    parser.add_argument("--stage", type=int, required=True,
                        choices=(0, 1, 2, 3), help="pp stage to shard")
    parser.add_argument("--ranks", required=True,
                        help="comma-separated tp ranks (e.g. 1,2,3)")
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    module = load_sharder()
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    ranks = [int(value) for value in arguments.ranks.split(",")]
    for rank in ranks:
        world_rank = arguments.stage * 4 + rank
        output = arguments.output_root / (
            f"dsv4_pro.tp4_pp4.rank{world_rank:02d}.spstage")
        result = shard_from_slice(module, arguments.input, output,
                                  rank, 4, arguments.stage)
        print(f"rank {world_rank:02d} (tp {rank}, stage "
              f"{arguments.stage}): {result['tensor_count']} tensors, "
              f"{result['bytes']} bytes")
        print(f"  sha256 {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
