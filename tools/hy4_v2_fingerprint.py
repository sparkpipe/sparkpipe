#!/usr/bin/env python3
"""Compare placed v1 and re-emitted v2 hy4 rank packs plane by plane.

Usage:
  python3 hy4_v2_fingerprint.py --rank 7 --old-dir DIR --new-dir DIR \
      [--samples N] [--full]

Every plane must carry the same dtype/shape in both packs and byte-equal
content at each pack's own declared offsets. The default samples across
the slice classes plus the router gate class; --full compares all planes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def load_header(path: Path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


def plane_bytes(path: Path, base: int, start: int, nbytes: int):
    with open(path, "rb") as fh:
        fh.seek(base + start)
        return fh.read(nbytes)


def classify(name: str) -> str:
    if "mtp_layers" in name:
        return "mtp"
    if "experts.gate_up_proj" in name:
        return "gather-expert-payload"
    if "experts.gate_up_proj_scale" in name:
        return "gather-expert-scale"
    if "experts.down_proj" in name and "_scale" not in name:
        return "gather-expert-down"
    if "o_proj.weight_scale" in name or "q_b_proj.weight_scale" in name \
            or "kv_b_proj.weight_scale" in name or "wq_b.weight_scale" in name:
        return "replicated-scale-contract"
    if "gate.weight" in name or "e_score_correction_bias" in name:
        return "router-gate"
    if "learnable_sink" in name or "hc_" in name:
        return "hyper-f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "vocab-slice"
    return "replicated-or-sliced"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)
    stem = f"model-fp8-tp16-rank-{args.rank:02d}.safetensors"
    old_header, old_base = load_header(args.old_dir / stem)
    new_header, new_base = load_header(args.new_dir / stem)
    names = sorted(k for k in new_header if k != "__metadata__")
    if set(names) != set(k for k in old_header if k != "__metadata__"):
        raise SystemExit("plane inventories disagree between packs")
    rng = __import__("random").Random(0x445934)
    pool = [n for n in names if "mtp_layers" not in n]
    picked = set(rng.sample(pool, min(args.samples, len(pool))))
    picked.add("model.hc_head.hc_head_scale")
    for name in names:
        if "mlp.gate.weight" in name or "e_score_correction_bias" in name:
            picked.add(name)
    for name in names:
        if args.full or name in picked:
            o = old_header[name]
            n = new_header[name]
            if o["dtype"] != n["dtype"] or o["shape"] != n["shape"]:
                raise SystemExit(
                    f"GEOMETRY MISMATCH {name}: {o['dtype']}{o['shape']} vs "
                    f"{n['dtype']}{n['shape']}")
            ob = o["data_offsets"][1] - o["data_offsets"][0]
            nb = n["data_offsets"][1] - n["data_offsets"][0]
            if ob != nb:
                raise SystemExit(f"EXTENT MISMATCH {name}: {ob} vs {nb}")
            old = plane_bytes(args.old_dir / stem, old_base,
                              o["data_offsets"][0], ob)
            new = plane_bytes(args.new_dir / stem, new_base,
                              n["data_offsets"][0], nb)
            if old != new:
                dif = next((i for i in range(ob) if old[i] != new[i]), -1)
                raise SystemExit(
                    f"CONTENT MISMATCH {name} ({classify(name)}) bytes={ob} "
                    f"first_diff={dif}")
            print(f"fingerprint {name} [{classify(name)}] "
                  f"sha256={hashlib.sha256(new).hexdigest()[:16]} EQUAL")
    print(f"FINGERPRINT PASS rank {args.rank:02d}: "
          f"{len(picked) if not args.full else len(names)} planes equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
