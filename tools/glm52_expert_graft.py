#!/usr/bin/env python3
"""Graft routed-expert entries from one glm52 rank pack onto the spine of another.

The glm52 front end of tools/glm5_next_expert_graft.py: the same streaming
copy, refusals, read-back and spine digest, over the glm52 v3 pack layout
(magic 0x32534C47, 264-byte header, expert kinds 22/23, 1,344 entries at
TP16). The output keeps the spine pack's directory order, spine entries and
spine bytes, takes the routed-expert entries verbatim from --expert-pack, and
rewrites the header's expert codec, model revision (offset 96), build
contract (offset 161, --contract-sha256) and pack recipe (offset 225, the
SHA-256 of the graft recipe: both headers and directories, codec, revision,
contract). The source-config digest stays the spine pack's.

--restage takes stage_count and stage_index from the expert pack instead of
refusing a difference. It exists for spine packs whose header stage fields
were written wrong while the payload verifies (spark5 rank 5 of the S1 BF16
spine, header stage 16/5); the receipt records both headers' values.

Usage (rank-local, CPU only; U2 = S1 BF16 spine + the published FP8 experts):
  nice -n 19 ionice -c3 python3 tools/glm52_expert_graft.py \\
      --spine-pack ~/sparkdata/glm53full.bf16.tp16/packs/glm53full.bf16.tp16-rank0.glm52sp \\
      --expert-pack ~/sparkdata/glm53full.fp8.tp16/packs/glm53full.fp8.tp16-rank0.glm52sp \\
      --expert-codec fp8 --tp-degree 16 --tp-rank 0 \\
      --model-revision $(python3 tools/glm52_model_contract.py --print-build-identity fp8_s1 | cut -d' ' -f1) \\
      --contract-sha256 $(python3 tools/glm52_model_contract.py --print-build-identity fp8_s1 | cut -d' ' -f2) \\
      --output ~/sparkdata/glm53full.fp8_s1.tp16/packs/glm53full.fp8_s1.tp16-rank0.glm52sp
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm5_next_expert_graft as engine  # noqa: E402
from glm52_model_contract import load_model_contract  # noqa: E402

CONTRACT = load_model_contract()
GLM52_LAYOUT = {
    "name": "glm52", "magic": 0x32534C47, "format_version": 3,
    "geometry": (CONTRACT["hidden_dimension"], CONTRACT["output_vocab_count"],
                 CONTRACT["moe_expert_count"], CONTRACT["layer_count"]),
    "codecs": {"bf16": 1, "fp8": 5, "nvfp4": 6},
    "identity": True, "receipt_kind": "sparkpipe.glm52.expert-graft-receipt.v1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spine-pack")
    parser.add_argument("--expert-pack")
    parser.add_argument("--output")
    parser.add_argument("--tp-degree", type=int)
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--expert-codec", choices=sorted(GLM52_LAYOUT["codecs"]))
    parser.add_argument("--model-revision")
    parser.add_argument("--contract-sha256")
    parser.add_argument("--restage", action="store_true")
    parser.add_argument("--arm", default="")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--chunk-bytes", type=int, default=64 << 20)
    parser.add_argument("--drop-spine-cache", action="store_true")
    parser.add_argument("--drop-expert-cache", action="store_true")
    parser.add_argument("--drop-output-cache", action="store_true")
    parser.add_argument("--spine-digest", metavar="PACK")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        if args.spine_digest:
            digest = engine.pack_spine_digest(Path(args.spine_digest), args.chunk_bytes, GLM52_LAYOUT)
            print(f"{digest}  {args.spine_digest}")
            return 0
        required = ("spine_pack", "expert_pack", "output", "tp_degree", "tp_rank", "expert_codec",
                    "model_revision", "contract_sha256")
        absent = [name for name in required if getattr(args, name) is None]
        if absent:
            parser.error("missing " + ", ".join("--" + name.replace("_", "-") for name in absent))
        receipt = engine.graft(Path(args.spine_pack), Path(args.expert_pack), Path(args.output),
                               args.tp_degree, args.tp_rank, args.expert_codec, args.model_revision,
                               args.chunk_bytes, {"spine": args.drop_spine_cache,
                                                  "expert": args.drop_expert_cache,
                                                  "output": args.drop_output_cache},
                               args.arm, args.source_commit, GLM52_LAYOUT, args.contract_sha256,
                               args.restage)
    except engine.PackFailure as error:
        print(f"GRAFT-REFUSED: {error}", file=sys.stderr)
        return 1
    print(f"GRAFT-PASS {receipt['output']['path']} bytes={receipt['output']['bytes']} "
          f"sha256={receipt['output']['sha256']} spine_digest={receipt['spine_digest']} "
          f"contract={receipt['contract_sha256']} expert_codec={receipt['expert_codec']} "
          f"tp{receipt['tp_degree']} rank {receipt['tp_rank']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
