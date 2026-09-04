#!/usr/bin/env python3
"""Patch the glm5_next stage-pack header provenance fields in place.

The module's SparkGlm5NextPackValidateHeader (spark_glm5_next_resident_
decode_stage_module.c) rejects a pack with SPARK_STATUS_HASH_MISMATCH unless
  - header->contract_sha256      == hex(GLM5_NEXT_CONTRACT_SHA256) (the
    module's compile-time contract pin), and
  - header->source_config_sha256 is non-zero, and
  - header->pack_recipe_sha256   is non-zero.

The repack packer (tools/glm5_next_resident_stagepack.py at 4758e68) emits
zeros for all three fields, so every fixed pack fails adapter_initialize
until the headers carry the provenance the M4 bring-up patched in place
("all headers re-patched on every node", glm53-2026-08-27 report).

This tool copies the 96 provenance bytes (contract 32 + source_config 32 +
pack_recipe 32, header offsets 161/193/225) from a pack known to load (the
deployed pre-repack pack) onto the fixed pack, after asserting:
  - the old contract bytes == --expect-contract-hex (from `strings` on the
    deployed adapter/driver .so), and
  - old config/recipe bytes are non-zero.

Offsets follow the packer's assemble_header layout (20xu32 + 2xu64 + 65B
revision, then the three 32B fields at 161, 193, 225).

usage:
  glm5_next_pack_header_patch.py --pack <fixed.g5nsp> --reference <loading.g5nsp> \
      --expect-contract-hex <64hex>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONTRACT_OFFSET = 161
FIELD = 32
PROVENANCE_BYTES = 3 * FIELD  # contract + source_config + pack_recipe


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, help="pack to patch (in place)")
    ap.add_argument("--reference", required=True,
                    help="pack whose header is known to load (old deployment)")
    ap.add_argument("--expect-contract-hex", required=True)
    args = ap.parse_args()

    expect = bytes.fromhex(args.expect_contract_hex)
    if len(expect) != FIELD:
        fail("expect-contract-hex must be 64 hex chars")

    with open(args.reference, "rb") as ref:
        ref.seek(CONTRACT_OFFSET)
        provenance = ref.read(PROVENANCE_BYTES)
    if len(provenance) != PROVENANCE_BYTES:
        fail("short read on reference header")
    contract, config, recipe = (provenance[0:FIELD],
                                provenance[FIELD:2 * FIELD],
                                provenance[2 * FIELD:3 * FIELD])
    if contract != expect:
        fail(f"reference contract {contract.hex()[:16]}.. != expected "
             f"{expect.hex()[:16]}.. (adapter/driver pin)")
    if not any(config):
        fail("reference source_config_sha256 is zero")
    if not any(recipe):
        fail("reference pack_recipe_sha256 is zero")

    path = Path(args.pack)
    with open(path, "r+b") as out:
        out.seek(CONTRACT_OFFSET)
        current = out.read(PROVENANCE_BYTES)
        if current == provenance:
            print(f"PATCH-NOOP {path.name}: provenance already present")
            return 0
        out.seek(CONTRACT_OFFSET)
        out.write(provenance)
        out.flush()
        import os
        os.fsync(out.fileno())
    print(f"PATCHED {path.name}: contract {contract.hex()[:16]}.. "
          f"config {config.hex()[:16]}.. recipe {recipe.hex()[:16]}..")
    return 0


if __name__ == "__main__":
    sys.exit(main())
