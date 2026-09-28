#!/usr/bin/env python3
"""Host pack-contract validation for dsv41_flash rank packs.

Synthesizes a two-layer TP8 rank pack with the module's pack synthesizer,
writes its .experts manifest with the manifest tool, and requires the
independent contract validator (tests/test_dsv41_flash_pack_contract.c) to
accept the pair and to reject the pack once an expert payload byte changes.
The three binaries are built by make test.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNTHESIZE = ROOT / "build" / "dsv41_flash_pack_synthesize"
MANIFEST = ROOT / "build" / "dsv41_flash_experts_manifest"
VALIDATOR = ROOT / "build" / "test_dsv41_flash_pack_contract"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
ZERO_DIGEST = "0" * 64
MANIFEST_HEADER_BYTES = 16


def run(argv):
    return subprocess.run([str(item) for item in argv], capture_output=True,
                          text=True)


def main() -> int:
    for binary in (SYNTHESIZE, MANIFEST, VALIDATOR):
        if not binary.is_file():
            print(f"FAIL missing {binary.relative_to(ROOT)}; run make test")
            return 1
    with tempfile.TemporaryDirectory(prefix="dsv41-pack-contract-") as tmp:
        pack = Path(tmp) / "rank3.spstage"
        manifest = Path(tmp) / "rank3.spstage.experts"
        steps = (
            [SYNTHESIZE, pack, REVISION, ZERO_DIGEST, ZERO_DIGEST, ZERO_DIGEST,
             8, 3, "mxfp4", 2],
            [MANIFEST, pack, manifest],
        )
        for argv in steps:
            result = run(argv)
            if result.returncode != 0:
                print(result.stdout, result.stderr)
                print(f"FAIL {Path(str(argv[0])).name}")
                return 1
        result = run([VALIDATOR, pack, manifest])
        if result.returncode != 0 or "PASS" not in result.stdout:
            print(result.stdout, result.stderr)
            print("FAIL validator rejected the synthesized pack")
            return 1
        with manifest.open("rb") as handle:
            handle.seek(MANIFEST_HEADER_BYTES + 16)
            offset = struct.unpack("<Q", handle.read(8))[0]
        with pack.open("r+b") as handle:
            handle.seek(offset)
            byte = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([byte[0] ^ 0xFF]))
        result = run([VALIDATOR, pack, manifest])
        if result.returncode == 0 or "record ck128 digest" not in result.stderr:
            print(result.stdout, result.stderr)
            print("FAIL validator accepted a changed expert payload")
            return 1
    print("PASS dsv41 flash pack contract (synthesized TP8 rank 3, 2 layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
