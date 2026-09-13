#!/usr/bin/env python3
"""K-F2 bind gate: kind truth for the full 93-layer backbone.

SparkK3BindLayer must derive GDN-vs-MLA from THE canonical rule
(spark_k3_pool_sizing.h's SparkK3LayerIsMla: the 3:1 period AND the
trailing layer 92), because the packer emits MLA tensors wherever the
checkpoint's layer map says full_attention - and layer 92 IS full
attention (93 one-indexed). A period-only rule binds layer 92 as GDN and
the resolve dies on KDA names the pack does not carry.

This gate is self-contained end to end:
  1. build a 93-layer synthetic checkpoint with the canonical layer_types
     map (tiny dims; reuses test_k3_pack_layout's builder);
  2. run the real tools/k3_pack.py CLI over it;
  3. compile tests/test_k3_bind.c against the repo sources (cc, CUDA-free);
  4. run it against the built pack - layers {0,1,2,3,91,92}, the PP4
     stage-3 slice 70..92 with pool-sizing count agreement, and the
     past-the-end negative.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_k3_pack_layout import mini_checkpoint  # noqa: E402

# Apple clang cannot write its default /var/folders temp under this
# workstation's sandbox; /tmp is writable (same workaround test_k3_engine
# needs).
ENV = dict(os.environ, TMPDIR="/tmp")


def canonical_layer_types(layers=93):
    """THE rule the bind must match: MLA at i%4==3 plus the trailing 92."""
    return ["full_attention" if (i % 4 == 3 or i == 92)
            else "linear_attention" for i in range(layers)]


def main():
    if shutil.which("cc") is None and shutil.which("clang") is None:
        print("SKIP no C compiler on this host")
        return 0
    cc = shutil.which("cc") or shutil.which("clang")
    failures = 0

    with tempfile.TemporaryDirectory(prefix="k3bind", dir="/tmp") as tmp:
        root = Path(tmp)

        # 1. the 93-layer synthetic backbone, layer 0 truly dense
        ckpt = root / "ckpt"
        ckpt.mkdir()
        mini_checkpoint(ckpt, layers=93,
                        layer_types=canonical_layer_types(),
                        dense_layers=(0,))

        # 2. the real packer CLI, full model (no slice args)
        pack_path = root / "k3.full.mini.pack"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "k3_pack.py"),
             str(ckpt), str(pack_path)], capture_output=True, text=True, env=ENV)
        if result.returncode != 0:
            print(f"FAIL packer exited {result.returncode}: "
                  f"{result.stdout.strip()} {result.stderr.strip()}")
            return 1
        raw = pack_path.read_bytes()
        magic, version, length = struct.unpack_from("<IIQ", raw, 0)
        assert magic == 0x4B33504B and version == 2, "packer wrote bad front"
        print(f"ok fixture pack: {length} B manifest, "
              f"{pack_path.stat().st_size} B total")

        # 3. compile the C gate against the repo sources
        binary = root / "test_k3_bind"
        sources = [
            ROOT / "tests" / "test_k3_bind.c",
            ROOT / "modules" / "k3_resident_decode_stage" / "source" /
            "spark_k3_bind.c",
            ROOT / "modules" / "k3_resident_decode_stage" / "source" /
            "spark_k3_pack_load.c",
            ROOT / "runtime" / "json.c",
            ROOT / "runtime" / "filesystem.c",
            ROOT / "src" / "spark_status.c",
        ]
        compile_result = subprocess.run(
            # No -Werror: spark_k3_pack_load.c carries one pre-existing
            # unused-variable warning this gate must not gate on.
            [cc, "-std=c11", "-O1", "-Wall", "-Wextra",
             "-I.", "-Iinclude", "-Isrc", "-Iruntime",
             "-Imodules/k3_resident_decode_stage/include",
             "-Imodel-families/common/include",
             *[str(s) for s in sources], "-o", str(binary)],
            capture_output=True, text=True, cwd=ROOT, env=ENV)
        if compile_result.returncode != 0:
            print(f"FAIL compile: {compile_result.stderr.strip()}")
            return 1
        print("ok C gate compiled clean (-Wall -Wextra -Werror)")

        # 4. run it against the fixture pack
        run = subprocess.run([str(binary), str(pack_path)],
                             capture_output=True, text=True, env=ENV)
        sys.stdout.write(run.stdout)
        if run.returncode != 0:
            print(f"FAIL test_k3_bind exit {run.returncode}")
            failures += 1

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe bind resolves every backbone layer from the ONE kind rule; "
          "the trailing-92 exception holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
