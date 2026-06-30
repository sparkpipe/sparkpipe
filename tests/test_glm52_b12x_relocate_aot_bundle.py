#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tool = root / "tools" / "glm52_b12x_relocate_aot_bundle.py"
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary) / "generated"
        generated.mkdir()
        (generated / "tvm_ffi_flags.mk").write_text(
            "TVM_FFI_CFLAGS := -I/home/spark2/runtime/tvm_ffi/include\n",
            encoding="utf-8",
        )
        (generated / "runtime_link_args.txt").write_text(
            "/home/spark2/runtime/libcute_dsl_runtime.so /home/spark2/runtime/libtvm_ffi.so\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--generated-dir",
                str(generated),
                "--spark-user",
                "spark7",
            ],
            check=True,
        )
        assert "/home/spark7/" in (generated / "tvm_ffi_flags.mk").read_text(encoding="utf-8")
        assert "/home/spark7/" in (generated / "runtime_link_args.txt").read_text(encoding="utf-8")
        assert "/home/spark2/" not in (generated / "runtime_link_args.txt").read_text(encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
