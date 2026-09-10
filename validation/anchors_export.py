#!/usr/bin/env python3
"""Export the gemma4 anchor fixtures to raw binary for the C oracle.

The oracle (validation/spark_gemma4_reference.c) is plain C with zero
dependencies; it cannot read npz. This dumps every array of every
fixtures/<tag>_<group>.npz as little-endian raw bytes, one file per key:

    <out>/<tag>__<group>__<key>.bin

plus manifest.txt lines "<file> <rows> <cols> <code>" (code: 0 u16 bf16,
2 f32, 3 i64) so the C side needs no parsing beyond whitespace.
Run: python3 validation/anchors_export.py [fixtures_dir] [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODES = {"uint16": 0, "float32": 2, "int64": 3}


def main() -> int:
    anchors = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "anchors/fixtures"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "build/anchors_bin"
    if not anchors.is_dir():
        print(f"SPARK_FAIL anchors_dir_missing: {anchors}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(anchors.glob("*.npz")):
        tag, group = path.stem.split("_", 1)
        with np.load(path) as data:
            for key in data:
                array = np.ascontiguousarray(data[key])
                code = CODES.get(str(array.dtype))
                if code is None:
                    print(f"SPARK_FAIL dtype_unsupported: {path.name}:{key} {array.dtype}")
                    return 1
                cols = array.shape[-1] if array.ndim >= 1 else 1
                rows = array.size // cols
                name = f"{tag}__{group}__{key.replace('.', '_')}.bin"
                array.astype(array.dtype.copy(order="<")).tofile(out / name)
                lines.append(f"{name} {rows} {cols} {code}")
    (out / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported {len(lines)} arrays -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
