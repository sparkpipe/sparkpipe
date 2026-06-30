#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_packer_module():
    repo_root = Path(__file__).resolve().parents[1]
    packer_path = repo_root / "tools" / "glm52_b12x_resident_pack.py"
    spec = importlib.util.spec_from_file_location("glm52_b12x_resident_pack", packer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load B12x packer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def byte_value(row: int, k_group: int) -> int:
    return ((row * 17) + (k_group * 31)) & 0xff


def row_major_bytes(rows: int, k_groups: int) -> bytes:
    return bytes(
        byte_value(row, k_group)
        for row in range(rows)
        for k_group in range(k_groups)
    )


def target_offset(row: int, k_group: int, rows: int, k_groups: int) -> int:
    del rows
    k_tiles = k_groups // 4
    m_tile = row // 128
    outer_m = row % 32
    inner_m = (row % 128) // 32
    k_tile = k_group // 4
    inner_k = k_group % 4
    return (((((m_tile * k_tiles) + k_tile) * 32 + outer_m) * 4 + inner_m) * 4 + inner_k)


def assert_scale_layout(module, rows: int, k_groups: int) -> None:
    source = row_major_bytes(rows, k_groups)
    output = module.scale_bytes_to_flashinfer_static_storage(
        "test scale",
        source,
        rows,
        k_groups,
    )
    assert len(output) == len(source)
    for row in range(rows):
        for k_group in range(k_groups):
            offset = target_offset(row, k_group, rows, k_groups)
            assert output[offset] == byte_value(row, k_group)


def main() -> int:
    module = load_packer_module()
    assert module.ABI_VERSION == 3
    assert_scale_layout(module, 128, 4)
    assert_scale_layout(module, 256, 8)
    try:
        module.scale_bytes_to_flashinfer_static_storage("bad rows", b"\0" * 64, 64, 1)
    except module.PackFailure:
        pass
    else:
        raise AssertionError("unaligned scale shape was accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
