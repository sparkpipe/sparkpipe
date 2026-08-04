#!/usr/bin/env python3
"""Host-side tests for DSV4 source fragment generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import dsv4_stage_source as fragment  # noqa: E402
from tools import dsv4_stagepack as pack  # noqa: E402


def write_source(root: Path) -> None:
    payloads = {"a": b"abcd", "b": b"01234567"}
    header = {
        name: {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0 if name == "a" else 4, len(payload) if name == "a" else 12],
        }
        for name, payload in payloads.items()
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    (root / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded + b"".join(payloads.values())
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in payloads}}),
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        output_root = root / "fragment"
        source_root.mkdir()
        write_source(source_root)
        record = type("Record", (), {"source_names": ("a",), "scale_names": ("b",)})()
        with pack.SafetensorSource(source_root) as source:
            result = fragment.write_stage_fragment(
                source, [record],
                output_root, 0, 1,
            )
        assert result["tensor_count"] == 2
        with pack.SafetensorSource(output_root) as generated:
            assert generated.read("a") == b"abcd"
            assert generated.read("b") == b"01234567"
        manifest = json.loads(
            (output_root / fragment.FRAGMENT_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["fragment_bytes"] == (output_root / fragment.FRAGMENT_NAME).stat().st_size
        shard_root = root / "shards"
        with pack.SafetensorSource(source_root) as source:
            shard_result = fragment.write_stage_shards(
                source, [record], shard_root, 0, 1
            )
        assert shard_result["source_mode"] == "shards"
        with pack.SafetensorSource(shard_root) as generated:
            assert generated.read("a") == b"abcd"
            assert generated.read("b") == b"01234567"
    print("PASS DSV4 stage source fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
