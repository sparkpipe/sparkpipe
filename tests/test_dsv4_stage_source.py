#!/usr/bin/env python3
"""Host-side tests for DSV4 source fragment generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import dsv4_stage_source as fragment  # noqa: E402
from tools import dsv4_stagepack as pack  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_failure(function, message: str) -> None:
    try:
        function()
    except pack.PackFailure as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"expected PackFailure containing {message!r}")


def write_source(root: Path) -> dict[str, object]:
    payloads = {"a": b"abcd", "b": b"01234567", "unused": b"wxyz"}
    weight_map = {}
    source_files = {}
    for index, (name, payload) in enumerate(payloads.items(), 1):
        shard = f"model-{index:05d}-of-00003.safetensors"
        header = {
            name: {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        }
        encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        path = root / shard
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
        weight_map[name] = shard
        source_files[shard] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    index_path = root / pack.SOURCE_INDEX_NAME
    index_path.write_text(json.dumps({
        "metadata": {"total_size": sum(len(value) for value in payloads.values())},
        "weight_map": weight_map,
    }), encoding="utf-8")
    source_files[pack.SOURCE_INDEX_NAME] = {
        "bytes": index_path.stat().st_size,
        "sha256": file_sha256(index_path),
    }
    return {
        "model_id": "test/dsv4",
        "source_revision": "1" * 40,
        "source_index_sha256": file_sha256(index_path),
        "source_tensor_count": len(payloads),
        "source_indexed_payload_bytes": sum(len(value) for value in payloads.values()),
        "source_files": source_files,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        output_root = root / "fragment"
        source_root.mkdir()
        contract = write_source(source_root)
        record = type("Record", (), {"source_names": ("a",), "scale_names": ("b",)})()
        assert pack.validate_source_identity(source_root, contract) == (
            contract["source_index_sha256"])
        unused_path = source_root / "model-00003-of-00003.safetensors"
        unused_bytes = unused_path.read_bytes()
        unused_path.write_bytes(unused_bytes[:-1] + bytes((unused_bytes[-1] ^ 1,)))
        expect_failure(
            lambda: pack.validate_source_identity(source_root, contract),
            "SHA-256 mismatch")
        unused_path.write_bytes(unused_bytes)
        unused_path.unlink()
        expect_failure(
            lambda: pack.validate_source_identity(source_root, contract),
            "source shard set mismatch")
        unused_path.write_bytes(unused_bytes)
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
                source, [record], shard_root, 0, 1, contract
            )
        assert shard_result["source_mode"] == "shards"
        assert shard_result["format"] == pack.STAGE_SOURCE_MANIFEST_FORMAT
        assert pack.validate_source_identity(
            shard_root, contract, [record], 0, 1) == contract["source_index_sha256"]
        expect_failure(
            lambda: pack.validate_source_identity(
                shard_root, contract, [record], 1, 1),
            "first_layer")
        with pack.SafetensorSource(shard_root) as generated:
            assert generated.read("a") == b"abcd"
            assert generated.read("b") == b"01234567"
        copied = shard_root / "model-00001-of-00003.safetensors"
        copied_bytes = copied.read_bytes()
        copied.write_bytes(copied_bytes[:-1] + bytes((copied_bytes[-1] ^ 1,)))
        expect_failure(
            lambda: pack.validate_source_identity(
                shard_root, contract, [record], 0, 1),
            "SHA-256 mismatch")
        copied.write_bytes(copied_bytes)
        manifest_path = shard_root / fragment.FRAGMENT_MANIFEST_NAME
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_manifest["source_files_present"] = False
        manifest_path.write_text(json.dumps(shard_manifest), encoding="utf-8")
        expect_failure(
            lambda: pack.validate_source_identity(
                shard_root, contract, [record], 0, 1),
            "source_files_present")
        shard_manifest["source_files_present"] = True
        manifest_path.write_text(json.dumps(shard_manifest), encoding="utf-8")
        reduced_index = shard_root / pack.SOURCE_INDEX_NAME
        reduced_index_bytes = reduced_index.read_bytes()
        reduced_index.write_bytes(reduced_index_bytes + b"\n")
        expect_failure(
            lambda: pack.validate_source_identity(
                shard_root, contract, [record], 0, 1),
            "reduced_index_sha256")
        reduced_index.write_bytes(reduced_index_bytes)
        index_only_root = root / "index-only"
        with pack.SafetensorSource(source_root) as source:
            fragment.write_stage_shard_index(
                source, [record], index_only_root, 0, 1, contract)
        expect_failure(
            lambda: pack.validate_source_identity(
                index_only_root, contract, [record], 0, 1),
            "source_files_present")
        assert fragment.main([
            "--model-dir", str(source_root),
            "--output-root", str(root / "forbidden-fragment"),
            "--first-layer", "0", "--layer-count", "1",
            "--source-mode", "fragment",
        ]) == 1
    print("PASS DSV4 stage source fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
