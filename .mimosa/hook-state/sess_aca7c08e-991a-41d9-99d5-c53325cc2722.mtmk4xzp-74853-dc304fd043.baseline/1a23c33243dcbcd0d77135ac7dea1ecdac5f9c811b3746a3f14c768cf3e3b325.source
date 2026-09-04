#!/usr/bin/env python3
"""Create per-stage safetensors source fragments for DSV4 stage packing.

This is setup-time tooling.  It copies only the tensors required by one
pipeline stage into a small, valid safetensors directory.  The destination
Spark can then run ``dsv4_stagepack.py`` locally without opening the full
checkpoint or writing a large pack through Spark0's warm archive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import os
import shutil
import struct
import sys
import tempfile
from typing import Iterable, List, Mapping, Sequence

from tools.dsv4_stagepack import (
    CONTRACT_PATH,
    SOURCE_INDEX_NAME,
    STAGE_SOURCE_MANIFEST_NAME,
    STAGE_SOURCE_MANIFEST_FORMAT,
    PackFailure,
    SafetensorSource,
    build_records,
    load_contract,
    sha256_file,
    validate_source_identity,
    validate_records,
)


FRAGMENT_NAME = "model-00001-of-00001.safetensors"
FRAGMENT_INDEX_NAME = "model.safetensors.index.json"
FRAGMENT_MANIFEST_NAME = STAGE_SOURCE_MANIFEST_NAME
PP13_RANGES = (
    (0, 3), (3, 3), (6, 3), (9, 3), (12, 3), (15, 3), (18, 3),
    (21, 4), (25, 4), (29, 4), (33, 4), (37, 4), (41, 2),
)
COPY_CHUNK = 16 * 1024 * 1024


@dataclass(frozen=True)
class FragmentTensor:
    name: str
    dtype: str
    shape: Sequence[int]
    start: int
    end: int


def ordered_tensor_names(records: Iterable[object]) -> List[str]:
    """Return record tensors once, retaining the packer's deterministic order."""
    names: List[str] = []
    seen = set()
    for record in records:
        for name in tuple(record.source_names) + tuple(record.scale_names):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _fragment_header(tensors: Sequence[FragmentTensor], data_bytes: int) -> bytes:
    offset = 0
    header: Mapping[str, object] = {"__metadata__": {"format": "pt"}}
    entries = dict(header)
    for tensor in tensors:
        end = offset + tensor.end - tensor.start
        entries[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    if offset != data_bytes:
        raise PackFailure(
            f"fragment payload accounting mismatch: {offset} != {data_bytes}"
        )
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    padding = (-8 - len(encoded)) % 8
    return encoded + b" " * padding


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            mode="w", encoding="utf-8", delete=False) as temporary:
        temp_path = Path(temporary.name)
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, path)


def _write_fragment_payload(source: SafetensorSource, tensors: Sequence[FragmentTensor],
                            output: Path) -> int:
    data_bytes = sum(tensor.end - tensor.start for tensor in tensors)
    header = _fragment_header(tensors, data_bytes)
    with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(struct.pack("<Q", len(header)))
        temporary.write(header)
        for tensor in tensors:
            source.copy(tensor.name, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, output)
    return 8 + len(header) + data_bytes


def _copy_file_atomic(source: Path, output: Path) -> int:
    if source.resolve() == output.resolve():
        raise PackFailure(f"source and destination are the same file: {source}")
    with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False) as temporary:
        temp_path = Path(temporary.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, temporary, COPY_CHUNK)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, output)
    return output.stat().st_size


def inspect_stage(source: SafetensorSource, records: Sequence[object]) -> Mapping[str, object]:
    names = ordered_tensor_names(records)
    shards = sorted({source.weight_map[name] for name in names})
    source_bytes = sum(
        (source.model_dir / shard).stat().st_size for shard in shards
    )
    payload_bytes = sum(source.meta(name).byte_count for name in names)
    return {
        "tensor_count": len(names),
        "source_shard_count": len(shards),
        "source_shard_bytes": source_bytes,
        "selected_payload_bytes": payload_bytes,
        "source_shards": shards,
    }


def write_stage_fragment(source: SafetensorSource, records: Sequence[object],
                         output_dir: Path, first_layer: int,
                         layer_count: int) -> Mapping[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ordered_tensor_names(records)
    tensors = [
        FragmentTensor(name, source.meta(name).dtype, source.meta(name).shape,
                       source.meta(name).start, source.meta(name).end)
        for name in names
    ]
    output = output_dir / FRAGMENT_NAME
    file_bytes = _write_fragment_payload(source, tensors, output)
    index = {
        "metadata": {"total_size": file_bytes},
        "weight_map": {name: FRAGMENT_NAME for name in names},
    }
    _write_json_atomic(output_dir / FRAGMENT_INDEX_NAME, index)
    manifest = {
        "format": "sparkpipe.dsv4.stage-source.v1",
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(names),
        "fragment": FRAGMENT_NAME,
        "fragment_bytes": file_bytes,
        "fragment_sha256": sha256_file(output),
        "source_index_sha256": sha256_file(source.model_dir / SOURCE_INDEX_NAME),
    }
    _write_json_atomic(output_dir / FRAGMENT_MANIFEST_NAME, manifest)
    return manifest


def write_stage_shards(source: SafetensorSource, records: Sequence[object],
                       output_dir: Path, first_layer: int,
                       layer_count: int,
                       contract: Mapping[str, object]) -> Mapping[str, object]:
    """Copy only required original shards and write a reduced source index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ordered_tensor_names(records)
    shards = sorted({source.weight_map[name] for name in names})
    manifest = write_stage_shard_index(
        source, records, output_dir, first_layer, layer_count, contract
    )
    copied_bytes = 0
    for shard in shards:
        copied_bytes += _copy_file_atomic(
            source.model_dir / shard, output_dir / shard
        )
    manifest = {**manifest, "source_files_present": True, "source_shard_bytes": copied_bytes}
    _write_json_atomic(output_dir / FRAGMENT_MANIFEST_NAME, manifest)
    return manifest


def write_stage_shard_index(source: SafetensorSource, records: Sequence[object],
                            output_dir: Path, first_layer: int,
                            layer_count: int,
                            contract: Mapping[str, object]) -> Mapping[str, object]:
    """Write the reduced index before shards are transferred independently."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ordered_tensor_names(records)
    shards = sorted({source.weight_map[name] for name in names})
    source_shard_bytes = sum(
        (source.model_dir / shard).stat().st_size for shard in shards
    )
    index = {
        "metadata": {"total_size": source_shard_bytes},
        "weight_map": {name: source.weight_map[name] for name in names},
    }
    _write_json_atomic(output_dir / FRAGMENT_INDEX_NAME, index)
    source_files = contract.get("source_files")
    if not isinstance(source_files, dict):
        raise PackFailure("shard staging requires pinned source file identities")
    files = []
    for shard in shards:
        entry = source_files.get(shard)
        if not isinstance(entry, dict):
            raise PackFailure(f"source shard is not pinned by the contract: {shard}")
        files.append({
            "name": shard,
            "bytes": entry.get("bytes"),
            "sha256": entry.get("sha256"),
        })
    manifest = {
        "format": STAGE_SOURCE_MANIFEST_FORMAT,
        "source_mode": "shards",
        "source_files_present": False,
        "source_model_id": contract.get("model_id"),
        "source_revision": contract.get("source_revision"),
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(names),
        "source_shard_count": len(shards),
        "source_shard_bytes": source_shard_bytes,
        "source_shards": shards,
        "source_index_sha256": sha256_file(source.model_dir / SOURCE_INDEX_NAME),
        "reduced_index_sha256": sha256_file(output_dir / FRAGMENT_INDEX_NAME),
        "files": files,
    }
    _write_json_atomic(output_dir / FRAGMENT_MANIFEST_NAME, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--first-layer", type=int)
    parser.add_argument("--layer-count", type=int)
    parser.add_argument("--stage-index", type=int)
    parser.add_argument("--all-pp13", action="store_true")
    parser.add_argument(
        "--source-mode", choices=("shards", "fragment"), default="shards"
    )
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    return parser.parse_args(argv)


def selected_ranges(args: argparse.Namespace) -> List[tuple[int, int, int]]:
    if args.all_pp13:
        if args.first_layer is not None or args.layer_count is not None:
            raise PackFailure("--all-pp13 cannot be combined with a layer slice")
        return [(index, first, count) for index, (first, count) in enumerate(PP13_RANGES)]
    if args.first_layer is None or args.layer_count is None:
        raise PackFailure("--first-layer and --layer-count are required")
    stage_index = 0 if args.stage_index is None else args.stage_index
    return [(stage_index, args.first_layer, args.layer_count)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        contract = load_contract(args.contract)
        ranges = selected_ranges(args)
        if (not args.inspect and args.source_mode == "fragment" and
                isinstance(contract.get("source_files"), dict)):
            raise PackFailure(
                "transformed fragment staging is forbidden for a pinned checkpoint; "
                "use --source-mode shards")
        if args.index_only and args.source_mode != "shards":
            raise PackFailure("--index-only requires --source-mode shards")
        validate_source_identity(args.model_dir, contract)
        with SafetensorSource(args.model_dir) as source:
            results = []
            for stage_index, first_layer, layer_count in ranges:
                records = build_records(contract, first_layer, layer_count)
                validate_records(source, records)
                stage_info = inspect_stage(source, records)
                stage_info = {
                    "stage_index": stage_index,
                    "first_layer": first_layer,
                    "layer_count": layer_count,
                    **stage_info,
                }
                if args.index_only:
                    output_dir = args.output_root / f"stage_{stage_index:02d}"
                    stage_info = {**stage_info, **write_stage_shard_index(
                        source, records, output_dir, first_layer, layer_count,
                        contract
                    )}
                elif not args.inspect:
                    output_dir = args.output_root / f"stage_{stage_index:02d}"
                    writer = (
                        write_stage_shards
                        if args.source_mode == "shards"
                        else write_stage_fragment
                    )
                    if writer == write_stage_shards:
                        staged = writer(
                            source, records, output_dir, first_layer,
                            layer_count, contract)
                    else:
                        staged = writer(
                            source, records, output_dir, first_layer, layer_count)
                    stage_info = {**stage_info, **staged}
                    print(
                        f"staged source stage={stage_index} "
                        f"shards={stage_info.get('source_shard_count', 0)} "
                        f"bytes={stage_info.get('source_shard_bytes', 0)}",
                        file=sys.stderr,
                        flush=True,
                    )
                results.append(stage_info)
            print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    except (OSError, PackFailure, json.JSONDecodeError, struct.error) as error:
        print(f"dsv4_stage_source: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
