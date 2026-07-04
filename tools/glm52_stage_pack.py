#!/usr/bin/env python3
"""
Build GLM-5.2 PP13 stage-local non-MoE packs from a Hugging Face checkpoint.

This is an offline setup tool.  The production C/CUDA path should load the
generated .spstage files and should not open Hugging Face shard files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any, BinaryIO, Dict, Iterable, List, Mapping, Sequence, Tuple


FORMAT = "sparkpipe.glm52.pp13.stagepack.v1"
STAGE_COUNT = 13
LAYER_COUNT = 78
LAYERS_PER_STAGE = 6
INDEX_FILE = "stagepack_index.json"
STAGE_FILE_TEMPLATE = "stage_{stage_index:02d}_non_moe.spstage"
REGION_ALIGNMENT = 4096
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class StagePackFailure(RuntimeError):
    pass


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def parse_stages(text: str) -> List[int]:
    if text == "all":
        return list(range(STAGE_COUNT))
    stages: List[int] = []
    for item in text.replace(";", ",").split(","):
        if not item:
            continue
        stage = int(item, 10)
        if stage < 0 or stage >= STAGE_COUNT:
            raise StagePackFailure(f"stage index out of range: {stage}")
        if stage not in stages:
            stages.append(stage)
    if not stages:
        raise StagePackFailure("no stages selected")
    return stages


def read_safetensors_header(path: Path) -> Tuple[Dict[str, Any], int]:
    with path.open("rb") as file:
        header_bytes = file.read(8)
        if len(header_bytes) != 8:
            raise StagePackFailure(f"short safetensors header length: {path}")
        header_length = struct.unpack("<Q", header_bytes)[0]
        if header_length == 0:
            raise StagePackFailure(f"empty safetensors header: {path}")
        header_text = file.read(header_length)
        if len(header_text) != header_length:
            raise StagePackFailure(f"short safetensors header body: {path}")
    header = json.loads(header_text.decode("utf-8"))
    if not isinstance(header, dict):
        raise StagePackFailure(f"safetensors header is not an object: {path}")
    return header, 8 + header_length


def load_weight_map(model_dir: Path) -> Mapping[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise StagePackFailure(f"missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise StagePackFailure("safetensors index is missing weight_map")
    return weight_map


def load_source_headers(
    model_dir: Path,
    shard_names: Iterable[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    headers: Dict[str, Dict[str, Any]] = {}
    payload_bases: Dict[str, int] = {}
    for shard_name in sorted(set(str(item) for item in shard_names)):
        shard_path = model_dir / str(shard_name)
        if not shard_path.exists():
            raise StagePackFailure(f"missing safetensors shard: {shard_path}")
        headers[str(shard_name)], payload_bases[str(shard_name)] = read_safetensors_header(shard_path)
    return headers, payload_bases


def tensor_stage(name: str) -> int | None:
    layer_match = LAYER_RE.match(name)
    if layer_match is not None:
        layer = int(layer_match.group(1), 10)
        if layer < 0 or layer >= LAYER_COUNT:
            return None
        return layer // LAYERS_PER_STAGE
    if name.startswith("model.embed_tokens."):
        return 0
    if name.startswith("model.norm.") or name.startswith("lm_head."):
        return STAGE_COUNT - 1
    return 0


def is_moe_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def collect_stage_tensors(
    weight_map: Mapping[str, str],
    selected_stages: Iterable[int],
) -> Tuple[Dict[int, List[str]], List[str]]:
    selected = set(selected_stages)
    stage_tensors: Dict[int, List[str]] = {stage: [] for stage in selected}
    shard_names: List[str] = []
    for name in sorted(weight_map.keys()):
        if is_moe_expert_tensor(name):
            continue
        stage = tensor_stage(name)
        if stage is None or stage not in selected:
            continue
        stage_tensors[stage].append(name)
        shard = str(weight_map[name])
        if shard not in shard_names:
            shard_names.append(shard)
    return stage_tensors, shard_names


def copy_exact(file_in: BinaryIO, file_out: BinaryIO, byte_count: int) -> None:
    remaining = byte_count
    while remaining > 0:
        chunk = file_in.read(min(remaining, 16 * 1024 * 1024))
        if len(chunk) == 0:
            raise StagePackFailure("unexpected EOF while copying tensor payload")
        file_out.write(chunk)
        remaining -= len(chunk)


def write_padding(file_out: BinaryIO, target_offset: int) -> None:
    position = file_out.tell()
    if position > target_offset:
        raise StagePackFailure("stage pack writer passed aligned target")
    if position < target_offset:
        file_out.write(b"\0" * (target_offset - position))


def tensor_record(
    name: str,
    shard_name: str,
    header: Mapping[str, Any],
    offset: int,
) -> Tuple[Dict[str, Any], int, int]:
    item = header.get(name)
    if not isinstance(item, dict):
        raise StagePackFailure(f"tensor metadata is missing: {name}")
    dtype = item.get("dtype")
    shape = item.get("shape")
    data_offsets = item.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(data_offsets, list) or len(data_offsets) != 2:
        raise StagePackFailure(f"tensor metadata is malformed: {name}")
    start = int(data_offsets[0])
    end = int(data_offsets[1])
    if start < 0 or end < start:
        raise StagePackFailure(f"tensor data offset is malformed: {name}")
    byte_count = end - start
    record = {
        "file": STAGE_FILE_TEMPLATE,
        "offset": offset,
        "bytes": byte_count,
        "dtype": dtype,
        "shape": [int(value) for value in shape],
        "source_shard": shard_name,
    }
    return record, start, byte_count


def write_stage_pack(
    model_dir: Path,
    output_dir: Path,
    stage_index: int,
    tensor_names: Sequence[str],
    weight_map: Mapping[str, str],
    headers: Mapping[str, Mapping[str, Any]],
    payload_bases: Mapping[str, int],
    reuse: bool,
) -> Dict[str, Any]:
    stage_file_name = STAGE_FILE_TEMPLATE.format(stage_index=stage_index)
    output_path = output_dir / stage_file_name
    stage_tensor_map: Dict[str, Dict[str, Any]] = {}
    if reuse and output_path.exists():
        return {
            "stage_index": stage_index,
            "file": stage_file_name,
            "tensor_count": len(tensor_names),
            "bytes": output_path.stat().st_size,
            "reused": True,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".stage_{stage_index:02d}_",
        suffix=".tmp",
        dir=str(output_dir),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        for name in tensor_names:
            shard_name = str(weight_map[name])
            header = headers[shard_name]
            aligned_offset = align_up(temp_file.tell(), REGION_ALIGNMENT)
            write_padding(temp_file, aligned_offset)
            record, tensor_start, tensor_bytes = tensor_record(
                name,
                shard_name,
                header,
                aligned_offset,
            )
            record["file"] = stage_file_name
            shard_path = model_dir / shard_name
            with shard_path.open("rb") as shard_file:
                shard_file.seek(payload_bases[shard_name] + tensor_start)
                copy_exact(shard_file, temp_file, tensor_bytes)
            stage_tensor_map[name] = record
    os.replace(temp_path, output_path)
    return {
        "stage_index": stage_index,
        "file": stage_file_name,
        "tensor_count": len(tensor_names),
        "bytes": output_path.stat().st_size,
        "reused": False,
        "tensor_map": stage_tensor_map,
    }


def load_existing_index(output_dir: Path) -> Dict[str, Any]:
    index_path = output_dir / INDEX_FILE
    if not index_path.exists():
        return {
            "format": FORMAT,
            "topology": "pp13_fixed6",
            "stage_count": STAGE_COUNT,
            "layers_per_stage": LAYERS_PER_STAGE,
            "tensor_map": {},
            "stages": {},
        }
    return json.loads(index_path.read_text(encoding="utf-8"))


def write_index(output_dir: Path, index: Mapping[str, Any]) -> None:
    tmp_path = output_dir / (INDEX_FILE + ".tmp")
    tmp_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_dir / INDEX_FILE)


def build_stage_packs(args: argparse.Namespace) -> Dict[str, Any]:
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    selected_stages = parse_stages(args.stages)
    weight_map = load_weight_map(model_dir)
    stage_tensors, shard_names = collect_stage_tensors(weight_map, selected_stages)
    headers, payload_bases = load_source_headers(model_dir, shard_names)
    index = load_existing_index(output_dir)
    index["format"] = FORMAT
    index["model_quantization"] = args.model_quantization
    index["topology"] = "pp13_fixed6"
    index["stage_count"] = STAGE_COUNT
    index["layers_per_stage"] = LAYERS_PER_STAGE
    index.setdefault("tensor_map", {})
    index.setdefault("stages", {})
    built: List[Dict[str, Any]] = []
    for stage_index in selected_stages:
        result = write_stage_pack(
            model_dir,
            output_dir,
            stage_index,
            stage_tensors[stage_index],
            weight_map,
            headers,
            payload_bases,
            args.reuse,
        )
        if "tensor_map" not in result:
            missing = [
                name
                for name in stage_tensors[stage_index]
                if name not in index.get("tensor_map", {})
            ]
            if missing:
                raise StagePackFailure(
                    f"reuse requested for stage {stage_index}, but {len(missing)} tensor index entries are missing"
                )
        for name, record in result.get("tensor_map", {}).items():
            index["tensor_map"][name] = record
        index["stages"][str(stage_index)] = {
            "file": STAGE_FILE_TEMPLATE.format(stage_index=stage_index),
            "first_layer": stage_index * LAYERS_PER_STAGE,
            "layer_count": LAYERS_PER_STAGE,
            "tensor_count": len(stage_tensors[stage_index]),
        }
        summary = dict(result)
        summary.pop("tensor_map", None)
        built.append(summary)
    write_index(output_dir, index)
    return {
        "format": FORMAT,
        "output_dir": str(output_dir),
        "model_quantization": args.model_quantization,
        "stages": built,
        "tensor_count": len(index["tensor_map"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build GLM-5.2 PP13 stage-local non-MoE packs")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-quantization", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_stage_packs(args)
    except StagePackFailure as error:
        print(f"glm52_stage_pack: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
