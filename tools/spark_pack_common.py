"""Shared packer core: checkpoint reading, tensor-parallel windows, hashing,
receipts, publication and the command plumbing every stage packer repeats.

SafetensorsSource reads the index, the config and each shard header without
torch; payloads are located by data_offsets and streamed by the caller.
Tensor-parallel windows fail closed on any extent that does not divide. Packs
and receipts are published atomically (a replace, or an exclusive link that
never overwrites an existing artifact) and every digest is SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

GLOBAL_LAYER = 0xFFFFFFFF
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
SHA256_CHUNK = 16 * 1024 * 1024


class PackFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(SHA256_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_safetensors_header(path: Path, label: str | None = None) -> Tuple[dict, int, int]:
    path = Path(path)
    label = path.name if label is None else label
    if not path.is_file():
        raise PackFailure(f"missing shard {label}")
    file_bytes = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise PackFailure(f"shard {label} is shorter than a safetensors header")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > file_bytes - 8:
            raise PackFailure(f"shard {label} declares a {header_bytes}-byte header in a {file_bytes}-byte file")
        header = json.loads(file.read(header_bytes))
    return header, 8 + header_bytes, file_bytes


def tensor_extent(meta: Mapping[str, Any], name: str, shard: str, payload_bytes: int) -> Tuple[int, int]:
    offsets = meta.get("data_offsets")
    if (not isinstance(offsets, list) or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] or offsets[1] > payload_bytes):
        raise PackFailure(f"tensor {name} in shard {shard} has data_offsets {offsets!r} outside the shard's "
                          f"{payload_bytes} payload bytes")
    return offsets[0], offsets[1]


class SafetensorsSource:
    def __init__(self, root: Path, index_name: str = INDEX_NAME, config_name: str = CONFIG_NAME,
                 config_required: bool = True) -> None:
        self.root = Path(root)
        index_path = self.root / index_name
        config_path = self.root / config_name
        if not index_path.is_file():
            raise PackFailure(f"missing {index_path}")
        if config_required and not config_path.is_file():
            raise PackFailure(f"missing {config_path}")
        self.index_sha256 = sha256_file(index_path)
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        if config_path.is_file():
            self.config_sha256 = sha256_file(config_path)
            self.config = json.loads(config_path.read_text())
        else:
            self.config_sha256 = None
            self.config = {}
        self.headers: Dict[str, dict] = {}
        self.data_start: Dict[str, int] = {}
        self.shard_bytes: Dict[str, int] = {}

    def check_config(self, expectations: Mapping[str, object], *, section: str | None = None) -> None:
        config: Mapping[str, object] = self.config.get(section, {}) if section is not None else self.config
        for key, expected in expectations.items():
            if config.get(key) != expected:
                raise PackFailure(
                    f"config.json {key}={config.get(key)!r}, expected {expected!r} "
                    "- this is not the checkpoint this packer is for")

    def shard_header(self, shard: str) -> dict:
        if shard not in self.headers:
            header, data_start, file_bytes = read_safetensors_header(self.root / shard, shard)
            self.headers[shard] = header
            self.data_start[shard] = data_start
            self.shard_bytes[shard] = file_bytes
        return self.headers[shard]

    def resolve(self, name: str) -> Tuple[str, dict, int]:
        if name not in self.weight_map:
            raise PackFailure(f"tensor not in checkpoint index: {name}")
        shard = self.weight_map[name]
        header = self.shard_header(shard)
        if name not in header:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        meta = header[name]
        begin, _ = tensor_extent(meta, name, shard, self.shard_bytes[shard] - self.data_start[shard])
        return shard, meta, self.data_start[shard] + begin

    def check_shape(self, name: str, rows: int, columns: int, *, dtype: str = "BF16") -> Tuple[str, dict, int]:
        shard, meta, offset = self.resolve(name)
        if meta["dtype"] != dtype:
            raise PackFailure(f"{name}: dtype {meta['dtype']}, expected {dtype}")
        shape = meta["shape"]
        if len(shape) == 3 and shape[1] == 1:
            shape = [shape[0], shape[2]]
        if len(shape) == 1:
            shape = [1, shape[0]]
        if shape != [rows, columns]:
            raise PackFailure(f"{name}: checkpoint shape {meta['shape']}, pack expects [{rows}, {columns}]")
        return shard, meta, offset


def write_receipt(receipt: Mapping[str, Any], output: Path, *, suffix: str | None = ".receipt.json") -> Path:
    path = Path(output) if suffix is None else Path(str(output) + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(receipt, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return path


def write_sha256_sidecar(path: Path, digest: str | None = None) -> str:
    path = Path(path)
    digest = sha256_file(path) if digest is None else digest
    (path.parent / (path.name + ".sha256")).write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_link(temporary: Path, path: Path) -> None:
    with open(temporary, "rb") as source:
        os.fsync(source.fileno())
    os.link(temporary, path)
    fsync_directory(Path(path).parent)


class HashingWriter:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        written = self.stream.write(data)
        self.digest.update(data)
        return written

    def tell(self) -> int:
        return self.stream.tell()

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()


def check_tp_rank(tp_degree: int, tp_rank: int) -> None:
    if tp_degree < 1 or tp_rank < 0 or tp_rank >= tp_degree:
        raise PackFailure(f"invalid tp rank {tp_rank}/{tp_degree}")


def tp_shard_range(dimension: int, tp_degree: int, tp_rank: int, block: int = 1) -> Tuple[int, int]:
    if tp_degree < 1 or tp_rank < 0 or tp_rank >= tp_degree:
        raise PackFailure(f"invalid tp shard {tp_rank}/{tp_degree}")
    if dimension % tp_degree != 0:
        raise PackFailure(f"dimension {dimension} not divisible by tp degree {tp_degree}")
    count = dimension // tp_degree
    if block > 1 and count % block != 0:
        raise PackFailure(f"tp shard extent {count} not aligned to quantization block {block}")
    return tp_rank * count, count


def tp_window(total: int, degree: int, rank: int) -> Tuple[int, int]:
    if degree <= 0 or total % degree != 0:
        raise PackFailure(f"tp_window: {total} not divisible by {degree}")
    per = total // degree
    return rank * per, per


class TpPlan:
    def __init__(self, kind: str, **fields) -> None:
        self.kind = kind
        self.__dict__.update(fields)


def packed_tp_shape(ref, plan) -> Tuple[int, int]:
    if plan is None:
        return ref.rows, ref.columns
    if plan.kind in ("rows", "experts"):
        return plan.row_count, ref.columns
    if plan.kind == "cols":
        return ref.rows, plan.col_count
    if plan.kind == "qkv":
        return sum(count for _, count in plan.segments), ref.columns
    raise PackFailure(f"unknown tp plan {plan.kind}")


def spark_pack_replicated_draft_rows(kind: int, layer: int, *, draft_layer_first: int, draft_layer_count: int,
                                     global_kinds: frozenset, draft_layer_kinds_slice: bool = False) -> bool:
    if layer == GLOBAL_LAYER:
        return kind in global_kinds
    if draft_layer_first <= layer < draft_layer_first + draft_layer_count:
        return not draft_layer_kinds_slice
    return False


def pump(fd: int, offset: int, length: int, out) -> None:
    chunk_bytes = 1 << 20
    remaining = length
    while remaining > 0:
        step = min(remaining, chunk_bytes)
        raw = os.pread(fd, step, offset)
        if len(raw) != step:
            raise PackFailure(f"short read at {offset}")
        out.write(raw)
        try:
            os.posix_fadvise(fd, offset, step, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        offset += step
        remaining -= step


def stream_range(file, offset: int, length: int, out, chunk_bytes: int, label: str, transform=None) -> None:
    file.seek(offset)
    remaining = length
    while remaining > 0:
        step = min(remaining, chunk_bytes)
        chunk = file.read(step)
        if len(chunk) != step:
            raise PackFailure(f"short read on {label}")
        out.write(chunk if transform is None else transform(chunk))
        remaining -= step


def window_chunks(read_at, base: int, row_bytes: int, row_start: int, row_count: int, column_offset: int,
                  column_bytes: int, budget: int, label: str):
    step_rows = max(1, budget // row_bytes) if row_bytes else 1
    row = 0
    while row < row_count:
        block = min(step_rows, row_count - row)
        data = read_at(base + (row_start + row) * row_bytes, block * row_bytes)
        if len(data) != block * row_bytes:
            raise PackFailure(f"short read {label}")
        if column_offset == 0 and column_bytes == row_bytes:
            yield data
        else:
            view = memoryview(data)
            for index in range(block):
                start = index * row_bytes + column_offset
                yield view[start:start + column_bytes]
        row += block


def check_pipeline_stage(stage_count: int, stage_index: int, first_layer: int, layer_count: int,
                         owns_embedding: bool, owns_tail: bool, total_layers: int) -> None:
    if not 1 <= stage_count <= total_layers or not 0 <= stage_index < stage_count:
        raise PackFailure("invalid pipeline stage count/index")
    if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > total_layers:
        raise PackFailure("pipeline layer span is outside the model")
    if owns_embedding and (stage_index != 0 or first_layer != 0):
        raise PackFailure("embedding ownership requires the first stage and layer")
    if owns_tail and (stage_index + 1 != stage_count or first_layer + layer_count != total_layers):
        raise PackFailure("head/MTP ownership requires the final stage and layer")


def stage_pack_name(prefix: str, suffix: str, tp_degree: int, tp_rank: int, stage_count: int, stage_index: int) -> str:
    pipeline = f".pp{stage_count}.stage{stage_index}" if stage_count != 1 else ""
    return f"{prefix}.tp{tp_degree}{pipeline}.rank{tp_rank}.{suffix}"


def run_tool(main, label: str) -> None:
    try:
        code = main()
    except PackFailure as failure:
        print(f"{label}: {failure}", file=sys.stderr)
        code = 1
    sys.exit(code)
