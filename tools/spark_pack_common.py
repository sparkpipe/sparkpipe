#!/usr/bin/env python3
"""Shared packer core: neutral data-movement primitives for every stage packer.

This module is the single home for the primitives that were re-implemented
across the six packers (see docs/PACKER_CORE_PLAN.md, DRY_CONSOLIDATION_PLAN
items 3 and 5). It is deliberately model-agnostic: it imports nothing
model-specific and holds no topology, geometry, kind, or sentinel policy. The
per-model config tables (geometry, tensor kinds, sentinels, wire structs, and
field order) stay in each packer.

Collapsed here:

  * PackFailure       - the one packer failure type (RuntimeError).
  * sha256 helpers    - streamed file hexdigest + raw bytes digest.
  * align_up          - round a byte offset up to an alignment.
  * SafetensorsSource - header-only safetensors reader (index, config,
                        per-shard headers, payload offsets) with no torch.
  * make_directory / pack_entry / pack_header - offset and framing helpers;
                        per-model structs and field order stay in the packer.
  * write_receipt     - atomic JSON receipt write.
  * tp_shard_range    - this rank's [start, count) slice of a dimension.
  * spark_pack_replicated_draft_rows - the one replicated-draft rule (item 5).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# The reserved marker for tensors owned by the whole model rather than one
# layer (embedding, head, norms, draft heads).
GLOBAL_LAYER = 0xFFFFFFFF

# Streaming chunk for sha256_file. The digest is identical for any chunk size;
# 16 MiB keeps the read loop cheap for multi-hundred-GiB checkpoints.
_SHA256_CHUNK = 16 * 1024 * 1024


class PackFailure(RuntimeError):
    """A source or wire-contract error that must stop pack generation.

    Standardizes a drift across the packers: qwen38_27b/qwen38 subclassed
    Exception (so a bare except did not catch it the same way), glm52_stagepack
    raised plain ValueError, and the rest used RuntimeError. The shared
    contract is always RuntimeError.
    """


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, streamed in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(_SHA256_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def sha256_bytes(data: bytes) -> bytes:
    """Raw SHA-256 digest of bytes (for binary header/receipt fields)."""
    return hashlib.sha256(data).digest()


def align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


class SafetensorsSource:
    """Header-only safetensors reader: index + config + per-shard headers.

    No torch: payload bytes are located by data_offsets and streamed by the
    caller. check_config and check_shape take their expectations as arguments
    so this class stays neutral; model-specific expectation tables live in the
    packer (which may subclass and add its own checks).
    """

    def __init__(
        self,
        root: Path,
        index_name: str = "model.safetensors.index.json",
        config_name: str = "config.json",
    ) -> None:
        self.root = root
        index_path = root / index_name
        config_path = root / config_name
        if not index_path.is_file():
            raise PackFailure(f"missing {index_path}")
        if not config_path.is_file():
            raise PackFailure(f"missing {config_path}")
        self.index_sha256 = sha256_file(index_path)
        self.config_sha256 = sha256_file(config_path)
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.config = json.loads(config_path.read_text())
        self.headers: Dict[str, dict] = {}
        self.data_start: Dict[str, int] = {}

    def check_config(
        self,
        expectations: Mapping[str, object],
        *,
        section: str | None = None,
    ) -> None:
        """Assert config values match expectations (optionally under a
        top-level section key, e.g. qwen38_27b's text_config)."""
        config: Mapping[str, object] = (
            self.config.get(section, {}) if section is not None else self.config
        )
        for key, expected in expectations.items():
            if config.get(key) != expected:
                raise PackFailure(
                    f"config.json {key}={config.get(key)!r}, expected {expected!r} "
                    "- this is not the checkpoint this packer is for")

    def shard_header(self, shard: str) -> dict:
        """Lazily parse (and cache) one shard's safetensors header."""
        if shard not in self.headers:
            path = self.root / shard
            if not path.is_file():
                raise PackFailure(f"missing shard {shard}")
            with path.open("rb") as file:
                header_bytes = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_bytes))
            self.headers[shard] = header
            self.data_start[shard] = 8 + header_bytes
        return self.headers[shard]

    def resolve(self, name: str) -> Tuple[str, dict, int]:
        """Return (shard, metadata, absolute payload offset) for a tensor."""
        if name not in self.weight_map:
            raise PackFailure(f"tensor not in checkpoint index: {name}")
        shard = self.weight_map[name]
        header = self.shard_header(shard)
        if name not in header:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        meta = header[name]
        return shard, meta, self.data_start[shard] + meta["data_offsets"][0]

    def check_shape(
        self,
        name: str,
        rows: int,
        columns: int,
        *,
        dtype: str = "BF16",
    ) -> Tuple[str, dict, int]:
        """Standard dense 2-D shape check for one tensor (drops a singleton
        middle dim, treats 1-D as a single row). Returns
        (shard, metadata, payload offset) on success."""
        shard, meta, offset = self.resolve(name)
        if meta["dtype"] != dtype:
            raise PackFailure(f"{name}: dtype {meta['dtype']}, expected {dtype}")
        shape = meta["shape"]
        if len(shape) == 3 and shape[1] == 1:
            shape = [shape[0], shape[2]]
        if len(shape) == 1:
            shape = [1, shape[0]]
        if shape != [rows, columns]:
            raise PackFailure(
                f"{name}: checkpoint shape {meta['shape']}, pack expects "
                f"[{rows}, {columns}]")
        return shard, meta, offset


# -- pack header/directory framing (bytes in, bytes out - no policy) ----------

def make_directory(
    records: Sequence[Any],
    header_bytes: int,
    entry_bytes: int,
) -> Tuple[List[Tuple[Any, int, int]], int]:
    """Compute payload/scale offsets for records laid out directly after a
    fixed header plus one entry per record.

    Each record must expose payload_bytes and scale_bytes. Returns
    (entries, file_bytes) where each entry is (record, payload_offset,
    scale_offset). Pure offset arithmetic; header/entry sizes stay with the
    packer.
    """
    cursor = header_bytes + entry_bytes * len(records)
    entries: List[Tuple[Any, int, int]] = []
    for record in records:
        payload_offset = cursor
        cursor += record.payload_bytes
        scale_offset = 0
        if record.scale_bytes:
            scale_offset = cursor
            cursor += record.scale_bytes
        entries.append((record, payload_offset, scale_offset))
    return entries, cursor


def pack_entry(entry: Tuple[Any, int, int], entry_struct: "struct.Struct") -> bytes:
    """Serialize one directory entry (record, payload_offset, scale_offset).

    entry_struct packs (kind, layer, weight_format, rows, columns, reserved=0,
    payload_offset, scale_offset); the per-model struct and field order stay
    with the packer.
    """
    record, payload_offset, scale_offset = entry
    return entry_struct.pack(
        record.kind, record.layer, record.weight_format, record.rows,
        record.columns, 0, payload_offset, scale_offset,
    )


def pack_header(fields: Sequence[Any], header_struct: "struct.Struct") -> bytes:
    """Serialize a pack header from an ordered field sequence. The caller owns
    the per-model field order and geometry; this only frames bytes out."""
    return header_struct.pack(*fields)


def write_receipt(receipt: Mapping[str, Any], output: Path, *,
                  suffix: str | None = ".receipt.json") -> Path:
    """Atomically write receipt as sorted, indented JSON.

    suffix is appended to output to form the receipt path (the default
    <pack>.receipt.json); pass suffix=None to write to output itself. Returns
    the path written.
    """
    path = Path(output) if suffix is None else Path(str(output) + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def tp_shard_range(dimension: int, tp_degree: int, tp_rank: int,
                   block: int = 1) -> Tuple[int, int]:
    """This rank's [start, start + count) slice of a dimension split across a
    TP group, with count required to be a whole number of quantization blocks
    so scale tensors slice on block boundaries. Fails closed on any
    misalignment rather than producing a shard whose scales cannot be
    represented."""
    if tp_degree < 1 or tp_rank < 0 or tp_rank >= tp_degree:
        raise PackFailure(f"invalid tp shard {tp_rank}/{tp_degree}")
    if dimension % tp_degree != 0:
        raise PackFailure(f"dimension {dimension} not divisible by tp degree {tp_degree}")
    count = dimension // tp_degree
    if block > 1 and count % block != 0:
        raise PackFailure(
            f"tp shard extent {count} not aligned to quantization block {block}")
    return tp_rank * count, count


# The replicated-draft rule must know whether a draft layer's own per-layer
# kinds slice like main layers or replicate whole. Two tables:
#   DSV4 - draft layers replicate whole (draft_layer_kinds_slice=False).
#   qwen - the MTP decoder reuses per-layer kinds, which slice like main
#          layers; only the MTP globals (fc + three norms) replicate
#          (draft_layer_kinds_slice=True).
def spark_pack_replicated_draft_rows(
    kind: int,
    layer: int,
    *,
    draft_layer_first: int,
    draft_layer_count: int,
    global_kinds: frozenset[int],
    draft_layer_kinds_slice: bool = False,
) -> bool:
    """True => this tensor's rows replicate full-width on every rank (no shard).

    The one replicated-draft rule (DRY_CONSOLIDATION_PLAN item 5):
      * a GLOBAL_LAYER kind in global_kinds (draft heads/norms/fc at the
        global marker) replicates;
      * a draft-layer marker in [draft_layer_first, draft_layer_first +
        draft_layer_count) replicates whole unless draft_layer_kinds_slice
        (qwen MTP).
    """
    if layer == GLOBAL_LAYER:
        return kind in global_kinds
    if draft_layer_first <= layer < draft_layer_first + draft_layer_count:
        return not draft_layer_kinds_slice
    return False


__all__ = [
    "GLOBAL_LAYER",
    "PackFailure",
    "sha256_file",
    "sha256_bytes",
    "align_up",
    "SafetensorsSource",
    "make_directory",
    "pack_entry",
    "pack_header",
    "write_receipt",
    "tp_shard_range",
    "spark_pack_replicated_draft_rows",
]
