#!/usr/bin/env python3
"""Repair the replicated fast-path tensors of a minimax.text.bf16.tp4 pack.

The wave-2 emission read every fully-replicated tensor from shard position
zero instead of the resolved tensor base (tools/minimax_stagepack.py
copy_entry fast path lacked the seek), so the placed packs carry shard
header bytes / shifted payload in EMBEDDING, FINAL_NORM, INPUT_NORM,
POST_ATTENTION_NORM, Q_NORM and K_NORM. Row- and column-sharded tensors
used the seeking paths and are correct. This tool rewrites a repaired COPY
of the pack, re-copying every affected tensor from the warm checkpoint and
verifying each written region against a fresh warm sha before the output is
committed. The source pack is never modified.
"""
import argparse
import hashlib
import os
import struct
import sys
import tempfile
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parents[1] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import SafetensorsSource, sha256_file  # noqa: E402

HEADER_BYTES = 120
ENTRY_BYTES = 56
ENTRY_STRUCT = struct.Struct("<6I4Q")
GLOBAL_LAYER = 0xFFFFFFFF
CHUNK_BYTES = 8 * 1024 * 1024
REPLICATED_KINDS = [0, 1, 3, 4, 5, 6]
KIND_LAYER_NAMES = {
    3: "input_layernorm.weight",
    4: "post_attention_layernorm.weight",
    5: "self_attn.q_norm.weight",
    6: "self_attn.k_norm.weight",
}


def tensor_name(kind, layer):
    prefix = "model.language_model."
    if layer == GLOBAL_LAYER:
        return {0: prefix + "embed_tokens.weight", 1: prefix + "norm.weight"}[kind]
    return prefix + f"layers.{layer}." + KIND_LAYER_NAMES[kind]


def sha256_at(file, offset, length):
    digest = hashlib.sha256()
    file.seek(offset)
    remaining = length
    while remaining > 0:
        step = min(remaining, CHUNK_BYTES)
        chunk = file.read(step)
        if len(chunk) != step:
            raise SystemExit("short region read")
        digest.update(chunk)
        remaining -= step
    return digest.hexdigest()


def copy_from(file, offset, length, out):
    file.seek(offset)
    remaining = length
    while remaining > 0:
        step = min(remaining, CHUNK_BYTES)
        chunk = file.read(step)
        if len(chunk) != step:
            raise SystemExit("short read")
        out.write(chunk)
        remaining -= step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = SafetensorsSource(args.checkpoint)
    file_bytes = args.pack.stat().st_size
    regions = []
    with args.pack.open("rb") as pack:
        pack.seek(0)
        header = pack.read(HEADER_BYTES)
        count = struct.unpack_from("<I", header, 16)[0]
        if count * ENTRY_BYTES + HEADER_BYTES > file_bytes:
            raise SystemExit("tensor count overruns the pack")
        for _ in range(count):
            raw = pack.read(ENTRY_BYTES)
            kind, layer, wf, rows, cols, sg, off, pb, so, sb = ENTRY_STRUCT.unpack(raw)
            if kind in REPLICATED_KINDS:
                if pb != rows * cols * 2:
                    raise SystemExit(f"region {kind}/{layer}: unexpected byte count")
                regions.append((kind, layer, off, pb, rows, cols))
        print(f"repair: pack {args.pack} tensors={count} replicated_regions={len(regions)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        with tempfile.NamedTemporaryFile(prefix=f".{args.output.name}.", suffix=".tmp",
                                         dir=args.output.parent, delete=False) as temp:
            temp_path = temp.name
            pack.seek(0)
            remaining = file_bytes
            while remaining > 0:
                step = min(remaining, CHUNK_BYTES)
                chunk = pack.read(step)
                if len(chunk) != step:
                    raise SystemExit("short pack read")
                temp.write(chunk)
                remaining -= step
            for kind, layer, off, pb, rows, cols in regions:
                name = tensor_name(kind, layer)
                shard, meta, base = source.resolve(name)
                shape = list(meta["shape"])
                if len(shape) == 1:
                    shape = [1, shape[0]]
                if shape != [rows, cols]:
                    raise SystemExit(f"{name}: warm shape {meta['shape']} != pack [{rows}, {cols}]")
                if meta["dtype"] != "BF16":
                    raise SystemExit(f"{name}: warm dtype {meta['dtype']} != BF16")
                with (args.checkpoint / shard).open("rb") as shard_file:
                    warm_sha = sha256_at(shard_file, base, pb)
                    temp.seek(off)
                    copy_from(shard_file, base, pb, temp)
                written_sha = sha256_at(temp, off, pb)
                if warm_sha != written_sha:
                    raise SystemExit(f"{name}: repaired region sha mismatch")
                print(f"repair: fixed kind={kind} layer={layer} name={name} bytes={pb}")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, args.output)
    print(f"repair: wrote {args.output} sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
