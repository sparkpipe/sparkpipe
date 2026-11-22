#!/usr/bin/env python3
"""Cut a speculation drafter and the target tensors it reads into one named pack.

    drafter_pack.py --drafter NAME --checkpoint DIR --target FILE [--target FILE ...] --revision REV --out DIR
        [--expect-sha256 FILE=HEX ...]

NAME is one of DRAFTERS: k31dspark (Kimi K3.1 Pro's DSpark), dsv41dspark (the DSpark built into DeepSeek V4.1
1017's checkpoint) and qwen27bdspine (Qwen3.8-27B's DSpine b16). For dsv41dspark the target tensors live in the
checkpoint's own files, so it takes no --target.

Every tensor of the drafter's files and of the target files is either packed, byte for byte, or left out by a named
rule; any other tensor, a dtype the rule does not name, a layer past the config's count, a missing target tensor, or a
drafter tensor that the checkpoint's model.safetensors.index.json places in a file that is not there refuses the pack. Drafter tensors keep their checkpoint names; the target's embedding, head and final norm are packed
as target.embed, target.head and target.norm whatever the target calls them. Integer version scalars go into the
pack's config, not its payload.

Each source file's sha256 is computed and recorded. A `<file>.sha256` beside a source, or an --expect-sha256 given
for it, must match, or the pack is refused.

The output is DIR/<arm>.rank0.sp, a named pack (tools/spark_named_pack.py) with magic DRPK, its .sha256 sidecar and
a .receipt.json naming every source and its digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import NAMED_PACK_ALIGN, PayloadWriter, assemble  # noqa: E402
from spark_pack_common import PackFailure, read_safetensors_header, write_receipt, write_sha256_sidecar  # noqa: E402

DRAFTER_PACK_MAGIC = 0x4B505244
DRAFTER_PACK_VERSION = 1
CHUNK_BYTES = 64 << 20
KINDS = {"BF16": "bf16", "F32": "f32", "F8_E4M3": "fp8", "F8_E8M0": "e8m0"}

DRAFTERS = {
    "k31dspark": {
        "arm": "k31dspark.bf16.tp1",
        "config": "config.json",
        "architectures": ["K3DSparkModel"],
        "layer_count": "num_hidden_layers",
        "layer_pattern": r"^layers\.(\d+)\.",
        "include": [r"^context_norm\.weight$", r"^context_proj\.weight$", r"^final_norm\.weight$", r"^layers\.\d+\.",
                    r"^markov_head\.markov_w[12]\.weight$", r"^confidence_head\.proj\.(weight|bias)$"],
        "dtypes": ["BF16"],
        "target": {"target.embed": "language_model.model.embed_tokens.weight",
                   "target.head": "language_model.lm_head.weight",
                   "target.norm": "language_model.model.norm.weight"},
        "skip": {r"^language_model\.model\.output_attn_res_(norm|proj)\.weight$":
                 "the target's AttnRes output projection, which the drafter does not read"},
        "target_from_checkpoint": False,
    },
    "dsv41dspark": {
        "arm": "dsv41dspark.mxfp4.tp1",
        "config": "inference/config.json",
        "architectures": None,
        "layer_count": None,
        "layer_pattern": r"^mtp\.(\d+)\.",
        "include": [r"^mtp\.\d+\."],
        "dtypes": ["BF16", "F32", "F8_E4M3", "F8_E8M0", "I8"],
        "packed_4bit": r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$",
        "target": {"target.embed": "embed.weight", "target.head": "head.weight", "target.norm": "norm.weight"},
        "skip": {r"^image_(start|end|newline)$": "image tokens; the drafter drafts text"},
        "target_from_checkpoint": True,
    },
    "qwen27bdspine": {
        "arm": "qwen27bdspine.bf16.tp1",
        "config": "config.json",
        "architectures": ["DFlashDraftModel"],
        "layer_count": "num_hidden_layers",
        "layer_pattern": r"^layers\.(\d+)\.",
        "include": [r"^fc\.weight$", r"^hidden_norm\.weight$", r"^norm\.weight$", r"^layers\.\d+\.",
                    r"^selected_prefix_bridge\.(codebook|early_read_weights|feature_write_weights|read_weight)$",
                    r"^selected_prefix_bridge\.(gate|out|predecessor_gate|prev)\.(weight|bias)$"],
        "dtypes": ["BF16"],
        "scalars": r"(^|\.)\w*_version$",
        "target": {"target.embed": "model.language_model.embed_tokens.weight",
                   "target.head": "lm_head.weight",
                   "target.norm": "model.language_model.norm.weight"},
        "target_dtypes": ["BF16", "F32"],
        "skip": {},
        "target_from_checkpoint": False,
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def checkpoint_index(directory: Path):
    index = directory / "model.safetensors.index.json"
    return json.loads(index.read_text())["weight_map"] if index.is_file() else None


def checkpoint_files(directory: Path, weight_map) -> list:
    files = sorted(directory.glob("*.safetensors"))
    if weight_map is not None and not {path.name for path in files} <= set(weight_map.values()):
        raise PackFailure(f"{directory}: the safetensors files and model.safetensors.index.json disagree")
    if not files:
        raise PackFailure(f"{directory}: no safetensors files")
    return files


def read_tensors(path: Path) -> dict:
    header, base, file_bytes = read_safetensors_header(path)
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        if base + end > file_bytes or end < begin:
            raise PackFailure(f"{path.name}: {name} lies outside the file")
        tensors[name] = {"file": path, "dtype": meta["dtype"], "shape": list(meta["shape"]),
                         "offset": base + begin, "bytes": end - begin}
    return tensors


def matches(patterns, name: str) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def skip_reason(spec: dict, name: str):
    for pattern, reason in spec["skip"].items():
        if re.search(pattern, name):
            return reason
    return None


def chunks(tensor: dict):
    remaining = tensor["bytes"]
    with tensor["file"].open("rb") as handle:
        handle.seek(tensor["offset"])
        while remaining > 0:
            part = handle.read(min(CHUNK_BYTES, remaining))
            if not part:
                raise PackFailure(f"{tensor['file'].name}: short read")
            remaining -= len(part)
            yield part


def scalar_value(tensor: dict) -> int:
    if tensor["dtype"] != "I64" or tensor["shape"] != [] or tensor["bytes"] != 8:
        raise PackFailure(f"version scalar of dtype {tensor['dtype']} shape {tensor['shape']}")
    raw = b"".join(chunks(tensor))
    return struct.unpack("<q", raw)[0]


def verify_digest(path: Path, computed: str, expected: dict) -> None:
    sidecar = path.parent / (path.name + ".sha256")
    wanted = [expected[path.name]] if path.name in expected else []
    if sidecar.is_file():
        wanted.append(sidecar.read_text().split()[0])
    for digest in wanted:
        if digest.lower() != computed:
            raise PackFailure(f"{path.name}: sha256 {computed}, its receipt says {digest}")


def plan(spec: dict, checkpoint: Path, targets: list) -> tuple:
    config_path = checkpoint / spec["config"]
    if not config_path.is_file():
        raise PackFailure(f"{config_path}: missing")
    config = json.loads(config_path.read_text())
    if spec["architectures"] is not None and config.get("architectures") != spec["architectures"]:
        raise PackFailure(f"{config_path}: architectures {config.get('architectures')}, expected {spec['architectures']}")
    layer_count = config.get(spec["layer_count"]) if spec["layer_count"] else None
    if spec["target_from_checkpoint"] and targets:
        raise PackFailure("this drafter's target tensors are in its own checkpoint; it takes no --target")
    if not spec["target_from_checkpoint"] and not targets:
        raise PackFailure("this drafter needs the target's files (--target)")
    weight_map = checkpoint_index(checkpoint)
    drafter_files = checkpoint_files(checkpoint, weight_map)
    sources = drafter_files + list(targets)
    checkpoint_names = set()
    seen = {}
    for path in sources:
        for name, tensor in read_tensors(path).items():
            if name in seen:
                raise PackFailure(f"{name} appears in {seen[name]['file'].name} and {path.name}")
            seen[name] = tensor
            if path in drafter_files:
                checkpoint_names.add(name)
    packed, scalars, skipped = [], {}, {}
    target_names = {source: canonical for canonical, source in spec["target"].items()}
    absent = sorted(name for name in (weight_map or {}) if name not in seen and
                    (name in target_names or matches(spec["include"], name)))
    if absent:
        raise PackFailure(f"{absent[0]} is in {weight_map[absent[0]]}, which is not in {checkpoint} "
                          f"({len(absent)} drafter tensors are absent)")
    for name in sorted(seen):
        tensor = seen[name]
        reason = skip_reason(spec, name)
        if reason is not None:
            skipped[name] = reason
            continue
        if spec.get("scalars") and re.search(spec["scalars"], name) and tensor["dtype"] == "I64":
            scalars[name] = scalar_value(tensor)
            continue
        if name in target_names:
            allowed = spec.get("target_dtypes", ["BF16"])
            if tensor["dtype"] not in allowed:
                raise PackFailure(f"{name}: dtype {tensor['dtype']}, the target tensor must be {'/'.join(allowed)}")
            packed.append((target_names[name], name, tensor, KINDS[tensor["dtype"]]))
            continue
        if name not in checkpoint_names or not matches(spec["include"], name):
            raise PackFailure(f"{tensor['file'].name}: {name} is neither packed nor left out by a named rule")
        layer = re.search(spec["layer_pattern"], name)
        if layer is not None and layer_count is not None and int(layer.group(1)) >= layer_count:
            raise PackFailure(f"{name}: layer {layer.group(1)} is past {spec['layer_count']} = {layer_count}")
        if tensor["dtype"] == "I8" and spec.get("packed_4bit") and re.search(spec["packed_4bit"], name):
            kind = "mxfp4_e2m1"
        elif tensor["dtype"] in spec["dtypes"] and tensor["dtype"] in KINDS:
            kind = KINDS[tensor["dtype"]]
        else:
            raise PackFailure(f"{name}: dtype {tensor['dtype']} is not one this drafter packs")
        packed.append((name, name, tensor, kind))
    missing = sorted(source for source in target_names if source not in seen)
    if missing:
        raise PackFailure(f"target tensors missing: {', '.join(missing)}")
    return config, sources, packed, scalars, skipped


def build(drafter: str, checkpoint: Path, targets: list, revision: str, out_dir: Path, expected: dict) -> Path:
    if drafter not in DRAFTERS:
        raise PackFailure(f"unknown drafter {drafter}; known: {', '.join(sorted(DRAFTERS))}")
    spec = DRAFTERS[drafter]
    config, sources, packed, scalars, skipped = plan(spec, checkpoint, targets)
    digests = []
    for path in sources:
        computed = sha256_of(path)
        verify_digest(path, computed, expected)
        digests.append({"file": path.name, "bytes": path.stat().st_size, "sha256": computed})
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / f"{spec['arm']}.rank0.sp"
    payload_path = out_dir / f"{spec['arm']}.rank0.sp.payload"
    identity = json.dumps({"drafter": drafter, "revision": revision, "sources": digests}, sort_keys=True)
    writer = PayloadWriter(payload_path, identity)
    for packed_name, source_name, tensor, kind in packed:
        writer.add_parts(packed_name, chunks(tensor), kind, tensor["shape"],
                         {"source": source_name, "source_file": tensor["file"].name, "dtype": tensor["dtype"]})
    tensors = dict(writer.manifest)
    writer.close()
    fmt = {"pack": "drafter", "drafter": drafter, "layout": "checkpoint", "version": DRAFTER_PACK_VERSION}
    pack_config = {"drafter": drafter, "arm": spec["arm"], "revision": revision, "checkpoint_config": config,
                   "scalars": scalars, "skipped": skipped, "sources": digests}
    assemble(pack_path, payload_path, DRAFTER_PACK_MAGIC, DRAFTER_PACK_VERSION, NAMED_PACK_ALIGN, fmt, pack_config, tensors)
    pack_digest = write_sha256_sidecar(pack_path)
    write_receipt({"pack": pack_path.name, "sha256": pack_digest, "drafter": drafter, "revision": revision,
                   "tensors": len(tensors), "sources": digests, "skipped": skipped, "scalars": scalars}, pack_path)
    print(f"{pack_path.name}: {len(tensors)} tensors, {pack_path.stat().st_size} bytes, sha256 {pack_digest}")
    return pack_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--target", action="append", default=[], type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-sha256", action="append", default=[])
    options = parser.parse_args(argv)
    expected = {}
    for item in options.expect_sha256:
        name, _, digest = item.partition("=")
        if not digest:
            raise PackFailure(f"--expect-sha256 {item}: want FILE=HEX")
        expected[Path(name).name] = digest
    try:
        build(options.drafter, options.checkpoint, options.target, options.revision, options.out, expected)
    except (PackFailure, OSError, ValueError, KeyError) as failure:
        print(f"drafter_pack: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
