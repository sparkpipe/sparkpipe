#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import platform
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t1_reference_common import f32_to_bf16_u16, write_fixture, write_manifest

HIDDEN = 6144
STREAMS = 4
LAYERS = 78
TOP_K = 8
HEAD_TOP = 5


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(checkpoint, output):
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(index.values()))
    lines = []
    for number, name in enumerate(files):
        lines.append(f"F {number} {os.path.join(checkpoint, name)}")
    for number, name in enumerate(files):
        with open(os.path.join(checkpoint, name), "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
        for tensor, entry in sorted(header.items()):
            if tensor == "__metadata__":
                continue
            if index.get(tensor) != name:
                raise SystemExit(f"{tensor} is in {name} but the index names {index.get(tensor)}")
            start, stop = entry["data_offsets"]
            shape = " ".join(str(dim) for dim in entry["shape"])
            lines.append(f"T {tensor} {number} {8 + size + start} {stop - start} {entry['dtype']} {len(entry['shape'])} {shape}")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest {output}: {len(files)} files, {len(lines) - len(files)} tensors")


def build_prompts(prompts, output):
    document = json.load(open(prompts))
    lines = []
    for prompt in document["prompts"]:
        capture = prompt["capture_layers"]
        ids = prompt["prompt_token_ids"]
        fields = [prompt["name"], prompt["new_tokens"], len(capture), len(ids)] + capture + ids
        lines.append(" ".join(str(value) for value in fields))
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_text(path):
    prompt = []
    generated = []
    heads = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[0] == "prompt":
            prompt = [int(value) for value in fields[1:]]
        elif fields[0] == "generated":
            generated = [int(value) for value in fields[1:]]
        elif fields[0] == "head":
            pairs = [field.split(":") for field in fields[2:]]
            heads[int(fields[1])] = ([int(token) for token, _ in pairs], [float(score) for _, score in pairs])
    return prompt, generated, heads


def build_fixtures(prompts, run_dir, output_dir, checkpoint, host):
    document = json.load(open(prompts))
    os.makedirs(output_dir, exist_ok=True)
    fixtures = {}
    for prompt in document["prompts"]:
        name = prompt["name"]
        ids, generated, heads = parse_text(os.path.join(run_dir, f"{name}.txt"))
        if ids != prompt["prompt_token_ids"]:
            raise SystemExit(f"{name}: run prompt ids differ from {prompts}")
        if len(generated) != prompt["new_tokens"]:
            raise SystemExit(f"{name}: run generated {len(generated)} of {prompt['new_tokens']} tokens")
        total = len(ids) + len(generated)
        processed = total - 1
        routes = np.fromfile(os.path.join(run_dir, f"{name}.routes.bin"), dtype=np.uint8)
        route_bytes = total * (LAYERS - 1) * TOP_K * 4
        if routes.size != 2 * route_bytes:
            raise SystemExit(f"{name}: routes file holds {routes.size} bytes, expected {2 * route_bytes}")
        route_ids = routes[:route_bytes].view(np.int32).reshape(total, LAYERS - 1, TOP_K)
        route_weights = routes[route_bytes:].view(np.float32).reshape(total, LAYERS - 1, TOP_K)
        capture = prompt["capture_layers"]
        streams = np.fromfile(os.path.join(run_dir, f"{name}.streams.bin"), dtype=np.float32)
        if streams.size != len(capture) * total * STREAMS * HIDDEN:
            raise SystemExit(f"{name}: streams file size disagrees with {len(capture)} capture layers")
        streams = streams.reshape(len(capture), total, STREAMS * HIDDEN)
        arrays = {
            "prompt_token_ids": np.asarray(ids, dtype=np.int32),
            "generated_token_ids": np.asarray(generated, dtype=np.int32),
        }
        for position in range(processed):
            for layer in range(1, LAYERS):
                arrays[f"pos{position:04d}_layer{layer:04d}_route_ids"] = route_ids[position, layer - 1].copy()
                arrays[f"pos{position:04d}_layer{layer:04d}_route_weights"] = route_weights[position, layer - 1].copy()
            if position < len(ids):
                for slot, layer in enumerate(capture):
                    arrays[f"pos{position:04d}_layer{layer:04d}_streams"] = f32_to_bf16_u16(streams[slot, position])
        for position in range(len(ids) - 1, processed):
            tokens, scores = heads[position]
            if position >= len(ids) - 1 and tokens[0] != (ids + generated)[position + 1]:
                raise SystemExit(f"{name}: head top1 at {position} is {tokens[0]}, generated {(ids + generated)[position + 1]}")
            arrays[f"pos{position:04d}_head_top1_token"] = np.asarray([tokens[0]], dtype=np.int32)
            arrays[f"pos{position:04d}_head_top1_score"] = np.asarray([scores[0]], dtype=np.float32)
            arrays[f"pos{position:04d}_head_top5_tokens"] = np.asarray(tokens, dtype=np.int32)
            arrays[f"pos{position:04d}_head_top5_scores"] = np.asarray(scores, dtype=np.float32)
        path = os.path.join(output_dir, f"{name}.t1r")
        write_fixture(path, arrays)
        fixtures[f"{name}.t1r"] = {"bytes": os.path.getsize(path), "sha256": sha256_path(path)}
        print(f"{name}: prompt {ids} generated {generated}")
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "checkpoint": {
            "config_sha256": sha256_path(os.path.join(checkpoint, "config.json")),
            "index_sha256": sha256_path(os.path.join(checkpoint, "model.safetensors.index.json")),
            "path": checkpoint,
        },
        "family": "hy4",
        "fixtures": fixtures,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/hy4_reference/hy4_fp8_reference.c",
        "generator_sha256": sha256_path(root / "tools" / "hy4_reference" / "hy4_fp8_reference.c"),
        "host": host,
        "numerics": "fp32 activations and accumulation, publisher MXFP8 weights dequantized exactly (E4M3 x E8M0 per 32 columns), BF16 planes widened exactly; DSA indexer selection is the identity below 2048 context tokens",
        "prompts_sha256": sha256_path(prompts),
        "python_version": platform.python_version(),
    }
    write_manifest(os.path.join(output_dir, "MANIFEST.json"), manifest)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--checkpoint", required=True)
    manifest.add_argument("--output", required=True)
    prompts = commands.add_parser("prompts")
    prompts.add_argument("--prompts", required=True)
    prompts.add_argument("--output", required=True)
    fixtures = commands.add_parser("fixtures")
    fixtures.add_argument("--prompts", required=True)
    fixtures.add_argument("--run", required=True)
    fixtures.add_argument("--output", required=True)
    fixtures.add_argument("--checkpoint", required=True)
    fixtures.add_argument("--host", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest(args.checkpoint, args.output)
    elif args.command == "prompts":
        build_prompts(args.prompts, args.output)
    else:
        build_fixtures(args.prompts, args.run, args.output, args.checkpoint, args.host)


if __name__ == "__main__":
    main()
