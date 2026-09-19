import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import (f32_to_bf16_u16, parse_llm_defines, sha256_file,
                                 write_fixture, write_manifest)


def sha_of_text(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_engine(family, checkpoint, header):
    module = importlib.import_module(f"t1_reference_{family}")
    defines = parse_llm_defines(header)
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    return module.ENGINE_CLASS(checkpoint, defines, config)


def run_prompt(engine, spec):
    prompt_ids = [int(t) for t in spec["prompt_token_ids"]]
    budget = int(spec["new_tokens"])
    anchors = [int(l) for l in spec.get("capture_layers", [])]
    states = {}
    caches = {}
    capture = {}
    arrays = {}
    tokens = list(prompt_ids)
    generated = []
    total = len(prompt_ids) + budget
    snap_params = "capture_streams" in inspect.signature(engine.decode_step).parameters
    for position in range(total):
        anchor_set = set(anchors)

        def snap(layer_index, layer_streams):
            if layer_index in anchor_set:
                arrays[f"pos{position:04d}_layer{layer_index:04d}_streams"] = \
                    f32_to_bf16_u16(layer_streams.reshape(-1))

        if snap_params:
            streams = engine.decode_step(tokens[position], position, states,
                                         caches, capture, snap)
        else:
            streams = engine.decode_step(tokens[position], position, states,
                                         caches, capture)
            for layer in anchors:
                arrays[f"pos{position:04d}_layer{layer:04d}_streams"] = \
                    f32_to_bf16_u16(streams.reshape(-1))
        print(json.dumps({"prompt": spec["name"], "position": position,
                          "done": True}), flush=True)
        token, score = engine.logits(streams)
        if position >= len(prompt_ids) - 1:
            arrays[f"pos{position:04d}_head_top1_score"] = \
                np.array([score], dtype=np.float32)
            arrays[f"pos{position:04d}_head_top1_token"] = \
                np.array([token], dtype=np.int32)
        if position + 1 < total:
            if position + 1 >= len(prompt_ids):
                if token == engine.eot:
                    raise ValueError(f"prompt {spec['name']}: eot at position {position}; "
                                     "the prompt set contract must avoid eot")
                tokens.append(token)
                generated.append(token)
    arrays["prompt_token_ids"] = np.array(prompt_ids, dtype=np.int32)
    arrays["generated_token_ids"] = np.array(generated, dtype=np.int32)
    for (position, layer), sink in capture.items():
        arrays[f"pos{position:04d}_layer{layer:04d}_route_ids"] = sink[0]
        arrays[f"pos{position:04d}_layer{layer:04d}_route_weights"] = sink[1]
    return arrays


def generate(arguments):
    engine = load_engine(arguments.family, arguments.checkpoint, arguments.header)
    prompts = json.load(open(arguments.prompts))
    out_dir = os.path.join(arguments.output, arguments.family)
    os.makedirs(out_dir, exist_ok=True)
    fixtures = {}
    for spec in prompts["prompts"]:
        arrays = run_prompt(engine, spec)
        name = f"{spec['name']}.t1r"
        path = os.path.join(out_dir, name)
        write_fixture(path, arrays)
        fixtures[name] = {"sha256": sha256_file(path),
                          "bytes": os.path.getsize(path)}
    index_path = os.path.join(arguments.checkpoint,
                              "model.safetensors.index.json")
    document = {
        "family": arguments.family,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "threading_env": {k: os.environ.get(k) for k in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "MKL_NUM_THREADS")},
        "generator": "t1_reference_decoder.py",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "checkpoint": {
            "path": os.path.abspath(arguments.checkpoint),
            "config_sha256": sha256_file(os.path.join(arguments.checkpoint,
                                                      "config.json")),
            "index_sha256": sha256_file(index_path) if os.path.exists(index_path)
            else None,
        },
        "llm_defines_sha256": sha_of_text(arguments.header),
        "prompts_sha256": sha_of_text(arguments.prompts),
        "defines_config_mismatches": engine.mismatches,
        "fixtures": fixtures,
    }
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    write_manifest(manifest_path, document)
    print(json.dumps({"family": arguments.family, "fixtures": len(fixtures),
                      "output": out_dir, "manifest": manifest_path}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    return generate(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
