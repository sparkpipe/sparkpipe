import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import (f32_to_bf16_u16, parse_llm_defines,
                                 sha256_file, write_fixture, write_manifest)


def load_engine(family, checkpoint, header):
    module = importlib.import_module(f"t1_reference_{family}")
    defines = parse_llm_defines(header)
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    return module.ENGINE_CLASS(checkpoint, defines, config)


def state_paths(state_dir, prompt_name):
    return (os.path.join(state_dir, f"{prompt_name}.streams.npy"),
            os.path.join(state_dir, f"{prompt_name}.caches.npz"),
            os.path.join(state_dir, f"{prompt_name}.tokens.json"),
            os.path.join(state_dir, f"{prompt_name}.log.jsonl"))


def load_state(state_dir, prompt_name, engine):
    streams_path, caches_path, tokens_path, _ = state_paths(state_dir,
                                                            prompt_name)
    if not os.path.exists(streams_path):
        return None, {}, []
    states = {"streams": np.load(streams_path)}
    caches = {}
    with np.load(caches_path) as data:
        for il in range(engine.layers):
            latents = data[f"latents_{il}"]
            pes = data[f"pes_{il}"]
            caches[il] = ([latents[i].copy() for i in range(latents.shape[0])],
                          [pes[i].copy() for i in range(pes.shape[0])])
    tokens = json.load(open(tokens_path))
    return states, caches, tokens


def save_state(state_dir, prompt_name, engine, streams, caches, tokens):
    os.makedirs(state_dir, exist_ok=True)
    streams_path, caches_path, tokens_path, _ = state_paths(state_dir,
                                                            prompt_name)
    np.save(streams_path, streams)
    arrays = {}
    for il in range(engine.layers):
        latents = np.stack(caches[il][0]) if caches.get(il) else \
            np.zeros((0, engine.kv_lora), dtype=np.float32)
        pes = np.stack(caches[il][1]) if caches.get(il) else \
            np.zeros((0, engine.rot), dtype=np.float32)
        arrays[f"latents_{il}"] = latents.astype(np.float32)
        arrays[f"pes_{il}"] = pes.astype(np.float32)
    np.savez(caches_path, **arrays)
    with open(tokens_path, "w") as fh:
        json.dump(tokens, fh)


def load_log(state_dir, prompt_name):
    log_path = state_paths(state_dir, prompt_name)[3]
    if not os.path.exists(log_path):
        return []
    with open(log_path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay(arrays, generated, lines, prompt_ids):
    for line in lines:
        position = line["position"]
        for key, value in line.get("snap", {}).items():
            arrays[key] = np.array(value, dtype=np.uint16)
        for il, (ids, weights) in line.get("routes", {}).items():
            arrays[f"pos{position:04d}_layer{int(il):04d}_route_ids"] = \
                np.array(ids, dtype=np.int32)
            arrays[f"pos{position:04d}_layer{int(il):04d}_route_weights"] = \
                np.array(weights, dtype=np.float32)
        if position >= len(prompt_ids):
            generated.append(line["fed"])
        if position >= len(prompt_ids) - 1:
            arrays[f"pos{position:04d}_head_top1_score"] = \
                np.array([line["score"]], dtype=np.float32)
            arrays[f"pos{position:04d}_head_top1_token"] = \
                np.array([line["top1"]], dtype=np.int32)


def run_prompt(engine, arguments, spec, arrays, generated, done, states,
               caches, tokens, stop_after):
    name = spec["name"]
    prompt_ids = [int(t) for t in spec["prompt_token_ids"]]
    total = len(prompt_ids) + int(spec["new_tokens"])
    anchors = set(int(l) for l in spec.get("capture_layers", []))
    new_positions = 0
    for position in range(done, total):
        if stop_after and new_positions >= stop_after:
            return done + new_positions
        new_positions += 1
        snap_data = {}
        capture = {}

        def snap(layer_index, layer_streams, position=position,
                 anchors=anchors, snap_data=snap_data):
            if layer_index in anchors:
                snap_data[f"pos{position:04d}_layer{layer_index:04d}_"
                          "streams"] = f32_to_bf16_u16(
                    layer_streams.reshape(-1))

        streams = engine.decode_step(tokens[position], position, states,
                                     caches, capture, snap)
        token, score = engine.logits(streams)
        entry = {"prompt": name, "position": position,
                 "fed": int(tokens[position]), "top1": int(token),
                 "score": float(score)}
        if snap_data:
            entry["snap"] = {key: value.tolist()
                             for key, value in snap_data.items()}
            for key, value in snap_data.items():
                arrays[key] = value
        routes = {}
        for (_route_position, il), (ids, weights) in capture.items():
            routes[str(il)] = [ids.tolist(), weights.tolist()]
            arrays[f"pos{position:04d}_layer{int(il):04d}_route_ids"] = \
                ids.astype(np.int32)
            arrays[f"pos{position:04d}_layer{int(il):04d}_route_weights"] = \
                weights.astype(np.float32)
        if routes:
            entry["routes"] = routes
        if position + 1 < total and position + 1 >= len(prompt_ids):
            generated.append(int(token))
        if position >= len(prompt_ids) - 1:
            arrays[f"pos{position:04d}_head_top1_score"] = \
                np.array([score], dtype=np.float32)
            arrays[f"pos{position:04d}_head_top1_token"] = \
                np.array([token], dtype=np.int32)
        with open(state_paths(arguments.state_dir, name)[3], "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if position + 1 < total:
            if position + 1 < len(prompt_ids):
                tokens.append(prompt_ids[position + 1])
            else:
                if token == engine.eot:
                    raise ValueError(f"{name}: eot at position {position}")
                tokens.append(int(token))
            save_state(arguments.state_dir, name, engine, streams, caches,
                       tokens)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="exit after this many fresh positions")
    arguments = parser.parse_args()

    engine = load_engine(arguments.family, arguments.checkpoint,
                         arguments.header)
    prompts = json.load(open(arguments.prompts))
    out_dir = os.path.join(arguments.output, arguments.family)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(arguments.state_dir, exist_ok=True)
    fixtures = {}
    for spec in prompts["prompts"]:
        name = spec["name"]
        prompt_ids = [int(t) for t in spec["prompt_token_ids"]]
        total = len(prompt_ids) + int(spec["new_tokens"])
        arrays = {}
        generated = []
        lines = load_log(arguments.state_dir, name)
        done = len(lines)
        replay(arrays, generated, lines, prompt_ids)
        if done >= total:
            print(json.dumps({"prompt": name, "status": "complete"}),
                  flush=True)
        else:
            states, caches, tokens = load_state(arguments.state_dir, name,
                                                engine)
            if done == 0:
                states = {}
                caches = {}
                tokens = list(prompt_ids)
            run_prompt(engine, arguments, spec, arrays, generated, done,
                       states, caches, tokens, arguments.stop_after)
        arrays["prompt_token_ids"] = np.array(prompt_ids, dtype=np.int32)
        arrays["generated_token_ids"] = np.array(generated, dtype=np.int32)
        fixture_path = os.path.join(out_dir, f"{name}.t1r")
        write_fixture(fixture_path, arrays)
        fixtures[f"{name}.t1r"] = {
            "sha256": sha256_file(fixture_path),
            "bytes": os.path.getsize(fixture_path)}
        print(json.dumps({"prompt": name, "fixture": fixture_path,
                          "generated": generated}), flush=True)
    complete = all(f"{spec['name']}.t1r" in fixtures
                   for spec in prompts["prompts"])
    if complete:
        index_path = os.path.join(arguments.checkpoint,
                                  "model.safetensors.index.json")
        document = {
            "family": arguments.family,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "host": os.uname().nodename,
            "threading_env": {k: os.environ.get(k) for k in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "MKL_NUM_THREADS")},
            "generator": "t1_reference_chunk_decoder.py",
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "checkpoint": {
                "path": os.path.abspath(arguments.checkpoint),
                "config_sha256": sha256_file(
                    os.path.join(arguments.checkpoint, "config.json")),
                "index_sha256": sha256_file(index_path)
                if os.path.exists(index_path) else None,
            },
            "llm_defines_sha256": hashlib.sha256(
                open(arguments.header, "rb").read()).hexdigest(),
            "prompts_sha256": hashlib.sha256(
                open(arguments.prompts, "rb").read()).hexdigest(),
            "defines_config_mismatches": engine.mismatches,
            "fixtures": fixtures,
        }
        write_manifest(os.path.join(out_dir, "MANIFEST.json"), document)
        print(json.dumps({"manifest": os.path.join(out_dir,
                                                   "MANIFEST.json")}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
