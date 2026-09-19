import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_reference_hy4 as engine
from t1_reference_common import parse_llm_defines

CHECKPOINT = os.environ.get("HY4_PROBE_CHECKPOINT",
                            "/mnt/model-warm/hy4-preview-fp8-official")
HEADER = os.environ.get(
    "HY4_PROBE_HEADER",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "model-families", "hy4", "include", "sparkpipe",
                 "llm_defines.h"))
OUT = os.environ.get("HY4_PROBE_OUT", "/tmp/hy4_probe")
BASELINE_STREAMS = os.environ.get(
    "HY4_PROBE_BASELINE", os.path.join(OUT, "baseline_streams.npy"))
PROMPT = [802, 8778, 299, 12749, 341]
GEN = int(os.environ.get("HY4_PROBE_GEN", "4"))


def log(message):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "m": message}),
          flush=True)


def load_engine(label):
    defines = parse_llm_defines(HEADER)
    config = json.load(open(os.path.join(CHECKPOINT, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    started = time.time()
    eng = engine.Hy4Engine(CHECKPOINT, defines, config)
    log(f"{label} engine built in {time.time() - started:.1f}s "
        f"sinks shape {eng.sinks.shape}")
    return eng


def trace_summary(trace):
    rows = []
    for entry in trace:
        sink_mass = entry["sink_mass"]
        entropy = entry["entropy"]
        rows.append({
            "layer": entry["layer"],
            "position": entry["position"],
            "sink_mass_mean": float(sink_mass.mean()),
            "sink_mass_max": float(sink_mass.max()),
            "collapsed_heads": int((sink_mass > 0.999).sum()),
            "entropy_min": float(entropy.min()),
            "entropy_mean": float(entropy.mean()),
        })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    baseline_streams = np.load(BASELINE_STREAMS)
    receipt = {"prompt": PROMPT, "gen": GEN,
               "baseline_pos0_top1": {"token": 2357,
                                      "score": 13.372605323791504}}

    fixed = load_engine("fixed")
    trace = []
    fixed.attention_trace = trace
    captured = {}
    states, caches = {}, {}
    tokens = list(PROMPT)
    trajectory = []
    total = len(PROMPT) + GEN
    for position in range(total):
        def snap(layer_index, layer_streams):
            if position == 0:
                captured[layer_index] = \
                    layer_streams.astype(np.float32).copy()

        started = time.time()
        streams = fixed.decode_step(tokens[position], position, states,
                                    caches, {}, snap)
        token, score = fixed.logits(streams)
        log(f"fixed position {position} forward+logits "
            f"{time.time() - started:.0f}s token={token} score={score:.4f}")
        trajectory.append({"position": position, "fed": int(tokens[position]),
                           "top1": int(token), "score": float(score)})
        if position == 0:
            receipt["fixed_trace_pos0"] = trace_summary(trace)
            receipt["per_layer_cosine_vs_baseline"] = [
                {"layer": il,
                 "cosine": float(np.dot(captured[il].ravel(),
                                        baseline_streams[il].ravel()) /
                                 max(1e-30, float(np.linalg.norm(
                                     captured[il]) *
                                     np.linalg.norm(
                                     baseline_streams[il])))),
                 "fixed_norm": float(np.linalg.norm(captured[il])),
                 "baseline_norm": float(np.linalg.norm(
                     baseline_streams[il]))}
                for il in sorted(captured)]
            trace = []
            fixed.attention_trace = trace
        if position + 1 < len(PROMPT):
            tokens.append(PROMPT[position + 1])
        elif position + 1 < total:
            tokens.append(int(token))
    receipt["trajectory"] = trajectory
    if position == 0:
        np.save(os.path.join(OUT, "fixed_streams_pos0.npy"),
                np.stack([captured[i] for i in sorted(captured)]))
    with open(os.path.join(OUT, "probe_receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=1)
    log("probe receipt written")


if __name__ == "__main__":
    main()
