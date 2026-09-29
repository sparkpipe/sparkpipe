import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import glm53flash_mtp_reference as reference
from t1_reference_common import bf16_round_f32, rmsnorm

HIDDEN = 8
EPS = 1e-5


class StubEngine:
    def __init__(self, half):
        self.eps = EPS
        self.hc = 2
        self.hidden = HIDDEN
        select = np.zeros((HIDDEN, 2 * HIDDEN), dtype=np.float32)
        select[np.arange(HIDDEN), half * HIDDEN + np.arange(HIDDEN)] = 1.0
        prefix = reference.MTP_PREFIX.format(layer=3)
        self.weights = {
            "model.language_model.norm.weight": np.full(HIDDEN, 3.0, dtype=np.float32),
            prefix + "enorm.weight": np.full(HIDDEN, 0.5, dtype=np.float32),
            prefix + "hnorm.weight": np.full(HIDDEN, 2.0, dtype=np.float32),
            prefix + "eh_proj.weight": select,
            prefix + "input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
            prefix + "post_attention_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
            prefix + "shared_head.norm.weight": np.ones(HIDDEN, dtype=np.float32),
        }
        self.attention_inputs = []

    def tensor(self, name):
        return self.weights[name]

    def embed(self, token):
        return np.arange(1, HIDDEN + 1, dtype=np.float32) * (token + 1)

    def dsa_attention(self, prefix, x, cache):
        cache.append(x)
        self.attention_inputs.append(len(cache))
        return np.zeros_like(x)

    def sparse_mlp(self, prefix, x, sink):
        return np.zeros_like(x)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    hidden = np.array([1.0, -2.0, 3.0, -4.0, 0.5, -0.25, 2.0, -1.0], dtype=np.float32)
    for half, weight_name, source in ((0, "enorm", "embed"), (1, "hnorm", "hidden")):
        engine = StubEngine(half)
        mtp = reference.MtpReference(engine, 3)
        values = engine.embed(4) if source == "embed" else hidden
        want = bf16_round_f32(rmsnorm(values, engine.tensor(reference.MTP_PREFIX.format(layer=3) + weight_name + ".weight"), EPS))
        got, _ = mtp.step(hidden, 4, "embed_hidden", [])
        expect(np.array_equal(got, want), f"embed_hidden half {half} must carry {source}: {got} vs {want}")
        swapped, _ = mtp.step(hidden, 4, "hidden_embed", [])
        expect(not np.array_equal(swapped, want), f"hidden_embed half {half} must not carry {source}")
    engine = StubEngine(0)
    mtp = reference.MtpReference(engine, 3)
    pre = np.stack([hidden, 3.0 * hidden])
    expect(np.array_equal(mtp.hidden_tap(pre, "hc_mean"), bf16_round_f32(2.0 * hidden)), "hc_mean is the plain stream mean")
    expect(np.array_equal(mtp.hidden_tap(pre, "final_norm"),
                          bf16_round_f32(rmsnorm(bf16_round_f32(2.0 * hidden), np.full(HIDDEN, 3.0, dtype=np.float32), EPS))),
           "final_norm applies the model norm to the stream mean")
    cache = []
    mtp.step(hidden, 1, "embed_hidden", cache)
    mtp.step(hidden, 2, "embed_hidden", cache)
    expect(engine.attention_inputs == [1, 2], f"sequence context keeps the MTP cache: {engine.attention_inputs}")
    print("PASS glm53flash_mtp_reference: eh_proj input is [enorm(embed) | hnorm(hidden)], taps and MTP cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
