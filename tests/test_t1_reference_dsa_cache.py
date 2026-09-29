import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (bf16_round_f32, bf16_to_f32,  # noqa: E402
                                 f32_to_bf16_u16, rmsnorm)
import t1_reference_glm53flash as flash  # noqa: E402
import t1_reference_glm5_next as next_ref  # noqa: E402

HIDDEN = 16
HEADS = 2
NOPE = 4
VDIM = 4
LATENT = 8
QA = 8
EPS = 1e-5
POSITIONS = 4
PREFIX = "layers.1."


class Store:
    def __init__(self, arrays):
        self.arrays = arrays

    def raw(self, name):
        return self.arrays[name]

    def pread(self, name):
        return self.arrays[name]


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def weights(kv_rows):
    rng = np.random.default_rng(7)

    def bf16(shape, scale):
        return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32)
                               * scale)

    a = PREFIX + "self_attn."
    return {
        a + "q_a_proj.weight": bf16((QA, HIDDEN), 0.5),
        a + "q_a_layernorm.weight": bf16((QA,), 1.0),
        a + "q_b_proj.weight": bf16((HEADS * NOPE, QA), 0.5),
        a + "kv_a_proj_with_mqa.weight": bf16((kv_rows, HIDDEN), 0.5),
        a + "kv_a_layernorm.weight": bf16((LATENT,), 1.0),
        a + "kv_b_proj.weight": bf16((HEADS * (NOPE + VDIM), LATENT), 0.5),
        a + "o_proj.weight": bf16((HIDDEN, HEADS * VDIM), 0.5),
    }


def inputs():
    rng = np.random.default_rng(13)
    return [bf16_round_f32(rng.standard_normal(HIDDEN).astype(np.float32))
            for _ in range(POSITIONS)]


def hand_attention(arrays, xs):
    a = PREFIX + "self_attn."

    def w(name):
        return bf16_to_f32(arrays[a + name])

    rows = []
    outputs = []
    kvb = w("kv_b_proj.weight")
    for x in xs:
        q_norm = bf16_round_f32(rmsnorm(bf16_round_f32(w("q_a_proj.weight")
                                                       @ x),
                                        w("q_a_layernorm.weight"), EPS))
        q = bf16_round_f32(q_norm @ w("q_b_proj.weight").T)
        kv = bf16_round_f32(w("kv_a_proj_with_mqa.weight") @ x)[:LATENT]
        rows.append(bf16_round_f32(rmsnorm(kv, w("kv_a_layernorm.weight"),
                                           EPS)))
        latent_rows = np.stack(rows)
        attn = np.empty((HEADS, VDIM), dtype=np.float32)
        for h in range(HEADS):
            base = h * (NOPE + VDIM)
            ql = bf16_round_f32(q[h * NOPE:(h + 1) * NOPE]
                                @ kvb[base:base + NOPE])
            scores = (latent_rows @ ql) * np.float32(NOPE ** -0.5)
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            al = bf16_round_f32(p @ latent_rows)
            attn[h] = bf16_round_f32(al @ kvb[base + NOPE:base + NOPE
                                              + VDIM].T)
        outputs.append(bf16_round_f32(w("o_proj.weight") @ attn.reshape(-1)))
    return rows, outputs


def flash_engine(arrays):
    engine = object.__new__(flash.Glm53FlashEngine)
    engine.raw = Store(arrays).raw
    engine.heads, engine.nope, engine.vdim = HEADS, NOPE, VDIM
    engine.latent, engine.eps = LATENT, EPS
    return engine


def next_engine(arrays):
    engine = object.__new__(next_ref.ENGINE_CLASS)
    engine.st = Store(arrays)
    engine.heads, engine.nope, engine.vdim = HEADS, NOPE, VDIM
    engine.eps = EPS
    return engine


def check(label, engine, arrays):
    xs = inputs()
    rows, want = hand_attention(arrays, xs)
    cache = []
    for position, x in enumerate(xs):
        got = engine.dsa_attention(PREFIX, x, cache)
        expect(np.array_equal(got, want[position]),
               f"{label}: position {position} attention diverges from the "
               f"hand-computed latent attention: max "
               f"{float(np.abs(got - want[position]).max())}")
    expect(len(cache) == POSITIONS, f"{label}: cache holds {len(cache)} rows")
    for position, row in enumerate(cache):
        expect(np.asarray(row).dtype == np.uint16,
               f"{label}: cached latent row {position} is "
               f"{np.asarray(row).dtype}, not bf16 codes")
        expect(np.array_equal(bf16_to_f32(row), rows[position]),
               f"{label}: cached latent row {position} does not decode to "
               f"the normalized latent")
    expect(float(np.abs(want[-1]).max()) > 1e-3,
           f"{label}: the hand-computed attention must be non-trivial")


def main():
    check("glm53flash", flash_engine(weights(LATENT + 4)),
          weights(LATENT + 4))
    check("glm5_next", next_engine(weights(LATENT)), weights(LATENT))
    try:
        bf16_to_f32(np.array([0.5, 3.0], dtype=np.float32))
        raise AssertionError("bf16_to_f32 must refuse float input")
    except TypeError as error:
        expect("uint16" in str(error), f"refusal must name uint16: {error}")
    codes = f32_to_bf16_u16(np.array([0.5, -3.0], dtype=np.float32))
    expect(np.array_equal(bf16_to_f32(codes),
                          np.array([0.5, -3.0], dtype=np.float32)),
           "bf16_to_f32 must decode uint16 codes")
    print("PASS t1_reference DSA latent cache: glm53flash and glm5_next "
          "cache bf16 codes and attend over every decoded row, bitwise "
          "equal to the hand-computed attention; bf16_to_f32 refuses "
          "non-uint16 input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
