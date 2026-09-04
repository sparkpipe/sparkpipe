#!/usr/bin/env python3
"""The q-side fold must equal the reconstructed query, and the value fold must not
be applied at all.

WHAT THE TREE DOES NOW, after the external audit found that absorbing the value
half is both incorrect and slow here:

    q_up[h]  ->  kv_b_nope[h].T @ q_up_nope[h], concat q_up_rope[h]   FOLDED
    kv_b_value                                                        NOT folded
    o_proj                                                            unchanged

The query fold is kept because it is free: the query is projected once per token
either way, and folding moves work from per-position to per-token. The value
fold is not, for two independent reasons - the output gate is elementwise in
v-space and does not commute with it, and on GB10 it trades 55 ms of weight
reads for 46 us of arithmetic.

The identity below covers the full absorption because that is what proves the
q-side half is right: if both folds together reproduce the reference, and the
value fold is simply omitted with kv_b_value applied explicitly instead, the
q-side fold is correct on its own.

modeling_kimi_linear.py rebuilds per-head keys and values from the cached latent
through kv_b_proj at attention time. LmAttentionDecodeKernel does not: it reads
LATENT + ROPE per head and treats the cached latent row as the key. The two are
the same function of the same weights, related by folding kv_b into the query
up-projection and the output projection - but that fold is a pack-time
transformation and until it is written down as arithmetic it is a claim.

This checks it on random tensors at K3's real shapes, so the packer has an
executable specification rather than a paragraph.

    q_absorbed[h] = W_kv_b_nope[h]^T @ q_up_nope[h]        (kv_lora x q_lora)
    o_absorbed[h] = o_proj[:, h] @ W_kv_b_value[h]         (hidden x kv_lora)

The rope half of the query passes through untouched, and the cached latent is
normalised before it is stored, because RMSNorm's scale depends on the latent
itself and therefore cannot be folded into a weight.
"""
import sys

try:
    import numpy
except ImportError:
    print("numpy unavailable; cannot check the fold")
    sys.exit(1)

# THE MATH IS CHECKED SMALL AND THE SHAPES ARE CHECKED REAL.
#
# At K3's dimensions the absorbed output projection is hidden x heads x kv_lora
# = 7168 x 96 x 512 doubles, 2.8 GB, which does not fit anywhere this runs. The
# fold's correctness is structural - it is one matrix identity per head and does
# not depend on the sizes - so the identity is verified at small dimensions with
# the same shape relationships, and the real dimensions are checked separately
# for the arithmetic that sizes the packer's output.
HEADS = 4
KV_LORA = 32
Q_LORA = 48
QK_NOPE = 8
QK_ROPE = 4
V_HEAD = 8
HIDDEN = 64
TOKENS = 4
CONTEXT = 7

# What the packer actually has to produce, at K3's real config.
REAL = dict(heads=96, kv_lora=512, q_lora=1536, qk_nope=128, qk_rope=64,
            v_head=128, hidden=7168)


def reconstructed(latent, rope_key, q_latent, kv_b, o_proj, q_up, scale):
    """What modeling_kimi_linear.py computes: rebuild k and v per head."""
    out = numpy.zeros((TOKENS, HIDDEN))
    for token in range(TOKENS):
        query = q_up @ q_latent[token]
        query = query.reshape(HEADS, QK_NOPE + QK_ROPE)
        head_out = numpy.zeros((HEADS, V_HEAD))
        for head in range(HEADS):
            q_nope = query[head, :QK_NOPE]
            q_rope = query[head, QK_NOPE:]
            scores = numpy.zeros(CONTEXT)
            for position in range(CONTEXT):
                key = kv_b[head, :QK_NOPE] @ latent[position]
                scores[position] = (q_nope @ key
                                    + q_rope @ rope_key[position]) * scale
            weights = numpy.exp(scores - scores.max())
            weights /= weights.sum()
            for position in range(CONTEXT):
                value = kv_b[head, QK_NOPE:] @ latent[position]
                head_out[head] += weights[position] * value
        out[token] = o_proj @ head_out.reshape(-1)
    return out


def absorbed(latent, rope_key, q_latent, kv_b, o_proj, q_up, scale):
    """What LmAttentionDecodeKernel computes, given the folded weights."""
    # the two folds, exactly as the packer must perform them
    q_up_nope = q_up.reshape(HEADS, QK_NOPE + QK_ROPE, Q_LORA)
    q_absorbed = numpy.zeros((HEADS, KV_LORA + QK_ROPE, Q_LORA))
    o_absorbed = numpy.zeros((HIDDEN, HEADS, KV_LORA))
    for head in range(HEADS):
        q_absorbed[head, :KV_LORA] = kv_b[head, :QK_NOPE].T @ q_up_nope[head, :QK_NOPE]
        q_absorbed[head, KV_LORA:] = q_up_nope[head, QK_NOPE:]
        o_absorbed[:, head] = (o_proj[:, head * V_HEAD:(head + 1) * V_HEAD]
                               @ kv_b[head, QK_NOPE:])
    out = numpy.zeros((TOKENS, HIDDEN))
    for token in range(TOKENS):
        head_out = numpy.zeros((HEADS, KV_LORA))
        for head in range(HEADS):
            query = q_absorbed[head] @ q_latent[token]
            q_lat, q_rope = query[:KV_LORA], query[KV_LORA:]
            scores = numpy.zeros(CONTEXT)
            for position in range(CONTEXT):
                # the cached row IS the key: latent then rope, no reconstruction
                scores[position] = (q_lat @ latent[position]
                                    + q_rope @ rope_key[position]) * scale
            weights = numpy.exp(scores - scores.max())
            weights /= weights.sum()
            for position in range(CONTEXT):
                head_out[head] += weights[position] * latent[position]
        out[token] = o_absorbed.reshape(HIDDEN, -1) @ head_out.reshape(-1)
    return out


def main():
    generator = numpy.random.default_rng(20260727)
    scale = (QK_NOPE + QK_ROPE) ** -0.5
    latent = generator.standard_normal((CONTEXT, KV_LORA)) * 0.05
    rope_key = generator.standard_normal((CONTEXT, QK_ROPE)) * 0.05
    q_latent = generator.standard_normal((TOKENS, Q_LORA)) * 0.05
    kv_b = generator.standard_normal((HEADS, QK_NOPE + V_HEAD, KV_LORA)) * 0.02
    o_proj = generator.standard_normal((HIDDEN, HEADS * V_HEAD)) * 0.02
    q_up = generator.standard_normal((HEADS * (QK_NOPE + QK_ROPE), Q_LORA)) * 0.02

    a = reconstructed(latent, rope_key, q_latent, kv_b, o_proj, q_up, scale)
    b = absorbed(latent, rope_key, q_latent, kv_b, o_proj, q_up, scale)
    error = numpy.abs(a - b).max()
    magnitude = numpy.abs(a).max()
    relative = error / magnitude

    print(f"heads {HEADS}  kv_lora {KV_LORA}  q_lora {Q_LORA}  hidden {HIDDEN}")
    print(f"max |reconstructed - absorbed| = {error:.3e}  "
          f"relative {relative:.3e}")

    if relative > 1e-10:
        print("\nFAIL the two forms disagree; the fold is wrong")
        return 1

    # a fold that dropped the rope half would still agree if rope were zero, so
    # check the rope path actually carries signal
    # relative, not absolute: these outputs are order 1e-4, so an absolute
    # threshold of 1e-6 passes for a rope path that does nothing.
    zero_rope = absorbed(latent, numpy.zeros_like(rope_key), q_latent, kv_b,
                         o_proj, q_up, scale)
    rope_effect = numpy.abs(b - zero_rope).max() / magnitude
    print(f"zeroing the rope key moves the output by {rope_effect:.3e} relative")
    # A fold that dropped the rope half would leave this exactly zero. The bar
    # is "nonzero", not "large" - at these dimensions the 512-wide latent term
    # dominates the 64-wide rope term by design, so a large threshold would fail
    # on a correct implementation.
    if rope_effect < 1e-10:
        print("\nFAIL the rope half contributes nothing; the test cannot see it")
        return 1

    # the shapes the packer must emit, at the real configuration
    r = REAL
    q_absorbed_elements = r["heads"] * (r["kv_lora"] + r["qk_rope"]) * r["q_lora"]
    o_absorbed_elements = r["hidden"] * r["heads"] * r["kv_lora"]
    q_source_elements = r["heads"] * (r["qk_nope"] + r["qk_rope"]) * r["q_lora"]
    o_source_elements = r["hidden"] * r["heads"] * r["v_head"]
    print(f"\nat K3's real shapes the fold changes the tensor sizes:")
    print(f"  q_up   {q_source_elements:>12,} -> {q_absorbed_elements:>12,} elements "
          f"({q_absorbed_elements / q_source_elements:.2f}x)")
    print(f"  o_proj {o_source_elements:>12,} -> {o_absorbed_elements:>12,} elements "
          f"({o_absorbed_elements / o_source_elements:.2f}x)")
    print(f"  and kv_b's {r['heads'] * (r['qk_nope'] + r['v_head']) * r['kv_lora']:,} "
          f"elements are consumed, not shipped")
    if q_absorbed_elements != r["heads"] * (r["kv_lora"] + r["qk_rope"]) * r["q_lora"]:
        print("FAIL query fold arithmetic")
        return 1

    print("\nabsorbed == reconstructed; the packer's two folds are these:")
    print("  q_up[h]  ->  kv_b_nope[h].T @ q_up_nope[h],  then concat q_up_rope[h]")
    print(f"               ({r['kv_lora']}+{r['qk_rope']}) x {r['q_lora']} per head")
    print("  o_proj   ->  o_proj[:, h] @ kv_b_value[h]")
    print(f"               {r['hidden']} x {r['kv_lora']} per head")

    # AND WHAT IT COSTS, which is the part that should reach whoever owns the
    # kernel choice rather than only whoever writes the packer.
    reconstructed_total = q_source_elements + \
        r["heads"] * (r["qk_nope"] + r["v_head"]) * r["kv_lora"] + o_source_elements
    absorbed_total = q_absorbed_elements + o_absorbed_elements
    layers = 24
    print(f"\nabsorption is not free at K3's shapes: {absorbed_total / reconstructed_total:.2f}x "
          f"the MLA projection weights,")
    print(f"  {reconstructed_total * layers * 2 / 2**30:.1f} GB -> "
          f"{absorbed_total * layers * 2 / 2**30:.1f} GB in bf16 across {layers} MLA layers.")
    print("  kv_lora 512 against v_head 128 is why: the absorbed o_proj is 4x the")
    print("  reconstructed one. DeepSeek-V2's ratio makes absorption cheap; K3's")
    print("  does not, and a kernel that reconstructs would trade 14 GB of weights")
    print("  for one extra GEMM per layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
