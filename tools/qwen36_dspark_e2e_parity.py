#!/usr/bin/env python3
"""W2 end-to-end parity: the module's DSpark block forward vs the numpy reference.

Reads the module's dumped inputs (taps + c0 + base_position, if present) and its
dumped B x vocab base logits, runs the reference forward (qwen36_dspark_reference)
on the SAME inputs, and compares the 7 mask-position logits (hidden[1:] @ lm_head)
bit-exact in BF16. This is the W2 forward gate: conv + attention + MLP + final
norm + shared lm_head, BEFORE the candidate-selector tail (W4/W3/W7).

Usage: python3 qwen36_dspark_e2e_parity.py [taps.bin] [c0.bin] [base.bin]
Defaults to the module's /tmp dump paths.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen36_dspark_reference as ref


def read_taps(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size == ref.TAPS * ref.HIDDEN, (raw.size, "taps")
    return ref.bf16_to_f32(raw).reshape(ref.TAPS, ref.HIDDEN)


def read_c0(path: str) -> int:
    return struct.unpack("<I", open(path, "rb").read(4))[0]


def read_base_logits(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size % ref.VOCAB == 0, (raw.size, "base logits")
    b = raw.size // ref.VOCAB
    return ref.bf16_to_f32(raw).reshape(b, ref.VOCAB)


def forward(taps: np.ndarray, c0: int, base_pos: float) -> np.ndarray:
    """Replicate qwen36_dspark_reference.main()'s forward; return full B x vocab logits (f32, BF16-truncated)."""
    drafter = ref.load_drafter()
    lm_head, embed_tokens = ref.load_target_shared()

    positions_q = np.arange(base_pos, base_pos + ref.BLOCK, dtype=np.float32)
    pos_ctx = base_pos - 1.0

    ctx = ref.bf16(drafter["fc.weight"] @ taps.reshape(-1))
    ctx = ref.bf16(ref.rms_norm(ctx, drafter["hidden_norm.weight"]))

    block = np.empty((ref.BLOCK, ref.HIDDEN), dtype=np.float32)
    block[0] = embed_tokens[c0]
    block[1:] = embed_tokens[ref.MASK_TOKEN_ID]

    x = block
    for L in range(ref.N_LAYERS):
        lw = {k.split(f"layers.{L}.")[1]: drafter[k] for k in drafter if f"layers.{L}." in k}
        x = ref.forward_layer(x, ctx, lw, positions_q, pos_ctx)

    hidden = ref.bf16(ref.rms_norm(x, drafter["norm.weight"]))
    logits = (hidden @ lm_head.T).astype(np.float32)
    return ref.bf16_to_f32(ref.f32_to_bf16(logits)), hidden


def main() -> int:
    taps_p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dspark_taps.bin"
    c0_p = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dspark_c0.bin"
    base_p = sys.argv[3] if len(sys.argv) > 3 else "/tmp/dspark_base.bin"
    base_pos = float(os.environ.get("SPARK_QWEN36_BASE_POS", ref.BASE_POS))
    # prefer the module's dumped base_position (RoPE must match) when present
    bp_p = "/tmp/dspark_basepos.bin"
    if os.path.exists(bp_p):
        base_pos = float(struct.unpack("<Q", open(bp_p, "rb").read(8))[0])

    taps = read_taps(taps_p)
    c0 = read_c0(c0_p)
    print(f"taps shape={taps.shape} c0={c0} base_pos={base_pos}")

    ref_logits, hidden = forward(taps, c0, base_pos)
    print(f"ref logits {ref_logits.shape}; hidden[0][:4]={hidden[0][:4].tolist()}")
    np.save("/tmp/dspark_ref_logits.npy", ref_logits.astype(np.float32))

    if os.path.exists(base_p):
        mod_logits = read_base_logits(base_p)
        print(f"module base logits {mod_logits.shape}")
        assert mod_logits.shape[1] == ref.VOCAB, (mod_logits.shape, ref.VOCAB)
        # module dumps B rows (anchor + 7 mask); compare the 7 mask rows (index 1..)
        b = mod_logits.shape[0]
        mask_rows = min(b - 1, ref.BLOCK - 1)
        a = mod_logits[1:1 + mask_rows]
        r = ref_logits[1:1 + mask_rows]
        # BF16 bit-exact on the mask rows
        np.testing.assert_array_equal(
            a.view(np.uint32), r.view(np.uint32),
            err_msg="mask-logit BF16 mismatch (module vs reference)",
        )
        d = np.abs(a - r).max()
        print(f"E2E_PARITY PASS ({mask_rows} mask rows bit-exact BF16; f32 max|diff|={d:.3e})")
    else:
        print(f"module base logits not found at {base_p}; wrote reference logits only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
