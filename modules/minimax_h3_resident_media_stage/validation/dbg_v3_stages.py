#!/usr/bin/env python3
"""Dump block-0 per-stage tensors from the anchor slab path (V3 driver bisect).

Runs on sparke inside ~/val-minimax (imports h3_reference + gen_v3_v4_real from
there, reads the real fixture inputs from ~/val-minimax/fixtures/real). Writes
v3_block0_stages.npz into argv[1]; each member is a bf16 tensor stored as
uint16, named ref_sNN_<stage> so minimax_h3_npz_to_raw.py produces
ref_sNN_<stage>__RxC.u16 binaries next to the v3 fixture files.

Stage map (driver kernel # -> member):
  1 ref_s01_norm1   rms_norm(h, norm1)
  2 ref_s02_adaln   n*(1+scale_msa)+shift_msa
  3 ref_s03_q       to_q GEMM
  4 ref_s04_qnorm   per-head rms_norm(q, norm_q)
  5 ref_s05_qrope   rope(q)
  8 ref_s08_krope   rope(k)
  9 ref_s09_v       to_v GEMM
 10 ref_s10_attn    sdpa output flattened
 11 ref_s11_attnout to_out GEMM
 12 ref_s12_hmid    h + gate_msa*attn
 13 ref_s13_norm2   rms_norm(h_mid, norm2)
 14 ref_s14_mlpmod  n*(1+scale_mlp)+shift_mlp
 15 ref_s15_ffnfused ff.net.0.proj GEMM
 16 ref_s16_ffnmid  first_chunk * silu(second_chunk)
 17 ref_s17_ffnout  ff.net.2 GEMM
 18 ref_s18_out     h_mid + gate_mlp*ffn  (== fixture block0_out, asserted)
"""

import os
import sys
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

VAL_HOME = "/home/sparke/val-minimax" if os.path.isdir(
    "/home/sparke/val-minimax") else os.path.expanduser("~/val-minimax")
sys.path.insert(0, VAL_HOME)
import h3_reference as ref
import gen_v3_v4_real as gen

OUT_DIR = sys.argv[1]
FIX = os.path.join(VAL_HOME, "fixtures/real/dit_blocks01_real.npz")


def bits(tensor):
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def main():
    d = np.load(FIX)
    h = torch.from_numpy(
        (d["packed_h"].astype(np.uint32) << np.uint32(16)).view(np.float32).copy()
    ).to(torch.bfloat16)
    temb = torch.from_numpy(d["temb"].astype(np.float32))
    cos = torch.from_numpy(d["rope_cos"].astype(np.float32))
    sin = torch.from_numpy(d["rope_sin"].astype(np.float32))
    inv = torch.from_numpy(d["timestep_indices"].astype(np.int64))
    tags = torch.from_numpy(d["token_tags"].astype(np.int64))
    adaln_indices = inv * 3 + tags

    raw = gen.build_raw_index("transformer")
    gen.verify_raw_index(raw)
    p = "transformer_blocks.0."
    dumps = {}

    def dump(name, tensor):
        if not torch.isfinite(tensor.float()).all():
            raise SystemExit("SPARK_FAIL stage.nonfinite %s" % name)
        body = tensor[0]
        dumps[name] = bits(body)
        print("%-18s shape=%s max=%.6g" % (name, tuple(body.shape),
            float(tensor.float().abs().max())), flush=True)

    def s_attn(x):
        query = gen.slab_linear_full(x, raw, p + "attn.to_q.weight")
        dump("ref_s03_q", query)
        key = gen.slab_linear_full(x, raw, p + "attn.to_k.weight")
        value = gen.slab_linear_full(x, raw, p + "attn.to_v.weight")
        dump("ref_s09_v", value)
        query = query.unflatten(-1, (56, 128))
        key = key.unflatten(-1, (56, 128))
        value = value.unflatten(-1, (56, 128))
        query = ref.rms_norm(query, gen.slab_vec(raw, p + "attn.norm_q.weight"), 1e-5)
        dump("ref_s04_qnorm", query.flatten(2, 3))
        key = ref.rms_norm(key, gen.slab_vec(raw, p + "attn.norm_k.weight"), 1e-5)
        query = ref.apply_rotary_emb(query, cos, sin)
        dump("ref_s05_qrope", query.flatten(2, 3))
        key = ref.apply_rotary_emb(key, cos, sin)
        dump("ref_s08_krope", key.flatten(2, 3))
        out = ref.attention(query, key, value, use_sdpa=True)
        out = out.flatten(2, 3)
        dump("ref_s10_attn", out)
        return gen.slab_linear_full(out, raw, p + "attn.to_out.0.weight")

    t_in = F.silu(temb).to(torch.bfloat16).float()
    mods = gen.slab_linear_full(t_in, raw, p + "adaln_proj.linear.weight",
        p + "adaln_proj.linear.bias", chunk=8192).to(torch.bfloat16).view(-1, 6 * 5376).chunk(6, dim=-1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    resid = h
    n = ref.rms_norm(h, gen.slab_vec(raw, p + "norm1.weight"), 1e-5)
    dump("ref_s01_norm1", n)
    n = n * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(0, adaln_indices)
    dump("ref_s02_adaln", n)
    a = s_attn(n)
    dump("ref_s11_attnout", a)
    h = resid + gate_msa.index_select(0, adaln_indices) * a
    dump("ref_s12_hmid", h)
    resid = h
    n = ref.rms_norm(h, gen.slab_vec(raw, p + "norm2.weight"), 1e-5)
    dump("ref_s13_norm2", n)
    n = n * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(0, adaln_indices)
    dump("ref_s14_mlpmod", n)
    fused = gen.slab_linear_full(n, raw, p + "ff.net.0.proj.weight")
    dump("ref_s15_ffnfused", fused)
    fm, fg = fused.chunk(2, dim=-1)
    mid = fm * F.silu(fg)
    dump("ref_s16_ffnmid", mid)
    fout = gen.slab_linear_full(mid, raw, p + "ff.net.2.weight")
    dump("ref_s17_ffnout", fout)
    out = resid + gate_mlp.index_select(0, adaln_indices) * fout
    dump("ref_s18_out", out)

    expected = torch.from_numpy(
        (d["block0_out"].astype(np.uint32) << np.uint32(16)).view(np.float32).copy()
    )
    drift = float(((out.float() - expected).norm() / expected.norm()).item())
    print("self-check ref_s18_out vs fixture block0_out rel=%.3g" % drift, flush=True)
    if drift > 1e-6:
        raise SystemExit("SPARK_FAIL stage.selfcheck rel=%r" % drift)

    mods1 = gen.slab_linear_full(t_in, raw, "transformer_blocks.1.adaln_proj.linear.weight",
        "transformer_blocks.1.adaln_proj.linear.bias", chunk=8192).to(torch.bfloat16).view(-1, 6 * 5376).chunk(6, dim=-1)
    for i, mod_name in enumerate(("shift_msa", "scale_msa", "gate_msa",
            "shift_mlp", "scale_mlp", "gate_mlp")):
        tensor = mods1[i]
        if not torch.isfinite(tensor.float()).all():
            raise SystemExit("SPARK_FAIL stage.nonfinite ref_b1_mod_%s" % mod_name)
        dumps["ref_b1_mod_%s" % mod_name] = bits(tensor)
        print("ref_b1_mod_%-10s shape=%s max=%.6g" % (mod_name, tuple(tensor.shape),
            float(tensor.float().abs().max())), flush=True)
    path = os.path.join(OUT_DIR, "v3_block0_stages.npz")
    np.savez(path, **dumps)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    print("wrote %s sha256=%s members=%d" % (path, digest.hexdigest(), len(dumps)),
        flush=True)


if __name__ == "__main__":
    main()
