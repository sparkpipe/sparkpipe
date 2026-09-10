#!/usr/bin/env python3
"""Dump V4 video/audio per-stage tensors from the anchor paths (driver bisect).

Runs on sparke inside ~/val-minimax (imports h3_reference + gen_v3_v4_real).
Reads the committed driver fixture inputs (z / latents .f32 files) so both
sides consume identical bytes. Writes v4_stage_dumps.npz into argv[1];
minimax_h3_npz_to_raw.py turns the members into refv_*/*.f32 binaries.

Video members (slab path of gen_v3_v4_real.raw_vae_video_decode, full 7-frame
decode == the fixture's single chunk):
  refv_in33        (33,2048) patches + registers + zero cls
  refv_b0          (33,2048) hidden after block 0
  refv_b1          (33,2048) hidden after block 1
  refv_b17         (33,2048) hidden after block 17
  refv_b35         (33,2048) hidden after block 35 (pre norm_out)
  refv_layernorm   (33,2048) decoder.norm_out output
  refv_pixels      (33,3072) proj_out output (pre unpatchify)
  refv_decoded     (3,22,32,32) chunk-trimmed pixels == fixture decoded

Audio members (safe_open path of the audio fixture generator):
  refa_dec_in      (2,2048,4)
  refa_conv_pre    (2,1024,4)
  refa_ups0        (2,512,20)
  refa_s0_d1       (2,512,20) after dilation-1 chain inside resblock 0
  refa_s0_b0       (2,512,20) resblock 0 output
  refa_s0_b1       (2,512,20) resblock 1 output
  refa_s0_b2       (2,512,20) resblock 2 output
  refa_s0_mean     (2,512,20) stage 0 average
  refa_s1_b0       (2,256,96) resblock 0 output
  refa_s1_b1       (2,256,96) resblock 1 output
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
MODULE_FIX = sys.argv[2]
VIDEO_BLOCKS = int(sys.argv[3]) if len(sys.argv) > 3 else 36
WARM = "/mnt/model-warm/minimax-h3"


def main():
    dumps = {}

    def dump_full(name, tensor):
        dump(name, tensor)

    def dump(name, tensor):
        if not torch.isfinite(tensor).all():
            raise SystemExit("SPARK_FAIL v4stage.nonfinite %s" % name)
        body = tensor.detach().cpu().contiguous().numpy().astype(np.float32).copy()
        dumps[name] = body
        print("%-16s shape=%s max=%.6g" % (name, tuple(body.shape),
            float(np.abs(body).max())), flush=True)

    video_z = np.fromfile(os.path.join(MODULE_FIX,
        "v4_video/z__1x24x7x2x2.f32"), dtype=np.float32)
    z = torch.from_numpy(video_z.copy()).view(1, 24, 7, 2, 2)

    raw = gen.build_raw_index("vae")
    gen.verify_raw_index(raw)
    zq = F.conv3d(z, gen.slab_weight(raw, "post_quant_conv.weight"),
        gen.slab_vec(raw, "post_quant_conv.bias"))
    b, c, f, hgt, wid = zq.shape
    x = zq.permute(0, 2, 3, 4, 1).reshape(b, f * hgt * wid, c)
    x = gen.slab_linear_full(x, raw, "decoder.proj_in.weight", "decoder.proj_in.bias")
    num_patches = x.shape[1]
    registers = gen.slab_weight(raw, "decoder.register_tokens").expand(b, -1, -1)
    cls_token = torch.zeros_like(x[:, :1, :])
    x = torch.cat([x, registers, cls_token], dim=1)
    dump("refv_in33", x[0])
    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0
        for size in (f, hgt, wid)]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(b, -1, -1)
    suffix = position_ids.new_zeros((b, 5, 3))
    position_ids = torch.cat([position_ids, suffix], dim=1)
    cos, sin = ref.vae_video_rope(position_ids, 48, 100.0)
    dump("refv_rope_cos", cos.reshape(33, 48))
    dump("refv_rope_sin", sin.reshape(33, 48))
    for i in range(VIDEO_BLOCKS):
        p = "decoder.transformer_blocks.%d." % i
        n = ref.rms_norm(x.float(), gen.slab_vec(raw, p + "norm1.weight"), 1e-5).to(x.dtype)
        if i == 0:
            dump("refv_b0_n1", n[0])
            query = gen.slab_linear_full(n, raw, p + "attn.to_q.weight", p + "attn.to_q.bias")
            key = gen.slab_linear_full(n, raw, p + "attn.to_k.weight", p + "attn.to_k.bias")
            value = gen.slab_linear_full(n, raw, p + "attn.to_v.weight", p + "attn.to_v.bias")
            query = query.unflatten(2, (32, 64))
            key = key.unflatten(2, (32, 64))
            value = value.unflatten(2, (32, 64))
            query = ref.rms_norm(query.float(), None, 1e-5).to(query.dtype)
            key = ref.rms_norm(key.float(), None, 1e-5).to(key.dtype)
            c = cos.to(query.dtype)
            s_ = sin.to(query.dtype)
            rd = c.shape[-1]
            qr, qp = query[..., :rd], query[..., rd:]
            kr, kp = key[..., :rd], key[..., rd:]
            q1, q2 = qr.chunk(2, dim=-1)
            k1, k2 = kr.chunk(2, dim=-1)
            query = torch.cat([qr * c + torch.cat([-q2, q1], dim=-1) * s_, qp], dim=-1)
            key = torch.cat([kr * c + torch.cat([-k2, k1], dim=-1) * s_, kp], dim=-1)
            dump("refv_b0_qr", query.flatten(2, 3)[0])
            dump("refv_b0_kr", key.flatten(2, 3)[0])
            dump("refv_b0_v", value.flatten(2, 3)[0])
            out = ref.attention(query, key, value, use_sdpa=True)
            out = out.flatten(2, 3)
            a = gen.slab_linear_full(out, raw, p + "attn.to_out.0.weight", p + "attn.to_out.0.bias")
            dump("refv_b0_ao", a[0])
        else:
            a = gen.raw_s_vae_attn(n, raw, p, cos, sin)
        x = x + a * gen.slab_vec(raw, p + "scale1")
        if i == 0:
            dump("refv_b0_attn", x[0])
        n = ref.rms_norm(x.float(), gen.slab_vec(raw, p + "norm2.weight"), 1e-5).to(x.dtype)
        ffn = gen.raw_s_ff(n, raw, p)
        x = x + ffn * gen.slab_vec(raw, p + "scale2")
        if i in (0, 1, 17, 35):
            dump("refv_b%d" % i, x[0])
    if VIDEO_BLOCKS >= 36:
        x = F.layer_norm(x, (x.shape[-1],), gen.slab_vec(raw, "decoder.norm_out.weight"),
            gen.slab_vec(raw, "decoder.norm_out.bias"), 1e-5)
        dump("refv_layernorm", x[0])
        x = gen.slab_linear_full(x, raw, "decoder.proj_out.weight", "decoder.proj_out.bias")
        dump("refv_pixels", x[0])
        x = x[:, :num_patches, :]
        ps, pst = 16, 4
        x = x.view(b, f, hgt, wid, 3, pst, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        full = x.reshape(b, 3, f * pst, hgt * ps, wid * ps)
        trim = torch.cat([full[:, :, 3:20], full[:, :, 23:28]], dim=2)
        dump("refv_decoded", trim[0])

    audio_latents = np.fromfile(os.path.join(MODULE_FIX,
        "v4_audio/latents__2x32x4.f32"), dtype=np.float32)
    latents = torch.from_numpy(audio_latents.copy()).view(2, 32, 4)
    shards = gen.build_index("audio_vae")
    get = gen.make_reader(shards)
    w = {}
    for name in shards:
        if name.startswith("decoder.") or name.startswith("dec_in_proj."):
            w[name] = get(name)
    w.update(dict(_num_upsamples=7, _num_kernels=3,
        _upsample_rates=(5, 5, 2, 2, 2, 2, 2),
        _upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        _resblock_kernel_sizes=(3, 7, 11),
        _resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))))

    def act1(t, prefix, mid_name=None):
        t = ref.audio_upsample1d(t, 2, 12, w["%s.upsample.filter" % prefix])
        t = ref.audio_snake_beta(t, w["%s.act.alpha" % prefix],
            w["%s.act.beta" % prefix])
        if mid_name is not None:
            dump_full(mid_name, t)
        return ref.audio_lowpass(t, w["%s.downsample.lowpass.filter" % prefix], 2, 12)

    def amp_block(t, w, prefix, kernel_size, dilation, marks, act_marks=(), mid_marks=()):
        for idx, dil in enumerate(dilation):
            a1 = act1(t, "%s.activations.%d" % (prefix, 2 * idx),
                "refa_s0_m%d" % (2 * idx) if (2 * idx) in mid_marks else None)
            if (2 * idx) in act_marks:
                dump_full("refa_s0_a%d" % (2 * idx), a1)
            r = F.conv1d(a1, ref.wn_weight(w["%s.convs1.%d.weight_g" % (prefix, idx)],
                w["%s.convs1.%d.weight_v" % (prefix, idx)]),
                w["%s.convs1.%d.bias" % (prefix, idx)], dilation=dil,
                padding=(kernel_size * dil - dil) // 2)
            a2 = act1(r, "%s.activations.%d" % (prefix, 2 * idx + 1))
            r = F.conv1d(a2, ref.wn_weight(w["%s.convs2.%d.weight_g" % (prefix, idx)],
                w["%s.convs2.%d.weight_v" % (prefix, idx)]),
                w["%s.convs2.%d.bias" % (prefix, idx)], dilation=1,
                padding=(kernel_size - 1) // 2)
            t = r + t
            if idx in marks:
                dump_full("refa_s0_d%d" % (idx + 1), t)
        return t

    h = F.conv1d(latents.to(w["dec_in_proj.weight"].dtype), w["dec_in_proj.weight"],
        w["dec_in_proj.bias"])
    dump_full("refa_dec_in", h)
    h = F.conv1d(h, ref.wn_weight(w["decoder.conv_pre.weight_g"],
        w["decoder.conv_pre.weight_v"]), w["decoder.conv_pre.bias"], padding=3)
    dump_full("refa_conv_pre", h)
    for i in range(7):
        rate = w["_upsample_rates"][i]
        kernel = w["_upsample_kernel_sizes"][i]
        h = F.conv_transpose1d(h, ref.wn_weight(w["decoder.ups.%d.0.weight_g" % i],
            w["decoder.ups.%d.0.weight_v" % i]), w["decoder.ups.%d.0.bias" % i],
            stride=rate, padding=(kernel - rate) // 2)
        if i == 0:
            dump_full("refa_ups0", h)
        residual = None
        for j in range(3):
            prefix = "decoder.resblocks.%d" % (i * 3 + j)
            r = amp_block(h, w, prefix, w["_resblock_kernel_sizes"][j],
                w["_resblock_dilation_sizes"][j],
                (0, 1, 2) if i == 0 and j == 0 else (),
                (0, 1) if i == 0 and j == 0 else (),
                (0,) if i == 0 and j == 0 else ())
            if i == 0 and j in (0, 1, 2):
                dump_full("refa_s0_b%d" % j, r)
            if i == 1 and j in (0, 1, 2):
                dump_full("refa_s1_b%d" % j, r)
            residual = r if residual is None else residual + r
        h = residual / 3
        if i == 0:
            dump_full("refa_s0_mean", h)

    path = os.path.join(OUT_DIR, "v4_stage_dumps.npz")
    np.savez(path, **dumps)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    print("wrote %s sha256=%s members=%d" % (path, digest.hexdigest(), len(dumps)),
        flush=True)


if __name__ == "__main__":
    main()
