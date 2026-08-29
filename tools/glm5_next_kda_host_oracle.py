#!/usr/bin/env python3
# glm5_next KDA host oracle — glm5-attractor lane.
#
# Recomputes the layer-0 KDA cell on the host from the CHECKPOINT's semantics
# (the fla/Kimi-Delta-Attention reference math + the checkpoint tensors) and
# compares every stage against the G5N-VEC full-vector dumps the module's diag
# build printed for the same request. The oracle shares no math with the
# module: its formulas come from the reference (g = lower_bound *
# sigmoid(exp(A_log) * (f + dt_bias)), per-(head, channel), decay-before-
# predict delta rule, q/k L2-normalized per head, RMSNorm-then-sigmoid-gate
# output) — not from linear_attn.cuh / layer.cuh.
#
# Stage ladder, per pass (waves are single-row: pass p IS position p):
#   normed (INPUT, dumped) -> qkv_beta fused -> split -> conv(+swish)
#   -> [decay latent -> decay logit -> retention | beta sigmoid]
#   -> delta rule (state tracked across passes from zero) -> o_norm+gate
#   -> o_proj partial; head: rmsnorm(hc_mean) @ lm_head -> argmax.
#
# usage: glm5_next_kda_host_oracle.py <residentd.log> <prompt_ids.json> [layer]
# Emit the dumps with: SPARK_GLM5_NEXT_PROBE=1 SPARK_GLM5_NEXT_PROBE_VEC=1
# (wave: tools/glm5_next_wave.sh --probe-vec full).
import json
import os
import re
import struct
import sys

import numpy as np

CKPT = os.environ.get("G5N_CKPT", "/mnt/model-warm/glm-5.3-flash")
LAYER = 0
TP = 16
RANK = 0
HIDDEN = 4096
KDA_HEADS = 64
KD = 128
VD = 128
CONV = 4
LOWRANK = 128
GATE_LB = -5.0
RMS_EPS = 1e-5
QK_L2_EPS_DELTA = 1e-6  # the delta rule's in-kernel l2norm epsilon (reference)

RANK_HEADS = KDA_HEADS // TP
RANK_QK = RANK_HEADS * KD
RANK_V = RANK_HEADS * VD

PREFIX = "model.language_model.layers."


# ---------------------------------------------------------------- safetensors
class Safetensors:
    def __init__(self, root):
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
        self.map = self.index["weight_map"]
        self.headers = {}
        self.fds = {}

    def _header(self, fname):
        if fname not in self.headers:
            fh = open(os.path.join(self.root, fname), "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            self.headers[fname] = (hdr, base)
            self.fds[fname] = fh
        return self.headers[fname]

    def raw(self, name):
        fname = self.map[name]
        hdr, base = self._header(fname)
        e = hdr[name]
        dt, shape, offs = e["dtype"], e["shape"], e["data_offsets"]
        fh = self.fds[fname]
        fh.seek(base + offs[0])
        return np.frombuffer(fh.read(offs[1] - offs[0]), dtype=self._np(dt)).reshape(shape)

    @staticmethod
    def _np(dt):
        return {"BF16": np.uint16, "F32": np.float32, "F16": np.float16,
                "U8": np.uint8}[dt]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_u16(x):
    """round-to-nearest-even bf16 of an f32 array -> uint16 patterns."""
    u = x.astype(np.float32).view(np.uint32).copy()
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def bf16_round_f32(x):
    return bf16_to_f32(f32_to_bf16_u16(x))


# ---------------------------------------------------------------- log parsing
def parse_log(path, max_pass=4096):
    """-> passes{p: {label: np.uint16/uint32 array}}, head_rows{p: (score,token)}"""
    vec_re = re.compile(r"G5N-VEC L(\d+) P(\d+) (\S+) (\d+)((?: [0-9a-f]{4,8})+)")
    head_re = re.compile(r"G5N-PROBE head score (\S+) token (\d+)")
    passes, head_rows = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = vec_re.search(line)
            if m:
                layer, p, label, count = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
                toks = m.group(5).split()
                if len(toks) != count or p > max_pass:
                    continue
                width = len(toks[0])
                arr = np.array([int(t, 16) for t in toks], dtype=np.uint32)
                if width == 4:
                    arr = arr.astype(np.uint16)
                passes.setdefault(p, {})[label] = (layer, arr)
                continue
            m = head_re.search(line)
            if m:
                head_rows[len(head_rows) + 1] = (float(m.group(1)), int(m.group(2)))
    return passes, head_rows


# ---------------------------------------------------------------- comparisons
class Report:
    def __init__(self):
        self.rows = []

    def add(self, stage, pass_, mine, dump_u, kind):
        """kind: 'bf16' (compare u16 patterns) or 'f32' (uint32 patterns)."""
        if kind == "bf16":
            a, b = mine.astype(np.int32), dump_u.astype(np.int32)
            diff = np.abs(a - b)
            ulp = diff
            fa = bf16_to_f32(mine)
            fb = bf16_to_f32(dump_u)
        else:
            fa, fb = mine.view(np.float32), dump_u.view(np.float32)
            ia, ib = mine.view(np.uint32), dump_u.view(np.uint32)
            ulp = np.abs(ia.astype(np.int64) - ib.astype(np.int64))
        denom = np.maximum(np.abs(fa), np.abs(fb))
        rel = np.where(denom > 0, np.abs(fa - fb) / denom, np.abs(fa - fb))
        exact = float(np.mean(ulp == 0))
        mx_ulp = int(ulp.max())
        mx_rel = float(rel.max())
        ok = exact > 0.90 and (mx_ulp <= 2 if kind == "bf16" else mx_rel < 1e-3)
        self.rows.append((stage, pass_, exact, mx_ulp, mx_rel, ok))

    def show(self):
        print(f"{'stage':<16} {'pass':>4} {'exact%':>7} {'max_ulp':>8} {'max_rel':>10}  verdict")
        bad = 0
        for stage, p, exact, mx_ulp, mx_rel, ok in self.rows:
            print(f"{stage:<16} {p:>4} {exact*100:>6.2f}% {mx_ulp:>8} {mx_rel:>10.3e}  {'ok' if ok else 'DIVERGES'}")
            bad += 0 if ok else 1
        print(f"\n{len(self.rows)} comparisons, {bad} DIVERGING")
        # first diverging stage in pass order
        for stage, p, *_rest, ok in self.rows:
            if not ok:
                print(f"FIRST DIVERGENCE: {stage} @ pass {p}")
                break
        return bad


def conv(raw_f32, cw, win):
    """short causal conv + swish; win carries the last CONV-1 raw taps
    (uint16, oldest first) across passes."""
    u = f32_to_bf16_u16(raw_f32).reshape(-1, 1)
    t = np.concatenate([win[:, 1:], u], axis=1)
    acc = (bf16_to_f32(t) * cw).sum(axis=1)
    sw = acc * (1.0 / (1.0 + np.exp(-acc)))
    return bf16_round_f32(sw), t


def l2_per_head(x_f32, heads=RANK_HEADS, dim=KD, eps=RMS_EPS):
    m = x_f32.reshape(heads, dim)
    inv = 1.0 / np.sqrt((m * m).sum(axis=1, keepdims=True) + eps)
    return (m * inv).reshape(-1)


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def _top(*names):
    """checkpoint tops vary across publishes: use the first name present."""
    import json as _json
    idx = _json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    for n in names:
        if n in idx["weight_map"]:
            return n
    raise KeyError(names)


def main():
    log = sys.argv[1]
    prompt_ids = json.load(open(sys.argv[2]))
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["prompt_token_ids"]
    prompt_len = len(prompt_ids)
    st = Safetensors(CKPT)
    P = PREFIX + f"{LAYER}.self_attn."

    # --- checkpoint weights, rank-0 shards (the audited byte-clean contract)
    q_w = bf16_to_f32(st.raw(P + "q_proj.weight")[RANK * RANK_QK:(RANK + 1) * RANK_QK])
    k_w = bf16_to_f32(st.raw(P + "k_proj.weight")[RANK * RANK_QK:(RANK + 1) * RANK_QK])
    v_w = bf16_to_f32(st.raw(P + "v_proj.weight")[RANK * RANK_V:(RANK + 1) * RANK_V])
    b_w = bf16_to_f32(st.raw(P + "b_proj.weight")[RANK * RANK_HEADS:(RANK + 1) * RANK_HEADS])
    fa_w = bf16_to_f32(st.raw(P + "f_a_proj.weight"))
    fb_w = bf16_to_f32(st.raw(P + "f_b_proj.weight")[RANK * RANK_QK:(RANK + 1) * RANK_QK])
    ga_w = bf16_to_f32(st.raw(P + "g_a_proj.weight"))
    gb_w = bf16_to_f32(st.raw(P + "g_b_proj.weight")[RANK * RANK_V:(RANK + 1) * RANK_V])
    qc_w = bf16_to_f32(st.raw(P + "q_conv1d.weight").reshape(-1)[RANK * RANK_QK * CONV:(RANK + 1) * RANK_QK * CONV])
    kc_w = bf16_to_f32(st.raw(P + "k_conv1d.weight").reshape(-1)[RANK * RANK_QK * CONV:(RANK + 1) * RANK_QK * CONV])
    vc_w = bf16_to_f32(st.raw(P + "v_conv1d.weight").reshape(-1)[RANK * RANK_V * CONV:(RANK + 1) * RANK_V * CONV])
    a_log = st.raw(P + "A_log")[RANK * RANK_HEADS:(RANK + 1) * RANK_HEADS].astype(np.float32)
    dt_bias = st.raw(P + "dt_bias")[RANK * RANK_QK:(RANK + 1) * RANK_QK].astype(np.float32)
    o_norm_w = bf16_to_f32(st.raw(P + "o_norm.weight").reshape(-1)).astype(np.float32)
    o_w = bf16_to_f32(st.raw(P + "o_proj.weight")[:, RANK * RANK_V:(RANK + 1) * RANK_V])  # [4096, 512]
    attn_norm_w = bf16_to_f32(st.raw(PREFIX + f"{LAYER}.input_layernorm.weight"))
    final_norm_w = bf16_to_f32(st.raw(_top("model.language_model.norm.weight", "norm.weight")))
    print(f"loaded checkpoint shards for layer {LAYER} rank {RANK}/{TP}")

    passes, head_rows = parse_log(log)
    got = sorted(p for p in passes if passes[p].get("normed"))
    if not got:
        sys.exit("no G5N-VEC normed dumps found in the log")
    print(f"log has passes {got[0]}..{got[-1]}, prompt_len={prompt_len}, head rows={len(head_rows)}")

    rep = Report()
    # oracle-owned recurrent state (KDA heads x KD x VD, key-major) + conv windows
    state = np.zeros((RANK_HEADS, KD, VD), dtype=np.float32)
    win_q = np.zeros((RANK_QK, CONV), dtype=np.uint16)
    win_k = np.zeros((RANK_QK, CONV), dtype=np.uint16)
    win_v = np.zeros((RANK_V, CONV), dtype=np.uint16)
    attn_norm_kind = None

    for p in got:
        d = passes[p]
        collapsed = d["collapsed"][1]
        normed = d["normed"][1]
        fused = d["fused_qkvb"][1]
        decay_latent = d["decay_latent"][1]
        q_pc, k_pc, v_pc = d["q_postconv"][1], d["k_postconv"][1], d["v_postconv"][1]
        decay_logit = d["decay_logit"][1]
        retention = d["retention"][1]
        write_gate = d["write_gate"][1]
        delta_out = d["delta_out"][1]
        delta_gated = d["delta_gated"][1]
        kda_gate = d["kda_gate"][1]
        out_partial = d["out_partial"][1]

        x = bf16_to_f32(normed).astype(np.float32)

        # 0) the attn-norm mapping: rmsnorm(collapsed*w) == normed?
        cx = bf16_to_f32(collapsed).astype(np.float32)
        if attn_norm_kind is None:
            for nm, w in (("input_layernorm", attn_norm_w),):
                if np.array_equal(f32_to_bf16_u16(rmsnorm(cx, w, RMS_EPS)), normed):
                    attn_norm_kind = nm
        if attn_norm_kind:
            rep.add("attn_norm", p, f32_to_bf16_u16(rmsnorm(cx, attn_norm_w, RMS_EPS)), normed, "bf16")

        # 1) fused q|k|v|beta projection: MY GEMM vs the dump (bf16 patterns)
        q_raw = bf16_round_f32(x @ q_w.T)
        k_raw = bf16_round_f32(x @ k_w.T)
        v_raw = bf16_round_f32(x @ v_w.T)
        b_raw = bf16_round_f32(x @ b_w.T)
        rep.add("fused_qkvb", p, np.concatenate([f32_to_bf16_u16(q_raw), f32_to_bf16_u16(k_raw),
                                                 f32_to_bf16_u16(v_raw), f32_to_bf16_u16(b_raw)]), fused, "bf16")

        # from here on every stage consumes the DUMPED buffer, so a divergence
        # convicts that stage's kernel (not an upstream accumulation):
        q_raw = bf16_to_f32(fused[0:RANK_QK]).astype(np.float32)
        k_raw = bf16_to_f32(fused[RANK_QK:2 * RANK_QK]).astype(np.float32)
        v_raw = bf16_to_f32(fused[2 * RANK_QK:2 * RANK_QK + RANK_V]).astype(np.float32)
        b_raw = bf16_to_f32(fused[2 * RANK_QK + RANK_V:]).astype(np.float32)

        # 2) short conv + swish (window tracked across passes), then the layer's
        # per-head L2 norm for q/k (eps 1e-5) - the dump is post-L2 for q/k
        q_conv, win_q = conv(q_raw, qc_w.reshape(RANK_QK, CONV), win_q)
        k_conv, win_k = conv(k_raw, kc_w.reshape(RANK_QK, CONV), win_k)
        v_conv, win_v = conv(v_raw, vc_w.reshape(RANK_V, CONV), win_v)
        rep.add("v_postconv", p, f32_to_bf16_u16(v_conv), v_pc, "bf16")
        rep.add("q_postconv", p, f32_to_bf16_u16(l2_per_head(q_conv)), q_pc, "bf16")
        rep.add("k_postconv", p, f32_to_bf16_u16(l2_per_head(k_conv)), k_pc, "bf16")

        # 3) decay chain: latent (dump) -> logit -> bounded retention
        lat = bf16_to_f32(decay_latent).astype(np.float32)
        dl = bf16_round_f32(lat @ fb_w.T)
        rep.add("decay_logit", p, f32_to_bf16_u16(dl), decay_logit, "bf16")
        lg = bf16_to_f32(decay_logit).astype(np.float32).reshape(RANK_HEADS, KD)
        dtb = dt_bias.reshape(RANK_HEADS, KD)
        scaled = np.exp(a_log).reshape(RANK_HEADS, 1) * (lg + dtb)
        sig = 1.0 / (1.0 + np.exp(-scaled))
        ret_mine = np.exp(GATE_LB * sig).reshape(-1)
        rep.add("retention", p, ret_mine.view(np.uint32), retention, "f32")

        # 4) write gate: sigmoid of the dumped beta section
        beta = (1.0 / (1.0 + np.exp(-b_raw))).astype(np.float32)
        rep.add("write_gate", p, beta.view(np.uint32), write_gate, "f32")

        # 5) delta rule per head, decay-before-predict, post-update read.
        # consumes the DUMPED post-L2 q/k/v, dumped retention and beta: any
        # divergence here is the recurrence/state path itself.
        ret = retention.view(np.float32).astype(np.float32)
        k2 = bf16_to_f32(k_pc).astype(np.float32).reshape(RANK_HEADS, KD)
        qv = bf16_to_f32(q_pc).astype(np.float32).reshape(RANK_HEADS, KD)
        vv = bf16_to_f32(v_pc).astype(np.float32).reshape(RANK_HEADS, VD)
        k2 = k2 / np.sqrt((k2 * k2).sum(axis=1, keepdims=True) + QK_L2_EPS_DELTA)
        q2 = qv / np.sqrt((qv * qv).sum(axis=1, keepdims=True) + QK_L2_EPS_DELTA)
        a2 = ret.reshape(RANK_HEADS, KD)
        b2 = beta.reshape(RANK_HEADS)
        o_all = np.empty((RANK_HEADS, VD), dtype=np.float32)
        for h in range(RANK_HEADS):
            pred = (state[h] * (k2[h] * a2[h])[:, None]).sum(axis=0)
            state[h] = a2[h][:, None] * state[h] + b2[h] * (vv[h] - pred)[None, :] * k2[h][:, None]
            o_all[h] = (state[h] * q2[h][:, None]).sum(axis=0)
        rep.add("delta_out", p, f32_to_bf16_u16(o_all.reshape(-1)), delta_out, "bf16")

        # 6) output norm (per head, RMS) then sigmoid gate, from dumps
        gs = (1.0 / (1.0 + np.exp(-bf16_to_f32(kda_gate)))).astype(np.float32)
        o32 = bf16_to_f32(delta_out).astype(np.float32).reshape(RANK_HEADS, VD)
        rms = np.sqrt((o32 * o32).sum(axis=1) / VD + RMS_EPS)
        normed_o = (o32 / rms[:, None] * o_norm_w[None, :])
        gated = (normed_o * gs.reshape(RANK_HEADS, VD)).reshape(-1)
        rep.add("delta_gated", p, f32_to_bf16_u16(gated), delta_gated, "bf16")

        # 7) o_proj rank partial, from the dumped gated rows
        part = bf16_round_f32(bf16_to_f32(delta_gated).astype(np.float32) @ o_w.T)
        rep.add("out_partial", p, f32_to_bf16_u16(part), out_partial, "bf16")

        # 8) the oracle's state vs the dumped state (f32 words)
        for h in range(RANK_HEADS):
            lab = f"state_h{h}"
            if lab in d:
                rep.add(lab, p, state[h].reshape(-1).view(np.uint32), d[lab][1], "f32")

    # 9) head arbitration: rmsnorm(head_mean) @ lm_head -> argmax (a few passes:
    # the lm_head read is 1.2 GB; every pass would dominate the runtime)
    hf = bf16_to_f32(final_norm_w).astype(np.float32)
    lm = st.raw(_top("model.language_model.lm_head.weight", "lm_head.weight"))
    want = sorted(set(got[:2] + [p for p in (prompt_len, prompt_len + 1) if p in head_rows] + got[-2:]))
    head_n, head_miss = 0, 0
    for p in want:
        hm = passes[p].get("head_mean")
        if hm is None or p not in head_rows:
            continue
        mean = bf16_to_f32(hm[1]).astype(np.float32)
        h = rmsnorm(mean, hf, RMS_EPS)
        best, best_i = -1e30, -1
        for c0 in range(0, lm.shape[0], 4096):
            chunk = bf16_to_f32(lm[c0:c0 + 4096]).astype(np.float32)
            sc = chunk @ h
            i = int(np.argmax(sc))
            if float(sc[i]) > best:
                best, best_i = float(sc[i]), c0 + i
        tok = head_rows[p][1]
        ok = (best_i == tok)
        head_n += 1
        head_miss += 0 if ok else 1
        print(f"head pass {p}: oracle top1 {best_i} (score {best:.4f}) vs emitted {tok} "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"head: {head_n} passes checked, {head_miss} argmax mismatches")

    bad = rep.show()
    sys.exit(1 if (bad or head_miss) else 0)


def decay_logit_latent(decay_latent_u16, fb_w):
    """decay latent (dumped bf16) @ f_b rank rows^T -> f32 [512]."""
    lat = bf16_to_f32(decay_latent_u16).astype(np.float32)
    return lat @ fb_w.T


if __name__ == "__main__":
    main()
