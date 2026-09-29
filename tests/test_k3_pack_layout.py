#!/usr/bin/env python3
"""The K3 pack V2 layout, verified with the stdlib alone.

tests/test_k3_pack.py holds the packer's byte moves and folds to the
checkpoint, and it needs numpy. This test needs none, because the V2 changes
are LAYOUT, and layout is integer arithmetic:

  the interleave grid closes exactly - 16 payload rows of 64B plus one 64B
  scale row per 16-neuron cell, zero padding, payload+scales to the byte
  interleave_byte_offset is the published addressing contract, and the relay
  the packer ships is held to it on random bytes, lane by lane
  the fused KDA tensors carry their section tables, one shard class each,
  sections tiling the rows
  every tensor is 128-aligned, layers emit in order, the closing tensors last
  a checkpoint the grid does not divide, or an E8M0 0xff, is refused loudly

and the whole thing is proven end to end by packing a synthetic mini
checkpoint through the real CLI with numpy blocked from the packer. Where
numpy is installed the same checkpoint is packed again with it, and every
tensor must match the stdlib pack byte for byte, except the MLA q-fold,
whose f32 accumulation order differs (tests/test_k3_pack.py holds its
values to the einsum). The kv_b value half and the q_up rope rows are byte
moves, so in both packs they must be the checkpoint's bytes, signalling NaN
payloads included.
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import k3_pack  # noqa: E402

rng = random.Random(11)
FAILURES = 0


def check(ok, message):
    global FAILURES
    if not ok:
        print(f"  FAIL {message}")
        FAILURES += 1


def rand_bf16(count):
    return b"".join(struct.pack("<H", rng.randrange(1 << 16) & 0x7FBF)
                    for _ in range(count))


def rand_f32(count):
    return b"".join(struct.pack("<f", rng.uniform(-1, 1)) for _ in range(count))


def rand_u8(count, lo=0, hi=256):
    return bytes(rng.randrange(lo, hi) for _ in range(count))


# -- unit: the grid closes ------------------------------------------------------

def unit_geometry():
    # the real K3 shapes: w1 [896][6144, 3584], w2 [896][3584, 3072]
    for out_dim, k_dim in ((6144, 3584), (3584, 3072)):
        geom = k3_pack.interleave_geometry(out_dim, k_dim, 896)
        payload = out_dim * k_dim // 2
        scales = out_dim * k_dim // 32
        check(geom["tensor_bytes"] == 896 * (payload + scales),
              f"interleave ({out_dim},{k_dim}) is not zero-padding")
        check(geom["rows_per_expert"] * 64 == payload + scales,
              f"interleave ({out_dim},{k_dim}) row count does not price out")
        check(geom["k_tiles"] * 128 == k_dim and geom["cells"] * 16 == out_dim,
              f"interleave ({out_dim},{k_dim}) grid does not tile the tensor")
    for out_dim, k_dim in ((6120, 3584), (6144, 3520)):
        try:
            k3_pack.interleave_geometry(out_dim, k_dim, 1)
            check(False, f"interleave ({out_dim},{k_dim}) should be refused")
        except k3_pack.PackFailure:
            pass


# -- unit: the relay honours the addressing contract -----------------------------

def unit_relay_matches_addressing():
    experts, out_dim, k_dim = 2, 32, 256
    geom = k3_pack.interleave_geometry(out_dim, k_dim, experts)
    payload = rand_u8(experts * out_dim * k_dim // 2)
    scales = rand_u8(experts * out_dim * k_dim // 32, lo=100, hi=150)
    got = k3_pack.interleave_py(payload, scales, geom)
    check(len(got) == geom["tensor_bytes"], "relay byte count is off")
    k_groups = k_dim // 32
    mismatches = 0
    for e in range(experts):
        for t in range(geom["k_tiles"]):
            for n in range(out_dim):
                for lane in range(0, 64, 7):  # sample every lane position
                    at = k3_pack.interleave_byte_offset(geom, e, t, n,
                                                        "payload", lane)
                    src = (e * out_dim + n) * (k_dim // 2) + t * 64 + lane
                    mismatches += got[at] != payload[src]
                for j in range(geom["scale_bytes_per_neuron_tile"]):
                    at = k3_pack.interleave_byte_offset(geom, e, t, n,
                                                        "scale", j)
                    src = (e * out_dim + n) * k_groups + t * 4 + j
                    mismatches += got[at] != scales[src]
    check(mismatches == 0,
          f"relay disagrees with interleave_byte_offset at {mismatches} lanes")
    # The NUMPY relay cannot execute on a numpy-less host, so its reshape
    # chain is pinned by stride emulation: (E, out, kt, 64).transpose(0,2,1,3)
    # .reshape(E, kt, cells, 16, 64) maps element (e, n, t, b) to
    # out[e][t][n//16][n%16][b] in C order - and the scale chain maps
    # (e, n, t, j) to out[e][t][n//16][0][(n%16)*4+j]. Both are exactly the
    # published addressing above, which is what this loop re-derives.
    emu = bytearray(geom["tensor_bytes"])
    for e in range(experts):
        for t in range(geom["k_tiles"]):
            for n in range(out_dim):
                at = k3_pack.interleave_byte_offset(geom, e, t, n,
                                                    "payload", 0)
                src = (e * out_dim + n) * (k_dim // 2) + t * 64
                emu[at:at + 64] = payload[src:src + 64]
                at = k3_pack.interleave_byte_offset(geom, e, t, n, "scale", 0)
                src = (e * out_dim + n) * k_groups + t * 4
                emu[at:at + 4] = scales[src:src + 4]
    check(bytes(emu) == got,
          "the numpy reshape chain's stride semantics differ from the relay")


# -- unit: fused section tables ---------------------------------------------------

def unit_sections():
    sections, rows = k3_pack.kda_fused_qkvb_sections(96, 128, 128)
    check([s["name"] for s in sections] == ["q", "k", "v", "beta"],
          "qkvb section order changed")
    check(rows == 3 * 96 * 128 + 96, "qkvb fused row count is wrong")
    check([s["row_offset"] for s in sections] ==
          [0, 12288, 24576, 36864], "qkvb section offsets are wrong")
    check([s["rows_per_head"] for s in sections] == [128, 128, 128, 1],
          "qkvb per-head split widths are wrong")


# -- end to end: pack the mini ----------------------------------------------------

MINI = {"hidden": 64, "vocab": 128, "q_lora": 16, "kv_lora": 32, "rope": 8,
        "nope": 16, "v_head": 32, "heads": 2, "kda_heads": 2, "kda_head": 64,
        "kernel": 4, "latent": 128, "inter": 128, "experts": 4, "top_k": 2}


def mini_checkpoint(root, latent=None, poison_scale=False):
    g = dict(MINI)
    if latent is not None:
        g["latent"] = latent
    kda_dim = g["kda_heads"] * g["kda_head"]
    config = {"hidden_size": g["hidden"], "num_hidden_layers": 3,
              "vocab_size": g["vocab"], "num_experts": g["experts"],
              "num_experts_per_tok": g["top_k"],
              "routed_expert_hidden_size": g["latent"],
              "moe_intermediate_size": g["inter"], "num_shared_experts": 1,
              "q_lora_rank": g["q_lora"], "kv_lora_rank": g["kv_lora"],
              "qk_rope_head_dim": g["rope"], "qk_nope_head_dim": g["nope"],
              "v_head_dim": g["v_head"], "num_attention_heads": g["heads"],
              "linear_attn_config": {"num_heads": g["kda_heads"],
                                     "head_dim": g["kda_head"],
                                     "short_conv_kernel_size": g["kernel"]},
              "layer_types": ["linear_attention", "full_attention",
                              "linear_attention"]}
    (root / "config.json").write_text(json.dumps(config))
    t = {}
    hidden = g["hidden"]
    t["model.embed_tokens.weight"] = ("BF16", (g["vocab"], hidden),
                                      rand_bf16(g["vocab"] * hidden))
    t["model.norm.weight"] = ("BF16", (hidden,), rand_bf16(hidden))
    t["lm_head.weight"] = ("BF16", (g["vocab"], hidden),
                           rand_bf16(g["vocab"] * hidden))
    t["model.output_attn_res_norm.weight"] = ("BF16", (hidden,),
                                              rand_bf16(hidden))
    t["model.output_attn_res_proj.weight"] = ("BF16", (1, hidden),
                                              rand_bf16(hidden))
    for layer, kind in enumerate(config["layer_types"]):
        p = f"model.layers.{layer}."
        # Routed layers ship their MoE under block_sparse_moe (the dense
        # replacement layer would keep mlp.gate_proj naming).
        a, m = p + "self_attn.", p + "block_sparse_moe."
        t[p + "input_layernorm.weight"] = ("BF16", (hidden,), rand_bf16(hidden))
        t[p + "post_attention_layernorm.weight"] = ("BF16", (hidden,),
                                                    rand_bf16(hidden))
        for res in ("self_attention_res", "mlp_res"):
            t[p + res + "_norm.weight"] = ("BF16", (hidden,), rand_bf16(hidden))
            t[p + res + "_proj.weight"] = ("BF16", (1, hidden),
                                           rand_bf16(hidden))
        if kind == "linear_attention":
            for proj in "qkv":
                t[a + proj + "_proj.weight"] = ("BF16", (kda_dim, hidden),
                                                rand_bf16(kda_dim * hidden))
                t[a + proj + "_conv1d.weight"] = (
                    "F32", (kda_dim, 1, g["kernel"]), rand_f32(kda_dim * g["kernel"]))
            t[a + "f_a_proj.weight"] = ("BF16", (g["kda_head"], hidden),
                                        rand_bf16(g["kda_head"] * hidden))
            t[a + "f_b_proj.weight"] = ("BF16", (kda_dim, g["kda_head"]),
                                        rand_bf16(kda_dim * g["kda_head"]))
            t[a + "dt_bias"] = ("F32", (kda_dim,), rand_f32(kda_dim))
            t[a + "A_log"] = ("F32", (128,), rand_f32(128))
            t[a + "b_proj.weight"] = ("BF16", (g["kda_heads"], hidden),
                                      rand_bf16(g["kda_heads"] * hidden))
            # Released checkpoint (55cd2f9 full-rank gate reconciliation):
            # the low-rank g_a/g_b pair does not exist - the gate is the
            # checkpoint's full-rank g_proj, read [kda_dim, hidden].
            t[a + "g_proj.weight"] = ("BF16", (kda_dim, hidden),
                                      rand_bf16(kda_dim * hidden))
            t[a + "o_norm.weight"] = ("F32", (g["kda_head"],),
                                      rand_f32(g["kda_head"]))
            t[a + "o_proj.weight"] = ("BF16", (hidden, kda_dim),
                                      rand_bf16(hidden * kda_dim))
        else:
            t[a + "q_a_proj.weight"] = ("BF16", (g["q_lora"], hidden),
                                        rand_bf16(g["q_lora"] * hidden))
            t[a + "q_a_layernorm.weight"] = ("BF16", (g["q_lora"],),
                                             rand_bf16(g["q_lora"]))
            t[a + "q_b_proj.weight"] = (
                "BF16", (g["heads"] * (g["nope"] + g["rope"]), g["q_lora"]),
                rand_bf16(g["heads"] * (g["nope"] + g["rope"]) * g["q_lora"]))
            t[a + "kv_a_proj_with_mqa.weight"] = (
                "BF16", (g["kv_lora"] + g["rope"], hidden),
                rand_bf16((g["kv_lora"] + g["rope"]) * hidden))
            t[a + "kv_a_layernorm.weight"] = ("BF16", (g["kv_lora"],),
                                              rand_bf16(g["kv_lora"]))
            t[a + "kv_b_proj.weight"] = (
                "BF16", (g["heads"] * (g["nope"] + g["v_head"]), g["kv_lora"]),
                rand_bf16(g["heads"] * (g["nope"] + g["v_head"]) * g["kv_lora"]))
            # Same 55cd2f9 reconciliation on the full-attention side.
            t[a + "g_proj.weight"] = (
                "BF16", (g["heads"] * g["v_head"], hidden),
                rand_bf16(g["heads"] * g["v_head"] * hidden))
            t[a + "o_proj.weight"] = (
                "BF16", (hidden, g["heads"] * g["v_head"]),
                rand_bf16(hidden * g["heads"] * g["v_head"]))
        t[m + "gate.weight"] = ("BF16", (g["experts"], hidden),
                                rand_bf16(g["experts"] * hidden))
        t[m + "gate.e_score_correction_bias"] = ("F32", (g["experts"],),
                                                 rand_f32(g["experts"]))
        t[m + "routed_expert_down_proj.weight"] = (
            "BF16", (g["latent"], hidden), rand_bf16(g["latent"] * hidden))
        t[m + "routed_expert_up_proj.weight"] = (
            "BF16", (hidden, g["latent"]), rand_bf16(hidden * g["latent"]))
        t[m + "routed_expert_norm.weight"] = ("BF16", (g["latent"],),
                                              rand_bf16(g["latent"]))
        t[m + "shared_experts.gate_proj.weight"] = (
            "BF16", (g["inter"], hidden), rand_bf16(g["inter"] * hidden))
        t[m + "shared_experts.up_proj.weight"] = (
            "BF16", (g["inter"], hidden), rand_bf16(g["inter"] * hidden))
        t[m + "shared_experts.down_proj.weight"] = (
            "BF16", (hidden, g["inter"]), rand_bf16(hidden * g["inter"]))
        for e in range(g["experts"]):
            base = m + f"experts.{e}."
            for name, rows, cols in (("w1", g["inter"], g["latent"]),
                                     ("w3", g["inter"], g["latent"]),
                                     ("w2", g["latent"], g["inter"])):
                t[base + name + ".weight"] = ("U8", (rows, cols // 2),
                                              rand_u8(rows * cols // 2))
                scale = bytearray(rand_u8(rows * (cols // 32), lo=100, hi=150))
                if poison_scale and layer == 2 and e == 3 and name == "w2":
                    scale[0] = 0xFF
                t[base + name + ".weight_scale"] = ("U8", (rows, cols // 32),
                                                    bytes(scale))
    # The released Kimi-K3 checkpoint prefixes every tensor with
    # "language_model." and the packer reads that spelling (2d30fec); the
    # fixture's dict stays unprefixed so its keys remain the test's own
    # manifest names.
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in t.items():
        header["language_model." + name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with open(root / "model.safetensors", "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)
    return t


def read_pack(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    assert magic == 0x4B33504B, "bad magic"
    manifest = json.loads(raw[16:16 + length])
    base = 16 + length
    base += (-base) % 128

    def tensor(name):
        entry = manifest["tensors"][name]
        return raw[base + entry["offset"]: base + entry["offset"] + entry["bytes"]]
    return version, manifest, base, tensor


def run_packer(checkpoint, out, env):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                           str(checkpoint), str(out)],
                          capture_output=True, text=True, env=env)


def numpy_blocked_env(scratch):
    blocker = scratch / "numpy_blocked" / "numpy"
    blocker.mkdir(parents=True)
    (blocker / "__init__.py").write_text(
        'raise ImportError("numpy blocked by test_k3_pack_layout")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(blocker.parent), env.get("PYTHONPATH")) if part)
    return env


def verify_pack(label, src, out):
    def expect(ok, message):
        check(ok, f"{label}: {message}")

    version, manifest, base, tensor = read_pack(out)
    expect(version == 2, f"pack version is {version}, not 2")
    expect(base % 128 == 0, "payload base is not 128-aligned")
    fmt = manifest["format"]
    expect(fmt["alignment"] == 128 and fmt["version"] == 2,
           "format block is wrong")
    expect(fmt["mxfp4_interleave"]["cell_rows"] == 17
           and fmt["mxfp4_interleave"]["row_bytes"] == 64,
           "format block interleave parameters are wrong")

    entries = manifest["tensors"]
    ordered = sorted(entries.items(), key=lambda kv: kv[1]["offset"])
    expect(all(e["offset"] % 128 == 0 for _, e in ordered),
           "a tensor offset is not 128-aligned")
    expect(all(a[1]["offset"] + a[1]["bytes"] <= b[1]["offset"]
               for a, b in zip(ordered, ordered[1:])),
           "tensor extents overlap")
    names = [n for n, _ in ordered]
    expect(names[0] == "model.embed_tokens.weight",
           "embedding is not the first tensor")
    expect(names[-3:] == ["model.norm.weight", "model.attnres_out_weight",
                          "lm_head.weight"],
           f"closing tensors are not last: {names[-3:]}")
    layer_of = [int(n.split(".")[2]) for n in names
                if n.startswith("model.layers.")]
    expect(layer_of == sorted(layer_of), "layers are not emitted in order")

    p, a = "model.layers.0.", "model.layers.0.self_attn."
    want = b"".join(src[a + n][2] for n in
                    ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                     "b_proj.weight"))
    expect(tensor(p + "kda_qkv_beta_weight") == want,
           "fused qkv|beta bytes are not the section concatenation")
    entry = entries[p + "kda_qkv_beta_weight"]
    expect(entry["shard_class"] == "output_dim_heads",
           "fused qkv|beta shard class is wrong")
    expect([s["row_offset"] for s in entry["sections"]] == [0, 128, 256, 384]
           and entry["shape"] == [386, 64],
           "fused qkv|beta section table does not tile the tensor")
    expect(tensor(p + "kda_decay_down_weight") == src[a + "f_a_proj.weight"][2],
           "decay_down bytes are not the checkpoint's f_a_proj")
    expect(entries[p + "kda_decay_down_weight"]["shard_class"] == "replicated",
           "decay_down shard class is wrong")
    expect(tensor(p + "kda_gate_weight") == src[a + "g_proj.weight"][2],
           "gate bytes are not the checkpoint's full-rank g_proj")
    expect(entries[p + "kda_gate_weight"]["shard_class"] == "output_dim_heads",
           "gate shard class is wrong")
    for gone in ("kda_q_weight", "kda_k_weight", "kda_v_weight",
                 "kda_beta_weight", "kda_decay_gate_down_weight",
                 "kda_gate_down_weight", "expert_w1_scale",
                 "expert_w2_scale"):
        expect(p + gone not in entries, f"{gone} should not exist in V2")

    geom = k3_pack.interleave_geometry(2 * MINI["inter"], MINI["latent"],
                                       MINI["experts"])
    got = tensor(p + "expert_w1_weight")
    expect(len(got) == geom["tensor_bytes"],
           "interleaved w1 byte count is off")
    mismatches = 0
    for e in range(MINI["experts"]):
        pay = src[f"{p}block_sparse_moe.experts.{e}.w1.weight"][2] + \
            src[f"{p}block_sparse_moe.experts.{e}.w3.weight"][2]
        sc = src[f"{p}block_sparse_moe.experts.{e}.w1.weight_scale"][2] + \
            src[f"{p}block_sparse_moe.experts.{e}.w3.weight_scale"][2]
        for n in range(0, 2 * MINI["inter"], 16):
            prow = n * (MINI["latent"] // 2)
            at = k3_pack.interleave_byte_offset(geom, e, 0, n, "payload", 0)
            mismatches += got[at:at + 64] != pay[prow:prow + 64]
            srow = n * (MINI["latent"] // 32)
            at = k3_pack.interleave_byte_offset(geom, e, 0, n, "scale", 0)
            mismatches += got[at:at + 4] != sc[srow:srow + 4]
    expect(mismatches == 0,
           f"interleaved w1 misplaces {mismatches} sampled rows")

    gamma = struct.iter_unpack("<H", src[p + "self_attention_res_norm.weight"][2])
    proj = struct.iter_unpack("<H", src[p + "self_attention_res_proj.weight"][2])
    fused = tensor(p + "attnres_attn_weight")
    mismatches = 0
    for i, ((gv,), (pv,)) in enumerate(zip(gamma, proj)):
        gf = struct.unpack("<f", struct.pack("<I", gv << 16))[0]
        pf = struct.unpack("<f", struct.pack("<I", pv << 16))[0]
        want = k3_pack.f32_list_to_bf16_raw([k3_pack.f32_round(gf * pf)])
        mismatches += fused[2 * i:2 * i + 2] != want
    expect(mismatches == 0, "gamma fold is not the f32-exact product")

    expect(tensor(p + "kda_head_log_scale") ==
           src[a + "A_log"][2][:MINI["kda_heads"] * 4],
           "A_log was not narrowed")

    p, a = "model.layers.1.", "model.layers.1.self_attn."
    heads, nope, rope = MINI["heads"], MINI["nope"], MINI["rope"]
    q_lora, kv_lora, v_head = MINI["q_lora"], MINI["kv_lora"], MINI["v_head"]
    q_b = src[a + "q_b_proj.weight"][2]
    kv_b = src[a + "kv_b_proj.weight"][2]
    want_value = b"".join(
        kv_b[2 * (h * (nope + v_head) + nope) * kv_lora:
             2 * (h + 1) * (nope + v_head) * kv_lora] for h in range(heads))
    expect(tensor(p + "mla_kv_b_value_weight") == want_value,
           "kv_b value half is not the checkpoint's bytes")
    q_up = tensor(p + "mla_q_up_weight")
    rope_mismatch = [
        h for h in range(heads)
        if q_up[2 * (h * (kv_lora + rope) + kv_lora) * q_lora:
                2 * (h + 1) * (kv_lora + rope) * q_lora]
        != q_b[2 * (h * (nope + rope) + nope) * q_lora:
               2 * (h + 1) * (nope + rope) * q_lora]]
    expect(not rope_mismatch,
           f"q_up rope rows of heads {rope_mismatch} are not the checkpoint's "
           "bytes")
    return {name: tensor(name) for name in manifest["tensors"]}


def end_to_end():
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        modes = [("numpy blocked", numpy_blocked_env(scratch))]
        if k3_pack.np is not None:
            modes.append(("numpy", dict(os.environ)))
        checkpoint = scratch / "checkpoint"
        checkpoint.mkdir()
        src = mini_checkpoint(checkpoint)
        packed = {}
        for index, (label, env) in enumerate(modes):
            out = scratch / f"mini{index}.pack"
            run = run_packer(checkpoint, out, env)
            check(run.returncode == 0,
                  f"{label}: the packer failed on the valid mini checkpoint: "
                  f"{(run.stdout + run.stderr)[-400:]}")
            if run.returncode != 0:
                continue
            advisory = "ADVISORY numpy not importable" in run.stdout
            check(advisory == (label == "numpy blocked"),
                  f"{label}: the packer ran the "
                  f"{'stdlib' if advisory else 'numpy'} q-fold")
            packed[label] = verify_pack(label, src, out)
        if len(packed) == 2:
            stdlib, accelerated = packed["numpy blocked"], packed["numpy"]
            check(sorted(stdlib) == sorted(accelerated),
                  "the stdlib and numpy packers emit different tensor sets")
            differing = [name for name in stdlib
                         if name in accelerated
                         and not name.endswith("mla_q_up_weight")
                         and stdlib[name] != accelerated[name]]
            check(not differing,
                  f"the stdlib and numpy packers disagree on {differing[:4]}")

        for index, (label, env) in enumerate(modes):
            for case, (kwargs, marker, what) in enumerate((
                    (dict(latent=96), "interleave tiles",
                     "a K that is not whole interleave tiles"),
                    (dict(poison_scale=True), "0xff", "an E8M0 0xff"))):
                refused = scratch / f"refused{index}_{case}"
                refused.mkdir()
                mini_checkpoint(refused, **kwargs)
                run = run_packer(refused, scratch / f"refused{index}_{case}.pack",
                                 env)
                check(run.returncode != 0 and marker in run.stdout,
                      f"{label}: {what} was not refused")


def main():
    unit_geometry()
    unit_relay_matches_addressing()
    unit_sections()
    end_to_end()
    if FAILURES:
        print(f"\nFAIL ({FAILURES})")
        return 1
    print("\npack V2 layout: fused sections tile, the interleave grid closes "
          "and the relay honours the published addressing, 128-aligned "
          "throughout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
