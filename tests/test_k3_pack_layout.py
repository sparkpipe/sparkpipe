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
checkpoint through the real CLI with no numpy installed.
"""
import json
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
            # released checkpoint (full_rank_output_gate): the gate is the
            # full-rank g_proj; the old low-rank g_a/g_b pair does not exist
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
            # released checkpoint (full_rank_output_gate): full-rank g_proj
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
    # "language_model." and the packer's source names follow it; like
    # tests/test_k3_pack.py, the prefix is applied here at write time so the
    # fixture dict's own keys stay unprefixed.
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in t.items():
        header["language_model." + name] = {
            "dtype": dtype, "shape": list(shape),
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


def assert_happy_pack(src, out):
    """The happy-path assertions, held to a pack the real CLI produced."""
    version, manifest, base, tensor = read_pack(out)
    check(version == 2, f"pack version is {version}, not 2")
    check(base % 128 == 0, "payload base is not 128-aligned")
    fmt = manifest["format"]
    check(fmt["alignment"] == 128 and fmt["version"] == 2,
          "format block is wrong")
    check(fmt["mxfp4_interleave"]["cell_rows"] == 17
          and fmt["mxfp4_interleave"]["row_bytes"] == 64,
          "format block interleave parameters are wrong")

    entries = manifest["tensors"]
    ordered = sorted(entries.items(), key=lambda kv: kv[1]["offset"])
    check(all(e["offset"] % 128 == 0 for _, e in ordered),
          "a tensor offset is not 128-aligned")
    check(all(a[1]["offset"] + a[1]["bytes"] <= b[1]["offset"]
              for a, b in zip(ordered, ordered[1:])),
          "tensor extents overlap")
    # consumption order: embed first; layers ascending; closers last
    names = [n for n, _ in ordered]
    check(names[0] == "model.embed_tokens.weight",
          "embedding is not the first tensor")
    check(names[-3:] == ["model.norm.weight", "model.attnres_out_weight",
                         "lm_head.weight"],
          f"closing tensors are not last: {names[-3:]}")
    layer_of = [int(n.split(".")[2]) for n in names
                if n.startswith("model.layers.")]
    check(layer_of == sorted(layer_of), "layers are not emitted in order")

    # the fused KDA tensors: bytes are the section concatenation, tables tile
    p, a = "model.layers.0.", "model.layers.0.self_attn."
    want = b"".join(src[a + n][2] for n in
                    ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                     "b_proj.weight"))
    check(tensor(p + "kda_qkv_beta_weight") == want,
          "fused qkv|beta bytes are not the section concatenation")
    entry = entries[p + "kda_qkv_beta_weight"]
    check(entry["shard_class"] == "output_dim_heads",
          "fused qkv|beta shard class is wrong")
    check([s["row_offset"] for s in entry["sections"]] == [0, 128, 256, 384]
          and entry["shape"] == [386, 64],
          "fused qkv|beta section table does not tile the tensor")
    # released checkpoint (full_rank_output_gate): decay_down is the
    # standalone replicated bottleneck, decay_up rides f_b unchanged, and the
    # gate is the checkpoint's full-rank g_proj - the decay|gate fusion does
    # not exist in what ships (docs/K3_GATE_RECONCILIATION.md, the same
    # contract tests/test_k3_pack.py holds)
    check(tensor(p + "kda_decay_down_weight") ==
          src[a + "f_a_proj.weight"][2],
          "decay_down is not the checkpoint's f_a_proj")
    check(entries[p + "kda_decay_down_weight"]["shard_class"] == "replicated",
          "decay_down shard class is wrong")
    check(tensor(p + "kda_decay_up_weight") ==
          src[a + "f_b_proj.weight"][2],
          "decay_up is not the checkpoint's f_b_proj")
    check(tensor(p + "kda_gate_weight") == src[a + "g_proj.weight"][2],
          "gate is not the checkpoint's full-rank g_proj")
    check(entries[p + "kda_gate_weight"]["shard_class"] == "output_dim_heads",
          "gate shard class is wrong")
    for gone in ("kda_q_weight", "kda_k_weight", "kda_v_weight",
                 "kda_beta_weight", "kda_decay_gate_down_weight",
                 "kda_gate_down_weight", "expert_w1_scale",
                 "expert_w2_scale"):
        check(p + gone not in entries, f"{gone} should not exist in V2")

    # the interleaved expert tensor, held to the checkpoint through the
    # published addressing, cell by cell - not through the packer's relay
    geom = k3_pack.interleave_geometry(2 * MINI["inter"], MINI["latent"],
                                       MINI["experts"])
    got = tensor(p + "expert_w1_weight")
    check(len(got) == geom["tensor_bytes"],
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
    check(mismatches == 0,
          f"interleaved w1 misplaces {mismatches} sampled rows")

    # the gamma fold, recomputed independently in f64: f32-exact then RNE
    gamma = struct.iter_unpack("<H", src[p + "self_attention_res_norm.weight"][2])
    proj = struct.iter_unpack("<H", src[p + "self_attention_res_proj.weight"][2])
    fused = tensor(p + "attnres_attn_weight")
    mismatches = 0
    for i, ((gv,), (pv,)) in enumerate(zip(gamma, proj)):
        gf = struct.unpack("<f", struct.pack("<I", gv << 16))[0]
        pf = struct.unpack("<f", struct.pack("<I", pv << 16))[0]
        want = k3_pack.f32_list_to_bf16_raw([k3_pack.f32_round(gf * pf)])
        mismatches += fused[2 * i:2 * i + 2] != want
    check(mismatches == 0, "gamma fold is not the f32-exact product")

    # A_log narrows to the runtime head count
    check(tensor(p + "kda_head_log_scale") ==
          src[a + "A_log"][2][:MINI["kda_heads"] * 4],
          "A_log was not narrowed")


def end_to_end():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        src = mini_checkpoint(root)
        out = root / "mini.pack"
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(out)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            # an aborted pack is a FAILURE, never a silent skip: record it and
            # still run the refusal cases below
            check(False, "packer: " + (run.stdout + run.stderr)[-400:])
        else:
            assert_happy_pack(src, out)

        # a checkpoint the interleave grid does not divide is refused
        for stale in root.iterdir():
            if stale.suffix == ".pack":
                stale.unlink()
        mini_checkpoint(root, latent=96)
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(out)],
                             capture_output=True, text=True)
        check(run.returncode != 0 and "interleave tiles" in run.stdout,
              "a K that is not whole interleave tiles was not refused")

        # a poisoned scale is refused even though it sits inside the fused stream
        for stale in root.iterdir():
            if stale.suffix == ".pack":
                stale.unlink()
        mini_checkpoint(root, poison_scale=True)
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "k3_pack.py"),
                              str(root), str(out)],
                             capture_output=True, text=True)
        check(run.returncode != 0 and "0xff" in run.stdout,
              "an E8M0 0xff was not refused")


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
