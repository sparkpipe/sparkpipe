#!/usr/bin/env python3
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "tools" / "motif3_tp16_pack.py"
TP = 2
G = {"hidden": 8, "layers": 3, "vocab": 8, "heads": 10, "groups": 2, "group": 5, "kv": 2, "head_dim": 6, "nope": 4, "rope": 2, "value": 4,
     "q_lora": 4, "latent": 4, "experts": 4, "expert_inter": 4, "dense_inter": 6, "streams": 4, "first_routed": 2}


def bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def f32(raw):
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)


def checkpoint_tensors(rng):
    h, s, signal = G["hidden"], G["streams"], G["groups"] * (G["group"] - 1)
    shapes = {"model.embed_tokens.weight": (G["vocab"], h), "lm_head.weight": (G["vocab"], h), "model.norm.weight": (h,),
              "model.mtp_layers.0.input_proj.weight": (h, 2 * h)}
    for layer in range(G["layers"]):
        p = f"model.layers.{layer}."
        shapes |= {p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,)}
        for site in ("mhc_attn", "mhc_ffn"):
            shapes |= {p + f"{site}.alpha_pre": (1,), p + f"{site}.alpha_post": (1,), p + f"{site}.alpha_res": (1,),
                       p + f"{site}.bias_pre": (s,), p + f"{site}.bias_post": (s,), p + f"{site}.bias_res": (s, s),
                       p + f"{site}.proj_pre.weight": (s, s * h), p + f"{site}.proj_post.weight": (s, s * h),
                       p + f"{site}.proj_res.weight": (s * s, s * h), p + f"{site}.rms_norm.weight": (s * h,)}
        a = p + "self_attn."
        shapes |= {a + "wq_a.weight": (G["q_lora"], h), a + "q_norm.weight": (G["q_lora"],), a + "wq_b.weight": (G["heads"] * G["head_dim"], G["q_lora"]),
                   a + "wq_b_gate.weight": (signal * G["value"], G["q_lora"]), a + "wkv_a.weight": (G["latent"] + G["rope"], h),
                   a + "kv_norm.weight": (G["latent"],), a + "wkv_b.weight": (G["kv"] * (G["nope"] + G["value"]), G["latent"]),
                   a + "lambda_proj.weight": (signal, h), a + "wo.weight": (h, signal * G["value"])}
        if layer < G["first_routed"]:
            m = p + "mlp."
            shapes |= {m + "gate_proj.weight": (G["dense_inter"], h), m + "up_proj.weight": (G["dense_inter"], h),
                       m + "down_proj.weight": (h, G["dense_inter"]), m + "act_fn.weight": (3,), m + "act_fn.bias": (1,)}
        else:
            e, i = p + "moe.", G["expert_inter"]
            shapes |= {e + "expert_bias": (G["experts"],), e + "router.gate.weight": (G["experts"], h),
                       e + "experts.gate_up_proj": (G["experts"], 2 * i, h), e + "experts.down_proj": (G["experts"], h, i),
                       e + "experts.act_fn.weight": (G["experts"], 3), e + "experts.act_fn.bias": (G["experts"], 1),
                       e + "shared_experts.gate_proj.weight": (i, h), e + "shared_experts.up_proj.weight": (i, h),
                       e + "shared_experts.down_proj.weight": (h, i), e + "shared_experts.act_fn.weight": (3,), e + "shared_experts.act_fn.bias": (1,)}
    return {name: bf16(rng.standard_normal(shape)) for name, shape in shapes.items()}


def write_checkpoint(directory, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        blob = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header).encode()
    with open(directory / "model-00001-of-00001.safetensors", "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": offset},
                                                                        "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}))
    config = {"architectures": ["MotifForCausalLM"], "model_type": "Motif", "dtype": "bfloat16", "attention_cls": "gdla", "diff_v2": True,
              "hidden_act": "poly_norm", "mhc_enabled": True, "num_shared_experts": 1, "interleave_moe_layer_step": 1}
    (directory / "config.json").write_text(json.dumps(config))
    contract = {"architecture": "MotifForCausalLM", "source_revision": "test",
                "model": {"hidden_dimension": G["hidden"], "layer_count": G["layers"], "vocabulary_size": G["vocab"]},
                "attention": {"query_head_count": G["heads"], "noise_head_count": G["groups"], "group_size": G["group"], "key_value_head_count": G["kv"],
                              "head_dimension": G["head_dim"], "value_head_dimension": G["value"], "qk_nope_head_dimension": G["nope"],
                              "query_lora_rank": G["q_lora"], "kv_lora_rank": G["latent"], "qk_rope_head_dimension": G["rope"]},
                "moe": {"routed_expert_count": G["experts"], "expert_intermediate_dimension": G["expert_inter"],
                        "dense_intermediate_dimension": G["dense_inter"], "first_dense_layer_count": G["first_routed"]},
                "hyper_connections": {"expansion_rate": G["streams"]},
                "source": {"config_json": {"sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()}}}
    (directory / "contract.json").write_text(json.dumps(contract))


def read_pack(path):
    data = path.read_bytes()
    _, _, length = struct.unpack_from("<IIQ", data)
    manifest = json.loads(data[16:16 + length])
    base = (16 + length + 127) // 128 * 128
    tensors = {}
    for name, entry in manifest["tensors"].items():
        dtype = np.float32 if entry["kind"] == "f32" else np.uint16
        tensors[name] = np.frombuffer(data, dtype=dtype, count=entry["bytes"] // np.dtype(dtype).itemsize,
                                      offset=base + entry["offset"]).reshape(entry["shape"])
    return manifest["config"], tensors


class MotifPackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.source = checkpoint_tensors(np.random.default_rng(7))
        write_checkpoint(root, cls.source)
        run = subprocess.run([sys.executable, str(PACKER), "--checkpoint", str(root), "--output-dir", str(root / "out"), "--contract",
                              str(root / "contract.json"), "--tp-degree", str(TP)], capture_output=True, text=True)
        if run.returncode != 0:
            raise AssertionError(run.stdout + run.stderr)
        cls.ranks = [read_pack(root / "out" / f"motif3.tp{TP}.rank{rank:02d}.pack") for rank in range(TP)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tensor(self, name):
        return self.source[name]

    def test_attention_follows_head_groups(self):
        groups, signal_per = G["groups"] // TP, G["group"] - 1
        for rank, (settings, packed) in enumerate(self.ranks):
            self.assertEqual(settings["head_group_first"], rank * groups)
            a = "model.layers.1.self_attn."
            heads = range(rank * groups * G["group"], (rank + 1) * groups * G["group"])
            signal = range(rank * groups * signal_per, (rank + 1) * groups * signal_per)
            q_b = self.tensor(a + "wq_b.weight").reshape(G["heads"], G["head_dim"], -1)
            np.testing.assert_array_equal(packed["layers.1.attn_q_b"], np.concatenate([q_b[h] for h in heads]))
            gate = self.tensor(a + "wq_b_gate.weight").reshape(-1, G["value"], G["q_lora"])
            np.testing.assert_array_equal(packed["layers.1.attn_gate"], np.concatenate([gate[s] for s in signal]))
            np.testing.assert_array_equal(packed["layers.1.attn_lambda"], self.tensor(a + "lambda_proj.weight")[list(signal)])
            out = self.tensor(a + "wo.weight").reshape(G["hidden"], -1, G["value"])
            np.testing.assert_array_equal(packed["layers.1.attn_out"], np.concatenate([out[:, s] for s in signal], axis=1))
            kv_b = self.tensor(a + "wkv_b.weight").reshape(G["kv"], G["nope"] + G["value"], -1)
            np.testing.assert_array_equal(packed["layers.1.attn_kv_b"], np.concatenate([kv_b[k] for k in range(rank * groups, (rank + 1) * groups)]))
            np.testing.assert_array_equal(packed["layers.1.attn_q_a"], self.tensor(a + "wq_a.weight"))

    def test_experts_are_whole_and_local(self):
        local = G["experts"] // TP
        for rank, (settings, packed) in enumerate(self.ranks):
            self.assertEqual((settings["local_experts"], settings["expert_first"], settings["expert_layout"]), (local, rank * local, "whole"))
            e = "model.layers.2.moe."
            window = slice(rank * local, (rank + 1) * local)
            np.testing.assert_array_equal(packed["layers.2.moe_w13"], self.tensor(e + "experts.gate_up_proj")[window])
            np.testing.assert_array_equal(packed["layers.2.moe_w2"], self.tensor(e + "experts.down_proj")[window])
            poly = np.concatenate([f32(self.tensor(e + "experts.act_fn.weight")), f32(self.tensor(e + "experts.act_fn.bias"))], axis=1)[window]
            np.testing.assert_array_equal(packed["layers.2.moe_poly"], poly)
            np.testing.assert_array_equal(packed["layers.2.expert_bias"], f32(self.tensor(e + "expert_bias")))

    def test_single_owner_mlps(self):
        for layer, owner in ((0, 0), (1, 1)):
            for rank, (_, packed) in enumerate(self.ranks):
                self.assertEqual(f"layers.{layer}.ffn_gate_up" in packed, rank == owner, (layer, rank))
        m = "model.layers.1.mlp."
        np.testing.assert_array_equal(self.ranks[1][1]["layers.1.ffn_gate_up"],
                                      np.concatenate([self.tensor(m + "gate_proj.weight"), self.tensor(m + "up_proj.weight")]))
        for rank, (_, packed) in enumerate(self.ranks):
            self.assertEqual("layers.2.shared_gate_up" in packed, rank == 2 % TP)

    def test_hyper_connection_site_folds_the_norm(self):
        p = "model.layers.0.mhc_ffn."
        flat = G["streams"] * G["hidden"]
        proj = np.concatenate([f32(self.tensor(p + "proj_pre.weight")), f32(self.tensor(p + "proj_post.weight")), f32(self.tensor(p + "proj_res.weight"))])
        norm = f32(self.tensor(p + "rms_norm.weight"))
        for _, packed in self.ranks:
            np.testing.assert_array_equal(packed["layers.0.hc_ffn_fn"], (proj * norm[None, :]).astype(np.float32))
            self.assertEqual(packed["layers.0.hc_ffn_fn"].shape, (6 * G["streams"], flat))
            np.testing.assert_array_equal(packed["layers.0.hc_ffn_scale"], [f32(self.tensor(p + f"alpha_{x}"))[0] for x in ("pre", "post", "res")])
            np.testing.assert_array_equal(packed["layers.0.hc_ffn_base"], np.concatenate(
                [f32(self.tensor(p + "bias_pre")), f32(self.tensor(p + "bias_post")), f32(self.tensor(p + "bias_res")).reshape(-1)]))

    def test_vocabulary_rows(self):
        rows = G["vocab"] // TP
        for rank, (_, packed) in enumerate(self.ranks):
            np.testing.assert_array_equal(packed["embed"], self.tensor("model.embed_tokens.weight")[rank * rows:(rank + 1) * rows])
            np.testing.assert_array_equal(packed["head"], self.tensor("lm_head.weight")[rank * rows:(rank + 1) * rows])


if __name__ == "__main__":
    sys.exit(unittest.main())
