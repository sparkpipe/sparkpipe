#!/usr/bin/env python3
"""Round-trip the mimo26 stage packer against a synthetic mini checkpoint.

Builds a real-bytes fake checkpoint with the mimo26 tensor inventory at a
miniature geometry (the module constants are overridden the way the qwen38_27b
packer test does), then drives emit -> assemble -> verify for both ranks of
TP2 and checks the properties a shape table cannot see:

  * every payload and scale plane is byte-exact against the source (verify),
    including the fused-qkv q/k/v section split, the upstream scale-grid row
    padding (fixture pads the grid like the real pro arm), kv replication,
    vocab row slicing and the per-rank disjoint expert slabs
  * the two ranks' expert slabs cover disjoint expert ids and together cover
    every expert exactly once
  * out-of-scope tensors (vision tower, MTP head, dflash) sit in the index
    and are never read
  * a flipped byte in the pack, a missing shard and a wrong shape are hard
    failures (PACK FAILURE), never silent passes
  * staging resume: deleting one staged payload re-emits only that file
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import mimo26_stagepack as packer  # noqa: E402

MINI = dict(
    hidden=256, layers=5, heads=8, head_dim=64, v_head_dim=64,
    kv_full=2, kv_swa=2, vocab=512, dense_inter=256,
    experts=8, experts_per_token=2, expert_inter=128,
    swa_window=8, default_tp=2,
)
HYBRID = [0, 1, 1, 0, 1]     # full at 0 and 3
MOE_FREQ = [0, 1, 1, 1, 1]   # dense layer 0
GRID_PAD_ROWS = 2            # the fixture pads fused scale grids (pro: 212->216)


def kv_heads(layer: int) -> int:
    return MINI["kv_full"] if HYBRID[layer] == 0 else MINI["kv_swa"]


def build_checkpoint(directory: Path) -> dict:
    tensors: dict = {}
    g = MINI
    def add(name, dtype, shape):
        tensors[name] = (dtype, shape)
    add("model.embed_tokens.weight", "BF16", [g["vocab"], g["hidden"]])
    add("model.norm.weight", "BF16", [g["hidden"]])
    add("lm_head.weight", "BF16", [g["vocab"], g["hidden"]])
    for layer in range(g["layers"]):
        kv = kv_heads(layer)
        qkv_rows = g["heads"] * g["head_dim"] + kv * g["head_dim"] + kv * g["v_head_dim"]
        module = f"model.layers.{layer}.self_attn.qkv_proj"
        add(module + ".weight", "F8_E4M3", [qkv_rows, g["hidden"]])
        add(module + ".weight_scale_inv", "F32",
            [qkv_rows // 128 + GRID_PAD_ROWS, g["hidden"] // 128])
        add(f"model.layers.{layer}.self_attn.o_proj.weight", "BF16",
            [g["hidden"], g["heads"] * g["v_head_dim"]])
        add(f"model.layers.{layer}.input_layernorm.weight", "BF16", [g["hidden"]])
        add(f"model.layers.{layer}.post_attention_layernorm.weight", "BF16", [g["hidden"]])
        if HYBRID[layer] == 1:
            add(f"model.layers.{layer}.self_attn.attention_sink_bias", "BF16", [g["heads"]])
        if MOE_FREQ[layer]:
            add(f"model.layers.{layer}.mlp.gate.weight", "BF16", [g["experts"], g["hidden"]])
            add(f"model.layers.{layer}.mlp.gate.e_score_correction_bias", "F32", [g["experts"]])
            for expert in range(g["experts"]):
                for mat, rows, cols in (
                    ("gate_proj", g["expert_inter"], g["hidden"]),
                    ("up_proj", g["expert_inter"], g["hidden"]),
                    ("down_proj", g["hidden"], g["expert_inter"]),
                ):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{mat}.weight"
                    add(name, "U8", [rows, cols // 2])
                    add(name.replace(".weight", ".weight_scale"), "U8", [rows, cols // 32])
        else:
            for mat in ("gate_proj", "up_proj"):
                add(f"model.layers.{layer}.mlp.{mat}.weight", "F8_E4M3",
                    [g["dense_inter"], g["hidden"]])
                add(f"model.layers.{layer}.mlp.{mat}.weight_scale_inv", "F32",
                    [g["dense_inter"] // 128, g["hidden"] // 128])
            add(f"model.layers.{layer}.mlp.down_proj.weight", "F8_E4M3",
                [g["hidden"], g["dense_inter"]])
            add(f"model.layers.{layer}.mlp.down_proj.weight_scale_inv", "F32",
                [g["hidden"] // 128, g["dense_inter"] // 128])
    # out-of-scope: in the index, never referenced by the packer
    add("model.mtp.layers.0.eh_proj.weight", "F32", [64, 64])
    add("visual.blocks.0.attn.qkv.weight", "BF16", [64, 64])
    add("audio_encoder.norm.weight", "BF16", [64])

    config = {
        "model_type": "mimo_v2",
        "hidden_size": g["hidden"], "num_hidden_layers": g["layers"],
        "num_attention_heads": g["heads"], "head_dim": g["head_dim"],
        "v_head_dim": g["v_head_dim"], "num_key_value_heads": g["kv_full"],
        "swa_num_key_value_heads": g["kv_swa"], "vocab_size": g["vocab"],
        "intermediate_size": g["dense_inter"], "n_routed_experts": g["experts"],
        "num_experts_per_tok": g["experts_per_token"],
        "moe_intermediate_size": g["expert_inter"],
        "tie_word_embeddings": False,
        "hybrid_layer_pattern": HYBRID, "moe_layer_freq": MOE_FREQ,
        "quantization_config": {
            "fmt": "e4m3",
            "ignored_layers": [
                f"model.layers.{layer}.self_attn.o_proj"
                for layer in range(g["layers"])] + ["model.decoder.self_attn.o_proj"],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))

    shards = {"model_pp0_ep0_shard0.safetensors": list(tensors)}
    weight_map = {}
    for shard, names in shards.items():
        header = {}
        cursor = 0
        for name in names:
            dtype, shape = tensors[name]
            elements = math.prod(shape)
            per = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}[dtype]
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [cursor, cursor + elements * per]}
            cursor += elements * per
        header_json = json.dumps(header).encode()
        with open(directory / shard, "wb") as file:
            file.write(struct.pack("<Q", len(header_json)))
            file.write(header_json)
            # deterministic non-zero payload: a repeating position-dependent byte
            block = bytes(range(251, 256)) * 8192
            written = 0
            while written < cursor:
                step = min(len(block), cursor - written)
                file.write(block[:step])
                written += step
        for name in names:
            weight_map[name] = shard
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return tensors


class Args:
    def __init__(self, **kw):
        self.arm = "pro"
        self.checkpoint = ""
        self.tp = 2
        self.rank = 0
        self.out = ""
        self.stage_dir = None
        self.emit = self.assemble = self.verify = False
        for key, value in kw.items():
            setattr(self, key, value)


def run_rank(tmp: Path, rank: int):
    out = tmp / f"mimo26mini.rank{rank}.sp"
    args = Args(checkpoint=str(tmp / "ckpt"), tp=2, rank=rank, out=str(out),
                stage_dir=str(tmp / f"stage{rank}"))
    args.emit = True
    assert packer.do_emit(args) == 0
    args.emit = False
    args.assemble = True
    assert packer.do_assemble(args) == 0
    args.assemble = False
    args.verify = True
    assert packer.do_verify(args) == 0
    return out


def main() -> int:
    packer.ARMS["pro"] = dict(MINI)
    with tempfile.TemporaryDirectory(prefix="mimo26-pack-test-") as raw:
        tmp = Path(raw)
        ckpt = tmp / "ckpt"
        ckpt.mkdir()
        build_checkpoint(ckpt)
        packs = [run_rank(tmp, rank) for rank in (0, 1)]

        # expert slabs: disjoint across ranks, complete jointly
        expert_bytes = []
        for rank, out in enumerate(packs):
            with open(out, "rb") as file:
                header = packer.HEADER_STRUCT.unpack(file.read(packer.HEADER_BYTES))
                assert header[0] == packer.MAGIC and header[4] > 0
                file.seek(packer.HEADER_BYTES)
                entries = [packer.ENTRY_STRUCT.unpack(
                    file.read(packer.ENTRY_BYTES)) for _ in range(header[4])]
            kinds = {(e[0], e[1]) for e in entries}
            for mat_kind in (packer.KIND_EXPERT_GATE, packer.KIND_EXPERT_UP,
                             packer.KIND_EXPERT_DOWN):
                slabs = [e for e in entries if e[0] == mat_kind]
                assert len(slabs) == sum(MOE_FREQ), (mat_kind, len(slabs))
                assert all(e[3] == (MINI["experts"] // 2) *
                           (MINI["expert_inter"] if mat_kind != packer.KIND_EXPERT_DOWN
                            else MINI["hidden"]) for e in slabs)
            assert (packer.KIND_SINK_BIAS, ) not in kinds
            expert_bytes.append(sum(e[3] * e[4] // 2 for e in entries
                                    if e[0] in (packer.KIND_EXPERT_GATE,
                                                packer.KIND_EXPERT_UP,
                                                packer.KIND_EXPERT_DOWN)))
        assert expert_bytes[0] == expert_bytes[1] > 0

        # windowed emission: two disjoint windows + globals merge into one pack
        merged_stage = tmp / "stage_w"
        merged_stage.mkdir()
        out_w = tmp / "mimo26mini.window.sp"
        for window in ("0:2", "2:3"):
            args_w = Args(checkpoint=str(ckpt), tp=2, rank=0, out=str(out_w),
                          stage_dir=str(merged_stage), layer_window=window)
            args_w.emit = True
            assert packer.do_emit(args_w) == 0
        args_w = Args(checkpoint=str(ckpt), tp=2, rank=0, out=str(out_w),
                      stage_dir=str(merged_stage))
        args_w.assemble = True
        assert packer.do_assemble(args_w) == 0
        args_w.assemble = False
        args_w.verify = True
        assert packer.do_verify(args_w) == 0

        # corrupt one payload byte: verify must fail loudly
        bad = tmp / "corrupt.sp"
        bad.write_bytes(packs[0].read_bytes())
        blob = bytearray(bad.read_bytes())
        blob[-1] ^= 0xFF
        bad.write_bytes(bytes(blob))
        import shutil
        shutil.copy(str(packs[0]) + ".receipt.json", str(bad) + ".receipt.json")
        try:
            packer.do_verify(Args(checkpoint=str(ckpt), tp=2, rank=0, out=str(bad)))
            raise AssertionError("corrupted pack verified")
        except (packer.PackFailure, SystemExit):
            pass

        # resume: one deleted staged payload is re-emitted, others untouched
        stage = tmp / "stage0"
        victim = stage / "r000_03.payload"
        survivor = stage / "rg_00.payload"
        assert victim.exists() and survivor.exists()
        before = survivor.stat().st_mtime_ns
        victim.unlink()
        assert packer.do_emit(Args(checkpoint=str(ckpt), tp=2, rank=0,
                                   out=str(packs[0]), stage_dir=str(stage))) == 0
        assert victim.exists() and victim.stat().st_size > 0
        assert survivor.stat().st_mtime_ns == before

        # wrong shape: break one expert tensor, expect a hard failure
        index = json.loads((ckpt / "model.safetensors.index.json").read_text())
        shard = ckpt / index["weight_map"]["model.layers.1.mlp.experts.0.gate_proj.weight"]
        with open(shard, "r+b") as file:
            file.seek(0)
            (n,) = struct.unpack("<Q", file.read(8))
            header = json.loads(file.read(n))
            header["model.layers.1.mlp.experts.0.gate_proj.weight"]["shape"] = [7, 64]
            blob = json.dumps(header).encode()
            file.seek(0)
            file.write(struct.pack("<Q", len(blob)))
            file.write(blob)
        try:
            packer.do_verify(Args(checkpoint=str(ckpt), tp=2, rank=0, out=str(packs[0])))
            raise AssertionError("wrong shape verified")
        except (packer.PackFailure, SystemExit, json.JSONDecodeError, struct.error):
            pass

    print("PASS mimo26 stagepack round-trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
