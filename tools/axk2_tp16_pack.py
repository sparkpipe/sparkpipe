#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from glm52_fp8_linear_repack import scale_groups, slice_fp8  # noqa: E402
from glm52_resident_stagepack import (  # noqa: E402
    ALIGNMENT, CODEC_ABI_VERSION, CODEC_BF16, CODEC_FP8, ENTRY_BYTES, FORMAT_VERSION, GLOBAL_LAYER, HEADER_BYTES,
    K_ATTN_NORM, K_ATTN_OUTPUT, K_DENSE_DOWN, K_DENSE_GATE_UP, K_EMBEDDING, K_EXPERT_DOWN, K_EXPERT_UP_GATE,
    K_FINAL_NORM, K_INDEX_HEAD, K_INDEX_K, K_INDEX_NORM_B, K_INDEX_NORM_W, K_INDEX_Q, K_KV_A, K_KV_A_NORM, K_LM_HEAD,
    K_POST_ATTN_NORM, K_Q_A, K_Q_A_NORM, K_Q_B, K_ROUTER, K_ROUTER_CORRECTION, K_SHARED_DOWN, K_SHARED_GATE_UP,
    MAGIC, MODEL_REVISION_BYTES, PAYLOAD_PACKED_WEIGHT, SCALE_F32, Entry, Fp8SourceReader, Packer, PlanItem)
from spark_pack_common import PackFailure, align_up  # noqa: E402

K_ATTN_NORM_GATE_DOWN, K_ATTN_NORM_GATE_UP, K_POST_NORM_GATE_DOWN, K_POST_NORM_GATE_UP = 26, 27, 28, 29
TP_DEGREE = 16
HEX = "0123456789abcdef"
GEOMETRY = {
    "layers": 61, "hidden_dimension": 7168, "output_vocab_count": 163840, "moe_expert_count": 256,
    "head_count": 64, "qk_nope_head_dimension": 128, "value_head_dimension": 128, "latent_dimension": 512,
    "first_routed_layer": 1, "moe_intermediate_dimension": 2048,
}


class Axk2Packer(Packer):
    def add_linear(self, kind, layer, names, shard):
        rows = 0
        cols = None
        for name in names:
            dtype, shape, _ = self.source.meta(name)
            if dtype != "F8_E4M3" or len(shape) != 2:
                raise PackFailure(f"{name}: {dtype} {shape}, the stage reads this linear as F8_E4M3")
            part_rows, part_cols = shape
            if shard == "rows":
                part_rows //= self.tp_degree
            elif shard == "cols":
                part_cols //= self.tp_degree
            if cols is not None and cols != part_cols:
                raise PackFailure(f"{names}: fused parts disagree on columns")
            rows += part_rows
            cols = part_cols
        entry = Entry(kind, layer, PAYLOAD_PACKED_WEIGHT, CODEC_FP8, SCALE_F32, 1, rows, cols)
        payload_bytes = rows * cols
        scale_bytes = rows * scale_groups(cols) * 4
        source, degree, rank = self.source, self.tp_degree, self.tp_rank
        held = {}

        def parts():
            if "parts" not in held:
                held["parts"] = [slice_fp8(source, name, shard, degree, rank) for name in names]
            return held["parts"]

        def produce_payload():
            for codes, _ in parts():
                yield codes.tobytes()

        def produce_scale():
            for _, scale in parts():
                yield np.ascontiguousarray(scale, dtype=np.float32).tobytes()
            held.clear()

        self.plan.append(PlanItem(entry, produce_payload, produce_scale, payload_bytes, scale_bytes, list(names)))

    def build_plan(self):
        c = self.c
        self.add_spine_bf16(K_EMBEDDING, GLOBAL_LAYER, "model.embed_tokens.weight", shard="rows")
        self.add_spine_bf16(K_FINAL_NORM, GLOBAL_LAYER, "model.norm.weight")
        self.add_spine_bf16(K_LM_HEAD, GLOBAL_LAYER, "lm_head.weight", shard="rows")
        for layer in range(c["layers"]):
            prefix = f"model.layers.{layer}"
            attn = f"{prefix}.self_attn"
            mlp = f"{prefix}.mlp"
            routed = layer >= c["first_routed_layer"]
            self.add_spine_bf16(K_ATTN_NORM, layer, f"{prefix}.input_layernorm.norm.weight")
            self.add_spine_bf16(K_ATTN_NORM_GATE_DOWN, layer, f"{prefix}.input_layernorm.W_down.weight")
            self.add_spine_bf16(K_ATTN_NORM_GATE_UP, layer, f"{prefix}.input_layernorm.W_up.weight")
            self.add_linear(K_Q_A, layer, [f"{attn}.q_a_proj.weight"], "")
            self.add_spine_bf16(K_Q_A_NORM, layer, f"{attn}.q_a_layernorm.weight")
            self.add_linear(K_Q_B, layer, [f"{attn}.q_b_proj.weight"], "rows")
            self.add_linear(K_KV_A, layer, [f"{attn}.kv_a_proj_with_mqa.weight"], "")
            self.add_spine_bf16(K_KV_A_NORM, layer, f"{attn}.kv_a_layernorm.weight")
            self.add_kv_b(layer)
            self.add_linear(K_ATTN_OUTPUT, layer, [f"{attn}.o_proj.weight"], "cols")
            if routed:
                self.add_spine_bf16(K_POST_ATTN_NORM, layer, f"{prefix}.post_attention_layernorm.norm.weight")
                self.add_spine_bf16(K_POST_NORM_GATE_DOWN, layer, f"{prefix}.post_attention_layernorm.W_down.weight")
                self.add_spine_bf16(K_POST_NORM_GATE_UP, layer, f"{prefix}.post_attention_layernorm.W_up.weight")
            else:
                self.add_spine_bf16(K_POST_ATTN_NORM, layer, f"{prefix}.post_attention_layernorm.weight")
            self.add_linear(K_INDEX_Q, layer, [f"{attn}.indexer.wq_b.weight"], "")
            self.add_linear(K_INDEX_K, layer, [f"{attn}.indexer.wk.weight"], "")
            self.add_spine_bf16(K_INDEX_HEAD, layer, f"{attn}.indexer.weights_proj.weight")
            self.add_spine_bf16(K_INDEX_NORM_W, layer, f"{attn}.indexer.k_norm.weight")
            self.add_spine_bf16(K_INDEX_NORM_B, layer, f"{attn}.indexer.k_norm.bias")
            if not routed:
                self.add_linear(K_DENSE_GATE_UP, layer, [f"{mlp}.up_proj.weight", f"{mlp}.gate_proj.weight"], "rows")
                self.add_linear(K_DENSE_DOWN, layer, [f"{mlp}.down_proj.weight"], "cols")
                continue
            self.add_spine_bf16(K_ROUTER, layer, f"{mlp}.gate.weight")
            self.add_f32(K_ROUTER_CORRECTION, layer, f"{mlp}.gate.e_score_correction_bias")
            self.add_experts(K_EXPERT_UP_GATE, layer, ["up_proj", "gate_proj"], c["moe_intermediate_dimension"],
                             c["hidden_dimension"], shard="rows")
            self.add_experts(K_EXPERT_DOWN, layer, ["down_proj"], c["hidden_dimension"], c["moe_intermediate_dimension"],
                             shard="cols")
            self.add_linear(K_SHARED_GATE_UP, layer, [f"{mlp}.shared_experts.up_proj.weight",
                                                      f"{mlp}.shared_experts.gate_proj.weight"], "rows")
            self.add_linear(K_SHARED_DOWN, layer, [f"{mlp}.shared_experts.down_proj.weight"], "cols")

    def sequential_layout(self):
        offset = align_up(align_up(HEADER_BYTES, ALIGNMENT) + len(self.plan) * ENTRY_BYTES, ALIGNMENT)
        for item in self.plan:
            item.entry.payload_offset = offset
            offset = align_up(offset + item.entry.payload_bytes, ALIGNMENT)
            if item.entry.scale_bytes:
                item.entry.scale_offset = offset
                offset = align_up(offset + item.entry.scale_bytes, ALIGNMENT)
        return offset


def header_bytes(packer, file_bytes, revision, contract_sha256, config_sha256, recipe_sha256):
    c = packer.c
    directory_offset = align_up(HEADER_BYTES, ALIGNMENT)
    header = struct.pack(
        "<20I2Q65s32s32s32s", MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, CODEC_ABI_VERSION, 0, len(packer.plan), 1, 0, 0,
        c["layers"], c["layers"], c["hidden_dimension"], c["output_vocab_count"], c["moe_expert_count"], CODEC_FP8, CODEC_FP8,
        CODEC_BF16, packer.tp_degree, packer.tp_rank, directory_offset, file_bytes,
        revision.encode("utf-8")[:MODEL_REVISION_BYTES - 1].ljust(MODEL_REVISION_BYTES, b"\0"), contract_sha256, config_sha256,
        recipe_sha256)
    blob = header.ljust(directory_offset, b"\0")
    for item in packer.plan:
        e = item.entry
        blob += struct.pack("<8I4Q", e.kind, e.layer, e.payload_type, e.weight_codec, e.scale_encoding, e.group_count, e.rows,
                            e.columns, e.payload_offset, e.payload_bytes, e.scale_offset, e.scale_bytes)
    return blob


class Sink:
    def __init__(self, rank, directory, name):
        self.host = f"spark{HEX[rank]}"
        self.path = f"{directory}/{name}"
        self.digest = hashlib.sha256()
        self.written = 0
        self.process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", self.host, f"mkdir -p {directory} && cat > {self.path}.partial"],
            stdin=subprocess.PIPE, bufsize=1 << 24)

    def write(self, data):
        self.digest.update(data)
        self.stream().write(data)
        self.written += len(data)

    def stream(self):
        return self.process.stdin

    def pad_to(self, offset):
        if offset < self.written:
            raise PackFailure(f"{self.host}: layout moved backwards at {offset}")
        if offset > self.written:
            self.write(b"\0" * (offset - self.written))

    def finish(self, receipt):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise PackFailure(f"{self.host}: the pack stream failed")
        name = Path(self.path).name
        script = (f"cd {Path(self.path).parent} && test $(stat -c %s {name}.partial) = {self.written} && "
                  f"printf '%s  %s\\n' {self.digest.hexdigest()} {name} > {name}.sha256 && "
                  f"cat > {name}.receipt.json && mv {name}.partial {name} && sha256sum -c {name}.sha256 --quiet && "
                  f"echo {self.host} {name} verified")
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, script], input=json.dumps(receipt, indent=1).encode(),
                                capture_output=True)
        if result.returncode != 0:
            raise PackFailure(f"{self.host}: verify failed: {result.stderr.decode()[-400:]}")
        print(result.stdout.decode().strip(), flush=True)


class LocalSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.digest = hashlib.sha256()
        self.written = 0
        self.handle = open(f"{self.path}.partial", "wb")

    write = Sink.write
    pad_to = Sink.pad_to

    def stream(self):
        return self.handle

    def finish(self, receipt):
        self.handle.close()
        partial = Path(f"{self.path}.partial")
        if partial.stat().st_size != self.written:
            raise PackFailure(f"{partial}: {partial.stat().st_size} bytes on disk, {self.written} written")
        Path(f"{self.path}.sha256").write_text(f"{self.digest.hexdigest()}  {self.path.name}\n")
        Path(f"{self.path}.receipt.json").write_text(json.dumps(receipt, indent=1))
        partial.rename(self.path)
        print(f"{self.path} written sha256={self.digest.hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--node-dir", default="sparkdata/axk2.fp8.tp16")
    parser.add_argument("--ranks", default="0-15")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--local-dir")
    args = parser.parse_args()
    first, last = (int(value) for value in args.ranks.split("-"))
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).digest()
    source = Fp8SourceReader(Path(args.checkpoint))
    if source.config.get("architectures") != ["AXK2ForCausalLM"]:
        raise PackFailure(f"{args.checkpoint} is not an A.X K2 checkpoint")
    if contract["source"]["config_json"]["sha256"] != source.config_sha256:
        raise PackFailure("config.json does not match the contract")
    recipe = {"tool": "axk2_tp16_pack.py", "linear": "fp8 verbatim + F32 scales per 128-row block expanded",
              "experts": "fp8 verbatim + F32 scales", "kv_b": "fp8 decoded exactly to bf16", "tp_degree": TP_DEGREE,
              "revision": contract["revision"]}
    recipe_sha256 = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).digest()
    packers = []
    for rank in range(first, last + 1):
        packer = Axk2Packer(source, GEOMETRY, (0, GEOMETRY["layers"] - 1), tp_degree=TP_DEGREE, tp_rank=rank, expert_codec=CODEC_FP8)
        packer.build_plan()
        packer.file_bytes = packer.sequential_layout()
        packers.append(packer)
    for packer in packers:
        print(f"rank {packer.tp_rank}: {len(packer.plan)} tensors {packer.file_bytes} bytes", flush=True)
    if args.plan_only:
        return 0
    sinks = []
    for packer in packers:
        name = f"axk2.tp16.rank{packer.tp_rank:02d}.pack"
        sink = LocalSink(Path(args.local_dir) / name) if args.local_dir else Sink(packer.tp_rank, args.node_dir, name)
        sink.write(header_bytes(packer, packer.file_bytes, contract["revision"], contract_sha256,
                                bytes.fromhex(source.config_sha256), recipe_sha256))
        sinks.append(sink)
    for index in range(len(packers[0].plan)):
        for packer, sink in zip(packers, sinks):
            item = packer.plan[index]
            sink.pad_to(item.entry.payload_offset)
            for chunk in item.produce_payload():
                sink.write(chunk)
            if item.produce_scale is not None:
                sink.pad_to(item.entry.scale_offset)
                for chunk in item.produce_scale():
                    sink.write(chunk)
        if packers[0].plan[index].entry.kind == K_EXPERT_DOWN:
            print(f"layer {packers[0].plan[index].entry.layer} packed", flush=True)
    for packer, sink in zip(packers, sinks):
        sink.pad_to(packer.file_bytes)
        sink.finish({"recipe": recipe, "contract_sha256": contract_sha256.hex(), "file_bytes": packer.file_bytes,
                     "sha256": sink.digest.hexdigest(), "tp_rank": packer.tp_rank})
    print("PACK DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
