#!/usr/bin/env python3
"""Convert the Qwen/Qwen3.6-27B BF16 safetensors checkpoint into qwen36 stage packs.

Setup-time code, never the serving path: reads safetensors shard headers and
streams payloads into the wire format of
modules/qwen36_resident_decode_stage/source/spark_qwen36_stagepack_format.h.

The checkpoint layout was pinned against transformers modeling_qwen3_5 and
verified against the shard headers on disk:

  * GDN layers carry in_proj_qkv (ONE fused tensor, conv channel order
    q 2048 | k 2048 | v 6144), in_proj_z (the output gate), and SEPARATE
    in_proj_b / in_proj_a 48-row projections (beta and decay - the pack's
    GDN_BETA / GDN_DECAY kinds map to them directly, no split).
  * Attention q_proj is [12288, 5120]: each head's 512 rows are 256 query
    then 256 gate, the fused output gate. The pack copies it verbatim; the
    module consumes the fused layout.
  * A_log and dt_bias are BF16 in the checkpoint but F32 in the pack (the
    format header's natural format for both); the converter upcasts.
  * conv1d.weight is [10240, 1, 4]; the singleton dim drops, the payload is
    already contiguous [10240, 4].
  * The vision tower (model.visual.*) is out of scope by contract and never
    referenced; the MTP decoder layer reuses the per-layer kinds at the
    reserved MTP layer marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "model_contracts" / "qwen36_authoritative.json"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# Wire constants, mirroring spark_qwen36_stagepack_format.h. The round-trip
# test cross-checks these against the header so the two cannot drift apart.
MAGIC = 0x50533651  # 'Q6SP'
FORMAT_VERSION = 2
HEADER_BYTES = 112
ENTRY_BYTES = 56
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
PAYLOAD_ALIGNMENT = 256

WEIGHT_BF16 = 0
WEIGHT_F32 = 1

HIDDEN = 5120
LAYER_COUNT = 64
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_KEY_DIM = 128
GDN_HEAD_VALUE_DIM = 128
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 24
ATTN_KV_HEADS = 4
ATTN_HEAD_DIM = 256
ATTN_ROPE_DIM = 64
FFN_INTERMEDIATE = 17408
VOCAB = 248320
MTP_LAYERS = 1
MXFP4_GROUP = 32

GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM          # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_VALUE_DIM   # 6144
GDN_CONV_CHANNELS = 2 * GDN_QK_DIM + GDN_VALUE_DIM     # 10240
ATTN_Q_DIM = ATTN_QUERY_HEADS * ATTN_HEAD_DIM          # 6144
ATTN_KV_DIM = ATTN_KV_HEADS * ATTN_HEAD_DIM            # 1024

HEADER_STRUCT = struct.Struct("<24I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
assert HEADER_STRUCT.size == HEADER_BYTES and ENTRY_STRUCT.size == ENTRY_BYTES

# Tensor kinds, mirroring SparkQwen36StagePackTensorKind.
(KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_ATTENTION_NORM,
 KIND_MLP_NORM, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN, KIND_GDN_QKV,
 KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY, KIND_GDN_OUTPUT,
 KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG, KIND_GDN_DT_BIAS, KIND_GDN_NORM,
 KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE, KIND_ATTN_OUTPUT,
 KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM, KIND_MTP_FC, KIND_MTP_EMBED_NORM,
 KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM) = range(27)

CHUNK_BYTES = 8 * 1024 * 1024
BF16_BYTES = 2
F32_BYTES = 4

LANGUAGE_PREFIX = "model.language_model."
MTP_PREFIX = "mtp."


class PackFailure(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def align(offset: int) -> int:
    return (offset + PAYLOAD_ALIGNMENT - 1) & ~(PAYLOAD_ALIGNMENT - 1)


def is_gdn_layer(layer_index: int) -> bool:
    return (layer_index % ATTENTION_PERIOD) != FULL_PHASE


# (rows, columns, weight_format) per kind, mirroring the Shape* functions in
# the format header. Norms are 1 x width rows; only A_log and dt_bias are f32.
def kind_shape(kind: int) -> tuple[int, int, int]:
    table = {
        KIND_EMBEDDING: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_LM_HEAD: (VOCAB, HIDDEN, WEIGHT_BF16),
        KIND_ATTENTION_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MLP_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_FFN_GATE: (FFN_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_FFN_UP: (FFN_INTERMEDIATE, HIDDEN, WEIGHT_BF16),
        KIND_FFN_DOWN: (HIDDEN, FFN_INTERMEDIATE, WEIGHT_BF16),
        KIND_GDN_QKV: (GDN_CONV_CHANNELS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_GATE: (GDN_VALUE_DIM, HIDDEN, WEIGHT_BF16),
        KIND_GDN_BETA: (GDN_VALUE_HEADS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_DECAY: (GDN_VALUE_HEADS, HIDDEN, WEIGHT_BF16),
        KIND_GDN_OUTPUT: (HIDDEN, GDN_VALUE_DIM, WEIGHT_BF16),
        KIND_GDN_CONV_WEIGHT: (GDN_CONV_CHANNELS, GDN_CONV_KERNEL, WEIGHT_BF16),
        KIND_GDN_A_LOG: (1, GDN_VALUE_HEADS, WEIGHT_F32),
        KIND_GDN_DT_BIAS: (1, GDN_VALUE_HEADS, WEIGHT_F32),
        KIND_GDN_NORM: (1, GDN_HEAD_VALUE_DIM, WEIGHT_BF16),
        KIND_ATTN_QUERY: (2 * ATTN_Q_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_KEY: (ATTN_KV_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_VALUE: (ATTN_KV_DIM, HIDDEN, WEIGHT_BF16),
        KIND_ATTN_OUTPUT: (HIDDEN, ATTN_Q_DIM, WEIGHT_BF16),
        KIND_ATTN_QUERY_NORM: (1, ATTN_HEAD_DIM, WEIGHT_BF16),
        KIND_ATTN_KEY_NORM: (1, ATTN_HEAD_DIM, WEIGHT_BF16),
        KIND_MTP_FC: (HIDDEN, 2 * HIDDEN, WEIGHT_BF16),
        KIND_MTP_EMBED_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MTP_HIDDEN_NORM: (1, HIDDEN, WEIGHT_BF16),
        KIND_MTP_FINAL_NORM: (1, HIDDEN, WEIGHT_BF16),
    }
    return table[kind]


# Checkpoint tensor name for a kind at a layer, or None for globals (named in
# the inventory builder). layer is a stack index or the MTP marker.
def layer_tensor_name(kind: int, layer: int) -> str:
    prefix = (LANGUAGE_PREFIX + f"layers.{layer}.") if layer != MTP_LAYER else (MTP_PREFIX + "layers.0.")
    gdn = {
        KIND_GDN_QKV: "linear_attn.in_proj_qkv.weight",
        KIND_GDN_GATE: "linear_attn.in_proj_z.weight",
        KIND_GDN_BETA: "linear_attn.in_proj_b.weight",
        KIND_GDN_DECAY: "linear_attn.in_proj_a.weight",
        KIND_GDN_OUTPUT: "linear_attn.out_proj.weight",
        KIND_GDN_CONV_WEIGHT: "linear_attn.conv1d.weight",
        KIND_GDN_A_LOG: "linear_attn.A_log",
        KIND_GDN_DT_BIAS: "linear_attn.dt_bias",
        KIND_GDN_NORM: "linear_attn.norm.weight",
    }
    attn = {
        KIND_ATTN_QUERY: "self_attn.q_proj.weight",
        KIND_ATTN_KEY: "self_attn.k_proj.weight",
        KIND_ATTN_VALUE: "self_attn.v_proj.weight",
        KIND_ATTN_OUTPUT: "self_attn.o_proj.weight",
        KIND_ATTN_QUERY_NORM: "self_attn.q_norm.weight",
        KIND_ATTN_KEY_NORM: "self_attn.k_norm.weight",
    }
    every = {
        KIND_ATTENTION_NORM: "input_layernorm.weight",
        KIND_MLP_NORM: "post_attention_layernorm.weight",
        KIND_FFN_GATE: "mlp.gate_proj.weight",
        KIND_FFN_UP: "mlp.up_proj.weight",
        KIND_FFN_DOWN: "mlp.down_proj.weight",
    }
    for mapping in (every, gdn, attn):
        if kind in mapping:
            return prefix + mapping[kind]
    raise PackFailure(f"kind {kind} is not a per-layer tensor")


GLOBAL_TENSORS = {
    KIND_EMBEDDING: LANGUAGE_PREFIX + "embed_tokens.weight",
    KIND_FINAL_NORM: LANGUAGE_PREFIX + "norm.weight",
    KIND_LM_HEAD: "lm_head.weight",
    KIND_MTP_FC: MTP_PREFIX + "fc.weight",
    KIND_MTP_EMBED_NORM: MTP_PREFIX + "pre_fc_norm_embedding.weight",
    KIND_MTP_HIDDEN_NORM: MTP_PREFIX + "pre_fc_norm_hidden.weight",
    KIND_MTP_FINAL_NORM: MTP_PREFIX + "norm.weight",
}

EVERY_LAYER_KINDS = (KIND_ATTENTION_NORM, KIND_MLP_NORM, KIND_FFN_GATE,
                     KIND_FFN_UP, KIND_FFN_DOWN)
GDN_LAYER_KINDS = (KIND_GDN_QKV, KIND_GDN_GATE, KIND_GDN_BETA, KIND_GDN_DECAY,
                   KIND_GDN_OUTPUT, KIND_GDN_CONV_WEIGHT, KIND_GDN_A_LOG,
                   KIND_GDN_DT_BIAS, KIND_GDN_NORM)
ATTN_LAYER_KINDS = (KIND_ATTN_QUERY, KIND_ATTN_KEY, KIND_ATTN_VALUE,
                    KIND_ATTN_OUTPUT, KIND_ATTN_QUERY_NORM, KIND_ATTN_KEY_NORM)


class TensorRef:
    """One pack tensor: its kind, layer marker, and checkpoint source."""

    def __init__(self, kind: int, layer: int, name: str):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows, self.columns, self.weight_format = kind_shape(kind)


def build_inventory(first_layer: int, layer_count: int) -> list[TensorRef]:
    """The slice's tensor list, in the synthesizer's emission order."""
    if layer_count == 0 or first_layer + layer_count > LAYER_COUNT:
        raise PackFailure(f"invalid slice {first_layer}+{layer_count} of {LAYER_COUNT}")
    refs: list[TensorRef] = []
    if first_layer == 0:
        refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
    for layer in range(first_layer, first_layer + layer_count):
        class_kinds = GDN_LAYER_KINDS if is_gdn_layer(layer) else ATTN_LAYER_KINDS
        for kind in EVERY_LAYER_KINDS + class_kinds:
            refs.append(TensorRef(kind, layer, layer_tensor_name(kind, layer)))
    if first_layer + layer_count == LAYER_COUNT:
        # The head stage's MTP chain embeds its own draft tokens and the
        # vocabulary is untied: a second embedding copy unless this pack
        # already carries it as the stage-zero global.
        if first_layer != 0:
            refs.append(TensorRef(KIND_EMBEDDING, GLOBAL_LAYER, GLOBAL_TENSORS[KIND_EMBEDDING]))
        for kind in (KIND_FINAL_NORM, KIND_LM_HEAD, KIND_MTP_FC,
                     KIND_MTP_EMBED_NORM, KIND_MTP_HIDDEN_NORM, KIND_MTP_FINAL_NORM):
            refs.append(TensorRef(kind, GLOBAL_LAYER, GLOBAL_TENSORS[kind]))
        for kind in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
            refs.append(TensorRef(kind, MTP_LAYER, layer_tensor_name(kind, MTP_LAYER)))
    expected = expected_tensor_count(first_layer, layer_count)
    if len(refs) != expected:
        raise PackFailure(f"inventory {len(refs)} tensors, format expects {expected}")
    return refs


def expected_tensor_count(first_layer: int, layer_count: int) -> int:
    """SparkQwen36StagePackExpectedTensorCount, restated."""
    full_below = lambda n: n // ATTENTION_PERIOD
    full = full_below(first_layer + layer_count) - full_below(first_layer)
    gdn = layer_count - full
    tensors = layer_count * 5 + gdn * 9 + full * 6
    if first_layer == 0:
        tensors += 1
    if first_layer + layer_count == LAYER_COUNT:
        tensors += 2 + 4 + 11 + (1 if first_layer != 0 else 0)
    return tensors


class SafetensorsSource:
    """The checkpoint's shards: index, per-shard headers, payload streams."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        index_path = checkpoint / INDEX_NAME
        config_path = checkpoint / CONFIG_NAME
        if not index_path.is_file():
            raise PackFailure(f"missing {index_path}")
        if not config_path.is_file():
            raise PackFailure(f"missing {config_path}")
        self.index_sha256 = sha256_file(index_path)
        self.config_sha256 = sha256_file(config_path)
        index = json.loads(index_path.read_text())
        self.weight_map = index["weight_map"]
        self.config = json.loads(config_path.read_text())
        self.headers: dict[str, dict] = {}
        self.data_start: dict[str, int] = {}

    def check_config(self) -> None:
        text = self.config.get("text_config", {})
        expectations = {
            "hidden_size": HIDDEN, "num_hidden_layers": LAYER_COUNT,
            "num_attention_heads": ATTN_QUERY_HEADS,
            "num_key_value_heads": ATTN_KV_HEADS, "head_dim": ATTN_HEAD_DIM,
            "linear_num_key_heads": GDN_KEY_HEADS,
            "linear_num_value_heads": GDN_VALUE_HEADS,
            "linear_key_head_dim": GDN_HEAD_KEY_DIM,
            "linear_value_head_dim": GDN_HEAD_VALUE_DIM,
            "linear_conv_kernel_dim": GDN_CONV_KERNEL,
            "intermediate_size": FFN_INTERMEDIATE, "vocab_size": VOCAB,
            "full_attention_interval": ATTENTION_PERIOD,
            "attn_output_gate": True, "tie_word_embeddings": False,
        }
        for key, expected in expectations.items():
            if text.get(key) != expected:
                raise PackFailure(
                    f"config.json text_config.{key}={text.get(key)!r}, expected {expected!r} "
                    "- this is not the checkpoint this packer is for")
        layer_types = text.get("layer_types", [])
        if len(layer_types) == LAYER_COUNT:
            for layer, layer_type in enumerate(layer_types):
                want = "linear_attention" if is_gdn_layer(layer) else "full_attention"
                if layer_type != want:
                    raise PackFailure(f"config layer_types[{layer}]={layer_type!r}, expected {want!r}")

    def shard_header(self, shard: str) -> dict:
        if shard not in self.headers:
            path = self.checkpoint / shard
            if not path.is_file():
                raise PackFailure(f"missing shard {shard}")
            with path.open("rb") as file:
                header_bytes = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(header_bytes))
            self.headers[shard] = header
            self.data_start[shard] = 8 + header_bytes
        return self.headers[shard]

    def resolve(self, name: str) -> tuple[str, dict, int]:
        """(shard, metadata, absolute payload offset) for a tensor."""
        if name not in self.weight_map:
            raise PackFailure(f"tensor not in checkpoint index: {name}")
        shard = self.weight_map[name]
        header = self.shard_header(shard)
        if name not in header:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        meta = header[name]
        return shard, meta, self.data_start[shard] + meta["data_offsets"][0]

    def check_shape(self, ref: TensorRef) -> tuple[str, dict, int]:
        shard, meta, offset = self.resolve(ref.name)
        if meta["dtype"] != "BF16":
            raise PackFailure(f"{ref.name}: dtype {meta['dtype']}, expected BF16")
        shape = meta["shape"]
        # The conv weight arrives [channels, 1, kernel]; the singleton drops.
        if len(shape) == 3 and shape[1] == 1:
            shape = [shape[0], shape[2]]
        if len(shape) == 1:
            shape = [1, shape[0]]
        if shape != [ref.rows, ref.columns]:
            raise PackFailure(
                f"{ref.name}: checkpoint shape {meta['shape']}, pack expects "
                f"[{ref.rows}, {ref.columns}] for kind {ref.kind}")
        return shard, meta, offset


def copy_tensor(source: SafetensorsSource, ref: TensorRef, offset: int, out) -> None:
    """Stream one tensor's payload, upcasting BF16 to F32 where the pack says."""
    path = source.checkpoint / source.weight_map[ref.name]
    elements = ref.rows * ref.columns
    source_bytes = elements * BF16_BYTES
    with path.open("rb") as file:
        file.seek(offset)
        remaining = source_bytes
        while remaining > 0:
            step = min(remaining, CHUNK_BYTES)
            chunk = file.read(step)
            if len(chunk) != step:
                raise PackFailure(f"short read on {ref.name}")
            remaining -= step
            if ref.weight_format == WEIGHT_BF16:
                out.write(chunk)
            else:
                # BF16 is the top half of F32: shift into place, zero the tail.
                widened = bytearray(step * 2)
                widened[2::4] = chunk[0::2]
                widened[3::4] = chunk[1::2]
                out.write(widened)


def convert(checkpoint: Path, output: Path, first_layer: int, layer_count: int,
            receipt: dict, dry_run: bool) -> dict:
    source = SafetensorsSource(checkpoint)
    source.check_config()
    refs = build_inventory(first_layer, layer_count)
    plans = []
    cursor = 0
    for ref in refs:
        shard, meta, offset = source.check_shape(ref)
        payload_offset = align(cursor)
        element_bytes = BF16_BYTES if ref.weight_format == WEIGHT_BF16 else F32_BYTES
        payload_bytes = ref.rows * ref.columns * element_bytes
        plans.append((ref, offset, payload_offset, payload_bytes))
        cursor = payload_offset + payload_bytes
    payload_base = align(HEADER_BYTES + len(plans) * ENTRY_BYTES)
    file_bytes = payload_base + cursor

    header = HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, len(plans),
        HIDDEN, layer_count, first_layer, LAYER_COUNT,
        ATTENTION_PERIOD, FULL_PHASE,
        GDN_KEY_HEADS, GDN_VALUE_HEADS, GDN_HEAD_KEY_DIM, GDN_HEAD_VALUE_DIM,
        GDN_CONV_KERNEL, ATTN_QUERY_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM,
        ATTN_ROPE_DIM, FFN_INTERMEDIATE, VOCAB, MXFP4_GROUP, MTP_LAYERS,
        HEADER_BYTES, file_bytes)
    entries = b"".join(
        ENTRY_STRUCT.pack(ref.kind, ref.layer, ref.weight_format, ref.rows,
                          ref.columns, 0, payload_base + payload_offset,
                          payload_bytes, 0, 0)
        for ref, _, payload_offset, payload_bytes in plans)
    receipt.update({
        "first_layer_index": first_layer,
        "layer_count": layer_count,
        "tensor_count": len(plans),
        "bytes": file_bytes,
        "source_index_sha256": source.index_sha256,
        "source_config_sha256": source.config_sha256,
    })
    if dry_run:
        print(f"qwen36_stagepack slice={first_layer}+{layer_count} "
              f"tensors={len(plans)} file_bytes={file_bytes} "
              f"file_gib={file_bytes / 2**30:.2f} (dry run)")
        return receipt

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp",
                                     dir=output.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(header)
        temp.write(entries)
        padding = payload_base - temp.tell()
        if padding < 0:
            raise PackFailure("directory overruns the payload base")
        temp.write(b"\0" * padding)
        for ref, source_offset, _, payload_bytes in plans:
            before = temp.tell()
            copy_tensor(source, ref, source_offset, temp)
            if temp.tell() - before != payload_bytes:
                raise PackFailure(f"payload size mismatch on {ref.name}")
            pad = align(temp.tell()) - temp.tell()
            if pad:
                temp.write(b"\0" * pad)
        temp.flush()
        os.fsync(temp.fileno())
    os.replace(temp_path, output)
    receipt["output_sha256"] = sha256_file(output)
    receipt["file"] = str(output)
    print(f"qwen36_stagepack slice={first_layer}+{layer_count} tensors={len(plans)} "
          f"file_gib={file_bytes / 2**30:.2f} wrote {output}")
    return receipt


def verify(pack_path: Path) -> dict:
    """Parse a pack back and check every rule the format header states."""
    file_bytes = pack_path.stat().st_size
    with pack_path.open("rb") as file:
        raw_header = file.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise PackFailure("short header")
        fields = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, period, phase,
         gdn_kh, gdn_vh, gdn_kd, gdn_vd, conv_kernel,
         attn_qh, attn_kvh, attn_hd, rope_dim, ffn, vocab, mxfp4, mtp,
         directory_offset, declared_bytes) = fields
        geometry = {
            "magic": (magic, MAGIC), "format_version": (version, FORMAT_VERSION),
            "header_bytes": (header_bytes, HEADER_BYTES),
            "directory_entry_bytes": (entry_bytes, ENTRY_BYTES),
            "tensor_count": (tensor_count, expected_tensor_count(first_layer, layer_count)),
            "hidden_dimension": (hidden, HIDDEN), "layer_count": (layer_count, layer_count),
            "total_layer_count": (total_layers, LAYER_COUNT),
            "attention_period": (period, ATTENTION_PERIOD),
            "full_attention_phase": (phase, FULL_PHASE),
            "gdn_key_head_count": (gdn_kh, GDN_KEY_HEADS),
            "gdn_value_head_count": (gdn_vh, GDN_VALUE_HEADS),
            "gdn_head_key_dimension": (gdn_kd, GDN_HEAD_KEY_DIM),
            "gdn_head_value_dimension": (gdn_vd, GDN_HEAD_VALUE_DIM),
            "gdn_conv_kernel": (conv_kernel, GDN_CONV_KERNEL),
            "attn_query_head_count": (attn_qh, ATTN_QUERY_HEADS),
            "attn_kv_head_count": (attn_kvh, ATTN_KV_HEADS),
            "attn_head_dimension": (attn_hd, ATTN_HEAD_DIM),
            "attn_rope_dimension": (rope_dim, ATTN_ROPE_DIM),
            "ffn_intermediate_dimension": (ffn, FFN_INTERMEDIATE),
            "output_vocab_count": (vocab, VOCAB),
            "mxfp4_group_size": (mxfp4, MXFP4_GROUP), "mtp_layer_count": (mtp, MTP_LAYERS),
        }
        for name, (actual, expected) in geometry.items():
            if actual != expected:
                raise PackFailure(f"geometry field {name}: {actual}, expected {expected}")
        if directory_offset != HEADER_BYTES or declared_bytes != file_bytes:
            raise PackFailure("directory offset or file size mismatch")
        raw_directory = file.read(tensor_count * ENTRY_BYTES)
        if len(raw_directory) != tensor_count * ENTRY_BYTES:
            raise PackFailure("short directory")
        payload_base = align(HEADER_BYTES + tensor_count * ENTRY_BYTES)
        cursor = payload_base
        seen = set()
        for index in range(tensor_count):
            entry = ENTRY_STRUCT.unpack_from(raw_directory, index * ENTRY_BYTES)
            (kind, layer, weight_format, rows, columns, scale_group,
             payload_offset, payload_bytes, scale_offset, scale_bytes) = entry
            expected_rows, expected_columns, expected_format = kind_shape(kind)
            if (rows, columns, weight_format) != (expected_rows, expected_columns, expected_format):
                raise PackFailure(f"entry {index} kind {kind}: shape or format mismatch")
            if (kind, layer) in seen:
                raise PackFailure(f"duplicate tensor kind {kind} layer {layer}")
            seen.add((kind, layer))
            if layer == GLOBAL_LAYER:
                if kind not in GLOBAL_TENSORS:
                    raise PackFailure(f"entry {index}: per-layer kind {kind} at the global marker")
            elif layer == MTP_LAYER:
                if kind not in EVERY_LAYER_KINDS + ATTN_LAYER_KINDS:
                    raise PackFailure(f"entry {index}: kind {kind} not valid at the MTP marker")
            else:
                if layer >= LAYER_COUNT:
                    raise PackFailure(f"entry {index}: layer {layer} out of range")
                if kind in GDN_LAYER_KINDS and not is_gdn_layer(layer):
                    raise PackFailure(f"entry {index}: GDN kind {kind} on attention layer {layer}")
                if kind in ATTN_LAYER_KINDS and is_gdn_layer(layer):
                    raise PackFailure(f"entry {index}: attention kind {kind} on GDN layer {layer}")
            if payload_offset != align(cursor) or payload_offset % PAYLOAD_ALIGNMENT != 0:
                raise PackFailure(f"entry {index}: payload offset {payload_offset}, expected {align(cursor)}")
            element_bytes = BF16_BYTES if weight_format == WEIGHT_BF16 else F32_BYTES
            if payload_bytes != rows * columns * element_bytes:
                raise PackFailure(f"entry {index}: payload byte count mismatch")
            if scale_group != 0 or scale_offset != 0 or scale_bytes != 0:
                raise PackFailure(f"entry {index}: BF16/F32 tensors carry no scales")
            cursor = payload_offset + payload_bytes
        if align(cursor) != align(file_bytes):
            raise PackFailure("trailing payload does not close the file")
    return {"file": str(pack_path), "bytes": file_bytes, "tensor_count": tensor_count,
            "first_layer_index": first_layer, "layer_count": layer_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint directory")
    parser.add_argument("--output", type=Path, help="pack output path")
    parser.add_argument("--stage-index", type=int, help="stage index into --recipe")
    parser.add_argument("--recipe", type=Path, help="PP recipe JSON (layer split + receipt hash)")
    parser.add_argument("--first-layer", type=int, help="explicit slice start")
    parser.add_argument("--layer-count", type=int, help="explicit slice length")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--receipt", type=Path, help="receipt output (default: <output>.receipt.json)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path, help="verify an existing pack and exit")
    args = parser.parse_args()

    if args.verify is not None:
        result = verify(args.verify)
        print(f"qwen36_stagepack verify ok: {result['file']} "
              f"slice={result['first_layer_index']}+{result['layer_count']} "
              f"tensors={result['tensor_count']} bytes={result['bytes']}")
        return 0

    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.stage_index is not None:
        if args.recipe is None:
            parser.error("--stage-index requires --recipe")
        recipe = json.loads(args.recipe.read_text())
        stage = recipe["pp"]["stages"][args.stage_index]
        first_layer, layer_count = stage["first_layer_index"], stage["layer_count"]
    elif args.first_layer is not None and args.layer_count is not None:
        first_layer, layer_count = args.first_layer, args.layer_count
    else:
        parser.error("name --stage-index with --recipe, or --first-layer with --layer-count")
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run")

    receipt = {
        "kind": "sparkpipe.qwen36.stagepack-receipt.v1",
        "tool": "tools/qwen36_stagepack.py",
        "checkpoint": str(args.checkpoint),
        "contract": {"path": str(args.contract),
                     "sha256": sha256_file(args.contract) if args.contract.is_file() else None},
        "recipe": None,
        "stage_index": args.stage_index,
        "weight_formats": {"projections": "bf16", "gdn_a_log_dt_bias": "f32"},
    }
    if args.recipe is not None:
        receipt["recipe"] = {"path": str(args.recipe),
                             "sha256": sha256_file(args.recipe),
                             "content_hash": recipe.get("content_hash")}
    result = convert(args.checkpoint, args.output or Path("/dev/null"),
                     first_layer, layer_count, receipt, args.dry_run)
    if not args.dry_run:
        receipt_path = args.receipt or Path(str(args.output) + ".receipt.json")
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"qwen36_stagepack receipt {receipt_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen36_stagepack: {error}", file=sys.stderr)
        sys.exit(1)
