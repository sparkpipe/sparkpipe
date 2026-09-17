#!/usr/bin/env python3
"""Generate independent DeepSeek V4 GA stage-boundary reference vectors.

This tool executes the checkpoint's reference model, not SparkPipe code. It
loads only the embedding and the first three transformer layers from the
Hugging Face safetensor shards and writes the final BF16 Hyper-Connection
boundary vector. A literal PyTorch FP8 GEMM replaces the bundled TileLang GEMM,
which is nondeterministic on GB10 for identical inputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from safetensors import safe_open
from torch import nn


CHECKPOINT_INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT_CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
CHECKPOINT_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
CHECKPOINT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
REFERENCE_BATCH_JSON_SHA256 = "6f7836819a9ecdbca117b18cb4717aa8cb91c230af5961c5d025968cef34f8bb"
REFERENCE_TOKEN_PAYLOAD_SHA256 = "f2f860f7843e755c4cdfcea408c647559ab604fde5c34a00bac314ba62289769"
REFERENCE_MODEL_SHA256 = "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
REFERENCE_KERNEL_SHA256 = "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"
REFERENCE_CONFIG_SHA256 = "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71"
REFERENCE_FIRST_LAYER = 0
REFERENCE_LAYER_COUNT = 3
REFERENCE_PROMPT_TOKENS = 128
REFERENCE_VOCABULARY_SIZE = 129280
REFERENCE_SOURCE_SHARDS = {
    "model-00001-of-00048.safetensors": (
        1059061856,
        "f3668ba4cccf1ca6a7eb84e888fb92c1cdc7204d472ba9db771e6fd3abf6b874",
    ),
    "model-00002-of-00048.safetensors": (
        3566321192,
        "77b26c939a0e25b3113c8d6bb04e1901a748bd4a7d2589e3bfdaabdf1e9bba14",
    ),
    "model-00003-of-00048.safetensors": (
        3566321192,
        "412abf4c906faadc221ef0cb50f90fe20bde8454a08ad4dc2364b6b79e7fda5c",
    ),
    "model-00004-of-00048.safetensors": (
        3596229272,
        "9610f56bc587fb0ff9a8b68a60299482ee8c433fe5b5587e4257aca98add4a2e",
    ),
}


class VectorError(RuntimeError):
    """A deterministic reference-vector generation failure."""


@dataclasses.dataclass(frozen=True)
class SourceTensor:
    name: str
    shard: Path


@dataclasses.dataclass(frozen=True)
class SourceBatch:
    token_rows: list[list[int]]
    request_id: int
    sequence_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-json",
        type=Path,
        required=True,
        help="SparkPipe schema-v1 B1 batch supplying the exact prompt token IDs",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_module(model_dir: Path) -> Any:
    inference_dir = model_dir / "inference"
    model_path = inference_dir / "model.py"
    if "kernel" in sys.modules:
        raise VectorError("checkpoint kernel module was imported before the reference model")
    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location("dsv4_ga_reference_model", model_path)
    if spec is None or spec.loader is None:
        raise VectorError(f"cannot import reference implementation: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = sys.modules.get("kernel")
    kernel_path = getattr(kernel, "__file__", None)
    if kernel_path is None or Path(kernel_path).resolve() != inference_dir / "kernel.py":
        raise VectorError("reference model imported a kernel outside the checkpoint")
    return module


def load_config(model_dir: Path, reference: Any, batch_size: int, sequence_length: int) -> Any:
    config_path = model_dir / "inference" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_batch_size"] = batch_size
    config["max_seq_len"] = max(128, sequence_length)
    return reference.ModelArgs(**config)


class ReferenceStage(nn.Module):
    def __init__(self, reference: Any, args: Any, first_layer: int, layer_count: int):
        super().__init__()
        self.first_layer = first_layer
        self.hc_mult = args.hc_mult
        self.embed = reference.ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleDict(
            {
                str(layer): reference.Block(layer, args)
                for layer in range(first_layer, first_layer + layer_count)
            }
        )

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.embed(token_ids)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        outputs = []
        for layer_id, layer in self.layers.items():
            hidden = layer(hidden, 0, token_ids)
            outputs.append(hidden.detach().clone())
        return outputs


def configure_reference(reference: Any, args: Any) -> None:
    reference.world_size = 1
    reference.rank = 0
    reference.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    reference.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    reference.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    reference.sparse_attn = sparse_attention_reference
    reference.rotate_activation = hadamard_reference
    reference.fp8_gemm = fp8_gemm_reference


def fp8_gemm_reference(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Literal block-scaled FP8 GEMM with deterministic FP32 accumulation."""
    del scale_dtype
    if not activation.is_contiguous() or not weight.is_contiguous():
        raise VectorError("FP8 GEMM inputs must be contiguous")
    inner = activation.shape[-1]
    rows = activation.numel() // inner
    outputs = weight.shape[0]
    if inner % 128 != 0 or weight.shape[1] != inner:
        raise VectorError("FP8 GEMM inner dimension must be a matching multiple of 128")
    block_count = inner // 128
    activation_2d = activation.view(rows, inner)
    activation_scale_2d = activation_scale.view(rows, block_count)
    if tuple(weight_scale.shape) != ((outputs + 127) // 128, block_count):
        raise VectorError("FP8 GEMM weight scale shape does not match the weight")
    result = torch.zeros(
        (rows, outputs), device=activation.device, dtype=torch.float32
    )
    expanded_weight_scale = weight_scale.float().repeat_interleave(128, dim=0)[
        :outputs
    ]
    for block in range(block_count):
        start = block * 128
        stop = start + 128
        partial = torch.matmul(
            activation_2d[:, start:stop].float(),
            weight[:, start:stop].float().t(),
        )
        partial *= activation_scale_2d[:, block].float().unsqueeze(1)
        partial *= expanded_weight_scale[:, block].unsqueeze(0)
        result += partial
    return result.to(torch.get_default_dtype()).view(*activation.shape[:-1], outputs)


def hadamard_reference(value: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester Hadamard transform used by the GA indexer."""
    width = value.shape[-1]
    if width < 1 or width & (width - 1):
        raise VectorError(f"Hadamard width must be a power of two, got {width}")
    output = value.float()
    stride = 1
    while stride < width:
        shaped = output.reshape(*output.shape[:-1], -1, stride * 2)
        lower = shaped[..., :stride]
        upper = shaped[..., stride:]
        output = torch.cat((lower + upper, lower - upper), dim=-1).flatten(-2)
        stride *= 2
    return (output * (width**-0.5)).to(value.dtype)


def validate_hadamard_reference() -> None:
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    expected = torch.tensor([[5.0, -1.0, -2.0, 0.0]], dtype=torch.bfloat16)
    if not torch.equal(hadamard_reference(value), expected):
        raise VectorError("Hadamard fallback failed the fixed H4 reference vector")


def sparse_attention_reference(
    query: torch.Tensor,
    key_value: torch.Tensor,
    attention_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Mathematical sparse-attention reference for the bundled kernel.

    The checkpoint TileLang kernel asks for 141312 bytes of dynamic shared
    memory at the GA shape, which GB10 cannot admit. This global-softmax
    fallback preserves the model formula, FP32 scores, and BF16 probability
    inputs while remaining independent from SparkPipe CUDA. It is not a
    bitwise transcription of TileLang's blockwise online softmax.
    """
    batch, sequence, _, _ = query.shape
    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp_min(0).long()
    batch_indices = torch.arange(batch, device=query.device).view(batch, 1, 1)
    gathered = key_value[batch_indices, safe_indices]
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), gathered.float()
    ) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attention_sink.float().view(1, 1, -1)
    maximum = torch.maximum(scores.amax(dim=-1), sink)
    weights = torch.exp(scores - maximum.unsqueeze(-1))
    weights = weights.masked_fill(~valid.unsqueeze(2), 0.0)
    denominator = weights.sum(dim=-1) + torch.exp(sink - maximum)
    output = torch.einsum(
        "bshk,bskd->bshd", weights.bfloat16().float(), gathered.float()
    )
    return (output / denominator.unsqueeze(-1)).to(query.dtype)


def weight_map(model_dir: Path) -> tuple[Path, dict[str, SourceTensor]]:
    index_path = model_dir / "model.safetensors.index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = document.get("weight_map")
    if not isinstance(mapping, dict):
        raise VectorError("checkpoint index has no weight_map")
    return index_path, {
        name: SourceTensor(name, model_dir / shard)
        for name, shard in mapping.items()
    }


def read_tensor(source: SourceTensor) -> torch.Tensor:
    with safe_open(source.shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(source.name)


def checkpoint_name(stage_name: str) -> str:
    if stage_name.startswith("embed."):
        return stage_name
    prefix = "layers."
    if not stage_name.startswith(prefix):
        raise VectorError(f"unexpected stage parameter: {stage_name}")
    relative = stage_name[len(prefix) :]
    layer_text, separator, suffix = relative.partition(".")
    if not separator:
        raise VectorError(f"malformed stage parameter: {stage_name}")
    return f"layers.{int(layer_text)}.{suffix}"


def dequantize_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.float8_e4m3fn:
        raise VectorError(f"wo_a weight has unexpected dtype {weight.dtype}")
    out_blocks = weight.shape[0] // 128
    in_blocks = weight.shape[1] // 128
    if tuple(scale.shape) != (out_blocks, in_blocks):
        raise VectorError(f"wo_a scale shape {tuple(scale.shape)} does not match weight {tuple(weight.shape)}")
    blocks = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
    blocks = blocks.float() * scale[:, None, :, None].float()
    return blocks.flatten(2, 3).flatten(0, 1).bfloat16()


def converted_tensor(
    name: str, sources: dict[str, SourceTensor], consumed: set[str]
) -> torch.Tensor:
    source = sources.get(name)
    if source is None:
        raise VectorError(f"checkpoint tensor missing: {name}")
    consumed.add(name)
    tensor = read_tensor(source)
    if name.endswith(".attn.wo_a.weight"):
        scale_name = name.removesuffix("weight") + "scale"
        scale_source = sources.get(scale_name)
        if scale_source is None:
            raise VectorError(f"checkpoint tensor missing: {scale_name}")
        consumed.add(scale_name)
        return dequantize_wo_a(tensor, read_tensor(scale_source))
    if ".ffn.experts." in name and name.endswith(".weight"):
        if tensor.dtype != torch.int8:
            raise VectorError(f"expert tensor {name} has unexpected dtype {tensor.dtype}")
        return tensor.view(torch.float4_e2m1fn_x2)
    return tensor


def load_stage(
    stage: ReferenceStage, sources: dict[str, SourceTensor]
) -> tuple[list[str], set[str]]:
    loaded = []
    consumed: set[str] = set()
    with torch.no_grad():
        for stage_name, parameter in stage.named_parameters():
            name = checkpoint_name(stage_name)
            tensor = converted_tensor(name, sources, consumed)
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise VectorError(
                    f"shape mismatch for {name}: checkpoint={tuple(tensor.shape)} "
                    f"reference={tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.append(name)
    return loaded, consumed


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu().view(torch.uint8)
    return value.numpy().tobytes()


def tensor_record(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    raw = tensor_bytes(tensor)
    path.write_bytes(raw)
    value = tensor.float()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "l2": torch.linalg.vector_norm(value).item(),
        "nonfinite": int((~torch.isfinite(value)).sum().item()),
    }


def token_record(path: Path, token_rows: list[list[int]]) -> dict[str, Any]:
    raw = b"".join(struct.pack("<I", token) for row in token_rows for token in row)
    path.write_bytes(raw)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "dtype": "uint32_le",
        "shape": [len(token_rows), len(token_rows[0])],
    }


def load_source_batch(batch_json: Path) -> SourceBatch:
    document = json.loads(batch_json.read_text(encoding="utf-8"))
    requests = document.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise VectorError("--batch-json must contain exactly one request")
    if not isinstance(requests[0], dict):
        raise VectorError("--batch-json request must be an object")
    request = requests[0]
    token_ids = request.get("prompt_token_ids")
    if not isinstance(token_ids, list) or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= REFERENCE_VOCABULARY_SIZE
        for token in token_ids
    ):
        raise VectorError("--batch-json request has invalid prompt_token_ids")
    if len(token_ids) < REFERENCE_PROMPT_TOKENS:
        raise VectorError(
            f"prompt has {len(token_ids)} tokens, cannot select {REFERENCE_PROMPT_TOKENS}"
        )
    request_id = request.get("request_id")
    sequence_id = request.get("sequence_id")
    for label, value in (("request_id", request_id), ("sequence_id", sequence_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VectorError(f"--batch-json request has invalid {label}")
    return SourceBatch(
        [token_ids[:REFERENCE_PROMPT_TOKENS]], request_id, sequence_id
    )


def require_checkpoint_identity(model_dir: Path) -> list[dict[str, Any]]:
    expected = {
        model_dir / "model.safetensors.index.json": CHECKPOINT_INDEX_SHA256,
        model_dir / "config.json": CHECKPOINT_CONFIG_SHA256,
        model_dir / "tokenizer.json": CHECKPOINT_TOKENIZER_SHA256,
        model_dir / "inference" / "model.py": REFERENCE_MODEL_SHA256,
        model_dir / "inference" / "kernel.py": REFERENCE_KERNEL_SHA256,
        model_dir / "inference" / "config.json": REFERENCE_CONFIG_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise VectorError(f"checkpoint identity mismatch: {path}")
    shard_records = []
    for name, (expected_bytes, digest) in REFERENCE_SOURCE_SHARDS.items():
        path = model_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != digest
        ):
            raise VectorError(f"checkpoint source shard identity mismatch: {path}")
        shard_records.append(
            {"path": name, "bytes": expected_bytes, "sha256": digest}
        )
    return shard_records


def require_consumed_shards(
    consumed: set[str],
    sources: dict[str, SourceTensor],
    shard_records: list[dict[str, Any]],
) -> None:
    consumed_shards = {sources[name].shard.name for name in consumed}
    expected_shards = {record["path"] for record in shard_records}
    if consumed_shards != expected_shards:
        raise VectorError(
            f"consumed source shard set mismatch: {sorted(consumed_shards)}"
        )


def build_manifest(
    model_dir: Path,
    batch_json: Path,
    loaded: list[str],
    consumed: set[str],
    source_batch: SourceBatch,
    shard_records: list[dict[str, Any]],
    tokens: dict[str, Any],
    vectors: list[dict[str, Any]],
) -> dict[str, Any]:
    inference_dir = model_dir / "inference"
    token_rows = source_batch.token_rows
    return {
        "format": "sparkpipe-dsv4-ga-reference-vectors-v1",
        "checkpoint": {
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": CHECKPOINT_REVISION,
            "index_sha256": CHECKPOINT_INDEX_SHA256,
            "config_sha256": CHECKPOINT_CONFIG_SHA256,
            "tokenizer_sha256": CHECKPOINT_TOKENIZER_SHA256,
        },
        "generator": {
            "path": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "reference": {
            "model_py_sha256": sha256_file(inference_dir / "model.py"),
            "kernel_py_sha256": sha256_file(inference_dir / "kernel.py"),
            "config_sha256": sha256_file(inference_dir / "config.json"),
            "sparse_attention_fallback": "torch-global-softmax-bf16-probability-v1",
            "hadamard_fallback": "normalized-sylvester-hadamard-v1",
            "fp8_gemm_fallback": "torch-block128-fp32-accumulation-v1",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "input": {
            "batch_json_sha256": sha256_file(batch_json),
            "source_request_id": source_batch.request_id,
            "source_sequence_id": source_batch.sequence_id,
            "token_artifact": tokens,
            "token_ids": token_rows,
            "positions": [list(range(REFERENCE_PROMPT_TOKENS))],
            "validation_sequence_ids": [1],
            "tensor_order": "batch,sequence,hc,hidden",
        },
        "layer_range": {
            "first": REFERENCE_FIRST_LAYER,
            "count": REFERENCE_LAYER_COUNT,
        },
        "loaded_parameter_count": len(loaded),
        "loaded_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(loaded)).encode("utf-8")
        ).hexdigest(),
        "consumed_checkpoint_tensor_count": len(consumed),
        "consumed_checkpoint_tensor_names_sha256": hashlib.sha256(
            "\n".join(sorted(consumed)).encode("utf-8")
        ).hexdigest(),
        "source_shards": shard_records,
        "vectors": vectors,
    }


def main() -> int:
    options = parse_args()
    model_dir = options.model_dir.resolve()
    output_dir = options.output_dir.resolve()
    batch_json = options.batch_json.resolve()
    if sys.byteorder != "little":
        raise VectorError("reference artifacts require a little-endian host")
    if not batch_json.is_file() or sha256_file(batch_json) != REFERENCE_BATCH_JSON_SHA256:
        raise VectorError("reference batch identity mismatch")
    source_batch = load_source_batch(batch_json)
    shard_records = require_checkpoint_identity(model_dir)
    token_rows = source_batch.token_rows
    sequence_length = len(token_rows[0])
    reference = load_reference_module(model_dir)
    args = load_config(model_dir, reference, len(token_rows), sequence_length)
    configure_reference(reference, args)
    validate_hadamard_reference()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    index_path, sources = weight_map(model_dir)
    with torch.device("cuda"):
        stage = ReferenceStage(reference, args, REFERENCE_FIRST_LAYER, REFERENCE_LAYER_COUNT)
    loaded, consumed = load_stage(stage, sources)
    require_consumed_shards(consumed, sources, shard_records)
    torch.use_deterministic_algorithms(True)
    torch.set_default_device("cuda")
    token_ids = torch.tensor(token_rows, dtype=torch.long, device="cuda")
    outputs = stage(token_ids)
    torch.cuda.synchronize()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = token_record(output_dir / "prompt_tokens.u32le", token_rows)
    if tokens["sha256"] != REFERENCE_TOKEN_PAYLOAD_SHA256:
        raise VectorError("reference prompt token payload identity mismatch")
    final_layer = REFERENCE_FIRST_LAYER + REFERENCE_LAYER_COUNT - 1
    vectors = [
        tensor_record(output_dir / f"after_layer_{final_layer}.bf16le", outputs[-1])
    ]
    if sha256_file(index_path) != CHECKPOINT_INDEX_SHA256:
        raise VectorError("checkpoint index changed during generation")
    manifest = build_manifest(
        model_dir,
        batch_json,
        loaded,
        consumed,
        source_batch,
        shard_records,
        tokens,
        vectors,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
