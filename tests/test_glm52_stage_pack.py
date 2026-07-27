#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile


def load_stage_pack_module():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "glm52_stage_pack.py"
    spec = importlib.util.spec_from_file_location("glm52_stage_pack", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load stage pack tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, body) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(body)],
        }
        payload.extend(body)
        offset += len(body)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def main() -> int:
    module = load_stage_pack_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "model"
        output_dir = root / "stagepacks"
        model_dir.mkdir()
        tensors = {
            "model.embed_tokens.weight": ("BF16", [4, 2], b"abcdefghijklmnop"),
            "model.layers.18.input_layernorm.weight": ("BF16", [4], b"12345678"),
            "model.layers.18.self_attn.q_proj.weight": ("U8", [2, 2], b"QWER"),
            "model.layers.18.mlp.experts.0.gate_proj.weight": ("U8", [2, 2], b"DROP"),
            "model.layers.78.enorm.weight": ("BF16", [4], b"MTPENORM"),
            "model.layers.78.self_attn.indexer.k_norm.weight": ("BF16", [4], b"MTPINDEX"),
            "model.layers.78.mlp.experts.0.gate_proj.weight": ("U8", [2, 2], b"SKIP"),
            "model.norm.weight": ("BF16", [4], b"abcdefgh"),
            "lm_head.weight": ("BF16", [8, 2], b"ABCDEFGHIJKLMNOP"),
        }
        write_safetensors(model_dir / "model-00001-of-00001.safetensors", tensors)
        index = {
            "metadata": {"total_size": sum(len(item[2]) for item in tensors.values())},
            "weight_map": {
                name: "model-00001-of-00001.safetensors"
                for name in tensors
            },
        }
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(index),
            encoding="utf-8",
        )
        args = type(
            "Args",
            (),
            {
                "model_dir": model_dir,
                "output_dir": output_dir,
                "model_quantization": "fp8",
                "stages": "0,3,12",
                "reuse": False,
                "mtp_only": False,
            },
        )()
        result = module.build_stage_packs(args)
        assert result["format"] == module.FORMAT
        stage_index = json.loads((output_dir / module.INDEX_FILE).read_text(encoding="utf-8"))
        tensor_map = stage_index["tensor_map"]
        assert "model.embed_tokens.weight" in tensor_map
        assert "model.layers.18.input_layernorm.weight" in tensor_map
        assert "model.layers.18.self_attn.q_proj.weight" in tensor_map
        assert "model.layers.18.mlp.experts.0.gate_proj.weight" not in tensor_map
        assert "model.layers.78.enorm.weight" in tensor_map
        assert "model.layers.78.self_attn.indexer.k_norm.weight" not in tensor_map
        assert "model.layers.78.mlp.experts.0.gate_proj.weight" not in tensor_map
        assert module.MTP_EMBEDDING_ALIAS in tensor_map
        assert "model.norm.weight" in tensor_map
        assert "lm_head.weight" in tensor_map
        assert tensor_map["model.embed_tokens.weight"]["file"] == "stage_00_non_moe.spstage"
        assert tensor_map["model.layers.18.input_layernorm.weight"]["file"] == "stage_03_non_moe.spstage"
        assert tensor_map["lm_head.weight"]["file"] == "stage_12_non_moe.spstage"
        assert tensor_map[module.MTP_EMBEDDING_ALIAS]["file"] == "stage_12_non_moe.spstage"
        for stage_file in ("stage_00_non_moe.spstage", "stage_03_non_moe.spstage", "stage_12_non_moe.spstage"):
            assert (output_dir / stage_file).stat().st_size > 0
        supplement_dir = root / "supplement"
        supplement_dir.mkdir()
        (supplement_dir / "stage_12_non_moe.spstage").write_bytes(b"base")
        (supplement_dir / module.INDEX_FILE).write_text(
            json.dumps({
                "format": module.FORMAT,
                "model_quantization": "fp8",
                "topology": "ring_fixed6",
                "stage_count": module.STAGE_COUNT,
                "layers_per_stage": module.LAYERS_PER_STAGE,
                "tensor_map": {},
                "stages": {"12": {"file": "stage_12_non_moe.spstage"}},
            }),
            encoding="utf-8",
        )
        supplement_args = type(
            "Args",
            (),
            {
                "model_dir": model_dir,
                "output_dir": supplement_dir,
                "model_quantization": "fp8",
                "stages": "12",
                "reuse": False,
                "mtp_only": True,
            },
        )()
        module.build_stage_packs(supplement_args)
        supplement_index = json.loads(
            (supplement_dir / module.INDEX_FILE).read_text(encoding="utf-8"))
        assert (supplement_dir / module.MTP_STAGE_FILE).stat().st_size > 0
        assert supplement_index["supplements"]["mtp"]["layer"] == 78
        assert supplement_index["tensor_map"]["model.layers.78.enorm.weight"][
            "file"] == module.MTP_STAGE_FILE
        assert supplement_index["tensor_map"][module.MTP_EMBEDDING_ALIAS][
            "file"] == module.MTP_STAGE_FILE
        w8_model_dir = root / "w8-model"
        w8_output_dir = root / "w8-stagepacks"
        w8_model_dir.mkdir()
        w8_tensors = {
            "model.layers.18.input_layernorm.weight": (
                "BF16", [4], b"12345678"
            ),
            "model.layers.18.self_attn.q_proj.weight": (
                "BF16", [2, 2], b"QWERASDF"
            ),
            "model.layers.18.mlp.experts.0.gate_proj.weight": (
                "BF16", [2, 2], b"SKIPSKIP"
            ),
        }
        write_safetensors(
            w8_model_dir / "model-00001-of-00001.safetensors",
            w8_tensors,
        )
        (w8_model_dir / "model.safetensors.index.json").write_text(
            json.dumps({
                "metadata": {
                    "total_size": sum(len(item[2]) for item in w8_tensors.values())
                },
                "weight_map": {
                    name: "model-00001-of-00001.safetensors"
                    for name in w8_tensors
                },
            }),
            encoding="utf-8",
        )
        w8_args = type(
            "Args",
            (),
            {
                "model_dir": w8_model_dir,
                "output_dir": w8_output_dir,
                "model_quantization": module.MODEL_QUANTIZATION_W8LUT,
                "stages": "3",
                "reuse": False,
                "mtp_only": False,
            },
        )()
        module.build_stage_packs(w8_args)
        w8_index = json.loads(
            (w8_output_dir / module.INDEX_FILE).read_text(encoding="utf-8")
        )
        assert w8_index["model_quantization"] == "w8lut"
        assert w8_index["non_expert_weight_dtype"] == "BF16"
        assert len(w8_index["source_model_index_sha256"]) == 64
        assert w8_index["tensor_map"][
            "model.layers.18.self_attn.q_proj.weight"
        ]["dtype"] == "BF16"
        assert "model.layers.18.mlp.experts.0.gate_proj.weight" not in w8_index[
            "tensor_map"
        ]
        nvfp4_output_dir = root / "nvfp4-stagepacks"
        nvfp4_args = type(
            "Args",
            (),
            {
                "model_dir": w8_model_dir,
                "output_dir": nvfp4_output_dir,
                "model_quantization": module.MODEL_QUANTIZATION_NVFP4,
                "stages": "3",
                "reuse": False,
                "mtp_only": False,
            },
        )()
        nvfp4_result = module.build_stage_packs(nvfp4_args)
        nvfp4_index = json.loads(
            (nvfp4_output_dir / module.INDEX_FILE).read_text(encoding="utf-8")
        )
        assert nvfp4_result["non_expert_weight_dtype"] == "BF16"
        assert nvfp4_index["model_quantization"] == "nvfp4"
        assert nvfp4_index["non_expert_weight_dtype"] == "BF16"
        assert nvfp4_index["tensor_map"][
            "model.layers.18.self_attn.q_proj.weight"
        ]["dtype"] == "BF16"
        bad_args = type(
            "Args",
            (),
            {
                "model_dir": model_dir,
                "output_dir": root / "bad-w8-stagepacks",
                "model_quantization": module.MODEL_QUANTIZATION_W8LUT,
                "stages": "3",
                "reuse": False,
                "mtp_only": False,
            },
        )()
        try:
            module.build_stage_packs(bad_args)
        except module.StagePackFailure as error:
            assert "must be BF16" in str(error)
        else:
            raise AssertionError("W8LUT stage pack accepted a non-BF16 weight")
        bad_nvfp4_args = type(
            "Args",
            (),
            {
                "model_dir": model_dir,
                "output_dir": root / "bad-nvfp4-stagepacks",
                "model_quantization": module.MODEL_QUANTIZATION_NVFP4,
                "stages": "3",
                "reuse": False,
                "mtp_only": False,
            },
        )()
        try:
            module.build_stage_packs(bad_nvfp4_args)
        except module.StagePackFailure as error:
            assert "must be BF16" in str(error)
        else:
            raise AssertionError("NVFP4 stage pack accepted a non-BF16 weight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
