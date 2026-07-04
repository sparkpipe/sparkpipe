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
        assert "model.norm.weight" in tensor_map
        assert "lm_head.weight" in tensor_map
        assert tensor_map["model.embed_tokens.weight"]["file"] == "stage_00_non_moe.spstage"
        assert tensor_map["model.layers.18.input_layernorm.weight"]["file"] == "stage_03_non_moe.spstage"
        assert tensor_map["lm_head.weight"]["file"] == "stage_12_non_moe.spstage"
        for stage_file in ("stage_00_non_moe.spstage", "stage_03_non_moe.spstage", "stage_12_non_moe.spstage"):
            assert (output_dir / stage_file).stat().st_size > 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
