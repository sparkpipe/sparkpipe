#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "glm52_dspark_manifest.py"


def write_config(path: Path) -> None:
    config = {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
        "block_size": 8,
        "dtype": "bfloat16",
        "draft_vocab_size": 154880,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 256,
        "max_anchors": 1024,
        "speculators_config": {
            "algorithm": "dspark",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "name_or_path": "zai-org/GLM-5.2-FP8",
            },
        },
        "transformer_layer_config": {
            "head_dim": 64,
            "hidden_size": 6144,
            "intermediate_size": 12288,
            "max_position_embeddings": 1048576,
            "model_type": "qwen3",
            "num_attention_heads": 64,
            "num_hidden_layers": 5,
            "num_key_value_heads": 64,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {
                "rope_theta": 8000000,
                "rope_type": "default",
            },
            "vocab_size": 154880,
        },
    }
    (path / "config.json").write_text(json.dumps(config))


def test_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        output = Path(tmp) / "manifest.json"
        model_dir.mkdir()
        write_config(model_dir)
        (model_dir / "model.safetensors").write_bytes(b"sparkpipe")
        subprocess.check_call(
            [
                sys.executable,
                str(TOOL),
                "--model-dir",
                str(model_dir),
                "--output",
                str(output),
                "--model-revision",
                "de0110be167c8da84eb7a253f07ba34eb172672e",
            ]
        )
        manifest = json.loads(output.read_text())
        assert manifest["format"] == "sparkpipe.glm52.dspark.speculator_manifest.v2"
        assert manifest["model_revision"] == (
            "de0110be167c8da84eb7a253f07ba34eb172672e"
        )
        assert manifest["training_verifier_model"] == "zai-org/GLM-5.2-FP8"
        assert manifest["verifier_contract"] == {
            "quantization_independent": True,
            "hidden_dtype": "bf16",
            "hidden_dimension": 6144,
            "vocabulary_size": 154880,
        }
        assert manifest["aux_hidden_state_layer_ids"] == [8, 23, 39, 55, 70]
        assert manifest["maximum_speculative_token_count"] == 7
        assert manifest["contract"]["verifier_hidden_dtype"] == 1
        assert manifest["contract"]["rope_theta"] == 8000000
        assert manifest["model_safetensors"]["path"] == "model.safetensors"
        assert manifest["config_json"]["path"] == "config.json"
        assert manifest["model_safetensors"]["sha256"] == (
            "19d151c5ad852a58ff2a464349b69bd14820375cb838ed046eae308a08523d7f"
        )


if __name__ == "__main__":
    test_manifest()
