#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = ROOT / "tools" / "glm52_dspark_manifest.py"
PREFLIGHT_TOOL = ROOT / "tools" / "glm52_dspark_artifact_preflight.py"
REVISION = "de0110be167c8da84eb7a253f07ba34eb172672e"


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
            "proposal_methods": [{
                "proposal_type": "greedy",
                "speculative_tokens": 7,
                "verifier_accept_k": 1,
            }],
            "verifier": {"name_or_path": "zai-org/GLM-5.2-FP8"},
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


def run_preflight(
    model_dir: Path,
    manifest: Path,
    quantization: str,
    mtp: bool = False,
) -> dict:
    command = [
        sys.executable,
        str(PREFLIGHT_TOOL),
        "--model-dir",
        str(model_dir),
        "--manifest",
        str(manifest),
        "--model-quantization",
        quantization,
        "--max-active",
        "64",
    ]
    if mtp:
        command.append("--mtp")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_dir = root / "model"
        manifest = root / "manifest.json"
        model_dir.mkdir()
        write_config(model_dir)
        (model_dir / "model.safetensors").write_bytes(b"sparkpipe")
        subprocess.run([
            sys.executable,
            str(MANIFEST_TOOL),
            "--model-dir",
            str(model_dir),
            "--output",
            str(manifest),
            "--model-revision",
            REVISION,
        ], check=True)
        for quantization in ("fp8", "w8lut", "nvfp4"):
            receipt = run_preflight(
                model_dir,
                manifest,
                quantization,
                mtp=quantization == "w8lut",
            )
            assert receipt["model_quantization"] == quantization
            assert receipt["quantization_contract"] == "independent"
            assert receipt["rows_per_lane"] == 8
            assert receipt["maximum_execution_rows"] == 512
        (model_dir / "model.safetensors").write_bytes(b"corrupt")
        failure = subprocess.run([
            sys.executable,
            str(PREFLIGHT_TOOL),
            "--model-dir",
            str(model_dir),
            "--manifest",
            str(manifest),
            "--model-quantization",
            "fp8",
        ], capture_output=True, text=True)
        assert failure.returncode == 2
        assert "DSpark weights bytes mismatch" in failure.stderr


if __name__ == "__main__":
    main()
