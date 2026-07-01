#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validator = (
        root
        / "modules"
        / "glm52_resident_decode_stage"
        / "validation"
        / "spark_glm52_resident_decode_stage_cuda_validation.cu"
    ).read_text(encoding="utf-8")
    makefile = (
        root
        / "modules"
        / "glm52_resident_decode_stage"
        / "Makefile"
    ).read_text(encoding="utf-8")
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN" in validator
    assert "routed_pipeline_from_hidden_final=1 final_stage=1" in validator
    assert "SparkValidationSetOutputHiddenOnly" in validator
    assert "run_final_outputs = final_token_stage != 0u" in validator
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1" in makefile
    assert "routed_from_hidden_final" in makefile
    assert "GLM52_CHAIN_ROUTED_FROM_HIDDEN_BF16=1 GLM52_CHAIN_ROUTED_FROM_HIDDEN_FINAL_TOKEN=1" not in makefile
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
