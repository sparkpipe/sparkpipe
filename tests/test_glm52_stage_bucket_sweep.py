#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "glm52_stage_bucket_sweep.py"
    spec = importlib.util.spec_from_file_location("glm52_stage_bucket_sweep", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load stage bucket sweep tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_sweep_module()
    sample = (
        "glm52_resident_decode_stage validation passed fixture=local_hidden_handoff "
        "routed_pipeline_from_hidden=1 intermediate_stage=1 first_routed_layer=11 "
        "routed_chain_layers=8 total_submissions=8 total_us=32000.000 "
        "maximum_us=4100.000 limit_us=1000000.000 pipeline_input_hidden=x "
        "pipeline_output_hidden=y layer3_selected_expert=42 layer3_bound_experts=256 "
        "launch_chains=8 graph_captures=1 graph_replays=1"
    )
    parsed = module.parse_result(sample)
    assert parsed["first_layer"] == 11
    assert parsed["layer_count"] == 8
    assert parsed["submissions"] == 8
    assert parsed["total_us"] == 32000.0
    assert parsed["maximum_us"] == 4100.0
    stage_ms = parsed["total_us"] / 1000.0
    tok_s = 32.0 * 1000.0 / stage_ms
    assert tok_s == 1000.0
    assert module.parse_csv_u32("8,16;32") == [8, 16, 32]
    assert module.parse_stage("75:3") == (75, 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
