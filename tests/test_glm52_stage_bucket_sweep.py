#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
    assert module.parse_stage("0:3") == (0, 3)
    assert module.parse_stage("0:9") == (0, 9)
    assert module.stage_validator_parameters(0, 3) == (3, 0, True)
    assert module.stage_validator_parameters(0, 9) == (3, 6, True)
    assert module.stage_validator_parameters(75, 3) == (75, 3, False)
    dense_prefix_sample = (
        "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 "
        "dense_prefix_routed_pipeline=1 intermediate_stage=1 production_b12x=1 "
        "dense_chain_layers=3 first_routed_layer=3 routed_chain_layers=6 "
        "total_submissions=9 total_us=45000.000 maximum_us=5200.000 limit_us=1000000.000 "
        "pipeline_output_hidden=y input_embedding_token=1000 layer3_selected_expert=42 "
        "layer3_bound_experts=256 launch_chains=9 graph_captures=1 graph_replays=1"
    )
    dense_parsed = module.parse_result(dense_prefix_sample)
    assert dense_parsed["first_layer"] == 0
    assert dense_parsed["layer_count"] == 9
    assert dense_parsed["submissions"] == 9
    dense_chain_sample = (
        "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 "
        "dense_chain_layers=3 dense_chain_submissions=12 dense_chain_total_us=96000.000 "
        "maximum_us=12000.000 limit_us=1000000.000 restricted_token=1 mtp_draft=2 "
        "mtp_reject=3 input_embedding_bf16=1 input_embedding_token=1000 layer0_reference_full=1 "
        "layer0_reference_full_max_error=0.00000000 real_lm_head=1 real_lm_head_max_logit_error=0.00000000 "
        "launch_chains=12 graph_captures=12 graph_replays=12"
    )
    dense_chain_parsed = module.parse_result(dense_chain_sample)
    assert dense_chain_parsed["first_layer"] == 0
    assert dense_chain_parsed["layer_count"] == 3
    assert dense_chain_parsed["submissions"] == 12
    args = SimpleNamespace(
        max_stage_us="1000000",
        graph=False,
        model_dir="/models/glm52",
        cuda_arch="sm_121a",
        nvcc="/usr/local/cuda/bin/nvcc",
        aot_env="/home/spark2/.config/sparkpipe/glm52_b12x_aot_env.sh",
        aot_output_dir="build/glm52_b12x_aot",
        b12x_moe_pack_dir="build/glm52_b12x_resident_moe_0075_0077_v3",
        b12x_moe_pack_layers="75,76,77",
        allow_pack_build=False,
        require_pack_reuse=False,
        verify_reused_sha256=False,
        warmup_runs=1,
        measure_runs=2,
    )
    command = module.build_package_command(args, 8, 75, 3, Path("in.bf16"), Path("out.bf16"))
    assert "B12X_MOE_PACK_REQUIRE_REUSE=1" in command
    assert "B12X_MOE_PACK_VERIFY_REUSED_SHA256=1" not in command
    dense_command = module.build_package_command(args, 8, 0, 9, Path("in.bf16"), Path("out.bf16"))
    assert "GLM52_VALIDATION_MODE=dense_to_layer3_routed" in dense_command
    assert "GLM52_VALIDATION_FIRST_ROUTED_LAYER_INDEX=3" in dense_command
    assert "GLM52_VALIDATION_ROUTED_CHAIN_LAYER_COUNT=6" in dense_command
    assert "GLM52_PIPELINE_INPUT_HIDDEN_BF16=in.bf16" not in dense_command
    args.allow_pack_build = True
    command = module.build_package_command(args, 8, 75, 3, Path("in.bf16"), Path("out.bf16"))
    assert "B12X_MOE_PACK_REQUIRE_REUSE=0" in command
    args.allow_pack_build = False
    args.verify_reused_sha256 = True
    command = module.build_package_command(args, 8, 75, 3, Path("in.bf16"), Path("out.bf16"))
    assert "B12X_MOE_PACK_VERIFY_REUSED_SHA256=1" in command
    args.driver_so = Path("model_driver.so")
    command = module.build_direct_command(Path("validator_b8"), args)
    assert command == ["validator_b8", "1000000", "model_driver.so"]
    args.driver_so = None
    command = module.build_direct_command(Path("validator_b8"), args)
    assert command == ["validator_b8", "1000000"]
    attempts = [
        {"status": "pass", "warmup": True, "attempt_index": 0, "total_us": 60000.0},
        {"status": "pass", "warmup": False, "attempt_index": 0, "total_us": 50000.0},
        {"status": "pass", "warmup": False, "attempt_index": 1, "total_us": 40000.0},
    ]
    best = module.summarize_attempts(args, attempts)
    assert best["total_us"] == 40000.0
    assert best["best_attempt_index"] == 1
    assert best["attempt_count"] == 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
