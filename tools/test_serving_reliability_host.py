#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "ownership": "memory_buffer arena work_transaction runtime_completion model_runtime launch release status",
    "serving": "model_serving_adapter model_resident_deployment model_resident_ipc model_resident_deadline model_resident_session model_resident_reconnect model_pipeline_client model_pipeline_client_mock model_batch_engine_mock model_api_text pipeline_runtime serving_cache_admission steploop_admission continuous_batch model_resident_end_to_end orchestrator",
    "transport": "tp_device_collective_mock tp_collective serving_tp_config distributed_work hidden_transport hidden_transport_rdma_control fabric_topology memlink weightd_mesh_doorbell weightd_mesh_mock serial_tp_replay",
    "cache": "kv_cache kv_page_layout kv_store nvme_tier jit_kv_slice jit_kv_wire jit_kv_c3c4 jit_kv_c5w2 topology_switch kv_model_table",
    "weights": "weightd weightd_lease weightd_working_set weightd_churn weightd_expert_stress weightd_worker weightd_fd_frames weightd_attach weightd_expert weightd_map stage_module_weightd glm5_next_lazy_dispatch",
    "model_contracts": "model_description module_library driver_compiler stage_module_common llm_module_contract llm_stagepack_format tokenizer tokenizer_sidecar json numerical_metrics weight_codec gemm_descriptor_cache gemm_tile_k_fallback glm52_stagepack kda_reference rope_plan tensor_map_geometry",
    "driver_fixtures": "dsv4_serving_adapter dsv4_tp16_serving_adapter dsv4_tp4_pp4_serving_adapter qwen38_27b_serving_adapter muse_glimmer_serving_adapter gemma4_serving_adapter ling_serving_adapter hy4_lifecycle_smoke k3_attach_contract k3_kv_cache k3_llm_defines glm52_dspark glm52_mtp_tree kv_mooncake dsv4_cache_plan dsv4_lane_continuity dsv4_paged_cache dsv4_parallel_shape dsv4_pool_layout dsv4_stage_runner dsv4_tp_graph_contract dsv4_w1_loader gemma4_defines gemma4_defines_moe gemma4_defines_negative gemma4_defines_moe_negative k3_run_equivalence qwen38_27b_work_control qwen38_work_control",
    "speculation": "draft_bridge dspark_drafter_pin dsv4_pro_dspark_drafter_pin speculation_policy_pin speculation_provider_slot speculation_seam speculation_tree_pin",
}
PYTHON = {
    "weights": "weightd_supervision weightd_supervised weightd_manifest weightd_lazy_pair weightd_map_fd_ownership",
    "glm": "tp_mesh_cancel_lifetime glm5_next_graph_failure glm5_next_stage_context glm5_next_embedding_collective glm5_next_adapter_config_load glm5_next_expert_shard_math glm5_next_geometry glm5_next_driver_probe",
    "deployment": "generate_model_resident_deployment deployment_config_drift fleet_registrar spark_queue multi_dev_orchestrate inference_smoke glm5_next_queue_build mesh_lane_ladder_receipt",
    "model_contracts": "stage_module_teardown model_families model_driver_contracts glm52_module_contract qwen4_flash_model_header laguna_model_header",
}
FUZZ = ("tp_allreduce_fuzz", "serving_fault_fuzz", "kv_lane_fuzz", "weightd_working_set", "system_loopback")
GAPS = [
    "B2+ tree and bounded transfers have host tests; real GPU overlap and topology qualification remain unrun (I36/I50)",
    "GPU graph fault/cancellation, numerical model parity and complete recurrent restore are unqualified",
    "RDMA/device completion and daemon replacement require real hardware tests",
    "TP4, TP16, TP4xPP4 sustained serving, fairness and matched throughput remain unqualified",
    "Driver fixtures and host CUDA stubs do not qualify any model's GPU implementation",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,7,73")
    parser.add_argument("--source-commit", help="required immutable source identity when running an exported archive")
    parser.add_argument("--rounds", type=int, default=128)
    parser.add_argument("--loopback-rounds", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds) or any(not 1 <= seed <= 4294967295 for seed in seeds):
        parser.error("provide 1..32 distinct seeds in 1..UINT32_MAX")
    if not 24 <= args.rounds <= 1024 or not 24 <= args.loopback_rounds <= 100000 or not 1 <= args.timeout <= 3600 or not 1 <= args.jobs <= 16:
        parser.error("invalid rounds, timeout or job bound")
    os.chdir(ROOT)
    if (ROOT / "build/obj").exists() or (ROOT / "build/reliability/results.json").exists():
        parser.error("use a fresh checkout: do not mix host CUDA stubs with production objects or prior receipts")
    output = ROOT / "build/reliability"
    output.mkdir(parents=True, exist_ok=True)
    if args.source_commit is not None:
        if len(args.source_commit) != 40 or any(char not in "0123456789abcdef" for char in args.source_commit):
            parser.error("source-commit must be a full lowercase commit SHA")
        commit = args.source_commit
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*")):
        name = path.relative_to(ROOT)
        if any(part in {".git", "build", "__pycache__", "docs", "runs"} for part in name.parts):
            continue
        if path.is_file() and not path.is_symlink():
            digest.update(str(name).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    report = {
        "source_commit": commit,
        "source_sha256": digest.hexdigest(),
        "profile": "host fixtures and explicit CUDA stubs; execution results, not full semantic qualification",
        "modules": sorted(path.name for path in (ROOT / "modules").iterdir() if path.is_dir()),
        "seeds": seeds,
        "qualification_gaps": GAPS,
        "results": [],
    }

    def save():
        output.joinpath("results.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, timeout, group):
        started = time.monotonic()
        with output.joinpath(name + ".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
                status = "PASS" if code == 0 else "FAIL"
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code, status = 124, "TIMEOUT"
        leaked = False
        if status != "TIMEOUT":
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leaked, status = True, "FAIL"
        result = {"name": name, "group": group, "command": command, "status": status,
                  "exit_code": code, "leaked_process_group": leaked,
                  "seconds": round(time.monotonic() - started, 3)}
        report["results"].append(result)
        save()
        print(f"{status} {name}", flush=True)
        return code

    targets = sorted({"test_" + name for names in GROUPS.values() for name in names.split()} | {"test_" + name for name in FUZZ})
    registered = set(re.findall(r"\btest_\w+\b", (ROOT / "Makefile").read_text().split("TEST_NAMES :=", 1)[1].split("TEST_BINARIES :=", 1)[0]))
    report["registered_c_targets"] = len(registered)
    report["unrun_registered_c_targets"] = sorted(registered - set(targets))
    selected_python = {"test_" + name + ".py" for names in PYTHON.values() for name in names.split()} | {"test_model_api_queue_lifetime.py"}
    report["unrun_python_test_files"] = sorted(path.name for path in (ROOT / "tests").glob("test_*.py") if path.name not in selected_python)
    save()
    run("build", ["make", "-k", f"-j{args.jobs}", "CUDA_HOME=/nonexistent", *["build/" + name for name in targets], "build/sparkpipe_weightd", "build/sparkpipe_registrar"], 900, "setup")
    for group, names in GROUPS.items():
        for name in names.split():
            binary = "build/test_" + name
            if not Path(binary).is_file():
                report["results"].append({"name": "test_" + name, "group": group, "status": "SETUP_FAIL", "reason": "binary missing; see build.log"})
                save()
                print("SETUP_FAIL test_" + name, flush=True)
                continue
            run("test_" + name, [binary], args.timeout, group)
    for group, names in PYTHON.items():
        for name in names.split():
            run("test_" + name, [sys.executable, "tests/test_" + name + ".py"], args.timeout, group)
    for seed in seeds:
        for name in FUZZ:
            binary = "build/test_" + name
            cases = [(name, [str(seed), str(args.rounds)])]
            if name == "system_loopback":
                cases = [(name, ["--fuzz", str(args.loopback_rounds), str(seed)])]
            elif name == "tp_allreduce_fuzz":
                cases = [(f"{name}-ranks{ranks}", ["--ranks", str(ranks), "--fuzz", str(args.rounds), "--seed", str(seed)]) for ranks in (2, 4, 8, 16)]
            for case, options in cases:
                label = f"test_{case}-seed{seed}"
                if not Path(binary).is_file():
                    report["results"].append({"name": label, "group": "fuzz", "status": "SETUP_FAIL", "reason": "fuzzer binary missing; see build.log"})
                    save()
                    continue
                run(label, [binary, *options], args.timeout, "fuzz")
        run(f"test_model_api_queue_lifetime-seed{seed}", [sys.executable, "tests/test_model_api_queue_lifetime.py", str(seed)], args.timeout, "fuzz")
    if Path("build/test_tp_allreduce_fuzz").is_file():
        run("test_tp_tree_qualification", ["build/test_tp_allreduce_fuzz", "--ranks", "4", "--fuzz", "0", "--qualify-tree"], args.timeout, "qualification")
    counts = {status: sum(row["status"] == status for row in report["results"]) for status in ("PASS", "FAIL", "TIMEOUT", "SETUP_FAIL")}
    report["counts"] = counts
    save()
    print(json.dumps(counts, sort_keys=True))
    print("Hardware/model qualification remains OPEN; see results.json and docs/SERVING_FUZZ_COVERAGE.md")
    return int(any(counts[status] for status in ("FAIL", "TIMEOUT", "SETUP_FAIL")))


if __name__ == "__main__":
    sys.exit(main())
