#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "model_contracts" / "spark_hardware_questions.json"
PROBE_PLAN_PATH = ROOT / "qualification" / "spark" / "probe_plan.json"
WORKLOADS_PATH = ROOT / "qualification" / "spark" / "workload_profiles.json"
MUST_WORK_PATH = ROOT / "model_contracts" / "must_work_targets.json"
PLAN_TOOL = ROOT / "tools" / "hardware" / "spark_qualification_plan.py"
VALIDATE_TOOL = ROOT / "tools" / "spark_hardware_qualify.py"


def load(path: pathlib.Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {command}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_registry() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    questions_document = load(QUESTIONS_PATH)
    probe_document = load(PROBE_PLAN_PATH)
    require(isinstance(questions_document, dict), "question registry is not an object")
    require(isinstance(probe_document, dict), "probe plan is not an object")
    questions_raw = questions_document.get("questions")
    probes_raw = probe_document.get("probes")
    require(isinstance(questions_raw, list) and questions_raw, "question registry is empty")
    require(isinstance(probes_raw, list) and probes_raw, "probe plan is empty")

    question_ids = [question.get("id") for question in questions_raw if isinstance(question, dict)]
    probe_ids = [probe.get("id") for probe in probes_raw if isinstance(probe, dict)]
    require(len(question_ids) == 33, f"expected 33 hardware questions, found {len(question_ids)}")
    require(len(question_ids) == len(set(question_ids)), "hardware question IDs are not unique")
    require(len(probe_ids) == len(set(probe_ids)), "probe IDs are not unique")

    probes = {str(probe["id"]): probe for probe in probes_raw}
    questions = {str(question["id"]): question for question in questions_raw}
    for question_id, question in questions.items():
        probe_id = question.get("probe")
        require(probe_id in probes, f"{question_id}: unknown probe {probe_id}")
        require(isinstance(question.get("production_required"), bool),
                f"{question_id}: production_required is not boolean")
        require(isinstance(question.get("question"), str) and question["question"],
                f"{question_id}: missing question text")
        require(isinstance(question.get("consumers"), list) and question["consumers"],
                f"{question_id}: missing consumers")

    question_counts = Counter(str(question["probe"]) for question in questions.values())
    for probe_id, probe in probes.items():
        require(question_counts[probe_id] > 0, f"probe {probe_id} owns no questions")
        components = probe.get("components")
        require(isinstance(components, list) and components, f"probe {probe_id} has no components")
        for component in components:
            require(isinstance(component, dict), f"probe {probe_id}: malformed component")
            source = ROOT / str(component.get("source", ""))
            require(source.is_file(), f"probe {probe_id}: missing source {source.relative_to(ROOT)}")
    return questions, probes


def validate_single_registry() -> None:
    candidates = []
    for path in ROOT.rglob("*.json"):
        if any(part in {"build", "__pycache__"} for part in path.parts):
            continue
        try:
            document = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("registry_id") == "spark_hardware_truth_v1":
            candidates.append(path.relative_to(ROOT).as_posix())
    require(candidates == ["model_contracts/spark_hardware_questions.json"],
            f"multiple hardware question registries exist: {candidates}")


def validate_workloads() -> None:
    workloads = load(WORKLOADS_PATH)
    must_work = load(MUST_WORK_PATH)
    require(isinstance(workloads, dict) and isinstance(must_work, dict), "invalid model contracts")
    models_raw = workloads.get("models")
    targets_raw = must_work.get("targets")
    require(isinstance(models_raw, list) and len(models_raw) == 5, "workload profile must contain five models")
    require(isinstance(targets_raw, list) and len(targets_raw) == 5, "must-work contract must contain five models")
    models = {str(model["family"]): model for model in models_raw}
    targets = {str(target["model_family"]): target for target in targets_raw}
    require(set(models) == set(targets), f"workload/must-work families differ: {set(models)} vs {set(targets)}")

    expected = {
        "k3": (7168, 96, 96, 896, 16, "mxfp4_e2m1", "bf16", "bf16"),
        "glm52": (6144, 64, 1, 256, 8, (
            "int6_block_f32",
            "int7_block_f32",
            "int8_block_f32",
            "fp8_e4m3_block_f32",
            "nvfp4_e2m1_ue4m3_global_f32",
            "mxfp4_e2m1_e8m0",
        ), "bf16", "bf16"),
        "qwen38_27b": (5120, 24, 4, 0, 0, "none", "bf16", "bf16"),
        "dsv4_flash": (4096, 64, 1, 256, 6, "fp4_native", "checkpoint_native", "fp8_e4m3"),
        "dsv4_pro": (7168, 128, 1, 384, 6, "fp4_native", "checkpoint_native", "fp8_e4m3"),
    }
    for family, values in expected.items():
        model = models[family]
        actual = (
            model.get("hidden_dimension"),
            model.get("attention_head_count"),
            model.get("kv_head_count"),
            model.get("moe_expert_count"),
            model.get("moe_top_k"),
            (tuple(model["expert_weight_formats"])
             if "expert_weight_formats" in model
             else model.get("expert_weight_format")),
            model.get("expert_activation_format"),
            model.get("non_expert_format"),
        )
        require(actual == values, f"{family}: workload geometry/precision mismatch {actual} != {values}")
        target = targets[family]
        require(target.get("production_ready") is False, f"{family}: unqualified target marked ready")

    batches = workloads.get("batch_sizes")
    contexts = workloads.get("context_buckets")
    pp_degrees = workloads.get("candidate_pipeline_degrees")
    require(batches == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "batch coverage drift")
    require(contexts == [2048, 16384, 65536, 262144, 1048576], "context coverage drift")
    require(pp_degrees == [1, 2, 4, 8, 13, 16], "pipeline-degree coverage drift")


def validate_generated_plans(questions: dict[str, dict[str, object]]) -> None:
    source_sha = "ab" * 32
    topologies = [
        ROOT / "qualification" / "spark" / "topologies" / "ring_13node_bringup.json",
        ROOT / "qualification" / "spark" / "topologies" / "single_switch_16node.json",
    ]
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        for topology in topologies:
            output = directory / f"{topology.stem}.json"
            run([
                sys.executable,
                str(PLAN_TOOL),
                "--topology", str(topology),
                "--source-package-sha256", source_sha,
                "--output", str(output),
            ])
            run([sys.executable, str(VALIDATE_TOOL), "validate-plan", str(output)])
            plan = load(output)
            require(isinstance(plan, dict), "generated plan is not an object")
            coverage = plan.get("coverage")
            require(isinstance(coverage, dict) and set(coverage) == set(questions),
                    f"{topology.name}: incomplete question coverage")
            node_count = len(plan["topology"]["nodes"])
            pp_candidates = coverage["TOPO-PP-001"]["axes"]["candidate"]
            require(all(int(value.removeprefix("pp")) <= node_count for value in pp_candidates),
                    f"{topology.name}: pipeline degree exceeds node count")
            if plan["topology"]["mode"] == "ring":
                require(coverage["NET-RING-001"]["applicable"] is True, "ring test disabled on ring")
                require(coverage["NET-SWITCH-001"]["applicable"] is False, "switch test enabled on ring")
            else:
                require(coverage["NET-RING-001"]["applicable"] is False, "ring test enabled on switch")
                require(coverage["NET-SWITCH-001"]["applicable"] is True, "switch test disabled on switch")
            jobs_by_question: dict[str, list[dict[str, object]]] = {}
            for job in plan["jobs"]:
                jobs_by_question.setdefault(str(job["question_id"]), []).append(job)

            expected_candidates = {
                "GB10-MEM-001": {"bandwidth"},
                "GB10-MEM-002": {"reuse"},
                "GB10-MEM-003": {"pointer_chase"},
                "GB10-UMEM-001": {"gpu_only", "cpu_read_contention", "cpu_write_contention"},
                "GB10-MAPPED-001": {"mapped_host"},
                "GB10-COPY-001": {"copy"},
                "GB10-GRAPH-001": {"direct", "graph"},
                "GB10-CALLBACK-001": {"stream_sync", "event", "host_callback"},
                "GB10-CONCURRENCY-001": {"copy_copy", "copy_compute", "compute_compute"},
                "GB10-SMEM-001": {"dynamic_shared"},
                "GB10-ATOMIC-001": {"contended", "distributed"},
                "GB10-THERMAL-001": {"sustained_memory_copy"},
                "NVME-RAW-001": {"direct_io"},
                "NVME-GPU-001": {"nvme_to_gpu"},
                "NET-PMTU-001": {"udp_df_binary_search"},
            }
            for question_id, candidates in expected_candidates.items():
                actual = {
                    str(job["parameters"].get("candidate"))
                    for job in jobs_by_question.get(question_id, [])
                }
                require(actual == candidates,
                        f"{topology.name}: {question_id} candidate drift {actual} != {candidates}")

            for job in jobs_by_question["GB10-LAUNCH-001"]:
                parameters = job["parameters"]
                require(parameters["mode"] in {"enqueue", "launch_sync"},
                        "launch mode is not implemented by the CUDA probe")
                require(parameters["load_mode"] in {"idle", "memory_loaded"},
                        "launch load mode is not implemented by the CUDA probe")
                require(parameters["batch_size"] > 0 and parameters["kernel_count"] > 0,
                        "launch cell has no executable work")

            for job in jobs_by_question["TOPO-PP-001"]:
                parameters = job["parameters"]
                require(parameters["candidate"] == f"pp{parameters['pipeline_degree']}",
                        "pipeline-degree candidate is not tied to its degree")

            for question_id, specification in coverage.items():
                require(isinstance(specification.get("expected_observation_count"), int),
                        f"{question_id}: no expected observation count")
                if specification.get("applicable", True) and questions[question_id]["production_required"]:
                    require(specification["expected_observation_count"] > 0,
                            f"{question_id}: required applicable question has no cells")

            coverage_text = json.dumps(coverage, sort_keys=True)
            for model_id in [model["id"] for model in load(WORKLOADS_PATH)["models"]]:
                require(model_id in coverage_text, f"{topology.name}: model {model_id} absent from plan")


def validate_packaging_contract() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import package_inventory  # type: ignore

    payloads = {entry.path for entry in package_inventory.source_inventory(ROOT)}
    require("qualification/spark/probe_plan.json" in payloads, "probe plan excluded from source package")
    require("qualification/spark/workload_profiles.json" in payloads, "workload profile excluded from source package")
    require(not any(path.startswith("qualification/raw/") for path in payloads), "raw probe data included")
    require(not any(path.startswith("qualification/receipts/") for path in payloads), "probe receipts included")


def main() -> int:
    questions, _ = validate_registry()
    validate_single_registry()
    validate_workloads()
    validate_generated_plans(questions)
    validate_packaging_contract()
    print("PASS Spark hardware question, probe, model, topology, plan, and package coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
