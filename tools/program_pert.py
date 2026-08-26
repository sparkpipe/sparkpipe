#!/usr/bin/env python3
"""Build and validate the full SparkPipe program PERT graph.

The existing orchestration/platform_tasks.json remains the executable queue of
fully scoped near-term agent contracts.  This file is the complete program WBS:
small work packages, external gates, hardware gates, UI work, and milestones.
It deliberately records no task as complete merely because related source or a
historical receipt exists.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


BASELINE_DATE = "2026-08-25"
VISUALIZATION_TEMPLATE = Path(__file__).with_name("program_pert_visualization.fragment.html")
AGENT_REDUNDANCY = {"implementer": 2, "auditor": 2}
PROVIDER_REQUESTS_PER_PAIR = sum(AGENT_REDUNDANCY.values())
MIN_PROVIDER_FAILURE_DOMAINS = 2
PROVIDER_SUPPLY_FRESHNESS_HOURS = 24

MODEL_DRIVER_PROGRAMS = (
    ("Q27", "Qwen 3.8 27B", "cuda1", "cuda2plus", "existing_needs_requalify"),
    ("D4F", "DSV4 Flash", "cuda4", "cuda4", "existing_needs_requalify"),
    ("GLM", "GLM 5.2", "capacity_selected", "cuda8", "correctness_blocked"),
    ("K3", "K3", "cuda16", "cuda16", "correctness_blocked"),
    ("D4P", "DSV4 Pro", "cuda16", "cuda16", "review_only"),
    ("QMAX", "Qwen 3.8 Max", "cuda16", "cuda16", "review_only"),
    ("H3", "MiniMax H3", "capacity_selected", "capacity_selected", "new_model"),
)

WORKSTREAMS = {
    "foundation": "Architecture and foundation",
    "agents": "OxAlpha development system",
    "reconciliation": "Main/unified reconciliation",
    "recipes": "Model recipes and compiler",
    "artifacts": "Artifacts, Ceph, and file agent",
    "hardware": "Hardware-neutral interface",
    "cuda": "NVIDIA CUDA",
    "amd": "AMD ROCm",
    "metal": "Apple Silicon Metal",
    "topology": "Topology and inventory",
    "collectives": "Collectives and transport",
    "runtime": "Inference runtime",
    "kv": "KV and prefix hierarchy",
    "scheduler": "Compute islands and scheduling",
    "api": "Public API",
    "metering": "Metering and capacity credits",
    "providers": "Compute-provider marketplace",
    "ui": "Customer, provider, and operator UI",
    "observability": "Observability and SOTA status",
    "security": "Security and privacy",
    "reliability": "SRE, HA, CI, and release",
    "models": "Model programs",
    "performance": "SOTA performance program",
    "business": "Commercial and external gates",
    "milestone": "Program milestones",
}

HARDWARE_REQUIREMENTS = {
    "host": [("host", 1)],
    "cpu": [("cpu", 1)],
    "accelerator": [("accelerator_lab", 1)],
    "cuda_sm121": [("cuda_spark", 1)],
    "cuda1": [("cuda_spark", 1)],
    "cuda2plus": [("cuda_spark", 2)],
    "cuda4": [("cuda_spark", 4)],
    "cuda8": [("cuda_spark", 8)],
    "cuda16": [("cuda_spark", 16)],
    "cuda4_storage": [("cuda_spark", 4), ("spark_storage", 4)],
    "spark_storage": [("spark_storage", 1)],
    "spark16_storage": [("spark_storage", 16)],
    "accelerator_storage": [("accelerator_lab", 1), ("storage_lab", 1)],
    "model_storage": [("storage_lab", 1)],
    "ceph": [("ceph", 1)],
    "rocm_toolchain": [("rocm_toolchain", 1)],
    "mi350": [("mi350", 1)],
    "mi350x4": [("mi350", 4)],
    "apple_silicon": [("apple_silicon", 1)],
    "apple_silicon_multi": [("apple_silicon", 2)],
    "multi_backend": [("multi_backend_lab", 1)],
    "multi_rank": [("multi_rank_lab", 1)],
    "single_node": [("single_node_lab", 1)],
    "fleet": [("fleet_window", 1)],
    "fleet_network": [("fleet_network", 1)],
    "fleet_and_provider": [("fleet_window", 1), ("provider_island", 1)],
    "provider_island": [("provider_island", 1)],
    "external": [("external_review", 1)],
    "capacity_selected": [("capacity_selected_cuda", 1)],
}

PLANNING_CAPACITY = {
    "worker:coordinator": 1,
    "worker:oxalpha_pair": 32,
    "worker:frontend_pair": 4,
    "worker:sdk_pair": 2,
    "worker:security_pair": 2,
    "worker:finance_pair": 2,
    "worker:legal_finance": 1,
    "host": 64,
    "cpu": 8,
    "api_provider_request": 128,
    "accelerator_lab": 8,
    "cuda_spark": 16,
    "spark_storage": 16,
    "storage_lab": 4,
    "ceph": 1,
    "rocm_toolchain": 2,
    "mi350": 4,
    "apple_silicon": 2,
    "multi_backend_lab": 2,
    "multi_rank_lab": 4,
    "single_node_lab": 8,
    "fleet_window": 1,
    "fleet_network": 1,
    "provider_island": 2,
    "external_review": 1,
    "capacity_selected_cuda": 16,
}


def _split_dependencies(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.replace(" ", "").split(",") if item]
    return list(value)


def _estimate(most_likely: float, uncertainty: str) -> dict[str, float]:
    if most_likely == 0:
        return {"optimistic": 0.0, "most_likely": 0.0, "pessimistic": 0.0}
    factors = {
        "low": (0.75, 1.35),
        "normal": (0.60, 1.80),
        "high": (0.45, 2.40),
        "research": (0.30, 3.20),
    }
    optimistic_factor, pessimistic_factor = factors[uncertainty]
    return {
        "optimistic": round(max(0.25, most_likely * optimistic_factor), 2),
        "most_likely": round(most_likely, 2),
        "pessimistic": round(max(most_likely, most_likely * pessimistic_factor), 2),
    }


def build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    def add(
        task_id: str,
        workstream: str,
        title: str,
        dependencies: str | list[str] | tuple[str, ...],
        days: float,
        acceptance: str,
        *,
        phase: int,
        kind: str = "implementation",
        resource: str = "oxalpha_pair",
        hardware: str = "host",
        gate: str = "G1",
        lock: str = "",
        uncertainty: str = "normal",
        planning_state: str = "planned",
        pairable: bool = True,
        recurring_days: int | None = None,
        freshness_hours: int | None = None,
        required_for_release: bool = True,
        provider_request_slots: int | None = None,
        agent_lane: str = "",
    ) -> None:
        labels = [item for item in hardware.replace(" ", "").split(",") if item]
        if not agent_lane and task_id.startswith("MOD-"):
            agent_lane = f"model-driver:{task_id.split('-', 2)[1].lower()}"
        requirements = [
            {"pool": pool, "quantity": quantity}
            for label in labels
            for pool, quantity in HARDWARE_REQUIREMENTS[label]
        ]
        locks = [item.strip() for item in lock.split(",") if item.strip()]
        if pairable and not locks:
            locks = [f"workstream:{workstream}"]
        if provider_request_slots is None:
            provider_request_slots = PROVIDER_REQUESTS_PER_PAIR if pairable else 0
        tasks.append(
            {
                "id": task_id,
                "workstream": workstream,
                "title": title,
                "dependencies": _split_dependencies(dependencies),
                "estimate_days": _estimate(days, uncertainty),
                "acceptance": acceptance,
                "phase": phase,
                "kind": kind,
                "resource": resource,
                "hardware": hardware,
                "hardware_requirements": requirements,
                "gate": gate,
                "write_locks": locks,
                "uncertainty": uncertainty,
                "estimate_basis": "coordinator_initial_heuristic",
                "planning_state": planning_state,
                "pairable": pairable,
                "recurring_days": recurring_days,
                "freshness_hours": freshness_hours,
                "required_for_release": required_for_release,
                "provider_request_slots": provider_request_slots,
                "provider_failure_domains_required": MIN_PROVIDER_FAILURE_DOMAINS if pairable else 0,
                "dispatch_contract_required": pairable,
                "dispatch_prerequisites": [],
                "agent_lane": agent_lane,
            }
        )

    # Architecture and immutable program contracts.
    add("FND-001", "foundation", "Current-state evidence baseline", "", 1.0, "Every model, branch, API, cache, backend, and fleet claim is labeled implemented, retained evidence, operator assertion, or plan.", phase=0, kind="analysis", resource="coordinator", gate="G0", pairable=False, planning_state="in_progress")
    add("FND-002", "foundation", "Complete product requirements and actor contracts", "FND-001", 1.5, "API tenant, platform operator, compute provider, model engineer, file agent, and auditor responsibilities have no conflicting authority.", phase=0, kind="design", resource="coordinator", gate="G0", pairable=False, planning_state="in_progress")
    add("FND-003", "foundation", "Exact model and checkpoint registry", "FND-001", 1.5, "Qwen 3.8 27B, Qwen 3.8 Max, DSV4 Flash/Pro, GLM 5.2, K3, and MiniMax H3 have exact source, revision, tokenizer, and identity fields.", phase=0, gate="G0", lock="model_contracts/**")
    add("FND-004", "foundation", "Qualification and evidence receipt schema", "", 1.5, "Receipts bind commit, clean-tree state, checkpoint, recipe, artifacts, hardware, commands, output parity, timing boundary, and evidence class.", phase=0, gate="G0", lock="schema/qualification*.json")
    add("FND-005", "foundation", "Canonical control-plane object model", "FND-002,FND-003,FND-004", 2.0, "Model contract, recipe, build, equivalent-build release, immutable deployment spec, mutable deployment state, alias, active set, island, offer, lease, request, receipt, and ledger identities have explicit ownership.", phase=0, kind="design", resource="coordinator", gate="G0", pairable=False)
    add("FND-006", "foundation", "Versioning, errors, events, and consistency ADR", "FND-005", 1.5, "Every mutable object has one transactional writer and generation fence; replay, idempotency, compatibility, terminal errors, leader transfer, and append-before-publish rules have executable vectors.", phase=0, kind="design", resource="coordinator", gate="G0", pairable=False)
    add("FND-007", "foundation", "Requirements-to-evidence traceability matrix", "FND-002,FND-004", 1.5, "Every program requirement maps to implementation, qualification, operations, and external-gate work packages.", phase=0, gate="G0")
    add("FND-008", "foundation", "Program risk and decision register", "FND-002", 1.0, "Hardware, Ceph, model identity, ABI, artifact volume, legal, provider, performance, and integration risks have owners and trigger conditions.", phase=0, kind="design", resource="coordinator", gate="G0", pairable=False)
    add("FND-009", "foundation", "Subsystem ownership and write-lock map", "FND-005", 1.0, "Every shared ABI, runtime core, schema, gateway, scheduler, artifact, and release path has one integration owner and lock scope.", phase=0, gate="G0")
    add("FND-010", "foundation", "Architecture freeze gate", "FND-003,FND-004,FND-005,FND-006,FND-007,FND-008,FND-009,FND-011", 0.0, "All foundation contracts are reviewed and downstream tasks consume versioned interfaces instead of editing assumptions independently.", phase=0, kind="milestone", resource="coordinator", gate="G0", pairable=False)
    add("FND-011", "foundation", "Independent estimate calibration and resource forecast", "FND-007,FND-008", 1.5, "Owners supply independent optimistic/most-likely/pessimistic estimates, basis, variance, scarce-hardware calendars, lock scopes, and confidence; the heuristic baseline is replaced before a delivery commitment.", phase=0, kind="planning", resource="coordinator", gate="G0", pairable=False)

    # Codex-owned development harness, provider racing, and pair workflow.
    add("OXA-001", "agents", "Provider pool and secret-loading contract", "", 0.8, "Enabled providers, independent failure domains, authorization, models, headers, redaction, and environment-only secrets validate offline.", phase=0, gate="GS", lock="orchestration/oxalpha_providers*.json", planning_state="candidate_unintegrated")
    add("OXA-002", "agents", "Strict provider response and finish validation", "OXA-001", 1.0, "Malformed choice indices, finish reasons, tool calls, JSON, byte limits, and fractional configuration fields fail closed.", phase=0, gate="G1", lock="tools/oxalpha_race.py", planning_state="candidate_unintegrated")
    add("OXA-003", "agents", "Race worker and dispatcher exact accounting", "OXA-002", 1.2, "Every success, cancellation, exception, timeout, and close settles exactly once; shutdown is idempotent with no stranded thread or in-flight count.", phase=0, gate="G1", lock="tools/oxalpha_race.py", planning_state="ready")
    add("OXA-004", "agents", "Provider-neutral context and session journal", "OXA-002", 1.0, "Exact messages, tool results, provider responses, request IDs, hashes, and continuation state survive provider switches and restart.", phase=0, gate="G1", lock="tools/oxalpha_harness.py", planning_state="candidate_unintegrated")
    add("OXA-005", "agents", "Bounded tool sandbox and native-process ownership", "OXA-004", 1.5, "Declared tests are contained; PIDs/process groups are durable; restart terminates survivors; output, workspace, and session growth are bounded.", phase=0, gate="GS", lock="tools/oxalpha_harness.py", planning_state="ready")
    add("OXA-006", "agents", "Durable implementer/auditor pair controller", "OXA-004,OXA-005", 1.5, "One task moves through implementer, immutable patch, fresh auditor, rejection/resume, and coordinator-ready states without crash windows.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py", planning_state="candidate_unintegrated")
    add("OXA-007", "agents", "Fail-closed audit contract and final fingerprint", "OXA-006", 1.0, "Malformed findings cannot approve; the final captured patch and verified receipt share one unchanged workspace fingerprint and SHA-256.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py", planning_state="ready")
    add("OXA-008", "agents", "Retry, resume, cooldown, and controller recovery", "OXA-003,OXA-006,OXA-007", 1.5, "Injected 429/5xx/truncation/process death/controller death retries exact context and preserves the documented number of resume attempts.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py,tools/oxalpha_race.py")
    add("OXA-009", "agents", "Provider transport and useful-work scorecards", "OXA-003,OXA-008", 1.2, "Raw events and 1h/24h/7d/lifetime rollups report latency, validity, errors, wins, turn limits, task completion, audit approval, and coordinator acceptance.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py")
    add("OXA-010", "agents", "Real-time agent and queue dashboard", "OXA-006,OXA-009", 1.2, "Every pair shows current role/task/attempt/provider/tokens/retries, global queue depth, hardware waits, event lag, and stale-state age.", phase=0, gate="GX", lock="tools/oxalpha_fleet.py")
    add("OXA-011", "agents", "Scale and chaos qualification at 4, 8, and 32 pairs", "OXA-008,OXA-010", 2.0, "Long runs reserve and exercise 32 logical pairs times four provider requests, span at least two fresh independent failure domains, and survive injected provider and controller failures with no lost task, duplicate integration, leaked child, or false approval.", phase=0, gate="G7", uncertainty="high", provider_request_slots=128)
    add("OXA-012", "agents", "Massively parallel development launch gate", "OXA-007,OXA-011,OXA-014,OXA-015,OXA-016", 0.0, "The controller, racer, harness, dashboard, provider scorecards, executable-contract admission, live dispatch evaluator, and provider-diversity supply are auditor-approved and integrated before broad dispatch.", phase=0, kind="milestone", resource="coordinator", gate="G9", pairable=False)
    add("OXA-013", "agents", "Executable implementation/audit contract schema", "FND-007,FND-009,OXA-007", 1.2, "A dispatch contract requires objective, non-goals, exact write set, fixtures, commands, immutable outputs, hardware, timeouts, implementer receipt, independent auditor checks, integration owner, and rollback.", phase=0, gate="G1", lock="schema/agent_task_contract*.json")
    add("OXA-014", "agents", "WBS refinement and queue-admission gate", "OXA-010,OXA-013", 1.5, "Every PERT work package is decomposed into bounded executable contracts; schema, dependencies, disjoint writes, hardware lease, immutable patch, and auditor assignment validate before queue admission.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py,orchestration/task_contracts/**")
    add("OXA-015", "agents", "Recurring provider discovery and independence qualification", "OXA-009", 1.0, "A recurring scan records available OxAlpha endpoints, signup/key requirements, authorization, model identity, ownership/failure-domain independence, latency, validity, quotas, and safe enable/disable without routing through another provider.", phase=0, gate="GS", recurring_days=1, freshness_hours=24, lock="orchestration/oxalpha_providers*.json")
    add("OXA-016", "agents", "Live execution-state overlay and dispatch evaluator", "OXA-006,OXA-010,OXA-013", 1.2, "Durable planned/refining/implementing/patch-sealed/auditing/rejected/coordinator-review/integrated/blocked states combine dependency integration, admitted contract, write-lock lease, typed hardware quantity, four provider-request slots, at least two fresh independent provider failure domains, role ownership, queue depth, and stale-heartbeat age into one fail-closed dispatch decision.", phase=0, gate="G1", lock="tools/oxalpha_fleet.py,orchestration/execution_state*.json")

    # Selective salvage from main, unified, and the dsh reference checkout.
    add("REC-001", "reconciliation", "Path-level source classification manifest", "FND-001", 1.5, "Every divergent path is classified import, rewrite, evidence-only, archive, or reject with source SHA and destination owner.", phase=0, gate="G0", lock="orchestration/reconciliation*.json")
    add("REC-002", "reconciliation", "Qwen 3.6 to 3.8 27B identity reconciliation", "FND-003,REC-001", 1.5, "Contracts, modules, recipes, packs, tests, fleet registry, and receipts use one Qwen 3.8 27B identity; stale identities fail tests.", phase=1, gate="G1", lock="model_contracts/**,model-families/qwen38_27b/**")
    add("REC-003", "reconciliation", "Main API history and unified API diff", "REC-001", 1.0, "Useful API fixes are mapped by semantic unit while subprocess, single-client, and stale compatibility behavior is explicitly rejected.", phase=1, kind="analysis", gate="G0")
    add("REC-004", "reconciliation", "Hardware-interface source extraction manifest", "REC-001,FND-009", 1.2, "The frozen interface and ROCm candidates have file-level provenance, ABI conflicts, contaminants, and target gates.", phase=1, gate="G0")
    add("REC-005", "reconciliation", "StagePack and packer convergence manifest", "REC-001", 1.2, "Four family readers/writers and shared candidates are compared at byte, alignment, tensor, and error-semantics level.", phase=1, gate="G0")
    add("REC-006", "reconciliation", "KV, prefix, and backing convergence manifest", "REC-001", 1.2, "Duplicate arenas, page stores, prefix caches, backing stores, and model-private paths have one chosen disposition.", phase=1, gate="G0")
    add("REC-007", "reconciliation", "Speculation implementation reconciliation", "REC-001", 1.2, "DSpark, MTP, draft packs, verify, rollback, recurrent state, and distributed transport variants are mapped without mixing evidence.", phase=1, gate="G0")
    add("REC-008", "reconciliation", "Model-family fix and maturity manifest", "REC-001,FND-003", 1.5, "Each DSV4, GLM 5.2, K3, Qwen 3.8 Max, Qwen 3.8 27B, and MiniMax H3 change is tied to an exact model gate and current-main conflict review.", phase=1, gate="G0")
    add("REC-009", "reconciliation", "Contamination and generated-debris denylist", "REC-001", 0.8, "Build trees, binaries, logs, temporary roots, stale names, private workspaces, and branch-only receipts cannot enter imports.", phase=1, gate="G1")
    add("REC-010", "reconciliation", "Clean current-main baseline and compatibility inventory", "REC-002,REC-003,REC-004,REC-005,REC-006,REC-007,REC-008,REC-009,FND-004", 2.0, "A clean checkout records host gates, ABI versions, known failures, package identity, and no imported temporary artifacts.", phase=1, gate="G1", lock="shared-baseline")

    # Recipe language, precision, codecs, and sharding compiler.
    add("RCP-001", "recipes", "Canonical model-contract schema", "FND-003,FND-010", 1.5, "Exact geometry, tensor names, tokenizer, attention, recurrent state, experts, position encoding, reference precision, and checkpoint hashes validate.", phase=1, gate="G0", lock="schema/model_contract*.json")
    add("RCP-002", "recipes", "Model-instance recipe schema", "RCP-001,FND-005", 1.5, "Precision, codecs, topology, batch/context, KV, target, promotion, co-residency, exclusivity, and qualification policies reject unknown fields.", phase=1, gate="G0", lock="schema/model_recipe*.json")
    add("RCP-003", "recipes", "Canonical recipe hash and mnemonic catalog", "RCP-002", 1.0, "Cross-language hash vectors are stable; aliases are revisioned; collisions, stale revisions, and semantic misspellings fail closed.", phase=1, gate="G1")
    add("RCP-004", "recipes", "Qualified full-precision compute policy", "RCP-001", 1.0, "The full-precision class requires a BF16/FP16 spine with FP32 accumulation/sensitive reductions; source weights, dequantized effective weights, storage error, and compute route are separate, and every narrower route gets a distinct reduced-precision build identity.", phase=1, gate="G0")
    codec_dependencies: list[str] = []
    for offset, bits in enumerate((4, 5, 6, 7, 8, 16), start=5):
        task_id = f"RCP-{offset:03d}"
        codec_dependencies.append(task_id)
        add(task_id, "recipes", f"{bits}-bit weight-storage codec contract", "RCP-004", 1.2 if bits not in (5, 7) else 1.8, f"Packing, byte order, group/scale geometry, exceptional values, deterministic dequantization, round trip, and numerical fixtures exist for {bits}-bit storage.", phase=1, gate="G1", uncertainty="high" if bits in (5, 7) else "normal", lock="include/sparkpipe/spark_weight_codec.h,runtime/pack/**")
    add("RCP-011", "recipes", "Calibration and codec provenance", codec_dependencies, 1.5, "Every quantized tensor block records calibration source, scales, error bars, codec version, checkpoint, and reproducible generator identity.", phase=1, gate="G1")
    add("RCP-012", "recipes", "Parallelism-axis schema", "RCP-002", 1.0, "DP, TP, PP, EP, ETP, CP, SP, KVP, prefill/decode, and interleave axes have explicit degree, nesting, and placement semantics.", phase=1, gate="G0")
    add("RCP-013", "recipes", "Tensor and scale-group sharding legality core", "RCP-001,RCP-012", 1.8, "Illegal splits of tensors, heads, experts, recurrent state, and codec groups produce deterministic source-level diagnostics.", phase=1, gate="G1")
    add("RCP-014", "recipes", "TP and PP placement compiler", "RCP-013", 1.8, "TP2/4/8/16 and PP2/4/8/16 produce exact rank/stage manifests or a reasoned rejection.", phase=2, gate="G1")
    add("RCP-015", "recipes", "EP and ETP placement compiler", "RCP-013", 1.8, "Expert ownership, replication, all-to-all, shared experts, and expert tensor splits are legal and byte-accounted.", phase=2, gate="G1")
    add("RCP-016", "recipes", "CP, SP, and explicit KVP compiler", "RCP-013", 2.0, "Context/sequence/KV ownership reports physical bytes per rank and any intentional replication.", phase=2, gate="G1")
    add("RCP-017", "recipes", "DP, disaggregated prefill/decode, and interleave compiler", "RCP-013", 1.8, "Replica, stage-boundary, prefill/decode, and virtual-pipeline plans include transport and consistency contracts.", phase=2, gate="G1")
    add("RCP-018", "recipes", "Batch-context-memory capacity envelope compiler", "RCP-004,RCP-014,RCP-015,RCP-016,RCP-017,KV-001", 2.0, "Every build publishes legal B1-B1024/context cells, active/parked KV, workspace, weights, and explicit rejected cells without extrapolation.", phase=2, gate="G1")
    add("RCP-019", "recipes", "Deterministic recipe-to-build compiler", "RCP-003,RCP-011,RCP-014,RCP-015,RCP-016,RCP-017,RCP-018,TOP-006,HWI-005", 3.0, "One recipe resolves exact topology, shards, low-level descriptions, kernels, collectives, capacity, and provenance; every fitting TP/PP/EP tuple over degrees 1/2/4/8/16 deterministically builds or returns a byte-proved geometry/capacity/topology rejection.", phase=2, gate="G1", uncertainty="high", lock="tools/generate_recipe.py,runtime/pack/**")
    add("RCP-020", "recipes", "Recipe recovery and mnemonic equivalence gate", "RCP-019,ART-010", 1.5, "A clean cache rebuild or catalog recovery resolves the same immutable objects and alias revision without hidden local state.", phase=3, gate="G7")

    # Artifact DAG, Ceph redesign, rank-local placement, and file agent.
    add("ART-001", "artifacts", "Immutable artifact DAG schema", "FND-005,RCP-002", 1.5, "Checkpoint blocks, normalized tensors, codecs, topology slices, rank packages, target kernels, builds, releases, and receipts have typed immutable edges.", phase=1, gate="G0")
    add("ART-002", "artifacts", "Exact checkpoint importer", "ART-001,RCP-001", 1.5, "Importer verifies source revision, file list, hashes, tokenizer, config, incomplete names, and resumable source acquisition.", phase=2, gate="G1")
    add("ART-003", "artifacts", "Normalized tensor-block store", "ART-002,ART-017", 1.8, "Canonical dtype, shape, byte order, alignment, names, block hashes, and slice boundaries are independent of downstream topology and are generated only after the signed pre-variant storage budget.", phase=2, gate="G1")
    add("ART-004", "artifacts", "Codec-block factory", "ART-003,RCP-011", 2.0, "Codec variants reuse normalized blocks, produce deterministic bytes and receipts, and do not decode unrelated tensors to slice one block.", phase=2, gate="G1")
    add("ART-005", "artifacts", "Canonical StagePack reader and writer", "ART-004,REC-005", 2.5, "Shared reader/writer preserves approved legacy fixtures, rejects malformed offsets, and removes family-private semantic drift.", phase=2, gate="G1", lock="runtime/pack/**,modules/*/source/*stagepack*")
    add("ART-006", "artifacts", "Topology slicer", "ART-005,RCP-019", 2.2, "The compiler emits exact rank/stage/expert/KV slices and verifies total tensor coverage without overlap or omission.", phase=2, gate="G1")
    add("ART-007", "artifacts", "Rank-local package builder", "ART-006", 2.0, "Each package contains only required tensors, scales, tokenizer/build metadata, target artifacts, and complete per-file hashes.", phase=2, gate="G1")
    add("ART-008", "artifacts", "Target kernel and link-unit artifact builder", "ART-007,HWI-005", 2.2, "CUDA, ROCm, Metal, and CPU artifacts bind backend capability, compiler, flags, ABI, source, and recipe identity.", phase=2, gate="G2", uncertainty="high")
    add("ART-009", "artifacts", "Compiled build and release manifest", "ART-007,ART-008", 1.5, "Manifest binds every rank package, library, topology, capacity envelope, receipt, and exact source generation.", phase=2, gate="G1")
    add("ART-010", "artifacts", "Artifact catalog and provenance query core", "ART-009", 2.0, "Catalog traverses provenance, availability, reuse, pins, qualification, revocation, and rebuild paths without trusting filenames.", phase=2, gate="G1")
    add("ART-011", "artifacts", "Ceph 14+2 warm-tier redesign and safety gate", "ART-001,SEC-001", 2.0, "The proposed 48 TB tier resolves the current stopped/masked operational conflict and has physical devices, failure model, namespace, rebuild, and benchmark plan before enablement.", phase=2, kind="design", resource="coordinator", hardware="ceph", gate="GS", uncertainty="high", pairable=False)
    add("ART-012", "artifacts", "Warm-cache materialization and pin planner", "ART-010,ART-011", 2.0, "Planner chooses exhaustive versus lazy variants from demand, rebuild cost, capacity, failure domains, and no-delete policy.", phase=2, gate="G1")
    add("ART-013", "artifacts", "Physical node-storage inventory", "TOP-004", 1.2, "Every storage claim binds block device, filesystem, mountpoint, capacity, free space, inodes, health, and active references.", phase=2, hardware="spark_storage", gate="G7")
    add("ART-014", "artifacts", "Signed placement planner and per-host file-executor contract", "ART-007,ART-013,ART-017", 1.5, "The global planner cannot touch files; signed plans bind host, physical device, scoped mount root, build/rank manifest, nonce, generation, lease, expiry, byte ceiling, exact operations, temporary peaks, protected files, and no implicit deletion.", phase=2, gate="GS")
    add("ART-015", "artifacts", "Per-host resumable transfer, verify, and atomic publish", "ART-014", 2.0, "A privileged host-local executor rejects stale, unsigned, wrong-device, and out-of-root plans; journals idempotent restart, resumes verified ranges, publishes only complete hash-verified rank manifests, and emits signed local receipts.", phase=3, hardware="spark_storage", gate="G7")
    add("ART-016", "artifacts", "Catalog-driven rank-local materializer", "ART-012,ART-015", 2.5, "Source-to-warm-to-rank materialization is idempotent, generation-fenced, provenance-complete, and recovers after process or node loss.", phase=3, hardware="spark_storage", gate="G7", uncertainty="high")
    add("ART-017", "artifacts", "Pre-variant per-Spark one-terabyte budget and deduplication plan", "ART-002,RCP-018,ART-013", 2.0, "Before any fleet placement move or normalized/codec/topology/rank shard generation, the checkpoint manifest and compiler envelope prove invariant reuse, desired rank-package estimates, temporary peaks, protected sources, no-delete migration, the one-terabyte model target, and remaining KV bytes per physical Spark.", phase=2, hardware="spark16_storage", gate="G8")
    add("ART-018", "artifacts", "Post-placement artifact and storage verification dashboard", "ART-016,ART-017,OBS-002", 1.5, "Desired/observed shards, hashes, warming, drift, physical free space, temporary peaks, actual versus planned model budget, and KV capacity are visible per node.", phase=3, gate="GX")
    add("ART-019", "artifacts", "Retention, legal hold, and authorized GC planner", "ART-010,SEC-004", 1.5, "Reachability and retention produce dry-run candidates only; deletion needs separate authorization, exact targets, and recovery evidence.", phase=4, gate="GS")
    add("ART-020", "artifacts", "Clean-cache build and disaster recovery", "RCP-020,ART-016,HA-002", 2.5, "A lost local catalog/cache is rebuilt from immutable source and backup with equivalent hashes, aliases, and qualification links.", phase=5, hardware="ceph,spark_storage", gate="GH")

    # Portable hardware interface and CPU reference behavior.
    add("HWI-001", "hardware", "Hardware capability schema", "FND-010", 1.5, "CUDA, ROCm, Metal, and CPU targets describe memory, matrix primitives, subgroup width, graphs, transfers, collectives, telemetry, and firmware ABI.", phase=1, gate="G0", lock="schema/hardware_capability*.json")
    add("HWI-002", "hardware", "Device and topology identity contract", "HWI-001", 1.0, "Target keys, device identity, local peers, fabrics, NUMA, storage locality, and capability generations are immutable and probe-verifiable.", phase=1, gate="G0")
    add("HWI-003", "hardware", "Portable memory, queue, event, and completion ABI", "HWI-001", 2.0, "Allocation, pinned/shared memory, asynchronous copies, queues/streams, events, graphs, and caller-owned completion work without vendor types in public headers.", phase=1, gate="G1", lock="include/sparkpipe/spark_hw_iface.h")
    add("HWI-004", "hardware", "Portable execution-island primitive descriptors", "HWI-001,RCP-004", 2.0, "GEMM, attention, normalization, activation, routing, MoE, codec, sampling, and KV primitives use fixed shape/precision descriptors.", phase=1, gate="G1", lock="include/sparkpipe/spark_hw_iface.h")
    add("HWI-005", "hardware", "Frozen model-device hardware interface", "HWI-002,HWI-003,HWI-004,REC-004", 2.5, "The reviewed DSV4-first interface is ported without branch debris and has explicit ownership, versioning, and no hot-path vendor branching.", phase=1, gate="G1", lock="include/sparkpipe/spark_hw_iface.h")
    add("HWI-006", "hardware", "Vendor-free portable-header conformance", "HWI-005", 1.0, "Portable headers compile as C/C++ on Linux and macOS without CUDA, HIP, Metal, or framework includes.", phase=1, gate="G1")
    add("HWI-007", "hardware", "Backend telemetry and profiler interface", "HWI-005,OBS-001", 1.5, "Backends report clocks, power, temperature, memory, queue, kernel, transfer, and collective evidence with bounded labels.", phase=2, gate="G1")
    add("HWI-008", "hardware", "CPU reference backend", "HWI-005", 2.0, "A slow deterministic backend exercises memory, execution descriptors, KV, sampling, and error behavior as a correctness oracle.", phase=2, hardware="cpu", gate="G3")
    add("HWI-009", "hardware", "Cross-backend conformance suite", "HWI-006,HWI-008", 2.0, "Mock and CPU backends pass lifecycle, bounds, precision, completion, cancellation, and malformed-descriptor tests shared by device backends.", phase=2, gate="G1")
    add("HWI-010", "hardware", "Hardware-interface freeze gate", "HWI-007,HWI-009", 0.0, "CUDA, ROCm, and Metal work consume a reviewed interface and add backend-private specialization without changing portable semantics.", phase=2, kind="milestone", resource="coordinator", gate="G1", pairable=False)

    # NVIDIA/GB10 remains the reference production backend.
    add("CUDA-001", "cuda", "Clean current-main CUDA baseline", "REC-010", 1.0, "Exact supported host and Spark tests, model receipts, compiler versions, SM target, clocks, and known failures are frozen before HAL extraction.", phase=1, hardware="cuda_sm121", gate="G2")
    add("CUDA-002", "cuda", "GB10 SM121 capability adapter", "HWI-010,CUDA-001", 1.5, "Probe emits exact GB10 capability generation and rejects unsupported shapes/formats instead of guessing.", phase=2, hardware="cuda_sm121", gate="G2")
    add("CUDA-003", "cuda", "CUDA memory, stream, event, and completion backend", "HWI-003,CUDA-002", 2.0, "Common lifecycle, asynchronous ordering, graph safety, cancellation, and cleanup tests pass on SM121.", phase=2, hardware="cuda_sm121", gate="G3")
    add("CUDA-004", "cuda", "CUDA graph and launch-descriptor backend", "CUDA-003", 1.8, "Graph capture/replay covers declared batch buckets with stable addresses and fails safely on unsupported descriptors.", phase=2, hardware="cuda_sm121", gate="G3")
    add("CUDA-005", "cuda", "CUDA GEMM and linear island registry", "HWI-004,CUDA-002", 2.0, "Production shapes select measured native implementations with fixed descriptors and no runtime framework fallback.", phase=2, hardware="cuda_sm121", gate="G3")
    add("CUDA-006", "cuda", "CUDA attention, RoPE, and KV island registry", "HWI-004,CUDA-002,KV-002,KV-018", 2.2, "Attention variants and every declared CUDA KV format match reference output, byte accounting, and capability rejection over named production shapes and context boundaries.", phase=2, hardware="cuda_sm121", gate="G3")
    add("CUDA-007", "cuda", "CUDA MoE, router, and codec island registry", "HWI-004,CUDA-002,RCP-011", 2.5, "Expert dispatch/combine, top-k, shared experts, and fused dequantization satisfy model accuracy and shape contracts.", phase=2, hardware="cuda_sm121", gate="G3", uncertainty="high")
    add("CUDA-008", "cuda", "CUDA NCCL, host-RDMA, and GPUDirect integration", "COL-005,CUDA-003", 2.5, "Transport selection, ordering, rank identity, timeout, cancellation, and all-rank completion pass on qualified fabrics.", phase=3, hardware="cuda4", gate="G6")
    add("CUDA-009", "cuda", "CUDA profiler and no-drop timing hooks", "CUDA-003,HWI-007,PERF-003", 1.5, "Profiles report GPU busy union, exposed idle, kernels, transfers, collectives, host gaps, and overlap without changing output or material timing.", phase=2, hardware="cuda_sm121", gate="G8")
    add("CUDA-010", "cuda", "CUDA zero-regression HAL gate", "CUDA-001,CUDA-004,CUDA-005,CUDA-006,CUDA-007,CUDA-008,CUDA-009,HWI-009", 3.0, "Retained production shapes preserve numerical output and meet declared end-to-end regression thresholds after interface extraction.", phase=3, hardware="cuda4", gate="G8", uncertainty="high")

    # AMD ROCm, first targeting gfx950/MI350-class hardware.
    add("AMD-001", "amd", "MI350/gfx950 target and toolchain contract", "HWI-001", 1.2, "Exact GPU target, ROCm/hipcc versions, supported precision, memory, graph, RCCL, and profiler requirements are pinned.", phase=1, gate="G0")
    add("AMD-002", "amd", "Reconcile duplicate unified ROCm island trees", "REC-004,AMD-001", 1.5, "One clean import plan excludes binaries, vendored debris, duplicate islands, host-only evidence, and unsupported readiness claims.", phase=1, gate="G0")
    add("AMD-003", "amd", "ROCm host and gfx950 compile CI", "AMD-001,AMD-002,CI-001,HWI-010", 2.0, "Portable host tests and real gfx950 device-code compilation run separately and preserve exact compiler receipts.", phase=2, hardware="rocm_toolchain", gate="G2")
    add("AMD-004", "amd", "MI350 capability and topology probe", "AMD-003,HWI-002", 1.5, "A real device records capabilities, memory, peer links, NUMA, RCCL, clocks, power, and unsupported operations.", phase=2, hardware="mi350", gate="G2", planning_state="blocked_hardware")
    add("AMD-005", "amd", "HIP memory, queues, events, and completion", "AMD-004,HWI-003", 2.5, "Common lifecycle and asynchronous ordering tests pass on MI350 with no CUDA compatibility assumption.", phase=2, hardware="mi350", gate="G3", uncertainty="high", planning_state="blocked_hardware")
    add("AMD-006", "amd", "HIP graph and fixed-launch backend", "AMD-005", 2.0, "Graph capture/replay or an explicit qualified alternative covers declared batch descriptors without hot-path allocation.", phase=2, hardware="mi350", gate="G3", uncertainty="high", planning_state="blocked_hardware")
    add("AMD-007", "amd", "ROCm BF16/FP16 spine and FP32 accumulator GEMM", "AMD-004,HWI-004,RCP-004", 3.0, "Production dense and projection shapes match the full-precision policy and reference tolerances on MI350.", phase=2, hardware="mi350", gate="G3", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-008", "amd", "ROCm FP8 storage/dequant execution", "AMD-007,RCP-009", 2.5, "FP8 layouts, scales, fused loads, accumulation, and model slices pass codec and accuracy gates.", phase=2, hardware="mi350", gate="G3", uncertainty="high", planning_state="blocked_hardware")
    add("AMD-009", "amd", "ROCm MXFP4 and packed low-bit execution", "AMD-007,RCP-005,RCP-011", 3.0, "Native or explicitly emulated packed execution has exact layout, memory, numerical, and performance receipts.", phase=2, hardware="mi350", gate="G3", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-010", "amd", "ROCm attention, RoPE, and KV islands", "AMD-007,KV-002,KV-018", 3.0, "DSV4 attention variants and every declared ROCm KV format match reference, byte accounting, and explicit capability rejection across production context shapes.", phase=3, hardware="mi350", gate="G4", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-011", "amd", "ROCm routing, MoE, dispatch, and combine", "AMD-008,AMD-009", 3.5, "Expert selection, grouped GEMM, shared experts, dispatch, combine, and router precision match reference layers.", phase=3, hardware="mi350", gate="G4", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-012", "amd", "RCCL collective backend", "AMD-005,COL-005", 2.5, "TP collectives pass ordering, precision, timeout, cancellation, and rank-loss tests on four AMD devices.", phase=3, hardware="mi350x4", gate="G6", uncertainty="high", planning_state="blocked_hardware")
    add("AMD-013", "amd", "DSV4 one-layer MI350 numerical gate", "AMD-010,AMD-011,ART-008", 3.0, "One real-weight complete layer matches the authoritative implementation with exact artifact and hardware identity.", phase=3, hardware="mi350", gate="G4", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-014", "amd", "DSV4 TP1 full-model MI350 gate", "AMD-006,AMD-013,RT-018,KV-010", 4.0, "Real-pack TP1 prefill/decode completes full-model numerical, lifecycle, memory, API, and soak tests.", phase=4, hardware="mi350", gate="G5", uncertainty="research", planning_state="blocked_hardware")
    add("AMD-015", "amd", "DSV4 ROCm TP4 and performance qualification", "AMD-012,AMD-014,PERF-006", 5.0, "RCCL TP4 full-model parity, throughput, latency, power, capacity envelope, and zero-drift release receipts pass.", phase=4, hardware="mi350x4", gate="G8", uncertainty="research", planning_state="blocked_hardware")

    # Apple Silicon is a complete product backend, not only a host build target.
    add("MET-001", "metal", "Apple-Silicon SoC support matrix", "HWI-001", 1.2, "Supported Mac families, memory budgets, Metal versions, tensor features, power counters, and minimum OS/toolchain are explicit.", phase=1, gate="G0")
    add("MET-002", "metal", "macOS and Metal compiler CI", "MET-001,CI-001,HWI-010", 1.8, "Portable C/C++ plus MSL compilation run on Apple Silicon and distinguish compile evidence from hardware execution.", phase=2, hardware="apple_silicon", gate="G2")
    add("MET-003", "metal", "Metal capability, memory, and topology probe", "MET-002,HWI-002", 1.5, "Real Macs report device family, unified memory, recommended working set, threadgroup/tensor features, counters, and peer/network topology.", phase=2, hardware="apple_silicon", gate="G2")
    add("MET-004", "metal", "Metal buffers, command queues, events, and completion", "MET-003,HWI-003", 2.5, "Common allocation, unified-memory accounting, transfer, queue ordering, completion, cancellation, and cleanup tests pass.", phase=2, hardware="apple_silicon", gate="G3", uncertainty="high")
    add("MET-005", "metal", "Metal command-buffer and fixed-dispatch plans", "MET-004", 2.0, "Pre-encoded command buffers or a qualified alternative cover declared batch shapes without request-path discovery or allocation.", phase=2, hardware="apple_silicon", gate="G3", uncertainty="high")
    add("MET-006", "metal", "MSL 16-bit spine and FP32 accumulation primitives", "MET-003,HWI-004,RCP-004", 3.0, "BF16/FP16 operand routes with FP32 accumulation/reductions pass tensor and layer accuracy gates on each supported SoC class.", phase=2, hardware="apple_silicon", gate="G3", uncertainty="research")
    add("MET-007", "metal", "Metal packed-weight codec kernels", "MET-006,RCP-011", 3.0, "Required 4/5/6/7/8/16-bit storage formats load deterministically; unsupported formats are rejected by capability and recipe compilation.", phase=2, hardware="apple_silicon", gate="G3", uncertainty="research")
    add("MET-008", "metal", "Metal dense GEMM, norm, activation, and residual islands", "MET-006", 3.0, "Qwen 3.8 27B production shapes match CPU/reference output and report measured memory and timing.", phase=3, hardware="apple_silicon", gate="G4", uncertainty="research")
    add("MET-009", "metal", "Metal attention, RoPE, and paged-KV islands", "MET-008,KV-002,KV-018", 3.5, "Prefill/decode attention and every declared Metal KV format match reference, byte accounting, and capability rejection across context boundaries and memory pressure.", phase=3, hardware="apple_silicon", gate="G4", uncertainty="research")
    add("MET-010", "metal", "Metal routing and MoE islands", "MET-007,MET-008", 3.5, "Router precision, top-k, expert dispatch/combine, and grouped low-bit execution pass selected model-layer gates.", phase=3, hardware="apple_silicon", gate="G4", uncertainty="research")
    add("MET-011", "metal", "Multi-Mac transport and collective backend", "MET-004,COL-005", 3.0, "The qualified network path reports topology, ordering, failure behavior, numerical parity, and measured crossover; unsupported collectives fail early.", phase=3, hardware="apple_silicon_multi", gate="G6", uncertainty="research")
    add("MET-012", "metal", "Qwen 3.8 27B complete-layer oracle", "MET-008,MET-009,ART-008", 3.0, "A real-weight layer matches the authoritative reference with exact recipe, codec, kernel, and SoC identity.", phase=3, hardware="apple_silicon", gate="G4", uncertainty="research")
    add("MET-013", "metal", "Qwen 3.8 27B single-Mac full-model gate", "MET-005,MET-012,RT-018,KV-010", 5.0, "A named supported Mac class completes full Qwen 3.8 27B prefill/decode with numerical, lifecycle, memory, context, and soak evidence; qualified low-bit weight storage is allowed but no smaller substitute can pass this gate.", phase=4, hardware="apple_silicon", gate="G5", uncertainty="research")
    add("MET-014", "metal", "Mac API and compute-island integration", "MET-013,API-013,SCH-020", 3.0, "A Mac advertises a signed island offer and serves authenticated streaming requests with correct usage, cancellation, and policy.", phase=4, hardware="apple_silicon", gate="G7", uncertainty="high")
    add("MET-015", "metal", "Metal v1 performance, power, and release qualification", "MET-011,MET-014,PERF-006,REL-004", 4.0, "Metal v1 passes the shared primitive suite, Qwen 3.8 27B full-model/API performance on a named Mac class, two-Mac collective correctness, power, capacity, thermal soak, and zero-drift release; other model/SoC cells remain unclaimed.", phase=5, hardware="apple_silicon", gate="G9", uncertainty="research")

    # Physical topology, live inventory, transport, and collectives.
    add("TOP-001", "topology", "Heterogeneous topology graph schema", "HWI-002", 1.5, "Nodes, devices, ports, links, rails, storage, NUMA, trust, provider, geography, and failure domains validate in one graph.", phase=1, gate="G0")
    add("TOP-002", "topology", "Node resource and deployment-demand schemas", "TOP-001", 1.2, "Fungible quantities and non-fungible capabilities represent weights, workspace, active/parked KV, bandwidth, queues, power, trust, and locality.", phase=1, gate="G0")
    add("TOP-003", "topology", "Failure-domain and locality semantics", "TOP-001", 1.2, "Device, host, rack, switch, provider, region, power, storage, and upstream failure domains have explicit independence rules.", phase=1, gate="G0")
    add("TOP-004", "topology", "Signed live node-inventory agent", "TOP-002,SEC-002", 2.5, "Renewable snapshots bind hardware, software, topology, storage, telemetry, nonce, generation, expiry, and signature; replay and stale agents fail.", phase=2, gate="GS")
    add("TOP-005", "topology", "Inventory history and capacity projection", "TOP-004", 1.5, "The control plane retains bounded capacity/health history and forecasts fit without treating stale inventory as available.", phase=2, gate="G1")
    add("TOP-006", "topology", "Deterministic locality-group enumerator", "TOP-002,TOP-003", 2.0, "Candidate groups satisfy every collective, stage, storage, trust, and failure-domain edge with deterministic diagnostics.", phase=2, gate="G1")
    add("TOP-007", "topology", "Fabric and PMTU probe framework", "TOP-004", 2.0, "Management, switched, direct, RDMA, and peer-memory paths are distinguished and produce route, MTU, duplex, latency, bandwidth, and error receipts.", phase=2, hardware="fleet_network", gate="G7")
    add("TOP-008", "topology", "Fleet compute-island formation", "TOP-006,TOP-007", 2.0, "Locally managed hardware forms an island only when model-stage latency, topology, storage, trust, and control ownership are qualified.", phase=2, gate="G1")

    add("COL-001", "collectives", "Collective semantics and ownership contract", "HWI-005", 1.5, "Dtypes, accumulation, ordering, buffers, completion, timeout, cancellation, rank loss, generation, and restart have one owner.", phase=1, gate="G0", lock="include/sparkpipe/spark_tp_device_collective.h")
    add("COL-002", "collectives", "Deterministic host reference collectives", "COL-001", 1.5, "All-reduce, reduce-scatter, all-gather, all-to-all, broadcast, and point-to-point reference fixtures pass positive and negative cases.", phase=2, hardware="host", gate="G1")
    add("COL-003", "collectives", "Topology-aware collective algorithm selector", "COL-001,TOP-006", 2.0, "Small, medium, large, sparse expert, pipeline, and KV payloads select a legal measured family with explicit crossover data.", phase=2, gate="G1")
    add("COL-004", "collectives", "Portable asynchronous collective backend ABI", "COL-001,HWI-005", 2.0, "NCCL, RCCL, Metal/network, host RDMA, GPUDirect, and reference backends share exact descriptors and completion semantics.", phase=2, gate="G1", lock="include/sparkpipe/spark_tp_device_collective.h")
    add("COL-005", "collectives", "NCCL, RCCL, Metal, TCP, and RDMA backend adapters", "COL-003,COL-004,TOP-007", 3.0, "Each adapter reports capability and passes backend-specific compile/lifecycle tests without contaminating portable headers.", phase=2, hardware="multi_backend", gate="G2", uncertainty="high")
    add("COL-006", "collectives", "Deterministic reduction and precision qualification", "COL-002,COL-005,RCP-004", 2.0, "Reduction order and accumulation meet declared numerical policy across supported degrees and backends.", phase=3, hardware="multi_backend", gate="G6")
    add("COL-007", "collectives", "Timeout, cancellation, rank-loss, and restart behavior", "COL-005", 2.5, "Injected process/network loss terminates or recovers every rank without hanging, cross-generation completion, or leaked gang resources.", phase=3, hardware="multi_backend", gate="G7", uncertainty="high")
    add("COL-008", "collectives", "Concurrent full-duplex fabric qualification", "COL-005,TOP-007", 2.0, "Every required pair sustains simultaneous bidirectional traffic at the named floor with raw per-direction receipts.", phase=3, hardware="fleet_network", gate="G8")
    add("COL-009", "collectives", "Cross-stage and disaggregated prefill/decode transport", "COL-005,RCP-017", 2.5, "Stage-homogeneous boundaries transfer activations/KV/state with exact identity, backpressure, deadline, and measured latency.", phase=3, hardware="multi_backend", gate="G6", uncertainty="high")
    add("COL-010", "collectives", "Collective and transport release gate", "COL-006,COL-007,COL-008,COL-009", 0.0, "Named topologies have numerical, duplex, timeout, cancellation, rank-loss, and restart receipts before model release use.", phase=3, kind="milestone", resource="coordinator", hardware="multi_backend", gate="G6", pairable=False)

    # Shared inference runtime: static execution descriptors, no hot-path discovery.
    add("RT-001", "runtime", "Driver/module/adapter/runtime ownership ADR", "FND-008,FND-009", 1.5, "Initialization, validation, admission, batching, KV translation, speculation commit, completion, cancellation, and release each have one owner.", phase=1, kind="design", resource="coordinator", gate="G0", pairable=False)
    add("RT-002", "runtime", "Model driver, module, adapter, and frame ABI freeze", "RT-001,REC-010", 2.0, "Version matrix, ownership, buffer lifetimes, shapes, status codes, and compatibility rules are pinned with negative tests.", phase=1, gate="G1", lock="include/sparkpipe/spark_model*.h,include/sparkpipe/spark_module_abi.h")
    add("RT-003", "runtime", "Driver compiler, loader, and unload qualification", "RT-002,ART-009", 2.0, "Generated source, shared object, hashes, ABI, dependencies, loading, unload, wrong-build rejection, and restart pass.", phase=2, gate="G1")
    add("RT-004", "runtime", "Shared StagePack runtime ingestion", "RT-002,ART-005", 2.0, "Runtime maps only manifested rank tensors, validates every range/format, and performs no family-private discovery in request paths.", phase=2, gate="G1", lock="runtime/stage_module_common.c")
    add("RT-005", "runtime", "Shared serving-adapter core", "RT-002,REC-008", 2.5, "At least two model families use common lifecycle, frame, cancellation, completion, and configuration code without numerical drift.", phase=2, gate="G1", lock="runtime/model_serving_adapter.c")
    add("RT-006", "runtime", "Resident daemon lifecycle and resource ownership", "RT-003,RT-005", 2.0, "Start, validate, load, prewarm, ready, drain, standby, destroy, socket closure, and crash cleanup are idempotent.", phase=2, gate="G7", lock="node/model_residentd.c")
    add("RT-007", "runtime", "Durable request state and continuation lease", "RT-006,FND-006", 2.0, "Request/attempt/sequence/build/allocation identities survive restart; stale continuations and duplicate commits fail.", phase=2, gate="G1")
    add("RT-008", "runtime", "Admission, completion, cancellation, and release invariants", "RT-007", 2.2, "Saturation, refusal, cancellation, terminal events, retries, and release pass leak, ordering, and exactly-once tests.", phase=2, gate="G1", lock="runtime/model_batch_engine.c")
    add("RT-009", "runtime", "Caller-owned memory pools and fixed launch descriptors", "RT-002,HWI-003", 2.0, "Steady-state request execution performs no heap allocation, module search, filesystem lookup, graph interpretation, or hardware probe.", phase=2, gate="G1")
    add("RT-010", "runtime", "Chunked multi-sequence prefill engine", "RT-004,RT-008,RT-009", 2.5, "Prefill chunks obey row/context limits, preserve sequence state, batch compatible work, and report exact progress/cancellation.", phase=3, gate="G3", lock="runtime/model_batch_engine.c")
    add("RT-011", "runtime", "Continuous decode engine", "RT-004,RT-008,RT-009", 2.5, "Decode forms legal dynamic batches, preserves per-sequence sampling/state, emits ordered tokens, and handles lane completion without stalls.", phase=3, gate="G3", lock="runtime/model_batch_engine.c")
    add("RT-012", "runtime", "Tokenizer and chat-template package runtime", "RCP-001", 2.0, "Pinned tokenizer/template packages reproduce authoritative token IDs, prompt rendering, special tokens, and incremental detokenization.", phase=2, gate="G1")
    add("RT-013", "runtime", "Sampling, RNG, logprob, and stop contract", "RT-012,RCP-004", 2.5, "Greedy, temperature, top-k/top-p, penalties, seed, logprobs, stop sequences, and distributed determinism have golden fixtures.", phase=2, gate="G1")
    add("RT-014", "runtime", "B1-B1024 continuous batch runtime", "RT-010,RT-011,RT-013,RCP-018", 3.0, "Mixed prefill/decode offered load fills supported buckets, rejects unsupported cells, and preserves fairness and output identity.", phase=3, gate="G7", lock="runtime/model_batch_engine.c", uncertainty="high")
    add("RT-015", "runtime", "Model-neutral speculation and rollback core", "RT-014,REC-007,KV-003", 3.0, "Draft/verify, accept/reject, KV/recurrent commit, rollback, sampling, and fallback equal the non-spec authoritative trajectory.", phase=3, gate="G4", uncertainty="high")
    add("RT-016", "runtime", "Multi-model co-residency execution seam", "RT-006,RT-014", 2.0, "Compatible models have independent ports, artifacts, KV, memory budgets, gangs, and performance-isolation policy; exclusive models drain safely.", phase=3, gate="G7")
    add("RT-017", "runtime", "No-drop runtime performance instrumentation", "RT-014,PERF-003", 1.5, "Request, prefill, decode, sampling, KV, host, collective, and idle spans correlate without unbounded labels or material slowdown.", phase=3, gate="G8")
    add("RT-018", "runtime", "Shared runtime conformance gate", "RT-003,RT-004,RT-005,RT-006,RT-008,RT-009,RT-014,RT-015,RT-016,RT-017,COL-010", 0.0, "Host and target lifecycle, numerical, failure, batching, speculation, co-residency, and instrumentation gates are green.", phase=3, kind="milestone", resource="coordinator", gate="G7", pairable=False)

    # KV, recurrent state, prefix sharing, and the 2.5 TB parked-session tier.
    add("KV-001", "kv", "KV/state ownership, precision, and byte contract", "RCP-004", 1.5, "Logical and physical bytes/token/rank, formats, scale geometry, recurrent state, replication, active capacity, and backing horizon are computed.", phase=1, gate="G0")
    add("KV-002", "kv", "Page geometry and layout-hash contract", "KV-001", 1.2, "Page tokens, block bytes, alignment, head/latent ownership, scales, model build, and layout hash have golden vectors.", phase=1, gate="G1")
    add("KV-003", "kv", "Recurrent, convolution, and speculation-state contract", "KV-001", 1.5, "Non-attention state has explicit ownership, snapshot, commit, rollback, paging, restore, and identity semantics.", phase=1, gate="G1")
    add("KV-017", "kv", "Selectable KV-resolution and scale schema", "KV-001", 1.5, "Recipes enumerate key and value formats independently, initially BF16, FP16, FP8 E4M3, FP8 E5M2, and INT8 where capable, with scale/group geometry, fallback, layout identity, and fail-closed unknown formats.", phase=1, gate="G0", lock="schema/kv_format*.json,include/sparkpipe/spark_kv*.h")
    add("KV-018", "kv", "Reference KV codecs, conversion, accuracy, and byte accounting", "KV-002,KV-017", 2.5, "CPU/reference encode/decode and conversion vectors cover every declared format, exceptional values, deterministic bytes, per-rank capacity, attention output error, long-context drift, and rejected error budgets.", phase=2, gate="G1", lock="cache/kv_codec*,tests/fixtures/kv_codec/**")
    add("KV-019", "kv", "Cross-backend KV capability and kernel matrix", "CUDA-006,AMD-010,MET-009", 2.5, "CUDA, ROCm, and Metal publish supported key/value format pairs, conversion and attention kernel receipts, numerical limits, capacity gains, and explicit unsupported-format rejection.", phase=4, hardware="multi_backend", gate="G4", uncertainty="high")
    add("KV-020", "kv", "Per-model legal batch-context-KV matrix", "KV-009,KV-018,RCP-018", 2.0, "Each model/build emits measured or byte-proved legal cells over B1-B1024, contexts through 256K, KV formats, active lanes, and parked lanes; no Cartesian B1024-by-256K implication is permitted.", phase=3, gate="G1")
    add("KV-004", "kv", "Common arena, page directory, and model table", "KV-002,KV-018,REC-006", 2.5, "One active implementation owns reserve, map, consume, release, rollback, per-format accounting, and malformed/stale handle rejection.", phase=2, gate="G1", lock="cache/**,include/sparkpipe/spark_kv*.h")
    add("KV-005", "kv", "Common prefix-cache reserve/commit/cancel core", "KV-004", 2.0, "Prefix identity binds build/tokenizer/template/policy; reference counts, lookahead, eviction, rollback, and collision tests pass.", phase=2, gate="G1", lock="cache/prefix_cache.c")
    add("KV-006", "kv", "Park/restore module ABI", "KV-002,KV-003", 1.5, "Whole-lane state transfer uses generation-fenced asynchronous descriptors and rejects partial, stale, or cross-build objects.", phase=2, gate="G1")
    add("KV-007", "kv", "Crash-durable backing slot journal", "KV-006,FND-004", 2.5, "Allocation, writeback, token boundary, digest, free, replay, torn-write, and process-death tests recover one authoritative slot map.", phase=2, gate="G1")
    add("KV-008", "kv", "Rank-local NVMe backing store", "KV-007,ART-013", 2.5, "Direct/buffered I/O, checksums, queue depth, cancellation, corruption, space exhaustion, and physical-device receipts pass.", phase=3, hardware="spark_storage", gate="G7")
    add("KV-009", "kv", "KVP, TP, and CP physical ownership compiler", "KV-002,RCP-016", 2.0, "Every legal topology reports approximately 1/N rank bytes where geometry permits and explicit replication otherwise.", phase=2, gate="G1")
    add("KV-010", "kv", "Active/parked lane residency state machine", "KV-005,KV-008,KV-009", 2.5, "Only full active lanes decode; transitions, shared prefixes, failures, cancellation, and release are generation-fenced and restart-safe.", phase=3, gate="G7")
    add("KV-011", "kv", "Whole-lane asynchronous restore pipeline", "KV-010,HWI-003", 2.5, "Required block sets restore in parallel before the decode quantum and expose deadline, progress, bandwidth, and cancellation futures.", phase=3, hardware="accelerator_storage", gate="G8")
    add("KV-012", "kv", "Next-lane prefetch predictor", "KV-011", 2.0, "Deterministic replay compares predictions, lead time, hit/miss, wasted bytes, deadline wins, and no-prefetch control.", phase=3, gate="G1")
    add("KV-013", "kv", "Pager admission feedback and anti-thrash policy", "KV-012", 2.5, "Pressure queues/rejects work, protects young active lanes, bounds churn, retains at least 95% matched-control throughput, misses under 0.1% of restore deadlines, wastes at most 10% prefetched bytes, and never loops unbounded eviction/restore.", phase=3, gate="G8")
    add("KV-014", "kv", "Shared-prefix privacy and deduplication", "KV-005,SEC-004", 2.0, "Cross-request/tenant sharing follows policy; hashes do not leak content; revocation and release remove unauthorized references.", phase=3, gate="GS")
    add("KV-015", "kv", "Backing corruption and restart recovery", "KV-007,KV-010,KV-014", 2.5, "Digest mismatch recomputes safely; crashes between every journal step recover without attaching bytes to another build or sequence.", phase=4, hardware="spark_storage", gate="GH")
    add("KV-016", "kv", "2.5 TB legal-cell parked-session qualification", "KV-013,KV-015,KV-019,KV-020,ART-018,MOD-D4F-007", 5.0, "Real hardware proves the published legal B/context/KV cells, 2.5 TB rank-local parked capacity, no active-token backing reads, at least 95% matched-control throughput, latency SLO, restore/prefetch budgets, crash recovery, and long-context output.", phase=4, hardware="cuda4_storage", gate="G8", uncertainty="research")

    # Hierarchical scheduler: global broker routes to islands; islands schedule devices.
    add("SCH-001", "scheduler", "Compute-island signed offer schema", "FND-005,TOP-008", 1.5, "Offers bind island/provider, member builds and release compatibility, features, batch/context, request/token/KV capacity, queue, policy, price/SLA, qualification, generation, nonce, and expiry.", phase=2, gate="G0")
    add("SCH-002", "scheduler", "Generation-fenced capacity lease store", "SCH-001,FND-006", 2.5, "CAS, acquire, renew, commit, release, expiry, replay, stale writer, and no-double-sale tests pass.", phase=2, gate="G1", lock="scheduler/lease_store.*")
    add("SCH-003", "scheduler", "Scheduler event log and restart recovery", "SCH-002", 2.0, "Offers, leases, requests, placements, deployments, and reconciliations replay from durable cursors without transient-only completion.", phase=2, gate="GH", lock="scheduler/event_store.*")
    add("SCH-004", "scheduler", "Global broker hard-filter core", "SCH-001,SCH-003", 2.0, "Alias revision resolves to one immutable release; release-member build, feature, context, region, trust, retention, qualification, expiry, and capacity filters fail closed with reasons before exact-build pinning.", phase=2, gate="G1")
    add("SCH-005", "scheduler", "Global broker scoring policy", "SCH-004,MTR-004", 2.0, "Queue, SLO, reliability, price, capacity-credit preference, affinity, energy, and fragmentation scoring is deterministic and policy-versioned.", phase=2, gate="G1")
    add("SCH-006", "scheduler", "Island-local final admission interface", "SCH-001,KV-010", 1.8, "The island atomically checks the selected member build, live weights, workspace, active/parked KV, batch, deadline, gang, and backing capacity before commit.", phase=3, gate="G1")
    add("SCH-007", "scheduler", "Island EDF/deficit/fairness scheduler", "SCH-006,RT-014", 3.0, "Priority, deadline, age, tenant deficit, quantum, cancellation, and starvation bounds pass deterministic traces.", phase=3, gate="G7", lock="runtime/model_batch_engine.c")
    add("SCH-008", "scheduler", "Dynamic B1-B1024 batch and context enforcement", "SCH-007,RCP-018", 2.5, "Compatible nonempty widths launch; unsupported cells cap or reject context explicitly; no hidden B1 fallback masks failures.", phase=3, gate="G8", lock="runtime/model_batch_engine.c")
    add("SCH-009", "scheduler", "Collective gang scheduler", "SCH-007,COL-010", 2.5, "All ranks commit one quantum; unrelated work cannot interleave; timeout/cancellation releases every rank exactly once.", phase=3, hardware="multi_backend", gate="G7")
    add("SCH-010", "scheduler", "KV-aware admission, restore, and backpressure", "SCH-008,KV-013", 2.5, "Admission accounts active/parked bytes and restore deadlines; saturation produces bounded queue/rejection rather than thrash.", phase=3, gate="G8")
    add("SCH-011", "scheduler", "Fleet gang-placement solver", "TOP-006,RCP-019,SCH-002", 3.5, "Solver forms legal locality groups, fits all resource tiers, minimizes fragmentation/cost, and returns auditable rejection causes.", phase=3, gate="G1", uncertainty="high")
    add("SCH-012", "scheduler", "Declarative active-model-set reconciler", "SCH-011,REL-002,RCP-003", 3.0, "Desired aliases converge to exact deployments with warming, readiness, drain, standby, rollback, and generation fencing.", phase=3, gate="G7", lock="scheduler/active_set.*")
    add("SCH-013", "scheduler", "Sub-minute warm promotion orchestration", "SCH-012,ART-016", 2.5, "Promotion treats copy/compile/restore as futures and publishes READY/ACTIVE only after all-rank prewarm and qualification.", phase=3, hardware="fleet", gate="G8")
    add("SCH-014", "scheduler", "Co-residency and exclusive-model policy", "SCH-012,RT-016", 2.0, "Small compatible models coexist under measured budgets; K3 or other exclusive deployments drain conflicting allocations safely.", phase=3, hardware="fleet", gate="G7")
    add("SCH-015", "scheduler", "Global per-request lease and attempt identity", "SCH-005,SCH-006", 2.0, "Every admission attempt is distinct; one commit atomically pins release member build, island, and allocation; only that attempt is billable and lease expiry cannot resurrect stale work.", phase=3, gate="G1")
    add("SCH-016", "scheduler", "Pre-token hedging and reroute policy", "SCH-015", 2.0, "Optional redundant island attempts cancel losers before first token and preserve idempotency, quota, usage, and latency evidence.", phase=3, gate="G7")
    add("SCH-017", "scheduler", "Post-token pinning and terminal failure semantics", "SCH-015,RT-008", 1.8, "After output begins, the exact build/island/allocation remains pinned; failure is terminal unless qualified continuation transfer exists.", phase=3, gate="G7")
    add("SCH-018", "scheduler", "Scheduler simulator and chaos matrix", "SCH-003,SCH-010,SCH-013,SCH-014,SCH-016,SCH-017", 3.0, "Trace replay covers overload, expiry, process loss, provider partition, promotion crash, KV pressure, and no-double-sale invariants.", phase=4, gate="GH", uncertainty="high")
    add("SCH-019", "scheduler", "Leader fencing and single-writer authority", "SCH-003,HA-001", 2.5, "Leader change and partition cannot create two active placement/broker writers or accept stale generations.", phase=4, gate="GH")
    add("SCH-020", "scheduler", "Hierarchical scheduler release gate", "SCH-018,SCH-019", 0.0, "Global broker, island scheduler, leases, placement, active set, promotion, reroute, pinning, and recovery pass sustained lifecycle tests.", phase=4, kind="milestone", resource="coordinator", gate="G7", pairable=False)

    # Public API: a hardened text/chat product, not the current token-ID shim.
    add("API-001", "api", "OpenAI compatibility profile", "FND-006,REC-003", 1.5, "Supported models, chat, responses, completions, streaming, tools, structured output, sampling, errors, usage, and extension semantics are explicit.", phase=1, gate="G0")
    add("API-002", "api", "Versioned protocol and client golden fixtures", "API-001", 2.0, "Request, response, SSE, tool, error, finish, usage, cancellation, idempotency, malformed, and SDK fixtures are deterministic.", phase=1, gate="G1")
    add("API-003", "api", "Hardened asynchronous gateway core", "API-002", 3.0, "Bounded HTTP parsing, connection lifecycle, backpressure, timeouts, graceful drain, health, and internal request identity pass fuzz and load tests.", phase=2, gate="G7", lock="gateway/**")
    add("API-004", "api", "Model catalog and alias endpoint", "API-003,FND-005,RCP-003", 1.5, "Callers see only authorized aliases, immutable build metadata, features, context, policy, price, availability, and stable pagination.", phase=2, gate="G7", lock="gateway/**")
    add("API-005", "api", "Chat, completions, and Responses request mapping", "API-003,RT-012", 2.5, "Text/messages render through the exact tokenizer/template package and map every supported or rejected field explicitly.", phase=2, gate="G7", lock="gateway/**")
    add("API-006", "api", "SSE streaming and ordered terminal events", "API-005,RT-008", 2.5, "Accepted, deltas, tools, usage, done, disconnect, errors, and one terminal event remain ordered under backpressure and retry.", phase=2, gate="G7", lock="gateway/**")
    add("API-007", "api", "Tool calls and structured-output conformance", "API-005", 2.0, "Tool schemas, arguments, finish reasons, JSON schema/format, invalid model output, and streaming assembly match fixtures.", phase=2, gate="G7", lock="gateway/**")
    add("API-008", "api", "Sampling, seed, logprob, penalty, and stop conformance", "API-005,RT-013", 2.0, "All supported controls propagate exactly and unsupported combinations return stable validation errors rather than being ignored.", phase=2, gate="G7", lock="gateway/**")
    add("API-009", "api", "Durable idempotent request lifecycle", "API-006,SCH-015,MTR-003", 2.5, "Submission/retry/restart recovers one request, attempt, quota reservation, route, stream cursor, usage, and terminal state without duplicate billing.", phase=3, gate="GH", lock="gateway/**")
    add("API-010", "api", "Cancellation and client-disconnect propagation", "API-006,SCH-017", 2.0, "Explicit cancel, deadline, socket loss, and gateway restart propagate to island/ranks and release quota/resources exactly once.", phase=3, gate="G7", lock="gateway/**")
    add("API-011", "api", "Stable errors, overload, and bounded backpressure", "API-003,SCH-010", 2.0, "Validation, auth, quota, routing, capacity, timeout, provider, cancellation, and internal errors have stable HTTP/SSE envelopes and no false 503 ambiguity.", phase=3, gate="G7", lock="gateway/**")
    add("API-012", "api", "Tenant identity, scoped keys, and policy enforcement", "API-003,SEC-002", 2.5, "Organization/project roles, key scopes, rotation, revocation, model/region/trust/data policy, and audit events fail closed.", phase=2, gate="GS", lock="gateway/**,auth/**")
    add("API-013", "api", "Gateway to global broker and island integration", "API-004,API-005,API-006,API-007,API-008,API-009,API-010,API-011,API-012,SCH-020", 3.0, "End-to-end requests retain exact alias/build/lease/island/allocation identity and independently reconcile terminal usage.", phase=4, gate="G7", lock="gateway/**")
    add("API-014", "api", "API sustained load, fuzz, restart, and chaos", "API-013", 3.5, "Named concurrency, B/context mix, streaming, cancellation, malformed input, process death, provider failure, and soak targets pass without leaks or double results.", phase=4, gate="G7", uncertainty="high")
    add("API-015", "api", "Official SDK compatibility matrix", "API-013,CI-003", 2.0, "Named Python and TypeScript OpenAI clients plus SparkPipe SDKs pass streaming, tools, cancellation, errors, and usage fixtures.", phase=4, gate="GX")
    add("API-016", "api", "Authenticated local API beta gate", "API-014,API-015,REL-004,MOD-D4F-007,MOD-Q27-007", 0.0, "At least Qwen 3.8 27B and DSV4 Flash serve sustained authenticated text/chat traffic from merged main with exact usage and rollback.", phase=4, kind="milestone", resource="coordinator", gate="G9", pairable=False)

    # Usage, cash settlement, and the 10% fee paid in qualified capacity credits.
    add("MTR-001", "metering", "Immutable usage-event schema", "FND-004,FND-005", 1.5, "Edge, broker, island, rank, model build, request attempt, token classes, timestamps, and evidence hashes have canonical integer fields.", phase=1, gate="G0")
    add("MTR-002", "metering", "Independent edge and island usage receipts", "MTR-001", 2.0, "Edge and island seal independently reconcilable prompt/cached/reasoning/speculative/output usage without provider authority to mint billing.", phase=2, gate="G1")
    add("MTR-003", "metering", "Transactional quota and worst-case reservation", "MTR-001,SCH-002,API-012", 2.5, "Concurrent reserve, commit actual usage, release remainder, cancel, expiry, retry, and restart never oversell or double charge quota.", phase=2, gate="GF")
    add("MTR-004", "metering", "Immutable price and SLA snapshots", "MTR-001,SCH-001", 1.5, "A request binds historical customer price, provider price, unit definitions, region/trust policy, SLA, currency, and effective generation.", phase=2, gate="GF")
    add("MTR-005", "metering", "Sold-capacity unit and settlement-window schema", "MTR-004", 1.5, "Sold capacity is measured in qualified comparable units with provider/island/build envelope, window, delivery evidence, and no idle-capacity fee.", phase=2, gate="GF")
    add("MTR-006", "metering", "Integer double-entry ledger core", "MTR-002,MTR-003,MTR-004,MTR-005", 3.0, "Balanced, append-only entries cover usage, charges, cash liability, capacity credit, consumption, resale, reversal, expiry, and replay.", phase=3, gate="GF", lock="ledger/**")
    add("MTR-007", "metering", "Customer charge, credit, and invoice ledger", "MTR-006", 2.5, "Every invoice line reconciles request usage and price snapshot; credits, refunds, taxes, and disputes remain balanced.", phase=3, gate="GF")
    add("MTR-008", "metering", "Provider cash-payout ledger", "MTR-006", 2.5, "Delivered sold service creates auditable payout liability separately from the capacity-in-kind fee, with hold and reversal states.", phase=3, gate="GF")
    add("MTR-009", "metering", "Ten-percent sold-capacity credit mint", "MTR-005,MTR-006,PRV-006", 2.5, "For sold capacity S, exactly 10% equivalent qualified capacity credits are minted in kind; advertised idle capacity and cash balances are untouched.", phase=3, gate="GF")
    add("MTR-010", "metering", "Capacity-credit consumption, resale, and fencing", "MTR-009,SCH-002", 2.5, "Platform internal use or resale consumes one generation-fenced credit once; quality equivalence, expiry, reversal, and no-double-sale tests pass.", phase=3, gate="GF")
    add("MTR-011", "metering", "Usage, payout, capacity-credit dispute workflow", "MTR-007,MTR-008,MTR-010,PRV-007", 2.5, "Edge/island/provider disagreement enters a hold with evidence, adjudication, correction, and immutable reversal rather than silent overwrite.", phase=4, gate="GF")
    add("MTR-012", "metering", "Financial and capacity-credit reconciliation gate", "MTR-011", 0.0, "Concurrent usage, invoice, payout, 10% capacity-in-kind fee, consumption/resale, disputes, restart, and backup reconciliation balance exactly.", phase=4, kind="milestone", resource="coordinator", gate="GF", pairable=False)

    # Compute-provider enrollment and decentralized island operations.
    add("PRV-001", "providers", "Provider identity and enrollment service", "SEC-002,TOP-004", 2.0, "Provider organization, island identity, roles, credentials, legal-reference fields, key rotation, revocation, and audit events are scoped.", phase=2, gate="GS")
    add("PRV-002", "providers", "Outbound-only provider node agent", "PRV-001", 3.0, "Provider agent enrolls, probes, warms artifacts, renews offers, executes leases, reports receipts, rotates credentials, and accepts no unauthenticated inbound control.", phase=2, gate="GS", uncertainty="high")
    add("PRV-003", "providers", "Hardware, topology, numerical, bandwidth, and soak qualification", "PRV-002,PERF-002", 3.5, "Qualification binds exact hardware/software/topology/build, expires, and distinguishes host, compile, numerical, performance, and sustained-load evidence.", phase=3, hardware="provider_island", gate="G8", uncertainty="high")
    add("PRV-004", "providers", "Provider trust, geography, retention, and data policy", "PRV-001,SEC-004", 2.0, "Trust tiers and tenant routing policies bind operator, region, hardware, artifact, logging, retention, and content visibility controls.", phase=2, gate="GS")
    add("PRV-005", "providers", "Signed health, offer, and maintenance renewal", "PRV-002,PRV-003,PRV-004", 2.5, "Offers use nonce/epoch/expiry, health generation, maintenance/drain, qualification, price/SLA, and capacity; stale/replay/correlated failures fail.", phase=3, gate="G1")
    add("PRV-006", "providers", "Sold-capacity and delivery evidence", "PRV-005,MTR-005", 2.0, "Sold capacity derives from committed leases and independently observed delivery in comparable units, excluding idle advertised capacity.", phase=3, gate="GF")
    add("PRV-007", "providers", "Provider quarantine, failure, and dispute evidence", "PRV-005,MTR-001", 2.5, "Lease revocation, stale offers, degraded delivery, security action, qualification expiry, partition, recovery, and payout hold have durable events.", phase=3, gate="GH")
    add("PRV-008", "providers", "External compute-island beta", "PRV-006,PRV-007,SCH-020,API-013,UI-P06", 4.0, "One external island serves qualified traffic with signed offers/leases, correct policy, usage, cash payout, 10% capacity credits, failover, and support workflow.", phase=5, hardware="provider_island", gate="G9", uncertainty="research")

    # Customer, provider, operator, and developer-facing product surfaces.
    add("UI-C01", "ui", "Customer-console information architecture and accessibility", "FND-002", 1.5, "Account, keys, models, playground, requests, usage, billing, status, and policy flows have responsive and keyboard-accessible contracts.", phase=2, resource="frontend_pair", gate="GX", lock="ui/customer/**")
    add("UI-C02", "ui", "Customer organizations, projects, roles, and API keys", "UI-C01,API-012", 2.5, "Create, scope, rotate, revoke, audit, and least-privilege flows match API authorization and never expose secret keys after creation.", phase=3, resource="frontend_pair", gate="GS", lock="ui/customer/**")
    add("UI-C03", "ui", "Customer model catalog, availability, and policy controls", "UI-C01,API-004,OBS-003", 2.0, "Users see model features, context, qualification, price, region, trust, retention, availability, incidents, and permitted choices from canonical APIs.", phase=3, resource="frontend_pair", gate="GX", lock="ui/customer/**")
    add("UI-C04", "ui", "Streaming API playground and generated examples", "UI-C01,API-013,API-015", 3.0, "Chat/responses, tools, structured output, usage, errors, cancel, curl, Python, and TypeScript examples reflect the exact request and stream.", phase=4, resource="frontend_pair", gate="GX", lock="ui/customer/**")
    add("UI-C05", "ui", "Customer request history, usage, quota, spend, and invoices", "UI-C01,MTR-007,OBS-003", 3.0, "Canonical request/usage/price/ledger data powers filters, budgets, alerts, exports, invoices, and retention-aware traces without a second calculation.", phase=4, resource="frontend_pair", gate="GF", lock="ui/customer/**")
    add("UI-C06", "ui", "Customer-console end-to-end qualification", "UI-C02,UI-C03,UI-C04,UI-C05,SEC-005", 2.0, "Role isolation, accessibility, responsive layout, API errors, stale status, key security, billing reconciliation, and browser E2E tests pass.", phase=5, resource="frontend_pair", gate="GX", lock="ui/customer/**")
    add("UI-P01", "ui", "Compute-provider console information architecture", "FND-002", 1.5, "Enrollment, setup, islands, qualification, health, maintenance, artifacts, storage, sold capacity, payout, 10% credits, disputes, and alerts are defined.", phase=2, resource="frontend_pair", gate="GX", lock="ui/provider/**")
    add("UI-P02", "ui", "Provider enrollment and node-agent setup UI", "UI-P01,PRV-002", 2.5, "Provider roles, credentials, guided setup, downloadable config, probe progress, remediation, rotation, and revoke flows use canonical APIs.", phase=3, resource="frontend_pair", gate="GS", lock="ui/provider/**")
    add("UI-P03", "ui", "Provider island health, capacity, offers, leases, and maintenance", "UI-P01,PRV-005,OBS-003", 3.0, "Live and stale capacity, active builds, leases, telemetry, incidents, drain/maintenance, offer expiry, and qualification status are visible.", phase=4, resource="frontend_pair", gate="GX", lock="ui/provider/**")
    add("UI-P04", "ui", "Provider qualification, artifacts, storage, and KV budgets", "UI-P01,PRV-003,ART-018", 2.5, "Exact failed gates, remediation, artifact warming, rank hashes, physical storage, model-data budget, KV capacity, and drift are displayed.", phase=4, resource="frontend_pair", gate="GX", lock="ui/provider/**")
    add("UI-P05", "ui", "Provider sold capacity, cash payout, and 10% capacity-credit UI", "UI-P01,MTR-011", 3.0, "Sold units, delivery evidence, cash payout, fee calculation, credit mint, platform use/resale, expiry, reversal, disputes, and exports reconcile exactly.", phase=4, resource="frontend_pair", gate="GF", lock="ui/provider/**")
    add("UI-P06", "ui", "Provider-console end-to-end qualification", "UI-P02,UI-P03,UI-P04,UI-P05,SEC-005", 2.0, "Tenant isolation, roles, accessibility, stale/offline states, security actions, financial reconciliation, and browser E2E tests pass.", phase=5, resource="frontend_pair", gate="GX", lock="ui/provider/**")
    add("UI-O01", "ui", "Operator and SRE fleet console", "OBS-003,SCH-020,PRV-007", 3.0, "Aliases, active sets, placements, offers, leases, promotions, islands, providers, artifacts, incidents, quarantine, and capacity credits are correlated.", phase=5, resource="frontend_pair", gate="GX", lock="ui/operator/**")
    add("UI-O02", "ui", "Operator incident, ledger, and override workflows", "UI-O01,MTR-012,SRE-002", 2.5, "High-risk mutations require scoped roles, reason, preview, generation fence, two-person policy where configured, and append-only audit events.", phase=5, resource="frontend_pair", gate="GS", lock="ui/operator/**")
    add("SDK-001", "ui", "SparkPipe Python SDK", "API-015", 2.0, "Typed sync/async clients cover streaming, tools, structured output, cancel, idempotency, usage, errors, policy extensions, and retries.", phase=4, resource="sdk_pair", gate="GX", lock="sdk/python/**")
    add("SDK-002", "ui", "SparkPipe TypeScript SDK", "API-015", 2.0, "Browser/Node clients cover the same canonical features with abort, streaming parsing, typed errors, and generated examples.", phase=4, resource="sdk_pair", gate="GX", lock="sdk/typescript/**")

    # Correlated product telemetry, SOTA freshness, security, CI, HA, and release.
    add("OBS-001", "observability", "Cross-plane identity and telemetry schema", "FND-004", 1.5, "Request, attempt, alias, build, lease, island, allocation, rank, artifact, usage, ledger, and agent events correlate with bounded labels.", phase=1, gate="G0")
    add("OBS-002", "observability", "Gateway, broker, island, runtime, storage, and backend exporters", "OBS-001,HWI-007", 2.5, "Metrics, traces, logs, profiles, and health expose declared counters without prompts, secrets, raw tokens, private addresses, or unbounded IDs as labels.", phase=3, gate="G1")
    add("OBS-003", "observability", "Operational query API and canonical dashboards", "OBS-002,API-003,SCH-003,MTR-001", 2.5, "Desired/observed active models, requests, capacity, KV, storage, offers, leases, usage, qualification, and incidents share one status calculation.", phase=3, gate="GX")
    add("OBS-004", "observability", "Provider and OxAlpha scorecards", "OBS-003,OXA-009,PRV-005", 2.0, "1h/24h/7d/lifetime transport, latency, validity, error, race, useful completion, audit, delivery, and settlement dimensions include sample counts.", phase=4, gate="GX")
    add("OBS-005", "observability", "Service SLOs, burn rates, and alerts", "OBS-003,SCH-020", 2.0, "TTFT, inter-token, availability, correctness, queue, cancellation, KV, capacity, and provider SLOs alert on injected violations and stale data.", phase=4, gate="G7")
    add("OBS-006", "observability", "Qualification and SOTA freshness status", "OBS-003,PERF-001", 1.5, "Each model/build/backend/topology cell shows highest gate, receipt age, source age, comparable target, observed gap, and blocker.", phase=4, gate="GX")

    add("SEC-001", "security", "Product threat model and trust boundaries", "FND-002", 2.0, "Tenant edge, control plane, provider island, artifact supply chain, model inputs, agent system, UI, and finance abuse cases have mitigations and owners.", phase=1, kind="design", resource="security_pair", gate="GS")
    add("SEC-002", "security", "PKI, mTLS, scoped credentials, rotation, and secrets", "SEC-001", 3.0, "Mutual auth, short-lived credentials, key rotation/revocation, least privilege, bootstrap, offline recovery, and log/patch redaction tests pass.", phase=2, resource="security_pair", gate="GS", lock="security/**")
    add("SEC-003", "security", "Artifact signatures, dependency inventory, and SBOM", "SEC-002,ART-009", 2.5, "Build/release/provider agents verify signatures, source/dependency provenance, revocation, tamper, wrong-target, and rollback metadata.", phase=3, resource="security_pair", gate="GS")
    add("SEC-004", "security", "Tenant retention, privacy, trust-tier, and data-policy engine", "SEC-001,FND-005", 2.5, "Content visibility, provider trust, geography, logging, prefix sharing, retention, deletion, and disclosure policies are enforced and audited.", phase=2, resource="security_pair", gate="GS")
    add("SEC-005", "security", "Public API, provider, artifact, UI, and finance security gate", "SEC-002,SEC-003,SEC-004,API-014,PRV-007,ART-019", 4.0, "Fuzz, dependency, authz, tenant isolation, replay, tamper, retention, abuse, UI, payout, and incident tests have independent review.", phase=5, resource="security_pair", gate="GS", uncertainty="high")

    add("CI-001", "reliability", "Host and multiplatform CI foundation", "FND-004,HWI-001", 2.5, "Linux and macOS host tests plus portable headers, schemas, generators, sanitizers, fuzzers, and deterministic fixtures run from clean checkouts.", phase=1, gate="G1", lock=".github/workflows/**")
    add("CI-002", "reliability", "CUDA, ROCm, Metal, fabric, storage, and fleet qualification queues", "CI-001", 2.5, "Hardware jobs declare scarce resources, exact target, clean deployment, timeout, artifact retention, and BLOCKED rather than simulated success.", phase=2, gate="G2", lock="qualification/**")
    add("CI-003", "reliability", "API, ABI, schema, recipe, and N/N-1 compatibility matrix", "CI-001,FND-006,API-002", 2.5, "Supported previous/current clients, schemas, artifacts, runtime ABIs, and rollback paths have positive and fail-closed fixtures.", phase=3, gate="G1")
    add("CI-004", "reliability", "Signed release train and promotion policy", "CI-003,SEC-003,REL-001", 2.5, "Only reviewed merged-main commits with complete receipts, signatures, SBOM, compatibility, and rollback artifacts can enter promotion.", phase=3, gate="G9", lock="deployment/**")
    add("HA-001", "reliability", "Replicated control state and leader fencing", "SCH-003,SEC-002", 4.0, "Leader loss, partition, stale writer, clock skew, replay, and split-brain tests preserve one authority for aliases, leases, requests, and placements.", phase=4, gate="GH", uncertainty="high")
    add("HA-002", "reliability", "Catalog, event, quota, ledger, alias, and key backup/restore", "HA-001,ART-010,MTR-012", 3.0, "Point-in-time backup, restore, hash reconciliation, key recovery, and missing/corrupt backup failures meet declared RPO/RTO.", phase=5, gate="GH")
    add("SRE-001", "reliability", "Runbooks, capacity planning, maintenance, and incident command", "OBS-005,REL-004", 2.0, "Drain, swap, rollback, quarantine, provider failure, storage pressure, network loss, and security incidents have executable evidence-preserving procedures.", phase=4, gate="GH")
    add("SRE-002", "reliability", "Fleet and provider chaos game days", "SRE-001,SCH-020,PRV-007", 3.0, "Timed process, rank, node, switch, storage, provider, broker, promotion, and billing failures meet recovery and evidence expectations.", phase=5, hardware="fleet_and_provider", gate="GH", uncertainty="high")
    add("HA-003", "reliability", "Island and regional disaster-recovery game day", "HA-002,SRE-002,REL-005", 4.0, "Measured failover/failback restores control, catalog, ledger, aliases, and qualified service within declared RTO/RPO without double billing.", phase=6, gate="GH", uncertainty="high")
    add("REL-001", "reliability", "Generation-atomic local release activation", "ART-009,RT-006", 2.5, "Power/process loss leaves either old or new complete generation active; mixed files and partial readiness cannot serve.", phase=2, hardware="single_node", gate="G7", lock="deployment/**")
    add("REL-002", "reliability", "All-rank readiness quorum and activation journal", "REL-001,COL-010", 2.5, "Every rank agrees build/artifact/topology/generation; mismatch, timeout, restart, and rollback never publish a partial deployment.", phase=3, hardware="multi_rank", gate="G7", lock="deployment/**")
    add("REL-003", "reliability", "Measured sub-sixty-second warm activation", "REL-002,SCH-013,ART-016", 3.0, "Warm LOCAL_READY builds activate, prewarm, agree, and publish in under 60 seconds with raw per-rank timing distribution and hashes.", phase=4, hardware="fleet", gate="G8", uncertainty="high")
    add("REL-004", "reliability", "Drain, rollback, and merged-main zero-drift deployment", "REL-003,CI-004", 3.0, "PR merge, target pull, clean rebuild/install, drain, restart, live parity, rollback, and process/hash evidence are one retained chain.", phase=4, hardware="fleet", gate="G9")
    add("REL-005", "reliability", "Canary-to-fleet rollout and compatibility rollback", "REL-004,HA-001", 3.0, "Canary health gates fleet progression; stale nodes cannot join; rollback restores the previous signed generation under live traffic.", phase=5, hardware="fleet", gate="G9", uncertainty="high")
    add("REL-006", "reliability", "Production platform release gate", "REL-005,HA-003,SEC-005,API-016,MTR-012,MOD-Q27-008,MOD-D4F-008", 0.0, "Local production has qualified models, API, scheduler, KV, accounting, security, HA/DR, UI, and zero-drift deployment evidence.", phase=6, kind="milestone", resource="coordinator", gate="G9", pairable=False)

    # Daily SOTA discovery and production-shape optimization work-order loop.
    add("PERF-001", "performance", "Daily primary-source comparable-SOTA ledger system", "", 1.5, "Every supported model observation binds source/date/revision, checkpoint, quality, hardware, topology, batch/context/output, speculation, metric, boundary, and evidence class; production receipts expire after 24 hours.", phase=0, gate="G0", recurring_days=1, freshness_hours=24, planning_state="candidate_unintegrated")
    add("PERF-002", "performance", "Benchmark comparability and plus-ten-percent protocol", "PERF-001,FND-004", 1.5, "COMPARABLE/PARTIAL/INCOMPARABLE decisions bind checkpoint, numerical gate, weight/compute/KV route, accelerator/count, topology/fabric, power/clocks, B/context/output distribution, prefix/speculation, TTFT/ITL SLO, metric/boundary, samples, confidence, and exact parity/110% rules.", phase=1, gate="G0")
    add("PERF-003", "performance", "No-drop profiler to work-order compiler", "PERF-002,OBS-001", 2.5, "Profiles separate GPU busy union, exposed idle, host, collectives, transfer, KV, and overlap; emitted tasks have hypothesis, write set, control, tests, and rollback.", phase=2, gate="G1")
    add("PERF-004", "performance", "Strong-reference production-shape kernel harness", "PERF-003,HWI-005", 2.5, "SparkPipe and current strong references run identical shapes, layouts, precision, quality, hardware, timing boundary, and numerical gates.", phase=2, hardware="accelerator", gate="G8")
    add("PERF-005", "performance", "Automated retain/reject hill-climb loop", "PERF-004", 3.0, "Randomized paired full-request A/B retains only quality-preserving end-to-end wins and archives rejected experiments to prevent repetition.", phase=3, hardware="accelerator", gate="G8", uncertainty="high")
    add("PERF-006", "performance", "Kernel-to-layer-to-model-to-service promotion gate", "PERF-005,RT-017", 2.0, "A microkernel or layer win cannot promote if resident full-model/API throughput, latency, memory, power, or correctness regresses.", phase=3, hardware="accelerator", gate="G8")
    add("PERF-007", "performance", "Per-model daily SOTA and bottleneck refresh policy", "PERF-006", 1.0, "A recurring service refreshes each supported model's comparable cells, clean-main baseline, exposed bottleneck, gap, and disjoint work-order queue daily; stale receipts fail release admission.", phase=4, hardware="multi_backend", gate="G8", recurring_days=1, freshness_hours=24, uncertainty="high")
    add("PERF-008", "performance", "Power, thermal, and capacity efficiency", "PERF-006,HWI-007", 2.0, "Throughput, latency, joules/token, thermal soak, device utilization, and per-capacity-unit efficiency accompany provider qualification and routing.", phase=4, hardware="multi_backend", gate="G8")
    add("PERF-009", "performance", "One-hundred-ten-percent economic target gate", "PERF-007,PERF-008,MTR-005", 1.5, "Named sold-capacity cells have fresh comparable SOTA and retained SparkPipe measurements at or above 110%; cells below parity cannot release and cells from 100% through under 110% cannot claim fee-neutral economics.", phase=4, gate="GF", freshness_hours=24)
    add("PERF-010", "performance", "Continuous performance regression and freshness gate", "PERF-006,CI-002", 2.0, "Named production cells enforce fresh <=24-hour comparable receipts and detect throughput, TTFT, inter-token, memory, power, and quality regression with retained controls and raw receipts.", phase=4, hardware="multi_backend", gate="G8", freshness_hours=24)

    # Seven model programs share one onboarding lifecycle and common runtime.
    model_release_ids: list[str] = []
    for code, name, minimum_hardware, production_hardware, initial_state in MODEL_DRIVER_PROGRAMS:
        prefix = f"MOD-{code}"
        model_lock = f"model:{code.lower()}"
        agent_lane = f"model-driver:{code.lower()}"
        model_release_ids.append(f"{prefix}-008")
        add(f"{prefix}-001", "models", f"{name} exact source, contract, and reference oracle", "FND-003,REC-010,RCP-001", 2.0 if code != "H3" else 3.0, f"{name} has exact checkpoint/tokenizer/config identity, geometry, tensor inventory, authoritative implementation, reduced fixtures, and no borrowed nearby-model constants.", phase=2, gate="G1", lock=f"model_contracts/{code.lower()}*,model-families/{code.lower()}*/**", uncertainty="high" if code in ("H3", "QMAX") else "normal", planning_state=initial_state, agent_lane=agent_lane)
        add(f"{prefix}-002", "models", f"{name} recipe, codec, topology, and capacity plan", f"{prefix}-001,RCP-019,KV-020", 2.0, f"{name} publishes legal precision/topology/KV variants, minimum and production hardware quantities, rank bytes, B/context cells, co-residency/exclusivity, and byte-proved rejected cells.", phase=3, gate="G1", lock=f"{model_lock}:contract", uncertainty="high", planning_state=initial_state)
        add(f"{prefix}-003", "models", f"{name} normalized tensors, rank packs, and compiled build", f"{prefix}-002,ART-009", 3.0 if code not in ("K3", "QMAX") else 4.5, f"{name} rank packages are complete, atomic, content-addressed, loader-validated, and recoverable from recipe without full-checkpoint duplication per rank.", phase=3, hardware="model_storage", gate="G2", lock=f"{model_lock}:artifacts", uncertainty="research" if code in ("K3", "QMAX", "H3") else "high", planning_state=initial_state)
        add(f"{prefix}-009", "models", f"{name} dense, norm, activation, and residual production shapes", f"{prefix}-003,CUDA-010", 2.5, f"All {name} dense/linear, normalization, activation, residual, and output-head shapes pass real-weight numerical, bounds, memory, and timing controls.", phase=3, hardware="cuda1", gate="G3", lock=f"{model_lock}:dense", uncertainty="high", planning_state=initial_state)
        add(f"{prefix}-010", "models", f"{name} attention, position, and recurrent-state production shapes", f"{prefix}-003,CUDA-010,KV-018", 3.0, f"All {name} attention/MLA/GDN or other state, position encoding, masks, prefill/decode seams, and declared KV formats pass real-weight controls.", phase=3, hardware="cuda1", gate="G3", lock=f"{model_lock}:attention", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-011", "models", f"{name} FFN, router, expert, and grouped-GEMM production shapes", f"{prefix}-003,CUDA-010", 3.0, f"Every required dense FFN or MoE router/top-k/shared expert/dispatch/grouped GEMM/combine path passes real-weight controls; non-applicable paths are contract-proved rather than silently skipped.", phase=3, hardware="cuda1", gate="G3", lock=f"{model_lock}:moe", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-004", "models", f"{name} real-weight complete-layer CUDA oracle", f"{prefix}-009,{prefix}-010,{prefix}-011", 2.5, f"One complete real-weight {name} layer including every model-required state, FFN/MoE, residual, normalization, and seam matches the authoritative implementation on one-rank layer fixtures.", phase=3, hardware="cuda1", gate="G4", lock=f"{model_lock}:layer", uncertainty="research" if code in ("K3", "QMAX", "H3") else "high", planning_state=initial_state)
        add(f"{prefix}-012", "models", f"{name} collective and production-topology plan", f"{prefix}-002,COL-010", 2.5, f"The selected {name} TP/PP/EP/ETP/KVP plan has exact rank quantity, tensors, collectives, stage boundaries, failure semantics, bandwidth floor, and deterministic rejection alternatives.", phase=3, hardware=production_hardware, gate="G6", lock=f"{model_lock}:transport", uncertainty="high", planning_state=initial_state)
        add(f"{prefix}-005", "models", f"{name} minimum-legal-topology full-model correctness", f"{prefix}-004,RT-018,KV-010", 4.0, f"{name} real-pack prefill/decode traverses the full model on its minimum legal topology with exact hardware quantity, prompt/output parity, memory, lifecycle, cancellation, restart, and no validation-time weight loading.", phase=4, hardware=minimum_hardware, gate="G5", lock=f"{model_lock}:fullmodel", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-006", "models", f"{name} production multi-rank topology correctness", f"{prefix}-005,{prefix}-012,SCH-009", 4.0, f"The selected production {name} topology passes all-rank pack identity, numerical, collective, failure, batch, and context gates on the declared hardware quantity.", phase=4, hardware=production_hardware, gate="G6", lock=f"{model_lock}:distributed", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-013", "models", f"{name} KV-resolution and long-context cells", f"{prefix}-005,KV-020", 2.5, f"{name} passes every released KV format and legal B/context cell with byte accounting, attention accuracy, active/parked lifecycle, prefix policy, and explicit rejected cells.", phase=4, hardware=production_hardware, gate="G7", lock=f"{model_lock}:kv", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-014", "models", f"{name} chunked-prefill production cells", f"{prefix}-005,RT-010", 2.5, f"{name} prefill passes declared chunk, batch, prompt/context, cancellation, memory, TTFT, and full-model parity cells.", phase=4, hardware=production_hardware, gate="G7", lock=f"{model_lock}:prefill", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-015", "models", f"{name} continuous-decode and batching cells", f"{prefix}-005,RT-011,RT-014", 2.5, f"{name} decode passes legal dynamic B/context/output cells, sequence completion, sampling identity, fairness, inter-token latency, and sustained throughput controls.", phase=4, hardware=production_hardware, gate="G7", lock=f"{model_lock}:decode", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-016", "models", f"{name} speculation, verify, and rollback cells", f"{prefix}-015,RT-015", 2.5, f"{name} non-spec control and every supported draft/MTP/DSpark route pass accept/reject, KV/state rollback, output identity, fallback, memory, and end-to-end performance gates; unsupported routes reject explicitly.", phase=4, hardware=production_hardware, gate="G7", lock=f"{model_lock}:speculation", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-007", "models", f"{name} API, KV, batching, speculation, and soak matrix", f"{prefix}-006,{prefix}-013,{prefix}-014,{prefix}-015,{prefix}-016,API-013,SCH-010", 3.5, f"{name} serves authenticated text/chat across released cells with park/restore, prefix, cancellation, exact usage, failure injection, and sustained load.", phase=4, hardware=production_hardware, gate="G7", lock=f"{model_lock}:service", uncertainty="research", planning_state=initial_state)
        add(f"{prefix}-017", "models", f"{name} fresh comparable-SOTA and regression gate", f"{prefix}-007,PERF-007,PERF-008,PERF-010", 3.0, f"A <=24-hour comparable receipt proves {name} at least matches SOTA for each released production cell, reports whether it reaches 110%, and passes paired quality, power, capacity, TTFT/ITL, and regression controls; PERF-009 is a separate sold-capacity economics gate, not a model-release prerequisite.", phase=5, hardware=production_hardware, gate="G8", lock=f"{model_lock}:performance", uncertainty="research", planning_state=initial_state, freshness_hours=24)
        add(f"{prefix}-008", "models", f"{name} zero-drift production release", f"{prefix}-017,REL-004", 3.0, f"{name} has fresh parity-qualified performance, merged-main clean deployment, rollback, signed artifacts/SBOM, and zero-drift release receipts; fee-neutral claims require the separate 110% gate.", phase=5, hardware=production_hardware, gate="G9", lock=f"{model_lock}:release", uncertainty="research", planning_state=initial_state, freshness_hours=24)

    # External commercial gates and whole-program milestones.
    add("BUS-001", "business", "Early capacity-in-kind commercial, tax, and data requirements", "FND-002,SEC-001", 3.0, "A reviewable draft fixes sold capacity, qualified units, the ten-percent in-kind/no-idle/no-cash fee, resale, payout, payment, tax, privacy, disputes, and target jurisdictions early enough to constrain engineering.", phase=1, kind="external_gate", resource="legal_finance", hardware="external", gate="GF", uncertainty="research", planning_state="external", pairable=False)
    add("BUS-002", "business", "Payment and provider cash-payout integration", "BUS-001,HA-002", 4.0, "Customer payment, provider payout, holds, reversal, reconciliation, tax records, outage, duplicate webhook, and backup/restore tests pass; capacity credits remain separate.", phase=6, resource="finance_pair", gate="GF", uncertainty="high")
    add("BUS-004", "business", "Final legal, tax, payment, and data-processing approval", "BUS-001,BUS-002,MTR-012,PRV-008,SEC-005", 0.0, "Authorized reviewers approve the production terms, jurisdictions, tax/payment operation, provider contract, capacity-credit treatment, privacy, retention, and data-processing controls; external calendar lead is reported separately.", phase=6, kind="external_gate", resource="legal_finance", hardware="external", gate="GF", uncertainty="research", planning_state="external", pairable=False)
    add("BUS-003", "business", "Production marketplace launch gate", "BUS-002,BUS-004,PRV-008,UI-C06,UI-P06,UI-O02,REL-006", 0.0, "Customer, provider, operator, inference, security, financial, capacity-credit, support, HA/DR, and legal gates are all approved.", phase=6, kind="milestone", resource="coordinator", gate="G9", pairable=False)

    add("MS-001", "milestone", "Architecture and contract freeze", "FND-010", 0.0, "Authoritative architecture and interfaces are frozen for parallel implementation.", phase=0, kind="milestone", resource="coordinator", gate="G0", pairable=False)
    add("MS-002", "milestone", "OxAlpha massively parallel launch readiness", "OXA-012", 0.0, "Dozens of paired agents can be dispatched with truthful state and independent audits.", phase=0, kind="milestone", resource="coordinator", gate="G9", pairable=False)
    add("MS-003", "milestone", "Authenticated local API beta", "API-016", 0.0, "Local customers can use qualified Qwen 3.8 27B and DSV4 Flash through the public API.", phase=4, kind="milestone", resource="coordinator", gate="G9", pairable=False)
    add("MS-004", "milestone", "Full 2.5 TB parked-session KV qualification", "KV-016", 0.0, "The rank-local backing tier satisfies the 2.5 TB parked-session contract and quantitative performance-loss budget.", phase=4, kind="milestone", resource="coordinator", gate="G8", pairable=False)
    add("MS-005", "milestone", "AMD ROCm production backend", "AMD-015", 0.0, "DSV4 runs correctly and competitively on qualified MI350-class hardware.", phase=5, kind="milestone", resource="coordinator", hardware="mi350x4", gate="G9", pairable=False)
    add("MS-006", "milestone", "Apple Silicon Metal v1 production backend", "MET-015", 0.0, "A named Mac class serves Qwen 3.8 27B through the common API and island scheduler and passes the two-Mac collective cell; no other model/SoC support is implied.", phase=5, kind="milestone", resource="coordinator", hardware="apple_silicon", gate="G9", pairable=False)
    add("MS-007", "milestone", "Initial seven-model production matrix", model_release_ids, 0.0, "Qwen 3.8 27B, DSV4 Flash/Pro, GLM 5.2, K3, Qwen 3.8 Max, and MiniMax H3 each have a production-qualified build.", phase=6, kind="milestone", resource="coordinator", hardware="fleet", gate="G9", pairable=False)
    add("MS-008", "milestone", "External compute-provider beta", "PRV-008", 0.0, "At least one external island serves and settles qualified traffic with the 10% capacity-in-kind fee.", phase=5, kind="milestone", resource="coordinator", gate="G9", pairable=False)
    add("MS-009", "milestone", "SparkPipe production marketplace", "BUS-003,MS-004,MS-005,MS-006,MS-007,MS-008", 0.0, "Every required task is in this super-sink's transitive closure and the full heterogeneous inference platform, provider marketplace, UIs, capacity credits, and model matrix are qualified.", phase=6, kind="milestone", resource="coordinator", gate="G9", pairable=False)

    for task in tasks:
        task["dispatch_class"] = "bootstrap" if task["workstream"] in {"foundation", "agents"} else "paired_after_oxa"
        if task["pairable"] and task["dispatch_class"] == "paired_after_oxa":
            task["dispatch_prerequisites"] = ["OXA-012"]

    referenced = {
        dependency
        for task in tasks
        for dependency in task["dependencies"]
    }
    terminal_ids = sorted(
        task["id"]
        for task in tasks
        if task["id"] != "MS-009" and task["id"] not in referenced
    )
    super_sink = next(task for task in tasks if task["id"] == "MS-009")
    super_sink["dependencies"] = sorted(set(super_sink["dependencies"] + terminal_ids))

    return tasks


def resource_constrained_schedule(
    order: list[str],
    by_id: dict[str, dict[str, Any]],
    expected: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    effective_dependencies = {
        task_id: sorted(set(by_id[task_id]["dependencies"] + by_id[task_id]["dispatch_prerequisites"]))
        for task_id in order
    }
    effective_indegree = {
        task_id: len(dependencies) for task_id, dependencies in effective_dependencies.items()
    }
    effective_successors: dict[str, list[str]] = defaultdict(list)
    for task_id, dependencies in effective_dependencies.items():
        for dependency in dependencies:
            effective_successors[dependency].append(task_id)
    effective_ready = deque(sorted(task_id for task_id, count in effective_indegree.items() if count == 0))
    effective_order: list[str] = []
    while effective_ready:
        task_id = effective_ready.popleft()
        effective_order.append(task_id)
        for successor in sorted(effective_successors[task_id]):
            effective_indegree[successor] -= 1
            if effective_indegree[successor] == 0:
                effective_ready.append(successor)
    if len(effective_order) != len(order):
        raise ValueError("dispatch prerequisites create a dependency cycle")
    slots = {
        pool: [0.0] * capacity for pool, capacity in PLANNING_CAPACITY.items()
    }
    lock_available: dict[str, float] = defaultdict(float)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for task_id in effective_order:
        task = by_id[task_id]
        start = max((finishes[dependency] for dependency in effective_dependencies[task_id]), default=0.0)
        requirements: dict[str, int] = defaultdict(int)
        if expected[task_id] > 0:
            requirements[f"worker:{task['resource']}"] += 1
            if task["provider_request_slots"] > 0:
                requirements["api_provider_request"] += task["provider_request_slots"]
            for requirement in task["hardware_requirements"]:
                requirements[requirement["pool"]] += requirement["quantity"]
        while True:
            candidate = start
            for pool, quantity in requirements.items():
                available = sorted(slots[pool])
                candidate = max(candidate, available[quantity - 1])
            for lock in task["write_locks"]:
                candidate = max(candidate, lock_available[lock])
            if abs(candidate - start) < 0.000001:
                break
            start = candidate
        finish = start + expected[task_id]
        starts[task_id] = start
        finishes[task_id] = finish
        for pool, quantity in requirements.items():
            indices = sorted(range(len(slots[pool])), key=lambda index: (slots[pool][index], index))[:quantity]
            for index in indices:
                slots[pool][index] = finish
        for lock in task["write_locks"]:
            lock_available[lock] = finish
    return starts, finishes


def validate_and_schedule(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for task in tasks:
        task_id = task["id"]
        if task_id in by_id:
            errors.append(f"duplicate task id: {task_id}")
        by_id[task_id] = task
        if task["workstream"] not in WORKSTREAMS:
            errors.append(f"{task_id}: unknown workstream {task['workstream']}")
        estimate = task["estimate_days"]
        if not (estimate["optimistic"] <= estimate["most_likely"] <= estimate["pessimistic"]):
            errors.append(f"{task_id}: invalid PERT estimate ordering")
        if task["pairable"] and not task["write_locks"]:
            errors.append(f"{task_id}: pairable work has no structured write lock")
        provider_slots = task["provider_request_slots"]
        provider_domains = task["provider_failure_domains_required"]
        if not isinstance(provider_slots, int) or provider_slots < 0 or provider_slots > PLANNING_CAPACITY["api_provider_request"]:
            errors.append(f"{task_id}: invalid provider request slots {provider_slots}")
        if not isinstance(provider_domains, int) or provider_domains < 0:
            errors.append(f"{task_id}: invalid provider failure-domain requirement {provider_domains}")
        if task["pairable"]:
            if provider_slots < PROVIDER_REQUESTS_PER_PAIR:
                errors.append(f"{task_id}: provider race capacity below 2+2 redundancy")
            if provider_domains < MIN_PROVIDER_FAILURE_DOMAINS:
                errors.append(f"{task_id}: provider failure-domain diversity below minimum")
        elif provider_slots != 0 or provider_domains != 0:
            errors.append(f"{task_id}: non-pairable work reserves agent-provider capacity")
        for lock in task["write_locks"]:
            if not lock.startswith(("workstream:", "model:", "shared-", "schema/", "tools/", "runtime/", "include/", "cache/", "gateway/", "scheduler/", "ledger/", "ui/", "sdk/", "security/", "qualification/", "deployment/", ".github/", "model_contracts/", "model-families/", "modules/", "node/", "auth/", "orchestration/", "tests/")):
                errors.append(f"{task_id}: untyped write lock {lock}")
        for requirement in task["hardware_requirements"]:
            pool = requirement["pool"]
            quantity = requirement["quantity"]
            if pool not in PLANNING_CAPACITY:
                errors.append(f"{task_id}: unknown hardware pool {pool}")
            elif not isinstance(quantity, int) or quantity < 1 or quantity > PLANNING_CAPACITY[pool]:
                errors.append(f"{task_id}: invalid {pool} quantity {quantity}")
    for task in tasks:
        for dependency in task["dependencies"]:
            if dependency not in by_id:
                errors.append(f"{task['id']}: missing dependency {dependency}")
            elif by_id[dependency]["phase"] > task["phase"]:
                errors.append(f"{task['id']}: phase precedes dependency {dependency}")
        for prerequisite in task["dispatch_prerequisites"]:
            if prerequisite not in by_id:
                errors.append(f"{task['id']}: missing dispatch prerequisite {prerequisite}")
    if errors:
        raise ValueError("\n".join(errors))

    indegree = {task_id: 0 for task_id in by_id}
    successors: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for dependency in task["dependencies"]:
            indegree[task["id"]] += 1
            successors[dependency].append(task["id"])
    ready = deque(sorted(task_id for task_id, count in indegree.items() if count == 0))
    order: list[str] = []
    while ready:
        task_id = ready.popleft()
        order.append(task_id)
        for successor in sorted(successors[task_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(tasks):
        cyclic = sorted(task_id for task_id, count in indegree.items() if count > 0)
        raise ValueError(f"dependency cycle involving: {', '.join(cyclic)}")

    required = {task["id"] for task in tasks if task["required_for_release"]}
    release_closure = {"MS-009"}
    pending = ["MS-009"]
    while pending:
        current = pending.pop()
        for dependency in by_id[current]["dependencies"]:
            if dependency not in release_closure:
                release_closure.add(dependency)
                pending.append(dependency)
    missing_release = sorted(required - release_closure)
    if missing_release:
        raise ValueError("required tasks outside production super-sink: " + ", ".join(missing_release))

    earliest_start: dict[str, float] = {}
    earliest_finish: dict[str, float] = {}
    expected: dict[str, float] = {}
    for task_id in order:
        task = by_id[task_id]
        estimate = task["estimate_days"]
        duration = (estimate["optimistic"] + 4 * estimate["most_likely"] + estimate["pessimistic"]) / 6
        expected[task_id] = duration
        start = max((earliest_finish[dep] for dep in task["dependencies"]), default=0.0)
        earliest_start[task_id] = start
        earliest_finish[task_id] = start + duration
    resource_start, resource_finish = resource_constrained_schedule(order, by_id, expected)
    program_duration = max(earliest_finish.values(), default=0.0)
    latest_finish = {task_id: program_duration for task_id in by_id}
    latest_start: dict[str, float] = {}
    for task_id in reversed(order):
        if successors[task_id]:
            latest_finish[task_id] = min(latest_start[item] for item in successors[task_id])
        latest_start[task_id] = latest_finish[task_id] - expected[task_id]

    enriched: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["id"]
        item = dict(task)
        item["expected_days"] = round(expected[task_id], 3)
        item["variance_days2"] = round(
            ((item["estimate_days"]["pessimistic"] - item["estimate_days"]["optimistic"]) / 6) ** 2,
            3,
        )
        item["earliest_start_day"] = round(earliest_start[task_id], 3)
        item["earliest_finish_day"] = round(earliest_finish[task_id], 3)
        item["resource_start_day"] = round(resource_start[task_id], 3)
        item["resource_finish_day"] = round(resource_finish[task_id], 3)
        item["resource_wait_days"] = round(max(0.0, resource_start[task_id] - earliest_start[task_id]), 3)
        item["slack_days"] = round(max(0.0, latest_start[task_id] - earliest_start[task_id]), 3)
        item["critical"] = abs(item["slack_days"]) < 0.001
        item["successors"] = sorted(successors[task_id])
        enriched.append(item)

    roots = sorted(task["id"] for task in tasks if not task["dependencies"])
    sinks = sorted(task_id for task_id in by_id if not successors[task_id])
    effort = sum(item["expected_days"] for item in enriched if item["kind"] != "milestone")
    critical_tasks = sorted(
        (item["id"] for item in enriched if item["critical"]),
        key=lambda task_id: (earliest_start[task_id], task_id),
    )
    critical_edges = sorted(
        [dependency, task["id"]]
        for task in enriched
        for dependency in task["dependencies"]
        if task["critical"]
        and by_id[dependency]["id"] in critical_tasks
        and abs(earliest_finish[dependency] - earliest_start[task["id"]]) < 0.001
    )
    terminal = min(
        (task_id for task_id in sinks if abs(earliest_finish[task_id] - program_duration) < 0.001),
        default=None,
    )
    representative_critical_path: list[str] = []
    while terminal is not None:
        representative_critical_path.append(terminal)
        candidates = sorted(
            dependency
            for dependency in by_id[terminal]["dependencies"]
            if dependency in critical_tasks
            and abs(earliest_finish[dependency] - earliest_start[terminal]) < 0.001
        )
        terminal = candidates[0] if candidates else None
    representative_critical_path.reverse()
    critical_variance = sum(
        next(item["variance_days2"] for item in enriched if item["id"] == task_id)
        for task_id in representative_critical_path
    )
    workstream_summary: dict[str, dict[str, Any]] = {}
    for workstream in sorted({item["workstream"] for item in enriched}):
        members = [item for item in enriched if item["workstream"] == workstream]
        workstream_summary[workstream] = {
            "title": WORKSTREAMS[workstream],
            "task_count": len(members),
            "expected_engineering_effort_days": round(
                sum(item["expected_days"] for item in members if item["kind"] != "milestone"),
                1,
            ),
            "earliest_start_day": min(item["earliest_start_day"] for item in members),
            "earliest_finish_day": max(item["earliest_finish_day"] for item in members),
            "resource_start_day": min(item["resource_start_day"] for item in members),
            "resource_finish_day": max(item["resource_finish_day"] for item in members),
            "critical_task_count": sum(1 for item in members if item["critical"]),
        }
    planning_roots = sorted(
        item["id"]
        for item in enriched
        if not item["dependencies"]
        and item["pairable"]
        and item["planning_state"] not in {"external", "blocked_hardware"}
    )
    pairable_tasks = [
        item for item in enriched if item["pairable"] and item["expected_days"] > 0
    ]
    schedule_points = sorted(
        {item["earliest_start_day"] for item in pairable_tasks}
        | {item["earliest_finish_day"] for item in pairable_tasks}
    )
    peak_pairs, peak_pair_day = max(
        (
            sum(
                item["earliest_start_day"] <= point < item["earliest_finish_day"]
                for item in pairable_tasks
            ),
            -point,
        )
        for point in schedule_points
    )
    peak_pair_day = -peak_pair_day
    resource_forecast = max(resource_finish.values(), default=0.0)
    model_driver_lanes = []
    for code, name, minimum_hardware, production_hardware, initial_state in MODEL_DRIVER_PROGRAMS:
        lane_id = f"model-driver:{code.lower()}"
        members = [item for item in enriched if item["agent_lane"] == lane_id]
        model_driver_lanes.append(
            {
                "id": lane_id,
                "model": name,
                "task_prefix": f"MOD-{code}",
                "task_count": len(members),
                "initial_state": initial_state,
                "minimum_hardware": minimum_hardware,
                "production_hardware": production_hardware,
                "provider_request_slots": PROVIDER_REQUESTS_PER_PAIR,
            }
        )
    return {
        "schema_version": 3,
        "program": "sparkpipe-full-heterogeneous-inference-platform",
        "baseline_date": BASELINE_DATE,
        "decisions": {
            "large_qwen_name": "Qwen 3.8 Max",
            "compute_precision": "The full-precision class uses a BF16/FP16 spine with FP32 accumulation and numerically sensitive reductions; narrower routes are separate reduced-precision build identities",
            "initial_models": ["Qwen 3.8 27B", "DSV4 Flash", "GLM 5.2", "K3", "DSV4 Pro", "Qwen 3.8 Max", "MiniMax H3"],
            "provider_fee": "10 percent of capacity actually sold through SparkPipe, paid in qualified capacity credits rather than cash",
            "hardware_backends": ["NVIDIA CUDA", "AMD ROCm", "Apple Silicon Metal", "CPU reference"],
            "agent_redundancy": AGENT_REDUNDANCY,
            "agent_continuation": "continue with the valid winner plus a different failure-domain backup using identical provider-neutral context",
            "model_driver_agent_policy": {
                "dedicated_logical_pair_per_model": True,
                "persistent_provider_neutral_context": True,
                "independent_auditor_context": True,
                "cross_model_edits_require_coordinator": True,
            },
            "model_driver_lanes": model_driver_lanes,
            "sota_release_policy": {"maximum_age_hours": 24, "parity_required": True, "economic_target_ratio": 1.10},
        },
        "dispatch_policy": {
            "broad_pair_gate": "OXA-012",
            "live_state_overlay_required": True,
            "states": ["planned", "refining", "ready_for_implementer", "implementing", "patch_sealed", "auditing", "audit_rejected", "coordinator_review", "integrated", "blocked"],
            "provider_request_slots_per_pairable_task": PROVIDER_REQUESTS_PER_PAIR,
            "minimum_independent_provider_failure_domains": MIN_PROVIDER_FAILURE_DOMAINS,
            "provider_supply_freshness_hours": PROVIDER_SUPPLY_FRESHNESS_HOURS,
            "model_driver_lane_affinity_required": True,
            "dispatchable_when": "semantic dependencies integrated, dispatch prerequisites integrated, executable contract admitted, lock and typed hardware lease available, enough request slots exist across at least two independently qualified provider failure domains with <=24-hour supply evidence, and no stale heartbeat",
        },
        "resource_forecast_assumptions": {
            "capacities": PLANNING_CAPACITY,
            "algorithm": "deterministic conservative greedy schedule using expected durations, worker pools, 2+2 provider-race request slots, typed hardware quantities, semantic/dispatch prerequisites, and exclusive write locks",
            "external_calendar_lead_excluded": True,
        },
        "workstreams": WORKSTREAMS,
        "summary": {
            "task_count": len(enriched),
            "workstream_count": len({item["workstream"] for item in enriched}),
            "expected_engineering_effort_days": round(effort, 1),
            "unconstrained_critical_path_days": round(program_duration, 1),
            "root_count": len(roots),
            "sink_count": len(sinks),
            "planning_pairable_root_count": len(planning_roots),
            "pairable_task_count": len(pairable_tasks),
            "unconstrained_peak_pairs": peak_pairs,
            "unconstrained_peak_pair_day": round(peak_pair_day, 1),
            "critical_task_count": len(critical_tasks),
            "critical_path_p90_days": round(program_duration + 1.282 * (critical_variance ** 0.5), 1),
            "resource_constrained_forecast_days": round(resource_forecast, 1),
            "required_release_closure_count": len(required & release_closure),
            "external_calendar_lead_excluded": True,
        },
        "roots": roots,
        "sinks": sinks,
        "critical_tasks": critical_tasks,
        "critical_edges": critical_edges,
        "representative_critical_path": representative_critical_path,
        "workstream_summary": workstream_summary,
        "planning_roots": planning_roots,
        "tasks": sorted(enriched, key=lambda item: (item["phase"], item["workstream"], item["id"])),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_visualization(graph: dict[str, Any]) -> str:
    payload = {
        "summary": graph["summary"],
        "workstreams": graph["workstream_summary"],
        "critical": graph["representative_critical_path"],
        "planning_roots": graph["planning_roots"],
        "tasks": [
            {
                "id": task["id"],
                "workstream": task["workstream"],
                "title": task["title"],
                "dependencies": task["dependencies"],
                "expected_days": task["expected_days"],
                "start_day": task["earliest_start_day"],
                "finish_day": task["earliest_finish_day"],
                "resource_start_day": task["resource_start_day"],
                "resource_finish_day": task["resource_finish_day"],
                "resource_wait_days": task["resource_wait_days"],
                "slack_days": task["slack_days"],
                "critical": task["critical"],
                "phase": task["phase"],
                "gate": task["gate"],
                "hardware": task["hardware"],
                "kind": task["kind"],
                "state": task["planning_state"],
                "resource": task["resource"],
                "write_locks": task["write_locks"],
                "hardware_requirements": task["hardware_requirements"],
                "successors": task["successors"],
                "recurring_days": task["recurring_days"],
                "freshness_hours": task["freshness_hours"],
                "pairable": task["pairable"],
                "dispatch_class": task["dispatch_class"],
                "dispatch_prerequisites": task["dispatch_prerequisites"],
                "provider_request_slots": task["provider_request_slots"],
                "provider_failure_domains_required": task["provider_failure_domains_required"],
                "agent_lane": task["agent_lane"],
                "acceptance": task["acceptance"],
            }
            for task in graph["tasks"]
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = VISUALIZATION_TEMPLATE.read_text(encoding="utf-8")
    if template.count("__PERT_DATA__") != 1:
        raise ValueError("visualization template must contain exactly one data placeholder")
    return template.replace("__PERT_DATA__", encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="write the validated graph JSON atomically")
    parser.add_argument(
        "--write-visualization",
        type=Path,
        help="write a self-contained interactive HTML fragment atomically",
    )
    parser.add_argument("--summary", action="store_true", help="print a compact summary")
    args = parser.parse_args()
    graph = validate_and_schedule(build_tasks())
    payload = json.dumps(graph, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        atomic_write(args.write, payload)
    if args.write_visualization is not None:
        atomic_write(args.write_visualization, render_visualization(graph))
    if args.summary:
        print(json.dumps(graph["summary"], indent=2, sort_keys=True))
        print("planning_roots=" + ",".join(graph["planning_roots"]))
    if args.write is None and args.write_visualization is None and not args.summary:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
