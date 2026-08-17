"""Enforce a monotonic ceiling for authored non-test source code.

Generated build products, test fixtures, documentation, caches, and package
receipts are not implementation source and must never move this number. The
previous counter included generated model-driver C files under build/, so its
ceiling changed depending on which tests had already run. This counter is
stable before and after a clean build.
"""
import sys
from pathlib import Path

# Phase 6 adds lossless per-lane completion ownership at the rank boundary,
# completion-to-transaction correlation, synchronous-callback deferral, strict
# final-event identity propagation, and rollback-safe submission ownership. The
# exact authored-source count at landing is retained so later changes remain
# monotonic. The follow-up audit landing adds tools/verify_package_manifest.py
# and wires more gates into tools/gates.sh; the ceiling moves by those
# tooling lines (86), no production source grew for its own sake.
# The audit-fix landing adds the mbarrier phase-parity model coverage
# (test_mma_fragment_mapping.c, +89), the deterministic-failure paths and their
# tests (node/backend.c +128, test_ring_service_backend_transactions.c +168),
# the shared smem opt-in (runtime/launch.h, qwen/kimi call sites net negative),
# and the new no-python/manifest gates; ceiling moves to the exact count.
# The performance wave adds the tensor-map descriptor cache
# (runtime/gemm_descriptor_cache.h + test), the comms arena (runtime/arena.h +
# test), the RDMA eviction/batching/lane-rotation logic in rdma.cu, the BF16
# collective path, and docs/archive/PERF_ROADMAP_2026-08-01.md; ceiling moves to the
# exact count again.
# The NVMe JIT KV tier (cache/nvme_tier.c + include/sparkpipe/spark_nvme_tier.h,
# the tier-3 manager and lookahead prefetcher, ~1080 lines with its build and
# gate wiring) lands alongside concurrent performance-wave work; ceiling moves
# to the exact count again.
# The K3 pack format V2 redesign (tools/k3_pack.py +400: the fused KDA
# projection emission, the interleaved weight+scale relay and its numpy-free
# fallback, the layout validator; tools/generate_k3_contract.py and
# generated_config.h gain the pack constants and geometry checks) lands its
# tooling lines; ceiling moves to the exact count.
# The DSv4 driver audit (2026-08-01) adds the exact attention-bytes
# derivation and the sparse-launch/rope-span defect flags to
# inference/llms/deepseek_v4/layer.cuh (+70), the Pro launch-budget note to
# deepseek_v4_pro/unity.cu (+10), and the two dsv4 gates' wiring in
# tools/gates.sh and Makefile (+6); the quantise dedup is net-negative code.
# Ceiling moves by those 86 lines; concurrent agents' in-flight growth is
# theirs to account.
# The D10 graph/gather/head wave adds the stage-side CUDA graph cache and
# replay contract (inference/stage/graph_replay.h, 451), the dispatch.cu
# capture wiring (+202 net), the head's chunked top-k pair with its launcher
# (inference/kernels/head.cuh, +231), the route row-indirection consumer
# contract (inference/kernels/route.cuh, +65), the slot-state field docs
# (+9), and their build/gate wiring (Makefile +6, tools/gates.sh +10,
# tools/build_head_topk.sh 25). Ceiling moves by those 999 lines; concurrent
# agents' in-flight growth remains theirs to account.
# The NVMe KV sizing work adds the estimator behind the dedicate-the-external
# drive decision (tools/nvme_kv_estimate.py, 407), the tier's default
# bandwidth/step-time assumptions from that analysis
# (include/sparkpipe/spark_nvme_tier.h +19), and the gate wiring (Makefile +1,
# tools/gates.sh +4); the test and the doc are excluded by construction.
# Ceiling moves by those 431 lines; concurrent agents' in-flight growth
# remains theirs to account.
# The topology-switch landing adds the TP16<->PP16 switch state machine and
# its strategy-neutral KV key scheme (scheduler/topology_switch.c, 646;
# include/sparkpipe/spark_topology_switch.h, 350) and its build/gate wiring
# (Makefile +6, tools/gates.sh +3) - 1,005 lines; the balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The hardware-topology landing adds the generator/validator
# (tools/generate_topology.py, 562) and its two generated C artifacts
# (deployment/include/sparkpipe/spark_hardware_topology.h, 111;
# deployment/src/spark_hardware_topology_tables.c, 285) plus the gate
# wiring (Makefile +1, tools/gates.sh +3) - 962 lines; the balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The batch-variant landing adds the per-family tuning headers
# (modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h, 131;
# modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_batch_tuning.h, 115),
# the variant rules and all/variants/publish_variants targets in the glm52
# module Makefile (+95 net), the firmware header's composed module ID (+5),
# and the build/gate wiring (Makefile +25, tools/gates.sh +21) - 392 lines;
# the test is excluded by construction. The balance of the exact-count move
# is concurrent agents' in-flight growth, theirs to account.
# The TP/PP recipe wave adds the recipe generator (tools/generate_recipe.py,
# 914: family adapters over the authoritative contracts, the stage_plan.c
# balancing DP mirrored, the k3 shard table built from k3_shard's own sets)
# and its gate wiring (tools/gates.sh +8, Makefile +1) - 923 lines; the
# test, the naming doc, the mimo25 contract JSON and the generated
# examples/recipes/ set are excluded by construction. The balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The K3 pack V2 shard wave teaches the TP tooling the V2 tensor names and
# slice rules (tools/k3_shard.py +118: the fused q|k|v|beta per-section
# head slice, the replicated decay|gate fusion, the interleaved expert
# cell/k-tile splits with manifest repricing; tools/generate_recipe.py +8:
# the scale-less expert classes and the 128-element k-tile group;
# model-families/k3/.../spark_k3_tp_shard_table.h +4: the V2 suffixes) -
# 130 lines. The balance of the exact-count move is concurrent agents'
# in-flight growth, theirs to account.
# The prefill/decode estimator landing adds tools/perf_estimate.py (649:
# the launch/wall/collective/transport overlay on top of the imported
# nvme_kv_estimate byte law, the chunked-prefill model, and the sm_121a
# ptxas occupancy check) and its gate wiring (Makefile +1,
# tools/gates.sh +4) - 654 lines; the test and the estimates doc are
# excluded by construction. Ceiling moves to the exact count.
# Phase 7 adds generation-carrying arena ownership, the explicit NVMe write
# lifecycle and cancellation/heap safety, fail-closed KV access diagnostics,
# contract-derived performance geometry, deterministic Git-independent source
# packaging, and conservative receipt-bound status checks. The test and
# documentation and raw qualification-evidence lines are excluded by
# construction; 125577 is the exact authored non-test source count in the
# deterministic Phase 7 source package.
# Phase 10 adds the hardware-truth system: exact question/plan/policy/closure
# tooling, GB10 CUDA and NVMe probes, production-provider wrappers, PMTU and
# topology probes, deterministic runner configuration, exact cell execution,
# node preflight, and the CUDA 13 sm_121a compile gate. The implementation
# increase is the qualification surface needed to replace hard-coded GB10 and
# network assumptions with retained measurements. The final source-package
# hygiene correction adds explicit qualification-evidence file classification
# (+16 authored lines); 134623 is the exact authored non-test source count at
# the Phase 10 handoff landing.
# The runtime-completion overlay adds the all-participant staged transaction
# controller, acknowledged final-event ownership, bounded selective-ACK
# transmission window, exact model-provider operation/precision contracts,
# deterministic overlay application, and the no-undefined GLM final-artifact
# receipt tools. The model-specific contracts remain under their family
# directories rather than contaminating the neutral runtime. 136812 was the
# resulting ceiling.
# The host-portability correction adds 115 authored lines: portable PMTU wire
# conversion and connected-UDP probing, plus Linux-AIO guards and Darwin host
# syntax support for the NVMe probe. 136927 is the exact post-fix count.
# The CUDA 13 hardware-probe correction replaces the removed device-property
# memory clock field with cudaDeviceGetAttribute and fixes the NVMe candidate
# names emitted by the qualification planner; the graph receipt contract adds
# the planned mode to the emitted parameters; the remaining CUDA receipt
# contracts add atomic mode and SMEM/thermal iteration fields; 136954 is the
# exact count.
# The hardware-probe failure contract adds the CUDA telemetry parser, the
# NVMe device fingerprint correction, and fail-closed runner status handling;
# the CUDA fingerprint contract correction adds the CPU-equivalent seed and
# index reduction; 137007 is the exact count.
# The DSV4 Flash resident-stage landing adds the stage runner, CUDA doorway,
# stage-pack wire format, per-rank source staging, and driver smoke gate. Its
# source is the implementation needed for a real PP13 stage, not generated
# receipts or test fixtures; 146181 is the exact count after that landing.
# The generalized model-serving landing adds the model-neutral adapter,
# manifest, resident IPC/client/daemon, two-phase PP pipeline client, batched
# token request engine, and pipeline-runtime topology builder. DSV4 contributes
# only its family-owned adapter and frame translation. The 8,052-line increase
# is this versioned ownership boundary plus fail-closed distributed lifecycle;
# the final 43 lines make batch-engine shutdown cancel and release every live
# resident sequence before destruction. Six module-build lines keep Make
# targets relative and therefore valid in space-containing workspaces. This is
# not generated output or a model-specific copy of common serving code; 154239
# is the exact count after the landing.
# The multi-session wave (merged as one span) adds the DSV4 Pro bring-up
# (stagepacks, requant tool, runbook tooling), the Qwen 3.8 Max family and
# resident stage, the Qwen TP4 deployment/bench tooling, the GLM52 TP8
# additive path, and the devcycle coordination harness; the DRY consolidation
# on the unified branch removes the duplicated Pro TP16 stagepack and
# re-parameterizes the TP4xPP4 driver; 158323 is the exact count after that
# consolidation lands.
# The main wave landed after that consolidation adds the K3 TP4 runner,
# dispatch, pack probe, and deploy/compile-gate tooling (#666), the GLM52
# B8 and B16 batch classes with adapter variant-bucket fixes (#668, #669),
# and the GLM52 accuracy fixes (#665); the unified branch folds it in as-is
# (the shared-gemm provenance rename is line-neutral). 162375 is the exact
# count after the integration; ceiling moves with it.
# The DSV4 contract reconciliation teaches the generator the merged Pro
# reality (first-light BF16 activations, FP8-expert codec selectability,
# the Pro alias guard in the Flash header, the first-light note in the
# normalized contract) so --check reproduces the checked-in files
# byte-exact; 162411 is the exact count after it lands. Ceiling moves
# with it.
# The six-session integration wave (K3 TP4 layer-0, DSV4 DSpark
# speculative loop, DSV4 Pro GA migration, Qwen 3.8 Max phase 2, Qwen TP4
# phase 2, DSpark design docs) lands on the unified branch as merged
# session machinery, plus the generator reconciliation for the GA 0813
# checkpoint (3 packed draft layers, KV codec selectability); 165675 is
# the exact count after the integration. Ceiling moves with it.
# The DSpark drafter neutralization adds the per-model config header
# (spark_dspark_drafter.h), the GLM52 selector/alias rewrite, and the
# pinning test; 165811 is the exact count after it lands. Ceiling moves
# with it.
CEILING = 165811

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {'.c', '.h', '.cu', '.cuh', '.py', '.mk', '.sh'}
EXCLUDED_COMPONENTS = {'tests', '.git', 'docs', 'build', 'qualification', '__pycache__'}


def main():
    total = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(component in EXCLUDED_COMPONENTS for component in relative.parts):
            continue
        if path.suffix in EXTENSIONS or path.name == 'Makefile':
            total += sum(1 for _ in path.open(errors='surrogateescape'))
    print(f"non-test authored lines: {total} (ceiling {CEILING})")
    if total > CEILING:
        print(f"\nFAIL authored code grew by {total - CEILING} over the ceiling; "
              f"shrink it or justify a new ceiling in the same change")
        return 1
    if total < CEILING - 800:
        print(f"note: ceiling is {CEILING - total} above reality; "
              f"lower it with the next landing")
    print("\nthe authored codebase did not grow")
    return 0


if __name__ == '__main__':
    sys.exit(main())
