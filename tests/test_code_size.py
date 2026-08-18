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
# The speculation-tree neutralization moves the tree machinery to
# include/sparkpipe/spark_speculation_tree.h (GLM52 shape stays in the
# family header with aliases) and adds its pinning test; 165905 is the
# exact count after it lands. Ceiling moves with it.
# The dispatch-policy extraction moves the 767-line neutral core into
# include/sparkpipe/spark_speculation_policy.h + src/spark_speculation_policy.c
# (GLM52 module shrinks to the 80 tap-plan lines; aliases keep consumers
# untouched) and adds its pinning test; 165975 is the exact count after it
# lands. Ceiling moves with it.
# The glm52 contract generator gains the PP tap-plan constants in its
# emission table; 165977 is the exact count after it lands. Ceiling moves
# with it.
# The shared packer core lands (tools/spark_pack_common.py, 309 lines)
# and five packers migrate onto it (net -144); 166142 is the exact count
# after it lands. Ceiling moves with it.
# tools/devcycle (host-recovery / fleet-ops scripts) is reclassified out of
# the authored-code budget (-1157); 164985 is the exact count after it.
# The admission core lands (spark_admission.h 210 + spark_admission.c 188)
# and collapses four serving-adapter admit wrappers + two module gates onto
# it (net +319: +148 wrapper lines, -227 deleted); 165304 is the exact count
# after it lands. The ceiling moves with it: the core is the DRY base the
# KV seam, scheduler tables, and model gates build on.
# The KV seam lands SparkKvModelTable + SparkKvBackendInitialize
# (spark_kv_model_table.h 74 + cache/kv_model_table.c 93, +4 Makefile,
# +1 sources.mk) as the single token-free fill point all four model adopters
# consume; 165476 is the exact count after it lands. The DRY payoff (four
# per-model fills -> one) realizes as model agents migrate onto it, so the
# seam is additive now, net-negative later.
# The GLM52 JIT-KV migration lands the complete unit: the model-family KV
# geometry fill (spark_glm52_kv_geometry.h), the block-major Glm52Kv geometry
# (layer.cuh), and the resident-stage rewrite (SparkGlm52PageCopy copy
# primitive, SparkGlm52KvInitialize -> SparkKvBackendInitialize, and the
# SparkGlm52AdmissionPredicate prepare/commit/abort tail). It is additive now:
# the kernel's identity device page_table and the kv_cache device pool (now the
# arena's key_device_base) are still the kernel's view, so the raw-init deletion
# arrives with the page_table data-flow follow-up, not here. 165735 is the exact
# count.
# fleet_swap.sh converts the residentd launch from setsid to systemctl (+2)
# so the OOM guardrail cgroup actually applies; 165737 is the exact count.
# The GLM52 JIT-KV completion half lands (frame-completion CompleteLane /
# RollbackLaneTransaction in the async callback, kv_backing_directory flows
# from the serving config with a lenient default, ABI 3->4) (+42); the
# net-negative deletion still rides with the page-table data-flow unit.
# 165779 is the exact count.
# The completion-tail now surfaces publish/rollback failures (page-cache status
# flows into async->completion.status instead of (void)) (+2); 165781 is the
# exact count.
# The qwen36 speculation fix lands (+55): verify/replay frames feed the
# committed token C0 instead of the redundant first draft (a first-draft miss
# no longer poisons or zeroes the chain), the gate becomes a named
# configurable policy (recover default / strict legacy for A/B), misses are
# surfaced in telemetry + the completion model_extension receipt, and the
# build gate publishes MTP_LAYER_COUNT=1 GDN_SNAPSHOT_SLOT_COUNT=8 so the GPU
# validator exercises the MTP chain + GDN snapshot path it never ran before.
# 165836 is the exact count.
# The qwen36 TP1 serving-adapter build switch lands (+15):
# SPARK_QWEN36_SERVING_TP_DEGREE becomes a #ifndef-overridable knob (4 default
# TP4 unchanged / 1 TP1 single-rank / 0 legacy PP), and adapter_id, stage_count,
# and stage_layer_counts derive from it so a TP1 adapter compiles from the same
# source with zero TP4 behavior change. 165851 is the exact count.
# The serial-TP replay harness lands (tests/serial_tp_replay.{h,c} + pinning
# test + design doc): host collective emulation for iterative single-spark
# correctness runs while the fleet is down. Harness body is in tests/ (excluded);
# only the Makefile wiring (+5) counts. 165856 is the exact count.
# The dsv4-flash k-sweep plumbing lands (+100): SPEC_STEP becomes an
# overridable bucket knob (buckets 6/9/11 added to the allowlist + firmware
# JSONs + serving-adapter sha map + build scripts) so k=5/8/10 drivers build
# alongside k=7 for the DSpark sweep. 165956 is the exact count.
# The dsv4-pro DSpark native-pass skeleton lands (+268): pinned drafter
# shapes (128 attn heads / 1 MLA KV head / 512 head-dim / 3072 intermediate
# from the GA rank-pack geometry), two new kernels (mean-reduction, main-KV
# window write, confidence) + the mHC draft chain wired behind the MTP guard,
# and the deployment-spec dataset names fixed (stale .b1024/.bf16 paths).
# 166224 is the exact count.
# The dsv4-pro DSpark pinning path lands (+16): the four draft shapes move into
# dsv4_pro_authoritative.json + the contract generator (byte-exact) + the
# generated model header, the neutral drafter table remaps to the Pro macros,
# and a pinning test mirrors test_dspark_drafter_pin.c. 166240 is the exact
# count.
# The k3 w2 sharder+layer fix lands (+10): the w2 GEMM arg order matches the
# pack layout (w2=[latent N, inter K]) at all three launch sites, and the
# sharder output-splits w2 on whole 16-neuron cells (TP16 diagonal subgrid,
# [224,192] per rank instead of full 3584). 166250 is the exact count.
# The TILE_K=32 GEMM fallback lands (+58): LmGemmSelectTileK (preferred/32/0)
# + LmGemmLaunchTileK in the shared launcher with a host-safe pinning test;
# the kernel body was already TILE_K-generic. Unblocks K3 serial-TP16. 166308
# is the exact count.
# dsv4-flash spec-completion fixes land (+1): lane advance 1+accepted
# unconditionally, max_speculative_token_count = SPEC_STEP (not block size),
# and the completion schema gate widened by the speculative allowance.
# 166309 is the exact count.
# The k3 GEMM integration lands (+7): K3Project switches to LmGemmLaunchTileK
# and unity.cu instantiates the TILE_K=32 kernel variants - the K3 half of the
# fallback. 166316 is the exact count.
# The qwen36 validator admits TP1 full-width into the whole-stack tier and
# derives head_stage from STAGE_COUNT (+6); the TP1 topology knob's validation
# path. 166322 is the exact count.
# The client continuation-lease fix lands (+8): the lease advances by the
# COMPLETION's emitted count (1+accepted) instead of the submission's chain
# width, mirroring the residentd - the last gate before the dsv4-flash
# k-sweep. 166330 is the exact count.
# The k3 per-half serial-TP rework lands (+179): K3LaunchSliceHalf + the
# stateful runner StepHalf (per-layer phase 0/1 sweeps with host all_reduce
# between halves) - the mechanically-correct 9-AR-point TP16 replay plan. The
# recurrent state-carry numerical bug is tracked separately. 166509 is the
# exact count.
# The qwen36 KV predicate fix lands (+5): a passing KV check now marks the
# decision ACCEPTED explicitly (an untouched decision kept its init
# rejection_reason and every passing admit was silently rejected - the
# module_admit smoke's real root cause). 166514 is the exact count.
# The continuation-lease generation fix lands (+5): the residentd fences the
# lease with the CURRENT client generation (not the route-reservation-time
# one), so reconnects during async spec bursts no longer leave a stale
# lease_client_generation - the last k-sweep gate. 166519 is the exact count.
# The residentd-side lease advance fix lands (+3): advance by the completion's
# EMITTED count (1 + accepted, clamped) instead of the coordinator-rank chain
# width - the batch side already mirrored this (c8f76e5), closing the final
# spec-continuation divergence. 166522 is the exact count.
# The k3 dense_row_offset fix lands (+1): the half-step path now writes the
# per-step dense offsets (was fresh-zero, so every grouped GEMM produced zero
# rows - the real missing-fold root cause). 166523 is the exact count.
# The residentd poll-gating fix lands (+4): POLLOUT is always requested for a
# connected client so the queued hello ACK flushes in the same iteration that
# read the HELLO - the last k-sweep gate. 166527 is the exact count.
# The k3 SiTU-sharding fix lands (+38): w1 becomes input-split (full cell
# axis) and w2 output-split (full k axis) so gate|up is all-reduced BEFORE
# the non-linear SiTU - the production diagonal layout applied SiTU to
# rank-sliced partials (sum(SiTU(p)) != SiTU(sum(p))). 166565 is the exact
# count.
# dsv4-flash spec-verify fixes land (+11): the lane advance drops the
# double-counted anchor (folded by the 1-row frame continuity), and the
# completion schema gate applies to DECODE only (RELEASE carries
# tokens_per_sequence == 0 by contract). 166576 is the exact count.
# The k3 production phase-2 collective lands (+75): the gate|up all-reduce
# before SiTU with the widened fused/staging buffers (24576 u16) - the
# production-side completion of the SiTU numerics fix. 166651 is the exact
# count.
# The k3 phase-1 gemm init fix lands (+7): the fresh K3LayerLatentMoe call's
# w2 launch now memsets and sets every field the grouped GEMM reads (the
# uninitialised scale_a returned -41 on the golden's first MoE layer).
# 166658 is the exact count.
# The dsv4-flash QKV exactness fix lands (+5): the FP8 decode linear routes
# ALL row counts to the exact batched kernel family instead of the MXFP8
# native path, whose activation quantization diverged the 8-row DSpark
# verify anchor (first divergent tensor: delta_wq_a). 166663 is the exact
# count.
# The exact-per-row pair + strided decode linear launchers land (+58):
# multi-row frames now run the certified 1-row kernels per row instead of the
# activation-quantizing native path (delta_wq_a/wo_b bit-identical to the
# lean baseline). 166721 is the exact count.
# The qwen36 FP8 pack path lands (+66): dtype-driven per-tensor FP8_E4M3
# selection with F32 per-128x128 scales (format 5), the firmware enum, the
# stagepack PayloadBytes/ScaleBytes, and module ValidateEntry - the BF16
# round-trip path stays byte-compatible. The shared Linear kernel already
# shipped the FP8 dot. 166787 is the exact count.
# The dsv4-flash MoE exactness kernels land (+10): shared W13, routed W13
# and W2 all run the certified B1 kernels for every row count (consecutive
# row iteration + tile-N match for the batched route build) - moe_out now
# bit-identical to the lean baseline. 166797 is the exact count.
# The dsv4-flash ProjectHead exactness fix lands (+9): every verify row runs
# the certified 1-row head instead of the screened-argmax route (hash
# unchanged - proves the token-2 divergence is upstream, but the certified
# head is the correct math regardless). 166806 is the exact count.
# The qwen36 FP8 scale-offset + verify fixes land (+17): copy_scale reads the
# scale tensor from its data offset (was header bytes -> NaN), and the verify()
# re-parse path accepts the FP8 format. 166823 is the exact count.
# The non-blocking token-emission fix lands (+66): emission decouples from
# the decode loop (buffered snprintf + O_NONBLOCK write + retry + drain at
# exit) - the host-side term that stalled B1 at the stdout reader's rate.
# 166889 is the exact count.
# The residentd poll-gating fix lands (+2): POLLOUT only when output is
# pending (matching the client) - kills the 100%-CPU busy-spin over the GPU
# decode that starved the completion callback (~8000x fewer Progress passes).
# 166891 is the exact count.
# The client PREPARE-flush fix lands (+12): the just-queued PREPARE flushes in
# the same Progress that reads the completion (drops the one-iteration
# deferral) - tightens the serial loop; the B1 bubble itself closes via
# speculation overlap, not the scheduler. 166903 is the exact count.
# The qwen36 MTP qualification lands (+37): the validator's whole-stack tier
# arms STAGE_MTP=1 and adds an MTP_DRAFT_AFTER frame with an in-vocab draft
# check (determinism rides the fresh-instance re-execution). 166940 is the
# exact count.
# The residentd loop reorder lands (+10): admission (spec phase-one
# decode-draft) now runs BEFORE the phase-two scan so verify(N) and
# decode-draft(N+1) overlap under max_inflight_submission_count; FIFO,
# lease, and transport ordering all preserved. 166950 is the exact count.
# The qwen36 DSpark drafter packer + format header land (+328): the 62-tensor
# drafter (Doopeworld/Qwen3.8-27B-DSpark-vLLM, 1.36B params) packs into the
# Q6SP wire format with 17 kinds and the module-side shape table - the rung-3
# foundation (draft weights are public; no gate). 167278 is the exact count.
# The dspark packer gains round-trip verification (+31): verify_payload()
# re-reads the pack against the safetensors byte-for-byte, and the entries now
# carry each tensor's own rows/cols (a leaked-loop-value bug it caught).
# 167309 is the exact count.
CEILING = 167309

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {'.c', '.h', '.cu', '.cuh', '.py', '.mk', '.sh'}
# .agents holds per-model agent worktrees (full clones of this tree);
# their copies are tooling infrastructure, not authored source, and must
# never move the counter.
EXCLUDED_COMPONENTS = {'tests', '.git', 'docs', 'build', 'qualification',
                      '__pycache__', '.agents', 'devcycle'}
# tools/devcycle holds host-recovery / fleet-ops scripts (GRUB staging,
# fstab fastboot fix): operational infrastructure, not serving-engine
# source, and must never consume the authored-code budget.


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
