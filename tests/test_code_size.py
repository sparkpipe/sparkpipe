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
# The qwen38_27b speculation fix lands (+55): verify/replay frames feed the
# committed token C0 instead of the redundant first draft (a first-draft miss
# no longer poisons or zeroes the chain), the gate becomes a named
# configurable policy (recover default / strict legacy for A/B), misses are
# surfaced in telemetry + the completion model_extension receipt, and the
# build gate publishes MTP_LAYER_COUNT=1 GDN_SNAPSHOT_SLOT_COUNT=8 so the GPU
# validator exercises the MTP chain + GDN snapshot path it never ran before.
# 165836 is the exact count.
# The qwen38_27b TP1 serving-adapter build switch lands (+15):
# SPARK_QWEN38_27B_SERVING_TP_DEGREE becomes a #ifndef-overridable knob (4 default
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
# The qwen38_27b validator admits TP1 full-width into the whole-stack tier and
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
# The qwen38_27b KV predicate fix lands (+5): a passing KV check now marks the
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
# The qwen38_27b FP8 pack path lands (+66): dtype-driven per-tensor FP8_E4M3
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
# The qwen38_27b FP8 scale-offset + verify fixes land (+17): copy_scale reads the
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
# The qwen38_27b MTP qualification lands (+37): the validator's whole-stack tier
# arms STAGE_MTP=1 and adds an MTP_DRAFT_AFTER frame with an in-vocab draft
# check (determinism rides the fresh-instance re-execution). 166940 is the
# exact count.
# The residentd loop reorder lands (+10): admission (spec phase-one
# decode-draft) now runs BEFORE the phase-two scan so verify(N) and
# decode-draft(N+1) overlap under max_inflight_submission_count; FIFO,
# lease, and transport ordering all preserved. 166950 is the exact count.
# The qwen38_27b DSpark drafter packer + format header land (+328): the 62-tensor
# drafter (Doopeworld/Qwen3.8-27B-DSpark-vLLM, 1.36B params) packs into the
# Q6SP wire format with 17 kinds and the module-side shape table - the rung-3
# foundation (draft weights are public; no gate). 167278 is the exact count.
# The dspark packer gains round-trip verification (+31): verify_payload()
# re-reads the pack against the safetensors byte-for-byte, and the entries now
# carry each tensor's own rows/cols (a leaked-loop-value bug it caught).
# 167309 is the exact count.
# The DSpark tap-capture foundation lands (+61): firmware ABI 4 with the
# DSPARK_DRAFT_AFTER flag + draft view, and the module's per-slot tap buffer
# capturing target hiddens at layers {4,16,28,40,52} during the decode loop.
# 167370 is the exact count.
# The dsv4-flash CSA isolation fixes land (+57): the prologue gates the
# compressor/indexer to the anchor row for verify/pad frames (draft rows
# never advance the committed state), and the commit replays rows 1..accepted
# through the committed compressor (rejected rows keep the old value) - the
# keep-old rollback contract. 167427 is the exact count.
# The DSpark parity reference lands (+236): the numpy golden draft script
# (tools/qwen38_27b_dspark_reference.py) whose rope view bug was the last blocker
# between the CUDA drafter and the reference tokens - the fixed script is
# the bit-exact golden for draft parity and must stay in-tree.
# 167663 is the exact count.
# The multi-row exact-linear routing lands (+58): DecodeLinearPair and
# StridedDecodeLinear multi-row launches now run the certified 1-row
# kernel once per row (per-row activation quantization) instead of the
# native MXFP8 per-tile route - the batch-coupled scale was the first
# divergent tensor (delta_wq_a) in the DSpark verify island.
# 167721 is the exact count.
# The parity-verified DSpark drafter port lands (+628 net): the 5-layer
# block-diffusion forward + device Markov head + serving-adapter dspark
# wiring + C0-anchor/draft-remap fixes, rebased onto the landed tap-capture
# foundation (the drafter is env-gated: SPARK_QWEN38_27B_SERVING_SPEC_METHOD=
# dspark + optional pack path; greedy baseline = 5.078 tok/s, 10.5% acc).
# 168349 is the exact count.
# The CUDA compile gate gains the qwen38_27b resident stage module (+9): the
# DSpark port landed without sm_121a gate coverage (the gate listed only
# glm52/dsv4 stages) - now the port's .cu is CI-compiled every push.
# 168358 is the exact count.
# DFlash2 W1+W8 lands (+170): the 21-kind drafter packer (81-tensor
# inventory, selector codebook slots, grouped-conv kinds) and the numpy
# parity oracles ported verbatim from the vLLM PR unit tests (the W2/W3
# kernel gate - no upstream reference implementation exists).
# 168528 is the exact count.
# W8 main() wiring lands (+18): the DFlash2 full-forward numpy smoke
# (sliding attention + conv wrapping + top-16 -> lattice -> walk) runs
# end-to-end against the downloaded checkpoint - the end-to-end parity
# reference for the W2/W3 kernels.
# 168546 is the exact count.
# W2 lands (+41): the fused grouped depthwise conv kernel (per-token
# coefficients, block-boundary zeroing, one elementwise pass) + launch
# wrapper - the DFlash2 drafter's first new kernel.
# 168587 is the exact count.
# +1 for the DRAFTER-path env override line in the DFlash2 reference.
# 168588 is the exact count.
# +1: the dspark .cuh includes the format header (W6 constants were
# undefined in the CI gate compile - the qwen38_27b gate coverage caught it).
# 168589 is the exact count.
# +79: the W2 conv parity harness (kernel vs _grouped_conv oracle,
# assert_allclose PASS) - the DFlash2 kernel regression gate.
# 168668 is the exact count.
# +9: W2 host wiring part 1 (conv weight struct + loading + side param).
# 168677 is the exact count.
# +111: DFlash2 module integration - selector weight loading + host mirrors
# replacing the Markov slots, the conv-wrapped block forward (prepare/finish
# per sublayer), the host top-16 + selector lattice + greedy walk pass, the
# dflash2 adapter method (block 8), and the fail-loud drafter-pack guard.
# 168788 is the exact count.
# +60: env-gated DFlash2 stage-dump bisection (SPARK_QWEN38_27B_DFLASH2_STAGE_DUMP)
# - the parity-debug surface, removed after forward parity lands.
# 168848 is the exact count.
# +107: conv kernel reads the fused [B, sides, taps, groups] delta directly
# (side param, full row stride - the pointer-offset variant read the wrong
# rows) + the stage-bisect tool (tools/qwen38_27b_dflash2_bisect.py).
# 168955 is the exact count.
# +11: spec acceptance clamped to the shared tokens-per-sequence ABI cap.
# 168966 is the exact count.
# +38: env-gated tap-capture/dump diagnostics (Nth-capture dump for
# spec-vs-no-spec tap parity at matched positions).
# 169004 is the exact count.
# +47: verify-tail re-draft - the drafter consumes the verify frame's row-0
# (anchor) hiddens so taps match the prefill-written committed trajectory
# (decode-frame taps drift 5-16% over prefill state; acceptance collapsed).
# 169051 is the exact count.
# +3: the draft hangs off the replay tail (taps = last committed row),
# not the verify tail (row 0 = the anchor itself - off-by-one vs training).
# +19: drafter-run tap dump (SPARK_QWEN38_27B_DFLASH2_RUN_DUMP=N) - the exact
# taps the drafter consumed, for oracle replay.
# 169089 is the exact count.
# +47: cache-based drafter context (upstream precompute_and_store_context_kv):
# per-position tap history, context-window fc/norm, per-layer staged K/V,
# k/q prep + cache attention kernels; supersedes the dual-source attention.
# 169136 is the exact count.
# +163: pair-atomic rope prep + layer-0 stage dumps + the
# cache-semantics checker tool (tools/qwen38_27b_dflash2_cache_check.py).
# 169299 is the exact count.
# +10: bonus-token convention (anchor = the frame's last input row;
# context window excludes the bonus position; adapter remap shifts).
# 169309 is the exact count.
# +6: serialize the GDN step + conv update rows (the recurrence races
# across multi-row frames found by the DSV4 session - verify/replay left a
# wrong GDN state while per-row outputs stayed golden; the tap-drift source).
# 169315 is the exact count.
# +4 more for the serialization (grid.y removals).
# 169319 is the exact count.
# +37: the replay walks the GDN STEP path (the DSV4 session's
# silent-divergence fix, unified 2bd2673) - the chunk/step fp32-rounding
# difference accumulated per round and decayed acceptance.
# 169356 is the exact count.
# +46: the drafter attention applies per-head norm+rope in f32 at
# attention time (the original dual-source rounding path); the bf16
# pre-prepped q/k flipped round-1 drafts.
# 169402 is the exact count.
# +4: tap-store grid.y channel tile.
# 169406 is the exact count.
# +5: the tap-layer evidence-trail comment (revert to [5,19,33,47,61]
# after the reference cross-test).
# 169411 is the exact count.
# +24: bonus-token anchor (the frame's last input row) + the shifted
# adapter remap - fixes the one-step-late drafts on natural text.
# 169435 is the exact count.
# +8: Nth-run neighborhood tap dump (context-convention sweep).
# 169443 is the exact count.
# +112: layer-0 stage dump Nth-gated to the ctx-dump round (the
# device-vs-oracle forward divergence bisection on GSM inputs).
# 169555 is the exact count.
# +6: anchor-id dump in the ctx-dump round.
# 169561 is the exact count.
# +7: anchor-id dump with the ctx round.
# 169568 is the exact count.
# +85: state-select verify (the vLLM shape) - per-row GDN state+tail
# checkpoints in the step kernels, checkpoint-select restore, 1-row replay.
# 169653 is the exact count.
# +2: checkpoint alloc moved after pool strides.
# 169655 is the exact count.
# -30: checkpoints reuse the GDN snapshot pool (no new buffers).
# 169625 is the exact count.
# +1: whitelist the select-restore flag.
# 169626 is the exact count.
# +8: the conv-tail per-row checkpoint (full bf16 elements - the
# first attempt never landed and the restore read stale tails).
# 169634 is the exact count.
# +23: SPARK_QWEN38_27B_DFLASH2_STATE_SELECT env gate (default OFF =
# the validated replay path; the select path stays opt-in WIP).
# 169657 is the exact count.
# +2: checkpoint slot-base wiring.
# 169659 is the exact count.
# +108: the bonus fold (the vLLM round shape, SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD
# env gate, default OFF) - the correction tail drafts (anchor = its emission)
# and the next round's verify row 0 walks the client token in the decode
# walk's place: 3 full-model frames per round drop to 2. Commit math is
# shape-aware (m+2 folded vs m+3) and the fold disarms on any plain decode,
# prefill, or bootstrap so desyncs self-heal.
# 169767 is the exact count.
# +19: deep-acceptance debugging instrumentation - the ctx dump's anchor
# read moves to the frame EMISSION (the token the block actually embeds;
# the input-row read mislabeled the oracle sweeps), the device's walk
# output dumps beside the ctxwin taps, and the round diag prints all 8
# draft/emitted positions (the collapse is position-2+ only).
# 169786 is the exact count.
# +154 (net, replacing the racy one-shot dumps): per-run parity captures at
# the walk-end stream sync - taps slice + the TRUE walk anchor + the device
# walk output, race-free, consumed round-by-round by the deep-parity scorer
# to localize the position-2+ acceptance collapse (WIP instrumentation).
# 169940 is the exact count.
# +416 (tools incl.): THE conditioning fix, from the live convention sweep on
# 44 O128 rounds (torch, tools/qwen38_27b_dflash2_conv_sweep.py): the drafter
# wants HF Qwen3 NeoX rope over the full 128-dim head (not interleaved-64),
# the context must INCLUDE the walked row's tap g_P at RoPE position base-1 -
# llama.cpp's seed pair (t_{P+1}, g_P) - and the walk emits own-position
# drafts, so output 0 is redundant and the verify remap shifts by one
# (draft_count caps at block-1). Sweep: p0 41%->70%, p1 31%->59%.
# 170356 is the exact count.
# +86: sweep-fix tail (adapter remap comment rewrite, module geometry comment,
# draft-count block-1 cap wording).
# 170442 is the exact count.
# +2: parity tool default moves to the fixed geometry.
# 170444 is the exact count.
# +92: the ONE-FRAME round (SPARK_QWEN38_27B_DFLASH2_BONUS_FOLD=2) - dspark view
# ABI v2 gains multi_block_count (block i anchored on verify row i's emission
# at base+i; the host picks block m post-accept), the module loops the block
# forward per row, and the adapter runs the round as ONE verify frame (row 0
# restores the previous accept's GDN checkpoint via GDN_RESTORE_VERIFY_ROW;
# no correction frame), committing m+1 and arming slot m.
# 170536 is the exact count.
# +12: the one-frame bring-up - the continuity rule treats a VERIFY_ROW
# restore like a full restore (the one-frame verify re-establishes the lane
# at the branch point), plus bring-up trace prints (tail drafter status).
# 170550 is the exact count.
# +18: padding-select - the verify-tail drafter computes the verify's own
# accept depth (emissions vs the walked rows, one tiny D2H) and drafts ONLY
# block m; the per-block selector is host-bound so k-1 of k blocks were pure
# overhead (measured: one-frame at 3 blocks ran 6.55 tok/s vs 8.76 2-frame).
# 170572 is the exact count.
# +78: the device-side selector front-end (fused top-16 + hidden projection
# kernel, exact two-key order and mul+add rounding to bit-match the host
# scalar path) + a compact ids/scores/hproj copy - removes the ~4MB logits
# D2H and the ~25-35ms scalar 7x248320 insertion pass per drafter call.
# 170650 is the exact count.
# +62: the selector parity oracle (SPARK_QWEN38_27B_DSPARK_SEL_CHECK=1 runs the
# original scalar host pass beside the device kernel and prints divergences)
# - it caught the kernel's rank>threads hproj hole (outputs 128..255 read as
# zero, halving the walk's edge scores: -6% O128). Zero mismatches post-fix.
# 170714 is the exact count.
# +128: the vLLM input-parity harness - the reference driver is patched
# (VLLM_DFLASH2_INPUT_DUMP) to dump its drafter-context hiddens/bonus/
# positions per round; the harness runs OUR block forward + walk on those
# exact inputs and scores draft agreement. Finding: perfect agreement on
# structured stretches, ~50% divergence on varied rounds - the acceptance
# mountain (their E=4.30 vs our ~1) localizes to the block invocation.
# 170881 is the exact count.
# +111: THE acceptance unlock in the engine - the drafter's NeoX-128 rope
# (HF rotate_half over the full head, re-applied; the trained convention)
# and the persistent block-KV history (per-layer raw k/v rows of every
# block the drafter ran, keyed by position, attended after the current
# block rows; SPARK_QWEN38_27B_DFLASH2_BLOCK_KV=1). Both validated on the
# reference input dumps: 87% pos-0 draft agreement, curve mirrors theirs.
# 170992 is the exact count.
# +199 stale at the handoff commit (506770e): the session's last tooling
# commits (the fp8/bf16 tapdiff and layer-bisect tools) landed without
# their ratchet - found when re-running the gate from a clean checkout.
# +233: the argmax selection unlock - draft selection is the per-mask-row
# full-vocab ARGMAX (the reference's SERVING path; the codebook walk is
# dflash2-speculator code that never loads in vllm serve). The engine walk
# is REPLACED by rank-0 selection (net engine shrink); the new lines are
# the stage-diff tool (live-reference per-stage tensor comparison) and the
# parity harness's SELECT_MODE (argmax default, walk modes kept for study).
# 171424 is the exact count.
# +213: the MX serving format - FP8_E4M3_E8M0B128 (format 6: E4M3 payload,
# per-row e8m0 scales per 128-K group, the native SM121 block-scaled fp8 MMA
# layout). Loader + byte-calcs + the SparkQwen38_27bLaunchLinear dispatch route
# (native launcher for its shapes, per-row loop for the rest), plus the
# repack tool (tools/qwen38_27b_stagepack_mx_repack.py - measured LOSSLESS: the
# pack's tile scales are already powers of two, so the conversion is a pure
# re-layout with zero quantization cost).
# 171637 is the exact count.
# +79: the engine-faithfulness proof - the numpy-oracle replay on the
# engine's own full-prefix taps (new tool) matches the engine's argmax
# drafts EXACTLY; the oracle's apply_rope_neox (the stale interleaved
# variant kept for history); the ctxrun dump widened to the full prefix.
# 171716 is the exact count.
# +66: the incremental context cache - position-keyed per-layer K/V plus
# the fc/normed watermark (the reference's precompute-and-store semantics:
# only the round's NEW committed rows are projected; the wide fc runs
# per-row to stay on the lean wide-B1 kernel). Stream-invariant (measured:
# cache and no-cache produce identical token streams), O512 wall 70.6s ->
# 67.7s and the window cost flattened (W=2048 == W=256, so the full window
# rides free with the better acceptance).
# 171782 is the exact count.
# +83: the native-linear micro-benchmark tool (tools/qwen38_27b_native_linear_bench.cu)
# - the measured record for the FFN kernel project: the byte-load variant
# WINS (125.6 GB/s vs 91.7/83.7 for 4-byte loads - the outstanding byte
# transactions are the needed memory-level parallelism), K-split does not
# help, and the path past ~125 GB/s is a cp.async shared-staged B tile.
# 171865 is the exact count.
# +225: the staged-B kernel exploration harness (tools/
# qwen38_27b_native_staged_bench.cu) - a bit-exact cp.async double-buffered
# variant of the native MMA linear plus the PIPELINE-DEPTH LAW: this kernel
# family's bandwidth ~= (in-flight bytes/CTA / DRAM latency) x resident
# CTAs x SMs (16KB/600ns x 4.5 x 10 ~= the measured 113-125 GB/s; pure
# coalesced reads reach 266+). The deep-ring (4+ stages) is the
# theory-backed path past it; the depth-2 ring is verified bit-exact at
# parity with the direct kernel.
# 172144 is the exact count.
# +328: the warp-specialized kernel SKETCH (tools/
# qwen38_27b_native_warp_specialized_bench.cu) - the design that escapes the
# shared-budget tension (producer warps stream the B ring continuously,
# consumer warps mma, collective A rendezvous). DEADLOCKS: the named-barrier
# arrival matrix needs re-derivation (documented in the header). The
# measured map it builds on lives in qwen38_27b_native_staged_bench.cu.
# 172472 is the exact count.
# +101: the warp-specialized kernel VERIFIED BIT-EXACT
# (139264/139264 at both depths) after fixing three subtle races (per-thread
# cp.async groups need a group barrier before publish; the 2-slot A ring
# needs consume-gating; spin gates need group barriers, not __syncwarp) and
# the 512-thread launch (SPARK_LM_CTA_THREADS is 256 - the producers never
# ran at 256). Perf at parity (126.4); the leaner-sync path to ~155 is
# documented in the harness header.
# 172573 is the exact count.
# +19: the warp-specialized perf LEDGER - the decisive
# STAGE_ONLY experiments (pure-B producers 179.6-182.2 GB/s; +A-inline 136;
# +consumers 133; A-on-consumer 127 WORSE) proving the A-quantize is orphan
# work needing overlap, the lean-sync recipe that reconstructs the verified
# 133.0, and the 180+ path (cp.async raw-A + shared-side quantize or a
# third warp group; TMA for the last 45 to the 266 pure-read ceiling).
# 172592 is the exact count.
# +291: the production WS header fixed for deployment - the
# prologue's b_scale_tile staging restored (the extraction regex had
# deleted it; garbage scales corrupted everything), the producer's A-input
# L2 prefetch restored (cold consumer reads cost 50 GB/s), and one-time
# cudaFuncSetAttribute. Result: 173.9-175.6 GB/s bit-exact on every
# production shape; O512 9.30 tps (+23%), E=4.94, bit-lossless.
# 172883 is the exact count.
# +49: the M=1 GEMV fix and the cross-request stability fix.
# SparkLmDotRowFp8E8m0 (the scalar decode for the e8m0 pack layout - at M=1
# the pure-streaming GEMV measures 228 GB/s vs the WS kernel's 171; the
# no-spec MX regression 103.5s -> 81.4s, parity with F32B128's 80.1s) plus
# rows 1-4 routing to it; and the block-KV history reset on a backward base
# (without it, later requests on the same daemon attend the previous
# sequence's stale rows and acceptance collapses run over run: 57.6 -> 79.7
# -> 85.2s measured on identical prompts; now stable 57.8/57.5/57.5s).
# 172936 is the exact count.
# +114: the frame-graph WIRING (opt-in, SPARK_QWEN38_27B_FRAME_GRAPH=1):
# per-(rows,prefill) capture in RunFrame, warm-then-capture-then-replay
# (the K3 pattern), eager uploads outside, capture-aware syncs (Finish +
# profile-head guarded), capture-fail = loud error (GDN rerun unsafe).
# Two sync blockers found and fixed; a THIRD invalidating call remains in
# the layer path (site=ffn cascade) - the hunt continues from
# graphs_broken diagnostics. Default OFF; production verified unaffected
# (80.9s no-spec, same stream).
# 173061 is the exact count.
# 173409 (2026-08-21): prefix caching - module prefix GDN pool + snapshot/
# restore transfers + continuity exception for borrow lanes; adapter prefix
# store (refcounted blocks, LRU entries), publish/borrow glue in the plain
# frame builder, RELEASE submission path (REQUIRES_RELEASE + residency echo);
# batch tool sequential_submissions mode (the arrival pattern the cache
# serves). Verified: borrow bit-identical output, 11.2s saved per repeat.
# 173452 (2026-08-21): drafter-history rollback hygiene - far-backward
# vs intra-sequence rollback split (watermark rewind + stale-row filter),
# lane continuity keyed on request generation (frame scalar[0]) with the
# base-0 restart rule (same-batch reruns against a live daemon).
# 173457 (2026-08-21): frame graphs default ON (kill-switch preserved) -
# the spec-graph anomaly proved resolved (FFN TP-reduce guard); bit-identical
# verified on both paths.
# 173994 (2026-08-21): timestamped spec_diag round telemetry (adapter) +
# the multi-row dot GEMM bench (tools/, four variants, negative-result
# ledger: 160.3 GB/s best vs WS 176.8 - the M=1 248 GB/s scalar does not
# generalize to M>=8).
# 174101 (2026-08-21): plain-B WS kernel variant (uint4 load+store B
# staging, default ON, kill-switch SPARK_QWEN38_27B_WS_PLAIN=0) + consumer
# acquire fence + publish barrier; bench ledger updated with the in-situ
# A/B (kernel-level +10%, end-to-end neutral, bit-exact).
# 174145 (2026-08-23): incident fixes - pack-load device-memory preflight
# (watchdog-restart SEGV -> clean capacity_exceeded), daemon client-session
# submission-id reset on hello, engine prefill lane concentration (full-
# width frames instead of 1-row-per-lane at B>1).
# 174603 (2026-08-23): daemon session-death slot unbind (the "all cells
# fail until restart" poisoning fix), submission-rejection logging (daemon
# + client), and node/model_api.c - the standard OpenAI-style HTTP entry
# point (single engine session, health endpoint working; completion path
# WIP - see the commit message).
# 174635 (2026-08-23): null guards on ALL debug-dump file writes (the
# "speculation does not work" report: CTX_DUMP/L0_DUMP env + failed fopen
# = fwrite(NULL) SEGV on the first decode round).
# 174923 (2026-08-24): JIT-KV design contract + backing-store tier 1
# (runtime/spark_kv_backing.c: slot file, alloc/free, 4 MiB block I/O,
# horizon exhaustion -> backpressure signal; unit test ALL PASS).
# The fleet-stability PRs #715-#717 (staged PXE rescue, brickproof
# expansion, Ceph startup quarantine removal) landed 18 authored lines
# past the ratchet without moving it; the qwen38 rename completion
# (firmware description files + references + validator identity) is
# line-neutral on this counter. 175279 is the exact count.
# The audit-response hardening adds the transactional KV-restore unwind
# in the qwen38_max module, explicit credit-buffer ownership in the 27B TP
# path (extracted AllocateCreditMemory), the stub fault-injection ledger,
# and its fault test; 175399 is the exact count.
# The correctness-audit response: K3 descriptor/submission/cleanup fixes,
# CoverLane incremental commits, uint16 block refs, the tap-plan
# declarations, nvcc host guard, and rename-completion for the test gate
# (work-control symbols, qwen36 test files and contents, fabric hosts,
# tp_degree in glm52 stagepack calls); 175433 is the exact count.
# The pipeline-e2e lease fix: a zero-emitted decode completion no longer
# fails the continuation-lease decode and kills the daemon - the lease
# holds at the lane context instead; net +7 with comments.
# The e2e restoration: the dsv4 flash TP4PP4 stage-layer table gains its
# correct 43-layer layout (plus build comment), and the pipeline test
# expectations move to the concentrated-prefill/overlapped-admission
# semantics with the capacity-tail invariant; 175449 exact.
# Audit-3 correctness fixes: QMax eviction reverse-map invalidation
# (stride field + guarded clear), GLM page-copy completion (the worker
# reuses its staging buffer immediately), and the API request/buffer
# reclamation (queue-locked event walk, unlink-then-free, body base
# ownership); 175483 exact.
# Audit-4 API/scheduler fixes: the API submits the whole queued set
# (not just the head) with aligned engine limits and explicit context
# for single-lane batches (aggregate kept at engine level), and the
# support table reflects GLM 5.2 deprecation and the flash models;
# 175507 exact.
# Qwen 3.8 Flash lane (qwen4_flash): the first module family port lands -
# modules/qwen4_flash_resident_decode_stage (module + serving adapter +
# CUDA stage + stagepack format + pack synthesizer + the validation harness
# ported from the proven qwen38_27b unit), model-families/qwen4_flash
# (geometry header + work control), tools/qwen4_flash_verify_source.py, and
# tests. Sibling-geometry re-parameterization of existing families, not new
# architecture code; 184913 is the exact count after it lands.
# The M1 contract freeze adds the authoritative JSON's census section and
# the M5-prep module work (MTP chain, TP narrowing, admission/snapshot
# ports) lands in the same window; 186345 is the exact count.
# The glm52 validator-fix lane (PR #728) + the packs-lane head remainder it
# carries: the expert-dimension dequant oracle fix with its host-executable
# tier-2 oracle gate, zero-mean fixture grids, the KV page-cache adapter
# lane + admission predicate + JIT_KV config fixes that carried the TP8
# band to a completed B1 decode, plus the packs lane's verify_source /
# deploy tooling and packer evolution (glm52_verify_source.py 278,
# glm52_deploy_packs.sh 88, stagepack +728, pack test +335).
# 192659 is the exact count.
# The glm52 fp8-source packer adoption (from the pack lane) lands the
# numpy-only fp8->bf16 spine dequant packer plus its hermetic test in the
# gate; it emits the deployed TP8 fleet packs.
# The qwen-flash M5 kernel port window (merged from lane/qwen-flash):
# whole-stack TP4 enablement in modules/qwen4_flash_resident_decode_stage
# (TP_STANDALONE bypass, vocab-sharded embedding gather + all-reduce,
# sharded head argmax with the maxloc u64 collective, rank-local GDN
# kernels, dual-width router gate, format-6 E8M0B128 grouped expert
# kernels incl. the family-local scalar variant, MTP draft chain fixes)
# + packer/verifier format-6 per-row plane and gate replication.
# The K3 pack lane lands its verification chain: k3_verify_pack.py (450:
# byte-level cross-check of every rank vs the stage pack, negative-
# controlled), k3_verify_source.py (262: 38-field contract check), the
# rank-0 smoke launcher, and the export-shim diagnostic (retired once the
# canonical-symbol fix merged). model_contracts/references/ joins
# EXCLUDED_COMPONENTS (vendored publisher modeling files pinned as
# semantics ground truth — modeling_qwen4_exp.py is 2707 lines of
# upstream code, not authored source), so the ceiling moves by the lane's
# 835 authored lines only. 188714 is the exact count.
# The qwen38max full-width lane lands its packer-at-scale chain: the
# tp4pp4 pack/deploy tooling (172+55) and the stagepack rewrite (net
# 37) that built the four 573 GiB stage packs with receipts. 189314 is
# the exact count.
# The dsv4pro pack lane (PR #729): parameterized deploy/scaffold/smoke
# tooling (4 scripts) + the report; the 16-rank pack chain with receipts.
# 192936 is the exact count.
CEILING = 192936

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {'.c', '.h', '.cu', '.cuh', '.py', '.mk', '.sh'}
# .agents holds per-model agent worktrees (full clones of this tree);
# their copies are tooling infrastructure, not authored source, and must
# never move the counter.
EXCLUDED_COMPONENTS = {'tests', '.git', 'docs', 'build', 'qualification',
                      '__pycache__', '.agents', 'devcycle', 'references'}
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
