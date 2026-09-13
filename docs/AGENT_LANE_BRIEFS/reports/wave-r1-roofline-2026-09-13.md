# WAVE-R1 ROOFLINE — qwen38max / qwen3flash / qwen38-27b (2026-09-13)

Operator directive 09-13: lanes go productive WITHOUT execution — deep code
audit + memory-roofline estimation + algorithm iteration toward >=50% of
roofline. Execution stays gated; every gain below is ESTIMATE until T2.

## 1. Measurement sources (all within the capped-probe law)

sparkcap probes ran on spark0/spark1 only, under
`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M --uid=1000`,
with `sync; echo 3 > /proc/sys/vm/drop_caches` purge after each:

| probe | result |
|---|---|
| userspace stream read, 20 threads, 1 GiB, spark0 (best of runs) | 107.8 GB/s (104.7–107.8 across runs) |
| userspace stream read, 16 threads, 1 GiB, spark0 | 106.9 GB/s |
| userspace stream write, 16 threads, spark0 | **112.0 GB/s** (best single result, 97.1–112.0 across runs) |
| userspace stream, spark1 cross-check (20T) | read 105.1 / write 108.4 GB/s |
| NVMe direct sequential read, 4 GiB, spark0 | **5.1 GB/s** |
| GPU-side achievable (retained receipts, `docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md`: eta_bw ~0.80, three independent derivations) | **~218 GB/s** (0.64–0.78 band per the 12x model) |

SPEC ceiling: 273 GB/s (LPDDR5X, platform spec). The userspace CPU probe
saturates at ~112 GB/s (~41% of spec) — a CPU-side lower bound on the bus,
not the GPU's stream rate; the 27b receipt independently implies ~240 GB/s
GPU-side (section 4). Both denominators are carried.

## 2. Canonical estimator

`tools/roofline_estimator.py` (PR #986) parses stagepack wire layouts
(magics, field order, tensor-kind numbering, weight-format codes) from the
repo's format headers, fetches pack manifests + receipts read-only over ssh,
and parses `/Users/mac/sparkpipe-coord/MEMORY_MODEL.md` for cross-check.
Derived constants are never hardcoded. Validation: parsed per-class byte
totals reconcile to file bytes on qwenmax.nvfp4.tp16 rank0 (98.11 GB) and
qwenmax.pp16 stage0 (96.92 GB). Probe source:
`tools/sparkcap_stream_probe.c` (-Werror green on spark0).

bytes/token/rank model (B1 decode): embedding row + spine stream/B +
lm_head/B + touched experts `moe_on_rank x (1-(1-k/N)^B)` (= moe x k/N at
B1, exact for replicated and expert-sharded placements) + lookup-rows term
(PLE/ngram touch rows, not the table) + KV read/append + GDN state R/W +
named activation round-trips (12/layer). HOT prices expert bytes at memory
BW; COLD prices them at the measured NVMe BW.

## 3. Roofline table (estimator output, packs parsed from spark0)

| lane | arm (node) | bytes/token/rank hot | SPEC 273 ceiling | GPU-eta 218 ceiling | probe 112 ceiling | COLD NVMe 5.1 ceiling | receipted tok/s | % of roofline |
|---|---|---|---|---|---|---|---|---|
| qwen38max | qwenmax.nvfp4.tp16 rank0 (TP16, 92L, 98.1 GB) | **12.411 GB** | 22.00 | 17.60 | 9.02 | 2.41 | N/A end-to-end; banked GPU-time multipliers: first deletion 2.04x (~79 ms/frame), bf16-staging 1.34–3.47x bit_exact (PR #928) | N/A (never fabricated) |
| qwen38max | qwenmax.pp16-stripped stage0 (PP16 stage, 6L, 96.9 GB) | 7.661 GB | 35.63 | 28.51 | 14.62 | 2.59 | N/A | N/A |
| qwen3flash | qwen3flash.nvfp4.tp8 rank0 (TP8, 48L, 23.9 GB) | **2.588 GB** | 105.5 | 84.7 | 43.3 | 18.5 | N/A (arms are smoketests only) | N/A |
| qwen3flash | qwenflash.tp8.fp8 rank0 (TP8, 48L, 30.5 GB) | 2.717 GB | 100.5 | 80.4 | 41.2 | 12.6 | N/A | N/A |
| qwen38-27b | qwen27b.tp4 rank0 (TP4, 64L, 10.5 GB) | **8.092 GB** | 33.74 | 26.99 | 13.84 | 13.84 (dense: no expert lease) | 8.00 no-spec B1 on the fp8/mixed ~29.9 GB TP1 pack (docs/archive/QWEN38-27B_HILLCLIMB.md A6, HWM c729071; pack per docs/archive/DFLASH2_BENCH_PLAN.md) | see below |

qwen38max TP16 arm split: spine 6.624 + lm_head 4.068 (replicated
full-width) + touched experts 1.628 (83.35 GB on-rank MoE x 10/512) + gdn
state 0.072 + activations 0.018 + embedding row.
qwen3flash nvfp4.tp8 split: spine 2.232 + lm_head 0.159 (sharded) + touched
experts 0.166 + gdn state 0.028 + activations 0.003 + PLE touch ~0.01; the
pack also carries 12.87 GB of replicated PLE/ngram lookup tables (finding F7).
qwen38-27b split: spine 7.373 + lm_head 0.636 (sharded) + gdn state 0.075 +
activations 0.008 + embedding row. Collective check (max lane): ~115
ops/token at d2a 78us = ~9 ms/token = ~111 tok/s — the byte roofline binds
first at B1 on every lane.

qwen38-27b % of roofline (receipt's own arm, arithmetic — that pack is not
on a readable node, bytes are archive-derived): implied stream ~240 GB/s =
87.6% of SPEC; against GPU-eta 218 GB/s the receipt sits at ~100-110%
(over-unity within the archive figures' rounding, flagging either the 29.9
GB pack figure or eta 0.80 as slightly understated for the TP1 config).
On the parsed TP4 arm the 8.00 receipt does not apply (different pack,
different topology). Verdict: the 27b lane already sits at the memory
roofline; only byte-cutting pays.

## 4. Dominant gaps and top-3 levers per lane (ESTIMATE gains)

qwen38max (12.411 GB/token/rank, dominant: spine_stream):
1. lm_head is replicated 4.07 GB/rank (33% of arm bytes) — vocab-parallel
   screened head: -4.05 GB/token/rank -> 8.36 GB; probe-BW ceiling 9.02 ->
   13.4 tok/s (+48%), GPU-eta 17.6 -> 26.1 (+48%). EST.
2. bf16-staging dense-linear occupancy — BANKED 2.30-3.47x per-stage
   (gdn_out 2.30x, attn_out 3.47x, gdn_linear_in 1.34x, attn_linear_in
   1.75x, all bit_exact); PR #991 unblocks running it on the placed TP16
   packs (they fail-closed on main's v1-only loader — finding F1).
3. Expert residency: COLD ceiling 2.41 tok/s binds if the 83.35 GB/rank MoE
   leases from NVMe; the 6 GB pool + sequential expert-gather keeps the HOT
   column. Keep pool hit-rate near 1 under co-residency. EST: preserves the
   hot roofline instead of a 6.2x cold stall.

qwen3flash (2.588 GB/token/rank, dominant: spine_stream incl. PLE residency):
1. PLE shard-narrowing or lease-streaming: 12.87 GB/rank replicated lookup
   tables; pack 23.9 -> ~11 GB/rank, residency and cold-lease bytes roughly
   halved; hot per-token traffic unchanged (lookup touches rows). EST.
2. TP8 multi-rank B>1 hill-climb (lane step 5 + B1): at B8 spine+head
   amortize 8x while touched experts rise to 8.49x(1-(1-10/512)^8) = 1.28
   GB -> ~1.6 GB/token/rank -> probe ceiling ~70 tok/s/rank, aggregate
   ~560 EST; transport trio re-eval is the gate (lane STATUS).
3. nvfp4 prefill tile-Mloop qualification (the gate task behind the removed
   comment): opens the nvfp4 wire for prefill; decode already runs the
   proven grouped-scalar path.

qwen38-27b (8.092 GB/token/rank, dense — at the roofline):
1. Dense-FFN nvfp4: FFN is the 65.4%-of-time term (rung-2 profile); halving
   its bytes moves the parsed 8.092 GB arm toward ~5.5 GB — probe ceiling
   13.8 -> ~20 tok/s, GPU-eta 27.0 -> ~39. EST. Needs packer + dense-nvfp4
   kernels (the calibration doc prescribes exactly this).
2. DFlash2 speculation (implemented, lane parked): amortizes the whole walk
   across accepted tokens — archive law ~x1.9 effective at full sweep,
   ~x1.0 at B1 top-k; acceptance-length lift is the variable to measure.
3. Batch: weight streams amortize /B (B1/B16 aggregate x3.71 -> x20.99 on
   the glm52 calibration corpus); the B-ladder is the free lever before any
   new kernel work. EST.

## 5. Audit findings ledger (file:line at HEAD c698e20; cross-checked vs
docs/CONSTANT_AUDIT.md #979 — its items are all glm5_next/weightd/
transport; ZERO overlap with these three lanes)

Fix-now (landed in this wave's PRs):
- F1 modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c:370 + source/spark_qwen38_max_stagepack_format.h:105 — loader rejects the PLACED v2 128B TP16 packs (format_version 2, tail tp_degree/tp_rank, directory at 128; wire-verified on qwenmax.nvfp4.tp16 rank0). Main could not serve the flagship arm; T2 unmeasurable. Severity: high. FIXED in PR #991 (v2 wire view + normalize + derived directory check + optional pack_load_common hook, no-op for other families).
- F2 modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_cuda.cu:1341 — SPARK_QWEN38_ROUTER_SORT_CAPACITY 512u restates SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT with no tie (bitonic sort silently corrupts if the count changes). Severity: medium. FIXED in #991 (derived + pow2 _Static_assert).
- F3 modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_cuda.cu:2063 — same 512 restatement under the WRONG family name (SPARK_QWEN38_* in the qwen4_flash file). Severity: medium. FIXED in #992 (renamed, derived, asserted).
- F4 Comments in code — 14 sites, law violation, all removed: max lane 6 (module.c 393/473/1681/1684/1703 pre-edit, validation .cu 1106, model.h 60), flash lane 4 (model.h 17, stagepack_format.h 525, module.c 1700, cuda.cu 2469), 27b lane 4 (module.c 1094, stagepack_format.h 468, validation .cu 1109, plus the serving-adapter spec notes re-homed to the PR). Severity: low. FIXED in #991/#992/#993.
- F5 modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:1796 — gdn snapshot clamp restates 7u where MAX_MTP_DRAFT_TOKENS-1u is the law. Severity: low. FIXED in #993.
- F6 modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_module.c:289 — profile log cadence `& 63u` bare literal. Severity: low. FIXED in #993 (named period define).

Fix-at-touch (flagged, owner+packer work):
- F7 qwen4_flash NarrowShape has no PLE case: 12.87 GB/rank of PLE/ngram lookup tables replicate across TP8 (parsed). Pack 23.9 -> ~11 GB/rank if sharded/leased. Flagged in #992; needs packer + module together. Severity: medium (residency/co-residency budget impact).
- F8 qwen38-27b dense-FFN nvfp4 (the lane's only paying lever; needs packer + dense-nvfp4 kernels). Flagged in #993. Severity: medium.
- F9 qwen38max TP16 lm_head replicated 4.07 GB/rank (33% of arm bytes). Vocab-parallel screened head. Severity: medium (largest single remaining byte term on the flagship arm).

Document-only:
- F10 Doc-vs-wire drift: docs/archive/QWEN38-27B_HILLCLIMB.md section B calls the deployed 27b weights "BF16"; the placed qwen27b.tp4 wire carries fp8 (family weight-format code 5) GDN/FFN payloads with bf16 embedding/attention. Record the mixed-precision reality where the receipt is quoted.
- F11 qwen27b.tp4 rank0 has 27,461,184 B of overlapping payload/scale ranges (parsed 10,551,882,624 vs file 10,524,421,440). Harmless for traffic accounting (each entry still streams its bytes); worth a line in the packer's next receipt.
- F12 Stagepack header revision policy: v2 shipped in placed packs before the format header landed on main. CONSTANT_AUDIT should require wire-format revisions to land with the packer that emits them (this is the root cause of F1).

Verified non-findings: all three stagepack headers carry full _Static_assert suites including sizeof wire pairs; all three modules validate pack geometry at boot (SparkStagePackHeaderMatches family / expected-geometry macros); the archived 27b A1 (build gate never exercised speculation) and A3 (replay_tokens OOB) findings are already LANDED on main (qwen38_tp4_build.sh:27 arms MTP=1/GDN_SNAPSHOT=8; the replay array is MAX_MTP_DRAFT_TOKENS+1u); no TODO/FIXME/scaffolding markers in any of the three lanes; dspark/mtp paths are live selectable code, not leftover scaffolding.

## 6. PRs and ledger

- PR #986 — tools: canonical roofline estimator + sparkcap stream probe (branch tools/roofline-estimator).
- PR #991 — qwen38max: v2 128B stagepack acceptance + derived router capacity + decomment (branch lane/qwen38max-roofline).
- PR #992 — qwen3flash: derive router capacity (drops wrong-family 512 restatement) + decomment; PLE replication flagged (branch lane/qwen3flash-roofline).
- PR #993 — qwen38-27b: derive snapshot clamp + profile cadence + decomment; roofline verdict (branch lane/qwen38-27b-roofline).
- This ledger: branch lane/wave-r1-roofline.

## 7. What was NOT verified

- No module/model execution anywhere (directive): every tok/s ceiling is analytic; the banked multipliers (2.04x, 2.30-3.47x) are prior T2-stage GPU-time receipts quoted from STATUS.md/SHARED_DECISIONS, not re-measured.
- The CUDA sources in the three lane PRs compile in CI only; locally only the pure-C paths (synthesize tools, work_control, stagepack_format runtime) were compiled (-Wall -Wextra, -Werror on the test harness) and the v1/v2 header harness was run (PASS).
- The 27b receipt's fp8/mixed TP1 pack was not found on readable nodes (spark3 is off-limits); its byte figure is archive-derived, not manifest-parsed.
- Probe coverage: spark0 + spark1 only, CPU userspace + one NVMe pass; GPU-side bandwidth is inferred from retained in-repo receipts, not directly probed.
