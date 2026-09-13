# Wave R3 — Roofline Program, lanes: ling, gemma4, laguna, muse, minimax (2026-09-13)

Operator directive 09-13: coredev stack unstable, execution gated. The lanes went
productive WITHOUT execution: deep driver audit + memory-roofline estimation +
algorithm iteration toward >=50% of roofline. No weightd/daemon contact, no
module/model execution, spark3/spark6 untouched.

## Measurement receipts (capped probes, sparkcap scope on spark0/spark1 only)

Probe (a), userspace stream read microbench, `tools/devcycle/bw_probe.cu`, 1.5 GiB
buffer, 8 iterations, under
`sudo -n systemd-run --scope -q -p MemoryMax=4096M -p MemoryHigh=2900M --uid=1000`:

```
spark0: bytes=1610612736 iterations=8 elapsed_ms=53.614 bandwidth_gbps=240.3
spark1: bytes=1610612736 iterations=8 elapsed_ms=52.819 bandwidth_gbps=243.9
```

MEASURED achievable memory bandwidth: 242.1 GB/s (88.7% of the 273 GB/s spec).
The `0.65 bandwidth convention` used by older estimate docs understates the
hardware; these ceilings use the measured number.

Probe (b), NVMe sequential read, 4 GiB O_DIRECT from a 21.7 GB stagepack, same
cgroup scope:

```
4294967296 bytes (4.0 GiB) copied, 0.916332 s, 4.7 GB/s
```

MEASURED NVMe seq read: 4.686 GB/s — the cold-expert bandwidth for MoE lanes.

Purge after each probe: `sudo -n sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`.

Wave-R1 `tools/roofline_estimator.py` was not on main at work time (verified by
`git ls-tree origin/main tools/` at c698e20); scratch math went into per-lane
estimator tools that parse the model headers per the CONSTANT_AUDIT law.

## Per-lane roofline summary

B1 decode, bytes/token per rank, resident weights at measured BW. Full block
tables live in each lane PR and are reproducible with the lane tool.

| lane | topology | spine weights/token | KV+state/token (ctx 4096) | active experts/token (bf16 pack) | TOTAL | ceiling tok/s [ESTIMATE] | cold-expert ceiling [ESTIMATE] | receipts |
|---|---|---|---|---|---|---|---|---|
| ling | TP16 | 519.8 MB | 42.6 MB | 5.99 MB | 0.569 GB | 425.9 | 48.9 (NVMe-bound) | none — N/A |
| gemma4 31B dense | TP4 | 15.35 GB | 293.6 MB | 0 (dense) | 15.646 GB | 15.5 | n/a | none — N/A |
| gemma4 26B A4B | TP4 | 0.93 GB | 94.4 MB | 23.79 MB | 1.051 GB | 230.4 | 49.2 | none — N/A |
| laguna | TP8 | 0.990 GB | 34.6 MB | 23.96 MB | 1.049 GB | 230.8 | 24.4 | none — N/A |
| muse | TP16 | 3.56 GB | 68.2 MB | 0 (dense, per contract + directive) | 3.629 GB | 66.7 | n/a | none — N/A |
| minimax | — | — | — | — | — | N/A | N/A | N/A |

Context sensitivity: ling 0.800 GB at ctx 32768 -> 302.7 tok/s; muse 3.819 GB at
ctx 32768 -> 63.4 tok/s. muse/ling expert bytes: muse is dense (emit 0 — no
routed experts exist); ling is MoE (512 experts top-8, emitted from the model
header — the directive's "partly-dense" note does not match the code, and the
code is authoritative). All receipts absent: the daemon gate holds, so
% of roofline is N/A for every lane; the ceiling + analytic model + T2 plan are
the deliverable.

## Deep audit findings

Classes: fix-now (correctness/drift hazard), fix-at-touch (hygiene at next
touch), document-only (recorded direction). CONSTANT_AUDIT.md (#979) items are
all glm5_next-scoped; nothing below duplicates them.

### fix-now

1. gemma4 `SPARK_GEMMA4_TP_STANDALONE` env —
   `modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c:171,508,570,610,617`.
   At tp_degree>1 the env skips collective init and every TP submit returns
   SPARK_STATUS_OK without combining; the module logs "results stay rank-partial"
   and serves. Env-var opt-out from correctness, success-reported wrong output
   (I03, I40). Fix: fail explicitly at degree>1, or force degree=1.
2. muse `tp_passive` silent fallback —
   `modules/muse_glimmer_resident_decode_stage/source/spark_muse_glimmer_resident_decode_stage_module.c:186-189,950,992`.
   Degree>1 with `SPARK_MUSE_GLIMMER_STAGE_TP_BACKEND_PATH` absent silently
   replays a tp-sliced pack per rank with no hidden combine (I03: a fallback
   selects an incomplete path; must fail explicitly).
3. ling/laguna `tp_collective_disabled` at degree>1 —
   `modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c:180,1300,1346,1382`
   and `modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_module.c:187,1298,1374,1420`.
   A zero `tp_collective_identifier` in the node context silently returns
   rank-partial success; laguna pins degree to 8 so only the identifier guards
   it. If this is the I38 single-rank component-probe seam, the probe needs an
   explicit probe identity and must fail at degree>1 instead of serving partial
   results.

### fix-at-touch

4. ling config.h restated literals —
   `modules/ling_resident_decode_stage/source/cuda/config.h:54-58`: KV bits/slots
   restated as 16u/64u and the KDA conv-window formula restated beside the model
   header's own define. FIXED in PR #982 with three static_assert ties.
5. gemma4 KV slot bare multipliers —
   `modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_cuda.cu:19-21`
   (`HEAD_DIMENSION*4u`, `*8u`). FIXED in PR #983 via
   `SPARK_GEMMA4_*_KV_SLOT_BYTES_PER_HEAD` + three static_asserts.
6. laguna bare launch literals —
   `modules/laguna_resident_decode_stage/source/spark_laguna_resident_decode_stage_cuda.cu:182,191,203,212,221`
   (255u/256u beside the named SPARK_LAGUNA_CUDA_THREADS). FIXED in PR #984.
7. gemma4 GDN misnomer —
   `model-families/gemma4/include/sparkpipe/spark_gemma4_model.h:60-61`,
   `spark_gemma4_moe_model.h:21-22`, `spark_gemma4_moe_model_aliases.h:56`,
   `modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c:289,313-322`,
   `source/spark_gemma4_stagepack_format.h:125-129`. Execution is sliding-window
   attention (stagepack has sliding K/V tensors; the reference oracle is
   1024-window GQA); the GDN names and the `#define gdn_* sliding_*` macro
   aliases are borrowed-shape leftovers. Rename at touch.
8. ling KDA state slot stride —
   `modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c:840-847`
   sizes the state pool with the full-width
   SPARK_LING_MODEL_KDA_STATE_BYTES_PER_LAYER (2 MiB) while
   `source/cuda/layer.cuh:512` pins the same value and
   `inference/kernels/linear_attn.cuh:148-175` addresses only
   rank_heads x 128 x 128 x 4 B (131 KB at TP16). 16x pool overallocation under
   TP sharding (capacity lever, not bandwidth).
9. gemma4 `SparkGemma4ConfigureCudaKernels` —
   `modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_cuda.cu:398-407`
   drops the cudaFuncSetAttribute status without a cudaGetLastError check.

### document-only

10. ling/gemma4/muse expert planes load eagerly through the common stage-module
    loader; laguna already uses the weightd lazy pack + leases (the I29
    direction). Residency fits today; revisit when packs grow.
11. muse placeholder protocol constant —
    `modules/muse_glimmer_resident_decode_stage/source/spark_muse_glimmer_resident_decode_stage_module.c:30,517,532`:
    `SPARK_MUSE_GLIMMER_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES 4096` feeds
    `kv_plan.gdn_record_bytes` for a model with zero recurrent layers. The value
    is protocol-coupled (KV tier plan), so it needs a designed constant, not a
    drive-by edit.
12. muse `lookahead_packet_count = 3u` (module.c:518) — unnamed magic in the KV
    plan.
13. laguna empty `AttentionPost`/`MlpPost` wave hooks
    (cuda .cu:557-569, called at module.c:1628,1657) — currently no-op phase
    seams; they are the TP-overlap seam for lever work, keep until overlap lands.
14. ling TP16 replicated reads (kv_a, router — 125.5 MB/token/rank, 22% of rank
    traffic) and laguna replicated router (73.9 MB, 7%) — the top byte levers in
    both lanes; owned by the PR #982/#984 lever lists.

Comment sweep: zero comments in any of the four lane trees (I48 clean).
Scaffolding sweep: one placeholder constant (finding 11), no `#if 0`, no
TODO/FIXME/dead debug paths; no `#ifdef DEBUG` clamps found in the four lanes.

## Algorithm-iteration PRs (1:1 per driver)

| PR | lane | content |
|---|---|---|
| #982 | ling | tools/ling_decode_roofline.py + config.h derive ties; levers: replicated-read broadcast (+28% [ESTIMATE]), expert codec, fp8 head |
| #983 | gemma4 | tools/gemma4_decode_roofline.py (31B + 26B) + derived KV slot bytes; levers: TP16 tables, batch amortization (quantization contract-barred) |
| #984 | laguna | tools/laguna_decode_roofline.py + named launch thread counts; levers: router broadcast (+7.5% [ESTIMATE]), expert codec, TP16-with-kv-replication |
| #985 | muse | tools/muse_decode_roofline.py (dense; expert bytes 0); levers: batch amortization (pack law bars quantization) |

Every gain is labeled ESTIMATE per the honest-benchmark law; T2 receipts are the
verdict. Compile evidence: host syntax check plus
`nvcc -gencode arch=compute_121a,code=sm_121a` object builds of ling unity.cu,
gemma4 dense TU, gemma4 `-DSPARK_GEMMA4_MOE_BUILD=1` TU, and the laguna CUDA TU,
all green on a GB10 spark.

## minimax blocker

MiniMax H3 has no driver in the repo: no model-families/minimax, no contract
JSON, no module. It appears only as a product row (docs/MODEL_SUPPORT.md:14) and
a TECHDEBT item ("Add exact checkpoint-derived contracts and native execution
packages for MiniMax H3", TECHDEBT.md). Per the never-fabricate law no roofline
table or audit was produced; the 1:1 PR is impossible without a driver. The lane
deliverable becomes: contract-first onboarding (checkpoint-derived geometry ->
model header -> estimator -> packer), then this wave's tooling applies as-is.

## T2 measurement plan (when the daemon stabilizes)

1. Deploy merged main per I33 on a TP set for one lane at a time.
2. Seed a reusable cache entry, verify the hit (I23), warm decode on the
   persistent engine (I41), B1 O128 per the GLM receipt convention.
3. Record `--receipt-tok-s <measured>` per lane tool; % of roofline prints
   against the 242.1 GB/s measured roofline. Gate: >=50% of ceiling per the
   directive (ling 213, laguna 115, muse 33, gemma4-26B 115, gemma4-31B 7.75
   tok/s at their modeled topologies).
4. Re-run the two probes after any driver change that moves ≥100 MB/token to
   keep the roofline denominator current.
