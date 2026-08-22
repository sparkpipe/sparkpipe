# Qwen 3.8 Max (Qwen3.8-2.4T-A95B) bring-up plan — round report

Scope: report-only round, per `max_brief.md`. Sources: `modules/qwen38_resident_decode_stage/`
(module.c 1870 ln, cuda.cu 1848 ln, serving adapter 1248 ln, firmware header, stagepack
format header, pack synthesize tool), `docs/QWEN38_MAX_PLAN.md`, fleet registry entry
`qwen38max` (`tools/devcycle/fleet_registry.json:129-159`), `docs/QWEN38_MAX_PERF.md`,
`docs/QWEN38_MAX_TP16.md`, `docs/QWEN38_MAX_AUDIT.md` refs, `docs/K3_PERF.md`,
`docs/HARDWARE_TOPOLOGY.md`, deployment examples, module git history (1c0d02f..4cfa095).
No files outside this report were modified.

---

## 1. What the driver implements vs what QWEN38_MAX_PLAN.md requires

Plan module items (QWEN38_MAX_PLAN.md §Module plan): (1) geometry, (2) routed-MoE swap,
(3) TP4xPP4 execution pattern, (4) serving adapter + firmware description. Verification
section additionally requires a Torch/HF reference gate and the exact 16-node run; the
packer section requires four rank-local packs per stage.

### 1.1 Implemented and verifiable in-tree

| Plan requirement | Status in driver | Evidence |
|---|---|---|
| Geometry: hidden 8192, 92 layers, 3:1 phase, 16/128 GDN heads, 64/4 attn heads, rope 64, ctx 262144 | Done, compile-time pinned and re-checked against every pack header field at load | `spark_qwen38_model.h`; `spark_qwen38_stagepack_format.h:129-142` (static asserts), `SparkQwen38StagePackHeaderMatches` |
| Routed MoE: gate GEMM + top-10 + shared expert, fused gate_up split to w1/w3 at pack time | Done. Router is softmax-over-top-k (fixed in math audit 1c0d02f), pair-reduce has overwrite semantics, shared expert scaled by scalar `shared_expert_gate` (Linear(8192,1) sigmoid) | `module.c:1499-1563` (`SparkQwen38ModuleRunMoe`), `stagepack_format.h:276-284`, `docs/QWEN38_MAX_MATH_AUDIT.md` via commit 1c0d02f |
| Vendor FP8 experts (E4M3 + F32 block-128 scale_inv), BF16 spine, FP32 GDN state | Done as the natural pack format; FP8 decodes to BF16 fragments under wmma on the tile path (B≥8), scalar grouped path below | `stagepack_format.h:280,285,425-431`; `cuda.cu:1730-1848`; `module.c:1325` tile gate |
| GDN layer kind (fused qkv 2048\|2048\|16384, conv k=4, beta/decay projections, A_log/dt_bias f32, gated head norm) | Done, decode recurrence + conv tails + fp32 state pool | firmware header `SparkQwen38GdnLayerWeights`; `module.c:1430-1464`; state pool `module.c:1592-1621` |
| Gated attention kind (fused q\|gate 32768 rows, q/k RMSNorm, partial rope 64, output gate, paged KV) | Done incl. head-parallel AttnPrepare/AttnDecode kernels taking tp_degree/tp_rank | `module.c:1474-1490`; commit fa1d78e |
| PP-N stages, first-class hidden transport, embedding/head ownership derived | Done; frame context refuses transport-flag disagreement with stage position; prefill/MTP/speculation fail closed (`SPARK_STATUS_UNSUPPORTED`) | `module.c:1692-1715`, `module.c:1837-1838` |
| Stage pack format + strict load validation (geometry, shapes, formats, coverage, duplicates) | Done; per-layer/per-kind seen-bits coverage proof before launch | `module.c:357-586` |
| Serving adapter + firmware description | Adapter done (`SparkModelServingAdapterInterface`, id `spark.qwen38.serving-adapter.tp4-pp4.v1`, tp_degree runtime 4→TP4xPP4 / 16→TP16xPP1); firmware JSON exists (`examples/model_descriptions/qwen38_resident_decode_stage_firmware.json`) | `spark_qwen38_serving_adapter.c:201-231,1164-1228` |
| KV tier (pluggable store, residency window, batched restores, per-lane table uploads) | Done behind `SPARK_QWEN38_STAGE_KV_STORE` (default `none` = all-resident, byte-identical) | `module.c:594-976`; commit 5b9e376 |
| Screened exact head (4-bit shadow + certified bounds + exact rescore) replacing the 4.07 GB/row matvec | Done on the head stage (shadow 1.02 GB + 63.5 MB scale plane) | `module.c:1213-1228,1668-1683` |
| TP expert-sharded MoE + one residual all-reduce per layer (delta, before shared expert) | Module-side wiring landed (env-driven `SparkTpDeviceCollective`), tp=1 byte-identical | `module.c:1002-1153,1538-1547`; commits 4cfa095, fa1d78e |

Single-spark evidence (plan §State, corroborated by perf docs): module compiles clean
(Makefile archive target); real-FP8 pack executes — 1-layer GDN pack (24.9 GiB) and
2-layer GDN+attention pack (49.8 GiB), both layer kinds and both MoE paths,
compute-sanitizer clean; measured microbenchmarks in `docs/QWEN38_MAX_PERF.md` (8.45
ms/layer at B=1 on the real pack). NOTE: the coordination brief says "never
hardware-tested" — reconcile as: *component-level* single-spark smokes/microbenchmarks
ran; **no full-model, no TP>1, no fleet, no serving-stack run has ever executed**, and
`model_contracts/qwen38_authoritative.json:66-69` still says `NOT_MEASURED` /
`production_ready: false`. Treat the plan's validation bullets as unverified until
receipts are re-produced on the target box.

### 1.2 Required by the plan but missing / partial (the bring-up blockers)

1. **TP4 rank-local packs do not exist** (plan open item, §State bullet 4). The packer
   (`tools/qwen38_stagepack.py`) packs whole PP-stage slices only; the docstring is even
   stale (says BF16→MXFP4; the code packs vendor FP8). Without rank-local packs every
   "16-rank" config is fictional — see §3 risk R1.
2. **TP collective never completed a single operation.** The two-process tp=2 smoke on
   spark4 reaches `SparkTpDeviceCollectiveCreate` and returns `CAPACITY_EXCEEDED`
   (backend sizing / mapped-host credit-plane suspects — `docs/QWEN38_MAX_TP16.md`
   "Blocked" section). Everything downstream (tp=2/4/16 smokes, head-sliced projections,
   strided o_proj, 4×-replicated KV heads for tp>4, GDN channel slicing) is queued
   behind it.
3. **The serving adapter cannot turn TP on.** `SparkQwen38ServingSetEnvironment`
   (`spark_qwen38_serving_adapter.c:318-336`) sets the `SPARK_QWEN38_STAGE_*` slice/KV
   env but **none of** `SPARK_QWEN38_STAGE_TP_DEGREE/_RANK/_BACKEND_PATH/_IDENTIFIER/
   _PORT_BASE/_HOSTS/_LOCAL_HOST` — the module therefore always initializes tp_degree=1
   through the serving path, while the stage json's `"tp_degree": 4` is parsed and then
   unused for the driver env.
4. **World-rank vs PP-stage conflation in the adapter.** `SPARK_QWEN38_SERVING_STAGE_COUNT=16`
   with `stage_index = world rank` (adapter comment lines 65-72) makes the module treat
   ranks 1..14 as pipeline middles: every TP peer of stage 0 demands a hidden *input*
   transport and every rank but 15 a hidden *output* transport — wrong for a 4-hop
   pipeline with 4-way TP. The plan itself flags "the TP-rank frame mapping inside the
   adapter is flagged outstanding."
5. **Prefill is refused by the module** (`module.c:1837-1838`) while the adapter
   advertises `CAPABILITY_PREFILL` and splits prefill submissions into frames
   (adapter header comment). Serving-side prefill fails closed today; effective prefill
   is decode-stepping (~1.3 prompt-tok/s, PERF §1b0).
6. **No Torch/HF reference gate** (plan §Verification open item): the math-audit fixes
   and the screened-head "identical argmax" claim are verified against CPU references
   only, never against HF `modeling_qwen3_5` on the real checkpoint.
7. **Batch ceiling inconsistency**: module refuses `max_active_sequences > 409`
   (16-row-tile guard, `module.c:308-317`) while the adapter/descriptor advertise 512.
8. **No CUDA graphs / multi-slot**: `enable_cuda_graph_replay` and slots 1-3 are dead
   config; Execute runs one slot with a per-frame `cudaStreamSynchronize`
   (`module.c:1818`), and the adapter honestly advertises `max_inflight=1`.
9. **Doc/provenance drift** (operator hazards): firmware header + adapter + synthesize
   comments still say "Qwen 3.6 27B" / "rows × 5120"; `qwen38_tp4_build.sh` /
   `qwen38_tp4_deploy.sh` actually drive **qwen36** artifacts; deployment expects
   `lib/libqwen38_tp4_pp4_serving_adapter.so` while the build seam produces
   `libqwen38_serving_adapter.so`; Makefile pins revision `d2dc3565…` while plan facts
   cite HF revision `207bd685…` (BF16) and the actual source is the LOCAL FP8 release.
10. **Qualification contradiction**: the module refuses to initialize without
    `SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION=1`, and the adapter force-sets it while
    calling that "the qualified execution path" (adapter comment line 15) — the contract
    says nothing is qualified.

---

## 2. Bring-up runbook — one GB10 box

Reality check first: **one box cannot run the full model** (§3, R1). The single-box
bring-up therefore proves the pipeline in tiers, each with a pass/fail gate. Paths are
given for `<ROOT>` = the qwen38 worktree, `<CKPT>` = the local FP8 release
(`/home/<user>/sparkdata/qwen38_2.4t_a95b/checkpoint`, stage-sliced shards 54/53/52/54),
`<PACKS>` = scratch with ≥300 GB free on the 4 TB internal NVMe.

### Tier 1 — build + synthetic smoke (~15 min, no weights)

```sh
cd <ROOT>/modules/qwen38_resident_decode_stage
make archive REPOSITORY_ROOT=<ROOT> NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
# build the two smoke tools
cc -O2 -o <PACKS>/qwen38_pack_synthesize tools/qwen38_pack_synthesize.c \
   -Isource -Iinclude -I<ROOT>/model-families/qwen38/include -I<ROOT>/include
cc -O2 -o <PACKS>/test_qwen38_execute <ROOT>/tests/test_qwen38_execute.c \
   -I<ROOT>/modules/qwen38_resident_decode_stage/include \
   $(for h in model-families/common model-families/qwen38; do echo -I\<ROOT\>/$h/include; done) \
   -I<ROOT>/include source/*.o \        # archive objects, or link libqwen38_resident_decode_stage.a
   -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lcuda -lstdc++ -lpthread -lm
# 2-layer GDN+attention synthetic slice (the proven shape)
<PACKS>/qwen38_pack_synthesize --output <PACKS>/q38_synth_l1_2.qwen38sp \
    --first-layer 1 --layer-count 2 --seed 7
```

Gate: `test_qwen38_pack_load` on the synthetic pack, then:

```sh
<PACKS>/test_qwen38_execute <PACKS>/q38_synth_l1_2.qwen38sp
# defaults already match this slice: TEST_QWEN38_STAGE_COUNT=4 INDEX=1 FIRST_LAYER=1 LAYER_COUNT=2
```

Expect: `initialize status=0`, `execute[0] status=0`, `execute[1] status=0`
(second step exercises the carried conv tail + fp32 GDN state). Any `VALIDATION_FAILED`
names the offending pack field — do not proceed on overrides.

### Tier 2 — real-FP8 slice smoke (~30–60 min, ~75 GB disk, ~55 GB device)

```sh
# 1-layer GDN slice (layers 0..0) — needs stage-0 shards present at <CKPT>
python3 <ROOT>/tools/qwen38_stagepack.py --checkpoint <CKPT> \
    --output <PACKS>/q38_fp8_l0.qwen38sp --first-layer 0 --layer-count 1 \
    --receipt <PACKS>/q38_fp8_l0.receipt.json
# ~24.9 GiB output; check the receipt's inventory/shapes against the pinned FP8 index
<PACKS>/test_qwen38_execute <PACKS>/q38_fp8_l0.qwen38sp
# 2-layer GDN+attention slice (layers 1..2), ~49.8 GiB — needs stage-0 shards
python3 <ROOT>/tools/qwen38_stagepack.py --checkpoint <CKPT> \
    --output <PACKS>/q38_fp8_l1_2.qwen38sp --first-layer 1 --layer-count 2
TEST_QWEN38_FIRST_LAYER=1 TEST_QWEN38_LAYER_COUNT=2 <PACKS>/test_qwen38_execute <PACKS>/q38_fp8_l1_2.qwen38sp
```

Gates: both runs `execute[0]/execute[1] status=0`; optionally re-run under
`compute-sanitizer` (the historical clean baseline). Timing sanity vs the measured
anchor: ~8.4 ms/step for the 1-layer slice at B=1 (`docs/QWEN38_MAX_PERF.md` §1).
Useful bisect aids on a failing/slow run: `SPARK_QWEN38_STAGE_DEBUG_SKIP_GDN=1` /
`_SKIP_MOE=1`.

### Tier 3 — first-token smoke, two-process tp=2 on the same box (the real milestone)

This is the first configuration that emits token ids from **real weights** on one box:
two module processes (GDN slice + GDN+attention slice), TP degree 2 over the expert
shards, ports 66620/66621 (inside the registered qwen38max collective block), env:

```sh
SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION=1
SPARK_QWEN38_STAGE_PACK_PATH=<PACKS>/q38_fp8_l0.qwen38sp      # rank A (l0)
SPARK_QWEN38_STAGE_COUNT=4 SPARK_QWEN38_STAGE_INDEX=1         # slice geometry per pack
SPARK_QWEN38_STAGE_FIRST_LAYER=0 SPARK_QWEN38_STAGE_LAYER_COUNT=1
SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES=1 SPARK_QWEN38_STAGE_PIPELINE_SLOTS=1
SPARK_QWEN38_STAGE_KV_BLOCKS=8
SPARK_QWEN38_STAGE_TP_DEGREE=2 SPARK_QWEN38_STAGE_TP_RANK=0|1
SPARK_QWEN38_STAGE_TP_BACKEND_PATH=<ROOT>/build/libhidden_transport_spark_host_rdma_verbs.so
SPARK_QWEN38_STAGE_TP_IDENTIFIER=<any u64>  SPARK_QWEN38_STAGE_TP_PORT_BASE=66620
SPARK_QWEN38_STAGE_TP_HOSTS=127.0.0.1,127.0.0.1  SPARK_QWEN38_STAGE_TP_LOCAL_HOST=127.0.0.1
SPARK_QWEN38_STAGE_TP_TIMEOUT_MS=120000
```

**Known blocker:** this smoke historically dies at `SparkTpDeviceCollectiveCreate →
CAPACITY_EXCEEDED` (`docs/QWEN38_MAX_TP16.md`). Debug order per that doc: backend .so
buffer/queue sizing at (512 rows, credit_count 2, route_count 1); mapped-host credit
plane vs probe mode; field-by-field diff against DSV4's working
`SparkDsv4ModuleInitializeTpCollective` (rail hosts/thresholds). Until it passes, the
Tier-2 status-0×2 gate is the box's "first token" proxy (argmax ids are produced but
meaningless on a middle stage; a head-stage synthetic variant can print an id but
validates nothing semantic).

### Tier 4 — serving-stack boot (documented for the fleet window; NOT single-box)

Artifacts already committed: `examples/deployments/qwen38_fp8_tp4_pp4_host_rdma.spec.json`
(control 22480, adapter `lib/libqwen38_tp4_pp4_serving_adapter.so`, runtime limits,
16 rank hosts, kv backing ≤ 4 TiB), `qwen38_fp8_tp4_pp4_stage.json` (revision
`d2dc3565…`, pack `packs/qwen38_fp8_tp4_pp4_stage.spstage`, ctx 262144, tp_degree 4,
collectives 66620-66635, rails 10.10.200.x / 10.10.100.x), release template
`examples/release/qwen38_tp4_pp4_b1_template/` (`sparkpipe.json` role:
`bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index {rank}`,
env `LD_LIBRARY_PATH={install_root}/lib`; b1 = `max_active_sequences 1`).
Registry slot: `qwen38max`, status `placeholder`, ports 22480 / 66620-35 / 63700,
coexistence via `tools/fleet_swap.sh` (fleet-exclusive measurement windows).

**Do not boot this until four corrections land** (each is a small diff, none exists yet):
1. Runtime-limit/KV sizing: adapter computes `kv_block_count = resident_sequence_capacity ×
   ceil(max_sequence_positions/64)` (adapter `:1212-1213`) = 4096 × 4096 ≈ 16.7 M blocks →
   multi-TB `cudaMalloc` → init death. The module's physical-page clamp only fires when a
   KV store provider is active, and the adapter pins `KV_STORE=none`. Fix sizing (or clamp
   unconditionally) and set `resident_sequence_capacity`/`max_sequence_positions` sanely
   (b1 template intent).
2. Propagate the TP env (§1.2 item 3) and fix the world-rank/PP-stage transport mapping
   (§1.2 item 4).
3. Host names: stage json lists `sparkN-mgmt` peers + rail IPs; verify against
   `docs/HARDWARE_TOPOLOGY.md` route contract (inference never on mgmt) and the
   host-rdma transport's interface binding.
4. Package names: reconcile `libqwen38_tp4_pp4_serving_adapter.so` vs the build's
   `libqwen38_serving_adapter.so`, and regenerate `sparkpipe.json` sha256 placeholders.

Then per box: install release tree → `sparkpipe_model_residentd` ×16 → first-token smoke
via the batch client (`sparkpipe_model_batch --batch /tmp/q38.json`) on rank 0, one
sequence, ≤64 tokens, with the sibling residentds (DSV4 Flash, Qwen 27B) paused per the
coexistence rule.

### Module environment reference (direct module runs)

`SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION` (mandatory 1),
`_STAGE_PACK_PATH`, `_STAGE_COUNT` (1-32), `_STAGE_INDEX`, `_STAGE_FIRST_LAYER`,
`_STAGE_LAYER_COUNT`, `_STAGE_MAX_ACTIVE_SEQUENCES` (1-409 effective), `_STAGE_PIPELINE_SLOTS`
(1-4, only slot 0 used), `_STAGE_KV_BLOCKS`, `_STAGE_KV_STORE/_SERVICE/_SOCKET/_POOL_BYTES/_WORKERS`,
`_STAGE_MTP`, `_STAGE_GDN_SNAPSHOT_SLOTS`, `SPARK_QWEN38_STAGE_TP_*` (8 vars, §Tier 3),
`SPARK_QWEN38_STAGE_DEBUG_SKIP_GDN/_SKIP_MOE`. The serving adapter derives the slice/KV
set from its stage json; it does not derive the TP set (gap).

---

## 3. Risks + bandwidth projections vs SOTA

### 3.1 Risks (ranked)

- **R1 — Fleet residency is impossible at FP8 (blocks hardware day).** 2.31 TiB / 16
  ranks = **148 GB/rank vs ~107 GB usable** on a GB10 (`docs/QWEN38_MAX_PERF.md` §3) —
  and the bound is topology-independent: *any* 16-way equal split exceeds memory.
  Whole-stage packs (23 layers ≈ 575 GiB) cannot even run one stage on one box; only
  1–2-layer slices fit (the 24.9/49.8 GiB smokes). The plan's own quality-first MXFP4
  policy resolves this (routed experts ≈ 12.4 GiB/layer → ~76-80 GB/rank, leaving ~25 GB
  for KV + fp32 GDN state) **and** unlocks the sm121-native MXFP4 B1 kernels (the
  measured 5-10× B1 lever). The FP8-as-shipped decision (contract `precision_policy`)
  reversed that without a capacity note. Decision needed before any pack day: MXFP4
  requant vs expert paging (KV-tier machinery exists but is not a weight path) vs >16
  nodes.
- **R2 — TP path has never completed one collective** (Create CAPACITY_EXCEEDED), and
  the serving adapter cannot even enable it (no TP env propagation). The entire
  TP4xPP4/TP16 premise is unvalidated end-to-end.
- **R3 — Adapter stage mapping** treats world rank as pipeline stage (§1.2 item 4);
  hidden-transport flag validation would reject/require the wrong peers at TP4xPP4.
- **R4 — KV pool sizing explosion** on first serving boot (§2 Tier 4 item 1); the b1
  template's intent (1 sequence) is not what the deployment's `runtime_limits` express.
- **R5 — Prefill fails closed** in the module while the serving stack advertises it;
  long prompts are decode-stepped (~1.3 prompt-tok/s) until the prefill path lands.
- **R6 — No numerical gate vs HF**; screened-head exactness and router/shared-gate
  semantics rest on CPU references only. First real-token quality judgment is currently
  impossible — the plan's Torch harness must precede any fleet measurement.
- **R7 — Latency hygiene debts, measured**: per-frame stream sync, no graphs
  (~0.3 ms/layer GDN launch overhead), 100 µs-poll host wait per TP all-reduce,
  single slot (`max_inflight=1`), 409-row batch ceiling vs advertised 512.
- **R8 — Operational**: fleet-exclusive 16-host measurement windows (scheduler-owned);
  port-block collisions governed via `fleet_swap.sh` (dsv4-pro 20480/64620/61700, k3
  21480/61620/62700, qwen38max 22480/66620/63700 — registry consistent); doc drift and
  the qwen36-misnamed `qwen38_tp4_*.sh` scripts invite wrong-artifact deploys; revision
  provenance (`d2dc3565…` vs `207bd685…` vs local FP8 release) must be reconciled in the
  receipt chain before packing.

### 3.2 Bandwidth projections vs SOTA

Conventions and anchors (all in-repo): GB10 = 48 SMs, sm_121a, **273 GB/s** LPDDR5x,
128 GB (≈107 usable); sustained-planning convention **273 × 0.65 ≈ 177 GB/s**
(`docs/K3_PERF.md`), anchored by K3's measured 55.5 ms warm stage step (54.2 ms under
graph replay — memory-bound, not launch-bound). Per-token active bytes for this model:
BF16 spine ≈ 78 GB + routed experts 10/512 × 50.3 MB(FP8) × 92 layers ≈ 46 GB + shared
expert ≈ 9 GB + head ≈ 1 GB ≈ **134 GB read per token fleet-wide** (≈95 B active params
— consistent with the A95B nameplate).

Measured on one GB10 with real FP8 packs (`docs/QWEN38_MAX_PERF.md`):

| slice | B | ms/step | implied rate |
|---|---|---|---|
| 1 GDN layer (scalar MoE) | 1 | 8.45 | ~500 MB expert weights/layer/token; ≈6 tok/s full-model single-stream |
| 1 GDN layer (tile path) | 16 | 26.9 | ~804 GB/s "effective" expert stream (137 experts) |
| 1 GDN layer (tile path) | 32 | 39.9 | flat scaling |
| 1 GDN layer (tile path) | 256 | 261.6 | 233 GB/s effective — CTA-occupancy bound, not BW bound |

Projections (derived, cross-checked against the doc's own rows):

| Configuration | B=1 per-sequence | Aggregate ceiling | Basis |
|---|---|---|---|
| TP4xPP4 **replicated** (today's whole-stage packs — *undeployable*, 575 GiB/rank) | ~0.76-0.78 s/token | ~39 tok/s @B=256 (doc anchor) | each rank streams its whole 23-layer stage: 33.5 GB/token ÷ 177 GB/s ≈ 190 ms × 4 stages |
| TP4xPP4 **rank-local FP8** (packs missing; *also undeployable*, 148 GB/rank) | ~190 ms/token ≈ **5.3 tok/s** | ~15-21 tok/s (weight-stream floor; head screen +2 ms) | 134 GB ÷ 16 ranks = 8.4 GB/rank/token ÷ 177 GB/s ≈ 48 ms/hop × 4 hops; matches the doc's TP16 row (~55 ms/token ≈ 18 tok/s) since TP16 has the same per-rank bytes without PP serialization |
| **MXFP4 rank-local** (the capacity-compatible config) | ~165 ms/token roofline ≈ **6 tok/s**; **12-18 tok/s** with the native sm121 MXFP4 B1 kernels (plan/TP16 estimate) | ~25-30 tok/s | active bytes drop to ~111 GB/token (7 GB/rank); B1 scalar-FP8 penalty removed |
| Single node, batch serving, post-fixes (graphs + MoE M-loop) | — | ~100 tok/s @B=256 (doc §4A) | 27 ms/layer weight-stream floor |

vs SOTA references: in-repo, K3 (TP4xPP4, measured) = 18.0 tok/s B1 (roofline 20.6) and
DSV4 Flash TP4 B1 = 33.5-38.1 tok/s measured, ~40 tok/s plain-vLLM TP4 reference. Qwen
3.8 Max at FP8 TP4xPP4 projects ~3-4× slower per stream than K3 and ~7× slower than DSV4
Flash — the honest consequence of ~2× K3's per-rank active-byte footprint plus the
missing FP8 B1 kernel; it is not a bandwidth-utilization failure. Against the hardware:
the design targets ~65% of the 273 GB/s peak (the repo's SOTA-anchored convention, which
K3 demonstrably reaches); the measured expert stream already *exceeds* DRAM peak at
B=16 ("804 GB/s effective" — almost certainly L2/harness overlap, revalidate before
trusting), and the real B1/high-B limiters are kernel generation (no FP8 B1 path, scalar
wmma fallback), CTA occupancy at large B, and launch/sync overhead — exactly the levers
queued in `docs/QWEN38_MAX_PERF.md` §5. Fabric is a non-issue at decode: the T1-T4
schedule needs ≤30 KB/rank/layer (≤0.5% of weight traffic; ~0.15-0.25 ms/token/stage at
TP4), and the ~110 Gb/s direct-link cap only matters if the residual is ever wire-quantized
(correctly rejected for now).

**Bottom line:** the driver is a real, largely complete, strictly-validated decode stage
whose single-spark component evidence is good — but the plan's two load-bearing open
items (rank-local packs, TP collective) sit exactly on the critical path, the FP8-as-shipped
decision makes the registered 16-rank deployment physically unbootable (148 > 107 GB/rank),
and no numerical gate against HF exists. Recommended order for the next hardware round:
(1) re-run Tier 1-2 gates for receipts, (2) fix the collective Create blocker and pass
tp=2 on one box, (3) take the MXFP4 capacity decision and re-pin the contract, (4) land
the adapter TP-env + stage-mapping + KV-sizing fixes, (5) Torch parity gate, (6) then
spend a fleet window.
