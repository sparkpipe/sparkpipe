# Qwen 3.8 Max (Qwen3.8-2.4T-A95B) resident decode stage — hardware bring-up runbook

Scope: `modules/qwen38_resident_decode_stage/` + `source/spark_qwen38_serving_adapter.c`,
the TP4xPP4 deployment configs, and the pack pipeline. This driver has **never been
hardware-tested as a deployment**: everything below up to S2 has passed once on a single
spark; S3 and beyond have never run end-to-end. Companions:
[QWEN38_MAX_PLAN.md](QWEN38_MAX_PLAN.md) (plan of record),
[QWEN38_MAX_AUDIT.md](QWEN38_MAX_AUDIT.md) (fixed/unfixed footguns, measured numbers),
[QWEN38_MAX_TP16.md](QWEN38_MAX_TP16.md) (TP collective state),
[QWEN38_DFLASH2_RUNBOOK.md](QWEN38_DFLASH2_RUNBOOK.md) (ops patterns this file mirrors).
Naming rule (.agents/QUALITY_LAW.md): report as **Qwen 3.8 Max / Qwen3.8-2.4T-A95B**;
never mint new qwen36-branded names.

## 0. Status box (do not skip)

| Item | State |
|---|---|
| Module archive build | ✅ compiles clean (`make archive`) |
| Pack loader vs real FP8 pack | ✅ 1-layer GDN pack (layers 1–1, 24.9 GiB) |
| Two-step execute vs real FP8 pack | ✅ GDN pack, and GDN+attention pack (layers 2–3, 49.8 GiB); compute-sanitizer clean; both layer kinds, both MoE paths on real weights |
| TP device collective | ❌ never succeeded: `SparkTpDeviceCollectiveCreate` returns CAPACITY_EXCEEDED in the two-process smoke (spark4; see QWEN38_MAX_TP16.md suspects list) |
| Rank-local (TP4-sharded) packs | ❌ do not exist; the packer emits whole-PP-stage slices only |
| Serving-stack fleet run | ❌ never attempted |
| Torch/HF reference gate | ❌ not built (plan §Verification unchecked) — no numeric reference exists beyond cross-rank agreement |
| Contract qualification | `model_contracts/qwen38_authoritative.json`: `status: NOT_MEASURED`, `production_ready: false` |

**Gating decision (audit §4):** vendor-FP8 full residency is ~2.31 TiB / 16 ranks ≈
148 GB/rank arithmetic against ~107 GB usable per GB10 — it does not fit. Until the
capacity decision lands (MXFP4 requant ≈74 GB/rank, more ranks, or NVMe expert paging),
no 16-rank resident run can allocate its weights. S0–S4 below are runnable today;
S5–S7 require that decision or an MXFP4 pilot pack (a packer change; the pack format,
module loader and common sm121 expert kernels already accept MXFP4-E2M1).

## 1. Identity and artifacts

| What | Value |
|---|---|
| Model | `Qwen/Qwen3.8-2.4T-A95B`, served revision pin **d2dc35658bcf77e66643428cb52e774cc3b5bd29** (compile-time `QWEN38_MODEL_REVISION`) |
| Contract pin | sha256 of `model_contracts/qwen38_authoritative.json`, baked at compile time (`QWEN38_CONTRACT_SHA256`) and re-checked when the adapter loads the driver — regenerate weights ⇒ rebuild |
| Module identifier | `spark.qwen38.resident_decode_stage.fp8.h8192.l92.gdn69.e512k10.v1` |
| Node target | `cuda.sm121.qwen38.resident_decode_stage.fp8` |
| Driver program name | `resident_decode` (required verbatim by the adapter) |
| Serving adapter id | `spark.qwen38.serving-adapter.tp4-pp4.v1`; driver model id `qwen38.2.4t-a95b.resident-decode-stage-firmware` |
| Firmware description | `examples/model_descriptions/qwen38_resident_decode_stage_firmware.json` |
| Geometry constants | 92 layers = 69 GDN + 23 attention (period 4, phase 3), hidden 8192, vocab 248320, MoE 512 experts top-10 + 1 shared, attn 64Q/4KV d256 rope64, GDN 16QK/128V d128 conv4; MTP packed, NOT served (`MTP_LAYER_COUNT=0`) |
| Precision | routed experts vendor FP8_E4M3 + F32 block-128 scales (codec 4); non-expert/KV/head BF16; accumulators + GDN state FP32 |

Source map: module `source/spark_qwen38_resident_decode_stage_module.c` (+ `.cu`),
adapter `source/spark_qwen38_serving_adapter.c`, pack format
`source/spark_qwen38_stagepack_format.h`, ABI `include/sparkpipe/spark_qwen38_resident_decode_stage_firmware.h`,
packer `tools/qwen38_stagepack.py`, synthetic-pack tool `tools/qwen38_pack_synthesize.c`.

## 2. Fleet etiquette and ports (operational policy)

Qwen 3.8 Max TP4xPP4 occupies the full-16 fleet slot. Per audit §5 (which matches the
checked-in specs): **control TCP 22480**, **collectives 66620 + rank (66620–66635)**,
**transport control 63700**. ⚠ `docs/QWEN38_MAX_PLAN.md` line 110 still says "control
TCP 20480" — stale; 20480 belongs to dsv4-pro. Verify against the fleet registry before
claiming the window and fix the plan line.

- Coexistence: sibling residentds may stay up idle anytime; a **measured window is
  exclusive on its hosts**. This model's windows are fleet-exclusive (16 nodes): pause
  DSV4 Flash TP4 (spark4–7) and Qwen 27B PP16 (spark0–3) first, measure, restore. The
  scheduler owns the timeslice.
- One daemon per GPU; kill stale residentds before any start (pattern in
  QWEN38_DFLASH2_RUNBOOK.md §2).
- Checkpoint placement is already done: the FP8 release sits per node on internal NVMe
  at `/home/<user>/sparkdata/qwen38_2.4t_a95b/checkpoint` as PP-stage shard slices
  (54/53/52/54 shards), 224/224 files hash-verified.

## 3. Stage packs: pack command, sizes, gaps

Packer: `tools/qwen38_stagepack.py` (setup-time only, never serving path):

```bash
python3 tools/qwen38_stagepack.py \
  --checkpoint /home/spark9/extnvme/models/hf/Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --output  build/stagepacks/qwen38_pp0_l000-022.spstage \
  --first-layer 0 --layer-count 23 \
  --contract model_contracts/qwen38_authoritative.json \
  --receipt  build/stagepacks/qwen38_pp0_l000-022.spstage.receipt.json
# add --dry-run first; expect inventory/shapes verified against the pinned index
```

- PP4 slices over 92 layers: stages get **23/23/23/23** layers. Stage 0 additionally
  carries the embedding; the last stage carries final norm + LM head + 4 MTP globals +
  16 MTP layer tensors + a second embedding copy (`SparkQwen38StagePackExpectedTensorCount`
  computes the inventory — trust its count in the receipt).
- Measured pack sizes (single-layer real packs): **≈25 GiB per layer**, i.e. a whole
  23-layer stage slice ≈ 550–600 GiB on disk. Load is serial per-tensor cudaMalloc +
  staged H2D — startup-only, minutes, acceptable.
- **Gap (blocks S5+):** these are WHOLE-STAGE packs. A 16-rank run needs FOUR rank-local
  packs per stage (TP4 column/row-parallel sharding; the module dispatches the rank-local
  view). The packer's TP slicing is explicitly outstanding (plan §State, packer header).
- Known cosmetic bug: the receipt labels routed experts `mxfp4_e2m1` although the pack
  carries vendor FP8 — fixed at next packer touch; don't let it confuse a provenance audit.
- MXFP4 pilot option (capacity fix candidate): format codes and the synthesize tool
  already understand MXFP4-E2M1 (group 32 + E8M0 plane); producing real MXFP4 expert
  packs is packer work (`quantize_mxfp4_e2m1` exists; geometry table natural formats
  need the pilot row set). Track as its own change.

## 4. Build recipe (on a spark, sm_121a)

```bash
cd modules/qwen38_resident_decode_stage
make archive REPOSITORY_ROOT=/home/spark4/sparkpipe-qwen38     # → libqwen38_resident_decode_stage.a
# link the smoke harnesses (recipe per plan §State):
cc -o $REPOSITORY_ROOT/build/test_qwen38_pack_load $REPOSITORY_ROOT/tests/test_qwen38_pack_load.c \
   $REPOSITORY_ROOT/build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a \
   -lcudart -lcuda -lstdc++ -lpthread -lm
cc -o $REPOSITORY_ROOT/build/test_qwen38_execute $REPOSITORY_ROOT/tests/test_qwen38_execute.c \
   $REPOSITORY_ROOT/build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a \
   -lcudart -lcuda -lstdc++ -lpthread -lm
```

Makefile knobs (`STAGE_COUNT/STAGE_INDEX/STAGE_FIRST_LAYER/STAGE_LAYER_COUNT/
MAX_ACTIVE_SEQUENCES/PIPELINE_SLOT_COUNT/KV_BLOCK_COUNT/ALLOW_UNQUALIFIED_EXECUTION/`)
parameterize the GPU-validator recipe identity, not the binary's runtime behavior — the
binary is configured by environment at Initialize (§6). The two compile-time pins that DO
ride in the binary: revision + contract sha256.

## 5. Deployment configuration

### 5.1 Per-stage adapter config — `config/qwen38_fp8_tp4_pp4_stage.json`
(schema_version MUST be exactly 3; member set is validated EXACT — extra/missing keys are schema errors)

| Member | Bring-up value | Notes |
|---|---|---|
| `schema_version` | `3` | hard-coded `SPARK_QWEN38_SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION` |
| `model_revision` | `"d2dc35658bcf77e66643428cb52e774cc3b5bd29"` | string-compared against the compiled-in pin |
| `stage_pack_path` | `"packs/qwen38_fp8_tp4_pp4_stage.spstage"` | resolved against the node's `runtime_root` |
| `max_sequence_positions` | ≤ `262144`; **start at 8192** | cap is the model native; also sizes the KV pool (below) |
| `tp_degree` | `4` | must divide STAGE_COUNT 16; drives PP geometry {23,23,23,23} and `pp_stage = world_rank/4` |

Reference: `examples/deployments/qwen38_fp8_tp4_pp4_stage.json` (also pins tp_collective:
backend `hidden_transport`, `lib/hidden_transport.so`, identifier `134217728`, listen
66620, timeouts 120 s, peer hosts \`{host}-mgmt\`, dual-rail 10.10.200.x/10.10.100.x,
algorithms recursive_doubling / direct_all_to_all (≤80 KiB) / counter_rotating_split_ring
(≥640 KiB), `step_rail_indices [0,1,1]` — inherited from dsv4, re-derive when the
collective schedule is real).

### 5.2 Fleet deployment spec — `examples/deployments/qwen38_fp8_tp4_pp4_host_rdma.spec.json`
16 rank hosts `spark0..sparkf`, stage_indices 0–15, transport
`libhidden_transport_spark_host_rdma_verbs.so` mode host-rdma, control_port_base 22480,
node_target as above, adapter config template per node.

⚠ Its `runtime_limits` are aspirational and will FAIL against today's module — derive a
bring-up variant instead:

| Spec field | Checked-in | Bring-up value | Why |
|---|---|---|---|
| `max_inflight_submissions` | 4 | 1 | module executes every frame on ONE slot with a stream sync; adapter advertises 1 |
| `max_active_sequences` | 1024 | 1 (b1) | module **refuses > 409** loudly (`config_batch_too_wide`, MoE tile guard); b1024 dataset naming cannot start yet |
| `max_input_rows` | 1024 | match active | same family of caps |
| `resident_sequence_capacity` | 4096 | 1 | KV pool = capacity × ceil(positions/64) blocks — 4096 × 4096 blocks at 262144 ctx cannot allocate |
| `kv_logical_page_capacity` / `kv_physical_page_capacity` | 1048576 / 16384 | ≥ resident capacity / ≥ active sequences | zero or undersized fails deployment validation (dflash2 §2 lesson) |
| `kv_backing_dataset` + 4 TiB cap | present | drop or keep provider `none` | stage KV tier is compiled but unused in decode (KV_PROVIDER none) |

### 5.3 Release layout — `examples/release/qwen38_tp4_pp4_b1_template/`
The b1 template (one active sequence) is the correct FIRST deployment shape: install root
`/home/{host}/sparkdata/qwen38.fp8.tp4_pp4.b1`, roles launch
`bin/sparkpipe_model_residentd --deployment …/config/model_resident.json --rank-index {rank}`
with `LD_LIBRARY_PATH={install_root}/lib`. Fill every `sha256` placeholder before
publishing; `restart_on_change` on configs, `resident_reload_boundary` on binaries +
pack. Deploy ALL artifacts together (stale-mix = subtle numerics bugs).

## 6. Environment variables (complete)

The adapter setenvs group A itself before driver create (one resident process hosts one
stage, process-wide env is the intended channel). Group B is read by the module via
getenv and **nobody in-tree sets it** (§8, risk R3). Group C is optional/debug.

| Var | Example (rank 5) | Who sets |
|---|---|---|
| `SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION` | `1` | adapter always sets 1; standalone runs MUST set it or Initialize returns MODULE_NOT_VALIDATED |
| `SPARK_QWEN38_STAGE_PACK_PATH` | absolute pack path | adapter (from stage json) |
| `SPARK_QWEN38_STAGE_COUNT` | `16` | adapter (constant) |
| `SPARK_QWEN38_STAGE_INDEX` | `5` (= world rank) | adapter |
| `SPARK_QWEN38_STAGE_FIRST_LAYER` / `_LAYER_COUNT` | `23` / `23` | adapter derives from tp_degree; ranks 0-3→0-22, 4-7→23-45, 8-11→46-68, 12-15→69-91 |
| `SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES` | `1` | adapter (from runtime_limits; ≤409 enforced) |
| `SPARK_QWEN38_STAGE_PIPELINE_SLOTS` | `1` | adapter (= max_inflight_submissions) |
| `SPARK_QWEN38_STAGE_KV_BLOCKS` | small | adapter (= resident_capacity × ceil(max_positions/64)) |
| `SPARK_QWEN38_STAGE_KV_STORE/_SERVICE/_SOCKET/_POOL_BYTES/_WORKERS` | none/none/none/0/0 | adapter (KV tier off) |
| `SPARK_QWEN38_STAGE_MTP` / `_GDN_SNAPSHOT_SLOTS` | `0` / `0` | adapter (MTP packed, not served) |
| `SPARK_QWEN38_STAGE_TP_DEGREE` | `4` | **launcher/export — unset ⇒ module silently runs tp=1 replicated** |
| `SPARK_QWEN38_STAGE_TP_RANK` | `1` | launcher (= world_rank % 4) |
| `SPARK_QWEN38_STAGE_TP_BACKEND_PATH` | `lib/libhidden_transport_spark_host_rdma_verbs.so` | launcher (required when degree>1, else INVALID_ARGUMENT) |
| `SPARK_QWEN38_STAGE_TP_IDENTIFIER` | `134217728` | launcher (must match across the TP group) |
| `SPARK_QWEN38_STAGE_TP_PORT_BASE` | `66620` | launcher |
| `SPARK_QWEN38_STAGE_TP_HOSTS` | `10.10.200.0,10.10.200.1,…` comma list, one host per rank in rank order | launcher (rail IPs) |
| `SPARK_QWEN38_STAGE_TP_LOCAL_HOST` | `10.10.200.1` | launcher |
| `SPARK_QWEN38_STAGE_TP_TIMEOUT_MS` | `120000` | launcher (optional, default 120000) |
| `SPARK_QWEN38_STAGE_DEBUG_SKIP_GDN` / `_SKIP_MOE` | set = skip | debug isolation only; never in a measurement |
| `SPARK_QWEN38_SERVING_PREFIX_CACHE` | `0` A/B toggle | adapter-side; the off switch IS the byte-identity gate |

## 7. Smoke-test ladder (each step gates the next)

**S0 — host-side sanity (any machine):** `make archive` clean; contract sha unchanged.

**S1 — pack integrity (spark, no GPU compute):** pack a small interior slice
(`--first-layer 1 --layer-count 1`), run `test_qwen38_pack_load PACK`.
Pass: `initialize status=0`, `destroy ok`. Failures name the offending header field
(`pack_geometry_mismatch`) — never edit a pack by hand; repack.

**S2 — two-step execute (the existing green baseline, re-run after ANY change):**
`test_qwen38_execute PACK` on (a) the 1-layer GDN pack (defaults), (b) the 2-layer
GDN+attention pack (`TEST_QWEN38_FIRST_LAYER=2 TEST_QWEN38_LAYER_COUNT=2`). Pass:
`execute[0] status=0`, `execute[1] status=0` (second step exercises carried conv tail +
FP32 GDN recurrence). Then the two slice-edge cases never hardware-tested yet:
embedding stage (`TEST_QWEN38_STAGE_COUNT=16 TEST_QWEN38_STAGE_INDEX=0
TEST_QWEN38_FIRST_LAYER=0 TEST_QWEN38_LAYER_COUNT=1`) and head stage
(`…INDEX=15 …FIRST_LAYER=90 LAYER_COUNT=2` — runs the 4-bit head-shadow screened argmax;
compare its token vs plain argmax once before trusting it). Re-run compute-sanitizer on
any kernel change. Optional isolation: DEBUG_SKIP_MOE / DEBUG_SKIP_GDN bisects layer kinds.

**S3 — TP collective two-process smoke (KNOWN BLOCKED):** two processes on one spark,
degree 2, group-B env above, rail IP 10.10.200.x, ports 66620/66621. Current failure:
`SparkTpDeviceCollectiveCreate → CAPACITY_EXCEEDED` after ApplyTopology/CreditBinding/
ProbeMemoryMode all pass — see QWEN38_MAX_TP16.md for the three suspects (backend .so
sizing, mapped-host credit-plane math, missing adapter-side topology fields vs dsv4's
working pattern). Do not schedule a fleet window until this passes.

**S4 — four-rank fanout on one PP stage (first true TP4xPP4 shape, needs rank-local
packs):** ranks 0–3, layers 0–22, expert-sharded MoE + residual delta all-reduce.
Pass per rank log: `tp_collective_open degree=4 rank=N port_base=66620` AND
`initialize ok slice=0+23 … owns_embedding=1 owns_head=0`; then identical emitted
tokens across the four ranks for the same submission (tools/qwen38_tp4_e2e.c pattern).
Any rank WITHOUT the tp_collective_open line is running tp=1 — stop, fix env.

**S5 — 16-rank residentd up, idle:** deploy the b1 layout to all 16 hosts; expect
`model_residentd ready rank=N …` everywhere within the load window (~minutes: serial
H2D of the big packs). Watch for `phase=adapter_initialize capacity_exceeded`
(stale daemon OR the §0 capacity wall). Idle coexistence is allowed by policy.

**S6 — gated window, first tokens:** scheduler-owned exclusive slice; pause siblings;
drive `bin/sparkpipe_model_batch --batch <file>` with ONE request, tiny context.
Bring-up acceptance (not production): submit_result 0 on all ranks, token events stream,
cross-rank agreement, stable second run on the same daemon. There is NO torch reference
yet — record hashes, don't claim correctness.

**S7 — measurement:** only after S6; projections live OUTSIDE PERFORMANCE_STATUS.md
until then. Audit-extrapolated expectation to beat (NOT measured): ≥160 ms/token/stage
MoE floor → ~6 tok/s single-stream ceiling on today's scalar path.

## 8. Risks and known blockers (ranked)

- **R1 — Capacity wall (gates S5+):** ~148 GB/rank vendor-FP8 residency > ~107 GB usable.
  Options: MXFP4 requant (≈74 GB/rank, unlocks sm121 tensor-core expert kernels — double
  win, quality-policy tradeoff), more ranks TP4xPP6/32 (≈99/74 GB — marginal/fits,
  topology change), NVMe expert paging (design absent; data placement already done).
  Decide BEFORE burning a fleet window.
- **R2 — Rank-local packs don't exist:** packer emits whole-stage slices; the TP-rank
  frame mapping inside the adapter is flagged outstanding (adapter header comment). S4+
  blocked on both halves landing together.
- **R3 — Silent tp=1 fallback:** the module takes its TP geometry ONLY from the group-B
  env vars; the adapter neither sets nor checks them. With env unset, a tp_degree=4
  config deploys a replicated tp=1 module (4× expert memory ⇒ likely capacity_exceeded,
  or duplicated work that quietly "works"). Bring-up mitigation: export group-B in the
  launcher AND grep every rank's log for `tp_collective_open degree=4`; longer term,
  fail closed when the stage json says 4 but the env is absent.
- **R4 — TP collective CAPACITY_EXCEEDED:** unresolved backend contract detail (S3).
- **R5 — Checked-in runtime_limits exceed module caps:** the b1024 spec (1024 active, 4
  inflight, 4096 resident) cannot initialize against the ≤409/single-slot module; use
  b1-shaped limits for bring-up.
- **R6 — Perf floor:** single-slot sync-per-frame execution, synchronous post_receive/send
  PP handoffs, scalar FP8 MoE (common kernels have tensor-core grouped paths for MXFP4
  only). Acceptable for bring-up; not serving-viable without R1 + kernel work.
- **R7 — Lane lifecycle absent:** Execute assumes cold identity lanes; Admit/Snapshot
  return UNSUPPORTED. Multi-sequence behavior after eviction/residency events is
  undefined — keep B=1 through bring-up.
- **R8 — Port-registry drift:** plan line 110 says control 20480 while spec/audit say
  22480. Re-verify against the fleet registry before the window; one wrong port collides
  with dsv4-pro. Fix the plan line in the same change.
- **R9 — Smaller tripwires:** `step_rail_indices [0,1,1]` inherited from dsv4,
  re-derive once the collective schedule is real; packer receipt mislabels routed experts
  mxfp4_e2m1 (cosmetic); head-shadow screened argmax parity unverified (S2 head-slice
  covers it); MTP tensors packed but never executed (expected, disclosed).

## 9. Evidence discipline

Measured results go to PERFORMANCE_STATUS.md, separate from projections; receipts
(`*.spstage.receipt.json`) travel with every pack; record window times, paused
siblings, and the exact artifact sha256 set alongside any token-stream hash. If a step
here contradicts the code, the code wins — update this file in the same change.