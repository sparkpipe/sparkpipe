# GLM5.2 driver diamond-pass audit + bandwidth projections

Scope: `modules/glm52_resident_decode_stage/` (+ `modules/glm52_dspark_draft_backend/`),
on the `unified` tree (= PR #673 bring-up base; #668 B8 and #669 B16 batch classes
merged). All file:line references are to this working tree. Code patches were
optional this round; none are attached — findings below are ordered so the next
round can lift them directly.

Driver size under audit:

| File | Lines |
| --- | ---: |
| `source/spark_glm52_resident_decode_stage_module.c` | 1,930 |
| `source/spark_glm52_serving_adapter.c` | 1,011 |
| `source/spark_glm52_resident_decode_stage_cuda.cu` | 520 |
| `source/cuda/layer.cuh` | 1,140 |
| `source/cuda/unity.cu` | 293 |
| `source/spark_glm52_stagepack_format.h` + headers | 713 |
| `glm52_dspark_draft_backend` (cu + validator + policy + header) | 2,979 |

---

## 1. DRY / structure findings

The quality law applied: Solutions/(production-codesize×2), simplest code, no
compat shims, no paranoid checks.

### 1.1 What collapses

1. **Four copies of the stagepack format/validator family.** `dsv4` (464 lines),
   `qwen36` (550), `qwen38` (432) and `glm52` (282) each carry a private
   `*_stagepack_format.h`, and each `*_module.c` re-implements the same
   header/entry/range/inventory validation chain (~250 lines in glm52 alone:
   `SparkGlm52PackValidateHeader/EntryGeometry/Ranges/ValidateInventory`,
   module.c 227–478). The only genuinely model-specific part is
   `SparkGlm52StagePackExpectedShape`'s shape table plus the row/column
   sharding policy — everything else (magic/version/alignment/offset/overlap/
   duplicate-bitmask logic) is parameterizable. One shared reader driven by a
   per-model geometry table removes roughly 1,200–1,500 lines across drivers
   with zero capability loss.
2. **Five hand-rolled serving-adapter skeletons** (glm52 1,011 / qwen36 2,116 /
   qwen38 1,248 / dsv4 1,515 / k3 502 lines): JSON member loading, the
   `tp_collective` loader family (~100–135 lines per model; glm52 adapter
   308–442 vs dsv4 508+ with per-model drift already visible), the pending-slot
   table, driver-completion translation, submission validation, frame build.
   k3's factored layout (`pack_load` / `bind` / `runner` over a 61-line module)
   shows the endgame; the same split over a common serving core cuts another
   ~2–3k lines fleet-wide.
3. **Round-major wave validation exists twice inside one driver.**
   `SparkGlm52ValidateRoundMajor` + `SparkGlm52RoundMajorWaveRows`
   (module.c 879–940) and `SparkGlm52ServingValidateRowOrder` (adapter
   499–530) are the same algorithm over resident-slots vs lane indices,
   including the identical wave/replay double loop. Collapse into one helper
   in `runtime/stage_module_common.c` taking an index accessor.
4. **Draft backend reimplements geometry-generic kernels that already exist in
   `inference/kernels/`.** Its own RMSNorm (`RmsNormRowsKernel`, .cu 323–369),
   SwiGLU (389–406), elementwise add (371–387), RoPE+head-norm (490+),
   argmax/confidence (730–856) duplicate `norm.cuh`, `topk.cuh`, `head.cuh`,
   RoPE kernels used by the resident stage. Worse, it introduces the tree's
   **only cuBLAS dependency** (`cublasGemmEx`, .cu 1341) — a second GEMM stack
   beside `LmGemmLaunch`. Keep the drafter's block/MTP orchestration; call the
   shared kernels; if cuBLAS genuinely wins on the dense draft shapes, that
   claim belongs as one sentence in the file that needs it (per the
   `config.h` doctrine), not as an unmeasured second stack.
5. **Two weight-loading doctrines coexist.** Resident stage: immutable
   content-addressed stagepack, compile-time-pinned contract SHA256. Draft
   backend: raw `manifest_path/config_path/safetensors_path` at runtime
   (header 39–41). When trained draft weights exist, the drafter must consume
   a packed artifact through the same reader — not a parallel safetensors path.
6. **Credit where due (landed by #673):** the dispatch-policy neutral core
   extraction (d6f8967), the admission/policy ladder (4cfa095), and the JIT-KV
   common seam (2a128d0) are real structural wins; glm52's module now consumes
   `spark_speculation_policy.c` and `stage_module_common` instead of private
   copies. Items 1–3 are the remaining big rocks.

Net under the law: items 1–3+5 remove an estimated 3–4k lines of production
C/CUDA while preserving every capability; item 4 removes a GEMM stack.

### 1.2 What is dead

1. **`SparkGlm52LaunchCudaWave` + `SparkGlm52RunLayers`** (cuda.cu 398–412,
   495–504; declared internal.h:132): zero callers anywhere. Production drives
   the chunk chain (`WaveBegin`/`LayerAttention`/`LayerMlp`/`WaveHead`) for TP
   reduce boundaries. Delete.
2. **unity.cu extern shims with no behavioral caller.** `Glm52HeadRestricted`
   (194–216) and `Glm52LayerAttentionBf16Graphed` (218–293 — a graph-replay
   wrapper although "no graph replay exists in the rewritten modules yet" per
   batch_tuning.h's own comment) have no references at all.
   `Glm52GemmBf16`, `Glm52GemmExpertWeightBf16Activation`,
   `Glm52Layer{Attention,DenseMlp,Moe}Bf16*` are referenced only by
   `tests/test_glm52_unity_precision_contract.py` /
   `test_glm52_quantized_cuda_contract.py`, which grep source *text*. They pin
   naming, not behavior; either give them a real caller or fold the precision
   contract into a test that compiles something.
3. **`glm52_dspark_draft_backend` is production-dead today.** Its archive has
   no consumer outside its own epoch-3 validator; the hidden-tap plan's only
   consumer is `tests/test_glm52_dspark.c`; and the base checkpoint ships no
   draft weights (PERFORMANCE_STATUS 2026-08-16 note). Legitimate bring-up
   scaffolding — but it must not be counted as serving capability, and it
   should not drag cuBLAS into production link units until it wires in.
4. **Stale README gates / missing GPU validator (most serious finding).**
   `modules/glm52_resident_decode_stage/README.md` documents eight
   `package_layer0_*` correctness gates; `validation/` was deleted
   (b45d714 "Delete the validation copy and dspark"), the targets do not exist
   in `resident_decode_stage_rules.mk`, and `GPU_VALIDATOR :=
   validation/validate_glm52_resident_decode_stage_cuda.sh` points at nothing.
   `require_gpu_validator` therefore fails → `validate`, `publish`, and
   `publish_variants` are all blocked, i.e. **the driver currently ships with
   no runnable in-tree GPU numerical gate** (host tests are string checks).
   Restoring a minimal retained-receipt validator is prerequisite #1 for
   everything else in this report, including PP7.
5. **Compat shims violating "no compat shims":** the lenient KV backing
   fallback `/tmp/sparkpipe_glm52_kv_%s` ("serving configs that predate the
   backing fields", module.c 182–184, 737–747) and the
   `tp_collective_identifier==0` degraded single-rank mode (module.c 193–197,
   adapter 361–363). Regenerate the configs; delete both.
6. **Paranoid checks to drop:** the lane-loop clamp in
   `SparkGlm52PrepareAsyncCompletion` whose own comment concedes upstream code
   makes it unreachable (module.c 1576–1588); the duplicated
   `context <= GLM52_DSA_SELECTED` guard in `Glm52LayerIndexer` (layer.cuh 243
   vs 359); `(void)boundary_bytes` dead computation in
   `SparkGlm52ServingValidateBoundaries` (adapter 795–808); the no-op
   `SparkGlm52ServingProgress`. Also fix the deployment pack filename typo
   `.glms52sp` in `tools/glm52_gen_deployment.py` while touching it.

---

## 2. Bandwidth-based prefill + output projections on GB10

### 2.1 Geometry and method

From `model-families/glm52/include/sparkpipe/spark_glm52_model.h`: 78 layers
(3 dense + 75 routed), hidden 6144, MLA with 64 heads over a 512-element
latent + 64 rope elements (**KV slot = 1152 B** in BF16 — the number that
makes this model tractable here, per `config.h` 34–41), top-8 of 256 experts
at intermediate 2048, dense intermediate 12288, vocab 154880, DSA keeps 2048
of context, indexer 32×128 shared per 4 layers (21 full-indexer layers).
Derived totals: **~743 B parameters, ~38.7 B active per token**
(≈516 M active params/layer × 75 + head).

Bytes are computed bottom-up from the exact pack shapes
(`spark_glm52_stagepack_format.h`) and codec math
(`spark_weight_codec.h`: fp8 e4m3 = 1 B payload + f32 scale per 128-column
block), with the declared TP sharding policy (rows-sharded: embedding,
lm_head, q_b, dense/expert/shared gate-up; cols-sharded: o_proj, down
projections; replicated: norms, q_a, kv_a, kv_b, indexer, router).
All byte figures below are decimal (10^9) to match the 273 GB/s anchor.

**Measured GB10 bandwidth anchor:** 273 GB/s LPDDR5x (PERFORMANCE_STATUS
2026-08-16 GLM5.2 section; ARCHITECTURE.md corroborates the fabric-side
numbers). All floors below are `bytes ÷ 273 GB/s`.

### 2.2 Decode arithmetic (per rank, TP8 fanout, expert_fp8 build)

| Component | Bytes/token/rank |
| --- | ---: |
| BF16 spine, 78 layers (projections, kv_b, norms, indexer on 21 layers) | 8.94 GB |
| lm_head local-vocab scan per token (19,360 rows × 6144 × 2 B; embedding contributes only a ~12 KB row gather) | 0.25 GB |
| Routed experts: 8 selected × 75 layers × 4.64 MiB/expert shard (W1 3,244,032 + W2 1,622,016 B incl scales) | 2.92 GB |
| **Total ideal reads** | **12.1 GB** |

Floor: 12.1 GB ÷ 273 GB/s = **44.3 ms/token → ~23 tok/s absolute ceiling**
with zero collective, launch, and non-weight cost. Measured B1 =
6.91 tok/s (144.7 ms) ⇒ **31 % of roofline**.

Codec×sharding cross-check of the expert stream alone (75 layers × 8
experts): fp8 sharded 2.92 GB · mxfp4 sharded 1.50 GB · fp8 unsharded
23.36 GB · mxfp4 unsharded 12.03 GB. The ledger's "~15.7 GB FP8 experts"
claim matches none exactly — it implies ~5.4× read amplification over the
sharded-fp8 ideal.
Decisive zero-cost check: read `reserved0/reserved1` (tp_degree/tp_rank) from
a deployed pack header (`SparkGlm52StagePackHeaderTpDegree`). If packs are
tp8-sharded, expert tile scheduling over-fetches ~5–6× and that — not
collectives — is the largest single lever; if they are tp1, every rank stores
and streams the full expert library and TP is fake-parallel on experts.

**Collective arithmetic (the measured first-order term).** The TP chain emits
exactly **158 reduces/token** (1 embedding + 2×78 layer chunks + 1 head max —
matches the ledger). Recursive doubling at TP8 = 3 exchange steps of
S = rows×12,288 B: at B1 that is 12.3 KB ≈ 3 µs of wire time at the ~100 Gb/s
practical link. The claimed median 912 µs/reduce is thus >95 % software
latency — and 158 × 912 µs = 144 ms ≈ the entire measured interval (the
ledger's own "~31 ms collectives" split is inconsistent with its median;
both readings agree the reduce path dominates). Getting each collective to
~40 µs turns 144 ms into ~7 ms and moves B1 into the 17–20 tok/s band even
before any bandwidth work. Note the ledger's B1 bandwidth decomposition
(~80 ms floor) used the unreconciled 21.7 GB figure; the sharded-ideal floor
is 44 ms, which makes the collective/launch overhead share even larger.

### 2.3 Output projections (decode aggregate)

Calibration uses the measured waves: B8 = 8/43.46 = 184 ms vs 111 ms weight
floor (60 % of ceiling); B16 = 212 ms vs 173 ms (82 %). The naive
distinct-expert model (E[distinct] = 256·(1−∏(248−i)/(256−i))) already
undershoots B16, meaning route reuse/L2 catches part of the expert stream —
so treat ≥B32 rows as planning bands, not predictions.

| Batch (rows/wave) | Weight floor ms/wave | Ceiling tok/s | Measured tok/s | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| B1 (single stream) | 44.3 | 23 | 6.91 | 31 % |
| B8 | 111 | 72 | 43.46 (#668) | 60 % |
| B16 | 173 | 92 | 75.55 (#669) | 82 % |
| B32 | ~260 | ~123 | — | projected 70–85 % |
| B128 | ~375 | ~340 | — | collective-bound band |
| B256 | ~386 | ~660 hard | — | unreachable: collectives scale |

Why the ladder flattens: at B≥128 all 256 experts sweep every step
(~102 GB/rank), while recursive-doubling payload grows linearly with rows
(B256: S=3.1 MB → ~750 µs wire × 158 reduces ≈ 118 ms/step) — collectives
become co-dominant with weights around B64–B128. Realistic plateau for this
8-Spark fanout: **~150–250 tok/s aggregate**, against the current 75.55.

Single-stream levers, ranked by arithmetic payoff:
1. Collective control-plane: 144 ms → <10 ms (158 × ~40–60 µs).
2. Expert-stream amplification, if the 15.7 GB reconciliation confirms it:
   up to −13 GB/token.
3. CUDA graphs: none are captured today (batch_tuning.h states graph replay
   does not exist yet); the ledger attributes ~25 ms/token to launch overhead.
4. Head: `config.h` promises a 256-token restricted head (1.6 MB) but the
   driver runs `Glm52HeadFullVocab` — a ~0.25 GB local-vocab scan every token.
   Free 2 % when a grammar/tool schema exists; also the natural place to port
   DSV4's screen-and-rescore if unrestricted sampling needs speed.
5. Replication waste (fleet view): ranks jointly read ~98.8 GB/token of
   spine+experts, but only 60.6 GB is unique — **~38 GB/token is replicated
   re-reads** of q_a/kv_b/indexer/router (per rank: 9.4 GB spine = 4.0 GB
   sharded-unique + 5.4 GB replicated). Fleet roofline without replication:
   60.6 GB ÷ (8×273 GB/s) = 27.7 ms → **36 tok/s single-stream** vs 23 with
   today's placement. This is the quantitative case for revisiting pure TP8
   fanout (see §3).
6. Long-context term (not yet binding at the 6-token benchmark): main
   attention is capped by DSA at 2048 slots × 1152 B × 78 = 184 MB/token/rank,
   but the index scan reads context×256 B on each of 21 full-indexer layers —
   0.54 GB/token at ctx=100k, 5.4 GB at ctx=1M, where it becomes dominant.

### 2.4 Prefill projection

Per dispatch (≤256 tokens, `SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH`),
token-expert assignments (256×8 = 2048) hit essentially all 256 experts, so a
dispatch sweeps the whole library: ~103 GB/rank → **377 ms floor**.
Compute: 256 tok × ~77 GFLOP = 19.9 TFLOP ≈ 160 ms at an assumed ~125 TFLOPS
dense BF16 (public GB10 spec recalled; web search unavailable this session —
flagged assumption) → prefill stays weight-bound on this box:

- Prefill ceiling ≈ **600–680 tok/s** per fleet at 256-token dispatches.
- TTFT, 128-token prompt: **~0.35–0.4 s** (single dispatch + connect).
- KV writes are noise (128 tok × 1152 B × 78 layers ≈ 11.5 MB).

### 2.5 SOTA reference statement

Primary reference: the **absolute roofline on the same silicon** — unique
active bytes/token ÷ 273 GB/s (vendor- and stack-independent; this is the
bound any implementation faces on one GB10). Secondary in-fabric anchor: the
retained plain-vLLM TP4 B1 measurement (37.79 tok/s, PERFORMANCE_STATUS
2026-08-14) — a different, smaller-active model, cited for stack-overhead
comparison only. Public prosumer anchors could not be verified this session
(web search unavailable): from memory, llama.cpp DeepSeek-R1-671B Q4 on a
M3 Ultra (819 GB/s) runs ≈18–20 tok/s single-stream, i.e. ~45–55 % of its own
roofline. Against that normalization, GLM5.2-on-GB10 sits at **31 % (B1) →
82 % (B16)** of roofline: the batch path is already in SOTA-normalized
territory; the single-stream path carries a 2–3× software overhead, which is
exactly where the levers above point. Per ARCHITECTURE.md, the DGX-Station
"50 % of B300" gate still requires matched-workload live comparison — no
analytical ratio closes it.

---

## 3. PP7 deployment plan

Motivation (arithmetic, not aesthetics): today's TP8 fanout stores ~107 GB/rank
(97 GB expert shards + 9.4 GB spine) of the 128 GB box, leaving ~21 GB for
KV/workspaces — and re-reads ~38 GB/token of replicated tensors (§2.3 item 5).
A pipeline split drops per-rank weight residency ~6× (≈15 GB for an 11-routed-
layer stage) and frees >90 GB/rank for KV, which is what makes the 1M-context
target (`SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS = 1,048,576`; pool
4,194,304 tokens) physically fittable: 4.19 M tokens × 1152 B × 11 local
layers ≈ 53 GB/rank.

### 3.1 Pack split

78 layers over 7 stages cannot be uniform; take **[12, 11, 11, 11, 11, 11, 11]**
(stage 0 = layers 0–11, absorbing the three dense layers):

- Byte balance is fine without cleverness: stage 0 ≈ 96 GB fleet-wide
  (3 dense + 9 routed), stages 1–6 ≈ 115 GB each (11 routed) — max/min 1.19,
  so stage 0 is never the bottleneck.
- The five DSpark tap capture indices {7, 22, 38, 54, 69} land in five distinct
  stages (0, 1, 3, 4, 6) under this split — no stage hosts two taps, and the
  drafter feed never crosses two hops.
- What already supports it: the pack header carries
  `stage_count/stage_index/first_layer_index/layer_count`; inventory masks are
  computed per requested layer range (`ExpectedGlobalMask/ExpectedLayerMask`);
  `owns_embedding`/`owns_final_head` logic exists; KV arena sizing keys off the
  local `layer_count`.
- Missing tooling: `tools/glm52_resident_stagepack.py` has `--tp-degree/
  --tp-rank` but no `--stage-index/--stage-count`; PP7 needs 7 stages × 8 ranks
  = 56 content-addressed packs plus contract/manifest identities for each, and
  `FirstLayer()` must become a per-stage table (uniform `stage_index × 78`
  math, firmware.h 115–118, breaks on uneven splits).

### 3.2 Per-stage configs

- Firmware constants: `SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT` is
  hard-wired to 1 and `ModuleConfigure` rejects anything else (module.c 174);
  re-widen to 7 with the layer table. `LAYERS_PER_STAGE=78` likewise.
- Serving adapter: today it is single-stage PARALLEL_FANOUT —
  `SPARK_GLM52_SERVING_STAGE_COUNT=8`, `stage_layer_counts={78×8}`, and it pins
  `node_context.stage_index = 0` (adapter 27–34, 761–764). Needs a stage-grouped
  topology: 7 groups × 8 TP ranks, per-group tp_rank mapping, and per-group
  collective identifiers/port plans (the current `control_port_base =
  peer_ports[0]`, ports `19480+rank` scheme collides across groups).
- Runtime limits per stage: `PIPELINE_SLOT_COUNT ?= 2` (max 4) cannot fill a
  7-deep pipeline; steady-state bubble fraction is (S−1)/(m+S−1), so raise the
  slot ceiling to ≥14 (2 waves/stage) and let admission keep ≥7 quanta in
  flight.
- Snapshot/aggregation: per-stage runtime snapshots must compose into one
  serving view (slot/lane counters already exist per stage).

### 3.3 Transport

- Boundary ABI survives from the PP13 era and is already validated per frame:
  HIDDEN_INPUT/OUTPUT (2×6144 BF16 = 24,576 B/row) and DSA sideband
  (2048×4 B = 8,192 B/row), carried when the source stage index is odd
  (firmware.h 120–133). Nothing needs to change in the frame contract.
- What does not exist: anything that **moves** those buffers between processes.
  `ring/transport/` provides the collective legs (hidden_transport, rdma,
  memlink) but no point-to-point boundary leg; and the adapter actively
  rejects multi-stage submissions today — `SparkGlm52ServingValidateBoundaries`
  requires hidden_input/output and sidebands to be zero (adapter 795–808).
  Needed: a p2p send/recv leg over the existing direct/switch rails with
  credit-based flow control sized by pipeline slot, plus recv-side wave
  staging in the slot workspaces.
- Boundary cost is negligible at serving batches: 32,768 B/row/hop → 262 KB
  at B8 ≈ 21 µs/hop at ~100 Gb/s useful; 6 hops ≈ 126 µs against a 184 ms
  wave (<0.1 %). At B1024 it is ~16 ms/wave (~9 %) — worth chunking, not
  blocking.
- If the drafter is fed through the pipeline later, define a second sideband
  kind for hidden taps: 5 layers × 12,288 B = 61,440 B/row (today's sideband
  carries only DSA positions).

### 3.4 What is missing (checklist, in order)

1. **Restore the GPU validator** (§1.2 item 4). No stage-split work should
   start before a numerical gate exists — a pipeline multiplies the places
   silent corruption can hide.
2. Firmware: stage-count/table re-widening + uneven-split support (pack, module,
   adapter, node-context cross-checks).
3. Packer + contract generator: `--stage-index/--stage-count`, 56-pack identity
   set, deployment generator emitting per-(stage,rank) configs and non-colliding
   port plans.
4. Boundary transport leg + credits; delete the adapter's reject-non-zero-
   boundary shim once real transport lands (it is the compat-shim-in-inverse:
   a stand-in for missing capability).
5. Scheduler: pipeline DAG awareness, fill depth ≥ 7, per-stage admission and
   snapshot aggregation, deadline propagation across hops.
6. De-hardcode the DSpark tap plan: `spark_glm52_dspark_dispatch_policy.c`
   validates `pp_stage_count == 13 && stage_layer_count == 6` (policy.c 61–63)
   — derive from the deployment config or retable for [12,11×6].
7. Two-stage shrink gate before fleet bring-up: run stages {0,1} back-to-back
   in one process with the boundary leg loopbacked, exact-vs-BF16-master, then
   scale out.
8. Keep TP8 fanout as the low-batch mode: at B1–B16 the pipeline buys capacity,
   not latency (bubbles + hops), and ARCHITECTURE.md already lets the scheduler
   pick layouts per dispatch.

---

## 4. Round summary

- The driver's structure is sound where #673 already landed (admission ladder,
  dispatch-policy core, JIT-KV seam) and still carries four collapse-grade
  duplications (stagepack validators, adapter skeletons, dual round-major
  validation, the drafter's private kernel+cuBLAS stack).
- Dead weight: the whole-wave launcher, seven unity shims kept alive by
  string-matching tests, the production-unwired draft backend, and a README
  describing eight correctness gates whose validator was deleted — leaving
  publish blocked and the driver without an in-tree GPU numerical gate.
- Bandwidth: 12.1 GB/token/rank ideal → 44 ms floor; measured 6.91 tok/s is
  31 % of roofline at B1, rising to 82 % at B16. First-order fixes: collective
  control latency (144→<10 ms/token), the unresolved ~5.4× expert-stream
  question (one pack-header read settles it), CUDA graphs, restricted-vocab
  head. Projection: single stream 17–20 tok/s near-term, ~150–250 tok/s
  aggregate plateau for the 8-Spark fanout; prefill ~600–680 tok/s, TTFT(128)
  ≈ 0.35–0.4 s.
- PP7: split [12,11,11,11,11,11,11], taps aligned 1-per-stage, boundary bytes
  negligible at serving batches; blockers are the missing validator, uniform
  geometry constants, absent p2p boundary transport (actively rejected by the
  adapter today), pipeline-depth scheduling, and the PP13-hardcoded tap plan.
