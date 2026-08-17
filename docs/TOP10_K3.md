# K3 TOP-10 — top-speed assessment and ranked improvements

Owner: K3 MODEL agent · Area: Kimi K3 driver (TP4xPP4 bring-up, pack V2 fused KDA
tensors, interleaved MXFP4 GEMMs, the K3 DSpark drafter). Verified by grep/read
against this clone at `afb43a8`; every claim cited file:line. Proposal/docs only.

## Top-speed assessment (what exists, what is measured, what is missing)

**Exists (code landed, compile/numerically gated):**

- **Interleaved MXFP4 expert GEMMs (pack V2) — DONE.** `expert_w1_weight`/`expert_w2_weight`
  carry payload+E8M0 scales co-tiled in 17-row cells; `K3LayerLatentMoe` launches the
  `INTERLEAVED_B` GEMMs at both `TILE_K=128` and `TILE_K=32` (`inference/llms/kimi_k3/layer.cuh:850-905`,
  selectors at :217,220). The TILE_K=32 TP16 variant is numerically gated
  (`tests/test_k3_interleave_gemm.cu`, `docs/K3_TP16_REPACK.md` item 5).
- **Fused KDA projections (pack V2) — DONE.** q|k|v|beta ship as ONE tensor
  (`kda_qkv_beta_weight`) + the full-rank gate reconciliation: `kda_decay_down_weight`
  standalone, `kda_gate_weight` = checkpoint g_proj. The layer projects two/three wide
  GEMMs over one activation read then splits sections (`layer.cuh:592-609`, split kernel :519).
- **DSpark verify half — DONE.** Tap capture after aux layers {7,23,51,67,83}
  (`slice.cuh:412-419`, slab :132-136) and the KDA accepted-prefix fold
  (`K3FoldAccepted`, `slice.cuh`) are landed; the engine's draft submit/verify loop is
  host-side (`inference/llms/kimi_k3/engine.h:144,407`).
- **KV geometry — DONE.** `SparkK3KvFillCapacityRequest` fills the common capacity request
  (`model-families/k3/include/sparkpipe/spark_k3_kv_geometry.h:49-62`).

**Measured:** single-spark (sparka, stage-0 real rank pack, 1 token): warm step
**55.5 ms** = 2.3 ms/layer, graph-replay 54.2 ms; roofline 48.6 ms (20.6 tok/s)
(`docs/K3_PERF.md:36-37,49,52,56`). Derived: decode B1 **18.0 tok/s**, TP16 PP1
**20.2 tok/s** at 49.5 ms latency (`docs/K3_PERF.md:36,42`). Bit-determinism holds at
4 ULP everywhere (`docs/K3_PERF.md:7-11`). The interleaved path and fused KDA path are
covered by the single-spark real-weight gate (bit-deterministic, warm step ~54.8 ms,
`docs/K3_TP4PP4_PREP.md` "Device-direct collective + TP16 round").

**Missing (the gap):**

1. **No DSpark draft backend for K3.** The drafter shape table is fully pinned — block
   7, 8 verify positions, aux {7,23,51,67,83}, 5 draft layers, 64 Q heads over **16 KV
   heads (GQA)**, head dim 64, intermediate 14336 (`inference/llms/kimi_k3/dspark.h:39-48`;
   `model-families/common/include/sparkpipe/spark_dspark_drafter.h:41-63`). But no K3
   draft kernels exist: only `modules/glm52_dspark_draft_backend` (11 kernels, GLM52
   shapes), and K3's serving tier disables speculation (`max_speculative_token_count = 0u`,
   `modules/k3_resident_decode_stage/source/spark_k3_serving_adapter.c:467`).
2. **Serving KV seam is a stub.** `capability_flags = 0u`, `kv_cache_codec = 0u`,
   `cache_block_token_count = 0u` (`spark_k3_serving_adapter.c:451,461,478`) — no paged KV
   despite the correct geometry header (PROPOSAL_KV_SEAM.md §3.6, fully approved).
3. **No admission admit.** `spark_k3_resident_decode_stage_module.c` is a 61-line
   initialize/destroy stub — no `ResidentDecodeStageAdmit` and no serving admit, so the
   admission core is greenfield for K3 (PROPOSAL_ADMISSION_CORE.md Phase C/D).
4. **No full 16-rank TP4xPP4 measurement.** The end-to-end run still needs a ring
   reservation; K3 has no entry in `PERFORMANCE_STATUS.md` and no speculation number
   anywhere (the 18.0/20.2 tok/s are single-spark-anchored estimates).
5. **Stale decay|gate-fusion remnants** (dead code, see #1 below).

---

## TOP-10 improvements, ranked

### 1. Delete the dead decay|gate-fusion remnants (DRY win)

- **What:** The full-rank gate reconciliation removed the `decay_down|gate_down` fused
  tensor, but left it half-removed: the dead scratch buffers `fused_decay_gate_bf16` /
  `gate_latent_bf16` (`layer.cuh:241-242`, still carved at
  `spark_k3_resident_decode_stage_cuda.cu:252-253,283-284`); the dead static_assert +
  `K3_KDA_GATE_DOWN_OFFSET` (`layer.cuh:62-64`); the dead generated constant
  `K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS` (`inference/llms/kimi_k3/generated_config.h:61`,
  `tools/generate_k3_contract.py:167`); the dead packer helper
  `kda_fused_decay_gate_down_sections` + stale docstring (`tools/k3_pack.py:19,164`).
- **Why it is right:** DRY/structural — deleting a line is a solution at zero cost. Worse,
  the layout test is **bitrotted and self-contradictory**: `tests/test_k3_pack_layout.py:344-347`
  still asserts `kda_decay_gate_down_weight` exists (and :350 asserts `kda_decay_down_weight`
  is gone), while the packer emits `kda_decay_down_weight` standalone (`tools/k3_pack.py:681`)
  and `tests/test_k3_pack.py:218-222` asserts the opposite (decay_down present, decay_gate absent).
  One of the two tests must be failing today.
- **Code-size delta:** net **−40 to −60 lines** + ~768 bytes/row of dead scratch reclaimed.
- **Owner:** K3 model agent (model files + tools + tests).
- **First step:** Rewrite `tests/test_k3_pack_layout.py:340-353` to assert the reconciled
  layout (`kda_decay_down_weight` standalone, no decay|gate fusion), then delete the dead
  helper/constant/scratch fields and the two scratch carves.

### 2. Instantiate the neutral DSpark draft backend for K3 (DRY + speculation level)

- **What:** Finish neutralizing `modules/glm52_dspark_draft_backend` into one shared backend
  keyed on `SPARK_DSPARK_TARGET_*`, then compile it for K3. The backend already reads the
  neutral `SPARK_DSPARK_*` shape macros (`spark_glm52_dspark_draft_backend.cu:20-23`) and
  `spark_dspark_drafter.h:41-63` already pins K3's block-7 / GQA-16 table — but the ABI
  structs are still `SparkGlm52Dspark*` and the header still includes
  `spark_glm52_dspark.h`. The ONE structural kernel delta is GQA: the attention and KV-scatter
  kernels (KERNEL_CONTRACT_CARDS.md G52-D-05/G52-D-08) must map 64 query heads → 16 KV heads.
- **Why it is right:** DRY win — one ~1000-line backend instead of a second copy — **and** the
  largest performance level in this lane: it buys speculative decode, which the K3 verifier
  already supports (`engine.h:144,407`; `K3FoldAccepted`) but which is currently disabled
  (`max_speculative_token_count = 0u`, `spark_k3_serving_adapter.c:467`). SGLang's DSpark
  reference reports **+68% throughput at B256** (accept length ~2.7) and +24% few-shot math
  (`dspark.h:96-101`) — this is the **match-SOTA → exceed-SOTA** rung, from a 0-speculation
  baseline.
- **Code-size delta:** **−~1000 lines avoided** (no duplicated module); GQA attention delta
  **+~80 lines** in the shared backend.
- **Owner:** K3 model agent (request/own) + CUDA-KERNELS agent (GQA kernel) + speculation
  subsystem (neutralization). Land via coordinator.
- **First step:** File a Part-1 contract card copying §0's K3 row into `shapes`
  (KERNEL_CONTRACT_CARDS.md §4), and write the K3-sized pin test modeled on
  `tests/test_dspark_drafter_pin.c` against the K3 constants in `dspark.h:39-63`.

### 3. Wire the KV seam: fill `SparkKvModelTable`, retire the stub adapter (DRY + landed spec)

- **What:** K3 is the cleanest KV-seam adoption (PROPOSAL_KV_SEAM.md §3.6, fully approved by
  the scheduler sign-off). Fill `SparkKvModelTable` from `SparkK3KvFillCapacityRequest`
  (`spark_k3_kv_geometry.h:49-62`: 24 MLA compressed latents, 69-KDA slab stays in the slot
  pool), add the one `SparkK3PageCopy` primitive, and set `cache_block_token_count = 64`.
- **Why it is right:** DRY win (retire the `capability_flags = 0` / `kv_cache_codec = 0`
  stub at `spark_k3_serving_adapter.c:451,461,478` in favor of the one approved token-free
  table) **and** a correctness/capacity level: K3 gains paged KV + prefix reuse on the common
  arena instead of no serving cache at all.
- **Code-size delta:** **+~40 lines** (table fill + copy primitive), − stub fields.
- **Owner:** K3 model agent.
- **First step:** Add `SparkK3KvModelTableFill()` in the model-family header, pinned by a
  test modeled on `tests/test_k3_kv_cache.c` (the "seam, crossed" reference).

### 4. Add the K3 admission admit from the shared admission core (DRY + landed spec)

- **What:** `spark_k3_resident_decode_stage_module.c` (61 lines) has no admit, and the
  serving adapter has no admit. Adopt `SparkAdmissionPolicyTable` +
  `SparkAdmissionEvaluateShape` (PROPOSAL_ADMISSION_CORE.md §2.2, Phase D — greenfield like
  qwen38's stub): `max_active_sequence_count = 16`, `max_input_row_count = 16`,
  `DECODE_EQUALS_SLOTS`.
- **Why it is right:** DRY win (no hand-rolled decision ladder) **and** correctness: today any
  frame is accepted because there is no gate. Buys the accurate-but-slow → well-formed
  admission rung at zero duplicated code.
- **Code-size delta:** **+~25 lines** (a static table + one call).
- **Owner:** K3 model agent.
- **First step:** Land the core first (scheduler-owned), then add the K3 table + pin with the
  serving-adapter driver fixture asserting the request fields the driver's `admit` sees.

### 5. BF16 KDA state: select the launch site (perf level: decode toward roofline)

- **What:** The `uint16_t` KDA state instantiation already exists and is host-gated
  (`tests/test_kda_bf16_state.py`); only the launch-site selection is missing — the layer
  still refuses `kda_state_bf16 != 0` (`layer.cuh` KDA path + `K3FoldAccepted`).
- **Why it is right:** The decode roofline is bandwidth-bound on the **fp32 KDA state
  read/write** (`docs/K3_PERF.md:37`). Halving the state pool to bf16 cuts that DRAM term and
  doubles resident sequence capacity — this buys toward the **20.6 tok/s roofline** (one rung
  up from the measured 18.0), plus a capacity level.
- **Code-size delta:** **+~30 lines** (dispatch at two launch sites + numerics receipt).
- **Owner:** K3 model agent.
- **First step:** Add the launch-site branch in `K3LayerKda` and `K3FoldAccepted` behind
  the existing flag, then run the kda gate's bit-equivalence decode-vs-verify-vs-fold.

### 6. TP16 balanced w2 half-tile repack (pack V3 + TILE_K=64) (perf level: TP16)

- **What:** The sharder currently REFUSES TP16 loudly — the w2 gate half (12 tiles) does not
  divide 16 ranks (`docs/K3_TP4PP4_PREP.md` "TP16 sharder audit"). The fix is the 64-element
  half-tile repack + a TILE_K=64 INTERLEAVED_B variant; the TILE_K=32 machinery already
  generalizes (`docs/K3_TP16_REPACK.md` item 3).
- **Why it is right:** Removes the last TP16 blocker; TP16 PP1 is measured at **20.2 tok/s /
  49.5 ms** with ~4x lower latency than pipelined TP4xPP4 (`docs/K3_PERF.md:42`). A latency
  level, not just throughput.
- **Code-size delta:** **+~60 lines** (TILE_K=64 instantiation + pack tile_k 64).
- **Owner:** K3 model agent (packer/sharder) + CUDA-KERNELS agent (TILE_K=64 variant).
- **First step:** Add `expert_tile_k 64` to the packer's geometry assertion, then the TILE_K=64
  GEMM instantiation in `unity.cu` + `tests/test_k3_interleave_gemm.cu` case.

### 7. De-duplicate the interleaved GEMM launch branches (DRY win)

- **What:** `K3LayerLatentMoe` holds four near-identical 3-way launch blocks (w1 indirect
  128/32, w2 direct 128/32) at `layer.cuh:850-905` — the only delta is `TILE_K` and the
  indirect flag.
- **Why it is right:** DRY/structural — a single templated dispatch on `expert_tile_k` removes
  the copy-paste and makes #6's TILE_K=64 a one-line addition rather than two more blocks.
- **Code-size delta:** **−~15 lines**.
- **Owner:** K3 model agent.
- **First step:** Factor a `template<uint32_t TILE_K> K3ExpertW1W2Launch(...)` helper.

### 8. Reduce-scatter + all-gather for the fused TP4 all-reduce (perf level: larger batch)

- **What:** The slot-encoded full-width all-reduce moves 4x the minimal bytes; a
  reduce-scatter + all-gather pair would halve per-rank wire traffic
  (`docs/K3_PERF.md:67`).
- **Why it is right:** Incremental — at B1 the path is weight-bandwidth-bound and the
  2×~5-15 us/layer AR is a small term vs 2.3 ms/layer (`docs/K3_PERF.md:67,49`); it pays at
  batch where the AR becomes load-bearing. Ranks below the speculation/KV/admission items
  because it does not move a whole METRIC level at the measured batch.
- **Code-size delta:** **+~50 lines** (in the device collective tier).
- **Owner:** K3 model agent (collective tier).
- **First step:** Prototype the reduce-scatter/all-gather pair against the fused 3×7168 buffer
  and re-run the TP4 offline equivalence gate.

### 9. Move the head exchange off the host tier (perf level: incremental AR)

- **What:** The head exchange still uses the host tier — its f32 slots have no NCCL f32
  collective; a bf16-splittable slot layout or an f32 NCCL op moves it device-direct
  (`docs/K3_PERF.md:67`).
- **Why it is right:** Same incremental AR class as #8; correctness-neutral, removes one host
  staging hop from the TP4 head path.
- **Code-size delta:** **+~20 lines**.
- **Owner:** K3 model agent.
- **First step:** Add an f32 NCCL op for the head slots (mirroring the embedding exchange that
  already moved to NCCL, `docs/K3_PERF.md:64-65`).

### 10. Measure + promote: full 16-rank end-to-end and the K3 PERFORMANCE_STATUS entry

- **What:** Run the reserved 16-rank TP4xPP4 end-to-end (`tools/fleet_swap.sh k3`) and
  promote K3's numbers from `docs/K3_PERF.md` into `PERFORMANCE_STATUS.md` (K3 has no entry
  there, and no speculation number exists anywhere).
- **Why it is right:** This is the "what is missing" that gates every perf claim above — the
  18.0/20.2 tok/s are single-spark-anchored estimates, not cluster measurements. No code, but
  it is the prerequisite that turns "match SOTA" claims into receipts.
- **Code-size delta:** **0 lines**.
- **Owner:** K3 model agent + coordinator (ring reservation).
- **First step:** Reserve the spark ring, run `fleet_swap.sh k3`, and record the receipt +
  the speculation-on number once #2 lands.

---

## Cross-model landed specs and how they apply to K3

- **PROPOSAL_ADMISSION_CORE.md** — applies as **greenfield** (Phase D): K3 has no admit today
  (`spark_k3_resident_decode_stage_module.c` is a stub); adopt the shared
  `SparkAdmissionPolicyTable`. → item #4.
- **PROPOSAL_KV_SEAM.md** — applies as **the cleanest adoption** (§3.6): geometry header done,
  serving adapter is the stub; fill `SparkKvModelTable`. Fully approved by the scheduler
  sign-off. → item #3.
- **PROPOSAL_DSV4_TREE_ADOPTION.md** — K3's verifier is already a linear-chain longest-prefix
  host loop (`engine.h:407-445`), so the shared `SparkSpeculationTreeResolve` is the same
  DRY adoption for K3 as a follow-up once #2 lands (not top-10 because it is a small win behind
  the missing drafter kernels).
- **KERNEL_CONTRACT_CARDS.md** — §0 pins the K3 drafter shape (block 7, GQA 16, head dim 64);
  §4 is the request path. K3's cards are **not yet written** — the GQA attention/KV-scatter
  cards (the G52-D-05/G52-D-08 equivalents) must be filed with `requestor: k3` and the
  concrete target number (beat 18.0 tok/s decode / add speculation). → item #2.
