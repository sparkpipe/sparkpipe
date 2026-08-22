# K3 driver audit + bandwidth projections — round report

Scope: the K3 TP16 MXFP4 stack as merged in PR #666 (TP4xPP4 deployment,
device-direct collectives, graph capture, TILE_K 32) and the open PR #667
items. Audited tree: local `unified` @ `9620f23`, cross-checked against
`origin/k3-tp4-layer0` (`bd34381`), `origin/k3-tp4pp4`, `origin/main`, and
PR bodies #666/#667 fetched live. Report only this round — no code was
changed. Classification discipline: every number below is labelled
**measured** (one anchor: the 55.5 ms single-spark stage step) or
**analytical**; K3 has never run on fleet hardware and its dashboard rows
were retracted once already (`5ec62ea`), so nothing here may be read as a
fleet measurement.

---

## 1. The #667 divergence — status, and what closes it

PR #667's body is stale relative to the tree. Of its three fronts, two are
root-caused and fixed on `unified`, one has an unmerged fix sitting on a
side branch, and two items (TP16 pack production, end-to-end run) are
exactly where the body left them.

### 1a. Tail nondeterminism / layer-0 equivalence — ROOT-CAUSED, FIXED (documented)

`docs/K3_PERF.md` §"2026-08-17" records it and the code agrees:

- The known tail variance (hidden[6144..7167], ~14 ULP, across runs AND
  across the TP4-vs-full equivalence) was the **KDA o_proj weight-only GEMM
  running with output == input == `attention_out_bf16`**. The persistent
  GEMM stages A per k-tile while storing finished output tiles, so a
  second-wave tile's A reads raced first-wave stores. Fixed in
  `inference/llms/kimi_k3/layer.cuh:683-691`: the projection lands in
  `hidden_bf16` (idle until the MLP-side retrieval overwrites it); the MLA
  path was already safe. The gate's 64-ULP tail exemption is gone; fresh-run
  determinism and capture fidelity hold at **4 ULP**, and the TP4 4-rank sum
  equals the full-stage run to bf16 rounding.
- The "plain BF16 GEMM breaks at K >= 512" repro was a **test artifact, not
  a kernel bug**: the probe filled activations with `((x % 11u) - 5)`, whose
  subtraction happens in UNSIGNED space and wraps negatives to ~4.29e9. The
  GEMM faithfully summed poisoned data; cross-process byte-exactness at the
  real K extents (7168/12288/33792) exonerates it. The gate
  (`tests/test_k3_interleave_gemm.cu`) now probes the o_proj's real 3072 k
  extent with 8-run determinism checks and passes
  ("direct + indirect + tile_k 32 + second wave + plain").

### 1b. The MoE TP4 sharding path — three more compiled-clean bugs, fixed on unified

The offline TP4 gate (`tools/k3_tp4_equivalence_check.py`: full pack at
tp_degree 1 vs the four rank packs; contract `full[k] ≈ Σ rank[k]`) flushed
out a chain that all type-checked:

1. `5385a63` — the expert **w2 GEMM arg order was swapped** (input/output
   transposed the interleaved grid; the finalize read moe_in columns of a
   w2_in-wide output).
2. `1f8b190` — the production numerics fix that supersedes the diagonal
   scheme: **w1 input-splits its k-tiles and keeps the gate|up output FULL,
   all-reduced BEFORE SiTU** (because `sum(SiTU(p)) ≠ SiTU(sum(p))`), and
   **w2 output-splits its latent cells with the intermediate input FULL**.
   This is what `tools/k3_shard.py` implements today
   (`_expert_gate_up` / `_expert_down` docstrings + code).

### 1c. The remaining open root: 1-D tensors that ride sharded axes — fix exists, NOT merged

`origin/k3-tp4-layer0` tip `bd34381` (Aug 17, hours before `5385a63`) found
the same w2 swap **and** a third defect that `unified` does not have:

- `kda_decay_bias`, `kda_head_log_scale`, `routed_norm_weight` ride the
  sharded head/latent axes but are classified **REPLICATED** in today's
  `tools/k3_shard.py:61-68`, so every rank pack carries the full array —
  while the kernels index them rank-locally from zero:
  `LmBoundedDecayKernel` reads `channel_bias[head*KEY_DIM+i]` and
  `head_log_scale[head]` over the rank's LOCAL heads
  (`inference/kernels/linear_attn.cuh:106-116`, launched with
  `K3_RANK_DIM(b,kda_heads_rank,...)` at `layer.cuh:654-655`), and the
  latent RMS norm reads `routed_norm_weight[0..moe_in)`
  (`layer.cuh:948-949`). Ranks 1..3 therefore consume rank 0's segments on
  **every KDA layer (69 of 93) and every MoE layer** — exactly the
  "bisected to the dense+KDA path; ranks agree bit-identically" signature
  #667 records. Neither the binder (`spark_k3_bind.c` BIND_ONE by name) nor
  the dispatch (`spark_k3_resident_decode_stage_cuda.cu` WF table) applies a
  per-rank offset; I found none anywhere else.

**What closes the divergence, in order:**

1. Port `bd34381`'s classification fix (slice those three tensors per rank,
   or apply per-rank pointer offsets in bind/dispatch — pick one mechanism)
   onto `unified`, with the `tests/test_k3_shard.py` reassembly coverage it
   already wrote.
2. Re-slice the rank packs with the corrected sharder and re-run
   `tools/k3_tp4_equivalence_check.py` on the layers-0-3 slice pack
   (the 4-layer ~65 GB pack that registers under the 48 GB chunking from
   `1febd9c`); record the max-relative-deviation receipt.
3. Re-validate capture fidelity (the gate-rework item: no-capture direct
   step-2 vs graph) — it was failing *because of* the GEMM bugs and must be
   re-proven on the fixed path.
4. Update PR #667's body (items 1a/1b are closed; cite receipts) and close
   it when 1-3 land. Bisect artifacts remain at `spark1:/tmp/k3h4.tar`.

Still open exactly as #667 says: **TP16 pack production** — code-complete
(packer takes `expert_tile_k`; at tile_k 32 the w1 grid is 112 k-tiles,
112 % 16 = 0, and w2's 384 cells % 16 = 0, verified against
`interleave_geometry`), one full-model packer run away; and the **end-to-end
run**, blocked on a ring reservation (K3 sits behind DSV4 Pro and
Qwen 3.8 Max in the `COORDINATION.md` queue; note that doc's claim
"chunked registration is still being fixed" is stale — `1febd9c` landed it).

---

## 2. DRY/structure findings vs the quality law

The law (`docs/archive/DRY_LEDGER.md` §"The law", restated as
`unified`'s "strict general-vs-model-specific law"): a model's name lives
only in `inference/llms/<model>/` and `model-families/<model>/` (plus, per
`UNIFIED.md`, `modules/<model>_*` and per-model tools); shared code never
names a model; a `Spark<Model>X` with no `<MODEL>_` constant is a defect of
the same class as a wrong constant.

**What passes.** `tests/test_dry_law.py` PASS (145 common files across
`include/sparkpipe`, `node`, `ring`, `runtime`, `inference/kernels`, ...);
hand grep agrees (zero `kimi`/`K3` word hits in shared paths, including the
files #666 touched — `inference/kernels/{gemm,tile,mxfp4,tensor_map}.cuh`,
`runtime/gemm.cuh`, `ring/transport/tp_collective.c`). `tests/test_code_size.py`
holds the ceiling exactly (175424/175424, no growth). Model facts sit where
the law wants them: `inference/llms/kimi_k3/` (kernels, `dspark.h`,
`engine.h`), the audited `model-families/k3/` shim whose constants match
`config.h` with the old wrong values documented as `/* was ... */` comments
(the config-drift incident class, closed), `modules/k3_resident_decode_stage/`,
`tools/k3_*`. `tests/test_k3_driver_contracts.py` is the law's spirit done
right: fail-closed options, graph-capturability, pack-V2-only bind — gated
as source because they are the kind of contract that "compiles while it
rots".

**Findings (each verified today):**

1. **A green gate that tests nothing: `tests/test_k3_pack_layout.py`.** Its
   `end_to_end()` mini fixture still names tensors without the
   `language_model.` prefix (`:170`), the reconciled packer refuses it
   ("missing tensor: language_model.model.embed_tokens.weight"), and the
   function **returns early** — `FAILURES` stays 0 and the suite prints its
   success banner with exit 0 (reproduced: true exit 0). Behind the early
   return sit assertions that contradict the shipped layout: `:344-353`
   require `kda_decay_gate_down_weight` present and `kda_decay_down_weight`
   "should not exist" — the exact inverse of the released checkpoint's
   full-rank gate reconciliation (`docs/K3_GATE_RECONCILIATION.md`) and of
   `tests/test_k3_pack.py:218-222`, which passes. Fix the fixture, fix the
   assertions, and make an early return count as a failure.
2. **Dead decay|gate-fusion remnants (TOP10_K3 #1, confirmed).** The
   full-rank gate reconciliation removed the fusion but left it
   half-removed: `fused_decay_gate_bf16` / `gate_latent_bf16` are carved per
   rank (`spark_k3_resident_decode_stage_cuda.cu:283-284`, budgeted at
   `:252`) and never read by any launch; the `static_assert` +
   `K3_KDA_GATE_DOWN_OFFSET` (`layer.cuh:62-64`); the generated constant
   (`generated_config.h:61`, `tools/generate_k3_contract.py:167`); the
   packer helper `kda_fused_decay_gate_down_sections` (`tools/k3_pack.py:164`);
   and `k3_pack.py`'s module docstring (`:10-19`) still advertises "TWO
   fused tensors per KDA layer" including `kda_decay_gate_down_weight`,
   contradicting its own emitter. Net −40-60 lines plus the dead scratch.
3. **`tools/k3_pack.py` is the last holdout off `spark_pack_common`.** All
   five sibling packers import the shared core
   (`dsv4_stagepack`, `dsv4_tp16_stagepack`, `glm52_resident_stagepack`,
   `qwen36_stagepack`, `qwen38_stagepack`); K3 still embeds its raw-bytes
   safetensors reader — the exact B-tier item DRY_LEDGER §"Tooling
   duplication" queued.
4. **Two sharding schemes claiming one contract.** `origin/k3-tp4-layer0`'s
   diagonal w1/w2 + sharded-norms scheme (`bd34381`, also described in
   `docs/K3_TP16_REPACK.md` and `K3_PERF.md`'s TP16 audit tail) vs
   `unified`'s input-split-w1/output-split-w2 (`1f8b190`). Worse,
   `tools/k3_shard.py` itself now carries **three** descriptions: the module
   header (`:25-42`, the diagonal era) contradicts its own method
   docstrings (the current scheme). One scheme must win; the loser's docs,
   tests and branch die. (Related arithmetic nit: `K3_TP16_REPACK.md:16`
   says "TP 1/2/4/8 keep the 128-element tiles (28 and 24 tiles divide
   evenly)" — 28 % 8 = 4, so TP8 also needs tile_k 32; only 1/2/4 pass at
   128.)
5. **The `device_collective` config dialect is a seed, not a debt — yet.**
   K3's serving adapter hand-rolls ~140 lines of JSON walking for it
   (`spark_k3_serving_adapter.c:99-236`); no other adapter parses it. Fine
   under the law (model module), but the moment a second model adopts the
   device tier this becomes the queued "tp_collective config extraction"
   copy #2 — extract into `ring/transport`/`include/sparkpipe` then, not
   after four copies exist.
6. **Process drift:** PR #667's body lists as open what `K3_PERF.md` marks
   fixed (§1a above); `COORDINATION.md` queue note contradicts landed
   `1febd9c`. `PERFORMANCE_STATUS.md`/dashboard honestly keep K3 at
   "none measured / RETRACTED (not hardware-measured)" — correct, and the
   bar to change it is a ring-day receipt.

---

## 3. Bandwidth-based prefill + output projections vs SOTA (GB10, K3 geometry)

Method: `tools/k3_tp4pp4_perf_estimate.py` (analytical; inventory read from
the deployed rank-pack manifest), repo bandwidth convention **273 GB/s
LPDDR5X × 0.65 = 177.45 GB/s usable per spark**, NCCL tree AR ≈ 8 µs × 2 per
layer (phase 0 + phase 1, correctness-mandated), anchored to the one
**measured** number: warm B1 stage step 55.5 ms (graph replay 54.2 ms,
bit-identical) on sparka, stage 0, real rank pack. Geometry per rank (TP4
slice): 93 layers = 69 KDA + 24 MLA; per-KDA-layer dense 327.2 MB, per-MLA
268.7 MB; expert set 1965.3 MB/layer/rank (MXFP4-E2M1 g32 + E8M0, all 896
experts), top-16 routing; embed/lm_head 587.2 MB each; fp32 KDA state
13.17 MB/token/layer ÷ 4 (TP) = 3.29 MB; MLA KV 1152 B/token/layer.

**Output (decode, B1):**

| quantity | value | class |
| --- | --- | --- |
| measured stage step | 55.5 ms → **18.0 tok/s** | measured (single-spark anchor) |
| roofline stage step (worst stage 70: 8.57 GB → 48.3 ms + 0.37 ms AR) | 48.6 ms → **20.6 tok/s** | analytical |
| TP16 (PP1) decode | 49.5 ms token latency → **20.2 tok/s** | analytical |

Worked arithmetic (worst stage, layers 70-92 = 16 KDA + 7 MLA): dense
16×327.2 + 7×268.7 = 7116 MB; experts 23 × 1965.3 × 16/896 = 807 MB; state
16 × 3.29 = 53 MB; lm_head 587 MB; activations ≈ 2 MB → **8.57 GB/token/rank**
→ 8.57e9 / 177.45e9 = 48.3 ms. The measured 55.5 ms sits at **87% of the
roofline** — the path is already weight-bandwidth-bound, exactly as the
capture audit concluded (capture's win is wall-clock host-enqueue, ~55 ms
serialized, not kernel time).

Absolute-hardware sanity check: even at 100% of 273 GB/s with no AR, the
worst stage needs 31.4 ms → **~32 tok/s is the GB10 B1 no-spec ceiling for
this topology**. We measure 18.0 (56% of absolute). Conclusion: no kernel
work moves B1 decode past ~32 tok/s; only **byte reduction** (FP8 KV, BF16
KDA state, batching the expert stream) or **speculation** (amortize weight
reads over accepted tokens) does — which is why the DSpark lever below
dominates.

**Prefill (B tokens per PP4 pipeline step, steady state = B / worst stage):**

| batch | stage ms | steady tok/s | single-prompt latency |
| --- | --- | --- | --- |
| 8 | 86.7 | 92 | 0.35 s |
| 32 | 208.9 | 153 | 0.84 s |
| 56 | 331.1 | 169 | 1.32 s |
| 128 | 356.1 | 359 | 1.42 s |
| 256 | 400.3 | 639 | 1.60 s |
| 512 | 488.9 | 1047 | 1.96 s |
| 1024 | 666.1 | **1537** | 2.66 s |

The expert stream saturates at **B=56** (16 × 56 = 896 experts); beyond it,
prefill bytes grow only with the per-token terms. At B1024 the worst stage
moves ≈116 GB: experts 45.2 + **fp32 KDA state 60.7 (52%!)** + dense 7.5 +
activations 2.1 + embed 0.6. The state term is strictly per-token, so the
**BF16-KDA-state lever is a prefill lever too**: halving it → ≈85.7 GB →
≈483 ms roofline → ≈**2119 tok/s (+38%)** at B1024. TP16 prefill is parity
(1511 tok/s at B1024) with 0.68 s single-prompt latency (no pipeline fill).
(Hand check of the tool's stage arithmetic agrees within ~2%.)

**vs SOTA at matched weight precision** (`docs/SURVEY_K3.md`, sources
2026-08-17): the only precision-matched published numbers are vLLM on
16× GB300 NVL72, TP16, MXFP4 experts + BF16 dense — **118 tok/s B1 no-spec;
370 with DSpark (controlled 3.14×); 464 peak**. Everything else (NVFP4,
2.5-bit, GGUF, MLX mixes) is a different codec; GPUStack 8×B300 reports
e2e durations only; NVIDIA's Dynamo recipe publishes no tok/s.

- Output gap: 118 vs our 18.0 measured = **6.6×** (5.7× on rooflines).
  Per-device streaming bandwidth is 273 GB/s LPDDR5X vs ~8 TB/s HBM3e
  ≈ 29×, but the observed gap is far smaller because **B1 decode is
  latency/launch-bound on both stacks**, not bandwidth-saturated — so the
  honest reading is: the SOTA box is a software-lever story (fused KDA +
  fused AttnRes we already have; DSpark 3.1-3.9× and FP8 KV we don't), not
  a kernel-efficiency indictment. Matching the reference's *software* state
  projects us to 18 × 3.14 ≈ **56-65 tok/s B1** (survey's own estimate) —
  past our ~32 tok/s no-spec hardware ceiling, which is precisely what
  speculation is for.
- Prefill: **no matched-precision published K3 prefill tok/s exists** — the
  92→1537 tok/s curve is unchallenged at matched precision. Nearest
  context points, not comparable: aiconfigurator Qwen3-32B-FP8 dense at
  684.8 tok/s/GPU (32× H200, TP4, FP8); GPUStack's 200K-context degradation
  battle (dcp=8: SGLang 1.25× vs vLLM 3.29×); prefix-cache multipliers
  (14-22×) from the 27B sibling on shared-prefix traffic. Our prefill rungs
  are #8 in the survey (ragged MLA prefill + query quantization) and the
  BF16 KDA state.

---

## 4. What K3 inherits from the qwen27B context-fix pattern

The pattern (qwen36 lane, Aug 19-20: `484fc3e` → `63cc76f` → `48c7f42` →
`6a9e536`/`f9a183f`, plus `2daa137`): the DFlash2 acceptance leak was not
numerics — it was **drafter context geometry**. The reference builds the
drafter context from the 5 tap layers' hidden states at EVERY position
(`[N,5H] fc → [N,H]`, attention over **N+8 keys**, window 2047); our port
used ONE position's taps → 9 keys, starving the drafter of the prompt
(proven: N=1 bit-identical to logged drafts, N≥2 diverges 5/7..7/7). The
fix: per-lane 2048-slot tap ring, tiled-K N-row projector for the 25600-wide
fc (the scalar path exceeds GB10's 101376-byte shared-memory opt-in cap),
N+8-key attention with absolute rope positions, per-slot workspaces
(+28 MiB/slot, +100 MiB/lane). Result: accepted +17-25% (1.81 honest),
credited 4.81 = 88% of published, **lossless held** (spec stream ==
no-spec golden, 128/128).

K3 inherits it concretely, because K3's next big level is the same problem:

- **K3's DSpark drafter is the same context-geometry machine.** Aux taps at
  layers {7,23,51,67,83}, block 7, 5 draft layers, and a **block-diffusion
  drafter with non-causal attention over tap-derived context**
  (`inference/llms/kimi_k3/dspark.h`, `docs/SURVEY_K3.md` #1). Build the
  N-position context from day one — no one-position scaffold, even as a
  bring-up shortcut. The shape table is fully pinned and the verify half is
  landed (tap capture `slice.cuh:412-419`; `K3FoldAccepted`;
  `engine.h:407` `K3EngineCommitVerify`); only the draft backend is missing
  (speculation disabled at `spark_k3_serving_adapter.c:467`).
- **Take the shared backend, not a copy** — the law's seam, already
  prepared: neutralize `modules/glm52_dspark_draft_backend` (reads
  `SPARK_DSPARK_TARGET_*`) against `model-families/common/.../spark_dspark_drafter.h`,
  with the ONE kernel delta being GQA 64 query → **16 KV heads** (TOP10_K3
  #2: ~1000 lines avoided, +~80 lines). `dspark.h` itself flags the two
  constants that would not announce themselves: KV heads 16 vs 64, and the
  **NoPE backbone vs roped drafter** — a gate that reasons "kimi_k3 is NoPE,
  therefore no rope kernel" would silently mis-build the drafter.
- **Interrogate the reference before tuning.** Port the harness pattern
  (`VLLM_DFLASH2_INPUT_DUMP` → replay OUR forward on the reference's exact
  per-round taps/inputs; layer-bisect against the reference model on the
  same dumps). It converts "acceptance is low" into a localized diff and is
  the only reason the qwen27B leak was found in days. Wire the
  accepted/proposed counters into `K3EngineResolveVerify` first (survey
  #10, ~15 lines) so the drafter lands with a measured acceptance rate.
- **Gate discipline, three rules.** (1) A multi-row verify gate's
  *reference side must actually run all rows* — the qwen27B 8-row gate ran
  2 and compared against calloc zeroes, producing phantom corruption
  verdicts (`63cc76f`); `K3FoldAccepted`'s `verify_row_begin` path needs
  that class of test. (2) Hunt the softmax/context-row corruption class
  (`2daa137`: `scores[0]=max` overwrote context row 0 every layer). (3)
  Lossless before speed: spec stream == no-spec golden, hash-pinned, and
  report accepted and credited separately — the dashboard's dual-metric
  honesty is what made the fix credible.
- **The GB10 constraint carries over**: any N-row projector K3's drafter
  needs (its fc input is 5 taps × 7168 = 35840-wide) hits the same
  101376-byte shared-memory opt-in cap — the tiled-K N-row projector is
  written, budget the same workspaces (+28 MiB/slot, +100 MiB/lane ring)
  into the K3 pool sizing (`spark_k3_pool_sizing.h`).

---

### Round bottom line

The #667 divergence is root-caused and fixed except one unmerged tensor-class
fix (`bd34381`) that is the leading candidate for any residual layer-0 gap —
port it, re-run the offline equivalence gate, re-prove capture fidelity, then
close #667. The K3 stack is law-clean where it counts (naming, ceilings,
contracts) but carries one silently-green layout gate, one dead-feature
tail, and one packer off the shared core. On paper the GB10 fleet delivers
~18-21 tok/s B1 output and ~1.5k tok/s prefill at B1024 against a matched-
precision SOTA of 118/370/464 on GB300 — a software-lever gap (DSpark, FP8
KV, BF16 state), not a kernel one, and nothing in this section is a
measurement until the ring day happens.
