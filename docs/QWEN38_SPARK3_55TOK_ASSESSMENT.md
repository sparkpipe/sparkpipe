# Qwen3.8-27B spark3: the ~55 tok/s question — lever assessment (2026-08-25)

Session target: close the gap from the 24.39 tok/s O512 baseline to sglang's
claimed ~55 tok/s for Qwen3.8-27B. Every number below is client-measured on
spark3 (GB10, deployment root `/home/spark3/sparkdata/qwen38.fp8.tp1`, port
17480) against the pinned production stack (pack `e61634942bf441aeca8a603e885c2480`,
release binaries 2026-08-25), exclusive window, warm daemon.

## TL;DR

1. **Baseline reproduced bit-exact**: O512 rows=8 → **20.94 s = 24.45 tok/s**,
   stream sha256[:16] **`d7f798801a6e43a6`** (== release pin), 77 rounds,
   steady accepted=7 (E≈5.66). Repeated under `SPARK_QWEN36_PROFILE=1`
   (20.75 s), `SPARK_QWEN36_FRAME_GRAPH=0` (21.18 s), and
   `DFLASH2_WINDOW=256` (20.92 s) — all four runs hash-identical.
2. **Lever 1 (BF16 lm_head): already shipped — no codec change needed.**
   The production pack stores LM_HEAD as **BF16** (tensor kind 2,
   weight_format 0, 248320×5120; verified by dumping the pack directory on
   spark3). The module builds an MXFP4 **shadow** of that BF16 head at init
   (`HeadShadowQuantize`, module.c:813) plus per-neuron certified error
   norms, uses it only to shortlist candidates, and rescores them **exactly
   against the BF16 weights** (`LaunchHeadScreenedArgmaxScore`,
   module.c:1923/1997; "certified screened argmax", commit `2411482`).
   Functionally this IS sglang's BF16 lm_head with a lossless pruning
   accelerator in front; the deployed driver carries all five
   certified-screen kernels (verified via `strings lib/model_driver.so`).
3. **Lever 2 (rows=64 prefill-only): confirmed on-box** — prefill-only
   (budget=1) rows=8 → 2.68 s (47.8 tok/s) vs rows=64 → **1.23 s (104.1
   tok/s)**. Generation cells MUST stay at rows=8 (M≥32 prefill rides the
   native-MMA path, not tap-stable: E 5.66→~2.8, stream diverges — see
   spark3 `OPS_NOTES.md`). Cell generators are now in-repo:
   `tools/qwen38_27b_spark3_bench_cells.py`.
4. **Lever 3 (CUDA-graph the verify path): premise corrected, upside
   bounded.** The "~76 ms launch overhead per frame" figure is a misread of
   commit `5b27165`: 76.2 ms was the FFN **frame time**, and that commit's
   launch-gap quantification measured overhead at **8–10 ms/frame no-spec,
   ~16 ms/round spec (~9%)**. Frame graphs are ALREADY default-ON for plain
   frames (commit `b4de6a2`); live A/B on spark3: graphs ON 20.75 s vs OFF
   21.18 s (**+2.1%**, both bit-exact). The verify frame remains eager by
   design (`graph_blocked` includes `SPECULATIVE_VERIFY`,
   module.c:2986) pending handoff §0 blockers a–c.
5. **The round budget makes ~55 tok/s unreachable from these levers.**
   Measured decomposition (profiled run, deltas over frames=80→88):
   frame GPU-busy ≈ **184 ms** = FFN 100 + GDN 47 + head 21 + attn 14 +
   ~2 gaps; median round cadence **229–231 ms**; residual ≈46 ms/round is
   drafter GPU + host orchestration/syncs. At 6.65 committed tokens/round,
   even a zero-overhead round keeps only the DRAM floor: one verify frame
   re-reads the full 27 GB weight set at ≤266–273 GB/s ⇒ ≥99–109 ms/round
   ⇒ ceiling ≈ **29–31 tok/s E2E** with ALL quality-lossless engine levers
   landed (GDN chunked scan −10 ms, drafter/host −15…30 ms).

## Measurements (all O512 canonical prompt, warm, exclusive)

| Config | Wall | Rate | Stream | Notes |
|---|---|---|---|---|
| rows=8 (canonical) | 20.94 s | 24.45 | `d7f798801a6e43a6` | release pin reproduced |
| + PROFILE=1 | 20.75 s | 24.70 | same | phase counters on |
| FRAME_GRAPH=0 | 21.18 s | 24.17 | same | graphs worth +2.1% |
| WINDOW=256 | 20.92 s | 24.48 | same | ctx-cache neutralizes W |
| prefill-only rows=8 | 2.68 s | 47.8 tok/s | n/a | matches findings doc |
| prefill-only rows=64 | 1.23 s | **104.1 tok/s** | n/a | prefill-only cells only |

## Where the remaining time actually is (per spec round, ~230 ms)

| Component | ms | Lever class |
|---|---|---|
| FFN sweep (64 layers) | 100 | AT its 248 GB/s class wall post-e8m0 fix (`5b27165`); TMA-class ceiling saves ≤~8% more |
| GDN scan (48 layers) | 47 | chunked scan ≈ −10 ms (quality-lossless, unlanded) |
| Head (screened argmax) | 21 | at wall |
| Attention (16 layers) | 14 | minor |
| In-frame launch gaps | ~2 | already graphed/captured cheap |
| Drafter GPU + host/sync/D2H | ~46 | padding-select sync drop (blocker a) + device-side selection (blocker b) + verify-frame capture (§0 plan) |

## Path to >30 tok/s (what would actually move the number, in order)

1. **Land the §0 verify-path plan** (blockers a→c; capture per K3
   pattern): bounded by the 46 ms/round residual — realistic +8–13%
   (24.5 → ~27).
2. **GDN chunked scan**: −10 ms/round (+4–5%).
3. **Acceptance beyond 7/8**: attributed to input-side fp8-target taps vs
   the bf16-teacher drafter (handoff §0) — training-side work (drafter
   fine-tuned on fp8 activations), NOT an engine fix. Each +0.5 E ≈ +7%.
4. **NVFP4/MXFP4 full weight stream** (~½ bytes/frame): lifts the hard
   floor to roughly 40–45 tok/s — a quantization-quality trade requiring
   its own audit (see the MX postmortem, handoff §0d, before any repack).
5. Aggregate (multi-request) throughput scales with batching but per-user
   rate does not; the sglang ~55 claim is not comparable to our
   greedy-bit-exact single-stream regime (community SOTA band on this
   hardware class: 28–32 decode-only; NVIDIA-forum SGLang NVFP4: 34–38).

## Session artifacts

- `tools/qwen38_27b_spark3_bench_cells.py` — in-repo cell generator
  (lands OPS_NOTES recommendation; replaces /tmp-ephemeral files).
- spark3 `OPS_NOTES.md` appended with this session's verification block.
- Daemon restored to canonical launcher state after experiments.
