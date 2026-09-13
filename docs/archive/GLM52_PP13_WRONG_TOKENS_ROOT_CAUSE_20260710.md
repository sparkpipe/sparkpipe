# GLM52 PP13 wrong tokens — status handoff (updated)

**Date:** 2026-07-10 (updated same day after PRs #248–#250)
**Repo:** `sparkpipe`
**Symptom (historical and current):** 13× SparkPipe ring inference can complete end-to-end while emitting **wrong** and/or **nondeterministic** tokens at temp=0.

**Important:** PR #250 archived a useful investigation note, but it is **not** “the bug is found and fixed.” This file supersedes the stale claims in that first revision.

---

## Current verdict (post #248 / #249)

| Item | Status |
|------|--------|
| RoPE half-split (`b5bc7a8` / PR #235) | **Fixed on current `main`** (ancestor of HEAD). Necessary historical root cause; **not sufficient** to explain today’s wrong tokens if the live release is truly current main. |
| Dense FP8 prepared MLP / silent BF16 fallback | **Fixed in PR #248 + #249** (after #247 left it incomplete). Live probe after deploy: `fbgate/fbinter/fbdown = 0` (old varying fallback path **not** active). |
| Absorbed-latent attention default, serial prefill race/pos=0, restricted-vocab greedy | Fixed earlier; keep on the deploy checklist for old releases. |
| **Remaining open issue** | **Still real.** After #249, inference runs end-to-end, but **three identical greedy prompts still produced different token ids: `17`, `2064`, `18300`.** Nondeterministic wrong generation on a path that no longer uses the old dense BF16 fallback. |

**Do not treat PR #250 or the RoPE/dense stories alone as the final answer for current bad inference.**

---

## What was real in the original analysis (and what changed)

### 1. RoPE interleaved vs `rotate_half` — real, already fixed

- Production once rotated adjacent elements `(2i, 2i+1)`; trained convention is **`rotate_half`** (`i` with `i + rope_dim/2`).
- Explains classic pattern: **token0 exact** (θ=0 identity), **token1+ corrupt**.
- Fixed in `b5bc7a8` / PR #235; that commit **is an ancestor of current `main` HEAD**.
- If live ring is on current main, RoPE alone does **not** close today’s bug.

### 2. Dense FP8 prepared MLP path — real, then fixed after #247

Original gap (correct at the time of PR #250 draft):

- Prepared dense path required `linear_plan->algorithm` as external scaled-GEMM backend.
- PP13 bind set `custom_launch_function` but did **not** bind that backend.
- #247 fail-closed too hard (`dense_mlp_fp8_gate status=1` observed live on rank0).

Then fixed:

| PR | Change |
|----|--------|
| **#248** | Route required FP8 dense MLP through valid quantized plans (`LaunchPreboundLinearPlan` on gate/up/down + SiluMul + residual) when prepared staging is unavailable. Forbid BF16 weight fallback for PREBOUND mode. |
| **#249** | Classify absent external scaled-GEMM backend (`algorithm == 0`) as **reroutable** (`NOT_FOUND`), not fatal mid-prepared path. |

Post-#249 probe: fallback probe slots **`fbgate/fbinter/fbdown = 0`** → old varying BF16 fallback is **not** where compute goes anymore.

### 3. PR #250 itself

- Useful as a **historical / structural** writeup and deploy checklist.
- **Stale** where it said dense FP8 was “still broken” — that specific hole is closed by #248/#249.
- **Must not** be read as “root cause found and fixed for current ring.”

---

## Open problem (what remains)

After dense-path fixes landed and the fallback probe went quiet:

```text
Identical greedy prompts (temp=0), three runs → token ids:
  17
  2064
  18300
```

So:

- End-to-end path **runs**.
- Output is **still wrong and nondeterministic**.
- Remaining bug is **not** “still on the dense BF16 fallback” (probes disagree).
- Remaining work is a **new** localization: which stage/layer/kernel/buffer still varies run-to-run with slots drained and dense on quantized plans.

Prior nondeterminism notes that may still matter (not closed by #248/#249):

- Prefill / mid-ring drain races (PRs #240–#241) — intended fix; re-verify on the release that produced `17/2064/18300`.
- Layer2-local divergence receipts in `docs/GLM52_KV_SLOT_PROBE_20260709.md` (from before dense reroute).
- Graph-capture / shared workspace / KV write visibility beyond slot mapping.
- Final epilogue / rank-final token selection (full-vocab path exists; still verify determinism).

This document does **not** claim a single remaining root cause. It claims the previous two are historical/fixed and the **open residual is nondeterministic greedy tokens after #249**.

---

## Deploy / interpretation checklist

1. Live release SHA must be **at or after** `40dc600` (merge of #249) or later including #250 doc-only if present.
2. Confirm RoPE half-split is in the binary (ancestor of `b5bc7a8`).
3. Confirm dense path: no `dense_mlp_fp8_gate status=1` hard fail; fallback probe slots stay zero if `SPARKPIPE_FP8_AMAX_PROBE=1`.
4. **Pass bar for “fixed”:** identical greedy prompts → **identical** token ids (and preferably match a trusted oracle).
5. Until (4), treat ring correctness as **open**.

---

## Historical evidence (still useful)

### Matched prompt

```text
Say OK. OK.
tokens: 45494 10397 13 10397 13
```

### Diagnostics

```text
diagnostics/glm52_pp13_diff_20260708/
diagnostics/glm52_pp13_diff_20260709/
diagnostics/glm52_pp13_diff_20260709_release231_seq1/numeric_diff.txt
docs/GLM52_FP8_PP13_ATTENTION_MODE_ROOT_CAUSE_20260709.md
docs/GLM52_KV_SLOT_PROBE_20260709.md
```

### Related merges

```text
b5bc7a8 / PR #235  RoPE half-split
2c2f2df / PR #222  tiled attention default
1b1bf45 / PR #240  prefill drain
c03f80d / PR #241  mid-ring drain
417c6a7 / PR #230  serial prefill positions
bf94fc6 / PR #238  full-vocab greedy epilogue
42e6b32 / PR #247  dense plan gate (incomplete alone)
8cfb47d / PR #248  route FP8 dense through quantized plans
9585e9b / PR #249  absent scaled-GEMM backend reroutable
dd953ce / PR #250  first handoff doc (superseded claims in this update)
```

### Key sources

```text
modules/glm52_resident_decode_stage/source/spark_glm52_sm121_required_decode_stage.cu
modules/glm52_resident_decode_stage/source/spark_glm52_pp13_node_context_builder_cuda.cu
modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_linear_plan.cu
```

---

## Suggested blurb for a teammate

> Updated handoff: `docs/GLM52_PP13_WRONG_TOKENS_ROOT_CAUSE_20260710.md`.
> RoPE (#235) and dense FP8 path (#248/#249) are real fixes but **not** the full current story.
> After #249, ring still produces nondeterministic greedy tokens (`17` / `2064` / `18300` on three identical prompts). Fallback dense probes are zero. Remaining issue is open.

---

## How this file should be used

- **Yes:** onboarding, “what we already fixed,” avoid re-fixing RoPE / dense BF16 fallback.
- **No:** closing the wrong-token bug as done.
- **Next correctness work:** localize **run-to-run divergence after #249**, not re-argue RoPE or the prepared-backend hole.
