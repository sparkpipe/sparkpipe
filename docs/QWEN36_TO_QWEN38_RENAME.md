# qwen36 → qwen38 rename plan (2026-08-25)

The model we serve is Qwen3.8-27B, not Qwen3.6. Every reference to
"qwen36" in the codebase is a deprecated-model name that confuses
everyone who reads the code. This document scopes the rename.

## The blocker: TWO modules exist

- `modules/qwen36_resident_decode_stage/` — 9+ files, OUR implementation
  (DFlash2 spec decode, prefix caching, WS GEMM, JIT-KV, all PRs)
- `modules/qwen38_resident_decode_stage/` — 4 files, an older/simpler
  variant that IS referenced in the Makefile

Both are alive. The qwen38 directory must be renamed or merged BEFORE
qwen36 can take the qwen38 name.

## Recommended approach

1. **Audit the existing qwen38 module** — determine if it's a subset of
   qwen36 (mergeable) or a genuinely different model variant (needs its
   own name). Check: does it share the same model header? Same stagepack
   format? Same adapter interface?

2. **If mergeable**: delete qwen38, rename qwen36 → qwen38
   **If different**: rename qwen38 → qwen38_legacy (or its actual model
   name), then rename qwen36 → qwen38

3. **Execute the rename** (in order, verifying build after each step):
   a. `git mv modules/qwen36_resident_decode_stage modules/qwen38_resident_decode_stage`
   b. `git mv model-families/qwen36 model-families/qwen38`
   c. Rename all files: `spark_qwen36_*` → `spark_qwen38_*`
   d. `sed -i 's/qwen36/qwen38/g; s/QWEN36/QWEN38/g; s/Qwen36/Qwen38/g'` on all files
   e. Update Makefile paths and target names
   f. Update test file names and references
   g. Update documentation
   h. Full rebuild + validation PASS + production verify

## Scope (measured 2026-08-25)

| Metric | Count |
|--------|-------|
| Files with qwen36 references | 169 |
| Directories to rename | 2 |
| Source files to rename | ~15 |
| Unique SPARK_QWEN36_* macros | 363 |
| SparkQwen36* symbols | ~100+ |
| String literals (logs, paths, configs) | ~500+ |

## Verification checklist

- [ ] Module builds and validates PASS at B8
- [ ] Full stack deploys (driver, adapter, residentd, batch, API)
- [ ] O512 benchmark: 512 tokens, d7f79880, ~21s
- [ ] API: completion request delivers tokens
- [ ] All tests pass
- [ ] Code-size gate passes (ceiling may need ratchet for rename-only delta)

## Timing

This is a 1-2 hour focused session with full context. DO NOT attempt
it piecemeal — a partial rename leaves the codebase in a mixed state
that's worse than either extreme.

## DRY analysis (updated 2026-08-25)

The two modules are NOT copy-paste duplicates — they implement DIFFERENT
models that share architectural patterns:

| | qwen36 module | qwen38 module |
|---|---|---|
| Model | Qwen 3.8-27B (dense hybrid) | Qwen 3.8 Max (MoE) |
| Lines (module.c) | 3,635 | 1,870 |
| Lines (adapter.c) | 2,408 | 1,248 |
| Features | DFlash2 spec, prefix cache, WS GEMM, JIT-KV, MTP | decode-only, PP, MoE |
| Stagepack | slice-validation, computed inventory | simpler layout |

After prefix normalization (QWEN38→QWEN36), the stagepack headers still
have 391 different lines out of ~500 — genuinely different evolution.

The DRY violation is at the PATTERN level (both implement the same
adapter contract, module ABI, stagepack concept, hidden transport), not
literal code duplication. The fix is RENAMING for clarity, not merging.

## Revised rename plan

Since qwen38 is a DIFFERENT model (Qwen 3.8 Max MoE), NOT a duplicate:

1. Rename `qwen38_resident_decode_stage` → `qwen38max_resident_decode_stage`
   (reflects its actual model: Qwen 3.8 Max)
2. Rename `qwen36_resident_decode_stage` → `qwen38_resident_decode_stage`
   (reflects our model: Qwen 3.8 27B)
3. Execute the full sed pass on all 169 files
4. Extract shared patterns into a common library as a follow-up
   (stagepack utilities, adapter helpers — reduces the pattern-level DRY)

The pattern-level DRY (shared adapter/module/stagepack abstractions) is
a separate refactoring from the rename. Do the rename first for
clarity, then extract common code.
