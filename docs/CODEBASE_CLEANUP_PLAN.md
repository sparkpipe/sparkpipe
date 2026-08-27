# Codebase Cleanup Plan — pre-parallel-development hardening

Written 2026-08-28 as 8 agents begin parallel development. The goal: no
merge conflicts, no duplicated logic, no silent fallbacks, no broken gates.

## Current state

### Duplication (measured, not estimated)

| Component | Duplicated across | Identical lines | Status |
|---|---|---|---|
| Serving adapter lifecycle | dsv4, qwen27b, glm52 | ~250 lines shared between each pair | **OPEN** — the interface is identical, the implementation differs only in family constants |
| Packer core | 6 packers | ~379 lines (qwen pair measured) | **PARTIALLY DONE** — spark_pack_common.py extracted, per-family packers still carry dead weight |
| TP collective config | qwen27b, dsv4, glm52, qwen4flash, qwen38max | ~500 lines per family | **OPEN** — same env parsing, same collective init, pasted per family |
| Stagepack format/verifier | 5 families | structurally identical | **OPEN** — magic/version/entry/directory logic is the same pattern |
| JSON config loader | dsv4, qwen27b, glm52 | ~80 lines each | **OPEN** — XServingJsonMember, XServingJsonUnsigned, XServingLoadConfiguration |
| Validation harness shape | all families | kernel-tier + module-tier pattern | **STRUCTURAL** — sharing is possible but the per-family independence is deliberate (control vs candidate must be independent) |
| Model header generation | dsv4 (generated), qwen4flash (hand), others | same #define shape | **OPEN** — should all be generated from contracts |

### Cyclomatic complexity (from the external audit, commit c635ee8)

| Metric | Result |
|---|---|
| Production functions | 3,330 |
| Mean / median CCN | 7.33 / 5 |
| P90 / P95 / P99 | 16 / 23 / 41 |
| Maximum CCN | 157 (DSpark block forward, 625 lines) |
| Functions CCN > 50 | 17 (0.5%) |

The complexity is concentrated in model-driver state machines and
initialization/cleanup paths. The worst functions are in the exact code
where 8 agents will be working.

### Non-DRY patterns (from the audits)

1. **Silent fallbacks** — partially fixed (per-row loop → hard fail);
   remaining: adapter declines, module init fallbacks, API mismatches
2. **Debug fprintf in hot paths** — production stderr writes in per-frame
   decode paths
3. **getenv() tuning knobs in hot paths** — DFlash2 env vars re-read
   per-frame
4. **No LICENSE** — needed before open-sourcing

## Cleanup priorities (before agents start modifying shared code)

### Priority 1: Serving adapter template (blocks all families)
Extract the shared adapter lifecycle into a parameterized template:
- `runtime/serving_adapter_template.c` — the lifecycle (Initialize,
  Destroy, ValidateConfiguration, Submit, Progress, Snapshot) as macros
  or inline functions parameterized by family constants
- Each family's serving_adapter.c shrinks from 1000-2400 lines to ~200
  lines of constants + family-specific submit logic
- **This is the single biggest DRY win**: eliminates ~3500 lines of
  pasted lifecycle code across 5 families

### Priority 2: TP collective config module (blocks TP work)
Extract `ServingLoadTpAlgorithms` + `InitializeTpCollective` (~500 lines
pasted per family) into `runtime/tp_collective_config.c`. One
implementation, parameterized by collective identifier and algorithm mask.

### Priority 3: Stagepack format/verifier library
One implementation of the pack magic/version/directory verification
pattern, parameterized by family magic and geometry constants.

### Priority 4: JSON config loader
The XServingJsonMember/Unsigned/LoadConfiguration trio (~80 lines × 3
families) → one shared implementation in the runtime.

### Priority 5: Hot-path hygiene
- Move getenv() calls to module init (read once, store in state)
- Guard debug fprintf behind a state->debug_enabled flag
- Both are per-family but follow the same pattern

### What NOT to consolidate (deliberate independence)
- Per-family kernel code (GDN, MLA, MoE — genuinely different math)
- Validation harness control vs candidate paths (must be independent)
- Per-family test files (each family's tests are its own regression gate)

## Merge protocol for 8 parallel agents

1. Each agent works on a branch in its own worktree — no shared working tree
2. Agents submit PRs to main; the coordinator (me) reviews and merges
3. Integration requests (changes to shared files) go through the
   coordinator — agents do not modify files outside their write set
4. Shared-file changes are serialized: one at a time, never concurrent
5. The code-size ratchet enforces that each merge stays within bounds
6. `make test` must pass before merge — I run it, not the agent (agents
   don't have the full build environment on the mac)

## Cyclomatic complexity ratchet

Following the external audit's recommendation:
- Changed production functions target CCN ≤ 15
- New functions above CCN 20 require justification in the commit
- Newly introduced functions above CCN 40 rejected unless there is a
  measured CUDA-performance exception
- The existing 17 functions above CCN 50 get split opportunistically
  when their family's agent touches them (not a dedicated refactor)

## Priority 6: Sharded pack format (blocks Qwen Max, benefits all)

The current module loads FULL-WIDTH packs — every TP rank loads the same
complete file and slices at runtime. This is wrong for large models:

| Model | Full-width pack | Per-rank (TP16) | Fits 119 GB? |
|---|---|---|---|
| 27B FP8 | 28.5 GB | 1.78 GB | ✓ (even TP4: 7.1 GB) |
| DSV4 Flash | 149 GB | 9.3 GB | ✓ |
| K3 MXFP4 | 390 GB (stage) | 24 GB (TP16) | ✓ |
| **Qwen Max FP8** | **576-606 GB (stage)** | **36-38 GB** | **✓ if sharded** |

The fix: add a TP shard axis to the pack wire format.
- Pack header gains `tp_rank` and `tp_degree` fields
- Packer produces per-rank files (each contains only that rank's slice of
  every tensor: its share of heads, experts, FFN columns)
- Loader validates the sharded shape (not full-width)
- Kernels index from rank-local offsets (already partially done — the TP
  kernels slice from resident buffers; they just need the buffers to be
  rank-local instead of full-width copies)

### Variable expert bit widths (part of the platform plan)

The pack format already has codec IDs (format 6 = FP8 E4M3, format 4 =
MXFP4). The loader needs to accept all of them:

| Bit width | Codec | Use case | Bytes/param | Qwen Max size |
|---|---|---|---|---|
| 8-bit | F8_E4M3 + scale_inv | maximum quality | ~1.0 | 2.3 TB |
| 6-bit | MXFP6 | balanced | ~0.75 | 1.7 TB |
| 4-bit | MXFP4 / NVFP4 | maximum capacity | ~0.5 | 1.15 TB |

For Qwen Max at 4-bit experts: ~1.15 TB / 16 ranks = 72 GB per rank.
Fits 119 GB nodes with room for KV cache and activations.

The quantization is an offline repack step (packer-side), not a runtime
change. The same kernels execute; they just read narrower payloads with
different scale layouts.
