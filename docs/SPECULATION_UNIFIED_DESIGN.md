# Unified speculation subsystem — design contract

This is the binding reference for the parallel implementation. Every change must satisfy it.
It exists so ten agents produce one coherent subsystem, not ten pastes.

## Operator rules (apply to every line)

- DRY at every level: shared logic becomes a parameterized function, not a copy.
- Less code beats more code; maximize Solutions/(Codesize^2). Test code is free.
- No backward compatibility to a nonexistent installed base; delete dead paths.
- No `#ifndef`-supplies-a-default and no silent env fallbacks: a missing required input fails
  loudly. `#ifndef ... #error` (fail at compile time) is the allowed form. No config-multiplying
  `#if` ladders.
- No magic numbers: named `#define`s.
- No comments in code. Rationale goes in the PR/commit.
- Cyclomatic complexity down, not sideways.
- Speculators are controllable, never disabled. Every built speculation path is individually
  selectable (explicit config/env: exactly `0` or `1`, anything else fails loudly) and defaults to
  on. A hard disable removes the path from test coverage — forbidden.

## The three layers (what is common vs per-model)

Layer 1 — provider slot (`include/sparkpipe/spark_speculation_provider.h` +
`runtime/speculation_provider.c`). The lifecycle boundary. Unchanged in shape. A model supplies a
draft function behind it. This is the ONLY thing the serving adapter sees.

Layer 2 — acceptance/resolve engine (`include/sparkpipe/spark_speculation_policy.h` +
`src/spark_speculation_policy.c`). Model-neutral. Owns: the per-sequence speculation state, the
draft-request lifecycle, the ONE acceptance accounting (longest accepted path + bonus token), and
the verify/commit counters. It does not know MTP from DFlash from DSpark.

Layer 3 — verify/commit, model-side. The model's wave machinery runs the verify frame; the model
implements state commit/rollback for its own state kinds. The common part is the resolution output
and the accounting; the per-model part is the state fold/rollback.

## Per-model customization — exactly three things

1. A geometry/contract descriptor filled at runtime (token budget, vocab, dtype, depth envelope,
   tap/anchor semantics). The hidden-tap specification (tap count, which layers, dtype, position
   relation) is part of this descriptor, supplied per drafter from its contract — never hardcoded
   in common or target code. This replaces the `SPARK_DSPARK_TARGET_*` build switch and the
   hardcoded-GLM52 validation. Validation becomes structural (ranges, coherence), never
   hard-equality to one model's constants.
2. A draft function: committed prefix in, candidate token ids (+ optional parent indices and
   confidence) out.
3. A state fold/rollback implementation for the model's recurrent/paged state.

Nothing else is per-model. If a fourth thing appears, it is a design smell — surface it.

## Acceptance semantics (the ONE accounting)

Greedy/deterministic first. Input: the proposed candidates (chain = implicit parent chain; tree =
explicit parent array) and the verifier's per-row chosen token ids. Output: the longest accepted
path from the root, the committed token count (accepted + 1 bonus), and the fallback token id.
Sampled acceptance is a later, separately-proved contract and is not built now.

Chain is the degenerate tree (parent[i] = i-1). The engine stores parents; a chain caller just
omits the array. Tree verification adds a parent-array walk, no second engine.

## What dies / what stays

- STAYS: provider-slot ABI v1; the verify kernels (`inference/kernels/speculate.cuh`); the
  shape-injected tree machinery (`spark_speculation_tree.h`, already fail-loud); the ReplaySSM fold
  (`LmReplayFoldKernel`, chain-shaped).
- DIES: the GLM52-shape hardcoding in the policy core; the dead-code paths in
  `spark_speculation_policy.c`; the per-family provider-shim pastes (k3, qwen38_max) once the common
  lifecycle covers them; the `SparkSpeculationDraftRequest` collision (Phase 0).

## glm5_next (GLM 5.3 Flash) — the first served model

Chain speculation via the MTP layer (45). The multi-row run machinery is the verify frame. Commit
flag on the wave (not hardwired 1). Rollback: KV page transaction (exists), KDA delta-state
ReplaySSM fold, conv-window stash+re-conv (new, tiny), DSA indexer rewind, MTP-layer draft KV. HC
streams need nothing. B1 greedy token-parity vs baseline decode is the gate before any timing.

## Migration of existing models (later, after glm5_next proves the seam)

dsv4 (the only live speculation) replaces its hand-rolled acceptance loop with the engine. glm52
gets a drafter only when its weights exist. k3/qwen38_max fail-closed shims become real when their
drafters land. Each migration is one small change: supply the three customization points.
