# Speculation provider abstraction (operator directive 2026-08-29)

Operator: support ALL speculation types — not every model gets today's
best speculator. Current set: basic MTP (multi-token-predict heads),
DFlash, DSpark, DFlash2. Coming: DSpark2, likely more. The design
must abstract cleanly into common code WITHOUT sacrificing per-method
performance — the tension is real (each method's hot path wants
method-specific batching, KV handling, and verify kernels), and the
resolution is the same one the module ABI already uses: a narrow
capability interface with the hot loop staying provider-owned.

## What exists today (surveyed 2026-08-29)

- MTP: dsv4 (spec layers in checkpoint, DSPARK env-gated), glm52
  (MTP layer in module), qwen-flash (MTP-1 in pack flags), k3 (none).
- DFlash2: qwen38_27b (the launch-env contract: SPEC_METHOD=dflash2
  DRAFT_COUNT=8 ... — a family-LOCAL convention, not an interface).
- DSpark: dsv4 (SPARK_DSV4_DSPARK=1 gate after the requalification),
  glm52 (modules/glm52_dspark_draft_backend — the only separate
  backend MODULE, i.e. the only real precedent for provider-as-unit).
- DFlash: legacy path in the 27B family.
- The verify side lives in each family's runner (chain walk, lease
  advance — the lease-advance bug was exactly a verify-side
  provider-contract mismatch).

## The design: provider as a capability unit behind the adapter

Mirror the serving-adapter pattern that already works:

```
SparkSpeculationProvider {
    kind: MTP | DFLASH | DSPARK | DFLASH2 | DSPARK2 | ...
    capability query:   supports(model_geometry) -> yes/no/why
    draft:              hidden+state -> draft tokens+scores (provider
                        owns batching, KV taps, its own graphs)
    verify contract:    accepted_token_count semantics, chain width,
                        tokens_per_sequence reporting (the lease-bug
                        class dies here — ONE implementation)
    KV interaction:     which frames the provider reads/writes, the
                        scratch-vs-tail split, block-KV history shape
    env/config schema:  ONE canonical launch contract (today's
                        SPEC_METHOD/DRAFT_COUNT sprawl is per-family)
}
```

PERFORMANCE RULE (the operator's expressed concern): the interface
covers LIFECYCLE + CONTRACT, never the inner loop. Draft generation
and verification kernels stay provider-owned .cu — the abstraction
buys dispatch, config, KV contract, and verify accounting, and costs
zero hot-path indirection (same as the module ABI's design).

## Why this is the right moment

The housecleaning sprint's W2 (adapter template) lands the shared
serving lifecycle NOW; folding the provider slot into that template
means every family gets speculation-as-capability for free, and the
auditor-visible sprawl (per-family launch envs, per-family verify
accounting — where the lease bug lived twice) consolidates into one
place. DSpark2, when it arrives, is a new provider module + a kind
enum entry — not five family edits.

## Sequencing

1. W2 lands the adapter template (running).
2. Extract the provider interface from the TWO existing clean units
   (glm52_dspark_draft_backend as module-provider; the 27B's dflash2
   as embedded-provider) — both shapes must fit.
3. Migrate family-by-family with cell-unchanged gates (the DFlash2
   env contract becomes the canonical schema last — it has the most
   receipts to keep identical).
4. The recipe compiler emits the provider block per model (the
   contract already records 'speculation provider + verification
   contract' per MODEL_SUPPORT.md).
