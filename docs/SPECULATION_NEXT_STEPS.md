# Speculation + cleanup — state and next steps (2026-09-02)

## Merged to main

- **PR #779 — cleanup.** Six consolidations (dsv4 config-matrix, qwen38 serving adapters,
  stagepack formats, env-var fail-loud discipline, qwen38 pack-loader spine, adapter headers,
  lifecycle migration) + the full comment purge (every authored `.c/.h/.cu/.cuh` stripped by a
  token-preserving lexer; preprocessed-identity-verified; full host test suite unchanged).
  Ratchet re-pinned to measured reality 237363 → 219005 (≈ −18k authored lines).
- **PR #782 — speculation foundation.** Header collision fixed (policy types renamed
  `SparkSpeculationPolicy*`); the policy core is now model-neutral (runtime contract descriptor,
  structural validation, no `SPARK_DSPARK_TARGET_*` build switch) and tree-capable (parent-array
  resolve; chain is the degenerate tree). No `glm52` names in general code. All tests green.

## Open

- **PR #783 — glm5_next MTP chain** (`lane/glm5next-spec`). Byte-exact parity gate green at TP1
  (`validate_mtp_parity`: tokens + KDA state + conv windows + KV + index all match baseline, across
  forced reject/mid/accept + organic). Opt-in `SPARK_GLM5_NEXT_MTP`, off by default, TP16 gated off.
  **Do not merge until** `lane/glm5next-mtp-accept` (the multi-row lane) is merged and the diag
  knobs (`SPARK_GLM5_NEXT_FORCE_WAVE_ROWS/LAYERDUMP/SYNC_EVERY/PACER_KB`) are stripped.

## Next steps, in order

1. **Merge the glm5_next multi-row lane** (the glm flash dev's). Its remaining blocker is the TP16
   NCCL transport race (deep 45-layer enqueue burst, proxy lag, ≥3 rows; masked by a heavy rank-0
   D2H read per layer). The handoff with the full evidence is
   `docs/AGENT_LANE_BRIEFS/reports/glm5next-multirow-handoff-2026-09-01.md` on that lane. Until it
   lands, glm5_next serving multi-row and the MTP fleet verification are blocked. The untested
   `PACER_KB` rank-0 D2H fence is deployed for them to try.
2. **Then merge PR #783** (glm5_next MTP). Strip the diag knobs first.
3. **Acceptance-rate measurement.** The MTP draft's attention is chain-local (its KV pool holds
   only the current K-step chain), so acceptance underestimates a full-context draft. Measure the
   real acceptance rate on a fixed corpus at TP1; only then claim a speedup. If acceptance is too
   low, build the MTP layer's full-prefix KV (run layer 45 over history) — that is the known gap.
4. **Fold/conv bit-exactness deep check.** The parity gate already diffs KDA state byte-exact; add
   a long-run sweep (many steps, varied accept lengths) to stress the ReplaySSM fold against serial
   decode before trusting it at scale.
5. **TP16 enablement for MTP.** The draft's TP collectives are deliberately not wired (init fails
   loudly at tp_degree>1). Wire `SparkGlm5NextMtpDraftOps` reduce callbacks and re-validate
   cross-rank draft-token identity. This is gated on the transport fix (1).
6. **Fleet verification.** B1 greedy token parity + tokens/s at TP16 with MTP on, once 1+5 land.
7. **Other speculators + composite tree.** The core is tree-capable; DFlash/DSpark remote drafters
   need the 5090 workstation + trained draft weights (do GLM 5.3 Flash draft weights exist? — check
   before building the remote path). Tree verification of a hybrid KDA model needs per-branch state
   lanes, not tree masks (the kimi-review caveat) — that is the hard part and is deliberately after
   MTP parity.

## Cleanup follow-ups (smaller, not urgent)

- The pack-loader second lineage (glm52/glm5_next share one) — consolidate when the spec work
  settles. glm5_next is mid-flight; do not touch it until the MTP work merges.
- dsv4 `spark_dsv4_batch_tuning.h` is a hand-pasted copy of the batch-variant ladder — adopt the
  common generator's bucket→sha macro when it exists.
- The Pro dsv4 serving-adapter target is broken on main (`SPARK_DSV4_MODEL_DSPARK_SPEC_STEP`
  undeclared under `SPARK_DSV4_PRO_BUILD`) — pre-existing, needs a model-config decision.
- `tests/cuda_stub` is missing a few error enums (`cudaErrorLaunchFailure` etc.) — pre-existing.

## Coordination notes

- The comment purge touches 553 files — any in-flight branch that edited them will conflict. Re-pin
  `tests/test_code_size.py` CEILING from measured reality after any conflict resolution (never
  carry an old ceiling forward).
- weightd is persistent; the queue owns setup/teardown. GPU work goes through the queue
  (priority 0 for glm5.3-family).
