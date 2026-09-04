# glm5_next multi-row prefill — handoff to glm flash dev (2026-09-01)

Branch: `lane/glm5next-mtp-accept`. Validator GREEN; serving has a separate, well-localized
transport-plane bug. This note is the full state so the lane resumes without re-deriving it.

## Fixed and verified (validator, TP1, spark0)

Three commits, all validated end-to-end (`glm5_next validator: PASS (0 failures)`, tiers
1/2a/3/4a/4b/4c all bit-exact):

- `0a454a3` — host staging-array overflow + deterministic `context_lengths`. The validator
  fixture's `host_{resident_slots,positions,token_ids}_stage` were `[1]` but `BuildRunWave` writes
  0..7, corrupting adjacent fields (the original "moving divergence band" was this, host-side, not
  a device OOB write). And `SparkGlm5NextWaveMetadataKernel` had a multi-writer race on
  `context_lengths[slot]` when rows share a slot; now one thread writes the deterministic max.
- `4438cc6` — the tier4 metadata capture read 16 u32 from an 8-element `positions` buffer
  (`meta_p1_copy` fail) and captured both passes after pass 2; capture pass-1 after pass 1, sized to
  `SPARK_GLM5_NEXT_VALIDATION_RUN_TOKENS`.
- The `SPARK_GLM5_NEXT_VALIDATION_RUN_TOKENS` macro is defined before the fixture struct.

So the multi-row run machinery (run-aware KDA recurrence, slot-keyed state, per-row causal DSA) is
proven bit-exact at run-of-1/2/8 through KDA layer 0 and DSA layer 3. That proof holds.

## The remaining bug is serving-only and is NOT module numerics

Controlled A/B on the fleet (same residentd/adapter, ONLY `lib/model_driver.so` swapped, fresh
fleet, first request, temperature 0): the rolled-back 1-row driver reproduces the verified
reference tokens exactly; the multi-row driver is deterministic (3/3 identical) but diverges at
token 0. Threshold is sharp:

- prompt rows 1 and 2: **bit-identical** to the 1-row reference.
- prompt rows >= 3: **diverge** at the first generated token.

## The decisive clue: a host-side per-layer sync MASKS it

Same binary, same fleet:

| run | tokens at prompt n=3 |
|---|---|
| plain multi-row | wrong (`923 76 22940 ...`) |
| `SPARK_GLM5_NEXT_LAYERDUMP=1` (rank 0: `cudaStreamSynchronize` + ~96 KB D2H read of `hidden_bf16` per layer) | **correct** (`11 5761 23716 406 ...` = reference) |
| `SPARK_GLM5_NEXT_SYNC_EVERY=1` (bare sync, all ranks) | wrong |
| `SPARK_GLM5_NEXT_SYNC_EVERY=1` rank-0-only bare sync | wrong |
| `SPARK_GLM5_NEXT_SYNC_EVERY=4` | wrong |
| `SPARK_GLM5_NEXT_PROBE=1` (sync + 1 KB read, all ranks) | wrong |
| `NCCL_PROTO=LL` | wrong |

Only the LARGE rank-0 D2H read per layer fixes it — a bare sync (any rank set) does not. This is a
timing/coherence race, not a stream-ordering hole: the module is single-stream end-to-end and the
NCCL allreduce is issued ON `slot->stream`, so device order is already correct. The completion is
delivered INLINE at enqueue (`tp_device_collective_nccl.c:860-891`), so the host enqueues the whole
45-layer wave (~92 collectives across the two comms) in one burst; the NCCL proxy/NIC plane falls
behind and the multi-row payload (narrow reduce = rows x 8 KB; 2 rows = 16 KB verified, 3 rows =
24 KB) crosses into the unverified region. Rank 0 is `coordinator_rank_index`, so a heavy rank-0
delay re-locksteps the fleet's rendezvous.

Not the fix: payload-envelope alone. The wide HC reduce crosses 64 KB->96 KB at the 2->3 row cliff
(once per wave, at BEGIN), but a per-layer sync would not repair a BEGIN-time corruption and yet it
does repair the tokens — so the live race is in the per-layer narrow collectives under deep enqueue,
not the BEGIN reduce size.

## Untested knob already deployed

`SPARK_GLM5_NEXT_PACER_KB` (rank 0, per layer: sync + tunable D2H read of `hidden_bf16`, no
fprintf) was built, validated, and deployed (`driver 33b2f95d...`) but NOT yet run — it isolates
whether a minimal D2H fence (vs the LAYERDUMP's fprintf volume) is sufficient and how small it can
be. Test: `G5N_EXTRA_ENV="SPARK_GLM5_NEXT_SYNC_EVERY=1 SPARK_GLM5_NEXT_PACER_KB=96"` and read the
n=3 tokens; then shrink PACER_KB. If a small fence fixes it reliably, that is a candidate stopgap
— but it is a timing workaround, not a root fix.

## Diag knobs on this branch (temporary — strip before merge)

- `SPARK_GLM5_NEXT_FORCE_WAVE_ROWS=N` — wave-row clamp. NOTE: N=1 currently makes the residentd
  exit with `model status 4` on the request (the new chain's multi-wave path differs from the old
  walk-and-clamp; do not trust it as a 1-row reference until fixed).
- `SPARK_GLM5_NEXT_LAYERDUMP=1` — per-layer all-rows hidden checksum dump, rank 0 (also the sync
  that masks the bug).
- `SPARK_GLM5_NEXT_SYNC_EVERY=N` — per-N-layer host sync (see table).
- `SPARK_GLM5_NEXT_PACER_KB=N` — rank-0 D2H fence (untested).
- `tools/glm5_next_wave.sh` gained `G5N_EXTRA_ENV` passthrough to arm these on the residentd.

## Recommended next steps

1. Convict the transport with an isolated repro: a 16-rank RoCE allreduce microbenchmark issuing
   back-to-back ~24 KB collectives in a deep burst on the fleet's NCCL 2.28.9 build; compare sums
   against a paced reference. If it corrupts, the bug is in the NCCL build's flow control, not the
   module.
2. The proper fix is in the collective plane (make completion device/proxy-complete before the
   chain advances, or repair the NCCL flow control) — not the model module.
3. Re-verify serving token-equivalence to the 1-row reference at prompt rows 1,2,3,8,372 after the
   transport fix; the validator tiers stay green throughout.

## Dev-cycle infrastructure (working — keep)

weightd is now a persistent per-node daemon holding the 22 GB rank packs; residentd attaches in
~5-10 s warm vs minutes cold. The warm-path identity bug that silently defeated it is fixed
(`f0ed177`: publish the pack digest before the already-up early return in `node/weightd_spawn.c`).
The fleet's deployed `bin/sparkpipe_weightd` had been a macOS binary (that's why it always fell
back to direct load); rebuilt as Linux ELF and deployed 16/16. weightd is persistent; the queue
owns setup/teardown. The stale pack `.sha256` sidecars were regenerated after the Sep-1 repack.
