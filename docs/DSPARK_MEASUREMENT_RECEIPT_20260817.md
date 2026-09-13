# DSV4 Flash DSpark measurement hour — receipt (spark4-7, 2026-08-17)

Status: **partial** — no-spec baseline re-confirmed; DSpark blocked by a pack
format mismatch and two infrastructure failures. No commits/pushes. Fleet left
in the known-good no-spec lean state.

## (1) No-spec B1 baseline — CONFIRMED

Lean control driver `3d962820…` (bucket 1, no spec), TP4 on spark4-7, O128
gate (exact token hash `a9385d0b…`), 1 warmup + 3 measured runs:

| Run | decode tok/s | exact |
| ---: | ---: | ---: |
| 1 | 40.276 | yes |
| 2 | 40.141 | yes |
| 3 | 40.157 | yes |
| **Mean** | **40.191** | 3/3 |

This re-confirms the retained ~40.2-40.5 ladder (devcycle control 40.46,
`PERFORMANCE_STATUS.md:97-166` lean 40.4553). Spec state: **no-spec** (the
only state that initializes on this band today).

## (2) DSpark k-sweep 5/7/8/10 — BLOCKED (two independent blockers)

**Blocker A — pack format mismatch.** The bucket-8 DSpark spec driver (built
today from `c0a4e433`, `/tmp/devcycle-build-lean-dspark`,
`model_driver.so` sha `95f09489…`, module id `…h4096.l43.e256.k6.ga0731.b8.v4`)
fails at `adapter_initialize` on all four ranks with:

    dsv4_stage pack_geometry_mismatch field=format_version
    model_residentd initialize=validation_failed status=9 phase=adapter_initialize

The deployed 40 GB stage pack (`dsv4_flash_stage.spstage`, built 2026-08-14
for the lean b1 control) has an older `format_version` than the b8 spec
driver expects (the spec build adds DSpark draft-layer packing). The b1 driver
loads the same pack fine. **A re-pack is required; downloads are forbidden this
hour, so this cannot be closed now.**

**Blocker B — k is compile-time.** Only bucket 8 (= spec_step 7) is a valid
DSpark bucket: `SPARK_DSV4_MODEL_DSPARK_SPEC_STEP 7u`
(`spark_dsv4_model.h:43`) gates on `SPARK_BATCH_BUCKET == 8`, and the
bucket allowlist is {1,2,4,8,16,32,64,128,256,512,1024}
(`spark_dsv4_batch_tuning.h:30-35`). k=5/8/10 ⇒ bucket 6/9/11 are not in the
allowlist. A k-sweep needs source edits (SPEC_STEP + allowlist + 4 rebuilds),
not a flag. Only **k=7** was even buildable, and it is blocked by Blocker A.

## (3) Probabilistic drafting on/off — NOT REACHED

The verify path is greedy Leviathan (`module.c:3582-3586`); the sampled
verifier kernel exists (`inference/kernels/speculate.cuh:78`, V-02) but is not
wired. Not measurable this hour because the spec driver does not initialize
(Blocker A), and switching requires a source change + rebuild.

## (4) Warmup-depth protocol — NOT MEASURED (spec path down)

The devcycle harness does 1 warmup; a deeper-warmup delta is only meaningful
once the spec path initializes. (No-spec warmup-depth was not exercised this
hour — time was consumed by the infra blockers.)

## Infrastructure failures found (both need coordinator action)

1. **fleet_swap.sh (systemd) launch is broken.** `systemctl start
   sparkpipe_model_residentd` fails immediately with
   `driver_load_error status=13 phase=adapter_initialize` on all ranks
   (3 restart attempts). Root cause not fully isolated: the unit's
   `Environment=LD_LIBRARY_PATH=${RUNTIME_ROOT}/lib` is not expanded by
   systemd (confirmed via `systemctl show` → literal `\${RUNTIME_ROOT}`),
   but a drop-in override (`20-ldpath.conf` with the absolute path) did NOT
   resolve it — the dlopen error string is not surfaced by the residentd.
   The **setsid path (`tools/devcycle.sh`) works** and was used for the
   baseline. MemoryMax=108G/MemorySwapMax=0 are correctly in place but only
   apply under systemd, so the DSpark run could not be MemoryMax-guarded this
   hour.
2. **fleet_swap.sh hangs on the glm52 band.** `fleet_swap.sh dsv4-flash`
   tries to stop the current big (glm52) on spark8-f, which are unreachable
   (timeout), so the script never reaches the flash-band start. The
   "always-on" dsv4-flash entry is mis-scoped as a band-swappable model.

## Receipts / artifacts

- Baseline runs: `/tmp/devcycle-lean-o128-r{1,2,3}-*.json` on the MacBook
  (exact O128 token hash `a9385d0b…`).
- Spec driver (b8): `tools/devcycle/drivers/lean-dspark/` (staged locally,
  sha `95f09489…`); source `c0a4e433`.
- Rank logs: `/tmp/devcycle-dspark-rank*.log` (spark4-7) — the
  `pack_geometry_mismatch field=format_version` evidence.

## What is needed to close DSpark next hour

1. Re-pack the stage pack at the b8 format_version (DSpark draft layers packed)
   on spark4-7, or stage a b8-compatible pack (needs a download/pack step the
   coordinator must authorize).
2. Fix the systemd unit's LD_LIBRARY_PATH expansion (and re-test the dlopen) so
   the MemoryMax=108G guardrail applies to the spec run.
3. For the k-sweep: add buckets 6/9/11 + set SPEC_STEP per k + rebuild (4
   drivers), or add a runtime k override.
