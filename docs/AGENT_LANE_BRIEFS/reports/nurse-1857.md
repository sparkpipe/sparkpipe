# Nurse sweep 1 — 2026-08-29 18:52-18:58 UTC (lane/nurse)

Scope: process-table liveness for remote pipelines + local lane worktrees +
staging phase labels + reservations. Read-only sweep; no re-arms needed this
sweep, nothing killed, nothing touched. Sweep 1 of 9.

## VERDICT

Fleet ALIVE. One LOUD-WATCH: the spark5 **bf16 pack build is STILL RUNNING**
while its **validate and place chain already ran** — validate rc=0 and the
first "placed" line are labels over mid-write files. Details below; the
pipeline's own pass2 re-verify is the designed correction, expected to fire.

## 1. Remote detached processes (ssh spark5, BatchMode, 18:52Z)

### LOUD-WATCH: glm53full bf16 pipeline — build raced by validate+place

Process table truth (pgrep, 18:52Z):

- **ALIVE** `glm52_resident_stagepack.py --source /mnt/model-warm/glm-5.3-bf16
  --expert-codec bf16 ...` pid 667294 (wrapper 667293), running since ~18:35Z.
- **ALIVE** validate+place chain pid 672489/672490 (started ~18:48Z): 16-pack
  `validate_pack.py` (xargs -P4) -> `place_packs.sh --bytes 98019368448
  --prefix glm53full.bf16.tp16` pass1 -> pass2. Place pass1 LIVE at sweep time
  (rsync rank0 -> spark0 in flight).

The race, with numbers:

- Packs on disk GROWING between polls: 52,652,813,312 B/rank at ~18:50Z ->
  60,404,452,864 B/rank at 18:52Z (~2.4 GB/min aggregate-write; target per the
  invocation = 98,019,368,448 B/rank). ETA at measured rate ~19:08Z.
- `glm53full_bf16_validate.log` already ends **"VALIDATE-DONE rc=0"**
  (18:48Z) — i.e. validation "passed" while every pack was ~55% of its target
  bytes. Structural validation over preallocated mid-write files is NOT
  content proof; **treat the bf16 validate rc=0 as VOID until re-run after the
  build process exits.**
- `glm53full_bf16_place.log` already says **"rank 0 -> spark0: placed"**
  (18:51Z) — that is a truncated copy of a growing file. With
  `--bytes 98019368448`, pass2 must re-push every rank placed undersized
  (rank0 confirmed; any rank placed before build completion). Expected cost:
  one wasted ~98G fabric push per early rank; expected outcome: pass2 lands
  byte-correct packs. **Coordinator-visible check: PLACE-PASS2-DONE in
  glm53full_bf16_place.log with all 16 ranks at exactly 98,019,368,448 B.**
- `glm53full_bf16_packbuild.log` is 0 bytes since 18:35Z — python stdout
  buffering, NOT a dead process (pids + growing pack files are the truth).
  Stated to prevent a repeat of the stale-label incident in reverse.

Nurse action: NONE (pack/place/validate are lane-owned; nurse re-arms are
FETCH/VERIFY only; never kill). Will re-check every sweep: build exit, pass2
byte proof, re-validation.

### Completed pipelines (one line each, log-line verified)

- fp8 fetch: `glm53full_fp8_fetch5.log` = "published repo=zai-org/GLM-5.3 ...
  destination=/mnt/model-warm/glm-5.3-fp8 bytes=755663688511" (23:42 KST).
  Earlier fp8 fetch/2/3 logs end in the known WAN-era RuntimeErrors —
  superseded by fetch5, no live fetcher. HEALTHY.
- fp8 pack/place/freeze: packbuild last line = rank15 written
  (00:34 KST), validate VALIDATE-DONE rc=0 (00:54), place
  PLACE-PASS2-DONE rc=0 (01:22), refreeze2 REFREEZE-DONE rc=0 (01:25).
  COMPLETE chain, no live process expected. HEALTHY.
- bf16 fetch: `glm53full_bf16_fetch3.log` = "published repo=zai-org/GLM-5.3-BF16
  revision=304b8051... destination=/mnt/model-warm/glm-5.3-bf16" (03:35 KST =
  18:35Z). No live fetcher. HEALTHY (status.json below confirms).
- No fetcher/verifier processes alive anywhere on spark5 (pgrep
  hf_fetch_resume = none) — correct: all fetches published.

### Disk

spark5 `/`: 880G free (76% used) vs ~616G still to write for the bf16 set
(952G written of ~1,568G target; fp8+nvfp4 local packs already freed per the
lane plan, 57M residue each). Build fits with ~264G margin. No wedge risk.
/mnt/model-warm: 40T free.

## 2. Phase-label rule: 25/25 .staging/*.status.json CLEAN

All 25 `/mnt/model-warm/.staging/*.status.json`: phase=published AND
destination dir PRESENT = healthy end-state, zero zombies, no re-arms owed.
The two live-relevant ones: `glm-5.3-bf16` published updated
2026-08-29T18:35:17Z; `glm-5.3-fp8` published updated 14:42:47Z (both with
resumed_by: tools/glm53full_hf_fetch_resume.py). Nothing sitting in
downloading/verifying with a dead process.

## 3. Local lane worktrees (61 in /tmp; full state in nurse state, deltas here)

origin refs refreshed by fetch first (many `lane/*` remote branches had been
pruned post-merge — pre-fetch "pushed" states were stale).

### ACTIVE lanes (quiet < 60 min) — all healthy WIP, none flagged

| lane | quiet | dirty | state |
|---|---|---|---|
| glm5dsa | 0m | 6 | WIP, unpushed (normal) |
| p1d2 | 0m | 6 | WIP (last commit 19m ago) |
| debts | 1m | 2 | WIP (last commit 12m ago) |
| toksidecar | 3m | 13 | WIP (last commit 47m ago — watch if quiet>60m next sweeps) |
| contbatch2 | 3m | 3 | WIP (last commit 29m ago) |
| k3finish | 6m | 0 | pushed (k3-finish-2026-08-30b.md) |
| cellrunner | 7m | 0 | WIP unpushed (last commit 12m ago) |
| cfgaudit | 16m | 0 | pushed |
| w3weightd | 33m | 0 | pushed |
| glm5attractor | 40m | 0 | pushed |
| r3flash | 44m | 0 | pushed |
| jikvc5 | 51m | 0 | pushed |
| nurse (self) | — | — | this report |

No lane is quiet >60min with uncommitted changes => no mid-edit death
suspects this sweep.

### DEAD lanes (quiet > 120 min) — 48 worktrees, coordinator respawns; nurse only reports

- 45 of them: tips verified ancestors of origin/main (work landed; remote
  branches pruned after merge). No at-risk bytes. (Full list available; e.g.
  contbatch 44h, glm52 44h, k3 41h, staging 26h, g53full 7.3h — glm53full's
  remote pipeline above IS its live work even though the local worktree is
  quiet 439m; last report glm53full-2026-08-28.md.)
- Informational (NOT in origin/main, branch gone or unpushed):
  - `lane-drywave1`: DETACHED HEAD, **dirty=8**, quiet 663m (11h), tip not in
    main, last report redgates-2026-08-28.md. Only lane with meaningful
    uncommitted+unlanded bytes at death.
  - `lane-dsv4flash`: clean, quiet 2448m (41h), tip not in main, last report
    dsv4flash-packs-2026-08-27.md (likely superseded by the dsv4pro/staging
    work — coordinator's call, nothing lost: clean tree, but unlanded).
  - `lane-qwen38max-shard`: dirty=1 + **1 unpushed local commit**, quiet
    2418m, branch still EXISTS on origin, last report
    qwen38max-shard-2026-08-28.md.
- `lane-g5kda`: quiet 803m (13.4h), dirty=2, merged — last report
  glm5-kda-2026-08-29.md.

## 4. Reservations (runs/reservations.json, report only)

All 16 nodes (spark0-9,a-f) held by `manual:lane-glm5attractor`, acquired
2026-08-29T18:11:32-33Z, TTL 90 min => **expire 19:41:32Z** (~43 min after
this sweep). Holder lane alive (quiet 40m) => legitimate live holding, within
TTL. No expired-TTL squatting observed this sweep. Note for the coordinator:
if the bf16 place pass2 (fabric pushes to spark0-f) collides with
glm5attractor's GPU work, the reservations are NOT the blocker (place is
disk-only rsync), no action needed from me.

## Actions taken this sweep

Read-only sweeps + this report. Zero re-arms (nothing qualified: no zombie
phase labels, no dead fetch/verify processes with unfinished work), zero
kills, zero writes outside docs/AGENT_LANE_BRIEFS/reports/nurse-*.md.

Next sweep ~19:05Z: re-check bf16 build exit (~ETA 19:08Z), pass2 byte proof,
re-validate question, toksidecar/p1d2/glm5dsa quiet progression.
