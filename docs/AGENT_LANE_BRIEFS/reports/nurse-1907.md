# Nurse sweep 2 — 2026-08-29 19:05-19:07 UTC (lane/nurse)

Sweep 2 of 9. Read-only; no re-arms owed; nothing killed or touched.

## VERDICT

bf16 pack build FINISHED CLEAN (process exited, receipts written). Two
coordinator-visible findings carry over: (a) the chain's `--bytes` spec is
wrong by 86,528 B so the place log can never show the "already placed x16"
proof and pass2 will re-push all 16 ranks (~1.5 TB avoidable fabric traffic —
but that re-push also repairs the truncated early ranks); (b) the
VALIDATE-DONE rc=0 in the log is VOID (ran mid-write) and no re-validation of
the final packs is scheduled in the chain.

## 1. Remote spark5 (19:05Z)

- bf16 packbuild pid 667294 **EXITED normally**; `glm53full_bf16_packbuild.log`
  0 -> 2550 B, last line "glm52sp written: ...rank15... bytes=98019454976"
  (04:02:43 KST = 19:02:43Z). All 16 packs now exactly **98,019,454,976 B**
  with 3.7 MB `.receipt.json` each, mtimes 04:02 KST. HEALTHY completion.
- **BYTE-SPEC MISMATCH (LOUD)**: the validate+place chain runs with
  `--bytes 98019368448` (the lane report's mock-plan number) but the actual
  pack size is **98019454976** — delta **86,528 B** (identical delta as the
  fp8 plan-vs-actual; fp8's live run passed the TRUE size, hence its clean
  pass2). Place test is exact `stat -c%s == bytes`
  (tools/glm53full_place_packs.sh:63), so:
  - NO rank can ever test equal => pass1 (running: ranks 0-5 "placed" by
    04:05 KST) pushes everything, and **pass2 will re-push all 16 again**
    (~1.5 TB avoidable) and end `PLACE-PASS2-DONE rc=0` WITHOUT the
    "already placed" lines that are the script's designed size proof. Anyone
    reading only the final line will over-trust it.
  - Silver lining: pass1's early ranks were placed from INCOMPLETE files
    (chain started 03:48Z, build finished 19:02:43Z; spark0-3 copies
    truncated). Because every target fails the wrong spec, pass2 re-pushes
    ALL ranks from the now-complete sources => final placed bytes should be
    correct. Fleet truth check (target size == 98019454976 on spark0-f) is
    what I will verify on later sweeps, not the log's rc=0.
- validate.log unchanged since the VOID 03:48Z rc=0 (see sweep 1). The chain
  does NOT re-validate post-build. Coordinator/lane should re-run the 16
  validators (~20 min) before serving trust is placed in them; receipts
  (written by the packer at completion) are present regardless.
- fp8 chain: unchanged, COMPLETE end-to-end (fetch5 published / pass2 all
  "already placed" / REFREEZE-DONE). HEALTHY.
- Staging labels: 25/25 status.json phase=published, destinations PRESENT.
  No zombies. No fetch/verify processes alive (correct — nothing left to run).
- Disk: build complete; no pressure change expected (880G was free pre-build
  with ~616G remaining — final margin ~264G).

## 2. Local lanes (61 worktrees, refs refreshed)

- ACTIVE (quiet<60m), all healthy: contbatch2 (quiet 0m, dirty 9, NEW report
  contbatch2-2026-08-29.md pushed 1m ago), cellrunner (0m, dirty 1, new commit
  5m ago), debts (1m, dirty 8 — grew 2->8, working), glm5dsa (2m, dirty 13,
  grew 6->13), toksidecar (4m, dirty 13 — **last commit 59m ago; if it goes
  quiet >60m with dirty=13 that becomes a mid-edit flag; watching**),
  p1d2 (11m, dirty 6), nurse self, k3finish (17m, pushed), cfgaudit (27m,
  pushed), w3weightd (45m, pushed), glm5attractor (52m, pushed), r3flash
  (55m, pushed).
- jikvc5 quiet 63m: clean + pushed => idle-between-turns, NOT flagged
  (flag rule is quiet>60m WITH uncommitted changes).
- No lane quiet>60m with dirty. Dead-lane set unchanged from sweep 1
  (drywave1/dsv4flash/qwen38max-shard remain the three informational
  not-in-main cases).

## 3. Reservations

Unchanged: all 16 nodes `manual:lane-glm5attractor`, expire 19:41:32Z (~34
min after this sweep). Holder alive. Report only; sweep 4 will verify release
or renewal.

## Actions taken

Read-only + this report. Zero re-arms, zero kills.

Next sweep ~19:18Z: place pass1 progress; re-check toksidecar; verify no new
zombie labels.
