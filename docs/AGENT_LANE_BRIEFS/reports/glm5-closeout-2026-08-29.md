# glm5-closeout — the fixed packs land on the fleet; GLM 5.3 Flash scoreboard cell

Lane: `lane/glm5-closeout` (worktree /tmp/lane-g5close), based on main
f63eedf (packer one-liner 4758e68). Fleet: spark0..sparkf, glm5_next.tp16
runtime roots, api spark0:8433. Every claim = command + raw output.
This session is the ONE wave owner (reservations: released the expired
lane-glm5rt / lane-glm5kda holds, reserved all 16 for `lane-glm5-closeout`,
TTL 360m, at session start).

## How this serves the scoreboard

The brief: land the fleet on the FIXED packs (the packer fused-shard fix,
4758e68), take the quality gate (COMPSEC-17) and the perf cell (M5
exact-32K B1 decode) — the GLM 5.3 Flash scoreboard number, per-spark
axis first per docs/GOALS.md's SOTA normalization.

## Pre-flight findings (commands + raw output)

1. Packer provenance per node — ranks 1-15 run the FIXED packer
   (sha256 of ~/g5rt2-src/tools/glm5_next_resident_stagepack.py):

       spark1..sparkf: 90b970b58ad14e52  == main 4758e68 (my worktree copy hashes the same)
       spark0:         3fbbb54eea4a433b  (1dac68b clean; rank 0's pack is the
                          earlier completed build — start=0 makes m[0:count]
                          accidentally right, and it independently verifies below)

2. The live driver changed under the lane before the swap: all 16 nodes hold
   lib/model_driver.so sha256-prefix 6ca5f16b8b4259c5 (2,290,016 B, mtime
   2026-08-29 12:50:56 UTC), deployed by the coordinator AFTER the
   LAUNCH-STATE.md receipt (0292a3e55f45b2fd — now stale). The live
   residentds started 13:29:51 UTC, i.e. ON this driver — so the
   BEFORE baseline below and the AFTER wave differ ONLY in packs.
   (LAUNCH-STATE.md updated at lane end.)

3. Reservations at session start were stale (lane-glm5rt 2026-08-28T22:28,
   TTL 480m; lane-glm5kda 02:00 TTL 240m — both expired); released and
   re-reserved 16/16 as lane-glm5-closeout.

## C1 — build monitor + per-rank verification

Monitor/controller: tools/glm5_next_pack_orchestrate.py (this branch),
packing tool: tools/glm5_next_pack_verify.py (committed here, shipped to
the nodes as a git bundle fetched into ~/g5rt2-src and `git archive`d to
/tmp/g5close-verify — never a hand-copied tools/ dir).

Verify checks per rank pack: header (magic/tp/file_bytes == rank-0 receipt
21,706,046,976), 1160 entries, layout bounds + payload byte-count rules,
FULL plan diff (the fixed packer's plan rebuilt against the checkpoint,
all 1160 entries field-by-field), spot payload round-trips regenerated
from the checkpoint through the packer's own produce closures
(default: L17 kda_qkv_beta, L17 kda_decay_gate_down, L0 dense up|gate,
embedding; --deep adds lm_head, L3 shared up|gate, L3 q_b, L3 288-expert
slab), plus the directory sha256 cross-rank invariant.

RANK 0 (--deep, spark0):

    VERIFY-PASS rank 0: glm5_next_stage.tp16.rank0.g5nsp 21706046976 bytes,
    1160 tensors, dir_sha 421ef0989c054f67
    (all spot rows PASS incl. kind=22 expert slab payload+scale)

RANKS 1-15: VERIFY-PASS with the SAME directory sha 421ef0989c054f67
(identical tensor inventory across all 16 ranks); --deep on ranks 1, 8
(and 15). Raw per-node logs: /tmp/g5verify-r<r>.log on each node.

Two tool bugs found and fixed during bring-up (both mine, both in the
verifier, committed separately): plan-vs-pack dict comparison must use the
shared field subset; expert slabs' produce streams payload+scale
interleaved, so the sized region (payload+scale) is the compare target.

Build anomalies: rank 4's builder was restarted from scratch by the
coordinator mid-lane (etime reset 1030s->307s, pack size 518MB->82MB) and
its ceph read rate crawled (~190KB/s vs ~4.5MB/s fleet average); tracked
for the brief's D-state>30min requeue rule. (Outcome section below.)

## C2 — the swap

tools/glm5_next_pack_swap.sh (this branch): the deployed pack entries are
SYMLINKS (packs/<name>.g5nsp -> ~/glm53_packs/<name>.g5nsp); the swap
renames the old symlink to .pre-closeout-bak (guarded: never overwritten
on re-run) and rename(2)s a new symlink to ~/glm53_packs_fixed/<name>.g5nsp
into place — atomic, nothing copied, old pack FILES untouched for rollback.
15/16 swapped at 06:2x UTC (rank 4 pending its rebuild); commands and
outputs appended in the transcript section.

## C3/C4/C5/C6 — (results appended as they land)

## INTEGRATION REQUEST

None new.
