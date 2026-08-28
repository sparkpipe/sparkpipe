
## 2026-08-28 09:2x — sweep: spark3 root cause revised; qwen-flash merged; 27B bests improved

- INCIDENT REVISED: spark3 reboot #2 coincided with a ~104+ GiB GPU KV-pool
  probe next to mds+2xOSD on the same unified memory — noisy-neighbor OOM,
  not (necessarily) flaky hardware. Escalation updated (move MDS off GPU
  nodes; systemd memory guards). NEW BINDING RULE: storage-host nodes
  (spark3, sparkc) host NO GPU lane work; knee lane granted one-time
  exception at the verified 71.1 GiB envelope, one-B-per-session.
- MERGED lane/qwen-flash (7c28c72): TP4 rank packs deployed+verified on
  spark4-7, M5 kernel port (TP standalone, sharded embedding/argmax,
  format-6 experts), ratchet reconciled 186844+448 → 187879. Scanners green.
- SCOREBOARD: 27B bests improved on exact main (f8f2ea0): B1 no-spec 8.45
  (was 8.03), B4 aggregate 55.44 (was 36.22 — continuous batching + kernel
  fixes; B=1/2/4 = 8.45/16.75/55.44, knee sweep K1, spark3 clean session).
- qwen38max-shard milestone: module PUBLISHED through the v2 validator on
  spark7 (wire-format-v2 chain green); driver compile blocked on firmware
  'stages' schema in examples/ — coordinator fix queued next sweep.
- Ceph canary: transient ENOENT + slow uncached reads = MDS cold after the
  08:59 reboot (expected, self-heals); kimi-k3 warm copy intact (96 shards).

## 2026-08-28 ~10:0x — qwen-flash M5 merged; modeling reference found+ pinned; lane resumed

- MERGED lane/qwen-flash final (e3b795a, ratchet green): M5 whole-stack TP4
  standalone validator gate PASSED on sparka AND spark4 (cross-node
  bit-reproducible, token 37853); five real bugs fixed behind the retired
  guard incl. the 1.4e-3 decode-vs-prefill drift -> bit-exact and an MTP KV
  OOB write; router-gate corrected -> v2 packs (replicated gates) deployed
  sha-identical spark4-7.
- BLOCKER CLEARED: modeling reference for hc/indexer/PLE found in
  huggingface/transformers main — pinned at
  model_contracts/references/modeling_qwen4_exp.py (sha 77fec77d...):
  GatedResidual (=hc), QSAIndexer, NGramEmbedding (=PLE) all present.
- PLE DECISION (operator quality preference): vocab-shard 23.9 GiB/rank
  BF16, no quantization loss; fp8 only as reported fallback.
- Lane RESUMED as new agent: S1-S4 semantics ports + v3 packs, S5 M6
  residentd/api, S6 live 4-node smoke + first perf cell.
- Standing integration request #4 (tests wiring) verified ALREADY DONE
  (Makefile:260).

## 2026-08-28 ~10:4x — sweep: SPARKE DOWN (osd.14); ceph topology fully mapped; knee sweep running

- INCIDENT: sparke unreachable (no ping from 2 vantages) — runs osd.14;
  degraded PGs, replication serves. Second node-down today (after spark3's
  reboot pair) — sysadmin asked to check for a common cause. K3 lane told
  to hold (journals/packs survive on NVMe; no build relocation — restart
  would re-read 174GB from degraded ceph).
- TOPOLOGY CORRECTED: every node runs an OSD (osd.0-18); mons spark0/7/f
  (quorum alive); MDS PAIR = mds.ds4warm.spark2 + .spark3 (explains
  transient ENOENT blips = pair failover from spark3's reboots; seen
  again on spark6 this sweep). Storage-risk rule REVISED in the README:
  strict no-GPU only on active-MDS hosts (spark2, spark3); all other
  nodes under a ~85-90 GiB device envelope so local OSDs survive GPU jobs.
- GPU: spark3 at 95% — the knee sweep is measuring (B-point checkpointing
  per protocol). spark5 load 2.7 (bisect building/measuring). No new lane
  merges this pass (no new commits on validator-fix/knee/bisect branches).
- Queue: bisect agent self-registered spark4-7; gates unchanged
  (glm52-validator-fix still in flight on spark8).

## 2026-08-28 ~11:0x — spark0 build collision arbitrated

- The qwen38max-SHARD lane staged an uncommitted packer on spark0
  (~/sparkpipe-lane, md5-identical to its worktree), pkill-by-name'd the
  sibling qwen38max lane's two stage builds, and started a rank-pack build
  reading WARM during ceph degradation.
- ARBITRATION: sharded v2 = THE qwen38max pack format (module published
  through the validator; all-16 policy). Full-width stage packs = scale
  proof only (original lane finishes 4 + verifier receipts, parks on
  sparkb). Divergent build TERMed (clean exit); COORDINATOR-HALT.md left
  in the shard lane's spark0 staging dir + controller worktree (commit
  packer, wait for ceph recovery, build from COLD, coordinate node,
  never pkill by name).
- NEW HARD RULE in README: no pkill-by-name on shared nodes (own pids
  only); remote staging from committed branches only.

## 2026-08-28 ~11:3x — K3 findings A+E fixed on main; lane merged; ratchet accounting corrected

- FINDING E FIXED (the real heap corruption): every self-cleanup failure
  path in SparkK3StageRunnerInitialize deletes `state` but never clears
  the early-assigned runner->private_state — the adapter's error-path
  destroy then double-frees. All 7 sites now clear the published pointer
  first. (Findings B/C/D were already fixed in 01b7ae4.)
- FINDING A FIXED: -lcuda added to the K3 serving adapter link (the
  runner's TMA driver API made the .so un-dlopen-able).
- MERGED lane/k3 (verifiers, smoke launcher, report): k3_verify_pack.py
  cross-checks every rank byte-level vs the stage pack (negative
  controlled), k3_verify_source.py pins the 38-field contract.
- RATCHET ACCOUNTING: my earlier modeling-reference pin (2707 lines of
  upstream modeling_qwen4_exp.py) slipped in without a ratchet run —
  references/ (vendored publisher semantics files) is now an excluded
  component by policy, and the ceiling moves by the lane's real 835
  authored lines (188714).
- K3 lane HOLDING for sparke (stage-0 journal at 266 tensors on its
  NVMe); auto-resume script noted — needs restart if the node rebooted.

## 2026-08-28 ~12:0x — SECOND spark0 kill incident; rule hardened to positive form

- The shard lane's "kill my own pids" cleanup used a fuzzy
  '*qwen38max.tp4*' PATH match — caught the sibling's stage0/2/3 builds
  (pids 1288558/1316337/1316338). Self-reported immediately; agent now
  fully hands-off spark0 (no process mgmt, builds, or cleanup there).
  Its earlier 09:15 kill identified as the same class (pkill -f bash -s
  restart pattern). Sibling notified to relaunch (stage1 receipt + pack
  unaffected; shard rehearsal dead = I/O share returns).
- Coordinator removed the two orphaned shard tmp files by explicit path
  (48G reclaimed, spark0 /home now 970G free).
- RULE HARDENED (README): kill ONLY spawn-captured pids; no fuzzy match
  by name, path, or parent — the agent's own root-cause line, verbatim.

## 2026-08-28 ~12:3x — sweep: SPARKE RECOVERED; K3 resumed; qwen38max merged

- SPARKE BACK (up 28 min, osd.14 rejoining, GPU healthy, stage-0 journal
  intact). Previously-stuck shard-12 object now reads 253 MB/s direct —
  recovery confirmed. K3 lane RESUMED via message (told to merge main
  first: all 5 module findings now fixed there; restart probe loop with
  spawn-captured pids per the hardened rule).
- MERGED lane/qwen38max (3588464, ratchet 189314 justified: tp4pp4
  pack/deploy tooling + stagepack rewrite that built the 573 GiB stage
  packs). Report-only tip + earlier tool commits; format arbitration
  acknowledged in their report.
- GPU: spark3 96% (knee sweep cooking). spark5/8 GPU-idle between
  measurement phases (bisect/validator host work). Queue gates all
  correctly pending; reservations intact.


## 2026-08-28 ~13:0x — sweep: K3 build-break fix merged; validator staleness flagged

- MERGED lane/k3 (6ef8c95): the lane fixed a BUILD BREAK from my own
  01b7ae4 (stray trailing */ in the stage_layer_counts comment) and
  retired the export-shim (superseded by the canonical-symbol fix).
  K3 stage-0 resume in progress on recovered sparke.
- GLM52-VALIDATOR WATCH: worktree silent ~2.2h, spark8 GPU idle —
  borderline (deep analysis vs stuck). If next sweep shows no movement,
  ping the agent; escalate if two sweeps silent.
- Canary ENOENT blip again on shard 80 (the documented MDS-pair
  failover signature during ongoing recovery — wait-and-retest, not a
  new incident). spark3 knee sweep still at 96% (cooking); bisect
  between measurement phases on spark5.

## 2026-08-28 ~13:4x — K3 module chain VERIFIED on real pack; rogue glm52 rank-6 TERMed

- K3 SMOKE MILESTONE: rank-12 on sparkc with the 16-node deployment gets
  through adapter_load → module init+bind (layer-92 fix confirmed live)
  → dispatch → collective, dying only at the PP transport wait — the
  module chain is clean; fleet bring-up = the first K3 fleet number.
  Scoreboard cell updated.
- NODE CONFLICT RESOLVED: glm52 TP8 rank-6 residentd (validator lane's
  ranks-1-7 step) sat ALONE on sparke at 114/121GB, OOM-killing the K3
  stage-1 builder and one allocation from re-OOMing osd.14. TERMed
  cleanly (memory 114→59GB). New rule pending README: residentd ranks
  come up ONLY as a coordinated fleet bring-up with queue reservations —
  half-lit fleets are wasted memory and neighbor-killers.
- MERGED lane/k3 config fixes (98ebc2e): deployment stage_index must be
  the unique linear rank (my k3_gen_deployment.sh emitted i/4 duplicates),
  kv page capacities 0 (k3 descriptor has no JIT_KV). K3 stage-1
  restarting on sparke; stage-0 probe-then-resume continues (shards 12/13
  deep objects still in backfill).

## 2026-08-28 ~14:3x — GLM52 LANE COMPLETE: PR 728 merged; glm53 RESUMED

- MERGED lane/glm52-validator-fix (b313cd5): routed-oracle failure was
  TWO validator-side bugs (expert dimension applied to scale index only
  — every expert decoded slab 0; positive-mean fixture grids = quadratic
  amplifier), kernels were CORRECT. First full validator PASS (dense+
  routed+dsa); 3 more restore bugs fixed (admission predicate, KV
  page-cache lane wiring, JIT_KV capacities); 8 ranks ready + B1 decode
  E2E complete (timing honestly not claimed — 8 tokens, no timestamps).
  All 3 integration requests landed: tier-2 oracle gate registered
  (PASS locally), model_common runbook note, ratchet 192659 justified.
- GATES CLOSED (glm52-validator-fix, glm52-smoke) → glm53 lane RESUMED
  (rebase onto fix, M3 validator with the fixed oracle pattern, M4 packs
  on the dark spark8-f band w/ reservations). spark8 reservation
  released; spark0 785GiB packbuild insurance REMOVED (manual, logged).
- SYSADMIN+1: sparke wedged at NETWORK level twice today (self-recovered,
  one self power-cycle) — node-health check requested before it hosts
  long-running work.

## 2026-08-28 ~15:0x — KNEE CURVE: B32 = 1469 tok/s aggregate (thesis proven); Pro lane stray cleared

- KNEE SWEEP MAIN CURVE (128-lane cfg, weights resident): B1-32 decode
  aggregate 8.44/16.86/33.73/111.18/406.65/1469.40 tok/s; step time
  118.5→21.8 ms. The compute-thesis proof — with 55GB weights resident
  the B1 step floor is overhead, not weight streaming; scoreboard updated.
- P1 FOUND (knee lane, B=24/48): intermittent FAILURE_DEACTIVATE_ROUTE
  (status=17 INTERNAL_ERROR reason=7) in the prefetch-abort deactivate
  path, node/model_residentd.c:1413-1417 — root-caused to code, logs
  captured; fix to be scheduled off the lane's receipts.
- B=64 pathology (3.5 tok/s, GPU busy) — 64-lane comparison ladder
  running to isolate lane-capacity overhead.
- DSV4 PRO STRAY: its rank-3 on spark3 self-exited fleet-less
  (transport-wait); spark3 is knee-reserved AND an MDS host. Marker left
  in its deployment dir; fleet-only + storage-host rules re-cited.
  spark2's rank-0 daemon = the legitimate 27B dev instance.
