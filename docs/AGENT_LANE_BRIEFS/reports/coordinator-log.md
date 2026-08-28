
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

## 2026-08-28 ~15:4x — DSV4 PRO LANE COMPLETE: PR 729 merged; Aug-15 ghost fixed

- MERGED lane/dsv4pro (3c83bf7 + ratchet d471853): 16/16 TP4xPP4 rank
  packs built+verified (~93GB EACH — the DSpark draft block is fully
  replicated per rank; fleet table corrected from ~52GB), 14/16 deployed
  (ranks 1/2 stashed on warm: spark1/2 unavailable). OLD PACKS WERE NEVER
  CORRUPT — the contract verifier derives full-width records and fails
  rank packs BY CONSTRUCTION; real failure was deployment-side.
- THE AUG-15 GHOST: examples/deployments/dsv4_pro...stage.json carried a
  placeholder model_revision since creation, pinned by a test — no Pro
  daemon could EVER pass adapter validation. Fixed on main + deployed
  configs (be03caa).
- REMAINING for Pro's ready line: driver-lane rebuild from current main
  (Aug-17 binaries fail validation status=9 after the config fix) —
  owner: follow-up dsv4 driver work; then ranks 1/2 when spark1/2 free.
  NOTE for the 16-rank fleet: spark2/3 are MDS hosts — the full-fleet
  bring-up needs the MDS-relocation decision or an envelope exception.
- Third fuzzy-match incident (Pro's smoke cleanup hit bisect's daemons on
  spark4-7; they self-restarted) — the spawn-captured-pids rule covers
  it; no further action.

## 2026-08-28 ~16:2x — sweep: MDS metadata INCONSISTENCY escalated; no merges

- ESCALATION 3 (ab4243b): warm kimi-k3 readdir/stat disagree (file
  listed, open ENOENT) — persistent, cross-node. Both MDS alive; hosts
  at 84-90G with co-located GPU daemons. Sysadmin: ceph fs status +
  likely standby-MDS restart; relocation/MemoryMin asks now backed by an
  integrity incident. K3 lane warned (stage-0 source affected; local
  NVMe packs/journals unaffected).
- No lane merges this pass (glm53's a6a3847 = the already-processed
  checkpoint report). spark3 knee sweep 96% (ladder finishing). Bisect
  fleet measurements live on spark4-7 (per Pro lane's accidental
  confirmation). All OSDs healthy; disks fine (spark0 1.7T, spark5 1.9T).

## 2026-08-28 ~16:5x — sweep: qwen-flash S4 advancing; MDS issue unchanged

- qwen-flash S4 in flight: v3 rank packs (hc+indexer+PLE vocab-sharded,
  56.06 GiB/rank, 1246 entries) rank0+rank1 verified 7 min ago — on pace
  for S5 (M6 driver) and S6 (live 4-node smoke + first perf cell).
- MDS metadata inconsistency UNCHANGED (shard 80 ENOENT persists) —
  sysadmin action still pending; K3 stage-0 warm reads remain blocked;
  all lanes on local-NVMe paths unaffected.
- No merges this pass. GPUs: spark3 96% (knee finishing), spark4 idle
  (qwen-flash between build/verify), spark2 idle (glm53 probe analysis).

## 2026-08-28 ~17:2x — KNEE LANE COMPLETE: PR 730 merged; two P1s queued

- MERGED lane/knee-sweep (3c2f1ac): full curve + L64 ladder + structural
  findings on the scoreboard with cell qualifications (B4 is config-
  specific per the attestation; the old ledger 'B16 ~9' is now 406.65).
  spark3 reservation RELEASED — node returns to MDS duty per storage rule.
- THE R-LAW (operational): per-step time ∝ B×64/lanes — configured lane
  capacity taxes every small batch (B=4: 64-lane 1.64× faster than
  128-lane). Deployment guidance: right-size lanes to workload.
- KNEE VERDICT: classical knee NOT reached; B*≈106 unproven, not
  contradicted. The walls are software —
  P1a: FAILURE_DEACTIVATE_ROUTE client-fatal bug (model_residentd.c:
       1410-1417, intermittent B≥24, 505-submission receipt) — FIX NEXT.
  P1b: ratio cliff at B ≥ lanes/2 (~3.5 tok/s, 96% GPU, no errors;
       identical at 64-lane B32 and 128-lane B64; not page/row-bound) —
       scheduler/dispatch structural bug.
- Exact-32K at 128 lanes mathematically infeasible on this hardware
  (517 blocks/lane → ~278 GiB KV); documented by the lane.

## 2026-08-28 ~17:5x — sweep: MDS recovered; P1a FIXED on main; retest agent on spark8

- MDS metadata inconsistency CLEARED (shard 80 stats + reads 238 MB/s
  from spark6). Residual: sparke's ceph CLIENT serves a stale negative
  cache (shard 12 ENOENTs sparke-local, fine elsewhere) — age-out or
  remount (sysadmin); K3 lane informed (source reads from other nodes).
- P1a FIXED (dcd3824): idempotent slot release + abort-path fences
  instead of client-fatal. Retest agent launched on spark8 (B=24/48
  reproduce-attempt + B=16/32 regression check, reserved). Verdict
  pending — all three outcomes (proven/not-reproduced/refuted) valid.
- Queue: spark3 released (MDS duty), spark8 → lane-p1a. No lane merges
  this pass (glm53/qwen-flash advancing, no new tips).

## 2026-08-28 ~18:3x — GLM53 M3 COMPLETE: validator full pass + module PUBLISHED

- The KDA step-0 ghost resolved via independent-reference arbitration
  (mac-side C transcription + xorshift32 grid replication + full-vector
  dumps): THREE real defects — (1) KDA o_norm must be F32 in the pack
  (k3 donor convention): the bf16 store made the gated-norm kernel read
  past the 128-element tensor, silently zeroing channels 64-127 of every
  head (the coordinator's first-position/edge hint pointed here);
  (2) fixture kv_slot 512-vs-8192 width (fixture-only overflow); (3) the
  oracle skipping the RMS norm of the HC-collapsed input (~1000x).
- Gates: tier1 KDA+dense+mHC rel_l2 0.0033 cos 0.99999 (4-token walk),
  bit-exact determinism both attention kinds, mid-pipeline PUBLISH
  SUCCEEDED (validation=executed, artifact 1987cb9c). Scoreboard updated
  (2907584). Honest non-claims: tier2a numeric, MoE leg, kpool 2b = the
  pack bring-up increment.
- LESSON for the memory design: the o_norm overread is a silent
  width-mismatch read past a tensor with no fault — cited as provenance
  for typed buffers/provenance tagging in the inference-OS memory model.
- Lane RESUMED for M4: 16-rank TP16 packs on spark8-d/f (reservations
  first; warm reads from non-sparke clients per the stale-cache caveat).

## 2026-08-28 ~14:3x KST — FLEET-WIDE AGENT PAUSE: 5h usage limit hit by all 8 lanes

All background agents died simultaneously on the account usage limit
(reset ~13:28 UTC). Remote NOHUP'D work survived; interactive
(agent-driven) work died:

SURVIVING (unattended, cooking):
- spark0: 3 qwen-max stage builds (journaled, self-resuming)
- spark7: 3 MXFP4 download loops (write to warm; receipt chain verifiable)
- spark2: the 27B dev daemon (normal resident)

DEAD MID-FLIGHT (resume points recorded):
- dsv4-bisect: active TP4 measurement on spark4-7 (client died with
  agent; daemons gone). Worktree /tmp/lane-dsv4bisect holds D-milestone
  state; packs v3+v4 built+validated on spark5. RESUME: D-measurement.
- glm53: M4 never started (agent died ~30s after resume). M3 complete
  and published. RESUME: M4 packs on spark8-d/f.
- qwen-flash: S4 rank2/3 builds died on spark4 (verify build state
  first). RESUME: S4 finish -> S5/S6.
- K3: probe/resume chain dead on sparke. RESUME: stage-0 journal.
- p1a-retest: died at launch, nothing started. RELAUNCH fresh.
- qwen38max full-width: agent dead, builds survive on spark0.

RULE for this window: no agent spawns until the limit resets; the sweeps
monitor the survivors; resumes happen in priority order (bisect verdict >
glm53 M4 > qwen-flash S4 > K3 > p1a retest) as budget returns.

## 2026-08-28 ~15:0x KST — budget reset confirmed-ish; MXFP4 download COMPLETE

- Usage-limit window passed (reset 13:28 KST < now 15:00 KST). Bisect
  agent resumed as the budget probe + top priority (D-measurement).
- MILESTONE: the AMD Quark MXFP4 source download is COMPLETE on warm —
  213/213 shards at /mnt/model-warm/packbuild/qwen38max/amd-mxfp4
  (12T used on warm total). The shard sprint's 16-pack build is
  unblocked on resume; receipt-chain re-hash still required (downloaded
  during ceph degradation).
- Survivors healthy: spark0 3 stage builds, spark7 download/verify
  loops winding down. glm53's spark9-f reservations persist (lane died
  post-reserve; resume keeps them).
- Resume stagger to conserve budget: bisect now; glm53 next sweep;
  qwen-flash/K3/p1a after, as capacity allows.

## 2026-08-28 ~15:3x KST — resumes issued; verification UNCERTAIN

- Bisect (resumed ~15:00) and glm53 (resumed ~15:30) both show NO
  failure notifications but ALSO no worktree activity yet (bisect wt
  stale 5.5h, glm53 wt stale ~30m). Either long read/plan phases after
  resume, or the resumes are not executing. DEFINITIVE check next sweep:
  worktree mtimes + any notifications. NO further spawns until resolved
  (budget conservation + uncertainty).
- Survivors healthy; queue reservations unchanged (glm53's spark9-f set
  persists). Scoreboard unchanged.

## 2026-08-28 ~16:0x KST — resume verification RESOLVED: glm53 alive, bisect re-attempted

- glm53 RESUME CONFIRMED ALIVE: commit 21bcf7c pushed 23 min post-resume
  (M4 pack work underway; the dir-mtime check was flawed — branch tips
  are the truth).
- Bisect first resume did NOT execute (zero activity 1h). Second attempt
  sent with a touch-your-worktree first action; if silent another 30 min,
  the bisect re-tasks to a fresh agent next window (state carries in its
  report/receipts).
- Scoreboard unchanged; survivors healthy.

## 2026-08-28 ~16:3x KST — both resumes CONFIRMED alive

- Bisect second resume EXECUTED: worktree touched 07:06 UTC (5 min
  after the resume message) — agent alive, early-phase. glm53 mid-M4
  (tip 21bcf7c, normal between-push cadence). Fleet healthy; scoreboard
  unchanged.
