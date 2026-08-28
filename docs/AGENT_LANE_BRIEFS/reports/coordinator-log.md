
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

## 2026-08-28 ~17:0x KST — bisect MEASURING (spark5 86% GPU); stagger holding

- Bisect is in its D-measurement phase: spark5 GPU 86%, residentd up,
  cooking the 40-vs-33.5 verdict cell. glm53 mid-M4 (tip 82m old —
  emission/re-verify work between pushes).
- Stagger intact: 2 concurrent agents (bisect, glm53); qwen-flash next
  when a slot frees (its 4c3ff91 = pre-pause S4 state), then K3, p1a.
- Queue TTL audit pruned expired manual reservations (spark4/5 from the
  bisect's old set; its live measurement unaffected — advisory among
  queue users). Scoreboard unchanged; verdict expected next bisect push.

## 2026-08-28 ~17:3x KST — bisect measurement ended; wrap-up or analysis (no notification yet)

- spark5 GPU returned to 0% and the bisect's queue reservations were
  released (by the agent or TTL audit) — consistent with either
  post-measurement analysis or lane wrap-up. No completion notification
  yet; next notification resolves. glm53 2h between pushes (mid-M4
  emission I/O — plausible). Scoreboard unchanged.

## 2026-08-28 ~18:1x KST — glm53 M4 COMPLETE: 16/16 TP16 packs shape-gated

- All 16 real packs (21.7GB each, 347.3GB; fleet table corrected +1.2GB
  replicated set) pass the module's own shape gate (1160 tensors, 0
  fail/rank). The gate caught 3 invalidating packer bugs pre-ship
  (payload_bytes collapse, magic byte-order, wrongly-TP-sliced
  replicated classes) + an --mtp loop bug. Lane RESUMED for bring-up:
  distribute → loader-init verify → whole-stack publish → all-16 deploy
  (fleet-only, reservations on all 16) → M5 exact-32K B1 cell.
- Stale lane-p1a reservation on spark8 already gone (TTL audit). Bisect
  mid-A/B on spark5 (ef8fa302-side deployment cycling).

## 2026-08-28 ~19:0x KST — DSV4 REQUALIFICATION LANDED (PR #731 merged, 71d52a6)

- THE FRAMING CORRECTION: main was BROKEN at the no-spec B1 cell (9
  tokens + FAILURE_CONTINUE_LEASE), not slower; the ledger's 33.55 was
  main@Aug-14. And the 40.4-at-exact-32K attribution was wrong — 40.4
  is the O128 cell; 32K tops at ~29 even on lean source (attention
  scaling ~10 tok/s). Scoreboard now carries the true cells: main+fix
  40.46/40.35/40.19 (O128, exact hash 211462f2).
- TWO REGRESSIONS FIXED: (1) the dspark always-on machinery gated behind
  SPARK_DSV4_DSPARK=1 (default off) — SPEC QUALS NOW NEED THIS ENV;
  (2) the lease-advance mirror bug (daemon advanced by accepted count=1,
  client by chain width 8). Lean reproduced at 39.81/39.77 exact-hash.
- Bisect lane doing one closing 32K re-run on the staged deployment.
- Stagger slot freed → qwen-flash resumes next sweep (S4 finish).

## 2026-08-28 ~19:3x KST — discovery: qwen-flash S4 already COMPLETE pre-pause

- lane/qwen-flash 4c3ff91 (pre-pause) = 'S4 COMPLETE - 4/4 v3 ranks
  verified' — the pause hit AFTER S4, so its resume is S5 (M6 driver/
  deploy) + S6 (live 4-node smoke + first perf cell), not S4.
- Bisect agent re-reserved spark4-7 for its closing 32K run (mid-cycle).
  glm53 bring-up continuing (tip = its checkpoint). Stagger holds at 2;
  qwen-flash resumes when the bisect's completion notification lands.
- Scoreboard unchanged this pass.

## 2026-08-28 ~20:1x KST — glm53 bring-up: loader-init PASS on real packs (9986eab)

- All 16 packs distributed (rank r → spark{hex r}); the module's FULL
  init chain runs clean against real 21.7GB TP16 packs (provenance
  digests, 1160 entries, device upload, caches, KV page-store) — the
  last verification-critical unknown retired.
- Three bring-up fixes: KV pool 64.96→5.9GB (DSA_LAYER_COUNT stride
  double-multiply), O_TMPFILE needs a DIRECTORY (fallback named a file),
  module link set completion.
- CROSS-LANE FINDING for glm52's follow-up bench: glm52's KV page-store
  fallback has the SAME O_TMPFILE file-path shape (ENOTDIR risk).
- Lane resumed: whole-stack publish → all-16 deploy (fleet-only,
  all-16 reservations, coordinate spark4-7 with bisect's closing run) →
  M5 exact-32K B1 = first glm53 scoreboard number.

## 2026-08-28 ~20:4x KST — glm53 whole-stack PUBLISHED; all-16 staged at 12/16

- glm53 whole-stack TP16 publish SUCCEEDED on the real rank-0 pack
  (validator PASS 0 failures, validation=executed, artifact 3219f204...).
  Deployment tree generated (16 per-rank configs, hidden_transport TP16,
  kv_backing as directory). HOLDS spark0-3+8-f (12/16); spark4-7 staged
  pending the bisect release. Staging runtime trees meanwhile so launch
  is one command per node.
- Bisect asked directly: 32K run complete / in-flight / done-unreported?
  (its TP4 fleet still up, GPUs idle, worktree quiet for hours).

## 2026-08-28 ~21:0x KST — 32K run in-flight w/ +25min cutoff; two meta-notes

- Bisect's exact-32K main+fix run in-flight (31m into prefill; lean's
  took 17m — the PREFILL ASYMMETRY is an open dsv4 item independent of
  the restored decode). Hard cutoff +25m then TERM+release; glm53
  launches on release.
- SWEEP RULE ADJUSTMENT: GB10 nvidia-smi reads 0% under real load —
  GPU-util is NOT an idleness signal on residentd-serving nodes; use
  process + queue state (the 'GPU idleness is a finding' rule now
  qualifies this).

## 2026-08-28 ~20:0x KST — second usage window: agents died, remote work survived; both resumed

- Second 5h window cut glm53 (mid-staging) + bisect (mid-32K-enforcement).
  Remote state INTACT: bisect's 32K batch client + fleet alive on spark4/5;
  glm53's staged trees + distributed packs + published module all persist.
  Queue emptied by TTL audit — re-reservation on resume.
- Both agents resumed (stagger 2): bisect collects/cuts its 32K run then
  releases spark4-7; glm53 waits for the clear, re-reserves all 16,
  launches simultaneous per fleet rule, checkpoints right after
  ready-lines, then M5.

## 2026-08-28 ~20:4x KST — BISECT LANE TRULY DONE; spark4-7 → glm53

- spark4-7 released + verified dark. The closing 32K run was cut at 141
  min (prefill, never reached decode) → recorded as OPEN ITEM: >=8x 32K
  row-serial prefill regression main-family vs lean's 17.3 min at the
  identical cell/pack/fleet; candidate axes: batch-engine prefill
  row-budget planning, v4 pack layout, prefill admission, client spin
  (perf-top/ptrace before rerun). Decode restoration stands (71d52a6).
  PR #732 = docs-only follow-up. Staged fix runtime + v4 packs remain on
  spark4-7/spark5 for the prefill investigation; glm53 may reclaim /tmp.
- glm53's last gate is GONE: all 16 nodes free, launch sequence live.

## 2026-08-28 ~21:1x KST — spark0 disk rescue; 27B aggregate roofline flag; MDS move news

- SPARK0 RESCUED: was 12G free/100% (disk-full killed the survivor
  stage builds — 0 procs, stage3 never finished). Removed unverified
  stage0+stage2 full-width packs (1.15T; rebuildable from cold); KEPT
  verified stage1 (615.3GB, receipt + verify.json = the scale proof).
  Now 585G free. Logged destructive action per policy.
- OPERATOR ROOFLINE CHALLENGE on the knee aggregate (their instinct):
  effective TFLOPS = tok/s x 54 GFLOP/token: B16 = 22 TFLOPS (plausible),
  B32 = 79.3 TFLOPS — possible only if GB10 dense BF16/FP8 peak is the
  125/250 class (63% MFU) vs the plan's assumed ~50 TFLOPS FP8 (would
  make it impossible). B1 no-spec = 8.44 (the old '~9' was an old-stack
  aggregate). The super-linearity is the r-law (step cost scales with
  CONFIGURED lanes; more rows/lane nearly free) but the B32 point needs
  PER-ROW token-stream verification (hash each row's output, not
  aggregate counts) before '1469' is trusted as decode tok/s. QUEUED as
  the verification follow-up; spec (24.5 old-stack) not yet measured on
  cont-batching main — pending.
- MDS RELOCATION to MacStudio announced by operator (sparks = backup) —
  when landed, spark2/3 return to full GPU duty and the storage-host
  rule relaxes. Best structural news of the day.

## 2026-08-28 ~21:4x KST — glm53 mid-launch (all-16 reservations, staging phase)

- glm53 holds the all-16 reservation set (10 confirmed in queue view)
  with zero fleet daemons yet — runtime-tree staging/sync across 16
  nodes; no failure notifications = tentatively alive. Watching; the
  ready-lines + M5 number are the next landmarks.
- ceph canary + spark0 disk post-rescue: healthy (585G free).
- Scoreboard unchanged.
  - LATE NOTE: canary ENOENT returned on shard 80 (spark6) — the
    intermittent MDS signature. Possibly the MacStudio MDS migration
    work itself (failovers during the move); glm53's launch is
    local-NVMe-only, unaffected. Watching.

## 2026-08-28 ~22:1x KST — glm53 died TRANSIENT (model error, not budget); retried

- Launch agent died on a model-request failure mid-staging (all-16
  reservations held). Resume retried with handoff-discoverability
  instruction (LAUNCH-STATE.md). Fallback if infra errors persist:
  coordinator launches manually — one command per node, fully staged.

## 2026-08-28 ~22:4x KST — GLM53 TP16 FLEET IS UP (16/16 daemons)

- All-16 reservations held + residentd running on every sampled node
  (spark0/3/8/c/f all carry their rank). Launch survived the transient
  death via the retried agent. Next: ready-line verification + M5
  exact-32K B1 = first GLM 5.3 Flash scoreboard number. LETTING IT COOK
  (no invasive checks during measurement).

## 2026-08-28 ~23:1x KST — glm53 submit-path debug underway; fleet steady

- Fleet 16/16 steady, all-16 reservations held; API healthy on spark0
  ({"status":"ok","served":1} — first served request recorded; whether
  that's the health probe or the first token, the lane's next push will
  say). Lane debugging the batch-engine submit blocker per the pointer.
- Scoreboard unchanged; M5 number pending the submit path.

## 2026-08-28 ~23:4x — CURVE RETRACTED (units bug); co-resident test framed

- OPERATOR VINDICATED (2nd challenge): knee-sweep awk multiplied the
  batch-total token count by B — every rate inflated xB (B32: 1469
  reported vs 41.3 true; receipts valid: wall+tokens were direct).
  Scoreboard retracted + tool fixed (3c98234). TRUE 27B curve: 8.31 B1
  → 41.3 tok/s saturation ~B16 (batching ~5x, not 174x). Open: flat B2
  (8.31 = B1 — first doubling gave nothing), step-ms re-derivation,
  K1's B2/B4 numbers (16.75/55.44 used the buggy column — re-derive).
- CO-RESIDENT TEST framed (operator request): glm5_next at 21.7GB/rank
  leaves ~76GB/node headroom (spark2 measured: 42G used with rank up).
  27B TP1 full pack (29.9GB) co-deploys per node without new packs.
  Plan: after glm53's M5, co-resident 27B on 4 glm53 nodes (distinct
  API ports), measure both models' tok/s solo-vs-paired + memory/OSD
  health — surfaces the co-residency issue class (mem pressure, one-
  client slots are per-daemon=OK, launch-wave interactions).

## 2026-08-28 ~00:1x KST — glm53 submit-debug window; fleet steady

- Fleet 16/16 steady (reservations + daemons); glm53 lane in its
  submit-path debug window (tip = blocker checkpoint 41m old; API
  restarted during debug, served counter reset). No merges this pass.
- Canary: healthy/known-blip class (MDS migration window). Scoreboard
  carries the RETRACTED-curve correction from the operator's catch.

## 2026-08-28 ~01:0x KST — first-token: deadlock FIXED on main; reject ranked for the lane

- API self-deadlock root-caused by the lane w/ gdb receipt (worker held
  the queue mutex across Submit→api_event re-entry) — fix (inflight/
  orphaned, submit-outside-lock) merged to main, verified live (my curl
  reaches admission). Ranked analysis handed to the lane for the
  admit→UNSUPPORTED: (1) publish-count mismatch @kv_page_cache ~996,
  (2) context==0-with-position!=0 underflow @586/800 (mis-filled field
  class), (3) dump block_token_count (geometry). One instrumented
  fprintf names the site; cache-side fix mine, adapter-side theirs.
- Follow-ups logged: client-generation recovery path (schema-error on
  2nd request after reject); glm52 adapter missing CONTINUE_LEASE (its
  next re-deploy hits the client gate).

## 2026-08-28 ~02:3x KST — CO-RESIDENT TEST phase 1 (mechanics): PARTIAL, findings real

Operator question: does two-model co-residency work? Phase-1 results:
- MECHANICS VALIDATED: 27B TP1 (29G pack) staged spark9+a alongside
  glm53 ranks; both daemons reached READY (module 71.1GiB pool + pack
  resident); config portability bugs found+fixed en route (absolute
  spark2 paths in deployment json; "host": "spark2" listener field;
  TIME_WAIT 45s rule bit twice).
- MEMORY ENVELOPE MEASURED: 114G/119G used with the FULL 71.1GiB KV
  pool — co-resident configs need a REDUCED KV pool (e.g. 4096 blocks
  ≈ 35GiB) to leave real headroom. This is the sizing datapoint.
- NOT PROVEN: both 27B instances exited SILENTLY post-ready (3-line
  clean logs, no error) under my hand-rolled ssh launch chains — root
  cause undetermined (orchestration pattern suspect: setsid-nohup-over-
  ssh + later TERM cycles; the lanes' proven gen_deployment launch
  tooling is the right harness, not hand curls).
- TRUE TWO-MODEL LOAD still gated on glm53 first token (admit bug in
  flight). Phase 2 = proper launch tooling + reduced-pool config +
  concurrent B-load on both models + interference measurement.

## 2026-08-28 ~03:0x KST — glm53: submit path FULLY OPEN (0482ccc)

- Lane's newest push (3 min old): submit path fully open, wave cycling
  during hunts 2-4. Mid-sequence — merging waits for the milestone
  (first token or the kv-admit fix). Fleet+API on spark0 mid-wave.
- Scoreboard unchanged.

## 2026-08-28 ~03:2x — OPERATOR CEILING: 110 GiB (the silent-exit root cause)

- User diagnosis confirmed: kernel log shows NVRM NV_ERR_NO_MEMORY at
  22:35:53 = the exact moment the sparka co-resident daemon exited
  silently (driver-allocator OOM at ~114/119 GiB; same class as this
  morning's spark3 crash). Not the Linux OOM-killer — the GPU driver's
  allocator; identical practical lesson.
- NEW HARD RULE (README): device allocation ceiling 110 GiB of 119.
  Deployments sized weights+KV+overhead <= 110; co-resident sums
  computed per node pre-launch, KV pools cut to fit. Co-resident
  phase-2 numbers: 27B (29G) + reduced 35G KV + glm53 rank (21.7G)
  ≈ 92G — comfortable.

## 2026-08-28 ~03:5x KST — glm53 hunt 5: exact failing site (fc10bba)

- Lane at the exact failing site (chainfail probe, 5-min-old tip); my
  live curl still rejects (status 4, empty tokens) — the instrumented
  hunt continues. Merging waits for the milestone. Scoreboard unchanged.

## 2026-08-28 ~04:4x — async-op debug: coordinator survey + lane resumed

- Reproduced the async TP failure myself against wave-18 (one curl; log
  trail: admit OK → CUDA wave → collective submit accepted → completion
  status 1 with ROWS 0 — the tell: submit took rows 1).
- MY SURVEY: transport CORE exonerated (hidden_transport.c = open-time
  checks only; tp_device_collective.c latches VALIDATION_FAILED not
  INVALID_ARGUMENT). Likely origin: the RDMA backend's async
  completion-delivery (rdma.cu, 89 sites; credit repost vs
  MAX_PENDING_RECEIVE_COUNT is a concrete candidate). Lane resumed for
  the co-debug (instrumented transport .so swap on spark0 + one curl).
- ALSO noted: tp_collective schema HARD-REQUIRES 2 rails + 3 step
  indices (single-rail = schema_error) — an inflexibility worth a
  follow-up.

## 2026-08-28 ~05:1x KST — phantom reservations cleaned; lane in transport-instrument window

- The resumed lane re-reserved with BARE hex hostnames (0-f, from the
  fleet-table's rank→spark{hex} shorthand taken literally) alongside the
  proper spark0-f set — 16 phantoms released. NOTE for lanes: node names
  are spark0..sparkf, never bare hex.
- glm53 API mid-wave during the lane's transport .so instrument work
  (curl empty; tip = the 33m-old handoff receipt). Scoreboard unchanged.
  - CORRECTION: the release also cleared the real set (the lane's
    reserves live in its worktree's runs/, advisory only; no contention
    at this hour — fleet exclusivity remains by lane ownership).

## 2026-08-28 ~05:4x KST — lane active on transport instrumentation

- glm53 worktree touched 20m ago — the resumed co-debug is working
  (transport .so instrumentation per the handoff). Fleet holding on
  spark0 (3 residentd procs — the lane's wave cycling; its node).
  Scoreboard unchanged; next event = the instrumented curl naming the
  async-op origin.

## 2026-08-28 ~06:1x KST — lane deep in remote build/test cycle (driver .so 5m old)

- model_driver.so on spark0 rebuilt 5 MINUTES ago — the lane is alive
  and mid instrument-build/deploy/test cycle (worktree quiet during
  remote builds; the earlier staleness was the build phase). 3
  residentd procs = its wave cycling. No intervention; next push or
  first tokens resolves. Scoreboard unchanged.

## 2026-08-28 ~06:4x KST — glm53 fully quiet this pass (no builds 35m+, API mid-wave)

- No pushes, no new .so builds in ~35m, API curl empty, worktree 80m
  stale. First fully-quiet sweep since the resume; if the next sweep
  shows the same, the lane gets a direct ping + re-resume (the harness
  error classes have killed it silently before). Scoreboard unchanged.

## 2026-08-29 ~0:0x KST — 5-agent fleet in setup phase; glm53 wave ready

- The debug wave's spark0 daemon shows READY (1) with the patched .so +
  debug env live; zero diagnostic lines yet (no curl since relaunch —
  the diag agent fires it). All 5 new lanes in setup (no pushes at
  30min; inventories/builds running). k3-fleet branched from main.
- Scoreboard unchanged.

## 2026-08-29 ~0:5x — K3-FLEET (PR733 merged f2be759): 16-rank contract proven; ASAN CLOSED

- The stage-3 4-rank slice CANNOT serve standalone (3 code locks + live
  transport receipt: rank12 receives from rank8). First K3 number needs
  ALL 16 stages. Delivered: module fixes verified, ranks 12-15 staged
  digest-clean, ASAN smoke PASS zero reports (the destroy-path item
  closes not-reproducible), k3_fleet_wave.sh (fleet-only rule built in),
  memory math ~96G/rank (fits 110 ALONE; exclusive window vs glm5_next).
- CRITICAL PATH: stage-0/2 packs — build loop RE-ARMED on sparke by
  coordinator (was stalled since the reboot). Stage-1 built-not-deployed.
- SEQUENCING ARBITRATION (mine): glm5_next's first number first (diag
  agent live), then K3's exclusive window for its wave.
- RATCHET AUDIT: ceiling drift +15K found (glm5_next module content
  through successive merges; several resolutions kept stale numbers).
  Reconciled 208305 with the process note: ratchet after EVERY conflict
  resolution. Staging agent's branch-in-main violation repaired earlier
  this sweep (lane/staging-v2 worktree).

## 2026-08-29 ~1:5x — qwen-flash: official FP8 found; decisions issued

- OFFICIAL Qwen3.8-Flash-Next-FP8 exists (publisher fine-grained FP8,
  172.8G, verified fetch running) — top tier under the quant policy.
  DECIDED: bf16 16-rank primary (first cell path, zero risk), official
  FP8 as the fast-follow alternate (~10.8G/rank; scale-plane mapping =
  the verified step). v3 self-quantized packs POLICY-VOID (validation
  only; staging agent informed via this log).
- Lane's pre-fixes: staged deploy_v3 had a HARDWIRED 12-rank adapter
  (would have served 16 layers, no head) — fixed whole-stack-TP/PP
  capable; new one-wave pid-file launcher (the old pgrep -f teardown
  would have killed glm5_next's fleet — third fuzzy-kill near-miss of
  the day). spark8 rank-8 + p1a's 27B = ~104G: under ceiling, wave
  sequenced after p1a completes.

## 2026-08-29 ~2:3x — LITELLM FRONT DOOR LIVE (PR734 merged 7aaa03a)

- Proxy on the Mac :4000 (bearer auth, config committed). CENTRAL
  FINDING: our token-ID contract requires LiteLLM's /vllm/ passthrough
  (transformed /v1/completions cannot serve it) — byte-for-byte
  forwarding proven via contract-exact mock. Routing proven by connect
  evidence (all upstreams down in-window: glm53 mid-debug, 27B staged).
  The lane followed the new merge gates unprompted (ratchet reconciled
  in-commit +101).
- OWED: one live completion curl when an upstream serves; spark9's port
  at its launch.
- DESIGN NOTE for the island/catalog discussion: the token-ID edge is
  why LiteLLM's native chat path can't serve us yet — the Phase-4
  tokenizer sidecar at the island edge is the named dependency for
  multi-island text-in routing.

## 2026-08-29 ~3:4x — INCIDENT OWNED: coordinator wave launcher caused spark8 crash

- The p1a lane's B=48 was killed by MY glm53 wave launcher's TERM sweep
  (pattern-matched sparkpipe_model* — the fuzzy-kill class I outlawed,
  violated by my own tooling), and the wave's relaunch stacked 80.6G on
  their residentd -> NV_ERR_NO_MEMORY -> spark8 dark ~7min, auto-reboot
  16:48. Node recovered; rank-8 reclaimed. THE RULE APPLIES TO THE
  COORDINATOR: launchers TERM by deployment-cwd-matched pids ONLY.
- P1a VERDICT: NOT-REPRODUCED (partial) — B=24 505-submission point
  clean (exit 0, no route_failed, no fencing lines); one pass can't
  prove a ~1/500 intermittent; completing points (B48/32/16, B24x2,
  ~40min exclusive) queued after glm53's number. PR 735.
- Also: sweep tool's pgrep -f guard self-matches its launcher shell
  (lane's fix suggestion in their report); my current wave-fire ssh
  hold-open recurs — launcher going ssh -f detached.

## 2026-08-29 ~4:4x — ROUTE-BINDING FIX PROVEN: 45/45 layers, status 0

- The fixed transport (20539c6) on a clean 16/16 wave: every TP
  completion arrives status 0; the execution walks ALL 45 layers (logs
  show layers 43→45 completing). The async-INVALID_ARGUMENT fleet-killer
  is DEAD. (Wave required: cuda_storage failures diagnosed as surviving
  API processes holding 31.8G CUDA contexts post-TERM — the API class
  joined the teardown checklist; cwd-matched sweep + refire = 16/16.)
- REMAINING (new, later, different): final completion emit reports
  status 17 with tokcnt 0 after the clean walk — final head/MTP/
  completion-accounting class. acc 10 in the emit line is a clue.
  Handed to the diag agent with the exact live state.

## 2026-08-29 ~5:1x — CONVERGENT CLOSURE: both fixes landed; ONE hunt remains

- The original glm53 lane independently root-caused + fixed the same
  transport bug (8b116c7: algorithm-guarded RouteBinding alias —
  convergent with my remap-everywhere 20539c6; both proven: 919
  consecutive status-0 completions, 45 layers + MTP). Lane branch
  merged (d4a648f) — its guard variant is canonical (preserves the
  ring alias's purpose), my remap is subsumed. Deep lesson recorded:
  the TP4 ring alias was latently incompatible with ANY recursive-only
  degree (log2 != 3) — first TP16 family found it.
- WAVE-OWNER RULE adopted (the lane's proposal; today's EADDRINUSE
  races): ONE fleet cycler at a time; tools/glm5_next_wave.sh is the
  encoded pattern.
- THE LAST BLOCKER (both lanes converged on it): final emit status 17,
  tokcnt 0, acc 10 after the clean full-model walk — final head/MTP/
  accounting class. Diag agent has the live wave-28.

## 2026-08-29 ~6:0x — AUDIT RESPONSE: top item FIXED by coordinator; hygiene agent on the rest

- PREFIX-CACHE CONTENT VERIFICATION (audit's #1, fd84988): every entry
  now carries a SHA-256 digest of its block tokens (32B/entry); both
  FindEntry match sites compare it; all callers compute from tokens
  they hold; LIVE_ONLY placeholders (pre-token, sequence-bound — not
  the collision class) zero-digested and skipped. Wrong-KV reuse:
  silent-hash-collision -> 2^-256 non-event. Clean compile; ratchet
  reconciled.
- HYGIENE AGENT (8 items): the 2 stale-red C gates, ~11 red python
  gates, the pipeline-client flake, Makefile:699 header dep, the
  qwen38max harness merge+wiring, memlink %n + dflash2 bound as exact
  integration-request diffs (my write set), LICENSE placeholder +
  OPERATOR DECISION (license choice is yours: Apache-2.0? MIT?).
- Auditor's asymmetry diagnosis accepted: static hygiene now has a
  dedicated lane so it paces the fleet.

## 2026-08-29 ~6:3x — fleet steady; final-emit hunt live; hygiene lane starting

- glm53 wave standing (spark0 ready-line fresh); the curl still returns
  status-4 empty (the final-emit bug under hunt by the diag lane, tip
  28m old). qwen-flash retarget pushed (16-rank bf16 repackaging).
  Hygiene lane just spawned (no pushes yet). Scoreboard unchanged.

## 2026-08-29 ~7:0x — hygiene lane landing fixes (Makefile dep 3m old); probes steady

- Hygiene lane's first fix pushed (hidden-transport test module header
  dep — the stale-dylib gate). Diag lane mid-hunt (35m since tip;
  deep-debug cadence). qwen-flash repackaging. Live probe: API down
  (mid-wave cycling during the diag's work). Scoreboard unchanged.

## 2026-08-29 ~7:2x — OPERATOR: no LICENSE for now (flexibility)

- License decision: NONE for now — maximum flexibility preserved. The
  hygiene lane's placeholder item resolves to 'deliberately absent';
  reopens on operator word. (Source-model licenses still apply to the
  models themselves — this is only about SparkPipe's code.)

## 2026-08-29 ~9:0x — HYGIENE LANE 8/8 MERGED; both fenced items FIXED

- Merged (48020f9): both stale C gates re-pinned w/ commit citations
  (757e6bb MTP-draft, 84efd5b native widths — tests were wrong, impl
  right); 9/15 red python gates fixed; the flake root-caused (SIGTERM-
  vs-EOF race in residentd teardown — both exits legit; race-tolerant,
  30/30); header dep; qwen38max harness wired (+1714); make-test host
  guards.
- COORDINATOR-APPLIED IRs: IR-9 memlink template validator (a0269a4 —
  config-is-never-a-format-string; %n/%s%x/%x%u all hard-rejected,
  strict-compile clean) and IR-10 dflash2 frame bound (7a4b645 — named
  2056 constant in the shared dspark_format.h, module+kernel twin
  guards, no clamps). THE AUDITOR'S TWO FENCED-NOT-FIXED ITEMS ARE
  FIXED. Remaining IRs (ring const one-liner, LmCopyRowsKernel dim3
  REAL BUG, rdma format strings, memory-contracts inventory,
  qwen38max lane merge) queued for the next coordinator window — the
  LmCopyRowsKernel 1D-grid bug is next priority (it's a correctness
  fix, not hygiene).

## 2026-08-29 ~9:3x — QWEN-FLASH PACK-COMPLETE + HELD (merged)

- 16/16 v4 packs (bf16 repackage of warm source, quant-policy clean),
  byte-trace verified, receipts per pack; serving stack staged
  IDENTICAL on all 16 nodes; wave = one command, HELD per operator
  pause. Two latent staging bugs fixed pre-launch (TP4xPP3 geometry,
  pgrep -f teardown). Verifier gained the bf16-expert branch. spark4-7
  reservations released to the housecleaning sprint.

## 2026-08-29 ~10:2x — W4 REDUNDANCY LANE MERGED (PR737, 2dea928)

- HEADLINE: runtime/node/cache/ring have ZERO duplication — the paste
  problem is entirely modules/-side (93 hits, all triaged: 44 to W2's
  libraries, 27 kernel templates parked pending W2, 22 justified as
  per-family validation independence).
- The 157-max cyclomatic named: DsparkBlockForward (619 lines) w/ its
  own plan; 151 hotspots dispositioned, 12 named plans.
- GENERATOR PROVEN: gen_geometry_header.py regenerates qwen38_27b's
  hand-written header BYTE-IDENTICAL; glm5_next/qwen4_flash cut over
  (one real correction each — the duplicated define + the exact
  32**-0.5 scale). RECIPE-COMPILER v0 EXISTS. The token-saver is real:
  geometry headers are never hand-written again.

## 2026-08-29 ~10:5x — SPECULATION PROVIDER DESIGN (operator directive)

- All spec types supported: MTP, DFlash, DSpark, DFlash2, + the coming
  DSpark2 and successors (not every model gets today's best — the
  fleet needs the portfolio). Design: docs/SPECULATION_PROVIDER_
  DESIGN.md; recorded in GOALS (060138b).
- Shape: provider = capability unit behind the adapter template
  (kind/capability/draft-lifecycle/verify-contract/KV-interaction/env
  schema); INNER LOOPS STAY PROVIDER-OWNED — zero hot-path
  indirection, the module-ABI principle applied to speculation.
- The verify-contract slot is where the lease-advance bug class lives
  (it hit twice, per-family) — one implementation kills the class.
- Sequenced into W2: the template gains the slot this sprint (design +
  two-shape mapping: glm52_dspark module-provider + 27B embedded
  dflash2); family migration is post-cleaning with cell-unchanged
  gates; DFlash2's env contract migrates last (most receipts). W2's
  agent session ended mid-flight — the addition rides its resume/PR
  review.

## 2026-08-29 ~12:3x KST — INCIDENT: spark0 load-spiral then unreachable

- spark0 observed at load 59.59/62.35 (4 residentd procs — stacked
  generations from the wave cycles) then SSH stopped answering
  (banner timeouts, 3 attempts over ~2 min). Node did NOT respond
  long enough to capture ps/dmesg. Other nodes healthy (spark4/8/e
  idle, GPUs free).
- The glm53 fleet (spark0 = rank 0 + API) is DARK for now — the
  diag lane's hunt is interrupted; NO REBOOT issued by us (rule).
  If it self-recovers (the sparke pattern), the wave relaunches; if
  not, sysadmin. The stacked-daemon class + load spiral is consistent
  with the earlier cuda_storage/EADDRINUSE generation-stacking we
  diagnosed — wave-owner rule exists precisely because of this; the
  stack likely predates it.
- W2 dry-template ACTIVE (dsv4 cutover, 5-min-old tip, spark2/5
  reserved); W1 staging report landed (2.53T, final gaps itemized).
  Scoreboard unchanged. NEEDS USER: spark0 may need a sysadmin look.

## 2026-08-29 ~13:0x — WEDGE ROOT-CAUSE ANALYSIS (spark0 OOM confirmed by operator)

EVERY wedge this week is one class: memory exhaustion via stacked
daemon generations during wave cycling, or GPU-alloc pressure against
co-located daemons. spark0 = 4 stacked residentds (~30G CUDA context
each) = >119G = OOM. CONTRIBUTING: TERM'd daemons exit slowly (D-state
I/O), launch loops didn't verify exit before refiring, cyclers raced
(EADDRINUSE), and the 110GiB ceiling is a RULE not CODE.
SYSTEMIC FIXES (queued, coordinator write set): (1) launcher PREFLIGHT
in the shared wave tools — refuse to fire unless free-mem >= envelope
AND no same-cwd daemon alive AND a generation counter guards double-
launch; (2) fast-exit TERM handling at daemon level; (3) weights+KV
envelope computed and checked at config time. Power-cycle spark0 =
operator.

## 2026-08-29 ~13:5x — spark0 unreachable AGAIN post-power-cycle; fleet stood down

- After the operator power-cycle, preflight PASSED (16/16 clean,
  107-110G free) and the wave fired — ranks 1-15 reached fabric-ready
  then route_failed (waiting on rank 0); spark0 went unreachable
  within ~3 min of launching its daemon. HYPOTHESIS: post-hard-cycle
  ceph recovery (mon.spark0 + osd.0) + the residentd's ~30G alloc
  raced — the preflight checked free memory but NOT ceph-recovery
  state (a gap; added to the preflight design).
- STOOD DOWN: 15 ranks TERMed by cwd (fleet dark, clean); NO further
  launch attempts until the operator confirms spark0 stable — the
  cycling-it-until-it-works pattern is exactly what wedges nodes.
- FLEET NOTE: post-power-cycle nodes hosting mon/osd need a settle
  window before GPU work; the preflight gains a ceph-recovery check.

## 2026-08-29 ~14:3x — spark0 STILL unreachable (2h+ post-second-failure)

- No recovery; holding the stand-down (no launches). W2 dry-template
  active (dsv4 cutover, spark2/5 reserved). NEEDS USER/SYSADMIN: spark0
  requires console attention again — after the second post-cycle
  failure with ceph-recovery racing GPU alloc, suspect (a) the node's
  ceph roles need settling/re-homing (mon.spark0 + osd.0 on the
  fleet's API node is structural risk — the MacStudio MDS move
  should extend to the mon), or (b) hardware. The glm53 fleet waits
  on spark0's stability; no cycling from us.

## 2026-08-29 ~15:0x — SPECULATOR PORTFOLIO ARRIVED (11 sources + license notes)

- Warm now carries 11 new model/speculator sources w/ pinned revisions
  + LICENSE-ADMISSION-v2 notes (policy: reject noncommercial; accept
  grants above business scale; permissive-upstream repos recorded):
  glm5.3-nvfp4-radixark (465G, Z.AI commercial grant), qwen3.8-27b-fp8
  OFFICIAL (Apache), 27b-dflash2-incoai, kimi-k3 dspark x3
  (redhatai/inferact/radixark) + k3-dflash2 x2 (lightseek/modal),
  dsv4-flash-dflash-redhatai, qwen-max dspark-radixark +
  dflash-modal, glm-5.3-flash-dflash2.
- THE PORTFOLIO IS COMPLETE ENOUGH FOR THE BAKE-OFF: 5 target models x
  2-4 speculator options each. Every source satisfies the quant/
  license policy as recorded (vetting receipts to be re-verified at
  first use per policy).
- NEXT (when the operator pause lifts + spark0 restores): the
  provider-abstraction bake-off harness — head-to-head per-spark
  acceptance + tok/s per (model, speculator) pair, feeding the
  tournament-provider decision with data.

## 2026-08-29 ~16:4x KST — spark0 RESTORED (3rd power-cycle)

- Up 3 min, 110G avail, ceph mon+osd both Sl (recovering), driver
  580.159.03 up, NO residentd actually running (earlier count was the
  pgrep shell itself). SETTLE WINDOW in effect per the post-cycle
  doctrine: NO wave until ceph recovery quiesces + the operator's
  go — this node burned us twice exactly here (recovery racing GPU
  alloc). The glm5_next relaunch is staged and waiting.

## 2026-08-29 ~17:1x KST — glm5_next FLEET RESTORED 16/16 (settle honored)

- spark0 settled 23 min (ceph Sl, warm 80MB/s) → full-16 preflight
  ALL CLEAN → wave fired → READY 16/16. API up; the first-token curl
  still status-4 (the final-emit bug — the diag lane's hunt resumes
  on live state). W5 NOTE: /v1/models returned "not found" — the
  deployed spark0 api binary predates 11260f4 (the /v1/models +
  error-shape build); W5 live-verify needs the api rebuilt+redeployed
  from main (queued — the wave's binaries came from the lane's staging).

## 2026-08-29 ~17:4x — fleet steady 16/16; lanes quiet (work in flight)

- glm5_next fleet holding (spark0 ready, API ok/served). W2's tip 2h
  old — the dsv4 cutover is the big one (validation builds run long);
  worktree reservations active (spark5). Diag lane quiet 3h — the
  final-emit hunt awaits its next window or the coordinator takes it
  (the live fleet is the repro). Scoreboard unchanged.

## 2026-08-29 ~18:1x — FINAL-EMIT HUNT: two mechanisms named

Status 17 = INTERNAL_ERROR (enum value 17; the emit's matches==1 —
the DRIVER itself failed). Route_failed status=1/reason=2 DECODED:
the completion violates the WIRE CONTRACT at model_serving_adapter.c
line 536 — a non-OK completion carrying accepted_token_count=4 (the
adapter copies driver_completion->accepted_token_count
unconditionally) is INVALID_ARGUMENT; the residentd rejects the
STRUCT before reading the true status, masking the real failure.

FIX-1 (adapter, glm5_next lane write set — unmask): zero
accepted_token_count + flags when status != OK. Every failure becomes
readable; the true status then reaches the client.
FIX-2 (the root cause, next hunt): why the driver returns
INTERNAL_ERROR after the clean 45-layer walk with acc=4 — partial
progress then failure; likely finalization (head/MTP/emit accounting).
The two conspired: FIX-1 alone won't produce tokens but without it no
failure is ever diagnosable through the wire.
  - HANDOFF: diag-lane agent not currently active (session window) —
    the finding + FIX-1/FIX-2 assignments are in this log and the
    report; the next diag-lane spawn or coordinator window executes.

## 2026-08-29 ~21:0x — SWEEP AUTOMATION STOPPED (operator)

- The 30-min sweep cron is DELETED — it was producing cadence-driven
  noise instead of judgment. The operator pings to keep on track;
  agent completions trigger self-prioritization instead (plenty of
  context exists in GOALS/HOUSECLEANING/ledger).
- HOUSECLEANING STATUS (honest): W3 hygiene 8/8 MERGED; W4 redundancy
  MERGED (PR737); W1 staging DONE+merged (2.53T, manifest tool in
  gate); W2 DRY-template IN FLIGHT — its dsv4-cutover commit is 2h old
  and the worktree went quiet ~18:03Z; NOT confirmed done. Next
  coordinator action on any wake: check W2, finish or re-task it.
- QUALITY STUDY queued in GOALS (full-res vs quant-spine+full-spine
  vs fully-quant, via ds4_eval 92x) — explicitly AFTER basics.

## 2026-08-29 ~21:2x — IDLENESS CORRECTED: W2-continuation + first-tokens agents spawned

- Operator caught agents idling with 27 defined tasks remaining —
  session-end passivity, not work exhaustion. CORRECTED: W2 items 2-4
  (+ provider slot) agent spawned; glm5_next FIX-1→first-tokens agent
  spawned (fleet live as reproducer). K3 pack build checked.
- Doctrine going forward: a lane session ending does NOT park its
  backlog — the next spawn carries the remaining items immediately.

## 2026-08-29 ~21:4x — NVFP4 WAVE ARRIVED (the precision portfolio)

- 11 new NVFP4 checkpoints on warm + NVFP4-MODEL-DEFAULTS-v1 notes:
  every major model now has an NVFP4 variant (dsv4-flash + pro,
  glm5.3 + 5.3-flash, k3, qwen27b ×4 quantizers, qwen-flash, qwen-max).
  POLICY in the notes: NVFP4 preferred, MXFP4 skipped when NVFP4
  exists (supersedes the earlier MXFP4-for-qwen-max plan), AWQ/GGUF
  excluded; QUALITY BOUNDARY = quantize routed experts/dense MLP ONLY,
  preserve spine/lm_head/MTP at source precision; full-res-spine
  variants present (qwen27b nvfp4a16-bf16-spine, qwen-max nvfp4-
  bf16-spine) enabling the 3-way quality study; every entry
  download-verified w/ pinned revision + license basis; runtime
  accuracy/perf honestly 'not_tested'.
- FLEET IMPLICATION: NVFP4 kernels become first-class (currently only
  probed; MXFP4 in-tree). The e2m1 dequant machinery overlaps; the
  grouped-expert kernel paths need an NVFP4 lane once the basics land.
  The island catalog's precision-variant axis is now fully populated.

## 2026-08-29 ~22:0x — BULK STAGEPACK agent launched (operator directive)

- Long-running data lane: ALL new sources → stagepacks → all sparks.
  Priority: speculators first (the bake-off is next), official 27B
  FP8 second, NVFP4 variants third (27B's four for the quality study,
  then the rest). Existing family packers only (repackage-never-
  quantize); NVFP4 natural-format gaps get RECORDED not converted
  (kernel-lane dependency); verify+receipts per pack; staging manifest
  stays the checked state; builds on spark4/5 (wedge-route pattern).

## 2026-08-29 ~23:4x — FIRST TOKENS (structurally): status 0, emission plumbing works; VALUE bug remains

- The unmask+emit-fix chain (merged 9cd3f61) WORKS: completions now
  status 0, tokcnt 1, tokens flow api→engine→client→residentd→adapter→
  45 layers→head→maxloc-reduce→D2H→completion→IPC (token_ids round-trip
  verified in the wire encode/decode)→engine AcceptToken→EVENT_TOKEN→
  the API's json. THE WHOLE STACK SERVES.
- REMAINING: emitted values are all ZERO (deterministic across calls).
  Emit line: status 0 flags 1 tokcnt 1 tps 1 acc 1 — one token, zero
  value. The head path: embedding→...→HcHeadMean→RMSNorm→HeadCandidate
  →HeadCommit→MaxlocPack→U64Max-reduce→Unpack. Candidates ranked:
  (1) the maxloc pack reads slot->output_score/ output_token written by
  HeadCommit — but the REDUCE_HEAD stage unpacks chain->wave_rows from
  head_maxloc_u64 whose pack ran on the same slot buffers PER-CHAIN;
  first_row offsets between pack (rows in chain) vs final D2H (base)
  mismatch would read ZEROS from the calloc'd staging tail;
  (2) embedding kernel writes value=0 when token < rank_offset — rank0
  owns tokens < vocab/16 so OUR tokens (154819 etc.) live on ranks 7-9;
  the CROSS-RANK embedding gather (each rank zero-fills outside its
  shard, then hidden allreduce) must run BEFORE layer 0 — verify the
  per-layer residual reduce includes the embedding buffer or an initial
  allreduce exists; if the embedding shards never sum, hidden stays
  ~zero on 15/16 ranks → head produces zeros on the head-owning rank.
  THIS IS THE PRIME SUSPECT: the initial hidden allreduce after
  embed-sharding appears MISSING from the chain stages.
- Hand to next agent window or coordinator: check for an initial
  hidden all-reduce (stage EMBEDDING_REDUCE) in the TP chain; the 27B's
  chain has one (its embedding is sharded the same way).
