
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
