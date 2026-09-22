# MGR2 HANDOFF — 2026-09-21 (post-compaction continuation doc)

Read this FIRST, then sparkpipe-coord/SHARED_DECISIONS.md tail (the full
ledger) and MEMORY.md. This doc carries everything the next session needs.

## MISSION + OPERATOR DIRECTIVES (the binding stack)

1. GPU-DEBUG READINESS: all drivers working, all B* (batch sizes), all
   topologies, all quantizations; every pack verified ON NVMe; ZERO ceph
   dependency when GPU debugging starts; all PRs merged properly.
2. NATIVE SPEED determines the ceiling — hill-climb per family is the core
   program. Speculation = PARALLEL project (2x+ later; kimi draft backends
   deprioritized; #1073's spec plumbing stays as landed infra).
3. QUALITY LAW: quantized expert weights + full-resolution spine; models
   with per-layer quantization subsets get NATIVE mixed-codec support
   (laguna); BF16 layers stay byte-identical to source; NO requantization
   to uniform anywhere, ever.
4. "it is there in warm storage" = warm reads are sanctioned for emission/
   census/reference work; outputs land on NVMe + git so GPU debugging never
   touches ceph.
5. GATE CONDUCT: never push to another dev's branch refs (operator
   call-out); verdicts are comments; force-with-lease only on mgr2's own
   dispatched-agent lanes; merges need a FRESH checks read naming the
   current head (three red-on-main incidents: #1045/#1053/#1056-class).
6. Timer STOPPED (the 30-min lane check automation deleted) — work is
   event-driven on completions.

## PROGRAM TOTALS

52+ merges this program (T1 wave #1037-#1071 + arch wave #1002-#1009
incl. 7 recovered-from-clobber + #1036 resilience + #1042 sweep +
#1046 emission-1 + #1051 feed-fix + #1055 regen + #1065/#1066/#1068/
#1069/#1070/#1071/#1072-pending/#1073/#1074 + coredev's #1067 regraded/
merged + #1057/#1054 ledger restores). Main GREEN at last verify.

## FAMILY VALIDATION STATE

- BYTE-EXACT/DETERMINISTIC references on main (post-#1051 feed-fix
  regeneration, prompt-identity cross-checks PASS): ling, laguna (streaming
  regen pending merge #1070 — MERGED), gemma4-31b, gemma4-26b, glm53full
  (streaming regen in #1070), muse, minimax, dsv41 (streaming regen in
  #1070), qwen38-27b both arms, k3 (merged #1040: KDA oracle 0.0,
  hand-layer-0 bit-identical, "Paris" 18.11), glm53flash (#1071: 5 arms
  mechanical + CPU reference engine + fixtures + 3 defect fixes),
  qwen3flash fp8 (tokens-exact). qmax: t1-offline fixtures STAND (rerun
  parked: engine blocking pathology — fix in PR #1072).

## IN-FLIGHT (agents running — collect their PRs)

1. Module codec wave 3 (re-dispatched): laguna per-layer mixed codecs +
   dsv41 nvidia convention + emit laguna fp8/nvfp4 + dsv41 nvfp4.
   Salvage: /Users/mac/sparkpipe-wave3 (prior agent's partial worktree).
2. Topology emission: the 9 skipped topologies (gemma-26b tp16 excluded —
   in-session loop on sparkc ~/topo-wave/g26tp16). Salvage:
   /Users/mac/sparkpipe-topo (preserved/wave3-partial branch).
3. minimax text-tower DRIVER (the family had none): mgr2's geometry
   header on lane/minimax-driver; agent ports the qwen4_flash module
   pattern; compile-gated; runtime execution-gated.
4. hy4: chat-template A/B agent running (spark1 ~/hy4-sinkprobe BOS loop,
   spark5 ~/hy4-fixtree fixture loops). The A/B decides: bare-prompt OOD
   (fixture convention) vs engine defect. post-fix evidence: whitespace
   collapse SURVIVES sinks/rope/feed fixes.
5. #1072 (qmax fix): root cause = the refs2-era dead 80GB cache + a 4GB
   single lm_head read; fixed engine 947MB plateau; awaiting CI + merge.
6. #1075: MERGED — coredev executed the full gate list (per-state
   preallocated overflow pool, retitle/rescope to the serving-ledger
   line, ledgers, the S2.5 payload-vs-control-plane distinction
   documented, #1014 closed superseded) and landed the V6 MESH PROTOCOL
   + THE FIRST COMPLETE 16-RANK GRAPH CHAIN (the S3-class delivery).
   The allreduce ladder (MEASURED): S0 1910-2700 -> S1 1334 ->
   **FIRST COMPLETE 16-RANK GRAPH CHAIN: status=0, 91 rounds,
   330us/round MEASURED (4x over S1)** -> next = relay burst pacing;
   the <=100us GOAL receipts follow. Coredev is landing continuously —
   re-read main's log before every action.
7. Emission relay: spark7 rank-7 stragglers (gemma-31b, lingfin,
   qwen27b-fp8 tp4pp4) building on sparkc ~/relay/wave1/ship_spark7.sh.

## THE OPEN QUEUE (everything red, owned)

- Module codec decisions RESOLVED by the quality law: laguna mixed-codec
  + dsv41 nvidia convention = wave-3 agent's scope (above).
- minimax packer: DONE (wave 2 wrote it; minimax.text.bf16.tp4 placed).
- glm53flash: #1071 merged (the family validated per operator order;
  node-f finding RETRACTED — stale .restore-rankf leftover; fp8.tp8
  rank6 pre-#877 defective generation still placed → re-emit; nvfp4.tp16
  node-f unpinned + spark8 stale sidecar → hygiene).
- qwen27b MTP-strip corruption: REPAIRED (wave 2, all 16 surviving packs,
  byte receipts).
- route.cuh markers: ROUTE ROW INDIRECTION restored via #1065; the
  cudaLaunchHostFunc glm52 question resolved STALE-TEST (#1065: the call
  moved to the common header; test greps the primitive now).
- mimo 2.6 pro + flash: download NOT yet visible on any warm mount;
  drivers/stagepacks queued behind minimax driver; model-families/mimo25
  is the pattern precedent.
- qwen-image-2.1: lane/qwen-image-driver (ARCHITECTURE.md + census
  landed — 297 tensors all BF16, dual-stream MMDiT; the fleet's FIRST
  flow-matching diffusion family; ONE diffusion resident stage design
  serves it AND the minimax-h3 video stack; the design doc is the next
  deliverable before the module).
- #1014: CLOSED superseded by coredev with the verdict comment.

## PROCESS LAWS (binding — each paid for with an incident)

- Ledgers (PACKAGE_MANIFEST.json + SHA256SUMS) resolve by WHOLESALE
  REGENERATION (tools/generate_*.py), never hunk-merge; conflicts repeat
  PER COMMIT — loop regen+continue; manifest self-digest hashes AFTER
  final content; total_bytes moves with size changes. NEVER sweep
  untracked files into the regen (the .mimosa/1000HSEM franken-tree
  class); keep gate logs OUTSIDE the tree.
- Rebase gotchas: `--theirs` = the BRANCH side; a resolved rebase can
  stop needing `--continue` silently (git rebase --quit + branch -f HEAD
  + force-with-lease); a rebase can AUTO-DROP commits as empty via bad
  auto-resolution (the #1005 KEY_SPACE_H case — grep the substance
  in-tree before trusting a clean rebase); prefer CHERRY-PICK of the
  content shas when a branch is clobbered.
- Branch clobber recovery: the content shas survive in gate receipts/
  `git log --all --grep` — worktree at the sha → rebase → regen →
  force-with-lease → `gh pr reopen` → CI → merge. Audit branch refs vs
  receipt shas before assuming a closed PR is dead.
- Merges: FRESH checks read naming the CURRENT head sha; after any
  force-push re-poll. The auto-merge loops are RETIRED (three
  red-on-main races). Post-merge, verify the tip's check-run = success.
- CI gate exits: exit 1 fast = a python check phase (regenerate FROM the
  contract, never hand-edit generated files — #1019); exit 2 after
  minutes of nvcc = silent link/compile death (diagnose from artifacts,
  or reproduce on sparkb). Actions stalls intermittently — the LOCAL
  GATE PRECEDENT: run the exact gate natively on sparkb (GB10 = sm_121a,
  CUDA 13.0 at /usr/local/cuda/bin; full run ~8 min; artifacts upload on
  failure carry the failing log — download via `gh run download`).
- Shared-worktree hazard: concurrent agents switching branches in a
  shared base tree clobber each other (the wave-2/#1071 file-loss class).
  Each agent gets its OWN worktree; verify final trees contain the
  agent's files before assuming completion.
- pgrep gotchas: alternation `\|` matches nothing (false DEAD);
  patterns self-match the probe's own cmdline (false writer counts) —
  census with `ps -C python3 | grep <unique>`; ps etime on nohup
  children reflects the DELAYED actual execution (background-task
  serialization), not issue time.
- 1GB tooling RSS law (operator, coredev-escalated): decoder launches
  wrapped MemoryHigh~1.2G/MemoryMax~1.5G; T1_REF_WEIGHT_CACHE_BYTES
  unset; non-fitting engines get slab-streaming (the muse zero-alloc
  pattern) or ENGINE-NEEDS-STREAMING honestly listed. All three whales
  FIXED and merged (#1070): laguna 536MB, dsv41 809MB, glm53full 1096MB.
- Warm-mount physics: no cache = every position re-reads all weights
  (~10-15 min/position cold); D-state = ceph client stall (restart that
  reader; three-strike rule moves the lane to a healthy node — spark2's
  mount is degraded); spark0's respawning weightd = coredev's own
  fleet_node_agent.sh loop (his to stop).
- Stagepacks-vs-warm: reference engines read WARM (provenance pinned by
  config/index sha in each MANIFEST); packs are DERIVED per-rank shards
  verified mechanically ON NVMe (#1042 matrix + #1066 tool fixes +
  re-runs: P 253+324 tool/sha, F 30, NT 0, M 33).

## INFRA

- Agents: spawns work (the account usage limit broke them 09-20/21,
  reset 09-25 20:56 — retest the gate on every dispatch round; do not
  assume). Subagent reasoning level: fixed via ~/.zcode/v2/
  agents-state.json (the stale legacy override map removed;
  backup agents-state.json.bak-0919) + a client restart.
- Weightsd: STOPPED fleet-wide per the operator's FULL STOP (the timer
  brief's "restart if inactive" clause is stale). The hourly flap was
  ROOT-CAUSED (the node agent's kill patterns; fixed in #1036) and
  structurally closed (#1052 mesh-dir split). Do not restart without the
  operator's word.
- spark0 hygiene: coredev's fleet_node_agent.sh (glm53flash.fp8.tp16
  lane) respawns the production weightd — his to stop; the root-owned
  foreign weightsd1 was killed via sudo; /tmp/weightd-mesh reset, then
  #1052 made record dirs per-deployment anyway.
- spark2's warm mount: degraded client (two D-strikes) — excluded from
  decode duty. spark1/spark3 mounts healthy. Warm throughput oscillates
  0.2 MB/s-1 GB/s fleet-wide (the single-consumer law keeps parallel
  readers in the slow band).
- #1067 (coredev allreduce hillclimb): MERGED after the regrade (the
  anti-fraud gate held: the title promised ≤100µs delivered; the doc is
  a plan; S1 1334µs measured is the standing number).

## NEXT ACTIONS (in order)

1. Collect the in-flight agent PRs (gate each: fresh CI/reads + the
   checklists in the briefs) and merge.
2. #1072 merge on CI green; then the qmax lane closes.
3. #1075 merge when coredev's fixes land green.
4. After the topology/codec/emission agents land: re-run the affected
   sweep rows; update the coverage docs; the complete-set tally moves to
   every model × quant × ruling-matrix topology, all NVMe-verified.
5. The minimax + qwen-image diffusion driver design docs precede their
   modules (the diffusion resident stage serves BOTH qwen-image-2.1 and
   the minimax-h3 video stack — the architecture record is landed on
   lane/qwen-image-driver).
6. When GPUs reopen: the EXECUTION-GATED lists in #1073/#1074/#1075/
   #1072 PRs are the measurement queue; the B*/native-speed hill-climb
   ladder resumes per the strategy stack.
