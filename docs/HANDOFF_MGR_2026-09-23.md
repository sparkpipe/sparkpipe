# MGR HANDOFF — 2026-09-23 06:41Z (timer stopped, all lane agents quota-dead until 14:39 local)

Read this FIRST, then tools/devcycle/lane_assignments.json CONVENTIONS (on main — binding), then
docs/MULTIDEV_QUICKSTART.md, then #1161 (the infra ledger). This doc carries the session's state.

## MISSION STATE

Goal: SOTA non-speculative per family; 3.5x community TP4 on the 16-Spark ring. Strategy
(operator-set): smoke tests + small B* first; exclusive fleet only for critical benchmarks
AFTER drivers work. ~49 driver x topology x quant combos have packs on NVMe; the matrix
fills one tok/s at a time.

## SCOREBOARD (honest)

- **Measured decode rates: GLM 5.3 flash FP8 TP16 = 12.81/11.77 tok/s isolated, 1.66-1.71
  each / 13.30 aggregate 8-way** (the PR #1082 campaign receipts — still the standing numbers).
- GLM concurrent 4-way on the NEW stack: 3.10-3.13 tok/s aggregate (hillclimb lane, attempt
  da89044b, PASS, valid).
- **Module tokens banked:** qmax 34020/107300 bit-exact (ranks 0+6, through the shared daemon).
- **Deepest receipts:** laguna residentd-ready + COMMIT-TRACE (011f); k3 request-accepted +
  slot-allocated (cold16); minimax HELLO_ACK + reconnect cycle (run9). NO family has served
  decode tokens yet — every chain reached its named final layer.

## MERGE STATE

- Main at 8608966a; ~100 session merges (659 commits incl. lane-side). EVERY family's final
  named fix is merged: #1185 laguna PP + #1191 laguna frame, #1190 k3 width contract (OPEN,
  gate-green), #1193 minimax reset, #1194 qmax MTP-skip, #1186 gemma4 CUDA link + #1183
  layer-scalar, #1189 ling collective shape + #1188 spine budget, #1192 qmax digest cache.
- OPEN PRs: #1195 (minimax decode cell), #1190 (k3 widths — MERGE ON GATE GREEN), #1179
  (gap tooling — RED, needs the regen fix), #1076 (parked for astra).

## EACH LANE'S NEXT ACTION (first thing on resume)

1. **qmax (lane 2)**: attach-j fires into the clean epoch; receipt client → serving tok/s.
   Merged stack complete (#1178+#1184+#1192+#1194). Honest budgets proven.
2. **k3 (lane 3)**: #1190 merges on its gate receipt → cold18 (honest device decl ~11264,
   see the budget ruling) → tokens. The width contract is the answer to cold16's diagnosis.
3. **gemma4 (lane 6)**: launch-16 fires in the next clean window (launch-15 proved every
   layer 16/16 ready; first-tokens driver staged).
4. **laguna (lane 8)**: 011g pipeline armed — on #1191 MERGED (done: 8608966a) it syncs,
   attaches, decodes. The frame-fix was the last named layer; expect tokens or the next
   precise failure.
5. **ling (lane 9)**: r10 decode races the post-k3 gap; attach proven twice; both fixes merged.
6. **minimax (lane 10)**: #1195 gates, then serve-run13 (the in-job decode cell) → tokens.
7. **gap bot**: g5p spine repair chain (pro r4-r7); then lingfin resume, dsv41 rank2, matrix
   update. #1179 needs its regen fixed and re-gated.
8. **hillclimb lane**: rates branch (f09124e5) — single-seq + TTFT decomposition pending.

## THE RIG (operator item #1)

`~/wdcore/sup.sh` on ALL 16 sparks: a core-dump reproduction rig, driven from 10.20.0.1 over
sshd sessions alive since Sep 18 (55s health-check cycles). Blocks all gpu-shared admission
(unaccounted GPU census), crash-loops shared units (latch squatting), and KILLED gemma4's
launch-15 fleet-wide. Cleared twice (simultaneous kills); evidence preserved in ~/wdcore/.
**THE SOURCE SESSION IS STILL LIVE — it will relaunch.** Kill signature: pkill -TERM -f
"my_weightd_run|sup.sh" fleet-wide simultaneously, then verify with a self-match-safe count.
Full identification on #1161.

## FLEET STATE

- All 16 shared weightd daemons: mesh16 binary (f09074e7), 90 GiB arena ceiling, active,
  tracked at 29184/41472 (the consistent values). One coherent epoch since the 14:06 wave
  (spark1's missed start fixed at 06:20Z).
- Controller: mac@mac-studio, /Users/mac/wk-sparkpipe-queue-main-20260922. The queue
  hotfixes (weightd census exemptions) are IN THE TREE and pinned (assume-unchanged +
  .hotfix-pinned backup) — git operations reverted them 4x tonight; RE-VERIFY after any
  controller checkout. Durable fix = a repo PR carrying the exemptions (astra).
- GPU-cell serialization RULE: two rank-pinned 16-node cells cannot co-run (mutual collective
  destruction — #1142 evidence). One clean cell at a time; first-merged-and-staged wins the
  window. The queue has no mutex for this yet.
- Honest-budget law: device declaration >= mapped pool span + spine + KV + context, by
  construction. Under-declaring blocks the whole fleet's admission (~14 min per incident).

## PROCESS LAWS (all paid for, all in the conventions)

Regen-last taxonomy (6 variants); committed-content verify via git cat-file; kind=run ALWAYS
(gate is a black hole); explicit --resources (exclusive-default starves); controller-ledger-
only for queue truth; single-job cells for arm+client (the two-job split cannot co-schedule);
no shared .git/config writes (identity corruption — per-invocation env only); worktrees never
share branches; the PR-only law permits testing on open PR branches (launch-from-branch is
the standing acceleration).

## OPEN ITEMS FOR THE OPERATOR (in priority order)

1. CUT THE RIG AT SOURCE (10.20.0.1 sessions — it destroys token runs).
2. The three design decisions for astra: #1138 (dense attach), #1142 (arena admission +
   cell mutex), #1155 (arena-basis pinning — blocks GLM serving instance_ready).
3. spark3/spark7 ceph clients (MDS plane degradation); spark1 client dead.
4. Workflow-scope token (#1124 — CI evolution blocked; Actions silent since 03:53Z, all
   qualification is on-fleet gates now).
5. The wipe provenance (dsv4_pro pre-deletion investigation — parked).
6. CCN lint regression (qwen38_27b, 88>75) — hygiene pass.

## AGENT MANAGEMENT

All 7 lane agents + gap bot quota-dead at ~13:41 local; reset 14:39:32 local. Respawn with
state-carrying briefs (the worktrees survive; every lane's state is documented above and in
the PR/branch state). 12+ agent deaths this session absorbed by the respawn protocol. The
lane-2 ownership lesson: ONE owner per lane; predecessors stand down and hand off named files.
