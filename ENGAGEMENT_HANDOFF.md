# glm5.3 Flash tree-transport engagement handoff — 2026-09-09

For the next dev with fresh eyes. Written after a two-day sprint on the
tree allreduce serving path, plus a survey of the 176 commits codex-astra
landed on main in parallel (PRs #867-#896). Read `docs/DEVCYCLE.md` for
the build/release loop, `docs/BUG_LEDGER.md` for the systemic patterns,
`docs/GLM_PERFORMANCE_GATES.md` for what a valid throughput claim needs,
and `docs/FLEET_STARTUP_PROTOCOL.md` for the registrar design (the
structural answer to the mesh-flap problems described below).

## Where inference stands (the honest numbers)

- E2E serving works on the tree: API → real tokens → status 0. Best
  verified: **0.585 tok/s at B1** (T8 22.8s / T32 63.8s), on the
  GPU-dataflow build. Parked-NCCL reference: 12.2 tok/s. The gap is
  execution model, not the wire (see "why slow" below).
- The GPU-dataflow refactor is live in my lane builds: chain advances
  inline after submit, per-credit op streams, device wait-kernels gate
  the compute stream on done-flags, host is out of the per-op callback
  path. It bought 0.50→0.585 — real but small.
- Per token: ~91 bf16 allreduces (2/layer × 45) + 1 u64 head-max.
  Roofline says bandwidth is irrelevant at B1 (collectives ~1MB/s vs
  50GB/s wire; weights ~33MB/s vs ~250GB/s HBM) — it is 100% latency
  and synchronization.
- WHY SLOW (measured, not theory): the tree op is a CPU-mediated
  4-stage state machine. Each stage transition needs the progress
  thread to notice an arrival, enqueue a fold, round-trip a
  `cudaEventQuery`, post the RDMA send. ~364 loop-mediated transitions
  per token × the loop's period ≈ the observed ~1.5-1.7s/token.
  NCCL does the whole collective as ONE stream-ordered kernel enqueue
  (~5µs host) — protocol stages execute inside the GPU pipeline. The
  operator's verdict is adopted as law: CPU-side orchestration cannot
  match NCCL; the fix is removing loop-mediated transitions, not
  tuning them.

## The two known live defects (both root-caused, fixes designed)

1. **Ordinal desync poisons serving after any failed/partial request.**
   Ordinals are minted per-rank by fetch-add; the exact-nonce wire
   protocol requires fleet-wide lockstep. One timeout leaves ranks'
   counters skewed → every later op's nonces never match → 30s
   OP-FAILs on everything until a fleet restart. This is why
   fresh-cycle first requests always worked and later ones failed.
   TTRACE evidence: op-0 arrivals never fire (a0 unset), pack never
   runs. FIX (also independently opened as PR #823 by another lane —
   check its state first): ordinals derive from the chain's dispatch
   base (identical on all ranks) + per-chain op index; ack gate
   compares recorded last-posted ordinal per (route, credit) instead of
   assuming contiguity. PR #823 says fleet bench validation is owed —
   do not merge without it.
2. **Wedged-daemon zombies survive TERM.** A daemon stuck in the
   admission spin ignores TERM and its alarm(60); it then holds ports
   and its stale log misleads every later diagnosis ("prior residentd
   not exited; NOT starting new" in the fleet-agent journal is THE
   tell). Law: check daemon AGE (`ps -o etime=`) before believing
   logs; sweep with kill -9 by /proc exe link.

## My lane state (branch lane/glm53-tree-2, NOT merged — main moved past it)

Last good commit: 6bc749b (slot-bench series). Everything earlier of
value was merged via PR #811 (see below). The branch's post-#811 work:
file-rendezvous transport, GPU-dataflow engine, TTRACE, slot bench.
IMPORTANT: main has since taken its own path (numerical fixes
#867-#896 + PR #823's own ordinal fix). Treat my branch as a design
reference, not a rebase candidate — port ideas, don't merge.

### Landed on main earlier via PR #811 (1a82e36)
Provenance gates deleted everywhere (runtime checks structural identity
only; orchestrator warns `mixed_versions` instead of failing); UPDATE
two-phase release protocol; exe-based stop sweep; hot-path print strip;
MTP gate (`state->mtp_active`, wide hc collective skipped when 0);
repo-driven build loop; `docs/DEVCYCLE.md`; gcc-15 `_POSIX_C_SOURCE`
purge. Pack ceremony deleted (validators synthesize weights; only dsv4
loads a real pack).

### In my branch, designed and half-proven (port the ideas)
- **File rendezvous (QPN publish/subscribe)**: each session publishes
  {boot_id, lane QPNs/PSNs/GIDs, identity, fixed-recv descriptor} to a
  shared dir; peers poll. Implemented against `/mnt/qpn` (NFS export
  `/srv/qpn` on rtx5090, mounted soft on all 16 sparks — this is set
  up and working). Order-independent wire-up, no TCP listener, no port
  tables, hot rejoin via boot_id change. Got 16/16 ready once; serving
  then hit the stale-record problem (152 leftover .rec files wired
  dead QPNs) — fixes: unlink-on-destroy (done), mtime TTL (done),
  clearing records on release (done in build script). STILL OWED: one
  clean serving run on the rendezvous build. Design note: awaiting the
  peer's fixed-descriptor inside Initialize DEADLOCKS the fleet (peers
  publish it only after their own Create returns) — the learn must be
  lazy, in `PersistentRemoteCreditReady` (this is in the branch).
- **TTRACE instrument** (env `SPARK_TREE_TRACE=1` via
  `env.local` in the runtime root; agent sources it): per-op line with
  submit/arrival-per-stage/terminal/callback CPU ns plus
  producer→pack and pack→result GPU µs (event pairs). This is how the
  ordinal desync and the 364-transitions/token were caught. Deploy via
  `tools/module_build_release.sh`, agent `start_root` reads env.local.
- **Slot bench** (`tools/slot_bench.cu`, branch only): one process, 16
  in-process ranks, REAL engine + REAL transport, serving-faithful
  serialized slot (producer kernel → submit → device wait-kernel →
  callback), serving-shaped one-block-per-row fold. This is the
  honest yardstick the operator demanded after the 66µs bench proved
  fraudulent (it pipelined back-to-back submits against pre-posted
  buffers — impossible in serving). Status: compiles clean; blocked at
  runtime by sparkf GPU OOM (fleet holds the GPU) — run it in a
  parked-fleet window. Target: reproduce ~0.5-0.6 tok/s-equivalent,
  then hill-climb per-op slot cost there.
- **CQ batch poll fix** (landed): `ibv_poll_cq` was handed batch=1
  against a 32-slot array; now batches properly.

## What codex-astra did in parallel (main #867-#896) — the context you need

The numerical-accuracy campaign: KDA norm/query-scaling/gated-output
fixes (#874), SwiGLU limits everywhere (#873), embedding reduction
(#872), paired expert-projection sharding (#877 — the packer was
slicing concatenated stacks, giving a rank only-up or only-gate rows;
75% relative-L2 error at layer 4; old TP packs must be REGENERATED),
rank-dependent TP reduction (#885), deterministic subgroup folding
(#886), logical batch policy across splits (#887), BF16 boundary
preservation series (#891-#896: RMS semantics, conv rounding before
SiLU, shared FP32 tree primitives, write-gate rounding, credit sizing
#895), full-checkpoint first-position reference with real-token
capture (#875/#879/#880), EOS required (#882), queue CPU
de-serialization (#878). PR #888 records: clean-main TP16 deployment,
layer-4 relative L2 improved 1.212%→0.607%, cached decode B1-B8
receipts, plus a CUDA context-allocation failure+recovery. Their docs
(GLM_FLASH_HILLCLIMB.md, GLM_PERFORMANCE_GATES.md, GLM_NUMERICAL_GATES.md)
are the source of truth for the numerics state. IMPLICATION FOR MY
LANE: every perf number I took was on packs/driver built before these
fixes — do not quote my 0.585 against their fixed builds without
re-measuring.

## Fleet ops state (as of handoff)

- Agents (systemd user unit fleet-agent) run the UPDATE protocol:
  touch `rtx5090:release/glm53flash.fp8.tp16/UPDATE` → nodes pull,
  append `down:host` after daemon exit, all-16-down gates starts,
  `up:host`, rename UPDATE.<n>. Idle = one ssh existence check, zero
  rsync.
- FLEET VIEW: the LIVE view is `sparkf:current/*.json` (one ssh
  answers fleet state). `rtx5090:current/` is a STALE copy (~25h) —
  the other lane was reading the wrong one.
- Build loop: `tools/module_build_release.sh <family> <codec> <root>
  <revision> <contract.json> [branch|sha]` on ANY spark (aarch64+GB10
  required; rtx5090 cannot link the driver). Build host sparkf
  (~/sparkpipe-build). Ships driver + transport DSO, clears stale qpn
  records, drops UPDATE. PARKS the local daemon for the GPU validator.
- The fleet was last seen 4 down / 11 ready / 1 starting with a LATCH
  on spark0 (ordinal desync signature). A clean UPDATE cycle (or the
  PR #823 fix deployed) restores it.
- weightd: NEVER kill (holds every model's packs). One per node.
  Agents start it only if absent.
- `/srv/qpn` NFS export on rtx5090 (192.168.50.4 to sparks;
  10.10.250.2 only reachable from sparkf) mounted soft at /mnt/qpn on
  all 16. Sparks also have /mnt/model-warm (ceph) — DO NOT poll it
  aggressively (MDS melts; that's why the rendezvous moved to NFS).

## Operator rulings in force (violations caused most of this week's pain)

- Tricky problem with no clear solution = misposed problem. Sync
  machinery growing in a design = STOP; write the minimal fact set +
  change frequency; check shared media (current/, NFS, repo) before
  any protocol.
- Qualify once at build (receipts), then LOAD. Never re-test at
  runtime. No provenance/lineage gates — structural checks only.
- No comments in code; explanations live in PRs/docs.
- New capability lands ONLY WITH deletion/conversion of what it
  replaces (grep every caller).
- Poll, never sleep. Fail fast (10-30s wiring timeout with
  route/host/port in the error). Missing node = peers WAIT and
  re-open, never suicide.
- pkill -f matches YOUR ssh wrapper — count processes by /proc exe.
- Daemon age before logs; module_library cache defeats source edits
  (rm link_units+active before publish, verify with strings).
- rtx5090 does API+tokenizing; drivers never tokenize. NCCL is ruled
  for total deletion. GPU-side detection (device spins the nonce) is
  the direction; verbs sends stay CPU-posted (host-only API).

## Recommended next steps (in order)

1. Check PR #823 state; if merged, deploy and verify the ordinal fix
   on-fleet (the poisoning defect dies). If not, its approach is
   correct — validate its bench matrix and land it.
2. Restore fleet to 16/16 + one serving request green (clean UPDATE
   cycle, trace env off — remove env.local from all runtime roots).
3. Re-measure E2E on the codex-astra fixed packs/driver; re-baseline.
4. Run slot_bench in a parked-fleet window; confirm it reproduces the
   serving number; then hill-climb per-op slot cost (each change =
   one bench run, minutes).
5. The big lever, in order of leverage: (a) engine-minted ordinals
   (needed regardless), (b) fold the 4 CPU-mediated stages into fewer
   loop transitions — target 364→<100/token, (c) GPU-side arrival
   detection (device spins nonce; CPU only posts sends), (d) resume
   the file-rendezvous port (kills the TCP wire-up class + gives hot
   rejoin), (e) weightd attach for glm5_next (packs mmap'd in
   residentd today — 21.7GB per boot vs 3ms attach; weightd IS the
   weights owner by design), (f) dual-rail (topology plumbing exists,
   dark).
6. MTP/spec payoff measurement (other lane's probe, PR #814) once the
   fleet is stable — their plan is sound; the speculation-blind
   admission accounting caveat they flagged is real.

## Where things live

- My lane worktree: /Users/mac/lane-glm53 (branch
  lane/glm5next-mtp-accept-takeover; do NOT commit from
  /Users/mac/sparkpipe — shared checkout).
- Build host: sparkf:~/sparkpipe-build. Hub: rtx5090 user spec
  (~/release/<root>/, fleet view sparkf:current/).
- API: spark0:8433 POST /v1/chat/completions with prompt_token_ids
  (no tokenizer wired in glm5_next deployment yet).
- Runtime root: ~/sparkdata/glm53flash.fp8.tp16 on every spark.
- Memory files (auto-memory) carry the full session-by-session state;
  the load-bearing ones: glm53-flash-tree-engine.md (everything tree),
  stop-reposing-problem-law.md (the problem-reposing law).
