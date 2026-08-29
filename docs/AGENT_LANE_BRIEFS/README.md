# Agent lane rules (shared, binding for every driver lane)

## Operating model (operator directive, 2026-08-29)

Agents are one step below the coordinator: briefs carry EXACT steps,
standards, and fleet-wide context (other lanes' state, the overall
goal) — not just missions. The coordinator neither solves everything
solo (bottleneck) nor babysits loops; it intervenes directly only for
(a) shared-write-set fixes, (b) cross-lane arbitration, (c) incidents,
then hands follow-through to a fresh, precisely-briefed agent. Every
lane report states how its current milestone serves the scoreboard /
fleet goal so all agents pull consistently.

You are one lane of a small pipeline. You own ONE model driver. Other
lanes and the coordinator work in parallel. These rules exist because a
larger agent fleet failed here; every rule below encodes a real failure.

## Write set (hard boundary)
- You work ONLY in your git worktree (given in your lane brief).
- You may create/edit files under: `modules/<your-family>/`,
  `model-families/<your-family>/`, `tools/<your-family>*`,
  `tests/test_<your-family>*`, `docs/` (your lane notes only).
- You may NOT edit: `Makefile`, `sources.mk`, `modules/resident_decode_stage_rules.mk`,
  `runtime/`, `node/`, `include/`, `cache/`, `ring/`, `src/`, other families' dirs.
  If your build needs shared wiring, implement it in your worktree, then
  record the exact diff under "INTEGRATION REQUEST" in your lane report.
  The coordinator integrates. Never push to `main`.
- Branch: `lane/<your-name>` (already created in your worktree). Commit
  often, small commits. Push with `tools/sparkpipe_github_pat.sh git push`.
  Open a PR at each milestone; never merge it yourself.

## Fleet pack policy (hard, 2026-08-27): every model on all 16 sparks

Every served model runs on a 16-rank topology fleet-wide, therefore
**every spark holds its rank's pack for EVERY model**. A pack build that
targets fewer nodes is incomplete by definition. Each packer emits 16
rank-addressed packs (wire format v2: tp_degree + tp_rank); rank r
deploys to sparke-hex-digit r (spark0..sparkf). The 16-rank topology is
per-model (TP16 where heads divide; TP4xPP4 otherwise — the topology
guide decides), but the deployment set is always all 16.

| Model | 16-rank topology | Pack/rank | Status |
|---|---|---|---|
| Qwen 3.8 27B | TP4xPP4 (4 KV heads) | ~3.4 GB | re-emit sharded |
| DSV4 Flash -0731 | TP16 | ~9.8 GB | building (spark4) → shard |
| Qwen 3.8 Flash | TP16 | ~10.5 GB | pending M5 kernels |
| GLM 5.3 Flash | TP16 | 21.7 GB (replicated indexer/router set) | 16/16 built+shape-gated, bring-up next |
| DSV4 Pro | TP4xPP4 | ~93 GB (DSpark draft replicated) | 16/16 built, 14/16 deployed |
| Qwen Max | TP4xPP4 | ~80 GB | sharding sprint (spark7) |
| K3 | TP16 | ~94 GB | building (sparke) → shard |

Per-node budget ≈ 269 GB of packs — fits every node's NVMe with KV head
room. GLM 5.2 is a kernel donor, NOT a serving target: no fleet pack.
Wire-format-v2 sharding lands via the qwen38max lane; until then lanes
build family packs as today and the coordinator re-shards + deploys at
merge time (packs are derived artifacts; re-sharding is a cheap pass).

## Hardware-independence contract (binding — status in docs/HARDWARE_INDEPENDENCE.md)

The HAL (Phase 1B `spark_device.h`) does not exist yet; lanes run
CUDA-direct at the module layer and that is correct. Three lines hold so
the future extraction is cheap and the CPU oracle keeps working:

1. KERNEL TREE STAYS HOST-COMPILABLE. Anything you add under
   `inference/kernels/` or `inference/llms/` uses the house loop shape
   (`for (i = threadIdx.x; i < N; i += THREADS)`; reductions at
   `THREADS/2` down) so THREADS==1 runs on the CPU shim, includes no
   cuda_runtime.h, and gains a host oracle test in your lane. TMA /
   async-pipeline intrinsics only through the existing tile.cuh
   patterns the shim already stubs — never bare.
2. NO PER-FAMILY ABSTRACTION LAYERS. Do not invent a device wrapper in
   your family "to be safe" — seven private HALs are worse than one
   direct surface. Call cuda directly, plainly, where your family
   already does. ANY NEW CUDA API CATEGORY (an API family you call that
   your family's existing code does not — events, graphs, host register,
   a new memory kind...) goes in your report's INTEGRATION REQUEST with
   the call sites listed, so the extraction inventory stays complete.
3. ABIs STAY DEVICE-OPAQUE OUTSIDE YOUR FAMILY. Shared headers take
   `void *` handles (see the serving adapter's `execution_stream`);
   `cudaStream_t` and friends stay inside your family-local headers.

MODULE_TARGET stays a namespaced tuple (`cuda.sm121.<family>...`) — it
is the backend-swap hook; do not collapse it to a bare arch flag.

MEMORY DISCIPLINE (the memory-space model is docs/INFERENCE_OS_DESIGN.md —
read it before writing allocation code):
4. EVERY ALLOCATION NAMES ITS SPACE KIND in a one-line comment:
   device-private (cudaMalloc), pinned (cudaMallocHost/Register),
   coherent (managed/UM), or file-backed (pack mmap). Zero-cost now;
   it is the extraction inventory later.
5. NO OPEN-CODED CROSS-SPACE COPIES. A cudaMemcpy between buffers whose
   spaces differ needs a comment saying which two spaces — pointer
   identity across spaces is exactly what breaks the first discrete
   port. Same-space copies are fine bare.
6. ANY NEW ALLOCATION KIND in your family (first managed alloc, first
   cudaHostRegister, a private mmap scheme) goes in the INTEGRATION
   REQUEST with call sites — the family pack loader's mapping is the
   sanctioned file-backed path; do not invent a second one.

## CAPACITY SIZING RULE (perf audit: measured 1.6-1.8x tax)

Deployments size max_active_sequences / lane pools to the SERVED
batch, not the maximum imaginable: GDN state is ~150 MB/lane on
unified memory and oversized configs cost measured 1.6-1.8x at mid-B
(TLB/page pressure). The spec defaults (1024 lanes / 4096 positions)
are footguns — a B8-serving deployment carries no 1024-lane pool.

## MERGE GATES (operator directive: no slop)

Every merge into main passes: (a) the code-size ratchet RE-RUN AFTER
ANY CONFLICT RESOLUTION (tonight's +15K drift came from resolutions
keeping stale numbers); (b) a cyclomatic check on changed files
(mean/max vs the 7.33/157 baseline; regressions justify in-commit);
(c) the value test Solutions/(Codesize^2) — additions must buy
disproportionate solution; (d) NO new high-level DRY violations
(pasted adapter lifecycle = refused; the DRY template is the cure).

## QUANTIZATION POLICY (hard, operator directive 2026-08-29 — docs/GOALS.md)

NO self-made quantizations. Weights arrive already-quantized from
official publisher releases or VETTED community quantizations (pinned
provenance + full receipt hashes + quality gate on first serve). Our
packers REPACKAGE (stagepack format, TP/PP sharding, scale planes) —
they never quantize a master. No acceptable source exists? Serve the
publisher's native precision (BF16 fits the fleet) instead of
inventing one. Precision changes = adopting a new official/vetted
SOURCE + full re-qualification.

## QUALITY GATE: ds4_eval at the "not horrible" transition (hard, 2026-08-28)

A perf cell alone does not make a model "not horrible" — inference can
be fast AND broken. When a model's first honest serving cell lands
(operator directive), the lane runs the ds4_eval capability-regression
suite against the deployed API BEFORE the model is called usable:

1. **COMPSEC-17 first** (the sanity tier): the 17 COMPSEC questions
   through the live /v1 endpoint, minutes at low concurrency. Working
   models score 15-17/17 (see the retained records); horribly broken
   inference (degenerate repetition, garbage, all-refusal) scores near
   zero or errors — that is a STOP, not a footnote.
2. **Full 92x** (the not-horrible tier): 25 GPQA + 25 SuperGPQA +
   25 AIME2025 + 17 COMPSEC. Reference bands from the retained records:
   API-class 70-81/92, local-quantized 73-78/92. A first 92x sets the
   model's quality baseline; big drops on later runs are regressions.

Records and protocol: qualification/ds4_eval/README.md (canonical archive
format: REPORT.md + INTEGRITY.json + summary.json + cases.json +
responses/*.json per run — copy the retained runs' structure; harness and
fixture SHAs are pinned there). New runs land at
qualification/ds4_eval/runs/<model>-<deployment>-<date>/ and the lane
report cites the score. The scoreboard ledger notes quality alongside
perf: a perf cell without a COMPSEC-17 pass stays marked provisional.

## Run queue + node reservations (hard, 2026-08-28)

GPU and heavy remote work is scheduled through the run queue
(`python3 tools/spark_queue.py`, state in `runs/`, controller-local):

- **Before GPU work on a node**: `tools/spark_queue.py reserve --node
  sparkX --holder lane-<name> --ttl-min <estimate>`; release when done
  (`release --node sparkX`). The coordinator's scheduler will not land
  queued runs on a reserved node, and the 30-min sweep audits stale
  reservations (TTL + holder process gone = released).
- **Script-expressible runs** (benches, sweeps, verifier passes, builds
  with a known command): submit them — `add --id <id> --nodes sparkX
  --cmd '<remote cmd>' --priority N [--after id,id]` — instead of running
  them by hand. The scheduler launches every entry whose node set is
  free; disjoint entries run in PARALLEL; a running entry HOLDS its
  nodes until exit (let-them-cook: no preemption).
- **Gates**: `--kind gate` entries block dependents until marked done —
  use them for cross-lane dependencies (e.g. "glm53 validator resumes
  after glm52 oracle fix merges").
- Entry kinds: run / gate / note. `after:` = dependency ids. Logs land
  at `/tmp/sparkq/<id>.log` on the node; `status --id` tails them.
- The queue REFUSES commands containing rm -rf, reboot, kill -9 and
  friends — destructive cleanup is a human-visible coordinator action,
  never a queued one.

## Cluster rules (hard)
- Nodes are `spark0..sparkf`, ssh BatchMode works from the controller mac.
- `spark2` hosts the 27B development instance. It CAN be used for pack
  deployment and testing — the daemon can be stopped/restarted when needed
  (coordinate via the report so concurrent users know). No special protection.
- Your lane's nodes are in your brief. Do not touch other lanes' nodes.
- NEVER reboot or restart a spark node. If a node is wedged, report it.
- GPU hygiene: `nvidia-smi -r` only when NO cuda processes are running
  (`pgrep -f sparkpipe_model`). After ANY reset wait 10s before starting
  daemons. A daemon SIGKILLed mid-CUDA leaks device memory; the next
  start fails with `cuda_storage` - that means reset (cleanly), not panic.
- **THE NO-KILL PROTOCOL (learned from three NVRM wedges, most recently
  spark3 2026-08-27 whose KILL also took down the model-warm ceph MDS
  fleet-wide):** TERM first, ALWAYS. Wait. If the process ignores TERM
  (R-state spin, D-state sleep), do NOT escalate to SIGKILL — capture
  `/proc/<pid>/stack`, `/proc/<pid>/status`, and 30s of `top -H` output,
  then REPORT the node as needing a sysadmin reboot with the process
  left running. A reboot cleans a stuck daemon without rolling the
  NVRM dice; a SIGKILL mid-CUDA wedges the GPU (and on MDS/OSD hosts,
  the storage for everyone). The daemon is already lost when TERM fails
  — the only question is whether the node goes with it.
- **WHY DAEMONS BECOME TERM-IMMUNE (the bug to fix, not the protocol):**
  an unbounded spin in a collective wait loop does not return to the
  signal handler. Every wait on a peer must carry a deadline (the
  config's operation_timeout_milli exists — honor it) and re-check
  shutdown flags. If you find a wait loop without one, that is a P1
  bug in the runtime, report it as an integration request.
- Each residentd accepts ONE client. The model_api holds that slot; for
  direct batch-tool tests, stop the api first, restore it after.
- **PROCESS HYGIENE ON SHARED NODES (hard, from TWO 2026-08-28 spark0
  incidents):** kill ONLY pids you captured at spawn time and recorded
  in your own log. There is NO safe fuzzy match on a shared node — not
  by process name, not by output-path pattern ('*qwen38max.tp4*' caught
  a sibling's builds TWICE: once by name, once by path), not by parent
  shell. Remote staging runs FROM YOUR COMMITTED BRANCH (a git checkout
  on the node), never a hand-copied tools/ dir — if the code isn't
  committed, the artifacts built from it are unreproducible. Before
  starting a heavy build on a node, check what else runs there (`ps` +
  the queue reservations) and coordinate via the reports.
- **RESIDENTD FLEET-ONLY RULE (from the 2026-08-28 sparke OOM):** ranks
  come up ONLY as part of a coordinated fleet bring-up (all ranks
  near-simultaneously per the deployment plan) and the node set gets
  queue reservations FIRST. A lone rank without its fleet sits in
  transport-wait holding its full weight in unified memory — one
  OOM-killed a sibling lane's pack build and nearly re-OOMed the local
  OSD on a storage node. Single-rank smoke tests run on YOUR OWN node.
- **FLEET-WAVE LESSONS (glm53 all-16 bring-up, 2026-08-28):** (1) a
  multi-rank launch is ONE SIMULTANEOUS WAVE — staggered launches die
  on the 180s hidden-transport connect window and late joiners are
  refused; TERM-sweep everything first, then launch all ranks together.
  (2) `pkill -f` catches your own ssh wrapper — use `pkill -x
  sparkpipe_model` (exact-name matching only; consistent with the
  spawn-captured-pids rule). (3) EADDRINUSE TIME_WAIT at wave start on
  some nodes: a 45s pre-launch sleep fixes it. Record per-node layout
  and launch commands in a LAUNCH-STATE.md in the deployment dir so any
  coordinator/agent can take over mid-bring-up.
- The DFlash2 launch environment is MANDATORY for spec runs (missing it
  produces degenerate output - this cost us a day):
  SPARK_QWEN38_27B_SERVING_SPECULATE=1 SPEC_METHOD=dflash2 DRAFT_COUNT=8
  DSPARK_PACK_PATH=<drafter> DFLASH2_STATE_SELECT=1 BONUS_FOLD=2
  DFLASH2_BLOCK_KV=0 DFLASH2_WINDOW=2048 DFLASH2_CTX_CACHE=1
- **CEPH TOPOLOGY + STORAGE-RISK RULE (full map in the infra advisory):
  EVERY node runs an OSD; mons on spark0/spark7/sparkf; the ACTIVE MDS
  pair is mds.ds4warm.spark2 + .spark3.** (1) STRICT no-GPU-work on the
  active-MDS hosts (spark2, spark3) — unified memory is shared and a GPU
  job past the envelope OOMs the MDS/OSDs (spark3 rebooted this way at
  ~104 GiB on 2026-08-28, and its reboots bounce the MDS pair fleet-wide:
  transient ENOENT + slow uncached reads = MDS failover/warm-up, wait and
  retest). (2) ALL other nodes run GPU work under the OPERATOR CEILING of
  110 GiB device allocation (of 119 GiB unified — kernel+OSD+driver
  need the rest; NVRM NV_ERR_NO_MEMORY at ~114 GiB kills daemons
  SILENTLY, confirmed twice 2026-08-28). Size deployments so
  weights+KV+overhead stays under 110; co-resident configs compute
  the sum per node BEFORE launch and cut KV pools to fit. (3) Fleet-wide SLOW UNCACHED
  reads + fast cached = MDS host trouble; SINGLE-OBJECT stall with healthy
  neighbors = degraded PG (node/OSD down — like sparke/osd.14).

## Build chain (the only supported path)
1. Module: `make -C modules/<family> publish NVCC=/usr/local/cuda/bin/nvcc
   CUDA_ARCH=sm_121a STAGE_PACK_PATH=<pack> <tier env>` - needs a readable
   pack; the whole-stack tier needs ~53GB free device memory (stop daemons).
   Mid-pipeline tier (STAGE_COUNT=2 STAGE_LAYER_COUNT=4 MTP_LAYER_COUNT=0)
   validates with a small synthesized pack when memory is tight.
2. Synthesized pack: `cc` the family's `tools/*_pack_synthesize.c`
   (include flags: -Iinclude -Imodel-families/common/include
   -Imodel-families/<family>/include -Imodules/<family>/include -Imodules/<family>/source).
3. Driver: `build/sparkpipe_model_compile --model
   examples/model_descriptions/<family>_firmware.json --stage <family>
   --library build/module_library --output <dir> --cc /usr/bin/cc --include include
   --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda
   --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl
   --cc-arg -pthread`
   (publish FIRST - the library links the published unit, not your .a).
   NOTE: links that need model_common add --cc-arg -lmodel_common (the
   glm52 F4 cycle hit this - driver compile failed without it).
4. Deployment dir: bin/{residentd,api,batch} + lib/{model_driver.so,
   model_serving_adapter.so,hidden_transport.so} + config (hostnames must
   match YOUR node!) + packs. Canonical launcher pattern: TERM-kill, wait,
   (reset if leaked), launch residentd, WAIT FOR "model_residentd ready",
   THEN the api.

## Truth rules (hard)
- Every claim = a command + its raw output. No "should work", no
  plausible text as correctness.
- Ground truth gates: validator PASS (all checks), decode-vs-prefill
  bit-exact, fresh-instance determinism, kernel cosine vs CPU oracle
  within the family's established thresholds. Spec paths additionally
  need in-vocab drafts and sane acceptance (~5-7/round at k=8).
- Report format (docs/AGENT_LANE_BRIEFS/reports/<lane>-<date>.md):
  what ran, raw numbers, what failed with exact stderr, INTEGRATION
  REQUEST section, next experiment. Honest negatives are valuable.
- The code-size ratchet (tests/test_code_size.py) applies on your branch:
  ratchet with a justification comment in the same commit.

## Script parameterization (hard rule)
Every script takes the spark host as a parameter (`--spark N` or
`SPARK_HOST=sparkN env`), NEVER hardcodes a node. The coordinator's
scripts burned an afternoon on hardcoded spark3 paths when that node
went down; a measurement chain must be movable to any healthy spark
by changing one argument. Batch/poll one-shots get this too.

## Escalation
Stuck > 45 min on one problem: write what you tried in the report and
move to the next independent item. Report wedged nodes immediately.
