# Parallel development on the Spark fleet

This guide covers running several driver jobs on the 16 Sparks while GLM 5.3
Flash keeps serving. The daemon contract behind it is
[WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md); queue mechanics (release builds only; the queue is otherwise retired) and PR evidence are in
[PARALLEL_DRIVER_DEBUG.md](PARALLEL_DRIVER_DEBUG.md). Statements marked
"observed 2026-09-28" come from read-only checks on the fleet that day and will
drift.

The API and resident daemon provide common serving; weightd owns byte residency,
leases and mesh transport. Drivers supply layouts, routing and model math. Each
API process serves one deployment; request `model` does not select another
model. Apply the model's chat template outside the API; it only joins message
content.

## When to use the queue

Owner ruling, 2026-09-28:
- `tools/spark_queue.py` is retired for day-to-day work. Its stale owners froze progress, so the lead dev cleared its state on 2026-09-28.
- Developers work in parallel, directly on their assigned nodes. Each uses its own runtime root, its own weightd mesh lane (lane 0 is production), its own ports, a memory budget from `tools/devcycle/lane_budget_calc.py`, and a `.wset` working set for MoE models.
- Start every process in its own transient unit (`systemd-run --user --unit=...`) so cleanup is exact. Never start one inside the `fleet-agent` cgroup.
- Performance measurements on the Sparks are serialized by the lead dev, one at a time, in short windows. This covers tok/s, TTFT, prefill time, concurrency sweeps and timed evals. Functional and correctness checks need no window.
- The one remaining queue use is `tools/module_build_release.sh`, which still demands a `SPARK_QUEUE_ID`. A release build runs as a single queue job until that guard is changed.

The queue mechanics below are kept for that release build and for reference.

## What every job shares

Each Spark runs the fleet agent as the systemd user unit `fleet-agent`. It
supervises one weightd on `/tmp/spark_weightd.sock` and the production GLM
residentd, and both run in the agent's cgroup
([production ownership](WEIGHTD_DESIGN.md#production-ownership)). Its drop-in
`20-serving.conf` (sha256 prefix `8324336487eecc38`, the one recorded in
`qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md`)
sets `G5_API_DISABLED=1`, `G5_WARMUP=0`, `G5_GRAPH_PATH=1`, `G5_PIN_EXPERTS=1`
and `SPARK_TP_WAIT_MODE=hardware`; it sets neither `SPARK_WEIGHTD_LANE` nor
`MemoryMax` (observed 2026-09-28 on spark0). `start_root` in `tools/fleet_node_agent.sh`
maps the two `G5_` switches to `SPARK_GLM5_NEXT_GRAPH_PATH` and
`SPARK_GLM5_NEXT_PIN_EXPERTS`, and passes `SPARK_WEIGHTD_EXPERT_POOL_BYTES`
34359738368 unless `G5_EXPERT_POOL_BYTES` overrides it. The production arena is
the whole GLM rank pack, pinned: the daemon held 20,874 MiB of device memory on
spark6 (nvidia-smi, observed 2026-09-28).

Lane jobs attach to that daemon. Do not start another weightd-line daemon on a
Spark:

- Any other `*/sparkpipe_weightd` process makes the agent stop managing the
  node until it exits. That covers the private daemons started by
  `tools/inference_smoke.py`, `tools/weightd_execute_receipt.py`,
  `tools/multi_dev_orchestrate.py` and `WEIGHTD_MODE=private` in
  `tools/devcycle/run-minimax-family-job.sh` and
  `tools/devcycle/run-dsv4-pro-family-job.sh`.
- `tools/devcycle/sparkpipe-weightd-shared.service` runs the 8-lane
  shared-serving-20260922 binary, which main-built clients cannot map
  (WEIGHTD_DESIGN.md, concurrent mesh lane reservations). It was inactive on
  all 16 Sparks (observed 2026-09-28).
- The separate weightsd channel (`tools/devcycle/sparkpipe_weightsd.service`,
  `tools/weightsd_deploy.sh`) is retired. One idle instance still ran on spark6
  (observed 2026-09-28). Its history is in [archive/WEIGHTSD.md](archive/WEIGHTSD.md).

Never restart `/tmp/spark_weightd.sock` or the agent by hand; either takes GLM
serving down. weightd changes only through the hub's `core/WEIGHTSD_BIN`
announce and a drain (`FLEET_RELEASE_RUNBOOK.md` §5.2). weightd bugs are fixed
on main and redeployed; do not patch a node's binary in place.

## Lanes

weightd has 16 mesh lanes (`SPARK_WEIGHTD_MESH_MAX_LANES`,
`include/sparkpipe/spark_weightd.h`). A lane keeps the first rank map configured
on it until the daemon restarts, a different map returns `UNSUPPORTED`
(`node/weightd_mesh.c`), and a daemon restart means draining production. Each
family therefore keeps one lane and one rank map. Set `SPARK_WEIGHTD_LANE` and
`SPARK_TP_MESH_RANKS` identically on every rank; an occupied lane returns
`NO_LANE` and is never redirected.

| Lane | Job | Default topology | Where the default is set |
| --- | --- | --- | --- |
| 0 | production GLM 5.3 Flash | TP16 identity | reserved; fleet agent |
| 1 | qwen38_27b | TP4, spark0-spark3 | `tools/qwen38_27b_lane_launch.sh` |
| 2 | qwen38_max | TP16 | `tools/qwen38max_multidev_run_family.sh` |
| 3 | k3 | TP4xPP4 | `tools/k3_multidev_run_family.sh` |
| 4 | dsv41_flash | TP4, spark4-spark7 | `tools/dsv41_flash_shared_lane.sh` |
| 5 | dsv4_pro | TP4xPP4 | `tools/devcycle/run-dsv4-pro-family-job.sh` |
| 6 | gemma4 | TP16 | `tools/gemma4_tp16_shared_socket.sh` |
| 7 | mimo26 | not chosen; no wrapper yet | `tools/devcycle/lane_assignments.json` |
| 8 | laguna | TP8xPP2 | `tools/laguna_multidev_run_family.sh` |
| 9 | ling, then muse_glimmer and hy4 | TP16 | `tools/ling_multidev_run_family.sh` |
| 10 | minimax | TP4 | `tools/devcycle/run-minimax-family-job.sh` |
| 11-15 | free: GLM development, second topologies | chosen per job | none |

Production GLM starts without `SPARK_WEIGHTD_LANE`, so it runs in automatic
mode and takes the lowest free lane on each rank (`runtime/spark_weightd.c`,
lane acquire). It lands on lane 0 on every rank only while lane 0 is free
everywhere, so no other job may use lane 0. Setting `SPARK_WEIGHTD_LANE=0` in
the drop-in would make this explicit; that is an open operator change.
`tools/devcycle/lane_assignments.json` still describes eight lanes and assigns
lane 0 to a GLM development lane; for lane numbers, the table above supersedes
it. To try another topology for a family (for example TP16, PP16 or TP4xPP4),
take a free lane instead of re-mapping the family's own lane.

Ports: lane L owns control 23000 + 16L and transport 64000 + 16L, 16 ports each
(`lane_assignments.json`; the TP collective uses the weightd mesh, not ports).
For lanes 11-15 the same formula gives 23176-23255 and 64176-64255 (arithmetic).
Reserve every listener with `--ports`.

The family wrappers default their socket to the fleet weightd at
`/tmp/spark_weightd.sock`; `SPARK_WEIGHTD_SOCKET` or the socket variable named in
each wrapper's header overrides it.

## Lane tiers

| Tier | Arena | Working set | Use |
| --- | --- | --- | --- |
| S | the production arena | whole pack, already pinned | unchanged production pack |
| W | its own arena | trace-recorded `.wset`, warmed before the run | changed pack or another model |
| E | its own arena, small pool | none; misses load on demand | functional debugging only |

**S, shared production arena.** Only for the unchanged production pack. The
attach must match the production identity (model, revision, topology, pack
SHA-256, geometry, ABI) and declare the same pool, 34359738368 bytes; a
different pool gets `INVALID_ARGUMENT` (`runtime/spark_weightd.c`, lazy attach
to an existing arena). The job adds no arena bytes, so budget only its
residentd: a B1, context-512 GLM resident measured 3,434 MiB of device memory
in the PR #1082 campaign
([archive/PARALLEL_RESIDENT_QUALIFICATION.md](archive/PARALLEL_RESIDENT_QUALIFICATION.md)).
A graph-mode GLM resident holds 24 of the arena's 256 leases, so at most 10
pinned residents fit per Spark, production included (arithmetic in
WEIGHTD_DESIGN.md, GLM graph residency today).

**W, own arena with a trace-recorded working set.** Attach with a pool smaller
than the pack, sized from the working set, so the arena stays per-chunk lazy.
Record the set by running the job's prompt set once: the daemon appends every
acquired key to `PACK.wset`. Copy that trace to a file named for the prompt set
(committed examples: `model-families/dsv4/smoke-standard-v1.wset`,
`model-families/glm5_next/glm53flash.fp8.tp16.smoke.wset`,
`model-families/k3/smoke-k3-v1.wset`, `model-families/ling/smoke-ling-v1.wset`)
and warm before each run:

```sh
build/weightd_warm /tmp/spark_weightd.sock PACK SHA256 REVISION TOPOLOGY --wset FILE
```

Add `--family dsv4_pro --world-rank R`, `--family dsv41_flash`, `--family k3`
or `--family ling` so the warm identity matches that module; without it the
tool uses the GLM identity (`tools/weightd_warm.c`). With `--family k3`,
TOPOLOGY is the runner's tp_degree: 4 for TP4xPP4, 16 for TP16; any other
value is refused, since the runner never attaches it. It prints
`WSET-WARM keys=N elapsed_ms=T`. Warm right before the run: an arena with no
attached consumer is cold, and the next attach that needs room frees the oldest
cold arena (`SparkWeightdServerReclaimCold`). `weightd_warm SOCKET --reclaim`
frees every cold arena on the daemon, including other developers' warmed
arenas, so do not use it on the shared daemon without checking whose they are.
A daemon holds at most 16 arenas (`SPARK_WEIGHTD_ARENA_COUNT_MAX`). A
graph-mode GLM job needs its whole pack pinned, so on W it must budget the full
pack; a partial `.wset` only serves eager mode.

**E, lazy eager.** Its own arena with a small pool, no warm-up and eager
execution (for GLM, `SPARK_GLM5_NEXT_GRAPH_PATH=0` and no pinning). Misses read
from NVMe during the run. Never use E results as performance evidence.

### Working sets and budgets

A MoE lane on W needs a `.wset` and the family's
`model-families/<family>/smoke_experts.json`. Dense families have no routed
experts (the gemma4 and minimax manifests record an empty set), so they budget
the full shard: spine bytes divided by the node count.

`python3 tools/devcycle/lane_budget_calc.py model-families/<family>/smoke_experts.json`
prints `DEVICE_MIB` and `TOTAL_MIB` per node. Device is (spine + working-set
experts, both divided by the node count when sharded, + KV floor + workspace)
× 1.08; total is 1.5 × device and at least device + 1024 MiB. Its fleet check
still multiplies by eight lanes against the 78,336 MiB envelope of the PR #1082
campaign, so treat that line as advisory. It knows nothing about the S tier,
whose arena bytes production already pays.

## Queue: gpu-shared and exclusive

There is one controller, `mac@mac-studio`, with ledger `~/.sparkpipe/queue`.
Do not start another dispatcher or set a private `SPARK_QUEUE_STATE`. Run the
client that matches the live dispatcher: on 2026-09-28 it was
`/Users/mac/.sparkpipe/station-20260924/spark_queue-913a2864.py` (started
2026-09-23), byte-identical to `tools/spark_queue.py` on main (sha256 prefix
`4a1a91b70442cd54`). Check with `ps` before relying on this. The older
`/Users/mac/wk-sparkpipe-queue-main-20260922` checkout sits at 729649e and its
client differs; do not use it. Pass your own checkout to `sync --repo`.

```sh
ssh mac@mac-studio
Q=/Users/mac/.sparkpipe/station-20260924/spark_queue-913a2864.py
python3 "$Q" doctor
python3 "$Q" list --all
```

The resource classes (`conflicts` and admission in `tools/spark_queue.py`):

- `cpu`: builds; may overlap within the summed memory budgets.
- `gpu`: excludes every other `gpu` or `gpu-shared` job on its nodes; no
  device census.
- `gpu-shared`: lane jobs, which may share nodes with each other. Requires
  `--memory-mib TOTAL` (host plus device) and `--device-memory-mib DEVICE`; the
  host cgroup gets TOTAL minus DEVICE. Admission rejects GPU processes outside
  tracked owners, observed device use above a declaration, and reservations
  beyond the smaller of 114,688 MiB and the node's memory minus 8,192 MiB
  headroom.
- `exclusive`: excludes every other queue job on its nodes. Use it for measured
  runs. It does not stop production serving, which runs outside the queue.

Blocker (observed 2026-09-28): `gpu-shared` admission fails on every serving
Spark. The ledger still tracks 32 persistent owners for
`sparkpipe-weightd-shared.service` and `sparkpipe-glm-serving-dd3526b2.service`,
which are inactive ("owner unit is not verifiably active"). Behind that, the
production weightd and residentd run in `fleet-agent.service` with
`MemoryMax=infinity`, so `track` refuses the unit ("persistent unit must have a
verified finite MemoryMax") and the census reports "unaccounted GPU process".
The operator fix: `untrack` the 32 stale owners, give `fleet-agent.service` a
finite `MemoryMax`, then on each Spark run
`track --node sparkN --unit fleet-agent.service --scope user --device-memory-mib DEVICE`,
where DEVICE covers weightd plus the production residentd (33,150 MiB on spark6
by nvidia-smi on 2026-09-28: 20,874 + 12,276, arithmetic). Until then lane jobs
cannot be admitted as `gpu-shared` on serving Sparks.

## Submitting a lane job

Commit the family wrapper and configuration, then sync the exact source:

```sh
python3 "$Q" sync --repo CHECKOUT --id dev-model-001 \
  --nodes spark4,spark5,spark6,spark7 --ref HEAD
```

Use the printed checkout path and build coherent family artifacts through the
queue (CUDA validation needs GPU admission). The family wrapper prepares a
private deployment under `$SPARK_QUEUE_RUNTIME_ROOT` and keeps children in its
queue cgroup. Its runtime `packs/` directory must contain exactly one valid
`*.sha256` digest ([WEIGHTD_SUPERVISED_STARTUP.md](WEIGHTD_SUPERVISED_STARTUP.md)).
Keep ports, runtime, KV and logs private. The resident launch uses the common
executable:

```sh
export SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock
export SPARK_WEIGHTD_LANE=4
export SPARK_TP_MESH_RANKS=4,5,6,7
export SPARK_WEIGHTD_EXPERT_POOL_BYTES=POOL_BYTES
exec /actual/verified/bin/sparkpipe_model_residentd \
  --deployment "$SPARK_QUEUE_RUNTIME_ROOT/deployment.json" \
  --rank-index "$SPARK_QUEUE_RANK"
```

A GLM job in the qualified configuration also sets
`SPARK_GLM5_NEXT_GRAPH_PATH=1`, `SPARK_GLM5_NEXT_PIN_EXPERTS=1` and
`SPARK_TP_WAIT_MODE=hardware`, with a pool larger than the pack. An unset wait
mode means spin (`ring/transport/tp_device_collective.c`), which is not the
qualified configuration; before PR #1255 (merged in cf64e90) spin mode also
failed single-sequence prefill waves above 8 rows.

Submit the wrapper as one per-node job:

```sh
python3 "$Q" add --id dev-model-run-001 \
  --nodes spark4,spark5,spark6,spark7 --per-node --cwd CHECKOUT \
  --resources gpu-shared --memory-mib TOTAL_MIB --device-memory-mib DEVICE_MIB \
  --ports FIRST:LAST --ttl-min 14 --by DEVELOPER --cmd-file run-family-job.sh
python3 "$Q" status --id dev-model-run-001
```

Use `cancel --id ID` for owned jobs. No nested SSH GPU launches. Logical rank is
the index in `--nodes`; keep this order identical to the map:

| Topology | `SPARK_TP_MESH_RANKS` | Node order |
| --- | --- | --- |
| TP16 identity | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` | spark0 through sparkf |
| TP16 permutation | `0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1` | spark0, spark2 through sparkf, spark1 |
| TP4 subset | `4,5,6,7` | spark4,spark5,spark6,spark7 |

### Connecting a driver to weightd

Drivers find the daemon through configuration only. Direct-environment
families set `SPARK_WEIGHTD_SOCKET`, `SPARK_WEIGHTD_ATTACH_LAZY=1` and
`SPARK_WEIGHTD_PACK_SHA256`. Deployment-JSON families set
`weightd.socket_path`; residentd then probes the socket and fails fast if it is
down, and never starts a daemon. `SPARK_WEIGHTD_ATTACH=0` together with a
configured socket fails by design.

## Known failure: mesh cudaHostRegister

The collective host-registers the daemon's mesh region with
`cudaHostRegister(PORTABLE | MAPPED)` (`ring/transport/tp_device_collective.c`).
In residentds attached to a shared daemon this has returned
`cudaErrorInvalidValue` (cuda=1): the 8-lane mixed-topology run at 44fe4af7
failed on 15 of 16 nodes with `MESH-REGISTER-FAIL`
([receipt](receipts/mixed-mesh-44fe4af7-failure.json)), and TECHDEBT.md records
the eight-lane attempt failing the same way. Two fixes landed: 498275a6 maps the
region's file offset 0 at an aligned address, and 5814bf2 (PR #1135) logs
`MESH-REGISTER-SKIP` for cuda=1 and continues unregistered, because GB10 memory
is coherent. `MESH-REGISTER-SKIP` in a lane log is therefore expected. Any other
error code still stops the job with `MESH-REGISTER-FAIL`; report it with the
full log.

## Performance evidence

A smoke run counts as performance evidence only when all of these hold:

1. An exclusive fleet: an `exclusive` queue job on every participating node and
   no other GPU work, production serving included. Stopping production is a
   planned GLM outage and needs the operator's approval.
2. Pinned identities: source commit, SHA-256 of every binary, driver, adapter,
   transport library and pack, plus the deployment and environment.
3. A working set that matches the run: diverse prompts, with expert misses
   counted. A warmed smoke `.wset` covers only its own prompt set.
4. Hashed, precomputed KV: the KV state decode starts from is precomputed and
   its hash recorded.
5. Warm-up discarded.
6. Variance reported: repeat count, median and spread.

Anything else is a functional result. Batched serving is not batch-invariant
(the COMPSEC-17 report above), so token comparisons at B > 1 need their own
reference.

## Model work rules

- Never quantize weights yourself. Packers slice, shard and repackage; they do
  not change precision. A lower-precision arm needs an official or vetted
  community release as its source.
- Model geometry and identity come from `model_contracts/<model>_authoritative.json`.
- Device memory per node is capped at 110 GiB, weightd's default
  `--device-bytes-max` (`SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT` in
  `include/sparkpipe/spark_weightd.h`). Lower it on a shared node; never raise it.
- Stop owned jobs with `cancel --id`. Stop a process outside the queue with
  TERM only after `/proc/<pid>/cwd` shows it is yours; never match by name
  alone and never `kill -9`. A wedged node (hung `nvidia-smi`) goes to the operator.
- Verify numerical agreement before timing; a mismatch stops the run. Record
  context, batch, topology and precision with every number.

## Other tools

There is no qualified one-command setup for several model families at once.
`tools/inference_smoke.py` replicas run one model, use GLM-specific flags and log
checks, accept lanes 0-7 only, and start a private weightd, so they cannot run on
a serving Spark. The older synthetic `tools/multi_dev_orchestrate.py` is not the
queue workflow. For the released GLM benchmark, binary verification and startup
failure signatures, see
[release recovery](SERVING_RELEASE_RECOVERY_20260924.md); its replay starts
private daemons and needs an operator maintenance window.
