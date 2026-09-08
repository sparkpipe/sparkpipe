# Parallel driver debugging

This is the shared workflow for every driver lane. Family code owns geometry,
tensor descriptions and model math. Reuse stage_module_common for ownership,
weightd for shared residency, the serving adapter/session lifecycle, the topology
generator and tp_device_collective. Do not fork these protocols per model.

## Queue v2 contract

One controller runs tools/spark_queue.py serve. All developer tasks on that
controller use the same default ~/.sparkpipe/queue directory. A private
SPARK_QUEUE_STATE is for hermetic tests only, never a second hardware scheduler.
Remote developers submit through the controller; they must not start their own
machine-local queue against the same Sparks.

The durable state is state-v2.json, atomically replaced and fsynced under .lock.
On first use, v1 queue/reservation/result files in that directory are imported
and retained unchanged as migration evidence. Stop the old dispatcher before
starting v2. Old running entries become legacy-review and fence their nodes;
they are not assumed dead. An operator must inspect and stop their actual
processes before migrating them. Reachability alone is not cleanup proof.

Each queued command gets a unique attempt ID and a systemd unit on every
participating node. Claims are durable before SSH. Lost launch acknowledgements
are reconciled using the retained unit, not by blindly launching another process.
Local transactions never wait for SSH; remote operations run concurrently with
bounded timeouts. CPU and GPU ownership are independent. Exclusive jobs exclude
both classes, and waiting exclusive jobs drain conflicting work before starting.

Jobs have 3-minute default / 15-minute maximum deadlines. systemd enforces the
deadline even if the controller disappears, with a 5-second termination grace.
Cancellation remains stopping until each control group is confirmed stopped.
A disconnected peer keeps its claim; successfully stopped peers become available
for assigned-node debugging. A fleet job waiting on an unresolved stopped peer
does not reserve otherwise healthy nodes. release affects manual reservations
only; it cannot release a running job.

MemoryMax defaults to 8192 MiB per job per node, with no swap. Declare enough
memory for compiler, driver, KV and workspace together. This bounds the job's
Linux cgroup, not allocations made by an independent shared weightd process.
The locked-memory allowance uses that same finite declared budget so RDMA/CUDA
registration is not restricted by systemd's default 8 MiB memlock limit.
The weightd pool must be budgeted independently. A queue memory limit alone is
not proof of bounded GPU expert residency.

Multi-node commands require --per-node: the same command runs on every node,
with SPARK_QUEUE_RANK (index in --nodes) and SPARK_QUEUE_SIZE. Do not spawn
remote peers with nested ssh, use sudo to escape the job, launch another service,
or daemonize outside its control group. All participants must belong to the job.
Use disjoint ports/socket paths for independent deployment roots.

## Commands

Run from a clean checkout of merged main on the controller:

```sh
python3 tools/spark_queue.py doctor
python3 tools/spark_queue.py list
python3 tools/spark_queue.py sync --id glm-flash-debug --nodes spark0
```

sync clones the current clean main into a temporary source checkout, rsyncs it
into a new node-local directory, verifies its Git identity and tracked content,
then renames the completed directory. It prints the exact cwd template. Reusing
an existing destination is an error; select a new lane ID. It copies committed
source and Git metadata, not local build products or model packs.

Use the printed cwd with a repo-owned command file. The command must call the
existing family validator/compiler/driver target for the intended test:

```sh
python3 tools/spark_queue.py add --id glm-cell-001 --nodes spark0 \
  --cwd '/home/{host}/srcdata/sparkqueue/glm-flash-debug/COMMIT_FROM_SYNC' \
  --ttl-min 3 --memory-mib 8192 --by glm-flash --cmd-file run-cell.sh
python3 tools/spark_queue.py status --id glm-cell-001
python3 tools/spark_queue.py cancel --id glm-cell-001
```

For a second family, use another node, lane ID and cwd. Queue all independent
jobs; one dispatch pass claims all disjoint runnable jobs. Build-only work can
use --resources cpu, but it still needs an honest memory budget.

For integration, list exactly the participating hosts and use --per-node. Each
rank starts its own residentd/validator using SPARK_QUEUE_RANK. A single-node
forward-cell pass cannot establish TP16 collective correctness. For a measured
fleet run, list all sixteen hosts and use --resources exclusive. Exclude other
out-of-queue services and prewarm the intended working set before measurements.

The controller runs:

```sh
python3 tools/spark_queue.py serve
```

dispatch and its schedule alias perform one pass. list --all includes receipts.
status returns attempt identity, per-node observations and exit status. Remote
logs are /tmp/sparkqueue-ATTEMPT.log on each participating node. systemd unit
names use that same attempt ID. Preserve logs needed for a PR before cleanup.

## Loading must fail closed

A missing or malformed PACK.experts is an error, with the path in the daemon
diagnostic. Generate the model-specific manifest using the existing family
manifest producer and verify it against the exact accepted pack. Generic byte
segments are not a substitute for routed-expert IDs.

An enabled weightd attach must return errors for missing identity/daemon,
rejected pack and failed consumer import. stage_module_common must propagate
them without allocating a direct copy. Explicitly disabled attach remains a
direct-load request; it is not a recovery path and is not lazy debugging.
The pending opt-in lazy attach in PR #829 must follow this rule as well.
Families with a specialized pack identity use SparkWeightdAttachMappedPack,
which owns attach/import validation and releases partial mappings on error.
DSV4 and the common region loader share this helper. Do not recreate their
former attach/import/fallback sequence in another driver.

Current shared lazy Ensure tests exercise the daemon's materialization and
eviction, including missing/corrupt metadata before residency. They do not yet
prove end-to-end driver lazy inference. A daemon VA is not a consumer mapping.
The remaining gate is a stable consumer import of the spine and routed working
set, held until GPU completion, with no silent whole-arena import or eager
fallback. Do not advertise the full driver workflow as lazy-qualified until
two real consumers pass cold/hit/eviction/reload, cancellation and numerical
checks inside their declared budgets.

## PR acceptance

Include source and loaded artifact SHAs, exact pack/checkpoint and generated
contract identities, model/topology/rank, command, shapes, token positions,
numerical reference and memory measurements. Compile, one-node numerical,
multi-rank functional and fleet performance receipts are separate evidence.

Deployment qualification follows merge, clean main installation on every
participant, rebuild, restart, readiness and testing. Never describe dirty or
unmerged builds as a main deployment receipt. Publish residentd, driver,
adapter, transport and config coherently; replacing only model_driver.so is
not a coherent release.

The B1 plan uses all-rank fan-out and local reduction. B2+ uses tree reduction
with compute overlap. Carry the logical batch policy across split chains:
splitting B2 into one-row halves must not accidentally select the B1 path.
Host callbacks must not call CUDA APIs. Use the existing host progress loop for
BUSY retry and submission.

## Collective regression runs

Build `build/mb_doorbell` and
`build/libhidden_transport_spark_host_rdma_verbs.so` in each synced main checkout
through a per-node CPU queue job. The benchmark target uses sm_121a and the same
shared transport library as the runtime. Then set `BENCH_CWD` to the sync result
and `BENCH_ID` to a unique job ID and run:

```sh
BENCH_ORDINAL_BASE_JUMP=1 tools/mb_run.sh 1 256 7 17 65536
```

Arguments are sync mode (0 async, 1 sync), iterations, live rows, credits and
the direct-all-to-all byte threshold (0 selects tree). This example tests seven
rows, seventeen credits and an ordinal jump during the timed interval. Use
`BENCH_SKEW_US=500` for rank-8 skew. The wrapper submits all sixteen participants
to the authoritative queue as one exclusive job; it does not launch detached
SSH children or decide completion from a readiness log line. Inspect the queue
receipt and every rank's numerical result. Logs include source and binary SHAs.
These are transport component measurements, not GLM serving throughput.
