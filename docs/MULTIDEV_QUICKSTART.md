# Shared development on the Spark fleet

The API and resident daemon provide common serving; weightd owns byte residency,
leases and mesh transport. Drivers supply layouts, routing and model math.
Each API process serves one deployment; request `model` does not select another
model. Apply the model's chat template outside the API; it only joins message content.

Use one controller, `mac@mac-studio`, and ledger `~/.sparkpipe/queue`. Verify the
merged commit in controller checkout `/Users/mac/wk-sparkpipe-queue-main-20260922`.
The operator runs one `spark_queue.py serve`; do not create another hardware queue
or set a private `SPARK_QUEUE_STATE`.

```sh
ssh mac@mac-studio
cd /Users/mac/wk-sparkpipe-queue-main-20260922
python3 tools/spark_queue.py doctor
python3 tools/spark_queue.py list --all
```

## Operator: establish the shared services once

Run one supervised weightd per physical node. Set finite systemd `MemoryMax`,
`MemorySwapMax=0`, finite memlock and weightd `--device-bytes-max`. Reserve its
ports, stable socket and private mesh-record directory. Set physical `--mesh-rank`,
`--mesh-rank-mask`, `--mesh-interface` and `--mesh-sgid-index`; exchange fresh peer
records and verify readiness. Model wrappers attach to this existing service.

Track each running service (replace the example unit and uppercase budgets/ports):

```sh
python3 tools/spark_queue.py track --node spark0 \
  --unit sparkpipe-weightd-shared.service --scope system \
  --device-memory-mib DEVICE_MIB --ports FIRST:LAST
```

Repeat per node. `track` checks the invocation, cgroup and finite host bound; it does not start a daemon.
Include CUDA overhead in its device budget; reconcile restarts with `untrack`/`track`.

Coordinate lanes **0 through 7**. Maps are immutable for the daemon lifetime;
changes require draining users and operator replacement. A crash can fence the mesh.

## Developer: submit one bounded model job

Choose an assigned lane, working set and measured budgets. Keep ports, runtime, KV and logs private.
Set `SPARK_WEIGHTD_SOCKET` and deployment `weightd.socket_path` to the shared socket.

Commit the family wrapper/configuration, then sync exact source:

```sh
python3 tools/spark_queue.py sync --id dev-model-001 \
  --nodes spark4,spark5,spark6,spark7 --ref HEAD
```

Use the printed `CHECKOUT`; build coherent family artifacts through the queue
(CUDA validation requires GPU admission). Put the remote command in controller
file `run-family-job.sh`. Its family wrapper prepares a private deployment under
`$SPARK_QUEUE_RUNTIME_ROOT` and keeps children in its queue cgroup. Its runtime
`packs/` directory must contain exactly one valid `*.sha256` digest for shared
weightd attachment. The resident launch uses the common executable:

```sh
export SPARK_WEIGHTD_SOCKET=/actual/shared/weightd.sock
export SPARK_WEIGHTD_LANE=3
export SPARK_TP_MESH_RANKS=4,5,6,7
exec /actual/verified/bin/sparkpipe_model_residentd \
  --deployment "$SPARK_QUEUE_RUNTIME_ROOT/deployment.json" \
  --rank-index "$SPARK_QUEUE_RANK"
```

After preparing the deployment and mesh backend, submit the complete wrapper:

```sh
python3 tools/spark_queue.py add --id dev-model-run-001 \
  --nodes spark4,spark5,spark6,spark7 --per-node --cwd CHECKOUT \
  --resources gpu-shared --memory-mib TOTAL_MIB --device-memory-mib DEVICE_MIB \
  --ports FIRST:LAST --ttl-min 14 --by DEVELOPER --cmd-file run-family-job.sh
python3 tools/spark_queue.py status --id dev-model-run-001
```

`TOTAL_MIB` includes host and device budgets; the host cgroup gets the difference.
Tracked weightd memory is additional. Reserve every listener; use `cancel --id ID`
for owned jobs. No nested SSH GPU launches or per-job replacement weightd.

Logical rank is the index in `--nodes`; keep this order identical to the map:

| Topology | `SPARK_TP_MESH_RANKS` | Node order |
| --- | --- | --- |
| TP16 identity | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` | spark0 through sparkf |
| TP16 permutation | `0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1` | spark0, spark2 through sparkf, spark1 |
| TP4 subset | `4,5,6,7` | spark4,spark5,spark6,spark7 |

Pin packs, manifests and working sets; cap expert pools, spine, KV and workspace.
GLM graphs require `SPARK_GLM5_NEXT_GRAPH_PATH=1`, `SPARK_GLM5_NEXT_PIN_EXPERTS=1` and a full pool.
Partial-pool eager mode (`0`/`0`) is not GPU-qualified by the parallel GLM receipts.

There is no qualified one-command setup for eight different model families.
`inference_smoke.py` replicas run the same model, use GLM-specific flags/log checks
and start a private weightd. Other families need validated shared-socket wrappers.
The older synthetic/SSH `multi_dev_orchestrate.py` is not the queue workflow.
See [parallel driver debugging](PARALLEL_DRIVER_DEBUG.md) for provenance/receipts.
