# WEIGHTSD — the stable weights-daemon channel

weightsd is the **stable weights-daemon channel** for driver/dev testing. It is
a **deployment identity of the same source tree as weightd** (same `node/weightd.c`,
same `runtime/spark_weightd*.c`, built by the same Makefile into its own binary
name). It is not a fork and carries no version pinning: the weightsd channel
tracks **origin main**. weightd remains coredev's debug instance and can be
restarted freely during debugging without touching anything that points at
weightsd.

## Service identity

| thing | value |
|---|---|
| binary | `build/sparkpipe_weightsd` (same sources as `build/sparkpipe_weightd`) |
| unit | `tools/devcycle/sparkpipe_weightsd.service` → `/etc/systemd/system/sparkpipe_weightsd.service` |
| env file | `/etc/sparkpipe/weightsd.env` (`WEIGHTSD_BIN`, `WEIGHTSD_SOCKET`, `WEIGHTSD_EXTRA_ARGS`) |
| socket | `/run/sparkpipe-weightsd/weightsd.sock` (RuntimeDirectory `sparkpipe-weightsd`) |
| state dir | `/var/lib/sparkpipe-weightsd` (StateDirectory) |
| deploy | `tools/weightsd_deploy.sh` (run on the node; build + install + enable + health check) |
| purge | `SPARK_WEIGHTD_SOCKET=/run/sparkpipe-weightsd/weightsd.sock build/weightdctl reclaim` |
| port block | **61000-61127** (PORT_LEDGER, service planes) |

The live client endpoint is the **unix socket** (the converged weightd-mesh
collective consumes only `SPARK_WEIGHTD_SOCKET`; ledger 09-11 E-3). The 61000-61127
block is the weightsd service plane reserved per ledger rule 2 for any future
control/transport bases; no endpoint in code or unit hardcodes it. The mesh is
opt-in: state `--mesh-rank/--mesh-interface/--mesh-sgid-index` via
`WEIGHTSD_EXTRA_ARGS` in the env file.

## The client seam — config only, zero code changes

Every weightd client discovers the daemon through configuration, never a
hardcoded default. With the seam unset, drivers use the resident path and no
daemon is contacted — nothing forces weightd. To point a driver at weightsd,
set the SAME fields you would set for weightd, to the weightsd socket.

Direct-environment families (glm52, glm5_next, dsv4, qwen38max stage modules,
`weightdctl`, `weightd_smoke`, `weightd_lazy_consumer`, `weightd_loadall`,
`weightd_execute_probe`):

```sh
export SPARK_WEIGHTD_SOCKET=/run/sparkpipe-weightsd/weightsd.sock
export SPARK_WEIGHTD_ATTACH_LAZY=1
export SPARK_WEIGHTD_PACK_SHA256=<pack digest from placement receipt>
```

`SPARK_WEIGHTD_ATTACH=0` conflicts with a configured socket and fails by design.

Deployment-JSON families (residentd-supervised, `docs/WEIGHTD_SUPERVISED_STARTUP.md`):

```json
{
  "weightd": {
    "socket_path": "/run/sparkpipe-weightsd/weightsd.sock"
  }
}
```

The deployment must also carry the pack digest sidecar (`packs/*.sha256`,
exactly one per rank runtime root). residentd probes the socket and fails fast
if the weightsd service is down — it never spawns or repairs the daemon; the
service owner (this unit) provides supervision, exactly as that document
requires for a shared cache daemon.

## State separation / coexistence with weightd

Lease state is **in-process**: a lease table of 64 leases x 512 expert groups
per map, eviction-epoch control page inside the client map reservation
(`runtime/spark_weightd_lease.h`, `runtime/spark_weightd_map.c`). There is no
shared lease database, so a co-resident weightd and weightsd cannot collide as
long as they do not share a socket — and they do not:

- weightd (debug): `/tmp/spark_weightd.sock` (its default)
- weightsd (stable): `/run/sparkpipe-weightsd/weightsd.sock`

Pack digest sidecars are per-deployment read-only inputs. Arenas are per-client
attach state inside each daemon process. Verified by the spark0 coexistence
receipt (both daemons up, independent sockets, independent execute waves,
independent reclaim counts).

## Restart policy

**weightsd restarts ONLY for a main-branch upgrade, announced in advance, never
in the middle of a running test.** The deploy tool enforces the guard:
`tools/weightsd_deploy.sh` refuses to restart an active service unless
`WEIGHTSD_DEPLOY_RESTART=1` is set (the announced-upgrade path). Crash recovery
(`Restart=on-failure`, 5s, StartLimitBurst 3/60s) is always on — that is not a
restart of the channel, it is the service staying up.

## Escalation

weightsd bugs are weightd bugs. There is one source and one fix path: file/PR
against the shared tree (`node/weightd.c`, `runtime/spark_weightd*.c`) on main.
Do not patch a node's weightsd in place; a node diverges from main only by
being stale, and the fix is redeploy from main.

## Acceptance

Per-node acceptance is D-1's execute-receipt rig
(`tools/weightd_execute_receipt.py` + `build/weightd_execute_probe`): 50 fixture
lifecycles plus real-pack lease/execute waves, daemon-alive and zero-daemon-error
gates, JSON receipt. Receipts for the fleet rollout are archived under
`runs/weightsd-channel/`.
