# Fleet agent: several roots per node

`tools/fleet_node_agent.sh` supervises every runtime root on a node: the
production root and any number of dev-lane roots beside it. A root is the
directory `~/sparkdata/<name>` (a symlink is fine; processes are matched on
the resolved path).

## Which roots the agent manages

- The roots named on the agent command line (`ExecStart=... fleet_node_agent.sh
  glm53flash.fp8.tp16 sparkf`), plus
- one root name per line in `~/.fleet_agent_roots` (override the path with
  `FLEET_AGENT_ROOTS_FILE`). The file is re-read every loop, so adding or
  removing a dev root never restarts the agent. Restarting `fleet-agent`
  would kill the production residentd (it runs in the agent's cgroup with
  `KillMode=control-group`).

Names must match `[A-Za-z0-9][A-Za-z0-9._-]*`; invalid names are logged and
ignored. Removing a name stops supervision only; it does not stop the process.

## Root layouts

A root named on the command line with none of the files below is a legacy
root and behaves exactly as
before: `--rank-index` is the fleet rank, `config/stage.json` links
`stage_<fleet rank>.json`, the environment comes from the agent unit (`G5_*`
knobs, `SPARK_TP_WAIT_MODE`), the residentd runs in the agent's cgroup and
the root is production. Today's GLM root is a legacy root. A root listed only
in `~/.fleet_agent_roots` must carry `agent.env`; without it the root is a dev
root in state `blocked-config`, so a half-written layout never runs as
production.

A root that carries any of these files uses the layout rules. `<RR>` is the
node's fleet rank as two decimal digits (spark0 = `00`, sparka = `10`).

| file | meaning |
|---|---|
| `config/rank_index_<RR>` | one integer: the residentd `--rank-index` on this node; `config/stage.json` then links `stage_<index as %02d>.json`, which must exist |
| `agent.env` | `KEY=VALUE` lines for every rank of the root |
| `config/env_<RR>.env` | `KEY=VALUE` lines for this node only; later keys win |

Values are taken literally (no quoting, no expansion). Blank lines are
allowed; any other line that is not `KEY=VALUE` blocks the root. Keys that
start with `AGENT_` control the agent and are not exported:

| key | values | meaning |
|---|---|---|
| `AGENT_ROLE` | `dev` (default for layout roots) or `production` | production roots are never held back by the headroom guard |
| `AGENT_MEMORY_MAX` | systemd size, e.g. `40G` | run the residentd as the transient user unit `sp-agent-<name>` with `MemoryMax` and `MemorySwapMax=0`; required for dev roots |
| `AGENT_MEMORY_NEED_GIB` | whole GiB | memory the root adds to the node when it starts (weightd arena + residentd host/unified memory); required for dev roots |
| `AGENT_SYNC` | `release` (default) or `local` | `local` roots are never fetched from the hub release |

Every other key is exported to the residentd, together with
`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0` and `LD_LIBRARY_PATH=<root>/lib`. Layout
roots get nothing else: no `G5_*` knobs and no expert pool default, so the
weightd lane, socket, pack sha, TP mesh ranks and model switches must all be
in the env files.

A release that changes `agent.env`, `config/env_*.env` or
`config/rank_index_*` restarts the root.

## Node KV reserve

weightd owns the KV device memory of every engine that uses the common KV
binding (GLM Full, GLM Flash, K3): each engine maps one weightd KV pool, and
weightd admits pools only inside `SPARK_WEIGHTD_KV_RESERVE_BYTES`, carved out
of `SPARK_WEIGHTD_DEVICE_BYTES_MAX` (weight arenas get the rest). The agent
passes every `SPARK_WEIGHTD_*` variable in its own environment to weightd, so
set the reserve in the agent unit's environment on every node. Pools resize
inside the reserve by chunk: an engine attaches with its full size when the
reserve is free and with at least its minimum (one full-length sequence)
otherwise, waiting while idle engines give chunks back. An engine whose pool
evicts or refuses for capacity grows it (`kv pool grew`), and weightd records
the chunks it could not grant; an engine without that pressure for four 50 ms
ticks sees those wanted chunks, vacates its top pages (discarding saved prefix
pages, parking the rest) and shrinks (`kv pool shrank`). Each engine logs
`chunks=granted/total minimum_chunks=` and `pages=limit/physical` on its
`kv pool weightd` line; size the reserve to at least the sum of the minimums
for the engines a node runs, and to the sum of the totals for them to run
unshared. Without a reserve those engines refuse to load. KV writes to NVMe
are never limited or refused: each engine counts every spill, recurrent-state
and snapshot write (`kv write meter` line at load), `kv_store_report` shows
`write_bytes`, `write_bytes_this_hour`, `write_bytes_previous_hour`,
`write_alerts` and `write_alerting`, and an engine that writes more than
20 GiB within an hour (about 0.5 TB/day, a quarter of a 4 TB Spark drive's
five-year endurance pace) logs one `KV-WRITE-ALERT` line per hour and is named
in the API's `/health` as `kv_write_alert_rank`. A pool
stays in weightd when its engine stops; a clean engine restart reattaches it
and adopts the sealed resident prefix pages, and detached pools are evicted
only when a new pool needs the room.

## Start gates

Before a dev root starts (at boot, after a crash, after a release) the agent
checks, in order:

1. `agent.hold` absent (state `held`),
2. the layout parses (state `blocked-config`, reason in the heartbeat),
3. every production root on the node is `ready` (state `waiting-production`),
4. `MemAvailable - AGENT_MEMORY_NEED_GIB >= 20 GiB` (state `blocked-headroom`).
   The 20 GiB comes from the owner's headroom rule; `FLEET_AGENT_HEADROOM_GIB`
   overrides it for the whole agent.

The gates are re-checked every loop; the agent logs each state change and the
heartbeat carries the current reason. Starts that pass the gates keep the
normal restart backoff (1 s doubling to 60 s). A dev root whose residentd died while its weightd arena stayed
resident is counted at its full need; set the need accordingly.

Production roots are only gated by `agent.hold` and a valid layout. When a
production root restarts and `MemAvailable` stays below `packs + 8 GB` for
60 s, the agent stops every running dev root it manages on that node (state
`yielded`) and waits again. The dev roots stay down until production is
`ready` and the headroom check passes. Their weightd arenas stay resident
until someone runs `weightd_warm <socket> --reclaim`.

## Stopping a dev root

`touch <root>/agent.hold` drains the residentd (SIGTERM, 15 s, then SIGKILL)
and keeps it down. Remove the file to let the agent start it again. Then free
its weightd arena with `weightd_warm /tmp/spark_weightd.sock --reclaim`.
Never put `agent.hold` in a production root unless you mean to stop
production.

## Process matching

A root's residentd is a process whose resolved cwd is the root's resolved
path and whose executable is named `sparkpipe_model_residentd`. State, pid,
drain, janitor and recycle decisions all use this match, so several
residentds on one node never see each other.

## Heartbeat

`~/current/<host>.json` (copied to the hub) lists every managed root, also
missing ones, with `state`, `reason`, `pid`, `rss_mb`, `log_age_s`,
`residentd` (disk sha), `running` (sha of the running executable), `driver`,
`role`, `rank_index`, `stage`, `unit`, `memory_max`, `need_gib` and `env`
(sha of the layout files). The host record carries `headroom_gib`.
