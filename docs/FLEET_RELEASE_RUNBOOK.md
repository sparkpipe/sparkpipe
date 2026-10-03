# Fleet release runbook

The one current guide for the production Spark fleet: hosts and routes, the
node agent, weightd ownership, releasing a runtime root, releasing the agent
and weightd, the GLM API on the hub, bootstrap and triage. It was checked on
2026-09-28 against origin/main `bbf5432` and read-only inspection of the hub
and all sixteen Sparks. Sections 4.2-4.6, 5.2 and 5.3 were rewritten on
2026-09-29 from the d37568d and a597ff0 releases and origin/main `dfa12a5`;
they use the release kit in `tools/fleet_release/`. The agent line references
(`:NNN`) in sections 2, 7 and 8 date from `bbf5432`; the agent has moved since,
so look functions up by name. It replaces `FLEET_RELEASE.md`, `FLEET_RUNBOOK.md` and
the unmerged 09-28 fleet handoff, which are now history in
[`archive/`](archive/). Queue and lane work is in [DEVCYCLE.md](DEVCYCLE.md)
and [PARALLEL_DRIVER_DEBUG.md](PARALLEL_DRIVER_DEBUG.md).

The rule: never deploy by killing and reloading engines by hand. Pause the
fleet rotation, hold the root, install verified files into the hub, write the
MANIFEST last, wait until every node has applied and verified it, release the
hold and let each node's agent converge. Bare line references (`:NNN`) are to
`tools/fleet_node_agent.sh`.

## 1. Hosts and network

| Role | Host | What lives there |
| --- | --- | --- |
| Hub | `rtx5090`: x86_64, one RTX 5090 32 GB, user `spec`, tailscale `100.123.97.61` | `~/release/<root>/` served on `:8802` by the user unit `fleet-release` (`tools/fleet_release_serve.py`, installed as `~/fleet_release_serve.py`); `~/current/<host>.json` heartbeats; `~/release/qpn/<host>/` mesh records; the GLM API, user unit `g53-api` on `:8433` (section 6) |
| Build host and rank 15 | `sparkf`: aarch64, `sm_121a` | `~/g5n-rd-build` build tree. sparkf is not the hub: no agent reads its `~/release` |
| Fleet | `spark0`-`spark9`, `sparka`-`sparkf` | `~/sparkdata/core/` core root, `~/sparkdata/weightd/` installed weightd, `~/sparkdata/<root>/` runtime root and packs, `~/current/<host>.json` local heartbeat |

A node's rank is its position in `FLEET_HOSTS` (`:8-23`). Pack names use hex
ranks.

Network (lead-dev facts and read-only checks, 2026-09-28):

- The hub's `eno1` (`10.10.250.2/30`) is cabled to sparkf's `enP7s7`
  (`10.10.250.1`) at 10 GbE. `enP7s7` was sparkf's management port
  ([SPARK_MANAGEMENT_FAILOVER.md](SPARK_MANAGEMENT_FAILOVER.md)).
- The other fifteen Sparks route `10.10.250.0/30` via sparkf's switched address
  `10.10.100.25` on `enp1s0f1np1`. The system unit
  `sparkpipe-hub-route.service`, which is not in this repository, runs
  `ip route replace 10.10.250.0/30 via 10.10.100.25 dev enp1s0f1np1`. All
  fifteen carried the route.
- Busy-polled 64 B UDP round trip: sparkf to hub 22.6 µs p50 and 25.2 µs p99;
  other Sparks, forwarded by sparkf, about 300 µs p50.
- Agents do not use this link for releases. They fetch over tailscale from
  `http://100.123.97.61:8802` (`:218`, `FLEET_HTTP_RELEASE` overrides) and scp
  heartbeats and mesh records to the `HUB` argument, `spec@100.123.97.61`
  (`:101-102`, `:228-248`). A `HUB` of `sparkf` is a legacy alias that the agent
  maps to the same address (`:5-6`).

The hub also runs unrelated workloads, the chess/nnue training among them.
Fleet work there touches only `~/release`, `~/release-staging`, `~/current`,
`~/g53-api-channel` (and its `.prev` copy) and its own build directories, and
restarts only `g53-api` or `fleet-release`.

## 2. The node agent

### 2.1 Unit, drop-in and cgroup

`fleet-agent` is a systemd user unit on every Spark. The base unit is the
one `tools/fleet_sync.sh start` writes (`fleet_sync.sh:37`), identical on all
sixteen on 2026-09-28:

```ini
ExecStart=%h/sparkdata/core/bin/fleet_node_agent.sh glm53flash.fp8.tp16 spec@100.123.97.61
Restart=always
RestartSec=3
```

The agent must run from `~/sparkdata/core/bin/`, the file that `sync_core`
replaces and `self_update` executes (`:370-379`).

Every Spark carries the same drop-in,
`~/.config/systemd/user/fleet-agent.service.d/20-serving.conf` (sha256 prefix
`8324336487eecc38` on all sixteen, 2026-09-28):

```ini
[Service]
Restart=always
RestartSec=3
Environment=G5_API_DISABLED=1
Environment=G5_WARMUP=0
Environment=G5_GRAPH_PATH=1
Environment=G5_PIN_EXPERTS=1
Environment=SPARK_TP_WAIT_MODE=hardware
```

| Variable | Effect |
| --- | --- |
| `G5_GRAPH_PATH=1` | Passed to the engine as `SPARK_GLM5_NEXT_GRAPH_PATH` (`:199`): CUDA-graph chains. |
| `G5_PIN_EXPERTS=1` | Passed as `SPARK_GLM5_NEXT_PIN_EXPERTS` (`:198`). The module pins every expert at boot (`spark_glm5_next_resident_decode_stage_module.c:664`). Graphs require full pinning since `78c2c21`. |
| `SPARK_TP_WAIT_MODE=hardware` | Inherited by the engine and read by the device collective (`ring/transport/tp_device_collective.c:1420`). |
| `G5_API_DISABLED=1` | No longer read. Since #1261 the agent has no `ensure_api` and never starts an API; the API is `g53-api` on the hub. The line is inert and can leave the drop-in. |
| `G5_WARMUP=0` | Rank 0 does not send its warmup completion to `G5_API_HOST`, which defaults to `100.123.97.61:8433` (`:529`, `:550`). |

With this drop-in, GLM 5.3 Flash measured 36 tok/s at B1: 23-24.5 ms/token,
8-step graph chains, all 12096 experts pinned (lead-dev measurement,
2026-09-28). A node rebuilt without it runs a different configuration. Check
that `sha256sum ~/.config/systemd/user/fleet-agent.service.d/20-serving.conf`
matches on all sixteen.

The agent starts weightd with `setsid nohup` (`:464`) and the engine with
`nohup ... &` (`:200`). Neither leaves the unit's cgroup, and the unit uses
the default `KillMode=control-group`. On spark0 on 2026-09-28, the cgroup held
the agent, `~/sparkdata/weightd/sparkpipe_weightd` and
`./bin/sparkpipe_model_residentd`. Therefore:

- `systemctl --user stop fleet-agent` or `restart fleet-agent`, a drop-in change
  (which needs `daemon-reload` and a restart), and any exit of the agent all
  take down weightd and the engine on that node. The engine then comes back
  cold and pins every expert again. Treat each of these as a planned outage.
- A new agent restarts nothing: `self_update` is an `exec` in place (`:378`).
- The user manager must linger (`sudo loginctl enable-linger <user>`).
  Without it, systemd stops the user manager, and with it fleet-agent,
  weightd and the engine, when the user's last login session ends, and does
  not start it after a reboot until someone logs in. On 2026-09-28 linger was
  enabled on spark0-spark7 and sparke only; spark8, spark9, sparka-sparkd and
  sparkf had no `/var/lib/systemd/linger` entry and were running on open
  sessions (read-only check).

### 2.2 One loop pass (every 1 s, `:559-578`)

1. `sync_core` (and `sync_root` per root in step 6) fetches the root's
   `MANIFEST`. If it differs from `.applied_manifest`, the agent fetches only the
   entries whose sha changed, verifies each sha, moves the file into place and
   sets `bin/*` to mode 755 (`:271-309`). Files are not re-verified while the
   MANIFEST is unchanged (`:279`). A file replaced on a node by hand stays until
   the hub changes that entry.
2. `install_core` acts when `core/WEIGHTSD_BIN` equals the sha16 of
   `core/bin/sparkpipe_weightd` and differs from the installed weightd. It
   installs the binary into `~/sparkdata/weightd/` (temp + `mv`) and restarts
   nothing (`:356-368`).
3. `self_update` executes a changed `core/bin/fleet_node_agent.sh`
   (`:370-379`).
4. `node_doctor` flaps the netdev of `rocep1s0f1` when the port is not
   `PORT_ACTIVE` (`:381-395`). `janitor` sends `kill -9` to every duplicate
   engine in the same root that is more than 1800 s old, keeping the
   longest-running one; its log line says the opposite (`:397-415`).
5. `ensure_weightd` (`:417-471`). If it fails, the pass only reports and stops
   there (`:565-569`): no root sync, engine start, API or warmup runs on that
   node. It fails when:
   - a `sparkpipe_weightd` runs from any other path ("unknown owner",
     `:428-430`);
   - the installed weightd differs from the running one while an engine runs
     ("requires dependent engines to drain", `:440-444`). With no engine
     running, it sends TERM and starts the new binary on a later pass
     (`:445-447`);
   - the control socket is not accepting yet, or weightd was just started.

   weightd is launched with `--socket /tmp/spark_weightd.sock --mesh-rank R
   --mesh-rank-mask 0xffff --mesh-interface rocep1s0f1 --mesh-sgid-index 3`
   (`:464-466`).
6. `sync_root` decides the restart scope (`:311-348`):
   - a changed engine binary, `lib/*`, `stages/*` or
     `config/model_resident.json` restarts the root: TERM, 15 s grace,
     `kill -9` fallback, memory gate, start;
   - a changed `bin/sparkpipe_model_api` only drains the API;
   - a changed `config/stage_*.json` only relinks `config/stage.json`.

   The root's deployment file is `model_resident.json` at the top of the root
   (`:186`, `:201`, and the hub MANIFEST), which matches none of these patterns.
   A change to that file alone lands on disk without a restart. Ship it with an
   engine, library or stage change, or drain the root (section 4.3).
7. The mesh records are not exchanged in this loop. `start_mesh_exchange`
   runs `mesh_exchange_loop` in the background from agent start, once per
   second, independent of roots and of `ensure_weightd`:
   - `mesh_push` publishes this node's `mesh-<rank>.rec` to
     `release/qpn/<host>/mesh/` with one ssh that writes a temporary file and
     renames it. Every 30 s it compares the hub copy with the live record and
     publishes again on a mismatch;
   - `mesh_pull` fetches the 15 peers' records in parallel with conditional
     GETs (`curl -z`, 2 s timeout each) and a full fetch every 60 s, and
     renames each download into place. It logs once when the release server
     becomes unreachable and once when it returns.

   A self-update stops the loop before `exec`; the new agent starts its own.
8. `ensure_root` starts a down root. It also recycles an engine whose binary or
   driver changed since boot (`:482-523`). The guards:
   - a node up less than 900 s does not start engines (`:516`);
   - a per-class backoff doubles up to 60 s (`:164-175`);
   - `MemAvailable` must reach the packs' `du -sBG` plus 8 GB (`:150-158`).
     `du -sBG` rounds small files up, so the threshold reads high;
   - a multi-rank root waits for `/tmp/weightd-mesh/.ready` (`:186-190`).
9. `warmup_hook` runs on rank 0 only; `G5_WARMUP=0` turns it off. The agent
   never starts an API.
10. `report_if_changed` (`:105-118`).

The agent writes its heartbeat locally and copies it to the hub only at agent
start, when an engine starts, and when a root's state or pid changes
(`:105-118`, `:203`, `:481`). A steady node does not report again, and a dead
node leaves its last `ready` file on the hub. The `epoch` field is therefore
not a liveness signal; check the nodes themselves (section 4.3).

### 2.3 Several roots per node

Dev-lane roots run beside production under the same agent. Each root is listed in `~/.fleet_agent_roots` and can carry its own:
- rank index (`config/rank_index_<RR>`);
- environment (`agent.env`, `config/env_<RR>.env`);
- `MemoryMax`.

The agent matches each root's residentd by its resolved cwd and exe, never by the first `pgrep` hit. A dev residentd with a lower pid therefore can no longer make GLM look down; on 2026-09-28 that bug recycled GLM rank 9 every minute.

Before starting a non-production root, the agent checks that at least 20 GiB of `MemAvailable` would remain. If not, it holds the root as `blocked-headroom`. The production root is never held.

See [FLEET_AGENT_ROOTS.md](FLEET_AGENT_ROOTS.md).

## 3. weightd ownership

Each node's production weightd is the agent's:
`~/sparkdata/weightd/sparkpipe_weightd`, socket `/tmp/spark_weightd.sock`,
started only by `ensure_weightd`. weightd holds a loopback TCP latch on port
61900 and exits if it cannot bind it (`node/weightd.c:72`,
`node/weightd.c:343-362`; `SPARK_WEIGHTD_LATCH_PORT` overrides the port). A
second weightd on the same node therefore needs its own latch port and socket. Even then, any
`sparkpipe_weightd` that runs from another path freezes that node's agent loop
(section 2.2, step 5).

Do not start another weightd on a production node, whether from a queue job,
a shared-weightd unit or the old weightsd system unit
(`/etc/sparkpipe/weightsd.env`). The mesh has 16 lanes
(`include/sparkpipe/spark_weightd.h:72`); a procedure for attaching dev lanes
to the production weightd is not documented yet.

## 4. Releasing a runtime root

### 4.1 Build (aarch64)

Build from a merged main SHA, in a clean checkout from `spark_queue.py sync`,
with a GPU-owned queue job that runs `bash tools/glm5_next_build_release.sh`
([PARALLEL_DRIVER_DEBUG.md](PARALLEL_DRIVER_DEBUG.md#glm-firmware-build-for-development-and-pr-testing)).
That script runs `tools/module_build_release.sh glm5_next_resident_decode_stage
fp8 glm53_release 84c6a6aa9497188e15a635ba793b0f95a79b1033
model_contracts/glm53_flash_authoritative.json`
(`glm5_next_build_release.sh:7`).

From the workstation checkout (the controller's ledger is
`~/.sparkpipe/queue`), the release lane runs:

```sh
SHA=<MERGED_MAIN_SHA>; S7=${SHA:0:7}
python3 tools/spark_queue.py sync --id release-glm53-$S7 --nodes sparkf --ref $SHA
python3 tools/spark_queue.py add --id release-glm53-$S7-build --nodes sparkf \
    --resources gpu --memory-mib 32768 --ttl-min 15 --by lane-release \
    --cwd /home/sparkf/srcdata/sparkqueue/release-glm53-$S7/$SHA \
    --cmd 'bash tools/glm5_next_build_release.sh'
python3 tools/spark_queue.py status --id release-glm53-$S7-build
```

On 2026-09-28 the job for `09fdad6` was admitted in under 10 s and finished
in about 50 s with `BUILD-PASS`. The job log is
`/tmp/sparkqueue-<attempt>.log` on sparkf. `build/glm53_release.tar.gz.sha256`
names the tarball relative to the checkout root, so run `sha256sum -c` from
there. The queue is used only for this build, because
`module_build_release.sh` requires `SPARK_QUEUE_ID`; do not set it by hand
and do not edit the guard.

`module_build_release.sh` takes exactly five arguments. It requires
`SPARK_QUEUE_ID`, a clean tracked tree, and no `build/obj` or `build/modules`
(`module_build_release.sh:3-7`, `module_build_release.sh:17-23`). It writes
only `build/glm53_release/`, plus `build/glm53_release.tar.gz` and its
`.sha256` (`module_build_release.sh:61-77`). It does not write `~/sparkdata/out`, publish
to the hub or restart an agent.

Verify before staging:

```sh
cd build/glm53_release
sha256sum -c --quiet SHA256SUMS
cat SOURCE_COMMIT                                                  # the merged main SHA
grep -o 'validation=[a-z]*' qualification/serving-receipts/publish.log   # executed
grep 'component validator' qualification/serving-receipts/publish.log  # glm5_next component validator: PASS
```

Two traps from 09-28, from the handoff's evidence:

- A build tree's `remote.origin.fetch` was pinned to a lane branch. Its
  `origin/main` stayed at a Sep-18 commit while the build reported success.
  Trust `SOURCE_COMMIT`, not a branch name.
- A reused module-library record (`validation=reused`) hid unverifiable
  provenance. A fresh checkout avoids it.

Do not publish with `tools/publish_local.sh` or `tools/publish_core.sh`:

- Both write the local host's `~/release`, not the hub's.
- `publish_core.sh` ignores `SPARKPIPE_BUILD_TREE` and always installs from
  `~/sparkpipe-build` (`publish_core.sh:4`).
- `publish_local.sh` installs `model_driver.so` from `~/sparkdata/out`
  (`publish_local.sh:32`), which the current build never writes. It pairs that
  driver with the engine, adapter and transport from the build tree, which
  makes a mixed release.

Never rsync sparkf's `~/release` to the hub. On 2026-09-28 it held:

- the de18669 agent `5c272d577aaf8647`, which lacks `--mesh-rank-mask`;
- a stale `WEIGHTSD_BIN` (`39c28190e92ea878`);
- the failed 53d33dd root.

(Audit, read-only.)

### 4.2 The release kit

`tools/fleet_release/` carries every step below as a script that checks
before it writes, logs to `$LOGS/<step>-<utc>.log`, and exits non-zero on
anything unexpected. The kit has no model knowledge. Everything
model-specific sits in a profile under the deployment, and everything
release-specific sits in one env file:

| File | Holds |
| --- | --- |
| `deployment/<deployment>/release/profile.env` | nodes, hub, root name, rank-pack template (`@hex@` = the node's position in `RELEASE_NODES`), driver path, build files, API channel, unit and port, x86 build command, required API log lines, rotation pause file and unit, memory gates, converge expectations |
| `deployment/<deployment>/release/checks.json` | engine log gates (`converge_logs`, `smoke_logs`, `perf_logs`), allowed ERRSITEs, the smoke requests, the perf windows and score gates |
| `deployment/<deployment>/release/probe_extra.sh` | extra node columns (for GLM Flash: `kvs`, `kvref`, `l2`, `gmode`) |
| `deployment/<deployment>/release/perf_expect.json` | the previous release's numbers, ranges and roofline inputs |
| `deployment/<deployment>/release/stage_drift.json` | why the served stage configs differ from the generator |
| release env (lane directory, from `tools/fleet_release/release.env.example`) | sources the profile, then `SHA`, old/new identities (residentd, driver, weightd, API, adapter, channel deployment, root MANIFEST), `EXPECT_CHANGED`, `CHANNEL_ADD`, `STAGE_CONFIG_ADD` |

GLM Flash's profile is `deployment/glm5_next_tp16/release/`. Every value is
required: a missing or `PENDING` value stops the step with exit 2, and
optional features are switched off with an explicit `none`. Run each step as
`RELEASE_ENV=<file> bash tools/fleet_release/<step>.sh`.

Preparation (production keeps serving):

1. `build.sh`: a shallow clone of the merged SHA, rsynced to the build host
   and built under a capped `systemd-run` unit; it refuses while
   `PERF_HOLDER` starts with a prefix in `RELEASE_GPU_HOLDER_BLOCK`.
2. `stage_config_patch.py patch` when the release changes stage configs: it
   appends the new members to every served config (byte format kept, checked
   against the served MANIFEST) and explains every member where the result
   differs from the generator (`--generator`, `--committed`, `--drift`). The
   hub re-checks the result with `stage_config_patch.py check` in
   `assemble_root.sh` and `precheck.sh`.
3. `stage.sh`: the tarball and new configs to the hub, `hub/assemble_root.sh`
   (the served root plus the build files, MANIFEST regenerated), the x86 API
   and adapter build from `git archive $SHA`, and `hub/stage_channel.sh`,
   which adds each `CHANNEL_ADD` block (for example the declared chat
   template, `chat_template=model-families/glm5_next/chat_template.json@<sha256>`)
   to the live channel deployment and checks that every stop marker is
   exactly one special token of the channel tokenizer. Copy the printed
   identities into the release env.
4. For a weightd change, stage the bundle first (section 5.2).

### 4.3 Before the window: the rotation

The fleet rotation (`tools/fleet_rotation.py`, [FLEET_ROTATION.md](FLEET_ROTATION.md))
drives production through the same `agent.hold` file, stops and starts the
production API, and runs `--reclaim-pack` on the production packs. A tick
that runs during a release would hold or unhold production underneath the
release scripts. So every release step starts with the rotation paused:

```sh
T="python3 ~/fleet-rotation/bin/fleet_rotation.py --config ~/fleet-rotation/rotation.json"
ssh rtx5090 "$T pause release <sha7>"
ssh rtx5090 'systemctl --user show -p ActiveState --value fleet-rotation.service'   # inactive, not activating
ssh rtx5090 "$T status"
```

`pause` does not wait for a running tick; the tick stops at its next step.
The kit's `rotation_gate` (in `precheck.sh`, `hold.sh` and `publish.sh`)
refuses unless `~/fleet-rotation/ROTATION_PAUSE` exists and the unit is
`inactive` or `failed`.

Then make production the only model on the fleet:

- In a `flash_plus` hour production already serves; its companion may keep
  running through a root-only release, but not through a weightd change or
  `perf.sh`.
- In a FULL or K3 hour production is held and the slot model owns the
  memory. Run `$T converge flash` (a manual converge ignores the pause):
  it stops the slot model, reclaims its packs by sha, and restores
  production alone with its API and smoke.
- For a weightd change or the perf gates, also stop the companion:
  `$T converge flash` leaves production alone.

After `SMOKE PASS` (or after `perf.sh`), run `$T resume`. The next tick
adopts the fleet as it finds it and moves to the hour's target. If the
release changed production's memory per node, update the `flash` entry's
`mem_gib` in `deployment/fleet_rotation/rotation.json` so the rotation's
floor prediction stays true.

A `lead-*` `PERF_HOLDER` also preempts the rotation within one tick, but only
once the coord-sync patch mirrors it to the hub; the pause file does not
depend on that.

### 4.4 The window: hold, publish, applied, unhold, converge, API, smoke

This is the flow that released d37568d on 2026-09-29 (MEASURED, lane logs in
`/Users/mac/wf/release-next/logs`, GLM Flash TP16, weightd unchanged):

| Step | Script | What it checks, then does | d37568d |
| --- | --- | --- | --- |
| 1 | `precheck.sh` | read-only: rotation paused, SHA on origin/main, CI green, hub assembled root and channel, build host clean, every node on production, weightd bundle staged when weightd changes | PASS 13:09:52Z |
| 2 | `hold.sh` | node gate (production identities, not held; for a weightd change also no other residentd and a staged bundle with valid receipts), then `touch agent.hold` on every node; each agent drains its residentd (TERM, `kill -9` after 15 s). Waits up to 3 min for `hold=yes eng_n=0` everywhere. **The model is down from here.** | HOLD DONE 13:09:57Z |
| 3 | `weightd.sh publish`, `announce`, `wait` | only for a weightd change (section 5.2) | - |
| 4 | `publish.sh` | hub `WEIGHTSD_BIN` is the new weightd, served MANIFEST is production, every node held and down with the new weightd running; then `hub/publish_root.sh` under `.publish.lock`: rollback copy of every changed file to `~/release-rollback-<ts>-<prev>` (path in `ROLLBACK_DIR`), changed files by temp + `mv`, **MANIFEST last** after `sha256sum -c` of the new MANIFEST against the served tree | PUBLISHED, 24 files |
| 5 | (inside `publish.sh`) | waits up to 5 min until every agent has **applied** the new MANIFEST (`.applied_manifest` sha) and the node's root **verifies** (`sha256sum -c .applied_manifest`, column `rootok`). Held roots sync files but never start | applied + verified 16/16 13:10:04Z |
| 6 | `unhold.sh new` | served MANIFEST is the release, every node held, down, applied, verified, weightd as expected, MemAvailable >= `RELEASE_MEM_UNHOLD_GIB`; then `rm agent.hold` everywhere | 13:10:07Z |
| 7 | `converge.sh new` | waits up to `RELEASE_CONVERGE_MIN` for every node: one residentd of the new sha from the root, new driver, applied MANIFEST, new weightd running (one process), ready, MemAvailable >= floor, plus `RELEASE_CONVERGE_EXPECT` (GLM Flash: `l2=on kvref=0 kvs=1of16 gmode=graph`). Prints the per-node pack-verify mode (`verify=receipt` means the verify-once receipt held, section 5.2) and runs the `converge_logs` gates | CONVERGED 13:10:22Z |
| 8 | `api_install.sh` | engines on the release everywhere; then `hub/api_install.sh`: the staged api, adapter and channel deployment by sha, the deployment equals the live one plus the `CHANNEL_ADD` keys only, tokenizer block and vocabulary checked, backup to `~/<channel>.bak-<prev>-<ts>` (path in `API_BACKUP_DIR`), temp + `mv`, `SHA256SUMS`, one unit restart, health with tokenizer, every `RELEASE_API_REQUIRED_LOG` line in the new `api.log` (GLM Flash: `tokenizer sidecar ready`, `chat_template declared stop_markers=3`) | API-INSTALLED 13:10:28Z |
| 9 | `smoke.sh` | `hub_smoke.py` sends the profile's requests (health, warmup ids, two chats with no think block and the old prompt length, system + user, role `tool` refused with 400, a text completion), scans `api.log` since the install for ERRSITEs outside `api_errsite_allow`, runs the `smoke_logs` gates (graph mode on every rank, no graph failures, a rows=1 capture, engine ERRSITEs outside the allow list), and re-checks the identities | SMOKE PASS 13:10:31Z |
| 10 | rotation | `$T resume` (or keep it paused for step 11) | - |
| 11 | `perf.sh` | production alone (other residentds refuse unless `ALLOW_OTHERS=1`); the profile's perf windows through `perf_window.py` as `lead-<sha7>-<label>`, its score gates, the `perf_logs` gates on copied engine logs, `tp_chain_budget.py`, and `perf_summary.py` against `perf_expect.json`, with a roofline line on every B1 number | see below |

The d37568d outage, hold to `SMOKE PASS`, was 39 s. The a597ff0 release,
which also swapped weightd, took 54 s from drain to converged with every
node in `verify=receipt` mode (MEASURED, `/Users/mac/wf/release-a597ff0/logs`).

d37568d perf gates (13:10-13:30Z, qwen up, non-spec, temp 0): B1 128 21.42
ms/token, 42.21 wall tok/s; B1 512 20.97 ms/token; TTFT 372/742 tokens 1.30/2.52 s
(a597ff0 1.02/1.96 s: flagged REGRESSION); warm B8/B16 118.5/155.8 aggregate
tok/s; COMPSEC-17 seq 15/17 twice (byte-identical), c17 13/17 and 14/17.
roofline @B=1 at 21.42 ms/token: memory 49% (2.56 GB/token/node vs 243 GB/s,
ceiling 95 tok/s) | compute ~1-2% | transport bw 1% + latency 71% (114 rounds
x 40 us floor against a 6.46 ms peer term). The perf gate failed on the two
TTFT numbers; that is a finding for the kv_shard rounds lane, not a release
blocker by itself.

### 4.5 Node gates and re-sampling

Every gate above is one call of `tools/fleet_release/nodes.py`: one ssh per
node runs `node_probe.sh` (plus the profile's `probe_extra.sh`) and prints one
`key=value` row; `--expect KEY=V|KEY!=V|KEY>=N|KEY<=N` rules are checked on
every row, and a column the probe does not print fails the rule. The probe
counts only processes whose `/proc/<pid>/exe` is `sparkpipe_model_residentd`
and whose cwd is the root, so a `pgrep` whose own command line names the
binary is not an engine.

- `--wait-min M --interval S` polls until every node passes or the deadline
  passes.
- At the final decision (at once for a gate without `--wait-min`, at the
  deadline for a wait), only the failing nodes are probed again, up to
  `--resample` times (default 2) `--resample-interval` apart (default 5 s).
  A node that passes on a re-sample is logged `TRANSIENT <host>: ... cleared
  on resample n/N`; one that still fails fails the gate.
- Every run appends its table to `--log`, and prints `NODES-OK n/n` or
  `NODES-FAIL k node(s): ...`.

Re-sampling exists because a single probe can catch an agent pass or a
heartbeat mid-write; it never turns a real failure into a pass, because the
node has to pass a complete fresh probe.

### 4.6 Rollback

`rollback.sh all` works from any state and ends non-zero unless the hub is
fully production (served MANIFEST, API, adapter, channel deployment,
`WEIGHTSD_BIN` and the hub's core `bin/sparkpipe_weightd`):

1. engines: skipped when every node already runs production unheld;
   otherwise `hold.sh any` (if not held), `hub/rollback_root.sh` (only if the
   release was published; it also restores a partial publish whose MANIFEST
   is still production), `weightd.sh rollback` if a new weightd was published or announced,
   wait until every node applied and verified the production MANIFEST,
   `unhold.sh old`, `converge.sh old`;
2. api: `hub/rollback_api.sh` restores the recorded backup channel (api,
   adapter and deployment together), or confirms the live channel is
   already production.

Parts: `rollback.sh engines|root|api`; `root` alone runs only while every node
is held and down. If step 2 ran and step 4 did not: `unhold.sh old` then
`converge.sh old`. Roll back when an engine is not ready 10 min after the
unhold or the smoke fails. Resume the rotation afterwards.

The hand commands this section replaced (copy into `~/release`, regenerate
the MANIFEST, drain by cwd-scoped TERM) are in git history; the laws in
section 9 still hold for any manual step.

## 5. Releasing core: agent and weightd

`~/release/core/` holds `bin/fleet_node_agent.sh`, `bin/sparkpipe_weightd`,
`MANIFEST` (the `bin/` files only) and `WEIGHTSD_BIN`. On 2026-09-28 the hub
served agent `a05e207588da2cef` (origin/main) and weightd `3da98597a88b60b5`,
which was the announced one. Write the core MANIFEST as `publish_core.sh:15-17`
does:

```sh
cd ~/release/core && find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST
```

### 5.1 Agent

An agent publish needs no announce and reaches every node within seconds,
because each `self_update` executes the new file (`:370-379`). On 09-28 a
stale agent in the hub core, one without `--mesh-rank-mask`, reached every
node this way and replaced the fixed agent. weightd refuses a mesh identity
unless all four mesh fields are given (`node/weightd.c:258-265`, exit 2), so the
cutover failed on all sixteen (handoff, section 3.3).

Publish only the file from the merged SHA, and check it first:

```sh
git show <SHA>:tools/fleet_node_agent.sh | grep -c -- '--mesh-rank-mask'    # 1
git show <SHA>:tools/fleet_node_agent.sh | ssh rtx5090 'set -e; cd ~/release/core
    cat > bin/fleet_node_agent.sh.new && chmod 755 bin/fleet_node_agent.sh.new
    mv -f bin/fleet_node_agent.sh.new bin/fleet_node_agent.sh
    find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST
    cut -c1-16 MANIFEST'
```

Then check the `agent` field in the heartbeats, and run
`journalctl --user -u fleet-agent | grep self-updating` on a node.

### 5.2 weightd: the bundle rollout (planned outage)

A weightd change restarts every engine cold, and each agent swaps weightd only
while no residentd of any root runs on its node (`ensure_weightd`,
"requires dependent engines to drain"). The bundle is three binaries built
from the release SHA in one tree whose host suite ran: `sparkpipe_weightd`,
`weightd_receipt`, `weightd_warm`. Set `OLD_WEIGHTD`, `NEW_WEIGHTD`,
`NEW_RECEIPT`, `NEW_WARM`, `WEIGHTD_BUNDLE_BUILD` and `BUILD_HOST` in the
release env; with `OLD_WEIGHTD = NEW_WEIGHTD` every weightd step refuses and
the root steps expect the running weightd unchanged.

**Verify-once receipts.** A weightd from #1336 on hashes a pack in full only
once. It then writes `<pack>.verified` (device, inode, size, mtime, ctime and
the digest; [WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md)), and every later cold path
trusts the receipt while the file's stat is unchanged, so a restart streams the
spine at disk speed instead of hashing 20.2 GiB. Each verification logs
`weightd pack-verify path=... mode=receipt|sha256|ck128`; the probe's
`verify` column is that mode for the node's rank pack.

Before the window (production keeps serving):

1. `weightd.sh stage`: checks the build tree is at `SHA` with the three
   shas, copies them to `~/weightd-bundle-stage/bin` on every node (KEEP line
   added), and checks `stage_wd`, `stage_rc`, `stage_warm` everywhere.
2. `weightd.sh receipts-adopt`: `weightd_receipt adopt <rank pack>` converts a
   legacy `/tmp/spark-weightd-spine` receipt without rehashing (`legacy=none`
   means the node verifies in full next). `HASH MISMATCH` stops everything:
   investigate that pack.
3. `weightd.sh receipts-verify`: only nodes without a valid receipt run
   `weightd_receipt verify` (nice 19, idle IO, one node at a time, about
   15-25 s per 20 GiB pack).
4. `weightd.sh receipts-check`: `receipt=rc0` on every node. `precheck.sh`
   and `hold.sh` require this whenever weightd changes.

In the window, between `hold.sh` and `publish.sh` (every other lane's
residentd stopped: `hold.sh` checks `others=none`):

5. `weightd.sh publish`: saves the old core binary as
   `~/release-staging/weightd-rollback/sparkpipe_weightd.<old>` (verified by
   sha), installs the new one into `~/release/core/bin` under
   `~/release/.core.publish.lock`, regenerates the core MANIFEST. Nodes fetch
   it; nothing restarts.
6. `weightd.sh announce`: `tools/weightsd_announce.sh` writes
   `WEIGHTSD_BIN`; each agent's `install_core` installs it and, with no engine
   running, replaces the running weightd.
7. `weightd.sh wait`: the new weightd installed and running (one process, no
   foreign weightd) on every node within 5 min.
8. `publish.sh` and the rest of section 4.4. After `converge.sh new`, run
   `weightd.sh verify`: every node on the new weightd with RECLAIM_PACK live
   (`reclaim_pack=freed=0 arenas=0 busy=0` for an all-zero sha) and the
   receipt mode count.

This order (hold, weightd, root, unhold) is DERIVED from the agent's rules and
the two MEASURED releases; it has not run end to end yet, so run its first use
supervised. a597ff0 used the older order: weightd publish and announce while
serving (agents then hold at `ensure_weightd`), root publish, then a
cwd-scoped drain of every engine; drain to converged took 54 s. Its first
drain attempt stopped because qwen still ran on four nodes: the gate that now
lives in `hold.sh`.

Rollback: `weightd.sh rollback` (while held) restores the saved binary and
`WEIGHTSD_BIN` and waits for the old weightd on every node; `rollback.sh all`
calls it when needed. The `.verified` receipts are ignored by an older weightd
and need no cleanup.

### 5.3 Freeing memory: `--reclaim-pack`, never `--reclaim`

`weightd_warm SOCKET --reclaim-pack PACK|SHA256 ...` (#1345) frees only the
cold arenas of the named packs and reports busy (attached or leased) arenas
instead of freeing them; exit 0 all freed, 3 busy, 2 bad argument, 1 daemon
error. The node-global `--reclaim` frees every cold arena of every lane: on
2026-09-28 it freed two cold qwen arenas while GLM Full was reclaimed. Use:

- `tools/fleet_window.sh reclaim [--list] [ROOT_GLOB...]` in the coordination
  directory: one `--reclaim-pack` per unique pack sha under
  `~/sparkdata/<ROOT_GLOB>` (sidecars and `SHA256SUMS` entries), with the
  staged bundle's `weightd_warm`; `--list` resolves packs and touches no
  weightd;
- the rotation, which reclaims a stopped model's own packs by sha and rejects
  any `--reclaim` in its config.

A weightd older than the bundle answers `reclaim_pack=unsupported`.

## 6. The GLM API on the hub (`g53-api`)

The serving API runs on the hub, not on spark0. The unit on 2026-09-28
(excerpt):

```ini
WorkingDirectory=/home/spec/g53-api-channel
Environment=SPARK_BATCH_INFLIGHT_BUDGET_NS=900000000000
Environment=SPARK_MODEL_API_MAX_PREFILL_ROWS=8
ExecStart=/home/spec/g53-api-channel/bin/sparkpipe_model_api --deployment /home/spec/g53-api-channel/model_resident.json --runtime-root /home/spec/g53-api-channel/runtime --port 8433
Restart=no
MemoryMax=2G
StandardOutput=append:/home/spec/g53-api-channel/api.log
```

`~/g53-api-channel/` holds:

- `bin/sparkpipe_model_api`;
- `runtime/lib/model_serving_adapter.so`;
- `runtime/tokenizer/tokenizer.json`;
- `model_resident.json`;
- `SOURCE_COMMIT` and `SHA256SUMS`.

The API loads the adapter from `--runtime-root` plus
`adapter.shared_object_path` (`lib/model_serving_adapter.so`). It loads the
tokenizer from the deployment's `tokenizer` block (`path`, `vocabulary_size`,
`sha256`), with the path relative to the runtime root
(`runtime/model_resident_deployment.c:19-24`,
`runtime/model_resident_deployment.c:390-413`).

`deployment/glm5_next_tp16/model_resident.json` carries this block
(`tools/glm5_next_gen_deployment.py` derives it from
`qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json`: path
`tokenizer/tokenizer.json`, the asset's sha256, vocabulary 154856). A GLM API
started from it refuses to start when `runtime/tokenizer/tokenizer.json` is
missing or differs. An API started from a deployment without the block starts,
logs `no tokenizer in deployment`, reports `"tokenizer":false` on `/health` and
answers every text prompt with HTTP 400 `tokenizer_unavailable`; COMPSEC-17
and 8-stream benches send text prompts. `tools/compsec17.py` and
`tools/glm5_next_api_bench.py conc` check `/health` first and exit 2 before
sending any prompt when the endpoint has no tokenizer.

The release root's API and adapter are aarch64 builds and cannot run on the
hub. Every root release therefore needs an x86 build of the API and the
adapter from the engines' `SOURCE_COMMIT`. The channel on 2026-09-28 was built
from `dd3526b`, the engines' source
([COMPSEC report](../qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md)).

Build in a clean directory on the hub. These commands were tested there on
2026-09-28:

```sh
SHA=<the engines' SOURCE_COMMIT>        # on the workstation
git archive --format=tar --prefix=g53-api-build-$SHA/ $SHA | ssh rtx5090 'tar -xf - -C ~'
ssh rtx5090
SHA=<the engines' SOURCE_COMMIT>        # again, on the hub
cd ~/g53-api-build-$SHA
make -j8 build/sparkpipe_model_api
make -C modules/glm5_next_resident_decode_stage adapter EXPERT_CODEC=fp8 \
    MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
```

`MODEL_REVISION` and `CONTRACT_SHA256` must be the values of the aarch64 build (`glm5_next_build_release.sh:7`,
`module_build_release.sh:35-36`). The adapter lands at
`build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so`
(`modules/resident_decode_stage_rules.mk:233`).

`tools/fleet_release/api_install.sh` (section 4.4, step 8) does this install
with the staged, declared channel deployment and checks the API's log lines;
use it. The commands below are the manual equivalent from 2026-09-28, before
the channel declared a chat template.

Install once the engines run the new root, from the same build directory.
The steps keep the previous channel, install with temp + `mv`, and refresh
`model_resident.json` from the release root plus the channel's tokenizer
block:

```sh
C=~/g53-api-channel
rm -rf $C.prev && cp -a $C $C.prev
cp build/sparkpipe_model_api $C/bin/sparkpipe_model_api.new && mv -f $C/bin/sparkpipe_model_api.new $C/bin/sparkpipe_model_api
cp build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so $C/runtime/lib/model_serving_adapter.so.new
mv -f $C/runtime/lib/model_serving_adapter.so.new $C/runtime/lib/model_serving_adapter.so
python3 - <<'EOF'
import hashlib, json, os
home = os.path.expanduser("~")
channel = f"{home}/g53-api-channel"
deployment = json.load(open(f"{home}/release/glm53flash.fp8.tp16/model_resident.json"))
tokenizer = json.load(open(f"{channel}/model_resident.json"))["tokenizer"]
with open(f"{channel}/runtime/{tokenizer['path']}", "rb") as f:
    assert hashlib.sha256(f.read()).hexdigest() == tokenizer["sha256"]
deployment["tokenizer"] = tokenizer
with open(f"{channel}/model_resident.json.new", "w") as f:
    json.dump(deployment, f, indent=2)
os.replace(f"{channel}/model_resident.json.new", f"{channel}/model_resident.json")
EOF
echo $SHA > $C/SOURCE_COMMIT
(cd $C && sha256sum bin/sparkpipe_model_api runtime/lib/model_serving_adapter.so \
    runtime/tokenizer/tokenizer.json model_resident.json SOURCE_COMMIT > SHA256SUMS)
systemctl --user restart g53-api && sleep 5 && systemctl --user is-active g53-api
```

Since #1359 the channel's `model_resident.json` also declares the chat
template (`chat_template` block from `model-families/glm5_next/chat_template.json`,
added by `hub/stage_channel.sh` through `CHANNEL_ADD`). The API renders chats
from it and logs `chat_template declared stop_markers=3`; roles it does not
declare (`tool`, `developer`) get HTTP 400 `role_unsupported`, and non-string
content gets 400 `invalid_messages`. The Paris chat still renders 16 prompt
tokens, as with the old hardcoded layout. A channel without the block
answers chats with `chat_template_missing`.

The unit has `Restart=no`, so a crashed API stays down until someone restarts
it. The rotation also stops and starts this unit around FULL and K3 hours. For a smoke check, send the agent's warmup request (`:550-553`):

```sh
curl -s --max-time 900 http://127.0.0.1:8433/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt_token_ids":[1,2,3,4,5,6,7,8],"max_tokens":4,"temperature":0}'
```

The release gate is COMPSEC-17: GLM chat template, `--thinking off`, 512
tokens, ds4_eval grading. On 2026-09-28 it scored 14/17, which is the tool's
default pass threshold (`glm5_next_compsec17.py:118`). Run it on the hub from
the build checkout:

```sh
python3 tools/glm5_next_compsec17.py --endpoint http://127.0.0.1:8433 --thinking off \
    --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
    --tokenizer ~/g53-api-channel/runtime/tokenizer/tokenizer.json \
    --out qualification/ds4_eval/runs/glm5-next-tp16-<date>-<sha7>-thinkoff
```

To roll back, stop `g53-api`, replace `~/g53-api-channel` with
`~/g53-api-channel.prev`, and start it again.

## 7. Bootstrap and `fleet_sync.sh`

`tools/fleet_sync.sh ROOTS_CSV [start|stop|status] [HUB]` (`fleet_sync.sh:8`,
`fleet_sync.sh:18-20`). `HUB` defaults to `$FLEET_HUB`, then to
`spec@100.123.97.61`.

- `status` is read-only. It prints the hub's heartbeat view.
- `stop` stops `fleet-agent` on all sixteen (`fleet_sync.sh:46-51`), which
  kills weightd and the engine on all sixteen (section 2.1).
- `start` restarts the whole fleet cold:
  1. It copies this checkout's `tools/fleet_node_agent.sh` over every node's
     `~/sparkdata/core/bin/` (`fleet_sync.sh:28-31`).
  2. It rewrites the base unit, then runs `daemon-reload`, `enable` and
     `restart` (`fleet_sync.sh:35-41`). Drop-ins survive.

  The hub does not replace the copied agent until the hub's core agent entry
  changes (section 2.2, step 1). Run `start` only from a checkout whose
  `tools/fleet_node_agent.sh` has the hub core's sha16.

On a serving fleet nothing needs starting: publishing to the hub is the
trigger. Use `start` and `stop` only to bootstrap or retire the whole fleet, in
a planned serving-down window.

To bootstrap one node:

1. Check the prerequisites:
   - `sparkpipe-hub-route.service` (section 1);
   - the node's SSH key is authorized for `spec@100.123.97.61` (the agent adds
     the host key itself, `:215-217`);
   - linger is enabled for the user (section 2.1);
   - the root's packs are in `~/sparkdata/<root>/packs/`, because the channel
     never syncs packs.
2. Fetch the hub's agent:
   `mkdir -p ~/sparkdata/core/bin && curl -sf http://100.123.97.61:8802/core/bin/fleet_node_agent.sh -o ~/sparkdata/core/bin/fleet_node_agent.sh && chmod 755 ~/sparkdata/core/bin/fleet_node_agent.sh`.
3. Write the base unit and the drop-in verbatim (section 2.1), then run
   `systemctl --user daemon-reload && systemctl --user enable --now fleet-agent`.
4. The agent then takes over:
   1. it syncs core and installs the announced weightd;
   2. it starts weightd and syncs the root;
   3. it waits for the mesh and starts the engine once the node has been up
      for 15 minutes.

## 8. Triage

- **Root shows `starting: <line>`.** The line is the engine's last log line.
  Failures print `ERRSITE file:line status=N count=M`
  (`include/sparkpipe/spark_error_site.h:22`).
- **Root shows `down`.** Run `journalctl --user -u fleet-agent` on the node. It
  shows the reason: backoff, the memory gate, `autospawn blocked`, or
  `waiting for weightd mesh`.
- **`weightd: unknown owner`.** A weightd from another path is running, and the
  node's loop stays frozen until it exits. Hand-started trees such as
  `~/sparkpipe/glm-serving-*` and `station-core-*` were retired on 09-28.
  Retire any others with a cwd- or exe-scoped TERM.
- **`requires dependent engines to drain`.** An announced weightd is installed
  and waiting; see section 5.2.
- **Shas differ across nodes.** The node missed a core or root sync. Run
  `curl -s http://100.123.97.61:8802/core/MANIFEST` on it to test tailscale
  reach. Also compare the drop-in sha (section 2.1).
- **`.wset` sidecars.** On 09-28, full-tape `.wset` sidecars next to the
  channel packs (12,096 keys, 96,768 bytes each) sent attach down the full-tape
  batch path. They were moved aside fleet-wide as `.wset.aside-20260928`
  (handoff, section 3.3).
- **NVRM** (09-13 to 09-15 record). Symptoms: `NV_ERR_NO_MEMORY` at context
  creation, or a node that refuses new CUDA contexts while old ones run.
  - `nvidia-smi` reports GB10 memory as "Not Supported"; list holders with
    `fuser -v /dev/nvidia*`.
  - A fresh boot that then fails in-kernel (`kgrctxA` OOM) had an exhausted
    driver pool. Killing the holders did not clear it; an operator-authorized
    reboot did. After a reboot the agent blocks engine starts for 15 minutes.
  - For CUDA startup failures that depend on cache state, see the
    cache-flush workaround in
    [SERVING_RELEASE_RECOVERY_20260924.md](SERVING_RELEASE_RECOVERY_20260924.md).
- **Cores.** They are written to `~/sparkdata/<root>/core.*` (`ulimit -c
  unlimited`, `:3`). Get a backtrace with `sudo -n gdb -p PID -batch -ex bt`.
  The engine runs with `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0` (`:195`), because
  the in-process GPU dump wedged engines past TERM on 09-18.
- **Logs.** The engine logs to `~/sparkdata/<root>/residentd.log`, rotated at
  each start with 20 kept (`:473-477`). weightd logs to `~/weightd.log`. The API
  logs to `~/g53-api-channel/api.log` on the hub.

## 9. Laws

Each of these was paid for in an incident.

1. Publish from a merged main SHA and check `SOURCE_COMMIT`; never publish from
   a lane tree.
2. The MANIFEST is written last and is the commit bit. A binary's sha is its
   version. A node has taken a release only when its `.applied_manifest` is
   that MANIFEST and its root passes `sha256sum -c` against it.
3. Install with temp + `mv`. Writing over a running binary fails with ETXTBSY.
4. Kill by `/proc/<pid>/cwd` or `exe`, never with `pkill -f`, which also matches
   your own ssh command. Drain with TERM before any `-9`.
5. Stage single verified files into the hub. Never rsync a staging tree over
   it.
6. A weightd change needs the `WEIGHTSD_BIN` announce and an engine drain
   (section 5.2).
7. Placed pack files are immutable: `chattr +i`, and clearing that needs
   `sudo chattr -i`.
   - The placement chain: verify gate, place, destination sha, `.sha256` and
     receipt, then `chattr +i`.
   - Before a deletion, verify the sha again and check `lsof` and `lsattr`.
8. A pack's content must match its declared source snapshot, checked by a
   router-gate fingerprint against the checkpoint. A pack whose declared
   revision misrepresents its content is a defect even when it decodes
   (glm53flash, 09-15).
9. Never `read_bytes()` a multi-GB file inside a capped cgroup; stream digests
   instead. A 30.5 GB verifier livelocked in `mem_cgroup_handle_over_high` for
   hours (09-13 to 09-15). The user slice, where GLM serving also runs, is
   capped at MemoryHigh 100G and MemoryMax 108G
   (`tools/devcycle/ds4_spark_brickproof.py:98-99`).
10. Commit tool scripts with the exec bit (`git update-index --chmod=+x`). The
    agent restores `bin/*` modes itself (`:301`).
11. `strings | grep -q` under `pipefail` exits 141 (SIGPIPE). Redirect to
    `/dev/null` instead.
12. Warm storage (Ceph, `/mnt/model-warm`) is a pack source only, never a
    serve path. Use one reader at a time fleet-wide by claiming
    `/Users/mac/sparkpipe-coord/CEPH_LEASE` on the controller Mac (holder,
    claim time, purpose; release right after the batch). Concurrent readers
    collapsed to kB/s (09-13 to 09-15: spark4 read at 64 KB/s); when one
    client crawls, dd-probe the candidates and move the read to an idle one
    (spark5 read the same data in 69 s against spark4's 84 min). Whether Ceph
    stays supported is an open question; see
    [INCIDENT_RECOVERY_PLAYBOOK.md](INCIDENT_RECOVERY_PLAYBOOK.md).
13. Required configuration fails loudly: no silent fallbacks and no `#ifndef`
    defaults. Grade every claim MEASURED, DERIVED or ASSUMED, with its artifact
    (file, node, date, build sha).
14. Pause the fleet rotation before any release step and resume it after the
    smoke; a rotation tick holds and releases the same `agent.hold` the
    release uses.
15. Free weightd memory by pack (`--reclaim-pack`), never with the
    node-global `--reclaim`, which frees other lanes' cold arenas.
