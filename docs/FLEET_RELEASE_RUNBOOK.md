# Fleet release runbook

The one current guide for the production Spark fleet: hosts and routes, the
node agent, weightd ownership, releasing firmware and the API, releasing the
agent and weightd, the serving stack on the hub, bootstrap and triage. It was
checked on 2026-09-28 against origin/main `bbf5432` and read-only inspection
of the hub and all sixteen Sparks. Sections 4 and 5.2 were rewritten on
2026-10-14 for the current release kit, `scripts/release_build.sh` and
`scripts/release_rollout.sh`, and section 6 for the swap-controller serving
stack. The agent line references (`:NNN`) in sections 2, 7 and 8 date from
`bbf5432`; the agent has moved since, so look functions up by name. It
replaces `FLEET_RELEASE.md`, `FLEET_RUNBOOK.md` and the unmerged 09-28 fleet
handoff, which are now history in [`archive/`](archive/). Queue and lane work
is in [DEVCYCLE.md](DEVCYCLE.md) and
[PARALLEL_DRIVER_DEBUG.md](PARALLEL_DRIVER_DEBUG.md).

The rule: never deploy by killing and reloading engines by hand. A release
goes through `scripts/release_rollout.sh`, which backs up every lane root and
hub API directory before it changes anything, checks that each install left
one hash on all sixteen nodes, and can put every backup back. Core files (the
agent and weightd) reach the nodes through the hub's `~/release/core`, with the
MANIFEST written last. Bare line references (`:NNN`) are to
`tools/fleet_node_agent.sh`.

## 1. Hosts and network

| Role | Host | What lives there |
| --- | --- | --- |
| Hub | `rtx5090`: x86_64, one RTX 5090 32 GB, user `spec`, tailscale `100.123.97.61` | `~/release/<root>/` served on `:8802` by the user unit `fleet-release` (`tools/fleet_release_serve.py`, installed as `~/fleet_release_serve.py`); `~/current/<host>.json` heartbeats; `~/release/qpn/<host>/` mesh records; the serving stack: swap controller, chat layer, LiteLLM and one model's engine API at a time (section 6) |
| Build host | `BUILD_HOST` of `scripts/release_build.sh` (default `lenovo7`): aarch64, `sm_121a`, not a fleet rank | `~/build-rp/fw-<sha>-src` build tree and `~/build-rp/fw-<sha>.out` log (section 4.1). The build host is not the hub: no agent reads its `~/release` |
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
Its root filesystem is small (216 GB) and filled on 2026-10-14, so anything
that grows lives on the workspace volume under `/srv/workspace/sparkpipe`
(owner rule, 2026-10-14): build trees, source tarballs, release captures, API
directory backups (`HUB_WORK` in `scripts/rollover_lib.sh`) and the API
engine logs (`/srv/workspace/sparkpipe/logs/<model>-kv-api.log`). The home
keeps only the live service directories (`~/<model>-kv-api`,
`~/sparkpipe-chat`, `~/sparkpipe-swap`), the release channel (`~/release`,
`~/release-staging`, `~/current`) and tool checkouts.

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
| `G5_GRAPH_PATH=1` | No longer read. GLM Flash takes `graph_path` from its stage config (`tools/glm5_next_gen_deployment.py`). |
| `G5_PIN_EXPERTS=1` | No longer read. GLM Flash takes `pin_experts` from its stage config. Graphs require full pinning since `78c2c21`. |
| `SPARK_TP_WAIT_MODE=hardware` | No longer read. The device collective takes `wait_mode` from the stage config's `tp_collective` (default `spin`). |
| `G5_API_DISABLED=1` | No longer read. Since #1261 the agent has no `ensure_api` and never starts an API; the model APIs run on the hub (section 6). The line is inert and can leave the drop-in. |
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
   engine, library or stage change, or drain the root.
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
   - a node up less than 300 s does not start engines (`BOOT_SETTLE_SECONDS`);
   - a node in boot-loop safe mode does not start engines: 2 of the last 3
     boots lasted under 20 min and ended within 6 h (`check_boot_loop`). The
     heartbeat's `safe_mode` field gives the reason. Clear it for the current
     boot with `cat /proc/sys/kernel/random/boot_id > ~/.fleet_agent_safe_clear`;
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
not a liveness signal; check the nodes themselves.

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

## 4. Releasing firmware and the API

A release builds one source commit into the aarch64 node files and every
TP16 serving adapter, builds the x86 API on the hub, and installs both into
every model's lane root and hub API directory in one planned outage. The
kit is `scripts/release_build.sh` and `scripts/release_rollout.sh`, both run
from the workstation, with their helpers in `scripts/rollover_lib.sh`.
Everything the kit writes on the hub lives under `HUB_WORK` (default
`/srv/workspace/sparkpipe`, section 1); the home keeps only the live API
directories it installs into.

### 4.1 Build

`scripts/release_build.sh [REV]` builds `REV` (default `HEAD`) of `REPO`
(default `/Users/mac/wf/integ`), named by its 9-character sha:

1. `git archive` of the revision to `$HUB_WORK/src/rp-src-<sha>.tar` on the
   hub and to `~/build-rp/` on the build host.
2. On the hub, in the background: unpack into `$HUB_WORK/api-build-<sha>`
   and build `sparkpipe_model_api`, `sparkpipe_tokenize_prompt` and every
   TP16 serving adapter for x86 (log `$HUB_WORK/logs/api-<sha>.out`).
3. On `BUILD_HOST` (default `lenovo7`; every aarch64 build runs on a
   Lenovo, never on a fleet node, so it takes no fleet lock): unpack into `~/build-rp/fw-<sha>-src` and build residentd, weightd,
   `weightd_warm`, `weightd_receipt`, `weightdctl`, `sparkpipe_model_batch`,
   the RDMA transport and the same adapters (log `~/build-rp/fw-<sha>.out`).
4. It prints the first error lines of both logs and writes the firmware
   directory `/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-release-<sha>`:
   the node files, the adapters, `tools/weightd_spine_budget.py`,
   `SOURCE_COMMIT`, `SHA16` and `SHA256SUMS`. The hub build gets a `SHA16`
   too.

The last lines are `FW-DIR <dir>` and `APIB $HUB_WORK/api-build-<sha>/build`,
the two inputs of the rollout. Both source trees carry `SOURCE_COMMIT`; trust
it, not a branch name.

### 4.2 Rollout

`FW=<FW-DIR> APIB=<APIB> scripts/release_rollout.sh <step>` runs one step.
`OLD_WD` is the production weightd sha16; the script's default is a fixed
sha, so set it, and `precheck` refuses unless the hub's `WEIGHTSD_BIN` is
that value. The step state (the backup stamp, the
staged tarball) is kept in `scripts/state-<sha>` (`STATE` overrides). The
kit knows every served model by its swap id: lane root, adapter, hub API
directory, port and KV label come from the model's record in
`deployment/lanes.json` (`tools/lane_config.py release-row`, docs/LANES.md),
and the script refuses to run when its `MODELS` and the records differ.
Captures, comparisons and perf receipts
go to `$HUB_WORK/release-<sha>/<model>/`; the published receipts of a
release are committed under `qualification/releases/<sha>/`.

| Step | Fleet | What it checks, then does |
| --- | --- | --- |
| `precheck` | serving | the swap controller is idle with `qwen3.8-max` active; all sixteen nodes answer; `FW` has every node file and every model's adapter, and `APIB` has the API, the tokenizer tool and every adapter; the hub's `WEIGHTSD_BIN` is `OLD_WD`; at least 20 GiB free on `HUB_WORK` and 2 GiB in the hub home. It copies the capture and compare tools to `$HUB_WORK/release-<sha>/kit` |
| `old` | serving, swapping | activates each model through the controller (`POST /activate` on `:8440`, with the controller token, through `swap` in `scripts/rollover_lib.sh`) and captures its reference outputs (`tools/release_capture.py`), plus logprob goldens for GLM Full, K3, the Qwen models and DeepSeek V4.1 Flash; it ends on `qwen3.8-max` |
| `stage` | serving | copies `FW` to `~/rp-release-<sha>` on every node and checks `SHA256SUMS`: one result on all sixteen, or it fails |
| `backup` | serving | stamps the backup (`STATE/utc`) and copies every lane root to `root.prev-<utc>` (without `kvsnapshot` and `kvcache`), every hub API directory to `$HUB_WORK/backups/<api>.prev-<utc>`, `weightd_warm` and the installed weightd to `.prev-<utc>` on every node, the production weightd to `~/release-staging/weightd-rollback/sparkpipe_weightd.<OLD_WD>` and `model_swap.json` to `.prev-<utc>` on the hub |
| `down` | **down from here** | stops the active model's API and engines with the controller's own `api_stop` and `node_stop` commands, and fails unless no residentd runs anywhere |
| `weightd` | down | section 5.2; it changes nothing when `FW` carries the production weightd |
| `install` | down | per model: residentd, `weightd_warm`, the spine budget, the batch client where the lane has one, the transport and the adapter into `~/<lane>/root`, each install one hash variant on all sixteen; the API, the tokenizer tool and the adapter into the hub's `~/<model>-kv-api`, dropping its `runtime/prefix_index.spi`; the shared `~/sparkdata/swap/weightd_warm`. This kit version also moves the GLM Full lane to the `glm53full.fp8lin.tp16` packs |
| `qualify` | serving, swapping | per model (`ONLY=<ids>` limits it): activate, capture and compare against `old`; `tools/ui_smoke.py` through LiteLLM; COMPSEC-17 (`tools/compsec17_openai.py`) when the outputs are not exact 6/6 or `COMPSEC` is set; `api_serving_perf.py` cases `o256` and `ttft6000`; the count of error lines in the rank logs. It ends on `qwen3.8-max` |
| `rollback` | down, then swapping | refuses unless every backup of the stamp exists; runs `down`; restores every lane root, hub API directory and `weightd_warm`; restores the production weightd and `WEIGHTSD_BIN` (section 5.2); then activates, captures and compares each model against `old` |

Run `precheck`, `old`, `stage` and `backup` while production serves, then the
window: `down`, `weightd`, `install`, `qualify`. The other steps exit
non-zero and name what failed; stop there. `old` and `qualify` go on to the
next model and print `OLD-FAILED <id>` or `NEW-ACTIVATE-FAILED <id>`. Roll
back when a model does not activate or its qualification regresses. `rollback` leaves weightd and the models down
when a restore fails or the old weightd is not ready on all sixteen, for a
manual repair.

## 5. Releasing core: agent and weightd

`~/release/core/` holds `bin/fleet_node_agent.sh`, `bin/sparkpipe_weightd`,
`MANIFEST` (the `bin/` files only) and `WEIGHTSD_BIN`. On 2026-09-28 the hub
served agent `a05e207588da2cef` (origin/main) and weightd `3da98597a88b60b5`,
which was the announced one. Write the core MANIFEST like this:

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

### 5.2 weightd

A weightd change restarts every engine cold, and each agent swaps weightd only
while no residentd of any root runs on its node (`ensure_weightd`,
"requires dependent engines to drain"). The release kit therefore changes it
inside the window, as `scripts/release_rollout.sh weightd` between `down` and
`install` (section 4.2):

1. It refuses while any residentd runs.
2. It copies `FW/sparkpipe_weightd` to the hub's `~/release/core/bin` and,
   under `~/release/.core.publish.lock`, checks its sha16, moves it into
   place, regenerates the core MANIFEST and writes `WEIGHTSD_BIN`: publish
   and announce in one step. Each agent's `install_core` installs it and,
   with no engine running, replaces the running weightd.
3. It waits up to 10 minutes until all sixteen nodes run the new weightd with
   `/tmp/weightd-mesh/.ready`, and otherwise stops with `WEIGHTD-NOT-READY`.

`backup` saved the production binary as
`~/release-staging/weightd-rollback/sparkpipe_weightd.<OLD_WD>` (checked by
sha), and `rollback` puts it and `WEIGHTSD_BIN` back the same way and waits
for the old weightd on all sixteen. The kit publishes only
`sparkpipe_weightd`; it does not build or publish `sparkpipe_mesh_status`
([WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md)).

`tools/weightsd_announce.sh [CORE_DIR]` (default `~/release/core`) is the
announce alone, for a binary already published into `CORE_DIR/bin`: it writes
that binary's sha16 to `WEIGHTSD_BIN`, and does nothing when it is already
announced.

**Verify-once receipts.** A weightd from #1336 on hashes a pack in full only
once. It then writes `<pack>.verified` (device, inode, size, mtime, ctime and
the digest; [WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md)), and every later cold path
trusts the receipt while the file's stat is unchanged, so a restart streams the
spine at disk speed instead of hashing 20.2 GiB. Each verification logs
`weightd pack-verify path=... mode=receipt|sha256|ck128`. The `.verified`
receipts are ignored by an older weightd and need no cleanup on a rollback.

### 5.3 Freeing memory: `--reclaim-pack`, never `--reclaim`

`weightd_warm SOCKET --reclaim-pack PACK|SHA256 ...` (#1345) frees only the
cold arenas of the named packs and reports busy (attached or leased) arenas
instead of freeing them; exit 0 all freed, 3 busy, 2 bad argument, 1 daemon
error. The node-global `--reclaim` frees every cold arena of every lane: on
2026-09-28 it freed two cold qwen arenas while GLM Full was reclaimed. Use:

- `tools/fleet_window.sh reclaim [--list] [ROOT_GLOB...]` in the coordination
  directory: one `--reclaim-pack` per unique pack sha under
  `~/sparkdata/<ROOT_GLOB>` (sidecars and `SHA256SUMS` entries), with
  `FW_WARM`; `--list` resolves packs and touches no weightd. Its default,
  `~/weightd-bundle-stage/bin/weightd_warm`, was staged only by the retired
  release kit; `~/sparkdata/swap/weightd_warm` is the current build, which
  `release_rollout.sh install` keeps up to date.

A weightd older than #1345 does not support `--reclaim-pack`.

## 6. The serving stack on the hub

Since 2026-10-08 one model holds the fleet at a time. The hub runs:

- the swap controller (`tools/serving/model_swap.py`, config
  `deployment/hub/model_swap.json`, unit `sparkpipe-swap` from
  `~/sparkpipe-swap`, `127.0.0.1:8440`, every call with the token in
  `/srv/workspace/sparkpipe/swap-controller/token`). It stops the other
  models, starts the requested model's engines on every node, then its
  engine API;
- one engine API at a time: `sparkpipe_model_api` from the model's
  `~/<model>-kv-api` (`bin/`, `runtime/lib/<adapter>.so`,
  `model_resident.json`), started by the controller as the transient unit
  `kv-<model>-api` on the model's port (`api_start` in `model_swap.json`,
  ports 8459-8476) and logging to
  `/srv/workspace/sparkpipe/logs/<model>-kv-api.log`;
- the OpenAI chat layer (`tools/serving/chat_frontend.py`, config
  `deployment/hub/chat_frontend.json`) on `127.0.0.1:8433`, unit
  `sparkpipe-chat` from `~/sparkpipe-chat`;
- LiteLLM on `:4000`, unit `sparkpipe-litellm`, configured from
  `config/litellm-config.yaml`.

[LITELLM_FRONTEND.md](LITELLM_FRONTEND.md) describes the APIs and the swap
order. `tools/hub/install_units.sh` installs the versioned units in
`tools/hub/systemd/` (the chat layer, LiteLLM and the DMA-guard listener)
into `~/.config/systemd/user` and reloads; `--enable` also enables them and
refuses if linger is off. The `sparkpipe-swap` unit is not versioned there
yet. The engine APIs have no persistent unit: the
controller owns their lifecycle, and a release installs into their
directories only while the fleet is down (section 4.2).

The engine API loads the tokenizer from the deployment's `tokenizer` block
(`path` relative to the runtime root, `vocabulary_size`, `sha256`;
`node/model_api.c`, `runtime/model_resident_deployment.c`) and refuses to
start when that file is missing or differs. A deployment without the block
starts, logs `no tokenizer in deployment`, reports `"tokenizer":false` on
`/health` and answers every text prompt with HTTP 400 `tokenizer_unavailable`.
The hub is x86, so the API and the adapters it loads come from the hub build
of section 4.1, never from a Spark build.

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
      for 5 minutes, unless the boot-loop breaker holds it in safe mode.

## 8. Triage

- **Root shows `starting: <line>`.** The line is the engine's last log line.
  Failures print `ERRSITE file:line status=N count=M`
  (`include/sparkpipe/spark_error_site.h:22`).
- **Root shows `down`.** Run `journalctl --user -u fleet-agent` on the node. It
  shows the reason: backoff, the memory gate, `autospawn waits`, `safe-mode`,
  or `waiting for weightd mesh`.
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
    reboot did. After a reboot the agent holds engine starts for 5 minutes.
  - For CUDA startup failures that depend on cache state, see the
    cache-flush workaround in
    [SERVING_RELEASE_RECOVERY_20260924.md](SERVING_RELEASE_RECOVERY_20260924.md).
- **Cores.** They are written to `~/sparkdata/<root>/core.*` (`ulimit -c
  unlimited`, `:3`). Get a backtrace with `sudo -n gdb -p PID -batch -ex bt`.
  The engine runs with `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0` (`:195`), because
  the in-process GPU dump wedged engines past TERM on 09-18.
- **Logs.** The engine logs to `~/sparkdata/<root>/residentd.log`, rotated at
  each start with 20 kept (`:473-477`). weightd logs to `~/weightd.log`. Each
  model's API logs to `/srv/workspace/sparkpipe/logs/<model>-kv-api.log` on the
  hub.

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
   (`USER_SLICE` in `tools/devcycle/ds4_spark_brickproof.py`).
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
    (spark5 read the same data in 69 s against spark4's 84 min). Ceph runs under
    cephadm on every node; the node policy is in
    [NODE_BASELINE.md](NODE_BASELINE.md).
13. Required configuration fails loudly: no silent fallbacks and no `#ifndef`
    defaults. Grade every claim MEASURED, DERIVED or ASSUMED, with its artifact
    (file, node, date, build sha).
14. Free weightd memory by pack (`--reclaim-pack`), never with the
    node-global `--reclaim`, which frees other lanes' cold arenas.
