# Fleet release & update RUNBOOK — the automated, order-independent path

Operational companion to `docs/FLEET_RELEASE.md` (the design). This file is the
do-this-then-that runbook: exact commands, what each step triggers, how to
verify, and how to retire a manual deployment. Written 2026-09-28 against the
code in `tools/` as ground truth, after the fleet drifted to manual
kill-and-reload deploys (`~/sparkpipe/glm-serving-<sha>` trees started by hand)
while the automated channel sat installed but inactive.

The one rule this runbook exists to enforce: **never deploy by killing and
serially reloading by hand.** Publish, flip the version, and let every node's
agent converge on its own — the daemons are load-order independent by design
(background accepts, retrying connects), so no choreography is needed and none
must be improvised.

## 0. Topology (who is what)

| Role | Host | What lives there |
| --- | --- | --- |
| Hub | `rtx5090` (user `spec`, tailscale `100.123.97.61`, fleet `10.10.250.2`) | `~/release/<root>/` — the served release base; `:8802` HTTP (`fleet-release.service`, user unit); `~/current/<host>.json` — per-node heartbeats |
| Build host | `sparkf` (user `sparkf`) | `~/g5n-rd-build` — aarch64/sm_121a build tree (do NOT touch `~/sparkpipe-build`, it is glmdev's); `~/release/` — staging copy of the roots; `~/sparkdata/out` — driver compile output |
| Fleet | `spark0..spark9, sparka..sparkf` (16 nodes, rank = index, hex in pack names) | `~/sparkdata/core/` — shipped core (weightd + agent); `~/sparkdata/weightd/` — installed weightd; `~/sparkdata/<root>/` — deployed runtime roots; `~/current/<host>.json` — local heartbeat |

Never kill anything on the rtx5090: it also hosts unrelated workloads (the
chess/nnue training among them) that must not be disturbed. Fleet operations
only ever write `~/release` and `~/current` files there.

## 1. The moving parts (what makes it "automated")

- **`tools/fleet_node_agent.sh ROOTS [HUB]`** — one systemd *user* unit
  (`fleet-agent`) per node, 1 s loop: `sync_core` (pull `core/MANIFEST` diff,
  sha-verify every file) → `install_core` (swap `~/sparkdata/weightd/`
  binary when the hub's `core/WEIGHTSD_BIN` announce differs from installed)
  → `self_update` (`exec` the freshly shipped agent) → `node_doctor` (flap a
  dead RoCE port) → `janitor` (reap duplicate/stale daemons, exe-scoped) →
  `ensure_weightd` (recycle on sha drift; start when absent) → `sync_root`
  per root (MANIFEST diff → fetch only changed files → restart scope:
  residentd/driver/lib → restart root; api-only → drain api; stage configs →
  relink) → `sync_rendezvous` (mesh/QPN records via hub `qpn/<host>/`) →
  `ensure_root` (boot a down root: 15-min post-reboot autospawn guard,
  `MemAvailable >= packs + 8 GB` gate, backoff) → `report` (heartbeat, local + hub).
  The agent never starts an API: the API runs only on the rtx5090 hub
  (`g53-api`).
- **The MANIFEST is the version.** Each root dir has `MANIFEST` (sha256 per
  file). Agents fetch the diff vs `.applied_manifest` and verify every sha
  before applying. A binary's sha is its version; there is no "deployed
  revision" anywhere else.
- **Restarts are scoped and exe/cwd-matched.** Kills go by `/proc/<pid>/exe`
  or cwd == the root being managed, never `pkill -f`. Installs are
  temp+`mv` (ETXTBSY). The agent only ever manages binaries under its own
  `~/sparkdata/` tree — a foreign (hand-started) daemon is invisible to it,
  which is exactly why a manual deployment must be *retired*, not "updated"
  (see §4).
- **Order independence.** No node waits for another. Engines accept and
  reconnect in the background, so a publish restarts each node's root
  independently and the fleet converges without choreography.

## 2. Publishing a release (the only sanctioned deploy)

All commands run on **sparkf** unless stated. The build tree is
`~/g5n-rd-build` (export `SPARKPIPE_BUILD_TREE=$HOME/g5n-rd-build` — the
scripts' default `~/sparkpipe-build` is glmdev's on sparkf).

### 2.1 Module/driver root (residentd + api + driver + adapter)

```sh
ssh sparkf
cd ~/g5n-rd-build
export SPARKPIPE_BUILD_TREE=$HOME/g5n-rd-build
git fetch -q origin main
# publish from a merged main SHA (never a lane):
git reset -q --hard <MAIN_SHA>          # e.g. origin/main after the PR merges

tools/module_build_release.sh <family> <codec> <root> <revision> <contract-json> origin/main
#   concrete glm53flash example:
# tools/module_build_release.sh glm5_next fp8 glm53flash.fp8.tp16 \
#     84c6a6aa9497188e15a635ba793b0f95a79b1033 \
#     model_contracts/glm53_flash_authoritative.json origin/main
```

What it does, in order: host-builds residentd/api/compile/transport; builds
the serving adapter with nvcc `sm_121a`; parks sparkf's own agent and drains
sparkf's residentds (the GPU publish needs the GPU); runs the module publish
with GPU receipts; compiles `model_driver.so` into `~/sparkdata/out`;
`publish_local.sh` installs everything into `~/release/<root>/` (flocked,
staging + mv), regenerates that root's `MANIFEST` (MANIFEST is written
LAST — it is the commit bit), restarts sparkf's agent, and re-publishes the
core agent via `publish_core.sh agent`.

The runtime-root guard inside `publish_local.sh` rejects a
`model_resident.json` whose `runtime_root` does not match `<root>`.

### 2.2 Core (weightd, agent) — "if needed"

```sh
# weightd binary (only when main changed node/weightd*, mesh, or budgets):
SPARKPIPE_BUILD_TREE=$HOME/g5n-rd-build tools/publish_core.sh weightd
# agent script:
SPARKPIPE_BUILD_TREE=$HOME/g5n-rd-build tools/publish_core.sh agent
```

### 2.3 Stage to the hub (rtx5090)

```sh
rsync -a --delete ~/release/core ~/release/<root> spec@10.10.250.2:release/
```

### 2.4 The weightd announce — the restart authorization

`install_core` on every node keys on `$HUB_HTTP/core/WEIGHTSD_BIN`: a file
containing the announced weightd sha16. **No announce, no weightd update** —
this is deliberate (the announce IS the authorization; it is not written
automatically by the publish). To authorize:

```sh
ssh rtx5090 'cd ~/release/core && sha256sum bin/sparkpipe_weightd | cut -c1-16 > WEIGHTSD_BIN'
```

Then every node's agent installs the new weightd (temp+mv) on its next loop
and `ensure_weightd` recycles the running one (TERM, then start the new sha).
Announce only after you are sure the build is good: it is fleet-wide.

### 2.5 Verify the publish (before the fleet sees it)

```sh
curl -s http://100.123.97.61:8802/<root>/MANIFEST | head    # served sha list
curl -s http://100.123.97.61:8802/core/WEIGHTSD_BIN         # announced sha
```

## 3. Activating / triggering the fleet update

The agents converge by themselves; "triggering" means making sure the units
are installed and active on all 16 nodes.

```sh
# from a repo checkout with the SSH aliases:
tools/fleet_sync.sh <reference-base> <ROOTS_CSV> start     # install+enable+start
tools/fleet_sync.sh - <ROOTS_CSV> status                    # per-node state
tools/fleet_sync.sh - <ROOTS_CSV> stop                      # stop agents (not daemons)
```

ROOTS_CSV example: `glm53flash.fp8.tp16` (comma-separated for multi-root
nodes). `HUB` defaults to `spec@100.123.97.61`.

> Note: `fleet_sync.sh` in tree generations past passed a reference-base arg
> to the agent that the current agent does not take (it takes `ROOTS [HUB]`).
> The unit must run the agent from `~/sparkdata/core/bin/fleet_node_agent.sh`
> — that is the path `sync_core` updates and `self_update` execs. Running it
> from a copy in `$HOME` defeats self-update.

Verify convergence (one read):

```sh
ssh rtx5090 'grep -h "\"state\":\"ready" ~/current/*.json | wc -l'   # expect 16
ssh rtx5090 'cat ~/current/spark0.json'   # state, weightd/agent/residentd/driver shas
```

weightd/agent shas must be uniform fleet-wide; a mismatch = that node cannot
reach `:8802` (check tailscale, `curl http://100.123.97.61:8802/core/MANIFEST`
from the node). Agent log: `journalctl --user -u fleet-agent` on the node.
A root stuck `starting: <line>` — the line is the residentd's last log line;
`ERRSITE file:line status=N` names the failure site.

## 4. Retiring a manual ("caveman") deployment

A hand-started daemon (e.g. `~/sparkpipe/glm-serving-<sha>/bin/…` or
`~/sparkpipe/station-core-<sha>/build/sparkpipe_weightd`) is invisible to the
agents by design: the agent never touches binaries outside its own
`~/sparkdata/` paths. Starting the agents therefore does NOT update the
manual stack — it starts a second, parallel stack that then fights over
ports. The manual stack must be retired first, once, explicitly:

```sh
# drain every node's manually-started residentd/api and weightd, cwd/exe-scoped:
for h in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 \
         spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  ssh -o BatchMode=yes $h '
    for p in $(pgrep -f "bin/sparkpipe_model_[r]esidentd"); do
      case $(readlink /proc/$p/cwd) in /home/*/sparkpipe/glm-serving-*) kill -TERM $p;; esac
    done
    for p in $(pgrep -f "bin/sparkpipe_model_[a]pi"); do
      case $(readlink /proc/$p/cwd) in /home/*/sparkpipe/glm-serving-*) kill -TERM $p;; esac
    done
    for p in $(pgrep -f "sparkpipe_[w]eightd"); do
      case $(readlink /proc/$p/exe) in /home/*/sparkpipe/station-core-*) kill -TERM $p;; esac
    done
    exit 0'
done
```

Drain, never `kill -9` first (collective teardown can take ~60 s). Then start
the agents (`fleet_sync.sh … start`) and let the channel bring weightd, the
roots, and the api up on its own. Afterwards keep or delete the orphaned
`glm-serving-<sha>` trees at leisure — nothing references them.

## 5. Diagnostics

- **View:** `sparkf:8801/summary` is the old aggregate; the live truth is the
  heartbeats: `rtx5090:~/current/<host>.json` (state per root =
  `ready | starting: <last log line> | down`, plus weightd/agent/residentd/
  driver shas, mem, load, heartbeat age).
- **Backoff:** repeated restarts double a per-class backoff to 60 s; a node
  flapping is visible as alternating `down`/`starting` with growing gaps.
- **Autospawn guard:** a node up < 15 min does not start engines
  (reboot-loop protection); the agent logs `autospawn blocked`.
- **Memory gate:** engine start waits for `MemAvailable >= packs-du + 8 GB`;
  `du -sBG` rounds small files up, so the threshold reads high — not a fault.
- **Cores:** `~/sparkdata/<root>/core.*` (agent sets `ulimit -c unlimited`);
  `sudo -n gdb -p PID -batch -ex bt` works (binaries have symbols; the driver
  .so does not — use addr2line against sparkf's `~/sparkdata/out` artifact).
- **Coredump wedge:** serving sets `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0`; the
  in-process GPU dump is what wedged engines past TERM historically.

## 6. Laws recap (each paid for; see FLEET_RUNBOOK.md for the incidents)

1. Publish from a **merged main SHA** on sparkf; never from a lane tree.
2. MANIFEST is written last and is the commit bit; a binary's sha is its version.
3. Kills are cwd/exe-scoped; drain (TERM, wait) before -9; the agent only
   manages its own `~/sparkdata/` paths.
4. weightd updates require the explicit `core/WEIGHTSD_BIN` announce.
5. Installs are temp+`mv` (ETXTBSY).
6. The rtx5090 hosts unrelated training (chess/nnue): **never kill or restart
   anything on the hub host**; publishing only writes `~/release`.
7. glm53flash root publishes remain coredev's channel per the isolation
   table — coordinate before publishing that root; the *mechanism* is
   identical for every root.
