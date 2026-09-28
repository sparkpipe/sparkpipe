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

On main (checked at `09fdad6`, 2026-09-28) `tools/module_build_release.sh`
takes exactly five arguments and no branch: `FAMILY CODEC OUTPUT_NAME
MODEL_REVISION CONTRACT`. It refuses to run outside a GPU-owned queue job
(`SPARK_QUEUE_ID`), refuses a checkout that already has `build/obj` or
`build/modules`, and never resets a tree, parks an agent, drains a residentd or
writes `~/release`. The older six-argument form with a trailing `origin/main`
or SHA fails with a usage error. The glm53flash wrapper is
`tools/glm5_next_build_release.sh` (no arguments).

```sh
SHA=<MERGED_MAIN_SHA>; S7=${SHA:0:7}
python3 tools/spark_queue.py sync --id release-glm53-$S7 --nodes sparkf --ref $SHA
python3 tools/spark_queue.py add --id release-glm53-$S7-build --nodes sparkf \
    --resources gpu --memory-mib 32768 --ttl-min 15 --by lane-release \
    --cwd /home/sparkf/srcdata/sparkqueue/release-glm53-$S7/$SHA \
    --cmd 'bash tools/glm5_next_build_release.sh'
python3 tools/spark_queue.py status --id release-glm53-$S7-build
```

Run these from the workstation checkout (the controller's ledger is
`~/.sparkpipe/queue`). `sync` gives a fresh checkout, which the build
requires. On 2026-09-28 the job for `09fdad6` was admitted in under 10 s and
finished in about 50 s with `BUILD-PASS`. The job log is
`/tmp/sparkqueue-<attempt>.log` on sparkf. `build/glm53_release.tar.gz.sha256`
names the tarball relative to the checkout root, so run `sha256sum -c` from
there.

A fresh queue-synced checkout per release gives the module publish an empty
`build/module_library`, so `publish.log` must show `validation=executed` and
`glm5_next component validator: PASS`. `validation=reused` means the checkout
was not fresh; do not publish it. Always name the merged SHA explicitly: never
trust the build tree's `origin/main` ref (the refspec trap in §6).

Output: `build/glm53_release/` with `bin/`, `lib/hidden_transport.so`,
`lib/model_serving_adapter.so` (the script finds the adapter under its real
product name, `libglm5_next_serving_adapter_fp8.so`), `stages/stage_000/`,
`qualification/serving-receipts/` (build, publish, driver link and inspect,
`ldd -r` receipts), `SOURCE_COMMIT`, `SHA256SUMS`, and the tarball with its
`.sha256`. Run `sha256sum -c SHA256SUMS` inside the directory before you use it.

Assemble the release root from the root that is being served, then replace only
the build's files:

| Root path | From `build/glm53_release/` |
| --- | --- |
| `bin/sparkpipe_model_residentd`, `bin/sparkpipe_model_api`, `bin/sparkpipe_model_batch` | same path |
| `lib/hidden_transport.so`, `lib/model_serving_adapter.so` | same path |
| `stages/stage_000/model_driver.so`, `spark_model_driver_generated.c` | same path |
| `stages/stage_000/link_units/*.a` | same path; drop the old link units from the MANIFEST |

`config/`, `model_resident.json`, `bin/sparkpipe_registrar` and the root's
inert `bin/sparkpipe_weightd` copy stay as they are unless the release changes
them. Regenerate `MANIFEST` with the `publish_local.sh` recipe
(`find lib bin stages config model_resident.json -type f ! -name stage.json !
-name MANIFEST | sort | xargs sha256sum`), written through a temp file and
`mv`, LAST.

The `.wset` sidecars next to the channel-root packs must stay aside
(`.wset.aside-20260928`): with a sidecar present the attach takes the
full-tape batch path.

Queue admission: the lead cleared the stale persistent owners on 2026-09-28
at 10:35Z. The queue is retired for day-to-day work. It is used only for this
build, because `module_build_release.sh` requires `SPARK_QUEUE_ID`. Do not set
`SPARK_QUEUE_ID` by hand and do not edit the guard.

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

Stage `~/release/core` only when this release just published it. sparkf's
staging copy of `core` can lag behind the hub. If a stale core is staged,
`self_update` rolls the old agent out fleet-wide (the 09-22 agent without
`--mesh-rank-mask` did this on 2026-09-28). When core has not changed, stage
the root alone.

Before staging, keep a rollback copy of the served root, core, API channel and
`g53-api.service` outside the served tree (for example
`rtx5090:~/release-rollback-<date>-<sha7>/`) and check each copy against its
`MANIFEST` or `SHA256SUMS`.

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

### 2.6 The x86 API on the rtx5090 (`g53-api`)

The serving API for glm53flash is the rtx5090 user unit `g53-api` (`:8433`).
It runs from `~/g53-api-channel`; the fleet agent starts no API on a Spark
(#1261). The API is an x86 build, so the aarch64 root cannot provide it: build it from the
same merged SHA as the engines, and only after the fleet reports 16/16 `ready`
with the new residentd and driver shas.

```sh
SHA=<MERGED_MAIN_SHA>; S7=${SHA:0:7}
git archive --format=tar --prefix=api-build-$S7/ $SHA | ssh rtx5090 'tar -xf - -C ~'
ssh rtx5090
cd ~/api-build-$S7 && export PATH=/usr/local/cuda/bin:$PATH
CONTRACT_SHA=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
make -j4 build/sparkpipe_model_api > build-api.log 2>&1
make -j4 -C modules/glm5_next_resident_decode_stage adapter EXPERT_CODEC=fp8 \
    MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256=$CONTRACT_SHA > build-adapter.log 2>&1
ldd -r build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so
```

Install through temp+`mv` (ETXTBSY):

- `bin/sparkpipe_model_api` comes from `build/sparkpipe_model_api`.
- `runtime/lib/model_serving_adapter.so` comes from the adapter above.
- `model_resident.json` comes from the release root's `model_resident.json`
  with the channel's existing `tokenizer` block (`path`, `sha256`,
  `vocabulary_size`) carried over. That block is the only intended
  difference. Check it with a key-by-key diff.
- `SOURCE_COMMIT` gets the merged SHA.

A workaround line in `~/.config/systemd/user/g53-api.service` is removed once
the release carries its fix. `SPARK_MODEL_API_MAX_PREFILL_ROWS=8` goes with
#1255. Then run:

```sh
systemctl --user daemon-reload && systemctl --user restart g53-api
cd ~/g53-api-channel && sha256sum bin/sparkpipe_model_api \
    runtime/lib/model_serving_adapter.so runtime/tokenizer/tokenizer.json \
    model_resident.json SOURCE_COMMIT > SHA256SUMS
```

`g53-api` is the one hub service a release restarts. Law 6 still covers every
other process on the rtx5090. Roll back by restoring the four files and the
unit file from the rollback copy (§2.3), then running daemon-reload and a
restart.

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
