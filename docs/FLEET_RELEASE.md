# Fleet release & update system

One channel delivers every artifact to the 16 sparks: binaries, configs,
the weights daemon, and the agent itself. No ssh fanout, no rsync, no
manual scp of binaries. Everything converges by itself after a publish;
a node that reboots or drops off the network catches up unattended.

## Topology

- **Hub** = sparkf. `~/release/<root>/` holds each release root;
  `fleet-release.service` serves `~/release` over HTTP on :8802
  (`http://10.10.100.25:8802/<root>/...`). The hub is also the build
  host, so publishing is a local `install`, not a network transfer.
- **Agent** = one systemd user unit (`fleet-agent`) per node, running
  `~/sparkdata/core/bin/fleet_node_agent.sh <roots> [hub]`. It is the
  only thing that starts daemons on a node.
- **Roots** = release directories. Driver roots (e.g.
  `glm53flash.fp8.tp16`) carry a serving deployment; the `core` root
  carries the model-generic node services: `bin/sparkpipe_weightd` and
  `bin/fleet_node_agent.sh` itself.
- **The MANIFEST is the version.** Each root has a `MANIFEST` (sha256
  per file); a binary's sha is its version. Agents pull manifests,
  fetch changed files over HTTP, verify every sha before applying.
- **Telemetry** = each agent writes `~/current/<host>.json` (locally
  and to the hub); `sparkf:8801` serves the aggregate
  (`/summary`, `/host/<name>`). Per root the `state` field is
  `ready | starting: <last log line> | down`, plus weightd / agent /
  residentd / driver shas. **One read answers "what runs where."**

## Layout

```
hub ~/release/
  core/bin/{sparkpipe_weightd, fleet_node_agent.sh}   shared, one per node
  <root>/                          one dir per model deployment
    lib/                           hidden_transport.so, model_serving_adapter.so
    bin/                           sparkpipe_model_residentd, sparkpipe_model_api
    stages/stage_000/model_driver.so
    config/stage_00..15.json       per-rank configs (stage.json symlink per node)
    config/model_resident.json
    MANIFEST, UPDATE.<n>           current manifest + archived ledgers

node ~/sparkdata/
  core/bin/fleet_node_agent.sh     the running agent (self-updated)
  weightd/sparkpipe_weightd        installed from core/
  <root>/{lib,bin,stages,config}   synced from release/<root>/
  <root>/packs/*.sp                NOT synced — placed by the pack pipeline
```

Each spark derives its rank from its hostname (`spark<c>` → 12; pack
names use **hex** ranks — `ranka..rankf`). All identity/ports live in
the configs as data (`session_ports` tables); nodes compute nothing.

## Publish (a dev's whole interaction)

```
tools/module_build_release.sh <family> <codec> <root> <revision> <contract> [branch|sha]
```

Run on sparkf. It fetches the ref, host-builds (residentd, api, weightd,
model_compile, transport DSO), parks sparkf's local daemon, runs the GPU
module publish with receipts, compiles the driver, then:

1. `publish_local.sh` installs driver/DSO/adapter/residentd/api into
   `~/release/<root>/`, regenerates the `MANIFEST`, touches `UPDATE`
   (the restart trigger).
2. `publish_core.sh` installs weightd + the agent into
   `~/release/core/` with its own `MANIFEST`.

A publish guard rejects a deployment whose `runtime_root` does not match
the release root (a bf16-rooted JSON inside the fp8 release once cost a
full day). Generator rule: `ROOT_NAME` is the single source — runtime
root, pack template, kv dir all derive from it.

## Agent convergence loop (every 5s)

1. `sync_core` — fetch `core/MANIFEST`; if changed, curl each file,
   sha256-verify, apply, `chmod 755 bin/*` (curl drops exec bits).
2. `install_core` — if the shipped weightd sha differs from the
   installed one: kill daemons **by /proc exe link** (never `pkill -f` —
   it matches your own ssh), install, start exactly one. This deliberate
   update restart is the only sanctioned weightd restart; it costs the
   node a cold reload of resident packs (lazy tiers fault back in on
   next attach, ~1-2 min/node).
3. `self_update` — if the shipped agent differs from the sha captured
   **at process start** (`START_SHA`), `exec bash` the new file. The
   baseline must be the startup sha: sync_core applies the new file onto
   the very path the process runs from, so comparing on-disk against
   on-disk never fires. An unknown/empty baseline skips (never spins).
4. `sync_root` per driver root — same fetch+verify+apply, then the
   two-phase UPDATE ledger on the hub: every node appends `down:<host>`
   after its daemon exits → the all-16 gate passes → every node starts
   and appends `up:<host>` → ledger archived `UPDATE.<n>`. The gate
   exists because a fresh sender wiring against a draining peer's
   listener parks QPs forever; the fleet stops together and starts
   together.
5. `ensure_root` — boots a down root. Restart discipline: TERM
   (cwd-scoped), wait for real exit, gate `MemAvailable >= pack + 8GB`,
   start exactly one, inside a systemd user scope with `MemoryMax`
   (default 32G — a weightd-mapped arena legitimately faults in up to
   one full pack; the cap exists to kill a second full load, not the
   first; `FLEET_RESIDENTD_MEMORY_MAX` overrides per node, `0`
   disables). The autospawn guard blocks only if **node uptime
   < 15 min** (reboot-loop protection); it never blocks on failed
   attempts — transient failures retry next loop so the fleet
   converges.
6. `ensure_api` (rank 0) — starts the model API once the fleet view
   reports 16 ready.

## Laws (each one paid for)

- weightd is never crash-restarted, never coupled to residentd; it
  restarts only via `install_core` on a shipped binary change.
- One daemon instance per node and per root; kills go by
  `/proc/<pid>/exe` links.
- The 15-minute autospawn guard reads `/proc/uptime` — node reboots,
  not residentd attempts.
- Every error-return site is macro-stamped: failures print
  `ERRSITE file:line status=N`. Grep the log, read the line.
- Lazy attach is the only load path. Direct pack loads by 14 parallel
  drivers kill sparks; the module fails UNSUPPORTED without weightd.
- Pack files are immutable on disk (`chattr +i` at deploy; clearing
  needs `sudo chattr -i` — plain users cannot).
- Sidecar debris (zero-byte `.sha256` from partial deploys) breaks the
  digest scan; the scan skips unreadable sidecars, fleet hygiene
  removes strays.
- Write-tool edits drop executable bits — tool-script commits must
  carry `git update-index --chmod=+x` or the publish breaks at the last
  step.
- `strings | grep -q` under pipefail SIGPIPEs (141) — redirect to
  /dev/null instead.

## Bootstrapping a new node (once)

The unit points at `~/sparkdata/core/bin/fleet_node_agent.sh`; ship the
agent + unit once (tar + scp + sha-verify + extract + `systemctl --user
daemon-reload && systemctl --user restart fleet-agent`). After that the
node updates itself forever through the channel. A reboot needs nothing:
systemd starts the agent, the agent ensures weightd, syncs roots,
converges. A node that loses the network is invisible in the view but
keeps retrying locally and rejoins without manual catch-up.

## Deliberately absent

- No registrar phase, no TIME_WAIT sleeps, no ready windows — daemons
  are load-order independent (background accepts, retrying connects).
- No derived ports — session port tables are data in the configs.
- No build-provenance gates at runtime — build-time receipts prove
  artifacts once; runtime checks structural identity only.

## Reading a stuck fleet

1. `curl sparkf:8801/summary` → which hosts, which states.
2. `starting: <line>` → the line is the residentd's last log line;
   `ERRSITE` names the exact source site.
3. `down` → `journalctl --user -u fleet-agent` shows the retry reason.
4. weightd/agent shas must be uniform fleet-wide; a mismatch means the
   node missed a core sync (check :8802 reachability).
