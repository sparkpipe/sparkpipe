# Fleet release & serving system

One hub, one reference tree, sixteen pull agents. Releasing a module means
copying it into the reference — nothing else. This document is the contract.

## Concepts

- **Hub**: the rtx5090 node (user `spec`). Two directories:
  - `~/release/` — the reference tree. The single source of truth.
  - `~/current/` — the fleet view: what every spark is running right now.
- **The reference IS the manifest.** No version files, no build lockstep.
  The sha256 of a binary is its version; the hub's copy is the current one.
- **Pull, never push.** Every spark runs one fleet agent (systemd user unit,
  `tools/fleet_node_agent.sh`) that rsyncs the parts it owns from the hub
  every 5 seconds. Builds never fan out to nodes; they write to the hub once.
- **Reload convention**: a file changed → TERM the daemon → start the new
  one. This applies ONLY where a restart is cheap (see weightd below).

## Layout

```
hub ~/release/
  weightd/sparkpipe_weightd              shared, model-generic, ONE per node
  <root>/                                one dir per model deployment, e.g.
    lib/                                 hidden_transport.so, model_serving_adapter.so
    bin/                                 sparkpipe_model_residentd, sparkpipe_model_api, ...
    stages/stage_000/model_driver.so     the compiled model driver
    config/stage_00..15.json             per-rank configs (stage.json symlink made per node)
    config/model_resident.json

node ~/sparkdata/
  weightd/sparkpipe_weightd              synced from release/weightd/
  <root>/{lib,bin,stages,config}         synced from release/<root>/
  <root>/packs/*.sp                      NOT synced — placed by the pack pipeline
```

Each spark derives its rank from its hostname (`spark<c>` → 12). All
identity/ports live in the configs as data (`session_ports` tables); the
nodes compute nothing.

## Updating a module

All updates are the same act: **build, then copy into the hub reference.**
The agents converge the fleet within one 5-second cycle.

| module | where it goes | reload |
|---|---|---|
| model driver (`model_driver.so`) | `release/<root>/stages/stage_000/` | automatic (residentd restart, ~15s fleet-wide) |
| serving adapter, hidden transport (`lib/*.so`) | `release/<root>/lib/` | automatic (same) |
| configs (`stage_*.json`, `model_resident.json`) | `release/<root>/config/` | automatic (a config change is a restart) |
| api / residentd binaries (`bin/*`) | `release/<root>/bin/` | automatic (cheap) |
| **weightd** | `release/weightd/` | **NEVER automatic. See below.** |
| packs (`*.sp`) | not in this system | per-node data via the pack pipeline; weightd lazy-loads on attach |

## weightd is special

weightd is the node's single model-generic weight daemon. It holds packs
resident for every model at once (lazy expert tiers, true-LRU, kv-reserve).
That residency is the entire point:

- **Starting it is free** (lazy — it loads nothing until a model attaches).
  Agents start it only if it is not running.
- **Restarting it is expensive**: it drops every resident pack for every
  model on the node. Agents therefore never TERM it. A new binary syncs
  into place and takes effect at the next deliberate restart (manual
  choice, or node reboot). Schedule that only when you accept a fleet-wide
  cold load.
- residentd must never load packs itself — the direct-load path is a
  fallback seam for a missing weightd, and agents guarantee weightd is up
  before any residentd starts, so the fallback never engages.

## Memory discipline on the sparks

The agent's restart protocol for a residentd, in order:

1. TERM the prior residentd (cwd-scoped to the root — never other models,
   never weightd).
2. Wait for the process to actually exit (up to 30s). If it does not, do
   not start a new one. One instance per root, ever.
3. Gate on memory: `MemAvailable >= pack_size + 8GB` before launching.
   A node that cannot afford the daemon stays down and says why.
4. Start exactly one; report versions.

## Fleet view

After any (re)start, the agent writes `~/current/<host>.json` on the hub:
sha16 of weightd and, per root, of residentd / driver / adapter /
transport. Reading one directory answers "what is running on every spark":

    tools/fleet_sync.sh <ref> <roots> status

## Commands

    tools/fleet_sync.sh <ref> <roots> start|stop|status
        install/control the per-node agents (systemd user units, survive
        reboots). <ref> e.g. rtx5090:release ; <roots> comma-separated.

    tools/fleet_serve.sh <root> stop|start|api|full
        manual relaunch: TERM in parallel, same-second launch, ready-or-
        error poll that fails in seconds (fail-fast prints the first
        error line from any node).

Build side (any capable host): produce artifacts in the runtime-root
layout (module `make publish` + `sparkpipe_model_compile` + adapter make),
then `rsync` the software parts into `hub:release/<root>/`.

## Deliberately absent

- No registrar phase, no TIME_WAIT sleeps, no ready windows — the daemons
  are load-order independent (background accepts, retrying connects,
  SO_REUSEADDR listeners).
- No derived ports — session port tables are data in the configs.
- No build-provenance gates at runtime — build-time receipts (pack
  validators, kernel tier tests) prove artifacts once; runtime checks
  structural identity only (revision string, geometry, codecs).
