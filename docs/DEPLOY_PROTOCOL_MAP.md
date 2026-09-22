# Fleet deploy protocol — as-deployed map (2026-09-14)

This document describes the historical global fleet publisher. For concurrent
development use [the current development cycle](DEVCYCLE.md).
`module_build_release.sh` now builds an isolated artifact and rejects the legacy
branch argument; it does not restart or publish to this global fleet.

Evidence collected read-only on 2026-09-14 17:10-17:45 JST via ssh to sparkf
and spark1 (no writes, no kills, no restarts, no daemon contact). `agent` =
`fleet_node_agent.sh`; line numbers refer to the deployed copy
sha256-16 `71ec91d332f2f076` unless stated. Repo copies differ (see 1.4).

## 0. Protocol in five lines

1. Hub sparkf serves release roots over HTTP :8802 (`fleet-release.service`);
   the version of a root is its `MANIFEST` (sha256 per file); no UPDATED
   marker exists in the deployed path — the trigger is a `MANIFEST` byte
   diff (`apply_manifest`, agent lines 236-273).
2. One systemd user agent per node (`fleet-agent.service`, `Restart=always`,
   `RestartSec=3`) polls every ~1-2s: sync core, sync roots (fetch +
   sha256-verify + apply), kill -9 and restart residentd/api, ensure
   weightd, sync rendezvous, ensure api on rank 0 (agent lines 370-382).
3. Weightd is a shared fleet resource: a mesh of 16 weightds (RDMA, mesh
   rank = hex hostname suffix); residentd start gates on
   `/tmp/weightd-mesh/.ready` (agent lines 117-121, 327-337).
4. In parallel, an out-of-band deploy path (`tools_local/p`, `b`,
   `fleet_serve.sh sync` on sparkf) tar-streams the hub release root into
   every node's `~/sparkdata/<root>` and runs an unconditional
   fleet-wide `kill -9` of residentd + weightd + api on every deploy.
5. Heartbeats: each agent writes `~/current/<host>.json` locally and scps
   it to `sparkf:~/current/` only when state/pids change;
   `fleet_view_serve.py` on :8801 aggregates.

## 1. The node side

### 1.1 Agent unit
- `~/.config/systemd/user/fleet-agent.service` (read on spark1):
  `ExecStart=%h/sparkdata/core/bin/fleet_node_agent.sh glm53flash.fp8.tp16 sparkf`,
  `Restart=always`, `RestartSec=3`; drop-in `restart.conf` repeats the
  Restart settings. One root per node is configured today, but ROOTS is a
  comma-separated list (agent line 3) and the loops (lines 375-378) handle
  many.
- Rank identity: `RANK=$((16#${HOST#spark}))` (agent line 6) — sparkf is
  rank 15 (0xf). Pack/config names use hex ranks.

### 1.2 Convergence loop (agent lines 370-382, `sleep 1`)
1. `sync_core` (284-288): apply `core/MANIFEST` into `~/sparkdata/core/`.
2. `install_core` (290-314): if `core/bin/sparkpipe_weightd` sha differs
   from installed `~/sparkdata/weightd/sparkpipe_weightd`: log "deliberate
   restart", kill -9 weightd by `/proc/<pid>/exe`, kill -9 ALL residentd
   (lines 298-303), install, wipe `/tmp/weightd-mesh/*`, restart weightd.
   LATENT BUG: line 309 starts `$home/sparkpipe_weightd` but `$home` is
   unset in this function; under `set -u` (line 2) the background fork
   dies with "home: unbound variable" (reproduced in scratch; journal
   confirms the start never lands). weightd actually returns via
   `ensure_weightd` one loop later (line 374/327-337).
3. `self_update` (316-325): if on-disk agent sha differs from `START_SHA`
   captured at process start, `exec bash` the new file.
4. `sync_root` per root (275-282): `apply_manifest` then, ONLY on manifest
   change, `unload_root` + `start_root`.
5. `ensure_root` per root (343-368): if running residentd's exe sha
   differs from disk (journal: "recycling"), mark down and restart; boot
   guard: node uptime must be >= 900s (361-365).
6. `ensure_api` (131-169, rank 0 only): starts `sparkpipe_model_api` on
   port 8433 once the hub view reports 16 "ready"; kills duplicate or
   older-than-residentd instances.

### 1.3 Kill and start mechanics
- `unload_root` (87-110): `kill -9` on `pgrep -f
  "bin/sparkpipe_model_(residentd|api)"` scoped by `/proc/<pid>/cwd` ==
  root dir, then waits up to 5s for exit, then gates start on
  `MemAvailable >= pack_GB + 8` for up to 60s. SIGKILL, not TERM —
  `docs/FLEET_RELEASE.md` line 99 claims "TERM (cwd-scoped), wait for
  real exit": the deployed code does not do this (drift).
- `start_root` (112-129): `cd $root; ln -sf stage_$(rank).json
  config/stage.json; rotate residentd.log; nohup residentd
  --rank-index $RANK`.
- No two-phase UPDATE ledger exists in the deployed agent (grep: no
  UPDATE token in the script). The `UPDATE` + `UPDATE.<n>` files on the
  hub (Jul 7-Sep 12 mtimes, last `UPDATE` Sep 12 04:00 holding 15
  `down:<host>` entries and no `up:`) are stale artifacts of an older
  generation. `docs/FLEET_RELEASE.md` lines 90-96 describe that older
  behavior.

### 1.4 Version drift, one artifact, three copies
- Deployed agent everywhere: `71ec91d332f2f076` (heartbeat `agent` field,
  all 16 `sparkN.json`; files on sparkf `~/release/core/bin/` and
  spark1 `~/sparkdata/core/bin/`).
- Hub `~/release/core/MANIFEST` (mtime Sep 12 00:53) still lists agent
  `d53a7d6d81796aef` (= repo `tools/fleet_node_agent.sh` today) and
  weightd `59548787f143287a`. Since `apply_manifest` triggers only on a
  MANIFEST diff, the manifest channel cannot deliver either current
  binary; both reached nodes out-of-band (see 2.3).
- Other copies on sparkf: `~/sparkpipe-build/tools/fleet_node_agent.sh`
  = `d53a7d6d81796aef` (repo build tree), `~/sparkpipe/tools/` =
  `561338d6dd029269` (older checkout), `~/sparkpipe-build-958/tools_local/`
  = `71ec91d332f2f076` (source of the deployed copy).

### 1.5 Heartbeats
- `report()` (agent lines 40-66) writes one JSON line: host, time,
  weightd sha16, agent sha16, per-root {state, residentd sha16, driver
  sha16} to `~/current/<host>.json`, then `scp`s it to
  `$HUB:current/`. `report_if_changed` (68-85) only fires when root
  states or pids change — so `sparkN.json` mtime is a change marker,
  NOT a liveness signal (observed: all mtimes 17:13 while heartbeats
  were healthy).
- `fleet_view_serve.py` (`/home/sparkf/fleet_view_serve.py`, served by
  `fleet-view.service` on :8801; lines 44, 59-67) reads `epoch`, `load`,
  `mem_avail_gb`, `pid`, `rss_mb`, `log_age_s` fields that the deployed
  `report()` never writes, so `age_s` values in `/summary` are
  meaningless (now - 0). Schema drift, no runtime harm.

## 2. The hub side (sparkf)

### 2.1 Directories
- `~/release/glm53flash.fp8.tp16/` — serving root: `bin/` (residentd,
  api, batch, registrar, weightd copy), `lib/` (hidden_transport.so,
  model_serving_adapter.so), `stages/stage_000/model_driver.so`,
  `config/stage_00..15.json`, `model_resident.json` (16 rank_index
  entries), `MANIFEST` (33 lines).
- `~/release/core/` — `bin/{sparkpipe_weightd, fleet_node_agent.sh}` +
  `MANIFEST` (2 lines).
- `~/release/qpn/<host>/{<root>,mesh}/` — rendezvous/mesh record uploads
  pushed by agents (`sync_rendezvous`, agent lines 186-234) and pulled
  by peers over :8802.
- `~/current/spark{0..9,a..f}.json` — heartbeat landing zone
  (fleet_view_serve reads here).
- Staging areas: `~/dsv4pro_manifest_stage/` (Sep 10), `~/sp4rbuild/`,
  `~/sp4rtools/` (Sep 14), `~/sparkpipe-build/` (publish build tree),
  `~/sparkpipe-build-958/` (active dev tree; `tools_local/` holds the
  deployed agent and the out-of-band deploy tools), `~/weightsd1/`
  (weights stable daemon workspace, Sep 14), `~/nccl-staged/`,
  `~/packtools-stage/`, `~/kvcache/`.

### 2.2 In-band publish (repo tooling)
- `tools/module_build_release.sh` (repo + `tools_local` variant): fetch
  ref, `git reset --hard` the SHARED `~/sparkpipe-build` (line 18 — no
  lock), host-build residentd/api/weightd/driver (line 27 — weightd is
  rebuilt on EVERY publish), park hub's own agent + residentd (33-40),
  GPU module publish with receipts, driver compile, then
  `publish_local.sh` (59) and `exec publish_core.sh` (61).
- `tools/publish_local.sh` (repo): installs driver/DSO/adapter/residentd/
  api into `~/release/<root>/` and regenerates `MANIFEST` (lines 32-35).
  It does NOT touch any UPDATE file — `docs/FLEET_RELEASE.md` line 65
  ("touches UPDATE (the restart trigger)") is stale; the real trigger is
  the MANIFEST diff.
- `tools/publish_core.sh` (repo and tools_local are identical): installs
  weightd + agent into `~/release/core/bin/`, regenerates core MANIFEST.
- Guard: publish rejects a deployment whose runtime_root does not match
  the release root (publish_local.sh lines 11-20).

### 2.3 Out-of-band deploy tooling (ACTIVE — used repeatedly today)
On sparkf `~/sparkpipe-build-958/tools_local/` (mtimes Sep 13 16:56):
- `p` (deploy, 4.1 KB): (8) `FLEET_REF=sparkf:release tools/fleet_serve.sh
  glm53flash.fp8.tp16 sync` — tar-streams hub release root into every
  node's `~/sparkdata/glm53flash.fp8.tp16` via
  `ssh hub "tar -C ... -cf - lib bin stages config model_resident.json" |
  ssh node "tar -C ~/sparkdata/<root> -xf -"` (`fleet_serve.sh` lines
  102-113) — this is the `tar -C .../glm53flash.fp8.tp16 -xf -`
  fingerprint D-1 observed. (9-16) md5 gate on driver.so across all 16.
  (17-28) pushes hub `release/core/bin/sparkpipe_weightd` into every
  node's `~/sparkdata/core/bin/` — WITHOUT regenerating core MANIFEST
  (root cause of the manifest/installed drift in 1.4). (29-36) same for
  the agent script. (37-43) per-node: if `core/bin` weightd differs from
  installed, install and `kill -9` weightd by exe link — fleet-wide,
  parallel. (44-47) SECOND wave: `kill -9` every process whose exe is
  `*sparkpipe_model*` with cwd in the root OR exe under
  `sparkdata/weightd` — unconditional. (48-68) poll 16 listeners
  (10.10.100.10+i:19560+i), API health, driver marker grep.
- `b` (bounce, 1.4 KB): "simultaneous full-stack bounce only (no sync,
  no build)": the line 44-47 wave of `p` as a standalone + same polls.
- `fullbuild_once.sh`: build + publish_local + `cp build/sparkpipe_weightd
  /tmp/weightd.new` (staging for a weightd swap).
- `module_build_release.sh` (tools_local variant): same skeleton as repo
  but does NOT end in publish_core (hub core MANIFEST untouched since
  Sep 12 00:53 while `release/core/bin/sparkpipe_weightd` mtime is Sep 14
  17:01 — replaced by the `p` path, not by publish_core).
- Transport: the waves arrive over ONE long-lived ssh connection:
  spark1 auth.log last `Accepted` before the afternoon waves is
  14:54:57.864 from `10.20.0.1` port 53122 (workstation gateway), and
  `ss -tn` on spark1 shows `10.20.0.11:22 <-> 10.20.0.1:53122` ESTABLISHED
  (sshd pid 2510880, started 14:54:57, root privilege side). All later
  tar streams and kill waves multiplex over it, which is why no sshd
  entries appear at wave times.

### 2.4 Observed cadence (spark1 `journalctl --user -u fleet-agent`, Sep 14)
Root-MANIFEST changes ("manifest changed; syncing") at 15:40:49, 16:01:43,
16:38:05, 16:51:04, 17:01:34, 17:21:40 (+ another publish by ~17:40: all
16 heartbeats moved to driver sha `9839bc634a7e237e`). Each completed
wave = SIGKILL of residentd and weightd within ~12s of the manifest
change (job "Killed" notices 17:01:34/45/46; weightd restart 17:01:47;
"waiting for weightd mesh" 17:01:47; mesh recs rebuilt by 17:30). The
16:38 event shows a partial publish: manifest changed, fetch errors, no
restart (apply_manifest returns 1 on fetch errors, agent line 270) —
agents race the publisher and silently retry; no lock exists.
No "deliberate restart" (install_core) fired all day: every weightd kill
came from the out-of-band wave, not the agent.

## 3. Channels inventory (what actually gets updated, and how)

| Channel | Content | Mechanism | Scope |
|---|---|---|---|
| Root release (in-band) | residentd, api, batch, registrar, driver .so, transport/adapter .so, stage configs, model_resident.json, (stale weightd copy) | MANIFEST diff on :8802 -> fetch+sha256 -> kill -9 residentd/api -> restart | per-root, fleet-wide within seconds; ANY file change restarts the root |
| Core release (in-band, DORMANT) | weightd, agent | core MANIFEST diff -> install_core/self_update | node-wide, fleet cascade on sha change; stale since Sep 12 |
| Out-of-band (ACTIVE) | whole root tree, weightd, agent | ssh tar-stream + scp + unconditional fleet-wide kill -9 | ALWAYS all 16 nodes, all daemons, regardless of what changed |
| Heartbeats | state + shas per node | scp on change -> hub `~/current/` | per-node |
| Rendezvous/mesh | .rec mesh records | push to `release/qpn/`, peer-pull :8802 | per-node, continuous |
| Packs | `<root>/packs/*.sp` (+experts, .mtp sidecars) | pack pipeline pushes build-host -> node NVMe directly (`tools/glm52_deploy_packs.sh` pattern); `chattr +i` immutable (verified spark1: `---i`, 82G, 10 files) | per-rank files; agent never touches packs |
| weightsd (in flight) | stable weights channel | `~/weightsd1/build/sparkpipe_weightsd --socket /run/sparkpipe-weightsd/weightsd.sock` on nodes (spark1 pid 74588, started manually 07:46 Sep 14, not a unit); `weightdctl` beside it | not yet integrated with agent or publish |

Nothing is per-driver scoped end to end: one root dir exists per driver
(40+ roots in spark1 `~/sparkdata`), but the bounce tooling kills
weightd (shared by every root) and the agent restarts on any manifest
byte change, so one dev's republish bounces the whole fleet.

## 4. Restart coupling trace — why a glm53flash republish kills weightd on all 16

1. Publisher regenerates hub root `MANIFEST` (publish_local.sh lines
   32-35) — 17:01:33-ish, 17:21:39 (hub mtimes).
2. `p` line 8 tar-streams the same tree into every node
   (`~/sparkdata/glm53flash.fp8.tp16`) — D-1's fingerprint, ~9s of
   transfer.
3. Node agent (`sleep 1` loop) diffs `MANIFEST` vs
   `.applied_manifest` (agent line 244), refetches and applies all 33
   files (247-269).
4. `sync_root` (275-282): `unload_root` kill -9 residentd + api
   (90) — first SIGKILL wave (per-node, near-synchronized fleet-wide
   because the loop is 1s).
5. `p` lines 44-47 (or `b` 10-17): ssh fanout kill -9 matching
   `*sparkpipe_model*` with cwd in the root OR exe under
   `sparkdata/weightd` — weightd SIGKILLed on ALL 16 nodes
   unconditionally, ~12s after the manifest change (17:01:46 on spark1).
   This is the wave D-1 saw; the agent merely logs the job as "Killed"
   and restarts weightd (`ensure_weightd`, 17:01:47).
6. weightd restart wipes `/tmp/weightd-mesh/*` (agent line 331);
   `start_root` blocks residentd on `.ready` (117-121); mesh rebuilds
   via rendezvous push/pull (recs 17:21 -> 17:30); residentd cold-loads
   (lazy tiers fault back, ~1-2 min/node per docs); `ensure_api`
   restarts the API at 16/16 ready.
Net effect: EVERY republish — even config-only — costs the fleet 16x
(residentd SIGKILL + weightd SIGKILL + mesh rebuild + cold reload + API
bounce). Observed seven times between 15:40 and 17:40 on Sep 14.

A second, currently dormant coupling: if anyone reruns `publish_core.sh`
(in-band), every node's `install_core` (298-303) kills weightd AND all
residentd fleet-wide; its start line then dies on unbound `$home` (set
-u), so weightd actually returns via `ensure_weightd`. Both the design
("weightd killed on any weightd rebuild — and weightd is rebuilt on
every module publish", module_build_release line 27) and the bug are
armed at all times.

## 5. NVMe staging and warm storage

- Node layout (`~/sparkdata/`): `core/bin/` (agent, weightd staging),
  `weightd/` (installed weightd), `<root>/{lib,bin,stages,config,packs,
  rendezvous}` + `residentd.log(.prev)`, `api.log`. spark1 holds 40+
  root dirs, ~1.6 TB, most stale (k3.*, qwen*, glm53full.*, dsv4_* ...);
  only `glm53flash.fp8.tp16` is in the unit's ROOTS.
- Packs: `packs/*.sp` per hex rank (+ `.experts`, `.mtp` sidecars),
  immutable via `chattr +i` (verified), placed by the pack pipeline
  straight onto NVMe, never synced by the agent (agent never references
  packs except size for the memory gate, line 101).
- Debris convention: docs/FLEET_RELEASE.md law "sidecar debris breaks the
  digest scan" — observed debris in packs/: `.old`, `.experts.old`,
  `.partial-20260908`, `.premtp-old` files; cleanup is manual.
- Ceph: warm storage per operator spec. ABSENT from all deploy tooling
  inspected (no ceph reference in agent, p, b, publish_*, fleet_serve,
  deploy_packs); the pack pipeline moves build-host -> node NVMe
  directly. Ceph-as-source enforcement does not exist yet.

## 6. Not verified / absent (stated explicitly)

- ABSENT: any UPDATED/UPDATE marker logic in the deployed agent or
  publish_local.sh (stale artifacts only).
- ABSENT: locks/leases anywhere in the publish path
  (module_build_release, p, b, publish_local/core, agent).
- ABSENT: drain/TERM handling in the deployed agent (docs claim it).
- ABSENT: ceph integration in deploy tooling.
- NOT PINNED: the human/operator behind the 14:54:57 multiplexed deploy
  session from 10.20.0.1 (key fingerprint RSA
  SHA256:WdsDgNV36DoDplaYmiYytHlCdi1QDJ3HKmD8OfnwOHU); and which exact
  wrapper invoked p/b at each wave time.
- NOT RUN: any write, kill, restart, or daemon contact by this audit;
  spark3/spark6 untouched (observation of shared state only).
