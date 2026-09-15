# Multi-dev deploy reliability proposal

Companion to `docs/DEPLOY_PROTOCOL_MAP.md` (evidence and file:line
references live there). Goal: multiple devs must be able to build and
deploy concurrently without bouncing each other's work. Items are
ordered: 0 is prerequisite hygiene, 1-6 are the operator's evaluation
points. Effort: S <= half day, M <= 2 days, L larger.

## 0. Prerequisite hygiene (do first)

0.1 One deploy path. Retire or wrap the out-of-band tools on sparkf
`~/sparkpipe-build-958/tools_local/{p,b}`: their unconditional
fleet-wide kill (p lines 37-47, b lines 10-17) and out-of-band core
pushes (p lines 17-36) are the direct cause of both the weightd waves
and the manifest/installed drift (map 1.4, 2.3, 4). Replace their body
with: run the repo publish tools, then wait for agent convergence
(poll `sparkf:8801/summary` to 16/16) — the agent already restarts only
what the manifest changed once item 2 lands. Effort S. Risk low:
removes a parallel today-active path, so coordinate with whoever owns
the 10.20.0.1 deploy session.

0.2 Reconcile the core channel. Regenerate `~/release/core/MANIFEST`
from the binaries actually installed fleet-wide (weightd `49b8e01e...`,
agent `71ec91d332f2f076`) so the in-band channel and reality agree
before any change to `install_core` semantics. Effort S. Risk low.

0.3 Fix the `install_core` start bug: agent line 309
`setsid nohup "$home/sparkpipe_weightd"` references unset `$home`
(dies under `set -u`; reproduced). Change `$home` to `$wd` (defined
line 291). Effort S. Risk low. Note: with 0.2 done, the next core
publish exercises this path for real.

0.4 Heartbeat/view schema: make `report()` (agent lines 40-66) emit
`epoch`, `pid`, `rss_mb`, `log_age_s`, `load`, `mem_avail_gb` —
`fleet_view_serve.py` lines 44, 59-67 already consume them. Restores
meaningful `age_s` (staleness detection) in `/summary`, which multi-dev
coordination needs to trust. Effort S. Risk low. Agent ships itself via
`self_update` (316-325) — no per-node work.

## 1. Per-channel release markers (one dev's deploy touches only that channel)

The structurally-per-driver layout already exists: one release root per
driver, one `MANIFEST` per root, one ROOTS list per node (agent line 3,
unit ExecStart). What is missing is scope discipline in two places:

- Node side: today ANY manifest change restarts the root
  (`sync_root` 275-282). With item 2 below, restart becomes
  content-scoped; no marker file is needed on the node — the diffed
  MANIFEST is the UPDATED marker per channel.
- Hub side: the deploy entry point must take the root name as its
  channel id and only ever write `~/release/<root>/` (publish_local.sh
  already does) — i.e., forbid tooling that streams across roots
  (0.1). For shared services the channel is `core`, and after item 4
  its blast radius is one daemon.

Adding a second driver = append to the unit's ROOTS CSV; the loops
(375-378) already iterate. Effort S (mostly 0.1 + item 2). Risk low.

## 2. Restart only what changed (content-hash scope)

`apply_manifest` (agent 236-273) already has old vs new manifest lines
in hand (`$root/.applied_manifest` vs fresh). Change:

- Diff the two manifests BEFORE fetching; fetch only lines whose sha
  changed (also cuts re-downloads; today it refetches all 33 files on
  any change, line 247).
- Map changed paths to restart scope: `bin/sparkpipe_model_residentd`,
  `lib/*`, `stages/*`, `config/model_resident.json` -> restart root;
  `config/stage_XX.json` -> rewrite `config/stage.json` symlink only,
  no restart (start_root already re-links per start, line 123);
  `bin/sparkpipe_model_api` -> rank 0 api only (ensure_api already
  kills stale api by proc_start order, 151-160);
  `bin/sparkpipe_weightd` in a driver root -> ignore (root copies of
  weightd are not the installed one; MANIFEST line 5 is dead weight).
- `sync_root` then restarts only if the residentd-relevant set is
  non-empty.

Effort S (~20 lines touched in one function). Risk low: fetch + sha256
verification untouched; failure path (any fetch error -> no restart,
line 270) unchanged. Payoff: config or doc edits stop bouncing the
fleet entirely.

## 3. Drain, don't kill (D-1 finding)

- Agent `unload_root` (87-110): send TERM first (it already scopes by
  `/proc/<pid>/cwd`), keep the existing exit-wait loop, extend the
  grace to ~15s, keep `kill -9` only as the deadline fallback. The
  correct drain logic already exists in the repo — port
  `tools/fleet_serve.sh` `stop()` (lines 26-51: parallel TERM, 30s
  exit poll, refuse-to-start-if-busy) into the agent.
- Out-of-band tools: `p` 44-47 and `b` 10-17 kill -9 by exe/cwd —
  replaced by 0.1; until deleted, switch their pattern to the same
  TERM-then-poll.
- `install_core` (298-303) keeps kill -9 by exe link for weightd (it is
  a deliberate binary swap) but should TERM residentd first and wait —
  residentd restart is implied anyway, and the memory gate (101-109)
  then measures a draining node instead of a post-kill cliff.

Effort S-M. Risk medium-low: drain lengthens the update window; the
fleet-stops-together behavior (all nodes hit unload before any starts,
because each node applies before starting) is preserved; QP re-wiring
risk that motivated the old stop-together gate is handled by the
load-order-independent listener design (FLEET_RELEASE "Deliberately
absent").

## 4. weightsd stable channel owns weightd restarts

Current state (map 3, 5): `sparkpipe_weightsd` already runs on nodes
(spark1 pid 74588, `~/weightsd1/build/`, socket
`/run/sparkpipe-weightsd/weightsd.sock`, `weightdctl` beside it) but is
started by hand and integrated with nothing; weightd restarts are
triggered today by the out-of-band bounce, and the dormant
`install_core` cascade is armed behind it.

Changes:

1. Make weightsd a system unit on each node
   (`ExecStart=.../sparkpipe_weightsd --socket ...`, `Restart=always`) —
   one-time bootstrap like the agent's (fleet_sync.sh start pattern).
2. Define the stable channel as an announced sha: weightsd serves
   `weightd/stable` = a sha16 published on the hub (e.g.
   `~/release/core/STABLE`); it installs and (drain-)restarts weightd
   only when the installed weightd differs from the announced stable —
   never on a dev publish. Reuses the existing retire-gate concept from
   `fleet_sync.sh retire-update` (lines 43-57).
3. Remove weightd from every other restart path: `p`/`b` kill patterns
   (`*sparkdata/weightd*`) die with 0.1; agent `install_core` becomes
   the fallback that defers to weightsd (or is reduced to installing
   the binary and letting weightsd own the process).
4. Publishing weightd becomes a two-step: publish to a candidate
   channel, then announce (move the sha to STABLE) when the fleet is
   idle — devs stop inheriting weightd restarts from unrelated module
   publishes because module_build_release line 27 stops rebuilding
   weightd unconditionally (build it only when `runtime/` changed).

Effort M. Risk medium: new owner for the most shared process; keep the
agent's `ensure_weightd` as a watchog fallback so a dead weightsd can
never leave a node weightless.

## 5. Data staged on NVMe, ceph as source

Already true for packs: per-rank `.sp` files live in
`<root>/packs/` on node NVMe, `chattr +i` immutable, placed by the pack
pipeline build-host -> node directly (map 5). Gaps and fixes:

- Ceph-as-source is ABSENT from tooling. Add one step to the pack
  pipeline: publish finished packs to ceph (warm) and deploy node NVMe
  FROM ceph (content-addressed, sha256 verified like apply_manifest)
  instead of host-to-host scp — rebuilds and node re-imaging then never
  depend on a dev's build tree staying alive. Effort M. Risk low.
- Retention: 40+ stale roots and pack debris (`.old`, `.partial-*`,
  `.premtp-old`) sit on NVMe (spark1 ~1.6 TB). Add a hygiene command
  (hub-driven, dry-run first): list roots absent from every node's
  unit ROOTS and from `~/release/` older than N days; delete packs
  debris only after a fresh immutable-set verify. Effort S. Risk low
  (deletion is the risky part — gate it behind the operator).

## 6. Concurrency: locks/leases per channel

- Publisher lock (the acute one): `module_build_release.sh` line 18
  `git reset --hard` on the SHARED `~/sparkpipe-build` — two
  simultaneous publishes stomp each other's tree and can publish mixed
  artifacts. Wrap the whole script in `flock ~/release/.publish.lock`
  (or per-root `~/release/<root>/.lock` once 1 lands). Effort S.
  Risk low.
- Agent-vs-publisher race (observed live 16:38: manifest changed while
  files were still landing -> fetch errors -> silent retry): make the
  publish atomic at the hub — write files into
  `~/release/<root>/.staging/`, then `mv` into place + regenerate
  MANIFEST last (publish_local.sh already writes MANIFEST last; extend
  it to stage the whole tree). Then a MANIFEST change guarantees a
  complete tree and the 16:38-class retry disappears. Effort S-M.
  Risk low.
- Multi-dev leases (only if concurrent SAME-channel deploys become
  routine): `~/release/leases/<root>` with pid + expiry, checked by the
  publisher; the agent needs no change (it converges to whatever
  manifest wins last). Effort M. Risk low. Do NOT add leases on the
  node side — the agent loop must stay lock-free to keep converging.

## 7. Sequencing and acceptance

Order: 0.1 -> 0.2/0.3 -> 2 -> 3 -> 0.4 -> 1 (no-op after 2) -> 6 ->
4 -> 5. After each step the acceptance check is the same one command
per class: a config-only publish changes no daemon (heartbeat residentd
shas unchanged); a residentd publish restarts 16 residentd and 0
weightd; two publishers on different roots never interleave; one
publish by any dev produces exactly one wave of exactly the daemons
whose inputs changed.

All changes are in: `tools/fleet_node_agent.sh` (deployed everywhere,
self-updating), `tools/publish_local.sh`, `tools/module_build_release.sh`,
`tools/fleet_serve.sh` (drain donor), the hub `tools_local/{p,b}`
(deletion), and the weightsd unit — no daemon source changes required.
