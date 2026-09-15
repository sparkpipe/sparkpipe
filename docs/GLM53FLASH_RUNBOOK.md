# glm53flash coredev runbook — build, publish to sparkf, deploy, test, debug

Owner: mgr2 (glm53flash.fp8.tp16 lane, TP16 on sparks 0-9a-f). Everything
below is reproducible from the lane worktree `/Users/mac/pr-bisect958`
(branch `glm53flash-hill1`, PR #1014). Build host: `sparkf`. Serving front:
`spark0:8433`. Everything runs over ssh with the workstation's keys.

## 1. The four trees

| Tree | Path | What it is |
|---|---|---|
| Lane worktree | `/Users/mac/pr-bisect958` | git; the only place source is edited. `tools_local/` holds the lane tools. |
| Build tree | `sparkf:~/sparkpipe-build-958` | compiled by `tools_local/m`. Synced by rsync (whole tree, no timestamps). |
| Release tree | `sparkf:~/release-glm53flash/glm53flash.fp8.tp16/` | the deploy payload (bin+lib+stages+config+model_resident.json+VERSION). Assembled by `m`, consumed by `p`. Lane-private by operator order — the manager's `sparkf:~/release/` is NOT ours anymore. |
| Node lane roots | `~/sparkdata/glm53flash.fp8.tp16` on all 16 sparks | the runtime. Engines + api run from here; supervised by the manager's `fleet_node_agent` (patched by us, see §7). |

VERSION in the release tree = the git short hash that built it. `p` refuses
to deploy if it does not match local HEAD. The MD5 gate checks the driver
.so on every node against the hub before any bounce.

## 2. Build + publish to sparkf (`tools_local/m`)

Run from the worktree root. Steps it performs, in order:

1. `HASH_ARG=$(git rev-parse --short HEAD)` — the version stamp.
2. scp the hot files (module sources, model header, firmware header,
   transport, weightd sources, mesh kernel headers), then
   `rsync -r --delete --exclude .git --exclude build --exclude __pycache__ --exclude tools_local ./ sparkf:~/sparkpipe-build-958/`.
   - `-r`, NOT `-a`: `-a` preserves mtimes and make then rebuilds nothing.
   - The `--delete` matters: stale remote files have bitten us twice.
3. Remote (ssh `bash -s "$HASH_ARG"`), inside `~/sparkpipe-build-958`:
   - `systemctl --user stop fleet-agent`; TERM all residentds; wait ≤15s.
   - `make -j16 -C modules/glm5_next_resident_decode_stage publish
     EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033
     CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d" " -f1)
     NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a`
   - `build/sparkpipe_model_compile --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json
     --library build/module_library --output ~/sparkdata/out --cc /usr/bin/cc --include include
     --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart
     --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread`
   - `tools/publish_local.sh glm5_next_resident_decode_stage fp8 glm53flash.fp8.tp16`
   - `make -j16 build/sparkpipe_weightd build/sparkpipe_model_residentd build/sparkpipe_model_api`
   - Assemble the release tree: fresh bin (residentd/api/weightd + batch/registrar),
     `lib/` from the lane root, `stages/` from `~/sparkdata/out/stages`,
     `config/` + `model_resident.json` from the lane root, `VERSION=$HASH_ARG`.
   - Gate: `strings ~/sparkdata/out/stages/stage_000/model_driver.so | grep -c GRAPH-CAPTURE-OK`
     must be ≥1 or the build is refused (marker-missing = the driver is wrong).
   - `systemctl --user start fleet-agent`.
4. Prints `BUILD-OK <driver-md5> ver=<hash>`.

Compile gates (config law): `SPARK_BATCH_BUCKET` must be `-D`'d on ALL THREE
compile lines — variant (Makefile line with `-USPARK_BATCH_BUCKET -DSPARK_BATCH_BUCKET=$(lastword $(MODULE_BATCH_VARIANT_BUCKETS))`),
host (`MODULE_COMPILE_FLAGS` in the module Makefile carries the same -U/-D),
and the cuda validator (`validation_nvcc_extra_args` in the validator .sh).
These were remote-only patches once and got silently lost; they are committed now.

If a rebuild produces an identical binary where you expected a change: make's
object deps have lied before — `rm` the specific `.o` + `.a` + binary on
sparkf and rebuild once.

## 3. Deploy (`tools_local/p`)

`FLEET_REF=sparkf:release-glm53flash tools/fleet_serve.sh glm53flash.fp8.tp16 sync`
(tars lib/bin/stages/config/model_resident.json — `stage.json` is excluded on
purpose, it is per-node), then the VERSION gate (release tree stamp vs local
HEAD — refuses on mismatch), then the MD5 gate (driver .so on every node vs
`sparkf:~/sparkdata/out`). Green = `MD5-GATE-OK`.

Deploy discipline (hard-won): every binary change makes the agent recycle
every engine (its hash check). So deploy ONCE, then converge, then test.
Never deploy mid-test. Expect ~2 min to 16/16.

## 4. Bounce + status + canary

- `~/lane-glm53/tools/b` — bounce engines + listeners (`BOUNCE-DONE listeners=16 health=200`).
- `tools_local/s` — hub/node/agent/mesh status.
- Convergence check: `ssh sparkf 'grep -l "\"state\":\"ready" current/*.json | wc -l'`
  → 16; plus `tail -1 api.log` on spark0 shows `model_api ready`. The api is
  spawned by the agent after 16/16 (af2 keeps it fresh vs engine restarts).
- `tools_local/t [N]` — canary (4tok) + N-token run; then greps the engine log
  counters. Success = first token `[3764,10]` and a `rate=` line. t=15.0s
  exactly = an expired request (stale api binary class); t=5.0s = a 5s
  transport spin timeout (see the print glossary).

## 5. weightd (ours; devs are on weightsd — different daemon)

- Runtime: `~/sparkdata/weightd/sparkpipe_weightd --socket /tmp/spark_weightd.sock
  --mesh-rank <N> --mesh-interface rocep1s0f1 --mesh-sgid-index 3` (argv differs
  per node — always preserve it).
- Restart with a fresh binary: `tools_local/w` — pulls
  `sparkf:~/sparkpipe-build-958/build/sparkpipe_weightd`, pushes per node,
  TERM→KILL, removes a stale socket file, relaunches with cwd = the binary dir
  (lazy-pack path resolution needs it), waits for `weightd-mesh: ready`.
- Relay log: `~/weightd.log` on each node (WD-SHIP / WD-MESH-CQ / WD-WIRED).
- Restarting weightd zeroes the mesh buffer (fresh cells) and forces engines
  to re-wire; expect one convergence cycle. A node reboot can bring weightd up
  before the NIC → `switch port not active` → degraded mesh → restart weightd
  once the link is up.

## 6. Ports

19560+rank = engine control listeners (16), 8433 = our api, 18433 = the (dead)
t1 rig's api. The T1 rig root `~/sparkdata/glm53flash.fp8.tp16.t1g53` is a
port-conflicted crash-loop zombie — its `/tmp/t1g53_supervise.sh` respawns it;
kill the supervisor first. d1-weightd cwd processes = old D-1 rig, kill on sight.

## 7. The agent (manager's, patched by us)

`~/sparkdata/core/bin/fleet_node_agent.sh` + systemd user unit `fleet-agent`.
Our patches (re-apply after the manager updates it; check for the markers):
- `af`: root_state matches ANY residentd pid whose cwd is the root (was
  oldest-pid — killed its own spawns in a 2s loop whenever older rigs existed).
  Marker: `af-any-pid`.
- `af2`: ensure_api's predates-check samples the LANE engine (was oldest pid)
  so a stale api restarts after bounces — the api has no engine reconnect.
  Marker: `af2-lane-rpid`.
Latent (unpatched, manager's call): the hash-recycle check kills engines on
every binary change by design; the exe readlink of a replaced binary reads
"none" sometimes (harmless recycle).

## 8. Debugging quick reference

- Engine log: `~/sparkdata/glm53flash.fp8.tp16/residentd.log` per node
  (`.prev` = previous boot). API log: `api.log` in the same dir.
- Hub states: `sparkf:~/current/*.json`.
- Agent journal: `journalctl --user -u fleet-agent` (Killed vs Segmentation
  fault vs down;starting tells you churn class).
- gdb ATTACH is blocked on some nodes (yama ptrace_scope; spark1 confirmed) —
  run the engine UNDER gdb as a child instead (`gdb -batch -ex run -ex bt
  --args ./bin/sparkpipe_model_residentd --deployment model_resident.json
  --rank-index N` from the lane root; stop the agent first, restore after).
- Core dumps are eaten by apport; don't count on them.

## 9. Failure-print glossary (all loud, all named)

| Print | Meaning |
|---|---|
| `BOOTCFG tp_degree=... graph_path=...` | which config the collective actually ran (first line of every boot) |
| `CKEY-WRITE rank=0 base=... cell=...` | rank 0 advanced the band cell for a new chain |
| `CKEY-ADOPT rank=N epoch=...` | rank N adopted an epoch from the cell |
| `CKEY-WAIT-TIMEOUT rank=N cell=... entry=... tail_epoch=...` | peer gave up waiting for a cell change (5s) — the barrier race (see blockers handoff) |
| `CKEY-RESET / CKEY-BAD` | cell held a poisoned/out-of-domain value |
| `MESH-SPIN-TIMEOUT rank=... peer=... want=... got=...` | an op's tail wait timed out; want/got decode as (epoch<<40)|(req<<16)|round — a want/got epoch difference is the sync defect; same epoch, different round = a lost publish |
| `MESH-CANCEL-ABORT` | a cancel arrived mid-wait (request unwind) |
| `REBASE-TIMEOUT / REBASE-CANCEL` | the non-chain-keyed rebase wait escaped loudly |
| `GRAPH-CAPTURE-OK bound=N` | decode graph captured (N = context runway) |
| `DEGRADE graph-fallback n=...` | a round fell off the graph path (rate-limited) |
| `DEGRADE graph-stuck slot=... alt=... err=...` | a replay wedged past the 15s watch; err = the seq the in-graph wait died on |
| `DEGRADE graph-wait-timeout ... seq=...` | an in-graph wait hit its 5s deadline |
| `KV-MATCH-FAIL ...` | admission kv-lane identity mismatch (status 9 class) |
| `G5N-DBG reduce submit -> N` | transport refused an op (N = status) |
| `WD-SHIP idx=... seq=... total=...` | relay shipped a doorbell entry (rate-limited print) |

## 10. Git / GitHub

All pushes via `tools/sparkpipe_github_pat.sh git push origin glm53flash-hill1`
(PAT wrapper, identity `sparkpipe`, never print the token). One PR per driver
(currently #1014). Commit BEFORE every build/deploy — the PR-bound law.
