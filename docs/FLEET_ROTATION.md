# Fleet rotation

The owner's ruling of 2026-09-29: GLM Full and Kimi K3 each get dedicated fleet time, every third hour. The third hour runs GLM Flash production plus one other driver, and the other driver rotates through the drivers that are not validated yet, so that every driver does at least slow inference. Four GLM Flash TP4 replicas were rejected because they need 4x the KV cache, so Flash stays TP16 with 1/16 KV per rank.

`tools/fleet_rotation.py` implements the rotation on the rtx5090 hub as a systemd `--user` timer. It has no model knowledge of its own: every model, command, port, pack and memory figure lives in `deployment/fleet_rotation/rotation.json`. `tests/test_fleet_rotation.py` checks the state machine against a fake fleet driven by a fake ssh.

## Schedule

The cycle is three UTC hours, anchored at `cycle.anchor_hour` (0). Hour `h` runs slot `cycle.slots[(h - anchor) % 3]`:

| hour mod 3 | slot | models | if its base model cannot run |
|---|---|---|---|
| 0 | `full` | GLM-5.3 Full owns the fleet (lane 6, pinned fp8, ~68 GiB/node, API rtx5090:8446) | `flash_plus` |
| 1 | `k3` | Kimi K3 owns the fleet | `flash_plus` (K3 is not runnable until the k3-serve lane delivers; lead experiment windows take this hour through the lead lock) |
| 2 | `flash_plus` | GLM Flash production (g53-api :8433) plus one companion | none; production is the floor |

The companion is picked once per hour: it is the next runnable entry of `companions` after the last one used. When no companion is runnable or fits, `default_companion` (Qwen TP4, validated) runs instead. `fleet_rotation.py schedule` prints the next hours. With today's config and the GLM Full release root not yet installed, `full` demotes itself to `flash_plus` at each FULL hour (see "Demotion" below).

## Each tick

The timer runs `tick` every 5 minutes at :20 s. A slot changes at the top of the hour, so the first tick of an hour performs the transition.

1. **Lock.** If `~/fleet-rotation/ROTATION_PAUSE` exists (manual or automatic pause), or the mirrored `lock/ROTATION_PAUSE` exists, or the mirrored `lock/PERF_HOLDER` starts with `lead-`, the tick touches nothing. It records `phase=paused|paused-coord|preempted` and marks the state for resync.
2. **Observe.** One ssh per node runs every runnable model's `engine.up` check, and the hub runs every `api.up`. A model is `up` (all nodes and its API), `down` (nothing), or `mixed`.
3. **Adopt or compare.** On the first tick, and after any lock or pause, the observed fleet is adopted as the current state; `mixed` models get a full stop before anything starts. Otherwise the observed fleet must match the recorded one:
   - a model that is up but not recorded means somebody changed the fleet outside the rotation: alert and auto-pause;
   - a recorded companion that died means fall back to production;
   - recorded production that is not up means alert and no transition until it is back. Production belongs to fleet-agent and the lead.
4. **Steady.** If the fleet already matches the target, only the APIs' `/health` is checked. A failing companion falls back to production.
5. **Plan** (read-only, before anything is stopped):
   - run each incoming model's `engine.precheck` on its nodes and `api.precheck` on the hub;
   - predict each node's MemAvailable after the swap: now + outgoing `mem_gib` − incoming `mem_gib`. It must stay at or above `floor_gib` (20).
   - **Demotion:** a slot base that fails either check demotes the slot for this hour (FULL → FLASH+1). A companion that fails either check is skipped for this hour in favour of the next one. Production is never held for a model that cannot start.
6. **Transition:**
   1. Floor check on every node (>= 20 GiB MemAvailable).
   2. Stop the outgoing models, companions first and production last. For each: stop its API, stop its engine units (for production, `agent.hold` on 16/16), and wait until `engine.up` is false everywhere (`stop_timeout_s`).
   3. Free memory by pack only: `weightd_warm <socket> --reclaim-pack <sha>` for each of the model's own packs (`packs`, sha files). It is refused on a node where the engine is still up. The config validator rejects any node-global `--reclaim` anywhere in the file.
   4. Start the incoming models, production first. For each: a memory gate on the nodes where it is not already up (`mem_gib` + floor), `engine.start`, then wait for `engine.ready` on every node (`ready_timeout_s`). After `grace_s`, an engine whose unit has exited fails the step early. Then start the API, wait for `api.health` (`api.timeout_s`), and run the smoke request (`smoke.body` to `smoke.path`; the reply must contain `smoke.expect`).
   5. Floor check on every node again.
7. **Failure** at any step (timeout, non-zero rc, ssh timeout, smoke mismatch, floor) triggers:
   - an `ERROR` line in `ALERT`;
   - **fallback:** stop every non-production model (API, units, reclaim-pack), then start production (idempotent when it is already up: unhold, ready, g53-api, Paris smoke). The failed hour stays on production alone; the same slot is not retried within the hour.
   - If the fallback fails: `phase=degraded`, a `CRITICAL` alert, and an automatic `ROTATION_PAUSE`. The lead takes over.

**Serving gaps.**
- A FULL or K3 hour starts with one transition during which nothing serves: production is held so the whole fleet is free.
- A failure costs one more transition, back to production.
- When a tick finds nothing serving (after a lead window, a degraded state, or a first run with the fleet held), it restores production before any slot model. The fleet is never left without a model for more than one transition.

Every command runs with a timeout (`ssh_timeout_s` per ssh call, plus the per-step timeouts). The tool never touches weightd binaries, the Flash release root or `~/release`. For production it only touches `agent.hold`, the `--reclaim-pack` of its own rank pack, and `systemctl --user start|stop g53-api`.

## Files on the hub (`~/fleet-rotation`)

| file | content |
|---|---|
| `state.json` | active models, phase, slot, per-hour companion choices, demotions, override/skip, last alert |
| `rotation.log` | append-only event log (`TRANSITION-BEGIN`, `STOP`, `RECLAIM-PACK`, `START`, `READY`, `SMOKE-PASS`, `FALLBACK-*`, `ALERT`, `ADOPT`, `HOLD-OFF`) |
| `ALERT` | append-only `WARN` / `ERROR` / `CRITICAL` lines (a repeated message is written once) |
| `STATUS` | one line: `ROTATION <time> phase= slot= active= target= next= lock= alert=` |
| `schedule.json` | the next 12 hours plus per-model availability, served by the responder |
| `ROTATION_PAUSE` | manual or automatic pause (reason inside) |
| `lock/PERF_HOLDER`, `lock/ROTATION_PAUSE` | mirror of the coordination lock, written by `sync` |
| `tick.out` | the timer's stdout/stderr |

## Lead lock and heartbeat (`sync`)

The coordination directory lives on the workstation, and the hub cannot read it. `fleet_rotation.py sync --lanes /Users/mac/sparkpipe-coord/lanes` runs on the workstation and does two things:
- pushes `lanes/PERF_HOLDER` and the presence of `lanes/ROTATION_PAUSE` to the hub's `lock/` directory;
- pulls `STATUS` and the last 3 `ALERT` lines into `lanes/ROTATION_STATUS`.

`lanes/fleet-rotation.coord-sync.patch` (lead action) makes `heartbeat.py` run `sync` and print the status line, and makes `window_lock.py` run `sync` right after taking the lock and right after releasing it. With the patch, a `lead-*` window blocks the rotation within one tick. Without it, pause by hand: `ssh rtx5090 python3 ~/fleet-rotation/bin/fleet_rotation.py --config ~/fleet-rotation/rotation.json pause <reason>`.

A lead window's own `fleet_window.sh hold/release` changes the fleet while the rotation is preempted. At the next unlocked tick the rotation adopts the fleet as it finds it, and then moves to the hour's target.

## Inactive models' APIs

Decision: **inactive ports stay closed.** The hub publishes availability at `http://rtx5090:8430/schedule.json` and `http://rtx5090:8430/v1/models`: every model with `active`, `port`, `next_slot_start`, `runnable`, `validated` and, if it cannot run, `not_runnable`. The responder is `fleet-rotation-schedule.service`; it reads `schedule.json`, which each tick rewrites.

Why not a 503 responder on each model port: every transition would then have to stop the responder before the real API could bind the same port, and restart it afterwards. That adds a step, and a failure mode ("port busy"), to every transition, exactly where the rotation must be most reliable. A client that finds a model's port closed reads `/v1/models` on :8430 to see when that model is next scheduled.

## Commands (on the hub; from the workstation prefix `ssh rtx5090`)

```
T="python3 ~/fleet-rotation/bin/fleet_rotation.py --config ~/fleet-rotation/rotation.json"
$T now                      # status, lock, the next 7 hours, last log lines
$T status                   # the one-line status
$T pause <reason>           # the timer keeps running; every tick holds off
$T resume                   # the next tick adopts the fleet as found, then follows the schedule
$T force <slot> [companion] # this hour runs <slot> (full|k3|flash_plus) [with that companion]
$T skip                     # the rest of this hour: production alone
$T schedule --hours 12      # predicted slots
$T --dry-run [--at 2026-09-29T15:00:30Z] tick   # print every command of the tick, run none, write nothing
$T converge <model...>      # manual converge with the same stop/reclaim/start/smoke steps
$T converge --rollback      # converge to rollback_models (production + Qwen)
$T tick                     # what the timer runs
```

`force`, `skip`, `pause` and `resume` only change the state; the next tick (within 5 min) acts. Run `tick` to act now; it takes the same state lock as the timer.

## Install and rollback (lead actions)

`tools/fleet_rotation_install.sh`, run from a checkout on the workstation:
- `check` (read-only): the hub is reachable, the hub reaches every node with BatchMode ssh from inside a `systemd-run --user` unit, port 8430 is free, and the config validates.
- `install`:
  - copies the tool and config to `rtx5090:~/fleet-rotation`, recording `INSTALLED.sha256`;
  - writes `ROTATION_PAUSE` ("waiting for the supervised first run");
  - writes `fleet-rotation.service` (oneshot `tick`, 100 min timeout), `fleet-rotation.timer` (`*:00/5:20`) and `fleet-rotation-schedule.service`;
  - runs `systemctl --user enable --now` on the timer and on the responder.
  
  The rotation is installed **paused**.
- `status`, `dry-run [UTC-TIME]`.
- `rollback`:
  1. disables the timer;
  2. waits for a running tick to finish (up to 60 min);
  3. disables the responder;
  4. writes `ROTATION_PAUSE`;
  5. runs `converge --rollback`, which puts back production + Qwen TP4 (the fleet before the rotation). Qwen's API is `qwen27b-api`.

## First supervised run (lead)

1. `bash tools/fleet_rotation_install.sh check`, then `install`. The rotation is now paused. The first tick of each hour records `phase=paused`, and `:8430/schedule.json` is up.
2. GLM Full prerequisite: the FULL slot demotes itself until `~/glmfull-lane6/root` carries release 8058e4b (`RELEASE`, `MANIFEST`) and `~/glmfull-lane6-api` carries the 8058e4b channel. Both are installed once by the lanes/glmfull-serve.md §3 window: `release_window.sh up` then `api`, verify, and `down reclaim`. The rotation starts and stops the lane itself after that; it never copies roots.
3. Low-risk first transition (production stays up), at any time in the hour:
   - `$T force flash_plus ling`, then `$T --dry-run tick` to read the plan, then `$T resume` and `$T tick`.
   - Watch `rotation.log`: Qwen stops and its lane-3 packs are reclaimed; Ling starts on 16 nodes, ready, `ling3-api` :8437, Paris smoke; floor >= 20 GiB.
   - During this hour, measure one Flash B1 through g53-api. A TP16 companion shares every GPU with production.
4. The first FULL hour, supervised:
   - at hh:55 run `$T --dry-run --at <hh+1>:00:30Z tick`;
   - at hh+1:00:20 the timer holds Flash, reclaims its packs, starts lane 6 (~3 min to ready), starts glmfull-api6 and runs the smoke;
   - expected outage ~5-8 min, and ~5 min again at the end of the hour when Flash comes back.
   - On any `ERROR`, the rotation has already fallen back. Read `ALERT` and `rotation.log`.
5. Leave it running. Apply the coord-sync patch so heartbeat shows the status line and lead windows preempt automatically.
6. Abort at any point: `$T pause`, or `fleet_rotation_install.sh rollback`.

## Driver inventory (2026-09-29 14:00Z)

Memory per node while serving. "Beside Flash" means MemAvailable stays >= 20 GiB with Flash (~38 GiB/node) also resident. Measured MemAvailable with Flash + Qwen: 68-83 GiB.

| model (config key) | nodes / lane | start / stop / API / smoke in config | GiB/node | beside Flash | status |
|---|---|---|---|---|---|
| GLM Flash (`flash`) | 16, lane 0, fleet-agent | `agent.hold` rm/touch; g53-api :8433; chat Paris | ~38 | - | production, fallback |
| GLM Full (`glmfull`) | 16, lane 6 | `sp-glmfull-rd6` (release env, pinned, graph + split); `glmfull-api6` :8446 from `~/glmfull-lane6-api`; chat Paris | ~68 pinned | no (own slot) | runnable once the release root is installed (prerequisite above) |
| Kimi K3 (`k3`) | 16 | - | ~93 (TP16 pack 99.6 GB) | no (own slot) | **not runnable**: k3-serve lane building (lanes/k3-serve.md) |
| Qwen3.8-27B TP4 (`qwen`) | spark0/1/2/5, lane 3 | `sp-qwen4-rd` no-spec from `~/sparkdata/qwen27b.mx2.tp4`; `qwen27b-api.service` :8435; completion Paris | ~15 | yes (53-55 left) | validated, default companion |
| Ling-3.0-flash TP16 (`ling`) | 16, lane 10 | `sp-ling-rd10` from `~/ling-lane10/root`; `ling3-api` :8437 from `~/ling-lane10-api`; completion Paris | ~22 | yes (46-61 left), but TP16 shares every GPU with Flash | companion |
| MiMo-V2.6-Flash TP4 (`mimo`) | spark4/6/7/8, lane 9 | `sp-mimo-rd9` from `~/mimo-lane9/root`; `mimo26-api` :8439 from `~/mimo-lane9-api`; completion Paris | ~46 | yes (28-35 left) | companion |
| Gemma 4 31B TP4 (`gemma`) | sparka-d, lane 4, mesh 10-13 | `gemma4_lane_resident.sh` release-11b1d2a, units `sp-g4l4-resident-r0..3`; `gemma4-api.service` :8436; Gemma-template completion Paris | ~36 | yes (44-46 left) | companion |
| Laguna-S 2.1 TP8xPP2 (`laguna`) | 16, lane 11 | - | ~20 | yes | **not runnable**: spark3 not restaged; start command incomplete in lanes/laguna-w6.md |
| DeepSeek V4.1-Flash (`dsv41`) | TP8 packs | - | - | - | **not runnable**: no module forward, adapter or firmware |
| DeepSeek V4-Pro (`dsv4pro`) | TP4xPP4 packs 16/16 | - | 94-100 | no | **not runnable**: no serving path |
| Hunyuan 4 (`hy4`) | TP16 packs | - | ~56 | at the floor | **not runnable**: module is a stub |
| MiniMax text, Qwen3.8 Max, Muse | packs only | - | - | - | **not runnable**: no serving lane recorded |

A driver joins the rotation when its entry gains `runnable: true` and the full set of fields: `nodes`, `mem_gib`, `engine.{precheck,start,stop,up,ready,grace_s,ready_timeout_s,stop_timeout_s}`, `packs` (the pack sha files for `--reclaim-pack`), `api.{precheck,start,stop,up,health,port,timeout_s}` and `smoke`. `check-config` enforces the list. Placeholders: `@rank@` (index in `nodes`), `@host@`, `@stamp@` (one per start, for log markers), `@model@`.
