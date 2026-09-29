# Fleet rotation

The owner's ruling of 2026-09-29: GLM Full and Kimi K3 each get dedicated fleet time, every third hour. The third hour runs GLM Flash production plus other drivers that rotate, so that every driver does at least slow inference. Owner, 19:40Z the same day: "if a third or fourth model fits, might as well give them runtime too". So each hour runs its slot's primary plus as many companions as fit in memory on their nodes (see "Packing"). Four GLM Flash TP4 replicas were rejected because they need 4x the KV cache, so Flash stays TP16 with 1/16 KV per rank.

`tools/fleet_rotation.py` implements the rotation on the rtx5090 hub as a systemd `--user` timer. It has no model knowledge of its own: every model, command, port, pack and memory figure lives in `deployment/fleet_rotation/rotation.json`. `tests/test_fleet_rotation.py` checks the state machine against a fake fleet driven by a fake ssh.

## Schedule

The cycle is three UTC hours, anchored at `cycle.anchor_hour` (0). Hour `h` runs slot `cycle.slots[(h - anchor) % 3]`:

| hour mod 3 | slot | primary | companion pool (config `slots.<slot>.companions`) | if its primary cannot run |
|---|---|---|---|---|
| 0 | `full` | GLM-5.3 Full (lane 6, pinned fp8, ~68 GiB/node, API rtx5090:8446) | the TP4 drivers (Qwen, Gemma, MiMo). No TP16 companion, so the GLM Full hour never shares every GPU. With `node_free_gib` 106 only Qwen fits beside it | `flash_plus` |
| 1 | `k3` | Kimi K3 (TP16 lane 8, full residency ~96 GiB/node, own floor 8 GiB, abort 6 GiB, API rtx5090:8448) | none: K3 owns the fleet | `flash_plus` (when K3's precheck or memory prediction fails; lead experiment windows take this hour through the lead lock) |
| 2 | `flash_plus` | GLM Flash production (g53-api :8433) | every other driver, runnable or not (the unrunnable ones are skipped) | none; production is the floor |

`fleet_rotation.py schedule` prints the next hours, and `plan` prints this hour's decision from the live fleet. With today's config and the GLM Full release root not yet installed, `full` demotes itself to `flash_plus` at each FULL hour (see "Demotion" below).

## Packing

The companions of an hour are chosen once, at the first tick of the hour, from live memory:

1. **Room per node.** `base = min(MemAvailable + mem_gib of every model resident on the node, node_free_gib)` is the node's free memory with nothing resident; `node_free_gib` (106) is the measured all-stopped MemAvailable of the lowest node, so a `mem_gib` that overstates a model's real use never inflates the room. `room = base - floor_gib - mem_gib(primary)` on the primary's nodes, `base - floor_gib` elsewhere.
2. **Candidates.** The slot pool's runnable entries, minus those skipped this hour, minus those whose `engine.precheck` or `api.precheck` fails (skipped this hour with a `WARN`).
3. **Fairness.** Candidates are ranked by the hour they last ran (never first), then by pool order. The first candidate that fits alone is the anchor: it always runs.
4. **Fill.** Among the sets that contain the anchor and fit on every node (for each node, the sum of the set's `mem_gib` on that node <= room), take the largest; among equally large sets, the one with the highest-ranked members. Node-disjoint TP4 companions therefore run together, and a companion that spans every node (Ling) joins when it fits or, when it does not, becomes the anchor of a later hour once it is the longest-waiting.
5. The decision is recorded in `state.json` (`picks.<hour>`: primary, companions, and for each waiting candidate the nodes that are short) and logged as `PACK`. Each chosen companion's `last_run` becomes that hour.

With the config figures (Flash 38, Qwen 15, Gemma 36, MiMo 46, Ling 22 GiB/node; room beside Flash 106 - 20 - 38 = 48 GiB), FLASH+1 hours alternate between Flash + Qwen + Gemma + MiMo and Flash + Ling + Qwen; the FULL hour runs GLM Full + Qwen (room 106 - 20 - 68 = 18 GiB: Qwen needs 15, Gemma 36). Measured today, Gemma used ~21 GiB/node and Flash ~24 (MemAvailable 82-85 with Flash alone); the config keeps the lane figures, and the live gates read real memory.

**Starting and stopping.** The primary starts first; the companions then start one at a time, each through its full start: a live MemAvailable gate on its nodes (`mem_gib` + floor, read just before the start), engine start, ready, API, health, smoke, and a MemAvailable check on its nodes (>= floor) right after it serves. A companion that fails any of these is stopped and reclaimed by pack, skipped for the rest of the hour (`dropped.<hour>`), and alerted as `ERROR`; the primary and the other companions are not touched, and the next companion still starts. A companion refused by the gate was never started, so nothing is stopped. Stopping runs in the reverse order: companions (last started first), then the primary, then production last.

**The next companion hour** considers a skipped companion again. A companion that cannot be stopped after a failure is `CRITICAL`: the rotation auto-pauses and leaves the primary serving.

## Each tick

The timer runs `tick` every 5 minutes at :20 s. A slot changes at the top of the hour, so the first tick of an hour performs the transition.

1. **Lock.** If `~/fleet-rotation/ROTATION_PAUSE` exists (manual or automatic pause), or the mirrored `lock/ROTATION_PAUSE` exists, or the mirrored `lock/PERF_HOLDER` starts with `lead-`, the tick touches nothing. It records `phase=paused|paused-coord|preempted` and marks the state for resync.
   The same lock is checked again at every step of a transition or fallback: before each stop and start, and on every poll while waiting for stop, ready or API health. A pause or `lead-*` lock that appears mid-transition stops the rotation at that step. It leaves the fleet as it is, writes a `WARN` alert naming the unfinished transition, and records the lock phase; when the lock clears, the next tick adopts the fleet as found (and restores production first if nothing serves). A manual `converge` (and so `rollback`) ignores the lock.
2. **Recover.** A `state.json` still in `phase=transition` or `phase=recovering` means the previous tick died mid-step (unit timeout, hub reboot, a kill, an unexpected error). The tick writes an `ERROR` alert and runs the fallback below, so a hold of production placed by the rotation is never left behind.
3. **Observe.** One ssh per node runs every runnable model's `engine.up` check, and the hub runs every `api.up`. A model is `up` (all nodes and its API), `down` (nothing), `api` (engine up on every node, API down) or `mixed`.
   **API repair.** A model in `api` state that the rotation recorded as serving (not production) gets its API restarted, health-checked and smoked with its engines kept (`API-RESTART`, `API-RESTORED`). This covers a lead window that stopped the API and did not restart it. If that fails (`WARN`), or the model was not recorded as serving, it is handled as `mixed`.
4. **Adopt or compare.** On the first tick, and after any lock or pause, the observed fleet is adopted as the current state; `mixed` models get a full stop before anything starts, except production: a `mixed` production (its API down, or a rank restarting) that the slot keeps is never held or reclaimed; its idempotent start (unhold, ready on every node, API start, smoke) repairs it, and the plan does not count its memory twice. Otherwise the observed fleet must match the recorded one:
   - a model that is up but not recorded means somebody changed the fleet outside the rotation: alert and auto-pause;
   - a recorded companion that died is stopped, reclaimed and skipped for the hour; the primary and the other companions keep serving. A slot primary other than production that died means fall back to production;
   - recorded production that is not up means alert and no transition until it is back. Production belongs to fleet-agent and the lead.
5. **Steady.** If the fleet already matches the target, the APIs' `/health` and every node's MemAvailable are checked. A companion whose health fails is stopped and skipped for the hour; a failing slot primary other than production falls back to production. A slot primary with `abort_gib` that has a node below it falls back to production. A node below the floor of the serving set (`floor_gib`, or an exclusive primary's own floor) sheds one companion per tick, the last started on that node (stopped, reclaimed by pack, skipped for the hour); with no companion on the low node the primary is left serving and a `WARN` is written.
6. **Reclaim idle arenas.** Before planning, every runnable model that is `down` on all its nodes gets `--reclaim-pack` of its own packs (`RECLAIM-IDLE`). Arenas left resident by an engine stopped outside the rotation (a lead window, a crash) are freed before memory is measured; a model with nothing resident frees nothing, and weightd refuses arenas still attached. A failed idle reclaim is a `WARN`; the start gates still read live memory.
   **Plan** (read-only, before anything is stopped):
   - run each incoming model's `engine.precheck` on its nodes and `api.precheck` on the hub;
   - predict each node's MemAvailable after the swap: now + outgoing `mem_gib` − incoming `mem_gib`. It must stay at or above the target's floor: `floor_gib` (20), or the lower `floor_gib` of an exclusive primary (see "Exclusive floor" below).
   - **Demotion:** a slot base that fails either check demotes the slot for this hour (FULL → FLASH+1). Production is never held for a model that cannot start.
   - **Packing** of the companions (above), the first time in the hour.
7. **Transition:**
   1. Floor check on every node (>= the floor of the outgoing set: 20 GiB, or an exclusive primary's own floor).
   2. Stop the outgoing models in the reverse of their start order: companions first, then a slot primary, production last. For each: stop its API, stop its engine units (for production, `agent.hold` on 16/16), and wait until `engine.up` is false everywhere (`stop_timeout_s`).
   3. Free memory by pack only: `weightd_warm <socket> --reclaim-pack <sha>` for each of the model's own packs (`packs`, sha files). It is refused on a node where the engine is still up. The config validator rejects any node-global `--reclaim` anywhere in the file.
   4. Start the incoming models: the primary first (production first when it is incoming), then the companions one at a time (see "Packing": a failing companion is stopped and skipped, the transition goes on). For each: a memory gate on the nodes where it is not already up (`mem_gib` + the model's floor), `engine.start`, then wait for `engine.ready` on every node (`ready_timeout_s`). After `grace_s`, an engine whose unit has exited fails the step early. A model with `abort_gib` fails the step when any of its nodes drops below it while it warms. Then start the API, wait for `api.health` (`api.timeout_s`), and run the smoke request (`smoke.body` to `smoke.path`, see "Smoke checks").
   5. Floor check on every node again, at the floor of the incoming set.
8. **Failure** at any step (timeout, non-zero rc, ssh timeout, smoke mismatch, floor, or any unexpected error in the tool) triggers:
   - an `ERROR` line in `ALERT`;
   - **fallback:** record `phase=recovering` first (so a fallback that is itself interrupted is redone by the next tick), then stop every non-production model (API, units, reclaim-pack), reclaim the packs of every model that is down (as in step 6), then start production (idempotent when it is already up: unhold, ready, g53-api, Paris smoke). The failed hour stays on production alone; the same slot is not retried within the hour. The failed hour is the hour the tick started in, even when the failure comes after the next hour began; the whole tick plans against the hour it started in.
   A model that cannot be stopped does not stop the fallback: production is still started (its live memory gate decides), then the rotation goes `degraded` with production serving.
   - If the fallback fails: `phase=degraded`, a `CRITICAL` alert, and an automatic `ROTATION_PAUSE`. The lead takes over. The rotation only auto-pauses with nothing serving after the idle reclaim and a production start were tried.

**Serving gaps.**
- A FULL or K3 hour starts with one transition during which nothing serves: production is held so the whole fleet is free.
- A failure costs one more transition, back to production.
- When a tick finds nothing serving (after a lead window, a degraded state, or a first run with the fleet held), it restores production before any slot model. The fleet is never left without a model for more than one transition.

Every command runs with a timeout (`ssh_timeout_s` per ssh call, plus the per-step timeouts). The tool never touches weightd binaries, the Flash release root or `~/release`. For production it only touches `agent.hold`, the `--reclaim-pack` of its own rank pack, and `systemctl --user start|stop g53-api`.

## Exclusive floor

A model that owns the whole fleet in its own slot may carry `floor_gib` and `abort_gib` (both, `0 < abort_gib < floor_gib <= floor_gib` of the config). Nothing shares the nodes with it, so there is no co-tenant for the global 20 GiB floor to protect. The validator accepts the pair only on a model that spans the whole fleet, is the base of slots whose `companions` pool is empty, and is never the fallback, in any slot's `companions` pool or a rollback model. With no companion to pack, the room of its hour is `base - floor_gib(model) - mem_gib(model)`, and the steady floor guard compares against the model's own floor (there is nothing to shed; the abort level falls back). The lower floor applies to the plan's prediction, the start gate and the floor check after the transition; the floor before the next transition out of that slot is the same lower floor, and the fallback's own floor check is the global one. While the model warms and at every steady tick the rotation reads MemAvailable on its nodes; a node below `abort_gib` fails the transition or triggers the fallback to production.

Lead decision 2026-09-29: Kimi K3 full residency (~95.3 GiB/node on a 104-109 GiB all-stopped baseline) runs with floor 8 GiB and abort 6 GiB in its hour.

## Files on the hub (`~/fleet-rotation`)

| file | content |
|---|---|
| `state.json` | active models, phase, slot, per-hour picks (`picks`), companions skipped per hour (`dropped`), `last_run` per companion, the last measured all-stopped estimate per node (`base_gib`), demotions, override/skip, last alert |
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

Decision: **inactive ports stay closed.** The hub publishes availability at `http://rtx5090:8430/schedule.json` and `http://rtx5090:8430/v1/models`:
- `serving`: every active model with `id`, `role` (`fallback`, `primary` or `companion`) and `port`;
- `slots`: the next 12 hours, each with `slot`, `primary`, `companions` and `models` (primary first). Future hours are predicted from the last measured memory and the fairness state, so a later hour's companions can still change when that hour is planned;
- `models` (also the `data` of `/v1/models`): every model with `role`, `active`, `port`, `next_slot_start`, `scheduled_starts` (every predicted hour it runs in), `runnable`, `validated` and, if it cannot run, `not_runnable`. The responder is `fleet-rotation-schedule.service`; it reads `schedule.json`, which each tick rewrites.

Why not a 503 responder on each model port: every transition would then have to stop the responder before the real API could bind the same port, and restart it afterwards. That adds a step, and a failure mode ("port busy"), to every transition, exactly where the rotation must be most reliable. A client that finds a model's port closed reads `/v1/models` on :8430 to see when that model is next scheduled.

## Commands (on the hub; from the workstation prefix `ssh rtx5090`)

```
T="python3 ~/fleet-rotation/bin/fleet_rotation.py --config ~/fleet-rotation/rotation.json"
$T now                      # status, lock, the next 7 hours, last log lines
$T status                   # the one-line status
$T pause <reason>           # the timer keeps running; every tick holds off
$T resume                   # the next tick adopts the fleet as found, then follows the schedule
$T plan                     # read-only: observe, MemAvailable, prechecks, and this hour's packing; starts and stops nothing, writes nothing
$T force <slot> [companion...] # this hour runs <slot> (full|k3|flash_plus) [packing only these companions]
$T skip                     # the rest of this hour: production alone
$T schedule --hours 12      # predicted slots
$T --dry-run [--at 2026-09-29T15:00:30Z] tick   # print every command of the tick, run none, write nothing; while paused or preempted it reports the lock and plans as if resumed
$T converge <model...>      # manual converge with the same stop/reclaim/start/smoke steps
$T converge --rollback      # converge to rollback_models (production + Qwen)
$T tick                     # what the timer runs
```

`force`, `skip`, `pause` and `resume` only change the state; the next tick (within 5 min) acts. Run `tick` to act now; it takes the same state lock as the timer. `pause` does not take the state lock: it succeeds while a tick is running, and that tick stops at its next step (see Lock above). `force`, `skip`, `resume` and `converge` are refused while a tick holds the lock.

A dry-run has no view of the fleet: it plans from the recorded `active` list, or from production alone when nothing is recorded (a fresh install). On the first supervised run the real tick also stops Qwen, which that dry-run does not show.

## Smoke checks

`smoke.body` goes to `smoke.path` on the model's API. The reply passes when:
- `smoke.expect` (a string), if set, occurs in the raw reply. Flash, GLM Full, Gemma (chat through its declared `chat_template`) and Qwen (a completion in its own chat format) keep `Paris`;
- `smoke.reply` (rules), if set, holds for `choices[0]` (`text`, or `message.content` for chat): `finish_reason` is one of the listed values; the text has at least `min_words` words and `min_distinct_words` distinct words (letters only). A reply that contains `look_for` passes the word rules at once. `SMOKE-PASS` logs the reply and whether `look_for` was found.

MiMo and Ling have no `chat_template` in their API channels (a `messages` request is refused), and a raw "The capital of France is" continues freely: MiMo answered " a city of romance, art, and" (`finish_reason` length) at 17:02Z, which failed the old `Paris` expectation and cost the hour. Both now use `reply` with `finish_reason` stop or length, at least 3 words of which 3 distinct, and `look_for` Paris; an empty, punctuation-only or single-token-loop reply still fails.

## Install, upgrade and rollback (lead actions)

`tools/fleet_rotation_install.sh`, run from a checkout on the workstation:
- `check` (read-only): the hub is reachable, the hub reaches every node with BatchMode ssh from inside a `systemd-run --user` unit, port 8430 is free, and the config validates.
- `install`:
  - copies the tool and config to `rtx5090:~/fleet-rotation`, recording `INSTALLED.sha256`;
  - writes `ROTATION_PAUSE` ("waiting for the supervised first run");
  - writes `fleet-rotation.service` (oneshot `tick`, 100 min timeout), `fleet-rotation.timer` (`*:00/5:20`) and `fleet-rotation-schedule.service`;
  - runs `systemctl --user enable --now` on the timer and on the responder.
  
  The rotation is installed **paused**.
- `upgrade [--no-resume]` (the rotation is installed and running):
  1. `check`;
  2. pause, unless it is already paused (then it stays paused afterwards);
  3. wait for a running tick (as `rollback` does);
  4. back up `bin/fleet_rotation.py`, `rotation.json`, `INSTALLED.sha256` and `state.json` to `~/fleet-rotation/backup-<UTC>`;
  5. copy the new tool and config as `.new`, run the new tool's `check-config` on the new config, then move them in (a failure leaves the old files; a failing `check-config` after the move restores the backup);
  6. `plan` (read-only) and `--dry-run tick`, printed for the lead;
  7. restart the schedule responder;
  8. resume if step 2 paused.
- `revert [BACKUP]`: the same steps, putting back the tool and config of the newest `backup-*` (or the named one); the files it replaces go to `revert-<UTC>`. `state.json` is kept: the older tool ignores the packing keys and converges the recorded companions to its own single companion at its next tick.
- `status`, `plan`, `dry-run [UTC-TIME]`.
- `rollback`:
  1. writes `ROTATION_PAUSE`, so a running tick stops at its next step;
  2. disables the timer;
  3. waits for a running tick to exit: it polls the unit's `ActiveState`, because `systemctl is-active` reports a running oneshot (`activating`) as not active (up to 105 min, above the unit's 100 min `TimeoutStartSec`);
  4. disables the responder;
  5. runs `converge --rollback`, which puts back production + Qwen TP4 (the fleet before the rotation). Qwen's API is `qwen27b-api`.

## First supervised run (lead)

1. `bash tools/fleet_rotation_install.sh check`, then `install`. The rotation is now paused. The first tick of each hour records `phase=paused`, and `:8430/schedule.json` is up.
2. GLM Full prerequisite: the FULL slot demotes itself until `~/glmfull-lane6/root` carries the release named in the `glmfull` precheck (`RELEASE` source_commit, `MANIFEST`) and `~/glmfull-lane6-api` carries the channel of the same commit (`SOURCE_COMMIT`). The config pins release dfa12a5 (main dfa12a5da, with the glm52 DSA indexer fix). The rotation starts and stops the lane itself; it never copies roots.
   - Changing the release is a swap outside a FULL hour: pause the rotation, wait for any running tick, check that `sp-glmfull-rd6` and `glmfull-api6` are inactive, move the root and API dir aside, copy the staged ones in, install the config whose precheck names the new commit, run that precheck, resume. A root or API dir that does not match the pinned commit demotes the FULL hour to FLASH+1; it never starts the wrong release.
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
| Kimi K3 (`k3`) | 16, lane 8 | `sp-k3-rd8` from `~/k3-lane8/root` (TP16, 1 sequence, host collective, pool 90,596,966,400 B = every expert chunk, state budget 1.5 GiB; the start drops the page cache of every pack under `~/sparkdata`) + `sp-k3-cache8` (`tools/pack_cache_trim.py` every 20 s on the rank pack); `k3-api8` :8448 from `~/k3-lane8-api`; chat Paris | 96, floor 8, abort 6 | no (own slot) | runnable once both roots are staged (lanes/k3-window.md); `validated: false` |
| Qwen3.8-27B TP4 (`qwen`) | spark0/1/2/5, lane 3 | `sp-qwen4-rd` no-spec from `~/sparkdata/qwen27b.mx2.tp4`; `qwen27b-api.service` :8435; completion Paris | ~15 | yes (53-55 left) | validated, companion |
| Ling-3.0-flash TP16 (`ling`) | 16, lane 10 | `sp-ling-rd10` from `~/ling-lane10/root`; `ling3-api` :8437 from `~/ling-lane10-api`; completion, reply rules (no chat_template) | ~22 | yes (46-61 left), but TP16 shares every GPU with Flash | companion |
| MiMo-V2.6-Flash TP4 (`mimo`) | spark4/6/7/8, lane 9 | `sp-mimo-rd9` from `~/mimo-lane9/root`; `mimo26-api` :8439 from `~/mimo-lane9-api`; completion, reply rules (no chat_template) | ~46 | yes (28-35 left) | companion |
| Gemma 4 31B TP4 (`gemma`) | sparka-d, lane 4, mesh 10-13 | `gemma4_lane_resident.sh` release-11b1d2a, units `sp-g4l4-resident-r0..3`; `gemma4-api.service` :8436 from `~/gemma4-api-channel` (declares the gemma4 `chat_template`); chat Paris | ~36 | yes (44-46 left) | companion |
| Laguna-S 2.1 TP8xPP2 (`laguna`) | 16, lane 11 | - | ~20 | yes | **not runnable**: spark3 not restaged; start command incomplete in lanes/laguna-w6.md |
| DeepSeek V4.1-Flash (`dsv41`) | TP8 packs | - | - | - | **not runnable**: no module forward, adapter or firmware |
| DeepSeek V4-Pro (`dsv4pro`) | TP4xPP4 packs 16/16 | - | 94-100 | no | **not runnable**: no serving path |
| Hunyuan 4 (`hy4`) | TP16 packs | - | ~56 | at the floor | **not runnable**: module is a stub |
| MiniMax text, Qwen3.8 Max, Muse | packs only | - | - | - | **not runnable**: no serving lane recorded |

Page cache: an engine whose weightd reads its pack through the page cache leaves up to tens of GiB of that pack cached. With little MemFree, CUDA context creation then fails (`NVRM ... kgrctxAllocMainCtxBuffer NV_ERR_NO_MEMORY`) although MemAvailable is high. `tools/pack_cache_trim.py` drops the page cache of named pack files only (never `drop_caches`); K3's start, stop and its `sp-k3-cache8` unit use it.

A driver joins the rotation when it is listed in a slot's `companions` and its entry gains `runnable: true` and the full set of fields: `nodes`, `mem_gib`, `engine.{precheck,start,stop,up,ready,grace_s,ready_timeout_s,stop_timeout_s}`, `packs` (the pack sha files for `--reclaim-pack`), `api.{precheck,start,stop,up,health,port,timeout_s}` (ports unique among runnable models, since companions serve side by side) and `smoke` (`path`, `body`, `timeout_s`, and `expect` or `reply`). `check-config` enforces the list. Placeholders: `@rank@` (index in `nodes`), `@host@`, `@stamp@` (one per start, for log markers), `@model@`.
