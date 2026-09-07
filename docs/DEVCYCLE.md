# Dev cycle

Edit on your machine, build once on any spark, the fleet converges. The
repo is the only sync mechanism: no scp'd trees, no hand-maintained build
directories. Any node can recreate the same source tree from a branch.

## The loop

    1. edit    your lane worktree (or any clone)
    2. push    git push origin <branch>
    3. build   ssh <any-spark> 'cd ~/sparkpipe-build && git pull && \
                bash tools/module_build_release.sh <family> <codec> <root> \
                    <revision> <contract.json> [branch]'
       e.g. bash tools/module_build_release.sh glm5_next_resident_decode_stage \
                fp8 glm53flash.fp8.tp16 84c6a6aa9497188e15a635ba793b0f95a79b1033 \
                model_contracts/glm53_flash_authoritative.json
    4. wait    the UPDATE cycle converges the fleet (poll, don't sleep)
    5. test    API request against spark0:8433

The build script is the whole step 3: it resets the tree to the branch,
builds host binaries, publishes the module (GPU receipts on the local
GB10 — it parks the local residentd for the validator and the UPDATE
cycle brings it back), compiles the model driver, installs it into the
hub reference, and drops the UPDATE file.

Build time ~2-4 min (nvcc dominates). Fleet cycle ~35-90s. Clean nodes
report `model_residentd ready` 16/16 about 12s after launch.

## What runs where

- **Your machine**: edits and pushes. Nothing else.
- **Any spark** (`~/sparkpipe-build`): the build. aarch64 + GB10 — the
  driver `.so` and the validator need them; an x86 host cannot link the
  driver and its GPU cannot run sm_121a kernels.
- **Hub** (`rtx5090:release`): the reference tree. Build outputs land
  here and nowhere else.
- **Every spark**: one agent (systemd user unit `fleet-agent`). Idle, it
  does one ssh existence check per cycle against the hub. When
  `release/<root>/UPDATE` exists, it pulls, then runs the two-phase
  restart: append `down:<host>` after the local daemon exits; when all
  16 are down, start; append `up:<host>`; rename to `UPDATE.<n>` when
  all 16 are up. The two phases exist so a fresh sender never wires
  against a draining peer's listener — that deadlock has no timeout
  that saves you.

## Testing against the running fleet

Runtime never checks build lineage. Codec, runtime limits, ABI, and
geometry are checked; model/revision/sha are data, not gates. The
orchestrator prints `mixed_versions` and keeps serving when endpoints
disagree on build shas — so a fleet may run mixed drivers while you
test.

Current caveat: restarting ONE residentd in place cannot rejoin a wired
mesh (the transport's sender side does not re-connect after a stall —
known, reported). Until that lands, mixed-driver testing means the
usual UPDATE cycle with one node's tree pointing at your build; a hot
solo swap parks the node instead.

## Receipts: qualify once, then load

`make publish` runs the GPU validator once per new artifact and records
the receipt (artifact sha + validator sha) in the module library.
Unchanged artifacts reuse the record — nothing re-runs. glm5_next and
the qwen/glm52 modules synthesize their own weights; dsv4's validator
loads a real pack (its Makefile names it). No pack requirement exists
anywhere else.

## Debug-cycle rules

- **Poll, never sleep.** Wait for the condition (ready line, UPDATE
  rename, process exit) with a short interval. A fixed sleep is 10-100x
  the event and hides regressions in cycle time.
- **Fail fast.** Wiring timeout is 10s; the error names route, role,
  host, and port. If it hasn't connected in 10s it will not connect in
  a year — go read the error.
- **Count processes by `/proc/<pid>/exe`**, not `pgrep -f`: your ssh
  wrapper matches your own pattern.
- One residentd per runtime root, ever. The agent's `start_root`
  refuses if one is running.
- Never kill `sparkpipe_weightd` — it holds every model's resident
  packs; agents start it only if absent.
- Don't touch networking, ssh config, or keys. Report; the operator
  owns that plane.

## Handy commands

    # fleet view (what every spark is running)
    ssh rtx5090 'cat release/../current/<host>.json'   # hub:current/

    # one node's daemon log
    ssh spark3 'tail -5 ~/sparkdata/glm53flash.fp8.tp16/residentd.log'

    # ready poll (all 16)
    for h in spark{0..9} spark{a..f}; do
        ssh $h "tail -1 ~/sparkdata/glm53flash.fp8.tp16/residentd.log" |
        grep -c 'model_residentd ready'; done | grep -c 1

    # request (needs token ids; no tokenizer wired yet)
    curl -s http://spark0:8433/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"prompt_token_ids":[151644,872,198,1131,151645,198],
             "max_tokens":8,"temperature":0.0}'

    # force a fleet restart without a new build
    ssh rtx5090 'touch release/glm53flash.fp8.tp16/UPDATE'
