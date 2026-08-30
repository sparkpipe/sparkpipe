# Model Dev's Guide — start every session here

You own ONE model. The fleet, its packs, and the task queue are shared
infrastructure run by the coordinator (PR-based). This guide is the
contract for getting your work done without stepping on anyone.

## What you can assume

- **Stagepacks are there.** Your model's packs are built, validated,
  and placed on the sparks (see docs/ROADMAP_TP16_FLEET.md state; if
  your model's set is missing/stale, file it with the coordinator —
  do NOT rebuild fleet packs yourself).
- **The spark task queue is debugged and is the ONLY GPU path.**
  `tools/spark_queue.py` — the dispatcher (5s loop) is the sole
  launcher of anything on the sparks. Never start a fleet/daemon by
  hand; never hold nodes while you think.

## Getting GPU work done

```bash
# a task = a real command; self-staged or fails loud naming the missing piece
python3 tools/spark_queue.py add --id <model>-<thing> \
  --nodes spark4,spark5,spark6,spark7 \
  --cmd 'bash /home/spark0/<model>_thing.sh' \
  --priority 5 --ttl-min 20          # declare duration; default is 15
```
- Priorities: 0 = glm5.3-family/coordinator; your model's routine work
  belongs at 5. FCFS within a class; 2h starvation auto-elevates.
- STAGED SCRIPTS, never inline quoting (ssh+case+$() mangles args):
  write the script, scp it to the node, and make the task run the file.
- Tasks must not end in `exit $rc` (it defeats the exit-file reap).
- Stage in home dirs, never /tmp (waves sweep /tmp).
- Cancel-orphan rule: if you cancel a task, TERM its remote processes
  yourself (cwd-filtered, see below).

## The acceptance gate: telemetry

Until the dashboard confirms it, it did not happen:
`curl http://127.0.0.1:8765/api/summary` — busy_nodes/avg_gpu/tok_s.
Check it IMMEDIATELY after any launch (loading shows in ~2s; do not
sleep-poll). Bursty short runs can hide from the 5s poll — use a
sustained generation for your receipt. Node-local `nvidia-smi` beside
the dashboard line is the standard proof pair.

## Generating your own stagepacks (operator directive: packs not yet
## done or in progress are YOURS to build)

- **Source law**: official or vetted community releases only, from
  warm ceph (/mnt/model-warm/...); NEVER quantize (see hard rules).
- **Tooling**: your family packer (tools/<family>_stagepack.py; the
  qwen38 one has --expert-codec nvfp4 for 4-bit sources). Stage-slice
  builds parallelize across sparks with zero read amplification —
  submit ONE queue task holding your nodes whose staged script fans
  out per-stage builders (the qmax_build2.sh shape: scp script, one
  ssh per node, layer-windowed --first-layer/--layer-count, outputs
  land directly on their nodes).
- **Deliverables = receipts**: per-stage .receipt.json, the family
  verifier green, and the two-pass placement proof ("already placed"
  on re-run). Drop caches before big loads; 110GiB ceiling law.
- The universal packer (docs/UNIVERSAL_PACKER.md) consolidates the
  per-family tools as codecs land in the core — build against the
  current family tool until its emitter ships, then switch.
- Coordinator-owned (do not duplicate): qwen-max nvfp4 (in flight),
  anything marked done in docs/ROADMAP_TP16_FLEET.md.

## PRs — the only way code lands

- **Your model's code** (modules/<your-model>/…): PR with your gates
  green. The coordinator merges by: offline gates BY EXIT CODE
  (`make offline-gates >log 2>&1; echo $?` — never grep output),
  ratchet re-measured exact, digests regenerate LAST.
- **Common code** (runtime/, node/, cache/, kernels/, tools/, the
  queue itself): PR with a justification of cross-model impact + which
  models you tested. The coordinator reviews before merge — expect
  questions, that's the point.
- **Perf claims** need a queued cell with the kill-switch discipline:
  exactness/bit-exactness verified BEFORE any timing; a mismatch is a
  RED stop, not a data point. Numbers recorded with context/batch/
  topology/precision.
- Every PR's report goes in docs/AGENT_LANE_BRIEFS/reports/ — receipts
  or it didn't happen.

## Hard rules (the graveyard that wrote them)

- **NEVER QUANTIZE WEIGHTS YOURSELF (operator policy, zero exceptions).**
  Packagers repackage (slice/codec-preserve/shard); they never change
  precision. If you want an FP8/4-bit arm of your model, the source
  must be an OFFICIAL or VETTED COMMUNITY release — hunt it, verify it,
  fetch it (the qwen-max-4bit/radixark pattern). Self-quantizing BF16
  →FP8/MXFP4 in a packer, script, or notebook = the PR is rejected.

- 110GiB device ceiling per node is law; GB10 page cache eats it —
  `sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` before
  big loads.
- TERM-only, by cwd-filter: `pgrep -f "bin/sparkpipe_model_[r]esidentd"`
  (bracket-trick) + check `/proc/<pid>/cwd` + exclude your own shell.
  pgrep -x matches nothing. Print-then-kill-all is NOT filtering.
- Never KILL -9. If a node wedges (nvidia-smi hangs), report it —
  power-cycles are the operator's.
- No model names in shared code (the dry-law gate is live).
- Fail fast: a task that can't succeed in its window gets cancelled,
  never left polling. Long processes: setsid+nohup, pid logged.
- The stale-phase-label lesson: a status file saying "downloading"
  proves nothing — the process table is the truth.

## Where things are

- Your model's geometry/truth: `model_contracts/<model>_authoritative.json`
- The fleet plan: `docs/ROADMAP_TP16_FLEET.md` · packer design:
  `docs/UNIVERSAL_PACKER.md` · weightd (weights stay resident between
  tests — use it): `docs/WEIGHTD_DESIGN.md`
- Known-good runbook shape: the q27b serve task (cache-drop in-cmd,
  exact-member batch schema, batch-while-alive ordering) in the
  coordinator log, 2026-08-30.
- Reports/receipts: docs/AGENT_LANE_BRIEFS/reports/ (yours named
  <model>-*.md). Queue state: runs/. Reservations map: runs/reservations.json.
