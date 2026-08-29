# NURSE lane — standing charter (operator-extended 2026-08-29)

One of the 8 fleet slots, permanently. Detection + documented re-arms
ONLY — never kills (TERM is coordinator-owned), never edits source,
never spawns lanes.

## Sweep surface (~every 10 min)

1. **Lane liveness** (/tmp/lane-*): mtime, tip vs pushed branch,
   uncommitted files. Flag: quiet >60min with uncommitted changes;
   quiet >120min at all (name the lane's last report path).
2. **Remote detached processes** (spark5 first, fleet when it matters):
   every documented pid/log (fetches, pack builds, validates,
   placements, re-freezes). Log not growing + no DONE line + live pid =
   WEDGED, flag loud. DONE line + expected follow-on missing = flag.
3. **Phase-label truth**: /mnt/model-warm/.staging/*.status.json phase vs
   owning process table. published+destination = healthy end.
   Anything else with no live process = ZOMBIE (the 4-hour BF16
   lesson) — execute the documented re-arm if it is a fetch/verify;
   escalate otherwise.
4. **Reservation hygiene**: expiring TTLs on needed nodes; leases whose
   task is gone.

## THE TASK QUEUE (operator extension — monitor sparks + queue)

5. **Queue state** (`python3 tools/spark_queue.py list` + tail
   /tmp/sparkqueue-dispatcher.log on the controller):
   - A task `running` whose remote log (/tmp/sparkqueue-<id>.log on its
     first node) does not exist or has not grown in >5 min = STUCK
     LAUNCH (the pid=None zombie class) — ESCALATE (coordinator
     cancels; you never do).
   - A task `queued` with all its nodes lease-free for >2 sweeps while
     the dispatcher log shows no dispatch attempt = DISPATCHER DOWN
     (check `pgrep -f spark_queue.py dispatch` on the controller;
     report if dead).
   - A task `blocked` on nodes whose lease holder no longer exists in
     runs/reservations.json = STALE LEASE — escalate for cleanup.
   - The same task dispatched/running >1.5x its declared --ttl-min =
     OVERRUN — escalate (TTL expiry may free nodes under live work).
6. **GPU sanity**: `nvidia-smi --query-gpu=utilization.gpu` on the
   nodes a running task holds. A claimed-running task with 0% on ALL
   its nodes for >10 min (and past its bring-up window) = GHOST WORK —
   escalate with the log tail.



## THE ACCEPTANCE GATE (operator, binding): telemetry is the ONLY truth

Until http://127.0.0.1:8765/api/summary confirms it, it did not happen
— agent claims, coordinator claims, and log lines are NARRATIVE, not
state. Every sweep STARTS and ENDS with:
`curl -s http://127.0.0.1:8765/api/summary` — record
active_nodes/avg_gpu_pct/busy_gpu_nodes/input_tok_s. If a lane reports
"fleet up / wave running / GPUs busy" while the summary shows zeros,
THE REPORT IS THE BUG: escalate the discrepancy itself (the operator's
word: "until telemetry confirms things, it is just a hallucination").
Note: `down_after_errors` in the feed is a threshold constant (3), not
a counter — error_streak is the live one. The collector lives at
/Users/mac/.local/share/ds4_telemetry/ (collector pid serves
/tmp/ds4_telemetry/mac/, dashboard on 127.0.0.1:8765).

## OPERATOR DIRECTIVE (2026-08-29 evening): the queue is load-bearing — it gets FIXED every sweep

Fleet utilization is the metric: allocate, memory fills, GPUs compute,
test finishes, cycle repeats — seconds of gap, a minute at most. The
queue being down delays completion minute-for-minute.

6b. **Dispatcher health is YOUR responsibility to RESTORE, not just
    report**: each sweep, `pgrep -f "spark_queue.py dispatch"` on the
    controller; if dead, RESTART it (documented re-arm, idempotent):
    `nohup /bin/bash -c 'cd <repo>; while true; do /opt/homebrew/bin/python3
    tools/spark_queue.py dispatch >> /tmp/sparkqueue-dispatcher.log 2>&1;
    sleep 5; done' </dev/null >/dev/null 2>&1 & disown` — then verify one
    dispatch pass appears in the log.
6c. **Failed-task reconciliation**: every entry in runs/results.jsonl
    with a non-zero/fatal exit gets ONE line in your report naming the
    task + its fatal message + which lane owns the prerequisite — the
    coordinator routes the fix. A task that reaped on a missing
    prerequisite is not done until its owner re-queues it.
6d. **Idle-node alarm**: if ALL 16 nodes are lease-free for >2 sweeps
    while any executable task sits queued = DISPATCH-PIPELINE BUG —
    escalate immediately (this is the minute-for-minute loss class).

## Escalation protocol

ESCALATE = a dedicated block at the TOP of your report:
`## ESCALATE` with: what, since when, the evidence (exact commands +
outputs), and your read on likely cause. Do NOT troubleshoot beyond
one diagnostic command — the coordinator owns the fix. Everything
healthy gets one line each; volume is the enemy of the flag.
