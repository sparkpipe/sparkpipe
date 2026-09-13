# Proposal: ring-window enforcement hook (consolidation 3.3)

Owner: SCHEDULER subsystem agent · Workspace: `.agents/scheduler` (clone of `unified`)
Follows `docs/PROPOSAL_ADMISSION_CORE.md:452-461` (§4.2). Proposal only — no commits,
no pushes, no shared/model edits. Cited `file:line`; verified at `HEAD ce93ed6`.

## 1. `SparkAdmissionPredicateFunction` semantics for the hourly rule
Reuses the type from `PROPOSAL_ADMISSION_CORE.md:130-132`:
`typedef SparkStatus (*SparkAdmissionPredicateFunction)(void *context, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);`
Mounted once at **`SparkOrchestratorSubmit`** entry (`src/spark_orchestrator.c:624-646`),
not per model or per endpoint — the rule is fleet-level. `request` is read only for `priority`; `context` is a `SparkRingWindowState*`.

**Inputs (state struct; model-neutral — passes `tests/test_dry_law.py:28-32`):**
```c
typedef struct SparkRingWindowState {
    char     holder_id[64];              /* registry id; "" = no holder        */
    uint64_t window_deadline_ns;         /* absolute slot end, CLOCK_MONOTONIC  */
    uint64_t artifact_counter;           /* monotonic durable-artifact count    */
    uint64_t artifact_counter_at_start;  /* snapshot taken when this hour began */
    uint32_t window_seconds;             /* 3600 (60-minute windows)            */
    uint32_t enforce;                    /* 0 = log-only, 1 = refuse            */
} SparkRingWindowState;
```
"current holder" = `holder_id`; "window start" = `window_deadline_ns - window_seconds*1e9`
(kept implicit so a late start ends on time, `COORDINATION.md:116`); "artifact counter" =
`artifact_counter` vs `artifact_counter_at_start`. The four durable-artifact kinds are the
rule's own (`COORDINATION.md:110-113`); each producer (branch hook, receipt script,
`PERFORMANCE_STATUS.md` writer, CI) bumps the counter idempotently (dedupe by kind + commit
sha / receipt path).

**State residence.** Authoritative copy at `/tmp/sparkpipe_ring_window_state.json` on
`spark0`, broadcast per-host exactly like the fleet state (`tools/fleet_swap.sh:17-18, 26, 29,
40-45`). Each rank's orchestrator reads its local projection once per submit; the predicate is
pure and side-effect-free, matching the contract that validation is side-effect free
(`spark_model_serving_adapter.h:298-300`).

**Refusal path.** Returns `SPARK_STATUS_OK` normally and expresses refusal in the
decision — the same idiom as `SparkStageModuleAdmissionDecisionReject`
(`runtime/stage_module_common.c:797-814`):
```c
SparkStatus SparkRingWindowPredicate(void *ctx, const SparkModelDriverAdmissionRequest *r,
                                     SparkModelDriverAdmissionDecision *d)
{
    SparkRingWindowState *w = ctx; uint64_t now = monotonic_ns();
    if (w->enforce == 0u) return SPARK_STATUS_OK;                 /* advisory */
    if (now < w->window_deadline_ns) return SPARK_STATUS_OK;      /* in window */
    if (w->artifact_counter > w->artifact_counter_at_start) {     /* progressed */
        w->artifact_counter_at_start = w->artifact_counter;       /* roll hour */
        w->window_deadline_ns += (uint64_t)w->window_seconds * 1000000000ull;
        return SPARK_STATUS_OK;
    }
    SparkModelDriverInitializeAdmissionDecision(d);               /* support.h:170-182 */
    d->accepted = 0u;                                             /* silent hour */
    d->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_REJECTED_DEADLINE;
    return SPARK_STATUS_OK;
}
```
Reusing `REJECTED_DEADLINE` (no new enum value) needs no ABI change and maps to
`SPARK_STATUS_BUSY` (`spark_model_driver_support.h:235-237`), so the submit path retries and
lands on the next holder after the swap. The orchestrator already short-circuits a non-accepting
endpoint (`src/spark_orchestrator.c:595-599`). A dedicated `WINDOW_EXPIRED` reason is an
optional append-only follow-up.

## 2. Composition with `tools/fleet_swap.sh` preemption

Order is **refuse → drain → swap → re-arm**:
1. **Refuse.** `enforce=1` + reject once the silent hour elapses — no new frames
   enter the holder's residentds.
2. **Drain.** In-flight work finishes under the existing quiesce idiom —
   topology-switch QUIESCE drains to a token boundary (`scheduler/topology_switch.c:350-354`);
   adapter quiesce closes admission before swap (`spark_model_serving_adapter.h:325-333`).
3. **Swap.** `tools/fleet_swap.sh <next>` evicts and starts the next
   (`fleet_swap.sh:81-116`), writing both fleet and ring-window state.
4. **Re-arm.** `fleet_swap.sh` broadcasts a fresh `SparkRingWindowState`
   (`window_deadline_ns = now + 3600s`, `artifact_counter_at_start = artifact_counter`)
   via `broadcast_state` (`fleet_swap.sh:40-45`).

**Idempotence.** `cmd_swap` no-ops when `current == model` (`fleet_swap.sh:88-91`);
the predicate is pure (same state → same decision, no side effects); the broadcast
is a plain overwrite; the artifact counter dedupes so a re-broadcast never
double-counts.

**Restore path.** Fleet-scope swap snapshots the running set and restores it on swap-out
(`fleet_swap.sh:93-103`); on restore the coordinator re-arms the holder (`window_deadline_ns =
now + 3600s`), so evicted time never counts against the continuation rule — the restored holder
gets a fresh hour. The old `holder_id`/counter are remembered in `running.prev_state`
(`fleet_swap.sh:97`).

## 3. Pinning tests (no hardware)

The predicate must leave these existing host tests green:
* `tests/test_dry_law.py` — predicate + `SparkRingWindowState` live under
  `src/`/`runtime/` and name no model (`COMMON` `13-27`, `MODEL_TOKEN`
  `28-32`); `holder_id` is a data string, not a source token.
* `tests/test_driver_compiler.c:138-139,183-193` — generated admit/merge untouched;
  the predicate mounts above it, never inside it.
* `tests/test_model_serving_adapter.c:463-503` — `SparkModelDriverAdmissionRequestIsValid`
  untouched; the predicate is not in the request-validation path.
* `tests/test_dsv4_stage_runner.c:36-40,220,230-232` — the runner still sees one
  `admit` call per submit; the predicate is orthogonal.
* `tests/test_model_resident_deadline.c:62,90-91` — per-route transport deadline
  stays distinct from the window lease (both use `CLOCK_MONOTONIC`,
  `node/model_residentd.c:1393-1399`); the hook does not touch residentd.
* `tests/test_topology_switch.c:324-363` — recipe-level admissions open/close unchanged.

New host-only tests (`tests/test_ring_window.c`, in the mock style of
`tests/test_topology_switch.c` / `tests/test_model_resident_deadline.c:62`):
(a) `now < deadline` → accept; (b) hour elapsed + counter advanced → accept and
window rolls exactly `+3600s`; (c) hour elapsed + no artifact → `REJECTED_DEADLINE`,
`accepted == 0`; (d) `enforce == 0` → always accept; (e) a late-start deadline is
honored as the absolute slot end (no `+3600s` from start). A companion script test
asserts `fleet_swap.sh` no-op idempotence (`fleet_swap.sh:88-91`) and the
state-file round-trip (`broadcast_state`) with a mocked `ssh`.

## Verification note

Cited `file:line` verified at `HEAD ce93ed6`. Net-new surface = one `SparkRingWindowState`
struct + one predicate + a state-file convention; reuses `SparkAdmissionPredicateFunction`,
`SparkModelDriverAdmissionDecision`, `SparkModelDriverInitializeAdmissionDecision`,
`REJECTED_DEADLINE` — no new wire types, no ABI change.
