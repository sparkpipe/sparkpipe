# Two-phase fleet startup protocol (operator design, 2026-08-30)

THE PROBLEM (diagnosed live): partial-restart mesh-flap — each rank
restart invalidates peers' transport sessions; peers re-open into
ranks mid-restart; cascading 180s open-timeouts. The heavy startup
(transport + CUDA + module + pack) RACES before membership exists.

## Phase 1 — REGISTRATION (lightweight, no CUDA, no GPU alloc)

A tiny registrar (separate binary, pure POSIX TCP — starts in
milliseconds, links nothing heavy) per node:
- CONFIG: the deployment's launch table IS the expected membership
  (rank, host, registration port). Known N — no discovery ambiguity.
- ANNOUNCE (every ~250ms): unicast to every expected member's
  registration port: {my rank, my host, my VIEW: members I have
  heard announce}.
- MERGE: heard announcements update my view + each member's last
  broadcast view.
- TWO-LEVEL READY CONDITION (the operator's 'all see all'):
  (a) my view == the expected set, AND
  (b) every expected member's LATEST announced view == the expected
      set (they also see everyone).
  Both levels computed locally; GO is deterministic when all N nodes
  independently satisfy it. Simple with known N — no consensus
  machinery needed; the two-level check IS the agreement.
- GO: rank 0's registrar broadcasts GO on ready (leader = lowest rank
  in the expected set; deterministic). All registrars relay GO once
  and exit 0.
- FAIL LOUD: after a timeout (config, default 120s), every registrar
  prints the actionable diff — MISSING: [ranks], PARTIAL-VIEW:
  rank->who-they-see — and exits nonzero. No silent hang.
- MAINTENANCE SUBSETS: --expect-subset r1,r2,... restarts converge on
  the subset (the registrar's expected set comes from flags/config).
  Default: the full table.

## Phase 2 — the existing startup, gated

The wave tools (and LAUNCH-STATE.md's canonical pattern) become:
  1. launch registrars on all N nodes (simultaneous, trivial — they're
     featherweight).
  2. WAIT for GO (or the loud failure).
  3. ONLY THEN fire the heavy residentd wave (TERM-by-cwd + settle +
     simultaneous fire — unchanged).
The heavy startup still races (it's a race by design — sessions open
fast when everyone's up); the difference is the race now has a
starting gun instead of a cloud of dust.

## Future (not now): the registrar can PERSIST as a membership
watchdog during serving — membership loss fences new submissions
before they hang in transport waits. Design only when needed.

## Acceptance
A: 16-node cold bring-up: GO in <10s after the last registrar starts;
   zero open-timeouts in the residentd logs.
B: The flap reproducer (kill 8 ranks, restart them as a sub-wave):
   registrars hold until the subset's two-level condition, no cascade.
C: Missing-node case: clean FAIL LOUD naming the missing rank.
