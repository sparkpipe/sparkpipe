# Registrar lane — fleet startup protocol phases 1+1b (2026-08-30)

Lane: `lane/registrar`, worktree `/tmp/lane-registrar` (based on `main`
at 6360b5d). Spec: `docs/FLEET_STARTUP_PROTOCOL.md` (ef2a52f + the
operator's phase 1b update 7b715f9). Registrar staging sha256
`a637985536c3e891…` on all 16 nodes.

Commits: `c24a6bb` (R1+R2), `8b626ea` (evidence timestamps + node-run
test fixes), settle-75s + TERM sweep commit, `8273e58` (gate wiring,
ratchet, manifest).

## Shipped

1. **`tools/sparkpipe_registrar.c`** (797 lines) + Makefile target
   `build/sparkpipe_registrar` (in TOOL_NAMES, builds in `make all`).
   Pure POSIX TCP, poll-driven, no threads, links NOTHING (libc only) —
   starts in milliseconds, deployable as one binary per node. Flags:
   `--rank --hosts --port-base --timeout-ms --announce-ms --term-wait-ms
   --deployment-cwd --daemon-name --expect-subset`.
   - ANNOUNCE ~250ms TCP unicast to every expected member's registration
     port (`SPGA <rank> <viewmask-hex> <stale_pid> <immune>`); the reply
     carries the peer's state, so one round trip merges both directions.
   - MERGE: my view, every member's latest announced view, every
     member's stale-daemon report.
   - THREE-LEVEL READY (two-level view agreement + phase 1b cleanslate):
     (a) my view == expected, (b) every expected member's latest view ==
     expected, (c) every member reports stale_daemon == 0 — self-check
     first: a registrar on a node whose old daemon lingers withholds GO.
   - CLEANSLATE remediation is TERM only, by exact exe basename + exact
     deployment cwd (never a pattern scan, never KILL). A daemon that
     survives `--term-wait-ms` (default 15s) is flagged TERM-immune and
     reported as `STALE-IMMUNE: rank->pid`; GO stays withheld.
   - GO from the lowest expected rank; all registrars relay GO once and
     exit 0. FAIL LOUD at `--timeout-ms` (default 120s):
     `REGISTRAR FAIL …` + `REGISTRAR FAIL_DIFF rank=N MISSING: […]
     PARTIAL-VIEW: rank->[…] STALE: rank->pid STALE-IMMUNE: rank->pid`,
     exit 1. `/proc`-gated: cleanslate enabled + no /proc → refuse to
     run; no `--deployment-cwd` → cleanslate disabled and every GO line
     says so. Every log line carries `[+s.mmm]` elapsed for GO-latency
     evidence.
2. **`tests/test_fleet_registrar.py`** — 16 fake ranks on loopback:
   GO path (deterministic leader, relay, <10s), missing-node (diff names
   the rank, exit nonzero), partial-view (fake peer with a frozen
   partial view → `PARTIAL-VIEW: 15->[15]`), subset (`--expect-subset
   5,9,14` over the 16-slot table), plus the /proc-gated cleanslate
   paths (stale holds GO; TERM clears it then GO; immune reports
   STALE-IMMUNE and is never KILLed). **22/22 PASS on spark0**
   (Linux aarch64); the 14 non-/proc checks also pass on the Mac. Wired
   into `PYTHON_TESTS`.
3. **`tools/glm5_next_wave.sh`** — phase 0 added: registrar fanout + GO
   wait before the heavy wave; the TERM/verify/settle/fire pattern is
   unchanged; `registrars` command and `--skip-registrar` escape hatch;
   `residentd.log` is now archived (`residentd.log.prev-<epoch>`)
   instead of rm'd, so flap evidence survives waves. Settle extended
   **45s → 75s** (see finding F3).
4. **`tools/registrar_stage.sh`** — compile once on spark0, hub through
   the workstation, copy+hash to all 16.
5. **`tools/devcycle/term_by_cwd_sweep.sh`** (excluded from the
   code-size counter, devcycle) — operator TERM sweep: exact exe+cwd
   discovery, TERM, wait, report survivors; never KILL.
6. Gates: code-size ratcheted 214793 → 215700 with lane justification;
   SHA256SUMS + PACKAGE_MANIFEST regenerated; `verify_package_manifest`
   PASS; python syntax PASS.

## Acceptance A — cold bring-up on the glm5_next fleet (PASSED with findings)

Baseline (captured before any action): fleet in the flap aftermath —
ranks 0,1 ready; rank 2 down with the flap signature in its log:

    hidden_spark_rdma_open_timeout route=tp-device.0023363482bdea87.0.2.3 role=sender host=spark3 port=63643 waited_ms=180000

and ranks 3-15 alive in `route_failed status=4`. spark5 additionally
carried 8 foreign single-spark procs from other lanes' checks (different
cwd — the cleanslate scan correctly ignores them; comm-matched procs are
handled by the wave's own stop step, as before).

First registrar phase over that mixed fleet: every node found its stale
daemon at `+0.003s`, TERMed it, observed exit at `+0.50s`, and **GO
fired at `+0.63s`–`+0.81s` per node** (bound: <10s) — 16/16 exit 0. The
only process left fleet-wide was the API (correctly out of cleanslate
scope). Then the full wave: 15/16 ready, **zero open-timeout lines on
all 16 fresh logs**. The one miss was rank 0: `GLM5_NEXT-ADAPTER
LoadDriver rc=15 → initialize=busy status=15 phase=adapter_initialize`
— the roving per-rank adapter-init flake (see F1), not a mesh failure.

Across every wave since the registrar went in (six fires, including the
collision window below): **zero `rdma_open_timeout` lines, fleet-wide,
every time.** The 180s-timeout cascade — the flap's signature — did not
recur once. The two-level-plus-cleanslate starting gun works.

## Findings (all live-evidenced)

- **F1 — roving adapter-init flake (pre-existing, separate hunt).** Each
  wave, exactly one (sometimes two) rank fails adapter_initialize with
  `LoadDriver rc=14|15` (route_not_found / busy): rank 0, ranks 7+8,
  rank 12, rank 7, rank 9 across five waves. It is independent of
  membership (GO fires first) and of the mesh (zero timeouts). The
  coordinator's spark2 history ("rc=15, flaky transport-open class")
  is the same beast. Fleet P(16/16 per wave) ≈ (15/16)^16 ≈ 0.36 given
  it; a flake costs a full re-wave since a **solo late rank cannot join
  a ready mesh** (see F4).
- **F2 — two wave owners recreated the race (coordination).** At
  18:34-18:37Z another actor ran a full registrar-gated wave over my
  18:01 wave's 15/16-serving fleet (their cleanslate TERMed my ready
  daemons — the mechanism works even mid-race — and their fire landed
  11/16; their API relaunch + state-bleed curl script was visible in
  `ps`). Racing waves produce fast `errno=98` listen failures instead of
  180s hangs — better, but still a degraded fleet. The one-wave-owner
  rule is load-bearing; the registrar cannot make two owners
  deterministic. I stood down once their traffic was identified.
- **F3 — cleanslate shifts TIME_WAIT into the settle window.** A
  TERM'd-but-late-dying daemon (see F5) leaves TIME_WAIT sockets that
  can outlive the classic 45s settle (observed as rank 7's
  `control_listen_failed port=63647 errno=98` at fire). Settle is now
  75s. Also added `tools/devcycle/term_by_cwd_sweep.sh` for the operator
  to pre-clear wedges.
- **F4 — the transport mesh admits no late joiners.** A solo rank-0
  restart (registrar subset {0}: TERM + GO in 0.50s) stalls at
  `fabric_ready` past the 180s window: ready peers never (re)open toward
  the new daemon. **The restart unit for TP16 glm5_next is the full
  fleet.** This retro-explains the flap era: sub-wave restarts could not
  re-join; the peers' re-opens cascaded. Acceptance B's "kill 8 +
  subset restart" is therefore expected to reproduce this stall for the
  restarted subset — the registrar gates it honestly (subset GO, no
  cascade to ready ranks), but the subset will not reach ready. I did
  not run the kill-8 form: the fleet was contested (F2) and then serving
  other actors' traffic; forcing it would have violated the ownership
  rules. Recommend: re-spec acceptance B for this deployment as
  "restart the FULL fleet via registrars, verify no ready-rank
  regression", or extend the transport to admit late joiners.
- **F5 — TERM-immune daemons are real and the fail-loud path carries
  the operator.** Three production demonstrations
  (`STALE-IMMUNE: 4->3192453`, `STALE-IMMUNE: 5->70841`,
  `STALE-IMMUNE: 4->3280539 15->1601842`): every registrar converged on
  full views, withheld GO for the full 120s, named the ranks+pids, and
  exited nonzero — no silent hang, no KILL. `SigCgt` showed SIGTERM
  caught by the daemon's handler; wedged loops defer it for minutes and
  all observed cases exited on their own afterwards. Two were fresh
  daemons TERMed mid-initialization (spawned by the concurrent actor
  between my sweep and my registrar phase). Operator path armed but
  never needed: the one pid I moved to SIGKILL had already exited on
  its own.

## Acceptance C — fail loud (PASSED)

Unit: missing-node and partial-view paths (22/22 on spark0). Live: the
three STALE-IMMUNE fleet events above — clean named diffs, nonzero
exits, GO withheld. Bonus evidence: `/proc`-less refusal and
`cleanslate=disabled` GO labeling are unit-covered.

## Bonus — state-bleed curl pair

Blocked from completing as a clean pair: the fleet spent this window
either contested (F2) or at 12-15/16. Raw result of the attempt at the
final state (15/16, rank 9 roving-flaked, API up):

    POST /v1/completions {"prompt":"The capital of France is","max_tokens":12}
    -> (empty body, connection hangs past 90s)

    POST /v1/completions {"prompt":"hello","max_tokens":4}
    -> HTTP:000 (request never completes)

i.e. **the roving rc=14/15 flake blocks serving** — a 15/16 mesh serves
nothing. That result is itself decisive for the next hunt: fix F1 and
the pair (below) becomes runnable; until then no completion-level
coherence test can even start.

The concurrent actor's exact pair, captured verbatim from `ps` on
spark0 (two DIFFERENT rotating token prompts back-to-back):

    curl -s --max-time 60 -X POST http://localhost:8433/v1/completions -H "Content-Type: application/json" -d '{"prompt_token_ids":[154819,11,1875,525],"max_tokens":4,"stream":false}'; echo E1
    curl -s --max-time 60 -X POST http://localhost:8433/v1/completions -H "Content-Type: application/json" -d '{"prompt_token_ids":[525,154819,11,13],"max_tokens":4,"stream":false}'; echo E2

## Fleet handoff state

FINAL SNAPSHOT (end of window): **15/16 ready, `open_timeouts` total
across all 16 fresh logs = 0**; the one miss is spark9 (rank 9,
`LoadDriver rc=14 → initialize=route_not_found`, the roving F1 flake).
API up on spark0:8433 (hangs requests while a rank is down — see the
pair result above). Compare the baseline: 2 ready / 1 down / 13
route_failed with 180s `rdma_open_timeout` cascades. The mesh flap the
operator diagnosed is gone in every wave the registrar gated.

Remaining work for the next lane: (1) F1 flake hunt
(adapter_initialize busy/route_not_found under simultaneous fire —
likely a driver-load race, NOT membership); (2) the state-bleed pair on
a 16/16 fleet; (3) decide acceptance-B's re-spec per F4; (4) the
closeout-lane's lingering procs on spark5 (cwd /tmp/dry2-*,
/home/spark5/sparkpipe) predate this window and were left untouched.

## Rules compliance

TERM never KILL by cwd (registrar TERMs exact pids by exact exe+cwd;
sweep tool same; the operator SIGKILL path was armed but never fired —
the one named pid had already exited). Spawn-captured pids in the wave
pattern unchanged. 110GiB envelope untouched (registrar is CPU-only,
~37KB binary). One wave owner: I stood down the moment another actor's
serving traffic was identified; my waves before that were the briefed
restoration. No model tokens in shared code — the registrar links
nothing and knows nothing about models; the launch table is flags.
Ratchet: code-size moved with justification in-commit; manifest
regenerated.
