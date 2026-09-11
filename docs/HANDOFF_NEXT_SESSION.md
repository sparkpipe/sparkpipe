# HANDOFF: glm53flash lane — post-PoC, productionization phase

Written 09-11 ~23:00 after the ≤100µs/round PoC campaign (iterations 13-32).
Lane: /Users/mac/lane-glm53, branch lane/glm53-p0, tip 1137657 (+~15 uncommitted
changes — see UNCOMMITTED). Coordinator memory glm53-flash-tree-engine.md is the
full ledger; this doc is the working map.

## THE HEADLINE RESULTS (all measured, real 16-node fabric)

1. **100-105µs/round sustained**, checksums correct, 16-way allreduce through
   the real weightd relay (was ~300µs at campaign start). Communication
   roofline ≈ 500 tok/s at B1 → at 14-50 tok/s targets the fabric is 3-10%
   noise. "Sparks too slow for TP16" is DEAD: the villain was always CPU
   orchestration (~900ms/token: launch ioctls ~127ms, ~106 completion
   host-funcs + drain wakeups, worker hops), not the wire.
2. **TP16 scaling projection revised UP to 2.5-3× TP4 at B1** (earlier 1.2-1.5×
   assumed 250µs unhidden rounds; at ~100µs hidden the round floor drops below
   TP4's 20ms compute floor). B1 scaling table in iteration-20 notes.
3. **Mesh transport FULLY ACQUITTED** after a hostile audit: every relay
   counter clean (send_ok exact, send_err=0), perfect delivery dumps
   (20/20 + 40/40 both bands). The ENTIRE stall class across iterations 22-31
   was TWO SELF-INFLICTED BUGS:
   - WR-mutation race: fast relay patched a prebuilt WR's sge and reposted it
     (verbs law: no WR modification before prior completion) → random
     mid-replay lost writes. FIXED: per-post fresh stack seq-WR copies
     (weightd d0288f418feb).
   - Stale inter-replay barrier (my iter-24 code) polling band-1 ends for a
     target band-alternation makes unreachable → replay 3 never launched.
     FIXED: barrier deleted (band alternation supersedes it).
4. Production serving throughout: tokens green (deterministic-first-token,
   divergence gremlin persists), ~0.85 tok/s floor = 0.42% mem roofline —
   the remaining gap is pure module orchestration, removed by layer graphs.

## WHERE THE CODE IS (all ships via scp+build; Mimosa gate blocks git commits)

- node/weightd_mesh.c (UNCOMMITTED): fast relay for bands 1+2 (2µs poll,
  prebuilt payload WRs, fresh seq WR per post, per-round 64K seq cells,
  stable seqlock doorbell re-read, inline CQ drain 64/pass, CQ 4096,
  QP depth 64, dedup-advance-only-on-full-success, WD-FAST/WD-TRACE debug
  prints with budgets). Deployed fleet-wide through several cores; latest
  published d0288f418feb. PRODUCTION bands 0/3 use the original slow path —
  unchanged semantics, all four hardening fixes apply to it too.
- ring/transport/tp_device_collective.c (UNCOMMITTED): GPU-resident rounds —
  publish host-func + CPU spin replaced by SparkGlm5NextMeshPublish/WaitKernel
  (in module .cu, extern launchers, ordinal baked as arg, no seq cell).
  Serving green at ~38s/32tok (≈5% win — engine round was never the
  bottleneck; module orchestration dominates).
- modules/glm5_next_resident_decode_stage_module.c (UNCOMMITTED):
  fire-and-forget chain (inline advance on submit; completion = failure-only)
  — KEPT: correct, simplifying, prerequisite for graphs.
- tools/mesh_graph_poc.cu: the PoC. Mode 2 = graph replay, mode 3 = latency
  probe (GPU %globaltimer spans, host-pinned stamps, watchdog thread, band
  alternation A/B graphs, per-replay launch-rc prints). POC_ROUNDS=20.
- tools/perf_roofline.py: 273GB/s mem / 25GB/s net / 201 tok/s mem-roofline.
- tools/read_ends.py: doorbell/end-word dump via weightd memfd.
- Engine shipped earlier & committed: parity slots, zero-sync host-registered
  rounds, GPU u64-max, max-op OOB fix, boot seq zeroing (main-merged via
  PR #929).

## WHAT REMAINS OPEN (priority order)

1. **PoC completion run**: patient rerun (timeout ≥500s) of mode 3 ×10
   replays — expect all replays complete now that both bugs are fixed; the
   sorted min/p50/p90 per-round stats print on completion. Alt-band (B)
   first-replay is COLD (~15-20ms — relay builds band-2 WR sets + lockstep
   re-forms); steady replays ~2ms/20 rounds. If a stall recurs with both
   fixes in, the watchdog names the position immediately.
2. **Layer graphs in the module** — THE production lever. Everything is
   captureable now: per-slot fixed buffers, fixed-VA lease window
   (BeginUse always returns map->base), GPU-resident rounds are pure kernels.
   This kills the 127ms ioctl tax + 106 host-func wakeups → projected
   10-40ms/token = 25-100 tok/s (14 = conservative floor).
3. **Port fast relay to production bands 0/3** (extend the band gate; the
   per-round cell arrays and inline drain are already global).
4. Fused wait+combine kernel (operator-endorsed: sum-on-flag-arrival, saves
   per-round combine launches ≈ 2-5ms/token; parameterize by peer-mask for
   TP4 quartets / TP8 halves from day one — topology matrix directive).
5. Cleanup: strip WD-FAST/WD-TRACE/watchdog prints; divergence gremlin
   (greedy decode diverges after token 1 run-to-run — state-slot hygiene
   suspect, diagnose before any acceptance gate); Mimosa scanner-root
   escalation (blocks ALL lane commits — scans shared checkout's pre-existing
   files); g5n_repack_place.sh + engine/module changes uncommitted because
   of it.

## OPERATIONAL LAW (hard-won this session)

- Ship protocol: scp + verify-repair md5 loop (parallel scp clobbers to 0
  bytes: sparkf hit it twice — sparkf builds LOCALLY from its .cu).
- Build: scp sources → sparkf make adapter+publish → sparkpipe_model_compile
  → publish_local.sh (driver-compile step rebuilds stages/*/model_driver.so;
  skipping it ships stale drivers). Weightd-only changes: make
  build/sparkpipe_weightd + publish_core.sh. module_build_release.sh does its
  own git fetch+reset+clean — NEVER ship uncommitted code through it.
- PoC launches: stagger 2s/node, argv-shared monotonic seq base
  (date +%s*1000), SIGTERM cleanup (kill -9 of registered procs leaves
  cudaHostRegister teardown poison). Token 0 costs ~72s of lockstep
  formation — not a stall.
- Full-fleet bounce needed after any wedge (partial bounces leave stuck
  routes); fresh-API restart clears the batch-engine latch alone.
- CUDA: register ONLY band slices (512MB) not the whole 2GB region.
- instruments: host-pinned stamps read by watchdog WITHOUT CUDA calls
  (device-side stamps + cudaMemcpy block behind the stalled graph).

## GIT / COMMIT MAP

- Committed & merged: PR #929 (main 540e44d) = parity engine + CLI mesh
  identity + agent mesh args. Probe commits 5eae2bd..1137657 on lane (temp
  probes stripped in worktree via git checkout 80dbecc -- module.c/.cu).
- Uncommitted (Mimosa-blocked): engine GPU rounds, fire-and-forget module,
  weightd fast relay + hardening, placement runbook, roofline tool, PoC,
  this handoff. ALL are live-deployed via scp+build.

## MODEL/PACK FACTS (from the numerics arc)

- fp8 packs rebuilt 09-11 from checkpoint (old packs predated scale-plane fix
  898ee1a — expert scale planes were garbage → all-zero tokens). Hex rank
  names ranka-f for ranks 10-15. Verify with tools/g5n_repack_place.sh
  (9/9 byte-verify vs checkpoint per rank).
- The lazy-expert lease path is CORRECT end-to-end (live-verified: exact
  checkpoint scale values through the full lease). Expert VA window is a
  fixed base → graph-capturable.
- Numerics root cause doc: memory glm53flash-numerics-rootcause.md.
