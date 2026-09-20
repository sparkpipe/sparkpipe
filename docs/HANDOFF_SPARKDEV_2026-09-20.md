# SPARKDEV HANDOFF — 2026-09-20 (post resilience + mock-suite era)

Read this first. It is the current operating state, the exact next steps, and
every trap that cost real hours. The older addenda live in
docs/HANDOFF_glm53_flash_2026-09-19.md (addenda 6-10 are this session).

## TL;DR

The fleet is 16/16 up on the fixed build but does NOT serve: the TP16
allreduce never completes a round (`rounds=0`, chains die at the 30s timeout).
The infra/ops layer is now solid and self-healing; the remaining wall is the
collective round path. A publish-ack backpressure fix (the proven wedge fix)
is built, tested, and PR'd but the fleet build carrying it was mid-deploy when
this was written — VERIFY whether it landed before debugging anything.

## PR landscape (check state with `gh pr list`)

Merged to main: #1045 (mock suites), #1047 (the 41-commit fleet-fix landing),
#1052 (weightd --mesh-dir), #1053 (mesh 64KB alignment restore), #1056 (cancel
on session reset), #1058 (BUSY-as-backpressure + pipeline mock wiring), #1059
(expert-loading stress), #1060 (allreduce fuzz harness + Destroy UAF fix),
#1061 (loopback full-system test + kill fuzz). Open: #1063 (publish-ack
backpressure + bench mode — the wedge fix). Verify #1063 merged before
assuming the fleet has it.

## The fleet build tree

The fleet should now build from **main** (everything is merged). The fleet
build happens on sparkf at ~/sparkpipe-build-main (rsync'd from a worktree).
The publish flow (all on sparkf):
  export PATH="/usr/local/cuda/bin:$PATH"
  SHA=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
  make -j16 build/sparkpipe_model_residentd build/sparkpipe_model_api build/sparkpipe_weightd build/sparkpipe_model_compile build/libhidden_transport_spark_host_rdma_verbs.so
  make -j16 -C modules/glm5_next_resident_decode_stage publish adapter EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 CONTRACT_SHA256="$SHA" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
  build/sparkpipe_model_compile --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json --library build/module_library --output /tmp/main-out --cc /usr/bin/cc --include include --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread
  SPARKPIPE_BUILD_TREE=$HOME/sparkpipe-build-main SPARK_DRIVER_SO=/tmp/main-out/stages/stage_000/model_driver.so tools/publish_local.sh glm5_next_resident_decode_stage fp8 glm53flash.fp8.tp16
  cp build/sparkpipe_weightd ~/release/core/bin/sparkpipe_weightd
  tools/weightsd_announce.sh ~/release/core   # then regen core MANIFEST + sync
  touch ~/release/glm53flash.fp8.tp16/UPDATE
  rsync -a ~/release/glm53flash.fp8.tp16/ spec@10.10.250.2:~/release/glm53flash.fp8.tp16/
  rsync -a ~/release/core/ spec@10.10.250.2:~/release/core/
The API runs on the 5090 (spec@100.123.97.61): rebuild model_api there
(x86) from the same source, cp to ~/glm53flash.fp8.tp16/bin/, systemctl --user
restart g53-api. The rsync set MUST include `inference/` now (runtime/gemm.cuh
needs it — a curated-list drift that bit twice).

## The resilience layer (all deployed; do not re-derive)

- Agent (tools/fleet_node_agent.sh on lane/fleet-resilience, deployed via the
  hub): weightd-generation coupling (an engine older than its weightd is
  recycled — the attach socket + mesh memfd are scoped to the weightd
  process); install_core sha16-vs-64char gate fixed (weightds had never
  updated); watchdog never kills a mid-bake weightd (CPU-advance liveness
  after the socket probes fail); --mesh-dir probed from the binary's --help
  so agent/weightd deploy in any order. Watchdog suite: tests/test_weightd_watchdog.sh
  (9 cases, Linux-only, skips if a live weightd exists).
- Engine/pipeline: a session reset broadcasts the collective cancel (peers
  fail fast, no 30s wedge); a rank's BUSY is backpressure (no fail-stop-all),
  the batch circuit breaker ignores BUSY.
- weightd: republishes its mesh record when a foreign generation overwrites
  it (the two-daemons-one-host collision); the fleet's weightd uses
  /tmp/weightd-mesh-fleet (the devs' weightsd keeps /tmp/weightd-mesh).

## The mock/test net (all green on main, run in seconds on any dev host)

- tests/test_model_pipeline_client_mock.c — 14 checks, the pipeline state
  machine over a mock resident boundary.
- tests/test_model_batch_engine_mock.c — 29 checks, 6 scenarios incl. rank
  death mid-decode, kill/revive, EOS, concurrent requests, BUSY backpressure.
- tests/test_tp_device_collective_mock.c — 11 checks incl. the two-rank
  cancel-fail-fast case.
- tests/test_weightd_mesh_mock.c — 147 checks over a behavioral ibverbs stub
  (wiring, peer-restart rewire, dead-QPN, CQERR repair, foreign-generation
  record overwrite -> republish).
- tests/test_weightd_churn.c — slot churn (130 cycles over the 128 cap),
  SIGKILL-mid-bake vortex, eviction under pressure.
- tests/test_weightd_expert_stress.c — concurrent same-expert, mid-acquire
  death, eviction+reload.
- tests/test_tp_allreduce_fuzz.c — N in-process collectives with verifiable
  sums; --fuzz N rank kills; --bench N timed rounds.
- tests/test_system_loopback.c + tests/fuzz_system_loopback.sh — the full
  stack on one host (no weights), kill/restart fuzz.
- tests/test_weightd_watchdog.sh — the agent watchdog (Linux only).
All are in `make test` (TEST_NAMES / SHELL_TESTS). The Mac build needs the
POLLRDHUP shim (landed) and cannot build weightd/mesh (no infiniband) — those
run on Linux.

## THE WALL (the one remaining blocker): TP16 allreduce rounds=0

Chains run (layers advance) but the first collective round never completes:
CHAIN-TIME status=15 total_ms=30000 rounds=0. Evidence at close:
- Epochs converge (all ranks adopt the same chain epoch).
- Mesh records converge (the republish fix healed the foreign-generation
  overwrite).
- Doorbells ship (WD-SEEN/WD-SHIP flow on rank 0).
- MESH-SPIN-TIMEOUT on rank 0 names a STABLE-then-shifting missing-peer
  subset — per-peer payload/tail delivery, not a protocol-wide wedge.
The publish-ack backpressure fix (PR #1063) makes the doorbell overrun
impossible; whether it clears the fleet wedge was NOT yet confirmed at write
time. If the fleet still wedges with #1063 in, the next instrument (from
addendum 10): dump the missing peers' slot tails from the weightd memfd
(/proc/<weightd pid>/fd/<spark-mesh fd>, doorbells at 0x400000000, 24-byte
entries, idx=band*16+rank) during a stuck round and compare against rank 0's
published tag — that splits "tail never shipped" from "tag mismatch".

## Benchmark state

In-process (the harness, no wire): TP16 steady-state ~1.4ms/round, ~700
rounds/s — the round's publish+wait+combine protocol on the host. macOS
nanosleep granularity inflates it; the wire number needs the fleet (blocked
on the wedge). The +8-18% cost of the backpressure fix is measured and honest
(see PR #1063). The operator's 50-100us/round target is far below the current
host-side protocol cost — the spin-wait/multi-handoff structure is the thing
to attack once the fleet serves. The dual-100gbps pairwise pipeline is parked
per the operator ("next version").

## Traps that cost real hours (don't relearn)

- The publish flow's curated rsync set drifts — use the full source set and
  include `inference/` (gemm.cuh needs it).
- Ledgers: any file add/change needs python3 tools/generate_package_manifest.py
  AND tools/generate_sha256sums.py, committed together, or the CI ledger gate
  fails.
- The fleet agents self-update from the hub (spec@100.123.97.61 release/core);
  a hub file alone does nothing — the MANIFEST must regen (sha256sum the three
  core files into MANIFEST, touch UPDATE) before agents pull.
- Restarting the fleet-agent systemd unit kills the weightd with it (same
  cgroup despite setsid). To restart only the agent, leave the weightd alone
  or accept the recycle.
- The API is single-session: during bringup fire ONE request and wait it out
  (cold chain ~508s, warm ~380ms). Repeated probes queue and starve.
- pgrep -f patterns need the bracket trick ([s]parkpipe) to avoid self-match.
- 5090 from the Mac: ssh spec@100.123.97.61. Nodes: ssh spark0..sparkf.
  spark0 has passwordless sudo for infra repair (ptrace_scope blocks gdb
  without it; sudo gdb -p works).

## Final state note (last update before handoff)

The publish-ack backpressure fix IS deployed fleet-wide (the driver carries
MESH-SHIP-TIMEOUT/MESH-SHIP-CANCEL) and the wedge PERSISTS — so the fleet's
rounds=0 is NOT the doorbell overrun. The remaining bug is the per-peer
delivery gap: rank 0's MESH-SPIN-TIMEOUT names a stable-then-shifting subset
of peers whose slot tails never arrive. The backpressure fix was still worth
landing (it removes a proven silent-drop). The next session starts at the
per-peer tail delivery: with a chain stuck, dump the missing peers' slot
tails from their weightd memfds and compare against rank 0's published tag;
that splits "tail never shipped" from "tag mismatch". Both the mesh mock
(rewire/CQERR) and the allreduce fuzz harness are the in-process repro
harnesses to build that case against.

## Spot-test result (the approved bypass experiment): the mesh DELIVERS fine

Direct memfd reads during a stuck round (tools: /tmp/meshdump.py +
/tmp/shipdump.py + /tmp/slotdump.py + /tmp/celldump.py on the sparks):
- rank 0's publish lands byte-identical in a peer's memfd (slot tail + payload
  head match exactly) — the weightd ship path WORKS.
- rank 0's base cell broadcast lands on the peer identically — the cell path
  WORKS.
- So "per-peer tail delivery" is NOT the wedge. The real shape: the chain
  needs all 16 ranks to accept the SAME submission in the same window. Engines
  still booting (post-restart) reject with BUSY at the resident slot claim;
  rank 0 gets in, runs the chain, and waits forever for the missing ranks'
  tails. MESH-SPIN-TIMEOUT's missing set IS the set of engines that were
  BUSY-locked when the chain started.

THE ACTUAL BUG: the batch engine dispatches a chain without confirming all 16
ranks accepted the submission — a partial cohort wedges the collective. The
fix direction: dispatch a chain only when every rank's pipeline reports the
submission accepted (the pipeline already aggregates per-rank results — the
chain must not start on a partial admit), OR the late ranks must join the
chain (the collective's slot assignment is per-request so a late joiner
misses the window). The former is a one-gate change in the batch dispatch;
the latter is a protocol change.

This composes with the slot starvation: a wedged partial-cohort chain holds
its resident slot for 30s, and enough of them BUSY-lock an engine. Fix the
partial-cohort dispatch and the slot pressure disappears with it.
