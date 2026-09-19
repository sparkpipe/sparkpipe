# SparkPipe GLM 5.3 Flash TP16 — full state handoff (2026-09-19)

Branch: `lane/glm53-graph-replay` (tip c0ef8d5). Agent: `lane/fleet-resilience`
(tip ac20795). This supersedes the 2026-09-18 handoff; the key facts carry over.

## The serving chain (what must all work for one request)

1. HTTP → `model_api` (5090, port 8433): parse, queue.
2. Batch engine (`runtime/model_batch_engine.c`): admit, bind resident KV
   lane, drive prefill → decode steps.
3. Pipeline client (`runtime/model_pipeline_client.c`): per-stage two-phase
   transactions to all 16 engines (prepare → result → decision commit →
   completion).
4. Engine (`node/model_residentd.c` + the glm5_next module): route, claim
   slot + KV lane, run the chain (per layer: attention, KDA state, MoE
   experts via weightd lazy load, mesh allreduce).
5. The mesh: publish kernel → doorbell → weightd ships payload+tail to
   peers → per-rank wait → fused combine.
6. Completion flows back engine → API → token.
7. Terminal → release the KV lane.

## Bugs fixed this week (each with the evidence that found it)

- **seq_cell NULL atomic** (the illegal-memory-access showstopper):
  the eager round's kernel publish did `atomicAdd(NULL)` because the device
  cells were only allocated in `ArmCapture`. Found by compute-sanitizer
  (Invalid __global__ atomic, access to 0x0). Fixed: `EnsureCells` runs at
  round entry and arm; seq_cell seeded from round_seq.
- **One-lane mesh registration**: `PrepareReceiveBf16` registered only one
  2GB lane of the 16GB mesh (static guard bug) → kernel access on other
  lanes faulted. Fixed: register the whole region once.
- **Fused allreduce kernel ignored `blockIdx.x`** — every block recomputed
  the whole output (up to 1024× redundant work). Grid-stride fixed.
- **Tail counter skew**: per-process global counters could never align
  across ranks. Wire tag is now `(chain_epoch<<32)|chain_round` — every
  rank computes the same tag per round.
- **Marker-ship was never deployed** — a stale WEIGHTSD_BIN kept the old
  weightd live; the recovery path (full-slot retransmit) masked it.
  Deployed for real.
- **residentd died on any late/route-less completion** — FailLocked
  removed; undeliverable completions drop the route.
- **Agent watchdog killed weightds mid-bake** (30s grace, one 2s probe) —
  the convergence vortex. Now 120s grace + 3 spaced probes.
- **API batch engine parked in-flight requests in quiet windows** —
  Progress only ran while the local HTTP queue was non-empty.
- **Pipeline late/duplicate responses were fatal** (SetFailure → FailStop
  all 16 connections → invalidate everything → cascade). Now dropped loudly.
- **Failed/expired requests leaked KV lanes permanently** (16-lane wall).
  Release now queues on error terminals too.
- **Publish pipeline was a franken-tree** (curated scp list that drifted;
  a Sep-12 residentd shipped next to same-day drivers). Now rsyncs the full
  source set with `-R` and `--no-times`; residentd+api build from the same
  tree as the driver.

## Mock test suite (the discipline the operator demanded)

Committed, green on the target (runs in milliseconds, no fleet):
- `tests/mock_model_resident_client.c/.h` — link-time mock of the
  resident-client boundary: scriptable per-rank results/decisions/
  completions, self-driving two-phase drive, once-per-submission gating.
- `tests/test_model_pipeline_client_mock.c` — 14 checks green: happy path
  with commit ordering, duplicate/late/unknown dropped-not-fatal, real
  errors surface, id-ordering, fingerprint semantics.
- `tests/test_tp_device_collective_mock.c` — 5 checks green: eager round
  publishes with live cells (catches the seq_cell NULL class); the
  cuda_stub's publish now does the real doorbell/tail/tag writes.
- `tests/chaos_fleet.sh` — randomized kill-and-converge harness
  (unproven — run it once the fleet serves).

**Building the mocks found real contracts the integration failures never
named**: commit-before-completion ordering, per-stage token placement,
residency identity echo, once-per-submission semantics, queue-capacity
preflight. The operator was right: these were all unit-testable with mocks,
and "can't unit test emergent bugs" was wrong.

**Not shipped**: `tests/test_model_batch_engine_mock.c` — the mock must
model the resident-decode continuation (positions advance per token). It
gets the request to prefill-inflight but not through the first token.
That's the next build.

## What still bites (the honest list)

1. **The watchdog-vs-crashloop vortex persists under load** (spark0/spark3
   as I write this): engine attach fails while weightd bakes, engine dies,
   agent restarts it, the reconnect storm fills the weightd's accept
   queue, the watchdog's probe times out, watchdog kills weightd mid-bake,
   repeat. The 120s grace isn't enough under a storm. The real fix: the
   weightd needs a dedicated health/accept thread so the main loop's long
   synchronous work (spine preload, big attaches) never blocks health, and
   the watchdog should measure PROGRESS (log advance / IO counters), not a
   bare connect.
2. **Pool never warms persistently** — the pool premap is disabled over a
   spine-VA conflict (my reorder-only fix broke the spine; reverted at
   5d8dc98). Every chain re-pays the lazy load. Fix: premap the pool in a
   VA range disjoint from the spine.
3. **Cold chains ~450s** vs the API's patience — the warmup hook exists
   (agent, single-flight, retry) but must survive the vortex to help.
4. **Mesh region is 16GB pinned** for ~128KB of live B1 data — the slot is
   sized 128 max rows × 32KB max row. Right-size as a coordinated change.

## Operational playbook (verified)

- Publish: `./tools_local/m` from the lane worktree (rsyncs the full source
  set to sparkf, builds driver + weightd + residentd + api, publishes);
  then on sparkf, touch UPDATE + rsync both release roots to
  `spec@10.10.250.2:~/release/`. Agents pull from `http://100.123.97.61:8802`.
- The 5090 api needs an x86 rebuild from the same commit:
  `cd ~/sparkpipe-build && git reset --hard <sha> && make -j16
  build/sparkpipe_model_api` + the adapter build, cp into
  `~/glm53flash.fp8.tp16/{bin,lib}`, `systemctl --user restart g53-api`.
- The 5090 is reached from the Mac via `ssh spec@100.123.97.61`.
- Engine sweep: `pkill -9 -f "sparkpipe_model_resident[d]"` per node (the
  bracket avoids pkill self-match).
- Foreign `~/weightsd1/build/sparkpipe_wei*` daemons hold ~16GB each on
  some nodes and trip the memory gate — the operator authorized killing them.
- Ground truth past log claims: read the mesh memfd directly at
  `/proc/<weightd pid>/fd/<spark-mesh fd>`; doorbell region at
  0x400000000 (24-byte entries, idx = band*16+rank).

## Next, in order

1. Break the vortex for good: weightd health on a dedicated thread +
   progress-based watchdog; engine attach must be idempotent and not
   crash-loop the process.
2. Batch engine + residentd mock suites green (the continuation-position
   modeling), then the whole mock suite gates the merge to main.
3. Pool premap with disjoint VA → warm chains on every token.
4. Fixture case 0 (176 tokens, answer "B") serving on the warm fleet.
5. Graph re-arm (the 695µs/round eager overhead is what the graph removes),
   then measure against the 50–100µs allreduce target.

## Addendum 6 (2026-09-19 late): mock-suite era + the two fleet-agent root causes

The mock program the operator demanded is on main (PR #1045): link-time
mock resident client (bounded in-flight queue, EnsureConnected reconnect
semantics, Kill/Revive, silent in-flight drop on death), batch-engine
chaos suite (24 checks: rank death mid-decode, kill/revive, EOS,
concurrent requests), weightd churn suite (130-cycle slot churn,
SIGKILL-mid-bake vortex, eviction under pressure), and a behavioral
ibverbs stub + mesh wiring suite (142 checks incl. the two-daemons
separation case). Watchdog shell suite (7 checks) on lane/fleet-resilience.

Bugs the suites found on arrival: the lane was missing #1034 (the mock
reproduced INTERNAL_ERROR 17); test_weightd_working_set was dead on every
branch (compile + two latent map teardown bugs: Destroy never unmapped the
span, imported chunk handles never released — fixed, suite green, landed
via #1050); SparkWeightdMeshInit was declared with args but defined (void)
— the mesh CLI flags were silently ignored (fixed).

Fleet-agent root causes found + fixed + deployed this session:
1. install_core compared sha16(file) against a full 64-char announced
   sha — NEVER matched, so weightd never updated once a full-length sha
   was announced. The fleet ran a stale weightd for days. Fix: announced
   cut to 16 chars. Test: watchdog case 5.
2. The watchdog's socket probe cannot distinguish "wedged" from
   "mid-bake" (the server never accepts during a synchronous load).
   Bakes > ~135s were killed mid-bake forever (spark0's vortex). Fix:
   after probes fail, read the youngest weightd's utime+stime twice 5s
   apart; advancing CPU = alive. Test: watchdog case 6.

spark0's final anatomy: a root-owned foreign weightsd1 (the devs' stable
daemon) squatted mesh rank 0 + /tmp/weightd-mesh (root-owned .ready our
agent couldn't unlink), plus a 39GB T1 decoder squeezed memory. Sysadmin
cleared both. The structural fix: weightd-mesh --mesh-dir /
SPARK_WEIGHTD_MESH_DIR (PR #1052) + the agent runs the fleet's weightd
with /tmp/weightd-mesh-fleet (lane/fleet-resilience). DEPLOY ORDER: the
weightd with --mesh-dir must be in the release BEFORE the new agent
self-updates, or the flag is unknown and weightd won't start.

Remaining known design gap (observed live, not yet fixed): the weightd
server loop is synchronous per request — a 512-key boot acquire with
cold disk->GPU copies occupies the server for minutes; the listen
backlog (128) fills with waiting engines and fresh connects get EAGAIN.
The queue-of-the-dead: a client that dies mid-request still has its work
completed before its EOF is seen. Consider: skip-if-dead peek before
dispatching a queued request, and/or chunk the big acquires into
per-Step slices.

## Addendum 7 (2026-09-20): merged-tree serving validation + the reset cascade

The merged tree (main + lane) serves: chains complete cold in ~508s
(CHAIN-TIME status=0, stage 3 = 508.5s of cold expert loads, 91 rounds,
allreduce 59.7s). The 30s "rounds=0" deaths were the API/client giving up
before the cold chain finished, not a transport break.

Deploy discipline that bit: publishing the merged ENGINES without the
merged WEIGHTD left the fleet protocol-mismatched (new tail contract vs
old ship path). Always publish weightd + engines from the same tree.

The reset cascade (cold-era amplifier): any engine-side failure -> API
pipeline IO_ERROR -> resident client reconnect -> engine session reset
("client reset generation=N" counted 536) -> in-flight work dropped ->
next request fails -> repeat. Warmup (one cold 900s-budget pass from
rank 0's agent) converges the fleet; once warm there are no failures and
no resets. The 39GB T1 decoder + foreign weightsd on spark0 are gone
(sysadmin); spark0's zzpin/zznowarm drop-ins removed (fleet-normal).

API is single-session: repeated probes queue behind the in-flight one —
during bringup, fire ONE request and wait it out.

## Addendum 8 (2026-09-20): the weightd->engine coupling law

The engine's weightd attachment (the unix socket AND the mesh memfd) is
scoped to the weightd PROCESS. A weightd restart orphans every attached
engine: the socket dies and the memfd belongs to the dead process, so the
engine's chains fail with instant IO_ERROR (spark_weightd.c:2963/3342)
while the engine looks alive. The agent recycles engines on BINARY sha
change but NOT on weightd process change — the install_core weightd
rollout orphaned the whole fleet's engines and every chain failed until
the engines were swept.

LAW: when weightd restarts, its engines must restart. The agent should
treat a weightd process change (pid/start-time) as an engine-recycle
trigger — same class as the driver-sha recycle it already does.

Chain progress signature for the record: cold chain ~508s (stage 3 =
expert loads), warm chain 380ms/91 rounds/~1ms allreduce per round,
first-round wedge = CKEY-CELL-TIMEOUT (peers waiting on rank 0's cell),
instant death = IO_ERROR (stale weightd attachment).
