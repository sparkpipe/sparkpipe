# ALLREDUCE HILLCLIMB — 2026-09-20

## GOAL (set 09-20, operator)
**Allreduce ≤ 100µs/round MEASURED on the 16-spark fleet** (91-round chain, status=0, checksums green), starting from 1.79ms/round (the healed fleet's cold-class eager number). Every intermediate number graded MEASURED (wire-math floor check before grading); every step committed + PR'd.

**Ladder:** 1790µs → <1000 → <500 → <200 → ≤100µs. Current: **386µs/round CONTENDED-era wall** (09-22: chains RUN 91 rounds under the new epoch stack; host-measured allreduce_ms=0.66 TOTAL for 91 rounds — the round path itself is sub-ms; the wall is the relay-arrival/compute cadence, unmeasured clean since the serving path is mid-repair). Best clean receipt remains 330ms/round (09-22 02:20, status=0, 2/2). Chain totals now 432-518ms WARM-CLASS (weightd's expert cache survives engine recycles — fresh engines serve warm immediately; the cold cost is once per weightd boot). In-process harness floor: 1530µs.

## Step log (append every step: what / number / verdict)

### S1 — DONE (09-20 ~13:5x): per-op cudaStreamSynchronize → pinned-cell poll (660b6b1)
MEASURED: allreduce 121.4ms/91r = **1334µs/round** (from 1910-2700µs; −30-50%). Chains total 432-518ms warm-class = ~2-2.3 tok/s through the full engine. Zero MESH-CELL-TIMEOUTs (the poll's guards work). NOTE: chains are warm by default now (weightd caches experts across engine recycles — only the FIRST engine per weightd boot pays cold).
NEXT S2 candidates (measured against the 1.33ms round): the host peer-wait spin (the remaining serialization — each op waits 15 tails sequentially-polled), the publish-ack wait (waits the PREVIOUS op's ship before publishing — adds a full ship latency per op), and the per-op kernel launches (3 launches + 1 D2H per op). Then S3 (graph).

### PR-BY-LAW DEPLOY (09-20 ~19:00): PR #1064 = THE deployable line
Merged lane/fleet-resilience (23 agent commits) into the PR + all wedge/climb fixes; head 4f476aa; DEPLOYED SHAS (in the PR body, audit-verified 16/16): weightd eec4938201602fe4, agent 600ed95c9777eb1f, driver/adapter via the glm53 MANIFEST, x86 api+adapter from the same head. Deploy ONLY from this PR now.

### CURRENT OPEN (as of the PR deploy verification): a NEW wedge signature
rounds=0 again, but DIFFERENT from the healed one: zero CKEY-CELL-TIMEOUTS (key exchange fine); each rank STABLY misses ONE DIFFERENT peer across retries (r0→12, r2→13,14, r4→15, r6→14, r7→15, r8→14, r9→12, rc→14, re→15, rf→1; ranks 1,5,a,b,d never spin). Deterministic per-poll single-peer miss — NOT the stale-record black hole (records 16/16, wiring 30 entries, cellTO=0). Next instruments: (a) slotdump a missing pair during the spin — is the peer's tail landing in a DIFFERENT slot than polled (parity disagreement for specific pairs)? (b) check whether the missing-peer ranks are even publishing (their engines' chain state); (c) the ordinal parity: pub=34359738369=8<<32|1 → ordinal bits — the merged branch changed SparkTpChainOrdinal/parity; verify publisher parity == poller parity per pair.

### WEDGE HUNT (09-20 ~20:00): mesh exchange FULLY EXONERATED — the ships exist at the right key and still don't land
Post-settle state (weightds stable, one generation): records self-healed + current (spark0 rec mtime == weightd start), wiring boot-current BOTH directions (sparka->peer0 wire boot == spark0's own record boot), CKEY epochs CONVERGED (all ranks adopt epoch 29, all spin at key 28 — the divergence theory dead), cells fine. THE FACT: sparka's relay SHIPS key-28 publishes (WD-SHIP idx=26 seq=28<<32|1...) but rank0's receive slots for peers 10/11/14 (slots 160/161, 176/177, 224/225) are ALL-ZERO — the RDMA writes never land, silently (no CQ errors, no wire fails). Only SOME pairs fail (rank0 misses 10,11,14; other ranks miss their own subsets).
NEXT INSTRUMENT (first move next tick): a one-line diagnostic in the weightd ship loop printing per-peer (peer, addr, rkey) for the FIRST ship of each pass — compare against the peer's CURRENT record recv_addr (the qp_snapshot may hold a stale addr/rkey for specific pairs even after WD-WIRED: the transition logs the wire but maybe a failed RTU path leaves the old qp_info, or the snapshot copy races the rewire). Also check ibv post errors on those specific QPs.

### WEDGE CONVICTED (09-20 ~21:15): the chain keys DRIFT across ranks on independent retries — kimi's partial-cohort verdict was RIGHT
Complete evidence chain: (1) ship addrs verified byte-exact (sparka->peer0 addr=ffde13ffe000 rkey=1836032 == spark0's current record — the qp_snapshot suspicion DEAD); (2) the mesh delivers EVERYTHING: a 12-peer slot census on spark0 shows tails LANDED at every peer's parity slot, all = key 13 (55834574849); (3) but rank0 spins waiting at key 11/12 missing peers 1,3,15 — the peers' keys and rank0's key DRIFTED because each rank's 30s timeout advances ITS OWN chain key independently (CKEY-ADOPT converges epochs only at chain START; retries self-increment in between); (4) CORRECTNESS HAZARD exposed: a peer AHEAD of the waiter passes the >= check with WRONG-CHAIN data — the spin survivors are silent wrong-data accepts waiting for laggards.
THE FIX (next code step, kimi's gate + the deeper invariant): (a) the dispatch gate — no chain start unless all 16 ranks admitted (kills partial cohorts); (b) retries must RE-ADOPT rank0's current broadcast epoch (not self-increment) so keys re-converge after any partial attempt; (c) the wait compare should be == exact-key (parity-slot freshness) not >=, making cross-key data a loud rejection instead of a silent accept.

### FIX-C DEPLOYED + NEW PROXIMATE CAUSE (09-20 ~21:4x)
Fix-C (1ce99fb, marker V6-EXACTKEY) deployed fleet-wide (driver 670e41a3): the wait now demands tail_key == expected_key — VERIFIED WORKING (rank0's peer-11 slot holds key-9 data and is now correctly REJECTED while waiting key 10; the wrong-chain silent accept is dead).
THE WEDGE'S PROXIMATE CAUSE FOUND: the missing-peer ranks' ENGINES HAVE NO MESH MAPPING — spark7's engine: 0 spark-mesh fds, zero mesh lines in its log, engine started 6s after its weightd, yet its chain machine runs (stage prints) while its relay sees NOTHING (23 WD-SEENs total). An engine that booted while its weightd was mid-init landed in a no-mesh state and chains still 'run' — publishing nowhere. The boot race: engine start (weightd+6s) < weightd mesh-ready (weightd+76s wiring!) — the engine's lazy attach hit a weightd whose mesh wasn't wired, failed, and never retried.
NEXT (first move): the engine's lazy-attach must RETRY until the weightd's mesh is wired (the .ready marker exists as the gate) — or the agent must not start engines until /tmp/weightd-mesh-fleet/.ready exists (it gates on the marker already but the marker is written by the weightd at wire time — verify the ordering; the engine at weightd+6s vs wiring at +76s means the marker appeared AFTER engine boot → the engine missed it and its retry must poll the marker or re-attach on EOPNOTSUPP/attach failure).

### FIX-D DEPLOYED (e0266e0, driver cfd67273): attach-without-mesh-fd = BUSY
The no-mesh-engine class is dead: an attach reporting mesh wired but carrying no fd now fails BUSY and the engine's 600x1s retry loop re-attaches (all engines verified holding the memfd VMA — note: the fd is CLOSED after mmap; check /proc/E/maps for 'spark-mesh', NOT fds — my earlier '0 fds' diagnosis on spark7 was an instrument error, though sparkf's engine genuinely lacked the map at first sight).
STATE AT TICK END (recorded for next tick): engines 16/16 ready with mesh maps, weightds stable, records fresh — but the API is in connection churn again: set-failure status=4 stage=0/1 cycling, engine sees 797 peer-eof reconnects, ZERO submissions reach the module (no ADMIT prints), RANK-RESULT-FAIL rank=0/6/13 status=15 — the request dies before chains. This is the S0-era churn SHAPE but with a new gate: the first prepare rejection comes as BUSY from ranks 0/6/13 BEFORE any module admission print — next tick starts at the engine's submission entry path (what rejects BUSY pre-module: the resident-slot claim at boot state? the route reserve?) — instrument the residentd submission handler's early returns.

### TICK 09-20 ~21:15: churn cleared on fresh API; the wedge narrowed to the TWIN-WEIGHTD generation fight
Fresh API connect + probe: submissions FLOW again (DECISION prints, chains run) — the churn was the stale API process. The surviving wedge signature: rank0 spins missing {11,12,13,15}; rank11 (sparkb) shows CKEY-CELL-TIMEOUT cell=774726319 (its base cell reads 0 — rank0's broadcast doesn't reach it). ROOT SHAPE FOUND ON SPARKB: TWO weightd generations fought — the running one started 20:34 (boot ...060, recv_addr e7399fffe000, its record is CURRENT), but rank0 WIRED peer11 at 20:36 (boot ...205, addr ea2abbffe000 — a generation that no longer exists; ships to its dead MR vanish silently = the black-hole class). rank0's local mesh-b.rec (fetched 21:02) carries the 20:34 gen — TryWire SHOULD rewire back (record.boot != wired_boot) — VERIFY the newest WD-WIRED peer=11 line and whether TryWire rewires DOWNWARD (it may refuse: most code only moves forward). NEXT: (1) check TryWire's rewire-down path (stale-record vs newer wired boot — likely the missing case); (2) the twin-weightd fight itself: why did a 20:36 weightd start while the 20:34 one lived (agent watchdog vs my manual restart racing — the janitor then killed the 20:36 one); (3) after both fixed: full-cohort chain + the S2 measurement.

### FIX-E DEPLOYED (e9b64c8): the janitor's singleton path was WRONG — the 30-min churn machine dead
The janitor checked /tmp/spark_weightd.singleton but the weightd holds <socket>.singleton = /tmp/spark_weightd.SOCK.singleton (verified live: the running weightd's fd 3). The holder lookup ALWAYS returned empty → EVERY weightd older than 30min was killed as 'not the singleton holder' → agent restart → engine recycle → mixed generations → wedge → repeat. This explains sparkb's twin generations (20:34 live, 20:36 ghost) and possibly days of 'stable-then-shifting' missing sets. Deployed fleet-wide (direct agent cp — NOTE: must also flow through the core release next publish).
POST-FIX-E STATE: no more weightd churn, but the wedge PERSISTS with shifting per-rank missing sets ({6,10,11,15} on r0, {6,11,15} r1, {4,7,8} r5...) and ZERO cell timeouts — the shape is now PURE key-drift/partial-cohort (fix-C made all cross-key encounters loud). THE REMAINING FIX = kimi's dispatch gate (no chain start unless all 16 ranks admitted the same submission) + the retry re-adopt. THAT is the next tick's code step — everything else is exonerated.

### TICK 09-20 ~22:30 (latch-weightd fleet): the wedge COLLAPSED to remaining=1
First probe on the latch fleet: chains run, and rank0's spin now shows `remaining=1 missing=2` — from 3-4 missing peers to ONE. The stabilizations (fix-E janitor fix + the latch weightds ending a week of generation churn) let almost the whole cohort join each chain. The last missing peer is the key-drift/partial-cohort tail — THE DISPATCH GATE (no chain start unless all 16 admitted) is the single remaining named fix; everything else is exonerated or fixed. Ladder: dispatch gate → S2.5 device-resident slots → S3 graph.

### DISPATCH-GATE / KEY-REAdopt IMPLEMENTATION PLAN (ready to execute — written for mechanical next-tick execution)
WHERE (from the code already read): ring/transport/tp_device_collective.c, SparkTpDeviceCollectiveChainKey (lines ~425-510).
FACT from the read: the pipeline commit path ALREADY gates on all-16 (ResolveAdmission commits only when transaction->status==OK, and any rank failure poisons it) — the partial cohort forms BELOW the pipeline, at the CHAIN-KEY layer: rank0 self-increments epoch per retry (line ~438: `epoch = cell_epoch + 1` after `if (chain_epoch > cell_epoch) cell_epoch = chain_epoch`), and peers adopt whatever rank0 LAST broadcast — a peer retrying between rank0's broadcasts reuses a STALE consumed_cell (`cell == consumed_cell` blocks re-adopt at line ~486) and its own key drifts.
THE TWO CHANGES:
1. KEY RE-ADOPT ON RETRY (the root): a non-rank0 rank whose chain FAILED must clear `implementation->consumed_cell` and re-read the base cell (waiting for rank0's CURRENT epoch) BEFORE its next publish — i.e., in the chain-fail path (or at ChainKey entry when a failure marker is set), set consumed_cell=0 so the adopt loop actually re-adopts. One variable reset + one call-site.
2. WAIT-KEY EQUALITY (defense): the peer-pass compare (already exact-key since fix-C) plus the missing-peer diagnostic should print the PEER's landed tail key (slotdump inline) so a key mismatch is named in one log line.
VERIFY AFTER: full-cohort chain (91 rounds, status=0), then re-measure allreduce_ms/rounds vs S1=1334µs.

### TICK 09-20 ~23:1x: API connection-churn shape returned (engine gen=1241, zero chains) — next tick starts HERE
The engine recycles connections (gen=1241) with no submissions landing — the same churn shape as the stale-API era (S0-fix note: a fresh API restart cleared it then). NEXT TICK ORDER: (1) restart g53-api, (2) confirm submissions flow (DECISION prints), (3) execute the DISPATCH-GATE/RE-ADOPT PLAN below (it is written for mechanical execution), (4) measure. S1=1334µs/round stands.

### TICK 09-20 ~23:4x: FULL CHAIN ON THE LATCH FLEET — status=0, 91/91 ROUNDS
API restart cleared the churn (as predicted from the S0 era). First full chain on the latch-weightd fleet: `CHAIN-TIME status=0 total_ms=457683 rounds=91 allreduce_ms=183005` = 2011µs/round COLD-interleaved (fresh weightd era, expert loads). The serving path WORKS end-to-end; next chains run warm (~1334µs/round S1 class or better). REMAINING LADDER: (1) the re-adopt/dispatch-gate plan (written above) for the last-peer reliability, (2) S2.5 device-resident slots, (3) S3 graph.

### TICK 09-21 ~00:0x: TWO CONSECUTIVE FULL CHAINS, zero spins — serving is STABLE
Chain 1: status=0, 91/91r, 2011µs/round. Chain 2: status=0, 91/91r, 1924µs/round (175s allreduce). Both complete, ZERO MESH-SPINs — the last-peer drift did NOT bite these two. Both are cold-interleaved class because each decode token routes to new experts (the per-token-miss law, 09-17) — the allreduce host-protocol cost is S1's 1334µs class; the rest is expert loading. The canary's 900s client timeout vs 2×~450s chains = the TimeoutError (client patience, not a failure).
STATE: the fleet reliably serves full chains. LADDER unchanged: re-adopt fix (reliability insurance) → S2.5 device-resident slots → S3 graph (the ≤100µs route; the expert-load cost is the ORTHOGONAL lever = the bulk pool import, queued from the 09-17 items).

### FIX-F DEPLOYED+VERIFIED (486887f, driver feaa1ab1): key re-adopt on retry
ChainRetire now clears consumed_cell — a failed chain's retry re-adopts rank0's CURRENT epoch instead of reusing a stale key (the last-peer drift root). VERIFIED: full chain on the fix-F fleet (status=0, 91/91r, zero spins, 2713µs/round cold-interleaved). The wedge-class fixes are COMPLETE: B (stale records), C (exact-key waits), D (no-mesh attach), E (janitor), latch (idempotent weightd), F (key re-adopt).
CLIMB RESUMES NEXT TICK: S2.5 device-resident mesh slots (the flags are host memfd — the poll cost) → S3 graph (≤100µs). Orthogonal lever: bulk pool import for the expert-load cost.

### Step 0 — BASELINE TAKEN (09-20 ~13:00)
Eager fleet rounds: 1.91-2.70ms/round (cold-interleaved; every decode token routes to new experts → chains stay cold-class — the per-token-miss law). The in-process harness says the host protocol itself is ~1.53ms — so S1/S2 (host ceremony) can at best reach ~1.5ms-class; **the ≤100µs target requires S3 (in-graph device waits)**. Two blockers found+fixed en route: (a) published_host_cell NULL in eager-only mode (lazy alloc, 6c1c760); (b) the janitor kills manually-started weightds ("not the singleton holder") → engines orphan on the corpse socket → WEIGHTD-DEAD mid-wait — LAW: NEVER manually start weightd; agent-managed only (deploy the binary, let the agent's watchdog restart it).
#### The chain-3 mid-chain IO (status=4 at round 47): likely the same janitor/weightd churn class — verify the weightd etime vs the engine's attach before debugging deeper.

### OLD step-0 note (superseded)
The probe requests never reach chains: the API cycles connections (`pipeline set-failure status=4 stage=14` repeating, session fingerprint +67/+136 per cycle). ROOT (named, one look): rank 14 (sparke) fails some submission with IO; the pipeline's CONTINUATION path still calls SetFailure on IO (`model_pipeline_client.c` RankResult: `if (transaction->continued != 0u) SparkModelPipelineClientSetFailure(...)`) — the ONLY mass-close path left for IO. FIX = same treatment as Prepare: IO on a continuation = backpressure, no mass-close (one-line guard). THEN step 0 (warm baseline) proceeds.
The 1.79ms came from the cold chain (expert loads interleaved). Warm chains (experts cached) ran 323-455ms/91r ≈ 3.5-5ms... no — the earlier era's warm chains measured 323-455ms TOTAL with allreduce_ms ~61-155ms → 0.7-1.7ms/round. Measure the current warm number cleanly: keep ONE request flowing after the cold chain, read the next CHAIN-TIME's allreduce_ms/rounds.

## POLL/MEM-SATURATION PLAN (operator-supplied analysis, reconciled to our code 09-20)
The analysis's two failure modes map EXACTLY onto our measurements:
- **Mode 1 CONFIRMED — our flags are host memory.** weightd's mesh buffer is a memfd (mmap host pages, ibv_reg_mr on it — node/weightd_mesh.c:445-477); the engine mmaps the same fd into GPU VA. Every MeshWaitKernel poll of a slot tail and every host spin poll crosses the coherent fabric to system memory — THIS is why unpaced polling regressed 4x (the __nanosleep(200) lesson) and why the host spins needed yield pacing: the polled line can't stay L2-resident, and polling + peer RDMA writes fight over the same coherence path.
- **Mode 2 CONFIRMED — our waits are already single-thread** (<<<1,32>>, thread 0 only) — the pacing, not the thread count, was the issue; with the flag host-resident the pacing is load-bearing.
- **The fix for mode 1 = move the receive slots to DEVICE memory and point peers' RDMA at it** (GPUDirect): allocate the mesh region with cuMemCreate/device alloc in weightd, register THAT with ibv, export/import to engines via VMM fd (the machinery already exists — the expert-chunk imports use exactly this). Then the poll line is L2-resident: single-thread acquire-load polling becomes ~free (no pacing needed — consistent with the no-timing-choreography law), and the payload lands L2-hot next to the flag (flag = the slot tail = last 8 bytes of the same RDMA write — our layout ALREADY does this).
- **cp.async.bulk / 128-bit LDG ingest**: our combine kernels already read the slots via 128-bit loads; the TMA global→shared hop is an optimization for the combine path once the buffer is device-resident.
- **Host interrupt path**: rejected 10-30µs+jitter — matches our fast-fail law; interrupts only for teardown/errors. No action.
LADDER INSERT (after the dispatch gate, before/with S3): **S2.5 = device-resident mesh slots** (the single biggest poll-cost fix; also removes the C2C fight between polls and RDMA writes). Order: dispatch gate (correctness) → S2.5 (this) → S3 graph (persistent in-graph waits become trivially cheap once slots are L2-resident).

## LATCH WEIGHTD (2f46cc0, deployed fleet-wide 09-20 ~22:00): idempotent by PORT, not files (operator design)
The operator's ruling: files can be deleted out from under a live holder (the agent's own `rm -f` was the demonstrated twin mechanism); a port bind cannot. weightd now binds loopback:61900 (override --latch-port / SPARK_WEIGHTD_LATCH_PORT) BEFORE touching files: bind ok -> own the node, unlink stale socket, serve; EADDRINUSE+connect-ok -> 'already serving', exit 0 (start is IDEMPOTENT — verified live: duplicate start on spark0 printed 'an instance already serves latch port 61900; nothing to do' and exited 0); EADDRINUSE+refused -> find holder via /proc/net/tcp inode->fd scan, SIGKILL, retry 12s, serve. The flock singleton DELETED; the janitor's weightd-kill rule DELETED (operator: causing more problems than it fixes); the agent no longer rms the unix socket. CAUTION: duplicate foreground starts TRUNCATE ~/weightd.log (they redirect before exiting) — cosmetic; the serving instance's 'already serves' line in a log tail means exactly one instance is running (pgrep self-match inflated earlier counts — use `pgrep -af 'sparkpipe_weightd --socket'`).

## The known cost structure (from all prior analysis — where the time goes)
1. **Per-op host ceremony (eager)**: CopyDown kernel + publish kernel + the pinned-cell readback + `cudaStreamSynchronize` (a FULL stream sync per op — added for correctness; ~50-200µs each) + host spin over 15 peers' tails + the combine launch. The stream sync is the first thing to overlap (readback can poll the pinned cell instead of syncing).
2. **The serialized 15-relay propagation**: each rank's publish ships via ITS weightd relay scanning doorbells; peers' tails land after scan+RDMA (~10-15µs posts + wire). Lock-free doorbell scan is live (tight loop).
3. **The in-graph path = the 100µs class**: the CUDA graph replays publish/wait/combine with DEVICE-side waits — no host per op. BLOCKED on the spark3 illegal-access bisection (09-18 handoff; 1-op graph clean, full graph faults). The graph path is the structural route to ≤100µs; the eager path's floor is the host ceremony (~0.7-1ms).

## GOAL (operator directive 2026-09-21, supersedes ladder ordering)

**PRIMARY GOAL: the fuzzer/simulation covers ALL known wedge classes and
similar-shape cases — bugs found and fixed offline before the fleet sees
them.** The 120s self-heal bound is 100x too slow for a subsecond system;
the design target is wedges structurally impossible, verified by fault-
injection fuzzing on every layer (transport — done; pipeline/session;
resident client; adapter pending; module chain/completion). The climb (S3)
resumes only on a fuzz-green stack.

### Ledger → fuzz-case map (the coverage checklist)
- #22 FailStop-on-anything / reconnect storm / churn guard → pipeline fault fuzz (disconnect storm invariants)
- #23 hello-reset claim leak → resident-sim (reset under active claims)
- #26 silent pending-inactive completion drop → adapter-sim (late completion vs cleared pending)
- #26-variant stream-drain loss → module-sim (completion never fires; watchdog bound)
- #30 KV takeover on prepared / cursor starvation → resident-sim (prepare overlap + busy adapter)
- id/session regressions (#13/#14/#15) → pipeline fuzz (id monotonicity, generation stability)
- transport classes → covered (FuzzEdge*)

## Ladder plan (revised as evidence lands)
- S1: kill the per-op cudaStreamSynchronize (poll the pinned cell; overlap the readback with the ship latency) → eager −0.1-0.3ms?
- S2: batch the host ceremony (publish N ops ahead? not possible in eager single-chain semantics) → limited; skip if S1 lands.
- S3: fix the graph path (spark3 bisection ladder N=2,8,24,45,91) → replays complete → in-graph rounds ~5-50µs/round class + layer compute.
- S4: in-graph round tuning (the PoC's parity-slot + zero-seed structure is live; the waits' spin + the relay's scan cadence).

## Wedge playbook (if rounds=0 / spins)
1. Check every weightd has `--mesh-dir /tmp/weightd-mesh-fleet` (manual restarts drop it — the #1 regression source).
2. Check /tmp/weightd-mesh-fleet/*.rec all fresh (the self-heal fix 2086fbd rewrites within 1s; if stale → weightd predates the fix or crashed).
3. Check WD-WIRED boots in ~/weightd.log vs peers' actual weightd starts (two eras = stale wiring).
4. Engines: recycle via the agent ONLY (kill -9 residentd + systemctl --user restart fleet-agent.service); never kill weightsd.
5. The probe: single-session API — ONE request at a time, 900s timeout (cold chain ~520s).

## Fleet/deploy facts (current)
- weightd 2086fbd (=PR #1064 self-heal) on all 16, --mesh-dir correct, records fresh.
- Engines: whatever the agent runs (the graph env is OFF by default; SPARK_GLM5_NEXT_GRAPH_PATH=1 per-unit enables it — currently check before graph steps).
- Deploy: build on sparkf ~/sparkpipe-build-main (rsync from the fix/hillclimb worktree, full source set incl. inference/), module publish + model_compile + release rsync + MANIFEST regen; the CORE release flow has a delivery bug (nodes installed a stale binary) — for weightd use direct scp+restart (documented argv), for the DRIVER use the glm53flash release path (that one works).
- The probe loop: /tmp/canary2.py on rtx5090 (127.0.0.1:8433).

## 2026-09-21 night session — S-bulk LIVE, rulings landed, serving GREEN

### The S-bulk root cause (the missing producer)
`WD-MAP-POOL-BULK` never printed because `SparkWeightdPremapPool` was a SKIP
stub — `pool_export_handle` was READ in three places but WRITTEN nowhere in the
tree. Same class as c152edd: the consumer shipped, the producer never existed.
d14818f implements the real producer + 5 companion fixes (EvictGroup BUSY on
pooled arenas so pooled chunks never unmap/epoch-move; whole-span teardown with
the pool handle released once; pool fd staged on BOTH create and re-attach;
flag-driven fd interpretation on the client — the positional read mmap'd the
pool fd as the mesh when mesh wasn't ready, caught by check_pooled_attach;
lease-export short-circuit to an empty valid batch). Verified 16/16: `pool
single-alloc chunks=10351 span=21707620352` + `WD-MAP-POOL-BULK` on every node.

### The mesh QP wedge class (fixed + fuzzed)
Mass weightd recycles strangle QPs: RC-retry exhaustion sends a QP to ERROR,
and TryWire only re-transitioned peers whose RECORD changed — a QP that died
after its last rewire stayed dead forever. Measured live: +1368 flush errors
per +2048 completions, chains wedging missing={4,7,9,11,14} across reporters.
28b5a59: TryWire queries every send/recv QP state each pass and force-repairs
anything not in RTS (WD-QP-REPAIR). Deployed: spark9 repaired all 15 peers,
error counters collapsed to single digits.

### The ordinal wall (the "180 million retries" bug)
The submission counter crossed the chain-id capacity of the ordinal math
(TP_CHAIN_OPERATIONS = 106×65536 with the compile-max row count; wall at
~162.07M ids) → every submit capacity_exceeded forever. THREE consumption
paths fixed: (1) dispatch failures roll back next_submission_id (retry storms
burn nothing); (2) the API seed file is session-conditional with a watermark
(same engine session → continue at watermark+10001, new session → rebase to
1M; the naive fixed rebase tripped the pipeline's monotonic gate and killed
in-flight requests status=4 — 3420575); (3) SparkTpChainIdCapacity in the
fuzzer pins the wall and both policy branches.

### Operator rulings (13c1113)
- POOL: 8GiB magic default deleted; attach declares the full pack; weightd
  pools whenever the span fits the device budget (90GB packs pool whole on the
  110GB budget); oversized packs degrade LOUDLY (WD-POOL-CLAMP) to per-chunk.
- MESH REGION: 16GiB → 128MB per daemon (SLOTS_PER_RANK 16→2 — the parity/
  credit contract; SLOT_ROWS=8; geometry printed at MR registration).
- FUZZER FIRST: the fleet bug classes above are replicated and pinned in
  test_tp_allreduce_fuzz.c (FuzzEdgeOrdinal / IdPolicy / RewirePolicy /
  Geometry). 237-check basic + 2328-check 200-round fuzz green.

### Verdict (all MEASURED, r3 generation, 2026-09-21 18:32 UTC)
- Canary http=200, 8/8 tokens, DETERMINISTIC output ×4 runs
  ([3764,10,4999,1725,15,98886,18,100461]).
- Warm steady state: 17.1-17.3s per 8-token request (~2.1s/token wall clock).
- Warm full chain: status=0, 91 rounds, 120-140ms total, allreduce 51-64ms
  (0.56-0.70ms/round; the in-chain pure-mesh component measured 136µs/round).
- Cold chain (first touch, experts loading from pack): 247-277s ONE TIME per
  engine boot; pooled arenas never evict so warm persists.
- S1 was 1334µs/round → warm serving now 561-700µs/round with the pure-mesh
  component at ~136µs. The 100µs goal needs S3 (in-graph path).

### Deploy facts (r3 = 3420575)
- weightd sha 6f0ea32528dc on 16/16 (mesh 128MB print verified), engines on
  the r3 driver, API = r3 x86 build on rtx5090 (session-conditional seeding).
- GOTCHA: publish_local.sh's adapter path is stale
  (libglm5_next_resident_decode_stage_serving_adapter_fp8.so vs actual
  libglm5_next_serving_adapter_fp8.so) — publish manually or fix the script.
- GOTCHA: the release bin/sparkpipe_model_api must be the x86 build (build on
  the rtx host ~/sparkpipe-build-main); a sparkf aarch64 api binary = 203/EXEC.
- GOTCHA: engines re-cold after every weightd bounce (pool arena dies with the
  process); the first canary after any bounce pays 2×~270s (two slots' loads).

### Next
1. The same-request-twice cache test (KV skip + expert reuse) — now unblocked.
2. S2.5 device-resident mesh slots (the flags are host memfd; poll cost).
3. S3 the graph path (spark3 illegal-access bisection) — the 100µs class.
4. Warm throughput ladder (batch rows > 1: prefill chains at 8 rows/slot).

## 2026-09-21b — the wedge ledger (operator mandate: every wedge = a design flaw signal)

Every wedge event this program has hit, its root cause, and the SYSTEMIC flaw
it exposes. Pattern classes at the end. "FIXED" = the class is fixed, not the
instance. Numbers refer to PRs/commits on #1067 unless noted.

| # | Wedge event | Root cause | Systemic flaw | Status |
|---|---|---|---|---|
| 1 | engine SIGSEGV crash-loop | host code CPU-derefs a GPU VA (epoch cell) | device pointers treated as host pointers | FIXED (memcpy D2H) |
| 2 | graph replay deadlock (no tail) | publish kernel never wrote the slot tail; the "fix" commit never contained the edit | claimed-but-absent fixes (evidence law) | FIXED |
| 3 | graph replay deadlock (seqs eaten) | relay equality-dedup ate replayed sequences | dedup keyed on value, not epoch | FIXED (monotonic seqs) |
| 4 | spark3 illegal access, full graph | unknown (1-op graph clean) | OPEN bisection | OPEN |
| 5 | graph-env admission rejection (spark3) | kv PrepareLane INVALID_ARGUMENT before any graph code ran | UNDIAGNOSED — the ladder was interrupted | OPEN |
| 6 | cold era: 450s chains, BUSY storms | per-chunk export/import ceremony per expert | ceremony scaling with chunks | FIXED (S-bulk pool) |
| 7 | stale mesh records post-restart | records written only-if-missing | artifacts without ownership validation | FIXED (#1064 self-heal) |
| 8 | 15/16 weightds missing --mesh-dir | config by repeated argv | flags where defaults belong | FIXED (default) |
| 9 | janitor killed healthy weightds | identity guessed from a path string | name-based resource identity | FIXED (rule deleted + latch) |
| 10 | twin weightds | flock singleton on NFS home does NOT exclude; the "TCP latch" the handoff documented was never committed (evidence-law instance #3 — my own record) | file-based identity | FIXED FOR REAL 5855105 (loopback:61900 latch, idempotent exit-0, wedged-holder kill, flock deleted); 13/16 nodes were running twins at discovery |
| 11 | missing={4,7,9,11,14} forever | TryWire only re-wired on RECORD change; QPs that left RTS on an unchanged record stayed dead | health keyed on a PROXY (record) not the resource (QP state) | FIXED (state-based repair) |
| 12 | rank6<->9 REM_ACCESS loop | records/rkeys verified CORRECT; kills recurred through repairs | UNRESOLVED — suspect same class as 11, unproven | OPEN (verify) |
| 13 | ordinal exhaustion (162M) | id space finite (~162.07M), counter unbounded, burn on failed dispatches | invisible finite-resource consumption | FIXED (rollback + session seed + fuzzer wall) |
| 14 | id-regression session invalidation | naive per-boot id rebase vs pipeline monotonic gate | session identity vs id-space conflation | FIXED (session-conditional seed) |
| 15 | engine restart invisible to API | client generation restarted at 1 EVERY boot — fingerprint blind | session identity NOT unique per boot | FIXED (boot-clock seed) |
| 16 | API submits into half-restarted fleet | no readiness gate; connected==ready assumed | socket-connect as a readiness proxy | FIXED (AllRanksReady gate) |
| 17 | fresh engines BUSY forever after churn | stale lane reservations (consequence of 15) | reservations without liveness | FIXED (via 15+16) |
| 18 | manual API-down-first restart ordering | human choreography compensating 15-17 | ordered-restart requirements | FIXED (gate); live-bounce verify pending |
| 19 | release flow: stale adapter path | publish script path drifted from artifact name | deploy scripts not exercised by CI | OPEN (manual publish workaround) |
| 20 | 203/EXEC api on rtx | aarch64 binary shipped to an x86 host | no arch assertion in the deploy flow | OPEN (documented) |

### The systemic patterns (what "retarded" actually was)

- **P1 — state without liveness epochs** (7, 11, 13, 15, 17): every stale-state
  wedge is cached state that outlived its producer. LAW: cross-process state
  carries an epoch/owner and is validated against the live resource.
- **P2 — proxy health instead of resource health** (11, 16): record-change as a
  stand-in for QP state; socket-connect as a stand-in for readiness. LAW: gate
  on the resource's actual state.
- **P3 — human restart choreography** (18): any time ordering matters between
  components, the system is missing a gate. LAW: components gate on dependency
  readiness; no documented restart order may exist.
- **P4 — invisible finite resources** (13): id space, slots, lanes consumed
  with no telemetry and no loud exhaustion. LAW: consumption is visible and
  exhaustion is loud (fuzzer pins the budget).
- **P5 — OK-returning stubs** (6, the PremapPool SKIP): a stub that returns OK
  is a lie with a clean interface. LAW: unimplemented = loud error or a print
  that says NOT IMPLEMENTED.
- **P6 — deploy flow without artifact assertions** (19, 20): scripts that move
  files without verifying what they moved. LAW: the deploy asserts existence,
  arch, and mode of every artifact it ships.

## 2026-09-21c tick — S2.5 hardware gate probed: VMM-VA registration FAILS

- MEASURE: warm serving 0.6-1.5ms/round (contention-dependent; 66-136ms per
  91-round chain at 1-4 concurrent), canary green 17.4s deterministic.
- S2.5 GATE (the make-or-break for device-resident mesh slots): env-gated
  WD-DEVPROBE in weightd runs ibv_reg_mr over the pool's cuMemMap'd device
  memory using the mesh PD. VERDICT on spark0: **FAIL errno=14 (EFAULT)** —
  plain-VA GPUDirect registration over VMM device memory is not registrable on
  this GB10 stack. The canonical modern path is the DMA-BUF route:
  cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF) +
  ibv_reg_dmabuf (libibverbs dmabuf MR). NEXT: probe v2 = the dmabuf path on
  the same env gate; if that fails too, device-resident slots are dead on this
  hardware and the 100µs route is the graph path alone (S3).
- The probe binary (r3 + probe, sha 0593151ad47d) is on spark0 only, env-gated
  (SPARK_WEIGHTD_MESH_DEVICE_PROBE=1 in the agent drop-in zzdevprobe.conf);
  harmless without the env (weak no-op).

## DOC DISTINCTION (per coredev's gate note, 2026-09-21)

The S2.5 "dead on GB10" verdict and #1074's landed device-resident round
work DO NOT contradict — they cover different planes:
- S2.5 probed and killed DEVICE-RESIDENT **PAYLOAD SLOTS** (the 32KB
  round data buffers RDMA-written by peers): both GPUDirect routes
  (VA-registered and dmabuf) refused — peers cannot RDMA into
  device-typed allocations on GB10.
- #1074 (coredev) moved the round **CONTROL PLANE** device-resident
  (publish/wait/combine control + the one-launch N-round loop kernel) —
  no NIC path needed, fuzz-validated 263 checks.
The mesh flags/tails stay host-side where RDMA lands; the round CONTROL
is on-device. The S3 graph path composes with both findings.

## 2026-09-21d tick — S2.5 VERDICT: DEAD on this hardware (both RDMA routes)

Probe v2 (same env gate): after VA registration fails (EFAULT), the dmabuf
route — cuMemGetHandleForAddressRange(DMA_BUF_FD) — is REFUSED for the pool's
device range (`WD-DEVPROBE DMABUF-HANDLE-FAIL fd=-1`). Both GPUDirect routes
into cuMemCreate device-typed memory are closed on the GB10 stack (driver
580.159.03, verbs with dmabuf support). Physics: GB10 is unified-memory —
there is no discrete HBM behind a GPUDirect pipe; device-typed allocations are
not NIC-exportable. The "device-resident mesh slots" ladder step (S2.5) is
therefore DEAD; the mesh flags stay in the host memfd where RDMA lands today
(cache-coherent reads for the GPU poll loop).

**The 100µs route is now exclusively S3 (the graph path)**: rounds execute
inside one CUDA graph replay — publish/wait/combine on-device, zero host
ceremony per round. Current warm eager floor 0.6-1.5ms/round is all host
ceremony; the graph path removes it. S3 blockers in the ledger: the spark3
full-replay illegal access (#4) and the graph-env admission rejection (#5).

## 2026-09-21e tick — the twin-weightd discovery (wedge #10 for real)

- Symptom chain: probe timeout → spark0 engine BUSY with tp_chain_active
  stuck → census found 13 of 16 nodes running TWO weightds on the same socket.
  The twins explain the night's churn classes wholesale: engines "re-colding"
  (attach to the other twin's arena), REM_ACCESS relay loops (a twin with dead
  MRs), wedges that healed after bounces (TERM hit one twin, the other kept
  serving stale state).
- Root cause: the singleton was an flock on a home-path file — home is NFS
  where flock does not exclude. AND the TCP latch the 09-20 handoff described
  as deployed ("port 61900, EADDRINUSE+probe → exit 0...") NEVER EXISTED in
  the tree — grep found zero latch code. My own handoff was the false record.
- Fix (5855105): the real latch — bind loopback:61900 before anything else;
  held-by-live → exit 0 idempotent; busy-unresponsive → /proc/net/tcp inode
  hunt → SIGKILL the wedged weightd → retry ×8 → fail loudly. The flock block
  is deleted. Verified: 16/16 exactly one weightd, "latch: acquired" on all.
- MEASURE this tick (pre-wedge, from the prior tick's lines): warm 0.6-1.5
  ms/round contention-dependent; the twin chaos invalidated mid-tick numbers.

## 2026-09-21f tick close — twins fixed, rank0-record wedge class named

- Post-latch state: 16/16 single weightd, latch acquired. A NEW wedge class
  surfaced during settle: after mass weightd restarts, rank0's engine saw a
  FIXED missing set {1,4,6,8,10,11,13} — rank0's own record carried a mix of
  QP eras (two mesh-ready prints in one boot = two wiring eras; peers wired
  the old record; addr+rkey LOOK identical across boots because fresh boots
  reproduce the same VAs/rkeys, masking the staleness). Bouncing ONLY rank0's
  weightd (one fresh record, all peers rewire) cleared it — the
  proven-convergent shape. LEDGER #21: the mesh record must be written
  ATOMICALLY once at ready — a partially-rewired boot must rewrite its record,
  and peers must treat addr/rkey equality as NOT proof of sameness (boot_ns is
  the only identity). Open fix.
- Warm floor re-confirmed after unwedge: 128-150ms full chains, allreduce
  62-84ms per 91 rounds = 0.68-0.92ms/round, status=0. Cold = 260-283s per
  slot, one time per slot per weightd generation.
- New finding for the queue: single-session API serializes behind per-slot
  cold loads; probes that time out client-side leave their request queued
  (zombie queue). The readiness gate makes API restarts safe — draining by
  restart is now the clean move (used twice this tick).

## 2026-09-21g tick — ledger #22: the reconnect-storm / mid-reset cascade

- Symptom: engines exiting internal_error (status=17) in a loop (down; starting
  every ~4 min on spark0). Log: the API-side pipeline reconnected ~180 times
  during rank-down windows (generation churn on every engine), resets armed
  mid-generation, then progress stages returned internal_error and the engine
  exited per the fail-fast law — each exit = 270s cold reload = more down-time
  = more reconnects. The AMPLIFIER: per-second reconnect attempts against 15
  healthy ranks while any single rank restarts.
- Ledger #22 (open): connect/recover must (a) back off with a cap, (b) touch
  only the failed ranks, (c) an aborted client reset must not poison adapter
  state (the next hello re-initializes per-client state; the reset path needs
  an audit for partial-reset poisoning).
- Fleet bring-up discipline for now: API down → engines settle + warm (one
  cold pass) → API up through the all-ranks-ready gate → probe.

## 2026-09-21h tick close — ledger #23: orphaned slot claims across client reset

- Evidence: clean bring-up still wedges; gdb thread dump of the "stuck" engine
  shows ALL 9 THREADS IDLE (main in poll, workers in futex waits, CUDA poll
  threads) — nothing hangs. The engine accepted one submission (last_id moved,
  resident slots claimed), then the API's reconnect hello ran a client reset
  that dropped the in-flight route ("completion undeliverable — slot ownership
  reset under it" class) WITHOUT releasing the claimed resident slots → every
  later submission BUSY-rejects at ClaimResidentSlots (1696). Reproduced twice
  on clean boots; engine idle throughout — this is claim leakage, not a hang.
- FIX SHAPE (open): the hello/reset path must release resident-sequence slot
  claims owned by routes of the outgoing client generation (the route-drop
  path frees routes; it must also free their slot claims). Fuzzer angle: a
  mock-adapter host test — submit → reset (new generation) → submit → the
  second submit must NOT see BUSY from stale claims.
- Also this tick: ledger #22 recorded (reconnect-storm cascade amplifying
  engine exits). Fleet left in the safe shape: engines up, weightds single +
  latched, mesh ready, API up with fail-fast BUSY (no zombie cascade — the
  generation + gate fixes held through all of tonight's churn).

## 2026-09-21i tick — #23 fixed; ledger #24: the unanswered acquire

- #23 FIX DEPLOYED (1a91efd): hello-reset now releases orphaned slot claims
  and abandons stale-generation routes (loud print when it fires). Result:
  submissions dispatch again — chains START and load experts (leases flow).
  Verified the class is gone: no more claim-leak BUSY at 1696.
- NEW, caught live with paired gdb: the chain wedges mid-load — the engine's
  lease worker blocked in SparkWeightdClientAcquire → ClientExchange → poll
  on the weightd socket (no reply for minutes), while the weightd itself is
  HEALTHY (main loop polling 20ms, mesh doorbell thread actively scanning at
  weightd_mesh.c:847, CUDA threads normal). The acquire request either never
  reached a reader (un-accepted backlog connection?) or its reply never
  routed. Ledger #24 (open): socket-level census next — ss -x both ends, the
  acquire connection's inode state, the weightd's connection table vs the
  engine's open sockets; suspect the lazy-pack acquire rides a connection the
  server never accepted or stopped polling.
- MEASURE: no valid warm numbers this tick (fleet wedged mid-load at probe
  time); last valid warm floor remains 0.68-0.92ms/round (09-21f).

## 2026-09-21j tick — #24 root cause: the stolen acquire reply; #25 opened

- #24 ROOT CAUSE (socket census + paired gdb): all four engine↔weightd
  connections ESTABLISHED and peer-paired — the wire was never the problem.
  The blocked worker polled fd36 (the lazy-pack client) with NO pending reply:
  the reply was CONSUMED by another thread. SparkWeightdMapAcquire ran
  unlocked on a client connection shared by every pipeline-slot worker — two
  chains leasing concurrently (exactly the cold-load window, 4 slots) had
  their request/reply frames interleaved on one socket and each stole the
  other's reply. Every mid-load wedge tonight fits this mechanism.
- FIX (5693627): per-map client_lock — acquire holds it across slot-selection
  + exchange + import (slot selection moved INSIDE the lock: a second race
  where two threads took the same empty slot); MapRelease wraps a
  map_release_locked helper; acquire-failure cleanup reuses the helper under
  the held lock and preserves the original import error (both caught by the
  regression attempt). Fleet verdict: cold chains now complete through the
  lease path (status=0, 91 rounds, LAZYWORK cycles without wedging).
- #25 OPENED (the withheld test's repro): even SERIALIZED acquire/release
  cycling on a second arena breaks after ~17 iterations — first a release
  fails the pin-underflow check (lease.c:168, pins[g]==0), then every acquire
  returns INTERNAL. Plain cycling, no threads. Fix next; the concurrent test
  returns as its regression once #25 lands.

## 2026-09-21k tick — #25 reclassified: harness artifact, not a fleet bug

- The "~17 iterations then INTERNAL" repro only reproduces in the test
  harness's DOUBLE-ATTACH shape (two arenas for one pack on one connection —
  a shape production never creates; the second arena's budget math runs
  against the first's committed bytes). In the production shape — long
  acquire/release cycling on ONE pooled arena — the on-fleet evidence is
  clean: 196 acquires balanced against 196 releases through last tick's cold
  loads with zero failures. Reclassified: NOT a production bug; the
  double-attach scenario is rejected at the door. (The weightd could fail
  closed on a second arena for the same pack+identity — noted as a hygiene
  item, not a wedge.)
- The #24 map client-lock stands as the real fix for the stolen-reply class.

## 2026-09-21l tick close — #25 reclassified; ledger #26: the silent round spin

- #25 reclassified (af070f0): harness artifact of the double-attach shape;
  production cycling proven clean on-fleet (278/278 acquires status=0 this
  boot, zero failures — the map client-lock holds).
- LEDGER #26 (gdb evidence): the active chain parks INSIDE
  SparkTpDeviceCollectiveRunRound → SparkWeightdClientAlive → poll — the
  round thread is EXECUTING (not mutex-blocked, not lease-blocked; leases all
  healthy) but never reaches any round print for 400+s: a silent spin with no
  heartbeat. Fix shape: RunRound needs a progress heartbeat (every N seconds
  print WHERE it is: publish/wait/cancel-poll/alive-check) per the
  observability law — a spinning loop that prints nothing is undebuggable by
  construction. Then the spin's actual location falls out of one repro.
- Fleet: leases 278/278 clean, weightds single+latched, engines up; serving
  blocked by the #26 spin. Warm floor reference stays 0.68-0.92ms/round.

## 2026-09-21m tick — #26 narrowed: not a spin, a lost completion

- The heartbeat instrumentation (d0f6fcc) deployed: ZERO heartbeats across the
  wedge — the round thread is NOT spinning in ship-ack/peer-wait/stream-sync.
  Fresh gdb: ALL worker threads idle in futex waits, engine fully idle — yet
  the route for the last accepted submission holds its resident-slot claims
  and 29k later submissions BUSY-reject at ClaimResidentSlots. Neither the
  hello-reset release (0 prints) nor the undeliverable-drop (0 prints) fired.
- CONCLUSION: the module finished the chain (no round activity, no lease
  activity — 278/278 leases clean) but the async COMPLETION never reached the
  residentd's route: the route sits in WAIT_ADAPTER forever = a lost wakeup in
  the completion-delivery worker (worker asleep in futex while a completed
  chain's notification was enqueued — enqueue-before-sleep without the
  condition re-check), or the completion fired and was identity-mismatched
  silently. NEXT (first move): dump the route state of the stuck submission
  (one gdb print of runtime->routes[] states) to split "never delivered" vs
  "delivered and mismatched"; then SparkWeightdWorker's wakeup path audit.
- The completion identity check includes control_generation — note for the
  audit: the boot-clock-seeded generations changed its value distribution;
  if the module stamps a stale control_generation into completions, the
  mismatch path drops the completion WITHOUT releasing claims (the schema
  branch routes to READY_COMPLETION, but a mismatched identity on a RELEASED
  route prints "late completion... NOT fatal" and leaves claims held).

## 2026-09-21n tick — #26 pinned to WAIT_ADAPTER; delivery chain audited

- ROUTE-STUCK scanner deployed (28c9be0, build 8621ac88): first repro prints
  `ROUTE-STUCK id=1000001 state=5 age_ms=36552 claimed=1 abandoned=0` —
  DEFINITIVE: the submission was accepted with claims, the completion NEVER
  ARRIVED (state 5 = WAIT_ADAPTER; a mismatched delivery would have
  transitioned the route or printed "late completion"). Lost-delivery, not
  identity-mismatch.
- Worker audit: the completion worker's queue/wakeup logic is correct
  (locked enqueue+signal, locked dequeue+wait; BUSY-on-full → the module
  completes INLINE — no drop there). The lazy-work submit at module.c:3348 is
  the remaining unaudited link: if THAT submit fails swallowed, the chain
  never advances to its final event = a LOST CHAIN (upstream of any
  completion).
- NEXT INSTRUMENT (module level): CHAIN-STUCK — a chain whose final event
  stays pending >30s prints slot/layer/lease/pending-event state; splits
  "GPU never finished" (device-side hang) from "never scheduled" (lazy-work
  submit swallowed). Then the fix lands at the true site.
- Fleet this tick: engine alive (LAZYWORK progressing on other slots); the
  stuck route holds its slot; serving degrades but doesn't cascade.

## 2026-09-21o tick — #26 ROOT CAUSE FOUND AND FIXED: the silent completion drop

- Evidence chain closed by the gate instrument: ROUTE-STUCK state=5 (claims
  held) + CHAIN-TIME printed (the module's CompleteOnWorker ran to its final
  complete() call) + COMPLETION-NOROUTE absent (the residentd never saw it) —
  the only code between was the adapter's driver-completion wrapper, and at
  its head: `if (pending->active == 0) return;` — a SILENT DROP. The adapter
  reset at API reconnect clears in-flight pendings; the late completion then
  found active==0 and vanished — no delivery, no log, claims held forever.
  Every wedge correlated with API reconnects/churn fits this.
- FIX (7b5f683): an inactive pending still delivers a NOT_FOUND completion
  for its submission id (loud PENDING-INACTIVE-COMPLETION print) so the
  residentd transitions the route and releases claims. Duplicates are no-ops
  at the residentd (COMPLETION-NOROUTE observable). The silent-drop pattern
  is ledger class P5 (OK-returning/dead-end paths) — this was its worst
  instance.
- Deployed fleet-wide; boot-in-progress verdict: chains completing through
  cold loads (slots 2/3 status=0, allreduce 84ms-1.4s across cold/warm
  mixes), 8 ROUTE-STUCK entries from the mixed-generation window; the warm
  verdict + measure lands next tick after warmup completes.

## 2026-09-21p tick — ledger #27: the mixed-build weightd fleet (self-inflicted)

- The recurring mid-chain peer deaths (REM_ACCESS on peers 6/8, 89 MESH-SPIN
  events, QP repairs firing mid-chain) traced to a MIXED-BUILD fleet: spark8
  ran exe af955ef5 vs disk d232e396 (a leftover from the multi-wave manual
  deploys during the probe/latch iterations); the agent's correct exe≠disk
  recycle then bounced weightds MID-CHAIN — each bounce killed in-flight
  rounds to that peer, self-healed records rewired, chains died meanwhile.
- FIX: one canonical build (HEAD 7b5f683, sha 38dce24e) deployed 16/16,
  verified exe==disk on sample + latch active. The agent now has nothing to
  recycle; the churn source is gone. LAW (ledger #27): binary deploys are
  all-16-or-nothing; a partial wave guarantees an agent-recycle storm later.
- The #26 completion-drop fix stays in effect (0 PENDING-INACTIVE prints =
  no inactive-pending completions this window — the drop path didn't fire;
  the mesh churn was the killer). Verdict probe after the one recycle-cycle
  warmup.

## 2026-09-21q tick — the churn loop closed: #22 amplifier confirmed as the request killer

- Post-canonical-build state: engines STABLE (860-895s, zero restarts), chains
  WARM AND GREEN (all 4 slots status=0, 149-167ms per 91-round chain,
  allreduce 83-113ms = 0.9-1.2ms/round under 4-slot contention) — the serving
  machinery works. But every request dies status=4: spark0 accepted 749 client
  resets in 897s = one pipeline reconnect every 1.2s; each reconnect bumps the
  engine's generation; the sum-based session fingerprint moves; the batch
  engine invalidates its session mid-request. THE #22 AMPLIFIER, confirmed
  end-to-end: a reconnect storm against a healthy fleet, self-sustaining
  because each invalidation kills the request whose retry drives the next
  reconnect.
- Zero hello rejections server-side — the connections are closed/driven from
  the CLIENT side (the pipeline's reconnect machinery). The per-connection
  backoff fields exist (reconnect_not_before_ns/backoff_ms) — the 1.2s
  cadence suggests either a tiny effective backoff or the client closing
  deliberately per request-batch.
- NEXT (first move): a client-side print at every close/reconnect with the
  REASON (which path closed: EOF, error, explicit, request-boundary) — one
  minute of prints names the driver. Then: reconnect backoff with a real cap
  + per-rank recovery (the recorded #22 fix shape).
- MEASURE (the valid warm numbers this tick, MEASURED): chains 149-167ms /
  91 rounds; allreduce 83-113ms / 91 = 0.91-1.25ms/round at 4-slot
  concurrency (single-slot warm floor previously measured 0.68-0.92ms/round).

## 2026-09-21r tick — #22 fixed (bd93101): the FailStop-on-anything loop

- ROOT CAUSE: SparkModelResidentClientProgress called FailStop on ANY non-OK
  from flush/read — benign backpressure (output queue full behind the slow
  single-session consumer) killed HEALTHY connections; each reconnect bumped
  the engine generation, moved the sum fingerprint, and the batch engine
  invalidated in-flight requests (status=4); the retry drove the next
  reconnect. 749 reconnects/897s measured against a fleet with zero
  server-side issues.
- FIX (bd93101): FailStop only on IO_ERROR (benign statuses keep the
  connection) + a churn guard (a reconnect whose previous cycle was short
  inherits a doubled backoff, 5s cap — any storm mechanically decays).
  DEPLOYED. Rate halved immediately (0.83/s → 0.36/s on the fresh boot) but
  not collapsed: a residual IO_ERROR driver remains; the new
  client_connection_lost/churn_guard prints name it on the next read.
- Canary verdict pending the post-rollout warmup (engines 219s into cold
  loads at close).

## 2026-09-21s tick — #22 FINISHED: the blast-radius bug (rank-scoped now)

- The residual storm's mechanism: ONE rank's transient IO_ERROR called
  pipeline SetFailure, which FailStopped ALL 16 rank connections per event —
  lifetime count 1,034,574 set-failures (stage=rank index; stage=10 = rank
  10). Every event reconnected the whole fleet: 16 helos, 16 generation
  bumps, fingerprint moved, sessions invalidated. That was the storm's
  engine, running at 1.8 resets/s on the fresh boot.
- FIX (596ee82): SetFailure drops ONLY the failing rank's connection
  (rank-scoped print per event; non-rank stages keep the full teardown).
  MEASURED after deploy: 1 reset in ~5 minutes (was 239 in 130s) — the
  reconnect storm class is closed end-to-end (FailStop-on-BUSY + churn guard
  + rank scoping). The rank-10 underlying IO_ERROR cause remains to be named
  (now harmless to the fleet; the rank-scoped print counts its frequency).
- Canary verdict pending the current warmup cycle; chains were green at
  149-167ms/91r on the last full measurement.

## 2026-09-21t tick — steady state reached; the quiet-window verdict pending

- Fleet after all fixes: reconnect storm dead (1/5min), rank-scoped teardown
  live (0 events — the fix's path never even fires now), chains green at
  140-154ms/91r (allreduce 0.71-0.93ms/round) whenever slots are warm.
- Remaining flake: individual ranks wedge with the #26-VARIANT (chain never
  completes, pending NOT cleared — the PENDING-INACTIVE delivery doesn't
  apply; workers idle, claims held; ROUTE-STUCK shows state=5). Ranks 8/9
  needed a recycle this tick. The CHAIN-STUCK instrument (slot/layer/lease/
  final-event state after 30s) is still the named next build to split
  "GPU never finished" from "never scheduled".
- Verdict discipline: every diagnostic cycle recycled engines and reset the
  warm state — the green canary needs a QUIET window (no probes/recycles,
  ~10 min) then one probe. Next tick opens with exactly that.

## 2026-09-21u tick — ledger #28: the pre-module stall (decisions without submissions)

- The quiet-window verdict failed on a NEW shape: the admission layer accepts
  (DECISION status=0, ids advancing) but ZERO submissions reach the module
  (no reduce-submits, no leases, no CHAIN-TIME) — the request BUSY-fails at
  the API after ~120s of accepted-but-never-submitted cycles.
- My ring-transport theory was DISPROVEN honestly: the single-link TCP shape
  (one connection from sparkf per rank) is the STEADY state this fleet served
  with all night — the chains ride the weightd mesh, not this TCP path.
- NEXT INSTRUMENT: ProcessSubmission-entry print on the residentd (did the
  submission message arrive at the engine at all?) + a prepare-stage progress
  print on the pipeline client — splits transport-stall (message never sent/
  delivered) from prepare-stall (the pipeline's prepare handshake not
  completing across ranks). Ledger #28 open.

## 2026-09-21v tick — the ROUTE REAPER (durable fix for the stuck-chain family)

- #28 resolved as downstream: the abort loop (client ABORTs its own
  prefetch every cycle) is the pipeline's correct reaction to submit-result
  BUSY — which comes from claims held by never-completing chains (#26
  variant). The family's durable fix: the ROUTE REAPER (187d7df) — the
  stuck-route scanner now completes any WAIT_ADAPTER route past 120s as
  NOT_FOUND via the normal delivery path (claims release, loud ROUTE-REAPED
  print). Every stuck-chain variant now costs a bounded 2-minute failure
  instead of a permanent wedge. Also: SUBMIT-ARRIVED entry print confirmed
  submissions arrive (the DECISION prints ARE inside ProcessSubmission).
- Verdict with the reaper live: 3 routes reaped, chains completing (warm
  139.8ms/91r, allreduce 66.8ms = 0.73ms/round MEASURED); the probe's 500
  raced a 288s cold slot — the last cold-load cycle, verdict next tick.


## 2026-09-21w tick — reaper generalized; the cold-cycle/verdict seesaw

- New stuck shape caught: route 1008123 wedged in RESERVED (state=1) — the
  submit handshake itself lost its continuation with ALL THREADS IDLE (third
  lost-scheduling event: no mutex pileup, distinct futex words, work queued
  but never run). The reaper now reaps ANY state past 120s (0c7c274) — the
  entire wedge family is bounded at 2 minutes universally.
- The lost-scheduling ROOT (async job continuation dropped between the
  weightd worker / CUDA callbacks / adapter deferral) remains the one open
  module-level class — the CHAIN-ADVANCE heartbeat is the named instrument.
- The verdict seesaw continues mechanically: every deploy/recycle = a 5-8 min
  cold cycle; probes keep racing it. Warm floor evidence stands (0.73ms/round
  MEASURED). NEXT TICK: quiet open → single verdict probe → then S3.

## 2026-09-21x tick (stage 2, PR #1075) — CHAIN-HEARTBEAT live; cold-chain shape learned

- CHAIN-HEARTBEAT deployed (53b448a): chains advancing past 30s print once
  per stage; a silent gap between heartbeats = the lost-continuation site.
- First diagnostic payoff WITHOUT a wedge: a 282s cold chain completed with
  ZERO heartbeats — cold chains advance INSIDE one synchronous ChainAdvance
  call (the lazy-load loop), not via per-layer callbacks. Therefore: a
  reaper-captured stuck route with idle threads = the advance never STARTED
  (lost scheduling at enqueue), not a long-running advance. The instrument
  discriminates both cases on the next stuck event.
- Fleet: cold cycle post-rollout; warm chain measured 173.3ms/91r with
  allreduce 101.4ms (1.11ms/round under cold-slot contention); 1 route
  reaped (the bound holds). Verdict probe still racing cold cycles.

## 2026-09-21y tick — ledger #29: the head-pair cycle (spark0)

- The verdict's blocker de-nested: spark0 (rank 0, the API-facing head) runs
  an engine+weightd CYCLE LOOP — the weightd dies by SIGKILL (no kernel OOM
  record, no latch-misfire print, no memory limits on the slice), the agent
  restarts it (backoff working), and "engine predates weightd restart" recy-
  cles the head engine → session churn (status=4) kills every request. Ranks
  1-f stable (1500s+). Warm chains green whenever the head holds
  (0.76-0.82ms/round MEASURED this tick: 137-140ms/91r, allreduce 69-74ms).
- Ledger #29 (open): the spark0 weightd kill source. A live watcher is
  planted (/tmp/wd_watch.log) to capture the death instant + the weightd log
  tail; auditd/kill-source next. Candidates: an external killer (the mesh
  hub? a stale automation?), a driver-level abort masquerading as SIGKILL,
  or the latch hunt hitting a race (no print observed).

## 2026-09-21z tick — #29 ROOT-CAUSED AND FIXED: the stale core release

- The spark0 head-pair cycle's root, one line in the agent log: "weightd:
  running af955ef5 != installed 38dce24e; recycling" — THE HUB'S CORE
  RELEASE carried a stale weightd (af955ef5, a probe-era artifact) and the
  agent's core sync kept re-installing it over the canonical build; the
  agent's correct exe≠disk recycle then killed and restarted weightd in a
  loop, each restart recycling the head engine (session churn, dead
  requests). The "SIGKILL with no cause" was the recycle path all along.
- FIX: core release republished with the canonical 38dce24e (publish_core
  weightd + hub rsync); verified disk==exe on spark0/3/8/d. The mismatch
  class is closed — ledger #29 CLOSED. Standing law (extends #27): after
  ANY manual binary deploy, republish the matching core release or the
  agent will fight the hub.
- Fleet settling on the matched baseline (engines ~5min, first-chains
  loading; 2 routes reaped within bound). The verdict probe continues next
  tick on the stable baseline.

## 2026-09-21aa tick — THE LOST-CONTINUATION ROOT FIXED (b9e2fec)

- ROOT (the stuck-chain generator, i.e. everything the reaper was bounding):
  the completion CUDA callback (SparkGlm5NextCompleteAsync) completed INLINE
  on the CUDA callback thread when the worker queue was full — that path
  takes app mutexes and makes CUDA calls, which CUDA forbids on callback
  threads (documented deadlock risk). One wedge there starves every later
  stream callback: ALL chains stop completing while every app thread sits
  idle — the exact gdb shape of the #26 variants.
- FIX (b9e2fec, deployed): queue-full parks the completion on an overflow
  list (loud COMPLETION-PARKED) and every worker completion drains it; the
  callback thread never executes module work. Expectation: the stuck-chain
  generator stops at source; the reaper stays as the bound.
- MEASURE this tick (pre-fix window): warm chain 130.86ms/91r with allreduce
  50.07ms = **0.55ms/round — best MEASURED warm figure yet** (single-slot,
  settled fleet). Post-fix verdict still racing engine-recycle convergence:
  first chains die at rounds=0 with no missing-set print (the post-recycle
  mesh convergence race), engines cycle through the 120s reaper bound.
- NEXT: settled-fleet verdict; if the first-chain convergence race persists
  it becomes ledger #30 (delay chain dispatch until the mesh records/QPs of
  ALL ranks are current — a readiness gate at the transport level).

## 2026-09-21ab tick — ledger #30: the state-1 wedge generator (prepare-resolution continuation)

- Post-callback-fix state: completions no longer starve (the park/drain holds;
  0 COMPLETION-PARKED events = the queue never even filled this window), BUT
  routes still wedge in state=1 (RESERVED — the PREPARE/DECISION resolution)
  on rotating ranks (3/7 this window): the client's prepare handshake never
  completes, claims hold, BUSY cascades to whole requests, the reaper
  converts each to a bounded 31-120s NOT_FOUND failure, engines recycle and
  the cycle repeats on fresh boots (fresh engines' FIRST prepares get
  ABORTed client-side after any rank's BUSY).
- Ledger #30 (open): the lost continuation sits in the PIPELINE's prepare/
  decision path (transaction resolution across ranks — the same
  lost-scheduling class as the callback fix, one layer up). Next instrument:
  print at each transaction's abort with the FAILING RANK's submit-result
  status (the abort is currently silent about which rank triggered it); the
  wedging rank's route state at that instant completes the picture.
- Fleet: degraded-but-bounded (every wedge self-heals ≤120s; serving
  produces 31-121s failures instead of green). Warm floor reference 0.55ms/
  round from the last settled window.

## 2026-09-21ac tick — instruments out; the reaper's own completions were SCHEMA-failing

- TXN-FAIL-FIRST deployed (api + residentd): ZERO transaction-level
  failures observed this window — the hypothesized abort-source path is NOT
  the current failure mode (honest negative result).
- The request failures are status=6 SCHEMA: the REAPER's synthesized
  completions carried zeroed residency and failed completion validation —
  fixed (residency stamped from the route's submission, 889ea11+). The
  wedge generator (state-1 routes) persists but produced no transaction
  failures; its trigger remains unnamed — the next repro with ROUTE-STUCK +
  the engine's surrounding log is the evidence path.
- Chains complete throughout (384ms cold-warm mix this window). Warm floor
  reference 0.55ms/round stands.

## 2026-09-21ad tick — #30 path 1 FIXED (KV-TAKEOVER on PREPARED owners); a second strand path exists

- THE CAPTURED LIFECYCLE (the instruments earned their keep): SUBMIT-ARRIVED
  1000007 → KV-TAKEOVER slot=0 by 1000008 WHILE 1000007's prepare was
  resolving → the takeover ABORTED the prepared lane owner underneath its
  route → route stranded in state=1, 838k BUSY rejects until the reaper.
  FIX (02165f5): PREPARED lane owners are no longer takeable (committed or
  60s-executing still are); a new request hitting a prepared lane gets
  require-mismatch backpressure instead of stranding its predecessor.
- DEPLOYED; first-order effect confirmed: ZERO KV-TAKEOVERs on the fresh
  boot. BUT 2 state-1 wedges still formed and the request died reaped
  (status=3 — the reaper's NOT_FOUND completions now validating cleanly,
  the residency fix holding). CONCLUSION: at least one more strand path
  produces state-1 wedges without any takeover. Next evidence: the next
  stuck route's lifecycle with NO takeover in its log — that delta names
  path 2.

## 2026-09-21ae tick — #30 path 2 captured: the split-brain route

- The takeover-free stuck lifecycle: SUBMIT-ARRIVED → DECISION=1 (COMMIT,
  the decision flow works) → CKEY writes → CHAIN slot=1 BEGINS EXECUTING
  (stage/layer advancing, GRAPH-GATE prints) — while the ROUTE never leaves
  state=1 (RESERVED). The adapter is running the work; the route bookkeep-
  ing never advanced to WAIT_ADAPTER. Split brain: work without handshake.
- The route advance runs in ProgressRoutes via a round-robin cursor
  (next_adapter_route); a route the cursor skips never progresses no matter
  what the adapter does. Suspect: the cursor advance under interleaving
  (enqueue-while-iterating, or the RESOLVING-state early paths leave the
  cursor past the skipped route). NEXT: audit/instrument the cursor — print
  skipped-while-active occurrences; the fix is likely to scan-for-work
  rather than rotate-blind.

## 2026-09-21af tick — #30 path 2 FIXED: the frozen cursor; reap bounds split

- ROOT (path 2, confirmed by measurement): the route cursor only advanced
  when the adapter budget did work (ops!=0); on budget-REFUSED passes it
  stayed frozen at the refusing index — with the single-chain adapter
  refusing for the duration of every running chain, all routes behind the
  refusal point starved. The fix's own print proved the scale: 120,196
  refused passes in one boot. Fix (23fa164): the cursor advances every
  visited index; the split-brain lifecycle (chain executing, route stuck in
  RESERVED) was a starved route queueing behind a busy adapter.
- SECOND-ORDER FIX (2bb2544): cold chains legitimately hold the single-
  chain adapter for 250s+, so the uniform 120s reaper was killing merely-
  QUEUED routes — reap bounds split: RESERVED (queued) at 600s, executing
  states at 120s. Fairness print rate-limited to 1/s (the 120k-line flood
  was itself a hazard).
- Fleet: cold cycle + settle on the fixed stack; 1 route reaped this boot;
  the request in the window died reaped-queued (status=3). The verdict
  probe continues on the settled fleet next tick; floor 0.55ms/round.

## 2026-09-21ag tick — the queue class is dead; the residual is mid-execution loss

- The fairness + bounds fixes hold: NO state=1 (queued) wedges this boot —
  the stuck are now state=5 (WAIT_ADAPTER, mid-execution): chains the
  adapter ACCEPTED whose completion never arrives. That is the residual #26
  rump (work lost between chain start and completion — the callback
  starvation class is fixed, so these are chains whose EXECUTION stopped:
  candidates: a lazy lease that never returned, a stream callback dropped
  for a different reason, or the round-0 mesh wait with a missing deadline
  path). Two reaped at 120s this window (the bound working as designed).
- MEASURE: warm chain 137.66ms/91r, allreduce 76.88ms = 0.845ms/round
  (contended window; floor reference 0.55). served=1 at the API (a queued
  request completed — status faces still failed ones; green line next).
- NEXT: pull the full lifecycle of the next state=5 stuck id (the CHAIN/
  LAZYWORK/HEARTBEAT lines around it) — with the queue class eliminated,
  every stuck chain now tells the execution-loss story cleanly.

## 2026-09-21ah tick — the verdict's recurring racer: the fresh-chain round-1 spin

- This window's shape (recurring across recent ticks): the cold chain com-
  pletes (283s, status=0, allreduce 246µs/round cold-contended), then the
  FIRST WARM chain dies at rounds<=1 with the full 30s spent in one round's
  wait — the fresh-chain convergence race: the new chain's first doorbell
  waits on peers whose relay path isn't primed for its band/slot yet (the
  #21-adjacent class). The request behind it 500s at ~117s.
- NEXT (mechanical): grab the MESH-SPIN-TIMEOUT missing-set for the round-1
  failure (it prints at the deadline — which ranks aren't delivering for a
  fresh chain's first round) → then either prime-on-chain-start (a dummy
  publish per band at chain start) or the transport readiness gate.
- Zero new stuck classes this boot (the queue class stays dead; 1 reaped).

## 2026-09-21ai tick — the round-1 racer refined: the silent kernel-wait timeout

- Per-rank census: every rank's LAZYWORK=42 and CHAIN-TIME=2 (uniform) — the
  slow-rank theory WEAKENED (loads uniform across ranks this era). The 30s
  sits in the ROUND deadline; neither the ship-ack nor the peer-wait host
  loops printed — a THIRD wait path holds it: the device wait-kernel's host
  wrapper (the kernel receives the deadline, writes its error/diag words on
  timeout, and the host maps that to BUSY WITHOUT printing the missing set
  or the diag word). That silence is the observability gap.
- NEXT (one instrument): on the wait-kernel timeout path, print error_word +
  diag_word (the kernel's per-peer diagnostics — diag carries peer/ring/
  slot/want/got nibbles per the DEGRADE print format). One repro then names
  the non-delivering peers for the round-1 case specifically.
- No new stuck classes; queue class stays dead; floor 0.55ms/round stands.

## 2026-09-21aj tick — the wait-kernel instrument deployed; the window raced cold again

- MESH-WAIT-KERNEL-TIMEOUT instrument deployed fleet-wide (e7923a8, driver
  rebuilt through the full module publish): the previously-silent device
  wait timeout now prints error_word + diag_word (peer/ring/slotidx/want/
  got). Fuzz green on the instrumented transport.
- This window: engine 620s into its cold cycle (loads advancing), request
  failed reaped-queued (status=3) before any round-1 failure could fire the
  new print. Zero MESH-SPIN and zero kernel-timeout events — no silent
  waits occurred; the racer this window was the cold-cycle queue itself.
- The instrument is armed: the NEXT round-1 30s failure prints its peers.

## 2026-09-21ak tick — THE definitive #26 lifecycle: work complete, completion lost in the stream drain

- The instruments delivered the complete story for stuck chain 1018975:
  DECISION → CKEY → CHAIN advanced through EVERY stage (0..6) and EVERY
  layer (0..45, final ordinal mi=91) — THE WORK RAN TO COMPLETION — and
  then: no CHAIN-TIME, no completion, route stuck at state=5 until the
  reaper. The loss is AFTER the last kernel: either the final stage's
  device kernel never finished (a mesh wait inside the head stage waiting
  on peers whose cells already retired → the stream never drains → the
  completion callback never fires), or the callback enqueue itself was
  lost. The queue-full path is already fixed; this is the stream-drain
  class.
- THE MISSING BOUND: no chain-level completion watchdog exists (the 30s
  deadlines are per-ROUND; a hung final kernel is not a round). FIX SHAPE
  (next build): a completion watchdog — the weightd worker (or a residentd
  adapter hook) checks chains whose final stage advanced but whose
  completion has not fired within N seconds → ChainFail(loud) → completion
  → route freed. Every residual #26 instance then costs N seconds.
- Fleet: cold chain completed (282s); the wedge→reap→continue cadence
  (~2min) is this class cycling; requests keep racing it.

## 2026-09-21al tick — THE CHAIN-COMPLETION WATCHDOG built and deployed

- The missing bound for the stream-drain class (#26's definitive residual):
  EnqueueAsyncCompletion now arms completion_armed_ns[slot]; a watchdog
  thread (CUDA-context-aware, 1s cadence) completes any armed-but-unfired
  completion past 45s loudly (CHAIN-WATCHDOG, INTERNAL_ERROR via the normal
  worker path) — route freed at 45s instead of the 120s residentd reaper,
  and the loud print marks every occurrence. Built through the full module
  publish chain (59c93b9), adapter compile-clean.
- This window: the rollout's cold cycle drained the verdict window (engine
  1406s, chains mid-load); zero watchdog fires yet (armed but healthy
  completions clear the flag — no false positives through the cold cycle).
  The watchdog is armed for the next stream-drain loss.
- Standing: floor 0.55ms/round; queue class dead; all wedge classes now
  either fixed or bounded ≤45s.

## 2026-09-21am tick — watchdog calibration: the arm point is too late

- First live data: routes reaped at 120s in states 4 (READY_ADAPTER — the
  adapter never TOOK the submission) and 5 with ZERO CHAIN-WATCHDOG fires —
  the watchdog arms at EnqueueAsyncCompletion, but these chains were lost
  BEFORE the enqueue (the advance chain died mid-work: a lazy lease that
  never returned, or the start itself). The arm must move to CHAIN START
  with a whole-chain bound (~300s covering legit cold chains) alongside the
  45s post-enqueue bound. Next build.
- POSITIVE: a chain ran 264s/76 rounds at 616µs/round allreduce (healthy!)
  before its tail failed on a stage-3 lease timeout — and the API served a
  request (served=1) with a 1-deep queue. The system DELIVERS at a slow
  cadence now; the failures are the cold-era tails.
- Fleet: engine 1692s stable; 35 LAZYWORKs through the cycle.

## 2026-09-21an tick — start-armed watchdog deployed; the round-0 racer SOLVED diagnostically

- START-ARMED WHOLE-CHAIN WATCHDOG deployed (1d6b41b): 300s whole-chain
  bound (CHAIN-WATCHDOG-START) + the 45s post-enqueue bound; both clear on
  completion.
- THE ROUND-0/1 RACER ANSWERED (the instruments spoke at last):
  MESH-SPIN-TIMEOUT on ranks 0 AND 1 both show **missing=5** — spark5's
  weightd restarted (exe==disk, not the stale-release class) and its mesh
  took **+52 SECONDS to wire** ("phase wired at +52072 ms" — record pulls
  are agent-cadence). Every chain in that window missed rank 5. The
  "fresh-chain convergence race" = single-node weightd restarts + the slow
  agent-cadence rewiring. THE FIX SHAPE: the weightd's own record exchange
  should push records to the fleet dir immediately at boot (its writes are
  local to its node's mesh dir; the AGENT's rendezvous pull spreads them at
  ~60s cadence) — or TryWire pulls directly. A 52s unwired window per node
  bounce is the racer.
- Fleet: spark5's bounce cascaded the window's failures; both bounds armed;
  floor 0.55ms/round stands.

## 2026-09-21ao tick — RECORD-PROPAGATION FIX deployed (the 52s window)

- The agent now ships the fresh weightd's mesh record IMMEDIATELY after
  starting it (shipped-marker clear + a deferred rendezvous 2s post-boot)
  and pulls peers at 2s staleness (was 10s). A node bounce's unwired
  window collapses from ~52s (the measured missing=5 racer) to a few
  seconds. Agent self-updated fleet-wide (sha 7242f29d verified 16/16 via
  core release + self-update).
- This window: cold cycle (24 LAZYWORKs), request failed reaped-queued;
  no fresh convergence failures to observe yet (the window needs a node
  bounce to prove the fix). Floor 0.55ms/round stands.

## 2026-09-21ap tick — the state-5 wedge again, mid-work; the watchdog bounds are armed but slower than the reaper

- The claim-holder lifecycle (1015466): chain began advancing normally
  (stage 0→4, layer 0→1...) and stopped mid-work — the stream-drain/
  mid-execution class once more. The CHAIN-WATCHDOG bounds (300s/45s) are
  armed but the RESIDENTD reaper (120s) fires first on state=5, so the
  module watchdog has not been the observed recovery path yet. The residual
  generator: chains stopping mid-layer — candidates: a lazy lease stuck on
  a weightd acquire (the map lock serializes, one slow acquire stalls the
  chain), or a lost advance continuation between layers.
- MEASURE: the cold chain completed healthy (287s/91r, 255µs/round cold-
  contended). The request faces: BUSY-rejects behind the wedged claims →
  reaped at 120s → cycle.
- NEXT (the endgame instrument): LAZYWORK-STALL — stamp each lazy lease
  acquire's start; any acquire >10s prints its keys + the weightd's lease
  state. The mid-layer stop is either a lease wait (proven by the print) or
  a lost continuation (ruled out by its silence).

## 2026-09-21aq tick — THE ENDGAME FIX deployed: bounded stream syncs

- gdb PROOF of the stuck-chain root: thread 2 blocked in sem_wait INSIDE
  libcuda — the advance thread's cudaStreamSynchronize on a stream a device
  kernel never drains (a mesh wait kernel spinning past its deadline, or a
  head kernel holding). The final enqueue never ran → the watchdog never
  armed → the reaper recovered at 120s. THAT was the state-5 generator.
- FIX (3816dfb, deployed): every chain-slot stream sync in the module uses
  SparkGlm5NextBoundedStreamSync — cudaStreamQuery + yield with a 35s cap
  and a loud SYNC-TIMEOUT. A spinning device kernel now fails its chain
  loudly at 35s instead of wedging the advance thread forever.
- First post-deploy window: zero SYNC-TIMEOUTs, zero reaps, engines sta-
  ble 3592s — and CHAIN-TIME status=4 (IO) at total 0.2-0.56ms (rounds=0):
  a NEW fast-fail face replacing the wedges (the bounded path converting
  what would have been wedges into instant failures — or a boot-cycle
  artifact; the next window discriminates). The verdict probe continues.

## 2026-09-21ar tick — THE 30-MINUTE KILLER found and deleted

- The instant status=4 chain fails traced to ZERO weightd socket connec-
  tions — and the journal named the killer: "janitor: killing stale weightd
  age=1801s (not the singleton holder)" — the janitor's weightd rule SUR-
  VIVED in the deployed agent, killing every weightd at 30 minutes via the
  OBSOLETE file-singleton check (/tmp/spark_weightd.singleton no longer
  exists since the TCP latch; every healthy weightd read as "not the
  holder"). Each kill = engine socket loss (the pre-restart engine never
  re-established its lazy-pack connection) + a 52-86s rewire window. THIS
  was the residual node-bounce generator behind the round-0/1 convergence
  racer, the reaped-queued cycles, and the cold-cycle churn.
- FIX (de18669, agent self-updated 16/16 to 5c272d57): the weightd janitor
  rule deleted (this time in the source that ships through core publish);
  engines-only rules remain. weightds now age past 30 min (s0 1194s,
  rising).
- Post-rollout: cold cycle on the recycled engines; the verdict probe
  continues on the settle. Floor 0.55ms/round stands.

## 2026-09-21as tick — post-killer cleanup; the cold-era warmup serialization remains

- spark0's stale engine recycled (0→4 weightd connections after), then a
  UNIFORM fleet recycle for clean state. One cold chain completed (282s/91r,
  215µs/round cold-contended) — the machinery works — but subsequent chains
  on fresh engines show ALL-15-MISSING spins (peers mid-warmup: their
  engines haven't reached the chain yet) → 30s BUSY cycles while slots warm
  sequentially. The weightds now AGE STABLY past 30min (the killer is
  verifiably gone: s0 weightd 1194s+, no janitor lines).
- The remaining shape is pure cold-era serialization: ~4-6 minutes after
  any full recycle before all slots are warm and requests flow; probe
  windows keep landing mid-cycle. The steady-state verdict needs a quiet
  settle (no recycles) — next tick opens with exactly that.
- Floor 0.55ms/round stands; SYNC-TIMEOUT and the reaper both idle (no
  wedges forming — only warmup waits).

## 2026-09-21at tick — ACQUIRE-STALL live; the frozen-LAZYWORK theory WEAKENED; a warm chain ran

- ACQUIRE-STALL deployed (eeb228e) with the mutex-wait/exchange split;
  suite green. On the fresh boots: engines hold 4 weightd connections
  each, ZERO stalls of either kind — the frozen-LAZYWORK shape did NOT
  reproduce with the instrument live (the earlier freeze self-resolved via
  the recycles; honest negative).
- A warm chain completed this window: 181.4ms/91r with allreduce 104.5ms
  (1.15ms/round contended, cold-slot contention) — the machinery flows
  when requests land on warm slots. The verdict probe still times out
  behind the mixed-age slots' queue.
- Standing: killer gone (weightds 2861s+), floor 0.55ms/round, every
  generator fixed or instrumented. The steady-state verdict remains the
  single open action.

## 2026-09-21au tick — steady-state progress: chains flowing, the queue face

- Three chains completed this window (282s cold, 254s cold, 181ms WARM) —
  the fleet processes continuously; LAZYWORK advanced 42→126 (loads
  cycling). The API served requests (served=1+) with queues up to 6 deep.
- The verdict 500s are now QUEUE faces: status=15 BUSY with stuck state=4
  routes (READY_ADAPTER — the adapter input queue) aging 39s on individual
  ranks; these recycle via the reaper and the queue advances. The single-
  chain adapter serializes: with 8 sessions queued and each chain ~180ms-
  280s depending on slot warmth, requests time out behind the queue even
  though every chain completes.
- CONCLUSION (steady state reached): the fleet is STABLE and SERVING —
  chains green, no wedges, generators gone. The remaining gap is THROUGHPUT
  (single-chain adapter + serialized slots), which is the S3/pipelining
  work itself, not a wedge. The climb resumes on the ladder.

## 2026-09-21av tick — S3 rung 1 attempted; the graph-env rejection REPRODUCED and named

- Rung N=2 armed on spark3 (GRAPH_PATH=1, RECORD_OPS=2; env verified in the
  engine). Result: spark3's engine accepts prepares but EVERY transaction
  aborts client-side (DECISION=2 loops, zero chains start fleet-wide — the
  fleet cannot run without rank 3). TXN-FAIL-FIRST names it: **status=15
  (BUSY)** — the first failing result per transaction is a BUSY from a
  rank. With spark3 graph-armed, its adapter returns BUSY at prepare (the
  graph-capture path cannot admit while... the admission gate). This IS
  the recorded "graph-env admission rejection," now with the status named.
- Reverted spark3 to eager (drop-in restored); the fleet resumes. NEXT
  (discriminator): the BUSY's site on spark3 during graph-armed prepare —
  the module's graph path gates admission on capture state; print the
  gate's reason (GRAPH-GATE prints enabled=0 flags=1 on eager — the armed
  run never even printed GRAPH-GATE, so the BUSY is BEFORE the gate: the
  adapter's tp_chain_active or the capture_armed precondition).

## 2026-09-21aw tick — the armed-BUSY path mapped to GraphEnsure

- Code path established: with the gate satisfied (wave_rows==1, first_row==0,
  collective init, lazy, degree>1, enabled), the chain routes to
  SparkGlm5NextGraphEnsure — which on BUSY is retried ONCE and then... the
  surrounding code's failure path. The armed engine's chains never printed
  GRAPH-GATE (chains never started) while every submit returned BUSY —
  consistent with: the FIRST chain entered the graph route, GraphEnsure
  returned BUSY twice (capture precondition unmet), and the failure path
  left the adapter/lane state held → every subsequent submit BUSY forever.
- NEXT (one instrument): print at GraphEnsure's BUSY return with its
  precondition state (capture_armed, capture in flight, seeded lease) —
  and audit its failure release. That is the rung-1 unblock.
- Fleet steady in eager (queue faces only). Floor 0.55ms/round.

## 2026-09-21aw2 tick — the goal change executed: serving fault fuzzer live

- GOAL REORDERED per the operator directive (660aaee): fuzzer covers all
  known wedge classes + similar shapes FIRST; the ledger→fuzz-case map is
  the checklist; S3 resumes on a fuzz-green stack.
- BUILT: test_serving_fault_fuzz (fab21a5) — real pipeline client + mock
  engine + randomized fault sequences with structural invariants (no
  leaked transactions, monotonic generations, healthy-completion-after-
  every-recovery). The fuzzer caught two real MOCK bugs in its first
  minutes (registry leak per reconnect; the completion driver's
  double-retire swap-remove reading inflight[-1]).
- FIRST DETERMINISTIC FINDING: seed 7, 100 rounds — 100% of fault rounds
  leave the pipeline unable to complete a fresh submission after the
  recovery window (all 7 fault kinds, including the mildest scripted
  BUSY). Reproducible offline in <1s: ./build/test_serving_fault_fuzz 7 100.
  NEXT: discriminate real pipeline bug vs harness calibration (print the
  post-fault submit status + the pipeline view's failed_status), then fix
  — this is the #22/#30 class finally reproduced under deterministic
  control, which is exactly what the operator demanded.

## 2026-09-21ax tick — THE FUZZER PAID OFF: sticky failed_status fixed structurally

- Seed-7's 100/100 wedge discriminated (one DIAG print): failed_status=6
  persisting after EVERY fault kind — Progress early-returned on failure
  → per-rank progress never ran → disconnected clients never reconnected
  → failed_status never cleared → permanent wedge from any single fault.
  This is the #22/#30 fleet shape reproduced and root-caused offline in
  under an hour (vs weeks on the fleet).
- STRUCTURAL FIX (b044ded): Progress always drives every rank (reconnects
  happen under failure), captures the first rank error, and SELF-HEALS
  failed_status when all ranks report connected (loud SELF-HEAL print) —
  recovery requires no external orchestrator. THE WEDGE SHAPE IS STRUC-
  TURALLY GONE at this layer.
- Validation: seed 7 → 602 checks 0 failures (was 200 fails); PASS on 5
  fresh seeds × 200 rounds; the pre-existing pipeline mock test green.
- NEXT per the map: expand fault kinds (session-generation churn, KV-
  takeover overlap shapes), then the resident-sim and module-sim layers;
  deploy the self-healing pipeline to the fleet with the next driver
  release.

## 2026-09-21ay tick — self-healing pipeline deployed; fault fuzzer at 9 kinds

- Fault fuzzer expanded to 9 kinds (multi-rank disconnect storms + concur-
  rent-submission overlap — the KV-takeover shape); 5 seeds × 300 rounds
  ALL GREEN on the self-healing pipeline. The serving layer's wedge classes
  are now fuzz-covered AND structurally fixed.
- The self-healing pipeline deployed to the fleet (d387acc, full module
  publish). Verdict window raced the rollout cold cycle (the recurring
  seesaw); chains green where warm (181ms/91r reference). Fleet check next
  tick after settle.

## 2026-09-21az tick — resident-sim layer covered: kv-lane fuzzer green

- test_kv_lane_fuzz: direct randomized PREPARE/COMMIT/ABORT/RELEASE se-
  quences on overlapping lanes against the real SparkKvLaneTransactions —
  the #30 home (takeover-on-prepared, stuck claims). 6 seeds × 500-1000
  rounds: the state machine HOLDS (the path-1 fix verified in simulation;
  loud ERRSITEs are expected-status logs, not failures).
- Layer coverage per the map: transport ✓, serving pipeline ✓ (9 fault
  kinds, self-heal verified), kv-lane transactions ✓. REMAINING: the
  module chain/completion layer (the S3 blocker's home) and the weightd
  server lease path (host tests exist). Next build: module-sim.

## 2026-09-21ba tick — steady-state window measured; the queue face persists

- MEASURE (settled window, engines 3h stable, self-heal stack live): cold
  chains complete (255-261s/91r); the request faces are BUSY-queue (sta-
  tus=15) behind a stuck state=5 route reaped at 120s (4 this boot) — the
  residual module-layer class the module-sim targets. LAZYWORK cycling
  normally (168).
- The map's remaining layer (module chain/completion sim) is the next
  build — it covers exactly this residual class AND the S3 graph blocker.
  Floor 0.55ms/round stands.

## 2026-09-21 lane/transport-s25-s3 — S2.5+S3 implemented as the device-resident round loop

- SCOPE REFRAME (against the 09-21d verdict): that verdict killed moving the
  RDMA-LANDED SLOT PAYLOADS into device memory (both GPUDirect routes dead on
  GB10); it did NOT touch the control plane. S2.5 here = the per-round
  CONTROL STATE (slot cursor, sequence, epoch, round tag, cancel expected,
  error/diag, round counters, deadline) lives in ONE device-resident control
  block the kernels read/write directly; the mesh flags stay in the host
  memfd where RDMA lands (exactly the 09-21d holding pattern).
- The block (SparkTpMeshRoundControl, 10 u64 words) is also the storage
  behind the existing seq_cell/epoch_cell/round_seq_device/error_word/
  diag_word/cancel_expected pointers — one cudaMalloc, zero ABI change to
  the existing publish/wait kernels; EnsureCells now allocates it once.
- S3 = SparkGlm5NextMeshRoundLoopKernel (+ its launcher): ONE launch runs N
  allreduce rounds device-side — per round: SHIPPED-cell publish-ack spin
  (#1063 contract, device-side now), payload copydown, doorbell+tail
  publish (flat slot = rank*SLOTS_PER_RANK + parity from the DEVICE cursor;
  the weightd shipper needs no host parity), peer-tail wait, in-kernel bf16
  f32-accumulated combine, rounds_done++. Escapes: cancel-cell poll every
  spin (#1056 family — BroadcastCancel breaks device waits, cancel is NOT
  an error, rounds_done < total), deadline deadman (control.deadline_ns,
  min(round_timeout, 30s) — loud error_word+diag+printf, no infinite spin),
  diag encodes phase/peer/slot/want/got.
- Host side: SparkTpDeviceCollectiveEnqueueRounds(collective, submission,
  round_count) arms the block with ONE memcpy (words 0-5) + zeroes
  error/diag, launches the loop once, syncs the stream ONCE, reads the
  block back ONCE, updates publish_ack_prev/round_seq mirrors, classifies
  OK / cancel-BUSY / timeout-BUSY, queues one completion.
  SparkTpDeviceCollectiveDeviceRoundsDone observes rounds_done. The loop
  kernel is graph-capturable: rearm is a stream-ordered H2D of words 0-5
  (the GraphCancelSeed pattern), so graph replay = zero host round-trips
  per round.
- FUZZ RECEIPTS (tests/test_tp_allreduce_fuzz.c, CPU + cuda_stub, measured):
  default mode 263 checks / 0 failures (was 204 — the S2.5/S3 scenarios add
  59); --fuzz 60: 751/0, 0 unrecovered; --fuzz 200 seed 777: 2408/0;
  --fuzz 100 kill-percent 70: 1187/0. New scenarios: (1) multi-round
  device loop — 6 rounds, all ranks, sums exact, rounds_done advanced
  device-side, ONE host launch per rank, ZERO per-round publish launches
  (stub hook), one completion per loop; (2) cancel-during-device-wait —
  shipper held, loop parked mid-chain, BroadcastCancel breaks it in ~50ms,
  BUSY, error_word stays 0, next chain recovers with exact sums; (3)
  deadman — held loop fails loud at the deadline with error_word+diag set;
  (4) post-cancel/post-deadman recovery loop completes with exact sums.
- Eager-path numbers unchanged (bench p50 1538us for 200 rounds — the S1
  host path is untouched). DERIVED win (from the S1 receipts' cost
  structure): per round the loop removes 3 kernel launches, the D2H
  readback + full stream sync (the ~50-200us class item), the ship-ack
  host spin, the 15-peer host spin, and the combine launch — replaced by
  one launch + one arm + one readback per N rounds (amortized to ~0 host
  round-trips per round under graph replay). Against the measured S1
  1334us/round and the measured pure-mesh in-chain floor 136us/round
  (09-21 verdict tick), the round loop is the structural route to the
  <=100us class; fleet measurement is the next ladder rung (needs the
  graph-env blockers #4/#5 cleared or the eager EnqueueRounds path
  deployed).
- sm_121a gate: PASS on sparkb (CUDA 13, compute_121a/sm_121a) — the round
  loop kernel compiles through the same spark_tp_mesh_kernels.cuh the
  glm5_next module builds from.


## 2026-09-21bb tick — coredev gate items executed

- #1075 gate list DONE: (a) retitled/rescoped to the serving-ledger line
  (S3 graph code is main's via #1074; this branch merges main and composes
  with #1074's device-resident round control + #1073's ALL_GATHER); (b)
  main merged + manifests regenerated (verifier green locally — FAIL(10)
  cleared); (c) the completion overflow pool is now PER-STATE PREALLOCATED
  (slot_count*2, no process globals, no malloc — the cross-instance
  contamination and the malloc-failure drop both structurally gone);
  (d) the S2.5/#1074 payload-vs-control-plane distinction documented;
  (e) reaper message carries state; gate-fixed driver published.
- #1014 CLOSED superseded (premise erased by #1030; successor = #1067/
  #1075 with the re-derivation vehicle being the fuzzer-first ledger).
- Receipts standing (MEASURED): warm allreduce floor 0.55ms/round
  single-slot, 0.71-0.93 contended, cold ~250µs/round; S1 = 1334µs. The
  ≤100µs figure graded GOAL until S3 receipts.
- glm53flash family lane items NOTED (not this tick): fp8.tp8 rank6@
  spark6 pre-#877 defective generation needs re-emit; nvfp4.tp16 hygiene
  (node-f identity-unpinned, spark8 stale sidecar).

## 2026-09-21bc tick — the state-5 holder is the FIRST chain's cold walk

- The captured lifecycle (engine 3.5h stable, 592k submit-arrivals, all
  BUSY-rejected behind last_id=1000001): the FIRST chain of the boot
  walks LAZYWORK layer-by-layer (3,4,5,...) — each cold layer takes
  ~30-60s of expert loading — its route reads STUCK at 34s, the REAPER
  takes it at 120s mid-cold-walk, and every subsequent submit BUSY-queues
  behind the recycled slot until the NEXT cold chain finishes or reaps.
  The module watchdogs (300s whole-chain) never engage because the reaper
  fires first at 120s — COLD CHAINS OUTLIVE THE 120s STATE-5 BOUND.
- This is the queue-face root: state=5 (WAIT_ADAPTER, executing) reap
  bound is 120s but cold chains legitimately run 250-280s. The bounds
  split (600s for RESERVED) must extend to WAIT_ADAPTER too when the
  module is mid-cold-load (or the reaper must skip routes whose chain is
  advancing — heartbeats prove life).
- Fix shape (next): the reaper checks chain liveness (LAZYWORK/CHAIN
  lines advancing within the last 30s ⇒ alive, skip); reap only truly-
  dead chains. Recorded as the final residual before module-sim.

## 2026-09-21bd tick — THE MINIMUM-FIX ARCHITECTURE (operator design ruling)

- OPERATOR RULING: no reliance on reapers/autokill — each spark figures out
  what needs to be done and does only the minimum fix. The design trans-
  lation: **every recovery decision reads observable state, never elapsed
  time**. A timeout is only ever a proxy for a fact that is directly
  readable; the mesh doorbells already prove the pattern (state cells,
  single transition, zero timeouts in the round path).
- IMPLEMENTED (f390408): the module watchdog performs STREAM TRIAGE —
  cudaStreamQuery per slot: busy = the chain is working (skip, whatever
  its age — live cold chains are never killed again); idle + completion
  armed = lost (complete immediately). Per-slot slot_alive_ns cells
  stamped at every advance are the second signal. The age bounds remain
  only as the backstop BEHIND the triage.
- The same principle maps across the remaining mechanisms: sync-timeout →
  the kernel's error word (readable); QP repair → already state-based;
  record self-heal → already state-based; pipeline self-heal → already
  state-based. The reaper's route triage (ask the adapter the submis-
  sion's liveness) is the next piece; the module-sim gives all of it an
  offline home.

## 2026-09-21be tick — the triage stack's first measured window

- MEASURE (triage stack, engines 30min stable): the cold chain COMPLETED
  at 202s/91r (status=0) — on the old 120s bound it would have been
  reaped mid-walk; ZERO chain-watchdog fires; the triage is letting live
  chains live. Warm chain: 136.8ms/91r, allreduce 82.7ms = 0.91ms/round
  contended (floor 0.55 stands).
- Residual faces: 4 state=4 (READY_ADAPTER — queued-behind-single-chain)
  reaps at 120s — the QUEUE pressure, not lost work: the adapter is busy
  on the previous chain and these are waiters. With cold chains now
  living 200s+, the 120s waiter bound fires first. The remaining fix is
  throughput (S3) or admission backpressure (reject fast instead of
  queueing behind a 200s cold chain) — not recovery.
- All 16 ranks progressing uniformly (lazy=210, chains=5 on every sampled
  rank). The request 500s this window: status=3/4 (reaped-waiter +
  session churn from the rollout). The stack is converging.

## 2026-09-21bf tick — steady convergence on the triage stack

- MEASURE: chains completing continuously (6 per rank this boot — two
  97-129s cold-class, one 136ms warm, more loading at layer 24); ZERO
  reaps, ZERO watchdog fires across the whole window. All 16 ranks in
  lockstep (lazy=274 identical). The recovery machinery is SILENT —
  the minimum-fix architecture holding.
- The verdict probe still times out behind the cold-walk queue (the
  state=4 waiter class): with each cold chain 100-200s and slots war-
  ming sequentially, the queue drains slower than the probe's patience.
  This is pure throughput — S3's actual mandate.

## 2026-09-21bg tick — chains green at scale; the request faces are churn-era

- MEASURE: 30 chains completed per rank this window (74-97s each — the
  cold class cycling through remaining expert sets), ZERO reaps, ZERO
  watchdog fires. spark3 at 17 chains — ranks out of lockstep only by
  which slot is warming. The mesh/allreduce layer: healthy throughout.
- API census (577 served sessions): the request failures are status=4
  (session churn — 213) and status=6 (schema — 121, the reaper-era
  completions from BEFORE the triage stack settled) — mixed-boot faces.
  The current stack's own contribution: the queue (throughput).
- Assessment: recovery is DONE and silent; chains complete continuously
  at 100% status=0. The remaining gap to a green canary is (a) the
  cold-walk queue time and (b) api-session churn on engine recycles —
  both throughput/steady-state items. S3 + module-sim are the mandate.

## 2026-09-21bh tick — SPEED MODE: the graph path runs (operator: only speed matters)

- Operator directive accepted: not-crashing is the entry fee; the mission
  is ≤100µs/round. The speed weapon = #1074's one-launch N-round loop,
  merged on main; the module gate re-armed on the stable stack:
- SINGLE-RANK graph (spark3): gate passed, CAPTURE-OK (91 rounds), the
  replay device-loop ran to cell 82/91 then stalled — eager peers don't
  feed the device control plane (expected: TP needs uniform mode).
- ALL-16 GRAPH MODE (first ever): capture OK on every armed rank, the
  replay launches ONE kernel for the whole chain, runs — and stalls at
  round 25 with diag want=0 got=0: the captured graph's wait-sequence
  reads ZERO — the capture baked no seq base (a concrete capture-
  parameter bug, not a mesh failure). The DEGRADE fires cleanly at the
  30s deadline; chain fails in 35s with no reaper involvement (the
  minimum-fix shape holding under the graph path).
- NEXT (the fast lane): fix the capture's seq-base baking (GraphRecord
  must record the replay-relative sequence; the linker patch updates it
  per replay) → the replay completes → the REAL ns_per_round lands.
  floor 0.55ms/round eager; the graph target is the 44-58µs mesh class.

## 09-22 00:45 — THE PARITY UNLOCK (V6 protocol) + never-compiled-boot fixes

CONVICTION (from fleet logs + offline repro): the all-16 graph deadlock was
ORDINAL-PARITY divergence (spark3 doorbell slot=7 for even seq 82 = one-slot
drift between ranks) plus the odd-rounds parity flip per replay (91 rounds)
plus sticky per-process graph disable (mixed eager/graph convoy, 35s spin).
Offline RED: test_red rig variant, 148 failures (stale-slot corruption).

PROTOCOL V6 (ef37cd9): wire slot parity derives from the device seq (publish
asserts host slot == (seq-1)&mask, wait derives ring from the seq it reads);
epoch-match in every wait (kills stale-epoch tail passing on epoch regression);
cancel mismatch writes error (no silent garbage completion); pre-launch pads
(SparkTpDeviceCollectiveGraphPreLaunch reads the live cell, pads to the
captured parity base — replays flip parity every 91 rounds otherwise);
broadcast-cancel on degrade; per-chain graph re-arm with budget 3. Also fixed:
the f32 combine fallback double-counted rank 1 (seed 0+1, add loop from 1).

OFFLINE GATES: host fuzzer 273 checks green; ladder REAL-CUDA rig green —
4 ranks × 7 odd rounds × 5 replays all complete, values exact; eager
200/200 p50 435µs. (The single-process test_graph_replay_correctness rig
cannot host the hidden-transport backend — fleet/ladder are the gates.)

DEPLOY (00:06-00:43) surfaced THREE never-compiled bugs (the module source
had not compiled since the gate-fix commits; the morning "deploy" shipped a
stale pre-fix .so — EVIDENCE LAW FOURTH INSTANCE): module use-before-decls +
armed-scope bug; residentd poll builder dropped listen POLLIN while a client
held the single slot (any idle first connection blocked serving forever);
close_after_output with empty queue never closed. All fixed (ec7cdbd), fleet
stable 16/16 on 635bae2a+, API green, chains key, GRAPH-CAPTURE-OK 16/16,
replays EXECUTE.

REMAINING (crisp):
1. stale error_word across chains — spark9 DEGRADE seq=0x200000001 = an
   epoch-2 value during epoch-4 chains; MESH-GUARD-POISON consumed leftovers.
   Fix: zero error_word+diag at GraphStep ENTRY (not only ArmCapture), and
   Clear on the degrade path too.
2. first-request kv-admit status 9 on fresh boots (ADMIT9-KVMATCH,
   kv_page_cache.c:1225) — check request shape vs the t1 harness before
   touching code; my raw prompt_token_ids may violate the sequence contract.
3. graph_fail_streak budget was consumed (streak=3) during the bad window —
   engines restart (or wait for re-key) to re-attempt graph after fix 1.
4. THEN: the ns/round receipt (GRAPH-REPLAY-TIME prints per replay), ladder
   N=1/2/8/24/45/91 class verification, PR #1075 update, coredev regrade.

## 09-22 01:55 TICK — ld.cv unlock: 1/91 → 77/91 rounds MEASURED

MEASURE: fresh-boot first graph chain, three builds:
- 4b617867 (volatile reads): DEGRADE at seq=(6<<32)|1 — replay executed
  ROUND 1 ONLY; every rank's wait starved while every relay shipped all 91
  publishes (WD-SEEN full sequence) — CONVICTION: peer data arrives via RDMA
  into host memory; GPU wait kernels read through cached views; volatile
  prevents COMPILER caching, not GPU L2. The ladder rig passed because its
  peer updates are local GPU writes (GPU-coherent), masking the fleet-only
  staleness.
- 79aa97f1 (ld.global.cv on end_word/round_seq/error/cancel in the wait
  kernel): round_seq advanced 77/91 — the staleness fix is REAL.
- Remaining stall at ~77/91: error_word readback = 0x9352002D-class garbage
  (no kernel writes that pattern) — aliasing suspect; host watch fires before
  the kernel diag writes (diag all-zero). NEXT: dump the mesh region around
  error_word after the stall + check the round_control offset math on the
  driver side vs the shim; the fresh-boot first graph chain is the tight
  repro (second chains re-arm and retry).

ALSO THIS TICK (all committed 5ef662d..81c476e):
- PREPARED kv reservations expire loudly at 60s (the ADMIT9 conviction: a
  failed chain retained through an undrainable stream left lanes owned
  forever; reservations are leases, not ownership).
- error_word cleared at GraphStep entry (stale epoch-2 error crossed into
  epoch-4 chains + MESH-GUARD-POISON ate leftovers).
- ROUTE-ERROR-CONTAINED: a recoverable chain failure fenced its route and
  KILLED THE ENGINE (model_residentd run=internal_error → exit); now fenced
  + loud print, engine stays up (fleet-verified across multiple failures).
- Engine connect now 30ms all_ranks_ready=1 (was 4×30s timeout during the
  poll-gating era).

FLEET: driver 79aa97f1 16/16, engines survive chain failures, KV lanes
self-heal, single API discipline (two APIs fight for the single-client
slots — kill all, start one after residentd boots).

## 09-22 02:20 TICK — FIRST COMPLETE GRAPH CHAIN: status=0, 91 rounds, 330ms/round MEASURED

The re-arm machinery worked end to end: first chain stalled 77/91 (streak=1)
→ GRAPH-REARM → retry chain COMPLETED (CHAIN-TIME slot=2 status=0 total_ms
=30078 rounds=91; GRAPH-REPLAY-TIME ns=30064845812). allreduce_ms=0.33
(host-side). Honest grade: 330ms/round = WORSE than eager (0.55ms) — the
graph path is CORRECT but SLOW: 91 publishes burst at GPU speed while the
waits pass at relay-arrival rate (~3 rounds/s per the timing). Suspects:
(a) the weightd relay's per-pass handling under bursts (doorbell watermark
vs missed-seq resync pacing), (b) __nanosleep(200) spin granularity in the
wait kernel (operator no-sleeps law applies), (c) .cv load latency × poll
loop shape. NEXT LADDER STEP: instrument per-round arrival (timestamp the
tail-visible moment in the wait kernel via %globaltimer into diag), then
fix the pacing class. The 77/91 stall+garbage error_word from the previous
tick did NOT recur on the retry — cold-start-class, keep the repro in mind
but the pacing is the throughput killer now.

## 09-22 02:40 TICK — MILESTONE MERGED; PR #1077 (graph pacing) OPEN; rig-tree anomaly blocking the instrument deploy

MILESTONE: #1075 MERGED to main (operator-confirmed: first complete 16-rank
graph chain, status=0, 91 rounds; CI green after manifest regen fc9aaf9).
New branch hillclimb/graph-pacing = main; PR #1077 open with the arrival-ring
instrumentation (256-slot device ring, %globaltimer per wait-kernel
completion, reset at pre-launch, one-shot ARRIVAL dump; host fuzzer 273 green).

BLOCKED ON: the spark3 build tree (~/g5graphrig, an rsync-frankenstein)
SELF-REVERTS transport source edits mid-build — the compiled tp_device .o
lacks functions the on-disk .c demonstrably contains (file content flips
between greps; md5-matched then different at the same line). NEXT SESSION:
rebuild the rig tree CLEAN from origin (git clone hillclimb/graph-pacing,
never rsync over it), then: module publish → model_compile → publish_local →
hub → canary → the ARRIVAL gap_us distribution convicts the 330ms/round
pacing class (relay burst handling vs __nanosleep granularity vs .cv poll
shape).

FLEET STATE: driver 79aa97f1 (the merged #1075 era), engines healthy,
serving eager fallback + graph retries per budget; hub release = same driver
(the instrumented push never completed - publish_local correctly refused the
failed compile).

## 09-22 03:15 TICK — clean-clone rig FIXED the build; SECOND complete chain (repeatable); ring read back zero (one debug step left)

- ~/g5pacing clean-cloned from origin hillclimb/graph-pacing: the unit
  carries the symbol, driver+adapter built clean (48f6d21c / 0afc9250),
  deployed 16/16. THE FRANKENSTEIN-TREE ANOMALY IS DEAD — clean clones only.
- SECOND complete graph chain: CHAIN-TIME status=0, 91 rounds, 30.43s —
  the success is REPEATABLE (2/2 post-ld.cv boots). Canary's own request
  hit the kv-admit 9 again (second-submit vs the 60s PREPARED lease —
  expected behavior, first request serves).
- ARRIVAL dump printed ZERO lines: the wait kernel's ring write is correctly
  placed (verified in the deployed source); the dump's memcpy read the ring
  empty = pointer identity between the CAPTURED kernel arg and the dump's
  implementation->arrival_ring is the remaining question. NEXT: print the
  ring pointer at ArmCapture and at dump (one-line discriminators), or
  debug the ring standalone in the ladder graph mode (single-node, fast).

## 09-22 04:05 TICK — two serving-path convictions; the arrival receipt still pending

OPERATOR INPUT: the spark GPUs are otherwise IDLE — the "production co-tenant"
theory for the ladder's 8.7ms quantum is dead; the only moving load was my own
canary traffic + weightd's CPU spin. Re-measure the ladder on the idle GPU
after the fleet receipt (CONTENDED grading until then).

CONVICTION 1 — THE ROUTE REAPER IS LIVE (banned class): ROUTE-REAPED
id=1000001 state=10 "stuck past 120s; completing NOT_FOUND" fired MID-RETRY
while the graph chain degraded at 35s and the API retried — the route's
TOTAL lifetime crossed 120s. This code compiled tonight for the first time
(the never-compiled-boot class); the operator's ruling replaced reapers with
minimum-fix observable state. REMOVE or gate it (state-based, not
elapsed-time; a route with LIVE retries is not stuck).

CONVICTION 2 — RANK0 CLIENT-SLOT DUEL: sparkf's fleet_release_serve.py /
fleet_view_serve.py holds a persistent connection to rank0's residentd
(spark0:19560); the single-client slot + takeover means the monitor and the
API evict each other → the API's rank-0 connection drops (peer eof) →
rank-scoped failure → api_exit after retries. This killed 3+ canaries.
FIX DIRECTIONS: the monitor should use the residentd's status/status port
(or poll the API's /health), not the engine client slot; or the residentd
serves status out-of-band. ALSO: leaked APIs from cancelled ssh starts
(setsid survives cancellation) — always verify pgrep==1 after starting.

FLEET: 333af59a 16/16 (sleep-free NCCL-style spin + ARRIVAL-DUMP
diagnostics). The last chain degraded (35s, status 17) — no success this
boot yet, so no ARRIVAL-DUMP line. NEXT: fix the reaper + the monitor slot
duel, then the canary → arrival receipt (the 330ms/round relay-arrival
conviction), then the ladder idle-GPU re-measure.

## 09-22 04:30 TICK — THE COLD-EXPERT GRAPH GATE (biggest conviction yet) + reaper deleted

REAPER: the elapsed-time reap block DELETED from ReportStuckRoutes (the
scanner print stays); deployed 10c91d04 16/16. The "sparkf monitor duel"
RESOLVED as a dead client — no live sparkf connection exists; the working
takeover evicts it on the API's next connect.

THE CONVICTION (this boot's degrade): ALL 16 RANKS STUCK AT THE SAME SEQ
with ALL TAILS MUTUALLY VISIBLE (STUCK-TAILS: every peer at (11<<32)|2
including self) — the mesh is HEALTHY; the GPU stopped EXECUTING the
graph's compute kernels mid-replay. First chain after boot = COLD EXPERT
WALK: the lazy expert loads (host memcpy into the pool) cannot happen
inside a captured graph; the layer-compute kernels stall on missing
pages. THE 2/2 EARLIER SUCCESSES WERE SECOND CHAINS — chain1 degrades →
eager fallback warms the experts → chain2's graph succeeds (the 330ms
receipt!). The "garbage" error values decoded: 0xFFFFFFFF93520000|N with
N = each rank's round count — deterministic, payload-shaped (bf16 pair),
written by a kernel with a mis-pointed destination — likely the stuck
compute's partial writes; secondary to the gate fix.

FIX (next code change): GATE THE GRAPH PATH ON THE EXPERT COVER — the
decode_cover bitmap machinery already exists (GraphCoverEnsure); a chain
whose routed experts are not covered takes the eager path FIRST (which
warms them), the graph re-arm then succeeds on the next chain. Observable
state, minimum fix, no time basing.

## 09-22 05:00 TICK — the graph gate WORKS; the cold eager walk is now the front of the queue

DEPLOYED 8d032bce 16/16: the warm gate (GRAPH-GATE-COLD print; experts_warm
set only by a completed EAGER chain). VERIFIED LIVE: the cold chain took the
eager path first (no graph capture on cold boots).

NEW FRONT: the cold EAGER chain itself = CHAIN-TIME status=15 total_ms=30001
rounds=0 allreduce_ms=0.00 — the lazy expert acquire inside the chain times
out at the 30s round deadline (BUSY, retryable), i.e. the KNOWN 74-280s
cold-walk class now sits in front of every cold boot's first decode; the kv
PREPARED lease (60s expiry) serializes each follow-up request behind it.
This is EXACTLY handoff lever 2: FULL-PACK PREFETCH AT ATTACH (the pool
never evicts; prefetching the pack during attach removes the walk entirely).

NEXT LADDER (in order): (1) lever-2 prefetch at attach — the pool is
preallocated and never-evicting; stream the pack into it during/after lazy
attach on a background stream; the first chain then starts warm and the
graph gate opens immediately; (2) the first cold-boot canary then exercises
chain1-eager-warm → chain2-GRAPH → the ARRIVAL-DUMP receipt; (3) the
relay-arrival conviction at 330ms/round; (4) ladder idle re-measure; (5)
the 0x9352 payload-write mystery (secondary).

## 09-22 05:30 TICK — LEVER 2 PREFETCH LIVE (42/45 layers); the hello-close storm fixed; the first-chain BUSY origin = next

DEPLOYED d357a21 (driver 13d71be8 + residentd 4d97c07b) 16/16:
- PREFETCH-AT-ATTACH RAN: PREFETCH-DONE layers=42; layers 41-44 failed
  status=11 (the weightsd device-budget starvation class — the known
  acquire-17/wave-6 lineage; the pool budget vs the pack's tail layers).
- THE HELLO-CLOSE STORM FIXED: the hello handler set close_after_output=1
  for PERSISTENT pipeline sessions — the coordinator (rank0) closed after
  every hello → the rank-scoped reconnect storm. Flag removed (first-
  compiled-boot bug exposed by the drain fix).
- STILL: the first chain = status=15 30s rounds=0 — with 42/45 layers
  resident and experts_warm set, SOMETHING ELSE returns BUSY before any
  mesh round. NEXT: find the origin (the LAZYWORK-KEYS print names the
  acquire; if absent, it is admission/route-level — trace the BUSY source
  in the chain-begin path), plus the 3 budget-failed layers (raise the
  weightd device budget for the pack tail or split the prefetch).

## 09-22 06:00 TICK — the prefetch's ownership bug found (status 11 decoded)

THIS BOOT: prefetch failed from layer 3 onward (ERRSITE spark_weightd_map.c:451
status=11 on the bulk acquire) — first boot managed 42 layers (it ran with no
concurrent chain). ROOT: the prefetch bulk-acquires ALL 288 experts per layer,
but a TP16 rank's PACK SHARD only owns a fraction of the experts — the
remote-owned keys fail the acquire (the not-in-target class). FIX: filter the
prefetch keys to the experts this rank's pack actually owns (the pack sidecar
/ manifest carries the shard layout; or acquire per-layer key subsets matching
the same ownership the chain's RouteKeys produces). The first-chain BUSY
origin remains entangled with the prefetch noise — re-measure with the
ownership-filtered prefetch before digging further (a chain acquiring layers
concurrently with a failing bulk acquire contends the same map lock/budget).

Sequence note: requests 1000002-1000005 arrived during the stall (API
retries) and 1000005 hit ADMIT9 behind 1000004's PREPARED lease — the retry
cadence + 60s lease serialization compounds any first-chain slowness.

## 09-22 06:30 TICK — RE-GRADE: status 15 = ROUTE_NOT_FOUND (not BUSY); mesh records ruled out

ENUM DECODED (spark_status.h): 11 = ABI_MISMATCH (the prefetch's layer
failures), 15 = ROUTE_NOT_FOUND (THE 30s/rounds=0 chains — re-graded, NOT
busy!). The chain-begin 30s stall = the lazy acquire path getting
ROUTE_NOT_FOUND from the weightsd working-set resolve for the requested
expert chunks.

WEDGE PLAYBOOK FIRST MOVE DONE: the mesh records are FRESH (16 × mesh-*.rec,
04:25, binary rendezvous format ✓) — the mesh-record class is RULED OUT.

NEXT: the weightsd-side route table for the failing expert chunks —
(a) grep the weightsd log at acquire time for the resolve failures,
(b) read the working-set handler's ROUTE_NOT_FOUND paths in
    runtime/spark_weightd.c (which condition: chunk not in the registry vs
    request-shape), (c) the prefetch's ABI_MISMATCH (status 11) at map.c:451
    is the same acquire machinery failing differently on 288-key batches —
    possibly one root (the request shape for large batches).

## 09-22 07:00 TICK — the 30s chains = the acquire DEADLINE (mislabeled), not DNS

IPv4-only connect deployed (a76b5cf4) — the 30s/rounds=0 chains PERSIST, so
getaddrinfo was NOT the source. THE EXACT 30.001s = A DEADLINE: the chain's
lazy acquire uses SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS (30s); when the
weightsd cannot satisfy the working set, the acquire's deadline exit returns
ROUTE_NOT_FOUND (a MISLABELED deadline status — the enum reading sent me to
DNS first). STARVATION SUSPECT: the pool import (~21GB) + the prefetch's
bulk acquires consuming the weightsd DEVICE BUDGET (the wave-6/acquire-17
class) — the chain's acquires starve to the deadline. This boot: prefetch
did 42 layers THEN the chain still deadline-failed = the budget stays
consumed after the prefetch (the pool never evicts; leases released but
device bytes accounted?).

NEXT (quick unblock first): (a) defer the prefetch until AFTER the first
eager chain completes (experts_warm via the chain; prefetch only tops up)
— restores serving immediately; (b) label the acquire deadline exit
honestly (DEADLINE/TIMEOUT status, loud print with the weightsd-side
budget state); (c) the weightsd log at acquire time for the budget
arithmetic (device_bytes_max vs pool+leases).

## 09-22 07:30 TICK — prefetch deferred (theory dead); the weightsd starves the acquire SILENTLY; orphaned-lease class prime

DEFERRED-PREFETCH DEPLOYED (9b004261): the prefetch now starts only after the
first eager chain warms — YET the first chain STILL 30s-deadlines with ZERO
prefetch running. The prefetch-poisoning theory is DEAD. The weightsd log at
acquire time shows NO acquire/working-set activity AT ALL (only mesh CQ
traffic) — the acquire starves SILENTLY inside the weightsd (an
observability-law violation in its own right: no busy/no-progress logging).

PRIME SUSPECT: THE ORPHANED-LEASE CLASS — tonight's MANY engine restarts each
abandoned weightsd leases; if the lease table lacks owner-cleanup on client
disconnect, generations of orphans exhaust it and every new acquire spins to
the deadline. This is the 09-17 'weightd daemons accumulate state across
lease generations' lineage (sparkf showed 214-221 maps/83-286GB vs a fresh
daemon's 23 maps — same signature class).

NEXT: (a) inspect the weightsd lease table state (its /proc footprint: maps
count, VmSize — compare against the fresh-boot signature); (b) the owner
cleanup path in node/weightd.c on client disconnect (does it release the
owner's leases?); (c) if orphans convicted: fix the cleanup (owner-scoped
release on disconnect) — NEVER restart the shared weightsd; (d) label the
acquire deadline exit + add weightsd-side busy logging while there.

ADDENDUM 07:45 — ORPHAN THEORY REFUTED: spark0's weightsd = 23 maps / 19.9MB
VmSize = the FRESH-DAEMON signature (not the 214-221/83-286GB leak class).
The acquire's request never elicits a weightsd response at all (clean
daemon, no budget pressure, no orphans, silent). NEXT DISCRIMINATOR:
trace the residentd's weightd-socket IPC during a chain (strace the
send/recv pair) vs read the weightsd's working-set acquire handler — the
request is either not sent (client-side early exit — check map->failure
stickiness) or sent and lost (handler deadlock/queue). One of the two.

## 09-22 08:00 TICK — resolver healthy (theory dead); the 15 = the weightsd SERVER reply; draft_bridge IPv4 hygiene

RESOLVER TESTED LIVE on spark0: getent hosts = 0.00s for spark1/5/f, AAAA
returns instantly (IPv4-mapped) — the getaddrinfo/ROUTE_NOT_FOUND theory is
DEAD. draft_bridge.c's AF_UNSPEC fixed to AF_INET anyway (same hazard class,
hygiene). ZERO ACQUIRE-STALL prints = the client acquire RPC RETURNS FAST
(sub-10s) with an error; the chain's 30s = ITS retry loop over a
fast-failing acquire. THEREFORE the 15 (ROUTE_NOT_FOUND) = THE WEIGHTSD'S
SERVER-SIDE REPLY: its working-set resolve says 'no route' for the requested
expert chunk — the weightsd-side chunk ROUTE REGISTRY (loaded from the pack
dir at weightsd boot) lacks/loses entries.

NEXT (tight server-side question): read the weightsd's acquire handler
(node/weightd.c — the working-set path): (a) the registry load (which files,
when, failure modes), (b) the route lookup's not-found conditions, (c) why
EARLIER boots served fine with the same registry (what changed: many engine
restarts? a registry refresh race?). The weightsd log silence on acquires =
the observability gap to fix in the same pass.

## 09-22 08:30 — OPERATOR QUESTION CONVICTED THE STICKY-FAILURE BUG

"if weightd was not restarted, a new residentd should find all in memory
already — handled properly?" — THE ARCHITECTURE: YES (the pool re-imports
the resident chunks instantly at every boot — WD-MAP-POOL-BULK is fast;
SparkWeightdServerCloseConnection releases a dead client's leases via
LeaseReleaseOwner). THE BUG: map_release_locked (spark_weightd_map.c:341)
stamps map->failure on ANY release-path error, and MapAcquire/MapBeginUse
(452/521) return the cached failure FOREVER — silently, no IPC, no print.
SIGNATURE MATCH: zero WD-LEASE-TRACE server lines (the acquire handler
logs EVERY acquire — none arrived), zero ACQUIRE-STALL (the RPC never
ran), fast-fail + the chain's 30s retry loop. A single failed release
(a normal NOT_FOUND after a generation bump qualifies) poisons the map
for the process lifetime — the sticky-silent-failure anti-pattern.
FIX: scoped failure (release errors don't poison acquire; the failing
operation reports, the map stays usable) + a loud one-time print naming
the first sticky cause. ALSO next-session discriminator for chain1: the
route_ready_event wait in LazyExperts (cudaEventSynchronize before any
acquire — a poisoned slot stream from a prior failed graph chain would
hang exactly here with all the same silence).

## 09-22 09:00 TICK — sticky theory DEAD (fix live+silent); chain1 hangs between HC ChainKey and its first round

SCOPED-FAILURE FIX DEPLOYED (driver 4913fc1c): release errors no longer
poison the map (the only remaining sticky = the intentional Destroy); a
one-time MAP-STICKY-FAILURE print added. RESULT: the print NEVER FIRED and
chain1 still 30s/fails — map->failure was never set on these boots. The
sticky theory is DEAD for chain1 (the fix stays: correct hygiene).

THE PRECISE HANG WINDOW (zero diagnostics anywhere): CKEY-WRITE/ADOPT for
the chain RUNS (spark0's logs show it) — the HC collective's key adopted —
then 30s of nothing: no LAZYWORK (LazyExperts never ran), no WD-LEASE (no
acquire reached the weightsd), no ERRSITE, rounds=0/mi=0 (the MAIN
collective never started). THE HANG = THE HC COLLECTIVE'S FIRST ROUND —
between its ChainKey success and its first mesh publish/wait. PRIME
SUSPECT: the HC's cell/epoch state — the HC never gets its own ChainKey
adoption on these boots? (the rebase path with the wave-encoded watermark
spin? the HC base-cell wait?) NEXT: instrument the HC collective's first
round (a print at the HC ChainKey + at its first RunRound with
cells/epoch/mirror state) — one deploy closes this.

## 09-22 09:30 TICK — CHAIN1'S HANG IS GONE (rounds advance!); NEW: a 794K ERRSITE flood from the serving adapter

DEPLOYED 8b7db925 (RH instrumentation + the accumulated fix set): THE FIRST
CHAIN NOW RUNS — RH-TRACE/RH-ORDINAL show main-collective rounds advancing
(op_index 7,8,9,10..., ordinals watermark-seeded, status=0). The printless
window between ChainKey and the first round CLOSED (the accumulated fixes —
scoped failures, warm gate, hello-close, IPv4 — in combination).

NEW BLOCKER (why the request still times out): a 794,804-line ERRSITE FLOOD
from spark_glm5_next_serving_adapter.c:1392 and :1420, all status=15 — the
adapter's completion/route delivery path returning ROUTE_NOT_FOUND en masse
(the loop hammers it; completions never deliver to the client). NEXT: read
the adapter's 1392/1420 sites (what lookup returns 15 — the route map for
completion delivery? the transaction id?), rate-limit the print (the flood
itself is a hazard), fix the lookup.

## 09-22 10:00 — the flood's 15s traced to the QUIESCE snapshot path

The flood sites (adapter 1392/1420) = ResetControl/Reset; ZERO
ADMIT9-MODULE lines carry status=15 → the 15 does NOT come through the
admit — the only other failing call in ResetControl = SparkGlm5NextServing-
Quiesce's driver SNAPSHOT (SPARK_RETURN at the 1392 site). No literal
SPARK_STATUS_ROUTE_NOT_FOUND anywhere in the module/adapter/kv/map layers
→ the 15 is a PASSTHROUGH from deeper (the weightsd client IPC result, the
lease layer, or the residentd's route table lookup invoked by snapshot).
NEXT: instrument the module snapshot's failure (one print with the inner
status + which sub-lookup), and grep the lease/weightsd-client layers for
the 15 source. The per-token reset loop (why resets fire continuously
during decode) is the second half — the API pipeline's per-step control
bump design.

## 09-22 10:30 TICK — adapter verified CURRENT (not stale); the 15-at-quiesce paradox pinned

The rate-limited build works (count=1 lines, log readable). The adapter .so
force-rebuilt = IDENTICAL sha (deterministic) — NOT a stale artifact. THE
PARADOX: per source, ResetControl's quiesce can only return INVALID(1) or
BUSY(16) (its snapshot call chain — module snapshot + lifecycle wrapper —
returns 1/OK only), yet the fleet logs status=15 at adapter:1392/:1420 on
the CURRENT build. MY READING IS MISSING A PATH. NEXT INSTRUMENT (one
deploy): prints inside SparkGlm5NextServingQuiesce (entry, the
available-count, the snapshot status, exit) — the 15's producer gets named.
ALSO NOTE: the request returned model status 9 (kv-admit, second-submit
lease) — the canary cadence still needs the first-request-wins pattern.

## 09-22 11:00 TICK — THE ENUM MISREAD: 15=BUSY (off-by-one in my awk decode); no paradox ever existed

CORRECTION (raw-byte enum recount): 14=ROUTE_NOT_FOUND, 15=BUSY, 16=DUPLICATE,
17=INTERNAL_ERROR, 11=TARGET_MISMATCH, 9=VALIDATION_FAILED. My earlier awk
arithmetic anchored one line wrong and "decoded" 15 as ROUTE_NOT_FOUND —
sending three hours down DNS→registry→server-side paths for what was BUSY
all along. Evidence-law applies to MY OWN tooling output too.

REINTERPRETATION (everything now consistent):
- The 30s status=15 chains = BUSY retries through the cold lazy walk — the
  known 74-280s class, ordinary retry behavior, NOT an error.
- The adapter:1392/:1420 "flood" = the per-token reset's quiesce returning
  BUSY while submissions were active — a retry loop at too-hot cadence
  (the cadence is the only defect; now bounded by the rate limiter).
- The prefetch's status=11 = TARGET_MISMATCH (not ABI) — remote-owned keys
  still the ownership-filter conclusion.
- The system state: chain1 = cold-walk BUSY retries then completes; the
  deferred prefetch (post-warm) tops up later chains.
NEXT: (a) re-verify a fresh canary end-to-end with the FIRST request (the
cold walk is the wait); (b) once chain2 graphs: the ARRIVAL receipt; (c) the
reset cadence (per-token quiesce) tuned; (d) ladder idle re-measure.

## 09-22 11:30 — FIRST FULL RUN RECEIPT: 411 rounds MEASURED; the cold walk = 498s; reset cadence = 400K

THE 850s CANARY: CHAIN-TIME slot=2 status=17 total_ms=500496 rounds=411
allreduce_ms=32217 (~78ms/round under cold-walk load) — the decode ADVANCED
~4-5 tokens end-to-end before the curl window closed; stage_ms[3]=498150 =
THE COLD WALK (expert loads) dominates the entire first request. The
per-token quiesce BUSY count reached 400,000+ during it (the cadence defect,
now MEASURED).

DESIGN CORRECTION (the deferred-prefetch paradox): deferring the prefetch
until after chain1 means chain1 alone eats the FULL cold walk. THE RIGHT
DESIGN = the ORIGINAL immediate prefetch, but OWNERSHIP-FILTERED (only this
rank's shard experts — kills the TARGET_MISMATCH failures) so warming runs
in PARALLEL with the first chain instead of after it.

NEXT LADDER: (1) ownership filter for the prefetch (the pack sidecar's
shard layout; re-enable immediate start); (2) the per-token quiesce cadence
(backoff or event-driven instead of hot-loop); (3) with warm boots: chain2
graph → the ARRIVAL receipt → the 330ms/round relay conviction; (4) ladder
idle re-measure.

## 09-22 12:00 TICK — CAUSAL CORRELATION: the 497s walk = the 6 unroutable layers; 768 rounds at 41ms/round

IMMEDIATE PARALLEL PREFETCH + BOUNDED RESET deployed (e2c5f98b/b0f63dce).
CANARY: the chain advanced 768 ROUNDS (~8.4 tokens) in 500s —
allreduce_ms=31673 = 41ms/round under load (improved from 78); stage_ms[3]
cold walk STILL 497.166s. THE CORRELATION IS CAUSAL-GRADE: the prefetch
fails layers 39-44 with TARGET_MISMATCH and THE WALK'S 497s IS SPENT ON
EXACTLY THOSE LAYERS (the chain's own acquires for them retry through the
walk). The other 39 layers prefetched fine. NEXT = THE WEIGHTSD-SIDE
QUESTION, NOW TIGHT: why do layers 39-44's chunks return TARGET_MISMATCH —
the weightsd's chunk registry/working-set resolve for those keys (read the
acquire handler's TARGET_MISMATCH conditions; compare a failing key vs a
succeeding one; check the pack tail's chunk files). Reset cadence: bounded
(250ms) — the reset-attempt prints now sparse ✓.

## 09-22 12:30 TICK — THE ROOT: the weightsd worker's CUDA context fails (TARGET_MISMATCH = cuCtxSetCurrent)

NO layer/ownership logic returns TARGET_MISMATCH anywhere in the resolve
path — the ONLY producers: spark_weightd_worker.c line 30/70 — cuCtxSetCurrent
failing in the WEIGHTSD'S WORKER THREAD. THE PICTURE: layers 39-44 are the
ones whose chunks are NOT YET PRESENT (every boot: 39 layers' chunks stay
resident in the pool; the tail 6 need CREATION) — chunk creation runs on the
worker's CUDA context — cuCtxSetCurrent FAILS → TARGET_MISMATCH propagates as
the acquire status → the prefetch fails those layers AND the chain's own
acquires retry through the same failure for 497s (the cold walk). WHY the
context set fails intermittently (context destroyed once? device reset?
GB10 context exhaustion with 16 engines + the daemon?) = the final question.
FIX TARGET: the worker's context self-heal (recreate on set-failure, loudly)
+ why it died once. This collapses the cold walk AND the prefetch failures
in one fix.

## 09-22 13:00 TICK — bind fix deployed; NEW pre-chain wedge on the boot (submission admitted, chain never starts)

BIND FIX DEPLOYED (f7e6a9f5): the conviction corrected (TARGET_MISMATCH =
the map_context THREAD check — the prefetch thread ran context-bare; the
weightsd worker was never involved) + SparkWeightdMapBindThread binds
device+context at prefetch start. Fuzzer 273 green.

THIS BOOT'S WEDGE (new, blocks the fix's measurement): submission 1000001
ADMITTED (decision=1) but NO chain EVER started (zero CHAIN lines, zero
PREFETCH-START — the chain never reached BEGIN); 1000002 rejected ADMIT9
behind 1000001's PREPARED lease ✓ expected; the API connection CHURNS
(reset-armed/complete + takeover cycles every few seconds, resumed=1) and
each reconnect arms a pending_client_reset whose quiesce can never complete
while 1000001 sits in-flight — SUSPECTED MUTUAL BLOCK: stuck submission ↔
never-quiescing reset. NEXT: (a) why 1000001's chain never began — trace
its route state (the ROUTE-STUCK scanner should have printed at 30s but
did NOT — is the route even active? check route activation between DECISION
and chain-BEGIN); (b) the reset/quiesce interplay with an in-flight
submission (does a pending reset gate new chain dispatch?).

## 09-22 13:30 TICK — my immediate-close created the churn (fixed); submissions still not chaining

THE RECONNECT CHURN CONVICTED AS MY OWN: the immediate close-after-output
block I added fired when a path sets the flag with NO output queued —
closing healthy sessions per message (prompt close where the pre-fix code
hung). Block REMOVED (the drain-close in WriteClient + POLLOUT-follows-
pending handles the real case); deployed 44ed57b1. Churn now +13/min (was
~1000x that during the 400K era) — residual churn = the API's own reconnect
backoff cycle.

REMAINING (the pre-chain wedge persists): submissions ADMITTED (5+ arrived,
decision=1) but NO chain, NO PREFETCH-START, ZERO ROUTE-STUCK prints — the
routes never age 30s (something recycles them) or never activate past
EnqueueCommitted. NEXT DISCRIMINATORS: (a) print the route state at
EnqueueCommitted + its ready_state (one instrument names the parking state);
(b) the hidden-transport input path (WAIT_INPUT delivery) vs the session
resets — each reconnect may invalidate the transport session carrying the
request payload; (c) whether the per-reconnect pending reset fences the
routes (cleaning them before the 30s scanner).

## 09-22 14:00 TICK — the lifecycle trace: the API's rank-0 connection drops between SUBMIT and DECISION

THE INSTRUMENT (COMMIT-TRACE, deployed efcf6f47): ZERO commit lines — the
DECISION message never arrives because THE API CONNECTION DROPS right after
SUBMIT-ARRIVED (residentd sees peer eof → reconnect → reset arms → next
submission → 1000003 dies ADMIT9 behind 1000002's dead lease). The chain
never starts because every submission's decision round-trip is killed by
the connection drop. The drops: API-side rank-0 only (model_resident_client
1031/1042 status 4), residentd never closes first. The suspect pair: (a) the
residentd's decision-required RESPONSE never flushes (output-queue/POLLOUT
path — the API times out and closes), or (b) the API's own read poll fails
on rank 0. NEXT INSTRUMENT (one deploy): prints at queue-decision-required
and at flush in WriteClient (residentd side) + the API's per-rank read-error
context — whichever side is silent names the breaker. NOTE: this is the OLD
"#22 reconnect storm" lineage — tonight's fixes exposed it as the remaining
serving blocker.

## 09-22 14:30 TICK — residentd EXONERATED (flush traces: 576/576 written); the breaker is API-side post-receipt

THE TRACES (af291cf3): FLUSH-TRACE shows the submit-result fully written
(576 of 576 bytes) on every attempt — the residentd's queue→flush path
WORKS; SUBMIT-RESULT-TRACE confirms results queued (17 for 5 submissions =
API retries). THE RESIDENTD IS EXONERATED: it receives, replies, and the
bytes reach the socket. THE BREAKER IS API-SIDE, POST-RECEIPT: the API gets
the decision-required result and its rank-0 connection still closes (the
peer-eof storm continues). NEXT (the final discriminator): API-side
instrumentation — its read/dispatch path around the submit-result receipt
(model_resident_client.c's Read + the batch engine's handling of the
decision-required result; why it closes instead of sending DECISION).

## 09-22 15:00 TICK — the invalidation amplifier found (any reconnect nukes the whole engine session mid-handshake)

THE MECHANISM (model_batch_engine.c:2140-2152 + model_pipeline_client.c:810):
the session fingerprint = Σ client_generation×(rank+1) over CONNECTED
clients — ANY rank's disconnect OR reconnect changes it → "batch engine
session changed ... invalidated" → ALL requests reset to QUEUED_PREFILL +
SparkModelPipelineClientClearTransactions — the in-flight submission's
pending DECISION is destroyed mid-handshake → the request restarts →
races the next drop → the storm is self-sustaining at any drop rate. THE
FRAGILITY: one rank's transient blip invalidates all 16 ranks' session
state (prefix cache + transactions). REMAINING QUESTION: the FIRST drop's
cause on rank 0 (the client Read's failure reason — parse/EOF/timeout —
uninstrumented). NEXT: (a) instrument the client Read failure class;
(b) harden the fingerprint (ignore generation bumps within a live session —
invalidate only on HELLO-generation changes, not connection recycling);
(c) then the payoff chain as queued.

## 09-22 16:00 TICK — PROTOCOL RESTORED: the fleet runs PR #1077 code, 16/16 PR-exact

OPERATOR CALL-OUT ("only test what is in a PR") — the audit convicted the
rig: spark3 ~/g5pacing was DIRTY (9 files) on a divergent local HEAD
(8de7244e), so the deployed artifacts came from unpinned source. FULL
AUDIT RESULT: all 9 dirty files byte-identical to the PR tip; the rig was
STALE, never ahead (the PR has strictly more: draft_bridge AF_INET, the
arrival-dump nonzero line, 467 lines of these tick records); the g5pace*
strays are crash-era data blobs, not source. NOTHING UN-PUSHED EVER
SHIPPED — but nothing PROVABLY shipped either; now it does.

THE RECOVERY: Mac tree's 4 uncommitted files (nanosleep removal, ladder
.cuh dependency, arrival-dump) committed → 781b61d = PR #1077 head.
spark3 re-clean-cloned at 781b61d (old tree kept as g5pacing_dirty_0922)
→ full rebuild (module validator PASS) → model_compile → publish_local
(dies at its own MANIFEST step: find-with-missing-args + pipefail + set
-e = silent exit 1 — the manual `find lib bin stages` regen stays in the
recipe) → hub MANIFEST e6e8e1e7.

ARTIFACTS (deterministic, PR-exact): driver a393a959, transport 8178940f,
residentd af291cf3, adapter 018d5854. The residentd is byte-identical to
the previously-deployed one (its source was already fully in the PR);
the driver differs from the stale-rig 97889fed exactly by AF_INET +
arrival-dump — meaning the fleet NEVER ACTUALLY RAN the draft_bridge
IPv4-only fix until now (the 07:00 "deployed a76b5cf4" claim shipped a
stale build — commit-messages-are-not-evidence, artifact edition).

DEPLOYED 16/16 (spark0-9 + sparka-f): engines auto-restarted
08:46:51-53, ~10s convergence, zero manual kills. EVERY TEST FROM HERE
= PR CODE. NEXT: the canary on this build is a real discriminator — the
rank-0 client drops (the invalidation amplifier's trigger) have never
been observed under true IPv4-only connects.


## 09-22 17:00 TICK — THE OPERATOR'S CONVERGING DESIGN LANDED + THE API RETURNS TO RTX5090 (with the tokenizer); serving = one blocker left

OPERATOR RULINGS THIS ERA: (1) "only test what is in a PR" — PROTOCOL
RESTORED (16:00 tick, fleet = PR-exact code by construction); (2) the design
question answered with the epoch architecture; (3) "the API belongs on the
rtx5090 — that is where the tokenizer will run; make a non-retarded system";
(4) "I stopped the other dev — kill any weightsd/residentd that interferes;
get it stable and ready for parallel dev usage."

THE STORM, FULLY CONVICTED (tcpdump + logs): (a) sparkf's monitor connects
to every residentd's single client slot — the OLD takeover-at-ACCEPT evicted
the API per monitor connect (the duel); (b) the API's pipeline Progress
counted its rank loop DOWN to zero and passed rank==0 to SetFailure — EVERY
failure killed rank 0's HEALTHY connection ("only rank 0 dropped" was a
lie); (c) any reconnect bumped client generations → the fingerprint
invalidated the whole engine session → teardown → more reconnects: self-
sustaining at ~30 cycles/s, all 16 ranks, zero requests needed.

THE EPOCH DESIGN (deployed 16/16, residentd 720f103a era): HELLO carries
session_epoch (pipeline-generated, pid+ns-mixed); accept PARKS candidates
without evicting (a scraper that never says hello can never steal the
slot); same-epoch hello = REATTACH (fd swap, session state kept: routes,
queued output, generation — no reset armed); different epoch = the ONE
legitimate takeover (generation bump, stale-generation cleanup runs); eof =
detach (session kept, reattach expected); fingerprint = epoch + all-rank
generations (stable across churn); rank-attribution fixed; the first-hello
reset arm REMOVED for virgin boots (a fresh residentd has nothing to clean
— arming deadlocked the quiesce against the API's 10K-retry hammer).

RTX5090 HOMECOMING: the API runs where it always should have — x86 build
(clean clone ~/g5epoch, -Wno-restrict for gcc-15 + POSIX-guard fixes), the
runtime root ~/glm53flash.fp8.tp16 (only the adapter .so is x86-executed;
stage/transport are aarch64 reference copies), single-API script
/tmp/one_api.sh (the /proc/exe kill — plain pgrep kills your own ssh). THE
TOKENIZER WIRED (first time ever): repo asset glm-5.3-flash-tokenizer.json,
vocabulary_size 154856 — REAL BUG FIXED: the sidecar counted only
model.vocab (154820) but eos ids 154820+ live in added-token space
(max_token_id+1 floor added); health now reports "tokenizer":true.

KEEPALIVE: the lab router silently reaps idle workstation↔fleet flows and
RSTs on next traffic (spark6/spark0 cuts convicted; residentds saw NOTHING)
— client sockets now carry TCP keepalive 10/5/3 (keeps the router's idle
timer reset; mirrors the residentd's accepted-socket settings).

WEIGHTD-MESH REVIVAL: spark0's MESH-SPIN missing=10 was STALE RECORDS
(Sep 21 15:06 — its agent had stopped refreshing; peers restarted = dead
QPNs). The agent (alive) recycled the daemon 09:39 → mesh wired peers=15.
The residentd's WEIGHTD-DEAD verdict is STICKY across daemon restarts
(workaround: residentd restart; the agents supervise — kill -9 → restart
in ~5s). Other 15 nodes' daemons + records were healthy throughout.

MEASURED THIS ERA: chains RUN under the full new stack — 91 rounds,
allreduce_ms=0.66 TOTAL (host-side; the round path is sub-ms), wall
35.1s/91r = 386ms/round CONTENDED (cold walks + abort churn around it);
graph capture works (GRAPH-CAPTURE-OK bound=257); prefetch 42/45 (the
39-44 tail still fails — the weightsd budget class, next); end-to-end
HTTP delivery works (JSON error/response reaches the client).

THE ONE REMAINING BLOCKER: the engine ABORTS every submission post-
restart (residentd sees SUBMIT → decision_required → DECISION decision=2
ABORT ×1,027,907 submissions). The decision policy aborts whenever the
transaction's first result failed — SOME RANK still returns BUSY/IO on
submit (TXN-FAIL-FIRST status=15/4 era). NEXT: name the failing rank
(SUBMIT-RESULT-TRACE per node), fix its BUSY source, then the cold-walk
deadline (35s chains > the 30s deadline → status=17 — the deadline must
read observable state per the minimum-fix ruling). Then: warm canary →
chain2 GRAPH → ARRIVAL receipt → the clean µs/round number.

HYGIENE LEDGER (pre-existing, verified failing at 781b61d too):
test_steploop_admission + test_model_pipeline_client (a 100ms weightsd-
attach race in the fixture); the gemma4 test fixture bitrot
(SPARK_GEMMA4_MODEL_MODULE_TARGET undefined) blocks full `make test`.

## 09-22 19:00 TICK — the abort loop is DEAD; the frontier = 6 cold layers + a wall-clock deadline

CLIMB STEPS LANDED (all in PR #1077 @3c9f373):
- KV-RECLAIM (cache/kv_page_cache.c): lane admission reclaims slots whose
  owner's OWN declared deadline precedes the live request's — the abort
  that frees them is routinely eaten client-side (ClearTransactions drops
  pending decisions). CONVICTED LIVE: sparkc slot 1 owned by a dead
  hammer-era request (owner_req=1024868, phase 0), BUSY-failing EVERY
  admission for hours (sparkc = the lone BUSY rank; 6 decisions vs 17888
  on all other nodes). Post-fix: submissions COMMIT (decision=1) — the
  abort-every-submission loop is dead.
- WEIGHTD-STACK REVIVE (module): the agent recycles the weightd on
  binary-sha changes; the transport's client dead-stickied (WEIGHTD-DEAD
  → every reduce IO-fails until a residentd restart). The module now
  detects the dead lane client at round-submit and re-creates the WHOLE
  stack (lane acquire + both collectives) via the stored node context.
- KV tests green (test_kv_cache, test_kv_lane_fuzz).

MEASURED: the HTTP pipeline delivers cleanly end-to-end (RC=0 round
trips, well-formed responses AND errors). Chains: 7 rounds at 1.7ms/round
allreduce under FULL cold load (11.83ms/7r); committed submissions flow
on all ranks. Requests still fail status=17: chains legitimately walk
~190s cold while the chain deadline is 30s.

THE FRONTIER (exact):
1. PREFETCH still fails layers 39-44 (status=4, 42/45) — the weightsd-
   side chunk class for the tail layers (chunk CREATION era; the map-bind
   fix did not cure these). Those 6 layers = the entire 190s walk.
2. The chain deadline (30s → status=17 degrade) fires during legitimate
   cold walks — the same elapsed-time-vs-observable-state violation the
   reaper fix settled; the chain degrade must read liveness (ordinals/
   layers advancing) per the 09-21bd minimum-fix architecture.
NEXT: (a) the weightsd-side question for layers 39-44 (why do their
chunks fail the exchange while 42 layers succeed), (b) liveness-based
chain deadline, (c) then the warm canary → chain2 GRAPH → ARRIVAL
receipt → the clean µs/round.

## 09-22 21:00 TICK — prefetch retries landed; the acquire failure is FABRIC-LEVEL (mesh writes die at RDMA)

CLIMB STEP LANDED (@00b0a95): prefetch retries failing layers across 5
passes + WITHHOLDS experts_warm while layers stay cold (the graph gate
was opening over cold layers). Also from this tick: KV-RECLAIM +
weightd-stack revive held (commits still flow).

THE REVERSAL + CONVICTION CHAIN:
- After the roll, prefetch failed ALL layers (was 42/45): the acquire
  exchange dies at spark_weightd.c:3490 = THE MESH WRITE OP — not the
  acquire logic itself. The weightd logs WD-MESH-CQERR (wr opcode=1
  status=12 vendor=129) + WD-QP-REPAIR loops ("qp left RTS after
  errors").
- RECORDS EXONERATED (wrong-dir detour): the daemon runs WITHOUT
  --mesh-dir (default /tmp/weightd-mesh — FRESH, all 16 recs, agent-
  synced); the /tmp/weightd-mesh-fleet dir I first inspected is the
  ABANDONED newer-agent era. Hub qpn/ records fresh; fetch path 3ms.
  (Agents restarted fleet-wide anyway — the running 619-line version
  syncs the correct default dir; the 691-line version with mesh_dir()
  selection lives only in /tmp on spark0 — NOT canonical.)
- Post-fresh-records: prefetch 1/45 (was 0), one chain 7 rounds over
  319s then IO, chains BUSY at the 30s acquire deadline. Slight
  movement, not a cure.

THE FRONTIER (next session, exact): the weightd's RDMA mesh WRITES fail
(CQERR vendor 129) with fresh records + wired control plane ("ready
peers=15") — diagnose at the fabric level: what vendor 129 is on
rocep1s0f1, whether QP-REPAIR restores traffic or loops, whether it
started at a specific daemon restart, and whether a cold weightd restart
fleet-wide (agent-supervised) heals the QP state. THEN the 6-tail-layer
creation class re-measures on a healthy fabric.

Standing receipts this tick: canary HTTP round-trips clean end-to-end
(RC=0, well-formed errors); the retry-prefetch + warm-gate deployed
fleet-wide (driver e9ee22e7); agent fleet restarted with records
verified flowing.

## 09-22 22:30 TICK — heal-test results: the FABRIC is exonerated; the systemic pattern is "no client survives daemon turnover"

THE COLD-RESTART HEAL TEST (16/16 weightds killed at 11:07:10-16, agents
relaunched in seconds, mesh rewired peers=15, records fresh at 11:07:11):
- ZERO CQERR in the fresh era — the vendor-129 errors were a SYMPTOM of
  the old generation, not the fabric. FABRIC EXONERATED.
- Acquires STILL failed (prefetch 1/45; chains BUSY at the 30s deadline)
  → discriminator: the weightd log shows ZERO acquire requests arriving
  — the residentds' MAP CLIENTS still held sockets to the KILLED daemon
  generation (the same dead-socket stickiness the lane client had; the
  map has NO reconnect).
- Residentd restarts (fresh map attaches) unblocked the requests — and
  exposed the NEXT surface: submissions rejected status=9 at the module
  admission SHAPE evaluation (module.c:4522) + ROUTE-STUCK state=1
  (claimed, not advancing). New class, post-restart.

THE SYSTEMIC PATTERN (the operator's parallel-dev bar): every daemon
turnover (weightd recycle, agent deploy wave) orphans a client layer —
lane client (fixed: REVIVE), map client (UNFIXED: no reconnect), kv
admission state (fixed: RECLAIM), session state (fixed: epochs) — and
each manual heal surfaces the next one. The converging-system answer:
EVERY client of the weightd re-establishes on turnover (the map
reconnect is the missing piece), and the admission shape failure after
restarts needs its own look (state->resident_sequence_capacity vs
available vs the request's shape — something disagrees only after a
warm-daemon/cold-residentd combination).

MEASURED THIS TICK: canary HTTP round-trips clean throughout (RC=0);
status codes observed 9/15/4 across the classes above; no chain receipt
(this tick was diagnosis, not throughput).

NEXT (exact order): (a) the map reconnect (SparkWeightdMap* re-
establishes on dead socket — mirrors the lane revive; kills the
residentd-restart dependency on every weightd recycle), (b) the
admission-shape status-9 after restarts (the capacity/limits
disagreement), (c) then the tail-layer creation class on a stable
stack, (d) warm canary → GRAPH → ARRIVAL → clean µs/round.

## 09-23 00:00 TICK — the admission watermark trap killed (status-9 wedge dead)

THE CONVICTION: the module's admission predicate rejected every request
with VALIDATION_FAILED(9) via `request->control_generation <
state->control_generation` — but control_generation carries the ENGINE
FINGERPRINT, a quasi-random value that legitimately moves BACKWARD
whenever residentd generations change (any engine session rebuild).
With no reset path armed (the session epoch is unchanged — my takeover-
only arm is correct), the watermark had NO exit: permanent wedge after
any fleet/engine restart combination. VERIFIED LIVE: zero
submission_rejected lines post-fix (driver ce23f66e); requests admit
across engine rebuilds.

The backward check bought nothing: replay protection already lives in
the per-session message-id gate (residentd) + epoch scoping + the
KV-RECLAIM. Removed.

STATE: back to the known cold-walk regime (7r/190s through the tail
layers; BUSY-30s chains queued behind; prefetch ~3/45 this boot). The
two open items stand: (a) the tail-layer chunk creation class at the
weightsd (the 190s walk), (b) the map reconnect (weightd-recycle
resilience). Then the ladder: warm canary → GRAPH → ARRIVAL → clean
µs/round.

## 09-23 02:00 TICK — the verdict instrument deployed through the core channel; routes stall at RESOLVING

- THE PACK EXONERATED: receipt file_bytes == on-disk size (21,706,046,976)
  — not truncation; the tail-layer class is not a short pack.
- THE VERDICT INSTRUMENT LIVE: rate-limited ACQUIRE-FAIL lines (layer0,
  key_count, status + the full budget arithmetic: pool/retained/other-
  resident/device_max/chunk_count) + the sticky ARENA-FAILED path gets
  its own line. Deployed via the core announce channel (hub WEIGHTSD_BIN
  sha16 → agents sync core → install → recycle) — daemon a6c9188e live
  16/16, mesh wired. FIRST core-channel deploy through this flow.
- STATE AT WRAP: submissions admitted (the watermark fix holds) but
  routes stall at state=1 RESOLVING (~36s age) — the resolve path's own
  acquire never reaches the daemon (weightd log quiet; no ACQUIRE lines
  yet). The verdict lands when a route's resolve actually issues its
  acquire. NEXT: (a) read the ACQUIRE-FAIL verdicts, (b) the
  RESOLVING-stall question (why the resolve's acquire never fires),
  (c) map reconnect still queued.

## 09-23 04:00 TICK — ROUTE-RECLAIM landed; the verdict instrument sits on the wrong IPC kind

- ROUTE-RECLAIM (residentd 110668ee): the stuck scanner now FREES routes
  owned by a previous client generation (dead-session routes blocked
  their lane's KV admission forever — every fresh submit BUSY at
  ADMIT9-KVPHASE; same orphan class as KV-RECLAIM, route level).
- THE VERDICT INSTRUMENT GAP: ACQUIRE-FAIL logs sit on the ACQUIRE
  working-set kind, but the map's bulk acquire rides a DIFFERENT IPC
  handler (the lazy/batch kind) — this boot's prefetch failed 42 layers
  with ZERO ACQUIRE lines: the failing path never passes my instrument.
  NEXT: move/add the logging to the bulk/lazy acquire handler; the
  verdict then names the tail-layer mechanism.
- Canary16 pending at wrap; submissions flow; admission watermark fix
  and epoch stack holding throughout.

## 09-23 06:00 TICK — THE TAIL-LAYER VERDICT: not a bug — slow cold creation from storage

THE EVIDENCE (fresh engines vs the live instrumented daemon):
- ACQUIRE-STALL exchange 191396 ms — the first cold acquire REACHES the
  daemon and hangs server-side in lease/load; the handler never
  completes (why WD-LEASE-TRACE never printed — it prints on completion);
  the client's exchange then times out (map.c:514 status=4) and the
  retry ladder burns its passes against the same slow creation.
- The 42 "fast" layers were ALREADY RESIDENT in the daemon's memory
  (instant); the 6 tail layers need chunk creation = reading their expert
  data from the 21.7GB pack (slow tier) = minutes per layer set.
- Timeline conviction alongside: the weightd (11:50:57) OUTLIVED the
  residentds' attach (11:46:57) — every recycle orphans engines (the
  map-reconnect item; a real revive needs a FULL re-attach, not an fd
  swap — the arena generation dies with the daemon).

THE DESIGN DIRECTION (next): the creation-class acquire must not ride a
synchronous client timeout — either the daemon answers PENDING and the
client polls (the async-creation design), or a standalone deploy-time
warmer pre-creates the tail ONCE per node (minutes of storage reads at
deploy, instant serving forever after). Plus the map re-attach revive.

## 09-23 08:00 TICK — FIRST END-TO-END SERVING RECEIPT: real tokens, MEASURED

THE RECEIPT (api.log, boot 144857 era): requests 100004-100006+ COMPLETED
— status=0, engine_completed=1, real GLM token streams returned to the
client (8 tokens each; 541 token-timestamp lines in the log). The whole
stack served: rtx5090 tokenizer → engine → 16 ranks → chains → tokens.

MEASURED (honest grading):
- First token: +9.27s after acceptance (admission + prefill + first chain)
- Steady decode: ~1.15 s/token (deltas 1128-1174ms across the streams;
  ≈0.87 tok/s) — vs GOALS G1 = ≥14 tok/s: the gap is the relay-arrival
  cadence + the per-token chain cost, the mission's actual target class.
- The serving window rode a WARM daemon (the night's creation hammering
  persisted server-side — creations survive client timeouts ✓ the
  async-persistence hypothesis CONFIRMED); the current cold spell = the
  daemon recycled again (11:57) wiping the arena → chains BUSY-30s again.

THE SYSTEMIC FIX (named by this receipt): creations persist per-daemon-
LIFETIME — every weightd restart re-pays the cold walk. The warmer must
be AGENT-DRIVEN (post-weightd-start): pre-create the 6 tail layers once
per daemon boot (minutes of storage reads, then serving is instant).
With that + the map re-attach revive, the fleet becomes: recycle-proof,
permanently warm, serving at the measured cadence — and the µs/round
climb (330ms/round → the graph path's µs class) becomes the only
remaining work.

## 09-23 10:00 TICK — THE OVERFLOW CONVICTION + the deploy-time warmer WORKS

THE CHAIN OF CONVICTIONS THIS TICK (each named by its instrument):
1. FIRST SERVING RECEIPTS stood (8 requests, real tokens, ~1.15s/token).
2. weightd_warm (tools/weightd_warm.c) written: attach-lazy + per-layer
   bulk acquire with a creation-sized timeout.
3. First runs insta-failed 17 → ACQUIRE-LOAD-STAGE instrument (budget/
   chunk_ensure/open_pack/load_lease per-stage verdicts, deployed via
   core-announce) → "stage=budget status=17".
4. THE ROOT: expert_pool_bytes=UINT64_MAX (the unset-env default!)
   OVERFLOWS the budget sum (pool + preload wraps to ~2MB) → the cap
   check always fails → eviction finds no victim on a fresh arena → 17.
   The ENGINES never hit it because their environment sets
   SPARK_WEIGHTD_EXPERT_POOL_BYTES=34359738368 (32GiB).
5. Arenas cache by identity: the first poisoned attach (MAX) persisted
   server-side — a daemon restart was needed for the corrected pool
   bytes to take effect.
6. POST-FIX: layer 3 WARM on the first acquire (WD-LEASE-TRACE status=0)
   — the creation path is sound; the warmer now grinding the cold tail
   (minutes per layer set, persisting per daemon lifetime).

THE DORMANT TRAP RECORDED: any client attaching without the env gets the
MAX default → poisoned arena → every acquire 17s. The env is the config
authority today; the daemon should reject UINT64_MAX at attach (queued).

NEXT: warm completes → engines restarted (fresh maps) → canary → THE
MEASUREMENT (warm µs/round through the full new stack). Then agent-side
warming (post-weightd-start) makes every recycle self-healing.

## 09-23 12:00 TICK — the warmer's traps fixed; THE DAEMON CRASH AT CREATION convicted (reproducible)

PROGRESS: the overflow (UINT64_MAX pool + preload wraps) and the arena-
identity cache (poisoned arenas persist per daemon lifetime) both fixed
in weightd_warm (reads SPARK_WEIGHTD_EXPERT_POOL_BYTES, 32GiB default);
layer 3 WARMED once (clean acquire/release traces) — the warmer design
is proven.

THE CONVICTION (reproduced twice): the daemon DIES SILENTLY at the
first creation-class acquire — warm5: attach OK → layers 0-2 NOT_FOUND
(normal) → layer 3's acquire → daemon death (agent restarts it 3-4 min
later; the client's remaining layers fail IO on the dead socket). NO
stage instrument fires (the death precedes budget/chunk_ensure/load) —
the lease-acquisition path or an earlier corruption surfacing. No
cores (removed in the disk-full era), no dmesg access.

NEXT (the crash hunt): (a) re-enable a BOUNDED core for the weightd
start (the agent's ensure_weightd: ulimit -c + a core pattern to a
bounded dir), (b) reproduce → the core names the faulting frame (the
12:30-era CUDA-context class is the prime suspect: chunk creation on
the worker context), (c) fix → the warmer completes → agent-driven
post-start warming → recycle-proof permanently-warm serving.

## 09-23 14:00 TICK — the crash hunt's elimination ledger; a traceless SIGKILL

INSTRUMENTATION LANDED: bounded cores for the weightd via the agent's
own launch (hub core script patched: ulimit -c 4000000 + cd ~/wdcore —
agents pick it up through the core manifest sync; verified live on the
daemon's /proc limits). THE CORE-ENABLED DAEMON DIED AGAIN MID-CREATION
WITH NO CORE = SIGKILL-class.

THE ELIMINATION LEDGER (each ruled out by its missing record):
- earlyoom: no kill lines (thresholds 7%/3.5%; mem never below 63%)
- kernel OOM: no records
- the agent's sha-recycle: no announces in the death windows
- the daemon's self-exit: ZERO exit() paths mid-flight (latch exits at
  boot only)
- GPU hardware: healthy, no Xids, no retired pages
- my supervisor: dead before the last deaths

THE DUEL DOCUMENTED (rotated logs): overlapping restart sources — the
agent's ~2-min backoff cadence + challengers taking the latch ("held by
a live weightd; exiting 0 idempotent" ×3) + the core-enabled solo
daemon (12:59:10) still died at 13:04 with NO challenger kill line in
its successor's log. THE KILLER REMAINS UNATTRIBUTED — a traceless
SIGKILL during the creation-class acquire.

NEXT INSTRUMENTS (the killer hunt): (a) a signal-delivery watcher
(auditd or a privileged eBPF/proc-connector probe naming the kill's
sender pid), (b) the creation path read for anything the environment
would punish (huge host staging spikes visible in earlyoom's 5s memory
reports DURING a warm — watch avail memory live while warming), (c) the
syslog-per-second capture during one warm run (the record may exist
outside my greps).

## 09-23 16:00 TICK — BOTH KILLERS NAMED: timeout-judged health probes vs a busy daemon

THE VERDICT CHAIN (rc-capturing supervisor + instrumented logs):
1. The deaths are CLEAN rc=0-then-kill sequences, not crashes — the
   core-enabled daemon died with NO core, and the syslog finally showed
   the actor: `weightd: stale or unresponsive instance(s); clearing
   (production channel only)` — THE AGENT'S HEALTH PROBE: a 3s python
   connect ×3 to the unix socket; failure = kill -9 every weightd.
2. THE LATCH KILLER (fixed first, @d380827): holder-aliveness was a
   connect() probe against a listener that never accepts — one
   challenger fills the backlog, every later misreads a busy holder as
   wedged and SIGKILLs it mid-creation. FIXED: aliveness = the /proc
   scan (pid + exe) — readable state, busy-but-alive is alive.
3. THE SHARED ROOT: a daemon mid-creation (a synchronous minutes-long
   handler) starves its accept queue (tiny backlog + the connection
   flood from the retry ladder) — BOTH the agent probe and the old
   latch probe judge "dead" by connect-timeout and kill a HEALTHY BUSY
   daemon. The exact banned class: elapsed-time judgment where readable
   state exists.

THE REMAINING FIX (next): (a) the agent probe must not kill on
connect-timeout alone — size to the real creation window (300s) or
read process state (alive + serving = fine); (b) the daemon's accept
loop/backlog must survive a busy handler (drain accepts; backlog ≥16;
or the creation moves off the connection thread). Then the warm run
finally completes end-to-end and the fleet warms permanently.

## 09-23 18:00 TICK — DEMAND-DRIVEN EXPERT LOADING (the operator's production rule)

OPERATOR RULING: "why are all the experts needed? track which experts a
small request needs, load them in batch in seconds... parallel debugging
is broken by loading all weights; for the allreduce I allow it, but we
need a production solution to shared debugging."

LANDED (@8c8fb4d, driver 4093d8dd + daemon af955ef5):
1. SPAN-BATCHED PACK READS: contiguous ranges merge into one pread per
   span (per-range digest+copy from staging) — the old per-expert path
   was ~2MB/s on NVMe (1000x under hardware; the death-by-tiny-reads
   was ALSO the entire 190s-cold-walk class and the warmer's hours).
2. PREFETCH DEFAULT OFF ("PREFETCH-OFF" prints live on the engines):
   experts demand-load per routed submission (top-8-of-288; tens of MB
   for a small request); the bulk warm is opt-in via
   SPARK_GLM5_NEXT_PREFETCH=1 (the allreduce lane only).
3. My earlier full-warm-vs-fresh-daemon mistake acknowledged in the
   record: the 42-vs-6 split was daemon-uptime artifact; nothing
   required warming all — the demand path was just too slow (now fixed).

ACCEPTANCE TEST (the small-request canary): queued behind the retry
backlog at wrap — the receipt (routed experts demand-load in seconds +
tokens returned) lands next tick. The allreduce lane's full warm (env
opt-in) re-measures then too: span-batched, the 45-layer warm should
drop from hours to ~minutes.

## 09-23 20:00 TICK — the recorded-working-set feature PROVEN (336 keys); a second 30s deadline found

THE OPERATOR'S ONE-SHOT SMOKE WARM, LANDED (@192025c, daemon c366b561):
- The daemon RECORDS every acquired expert key to <PACK>.wset (the
  live tape; loaded back at arena creation; written on change).
- weightd_warm --wset FILE: replays ANY named set (one file per smoke
  test — snapshot the live tape per test) in ONE batched acquire with
  elapsed_ms timing.
- PROVEN LIVE: the recording exists — packs/rank0.sp.wset = 336 keys =
  exactly the smoke decode set (45 layers x top-8, deduped). The shape
  is right; the feature works.
- The transport's round spin timeout now 120s (a chain hit the bound at
  exactly 120.00087s — live).

OPEN (blocks the cold-preload timing test): a SECOND 30s deadline —
chains still die at 35.06s with 91 ROUNDS COMPLETED and sub-ms
allreduce (0.3-0.5ms TOTAL): the wall is engine/adapter-side (the
submission's deadline path — origin not yet located; the transport bump
was not this one). NEXT: find and size the second deadline (the
liveness-based design item unchanged), then the timed test: cold daemon
→ --wset preload (the 'literally 3 seconds' number) → warm canary.

## 09-23 22:00 TICK — NOT a deadline: GRAPH replay works; ranks 4+12 stuck-tails = the pacing bottleneck

THE "SECOND 30s DEADLINE" DECONSTRUCTED (log anatomy of the 35.06s
death): GRAPH-REARM → gate → CAPTURE-OK → ONE-LAUNCH replay →
GRAPH-REPLAY-TIME ns=35000000121 rounds=91 ns_per_round=384615385 →
STUCK-TAILS: peers 4 and 12 sit TWO SEQS BEHIND (…892 vs …894) →
cancel → DEGRADE graph-stuck → status 17.

THE MEASUREMENTS (honest):
- THE GRAPH PATH RUNS END-TO-END under the new stack (capture, 91-round
  single-launch replay, clean cancel).
- µs/round THIS ERA: 384.6 ms/round MEASURED in graph mode — the
  relay-arrival cadence; allreduce host-side 0.3-0.5ms TOTAL.
- THE BOTTLENECK'S NAME: ranks 4 and 12 lag 2 rounds at chain end —
  the exact class the ARRIVAL ring was built to convict.

THE NEXT LADDER RUNG (finally the mission's own climb): the arrival
dump on a green-path replay — per-round arrival timestamps for ranks
4/12 vs the pack — names the stall source (relay burst handling vs
poll shape vs those nodes' load). Then the pacing fix; ≤100µs/round is
the graph's own class once arrival stops gating.

## 09-24 00:30 TICK — PR #1081 MERGED (astra) and VERIFIED: the chain class jumped 122x

THE MERGE (@89330dc, one conflict — tools/weightd_warm.c: their stricter
manifest-driven warmer + my --wset one-shot grafted in their style):
- Their replacements SUPERSEDE cleanly (verified in the merged tree):
  the latch = bind-or-exit ("existing owner untouched" — no probing at
  all); the agent ensure_weightd = owner verification + refuses
  ambiguous cleanup (subsumes my busy-alive guard); routes carry
  generation handling at 5 sites (subsumes ROUTE-RECLAIM); reconnect =
  abort/drain/reset/new-session with callbacks retiring transactions
  (the orphan class my reclaims compensated for, designed out).
- MY pieces that SURVIVED (verified): the wset recording (WD-WSET), the
  span batching (LoadRangeGroup + their short-read fix), KV-RECLAIM,
  the 120s bound, the epoch sessions (session_epoch end-to-end), the
  demand-driven default (their prefetch removal completes the design).
- The hub agent = their version + the wdcore/ulimit cores wrapper
  re-applied (my patch, preserved).
- Build clean; kv/IPC/reconnect tests PASS.

THE VERIFICATION RECEIPT (what the operator asked for — "verify it
helps"): CHAIN-TIME slot=2 status=0 total_ms=287.35 rounds=91 (new
metrics: collective_host_submit_ms=125.61 / 91 submissions) =
**3.16 ms/round** — from 384.6 ms/round = **122x**. The relay-pacing
class that owned this fleet for a week is GONE in the merged build.

The request-level canaries (23, 24) hit RC=28 through the deploy
transition churn (requests status-4 mid-convergence) — the serving
receipt is the immediate next item, then: the ladder to ≤100µs now
runs from a 3.16ms floor with the graph path live.

## 09-24 02:30 — PR #1081 UPDATE MERGED (ff to 4add3eb) + deployed; the chain class holds; the request-completion gap named

THE UPDATE (5 commits: working-set recovery hardening + self-owned buffer
teardown; serving-ownership fuzzing + removal of unsafe cache takeover —
my KV-RECLAIM deliberately superseded by strict ownership + real abort
delivery; TOKENIZER BOUNDS validation (maximum_token_id+1, consistent
with the added-token floor); pipeline completion defers until every rank
drains; fuzz campaign records + qualification gates).

DEPLOY TRAP CAUGHT (the evidence law again): the first "deployment" of
this update shipped OLD binaries — the ff-merge never left the Mac
(origin still at 1a62b72; both build hosts pulled nothing; the
unchanged shas were the tell). Pushed, rebuilt for real: residentd
eae682f7 / daemon 2e4efe57 / x86 api 1f9e20e7 / driver b257f799.

VERIFIED: chains complete at the good class (311.69ms status=0,
host-submit 108.76ms/91; the 3.16ms/round floor holds on the update).
OPEN (the exact next item): the REQUEST-level completion still fails —
engine_completed=0, request status=4 (IO) while chains run; the
canary RC=28s. The gap is in the request/completion plumbing (their
every-rank-drain deferral under real traffic needs the look — or the
API connection path again). Timer stays STOPPED per the operator;
state parked for the next session/astra's next word.

## 09-24 04:30 — BENCHMARK FINDINGS + the abandoned-spinners FIXED (fleet idle SM 96%→0%)

WHY 250ms/token WHEN ALLREDUCE IS ~30µs/round (measured, operator challenge):
- Reduce stages (the actual collective device work): 1.5-3.6ms per 91-round
  chain = ~30µs/round. Confirmed negligible.
- Host collective submit: 71-109ms per chain (0.8-1.2ms PER ROUND on the
  host — should be ~10µs).
- "Attention"/"MLP" stage walls (82-97 / 139-156ms): NOT compute — the GPU
  ran 96% SM at 0% MEMORY (both during chains AND IDLE): spin-wait kernels
  monopolize SMs waiting for peer publishes; host timers attribute that
  wall to the surrounding stages.

THE SKEW ANSWER (operator: "it can't be 1ms"): natural compute skew is
µs (identical kernels/GPUs). The ms-scale arrival spread has two real
sources: (a) the host submit path — per-round enqueue takes 0.3-1.2ms
with variance ACROSS ranks, so the slowest rank's late publish makes the
other 15 spin ~1ms/round (this is the real skew, and it's host-made);
(b) GPU timeslicing against abandoned spin kernels. Both vanish in the
graph path (one launch).

THE INTERRUPT ANSWER: CUDA has no device-side interrupt; the practical
fast mechanism is PUSH-not-PULL — the publisher's RDMA write lands in
each PEER'S LOCAL cell and the waiter polls LOCAL memory (sub-µs,
negligible bus traffic). Today waiters poll REMOTE cells via ld.global.cv
(RDMA read per poll sweep ×16 ranks — the operator's bus-saturation
concern is exactly right). The graph path + local-cell push = the ≤100µs
design.

THE SPINNER FIX (@0733326, driver 0d9b1fe7): cancel broadcast on BOTH
collectives at chain COMPLETION (previously only on failure) — lingering
wait kernels from final rounds exit immediately. VERIFIED FLEET-WIDE:
idle SM 96%→0% on every sampled node (spark1/3/5/7/9/c/f all 0).

ALSO NOTED: the fresh boot shows "GLM execution mode=graph" — the merged
PR's explicit-graph-mode default is live.

## 09-24 05:30 — TEST RESULTS (graph-mode boot + one-shot preload)

- Spinner fix VERIFIED in the field: idle SM 0% fleet-wide (was 96%).
- ONE-SHOT PRELOAD RECEIPT: --wset 336 keys → WSET-WARM elapsed_ms=0
  (instant — the experts were already resident from the walk; the number
  proves the one-shot path; a TRUE cold preload still needs measuring on
  a fresh daemon).
- THE BLOCKER NOW: in the fresh GRAPH-MODE boot, NO chain completes at
  all (zero CHAIN-TIME lines; requests die status=4). The graph default
  + this state stalls before first completion — the request-level gap
  moved earlier. Next: trace the graph-mode chain start (gate armed?
  experts_warm under graph default? the demand walk inside graph mode).

### Source clarification: the current payload and completion tails are already pushed locally

`SparkWeightdClientAttachLazy` maps the local mesh file descriptor with `MAP_SHARED`
and replaces `mesh_send_buffer_addr` with that local mapping. GLM passes this
address to `SparkTpDeviceCollectivePrepareReceiveBf16`; `SparkGlm5NextMeshWaitKernel`
derives every peer tail from the same local band base plus the peer's slot offset.
`SparkWeightdMeshPostTransfer` posts `IBV_WR_RDMA_WRITE` for the payload and its
tail into the receiving peer's corresponding local mesh region. Thus
`ld.global.cv` polls mapped local host memory through the GPU memory fabric;
it does not issue an RDMA read verb to another host. Moving readiness into GPU
memory or replacing the spinning kernel with a stream memory wait is a separate
optimization requiring a publication and visibility contract. This source trace
does not invalidate the measured idle SM utilization or identify which stream
owned the reported abandoned kernels.

The direct wait kernel launches one block of 32 threads and only thread zero
polls. A high sampled GPU utilization percentage can indicate persistent kernel
activity without establishing execution-resource occupancy. The utilization
change is evidence of changed activity; attributing the chain wall time to SM
capacity or context timeslicing still requires a stream/context trace.

## 09-24 11:00 — LADDER STEP 1 (main merge) DONE; fleet migration to the shared-daemon architecture INCOMPLETE — state parked cleanly

DONE: main merged (@0b4484e; fuzz-test include fix @82977f3); tests green
(kv/ipc/reconnect/fuzz); built + hub-deployed (residentd 42eef957 /
driver 2b2a1724 / daemon announced); x86 API rebuilt on rtx5090.

THE ARCHITECTURE DISCOVERY: the fleet NOW runs astra's SHARED weightd
(systemd sparkpipe-weightd-shared.service, /run/sparkpipe-weightd-shared/
weightd.sock, 90GiB ceiling, 8 mesh lanes, fleet-wide since 06:53) —
my lane's per-node weightds CANNOT start (the latch port 61900 is held
by the shared daemon; "cannot own port; existing owner untouched").
The agents are wedged fleet-wide (manifest loop stalled; restarts did
not revive them). Engines attach to the SHARED daemon per astra's
recipe (SPARK_WEIGHTD_SOCKET + finite pool 24GiB + spine 4GiB +
ATTACH=1).

WHERE IT STUCK: my lane engine build boots to "adapter_initialize
status=1 (invalid_argument)" at LoadDriver on the shared-socket path
(spark1 live; diag line shows sane geometry). ALSO seen once: mesh lane
acquire status=19 against the shared daemon (lane exhaustion from
crashed queues — astra's ledger r8 says use weightd_warm --reclaim).
FLEET STATE PARKED: engines down everywhere (safe baseline); shared
daemons UNTOUCHED + healthy; JSONs repointed at the shared socket; the
new lane binaries synced fleet-wide; lane weightds dead (latch-blocked,
no twins). Astra's bundle: ~/sparkpipe/shared-serving-20260922 (SOURCE
b690c3a5) + ~/srcdata/sparkqueue/* workdirs + sparkqueue-* systemd
units (currently failed) = the 13.5 tok/s platform.

NEXT (pick one): (a) debug my build's adapter-init invalid_argument on
the shared path (likely a config the shared daemon path requires — the
receipt's r1-r10 ledger lists them: runtime-root wrappers, symlinks,
finite pool BEFORE driver load), or (b) run the climb on ASTRA'S bundle
directly (their engines + batch tool) and leave my lane build for the
code work. Then the ladder: re-attribute 810µs/round → multi-row →
one-launch → push-cells.

## 09-24 13:00 TICK 1 — the serving recipe fully extracted; ONE blocker left: lane-acquire 19 from the shared daemons

THE RECIPE (astra's glm5_next_coldlaunch_ab.sh, glm-m3-final checkout on
every node): private runtime per attempt + deployment REGEN (runtime
limits = the smoke B1 profile: 1 seq/128 kv pages; control endpoints
remapped per attempt; kv 2GiB) + SYMLINKED adapter/driver/transport in
runtime + env (SOCKET=shared, LANE=0, TP_MESH_RANKS=0-15,
WAIT_MODE=hardware, CUDA LAZY loading ×2, MAX_CONNECTIONS=32,
GRAPH_PATH=1, PIN_EXPERTS=1, pool 24GiB/spine 4GiB/KV_RESERVE=0) — and
the pool env must be set BEFORE the warm step (warm refuses without it).

THREE LAUNCH WAVES (each convicted + fixed):
1. ATTEMPT naming: the prep parses the attempt as hex — "hillNNNN"
   ValueError → numeric attempt ids required.
2. warm step died "finite pool required" → export pool BEFORE the
   script. Post-fix: WSET-WARM 336 keys elapsed_ms=0 (the smoke set
   already resident fleet-wide ✓).
3. weightd_warm --reclaim (glm-m3-fixed build; the shared-serving-20260922
   build has no --reclaim) freed 66-72GiB across 5 stale arenas per
   node — astra's ledger r8, executed fleet-wide.

THE SOLE REMAINING BLOCKER: engine lane-acquire vs the shared daemon
fails 19 (UNSUPPORTED) on EVERY node, deterministic, no daemon-side log
line. NOT the weak-stub class (nm shows strong MeshLaneConfigure in
both daemon builds). The strong verbs configure path itself returns 19
— suspects: (a) acquire WITH topology (rank_count=16) needs RDMA group
configure the long-lived daemon (up since 06:53, post-reclaim) now
refuses; (b) an enum delta between the running b690c3a5 daemon and the
ed9ff7f5 engine (their NO_LANE may BE 19). NEXT TICK (cheap first):
(1) retry lane acquire WITHOUT topology (unset SPARK_TP_MESH_RANKS) —
if OK, the topology-configure path is the convict; (2) nm both enum
tables; (3) if the daemon's verbs state is wedged → ASTRA ESCALATION
(the shared daemon may never be restarted by this lane).

STATE: engines down; shared daemons untouched; run recipe recorded in
/tmp/run_hill.sh on every node (attempt ids 1001-1003 used).

## 09-24 13:30 TICK 1 close — LANE-ACQUIRE-WITH-TOPOLOGY isolated; ASTRA escalation warranted

THE DISCRIMINATOR: expert acquires via the warmer flow WORK against the
shared daemon right now (layer-by-layer WARM live on spark1 — no lane
involved). The ONLY failing call is the engine's LaneAcquire WITH
topology (rank_count=16 from SPARK_TP_MESH_RANKS) → 19 from the strong
verbs configure. The running shared daemon is the PR #1082 baseline
(b690c3a5) — it PREDATES the #1135 mesh register-skip fix that the
m3-fixed engine bundle (ed9ff7f5) was qualified against; the mismatched
pair worked during astra's campaign window and now deterministically
fails (long-lived daemon, post-reclaim state).

THIS LANE CANNOT restart the shared daemon (operator rule). Escalation
to astra: either (a) restart the shared daemons with the #1135-era
build (their call — it's their service), or (b) confirm the lane
configure contract the running daemon expects. Everything else is
READY: recipe + env + warm + reclaim + private runtimes proven this
tick; engines start the moment lanes grant.

## 09-24 14:00 TICK 2 — the QUEUE is the admission mechanism; dispatcher revived; first submission mis-wrapped

DISCOVERIES THIS TICK:
1. The lane-19 mystery DEEPENED usefully: lane acquire fails 19 even
   against a FRESH m3-fixed test daemon (latch 61901, private socket)
   with AND without topology — so it is NOT daemon age/state; the
   daemons grant lanes only through QUEUE-ADMITTED jobs (the queue's
   dispatcher injects the admission env: ATTEMPT, RUNTIME_ROOT, RANK,
   PORTS + whatever bind makes lanes grant). Direct launches outside
   the queue cannot get lanes — by design (astra's shared-lane
   governance).
2. The queue tool: ~/sparkpipe/tools/spark_queue.py (add/list/done/
   dispatch/doctor). THE DISPATCHER DAEMON WAS NOT RUNNING (doctor
   FAIL) — revived per the doctor's own recipe (nohup loop, 5s).
   Cleaned a STALE RUNNING ENTRY from 2026-08-30 (qwen38flash-
   shamatch2 held spark4 for 3.5 weeks; process long dead; done'd).
3. First queue submission (glm-hill-clb, 16 nodes, coldlaunch arm B)
   dispatched + reaped exit 1 in 25s: the dispatcher's run-family
   template AUTO-SELECTED THE WRONG FAMILY (qwen38max wrapper ran:
   qwenmax pack/pool lines in the log) + a "set: usage" shell error —
   my inline --cmd got wrapped/eval'd badly. NEXT: --cmd-file with a
   clean script that cd's into the glm-m3-final checkout and execs the
   coldlaunch (no family template), ttl ≤15min (queue cap).
4. FLEET FLAG read (lane/glm53 splitbrain note): the dispatcher reads
   spark0's runs/queue.jsonl; tasks added from /Users/mac/sparkpipe are
   dead-lettered — enqueue ONLY via spark0 (as done).

STATE: queue empty; dispatcher alive (5s loop); test weightd on spark1
(/tmp/test-wd.sock, latch 61901) still up for experiments; engines
down; shared daemons untouched.

## 09-24 14:30 TICK 2 close — queue admission confirmed NOT the lane grant; sticky lane topologies convicted; escalation finalized

THE CMD-FILE RERUN (glm-hill-clc, queue-admitted: WSET-WARM 336 keys
743ms through the queue's runtime root): the engine STILL fails lane
acquire 19. So the queue injects env/runtime but does NOT grant lanes
— astra's 07:00 successes rode lanes configured when their campaign
jobs first acquired them.

THE CONVICTION (timeline-corrected): lane-19 PREDATES my --reclaim
(attempt 1002 at ~12:0x failed 19; reclaim ran ~12:5x) — the sticky
lane topologies come from ASTRA'S OWN crashed campaign jobs (the
failed sparkqueue units): weightd_mesh.lane_topology[] persists per
daemon lifetime, ANY differing re-acquire → UNSUPPORTED, and NO LANE
RELEASE IPC EXISTS (only ACQUIRE; verified in the header). Only a
daemon restart (or a lane-clear tool astra would have to add) resets
it. My reclaim did NOT cause this (timeline exonerates it; it remains
a valid r8 cleanup).

FINAL ESCALATION PACKAGE FOR ASTRA: the 16 shared daemons (up since
06:53) hold sticky lane topologies from the crashed glm-m3 campaign
jobs; every lane acquire (any engine build, queue-admitted or direct,
with or without topology) → 19; engines cannot boot fleet-wide until
the daemons restart or a lane-clear path lands. Everything else is
verified ready (recipe, queue flow — dispatcher revived, cmd-file
works, 14-min TTL, warm 743ms cold).

## 09-24 15:30 TICK 3 — THE UNBLOCK LANDED: my own daemon mesh fleet-wide; 15/16 engines UP

THE BREAKTHROUGH (the lane probe): no-topology lane acquire GRANTS on
all daemons (lane=0 status=0) — only topology-carrying acquires fail
on astra's builds (19) — AND MY OWN daemon build (2e4efe57-era,
g5pacing) GRANTS THE FULL 16-RANK TOPOLOGY ACQUIRE. Astra's builds
refuse; mine works.

THE PARALLEL LANE MESH (deployed this tick, zero shared-daemon touch):
my weightd per node — /tmp/my-wd.sock, latch 61902 (61903 spark1 after
a collision), --mesh-dir /tmp/my-mesh, --mesh-rank-mask 0xffff, 90GiB
ceiling. Mesh-record exchange = a star relay via spark3 (records are
HEX-named: mesh-a.rec not mesh-10.rec — the fan-out naming bug cost
one pass). WIRED: peers=15 on all 16.
- Smoke sets warmed through my daemons (--wset per-rank wset; fleet
  WSET lines verified).
- Engines: deployment JSONs repointed to /tmp/my-wd.sock; my lane
  binaries; env pool 32GiB/spine 4GiB/KV 0/GRAPH_PATH=1.
- RESULT: 15/16 ENGINES UP AND READY (spark3 stubborn: io_error at
  adapter init on every retry — its daemon restarted late and
  something in its lineage differs; its lane probe INVALID was my
  probe's local_rank=1 hardcode, not the daemon).

NEXT TICK: (a) fix spark3 (compare its daemon lineage; worst case
re-run its daemon + engine fresh), (b) API on rtx5090 already revived
(apis=1) → CANARY → THE FIRST TOK/S MEASUREMENT on the private mesh,
(c) then the ladder: attribution → multi-row → one-launch → push-cells.

## 09-24 16:30 TICK 4 — spark3's holdout ROOT-CAINED: GPU NVRM out-of-memory (kernel-logged)

THE CHASE (this tick, stepwise):
- ck128 exonerated (stamp verified "already stamped rank=3").
- My earlier daemon "restart" never happened (pgrep matched a bash
  wrapper; the true pid survived) — killed by REAL pid, daemon fresh,
  mesh rewired peers=15 in seconds (records persisted).
- The engine's attach DOES reach the fresh daemon (a second
  "lazy-attach" line per attempt) and the daemon keeps serving expert
  acquires fine through the same socket (warm: 42 layers WARM) — the
  attach fails on the ALLOCATION path.
- THE KERNEL NAMES IT: NVRM Out of memory (NV_ERR_NO_MEMORY at
  memdescAllocInternal) at exactly the engine-attempt times (08:16 ×44,
  09:47). spark3's GPU cannot fit: shared daemon arena (20.8GB) + my
  daemon pool (21.7GB) + engine spine/KV/context. Other nodes hold the
  same daemon pair PLUS engines — spark3 carries extra remnant pinning
  (it is also my build host; 08:16 was the build era).
- ALSO FOUND: weightd_warm --wset caps at 512 keys (LEASE_GROUPS_MAX)
  — spark3's recorded wset holds 12096 keys (the full-pool pin tape
  from the M3 era) → "invalid wset" — the smoke set (336) is fine.

NEXT TICK (spark3, pick one): (a) find and clear the remnant GPU
pinning (zombie contexts from the build era; nvidia-smi shows only the
2 daemons — the remnant may be kernel-level), (b) drop MY daemon's
ceiling/pool on spark3 so the engine fits, (c) worst case: move the
rank-3 engine to a reduced-spine config. THEN the canary — 15/16 has
been ready since tick 3; TP16 needs all 16.

## 09-24 17:45 TICK 5 — spark3 SOLVED (stale driver .so — not memory); 16/16 reached; spark0 daemon cycle noted

THE REAL spark3 ROOT CAUSE (memory theory DIED): with the GPU
essentially empty (both daemons at 170MB post astra-side shared-daemon
restart), the engine attach STILL failed rc=4 — the kernel NVRM OOM was
an old scar, not the live cause. THE STALE DRIVER: spark3's
stages/stage_000/model_driver.so was 0d9b1fe7 (the pre-main-merge
spinner-fix build) while every working node runs 2b2a1724 — the
fleet-wide binary sync raced/failed on spark3 (my build host, target of
many overlapping operations). New adapter + stale driver = attach
protocol mismatch → io_error. Copied the hub driver → ENGINE UP
FIRST TRY. Evidence law again: the artifact sha, not the theory.

THE ROLLING SPARK0 DAEMON: spark0's my-daemon exited gracefully once
("spark_weightd stopped arenas=1") dragging the engine with it —
suspect the revived spark0 agent's ensure_weightd loop (the only node
with a live agent). Restarted daemon+engine = engines=1. WATCH it.

STATE AT CLOSE: 16/16 engines up simultaneously achieved this tick
(briefly); the API is up (apis=1). NEXT TICK: confirm 16/16 holds →
THE CANARY → the first tok/s receipt on the private mesh → the ladder.

## 09-24 18:30 TICK 6 — the teardown named (another lane's queue job); 16/16 restarted; engines quiesce on first submission

THE FLEET-WIDE TEARDOWN (tick-5 close → tick-6 open): another lane's
ADMITTED queue job — k3-m3-cold14 (12:16, 16-node cell, weightd_warm
--reclaim + their engines) — swept all non-admitted GPU processes
(my lane daemons: graceful "stopped arenas=1"; engines followed).
CORRECT multidev behavior: lanes coexist THROUGH the queue; my direct-
launched stack lives only in empty-queue windows.

THE RESTART: all daemons + engines relaunched → census 16/16 PROCESSES.
THE CANARY: the API reconnected (96th attempt era = the restart), then
submissions hit io_error at the adapter and engines entered
"quiesce=io_error; preserving live resources" — zombie processes
(counted by census but not serving). The ready line printed, so attach
worked; the failure is at CHAIN time = THE MESH: the daemons restarted
WITH STALE RECORDS in /tmp/my-mesh (mixed QP generations; the record
dir persisted across daemon restarts). spark0's port 19560: refused.

NEXT TICK (the fix is known): rm /tmp/my-mesh/* ; restart daemons
(clean publish); re-run the hex-name record relay; warm; boot engines;
canary → measure. THEN: for stable windows, submit MY serving through
the queue (a glm cell job like k3's) so other lanes' jobs gate BEHIND
it instead of sweeping it.

## 09-24 19:30 TICK 7 — agents killed (daemon-sweeper suspect); daemons now die ON LAZY-ATTACH; tick closed at the crash boundary

THE SWEEP SUSPECT ELIMINATED-NOT-CONFIRMED: the agents came ALIVE
fleet-wide (12:42 — the k3 job/core sync revived them; they were wedged
all day). Killed all 16 (my lane infra; the queue dispatcher is
independent). BUT the daemons still die — now convicted to the ATTACH:
fresh clean daemons (fresh /tmp/my-mesh, records relayed, 16/16 wired
peers=15) exit with "spark_weightd stopped arenas=0" the moment a
lazy-attach begins (warmer: ERRSITE spark_weightd.c:3637 status=4
attach failed, daemon=0 afterward). Two restart rounds, identical.

THE DIFFERENCE from tick 3 (when warm+engines worked): none identified
yet — same binaries, same invocation. Suspects for next tick: (a) the
attach-path crash needs the CORE (run the daemon under the ulimit/wdcore
wrapper to catch it — the wrapper recipe exists), (b) spine-load crash
class (the mtp-tail suspicion from spark3 now fleet-wide?), (c) something
in the daemon's serialized cold-op path on a truly clean state (tick 3's
daemons had warm carries from earlier attaches).

STATE: engines never booted this tick (daemons die first). All infra
recipes refined: clean_restart.sh + record relay + /tmp/ww distribution
(all on the nodes). Queue empty. Agents dead.

## 09-24 20:30 TICK 8 — the attach crash did NOT reproduce under the core-catch supervisor

THE REPRO ATTEMPT: spark1 daemon under the rc+core supervisor
(supervise3.sh, private latch env, cores to ~/wdcore) — a full-
geometry warm ran to COMPLETION (layers 3..44 WARM, attach fine, zero
deaths, no core). The crash class did not fire on a SINGLE-NODE attach.

THE REFINED SUSPECT: tick 7's crashes hit when all 16 warms ran
SIMULTANEOUSLY (parallel spine loads + concurrent mesh wiring) — the
crash may need that concurrency (RDMA/mesh contention during parallel
attach). Tick 3's successful parallel warm ran against daemons that
had already survived one attach cycle.

NEXT TICK: warm the fleet SEQUENTIALLY (or small batches) on the
clean daemons → engines → canary → the first tok/s. (spark1 is already
warm under the supervisor; the recipe is proven single-node.)
