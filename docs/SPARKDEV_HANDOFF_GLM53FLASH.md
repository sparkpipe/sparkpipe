# SPARKDEV HANDOFF — glm5.3 flash TP16 fleet: serving WORKS, now optimize

**Date:** 2026-09-16 · **Author:** kimi (sparkdev takeover) · **Branch:** `lane/glm53-takeover` · **PR:** #1027

## 1. Where things stand

**The fleet serves end-to-end.** 16/16 engines, TP16, glm5.3 flash fp8. The qualification fixture (176-token prefill + 32-token decode, `quality-fixtures-glm5.3-flash.json` case 0) completes with coherent, correct output. A 16-token probe returns perfect English continuations. The system survives every failure class we threw at it: engine kills, weightd kills, api restarts, connect storms, dead NATted sessions, and fleet-wide recycling.

**Perf status (measured live, warm fleet):**
- ~500ms per 1-row prefill chunk end-to-end ≈ 1.9 tok/s/lane steady state.
- `CHAIN-TIME` instrumentation (in the tree, prints per chain): **~500ms total = ~130ms allreduce (91 rounds ≈ 1.3ms/round) + ~370ms host orchestration/kernels.**
- The 50-100µs/round allreduce target is NOT met (1.3ms ≈ 20×). The GPU-side graph path is the 10× lever (§7.1). The ~370ms/chunk host overhead (~7.5ms/layer) is what multi-row prefill amortizes (glmdev's bug, §7.2).

## 2. PR / merge state

- PR #1027 contains everything below. `origin/main` was merged in (commit `70209dc` + build-fix `3d67965`). Resolution choices:
  - `spark_weightd.h`: kept 16 slots/rank + u64 cast, kept main's export-lease API.
  - firmware.h: main's `#ifndef` TAP defines with the 5-layer values as defaults.
  - module.c: graph path stays **env opt-in, default OFF** (`SPARK_GLM5_NEXT_GRAPH_PATH=1`). Main defaults it ON at compile time — deliberately overridden; the fleet is validated OFF.
  - fleet agent: kept `G5_API_DISABLED` gate + main's `api_root`; dropped duplicate weightd start (ensure_weightd owns it).

## 3. Root causes fixed this session (all verified live)

1. **Fleet-killer SIGSEGV** — `SparkGlm5NextCompleteOnWorker` NULL-deref: `epoch_device` is armed at lazy-pack attach (boot), `decode_miss_host` is allocated on first chain setup; completions in that window derefed NULL+264. Caught via new fleet-wide cores (si_addr=0x108). Guard added.
2. **Idle-timeout wedge (self-latching)** — `last_activity_ns` never reset at accept/close: after any 30s-stale disconnect, every new connection was idle-closed before its hello was read. The fleet could never serve a reconnecting api again.
3. **Client generation pinned at 1** — module reset dedup (`generation > reset_generation`) skipped the KV-lane purge for every session after the first; leaked lanes accumulated to permanent admission BUSY. Now: generation bumps per accept, **every hello arms a reset**.
4. **Api→engine state desync** — api prefix cache believed prefixes were engine-resident after engine resets purged them → continuation chunks at nonzero positions into a purged cache (NOT_FOUND). The batch engine now fingerprints the session (per-rank client generations) and invalidates prefix cache + resident bindings + requeues requests on any change (`SparkModelPipelineClientSessionFingerprint`).
5. **IO_ERROR killed requests** — dead engine mid-dispatch restored the request, then the error path massacred it with the same status. IO_ERROR is now always transient-restore; poll descriptors tolerate disconnected ranks (fd=-1); per-rank reconnect backoff (100ms→5s, was zero-delay SYN storms vs listen backlog 1).
6. **KV lane leaks** — PREPARED/COMMITTED lanes of dead requests never freed: takeover by any new request id; EXECUTING lanes reclaimed after >60s (frames finish in ~1.4s).
7. **Expert lease exhaustion (64/64)** — three layers: (a) sweep-path chainfails skipped lease release → quiet-release at all four chain terminal sites; (b) weightd never reaped dead connections' leases → `SparkWeightdLeaseReleaseOwner` at close (a dead consumer's mesh endpoints die with it — safe); (c) table 64→256 because releases legitimately lag a full layer (16 chains × lag pinned 63-64/64, trace-verified).
8. **The silent fleet freezer** — `LazyRetryRetained` was only submitted from module **Destroy**; every transient chain-drain/cleanup failure parked a pipeline slot + its lease forever. Retry now scheduled from both retain sites + nudged at chain start.
9. **The "squatter"** — sparkf NATs 10.10.250.0/30→fleet (nftables masquerade). Dead api sessions held engines' single-client slots via stale conntrack. `ss -K` clears; **TCP keepalive** on accepted sockets (10s/5s/3) reaps dead peers permanently.
10. **Allreduce spin** — `nanosleep(1µs)` (rounds to ~55µs tick) then `sched_yield`, now pure spin: ~25% off round time. Floor is now genuine peer-skew/propagation.

## 4. Architecture map (what runs where)

- **rtx5090 (10.10.250.2, tailscale 100.123.97.61, user `spec`):** the api (`sparkpipe_model_api`, port 8433, systemd user unit `g53-api`), release hub (`fleet-release.service` serving `~/release` on :8802), draft/tokenizer host. Reachable from the workstation as `rtx5090` (ProxyJump sparkf).
- **sparkf (10.10.100.25):** aarch64 build host ONLY (`~/g5n-rd-build` — do NOT touch `~/sparkpipe-build`, it's glmdev's). Also NATs the 5090's traffic to the fleet (masquerade — api appears as 10.10.100.25 on some ranks; this is NORMAL, not a squatter).
- **16 sparks:** `fleet-agent` (systemd user) runs `~/sparkdata/core/bin/fleet_node_agent.sh` → pulls releases over tailscale HTTP from the 5090, runs `weightd` (mesh + expert arena) and `residentd` (the engine) per node. Engines listen on 19560+rank. Single-client control protocol (hello → reset → submissions).
- **weightd** owns: expert pack lazy-attach arena (32GB pool, `G5_EXPERT_POOL_BYTES`), lease table (256), mesh QP wiring + records. Mesh records ship via scp to `spec@100.123.97.61:release/qpn/<host>/...` and fetch via HTTP from same.
- **KV lane protocol:** EMPTY→PREPARED→COMMITTED/EXECUTING→EMPTY per resident slot; prefill chunks are 1-row submissions (`SPARK_MODEL_API_MAX_PREFILL_ROWS=1` on the api — multi-row blocked by glmdev's prefill bug).

## 5. Build/publish/test loop (exact commands)

All GitHub ops via `tools/sparkpipe_github_pat.sh` (never raw gh/git push; identity must be `sparkpipe`). Worktree on the workstation: `/Users/mac/lane-takeover`.

**Full publish (driver+residentd+weightd):**
```sh
ssh sparkf 'cd ~/g5n-rd-build && git fetch -q origin lane/glm53-takeover && git reset -q --hard FETCH_HEAD && export PATH=/usr/local/cuda/bin:$PATH && make -j8 build/sparkpipe_model_residentd build/sparkpipe_weightd && SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d" " -f1) && make -C modules/glm5_next_resident_decode_stage adapter publish EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 CONTRACT_SHA256="$SHA" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a && build/sparkpipe_model_compile --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json --library build/module_library --output /home/sparkf/sparkdata/out --cc /usr/bin/cc --include include --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread'
```
Then install into `~/release/{glm53flash.fp8.tp16,core}` (residentd→bin/, weightd→core/bin/, model_driver.so→stages/stage_000/, adapter .so→lib/), regenerate each root's MANIFEST (`find . -type f ! -name "MANIFEST*" ! -name "UPDATE*" -exec sha256sum {} + | sed "s| \./| |" | sort -k2 > MANIFEST`), `touch UPDATE`, and `rsync -a --delete ~/release/core ~/release/glm53flash.fp8.tp16 spec@10.10.250.2:~/release/`. Agents converge in ~3 min. **SPARK_BATCH_BUCKET -D law from the runbook still applies** — check `docs/GLM53FLASH_RUNBOOK.md` if the module Makefile changed on main.

**Api-only change:** build on the 5090 (`cd ~/sparkpipe-build && git fetch/reset && make -j16 build/sparkpipe_model_api`), `systemctl --user stop g53-api`, cp, start.

**Validation:**
```sh
# fixture (176 tokens, expect coherent output):
ssh rtx5090 'systemctl --user restart g53-api && sleep 12 && python3 -c "
import json, urllib.request
d = json.load(open(\"/home/spec/sparkpipe-build/qualification/ds4_eval/quality-fixtures-glm5.3-flash.json\"))
c = (d[\"cases\"] if isinstance(d,dict) and \"cases\" in d else d)[0]
ids = c.get(\"prompt_token_ids\") or c.get(\"prompt_ids\")
req = urllib.request.Request(\"http://localhost:8433/v1/completions\",
  data=json.dumps({\"model\":\"sparkpipe-model\",\"prompt_token_ids\":ids,\"max_tokens\":32,\"temperature\":0}).encode(),
  headers={\"Content-Type\":\"application/json\"})
print(urllib.request.urlopen(req,timeout=1100).read().decode()[:400])"'
```
Detokenize with `/home/spec/sparkpipe-build/qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json`.

**Fleet ops:** cores at `~/sparkdata/glm53flash.fp8.tp16/core.*` (core.%p set, agent has ulimit -c unlimited). gdb any live process with `sudo -n gdb -p PID -batch -ex bt` (weightd/engines have symbols; driver .so has NO line info — use `info proc mappings` + `nm`/`addr2line` on sparkf's `~/sparkdata/out/stages/stage_000/model_driver.so`). Lease table inspection: `WD-LEASE-TRACE` in `~/weightd.log` (acquire/release/occupied per op — diag, strip when done).

## 6. Fleet gotchas (hard-won, each cost hours)

- **Single-client engines**: one dead connection wedges an engine forever. Keepalive covers silent death; `ss -K dst <ip>` on the engine host clears stuck ones.
- **sparkf NAT**: api traffic appears from 10.10.100.25 on some ranks — never "fix" this as an intruder.
- **spark5** gets foreign GPU squatters (`t1_reference*.py` respawning from an external harness) — kill on sight; operator has authorized.
- **Memory gate**: agent blocks engine start until `MemAvailable >= packs-du + 8GB`. Stale pack variants (`.old`, `.mtp-restored`) inflate du — delete them fleet-wide if the gate livelocks (sparkf hit this).
- **du -sBG rounds every small file to 1G** — the gate's pack_gb is inflated ~20GB by tiny receipt files. Don't panic at the number; it's still just a threshold.
- **Release server staleness**: `fleet_release_serve.py` chdirs at start — restart `fleet-release.service` on the 5090 if `~/release` was ever replaced; after rsync publishes it's fine (subdirs only).
- **Stranding risk**: agents pull from the hub; if you change the hub address, fan the new agent out by scp first or nodes can never reach the new hub to learn it.
- **`(void)runtime;` stub on main** in `FailContinuationLocked` didn't compile — deleted in the merge; if glmdev's main reintroduces it, delete again.
- **Engines exit on INTERNAL_ERROR by design** (failed latch) — a route-level error kills the rank; agent restarts. If you see `progress stage=... internal_error`, the fix belongs in the route path, not the loop.
- **api single-session queue**: zombie requests block new sessions behind `queued_behind`; restart g53-api to clear.

## 7. What remains (priority order)

### 7.1 The allreduce: 1.3ms/round → 50-100µs target

**PR #1030 state (lane/glm53-graph-replay) — as of 2026-09-17:** the graph path
captures and replay #1 runs; replay hangs with ALL peers' slot tails reading 0
while doorbells land (relay/copy layer, not the sequence protocol — per-slot
generation counters are in, ruling out cross-rank counter skew). The fleet run
also exposed: `apply_manifest` scope-arg crash-loop (`set -u`), the env-word
bash trap in the agent launch, and an eager-admission regression on the branch
(KV_CAPACITY rejects on fresh engines — from the -614-line module surgery).
Do NOT merge until: (a) the admission regression is found, (b) a replay #N≥3
completes repeatedly on the live fleet, (c) CHAIN-TIME shows replay beating
1.3ms/round warm eager. Graph stays opt-in (`G5_GRAPH_PATH=1` drop-in per node;
agent passes it as SPARK_GLM5_NEXT_GRAPH_PATH).

`CHAIN-TIME` shows 91 rounds/chain ≈ 130ms of ~500ms. Pure host spin is already at its floor (peer-skew + 5 stream-ordered memcpys + RDMA/doorbell propagation per round). **The real lever is the GPU-side mesh path** (`capture_armed`: `SparkGlm5NextLaunchMeshPublish/Wait` run publish+wait on-GPU, no host round-trips). It's wired but disabled: `SPARK_GLM5_NEXT_GRAPH_PATH=1` opts in (default off since merge). Main's compile-time default is ON — re-validate before flipping the default. Steps: (1) enable env on ONE node config, probe; (2) watch for the historical graph-path issues (chainfail poisoning, EPOCH-MOVE lease drops); (3) measure CHAIN-TIME allreduce_ms. The characterize bench `tools/hardware/spark_tp_device_collective_characterize.cu` doesn't link (needs `SparkGlm5NextLaunchMesh*` from the module + `-lcuda`) — fix its Makefile rule when a low-level A/B is needed.

### 7.2 Host overhead: ~370ms/chunk (~7.5ms/layer)
45 layers × per-layer admission/route/lazy machinery on the host. Amortized by multi-row prefill (glmdev's KV-nondeterminism bug, `docs/CONSULT_multirow_kv_nondeterminism.md` on `lane/glm5next-mtp-accept` — glmdev owns it). Until multi-row lands, `SPARK_MODEL_API_MAX_PREFILL_ROWS=1` stays. Do NOT lift it unvalidated.

### 7.3 Fuzz harness (operator requirement)
Random `kill -9` of residentd/weightd on random sparks during serving; assert time-to-serve and never-wedge. All kill classes are handled in code (§3) but never fuzz-tested continuously. Build it as a spark-queue job (`tools/spark_queue.py add`) driving the fixture in a loop. Expect: engine restart ≈ 1-2 min to serve again (weightd mesh rewire + lazy attach + memory gate).

### 7.4 Engine boot time
Currently ~2-3 min to ready (lazy-attach bake + mesh). Profile before optimizing: the memory gate, lazy-attach retries (600×1s), mesh rendezvous, weight load order are the suspects.

### 7.5 Cleanup debt
Diag prints to strip when green: `ADMIT9-*`, `KV-TAKEOVER`, `KV-RECOVER`, `CHAIN-TIME`, `LAZYWORK-*`, `WD-LEASE-TRACE`, `CKEY-*`, `MESH-*`, `G5N-DBG`, `client_reconnect_fail`, `batch_rejected`, slot_mismatch/lease_reject in residentd. Keep ERRSITE. Also: `test_model_pipeline_client` host test fails at line 1632 (pre-existing, also fails on merge-base — not ours).

## 8. First three actions for the next dev

1. Verify READY=16/16 on the merged publish, rerun fixture case 0, then merge PR #1027.
2. Graph-path experiment per §7.1 (this is the 10× allreduce lever and the operator's stated goal: 50-100µs/round, 50% of memory-BW roofline ≈ 100 tok/s B1).
3. Fuzz harness per §7.3 — the operator explicitly asked for "random kills, measure time-to-serve, never wedge."

The fleet is yours. It serves. Don't let it regress: every change through PR, every publish through the loop in §5, every claim measured with CHAIN-TIME or the fixture.
