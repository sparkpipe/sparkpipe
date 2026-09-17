# GLM 5.3 Flash TP16 Takeover — Status Handoff (2026-09-15)

Branch: `lane/glm53-takeover` (all pushed). Fleet: serving mechanically end-to-end on TP16.

## What works now (verified on fleet)

- Full requests complete with status 0: 16-token, 100-token, and 176-token prompts,
  32-48 output tokens each, over the mesh allreduce (16 ranks, both bands).
- The mesh epoch protocol is rewritten and converges cleanly:
  - The band base cell carries the full chain key `(epoch<<24)|request_id`.
    Rank 0 writes it on every ChainKey and bumps the epoch EVERY chain
    (per-chain unique epochs — stale slot data can never exact-match again).
  - Peers spin for their own request id in the cell and reject their
    already-consumed cell (`consumed_cell`) — api restarts no longer skew the
    fleet one epoch apart (request ids reset to 1 per api session).
  - `cancel_seen` is captured on BOTH ChainKey branches (its absence made
    every chain insta-abort on the persistent storm-era cancel cell).
- Mesh liveness: parallel all-peer wait scan (was sequential per-peer blocking —
  fast ranks lapped the 4-slot ring and overwrote evidence slower readers
  needed); slots per rank 4 -> 16 (lap window 16 rounds).
- Chainfail reentrancy SEGV fixed (ChainFail cleared `active` only at the end;
  BroadcastCancel reentered the completion callback for the same chain ->
  double free; caught under gdb).
- API robustness redesign (the "fragile api" class):
  - Submissions stamp the live control epoch learned at hello (max over rank
    client generations) instead of constant 1 — the adapter reset gate can no
    longer wedge every engine (ADMIT9-RESETGEN: reset_generation=2 vs
    control_generation=1).
  - `SparkModelPipelineClientRecover` clears the permanent `failed_status`
    latch and reconnects FailStopped rank clients; called from
    `SparkModelBatchEngineProgress` (every poll cycle) and ReopenAdmission.
  - In-flight request budget default 240s -> 900s, env override
    `SPARK_BATCH_INFLIGHT_BUDGET_NS`.
- Infra: 64-bit `SPARK_WEIGHTD_MESH_BUFFER_BYTES` (u32 overflow 4MB*256*4=2^32
  -> memfd ftruncate'd to 8KB -> SIGBUS at boot); cudaHostRegister of the 2GB
  lane span is best-effort on GB10 unified memory (pageable fallback);
  CUDA-graph wave path disabled by default (`SPARK_GLM5_NEXT_GRAPH_PATH=1`
  opts in — it wedged in cudaStreamSynchronize inside SparkGlm5NextGraphEnsure).
- weightd update path: release tree -> deployment bin -> `core/bin` ->
  install_core deliberate restart (fleet-wide, done; new weightd serves the
  4GB region).

## THE open problem: numerics are degenerate

Requests complete but output is garbage:
- "The capital of France is" -> 16 x token 154822.
- Fixture case 0 (176-token QA prompt) -> 'C. 0C. 0C. 0...' (expected "B").
- 16-token prompt -> ' intelligent intelligent ... intelligenthing'.

Output is prompt-dependent but degenerate/repetitive. The mesh protocol
exact-match makes stale-data reduce impossible by construction (chain_key +
round exact match, 16-slot lap window). So the suspects are the compute path:

1. Allreduce combine: payload/seq offsets per band, the two instances
   (main band0 8192B ops, HC band1 32768B ops), pageable-memory D2H path
   (registration fallback) — verify the combine actually sums fresh partials.
2. Module partial computation (attention/MLP partials per rank before reduce).
3. KV/position state across waves (hi reaches 176 rows now; >128-row waves
   use the multi-wave path — the glmdev's prefill bug territory).
4. TP16 stagepacks themselves (sysadmin-regenerated; MTP split).

Bisect plan (next session, first thing):
- Deploy `glm53flash.fp8.tp4pp4` (TP4xPP4 reference path, same module/kernels,
  narrower allreduce). If tp4pp4 is ALSO degenerate -> module/pack math, not
  the mesh. If tp4pp4 is correct -> TP16 allreduce path.
- Oracle: ds4_eval on rtx5090 (`qualification/ds4_eval/`,
  `quality-fixtures-glm5.3-flash.json` 92 cases with expected answers;
  tokenizer at `ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json`).

## Performance state (post-numerics task)

- ~10ms per mesh round (91 rounds per 45-layer decode chain) => ~1.1 tok/s
  decode, ~15s prefill(16). Target: 50-100us allreduce => ~100 tok/s B1.
- The round cost is NOT the relay (RDMA ships land in ~us) and NOT GPU compute
  (GPU idle during waits) — profile the host publish path (5x cudaMemcpyAsync
  per round, stream-queued) + spin + combine next.

## Ops notes (hard-won)

- Build/deploy loop on sparkf `~/g5n-rd-build`: `git fetch/reset` then
  `make build/sparkpipe_model_residentd build/sparkpipe_model_api
  build/sparkpipe_weightd`, module `adapter publish`, `sparkpipe_model_compile`,
  copy 5 artifacts into `~/release/glm53flash.fp8.tp16/`, regenerate MANIFEST
  with `sha256sum | sed 's| \./| |' | sort -k2`, `touch UPDATE`.
  **weightd needs a manual copy to `core/bin` fleet-wide** for install_core.
  The Makefile rule for sparkpipe_weightd lacks header deps — after editing
  spark_weightd.h, `touch` the .c files or verify the rebuild happened.
- weightd is NOT supervised for liveness (install_core only fires on hash
  change); a dead weightd stays dead. Restart manually with the per-node
  `--mesh-rank N` args (spark0-9 => 0-9, sparka-f => 10-15).
- Engines rotate residentd.log -> .prev on EVERY start; crash evidence dies
  with the next restart. Catch crashes with gdb-as-child (attach froze the
  engine once; child mode works).
- The fleet agent kills/restarts the api whenever the engine recycles;
  long requests need a quiet fleet window.
- Probe: `bash /tmp/probe_req.sh` on spark0 (16 ids, 32 tokens, greedy);
  fixture probes `/tmp/probe_fixture0.sh`, `/tmp/probe_100.sh`.
- Mesh forensics: read the weightd's memfd directly
  (`/proc/<weightd_pid>/fd/<spark-mesh fd>`), slot seq at
  `band*64*4MB + rank*SLOTS_PER_RANK*4MB + (round&SLOTS_MASK)*4MB + bytes`.

## Diag prints still in the tree (strip when green)

`ADMIT9-*` (admission 9-sources), `CKEY-*` (ChainKey), `MESH-*` (spin/staging),
`G5N-DBG` (chainfail), model_residentd client-reset prints.
