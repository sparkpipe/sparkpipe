# cell-runner lane report — 2026-08-29

Lane `lane/cell-runner` (worktree /tmp/lane-cellrunner). Mission: own the
QUEUED GPU receipts — run them, collect, diagnose-on-mismatch, report.
Base: origin/main `e77fca9` → merged `f3a2772` (task-based dispatcher);
lane tips: `609bb59` (r3 cell repair + W3 fd-by-value fix), `ee3d5ea`
(report + ledger correction), `ab1848a` (stub fidelity + manifests),
`e379da0` (import diag), `31d5302` (durable cell roots). All pushed;
nothing self-merged. `make offline-gates` PASS on the final tree
(`OFFLINE GATE PASS: build-all run-tests package-manifest`, 229922 exact).

## Receipts table (numbers only from kill-switch-clean runs)

| Cell | Nodes | Context / batch / topology / precision | Kill-switch | Receipt |
|---|---|---|---|---|
| weightd VMM verify: granularity + cold cuMemMap arena readback + warm attach refcount=2 | spark0 (1) | 8 MiB arena, 4×2 MiB chunks, real driver 580.159.03 / CUDA 13.0 / GB10 | exit 0 + `VMM VERIFY PASS` overall | **GREEN** (first hardware contact; these legs ran on the UNFIXED binary) |
| weightd W3 import leg, in-process | spark0 | same arena, production `SparkWeightdAttachImportMap` | byte-exact D2H through the consumer-local VA | **GREEN after fix** (`in-process import map verified chunks=4 chunk_bytes=2097152`) |
| weightd W3 import leg, cross-process | spark0 | real second PROCESS, SCM_RIGHTS, exec'd child | byte-exact readback in the child | **RED → diagnosed → fix landed; diag re-run queued** (`weightd-vmm-verify-gpu-t4`) |
| R3 flash-decode O128 threshold-0 control | spark8..sparkf (8) | O128 decode batch (md5 2f8f2a57…), glm52.tp8.fp8 TP8, fp8 packs b4734de4 | — (control) | QUEUED `r3flash-glm52-exact-cell-t5` |
| R3 O128 threshold-2048 split | spark8..sparkf | same batch; one key: `decode_split_context_threshold=2048` in config/glm52_stage.json | bit-exact vs control required | QUEUED (same task, phase B) |
| R3 32K decode split vs control | spark8..sparkf | synthesized 32,768-token prompt (deterministic tile of the O128 prompt ids), 256-token budget, `max_context_tokens` 33024 | only after kill-switch PASS | QUEUED (same task, phase C) |
| R2c b1 O128 gate (rows==1) | spark4..spark7 | O128 batch, dsv4 flash TP4, lean packs | pinned hash `a9385d0b…` | **BLOCKED — staging wiped, see below** |
| R2c b8 O128 gate (width-8 bulk) | spark4..spark7 | `max_input_rows` 8 / `max_prefill_rows_per_submission` 8 | SAME pinned hash | BLOCKED (same task) |
| R2c exact-32K prefill (bulk) | spark4..spark7 | 32768-token prompt, 256-token budget | only after both gates PASS | BLOCKED (same task) |

No perf claims beyond the raw receipts; timing legs run only behind their
gates.

## W3 GPU receipt: RED on first hardware contact — two real defects found and fixed

The staged `tools/sparkpipe_weightd_vmm_verify.sh` had never touched a GPU.
First run (spark0): granularity 2 MiB PASS, cold cuMemMap arena readback
PASS, warm attach refcount=2 PASS — then every W3 import failed
`reason=import_handle`.

1. **Consumer side (fixed, `609bb59`):** `cuMemImportFromShareableHandle`'s
   `osHandle` carries the POSIX fd **by value** (cuda.h: "Shareable Handle
   representing the memory allocation");
   `runtime/spark_weightd_attach.c` passed `&batch.fds[index]` — the driver
   read a stack address as an fd number (`CUDA_ERROR_INVALID_HANDLE`),
   deterministically on the first chunk of the first batch. The host stub
   modeled the same dereference, which is why every W3 stub suite and TSan
   run was green while hardware failed. After the fix the **in-process
   import leg went GREEN on real driver** (`chunks=4 chunk_bytes=2097152`,
   byte-exact D2H).
2. **Stub side (fixed, `ab1848a`):** with the code fixed, the stub's
   dereference model made `test_weightd_map` wedge (garbage-fd reads) —
   the stub now reads `fd = (int)(uintptr_t)shareable_handle` and the
   65-chunk two-batch + cross-process legs run green under the REAL
   contract on both sides. This also cleared a full offline-gates hang.
3. **Cross-process leg still RED** — but a standalone platform probe
   (create → export → same-process import → re-export → SCM_RIGHTS →
   fresh-exec child with the production `cudaFree(0)` bootstrap → import)
   passes `rc=0 SUCCESS` on this exact driver/GPU, so the remaining failure
   is consumer-path, not platform. `e379da0` added
   `SPARK_WEIGHTD_IMPORT_DIAG` (prints the exact CUresult + fd at the
   failing import); the diag re-run is queued (`weightd-vmm-verify-gpu-t4`).

## The r3 cell could not have run as committed — repaired on this lane

`tools/devcycle/r3flash_exact_cell.sh` (as queued by lane-r3flashdecode)
had four defects. Fixed in `609bb59`/`31d5302`:

1. **Wrong fleet band**: HOSTS `spark0..spark7` → the deployed
   glm52.tp8.fp8 band `spark8..sparkf` (ranks 0..7 per
   `tools/glm52_gen_deployment.py`; spark0-7 carry no fleet roots).
2. **Kill-switch key on the wrong surface**: the script wrote
   `decode_split_context_threshold` to the top of `model_resident.json`;
   the serving adapter parses it from the ADAPTER stage config
   (`config/glm52_stage.json`, exact-member list — missing = SCHEMA_ERROR).
   The deployed stage configs predate r3 and lack the key; the cell now
   renders both configs per rank.
3. **32K leg inadmissible**: deployed stage configs cap
   `max_sequence_positions` 4096; the cell renders 32768 (module ceiling
   1,048,576). A sequence the KV pool cannot hold is refused loudly at
   admission — recorded, never faked.
4. **Ready-check port**: 19480+rank (the band's own control ports), was
   dsv4's 18480. Plus: runtime roots at the cell root so the staged r3
   driver loads; per-rank packs symlinked from each rank's own fleet root
   (the t3 run proved `$HOME` expands on the driver host — packs are
   resolved per-rank by fleet root now); ROOT/RES moved out of /tmp after
   wave3's deploy swept /tmp mid-flight (t4 died of that sweep ten seconds
   in).

Supporting work on spark8 (the r3 lane left the publish unfinished —
`lane-r3flash/build` had objects and static libs but no binaries/.so):
finished it with the FLEET's exact identity (`MODEL_REVISION=b4734de4…`,
`CONTRACT_SHA256=ec5afd74…`), `validation=executed` — `glm52_validation
PASS` all tiers incl. split (`split_forward_hidden 0.00610/0.99998`,
`split_determinism bit_exact=1`), module artifact `0584e3f7d492705f…` —
then `sparkpipe_model_compile`. The driver link needed the KV runtime
backend (the glm52 module references `SparkKv*` but its
`MODULE_ADDITIONAL_HOST_SOURCES` never grew the KV tail its module source
grew — the glm5_next module's did; recorded for the owners). Artifacts
(sha256): driver `6fd25fc72958dd9e…`, adapter `1c51f1d8766caa3f…`,
transport `b53f5a9ebf8c5add…`, residentd `5b7e0656b8edfa19…`, batch
`0d214cf699d0879c…`.

## R2c cell: BLOCKED on wiped staging — exact missing prerequisites

The /tmp sweep that accompanied the wave deployments took the r2 cell's
staging with it (spark4:/tmp): the b1+b8 devcycle builds
(`devcycle-build-r2c-b{1,8}` — binaries + per-bucket driver/adapter .so),
the `dsv4bisect-fix` checkout (rank0 pack source), the staged 170 GB rank
packs, the cell script, and the devcycle validator wrapper
(`/tmp/sparkpipe_dsv4_devcycle_validator.sh`). Survivors:
spark5:/home/spark5/lane-dsv4bisect/packs (v3_tp4rank1/2 = exact bytes of
the wiped rank1/2 packs by size; v4_tp4rank0 = exact bytes of the wiped
rank0 pack, 50,996,758,576 B), the 32K batch json, spark7:/tmp/dsv4bisect-
main (but its pack is now 50,995,710,000 B — it CHANGED since the pinning
run — so the rank3 bytes that produced the pin are not reliably
reconstructible), and R2c itself is merged in main (`4c65b23`) so the
binaries are rebuildable.

Re-running before the rank0/rank3 pack bytes are pinned again risks a
FALSE RED (staging drift, not kernel) — which per the discipline would
stop everything for a staging ghost. **Arbitration requested**: either (a)
confirm the pack set to pin (v4_tp4rank0 bytes on rank0+rank3, v3_tp4rank1
/2 on rank1+2, as measured in the pinning deployment) and I re-stage +
re-queue the full cell; or (b) have the r2-prefill lane re-publish its
retained bisect staging. Everything else is staged and self-sufficient.

## Queue-mechanics findings for the coordinator (not my rocks)

- **Double-launch at the transition** (old sweep + new dispatcher both
  fired one task): two cell instances raced one runtime root; ranks 1/2
  died `free(): corrupted unsorted chunks` in `adapter_initialize`. The
  19:11Z cancel removed the LEASE but not the remote processes — an
  orphaned cell with no lease on spark4-7 while a 16-node wave was queued;
  I TERMed it (TERM-only, including the cwd-matched rank daemons) before
  the wave launched. A cancel-side kill of the recorded pid on nodes[0]
  closes the gap; a "old sweep retired" check closes the other.
- **Cancels leave remote work running** (above), and **my own `exit $rc`
  task-cmd suffix defeats the dispatcher's exit-file append** (the appended
  `echo` never runs; the task reaps as "pid gone without exit file" even
  on success). Task cmds should end with a command whose status IS the
  cell's (`test "$rc" -eq 0`).
- **Head-of-line + priority barrier**: the dispatcher runs one task per
  pass; a p0 head with busy nodes blocks everything behind it, and the new
  barrier deliberately holds lower-priority tasks behind a queued p0 wave.
  Correct for wave priority; worth knowing that disjoint-set p0 tasks
  (my r3 on spark8-f vs r2 on spark4-7) cannot overlap by design now.
- **/tmp on the nodes is swept by wave deployments** — task staging must
  live in home dirs (both my cells now do).
- A wedged cell (ssh with no ConnectTimeout through the flaky
  `ds4-spark-fleet-proxy`) holds its lease indefinitely; I TERMed mine.
  Consider a dispatcher-side lease TTL that reaps (and kills) stale pids.

## Staging map (what the next runner needs — all durable home paths)

- spark8:/home/spark8/lane-r3flash/cell/: `r3flash-cell.sh` (md5
  4a38d492…), `r3flash-o128-batch.json` (2f8f2a57…),
  `r3flash-32k-batch.json` (synthesized, deterministic). Queued:
  `r3flash-glm52-exact-cell-t5`. Receipts →
  /home/spark8/lane-r3flash/cell/results/.
- spark0:/home/spark0/vmm_stage.tgz (repo subtree with the W3 fixes).
  Queued: `weightd-vmm-verify-gpu-t4`.
- spark4: r2 cell BLOCKED pending arbitration (above); the cell script
  source is tools/devcycle/r2prefill_exact32k_cell.sh on this branch (md5
  ec8e0ad2…, byte-identical to what was staged).
