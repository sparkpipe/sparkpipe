# cell-runner lane report — 2026-08-29

Lane `lane/cell-runner` (worktree /tmp/lane-cellrunner). Mission: own the
QUEUED GPU receipts — run them, collect, diagnose-on-mismatch, report.
Base: origin/main `e77fca9` → merged `f3a2772` (task-based dispatcher) at
the session start; lane tip `609bb59`. Pushed; nothing self-merged.

## Receipts table (numbers only from kill-switch-clean runs)

| Cell | Nodes | Context / batch / topology / precision | Kill-switch | Receipt |
|---|---|---|---|---|
| R3 flash-decode O128 threshold-0 control | spark8..sparkf (8) | O128 decode batch (`o128_batch.json`, md5 2f8f2a57…), glm52.tp8.fp8 TP8, fp8 packs b4734de4 | — (control) | PENDING |
| R3 flash-decode O128 threshold-2048 split | spark8..sparkf (8) | same batch, one key: `decode_split_context_threshold=2048` in config/glm52_stage.json | bit-exact vs control required | PENDING |
| R3 32K decode split vs control | spark8..sparkf (8) | synthesized 32,768-token prompt (deterministic tile of the O128 prompt ids), 256-token budget, `max_context_tokens` 33024 | only after kill-switch PASS | PENDING |
| R2c b1 O128 gate (rows==1) | spark4..spark7 (4) | O128 batch, dsv4 flash TP4, lean packs (bisect-lane retained) | pinned hash `a9385d0b…` | PENDING |
| R2c b8 O128 gate (width-8 bulk) | spark4..spark7 (4) | same O128 batch, `max_input_rows` 8 / `max_prefill_rows_per_submission` 8 | SAME pinned hash | PENDING |
| R2c exact-32K prefill (bulk path) | spark4..spark7 (4) | 32768-token prompt, 256-token budget, width-8 bulk | only after both gates PASS | PENDING |
| weightd VMM verify (W2b+W3 legs) | spark0 (1) | 8 MiB arena, 4×2 MiB chunks, real driver + real second process | exit 0 + `VMM VERIFY PASS` | RED→FIXED, re-run PENDING (see below) |

All cells recorded with context/batch/topology/precision as above. No perf
claims beyond the raw receipts; timing legs run only behind their gates.

## What ran, in order (the operator's task queue did the arbitrating)

1. **19:04Z — the r2 cell fired during the scheduler transition and was
   double-launched.** The old 30-min sweep AND the new dispatcher both
   started `r2prefill-exact32k-cell` (old wrapper pid 3709433, new wrapper
   pid 3712011 on spark4). Two cell instances raced on one runtime root;
   ranks 1/2 died in `adapter_initialize` with `free(): corrupted unsorted
   chunks` after RDMA route failures — transition debris, not a code red.
   The coordinator cancelled the queue entry at 19:11Z; the cancel removed
   the LEASE but not the remote processes, leaving an orphaned cell with no
   lease on spark4-7 while `g5dsa-wave1` (all 16 nodes) was queued — the
   next dispatch tick would have landed a 16-rank wave on top of it.
2. **19:5xZ — orphan terminated, TERM-only** (cell wrappers + the one
   cwd-matched residentd that had come up on spark4; ranks 5/6/7 had none).
   Leases and nodes were clean before `g5dsa-wave1` launched.
3. **Re-queued as `r2prefill-exact32k-cell-t2`** (id reuse is refused by
   design after a cancel). Script unchanged and idempotent
   (spark4:/tmp/r2prefill-cell.sh, md5 `ec8e0ad2…`, byte-identical to
   `tools/devcycle/r2prefill_exact32k_cell.sh`); packs verified present on
   all four ranks (rank0/rank3 50,996,758,576 B; rank1 40,130,454,508 B;
   rank2 40,129,405,932 B — the retained bisect-lane per-rank packs).

## The r3 cell could not have run as committed — repaired on this lane

`tools/devcycle/r3flash_exact_cell.sh` (as queued by lane-r3flashdecode)
had four defects that made the receipt unrunnable. Fixed in `609bb59`:

1. **Wrong fleet band.** HOSTS was `spark0..spark7`; the glm52.tp8.fp8
   fleet roots exist on `spark8..sparkf` (ranks 0..7 per
   `tools/glm52_gen_deployment.py`). spark0-7 carry no
   `sparkdata/glm52.tp8.fp8` at all.
2. **Kill-switch key on the wrong surface.** The script wrote
   `decode_split_context_threshold` to the top of `model_resident.json`;
   the serving adapter parses the key from the ADAPTER stage config
   (`config/glm52_stage.json`, exact-member list — a config without it is
   rejected SCHEMA_ERROR at load). The deployed stage configs predate the
   r3 lane and lack the key, so the cell now renders both configs per rank.
3. **The 32K leg was inadmissible.** Deployed stage configs cap
   `max_sequence_positions` at 4096; the cell renders 32768 (module ceiling
   `SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS` is 1,048,576). The O128 gates
   never reach the cap; a 32K sequence that the KV pool cannot hold is
   refused loudly at admission — recorded, never faked.
4. **Fleet-ready checked the wrong port.** 18480 is the DSV4 block; the
   glm52 band's control ports are 19480+rank. Also: runtime roots point at
   the cell root so the staged r3 driver .so is the one that loads, and the
   per-rank packs are symlinked in (no 40 GB copies).

Supporting work on spark8 (the r3 lane left the publish unfinished —
`lane-r3flash/build` had objects and static libs but no binaries/.so):
finished the publish with the FLEET's exact identity
(`MODEL_REVISION=b4734de4…`, `CONTRACT_SHA256=ec5afd74…`, the values the
deployed packs/carrier configs carry), `validation=executed` —
`glm52_validation PASS` all tiers including `split_forward_hidden
0.00610/0.99998` and `split_determinism bit_exact=1` — module artifact
`0584e3f7d492705f…`, then `sparkpipe_model_compile`. The driver link
needed the KV runtime backend (`SparkKv*` referenced by the glm52 module
archive but not in it); linked the same four cache sources the glm5_next
module archive carries (`kv_cache/kv_page_cache/kv_page_store/
kv_model_table.c`) as a driver-link archive — the glm52 module's
`MODULE_ADDITIONAL_HOST_SOURCES` never grew the KV tail its module source
grew (recorded for the owners). Published artifacts (sha256):
driver `6fd25fc72958dd9e…`, adapter `1c51f1d8766caa3f…`, transport
`b53f5a9ebf8c5add…`, residentd `5b7e0656b8edfa19…`, batch `0d214cf699d0879c…`.

## W3 GPU receipt: RED on first hardware contact — exact divergence found, one-expression fix

The staged `tools/sparkpipe_weightd_vmm_verify.sh` had never touched a GPU.
First real run (spark0, GB10, CUDA 13):

```
granularity=2097152                       (2 MiB law — PASS)
cold arena bytes verified over cuMemMap   (leg 2 — PASS)
warm attach shared handle=… refcount=2    (leg 3 — PASS)
helper import map failed reason=import_handle   (leg 4 — FAIL)
```

Exact divergence: `cuMemImportFromShareableHandle`'s `osHandle` carries the
POSIX fd **by value** (cuda.h: "Shareable Handle representing the memory
allocation"); `runtime/spark_weightd_attach.c` passed `&batch.fds[index]` —
the driver read a stack address as an fd number and every import failed
`CUDA_ERROR_INVALID_HANDLE`, deterministically, on the first chunk of the
first batch. The host stub models the by-value contract, which is why the
W3 suites and TSan runs were green and only hardware exposed it. The
daemon-side legs (export, cold map readback, warm refcount) are receipts
from the UNFIXED binary and are unaffected.

Fix (`609bb59`): pass `(void *)(uintptr_t)batch.fds[index]`. The re-run is
queued (`weightd-vmm-verify-gpu`, spark0) with the fixed tree staged at
spark0:/tmp/cellrunner-vmm. W3's merged code was RED on hardware until this
fix; the fixed receipt gates the dsv4 attached-binding path, not this lane.

## Ratchet

229907 → **229914 exact** (+7: the fd-by-value fix and its justification
comment; the r3 cell repair's +82 is ledgered in the same entry),
`tests/test_code_size.py`.

## Queue-mechanics findings for the coordinator (not my rocks)

- The dispatcher runs ONE task per tick and blocks the whole queue behind a
  head task whose nodes are busy (`dispatch` picks the single best runnable
  by (-priority, submitted_at)). While dry2-dry2phase-v2 held spark5, every
  priority-0 task wanting other free nodes stayed blocked behind it. A
  "skip busy head, try next runnable" pass would let disjoint-set tasks
  overlap (my r3 on spark8-f and r2 on spark4-7 are disjoint and would
  have run concurrently).
- Cancelling a running task does not TERM its remote process (the lease
  dies, the work doesn't). The 19:11Z cancel orphaned a live cell on
  spark4-7. A cancel-side `kill` of the recorded pid on nodes[0] (plus a
  note that cell scripts must be cwd/anchor-killable) closes the gap.
- The transition window double-dispatched one task (old sweep + new
  dispatcher). One transition cost a cancel + a clean re-queue; worth a
  one-line "old sweep is retired" check before the next infra flip.

## Staging map (what the next runner needs)

- spark4:/tmp/r2prefill-cell.sh (md5 ec8e0ad2…) — queued as
  `r2prefill-exact32k-cell-t2`, receipts → spark4:/tmp/r2prefill-results.
- spark8:/tmp/r3flash-cell.sh (fixed script) + /tmp/r3flash-o128-batch.json
  (2f8f2a57…) + /tmp/r3flash-32k-batch.json (synthesized, deterministic) —
  queued as `r3flash-glm52-exact-cell`, receipts →
  spark8:/tmp/r3flash-results; lane artifacts in
  /home/spark8/lane-r3flash/build.
- spark0:/tmp/cellrunner-vmm (fixed W3 tree) — queued as
  `weightd-vmm-verify-gpu`.
