# origin/main to unified divergence plan

Status: audit document, every figure re-verified mechanically against the repo.
Branches audited at `unified @ c834436` ("dsv4-flash serving restored + measured
40.67 tok/s B1 on spark4-7") and `origin/main @ 2e17659`.

| Fact | Value |
| --- | --- |
| Merge base | `2deef65` (incident-fixes merge, PR #682) |
| Commits on main not on unified | 75 total = 41 content commits + 34 GitHub merge commits |
| Commits on unified not on main | 159 |
| Trial `git merge origin/main` into unified | completes with exactly **17 conflicted paths** (13 UU + 4 UD, section 3c) |
| Working tree at audit time | clean except pre-existing untracked docs/ + tools/fleet/ |

The lineages diverged after #682. main carries the Qwen 3.8-27B rename
(`modules/qwen38_27b_resident_decode_stage` — the BEST qwen38-27B driver), the
standard HTTP API (`node/model_api.c`), the JIT-KV backing store, fleet
hardening, and merge-safe packaging. unified carries a parallel evolution of the
same driver under old names (`modules/qwen36_*` / `modules/qwen38_*`: DFlash2
selector work, paged KV, extra validation suites — 26 files, +7100/−826 lines)
plus the dsv4-flash milestone.

---

## 1. Categorized commit inventory (unified..origin/main)

All 41 content commits, categorized by area. "Pick" = result of a mechanical
`git cherry-pick` of that single commit onto `unified @ c834436`
(C = applies clean, X = conflicts). Most X results are *prerequisite*
conflicts — the commit edits files created or last touched by earlier
main-only commits (e.g. `node/model_api.c`) that are absent when picking out of
order — rather than true content clashes. True hot files are listed in 3c.

### 1a. qwen38_27b driver / firmware / validation / rename (14)

| Commit | Subject (abridged) | Pick | Note |
| --- | --- | --- | --- |
| 038c2d4 | qwen36: null-guard all debug-dump file writes | X | module.c + test_code_size.py; port into renamed module |
| d655e5e | JIT-KV paged cache: design contract + backing-store tier 1 | X | adds spark_kv_backing.{h,c}; Makefile conflict |
| c7230c8 | full-stack fuzz matrix harness + API poll-loop fix (WIP) | X | needs model_api.c from 91ed563 first |
| 2d44cc1 | storm stress test (10K+ target) + sequential-client findings | X | tools only |
| 7ac8e91 | production limits — B1024, 256k ctx, auto-configuring API | X | serving_adapter + model_api |
| da78f91 | firmware header B1024 + storm bug isolated (seq-ID collision) | X | later reverted by 33a60bd |
| fa0eef8 | JIT-KV backing-store stress (ALL PASS) + storm v2 auto-restart | X | test_code_size.py ratchet only |
| 33a60bd | revert firmware MAX_ACTIVE_SEQUENCE_COUNT back to 512 | X* | *standalone pick is an empty-patch conflict because da78f91 was skipped; applies C immediately after da78f91. Net firmware state after the pair: B1024 KV blocks runtime-configured, MAX_ACTIVE_SEQUENCE_COUNT = 512 |
| a9b54cd | docs: qwen36→qwen38 rename plan + DRY analysis | C | |
| 0b5955e | docs: corrected naming plan (qwen36=Qwen3.8-27B, qwen38=Max) | C | |
| fa7af0c | B16 fix: validator KV_LANES uses compile-time MAX_ACTIVE_SEQUENCES | C | qwen36-named validation file; applied clean |
| 499738b | validator KV_LANES fallback + build-system note for B16 | C | |
| ab57132 | validator: -D flag for KV_LANES in renamed code | C | assumes rename landed |
| 33c69b6 | RENAME qwen36→qwen38_27b, qwen38→qwen38_max (169 files) | X | THE hard one; see sections 3c/4 |

### 1b. API-serving improvements (10)

| Commit | Subject (abridged) | Pick | Note |
| --- | --- | --- | --- |
| 91ed563 | daemon session-hardening + standard API entry point (WIP) | X | creates node/model_api.c; Makefile + model_residentd.c + test_code_size.py clash |
| b97dfd6 | model_api: pre-submit Progress pump (submit-then-wait deadlock) | X | prerequisite: 91ed563 |
| 2846096 | model_api: engine config matched to batch tool | X | prerequisite chain |
| fa8f160 | model_api: HTTP body reading + subprocess backend fixes | X | prerequisite chain |
| f22dbbc | model_api: clean production architecture | X | prerequisite chain |
| f576ad9 | model_api: WORKING — completions end-to-end | X | prerequisite chain |
| 91755bd | model_api: single-request OK, multi-request abort (heap corr.) | X | prerequisite chain |
| e36fc38 | model_api: ASan finding — BUSY on 2nd request (admission) | X | prerequisite chain |
| 1f36d5e | model_api + engine: FULLY WORKING at volume (50/50 sequential) | X | engine + header deltas apply clean vs unified |
| 93cadff | engine: admission lifecycle fix VERIFIED at 200+ requests | X | test_code_size.py ratchet only |

### 1c. Fleet ops (8)

| Commit | Subject (abridged) | Pick | Note |
| --- | --- | --- | --- |
| 1ac239d | Add parallel PXE login rescue | C | tools/devcycle only |
| e91d08a | Document boot-independent emergency SSH | C | |
| d744a8a | Control systemd automounts during fsck | C | |
| 9157a67 | Make Spark fleet recovery canonical | C | INCIDENT_RECOVERY_PLAYBOOK rewrite |
| 6bce33e | fleet: fail closed on degraded maintenance | C | |
| bc616ea | fleet: automate hardware duty checks | C | |
| a52ca66 | fleet: quarantine Ceph startup | X | PACKAGE_MANIFEST.json / SHA256SUMS regen noise |
| 8ac2e05 | fleet: clear recovered root user failure | X | same manifest regen noise |

### 1d. Packaging / build / tests (7)

| Commit | Subject (abridged) | Pick | Note |
| --- | --- | --- | --- |
| 0f9b6c2 | code-size ratchet to 174972 | X | tests/test_code_size.py diverged on both sides |
| 61680f6 | ratchet to 175066 (fuzz + API poll loop) | X | same |
| 6844712 | build: restore GLM52 DSpark host gate | X | Makefile conflict |
| c9f3555 | package: exclude private controller state | C | .gitignore |
| 7c44b43 | package: ignore private tool workspaces | C | .gitignore |
| 1967890 | package: enforce merge-safe manifests | C | PACKAGE_MANIFEST.json shrinks ~93k lines |
| c70913e | package: close pre-manifest-gate merge window | X | manifest regen noise |

### 1e. Docs / architecture (2)

| Commit | Subject (abridged) | Pick |
| --- | --- | --- |
| 6b16b18 | docs: SparkPipe serving charter (B1024/256k audit, continuous batching) | C |
| 6bf9fb4 | docs: charter addendum — the 2.5TB backing-store arithmetic | C |

Tally: 19 C / 22 X across the 41 content commits. The remaining 34 GitHub merge
commits are subsumed by any merge or ordered cherry-pick of the above.

---

## 2. qwen38_27b module completeness verdict (origin/main)

Inventory at `origin/main:modules/qwen38_27b_resident_decode_stage/`:

| Component | Present | Artifact |
| --- | --- | --- |
| Firmware header | yes | include/sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h (24.7 KB). Net state after da78f91+33a60bd: MAX_ACTIVE_SEQUENCE_COUNT = 512u; KV capacity is runtime-configured (STAGE_KV_BLOCKS env), so B1024 is a serving-config ambition, not yet a firmware constant |
| Module core (.c) | yes | source/spark_qwen38_27b_resident_decode_stage_module.c (200 KB); strict env config ("a missing value is a refused Initialize, never a default") |
| CUDA kernels | yes | source/spark_qwen38_27b_resident_decode_stage_cuda.cu (117 KB) + native_ws / dspark .cuh |
| Serving adapter | yes | source/spark_qwen38_27b_serving_adapter.c (117 KB) + include/sparkpipe/spark_qwen38_27b_serving_adapter.h; schema-v3 JSON config |
| TP support | yes | source/spark_qwen38_27b_tp.{c,h} |
| Packer | partial | module-level tools/qwen38_27b_pack_synthesize.c (16 KB, byte accounting closes on the 27B param count); repo-level packers remain qwen36-named (tools/qwen36_stagepack.py, qwen36_dspark_stagepack.py, qwen36_stagepack_mx_repack.py, qwen36_stagepack_rans.c, plus tools/qwen38_stagepack.py for Max) |
| Validation harness | yes | validation/spark_qwen38_27b_reference.c CPU oracle (23 KB) + _cuda_validation.cu + validate_qwen38_27b_resident_decode_stage_cuda.sh with SHA-256-pinned recipe in the module Makefile |
| Adapter tests | BROKEN WIRING | tests/test_qwen36_serving_adapter.c drives the qwen38_27b adapter (content renamed, filename not), but main's root Makefile lists target `test_qwen38_27b_serving_adapter` (Makefile:206) and builds it from `tests/test_qwen38_27b_serving_adapter.c` + `tests/fixtures/qwen38_27b_serving_adapter_config*.json` (Makefile:767–768) — NONE of which exist in the tree (0 fixtures match qwen38_27b). That make target fails on main as-is |
| Deployment examples | partial (name debt) | examples/deployments/qwen36_pp13_host_rdma.json targets node_target `cuda.sm121.qwen38_27b.resident_decode_stage.bf16` on all 13 stages with real runtime_root paths, but filenames keep qwen36 names; model_contracts carries both qwen36_authoritative.json and a byte-identical qwen38_27b_authoritative.json |

Verdict: **production-shaped for bring-up, not rename-complete.** The
kernel/module/adapter/validation stack is coherent and hardware-proven (see
receipts below). The rename commit 33c69b6 covered model-families/ and modules/
but left tests/, tools/, examples/, model_contracts/ under old names with
updated content — cosmetic debt, plus one genuinely broken make target.

Packs / hardware receipts found:

- README records a green `make validate` on GB10 against BOTH the synthetic
  slice pack AND the real PP13 stage-0 pack, dated 2026-08-08, with measured
  numbers (rel_l2 ~1.7e-3, cosine ~0.9999986, bit-exact decode-vs-prefill
  agreement and fresh-instance determinism).
- model_contracts/qwen38_27b_authoritative.json pins HF revisions
  (Qwen/Qwen3.8-27B @ 1d4bf0f2..., FP8 variant @ 017b9c7a...) marked
  "AUDITED 2026-08-17".
- docs/QWEN38-27B_HILLCLIMB.md is an audit-grade receipt document (file:line
  citations of MTP speculation defects A1–A4; MTP execution gated behind
  SPARK_QWEN38_27B_SERVING_SPECULATE, off by default).
- The serving_adapter "receipt" hits are telemetry about first-draft misses,
  not pack provenance; pack identity itself is enforced by fingerprinting
  stage-pack geometry into KV keys (stale-pack KV reuse impossible).

---

## 3. API-level improvement inventory and conflict analysis

### 3a. Concrete improvements main carries that unified lacks

1. Standard serving entry point — node/model_api.c (NEW, 524 lines): HTTP
   completions endpoint over the batch engine; verified end-to-end (f576ad9),
   hardened to 50/50 sequential requests at volume (1f36d5e), admission
   lifecycle verified at 200+ requests (93cadff).
2. Engine admission lifecycle fix — runtime/model_batch_engine.c (+25/−7) +
   include/sparkpipe/spark_model_batch_engine.h (+8):
   - new `SparkModelBatchEngineReopenAdmission()` so a serving process can
     re-open admission per submit and clear sticky failure;
   - a failed prefix-cache release no longer closes admission/SetFailed
     (which bricked the engine for all future requests);
   - per-request failures route through HandleRejected only (one debug
     fprintf added there).
3. Connection-leak / poisoned-slot fix — node/model_residentd.c (+24): on
   client session death the daemon unbinds ALL lane slots; previously a dead
   client left REQUIRES_RELEASE bindings rejecting every later client until
   daemon restart. Plus a submission-rejected log line for ops visibility.
4. JIT-KV backing store tier 1 — include/sparkpipe/spark_kv_backing.h (NEW,
   66 lines) + runtime/spark_kv_backing.c (NEW, 191 lines: slot file,
   pread/pwrite, fixed 4 MiB stride, live bitmap, single-writer discipline)
   + tools/spark_kv_backing_test.c (NEW, 83 lines, ALL PASS) +
   docs/JIT_KV_DESIGN.md contract (park/restore whole lanes, never per-token
   paging; 2.5 TB LRU horizon ≈ 38M tokens).
5. Stress/fuzz tooling: tools/qwen38_fuzz_matrix.sh, tools/qwen38_storm.sh
   (10K+ request storm, v2 auto-restart), code-size ratchets 174972/175066.
6. Ops hygiene: merge-safe PACKAGE_MANIFEST.json (−93k lines), fleet recovery
   canonicalization, PXE login rescue, Ceph quarantine, duty checks.

### 3b. What unified has that main does not (preserve these)

1. Deferred continuation-lease position fence (SEMANTIC, must survive):
   runtime/model_continuation_lease.{h,c} (+22/−1 incl. new
   `SparkModelContinuationLeaseEstablishDeferred()` and
   SPARK_MODEL_CONTINUATION_LEASE_DEFERRED_POSITION) + node/model_residentd.c:
   a transported non-final stage publishes status-only DECODE completions
   (tokens_per_sequence == 0); feeding the zero to the lease decoder rejected
   with INVALID_ARGUMENT and killed the run loop. unified defers the position
   fence instead (generation/step fences stay armed).
2. Residency-mismatch hex dump in node/model_residentd.c (prints expected vs
   actual 32-byte residency tokens on completion validation failure).
3. ~+7100 lines of parallel driver evolution under OLD names (26 files,
   +7100/−826 across modules/qwen36_* / qwen38_*: DFlash2 selector host/device
   paths + 4 validation .cu suites, spark_qwen36_paged_kv.{c,h},
   ISLAND_MAPPING.md, firmware/adapter/module changes).
4. The dsv4-flash milestone (40.67 tok/s B1 measured, hash 3/3).
5. NOTE: runtime/model_pipeline_client.c was NOT changed by main since base;
   unified's own 16-line delta there is PC_TRACE fprintf debugging (8 sites,
   commit 7cfc7a6) — leftover debugging to strip before closing the merge.

### 3c. Conflict analysis vs unified's copies (trial merge: 13 UU + 4 UD)

Targeted diffs vs base `2deef65` for the five audited overlap files:

| File | main delta since base | unified delta since base | Trial-merge outcome |
| --- | --- | --- | --- |
| node/model_batch.c | none (file exists, unchanged) | none | no conflict |
| runtime/model_pipeline_client.c | none | 16 lines changed (8 PC_TRACE fprintf sites) | auto-merged clean; strip before landing |
| runtime/model_continuation_lease.c | none | +22/−1 deferred-position fence | auto-merged clean; take-unified |
| node/model_residentd.c | +24 (session-death slot unbind @~1330, rejection log @~1812) | +47/−3 (deferred lease @~777, residency dump @~1060, TRACE fprintf @~2543) | auto-merged clean (disjoint regions); keep BOTH sides |
| runtime/model_batch_engine.c | +25/−7 admission lifecycle | none | clean take-main |
| include/sparkpipe/spark_model_batch_engine.h | +8 ReopenAdmission decl | none | clean take-main |

Remaining conflicted paths from the real (rename-aware) trial merge:

UU content conflicts (13):
- Makefile (both sides add module wiring — union them);
- modules/qwen38_27b_resident_decode_stage/: firmware.h, _dspark_cuda.cuh,
  _cuda.cu, _module.c, _serving_adapter.c (rename detection applies unified's
  edits of the OLD qwen36 module onto main's RENAMED targets);
- modules/qwen38_max_resident_decode_stage/: _module.c, _serving_adapter.c (same effect);
- tests/fixtures/qwen36_serving_adapter_driver.c, tests/test_code_size.py,
  tools/cuda13_sm121a_compile_gate.sh, tools/qwen36_dflash2_conv_parity.cu,
  tools/qwen36_dspark_reference.py (small divergences both sides).

UD modify/delete (4) — old-named files unified edited that main deleted in the
rename; accept deletion, port unified's delta into the renamed home:
- modules/qwen36_resident_decode_stage/source/spark_qwen36_dspark_format.h
- modules/qwen36_resident_decode_stage/source/spark_qwen36_stagepack_format.h
- modules/qwen36_resident_decode_stage/validation/spark_qwen36_resident_decode_stage_cuda_validation.cu
- modules/qwen38_resident_decode_stage/source/spark_qwen38_stagepack_format.h

PACKAGE_MANIFEST.json / SHA256SUMS churn (from a52ca66, 8ac2e05, c70913e)
auto-merges textually but should be regenerated, not hand-merged.

---

## 4. Merge strategy

Two viable routes. Route A (recommended) does ONE merge with a prepared
conflict playbook; Route B (fallback / review-heavy) cherry-picks in dependency
order. Do NOT cherry-pick the rename piecemeal.

### 4a. Route A — single merge of origin/main into unified (recommended)

1. Precondition: decide fate of untracked tools/fleet/; run unified's test
   suite once for a green baseline.
2. `git checkout -b merge-main-into-unified && git merge origin/main`.
   Expect exactly the 17 conflicted paths from section 3c.
3. Resolve in this order:
   a. Trivial take-main (main-only changes): runtime/model_batch_engine.c,
      include/sparkpipe/spark_model_batch_engine.h,
      include/sparkpipe/spark_kv_backing.h, runtime/spark_kv_backing.c,
      node/model_api.c, tools/spark_kv_backing_test.c.
   b. Keep BOTH sides' residentd changes (disjoint regions — already
      auto-merged in the trial; verify the deferred-lease hunk survived).
   c. Makefile: union both module wirings — keep unified's dsv4 targets AND
      main's qwen38_27b/qwen38_max/KV-backing wiring and GLM52 host gate;
      point `test_qwen38_27b_serving_adapter` at the real
      tests/test_qwen36_serving_adapter.c (or create the missing test file)
      instead of leaving a broken target.
   d. Rename-detection UU set (module files): start from main's renamed
      content, then port unified's post-base deltas per file (DFlash2 selector
      paths, paged-KV hooks, dump guards 038c2d4). Where semantics clash,
      main's hardware-measured state wins; re-validate on hardware.
   e. UD old-named files: accept deletion; ensure their unified-only additions
      now live at the renamed paths.
   f. tests/test_code_size.py: union entry lists, adopt main's ratchet value,
      re-measure after the build, bump once.
   g. Regenerate PACKAGE_MANIFEST.json / SHA256SUMS with the packaging tool
      after all other resolutions; do not hand-merge.
4. Strip unified's PC_TRACE instrumentation from
   runtime/model_pipeline_client.c (8 sites, 7cfc7a6).
5. Verification gates (all must pass): full build incl. cuda13_sm121a gate;
   adapter test (drives qwen38_27b adapter); code-size ratchet;
   spark_kv_backing_test; qwen38_fuzz_matrix.sh smoke; dsv4-flash regression
   (40.67 tok/s B1, hash 3/3); qwen38_27b `make validate` on GB10 if
   hardware available.

### 4b. Route B — ordered cherry-picks (if per-commit review is mandated)

Order respects prerequisites; X-marked ones need the resolutions from 4a:

1. Docs/architecture first (clean): 6b16b18, 6bf9fb4, a9b54cd, 0b5955e.
2. Fleet ops (clean): 1ac239d, e91d08a, d744a8a, 9157a67, 6bce33e, bc616ea;
   then packaging c9f3555, 7c44b43, 1967890.
3. Driver fixes that apply before the API chain exists: 038c2d4 (dump guards).
4. KV/stress tooling: d655e5e, fa0eef8, then ratchets 0f9b6c2, 61680f6
   (resolve test_code_size.py unions), fuzz/storm 2d44cc1, c7230c8.
5. API chain IN ORDER (X until its predecessor lands):
   91ed563 → b97dfd6 → 2846096 → fa8f160 → f22dbbc → f576ad9 → 91755bd →
   e36fc38 → 1f36d5e → 93cadff. Resolve Makefile / model_residentd.c /
   test_code_size.py clashes at 91ed563.
6. Driver limits pair: da78f91 then immediately 33a60bd (net = B1024 blocks
   via config, MAX_ACTIVE_SEQUENCE_COUNT 512), then 7ac8e91 prod limits.
7. LAST: 33c69b6 RENAME + manual reconciliation (the 3c UU/UD work), then
   post-rename fixups fa7af0c, 499738b, ab57132, 6844712, c70913e, a52ca66,
   8ac2e05 (regenerate manifests rather than merging them).

### 4c. What to drop

- unified's PC_TRACE fprintf instrumentation (runtime/model_pipeline_client.c,
  8 sites; plus the TRACE-labelled fprintfs inside unified's model_residentd.c
  hunks — keep the semantic deferred-lease/residency-dump changes, drop the
  bare "TRACE" lines if desired).
- Nothing from main: even the reverted pair da78f91+33a60bd should land as a
  pair (net-zero firmware churn but preserves the storm root-cause analysis).

### 4d. Known risks

- The qwen38_27b module on main and unified's qwen36/qwen38 modules are two
  diverged implementations of the same driver; resolving the 7 module UU files
  is a semantic reconciliation, not a text merge. Budget engineering time.
- main's hillclimb doc documents open defects in the MTP speculation path
  (A1–A4); do not treat the module as fully production-proven there.
- Stale naming residue (tests/tools/examples/model_contracts filenames, README
  title still "Qwen 3.6 27B", broken `test_qwen38_27b_serving_adapter` target)
  should be finished as an immediate follow-up so both lineages converge on
  one vocabulary.
- After landing, force-align: fast-forward origin/main to the merged result
  (or retire it) to stop re-divergence.
