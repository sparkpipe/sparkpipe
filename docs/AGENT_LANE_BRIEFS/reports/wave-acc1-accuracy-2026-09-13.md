# ACC-1 accuracy wave — reference-anchored pack-verification receipts (2026-09-13/14)

Agent: ACC-1 (fleet ACCURACY wave, mgr2 dispatch). Branch
`lane/wave-acc1-accuracy` off origin/main `d1c3822` (fresh clone
`/Users/mac/acc1`). Lanes: qwen38max, qwen3flash, qwen38-27b, glm53full, hy4.

Mission: enforce the EXISTING pack-verification law — **pack == packer +
packer == checkpoint** — with reference-anchored offline evidence per lane,
grade every headline accuracy claim (MEASURED / DERIVED / ASSUMED), and name
exactly what is missing for a T1 decode-token receipt per lane.

SP-4R boundary honored: placed bytes for `glm53flash.fp8.tp8`,
`qwen3flash.fp8.tp8` and `qwen3flash.fp8.tp4pp4` were NOT verified (rebuilding
under us; their stage3 source-side verify was observed live on spark0/spark5,
pids from `~/sp4rtools`, `qwen4_flash_pack_verify --tp-degree 4 --tp-rank 3/1
--checkpoint /mnt/model-warm/qwen3.8-flash-next-fp8`). spark3 and spark6 were
never contacted. No daemon contact, no model execution, no GPU work.

## Deliverable: the parity oracle

Two committed tools; the family verifiers stay the byte-level authorities and
are re-used, never duplicated:

- `tools/wave_acc1_parity_oracle.py` — three-leg driver per arm:
  1. **pack == packer**: streamed sha256 of the placed file vs the packer
     receipt `output_sha256` / `<pack>.sha256` sidecar / manifest
     `file_sha256`.
  2. **packer == checkpoint**: receipt `source_index_sha256` (+
     `source_config_sha256`) vs the LIVE warm checkpoint's digests. Fails
     closed when no checkpoint is given.
  3. **pack == checkpoint**: runs the lane's own verifier as a child process
     (`--family-verify` + `--family-args` with `{pack}`/`{checkpoint}`/
     `{receipt}` substitution).
  Exit 0 only on full pass; legs with no usable input report SKIP and fail
  the run (UNVERIFIABLE), never silently pass.
- `tools/wave_acc1_glm53full_rank_verify.py` — glm53full rank-pack oracle that
  RE-RUNS the lane packer's own plan producers
  (`glm52_resident_stagepack.Packer.build_plan` + per-item
  `produce_payload()`/`produce_scale()`) against the live warm checkpoint and
  compares those bytes against the placed rank pack (directory re-parsed with
  the packer's own `<8I4Q` entry struct; scale offsets included). Anchor set:
  embedding/final-norm/lm_head globals; at layers 0/middle/last: attn_norm,
  post_attn_norm, q_a, q_b, o_proj, and at routed layers router,
  f32 correction bias, shared experts; the MIDDLE layer's routed-expert
  payload+scale planes for experts {0, 128, 255}. Works for codec bf16, fp8
  (verbatim payload + expanded-f32 scale planes) and nvfp4 (verbatim e2m1 +
  per-16 e4m3 planes + per-expert f32 globals).

Rope coverage note: none of the five families stores precomputed rope tables
in packs. Rope enters packs as header constants (attention period, full
attention phase, rope dimension) and every family verifier used below checks
those header words against the pinned geometry — the strongest statement the
pack format itself supports. GLM families (glm53full, hy4) carry no rope
fields at all (MLA + indexer, rope implied in projections).

## Oracle results (all commands node-local, heavy reads under
`sudo -n systemd-run --scope -q -p MemoryMax=...M -p MemoryHigh=...M
--uid=1000`, disk-cache purge after each node batch)

### glm53full — VERDICT: CHECKPOINT-FAITHFUL (bf16, fp8, nvfp4 arms, rank-level)

| check | result |
|---|---|
| bf16.tp16 rank5 (spark5), 3-leg oracle | PASS — pack digest == `.sha256` sidecar; receipt `source.index_sha256` == live `/mnt/model-warm/glm-5.3-bf16`; 28 anchor entries re-derived from the packer plan byte-exact (`28 anchor entries checked, 0 failures / RESULT: PASS`) |
| bf16.tp16 rank4 (spark4), rank8 (spark8) digests | PASS — recomputed sha256 == per-node `.sha256` sidecars (`addc2b4d…`, `a4032bb2…`) |
| fp8.tp16 rank0 (spark0 pack verified on spark9), plan oracle | PASS — `arm=fp8 rank=0: 28 anchor entries checked, 0 failures` vs live `/mnt/model-warm/glm-5.3-fp8` (fp8 expert payload planes verbatim, expanded-f32 scale planes, bf16-dequant spine re-derived through the packer's own producers) |
| nvfp4.tp16 rank0 (spark0), plan oracle | PASS — `arm=nvfp4 rank=0: 28 anchor entries checked, 0 failures` vs live `/mnt/model-warm/glm-5.3-nvfp4-radixark`; digest leg: spark0 rank0 pack sha == build-tree `SHA256SUMS` entry `36fef980…` |

Prior receipts confirmed and still standing: 2026-08-30 full 16-rank bf16
source verify (experts L{3,40,77} × 16 ranks + full 256-expert sweep + spine +
replicated + tp16 partition, all byte-exact); GPU validator cosines bf16
dense 0.99998-0.99999 / routed 0.99998819 / dsa 0.99999; fp8 publish dense/
routed/dsa PASS (2026-09-10, on post-d2a main); nvfp4 routed cosine 0.99999.

### qwen38max — VERDICT: CHECKPOINT-FAITHFUL (nvfp4.tp16 arm, full content); tp4pp4 arm HALF-VERIFIED

| check | result |
|---|---|
| nvfp4.tp16 rank0 (spark0), `qwen38max_tp16_rank_verify.py` FULL content | **PASS — 1682/1682 tensors content-compared vs live `/mnt/model-warm/qwen3.8-max-nvfp4-radixark-bf16-spine`, `content_errors: 0`, `verdict: PASS`; recomputed whole-file sha256 `3896939f…` == receipt `output_sha256`; receipt `source_index_sha256` == live warm index** |
| tp4pp4 rank00 (spark0), digest leg | PASS — 93,243,509,760 B pack sha256 == receipt `output_sha256` `ae4db276…` |
| tp4pp4 rank00, identity leg | PASS by transitivity today: tp4pp4 receipt `source_index_sha256` `2b8d7065…` == nvfp4.tp16 receipt's, whose live-warm equality was measured above the same day on the same warm dir |
| tp4pp4 rank00, structure/content leg | **UNVERIFIABLE with in-tree tools** — `qwen38_pack_verify.py` hard-decodes a 26×u32+2×u64 header with expert fields at words 20-25; the placed tp4pp4 packs carry the family v3 order (`FFN_INTERMEDIATE, VOCAB, MXFP4_GROUP, MTP_LAYERS, tp_degree, tp_rank` mid-header, no expert words). On main today this verifier CRASHES (`AttributeError: ... has no attribute 'EXPERT_COUNT'`) rather than failing cleanly, and against a v3 pack it reports garbage header fields. Gap: a layout-aware max/27b verifier (see findings) |

### qwen3flash — VERDICT: CHECKPOINT-FAITHFUL (bf16.tp8, nvfp4.tp8 arms); fp8 arms at the SP-4R boundary
| check | result |
|---|---|
| bf16.tp8 rank5 (pack mesh-copied spark5 → spark9, byte-identical copy), `qwen4_flash_pack_verify.py` | **PASS — `header geometry, 1246 directory entries (tp 8/5), 6 byte-traced samples`** vs live `/mnt/model-warm/qwen3.8-flash-next` (first placed-bytes content verification of this arm; the 08-30 attempt was blocked by the ceph wedge, honestly recorded in `qwen-flash-wave-ready-2026-08-30.md`) |
| nvfp4.tp8 rank0 (spark0), digest+receipt legs | PASS — 23,908,971,008 B pack sha256 == packer receipt `output_sha256` `68a5fdc4…` == `.sha256` sidecar |
| nvfp4.tp8 rank0, content leg | **PASS — `header geometry, 1215 directory entries (tp 8/0), 6 byte-traced samples`** vs live `/mnt/model-warm/qwen3.8-flash-next-nvfp4-nvidia` (pack mesh-copied to spark9 after a hard per-object ceph stall on spark0; the default MTP-carrying expectation had correctly FAILED `header mtp_layer_count=0 expected 1` first — a fail-closed catch, not drift; the arm is MTP-free by design and `--no-mtp` is the documented gate) |
| fp8.tp8 / fp8.tp4pp4 placed bytes | NOT verified — SP-4R rebuilding under us (coordination law). SP-4R's own source-side verifies were observed running against `/mnt/model-warm/qwen3.8-flash-next-fp8` |

Prior receipts confirmed: publication smoke 17/17 vs placed bf16.tp8 rank0
(gdn_chunk_output rel_l2 0.00166 / cosine 0.99999863, hc_residual 0.00242,
indexer_pooled 0.00497, ple_hash bit_exact, decode-vs-prefill bit_exact,
determinism bit_exact — 2026-09-08 and post-astra re-run); coexistence smoke
bf16+nvfp4 both rc=0 (2026-09-10).

### qwen38-27b — VERDICT: CHECKPOINT-FAITHFUL (tp4pp4 rank00, strongest possible form)

| check | result |
|---|---|
| tp4pp4 rank00 (spark0), packer's own `verify()` | PASS — `structure ok: {'bytes': 4359934976, 'tensor_count': 213}` |
| tp4pp4 rank00, packer RE-RUN (convert, world_rank 0, tp4/rank0) vs placed | **PASS — `placed_sha 4c4e6efc…` == `rebuilt_sha 4c4e6efc…`; PACK_EQUALS_PACKER_RERUN: PASS.** The placed pack is bit-identical to a fresh deterministic packer run reading the live `/mnt/model-warm/qwen3.8-27b-fp8` — this is pack==packer AND pack==checkpoint in one digest |

Prior receipts confirmed: GPU validation vs the real 29 GB pack (decode-vs-
prefill bit_exact=1, mtp_draft in_vocab=1, determinism bit_exact=1,
`qwen38_27b_validation PASS`, artifact `c031daec…`, 2026-08-30); ledger
TP1 stream hashes (spec `d7f79880…` 24.5 tok/s, no-spec `5d6ee525…`).

### hy4 — VERDICT: CHECKPOINT-FAITHFUL (fp8.tp16 rank00)

| check | result |
|---|---|
| fp8.tp16 rank00 (pack + manifest mesh-copied spark0 → spark9), `hy4_fp8_pack_verify.py` | **PASS — `sha OK (a71ea082…)` == manifest == `.sha256`; 6 samples per slice class byte-exact vs live `/mnt/model-warm/hy4-preview-fp8-official`: range (experts down_proj 201,326,592 B blocks, q_b_proj), gather (o_proj incl. replicated scale planes), full (kv_a_proj_with_mqa, kv_a_layernorm, indexer.k_norm.bias, hc norms/scale), mtp (shared_experts up_proj), scale companion — `VERIFY PASS`** |

Prior receipts confirmed: FP8 numerical rung 2026-09-10 — production grouped-
dot kernel vs independent double-precision CPU oracle over the PLACED rank-02
pack, 573 FP8 pairs, 4584 dots, maxabs 1.110e-06 / maxrel 1.698e-04, p50
1.887e-07 (f32-epsilon class), no bias shape, no layer jump: NOISE-CLASS.
Scale contract: 72 C/python agreement cases, vendor reference
`hyv4_reference.cpp` pinned at sha `514ef62a…` (llama.cpp hyv4.cpp @
`0cea36222`), 7 fail-closed codes. 837/837 `.experts` sidecar chunks CK128
GREEN. Original build byte-exact vs source checkpoint (0 differing 1 MB
blocks on all four layer-1 planes).

## ACCURACY INVENTORY — every headline claim, graded

Grading: MEASURED = retained artifact with node/date/digest; DERIVED = computed
from named inputs + assumptions; ASSUMED = no artifact or an unquantified
ruling. "Self-consistency" (decode-vs-prefill, determinism, replay hashes) is
worth having but is NOT reference-anchored accuracy.

### qwen38max

| claim | grade | evidence |
|---|---|---|
| B1 1.29 tok/s (TP4xPP4 anchors) | MEASURED | PERFORMANCE_LEDGER row, "measured anchors" |
| nvfp4.tp16 packs faithful to checkpoint | MEASURED | this wave, 1682/1682 + digest + identity |
| tp4pp4 packs == packer output | MEASURED (digest) | this wave; content leg UNVERIFIABLE in-tree (layout gap) |
| nvfp4 = "the ONLY fitting form" / pack sizes | DERIVED | geometry arithmetic; roofline R1/R3 tools (reference, not duplicated) |
| stagepack v2 wire audit (F1/F2 defects) | MEASURED | qwen38max-v2-cpu-audit-2026-08-30 (CPU proof on real bytes, fixed) |
| numerical validation harness | ABSENT | ledger Honest Gap #5: "qwen38_max has no validation harness" — still true on main; nothing runs decode tokens vs a reference for this family |

Accuracy evidence that EXISTS: pack byte-parity (now strong), wire-contract
audits. MISSING: any decode-token-vs-reference check, any COMPSEC/92x run,
any determinism receipt at the serving layer.

### qwen3flash

| claim | grade | evidence |
|---|---|---|
| bf16.tp8 / nvfp4.tp8 packs faithful | MEASURED | this wave (bf16 content PASS; nvfp4 digest PASS, content in flight) |
| validator 17/17 (cosines vs host oracle) | MEASURED | publication smoke 2026-09-08 + post-astra re-run; coexistence 2026-09-10 |
| nvfp4 repackage pick (nvidia 0.081 vs radixark 0.095 relerr) | DERIVED | receipt cites operator ruling + calibration comparison; the relerr derivation itself is not retained as an artifact in-repo |
| decode-vs-prefill bit_exact, determinism bit_exact | MEASURED but SELF-CONSISTENCY | publication/coexistence smokes |
| "byte-trace verified" (ledger 2026-08-29) | was DERIVED-then-BLOCKED | 08-30 addendum honestly corrected it to size-only; NOW measured by this wave |

Missing for T1: reference decode tokens (see gap list); multi-rank TP8
functional; tp4pp4 waves; fp8 arm placement verification (SP-4R).

### qwen38-27b

| claim | grade | evidence |
|---|---|---|
| spec 24.5 / no-spec 7.7-8.03 / prefill ~21.7 tok/s (TP1) | MEASURED | ledger rows with stream hashes + receipts |
| aggregate B1-32 "174x" | RETRACTED | ledger: units bug caught via roofline; corrected curve 8.31→41.3 |
| GPU validation vs real pack (bit-exact decode-vs-prefill, in-vocab MTP) | MEASURED, self-consistency | qwen27b-serve-2026-08-30 |
| tp4pp4 rank packs faithful | MEASURED | this wave (rank00 bit-identical re-derivation; ranks 1-15 not re-derived this wave, placement hashes unknown to this wave) |
| TP4xPP4 199/794/~8900 tok/s ladder | DERIVED, explicitly "projections, not measurements" | PERFORMANCE_LEDGER; never served on tp4pp4 |
| community baselines (7.88 beaten; DSpark 58.5 target) | ASSUMED-external | ledger cites community numbers, "cite and verify" per I42 not done in-repo |

Missing for T1: ds4_eval/COMPSEC receipts (none retained for 27B serving);
reference-token comparison; tp4pp4 serving.

### glm53full

| claim | grade | evidence |
|---|---|---|
| bf16 packs byte-exact vs source (16 ranks) | MEASURED | 2026-08-30 full verify + this wave rank5 3-leg + rank4/8 digests |
| fp8/nvfp4 packs faithful (rank0) | MEASURED | this wave plan-oracle PASS both arms |
| GPU validator cosines (bf16/fp8 tiers) | MEASURED, component-tier reference (host oracle) | lane report 2026-08-30 / 2026-09-10 |
| nvfp4 "correctness banked, routed cosine 0.99999" | MEASURED, component-tier | lane report 2026-08-30 |
| "ready for the Phase-2 fleet gate" | DERIVED | staging receipts complete; the gate itself never ran |
| served decode accuracy | ABSENT | "glm53full has NEVER been served end-to-end" (lane audit 2026-09-07) — still true |

Missing for T1: first serving cell, 8-token gate, reference comparison,
ds4_eval; nvfp4 ranks 1-15 content sweep (offline, tooling now exists).

### hy4

| claim | grade | evidence |
|---|---|---|
| FP8 kernel numerics NOISE-CLASS on placed bytes | MEASURED | rung-2 report 2026-09-10 (4584 dots vs CPU double oracle) |
| fp8.tp16 rank packs byte-faithful | MEASURED | rung receipts + this wave rank00 VERIFY PASS |
| scale-row-offset contract | DERIVED + MEASURED agreement | 72-case C/python test, vendor reference sha-pinned |
| UD-IQ1_M ~1-bit arm quality ("image workload tolerates the error profile") | ASSUMED | operator ruling 2026-09-01 recorded in the contract; zero quantifying evals in-repo |
| 259 replicated/dim1-split scale planes (o_proj/q_b/kv_b) kernel coverage | NOT COVERED | rung report: attach-milestone acceptance case, pending weightd seam |

Missing for T1: lazy-attach acceptance (blocked on the shared seam, k3
A-0023), end-to-end decode vs reference, ds4_eval.

### Fleet-wide accuracy truth

- NO locally-served SparkPipe model has a retained ds4_eval run. The two
  retained 92x runs are API baselines (Kimi K3 API 81/92, DeepSeek API
  70/92). The only local COMPSEC-17 attempt on record (glm53flash pre-fix,
  2026-08-30) correctly scored 17/17 degenerate — the gate worked and refused
  to bless a broken model.
- The pack-verification law (pack==packer==checkpoint) is now MEASURED at
  rank level for every lane in this wave; T1 decode-token reference evidence
  is the remaining accuracy plane and is absent fleet-wide.

## Findings (defects and laws discovered on the way)

1. **`tools/qwen38_pack_verify.py` is dead code for the family it names.** It
   imports `qwen38_27b_stagepack` tables but assumes a v1 28-word header with
   expert fields at words 20-25; the family v3 header (which the 27b packer
   itself writes and its `verify()` checks) orders `FFN_INTERMEDIATE, VOCAB,
   MXFP4_GROUP, MTP_LAYERS, tp_degree, tp_rank` there. Result: on a real 27B
   pack it crashes with `AttributeError: EXPERT_COUNT` (structure path) or
   reports garbage geometry; on the max family it is wrong by construction.
   The tp4pp4 packs were in-tree unverifiable until this wave's re-derivation
   approach. Fix owner: qwen38max/27b lanes (verifier should delegate to the
   packer's own `verify()` + a convert-re-derive digest, as this wave did).
2. **Byte-trace verifiers need a memory ceiling declaration.** Under a 4096M
   cap, `qwen4_flash_pack_verify` thrashed (276 s of system time, zero read
   progress, RSS pinned at ~3 GB) on multi-GB single-tensor compares; the
   same run passed in ~7 min under MemoryMax=14336M. Wave law: run big-family
   byte-traces with `MemoryMax=12288M+`, and always `python3 -u` (two runs
   silently lost their buffered output when inner timeouts fired — empty logs
   are ambiguous; unbuffered logs are not).
3. **Ceph stalls are per-(dataset, node), not global.** Live probes this wave:
   `qwen3.8-flash-next` at 1.1 GB/s on spark9 but the SAME mount stalled
   ~30 KB/s on spark0 for `glm-5.3-fp8`; `qwen3.8-27b-fp8` at 1.1 GB/s on
   spark0 but 6.1 MB/s on spark1 and <5 MB/s on sparkc. Playbook (matches
   wave-b1's incident): before any heavy warm read, dd-probe 150 MB of the
   target shard per candidate node, run on a fast node, mesh-copy the PACK to
   the verifier (packs are node-local NVMe, cheap to move, and byte-copied
   packs are the same evidence).
4. **Ratchet drift is pre-existing on main.** Clean origin/main `d1c3822`
   measures +1587 over the committed ceiling (283170); this wave's tools add
   +422 (ceiling bumped to the exact 285179 with justification in the same
   commit). The +1587 belongs to the landings that produced it — coordinator
   to adjudicate.

## GAP LIST — what stands between each lane and a T1 accuracy receipt
(decode tokens vs a pinned reference within tolerance)

Common shape: (a) a pinned reference forward (HF transformers / publisher
reference) producing expected tokens or layer outputs for a fixed prompt;
(b) a serving path that can produce those tokens; (c) the COMPSEC-17 gate.
The pack plane is no longer the gap anywhere in these five lanes.

| lane | gap | blocked? |
|---|---|---|
| all | No family reference-decoder harness in-repo except glm52's `glm52_transformers_stage_reference.py` (donor pattern). Port it per family; pin publisher modeling-file shas (qwen3flash receipts already cite `modeling_qwen4_exp.py sha 77fec77d`) | NO — offline now |
| all | ds4_eval COMPSEC-17 + 92x runs land in `qualification/ds4_eval/runs/` | YES — needs each model serving first (daemon/GPU); harness and protocol are offline-ready |
| qwen38max | validation harness absent (ledger gap #5); tp4pp4 layout-aware verifier; first serving cell (anchors only today) | verifier: offline now. Serving: needs queue window |
| qwen3flash | fp8 arm placement verify (SP-4R owns); multi-rank TP8 functional; wave B1 → COMPSEC | pack plane closed this wave; rest: serving |
| qwen38-27b | tp4pp4 ranks 1-15 re-derivation sweep (tooling proven on rank00); ds4_eval on the TP1 known-good serve | sweep: offline now. ds4_eval: TP1 serve exists (spark5) — schedulable, not code-blocked |
| glm53full | first end-to-end serve (never served); nvfp4/bf16 remaining ranks' content sweeps | sweeps: offline now. Serve: needs the lazy-arena chain (#829 lineage) + window |
| hy4 | lazy-attach acceptance incl. the 259 replicated/dim1-split scale planes (blocked on shared seam k3 A-0023); UD-IQ1_M arm has NO accuracy evidence beyond the operator ruling | attach: blocked on shared code. UD-IQ1_M eval: needs serving; nothing offline can quantify a ~1-bit quantization without a reference decode |

## INTEGRATION REQUEST

1. (coordinator) Adjudicate the +1587 pre-existing ratchet drift on main
   attributed in `tests/test_code_size.py` history.
2. (qwen38 lanes) Retire or fix `tools/qwen38_pack_verify.py` per Finding 1 —
   either delegate to `qwen38_27b_stagepack.verify()`/re-derivation digests or
   make it fail closed on unknown layouts (this wave's wave_acc1 tools show
   the pattern).
3. (owner of the no-comments law enforcement) `qwen4_flash_pack_verify.py`
   defaults to the MTP-carrying expectation and needs `--no-mtp` for the
   MTP-free nvfp4 arm — consider auto-deriving the expectation from the pack
   header's own `mtp_layer_count` word instead of a flag (fail-closed today,
   but the flag is a footgun; this wave hit it).

## Ledger

- Branch: `lane/wave-acc1-accuracy` (base origin/main `d1c3822`).
- Tools: `tools/wave_acc1_parity_oracle.py`,
  `tools/wave_acc1_glm53full_rank_verify.py` (committed at
  `1fa44b8` + this report's commit).
- Node-side checkouts used for verification: `~/acc1-src` on spark0, spark5,
  spark9 (all at the pushed branch tip, `git reset --hard` verified).
- Mesh-copied evidence packs: spark9 `~/acc1-packs/`
  (`qwenflash.tp8.rank5.pack` 46,333,527,808 B,
  `glm53full.fp8.tp16-rank0.glm52sp` 54,136,549,376 B,
  `model-fp8-tp16-rank-00.safetensors` 56,131,340,104 B,
  `qwen38_27b.tp4_pp4.rank00.spstage` 4,359,934,976 B,
  `qwen3flash.nvfp4.tp8.rank0.sp` 23,908,971,008 B) — deletable scratch;
  results above do not depend on the copies persisting.
- Raw verdict JSONs/logs: node `/tmp/acc1-*.json`, `/tmp/acc1-*.log`
  (`acc1-qmax-nvfp4-r0.json`, `acc1-q3f-nvfp4-r0.json`, `acc1-hy4-r00.json`,
  `acc1-g53-nvfp4-r0.json`, `acc1-glm53full-bf16-rank5.json`,
  `acc1-q3f-nvfp4-s9.log`, `acc1-hy4-r0b.log`, …).
