# The family pack-builder DRY study — definitive shape comparison and consolidation proposal (DRY-1, 2026-09-13)

Operator finding: "every dev did their own thing and we have a massive DRY
violation. It seems most of pack building is the same shape and having a
common packbuilder would make it easier to support new models." This document
is the measured proof of that finding and the executable migration plan behind
docs/UNIVERSAL_PACKER.md (the 2026-08-30 design sketch, here quantified).
Read-only study; no code changed. Baseline: main `ce2f8d2`; the dsv5 packer
reads from `origin/lane/dsv5-flash-dev` (`83b1a09`) where main has not yet
merged it.

## 1. What already exists (do not re-propose it)

* `tools/spark_pack_common.py` (309 LOC): wave-1 shared core —
  `PackFailure`, sha256/align, `SafetensorsSource`, `make_directory`,
  `pack_entry`, `pack_header`, `write_receipt`, `tp_shard_range`, the
  replicated-draft rule. 14 tools import it (12 packers, 2 verifiers).
* `docs/archive/DRY_CONSOLIDATION_PLAN.md` + `docs/archive/PACKER_CORE_PLAN.md`:
  the wave-1 inventory (primitives) — landed.
* `docs/UNIVERSAL_PACKER.md`: the operator's universal-packer directive —
  codecs/source/topology/receipts core + byte-compatible emitters. Design
  only; no LOC data, no verifier plan, no per-family gates. This document
  supplies those.

## 2. The corpus

Task packers (10 families, 11 files) and the wider same-shape corpus the
conversion eventually absorbs (dsv4 trio, gemma4, laguna, qwen36_dspark,
k3_dspark, stagepack_mtp_strip, the four tp4pp4/fanout drivers):

| packer | file | LOC |
|---|---|---|
| qwen38max | tools/qwen38_stagepack.py | 1,100 |
| qwen38-27b | tools/qwen38_27b_stagepack.py | 900 |
| qwen4_flash | tools/qwen4_flash_stagepack.py | 1,412 |
| dsv5 | tools/dsv41_flash_stagepack.py (branch) | 657 |
| muse | tools/muse_glimmer_stagepack.py | 427 |
| ling | tools/ling_stagepack.py | 898 |
| glm52 PP13 | tools/glm52_stagepack.py | 599 |
| glm52 resident v3 | tools/glm52_resident_stagepack.py | 1,051 |
| glm5_next | tools/glm5_next_resident_stagepack.py | 994 |
| k3 | tools/k3_pack.py | 860 |
| hy4 | tools/hy4_fp8_stagepack.py | 392 |
| corpus total (task 11) | | 9,290 |
| same-shape corpus remainder | 12 more files | ~7,600 |
| per-family verifiers/auditors | 14 files | 4,458 |

## 3. The shape comparison (a–g per packer)

(a) input, (b) topology plan, (c) codec handling, (d) output emission,
(e) receipt/manifest, (f) verifier, (g) LOC.

| family | (a) input | (b) topology | (c) codecs | (d) output | (e) receipt/manifest | (f) verifier | (g) LOC |
|---|---|---|---|---|---|---|---|
| qwen38max | shared-core `SafetensorsSource` subclass; config expectation table; contract sha | PP slices + conforming TP plans (`TpPlan` rows/cols/expert-ranges/qkv-segments); GDN/ATTN 4-period layer classes | bf16 pass, f32 widen, fp8 e4m3 + BF16→F32 scale_inv b128, nvfp4 radixark pass (U8 + F8 g16 + F32 globals), fused-BF16 MTP experts | v1 120 B / v2 128 B header (carries tp_degree/tp_rank), 56 B entries, 256 B align, temp+replace, write-through sha256, page-cache eviction | `.receipt.json` v1 via shared `write_receipt` | `qwen38_pack_verify.py` (416; imports the 27b packer's tables) + `qwen38max_tp16_rank_verify.py` (318) + `qwen38_stagepack_layout_audit.py` (278) | 1,100 |
| qwen38-27b | same skeleton, 27b geometry | TP4 rows/cols | fp8 b128 experts; bf16 spine | same qwen wire (v3 header) | `.receipt.json` | shares `qwen38_pack_verify` (loads these tables) | 900 |
| qwen4_flash | shared-core source + per-release mixins (`Fp8OfficialSource`, `Nvfp4OfficialSource`); PLE/indexer/HC kinds | `shard_ref` TP narrowing on refs; PP windows; MTP flag | 5 arms: fp8-f32b128 requant, fp8-e8m0b128 per-row MX requant, bf16 repackage, fp8-official pass-through, nvfp4-official pass-through; ngram F8→bf16 widen | same qwen wire | `.receipt.json` v1 + hc-semantics block | `qwen4_flash_pack_verify.py` (540; sampled dequant trace) | 1,412 |
| dsv5 | staged pipeline `headers`/`plan`/`copy`; flat header table; consumed-names census + engram/sidecar/vision accounting that must close | per-kind slice modes (repl/rows/sink/ob/expert) in `checkpoint_spec`; all-ranks piece list; shard-bounded copy for burst rotation | fp8 e4m3 + E8M0 32×32 scales; mxfp4 experts; bf16/f32 pass | `.spstage` (257 B header with revision + contract/config/recipe digest slots, 64 B entries, 256 B align); `plan.json` resumable | placement receipts (repack chain), `copy_LO_HI.done` markers | `dsv41_flash_pack_verify.py` (488; independent piece re-derivation, C-packer parity 1038/1038) | 657 |
| muse | shared-core source + `tensor_patterns.json` census expectations | TP16: per-head qgkv row spans, kv replicate-by-rank, gate_up fusion | bf16 only | same qwen 120 B/56 B framing | receipt with two-pass placement proof | NONE dedicated — receipt-only (gap) | 427 |
| ling | own mmap `SourceReader` with LRU (duplicates the shared core's reader) + `name_map.json` + pinned census count | `Packer.add_*` plan builders (replicated/rows/cols/fused-sections/up-gate/kv_b-transpose/conv/experts); `tp_shard_range` from the shared core | bf16 pass, f32, F32→bf16 RNE; experts bf16 pass-through only | 264 B header (codec-ABI), 64 B entries, 256 B align; `os.link` atomic + dir fsync; ck128 per-expert slab digests | `receipts/rank{r}.json` + `.sha256` sidecar + `.experts` v2 manifest (48 B records, ck128) | `ling_verify_pack.py` (270) | 898 |
| glm52 PP13 | torch `TensorSource` + model contract JSON | PP13 stages (6 layers/stage); no TP | `CODECS` table int6/int7/int8/fp8/nvfp4/mxfp4 — REQUANT from bf16 on CUDA | v3 264 B header + revision + 3 digest slots; 64 B entries | `.receipt.json` + recipe sha | `glm52_validate_pack.py` (254) + 4 test fixtures | 599 |
| glm52 resident v3 | mmap `Fp8SourceReader` (fp8 spine + scale planes, nvfp4 readers, source name mapping) | TP16 rank packs + PP stages; MLA kv_b split, indexer, fused gate_up | fp8 payload+scale copy-through; nvfp4 pass (block scales + per-expert globals); bf16 spine | 264 B header v3, 64 B entries | `build_receipt` + `write_all_ranks` | `glm52_validate_pack.py` + `glm53full_bf16_tp16_source_verify.py` (303) | 1,051 |
| glm5_next | same shape as glm52 resident (textually drifted) | TP + PP stages; KDA vs MLA layer classes | fp8, nvfp4, bf16 | same 264 B/64 B framing (v1) | receipt + stage pack names | `glm5_next_pack_verify.py` (327) + region tests | 994 |
| k3 | `SafetensorDir` + config-driven geometry (nested text_config, layer_types) | NO TP in the packer — `shard_class` annotations in the manifest; slicing is downstream (`k3_tp16_shard.py`, `k3_tp4_slice.sh`) | mxfp4 INTERLEAVED (tile_k 128, group 32 — bespoke transform); gamma_fold; MLA `q_fold_absorb` (compute-in-pack) | K3PK v2: JSON manifest + payload blob, 128 B align; resumable `.journal` writer | the manifest IS the receipt; `.experts` v2 via `k3_experts_v2_build_and_gen.sh` | `k3_verify_pack.py` (450; expert-cell `cross_verify`) + `k3_verify_source.py` (262) + `k3_deployed_audit.py` (46) | 860 |
| hy4 | raw index + own header loader | suffix-rule `SPLIT_RULES` table (dim0/dim1/replicate) + the scale contract (`SCALE_REPLICATED` four planes) | verbatim copy only (fp8 e4m3 + E8M0 companions) | SAFETENSORS-PER-RANK (format outlier) + `manifest-rank-XX.json` + `.sha256`; `--manifest-only` regen | manifest + sha sidecar | `hy4_fp8_pack_verify.py` (147; sampled) | 392 |

## 4. The deltas: what is family data vs copied shape

Every packer is the same seven-phase engine with different data plugged in:

1. family tables (kinds, geometry, name maps, config expectations) — DATA
2. source reader (index, shard headers, resolve, shape check) — SHAPE
3. plan builder (inventory walk + TP/PP slice math) — SHAPE
4. codec/copy (pass-through, widen, requant, scale planes) — SHAPE
5. emit (header, directory, alignment, atomic write, digest) — SHAPE
6. census/receipt (consumed-names closure, receipt JSON, sidecars) — SHAPE
7. CLI — SHAPE (trivially parameterized)

Measured evidence of the copied shape:

* Pairwise textual similarity (best-match alignment, difflib, exact): the
  eleven packers match another file 21% of lines on average (range 7–32%).
  The number is LOW because the copies drifted — which is the finding: same
  shape, independently mutated text. Structural (function-role) match from
  the AST census is far higher: `check_shape` ×7, `f32_to_bf16_u16` ×4,
  `kind_shape`/`layer_tensor_name`/`TensorRef`/`expected_tensor_count`/
  `build_inventory`/`SafetensorsSource`/`convert` ×3, `Entry`/`PlanItem`/
  `Packer`/`produce` ×3–18.
* glm52_resident vs glm5_next_resident: same reader/plan/emit skeleton
  (mmap reader, `Entry`/`PlanItem`/`Packer.add_*`, `emit_region`,
  `serialize_entry`, nvfp4 payload/scale/global readers), 25% textual match —
  a fork that drifted.
* qwen38 vs qwen38_27b: measured 379 identical lines at the 2026-08 audit;
  still one skeleton with two geometry tables.
* The 64-byte directory entry is ALREADY cross-family identical:
  glm52/glm52_resident/glm5_next/ling/dsv41 all pack the same 12 fields
  (kind, layer, payload_type, codec, scale_encoding, groups, rows, columns,
  payload_offset, payload_bytes, scale_offset, scale_bytes) into `<8I4Q>`-
  class structs. Only the qwen 56-byte entry and the two outlier formats
  (k3 JSON manifest, hy4 safetensors) differ.
* Dead copy artifacts found while reading (deletable today):
  `qwen38_stagepack.py` defines `copy_fp8_experts` TWICE (line 557 shadowed
  by 596) and `sharded_bf16_plan` is dead below its `if True: return`
  (the retired runtime-slice path, superseded by `build_tp_plan`);
  ling carries its own reader although it imports the shared core.

Phase LOC split across the 11 task packers (read-classified, ±10%):

| phase | LOC | share |
|---|---|---|
| family tables + geometry (data) | ~2,300 | 25% |
| engine shape (phases 2–6) | ~5,600 | 60% |
| CLI/main | ~750 | 8% |
| dead/shadowed | ~640 | 7% |

THE DRY PRIZE: ~5,600 engine LOC re-implemented 10× over, plus ~4,458
verifier LOC of which roughly 3,200 re-implement the same verify engine
(header parse → directory parse → expected-shape re-derivation from a SECOND
copy of the family kind map → sampled/full byte-compare vs source → receipt
cross-check). The kind map is duplicated packer-side AND verifier-side per
family — every D-2-class fix lands twice today.

## 5. The common pack-builder design

One engine, one CLI, per-family descriptors, byte-compatible emitters —
the UNIVERSAL_PACKER.md shape, made concrete:

```
tools/sparkpipe_stagepack.py        ONE CLI (build/verify/census/plan)
tools/stagepack_core/
  source.py      safetensors reading (extends spark_pack_common;
                 mmap-LRU and staged-headers variants behind one interface)
  topology.py    slice plans: rows | cols | expert-range | segments |
                 replicated; TP windows; PP stage windows; per-kind slice
                 modes (the dsv41 mode table generalized; hy4's suffix
                 rules become descriptor data)
  codecs.py      THE codec table (glm52_stagepack CODECS is the seed):
                 bf16 | f32 | fp8-e4m3{f32b128,e8m0b128,per-row-mx} |
                 nvfp4-g16-ue4m3(+globals) | mxfp4-e2m1-g32(+interleave) |
                 int6/7/8 — payload+scale plane pricing, requant engine
                 (CUDA torch behind an import guard), pass-through paths
  emit.py        header/directory framing (120/128/257/264-byte header
                 variants and the 56/64-byte entries as descriptor data),
                 256-align, atomic write, digest sinks (sha256 whole-file,
                 ck128 per-slab), page-cache discipline
  receipts.py    receipt JSON, .sha256 sidecars, .experts v2 manifest,
                 placement proofs (the muse two-pass check generalized)
  descriptors/   ONE family descriptor per model: kind table (kind → source
                 name template, shape, slice mode, layer membership),
                 geometry, codec allowlist, header variant, validation
                 hooks (config expectations, census closure, MTP law)
                 — model_contracts/*_authoritative.json stays the pin
  emitters/      per-format byte-compat writers: qwen-wire, codec-ABI-wire
                 (glm/ling/dsv41 entries), k3 K3PK (interleave + folds as
                 the one compute emitter), hy4 safetensors-per-rank
```

Dry-law: no family names in the core, ever. A new model = one descriptor +
one emitter selection; the engine is already tested.

The verifier consolidates onto the same engine: `stagepack_core/verify.py`
parses the pack, re-derives expectations from the SAME descriptor the packer
used (single-sourced kind map — the structural fix), byte-compares payload
against source (full or sampled), cross-checks receipts/manifests. Per-family
verifier files shrink to a descriptor reference + invocation, or vanish into
the CLI's `verify` verb. muse gains the dedicated verifier it lacks today.

Fan-out: the four tp4pp4/fanout drivers (88–192 LOC each) collapse into one
`--fleet-build` flag over the queue (UNIVERSAL_PACKER.md step 4). The six
`*_experts_manifest.c` C emitters of `.experts` v2 are one C utility keyed
by the directory layout, not six copies.

## 6. LOC projection

| | today | post-consolidation |
|---|---|---|
| engine (core) | ~5,600 copied | ~1,300 (written once) |
| family data (descriptors) | ~2,300 | ~2,200 (data stays; k3 compute emitter +300) |
| CLI + fan-out | ~1,150 | ~350 |
| dead/shadowed | ~640 | 0 |
| task-11 packers total | 9,290 | ~4,150 |
| verifiers | 4,458 | ~700 (engine ~600 + invocations) |
| TOTAL | 13,748 | ~4,850 (−65%) |

New-family cost: today a packer + verifier ≈ 1,400 LOC of copied engine
before any family data; after, the descriptor + emitter choice ≈ 300–400
LOC of data. That is the "easier to support new models" dividend.

The same-shape corpus remainder (dsv4 trio 1,840, gemma4 785, laguna 791,
k3_dspark 472, qwen36_dspark 313, mtp_strip 697, drivers 510) converts on the
same rails afterwards; DRY_CONSOLIDATION_PLAN item 1 (dsv4 family fold) is
absorbed by the descriptor mechanism.

## 7. Risks

1. BYTE COMPATIBILITY IS THE CONTRACT. Every family has sha-receipted,
   placed 16/16 sets (STAGEPACK_AUDIT_2026-08-31 §10: 16 sets PASS). The
   engine does NOT change any format; each family's conversion is gated on
   rebuilding a placed pack byte-identical (the wave-1 identity proof: 78.5 GB
   hashed, zero differing bytes). A family that cannot reproduce its bytes
   does not convert — the old packer retires only behind the identity gate.
2. Placement-matrix coupling: packs already placed stay untouched; the gates
   run warm-side rebuild + sha256 compare against the placed receipts. No
   fleet re-placement is required by this program (a rebuild is only pulled
   if the operator wants new arms anyway).
3. dsv5 is mid-flight on its lane (r2 packs placed from the branch; the
   C-packer parity gate converter==C-packer 1038/1038 is that lane's own
   contract). Conversion of dsv41 must not perturb the lane: schedule it
   after the lane merges or run it against the branch tip with the lane's
   consent.
4. The requant path (glm52 CODECS) needs CUDA torch; the engine must keep it
   behind an import guard so copy-through families never require torch (today
   only glm52_stagepack imports it).
5. Wire limits are descriptor data, not bugs to fix silently: the qwen38_max
   1,024-entry directory capacity overflow (wave-1 receipt, cluster 1) and
   the k3 64 MiB expert-range cap stay as declared limits until the operator
   rules on a wire change.
6. The k3 packer computes (interleave, gamma fold, MLA q-fold) rather than
   copies. Its emitter is the one genuine porting effort; bit-exactness is
   pinned by test_k3_pack* fixtures and the placed k3 sets.
7. Two engine modes must coexist (copy-through vs requant) plus three output
   formats (qwen wire, codec-ABI wire, k3/hy4 outliers). The descriptor
   carries the choice; the risk is descriptor schema creep — keep the schema
   closed: kinds, geometry, codecs, topology class, header variant, hooks.

## 8. Migration path (conversion order)

Reference rule: first emitter = the most recently landed, most-armed family
with a live byte verifier. Each step lands only with its identity receipt.

1. CORE EXTRACTION — pull `codecs.py`/`emit.py`/`receipts.py` out of the
   highest-adoption packers (glm52 CODECS, qwen emit, ling receipts);
   `source.py`/`topology.py` extend `spark_pack_common.py`. Unit tests
   against existing pack fixtures. No family converts yet.
2. qwen4_flash — first emitter (5 arms, TP+PP+MTP coverage, byte verifier
   already imports the shared core). Gate: byte-identity vs the placed
   qwenflash TP8 and TP4×PP4 sets.
3. qwen38max — the D-2-fixed directory path. Gate: identity vs
   qwenmax.pp16 receipts. Deletes the dead `copy_fp8_experts` twin and the
   dead `sharded_bf16_plan` body in the same change (deletion is part of
   conversion, not cleanup).
4. qwen38-27b — identity vs qwen27b.tp4; then delete the 27b fork body
   (frozen-deprecated family becomes descriptor-only).
5. glm52_resident + glm5_next — convert TOGETHER (isomorphic pair, one
   review). Gates: identity vs glm53full six variants + glm5_next sets;
   test_glm5_next_pack_regions + glm52 pack fixtures.
6. ling — identity vs ling packs AND `.experts` v2 byte-compare; absorbs the
   ck128 slab-digest sink into emit.py.
7. dsv5 (post lane-merge) — identity vs the r2 rank packs via the lane's own
   piece-re-derivation verifier.
8. k3 — the compute emitter (interleave + folds) against test_k3_pack*
   fixtures and the placed k3.mxfp4 sets.
9. hy4 — the safetensors-per-rank emitter variant + scale-contract hooks.
10. Retire the old packers behind drift/verify gates; sha receipts pin the
    transition; the verifier consolidation rides each step (the family's
    verifier dies when its descriptor-driven verify passes the same gates).

Sequencing note: steps 2–4 are one reviewer's week each including the
identity runs; 5 is the largest single diff; 8–9 are the only genuinely
new engine code (compute emitter, format variant).

## 9. What this proposal deliberately does not do

No implementation, no format change, no packer retirement, no fleet touches.
The deliverable is the measured case and the gated order. Implementation
routes to the stagepack campaign on operator approval of this document.
