# CONSTANT_AUDIT — fleet-wide hardcoded-constant audit (CONST-1, 2026-09-13)

Audit of origin/main `6b1357a19a17870754e355499bab8496d9874286` for constants
that drift: duplicated magic numbers, derived-but-hardcoded values, cross-file
sync pairs without enforcement, and config-vs-code duplicates. Scanner:
`tools/constant_audit_scan.py` (numeric-literal extraction + frequency + file
spread over .c/.h/.json/.sh/.mk; findings below verified by reading each hit in
context). Raw scan noise (small integers, token ids, doc line numbers, hashes)
was discarded; every ledger entry below has file:line evidence read in context.

Risk classes: HIGH = cross-file sync pair with no mechanical enforcement
(drift-active). MEDIUM = derived-but-hardcoded (parent change rots children
silently). LOW = documented single-use, tolerated.

## 1. Findings ledger

| # | Class | Where | Value(s) | Risk | Generative fix |
|---|-------|-------|----------|------|----------------|
| F1 | config-vs-code | `model_contracts/mimo25_authoritative.json` (model/attention/moe) vs `model-families/mimo25/include/sparkpipe/spark_mimo25_model.h:8-40` vs `inference/llms/mimo_2_5/config.h:7-32` | full architecture, ~20 values (4096, 48, 152576, 64, 192, 128, 64, 4, 8, 12288, 13568, 14848, 8192, 16384, 2048, 256, 8, 1e-5, 1e7, 1e4, 0.707) hand-written in both headers + literal in contract | HIGH (3-way) | contract is the parent; both headers generated (see G1/G2) |
| F2 | config-vs-code | k3: `inference/llms/kimi_k3/config.h:14` + `inference/llms/kimi_k3/generated_config.h:12` + `model-families/k3/include/sparkpipe/spark_k3_model.h:20` | 896 experts (plus shared 2, intermediate 3072, dense 33792, 93 layers, 96 MLA heads) | HIGH (3-way) | k3 model.h generated from k3 contract; config.h includes it (G2) |
| F3 | config-vs-code | glm52: `model-families/glm52/include/sparkpipe/spark_glm52_model.h` (hand-written, e.g. `:16` 256u) vs `model_contracts/glm52.json` vs `model_contracts/glm52_authoritative.json`; generator `tools/glm52_model_contract.py` reads the HEADER to emit the contract | 256 experts, 6144, 78, 64, 512, 12288 dense-intermediate | HIGH (direction inverted: for glm52 the header parents the contract; for glm5_next the contract parents the header via `tools/gen_geometry_header.py`) | one direction only: contract → header (G1) |
| F4 | config-vs-code | 18 `model_contracts/*_authoritative.json` exist; `tools/gen_geometry_header.py` FAMILIES covers only qwen38_27b, glm5_next (glm53_flash contract), qwen4_flash; `tools/generate_k3_contract.py` covers llms/kimi_k3 | all remaining family headers hand-maintained (mimo25, laguna, ling, gemma4, hy4, dsv4, muse_glimmer, qwen38_max, k3 model.h, glm52) | HIGH | extend the existing generator to all families (G1) |
| F5 | cross-file sync (already diverged) | wire code `FP8_E4M3_F32B128`: `include/sparkpipe/spark_stagepack_format.h:14` = 4u; `modules/qwen38_max_.../include/sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h:41` = 4u; `modules/qwen4_flash_.../include/.../spark_qwen4_flash_resident_decode_stage_firmware.h:41` = 4u; `modules/qwen38_27b_.../include/.../spark_qwen38_27b_resident_decode_stage_firmware.h:45` = **5u**; `tools/qwen38_27b_stagepack.py` `WEIGHT_FP8_E4M3_F32B128 = 5`; kernel `model-families/common/include/sparkpipe/spark_lm_kernels.cuh:40-41` has 4u = plain `FP8_E4M3` and 5u = `FP8_E4M3_F32B128` | same name, values 4 and 5, three numbering spaces | HIGH (FIX-NOW) | single format-code registry header; family codes `_Static_assert`ed against it (G4); qwen38_27b format.h pins BF16 (line 313) and NVFP4 (315) against core but omits the FP8 pin — the omitted assert is the smoking gun |
| F6 | cross-file sync | `model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_serving_constants.h:11-17` (parsed by `tools/gen_geometry_header.py:49`) vs `modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:59-66` re-defines MODEL_ID, DRIVER_MODEL_ID, STAGE_NAME, TARGET, PROGRAM_NAME, MAX_SEQUENCE_POSITIONS_CAP = 262144u; the .c does NOT include the constants header | dual-maintained identity block | HIGH (FIX-NOW) | .c includes the header; header is sole parent (G5) |
| F7 | cross-file sync | port bases in 8 generators: `tools/glm53full_gen_deployment.py:30-32,61,63` (19500/63700/60900/63500/64500), `tools/glm52_gen_deployment.py:21-23,29` (19480/63620/60700/64500), `tools/glm5_next_gen_deployment.py:27-33` (19560/63640/60710/61500/62550), `tools/laguna_gen_deployment.py:43-45` (19560/63640/60710), `tools/ling_gen_deployment.py:32-38` (19590/63560/60730/12288, block 12288-13311 per PORT_LEDGER), `tools/muse_gen_deployment.py:21-24` (13312), qwen38_27b/qwen38max generators | glm5_next and laguna defaults are IDENTICAL (19560/63640/60710) ±10%: co-deployment collision not proven, but identical defaults are one env-var oversight away; ledger lives outside the repo (sparkpipe-coord/PORT_LEDGER.md) | HIGH | in-repo `tools/port_ledger.json` parent; generators parse it; overlap assert; block size derived from TP/session matrix (G6) |
| F8 | cross-file sync (assert gap) | `_Static_assert` counts per family stagepack format: qwen4_flash 21, muse_glimmer 20, qwen38_max 19, qwen38_27b 18, gemma4 14, hy4 12 — vs **0** in `spark_ling_stagepack_format.h`, `spark_laguna_...`, `spark_glm5_next_...`, `spark_glm52_...`, `spark_dsv4_...` | 5 of 11 family wire formats carry no layout proofs | HIGH | G4: core proof macros in all 12 formats |
| F9 | cross-file sync (assert gap) | `include/sparkpipe/spark_stagepack_format.h:111` defines `SPARK_STAGEPACK_HEADER_BYTES 120u` + `SPARK_STAGEPACK_HEADER_LAYOUT_PROOF` (117-149); used by only 4 families (hy4:463, muse_glimmer:145, qwen4_flash:212, qwen38_max:180); gemma4/hy4/q38max/q38_27b/qwen4/muse hardcode `120u`/`56u` locally (e.g. gemma4:105-106, q38_27b:99-100); no core `SPARK_STAGEPACK_ENTRY_BYTES`, no entry layout proof | 56u duplicated in 6 headers; 120u in 7 | MEDIUM | core entry constant + entry proof macro (G4) |
| F10 | duplicated magic | python packers duplicate wire framing per family with textual drift: `docs/DRY_PACKBUILDER_PROPOSAL.md:58-63` — 264B header + 64B entries across ling/glm52/glm5_next/dsv5 (257B dsv5), "glm5_next same shape as glm52 resident (textually drifted)"; C-side counterpart is 120B/56B stagepack | 64B entry cross-family identical but copy-pasted | MEDIUM | shared pack core framing (the proposal's own emit.py path); pack framing constants from one module (G4) |
| F11 | derived-but-hardcoded | mimo25: `Q_DIM 12288` = HEADS 64 x HEAD_DIM 192; `FULL_QKV_DIM 13568` = Q_DIM + 4 x (192+128); `SWA_QKV_DIM 14848` = Q_DIM + 8 x (192+128); `O_INPUT_DIM 8192` = 64 x 128 — literals in `inference/llms/mimo_2_5/config.h:16-19`, `spark_mimo25_model.h:20-23`, AND literal contract keys `full_qkv_rows`/`swa_qkv_rows`/`o_input_dimension` in mimo25_authoritative.json | 4 derived values x 3 places | MEDIUM (HIGH inside contract: the "authoritative" artifact hardcodes derivable rows) | derive in generated header + `_Static_assert` vs contract keys (G1/G3) |
| F12 | derived-but-hardcoded | mimo25 layer schedule encoded two ways: period macro `MIMO25_LAYER_KIND`/`MIMO25_ATTENTION_PERIOD 7u` (`inference/llms/mimo_2_5/config.h:34-38`) vs 48-entry literal tables `SPARK_MIMO25_MODEL_LAYER_KIND`/`LAYER_IS_MOE` (`spark_mimo25_model.h:44-56`, table also specials-case layer 0) | same schedule, two encodings | MEDIUM | generate the table from (period, first_full_layer) in the header template; assert both encodings agree (G3) |
| F13 | derived-but-hardcoded | `tools/qwen38_27b_stagepack.py:72-89` hardcodes the parent geometry (HIDDEN 5120, LAYER_COUNT 64, PERIOD 4, heads, VOCAB 248320, ...) while naming the contract as DEFAULT_CONTRACT (line 51); it derives children correctly (`GDN_QK_DIM = GDN_KEY_HEADS * GDN_HEAD_KEY_DIM` etc., lines 91-96) but the parents are literals | 18 parent literals in the tool | MEDIUM | read parents from the contract json (G1) |
| F14 | duplicated magic (multi-semantics) | 262144: qwen38_27b serving cap (2 files, F6), context cap in 6 family headers (ling/qwen38_max/qwen38_27b/qwen4_flash/gemma4 x2), tokenizer piece cache `include/sparkpipe/spark_tokenizer.h:30` | same value, several semantics | LOW | per-family caps stay; serving cap single-sourced via F6 fix; document tokenizer cap as independent |
| F15 | duplicated magic (multi-semantics) | 12288: mimo Q dim, glm52/glm5_next/laguna dense-intermediate (`spark_glm52_model.h:20,61`, `spark_glm5_next_model.h:108`, `spark_laguna_model.h:77`), k3 gate rows (`tests/test_k3_bind.c:33`) | same value, distinct per-family semantics, family-namespaced | LOW | document; no action |
| F16 | cross-file sync | arm-name strings (`*.resident-decode-stage-firmware`, `serving-adapter.tp4.v1`, `cuda.sm121.*.bf16`) duplicated between family constants/adapter sources and `model_contracts/must_work_targets.json` + `examples/model_descriptions/*` | placement matrix as strings | MEDIUM | emit arm names from the same generated constants blob the adapter descriptor uses (gen_geometry_header already emits adapter constants for qwen38_27b — generalize) |
| F17 | duplicated magic | k3 pack shapes asserted in tests (`tests/test_k3_pack_load.c:42`, `tests/test_k3_bind.c:33`: 12288x7168) — tests hardcoding geometry is correct (they pin behavior); they must read the same parent once G1 lands | — | LOW | keep, re-point at generated header |

## 2. Generative-set proposal (minimal parents → all values)

Parents (the only places a number may be written):

- **G1 — contract parent**: `model_contracts/<family>_authoritative.json`,
  checkpoint-derived (freeze tools already exist: `glm53_contract_freeze.py`,
  `generate_k3_contract.py`, `generate_dsv4_contracts.py`,
  `generate_hy4_contracts.py`). Contract carries PRIMITIVES ONLY: heads,
  head dims, kv head counts, rope dim/window, layer count, period, phase,
  expert count, expert intermediate, vocab, epsilons, thetas, context cap.
  Derived keys (`full_qkv_rows`, `swa_qkv_rows`, `o_input_dimension`,
  `derived_geometry/*`) are either dropped or emitted by the generator and
  re-checked (G3), never hand-written.
- **G2 — geometry emission**: `tools/gen_geometry_header.py` FAMILIES extended
  to all 14 families; each `model-families/<f>/include/sparkpipe/spark_<f>_model.h`
  geometry block becomes generated (`--check` byte-identity gate, the existing
  qwen38_27b proof pattern). The `inference/llms/*` CUDA `config.h` files
  become `#include` shims of the generated family headers (k3 already
  half-does this via `generated_config.h` — keep that generator, feed it the
  same contract). Deletes ~20-value hand-written blocks in mimo25, laguna,
  ling, gemma4, hy4, dsv4, muse_glimmer, qwen38_max, k3, glm52 (F1-F4).
  glm52 contract direction inverted to match (F3).
- **G3 — derivation + asserts in the header template**: the template derives
  children with `_Static_assert` chains, e.g.
  `Q_DIM == HEAD_COUNT * HEAD_DIM`,
  `FULL_QKV == Q_DIM + FULL_KV_HEADS * (HEAD_DIM + VALUE_DIM)`,
  `SWA_QKV == Q_DIM + SWA_KV_HEADS * (HEAD_DIM + VALUE_DIM)`,
  `O_INPUT == HEAD_COUNT * VALUE_DIM`,
  `GDN_LAYER_COUNT == (LAYERS / PERIOD) * (PERIOD - 1)` (qwen4_flash already
  shows this pattern, `spark_qwen4_flash_stagepack_format.h:129-133`),
  `MOE_EXPERT_COUNT % TP_DEGREE == 0` (laguna/ling/glm5_next already guard
  this with `#if`), layer-kind table == period macro (F12). Cross-domain
  pairs (contract key vs header macro) get the same assert inside the
  generator's `--check`.
- **G4 — wire-code registry**: one header owns the pack wire vocabulary:
  format codes (fixing F5: one value per format name fleet-wide), 120B header
  + 56B entry + 264B/64B pack framing as named constants, and the proof macros
  (`SPARK_STAGEPACK_HEADER_LAYOUT_PROOF` + a new entry proof) adopted by all
  12 family stagepack formats and the shared python pack core (F8-F10).
  Rule: a family code that differs from the registry must fail to compile,
  not silently differ (the qwen38_27b FP8 omission shows the failure mode).
- **G5 — serving identity single-source**: the family constants header is the
  only definition; adapter .c includes it (F6). Generalize the
  `ADAPTER_CONSTANTS` emission so every family's identity macros and arm-name
  strings come from one generated blob (F16).
- **G6 — port ledger in-repo**: `tools/port_ledger.json` mirrors
  sparkpipe-coord/PORT_LEDGER.md; all 8 deployment generators read it and
  assert non-overlap; block sizes derive from (TP degree, session matrix)
  instead of literals (F7).

Net new constant count: parents stay ~1 per (family x primitive) in the
contracts + 1 wire registry + 1 port ledger. Every other audited occurrence
becomes an expression or an assert. The generative core is 6 mechanisms
(G1-G6); no new value is introduced anywhere.

## 3. Fix classification

- **FIX-NOW** (drift-active): F5 (FP8 wire code already 4-vs-5 across
  families sharing the name — the omitted `_Static_assert` proves the pair
  was known); F6 (qwen38_27b identity block dual-maintained, script-parsed on
  one side); F3 (glm52 contract direction inverted while glm52 lanes are
  active); F7 (port defaults identical across two arms; ledger not in repo).
- **FIX-AT-TOUCH**: F1, F2, F4, F8, F9, F10, F11, F12, F13, F16 — convert a
  file to the generated path when next edited rather than in one sweep.
- **DOCUMENT-ONLY**: F14, F15, F17 (same-value-different-semantics literals
  and test pins; the law tolerates single-source-per-semantics).

## 4. Honesty markers (±10%)

- F7 collision impact: glm5_next and laguna sharing control/collective/
  transport defaults is verified text; whether both arms are ever co-deployed
  on the same hosts was NOT verified.
- F5: modules translate family format codes to kernel codes explicitly
  (e.g. `spark_qwen38_max_resident_decode_stage_cuda.cu:1598-1613` checks the
  family constant, passes `SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128`), so no
  live misdecode path was found; the finding is the same-name-different-value
  hazard plus the un-pinnable assert, not a proven misdecode.
- The scanner's raw frequency table is dominated by benign noise; findings
  rest on the semantic review, not the counts.
- examples/, deployment JSON outputs, and docs were treated as generated
  artifacts (audited only where they are the other side of a sync pair).
