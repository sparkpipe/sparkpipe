# DSV4 DRY audit (post-unified rebase)

Audited against `dsv4-pro-ga-pr` at the unified six-session tree. Findings are
ranked by blast radius: how many places must change when one fact changes.

## HIGH

### H1. Tensor geometry has three sources of truth (packer, C shape table, sharder)
Every tensor's rows/columns/format is declared three times:

- Python packer, per call: `tools/dsv4_stagepack.py:539` `add_layer_records`
  (e.g. `wq_a: query_rank x hidden, WEIGHT_FP8`) and the Pro copy
  `tools/dsv4_pro_stagepack.py:63` `pro_add_layer_records`.
- C header, per kind: `spark_dsv4_stagepack_format.h:193/237/244`
  `SparkDsv4StagePackShapeOf(Layer|Global)` — 105 kind references re-deriving
  the same shapes from model constants.
- Python sharder, per kind: `tools/dsv4_tp16_stagepack.py` (own KIND_* set +
  `shard_shape`/row/column logic).

A shape change (or a new tensor kind) must be made in all three, and nothing
checks them against each other at build time (only the runtime pack load
refuses a mismatch, late). Recommendation: generate the C shape tables and the
sharder's kind map from one Python geometry table (the
`MODEL_GEOMETRY` pattern the sharder already uses for flash/pro), or export
the pack's directory as the single shape source and have the C loader consult
the entries it already reads.

### H2. The Pro packer re-implements the Flash packer's layer builder verbatim
`tools/dsv4_pro_stagepack.py:63` `pro_add_layer_records` is a ~90-line copy
of `tools/dsv4_stagepack.py:539` `add_layer_records` with the geometry
constants swapped (7168/1536/384/16/64 vs 4096/1024/256/8/…), forced because
the Flash function hardcodes geometry as locals
(`tools/dsv4_stagepack.py:542`: `hidden, query_rank, q_dim, head_dim = 4096,
1024, 64*512, 512`). Same story for the MTP block
(`dsv4_pro_stagepack.py:175` vs `dsv4_stagepack.py:660`) and the header
packing (`pro_pack_header` at `:234` vs `pack_header` at `:790` — the
same 16I2Q struct.pack argument list).

Fix: a geometry table (like the consolidated sharder's `MODEL_GEOMETRY` in
`tools/dsv4_tp16_stagepack.py:37-83`) parameterizing one
`add_layer_records`; the Pro wrapper then shrinks to a config dict +
`--kv-codec` and the MTP block moves into the shared builder.

### H3. Five standalone stagepack frameworks (cross-family)
`tools/glm52_stagepack.py` (606), `qwen36_stagepack.py` (745),
`qwen38_stagepack.py` (633), `dsv4_stagepack.py` (968) — each re-implements
safetensors reading, record/directory/header serialization, payload/scale
copying, sha receipts, and verify. qwen38's docstring says it outright:
"Mirrors tools/qwen36_stagepack.py". Unified already collapsed the two DSV4
sharders into one geometry-parameterized file (`dsv4_tp16_stagepack.py`); the
same treatment for the packers would remove roughly 1500-2000 duplicated
lines. Per-model deltas should be config tables + the handful of genuinely
different tensor names (GDN conv for qwen36/38, k3's layout, etc.).

## MEDIUM

### M1. Model identity is hand-synchronized across 5+ files
`ga0813` / module-id / target / description shas live in:
`modules/dsv4_resident_decode_stage/Makefile.pro` (MODULE_IDENTIFIER_PREFIX +
suffix), `model-families/dsv4/include/sparkpipe/spark_dsv4_pro_model_aliases.h`
(ids, DRIVER_REVISION, DESCRIPTION_SHA256, MODULE_ID), the two model
description JSONs (`examples/model_descriptions/dsv4_pro_resident_decode_stage_firmware{,_b1}.json`),
`model_contracts/dsv4_pro.json`, and the generated manifests. I edited all of
them by hand for the GA migration. Recommendation: generate the Makefile
fragments + description JSONs from the contract (extend
`tools/generate_dsv4_contracts.py`, which already regenerates part of this).

### M2. The bucket-pair description JSONs are 95% identical
`dsv4_pro_resident_decode_stage_firmware.json` vs
`..._firmware_b1.json` differ in exactly 4 fields (max_inflight,
module id bucket suffix, max_active_slots, max_resident_sequences). Generate
the bucket variants from one template.

### M3. devcycle script family duplication
- `build_remote.sh` vs `build_pro_remote.sh`: same publish/compile/artifact
  tail, differing only in bucket/paths/module-id/adapter — the former already
  parameterizes BUCKET; merge into one script with a `--pro` geometry mode.
- `validate_ga_pro.sh:24-25` vs `publish_validator_wrapper.sh:14-31`:
  identical 13-variable `SPARK_DSV4_STAGE_*` env block and near-identical
  config-string formula; validate_ga_pro.sh additionally repeats the env block
  inline inside its ssh command (:25) — three copies of one environment. Move
  to one shared `dsv4_pro_validation_env.sh`.
- `finish_ga_download.sh` vs `finish_ga_parallel.sh`: same tree-refresh +
  missing-list + marker logic (sequential vs xargs -P4). One script with a
  `--parallel N` flag.
- `deploy_pro.sh`, `rankpacks_ship_pro.sh`, `split_ship_pro.sh`: three
  host-loop/ship variants; fold into deploy_pro.sh steps.
- Host paths are baked into the scripts (`sparkb` in validate_ga_pro.sh:21,25,
  `spark1` in the wrapper default) and drift every time the build host moves.

## LOW

### L1. Wire-kind constants defined twice in Python
`tools/dsv4_stagepack.py:83+` and `tools/dsv4_tp16_stagepack.py:113+` each
define the full KIND_* set (and MTP_LAYER_FIRST). The sharder already imports
the packer's module for some things; import the constants instead.

### L2. Serving-adapter capability chains hand-rolled per family
Four distinct `.capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | ...`
chains (`spark_dsv4_serving_adapter.c:282`, `glm52…:124`, `qwen36…:245`,
`qwen38…:204`), each re-listing the common decode/release/prefetch/kv-ownership
flags. Provide a shared base-chain macro + per-family extras.

### L3. Magic numbers repeated across packers/headers
hc mix rows 24, hc scale cols 3, index dim 64x128, index compressor channels
256, markov rank 512/256, vocab 129280 — each appears in the Flash packer, the
Pro packer, the C headers, and the sharder. They belong in the per-model
geometry/constants tables (H2/H1 fix absorbs most).

## Not violations (verified)
- `SparkLm*` helpers in `spark_lm_kernels.cuh` are shared by reference
  across glm52/qwen36/qwen38/dsv4 module kernels — no per-module copies.
- The validation harness (`spark_dsv4_resident_decode_stage_cuda_validation.cu`)
  intentionally re-derives fusion semantics (control vs candidate) for
  independence; keep, though its tiny bf16/linear helpers could import from a
  shared header without weakening the check.
- Unified already deleted `tools/dsv4_pro_tp16_stagepack.py` and
  `tools/dsv4_pro_tp4_pp4_stagepacks.py` and consolidated them into the
  geometry-parameterized `dsv4_tp16_stagepack.py` — the pattern to repeat
  everywhere above.

## Suggested order
1. H2 + H1 together: single geometry table driving the packer, C shape tables,
   and sharder kind map (largest blast-radius reduction, enables the rest).
2. H3 packer-core extraction (follows naturally from the shared table).
3. M1/M2 identity generation.
4. M3 devcycle consolidation.
5. L1-L3 cleanups.
