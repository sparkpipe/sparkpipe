# STAMP-CHECK RECEIPT — qwen3flash.fp8.tp8 (Q3F-T1, 2026-09-15)

Verdict: **FOUND DEFECT** — stamp/geometry contract clean, scale-plane
CONTENT broken on every placed pack. Decode per the codec contract is
impossible; accuracy-fatal on this arm.

## What was checked (before decode, per dispatch)

Instrument: `tools/qwen4_flash_pack_stamp_check.py` (this lane). For every
directory entry of a pack: format stamp vs the contract natural format
(`spark_qwen4_flash_stagepack_format.h`: routed experts = 4
FP8_E4M3_F32B128, everything else bf16/f32/i64), payload/scale byte math
(`SparkQwen4FlashStagePackPayloadBytes/ScaleBytes`), TP8 narrow geometry,
offset alignment, file bounds, range overlap. Plus hand-checks (K3A
pattern): bf16 decode of layer-0 GDN QKV, fp8 block decode of layer-0
MOE W1, PLE i64 constants. Streaming reads only (header + directory +
small slices; no multi-GB read).

## Result, rank0 `~/sparkdata/qwen3flash.fp8.tp8/packs/qwenflash.tp8.fp8.rank0.spstage`

- Header: magic 0x50533451, format_version 2, 1215 entries, 48 layers,
  mtp 0 — matches the receipt and the expected tensor count.
- Directory contract walk: PASS (stamps, sizings, geometry, alignment,
  bounds, no overlaps, file covered exactly).
- bf16 hand-check: layer-0 GDN QKV decodes finite, std 0.0201 — sane.
- fp8 payload codes: sane e4m3 exponent histograms.
- **fp8 scale planes: 144 of 144 entries carry non-f32-scale content.**
  Layer-0 W1 plane f4[0] = 2.362e10, plane median 6.8e22. The bytes are
  u16->u32-widened text of the SOURCE checkpoint's safetensors header:
  `{"model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv":
  {"dtype":"BF16","shape":[5,20],"data_offsets":[0,200]},...` — the
  repair pass re-copied from the source FILE START (header region)
  instead of the scale tensor's data offset, then widened bf16 to f32.

## Fleet scope (MEASURED, one probe per host)

All 16 placed packs identical-broken (rank r on spark r and spark r+8):
f4[0] = 2.362e10 and the same widened-header bytes on every rank.
Also broken: qwen3flash.fp8.tp4pp4 stage3 rank13 on sparkd (finisher
placed; kind-6 layer-36 probe, f4[0] = 1.548e15, same signature).
Control: qwen3flash.bf16.tp8 rank0 — zero fp8 entries, unaffected.

## Why the finisher gate missed it

`tools/qwen4_flash_pack_verify.py` sample_trace forces nvfp4 (wire-8)
sample coverage but not fp8; on an all-fp8 arm the 8 samples can land on
zero fp8 entries, so the dequant-vs-source check that would have
convicted never ran. The repair receipt then re-verified the streaming
sha only. The pack's own receipt cites "8-sample byte-trace PASS" from
the pre-repair build; the repair (replaced_sha256 4690c141... ->
output_sha256 59872174...) broke the scale planes after that verify.

Raw checker output: `STAMP-CHECK-rank0.json` (failures list).
