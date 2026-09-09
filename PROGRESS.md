# PROGRESS — lane muse (driver), branch lane/muse-driver

Coder stage output. Design contract: DESIGN.md (untracked, per criterion 9).
Anchors/oracle live in /Users/mac/batch-muse-val (separate lane; untouched).

## Per-criterion status (DESIGN.md section 9)

1. **DONE** — `model-families/muse_glimmer/` + contract + pinned modeling
   reference (commit 4177486a) exist; `python3 tests/test_muse_glimmer_model_header.py`
   exits 0: 24 bindings + 14 composed (>= 20).
2. **DONE** — `spark_muse_glimmer_stagepack_format.h` compiles standalone
   `-std=c11 -Wall -Wextra -Werror` (mac, cc Apple clang): 120-byte header
   proof, 56-byte entries, magic 0x47534D55 (grep of all STAGEPACK_MAGIC
   values showed no collision), shape assertions incl. 512B slot,
   768/2496/1248 per-rank rows, 12628 vocab rows.
3. **DONE** — `inference/kernels/norm.cuh` gains exactly the two kernels
   (`LmCenteredRmsNormKernel`, `LmHeadRmsNormKernel`), no model names;
   both compile on sparkc under build-all and are exercised by V0.
4. **DONE (V0)** — module builds on sparkc (aarch64, nvcc sm_121a, queue
   jobs muse-v0-build-004/008: exit 0) and the V0 validator passes
   end-to-end (queue job muse-v0-gpu-012, exit 0):
   `muse_glimmer_validation PASS` — centered norm ulp1, qk 3.87 exact vs
   hand-frozen constant, window walk at context 2049 (position 0 leaves
   the read set), NoPE/full decode path, output gate + silu-mul bitwise,
   module decode 8 steps x2 identical inputs -> identical tokens.
5. **DONE** — audit greps: 0 private norm/rope/kv/decode/gate/silu kernels
   in the module tree; 18 shared-kernel call sites; layer dispatch is
   `LAYER_IS_FULL_ATTENTION(layer)` (pure function of layer index); the 12
   GDN/MTP grep hits are shared-vtable constants (counts = 0) and the
   kv-tier protocol placeholder, not logic.
6. **READY-TO-RUN (blocked-on-download)** — `tools/muse_glimmer_stagepack.py`
   (dry-run proven: rank0/rank15 receipts, 419 tensors,
   3,639,537,920 bytes at tp16 = 3.4 GiB, kv heads 0 vs 1, two-pass
   placement proof) fires on the warm copy; A7 audits run with it.
7. **BLOCKED-ON-DOWNLOAD** — V1 anchors A1-A7: goldens frozen in
   /Users/mac/batch-muse-val; the real-pack runner consumes
   `tools/muse_glimmer_stagepack.py` + the anchor harness.
8. **BLOCKED (fleet)** — V2 needs 16 live ranks + warm weights.
9. **MANIFEST+SUMS** — regenerated in this commit (last tree change).

## V0 evidence

Queue (v2 tool) job muse-v0-gpu-012 on sparkc, exit 0:

```
muse_stage tp_passive degree=16 rank=0 (single-rank replay of a tp-sliced pack)
muse_stage initialize ok slice=0+52 tp=0/16 owns_embedding=1 owns_head=1
muse_glimmer_validation check=centered_norm tolerance=ulp1 worst_distance=1
muse_glimmer_validation check=qk_norm_3_87 tolerance=ulp1 worst_distance=0
muse_glimmer_validation check=window_decode tolerance=ulp2 worst_distance=1
muse_glimmer_validation check=full_decode tolerance=ulp2 worst_distance=0
muse_glimmer_validation check=window_walk context=2049 window=2048 boundary_drop=verified
muse_glimmer_validation check=output_gate_and_silu_mul bitwise=exact
muse_glimmer_validation check=module_decode steps=8 deterministic=exact tokens=1382 x8
muse_glimmer_validation PASS
```

## Design deltas found while coding (for the record)

- DESIGN section 2 says the final norm is centered; the pinned publisher
  source (line 467, MuseGlimmerRMSNorm) says PLAIN weighted RMS. The
  contract + module follow the source (final norm = LmBf16RmsNormKernel),
  matching the independent anchor finding.
- DESIGN section 5's `LmKvHeads<16, 1, 128, 128, 64>` has one parameter too
  many; the real template is `LmKvHeads<16, 1, 128, 64>` (slot 512 B, page
  64 slots) — same numbers it intended.
- Census: 52x12 kept layer tensors + 3 globals = 627 source tensors vs the
  contract's 626. Packer asserts pattern counts and reports the delta;
  resolve against the warm index at real-pack time. rotary inv_freq buffers
  are non-persistent (absent from safetensors).
- MLP is separate gate_proj/up_proj in HF (fused gate_up is a stagepack-side
  fusion); the fused qgkv interleaves q|gate per head because the shared
  LmSplitQueryGateKernel splits head-major.
- Attention rounding (scores->bf16, probs->bf16) is inside the frozen shared
  LmGqaAttentionDecodeKernel (fp32 unrounded); HF-exactness at those points
  is delegated to the anchor oracle's fp32-unrounded variants per the
  anchor report.

## Offline-gates status (important)

`make offline-gates` (build-all) is RED at pristine origin/main 8f3a6f2,
independent of this lane: `modules/k3_resident_decode_stage/source/
spark_k3_serving_adapter.c` is a half-committed refactor (orphaned
`if ( status == SPARK_STATUS_OK )` at line 516, undeclared
`seqslot_device/seqslot_host/dispatch/rows/submission`) - reproduced on
the untouched synced main checkout on sparkc (job muse-gates-001 and a
direct pristine-tree make). The manager should bounce the k3 lane.
Targeted gates on the muse tree all pass (job muse-gates-003, exit 0):
verify_package_manifest (with the lane DESIGN.md present), the header
bindings test, and the muse module archive (nvcc sm_121a). The manifest
covers the lane-local DESIGN.md by intent; the file ships with the tree.

## Environment notes

- The shared spark_pack_load/synthesize commons are MTP/GDN-entangled past
  their macro surface; muse uses standalone loader + synthesize tool with
  identical wire behavior (muse is the first family with no GDN/MTP).
- `runtime/stage_module_lifecycle.c` is in no library sources list; the
  module links it family-side (pre-existing gap, qwen38_max inherits it).
- Kernel host-side error reporting: module.c/synthesize use SPARK_FAIL per
  operator directive; device-side stays on LmFrameError/LmKvReport paths.
- Warm dir `/mnt/model-warm/muse-glimmer-30b` still ABSENT on sparkc as of
  this writing (polled via short-timeout ssh; no blind sleeps).
