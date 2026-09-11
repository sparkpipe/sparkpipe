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

## Round 2 (validation-debugger, 2026-09-11): V1 real-weight chain

Branch state: rebased onto origin/main 14df85a via origin/lane/muse-driver
845ee15 (coredev alignment) + repair commits. The lane delta vs main is now
28 files, +7388/-0: the muse family tree, the two shared norm kernels
(norm.cuh, purely additive), the muse tools/tests/contract, PROGRESS.md and
the code-size ceiling. The first rebase had silently kept old-main/donor
versions of ring/, node/, cache/, src/, sources.mk, the root Makefile, other
families' module Makefiles, the legacy tp test suite and the deleted-on-main
tp_device_collective_nccl.c (65 files, ±20K lines); 15aef76 restores every
shared file to main's content. The muse module Makefile deliberately keeps
the fad1fff nccl drop - main's tp_device_collective.c no longer dispatches
to nccl.

V1 evidence chain (sparkc, GB10, packs + warm copy):
- Frozen synthetic fixtures A1-A7: PASS on the golden-freeze host env
  (python 3.14.5, numpy 2.5.0, darwin arm64). A6 greedy token 191704,
  runner-up 140134, gap 0.0469 - exactly the frozen values.
- Real-weight goldens (expected_real/): A1-A7 built on sparkc against
  /mnt/model-warm/muse-glimmer-30b with every internal cross-check green
  (a6 full 52-layer prefill 86.7s). A6 real greedy token 1418; A5 real
  drop0 delta 5.9e-4 recorded in meta. Provenance (host, numpy, shas)
  frozen in each npz meta.
- A7 real-pack audit (real_pack_audit.py, independent wire transcription):
  PASS all checks on all rebuilt packs - rank 0/15 header+geometry+receipt
  shas, kv-head replication law (ranks 0/2/7 cached-K bitwise equal; 7 vs 8
  bitwise different), norm replication, and source spot-checks (embed,
  lm_head, k, gate, up, o_proj vs the warm safetensors).
- GPU module tier on the real audited rank00 pack: PASS - centered_norm
  ulp1, qk_norm_3_87 ulp1, window_decode ulp2, full_decode ulp2,
  window_walk boundary verified + decode bitwise, output_gate_and_silu
  bitwise exact. Receipts: pack run 16/16 exit=0 (~67s/rank, maxrss 645MB),
  index sha 7d817b4d, config sha 5a9df2d8 (PINS byte-identical).

Gate fixes committed this round:
- Packer wrote header.directory_offset = header+directory end; the frozen
  wire semantic (C loader + synthesize tool) is the directory start. This
  made the module reject every real pack with pack_geometry_mismatch.
- Validator module_decode tier now opts in via
  SPARK_MUSE_GLIMMER_VALIDATION_MODULE_TIER (default skip with printed
  platform reason): main refuses standalone direct pack loads - weightd
  attach via model_residentd is mandatory, and main's own glm5_next
  validator carries kernel tiers only. The module lifecycle E2E moves to
  the residentd/serving qualification step (needs the weightd lazy-pack
  attach plumbing; same platform-gap class as the credit-binding strip).
- Code-size ceiling re-measured: 245992 (muse stack +4665 over main's
  measured 241327 at 14df85a; main itself was +275 over its stale 241052
  pin - landing debt outside this lane).
- Contract: census RESOLVED 627 text / 809 vision (the pre-download 626/810
  estimate was off by one; the patterns side was right), digest_freeze
  filled (both shard shas + index/config/tokenizer files), modeling
  reference mispin corrected (4177486a was the HF model revision; the
  transformers commit is 4815a0a6, verified by sha256 against PINS).

Validation-side (batch-muse-val tree, not the PR): three latent reader bugs
fixed in the anchor tooling - muse_realweights/real_pack_audit resolved
safetensors reads to the shard data start instead of the per-tensor
data_offsets (mid-shard spot-checks compared the wrong bytes; first-tensor
checks passed by coincidence), a stale 3-tuple unpack, and a missing *2
(element vs byte) in the auditor's up-half offset. The A5 real-mode
negative-control canary is now noise-relative (the synth-tuned absolute
1e-3 threshold does not transfer to real weights; real drop0 delta 5.9e-4
is recorded in the golden meta and sits below the driver atol - the
boundary read-set checks carry the real-mode observability).

Deployment generator: no change needed. Main's own glm5_next generator
still emits session_ports tables post-rewrite; the muse generator already
matches that shape.

Next step for the lane: module lifecycle E2E (module_decode tier with
SPARK_MUSE_GLIMMER_VALIDATION_MODULE_TIER=1) once the weightd lazy-pack
attach plumbing lands, then fleet qualification under model_residentd.

Second hop (same round): main advanced 190 commits under the lane during
validation (14df85a -> 167cde7, qwen38max-sota + k3 fleet wave merges). The
branch was rebased again (-X ours: every shared-file conflict takes main;
the muse family files are additions and apply clean), the lane delta vs
167cde7 re-verified as muse-only + norm.cuh + ceiling, the code-size
ceiling re-pinned to the measured exact 238091 (muse stack +4672 over
main's measured 233419 at 167cde7), and the full GPU tier re-run GREEN on
the final tree (ab8679b) against the same real audited rank00 pack.
