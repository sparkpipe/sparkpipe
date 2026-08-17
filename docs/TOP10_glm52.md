# TOP10 — GLM 5.2 driver (top-speed assessment)

Area: resident stage + DSpark backend + neutralized dispatch policy + MTP tree +
six expert codecs. Measured TP8 receipts: B1 6.91 / B8 43.46 / B16 75.55 tok/s.

## Where it stands today

**Accurate but bandwidth-bound, and the speculation machinery is inert.**

- The resident stage is token-exact and TP8-measured: B1 single-stream **6.91
  tok/s**, B8 aggregate **43.46**, B16 **75.55**
  (`PERFORMANCE_STATUS.md:590,592,621`). Every run emits the identical stream
  (`PERFORMANCE_STATUS.md:573`).
- B1 is the bandwidth floor: ~15.7 GB FP8 experts + ~6 GB BF16 spine per token at
  the 273 GB/s LPDDR5x ceiling = ~80 ms hard floor, plus ~31 ms of TP collectives
  (158 reduces/token) and ~25 ms launch overhead
  (`PERFORMANCE_STATUS.md:594-599`, `KERNEL_PLAYBOOK.md:389-391`). 6.91 tok/s
  is ~55% of the ~12.6 tok/s ceiling → **level 1 (accurate-but-slow)** on the
  METRIC ladder for B1; B8/B16 amortize weight bytes into **level 2-3**.
- The entire speculation path — 2344-line draft backend
  (`spark_glm52_dspark_draft_backend.cu`), 798-line neutral policy
  (`src/spark_speculation_policy.c`), 92-line tap-plan remainder, 107-line MTP
  tree — is **landed but buys 0 tok/s**: "no speculation … no speculative draft
  model … draft weights must be trained first; the base checkpoint ships none"
  (`PERFORMANCE_STATUS.md:581,586,600-601`).
- The landed policy does only **per-request** confidence thresholding
  (`SparkSpeculationPolicyConfidenceThresholdForRequest`
  `src/spark_speculation_policy.c:438`, `…AcceptedDraftLengthByConfidence` :449),
  not DSpark's global SPS(B)-aware greedy admission — the paper's whole
  "confidence-scheduled verifier" differentiator is still missing.
- Six codecs are packed (`glm52_stagepack.py:63-70` int6/int7/int8/fp8/nvfp4/
  mxfp4) but only **FP8** is served (kernel target `bf16.expert_fp8`,
  `PERFORMANCE_STATUS.md:580`).

## TOP-10 (ranked by Solutions / code size)

1. **Unlock the landed DSpark path with real draft weights.** Buys B1 from level 1
   toward level 3-4 (DSpark's 60-85% per-user gain) by activating ~3.2K already-
   landed lines. Delta **0 code** (weights). Owner: training/coordinator. First
   step: pin `RedHatAI/GLM-5.2-speculator.dspark` (`KERNEL_CONTRACT_CARDS.md:91`)
   and run `validate_glm52_dspark_epoch3_cuda.cu` against it.

2. **Implement the global confidence-scheduled verifier (SPS(B) greedy admission).**
   Buys B8/B16 aggregate (the real DSpark scheduler: pool+sort marginal survival
   values, greedily admit while Θ=τ·SPS(B) improves). Delta **+200-400** in
   `src/spark_speculation_policy.c`. Owner: glm52 + coordinator. First step: add
   an SPS(B) profile table and the pool/sort/admit loop beside :449.

3. **B1 reduce-path squeeze.** Fuse the 158 reduces/token (~31 ms,
   `PERFORMANCE_STATUS.md:596`) to the 3-step TP8 dual-link path
   (`PAIRED_DUAL_LINK_ALLREDUCE.md`). Buys B1 6.91 → ~9-10 (level 1→2). Delta
   small. Owner: coordinator (ring/transport). First step: profile the reduce pairs.

4. **B1 GEMV expert kernel.** "True B1 is a matrix-vector" (`KERNEL_PLAYBOOK.md:333`);
   the padded tensor-core atom wastes the single-stream expert pass. Buys B1
   (level 1→2). Delta **+150**. Owner: glm52. First step: measure B1 expert GEMM shape.

5. **Wire mxfp4/int codecs into the resident kernel.** mxfp4 halves expert bytes
   (~15.7→~7.9 GB) → lowers the B1 floor AND doubles KV-adjacent capacity; the
   template already exists (`layer.cuh:770-778`, `weight_codec.cuh`). Buys B1 +
   capacity (level up). Delta **+50**. Owner: glm52. First step: expert bytes/token
   under mxfp4 vs fp8.

6. **CUDA-graph the decode step (launch-overhead squeeze).** ~25 ms launch overhead
   (`PERFORMANCE_STATUS.md:597`). Buys B1 (level 1→2). Delta **+100**. Owner: glm52.
   First step: graph-capture one layer's kernel chain and measure.

7. **Migrate `glm52_stagepack.py` onto `spark_pack_common.py` (DRY).** It is the
   last un-migrated packer — still inlines `align` (:180) and `sha256_file` (:508)
   and uses plain `ValueError`. Delta **negative**. Owner: glm52. First step:
   swap the inline primitives for the shared names.

8. **Delete the dormant `glm52_resident_pack_common.py` (DRY).** 99 lines imported
   by nothing, side-effect-loads the contract, fully superseded by
   `spark_pack_common.py`. Delta **-99**. Owner: coordinator. First step: move its
   `tp_shard_range`/`parse_layers`/`tensor_name` into the shared module, then delete.

9. **Wire the MTP layer for MTP-1 speculation.** GLM52's checkpoint has an MTP layer
   (`MTP_LAYER_INDEX = 78`, `spark_glm52_model.h:44-46`) that is not served; MTP-1
   is the DSpark paper's baseline and needs no trained drafter. Buys B1 (level up)
   with near-zero new code. Owner: glm52. First step: confirm the MTP weights load.

10. **Neutralize the DSpark draft backend .cu for K3/DSV4-Pro reuse (DRY).** The
    2344-line `.cu` is GLM52-named; the audit's step-1 table made constants neutral
    but the kernels stay per-model — K3/DSV4-Pro would get a working drafter with
    zero new kernels. Delta **0→negative**. Owner: coordinator. First step: move the
    remaining `SPARK_GLM52_MODEL_*` in the .cu behind the neutral drafter table.
