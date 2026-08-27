# GLM 5.2 validator routed-oracle fix — 2026-08-27

Worktree /tmp/lane-glm52-fix, branch lane/glm52-validator-fix (from
origin/lane/glm52). Scope: the M2 routed-expert validator gate that blocks
`module publish` (pack lane's P4). Node: spark8 (GPU GB10, driver 580.159.03,
CUDA 13.0). Inputs: the synthetic validator fixture + the deployed rank00
pack `/home/spark8/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank00.fp8.glms52sp`
(102,835,957,760 bytes). The validator never reads the pack (synthetic
weights); it gates the publish receipt.

## F1 Reproduce — exact failure captured

Command (spark8, workspace /home/spark8/glm52_fix_repo, this branch):

```
$ make -C modules/glm52_resident_decode_stage validate EXPERT_CODEC=fp8 \
    MODEL_REVISION=b4734de4facf877f85769a911abafc5283eab3d9 \
    CONTRACT_SHA256=ec5afd74d2ba0c913474f30d78209b4a49fa87667845f2968039fddad0fead7a \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
    STAGE_PACK_PATH=/home/spark8/sparkdata/glm52.tp8.fp8/packs/glm52_tp8_rank00.fp8.glms52sp
```

Raw output (validator stdout; full log /home/spark8/glm52_fix_f1_reproduce.log):

```
glm52_validation check=layer_forward_hidden elements=6144 relative_l2=0.00603604963 cosine=0.999981849 max_abs=0.001953125
glm52_validation check=layer_forward_residual elements=6144 relative_l2=0.0035535041 cosine=0.999993692 max_abs=0.001953125
glm52_validation check=determinism elements=12288 bit_exact=1
glm52_validation tier=dense PASS
glm52_validation check=routed_selection token=0 experts=5,1,0,3,7,6,4,2
glm52_validation check=routed_selection token=1 experts=1,7,5,3,4,0,6,2
glm52_validation check=routed_selection_set elements=16 exact=1 max_weight_delta=6.24359e-05
glm52_validation check=routed_expert_forward elements=6144 relative_l2=0.335913807 cosine=0.942554536 max_abs=882688
glm52_validation failure=routed_expert_forward detail=relative_l2
make: *** [../resident_decode_stage_rules.mk:200: validate] Error 1
```

Identical numbers to the pack lane's report and commit 9f56300 — fully
deterministic on the synthetic fixture (pre-dates the real packs, as reported).

## F2 Root cause — the oracle's payload read drops the expert dimension

The coordinator's working hypothesis (fixture allocates only 8 compact
payload slabs while the kernel indexes payload expert-major across 256
experts) is HALF right about the layouts but wrong about the disagreement
side. Byte-level audit of all three parties:

1. DEVICE (kernel): `LmGemmEncodeWeightMap` (runtime/gemm.cuh) encodes the
   expert payload as `rows = per-expert rows, columns = input dim,
   groups = group_count` — one TMA group per expert, so expert e's slab sits
   at `payload + e * rows*cols*bits/8`, expert-major across ALL 256 experts.
   Scales: `LmWeightCodecScaleTensor<ExpertCodec>(scale, GLM52_EXPERTS=256,
   rows, hidden)` + `LmScaleTensorIndex` = `group*group_stride +
   row*row_stride + k/k_group` — expert-major across 256 groups,
   row_group_size=1. This matches the real-pack layout the pack lane froze:
   payload fp8 codes verbatim expert-major, scales F32 scale_inv
   row-expanded per 128-block.
2. FIXTURE (SparkGlm52ValFixtureSetup): payload slabs for lanes 0..7 at
   `lane * payload_bytes` — for the pinned selection {0..7} this IS the
   expert-major prefix, inside an 8-slab allocation the kernel only ever
   addresses at offsets 0..7 (selection is pinned by the +4/-4 correction
   bias and `routed_selection_set exact=1` proves experts {0..7} ran).
   Scale buffer priced for all 256 experts (`ScaleBufferBytes(codec,256,...)`),
   planes 0..7 filled at their 256-expert offsets. THE FIXTURE IS CORRECT —
   it already mirrors the real pack layout verbatim. No out-of-bounds or
   zero reads exist: the device read real, distinct slabs.
3. ORACLE (the bug): `SparkGlm52ValDequantWeight` applies the expert
   dimension to the SCALE index only:

   ```c
   uint64_t scale_index = SparkGlm52ValScaleIndex(codec,rows,columns,expert,row,column);
   ...
   raw = SparkGlm52ValE4m3Decode((uint8_t)SparkGlm52ValReadCode(payload,codec,row,columns,column));
   ```

   `SparkGlm52ValReadCode` indexes `payload + row*row_bytes` with NO expert
   offset — every expert's forward reads slab 0's codes. Scales are uniform
   powers of two in the fixture, so the wrong slab is the ONLY error: for
   expert 0 the oracle is right; for experts 1..7 it computes expert 0's
   weights with expert e's (identical) scale.

Why the numbers fit exactly: route 0 of token 0 selected expert 5 (raw
output above: `experts=5,1,0,...`), so `routed_expert_forward` compares the
device's slab-5 forward against the oracle's slab-0 forward. The fixture's
fp8 code grid {0x08,0x2c,0x30,0x34,0x38,0x3c,0x40,0x44,0x88,0xb0} has a
positive mean (~0.76), so two independent slabs' forwards are strongly
correlated through the common-mean component: host simulation of two
independent slabs through the fixture's exact expert MLP gives
cosine 0.994 / rel_l2 0.11 at unit activation scale (see scratch
reproduction below) — the same regime as the observed cosine 0.9426 /
rel_l2 0.336 at the fixture's real activation scale. The alternatives are
excluded: identical slabs would give cosine 1.0; zeros/garbage (the
coordinator's missing-mixture theory) would give cosine ~0 and rel_l2 ~1.4.
Equal-norm check: rel_l2 = sqrt(2*(1-cosine)) = 0.339 at cosine 0.9426 —
the observed pair (0.3359, 0.94255) is self-consistent with a
partially-correlated wrong-slab read.

Scratch simulation (controller mac, python3, fixture grid + expert MLP):

```
$ python3 - <<'EOF'
grid = [0x08,0x2c,0x30,0x34,0x38,0x3c,0x40,0x44,0x88,0xb0]
... e4m3 decode: [0.015625 0.375 0.5 0.75 1. 1.5 2. 3. -0.015625 -0.5]
two independent slabs: cosine=0.993975 rel_l2=0.109633
EOF
```

The oracle selftest (`SPARK_GLM52_VALIDATOR_ORACLE_SELFTEST`) cannot catch
this: it fills ONE slab at offset 0 of a single-slab buffer and reads it
back with `expert=1` — it inherited the same wrong convention (the read at
expert 1 aliasing slab 0 was invisible), and it asserts only finiteness.
The referenced harness tests/test_glm52_cuda_validator_tier2_oracle.py did
not exist.

## F3 Fix

Mirror the kernel's expert-major payload addressing in the oracle:
`SparkGlm52ValDequantWeight` now offsets the payload by
`expert * payload_bytes_per_expert` (new helper `SparkGlm52ValPayloadExpertOffset`,
the mirror of `SparkGlm52ValScaleExpertOffset`), matching the weight tensor
map's group stride. Callers are unchanged (they already pass the
expert-major base). The selftest now fills TWO distinguishable slabs and
asserts per-expert reads differ (payload plane) and scale plane patches
scale the result bit-exactly (scale plane). A new host-executable gate
tests/test_glm52_cuda_validator_tier2_oracle.py compiles the validator TU's
selftest entry with tests/cuda_stub on any host (no GPU needed).

(sections F4/F5 below filled as the milestones land)

## INTEGRATION REQUESTS

(to be finalized in F5)
