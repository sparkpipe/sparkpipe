# GLM52 SM121 B12x Resident MoE Pack

The resident GLM52 routed-MoE path is strict: routed NVFP4 layers require a
real FlashInfer B12x compiled MoE primitive and a real resident expert-weight
pack.  The stage does not build synthetic buffers, does not expose the old
local MoE kernels, and does not fall back to generic CUTLASS or scalar CUDA.

The one-time packer is:

```bash
./tools/glm52_b12x_resident_pack.py \
  --model-dir "$GLM52_MODEL_DIR" \
  --aot-manifest build/glm52_b12x_aot/generated/aot_manifest.json \
  --layers 3,4,5,6,7,8,9,10 \
  --output-dir build/glm52_b12x_resident_moe
```

The packer is allowed to use Python, Torch, and safetensors.  Serving runtime is
not.  Serving runtime restores the produced `.spb12x` files with the native C
binder:

```c
SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateFromPackFile(...)
SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(...)
```

Each pack contains all 256 GLM52 experts for one routed layer.  Pack ABI v3 is
the first ABI that stores B12x scale factors in the required FlashInfer static
storage order; ABI v2 packs must be regenerated.

```text
w1 FP4 static view storage:        expert-major [256, 4096, 3072] uint8
w1 scale static storage:           expert-major [256, 4096, 384]  UE4M3 bytes
w1 alpha:                          [256] fp32 ones
fc2 input scale:                   [256] fp32 ones
w2 FP4 static view storage:        expert-major [256, 6144, 1024] uint8
w2 scale static storage:           expert-major [256, 6144, 128]  UE4M3 bytes
w2 alpha:                          [256] fp32 ones
```

The scale storage is not row-major `[expert][row][k_group]`.  For each expert,
the byte order is:

```text
m_tile
k_tile
outer_m = row % 32
inner_m = (row % 128) / 32
inner_k = k_group % 4
```

Equivalently, for one expert:

```text
row = (m_tile * 128) + (inner_m * 32) + outer_m
k_group = (k_tile * 4) + inner_k

target =
    (((m_tile * k_tiles + k_tile) * 32 + outer_m) * 4 + inner_m) * 4
  + inner_k
```

This matches FlashInfer's `convert_sf_from_mma_layout(...).contiguous()`
storage consumed by the generated SM121 native backend.  The packer remaps
checkpoint row-major scale tensors into that storage before writing `.spb12x`.

Gate/up order is fixed to the FlashInfer B12x contract:

```text
w1 rows 0..2047     = up projection
w1 rows 2048..4095  = gate projection
```

The packer bakes each projection's `weight_scale_2` into the corresponding
UE4M3 block-scale tensor and sets `w1_alpha`, `w2_alpha`, and
`fc2_input_scale` to one.  That matches the FlashInfer/vLLM B12x convention
that the runtime receives already-restored B12x weight views and scale storage.

The native binder validates the pack header, copies all regions to resident CUDA
buffers, initializes the B12x primitive through:

```c
SparkGlm52Sm121FlashInferB12xMoeCreate(...)
```

and exposes the stage dispatch plan:

```c
SparkGlm52ResidentDecodeStageB12xMoeDispatchPlan
SparkGlm52ResidentDecodeStageB12xMoePlan
```

A routed GLM52 stage without this bound plan is invalid.  Missing packs, missing
compiled B12x archive, bad shapes, bad layout IDs, zero qualification hashes, or
insufficient token capacity all fail before the routed launch.

The B12x implementation and scale/weight layout contract are derived from the
vendored FlashInfer B12x fused-MoE source under `third_party/flashinfer/` and
must retain the upstream attribution and license notices in that directory.
