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

The PP13 model contract keeps attention, DSA, dense layers, shared experts,
embedding, normalization, vocabulary head, and MLA KV in BF16. Only routed
experts are stored as NVFP4.

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

A PP rank owns one compiled B12x state and generated AOT workspace. Local
routed layers execute in model order against that shared state while retaining
layer-local expert weight buffers. The first layer owns the state; later layers
bind it as external state and cannot destroy it.

A routed GLM52 stage without this bound plan is invalid.  Missing packs, missing
compiled B12x archive, bad shapes, bad layout IDs, zero qualification hashes, or
insufficient token capacity all fail before the routed launch.

`maximum_token_count` is the number of execution rows accepted by the linked
AOT table:

```text
plain decode: execution_rows = logical_lanes
MTP/DsPARK:   execution_rows = logical_lanes * 7
```

Therefore plain B1024 requires an AOT maximum of at least 1024, while MTP or
DsPARK B1024 requires at least 7168. A 1024-row AOT table supports at most 146
logical lanes at seven rows per lane.

Validate one rank-local artifact set before building:

```bash
python3 tools/glm52_nvfp4_artifact_preflight.py \
  --rank 2 \
  --stagepack-root /path/to/rank2/stagepack \
  --nvfp4-pack-root /path/to/rank2/b12x \
  --aot-manifest /path/to/generated/aot_manifest.json \
  --max-active 1024
```

Add `--mtp` for the seven-row capacity contract. The preflight checks source
identity, exact pack layout, scale metadata, AOT object hashes, generated CUDA
files, runtime link paths, and the linked-kernel manifest hash. It has no
fallback mode.

The B12x implementation and scale/weight layout contract are derived from the
vendored FlashInfer B12x fused-MoE source under `third_party/flashinfer/` and
must retain the upstream attribution and license notices in that directory.
