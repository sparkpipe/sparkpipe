# K3 pack format V2 — the consumption contract

This is the binding contract between `tools/k3_pack.py` (the only writer) and
the driver that turns a `.pack` into a resident `K3LayerWeights[]`. It is
written so that bind is pointer arithmetic: every number below is either in
the manifest header block or derivable from `generated_config.h`
(`K3_PACK_*`, `K3_KDA_*_FUSED_ROWS`), and both are checked by
`tools/generate_k3_contract.py` and `tests/test_k3_pack_layout.py`.

V2 exists to remove three V1 defects:

1. **Six GEMM launches read the same KDA activation.** V1 shipped q, k, v,
   beta, decay_down and gate_down as six tensors, so the layer launched six
   projections against `normed_bf16` (`inference/llms/kimi_k3/layer.cuh`
   K3-PERF-003). One wide GEMM needs one base pointer and one row stride,
   which V1's name-scattered placement could never give it.
2. **Weights and scales were two planes.** V1 shipped `expert_w{1,2}_weight`
   and `expert_w{1,2}_scale` as separate tensors, so a weight tile cost a TMA
   box plus a far LDG stream for its E8M0 scales — two address streams, two
   base pointers, no shared prefetch locality.
3. **Placement was unspecified.** V1 aligned tensors to 64B in dump order, so
   no consumer could assume TMA-base alignment or sequential prefetch.

There is no compat constraint: no V1 pack was ever generated.

## Container

```
offset  size  content
0       4     magic 0x4B33504B ('K3PK'), little-endian
4       4     format version = 2 (K3_PACK_FORMAT_VERSION)
8       8     manifest length M in bytes
16      M     manifest JSON (header block, below)
16+M    pad   zero to K3_PACK_ALIGNMENT (128)
...     ...   tensor payloads, every tensor 128-aligned, in emission order
```

Single file. The manifest is parsed once at load; nothing below requires
walking it more than once.

## Manifest

```json
{
  "format":  {"version": 2, "alignment": 128,
              "mxfp4_interleave": {"tile_k": 128, "group": 32,
                  "stored_bits": 4, "cell_payload_rows": 16, "cell_rows": 17,
                  "row_bytes": 64, "scale_bytes_per_neuron_tile": 4},
              "kda_fused": {"qkvb_sections": ["q","k","v","beta"],
                  "decay_gate_down_sections": ["decay_down","gate_down"]}},
  "config":  {"hidden": 7168, "layers": 93, "experts": 896, "latent": 3584,
              "intermediate": 3072, "kda_heads": 96, "kda_head": 128, ...},
  "tensors": {"<name>": {"offset": O, "bytes": B, "align": 128,
              "kind": K, "shape": [..],
              "sections":  [..],   // fused projections only
              "shard_class": "..", // fused projections + expert tensors
              "interleave": {..}}, // expert tensors only
              ...}
}
```

`offset` is relative to the payload base (the byte after the header pad).
`kind` is `bf16`, `f32`, or `mxfp4_ws_interleaved_v1`. Every tensor the
loader binds is a name lookup plus a base add.

## Tensor order

Emission order is the prefetch contract: `model.embed_tokens.weight`, then
for each layer 0..92 its tensors in consumption order (attention norm,
attention-side AttnRes row, attention projections, MLP norm, MLP-side AttnRes
row, router, routed experts, shared experts, latent projections), then
`model.norm.weight`, `model.attnres_out_weight`, `lm_head.weight`. A loader
that maps the file and prefetches sequentially never re-seeks.

## Fused KDA projections (one shard class per tensor)

The six projections that read one normed KDA input ship as TWO tensors,
because the TP shard table gives them two different classes
(`spark_k3_tp_shard_table.h`) and one tensor cannot carry two:

### `model.layers.N.kda_qkv_beta_weight` — OUTPUT_DIM_HEADS

BF16 `[K3_KDA_QKVB_FUSED_ROWS, hidden]` row-major = `[36960, 7168]`. Row
ranges (`sections` in the manifest, in order):

| section | row offset | rows | rows_per_head |
|---------|-----------:|-----:|--------------:|
| q       | 0          | heads*key_dim = 12288 | key_dim = 128 |
| k       | 12288      | 12288 | 128 |
| v       | 24576      | heads*value_dim = 12288 | value_dim = 128 |
| beta    | 36864      | heads = 96 | 1 |

Each section is head-major, so a TP rank holding heads `[h0, h1)` owns rows
`[off_s + h0*rph_s, off_s + h1*rph_s)` of every section — one contiguous
slice per section, per-head widths varying by section. Section bases are
128-aligned because a hidden row is 14336B = 112 lines.

Consumption: one GEMM over `normed_bf16` with output width 36960 into a wide
scratch, then the split kernel reads q/k/v/beta by the section table — the
shared fused-projection idiom implemented by the common model kernels. Four
launches become one; the activation is read once.

### `model.layers.N.kda_decay_gate_down_weight` — REPLICATED

BF16 `[K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS, hidden]` = `[256, 7168]`:

| section    | row offset | rows | rows_per_head |
|------------|-----------:|-----:|--------------:|
| decay_down | 0          | key_dim = 128 | 0 (replicated) |
| gate_down  | 128        | 128 | 0 (replicated) |

The bottlenecks replicate across TP, so this fused tensor replicates too and
every rank binds the whole thing. The second wide GEMM is one launch.

`decay_up` and `gate_up` are NOT fused into anything: their input is the
128-wide bottleneck, not `normed_bf16`. They stay separate tensors.

## Interleaved expert weight+scale (`mxfp4_ws_interleaved_v1`)

`model.layers.N.expert_w1_weight` and `expert_w2_weight` are each ONE tensor
carrying payload and scales; there are no `expert_w*_scale` tensors in V2.
The names keep their V1 meaning: w1 is the gate|up concatenation
`[experts][2*inter, latent]`, w2 is down `[experts][latent, inter]`.

### The grid

Per expert, the tensor is a byte grid of 64-byte rows:

```
row(e, t, c, r) = e * rows_per_expert + (t * cells + c) * 17 + r
    e  expert in [0, 896)
    t  k-tile in [0, K/128)          (128 elements = 64 payload bytes)
    c  16-neuron cell in [0, out/16)
    r  sub-row in [0, 17)

r in [0,16):  payload row — k elements [128t, 128t+128) of neuron 16c+r,
              MXFP4 nibbles exactly as the checkpoint ships them
r == 16:      scale row — byte 4n+j is the E8M0 of neuron 16c+n,
              k-group 4t+j (j in [0,4)), n in [0,16)
```

The closure is exact: 16 neurons × 4 scale bytes = one 64-byte row, so the
grid is **zero padding** —
`rows_per_expert * 64 == out*K/2 + out*K/32` to the byte, which
`interleave_geometry` in the packer asserts and the layout test re-proves for
the real K3 shapes (w1: 28 k-tiles × 384 cells × 17 rows; w2: 24 × 224 × 17).

### Why this geometry

- **64B rows** keep the existing TMA plan legal: the box inner extent is 64B,
  which `LmTensorMapPlanBuild` maps to `CU_TENSOR_MAP_SWIZZLE_64B` — the
  fragment path's `LmSwizzleSpanFor` agreement is untouched
  (`inference/kernels/tensor_map.cuh:18-24`).
- **The k-tile is 128 elements** because that is `LmMxfp4::kTileK`: the k
  extent whose payload fills one swizzle span (`formats/mxfp4.cuh:41-43`).
- **The scale row is co-tiled**, so one box covers a stage's payload AND its
  scales: the V1 far LDG scale stream is gone.

### TMA recipe (per expert GEMM operand)

One rank-3 descriptor per tensor:

```
data type   UINT8 (as today; 4-bit is described as bytes)
globalDim   [64, rows_per_expert, experts]
strides     [64, 64 * rows_per_expert]          (rank-1 entries, as today)
box         [64, 17 * (TILE_N/16), 1]           (TILE_N=128 -> 136 rows, 8704B)
swizzle     64B
```

Per k-tile stage the producer issues ONE `cp.async.bulk.tensor` at
coordinates `(0, (t * cells + neuron_base/16) * 17, expert)` — note the k
coordinate moved from a byte offset on dim 0 to the row coordinate on dim 1,
because dim 0 is now always exactly one span. `expect_tx` is the box bytes;
the stage barrier pricing is unchanged in kind.

Staged shared memory holds 17-row cells: rows 0..15 are payload (fragment
path unchanged, same swizzle agreement), row 16 is the cell's scales. A scale
byte read applies the same staged-row xor the payload path uses — row index
within the staged box, 64B pitch, `LmSwizzledOffset(row, col, 64, 64)`. The
`LmScaleTensor`/`LmScaleTensorLoad` far-plane path is not used for these
operands in V2; the scales are already in the stage.

The cost accounting: the box is 8704B where the V1 payload-only box was
8192B — the extra 512B is the stage's own scale bytes, previously fetched
separately. Net DRAM traffic is identical in bytes and one stream fewer;
smem grows 6.25% per stage.

### Scale-group alignment

A k-group (32 elements) is 16 payload bytes at offset `(k_group % 4) * 16`
inside its payload row — 16B aligned, never straddling rows — and its scale
byte sits in the co-tiled scale row. A group never crosses a TMA box.

## Alignment and validation

Every tensor offset is a multiple of 128 (`K3_PACK_ALIGNMENT`); the payload
base is 128-aligned by the header pad. The packer re-derives every layout
identity from the config before writing (`validate_layout`) — offsets aligned
and non-overlapping, interleave byte counts equal to payload+scales exactly,
section tables tiling the fused rows — and refuses the pack otherwise. A
checkpoint whose geometry the grid does not divide (K not a multiple of 128
elements, output not a multiple of 16 neurons) is a loud `PackFailure`, as is
an E8M0 `0xff` anywhere in the scale stream.

## TP sharding notes (for the shard wave)

- `kda_qkv_beta_weight` slices per section on whole heads, per-head widths
  from the section table; `kda_decay_gate_down_weight` replicates.
- `expert_w1_weight` output-splits on whole 16-neuron cells; the gate|up
  boundary (row `inter`) is a cell boundary whenever `inter % 16 == 0`, which
  the contract generator asserts.
- `expert_w2_weight` input-splits on whole 128-element k-tiles — a
  contiguous row range per rank. The V1 shard tool split K on 32-element
  groups; the interleaved grid coarsens that granularity to the k-tile
  (TP degrees 1/2/4/8 divide K3's 24 w2 k-tiles; 16 does not). This is the
  one granularity trade the interleave forces, and it is deliberate: the
  alternative — interleave units smaller than the swizzle span — would
  narrow every TMA box.
- The K3 shard tables (`spark_k3_tp_shard_table.h`, `tools/k3_shard.py`)
  classify by tensor name and know the V2 names (the shard wave):
  `kda_qkv_beta_weight` slices per section on whole heads,
  `kda_decay_gate_down_weight` replicates, and the scale-less
  `expert_w{1,2}_weight` slice on the interleave grid above. Unknown names
  are still refused loudly, so nothing mis-slices.

## What the bind wave consumes

Per KDA layer, the new weight-table entries are: `kda_qkv_beta_weight` (+ its
section table), `kda_decay_gate_down_weight` (+ its table),
`kda_{q,k,v}_conv_weight`, `kda_decay_up_weight`, `kda_decay_bias`,
`kda_head_log_scale`, `kda_gate_up_weight`, `kda_out_norm_weight`,
`kda_out_weight` — the V1 names `kda_{q,k,v,beta}_weight`,
`kda_decay_down_weight`, `kda_gate_down_weight` no longer exist. Per MoE
layer: `expert_w1_weight`, `expert_w2_weight` interleaved as above, no scale
tensors. MLA layers and everything else are unchanged from V1. The stale
callers — `K3LayerWeights`/`K3BindLayer`
(`inference/llms/kimi_k3/slice.cuh:34-211`), the six-launch block at
`inference/llms/kimi_k3/layer.cuh:417-440`, and the K3-PERF-003 comment
itself — are the driver wave's to update against this document.
