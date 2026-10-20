# K3 rank pack format 3

Format 3 is the cell-split K3 rank pack. Compared with format 2 it holds three lossless changes:

- each rank's own rows of the router;
- each rank's own rows of the MLA down projections;
- expert scales stored as 4-bit codes.

The decoded values are bit-identical to the format 2 pack it was converted from. At TP16 the pack is 95,118,784,896 bytes instead of 99,566,844,288, which frees 4,448,059,392 bytes of weight memory on every node.

Format 2 is still the format for whole-router packs: TP1 packs and the K-split packs from `tools/k3_shard.py`. A cell-split pack has to be format 3.

## Header

The header is the same as format 2: magic `K3PK` (`0x4B33504B`), a u32 version (here `3`), and a u64 manifest length. The JSON manifest is padded with spaces to that length. Tensors are 128-byte aligned. `tools/k3_repack.py` keeps the source's manifest reserve, so the payload base stays at 1 MiB.

`format` is `{"version": 3, "alignment": 128, "expert_codec": "mxfp4_ws_paired_v1", ...}`. The stale global `mxfp4_interleave` block is removed; each expert tensor carries its own geometry.

`config` keeps every format 2 key, including `tp_degree`, `tp_rank` and `"expert_split": "cells"`, and adds:

| Key | Meaning |
|---|---|
| `source_format` | `2` |
| `source_pack_sha256` | sha256 of the format 2 rank pack it was converted from |
| `weights_identity` | the same sha256; a lossless repack keeps it |

## Rows per rank

Three tensors hold only this rank's rows:

| Tensor | Format 2 shape | Format 3 shape at degree d | `shard_class` |
|---|---|---|---|
| `model.layers.L.router_weight` | `[896, 7168]` | `[896/d, 7168]` | `expert_rows` |
| `model.layers.L.mla_q_down_weight` | `[1536, 7168]` | `[1536/d, 7168]` | `output_rows` |
| `model.layers.L.mla_kv_a_weight` | `[576, 7168]` | `[576/d, 7168]` | `output_rows` |

Each of these entries carries `"rows": {"first": tp_rank * count, "count": count, "total": total}`.

`router_bias` stays whole, because top-k runs on the gathered 896 logits.

**Router.** The cell path already computed only this rank's slice of the router logits and gathered them with the latent. It now reads the stored rows directly.

**MLA down.** Every MLA wave computes this rank's 96 `q_down` and 36 `kv_a` outputs:

- 16 rows or fewer use the skinny kernel;
- wider waves use the stream GEMM.

Both write the same 132-wide rows. The `MLA_DOWN` all-gather then lands them in a dedicated buffer, and the reorder kernel produces the full query latent and KV slot. This covers:

- the scatter path;
- gather prefill, including windowed and expanded-value passes;
- both paired half-wave lanes. Each lane fences its stream around the gather on the main stream.

The RMS norms run after the gather, as before. K3 MLA is NoPE, so there is no rotary step to place.

## Paired 4-bit scale cells

Expert tensors keep their shape and `shard_class`. Their kind is `mxfp4_ws_paired_v1` (format 2 used `mxfp4_ws_interleaved_v1`).

The layout is `[k_pair p][cell c][33 rows][16 bytes]`:

| Rows | Content |
|---|---|
| 0-15 | E2M1 payload of neurons `16c..16c+15` for k-tile `2p` (16 bytes per neuron, as in format 2) |
| 16-31 | the same for k-tile `2p+1` |
| 32 | byte `n` = `(e8m0(n, 2p) - base) \| (e8m0(n, 2p+1) - base) << 4` |

The `interleave` block records:

- `tile_k 32`, `group 32`, `stored_bits 4`, `row_bytes 16`, `cell_payload_rows 16`;
- `cell_rows 33`, `tiles_per_cell 2`, `cells`, `k_tiles` (even), `k_pairs`;
- `scale_bits 4`, `scale_base` in 1..239, `scale_span 16`;
- `rows_per_expert`, `expert_bytes = k_pairs * cells * 528`, and `tensor_bytes`.

**Decode.** The scale of a block is the float whose bits are `(scale_base + code) << 23`. That equals the format 2 decode `LmUe8m0ToFloat(e)` for every `e` in 1..254, which `scale_base` and the 4-bit code guarantee.

**Kernels.**

- The skinny kernel (`LmSkinnyPairedCellExperts`) gives warp half `h` the k-tiles `h, h+2, ...`, which is position `h` of every pair. Each lane sees the same tiles in the same order as with 17-row cells.
- The tensor-core kernel (`LmCellMmaExperts`) maps tile `t` to pair `t/2`, position `t%2`.
- `tests/test_skinny_cell.cu` requires both kernels to equal their 17-row counterparts bit for bit. The tensor-core counterpart is a test-only copy of the format 2 kernel.

**Size.** One `(layer, expert)` pair shrinks from 1,096,704 to 1,064,448 bytes.

## Refusals

- **Open.** A pack whose version is neither 2 nor 3 is refused, with a message that names `tools/k3_repack.py convert`.
- **Kinds.** Each format accepts only its own expert kind.
- **Bind (`K3PackFormatFits`):**
  - a format 3 pack must be a cell split at `tp_degree >= 2`, and its router and MLA down rows must equal `total / degree`;
  - a format 2 pack must hold the whole router and MLA down tensors;
  - a format 2 cell-split pack is refused with the conversion command.
- **Scale base.** Bind requires `tile_k 32` and a valid `scale_base` on both expert tensors of every routed layer.
- **Rank.** The stage model requires the pack's `tp_rank` and `tp_degree` to equal the stage's own, so a pack from another rank cannot bind.
- **Kernels.** The full-router and full-MLA-down kernels return a shape error on split tensors.

## Conversion

`tools/k3_repack.py` runs on each node against its local rank pack. It needs numpy and a `libk3manifest.so` built from `src/spark_ck128.c`, `src/spark_status.c` and `runtime/spark_weightd_manifest.c`.

**`check <v2.pack>`** is read-only (O_DIRECT). It reports each expert tensor's scale codes. A tensor is convertible when its codes fit one 16-code window inside 1..254. On 10-19 all 16 production ranks fit: w1 codes 109-123, w2 codes 119-124.

**`convert <v2.pack> <v3.pack> <libk3manifest.so>`:**

1. Hashes the source.
2. Takes each expert tensor's base from its lowest code.
3. Writes `<v3.pack>.partial`, with periodic fdatasync and page-cache drop so dirty memory stays bounded.
4. Runs `verify` on it, then renames the pack, its `.sha256` and its weightd `.experts` sidecar into place.

**`verify <v2.pack> <v3.pack> <libk3manifest.so>`** re-reads both packs from disk with O_DIRECT and checks that:

- the manifest is exactly the one the source implies;
- copied tensors are byte-equal;
- split tensors equal the source rows;
- every expert inverse-transforms to the source bytes;
- padding is zero.

It also computes the sha256 of the bytes on disk and writes a sidecar that weightd's own manifest loader accepts.

**Lane wiring.** The lane points at the format 3 pack with:

- links `packs/<name>`, `packs/<name>.experts` and `packs/<name>.sha256`;
- `packs/pack.sha256`;
- `spine_budget`, from `tools/weightd_spine_budget.py`;
- `config/adapter.json` `stage_pack_path`.

The `.wset` file is copied unchanged. The swap controller's `kimi-k3` `node_warm` warms the pack the lane links, `$HOME/k3-lanes/lane8/root/packs/k3.stage0.rank*.pack`.
