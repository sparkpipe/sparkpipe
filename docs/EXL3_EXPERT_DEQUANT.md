# EXL3 routed experts decoded to BF16 (quant A/B lane L11)

Design: quant-ab-design §11.4. These tools turn a community EXL3 checkpoint's
routed experts into BF16 expert entries on the S1 spine (the F1 layout), so an
EXL3 quantization can be scored through the same BF16-expert kernel as the
ceiling. A dq arm measures **weight error only**. It excludes exllamav3's
runtime approximations: FP16 activation rounding, the FP16 Hadamard stages,
and the Blackwell int8-activation GEMV (`EXL3_INT8_GEMV`, default 2).

The decode reimplements exllamav3 (MIT, see `NOTICE`).

## Tools

| File | Scope |
|---|---|
| `tools/exl3_expert_dequant.py` | Model-neutral. Codebook tables, trellis state extraction, tile permutation, exact fp64 reconstruction, one RNE rounding to BF16, safetensors reader, TP slicing on 128-wide Hadamard blocks. `lut` and `decode` subcommands. |
| `tools/exl3_dequant_parity.py` | Model-neutral. Parity gates against exllamav3 itself (needs torch + exllamav3 + a GPU). |
| `tools/glm5_next_exl3_graft.py` | GLM-5.3 Flash adapter. `slice` (one Ceph pass: per-rank slice files), `graft` (rank-local decode onto the F1 pack layout, via `glm5_next_expert_graft.py`), `check` (re-decode sample experts from the full checkpoint and compare with a placed pack). |

## Math

For a linear with `trellis` int16 `[k/16, n/16, 16K]`, `suh` fp16 `[k]` and
`svh` fp16 `[n]` (exllamav3 stores `W` as `[in, out]`):

    W = diag(suh) · H · W_inner · H · diag(svh) / 128

- `H` is the ±1 Sylvester Hadamard matrix of order 128, applied blockwise.
- `W_inner[r, c]` is `LUT[state]`. The state for tile position `t` is the 16
  bits of the tile's bit stream that end at bit `(t + 1)·K`, tail-biting.
  The stream is the tile's uint32 words (little-endian pairs of the int16
  words), MSB first.
- Position `t` maps to a row-major tile element through the tensor-core
  permutation (32 lanes × 8 values).
- `LUT` is the 65,536-entry table for the linear's codebook: `mcg` (marker
  tensor `.mcg`, multiplier 0xCBAC1FED), `mul1` (`.mul1`, 0x83DCD12D), or
  3INST without a marker. The tool refuses a table whose sha256 differs from
  the pinned value that was checked against exllamav3 on all states.
- K is read per tensor from the trellis width, so mixed K3/K4 needs no map.

Exactness: `W_inner` values are fp16 with magnitude below 4, so `H·W_inner·H`
is an integer multiple of 2⁻²⁴ below 2⁴⁰ and is exact in fp64; the product
with `suh` is exact too. Only the product with `svh` rounds (once, in fp64).
The BF16 value is then rounded RNE from the fp64 value. When the fp64 value
lands exactly on a BF16 tie, the exact product decides the direction, so the
BF16 image is the correctly rounded exact weight.

TP slicing: rank `r` of TP16 takes intermediate columns `[128r, 128r+128)`:
`trellis[:, 8r:8r+8]` and `svh[128r:…]` for up/gate, `trellis[8r:8r+8]` and
`suh[128r:…]` for down. A 128-wide slice is one Hadamard block, so the slice
decode is bit-identical to slicing the full decode (tested). Any other
slice width is refused.

## Parity gates (hard, before any graft)

`tools/exl3_dequant_parity.py --plan plan.json --output report.json` runs:

1. **Codebook tables:** on-GPU `reconstruct` of synthetic trellises built so
   every one of the 65,536 states appears, compared bit for bit with the
   table, for 3INST, MCG and MUL1.
2. **Inner weight:** random trellises (K 1-8, all three codebooks) and every
   sample tensor: `decode` equals `ext.reconstruct` bit for bit.
3. **BF16 rounding:** every element within ½ ulp of the fp64 weight.
4. **Against `get_weight_tensor()`:** relative Frobenius error and max
   absolute error over max |W| both ≤ 2⁻¹⁰. The element-wise relative error
   is reported but not gated: `get_weight_tensor()` rounds its intermediates
   to FP16 three times, so near-zero elements carry large relative error.
5. **Forward:** `LinearEXL3.forward` with `EXL3_INT8_GEMV=0`, rows 1, 8 and
   256, against `x·W` in fp64: relative Frobenius error ≤ 4·10⁻³.
6. **Slices:** a 128-wide TP slice decodes to exactly the columns (or rows)
   of the full decode.
7. **Format audit:** each sample is compared with the publisher BF16 weight.
   A permutation probe checks whether the checkpoint reorders an expert's
   intermediate units (see the lane notes for what was found).

## Pipeline for one arm (GLM-5.3 Flash, TP16)

1. `sha_verify` of the downloaded repo against the LFS sha256 in
   `SPARKPIPE_FETCH.json` (one Ceph window).
2. `glm5_next_exl3_graft.py slice` on one node inside a Ceph window: one pass
   over the checkpoint in file order, 16 slice files (`<arm>.exl3slice.rank<h>.safetensors`
   plus `.sha256`, ~9.5 GB each for K4). The slice metadata records the repo,
   revision, expert shard sha256s and the sha-verify receipt.
3. Copy rank `h`'s slice to node `h` over the fabric (no Ceph).
4. `glm5_next_exl3_graft.py graft` rank-locally: spine from the F1 pack,
   experts decoded from the slice, same directory and offsets as F1, header
   codec 1 (bf16), read-back of every region, spine digest equal to F1's,
   receipt `<pack>.receipt.json` (kind `expert-graft-receipt.v1`, expert
   source `exl3-offline-decode`) and `<pack>.sha256`.
5. `glm5_next_exl3_graft.py check` on sample ranks: re-decodes sample experts
   from the full checkpoint and compares the pack bytes.
