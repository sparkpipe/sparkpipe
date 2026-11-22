# Fused kernels

`tools/fused_kernels.py` generates fused CUDA kernels from a plan. A fused
kernel takes one launch where a step launched several kernels, and its blocks
run the same device functions those kernels run, so every value is computed
with the same arithmetic in the same order. The generated test compares the
fused kernel's outputs byte for byte with the launches it replaces.

## Generating

```sh
python3 -I tools/fused_kernels.py model-families/dsv41_flash/fusion/decode.json
python3 -I tools/fused_kernels.py model-families/dsv41_flash/fusion/decode.json --check
```

A plan writes three files, which are committed with it:

- the header the stage includes, with each kernel, its argument struct and its
  launcher;
- the exactness test (a CUDA program);
- a manifest: the plan's SHA-256, and for each kernel its roles, the body each
  role runs, the kernels it replaces, the rows it accepts and what a launch
  patch may change (the row count and the grid width that follows from it).

`--check` regenerates the files in memory and fails when a committed file
differs. `tests/test_fused_kernels.py` runs it for every plan.

## Plans

A plan (`model-families/<model>/fusion/<step>.json`) names its three output
files, the headers the generated header and test include, its model's own ops
(under `ops`, in the same form as the catalogue below, with bodies in the
model's directory) and its kernels. Each kernel gives:

- its name, its launcher, its argument struct and its row-limit constant, all C
  identifiers of at most 64 characters;
- the threads per block (whole warps, at most 1,024), the most rows (1 to 16)
  and the dynamic shared memory as a C expression;
- its arguments. A pointer has an element count (a C expression in `rows`)
  and a test fill: random BF16 or FP32 values, zeros, or a C expression of the
  element's `index`. A scalar (`uint32_t`, `int32_t`, `uint64_t`, `float`) has
  either a value expression or a `derived` expression in `rows`, which the
  launcher evaluates at each launch; the manifest lists derived scalars as
  patchable. An output that a launch writes only in part (a ring slot, say)
  is marked `partial`. An output whose row 0 is not its first elements lists
  row 0's `[offset, count]` regions in `row_zero`, and an output of the whole
  launch (offsets over all rows, say) has `row_zero` none;
- scratch buffers that only the reference calls use;
- its roles: an op from the catalogue, the blocks the role gives each row (a C
  expression), and the binding of each of the op's parameters to a C
  expression over `arguments.<name>`. A role may end in a tail, an op that
  the last arriving block of that role runs, either for its row (`scope`
  row) or once for the whole launch (`scope` launch);
- the reference calls (the unfused launches), the outputs the test compares,
  the input it perturbs and the kernels it replaces. The test writes the
  type's default (1.5 for a float) into the perturbed input's first element;
  `bump` gives another literal when that one would not change the outputs, as
  for the router's scores, where 1.5 does not put expert 0 among a row's top
  six.

The grid is `rows` times the sum of the roles' blocks per row. A block belongs
to the role whose range holds its index, with its row and its item within the
row. Roles run concurrently and no block waits for another block, so a fused
kernel completes however few of its blocks are resident at once. A tail runs
after `LmFusedLastArrival`: each block of the role fences its writes and adds
one to its row's counter; the block that brings the counter to the role's
block count resets it to zero and fences again before it reads. The counters
are therefore zero between launches and between graph replays.

The generator refuses a plan with an unknown op, an op missing a parameter or
given one it does not take, a binding to a name that is not an argument, a
name that is not a C identifier of at most 64 characters, an output file in
shared code (`include/`, `runtime/`, `inference/`, `node/`, `cache/`,
`serving/`, `src/`, `common/`, `model-families/common/`) or outside
`model-families/` and `tests/`, an output or a perturbed input that is not an
argument, a tail without zeroed `uint32_t` counters or one that runs neither
per row nor once per launch, more than 16 rows, a block size that is not whole
warps, a name given twice, a model op that shadows a shared op or whose body
is in shared code, a scalar given a fill or both a value and a derived value,
an index fill without its expression, a constant or scalar output, a perturbed
scalar, a bump that is not a numeric literal, a read-only scratch buffer, or a
row-0 region that is neither a list nor none.


## Op catalogue

| op | body | as run by |
|---|---|---|
| `hc_sinkhorn_partials` | `LmHcSinkhornPartialTask` (`inference/kernels/hc.cuh`) | `LmHcSinkhornPartialKernel`: the per-warp partial dot products of one mix (or the sum of squares) over a row's flattened streams |
| `hc_sinkhorn_row` (a tail, one warp) | `LmHcSinkhornRow` (`hc.cuh`) | `LmHcSinkhornKernel`: the partials summed in warp order, the pre and post weights and the Sinkhorn combination |
| `hc_collapse_norm` | `LmHcCollapseNormRow` (`inference/kernels/fused/collapse_norm.cuh`) | a row's streams collapsed with the pre weights, RMS-normed with the whole row's sum, and one slice of it written in BF16 and through a family's quantizer |
| `topk_route_row` | `LmTopkWarpsRow` (`inference/kernels/topk_warp.cuh`) | `LmTopkWarpsKernel`: one row's top-k routes on `LM_TOPK_WARPS` warps, with the route fault and the renormalised weights. The keys are unique, so it selects the same routes as `LmTopkWarpKernel` in the same order, and the weights use the same sum. |

The kernels that run these bodies keep their names, launches and code: each is
a wrapper that calls its body with its block indices. The bodies are shared
code and carry no model name; a family passes its own pieces as template
arguments (DeepSeek V4.1 Flash's FP8 quantizer is `SparkDsv41FlashFp8QdqLaneOp`,
and its `D41RmsNormQdqKernel` is the `hc_collapse_norm` body).

## DeepSeek V4.1 Flash decode: the hyper-connection block

`D41FusedMixNormKernel` (`model-families/dsv41_flash/source/dsv41_flash_fused.cuh`)
replaces `LmHcSinkhornPartialKernel<4>`, `LmHcSinkhornKernel<4>` and
`D41RmsNormQdqKernel<256>` at both hyper-connection sites of every layer, for
1 to 16 rows: each row has 25 blocks for the mix partials, the last of which
runs the Sinkhorn row, and 20 blocks that collapse, norm and quantize 256
elements of the row each. The collapse does not depend on the mix the kernel
computes: the attention input collapses with the pre weights of the previous
layer's FFN mix (the initial weights at layer 0), the FFN input with this
layer's attention mix (`D41CollapsePre`). The stage launches the fused kernel
when no rows are owned, dumps are off and there are 1 to 16 rows
(`D41MixCollapses`), and the three kernels otherwise.

## DeepSeek V4.1 Flash decode: the attention output block

`D41FusedOutputKernel` replaces, after the window fold of the four-rank step:

- the window ring store, `D41WindowRingStoreKernel`;
- the sink merge, `D41MergeSinkKernel`;
- the inverse RoPE, `D41RopeRowsKernel`;
- for each output group, the copy of the group's columns into its input
  (`D41ColumnsKernel`) and their FP8 quantization (`SparkDsv41FlashFp8QdqKernel`).

Each row has one block for each of the rank's heads and one for the ring
store. A head's block:

1. merges the head's record with the sink into shared memory
   (`D41MergeSinkHead`, the body of `D41MergeSinkKernel`);
2. turns the last 64 elements back (`D41RopePair`, the body of
   `D41RopeRowsKernel`);
3. writes the head's 512 elements through the FP8 quantizer into its group's
   input. Every warp holds whole 32-element blocks, so each block's scale is
   the one the quantization kernel computes.

The group inputs are laid out group by group (`[group][row][4096]`), so each
group's o_a projection reads its input as it is. The projections write
straight into `wo_a_out` at the group's column offset (`D41SkinnyLinear` with
an output row stride), so the two output-side column copies go as well.

The plan declares the two model ops: `d41_output_head` (`D41OutputHead`) and
`d41_ring_store` (`D41WindowRingStoreRow`, the body of
`D41WindowRingStoreKernel`).

The stage takes this path (`D41OutputFuses`) when all of these hold:

- the step is not a tensor-core attention wave;
- no rows are owned and dumps are off;
- there are 1 to 16 rows;
- the group input does not go through the MX quantizer.

A layer's attention output block then launches 5 kernels instead of 13: the
fused kernel, the two o_a projections, the quantization of `wo_a_out` and o_b.

## DeepSeek V4.1 Flash decode: the router block

`D41FusedRouteKernel` takes everything between the router's GEMV and the
experts:

- the top six of 384 experts, `LmTopkWarpKernel`;
- the mapping to the rank's local experts, `D41RouteLocalKernel`;
- the expert-grouped route table, `LmRouteBuildKernel`;
- the packed route weights, `LmPackedRouteWeightKernel`.

Each row has one block. The block's eight warps score the row's candidates
and select its routes (`LmTopkWarpsRow`); one warp per row did this before.
The last block of the launch then runs the local mapping, the route table
and the packed weights for all rows (`D41RouteTail`, from the bodies
`D41RouteLocal`, `LmRouteBuildBlock` and `LmPackedRouteWeight`). The route
table's tile size depends on the row count, so the launcher derives it
(`tile_m`).

The stage takes this path when no rows are owned, dumps are off and there
are 1 to 16 rows (`D41RoutesFuse`). Four kernels become one.

## Tests

- `tests/test_fused_kernels.py` (host): every plan regenerates its committed
  files, `--check` fails on a changed generated file and passes after
  regeneration, and each refusal above is made.
- `tests/test_dsv41_flash_fused_exact.py` (sm_121): for each fused kernel,
  for 1 to 16 rows, with ordinary and edge-valued inputs, every output is the
  same bytes as the unfused launches write. Row 0 is the same at 1 and at 16
  rows, and a changed input changes the outputs. Three graph replays give the
  same bytes and leave the counters at zero; before each replay, fully
  written outputs are overwritten and partially written ones are restored.
  The test is clean under `compute-sanitizer` memcheck, synccheck and
  racecheck.

## Measurements (2026-11-22, one GB10 host, CUDA flavour)

- **Code of the refactored kernels** (`tools/kernel_codegen_diff.py` on the
  four-rank adapter's cubins): `LmHcSinkhornPartialKernel<4>`,
  `LmHcSinkhornKernel<4>` and the other 332 kernels are unchanged.
  `D41RmsNormQdqKernel<256>` schedules its integer index arithmetic
  differently (35 registers instead of 33); its floating-point instructions
  are the same in kind and number.
- **Whole step** (`docs/DECODE_REPLAY.md`, 30 steps after 5, two runs each):
  the final streams are the same bytes with and without the fused kernel at
  1, 8 and 16 rows.

| rows | step, unfused | step, fused | hyper-connection block, unfused | fused | kernels a step |
|---|---|---|---|---|---|
| 1 | 27.30-27.34 ms | 26.94-26.95 ms | 20.8-21.2 us | 16.3-16.7 us | 1,921 -> 1,761 |
| 8 | 66.19-66.21 ms | 65.81-65.83 ms | 25.6-25.7 us | 20.4-21.5 us | 1,921 -> 1,761 |
| 16 | 110.57-110.73 ms | 110.02-110.15 ms | 41.6-42.3 us | 36.1-37.4 us | 1,921 -> 1,761 |

  Step times are the traced spans; event-timed step medians drop by
  0.41-0.83 ms. At one row the unfused partial kernel takes 12.9 us with its
  launch gap, reading 1.97 MB of mixer weights (about 165 GB/s); the fused
  kernel adds the Sinkhorn row's 3-4 us to it, during which memory is idle.
- **What does not help.** Adding the mixer weights to the preceding round's
  prefetch group makes the fused kernel 6.4 us faster at one row, but the
  weights it displaces from the group slow the next projections by as much
  (step -0.09 ms at one row, +0.19 ms at eight). Prefetching each partial
  block's mixer column into L2 from inside the kernel changes nothing at one
  row and costs 2.2 us at eight.

### The attention output block (2026-11-22, a second GB10 host, against the hyper-connection fusion)

- **Code of the refactored kernels.** `D41WindowRingStoreKernel` and the
  other 333 kernels of the four-rank adapter are unchanged. `D41MergeSinkKernel` and `D41RopeRowsKernel`
  schedule their addressing differently; their floating-point instructions
  are the same in kind and number.
- **Whole step.** The final streams are the same bytes with and without the
  fused kernel at 1, 8 and 16 rows.

| rows | step, before | step, fused | output block, before | fused | kernels a step |
|---|---|---|---|---|---|
| 1 | 27.87 ms | 27.32-27.33 ms | 115.6 us + 2.4 us ring store | 104.7-104.8 us | 1,761 -> 1,441 |
| 8 | 65.09-65.12 ms | 64.54-64.58 ms | 113.3-113.5 us + 2.6 us | 102.4 us | 1,761 -> 1,441 |
| 16 | 108.13-108.27 ms | 107.59-107.81 ms | 159.5-160.0 us + 2.6 us | 148.4 us | 1,761 -> 1,441 |

  Event-timed step medians drop by 0.54-0.68 ms. Of the fused block, the
  three projections are 98 us at one row and 95 us at eight. Their weights
  (19.5 MB with the scales) take at least 84.6 us at 230 GB/s. The rest is
  the boundaries between the projections, which a dependent launch that loads
  its weights before it waits would close.
