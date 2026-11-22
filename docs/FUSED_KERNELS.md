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
files, the headers the generated header and test include, and its kernels.
Each kernel gives:

- its name, its launcher, its argument struct and its row-limit constant, all C
  identifiers of at most 64 characters;
- the threads per block (whole warps, at most 1,024), the most rows (1 to 16)
  and the dynamic shared memory as a C expression;
- its arguments: name, pointer type, element count as a C expression in
  `rows`, and how the test fills it (BF16, FP32 or zeros);
- its roles: an op from the catalogue, the blocks the role gives each row (a C
  expression), and the binding of each of the op's parameters to a C
  expression over `arguments.<name>`. A role may end in a tail, an op that the
  row's last arriving block of that role runs;
- the reference calls (the unfused launches), the outputs the test compares,
  the input it perturbs and the kernels it replaces.

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
argument, a tail without zeroed `uint32_t` counters or one that is not per
row, more than 16 rows, a block size that is not whole warps, or a name given
twice.

## Op catalogue

| op | body | as run by |
|---|---|---|
| `hc_sinkhorn_partials` | `LmHcSinkhornPartialTask` (`inference/kernels/hc.cuh`) | `LmHcSinkhornPartialKernel`: the per-warp partial dot products of one mix (or the sum of squares) over a row's flattened streams |
| `hc_sinkhorn_row` (a tail, one warp) | `LmHcSinkhornRow` (`hc.cuh`) | `LmHcSinkhornKernel`: the partials summed in warp order, the pre and post weights and the Sinkhorn combination |
| `hc_collapse_norm` | `LmHcCollapseNormRow` (`inference/kernels/fused/collapse_norm.cuh`) | a row's streams collapsed with the pre weights, RMS-normed with the whole row's sum, and one slice of it written in BF16 and through a family's quantizer |

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

## Tests

- `tests/test_fused_kernels.py` (host): every plan regenerates its committed
  files, `--check` fails on a changed generated file and passes after
  regeneration, and each refusal above is made.
- `tests/test_dsv41_flash_fused_exact.py` (sm_121): for 1 to 16 rows, with
  ordinary and edge-valued inputs, every output of the fused kernel is the
  same bytes as the unfused launches write; row 0 is the same at 1 and at 16
  rows; a changed input changes the outputs; and three graph replays, with the
  outputs overwritten before each, give the same bytes and leave the counters
  at zero. It is clean under `compute-sanitizer` memcheck, synccheck and
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
