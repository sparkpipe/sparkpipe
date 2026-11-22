# Decode replay: one rank of the four-rank DeepSeek V4.1 Flash step on one GPU

`build/dsv41_flash_decode_replay` (`tools/dsv41_flash_decode_replay.cu`) runs
the decode step of rank 0 of the four-rank DeepSeek V4.1 Flash stage on a
single GPU, so kernel work on that step can be measured without the fleet.
It is a measurement tool, not a test of the model's numbers.

## What runs

- **The stage's own code.** Every layer runs through the schedule body
  (`D41Schedule<D41Cuda>::Body`) and the stage's hooks, over a model state laid
  out by the stage's own arena plan (`D41Cuda::Plan`) for 16 rows and 16
  sequence slots. A change to the stage's launch sequence is measured as it
  will run.
- **Synthetic weights in the production formats and shapes:** MXFP8 dense
  projections with UE8M0 scales, the BF16 router, the hyper-connection mixers,
  and the routed experts in the MXFP4 cell layout (one pool of 96 experts that
  every layer shares). Values are bounded so that every step stays finite; the
  run checks the streams and the route fault word at the end and prints a
  digest of the final streams, so two builds that must compute the same values
  can be compared.
- **Production initial state.** The arena starts zeroed, as the stage runner
  allocates it, and `D41Cuda::Upload` writes the first layer's collapse
  weights; the streams and the window ring start random.
- **One state per step.** The first kernel of every step copies the same
  starting streams, so every step computes the same values and every run routes
  the same tokens to the same experts. Two builds or two variants therefore
  compare the same work.
- **Rows and context.** `--rows` (1 to 16) decode rows, one per sequence, at
  positions from `--position` (default 160: the conversations of the reference
  trace held a 96-token prompt and up to 128 generated tokens, so the window is
  full and the compressed attention reads what they held). The selection is
  512 fixed compressed positions; positions past a sequence's compressed length
  are skipped, as in the stage.
- **Graph replay.** The 40 layers are captured into one CUDA graph, with the
  weight prefetch plan armed as the stage runner arms it, and replayed
  `--warmup` plus `--steps` times.

## What stands in

- **Collectives.** The stage's round service is replaced by three kernels: a
  copy of the local rows, an arrival that holds for the wait traced for that
  layer and site (attention reduce or MoE gather, `tools/dsv41_flash_decode_holds.txt`)
  while a second warp prefetches the round's weight group exactly as the mesh
  arrival kernel does, and a copy into the result. Other ranks' rows are zero.
- **Layers the replay cannot run alone.** The engram layers (1 and 14), the
  compressed-KV sources (2, 8, 14, 20) and the indexer layers (24, 28, 32, 36)
  need disk rows, peers or a filled index. Each runs as the next layer of the
  same compression ratio, with its own weights. The embedding, the head and the
  stage edges are not run.

## Modes

| mode | what it measures |
|---|---|
| `replay` | the production step |
| `prefetch` | the attention block re-sequenced in the tool (the same kernels and launch helpers as `D41AttentionWhole` on these layers), once as it is and once with a side-stream kernel that prefetches the o_a and o_b weights into L2 while the window fold runs (`--prefetch-mib`, `--prefetch-threads`) |
| `boundary` | 48-link chains of two kernels: a copy and a copy; the FP8 QDQ and an MXFP8 GEMV of the o_b shape; a copy and the GEMV; a copy that releases its dependents first and the GEMV; and the same with a streaming kernel that waits before, or loads before, `griddepcontrol.wait` |

## Running it

```sh
make build/dsv41_flash_decode_replay
python3 -I tools/tp_cupti_trace_build.py --compile --cuda-root /usr/local/cuda --output-directory "$PWD/build/cupti"
CUDA_INJECTION64_PATH="$PWD/build/cupti/libspark_cupti_trace.so" SPARK_CUPTI_TRACE_FILE=/abs/run.cupti \
    build/dsv41_flash_decode_replay --mode replay --rows 8 --holds tools/dsv41_flash_decode_holds.txt > run.out
python3 -I tools/dsv41_flash_decode_replay.py --trace /abs/run.cupti --stdout run.out --json run.json
```

The summary splits the trace into steps at `ReplayMarkKernel`, drops the
warmup steps and takes, for each kernel position, the median duration and the
median gap since the previous kernel ended. It reports six sequences by
kernel names in order (attention output, attention core, attention input,
shared expert, mHC, router) against `tools/dsv41_flash_decode_reference.json`,
the same sequences on rank 0 of the four-node lane, and the sequences of the
fused kernels (`docs/FUSED_KERNELS.md`): the mHC kernel, and the attention
output kernel with the projections that follow it. `tests/test_dsv41_flash_decode_replay.py`
checks the summary on a synthetic trace.

## Measurements (2026-11-22, CUDA flavour, two GB10 hosts, 30 steps after 5)

Sequences, microseconds per instance at the default position, replay (both
hosts) against the four-node trace:

| sequence | 1 row | 8 rows |
|---|---|---|
| attention output (12 kernels) | 112.2-115.4 / 113.7 | 110.7-112.4 / 123.7 |
| attention core (3) | 82.5-82.7 / 78.6 | 99.4-99.7 / 109.7 |
| attention input (5) | 78.0-80.7 / 86.3 | 87.3-88.2 / 100.7 |
| shared expert (5) | 43.1-45.3 / 58.0 | 55.0-56.6 / 65.0 |
| mHC (3) | 21.2-22.0 / 26.1 | 25.3-25.4 / 31.0 |
| router (5) | 24.4-24.8 / 28.1 | 27.2-27.3 / 31.5 |

- Steps of 1,921 kernels take 27.0-27.7 ms (1 row) and 66.2-69.4 ms (8 rows)
  with the traced waits; the 8-row span moves with the routing the starting
  streams give the expert kernels. The two hosts agree within 3%. Against the four-node
  trace most kernels run 7-14% faster here, while the step's boundary structure
  is the same: 0.48 us median between kernels, 1.15 us before every
  dependently launched skinny or cell kernel, 1.2 ms of gaps a step. The tool
  compares two builds or two variants on one host; it does not stand in for
  the fleet's absolute numbers.
- **Prefetch during the window fold** (position 400): prefetching o_a and o_b (16 MiB) cuts
  their GEMVs by 32% (8 rows) and 51-53% (1 row), but the fold, which is bound
  by the latency of its own reads, is slowed by 63-70% while the prefetch
  streams: the step is 0.5-1.2 ms slower at 8 rows and 0.36-0.46 ms faster at
  1 row. With 8 MiB the GEMVs gain 15-26% and the fold loses 31-35%. Eight
  prefetch threads behave as 32.
- **Boundaries:** in the two-kernel chains every gap is 0.00-0.16 us, against
  0.48 and 1.15 us for the same kernels in the step, so most of a boundary's
  cost in the step comes from the surrounding graph rather than from the pair.
  After a 128-block copy the GEMV starts with no gap with or without an early
  release, and 0.13-0.16 us after the FP8 QDQ kernel; loading the first
  weights before the wait shortens a 10.8 MB stream by 0.4-1.3%.
