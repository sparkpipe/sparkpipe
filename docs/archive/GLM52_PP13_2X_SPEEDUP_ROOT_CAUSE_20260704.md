# GLM52 PP13 2x Speedup Root Cause - 2026-07-04

## Short Answer

The 2x+ speedup came from switching the local PP13 path from the old
per-layer routed-chain validator to the exact PP13 six-layer stage-slice path.

The important transition was:

```text
old routed-chain stage:
    one submit/launch chain per layer
    no exact PP13 stage-slice plan
    typical routed stage: 8 submissions for 8 layers

new exact PP13 stage:
    one submit/launch chain for the whole six-layer Spark stage
    built-in exact PP13 AOT dispatch table
    stage-level graph capture/replay
    typical PP13 stage: 1 submission for 6 layers
```

CUDA graph replay helped, but it was not the main 2x cause.  On the final
`72:6` stage, graph replay changed the measured time from about `17.20 ms` to
about `16.56 ms`, roughly a 4 percent improvement.  The large gain came from
changing the execution shape.

## Commits That Matter

The speedup path came from these commits:

```text
ba54092 Use exact PP13 local pipeline gate
3991301 Measure exact PP13 stages with full GPU sync
```

The earlier design target is described in:

```text
docs/GLM52_PP13_EXACT_SLICE_AND_VLLM_KERNEL_WORK_20260701.md
docs/GLM52_PP13_BUILTIN_AOT_STAGE_SLICE_20260702.md
```

The old routed-chain baseline is recorded in:

```text
docs/GLM52_B64_STAGE_SWEEP_20260630.md
```

## Evidence

Current Spark2 exact PP13 timing from merged main:

```text
stage  total_ms  submissions  launch_chains  graph_captures  graph_replays
0:6    20.740    1            1              1               2
6:6    16.491    1            1              1               2
12:6   17.381    1            1              1               2
18:6   17.140    1            1              1               2
24:6   16.969    1            1              1               2
30:6   16.168    1            1              1               2
36:6   16.562    1            1              1               2
42:6   17.091    1            1              1               2
48:6   15.926    1            1              1               2
54:6   17.386    1            1              1               2
60:6   16.474    1            1              1               2
66:6   16.017    1            1              1               2
72:6   17.734    1            1              1               2
```

The slowest current B1 PP13 stage is:

```text
0:6 = 20.740 ms
B1 filled-pipeline ceiling = 48.22 tok/s
```

Old routed-chain evidence in the same Spark2 tree still shows the old shape:

```text
stage   total_ms  submissions  launch_chains  graph_captures  graph_replays
11:8    63.806    8            8              0               0
19:8    63.702    8            8              0               0
27:8    63.945    8            8              0               0
35:8    64.154    8            8              0               0
43:8    63.997    8            8              0               0
51:8    63.613    8            8              0               0
59:8    63.426    8            8              0               0
67:8    62.988    8            8              0               0
```

Normalized per layer:

```text
old routed-chain:
    about 7.9-8.0 ms/layer

new exact PP13:
    about 2.7-3.5 ms/layer
```

That is the 2x+ improvement.

## Graph Replay Ablation

Same Spark2 main checkout, same `72:6` final-stage input hidden:

```text
exact PP13 graph on:
    total_us=16557.305
    total_submissions=1
    launch_chains=1
    graph_captures=1
    graph_replays=2

exact PP13 graph off:
    total_us=17202.295
    total_submissions=1
    launch_chains=2
    graph_captures=0
    graph_replays=0

old routed-chain graph on:
    total_us=20510.144
    total_submissions=6
    launch_chains=6
    graph_captures=6
    graph_replays=6

old routed-chain graph off:
    total_us=22052.480
    total_submissions=6
    launch_chains=6
    graph_captures=0
    graph_replays=0
```

Conclusion:

```text
CUDA graph replay helps a little.
Exact PP13 whole-stage execution is the big win.
```

## What Changed Mechanically

`ba54092` changed `tools/glm52_spark2_local_pipeline_gate.sh` so the local
pipeline no longer drives the old modes:

```text
dense_prefix
hidden
final
```

as routed-chain or package-style per-layer stages.  It drives:

```text
0:6 exact
6:6 exact
...
72:6 exact_final
```

with:

```text
GLM52_EXACT_PP13_STAGE_SLICE=1
GLM52_ENABLE_CUDA_GRAPH_REPLAY=1
```

The validator and CUDA path then use the exact PP13 stage-slice plan:

```text
SparkResidentDecodeStageBackendSubmitStageSlice
SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice
built-in exact PP13 AOT dispatch table
```

This collapses a Spark stage from several independent layer submits into one
stage-slice submit and one launch chain.

## What Did Not Cause The 2x

The later FP8 handoff did not explain the measured 2x jump in this run.  The
current local PP13 gate still uses the NVFP4/B12x pack root:

```text
/home/spark2/sparkpipe_artifacts/glm52_b12x_resident_moe_all_v3
```

The exact final-stage restricted-logit fix also did not create the speedup.  It
made the fast path measurable and accurate by writing the restricted logits that
the fused final epilogue already computes.

