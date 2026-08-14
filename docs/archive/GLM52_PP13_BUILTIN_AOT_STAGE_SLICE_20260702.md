# GLM52 PP13 Built-in Exact Stage AOT Launcher

This pass finishes the exact PP13 six-layer stage execution path as an internal Sparkpipe fast path instead of an external callback placeholder.

The fixed PP13 production layout remains:

```text
0:6, 6:6, 12:6, 18:6, 24:6, 30:6, 36:6,
42:6, 48:6, 54:6, 60:6, 66:6, 72:6
```

## What changed

`SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_BUILTIN_EXACT_PP13_AOT` marks an exact PP13 slice plan that must use the built-in exact-stage AOT dispatch table.

The SM121 required CUDA path now contains a compiled dispatch table for every exact fixed stage and batch bucket:

```text
13 stage indices x B16/B32/B64
```

The selected launcher checks the exact stage index, first layer, six-layer count, batch bucket, active sequence count, and final-token stage, then runs the six layers through an unrolled built-in exact-stage body under the stage-level graph capture/replay path.

Runtime KV block tables remain connected through this path. If a dispatch carries a `SparkKvBlockTableView`, each layer receives the execution-time physical KV block table selected by the scheduler/request API.

## No fake AOT

A plan advertising `SPARK_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_AOT_STAGE_LAUNCH` without either a real external launch function or the built-in exact PP13 AOT capability is now rejected during resident module validation.

`SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PRODUCTION_PP13_CAPABILITIES` now means the built-in exact PP13 AOT path, not a set of required-but-unimplemented fused callbacks.

## Validation added

The resident firmware tests now cover:

```text
exact PP13 production plan with launch_function == NULL is accepted only when the built-in exact AOT capability is present
unbacked AOT exact-stage plan is rejected
```

Local validation:

```text
make -j2 test
```

passed. CUDA hardware validation still needs Spark2 because this container has no `nvcc`.
