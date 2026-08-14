# GLM-5.2 PP13 exact-slice and vLLM-inspired kernel work, 2026-07-01

The current production layout is the measured fixed PP13 plan from PR #55:

```text
0:6, 6:6, 12:6, 18:6, 24:6, 30:6, 36:6,
42:6, 48:6, 54:6, 60:6, 66:6, 72:6
```

The B64 measured slowest stage is `0:6 = 50.660288 ms`, so the filled-pipeline
ceiling before transport is approximately `64 / 0.050660288 = 1263 tok/s`.
The final stage is no longer modeled with validation/debug timing inflation:
`72:6` measured `46.268929 ms` hidden-only and `46.449792 ms` with final-token
checks enabled.

## Runtime additions

The SM121 resident decode-stage CUDA path now has a PP13 exact six-layer slice
launcher:

- `SparkGlm52Sm121RequiredDecodeStageLaunchExactPp13StageSlice` validates the
  exact `stage_index * 6` layer range, active B16/B32/B64 bucket, final-stage
  position, and device-only hidden handoff between the six resident layer
  contexts.
- The exact launcher captures/replays one CUDA graph for the whole six-layer
  spark stage instead of replaying one graph per layer.
- Stage-slice plans can advertise internal graph counters so the external
  stage-plan wrapper does not double-count graph replays.
- The exact-slice plan exposes two branch streams and three events so raw Q and
  KV projection branches can overlap inside each layer of the slice while the
  main stream waits only at the branch join.

The bulk-prefill path also has a paged chunk prefill plan ABI:

- `SparkResidentDecodeStagePagedPrefillPlan` carries vLLM-style prompt
  positions, slot mapping, context lengths, and block-table pointers.
- `SparkGlm52Sm121RequiredDecodeStageLaunchPagedChunkPrefill` validates the
  paged/chunked plan and launches device kernels for block metadata staging and
  prompt-hidden staging.
- The scheduler keeps the vLLM-inspired token-budget, chunked-prefill, prefix
  cache, and graph-padding metadata from the previous port.

## What this is expected to improve first

The measured validator still showed `graph_captures=6 graph_replays=6` for a
single six-layer stage. The exact PP13 launcher targets that specific overhead
by reducing the stage to one graph replay and one submit path for all six
resident layer contexts. The branch-overlap hooks target the next intra-layer
hot spot: independent Q and KV projection work before the shared attention join.

CUDA hardware measurement on Spark2 is still required to quantify the realized
speedup. The implementation should be validated by checking that a six-layer
stage reports one stage-level capture/replay path instead of six per-layer graph
replays when an exact PP13 stage-slice plan is attached.
