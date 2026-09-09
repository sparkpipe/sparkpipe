# GLM real-pack local driver probe

`build/glm5_next_driver_probe` loads the published driver through
`SparkLoadModelDriver` and uses `SparkAdmissionRequestFromFrame` and
`SparkAdmissionEvaluateAndApply`. It exercises a TP16 rank0 FP8 pack with
collectives explicitly disabled. This is a local token smoke test, not a full
model oracle or a throughput benchmark.

Build and run only in an assigned queue `run` job from a clean, merged-main
checkout, following the queue's normal verified rsync and resource limits:

```sh
make -j4 CUDA_HOME=/usr/local/cuda build/glm5_next_driver_probe
build/glm5_next_driver_probe DRIVER TP16_RANK0_PACK resident 1
build/glm5_next_driver_probe DRIVER TP16_RANK0_PACK resident 3
```

Build `DRIVER` from that same commit with `tools/glm5_next_build_release.sh`.
Its usual path is `build/glm53_release/stages/stage_000/model_driver.so`.
The resident comparison requires `SPARK_WEIGHTD_SOCKET` to be absent. For lazy
execution, pass `lazy` and configure the socket, pack SHA256, expert pool bytes
and spine budget exactly as described in `GLM_LAZY_DRIVER_INTEGRATION.md`.
A lazy invocation with no socket fails before driver creation. The driver's
strict manifest validation and load errors remain authoritative.

Each invocation starts fresh state and submits four fixed-input positions per
sequence: one prefill row per sequence, then three decode steps. Batch size 3
deliberately exercises a non-power-of-two occupancy. Completion identity and
token bounds are checked; the next step waits for dispatch-slot release, not
just callback entry. A failed or timed-out invocation exits without pretending
that uncertain in-flight GPU ownership can be freed safely. The owning queue
must supervise the process and its daemon and clean their cgroup together.

Compare the `TOKEN` lines between resident and lazy runs at each batch size.
Matching local argmax tokens are weaker than matching logits, hidden states or
KV/recurrent state. Full numerical qualification still requires those checks
and actual TP16 / TP4xPP4 collectives. This probe neither claims those results
nor measures serving performance. The host fixture tests only the probe's
control flow, with fake CUDA/driver functions and real shared admission code.

`tools/glm5_next_driver_compare.py` owns the comparison processes and daemon
within a queue GPU job. Pass `--probe`, `--driver`, `--daemon`, `--pack`, a new
`--output` directory, verified `--pack-sha256`, and explicit `--pool-bytes` and
`--spine-bytes`. It records resident B1/B3 baselines, lazy B1, then starts two
lazy B3 processes before waiting for either. All token lines must match the
corresponding baseline. Only complete success writes `RESULT.json`. Logs stay
in the output directory on failure. A shared 12-minute deadline bounds child
waits; owned children are terminated and reaped on errors. Set the queue TTL
to 15 minutes and budget for the resident pack or both lazy spines plus the
shared expert pool and driver workspaces. Starting both processes does not
itself prove simultaneous lease ownership; use the dedicated lazy pair test
for deterministic pin/eviction assertions.
