# Isolated rank timing trace

`tools/tp_cupti_trace_build.py` builds a private adaptation of NVIDIA's installed
`cupti_trace_injection` sample. It does not create a CUDA context or launch work.
The original sample and helper remain unchanged. An unexpected source layout
fails compilation rather than substituting another profiler.

```sh
python3 tools/tp_cupti_trace_build.py --compile --cuda-root /usr/local/cuda --output-directory /tmp/sparkpipe-cupti-rank0-20260922/build
```

The build receipt records the actual compiler command, original/adapted source
hashes and binary hash. On Spark0 this compiled against CUDA 13.0 and CUPTI
`libcupti.so.2025.3.1`. The initial compiled library SHA256 was
`1aef6628a749f2af63b2ad13470f111ba11382f61fb23f2fc3e487ef236e4194`.
This is a build receipt; no injected model run was performed for that receipt.

For an already authorized isolated experiment, set these variables only on the
rank-zero resident process, before CUDA initialization:

```sh
CUDA_INJECTION64_PATH=/tmp/sparkpipe-cupti-rank0-20260922/build/libspark_cupti_trace.so
SPARK_CUPTI_TRACE_FILE=/absolute/unique/rank0.cupti.log
```

The trace file must be an absolute, nonexistent path. An unavailable output file
fails explicitly. Other ranks and weightd do not need this injection. Keep the
resident's existing stdout/stderr so token events and CHAIN-TIME counters can
be compared with the separate trace.

The trace enables concurrent kernels, memory copies, memory sets, runtime/driver
calls and profiler overhead. The installed sample emits kernel names, start/end
timestamps, durations, stream/context IDs and graph/node IDs. Kernel10 and
Memcpy6 activity records retain those graph IDs for captured work, so graph
execution does not collapse into a single launch duration. Memory-operation
waits are not kernel intervals; compare their gaps with the collective's
source/peer-wait counters. Do not add overlapping kernel durations and call that
elapsed token time.

`TRACE_ANCHOR` records PID and nearby CUPTI, monotonic and realtime timestamps
for log alignment. Each returned activity buffer emits `TRACE_BUFFER` including
its byte count and `dropped_records`, even when zero. Any nonzero lost-record
count makes a complete timing decomposition unqualified. Buffer printout is
serialized to preserve each record's multiple lines.

CUPTI's own worker flush period is 1,000 ms; it returns full, completed buffers.
The official sample also flushes at profiler stop, device reset and normal
process exit. After the experiment, use the isolated resident's graceful
shutdown and allow at most ten seconds for normal exit before the experiment's
existing hard cleanup. Require `TRACE_EXIT` and zero dropped counts before
claiming a complete trace. SIGKILL, a crash, a missing exit marker or unfinished
records leave a partial trace; a forced flush does not invent completed GPU
activity. Do not call CUDA/CUPTI from an asynchronous signal handler.

Tracing perturbs timing. Use the trace to attribute time, then measure throughput
with the same model/configuration and injection disabled. No serving daemon or
persistent configuration is changed by the build tool.

## Interpret a completed trace

```sh
python3 tools/tp_cupti_trace_report.py /absolute/unique/rank0.cupti.log --output /absolute/unique/rank0.cupti.json
```

The report merges overlapping GPU intervals before calculating covered time
and gaps. It separates compute kernels, names containing `SparkGlm5NextMesh`,
memcpy and memset, and lists the largest uncovered intervals and kernel names.
Category intervals can overlap; their coverage is not additive. An uncovered
interval is not automatically a network wait: it may also include CPU work,
launch delay or untraced device operations.

For an exact token/request window recorded on that same host, pass
`--clock monotonic --start-ns N --end-ns M` (or `--clock realtime`). The parser
converts the window using `TRACE_ANCHOR` and clips intersecting GPU intervals.
The nearby clock samples have sampling error and do not synchronize different
hosts. A trace lacking final flush, buffer-loss accounting or valid completed
records produces an explicit partial report and exit code 2.
