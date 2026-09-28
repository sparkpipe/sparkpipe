# GLM real-pack local driver probe

`build/glm5_next_driver_probe` loads the published driver through
`SparkLoadModelDriver` and admits frames with `SparkAdmissionRequestFromFrame`
and `SparkAdmissionEvaluateAndApply`. It runs a TP16 rank-0 FP8 pack on one
GPU with collectives disabled. It is a component smoke test (I38), not a
full-model oracle or a throughput benchmark.

## Lazy mode only

The probe takes `DRIVER TP16_RANK0_PACK resident|lazy 1|3|5 [prefix]`, but
only `lazy` can run. Since `c67be23` (2026-09-09) the module has no direct
pack loader. With `SPARK_WEIGHTD_SOCKET` unset, `SparkWeightdAttachRequested`
returns `BUSY` (`runtime/spark_weightd_attach.c`), `SparkGlm5NextLazyOpen`
maps that to `UNSUPPORTED`, and driver creation fails. The probe itself exits
with status 3 when the mode disagrees with the socket setting.

## Running it

Build and run from a clean, merged-main checkout. `tools/glm5_next_build_release.sh`
builds `DRIVER` and needs a GPU-owned `spark_queue` job (`SPARK_QUEUE_ID`); the
driver lands at `build/glm53_release/compiled/stages/stage_000/model_driver.so`.

```sh
make -j4 CUDA_HOME=/usr/local/cuda build/glm5_next_driver_probe build/sparkpipe_weightd
build/sparkpipe_weightd --socket SOCKET --device-bytes-max POOL_BYTES &
SPARK_GLM5_NEXT_GRAPH_PATH=0 \
SPARK_WEIGHTD_SOCKET=SOCKET \
SPARK_WEIGHTD_PACK_SHA256=PACK_SHA256 \
SPARK_WEIGHTD_EXPERT_POOL_BYTES=POOL_BYTES \
build/glm5_next_driver_probe DRIVER TP16_RANK0_PACK lazy 3
```

`SPARK_GLM5_NEXT_GRAPH_PATH` is mandatory: the module fails with
`INVALID_ARGUMENT` unless it is `0` or `1`. With collectives disabled, neither
graphs nor linear chains are eligible, so the probe always runs the chain
state machine. `SPARK_WEIGHTD_SPINE_BUDGET_BYTES` is optional and defaults to
8 GiB. The other lazy settings are described in
[GLM_LAZY_DRIVER_INTEGRATION.md](GLM_LAZY_DRIVER_INTEGRATION.md).

## What it checks

Without `prefix`, each invocation starts fresh state and submits four
fixed-input positions per sequence: one prefill row, then three decode steps.
B3 and B5 exercise occupancies that are not powers of two. The probe checks
completion identity and token bounds, prints one `TOKEN` line per row and
step, and ends with `PASS local-token-smoke`. Each step waits for the dispatch
slot to be released, not just for the callback to run. A failed or timed-out
invocation exits without freeing GPU state whose ownership is uncertain, so
the owning job must supervise the probe and its daemon and clean up their
cgroup together.

With `prefix`, the probe runs a 63-token prefix plus four continuation steps.
It then restores the prefix into moved pages, joins new lanes into a running
batch and resets. It requires the restored continuation to reproduce the
tokens, the captured state bytes and the selected logit exactly. It prints
`RESTORE`, `JOIN` and `TEMPORAL` receipts and ends with
`PASS local-prefix-reuse`. This is the single-rank form of invariant I27.

Matching argmax tokens are weaker evidence than matching logits, hidden states
or KV and recurrent state. Distributed numerical qualification still needs the
real TP16 or TP4xPP4 collectives.

## The compare tool is broken

`tools/glm5_next_driver_compare.py` runs resident B1, B3 and B5 baselines
first, then lazy B1, then lazy B3 and B5 concurrently, and requires every
lazy token line to match its baseline. Because the resident runs always fail,
the tool raises before writing `RESULT.json`. Until its baseline is replaced,
compare lazy runs by hand, or use the path parity checks in
[GLM_PERFORMANCE_GATES.md](GLM_PERFORMANCE_GATES.md).
