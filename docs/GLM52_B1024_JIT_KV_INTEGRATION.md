# GLM-5.2 PP13 B1024 JIT-KV integration handoff

This branch is the Sparkdev integration base for the 13-Spark production path. It combines current `upstream/main` batching/prefill work with layer-major MTP execution, an internal stage-local KV directory, NVMe-backed KV records, and fail-closed backend receipts.

It does not claim measured GB10 throughput. Host tests and CUDA structural compilation pass; the SM121 build, numerical gate, and full-ring performance gate must run on the Sparks.

## Production shape

| Contract | Value |
|---|---:|
| Pipeline stages | 13 |
| Layers | table-driven `6 × 13` default |
| GPU dispatch width | 1,024 logical lanes |
| MTP execution capacity | 7,168 rows (`1024 × 7`) |
| Pipeline in-flight request capacity | 13,312 (`13 × 1024`) |
| KV block size | 64 tokens |
| GPU KV pool | 4,194,304 tokens / 65,536 blocks per stage |
| NVMe KV capacity | 1,048,576 blocks per stage |
| Expert execution | stage-local grouped FP8 MoE; no cross-node expert parallel |

The dispatch-width constant and the pipeline-request-capacity constant are intentionally separate. A B1024 kernel is still limited to 1,024 lanes, while the gateway, service request table, request map, and stream backlog can retain thirteen full pipeline cohorts.

## Fail-closed path

The production resident accepts prefill and decode only through `submit_work`:

1. The service builds one flexible-lane work packet per prefill token or decode chunk.
2. The same packet is sent to the local CUDA resident and forwarded to the next rank.
3. The resident validates it against the 1,024-lane / 7,168-row rank plan and queues it.
4. The builder resolves KV blocks through its internal directory, performs any required NVMe store/load batch, and submits asynchronous CUDA work.
5. Sequence-release packets remove both GPU and NVMe directory state.

The service refuses to attach to a resident unless all of these are true:

- all routed layers bind the grouped FlashInfer FP8 MoE backend;
- every FP8 linear plan binds the native scaled-GEMM backend;
- logical and execution capacities exactly match the rank plan;
- B1024 has JIT NVMe enabled with nonzero record, pool, and batch capacities.

There is no Q/KV/O BF16-WMMA fallback in this path. Missing FP8 scaled-GEMM bindings prevent READY state and report `bound/expected` counts.

## Exact KV budget

Build and run:

```sh
make -j build/sparkpipe_glm52_kv_jit_budget
build/sparkpipe_glm52_kv_jit_budget --average-context 4089
```

With MTP enabled, 4,089 existing tokens plus seven reserved positions exactly fill 4,096 tokens per active sequence. For 13,312 requests, the per-node requirements are:

| Rank shape | GPU KV pool | NVMe record | NVMe required |
|---|---:|---:|---:|
| layers 0:6, 3 local DSA sources | 30.094 GiB | 488 KiB | 396.5 GiB |
| 6-layer stage, 2 local DSA sources | 29.063 GiB | 472 KiB | 383.5 GiB |
| 6-layer stage, 1 local DSA source | 28.031 GiB | 456 KiB | 370.5 GiB |
| final 6-layer stage plus MTP | 32.531 GiB | 528 KiB | 429.0 GiB |

The default `--average-context 4096` intentionally fails the GPU-fit gate: adding seven MTP positions crosses into a 65th block, so the current pool fits only 1,008 such lanes. Do not hide this with overcommit. Either cap pre-draft context at 4,089, increase the physical pool after a memory receipt, or reduce the active batch.

## Current NVMe limitation

`BATCHED_COHORT_JIT` is a correct capacity implementation, not yet a throughput result. It uses aligned, batched records and keeps expert weights resident, but a cache miss still reads or writes a complete stage-local 64-token KV record and synchronizes the KV stream at batch flush.

Thirteen round-robin B1024 cohorts can therefore cause severe NVMe churn. The hardware acceptance run must report `kv_nvme_read_bytes`, `kv_nvme_write_bytes`, batch flushes, and synchronous waits per generated token. A large aggregate tok/s claim is invalid without those counters. `ASYNC_SELECTED_JIT` remains a named future mode; this branch does not pretend it is active.

## Sparkdev integration sequence

```sh
git fetch upstream
git switch codex/b1024-jit-kv-integration
make clean
make -j test
make -j build/sparkpipe_glm52_cuda_residentd \
  build/sparkpipe_glm52_cuda_resident_gate \
  build/sparkpipe_glm52_kv_jit_budget \
  glm52_pp13_service_backend
```

On an SM121 Spark, rebuild the CUDA module with the normal production archives. Start every rank with:

```sh
--max-active 1024 \
--kv-nvme-path /fast-local-nvme/kv_rankN.cache \
--kv-nvme-blocks 1048576 \
--kv-nvme-batch-blocks 32
```

Start the attached gateway with the matching logical capacity:

```sh
--kv-logical-blocks 1048576
```

Add `--mtp` for MTP and `--dspark` for DSpark. The packed DSpark backend uses
eight verifier rows per lane and can be enabled together with MTP; no
single-lane verifier fallback is permitted.

After a clean resident start, the log must contain:

```text
logical_lane_capacity=1024
execution_row_capacity=8192
fp8_moe_layers=N/N
fp8_scaled_gemm_plans=N/N
kv_nvme_mode=batched_cohort_jit
state=ready
```

After an idle B1024 MTP verification run, query each rank:

```sh
build/sparkpipe_glm52_cuda_resident_gate \
  --socket /run/sparkpipe/cuda_resident_rankN.sock \
  --rank N --require-work --require-layer-major
```

The gate rejects missing FP8 backends, non-JIT KV, pending NVMe operations, resident queue leftovers, rejected work, asynchronous failures, and layer-major counter mismatches.

## Hardware acceptance receipts

Do not merge a performance claim until all are retained:

1. SM121 compilation from this exact commit.
2. Numerical parity for B1, B128, and B1024 prefill/decode.
3. MTP token/confidence parity and B1024 layer-major verification.
4. All 13 resident gates passing after the same full-ring run.
5. Aggregate tok/s at B128, B256, B512, and B1024.
6. Per-rank stage time, FP8 tensor-core utilization, grouped-MoE expert coverage, NVMe bytes/token, and cohort-switch stall time.
7. Zero fallback, rejection, asynchronous-failure, and work-queue-error counters.

The honest expected result of this handoff is an integration-ready, fail-closed implementation. Its performance remains `NOT_MEASURED` until those Spark receipts exist.
