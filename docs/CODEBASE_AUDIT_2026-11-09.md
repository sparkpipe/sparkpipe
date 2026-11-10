# SparkPipe production codebase audit, 2026-11-09

This audit answers three questions about the production code: whether it is
DRY, where its footguns are, and whether the architecture can reach the
fleet's performance floors while a new LLM driver stays a one-day job. It
follows the performance audit of the same day, which established that TP16
decode is bound by collective rounds times 50-95 us and TP16 wave prefill by
wire bytes per token, and recommended DP4xTP4 replicas for models that fit
four nodes and a latency-floor program for K3 and GLM 5.3 Full.

## Scope and method

Production code is about 184,000 lines of C and CUDA (`node/`, `runtime/`,
`src/`, `scheduler/`, `cache/`, `ring/`, `inference/`, `modules/`,
`model-families/`, `include/`, `text/`, `deployment/`), the serving Python in
`tools/serving/`, the stage pack builders and the build. Tests and the
archived docs were read only as evidence of production behaviour.

Four subsystem maps (resident process, API, batch engine, IPC and adapters)
and three cross-cutting inventories (layer-execution paths, configuration
surface, hot-path synchronizations) were produced by readers and are quoted
here with their `file:line` citations. The highest-impact claims were then
confirmed by reading the cited lines; those carry the mark (confirmed).
The rest of the tree was covered by direct reads of the runner and model
interfaces, the family schedule template, the module files, the transport
launcher list, and by measurements over the tree: a kernel census, a
normalized diff of the 18 module files, a function census of the 15 packers,
and reference counts for the suspected dead subsystems. What was not read in
depth is listed at the end.

## Verdict

**Hardware independence did not slow the served path.** A family schedule
(`GemmaSchedule<G, B>::Step`, `model-families/gemma4/include/sparkpipe/spark_gemma4_family.h:95-190`)
is a compile-time template that calls static backend functions; the common
runner calls `cudaStream*`, `cudaGraph*` and the mesh kernels directly
(`inference/runner/stage_runner.cu`); the only abstraction on the hot path is
the `void *stream` handle in the model interface
(`include/sparkpipe/spark_stage_runner_model.h:134-157`). The device layer
(`runtime/device/`) and the device-neutral runner (`inference/runner/device_*`)
link only into `build/metal_stage_run` (`Makefile:1534-1535`); no serving
adapter links them. Their cost is maintenance, not time: about 19,500 lines
kept in lockstep with the CUDA path and serving no request today.

**The performance is lost in five places that have nothing to do with device
neutrality**, and every one of them lives in common code, so fixing them does
not touch the drivers:

1. The runner observes and the model drives. Each family's `Step` walks its
   own layers and issues its own launches and rounds; the runner wraps the
   whole step in one graph or one eager chain. It therefore cannot split a
   batch into two micro-batches, overlap a collective with compute, fuse
   across operations, or chunk rows: those decisions sit in nine copies of
   model code.
2. Every step and wave is a host transaction. A chain step ends in five to
   nine blocking host round trips; a wave boundary costs 0.5-1.1 ms with all
   sixteen GPUs idle; a non-continuation submission is a two-phase commit
   over sixteen ranks; a publish frame blocks the wave while 9-28 MB of
   recurrent state is written.
3. Decode chains collapse with batch width. The chain is
   `max_output_token_count / lane_count` capped at 8
   (`runtime/model_batch_engine.c:3375-3379`, confirmed): 8 steps at one
   lane, 2 at eight lanes, 1 at sixteen, and 1 whenever a prefill is queued.
   The per-frame host cost is paid every token at the served batch widths.
4. The collective plan is per model, not per schedule: 2 rounds a layer on
   GLM Flash, 3 on GLM Full, 4-5 on DeepSeek V4.1 Flash, 5.6 on K3, each
   written into the family's step.
5. The KV-sharding rule written for K3 is applied to a model whose whole 1M
   KV is 0.9 GB (DeepSeek V4.1 Flash, 890 B per token).

**It is DRY where it matters least and duplicated where it costs most.** The
math is shared (schedules, `inference/kernels/`), but the execution
machinery is not: seven layer-execution paths of which two serve, four row
splitters, nine CUDA backends each carrying 300-400 lines of model-interface
plumbing against one 683-line shared base on Metal, hyper-connections
written four times, 15 packers sharing seven function names, 18 module files
that are 56-68% identical text, two stage-config parsers of which the one in
checked-in deployments is dead, and about 10,000 lines of production-linked
code with no production caller.

**The footguns are concentrated in configuration and in the host protocol.**
The worst: an absent `wait_mode` silently selects spin waits and disables
graphs; the API refuses Gemma 4 token ids above 260000; dropping one
hand-copied capability bit silently turns decode into one-token frames.

## Findings

Severity: high means at least 10% of a step, a correctness or data-loss risk
on the served path, or a structural blocker to the recommended direction;
medium is measurable but bounded; low is hygiene.

### 1. Execution architecture: the runner observes, the model drives

| Sev | Finding | Where |
| --- | --- | --- |
| high | Seven paths run a model layer; two serve traffic. The CUDA stage runner (4,659 lines plus the 2,785-line collective and the 2,389-line mesh kernels) serves all 18 lanes; the device-neutral runner (6,641 lines), nine family device stages (5,321), `runtime/device` (3,466), the Metal kernels (2,801) and the pass engine (1,295, K3 only) are the other five. The runner only wraps the family's `Step`, so micro-batching, fusion and overlap cannot be added once. | `inference/runner/stage_runner.cu:1834`, `Makefile:1429-1535` |
| high | Row splitting exists four times with four thresholds and none serves decode: `StageRunnerLanesSplit` at 256 rows for one-sequence prefill (`stage_runner.cu:1737-1750`), the MoE families refuse below 1,024 (`glm52_stage.cuh:57`, `glm5_next_stage_model.cuh:376`, `qwen38_hybrid_stage.cuh:458`), GLM chunks again inside `NextWave` (`glm52_stage.cuh:2325-2348`), DSV4.1 through tail rows, K3 through `SlicePair`. Five `Lane()` clones of 25-35 lines copy the whole `State`. | `inference/runner/stage_runner.cu:1737` |
| medium | Nine CUDA backends each carry their own `AttachKv`, `RecurrentCopy`, `ResetSlot`, `Sideband`, `ProbeBuffers`, `Report`, `Interface`, `Open`, `Bind`, `Lane` and a `Round()` wrapper: about 4,000 lines against 683 shared on Metal (`inference/runner/device_stage_model.c`). A new CUDA family writes 300-400 lines of this before any math. | `model-families/gemma4/source/gemma4_stage.cuh:1036-1131`, `mimo26_stage.cuh:718-788` |
| medium | DeepSeek V4.1 Flash's backend, pack reader and kernels live in its module (5,169 lines), where schedule families keep a 160-260-line module; hyper-connection kernels exist four times on CUDA and once on Metal. A second DeepSeek-lineage model would copy 2,249 lines. | `modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_model.cu:1107` |
| medium | The pass engine is model-neutral in name only: `LmPassRun<Model>` drives hooks that only `K3Streamed` implements, with its own row collectives beside the runner's round service and seven stream drains per pass. GLM Full, GLM Flash and DSV4.1 stay on wire-bound TP16 waves because a second 1,400-line backend per family is the price of joining it. | `inference/runner/pass_prefill.cuh:1162-1290` |
| medium | Taps and lanes exclude each other through a vtable slot: `SparkStageRunnerSetTaps` refuses when the model exports `lane` (confirmed). A.X K2 has a DSpark drafter but its GLM backend exports `Lane`, so no tap stream can be configured for it, while GLM's lanes never fire. | `inference/runner/stage_runner.cu:2587-2591` |
| low | K3 keeps a second, unserved TP4xPP4 adapter build, its own launcher (`tools/k3_lane.sh`, the one `k3_lane` entry in `deployment/lanes.json` against 18 `stage_lane`), and a compile flag that decides whether verify and taps exist. | `Makefile:810,1600-1610`, `spark_k3_serving_adapter.c:13-56` |

### 2. The host protocol around every step and wave

| Sev | Finding | Where |
| --- | --- | --- |
| high | A chain step ends in 5 (greedy) to 9 (sampled) blocking host round trips on residentd's thread: `StageRunnerCopy` is `cudaMemcpyAsync` followed by `cudaStreamSynchronize` into pageable `new[]` buffers (confirmed), four calls per step after `GraphSettle` already synchronized, plus `HardwareStats`' two synchronous `cudaMemcpy` for a report printed every 64 steps. 110-300 us per step with the GPU idle, and the reason a K-step chain is K graph launches. | `inference/runner/stage_runner.cu:478-486, 2115-2118, 2018-2030`; `ring/transport/tp_device_collective.c:2411-2413, 2719-2724` |
| high | Every non-chained wave pays two weightd IPC round trips, a rank-0 broadcast that fifteen ranks spin on, the token-upload drain and the adapter's own sync: about 0.5-1.1 ms per `SparkStageRunnerSubmit` on all sixteen ranks with the GPUs idle, paid by every prefill, verify and publish wave. | `ring/transport/tp_device_collective.c:577-835`; `inference/runner/stage_runner.cu:2355-2376` |
| high | Decode chains collapse to one token per IPC round at the served batch (confirmed): `chain_tokens = max_output_token_count / lane_count`, capped at `SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE` (8, `include/sparkpipe/spark_model_driver.h:15`); 17 modules declare 16. At 16 lanes every token pays the full control plane: two network hops and about eight host hops. Any waiting prefill also forces one-step chains (`SparkModelBatchPrefillWaiting`), and the block-end clamp shortens them further (mean 5.4 at eight lanes). | `runtime/model_batch_engine.c:3371-3401` |
| high | The whole frame runs synchronously on residentd's IPC thread, so step N+1 cannot be read, prepared or committed during step N's GPU time; decode queues behind a prefill chunk's whole GPU time on one strict FIFO. Required for collective alignment, but it is why no two submissions can overlap. | `node/model_residentd.c:2787-2839, 1033-1057` |
| medium | Two-phase commit for every non-continuation submission: two client round trips to sixteen ranks before the GPU starts, a third for completion. The fan-out is serial on one thread, 10-20 us per rank, so the last rank sees the step 160-320 us after the first; that host skew feeds the collective skew the performance audit measured. | `runtime/model_pipeline_client.c:399-432, 903-918` |
| medium | A publish frame blocks the wave on the KV completion thread while it writes 9.3 MB (GLM 5 Next at TP16) to 28 MB (K3) of recurrent state per publishing lane, every 64-token block. `SparkStageKvBindingFinishAsync` exists and is unused. | `inference/runner/stage_serving_adapter.c:1286-1301`; `runtime/stage_kv_binding.c:2482-2512, 2643-2690` |
| medium | A sampled step runs the head twice (the FP8 certified copy for the argmax, then the BF16 slice for the distribution) and all-gathers the f32 logits of every row to every rank (confirmed), about 620 KB per row for GLM at the mesh's measured ~5 GB/s. | `inference/runner/stage_runner.cu:1218-1272` |
| medium | Pageable host mirrors make every per-wave input copy a synchronous staged copy (six per wave) and the runner drains the stream after the token upload before the first kernel. `SPARK_MEMORY_SPACE_HOST_PINNED` already exists in `runtime/memory_buffer.c:31-36`. | `inference/runner/stage_serving_adapter.c:1386-1395, 977-989`; `stage_runner.cu:2374-2376` |
| medium | Validation inflation: a continue step runs 49 `Prevalidated` calls, 98 lane/row passes and 1.7 MB of memset on the coordinator, then each residentd decodes twice and validates three times. | `runtime/model_pipeline_client.c:892-906`; `node/model_residentd.c:2180-2190` |
| medium | 37.5 KB completions copied by value and 168-byte lanes resent on every pure decode continuation: 0.6 MB zeroed and 75 KB copied per token at TP16 on the coordinator. | `include/sparkpipe/spark_model_serving_adapter.h:199-219, 299-302` |
| medium | The engine's dispatch thread does 15-20 scans over 128 request slots per tick, re-runs every queued prompt's prefix lookup (one SHA-256 per block) on every publication epoch, and calls the hub drafter synchronously while no decode submission is in flight. | `runtime/model_batch_engine.c:2283-2338, 3107-3199, 4005-4081` |
| low | Prefill-wave diagnostics are always on (one event per layer and two per round, then 565 elapsed-time queries per GLM Full wave on residentd's thread), and `glm52` drains the stream at step end. `STAGE-STEP-TIMING`, `SUBMIT-ARRIVED` and `DECISION` write unbuffered stderr per submission per rank. | `inference/runner/stage_runner.cu:2414-2415, 1356-1419`; `glm52_stage.cuh:2350-2355`; `node/model_residentd.c:2178, 2427` |

### 3. Collectives and the KV sharding rule

The performance audit covers the arithmetic; the code adds three points.

| Sev | Finding | Where |
| --- | --- | --- |
| high | The round plan is model code. Each family calls `B::AllReduce` (or its own `Round`) where it likes: GLM Flash twice a layer, GLM Full three times, DSV4.1 Flash four to five (query all-gather, fp32 partial all-to-all to head owners, pair gather, MoE all-to-all, all-gather), K3 5.6. One hardware round is at least a publish kernel, an arrival wait and a combine kernel around the CPU doorbell hop (`SparkTpMeshHardwarePublish*`, `SparkTpMeshHardwareArrival*`, `SparkTpMeshHardwareCombine*`), so the round count is also a kernel count. | `model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh:1037-1289, 1700-2366` |
| high | DeepSeek V4.1 Flash context-shards an 890 B per token KV (`model_contracts/dsv41_flash_authoritative.json`, `kv_cache.per_token_global_bytes`). A 1M context is 0.9 GB fleet-wide; the sharding buys nothing and forces the partial exchange that costs about 5 MB per token per rank in prefill and two to three rounds a layer in decode. | `modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_model.cu:762-831` |
| medium | `algorithm_mask` is hard-coded to `TREE` in the live parser and the adaptive-fabric fields of the checked-in stage configs (`algorithms`, `direct_*`) are read by nothing; the regime table in README is implemented as wait-mode branches (TECHDEBT, I36). | `inference/runner/stage_serving_adapter.c:262` |

### 4. Configuration surface and silent modes

| Sev | Finding | Where |
| --- | --- | --- |
| high | `wait_mode` is optional and an absent member means spin (confirmed): a stream drain per round (92 to 521 per token), host spins on fifteen peer words, and `StageRunnerGraphEligible` refuses graphs because `SparkTpDeviceCollectiveDeferred` is 0, so every step runs eager with 1,400-2,000 launches. Nothing fails; the rank only logs its path. The 16 checked-in Laguna stage configs have no `wait_mode`. | `inference/runner/stage_serving_adapter.c:269-273`; `stage_runner.cu:850, 1866-1872` |
| medium | Two stage-config schemas. `runtime/serving_adapter_template.c` (635 lines, `tp_collective`, exact-member validation) has no production caller; the live runner reads `device_collective` with fallbacks and zero exact-member checks, so a misspelt `max_sequences`, `max_rows`, `kv_pages` or `wait_mode` is silently replaced by a default. `deployment/glm5_next_tp16/config/stage_00.json:11` and the Laguna configs carry `tp_collective`, which the live parser refuses at configure; five generators and two docs still emit or describe it. | `inference/runner/stage_serving_adapter.c:126-133, 683-771`; `runtime/serving_adapter_template.c:498` |
| medium | 28 environment switches (5 dead, 10 undocumented, none in README or SPEC). `SPARK_TP_WAIT_MODE` has no reader but the fleet drop-in sets it and twelve docs describe it. weightd applies three precedence rules to its flag/environment pairs and the agent forwards every ambient `SPARK_WEIGHTD_*`. The expert pool and spine budgets, resource declarations under I04, arrive only through the environment next to the JSON that could carry them. | `node/weightd.c:459-474`; `tools/fleet_node_agent.sh:942-943`; `inference/runner/stage_serving_adapter.c:151` |
| medium | Runner-owned capability bits (`PARALLEL_FANOUT`, `CONTINUE_LEASE`, `RESIDENT_DECODE_CHAIN`, `PREFIX_REUSE`, `CACHE_PUBLISH`) are hand-copied in all 18 modules; only the two cache bits are refused when absent. A module that drops `RESIDENT_DECODE_CHAIN` loads and serves one-token frames; one that drops `CONTINUE_LEASE` runs two-phase admission every step. | `runtime/model_serving_adapter.c:207-218`; `runtime/model_batch_engine.c:3371-3373`; `runtime/model_pipeline_client.c:238-241` |
| low | Layer-dump environment switches change execution shape in release builds (graphs off, lanes off for the rank's life) and exist in three implementations under four names. | `spark_dsv41_flash_stage_model.cu:1643`; `qwen38_hybrid_stage.cuh:1182-1190`; `inference/runner/device_stage_model.c:517` |
| low | `/v1/models` answers the environment's `SPARK_MODEL_ID` or `"sparkpipe-model"` for every model; `SPARK_BATCH_INFLIGHT_BUDGET_NS` sets the request expiry outside the deployment; `SPARK_WEIGHTD_SOCKET` is read at eight sites and one defaults silently to `/tmp/spark_weightd.sock`. | `node/model_api.c:1854, 2088`; `ring/transport/rdma.cu:29` |

### 5. Correctness footguns on the served path

| Sev | Finding | Where |
| --- | --- | --- |
| high | The API refuses any token id above 260000 in `prompt_token_ids` and stop ids (confirmed). Gemma 4's vocabulary is 262,144 (`model_contracts/gemma4_31b_authoritative.json:23`), so ids 260001-262143 are rejected as `prompt_token_ids required` or `invalid_stop_token`; the chat layer sends token ids, so Gemma prompts containing them fail with 400. A model-specific literal in model-neutral code. | `node/model_api.c:962` |
| medium | EOS and request-stop accounting diverge: the engine flags both as `STOP_TOKEN`, the API suppresses only request stops, so a model EOS is appended to `tokens` and counted in usage. `stop_token_ids` on a `messages` request replaces the template's turn stops. | `node/model_api.c:330-348, 1617-1621`; `runtime/model_batch_engine.c:957-959` |
| medium | The IPC cannot encode `SPARK_STATUS_NO_LANE` or `EVICT_DENIED` (appended after `UNSUPPORTED`); the mapping helpers that would translate them run only in tests. | `runtime/model_resident_ipc.c:245-427`; `include/sparkpipe/spark_model_serving_adapter.h:111-123` |
| medium | One malformed or duplicate message resets the whole fleet session; a second result for a known submission closes admission with `SCHEMA_ERROR`; `Submit` returns OK after a partial fan-out when rank k fails with `IO_ERROR`. | `runtime/model_resident_client.c:895-969`; `runtime/model_batch_engine.c:1559-1567`; `runtime/model_pipeline_client.c:937-947` |
| medium | Expired requests keep their resident slot and page reservation until the stalled submission completes; `ReopenAdmission` reopens an engine that failed with `IO_ERROR` on the next request with no operator action. | `runtime/model_batch_engine.c:3803-3814, 4092-4105` |
| low | The 512-byte hello buffer is 8 bytes larger than the 504-byte wire struct with no static assert; listen backlog is 1; shutdown timers that must agree live in three files; the IPC message cap is 10-18% above the worst-case submission and trips as a runtime `CAPACITY_EXCEEDED`. | `node/model_residentd.c:260, 570, 277-288, 751-776` |
| low | Six JSON escapers with different escape sets, two UTF-8 validators, eight read/write-all loops, eleven monotonic clock helpers, three stop-token scans per token. | `node/model_api.c:974-1049`; `runtime/net.h` |

### 6. Duplicated and dead code

Measured over the tree on 2026-11-09.

| What | Size | Evidence |
| --- | ---: | --- |
| The 18 module serving files are 56-68% identical text after normalizing family names (53 normalized lines appear in at least 12 of 17 files). Each declares the same nine capability bits, the same descriptor shape and the same `SparkStageServingModel` wiring; the model facts are about 25 lines. | 17 x 95-108 lines | normalized diff of `modules/*/source/*_stage_serving.c` |
| 15 packers share seven function names (`main`, `build_inventory`, `f32_to_bf16_u16`, `convert`, `to_bytes`, `fp8_e4m3_lut`, `kind_shape`); 11 import the 378-line `tools/spark_pack_common.py`. The safetensors walk, codec conversion, TP slicing, scale planes, manifest and hashing are written per family. | 12,228 lines | function census of `tools/*_stagepack.py`, `tools/k3_pack.py` |
| `spark_lm_kernels.cuh` is compiled into every adapter for three certified-head functions and their five kernels; the other ~5,200 lines have includers only in tools, tests and probes, and present a second attention and GEMM library beside `inference/kernels/`. | 5,462 lines x 20 builds | `model-families/common/include/sparkpipe/spark_lm_kernels.cuh:4727-4822`; `inference/runner/stage_head.cuh:9` |
| `runtime/stage_module_common.c`: 50 of 57 exports have no production caller (the load pipeline, stage timing, `CudaWait*`, `CudaFork*`, the pack-arena attach). The live 230 lines are the ledger, the atomic claims and `CudaStatus`. The header includes `cuda_runtime.h` into every module's collective path and is the largest device-neutrality debt the gate records. | 2,138 lines | `runtime/stage_module_common.c:21-180, 329-468, 921-1088, 1121-1475, 1929-2091` |
| The transaction ledger (`runtime/runtime_completion.c`, `work_transaction.c`, `model_runtime.c`) is linked into every binary and used by nothing in production; the pipeline client re-implements two-phase commit without it. | 1,878 lines | `sources.mk:37-38, 80`; `Makefile:1684, 1708` |
| `runtime/serving_adapter_template.c` is production-dead end to end except one macro. | 635 lines | `include/sparkpipe/spark_serving_adapter_template.h:21-30` |
| `scheduler/continuous_batch.c` (a second admission scheduler) links only into the gate binary; `cache/nvme_tier.c` and `cache/store/*` are reached only from the unwired `scheduler/topology_switch.c`; `runtime/spark_graph_reloc*.c` has no caller; the K3 range kernels are launched by `K3MlaDecodeRange`, which nothing calls; `node/memlink_tool.c` (1,664 lines, three binaries) is never spawned. | about 5,600 lines | reference counts over `node runtime cache scheduler inference` |
| The NCCL enum and parsers, the one-phase `SUBMIT` message kind and its server branch, the `gpudirect-rdma` and `host-staged` transport modes, `SPARK_BATCH_BUCKET`, the speculation tree verifier and drafter mix: built, unreachable from any lane. | | TECHDEBT and `node/model_residentd.c:2186-2237, 503-515` |
| Kernel census: 442 `__global__` definitions, 395 names, 70 in `model-families/common`, 60 in the DSV4.1 module, 56 in `inference/llms`. Operation families with many spellings: head 35, gather 18, hyper-connections 17, RMS norm 13, latent attention 12, top-k 11. | | regex census over `inference/`, `model-families/`, `modules/` |
| The hidden transport (`ring/transport/hidden_transport.c`, `host_staged_tcp.c`, `rdma.cu`) is the PP stage-boundary path, loaded by residentd only for `HYBRID_TP_PP` deployments; no lane uses it today. Keep for TP4xPP4; it is not dead. | 2,505 lines | `node/model_residentd.c:1496-1517` |

### 7. The driver path today

A new model on an existing family writes: a contract JSON; a geometry struct
in its module header; one `*_stage_model.cu` instantiation (32 lines for GLM
5.3 Full); a ~100-line serving file; a lane record and stage config; a
packer; a reference tool. That is the one-day path and it holds for the
seven schedule families (Gemma, Qwen hybrid, MiMo, Ling, HY4, GLM Flash, GLM
Full) and the models that ride them (K-EXAONE, Laguna, Muse Glimmer, Solar,
A.X K2). It does not hold for a new family: that costs a schedule header plus
a CUDA backend of 1,000-3,000 lines, of which 300-400 are the plumbing in
finding 1.3, plus the HC or attention kernels the shared library lacks. K3
and DSV4.1 Flash each took the long path and stayed off the schedule.

What a driver author can get wrong without an error: dropping a runner-owned
capability bit (4.4), omitting `wait_mode` (4.1), a misspelt stage-config key
(4.2), leaving `kv_pages_per_sequence` at 0 (fails at `adapter_limits` with a
message naming the symptom), exporting `lane` on a model that needs taps
(1.6), a vocabulary above 260000 (5.1), a page size that makes chains publish
every few steps (2.3).

## Target architecture: performance without losing the one-day driver

### Principle

Keep the three-part family: geometry, schedule, operations. Change who runs
the schedule. Today the schedule is code the family executes; make it data
the runner executes. Then every performance mechanism (micro-batches,
overlap, fusion, chunking, the round plan, K-step graphs, placement) is
written once in `inference/runner/` and every family gets it on the day it
lands, including the eleven that exist. The device-neutral and model-neutral
rules stay as they are; they were never the cost.

### 1. The runner drives the schedule

The family exports a per-layer operation table (the body of today's `Step`
loop without the loop: `B::RmsNorm`, `B::Attention`, `B::AllReduce`, ...) and
a `chunk_rows(step)` query for its minimum chunk and row regimes. The runner
iterates layers, inserts layer marks, lane and micro-batch fences, expert
leases and tap copies, and owns row chunking. The nine `Step` loops of 40-110
lines lose the loop and keep their bodies; the five `Lane()` clones and the
four row splitters go. This is the one change that makes two-stream
micro-batch overlap a runner feature instead of nine rewrites, and it is the
prerequisite for 3 and 4.

### 2. One host turn per chain, none per wave

Runner, collective, adapter and residentd only; drivers keep `step`,
`lane` and `recurrent_copy`:

- capture a chain of K steps as one graph (`StageRunnerChainAdvanceKernel`
  already feeds tokens on the device) with the head writing each step's
  token and score into a pinned ring; settle once per chain;
- keep mesh activity and the chain epoch for the lane's session and derive
  the chain key from the device round counter, so a wave needs no IPC, no
  broadcast and no spin;
- read `HardwareStats` from `published_host_cell`; pinned host mirrors; no
  drain after the token upload;
- complete routes from the KV completion callback (`FinishAsync`) so
  residentd serves its socket during the frame, and let step N+1 be prepared
  while step N runs;
- raise `max_output_token_count` with lanes (or make the chain length a
  runner decision) so chains stay at 8 steps at 16 lanes, and stop cutting
  chains to one step when a prefill waits.

Expected: the 0.5-1.1 ms wave boundary and the 110-300 us per step go to one
turn per chain; at eight to sixteen lanes this is worth more than any kernel.

### 3. The collective plan comes from the schedule

With the runner holding the operation table it can compile the round plan:
two rounds a layer for every family (K3's output-sharded W1 and folded norm,
GLM Full's query folded into the index gather, DSV4.1 Flash's attention
head-sharded over replicated KV), the publish fused into the producing
kernel's last block and the wait into the consumer's prologue through the
PDL primitives already in `spark_tp_mesh_kernels.cuh`, the algorithm chosen
from degree, operation, payload and logical batch (I36) rather than the wait
mode, and the second micro-batch's rounds interleaved with the first's
compute. Fusion across operations (norm into GEMV, QDQ into epilogues,
eventually one persistent kernel per layer) is generated from the same table.

### 4. Shared CUDA stage-model plumbing

Add a model-neutral `inference/runner/stage_model_cuda.cuh` mirroring
`device_stage_model.c`: named-pack bind over a geometry's tensor list
(`SparkNamedPackBindSized` exists), KV attach over declared pools, recurrent
copy and reset over declared state, lane state as a shallow clone, the
interface table. A CUDA family then writes its operations and nothing else,
as a Metal family does today. Move `D41Cuda` to
`model-families/dsv41_flash/source/` and the four HC kernel sets to
`inference/kernels/hc.cuh`; move K3's device code under `model-families/k3/`
as TECHDEBT already asks.

### 5. Streamed prefill as a schedule backend

Make the pass engine a backend `B` whose per-layer operation binds weights
from the pass's landing blocks and whose `AllReduce` is the pass row
exchange. GLM Full, GLM Flash and DSV4.1 Flash then leave the wire-bound
TP16 waves with a ~100-line binding hook each instead of a 1,400-line
backend; K3Streamed's 1,431 lines collapse into the K3 schedule.

### 6. Placement as deployment data

DP4xTP4 needs a replica set above the pipeline client and a deployment
schema that no longer assumes `node_count == stage_count`
(`runtime/model_resident_deployment.c:952`); the API already holds exactly
one engine (`node/model_api.c:112-113`), so a replica is one engine process
behind a prefix-affinity router in the chat layer. The TP degree is already a
template argument of the geometry; `kv_shard` becomes a per-model declaration
(replicated for small-KV models) rather than a rule.

### 7. One configuration schema

Every per-rank input lives in the stage or deployment JSON with exact-member
validation; `wait_mode` required; the spin regime removed from the served
path or kept only behind an explicit value; the environment carries the
residentd-to-adapter handoff (socket, pack digest) and nothing else;
diagnostics compile under `DEBUG`. Runner-owned capability bits are set by
`SparkStageServingAdapterInterface()`; modules declare model facts only.

### 8. Deletions

`serving_adapter_template.c`, the dead 1,900 lines of
`stage_module_common.c` (keep the seven live symbols in a device-neutral
unit), the transaction ledger, `continuous_batch.c`, the NVMe tier and store
client with the topology switch, graph relocation (or wire it), the range
kernels, the NCCL enum and parsers, the one-phase `SUBMIT` kind, the unused
transport modes, `memlink_tool`, the K3 TP4xPP4 build and launcher until the
PP runtime exists, the dead layer-dump switches, and `spark_lm_kernels.cuh`
down to the certified head. About 20,000 lines, none of it serving.

## Order of work and expected effect

| Step | What | Buys | Driver impact |
| --- | --- | --- | --- |
| 1 | Configuration schema (7) and the correctness fixes (5.1, 2.3's chain cap, 4.4's bits) | Removes the silent-spin cliff, the Gemma 400s, the 1-token chains at batch | None; modules shrink |
| 2 | One host turn per chain (2) | 0.5-1.1 ms per wave boundary and 110-300 us per step; chains of 8 at 16 lanes | None |
| 3 | Runner drives the schedule (1) and shared CUDA plumbing (4) | Two-stream micro-batches, row chunking, fusion points once; 4,000 lines of plumbing gone | Each family's `Step` body becomes a table; one mechanical pass per family, validated by the existing byte-identical gates |
| 4 | Collective plan from the schedule (3) | Rounds per layer to 2 everywhere; publish and wait fused; K3 from 521 to 186 rounds | None after step 3 |
| 5 | Streamed prefill as a backend (5) | GLM Full, GLM Flash, DSV4.1 off the wire-bound waves | ~100 lines per family |
| 6 | Placement as data (6) | DP4xTP4 for the models that fit four nodes | None |
| 7 | Deletions (8) | 20,000 lines; a build that compiles only what serves | None |

Steps 1 and 2 are independent of the architecture change and can start now.

## What a new driver writes afterwards

A model of an existing family: contract JSON, geometry struct, one
instantiation, a module of about 40 lines (identity and model facts), a lane
record, a packer emitter on the common packer. A new family: a schedule header
(layout plus the per-layer operation table, 150-250 lines) and only the
operations whose math is new; the plumbing, chunking, overlap, fusion, round
plan, graphs and placement come from the runner. Both stay inside one day,
and both get every performance mechanism the runner has.

## What landed and what remains

Landed in the pull request that carries this page:

- Step 1 in full. A stage config carries only the members the adapter reads;
  `wait_mode` is required; `tp_rank` is checked; the API bounds token ids by
  the tokenizer's vocabulary and counts no stop token as output; a chat
  request's `stop_token_ids` add to the template's turn stops; modules declare
  model facts and the adapter builds the descriptor with the runner-owned
  capability bits; the chain capacity is 128 lanes times 8 steps for every
  model; a chain built while a prefill waits runs two steps, the draft
  planner's bound, instead of one.
- Step 2's runner half. A decode chain is one host turn: pinned mirrors,
  device-side step offsets and distribution positions, the sampled
  distribution inside the step's graph, one settle, one deferred-round
  verification and one timing read per chain, a pinned token upload without a
  drain, and pinned adapter mirrors with a stream-ordered token readback.
- Step 3's first half, in the pull request stacked on the first. Every
  schedule family (Gemma, Hy4, MiMo, Ling, Motif, GLM 5.3 Flash, GLM 5.3
  Full, Qwen hybrid) and DeepSeek V4.1 Flash export `begin`, `layer` and
  `end` (GLM 5.3 Full also `wave` and `wave_end` for its row waves) through
  the model interface, and the common runner drives the layer loop for the
  whole step and for each prefill lane; a model's `Step` remains only as the
  composition the Metal device runner still calls. K3 stays on `step` until
  its dispatch engine is split the same way. The two prefill lanes are two
  frames the runner drives layer by layer; the second runs on its worker
  thread (a one-thread interleave was tried and cost 12% of prefill on the
  fleet, see below). A model declares
  its minimum rows per frame (`chunk_rows`) and the runner's split honours it
  before any lane is set up, and the MoE families' own row guards are gone.
  The engine now builds a prefill wave's rows sequence by sequence instead of
  interleaving the lanes token by token, so a wave's rows are contiguous per
  sequence; the runner still splits only a one-sequence wave, and the
  sequence-boundary split of a multi-sequence wave is the next piece.
- Step 7 in part: the `tp_collective` parser and template, the transaction
  ledger, the external KV provider client, the NVMe tier and topology switch,
  graph relocation, memlink and its tools, the NCCL enum and parser branch,
  and the K3 TP4xPP4 adapter variant are deleted.

- Step 8 in full, on 2026-11-10: `runtime/stage_module_common.c` is deleted
  (one of its 45 exports had a caller, the CUDA status helper of the mesh
  round bench, which the bench now owns; the slot and index-set claims keep
  their own test); `spark_lm_kernels.cuh` is the certified head's closure,
  599 lines of 5,192, and the probes that included the rest are gone; the
  one-phase `SUBMIT` message, its encoder, guard and residentd branch are
  gone (IPC ABI 26); `host-rdma` is the one transport mode, with the
  gpudirect variant and the host-staged TCP module deleted; the K3 range
  kernels and their context-split plan are deleted; the batch tool's second
  admission scheduler (`scheduler/continuous_batch.c`) is deleted; the five
  generators that wrote the `tp_collective` schema, the TP4xPP4 tools and
  the MiniMax deployment are deleted; `SPARK_BATCH_BUCKET` left the build;
  K3 builds for one topology. About 12,900 lines in this pass.
- Step 7's diagnostics: the three layer-dump switches and the collective's
  round trace are read only in `DEBUG` builds, so a release build's
  execution shape does not depend on the environment.
- Step 7's inputs and finding 4.3, on 2026-11-10: a lane's mesh lane, rank
  map, expert pool cap and read-only arena share are stage config members
  (`device_collective.mesh_lane` and `mesh_ranks` required, every member
  parsed strictly); the serving process reads only residentd's own socket
  and digest handoff, which the adapter passes on in configurations; the
  spine budget input, the `SPARK_WEIGHTD_ATTACH` switch and the RDMA
  transport's silent socket default are gone; weightd takes flags first
  under one rule and the fleet agent passes it explicit flags pinned to
  `deployment/lanes.json` instead of forwarding the node's environment;
  `docs/ENVIRONMENT.md` lists every variable still read. Twenty launchers
  that existed to set the retired variables are deleted. A Qwen3.8-27B
  test lane booted from the stage config alone and matched the previous
  head token for token and logprob for logprob on every path of six of the
  seven release prompts (the seventh ran on a different source file); the
  K3 lane's start command changed the same way and waits for a production
  window.

A first fleet gate ran on 2026-11-10 on a Qwen3.8-27B test lane
(`qualification/fleet-gate-20261110/README.md`): this branch before the Metal
line is bit-identical to `main` on the release kit's six prompts, the head
keeps every token with the Metal line's publisher numerics in the logprobs,
eight streams gain 13%, and prefill had lost about 12% to the two lanes
interleaved on one host thread, which serialised one lane's collective rounds
with the other's enqueues. The second lane is back on its worker thread
(prefill returns to `main`'s time, outputs unchanged); the one-thread
interleave needs rounds that do not block the host first, which is step 2's
collective half. The gate also found that the release kit did not ship the
device library every adapter now loads (fixed).

The second fleet run, Gemma 4 26B-A4B at 16 sequences, is in the same receipt: the work before the Metal line is bit-identical to `main` there too, sixteen streams gain 24%, and the Metal line's numerics leave one open defect (a short prompt's first token depends on its batch) that blocks the release until it is closed.

Still open, in the order above:

- Step 2, the collective half: the wave-boundary IPC and chain key,
  `FinishAsync` for publish frames, the two-phase commit and validation
  inflation (2.5, 2.8, 2.9).
- Step 3, the second half: the runner's sequence-boundary split of a
  multi-sequence wave, GLM 5.3 Full's wave chunking and DSV4.1's tail rows in
  the runner, K3's dispatch loop; then the collective plan from the
  schedule.
- Step 4: the shared CUDA stage-model plumbing.
- Step 5: streamed prefill as a schedule backend.
- Step 6: placement as deployment data.

Each of these changes the serving path's behaviour and none can be proven
by a host test or a compile: they need a lane on the fleet, which the
budget of 2026-11-10 did not allow.

Found while deleting, and left: the committed static deployments
`deployment/glm5_next_tp16` and `deployment/laguna-s-2.1.bf16.tp8pp2` carry
stage configs in the `tp_collective` schema the adapter refuses (the
deployment loader test and the launcher test use them as fixtures); the
hardware characterisation tools keep a `gpudirect` candidate, which measures
the hardware rather than selecting a serving mode. `tools/k3_multidev_lane.py`
and its TP4xPP4 default went with step 7's inputs: K3 test lanes render
through `tools/stage_lane.py`.

Merged on 2026-11-10: the tree verification of #1902 (trunk plus single-level
branches in a verify frame, on Metal for Qwen3.8-27B and on CUDA for Solar
Open 2; `docs/TREE_VERIFICATION_DESIGN.md`), with the module's tree slot
grain as a fact and the `TREE_VERIFY` capability derived from it. Folded in
as their own commits: GLM 5.3 Flash's index round gathers each rank's top
pools (#1837); verify frames of 2-16 rows read each dense weight once
(#1891); the shared spine loads through the direct reader (#1871); the KV
headroom is the measured 2.5 GiB and GLM 5.3 Full BF16 serves 1M positions
(#1753, confirmed by the first 1M run).

The Metal line merged on 2026-11-10 as well, on the owner's call, in
dependency order: numerics that follow the publisher with FlashAttention-2's
prefill form (#1739, #1865), Qwen 3.8 Max on the Qwen Metal backend (#1760),
Metal serving through the common stack (#1901: the stage serving adapter, KV
binding and copier on the device API, Qwen 3.8 27B served on one M5 Ultra),
Qwen3.5-4B and 2B on the Qwen device stage (#1900) and declared cache points
(#1905). The adapter keeps the module facts, the tree frames and the required
wait mode across the device API; the Qwen3.5 geometries take the 27B's
rounded rope terms until their reference run says otherwise. The merged
Metal kernels were not compiled here (no Metal toolchain on this Mac), which
the Metal session does on M5U0.

Carried in from the open pull requests on 2026-11-10, each as its own
commit on this branch: the all-reduce switch to reduce-scatter plus
all-gather from 10,240 elements at 16 ranks on dual rail (#1863); a failed
streamed prefill makes the lane fatal, so the runner refuses every later
submit and reset and residentd exits the rank for a restart (#1874); every
slow path logs once per shape, a decode graph that cannot be captured or a
decode step before the experts are resident aborts the rank, and `kv_pages`
is required in an adapter configuration (#1841, the owner's rule: a
fallback turns a bug into a number). The draft planner's calibration fix
(#1873) went to the tree-verification branch (#1902), whose draft server
already keeps contexts per engine session and request (#1765's fix).

Closed as superseded or dropped, with what they measured kept here so the
next performance pass starts from it:

- Two-lane prefill (#1889): on the 11-08 GLM 5.3 Full 16K launch profile,
  22-31% of the span is collectives with no compute and compute overlaps a
  collective only 18% of the time, about 1.3x on prefill from overlap
  alone. The fix it proposed, the second lane's rounds on its own mesh
  band borrowed through the lane's weightd client, belongs in step 3's
  runner now that both lanes run on one host thread: the runner opens the
  band, not the lane thread the PR was written against.
- Drafter taps from the GLM and Gemma 4 stages (#1639): with the runner
  walking the layers, a tap is an export after layer `k` in the runner's
  loop, one hook for every family, not a change per stage.
- K3 fill lane (#1853): a second lane configuration for prefill filled
  245,760 tokens in 165 s plus a 76 s reload against 494 s on waves. The
  number stands as the target for the streamed prefill in the decode lane;
  the second configuration does not.
- Concurrent agent sessions speculating (#1765): the engine side (one draft
  connection per lane, drafts sent before the frame is planned) is to be
  redone on the tree-frame engine if agent-turn measurements call for it.
- A per-request GPU clock mode (#1676), the asynchronous bulk collectives
  experiment (#1705) and the PP4 x TP4 DeepSeek prefill (#1896, the owner
  dropped the topology on 2026-11-09) are closed without carry-over.

Two items for the next performance pass, from the closed pull requests and
the November perf audit: the second lane's collectives on their own mesh band,
and a roofline meter in the runner (bytes moved and rounds per step against
the measured bandwidth and round cost, logged per rank) so that attainment
is a number the rank reports, not an estimate.

## Not covered, and numbers to measure

Not read in depth: the weightd daemon and mesh internals beyond the round
structure, the KV page cache and prefix cache internals (TECHDEBT's KV
sections stand), the tokenizer, the speculation sources, the serving Python
beyond its interfaces, and the per-family CUDA backends beyond their
interface plumbing and round counts. The three inventories and four maps
were not independently re-verified beyond the spot checks marked
(confirmed); their line citations are current as of this commit.

To measure on the fleet before step 3: the inter-frame bubble per rank
(completion send to next CONTINUE read), one `Progress` tick at 1 and 16
lanes, the host-side fan-out skew into the first round of a step, and the
count of PREPARE against CONTINUE steps in steady-state decode.
