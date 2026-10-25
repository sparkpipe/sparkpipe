# SparkPipe

SparkPipe serves open-weight frontier models from a fleet of small
unified-memory machines. The reference fleet is sixteen NVIDIA DGX Spark
(GB10) nodes. Each node has:

- 128 GB of unified memory at 273 GB/s;
- one 100 Gb/s RDMA port on a switch;
- a second port cabled directly to one neighbour.

Together the sixteen nodes hold 2 TB and can stream about 4.4 TB/s. That is
enough for models that normally need a datacenter server, but only if the
fleet behaves like one machine. SparkPipe is the software that makes it
behave that way.

It is written from scratch in C and CUDA. The GEMM, attention, MoE and
collective kernels, the RDMA transport, the tokenizer, the scheduler and the
cache all live in this tree. There is no third-party inference framework or
kernel library.

This file describes the system SparkPipe is built to be.
[`TECHDEBT.md`](TECHDEBT.md) lists where the code does not match it yet, and
[`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md) holds the measurements.

## Performance today

Sixteen DGX Spark nodes at TP16, measured 2026-10-18. Methods and receipts
are in [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md).

**Decode without speculation**

| | Kimi K3 | GLM 5.3 Full | GLM 5.3 Flash |
| --- | ---: | ---: | ---: |
| 1 request | 16.2 tok/s | 28.9 tok/s | 55.0 tok/s |
| 1 request, long context | 15.1 tok/s at 73K | 24.1 tok/s at 197K | 46 tok/s at 71K |
| 8 requests, total | 64 tok/s | 74 tok/s | 176 tok/s |
| 16 requests, total | – | – | 227 tok/s |
| Of the floor for the bytes it reads | 58% | about 50% | 59% |
| Prefill, 8K prompt | 590 tok/s | 1,162 tok/s | 876 tok/s |
| Context window | 1,048,576 tokens (needle found at 998K) | 1,048,576 tokens (needle found at 1.01M) | 131,072 tokens |

**Speculation, one request**

| Workload | Kimi K3 | GLM 5.3 Full |
| --- | --- | --- |
| Whole-file code rewrite | 15.3 → 140 tok/s | 27.6 → 233 tok/s |
| SEARCH/REPLACE edit | 1.9× | 1.3× |
| Short edits with the draft model | 1.4-1.6× | – |

Drafts come from the request's own context (lookup) and, on K3, from a
draft model on the hub's RTX 5090. The target model verifies every draft.

**Long context and cache**

- Each node stores 1/16 of every sequence's cache, so K3 and GLM 5.3 Full
  serve 1,048,576-token windows. A needle placed in a 998K-token prompt
  (K3) and a 1.01M-token prompt (GLM 5.3 Full) was found after a cold
  prefill of 66 and 55 minutes.
- The next turn of a long conversation reuses its cached prefix: 5 s at
  137K tokens instead of 299 s, 8 s at 337K instead of 982 s.
- Cached prefixes survive model swaps: 137,600 tokens restore in 2.7 s.

**Accuracy**

- Every release reproduces the previous build's tokens for all 16 models.
- ds4_eval, 92 questions: Qwen 3.8 Max 82/92 and Qwen 3.8 Flash 79/92 as
  served by SparkPipe.
- Number formats ship only after a teacher-forced A/B shows no loss beyond
  a same-precision floor.

Limits today: K3 and GLM 5.3 Full serve up to 8 requests at once and GLM
5.3 Flash up to 16; prefill reaches at most 5% of compute; and only K3 and
GLM 5.3 Full speculate.

## The constraint everything follows from

Decoding one token is bound by memory bandwidth: a step takes about as long
as it takes to read that step's weights and cache bytes once. For GLM 5.3
Flash at TP16, each rank reads 2181 MB per token, which gives a floor of
8.0 ms per token at 273 GB/s
([`docs/GLM5_NEXT_ROOFLINE.md`](docs/GLM5_NEXT_ROOFLINE.md)).

Three rules follow, and most of the design is one of them applied to a
different part of the system:

1. **Read every byte once per step, on one node.** Weights and cache are
   sharded so that the fleet's bytes are disjoint. Anything replicated on
   every rank multiplies its cost by the fleet size. For example, 30% of
   GLM 5.3 Flash's per-rank bytes are replicated tensors.
2. **Amortize each read over as many rows as possible.** A weight tile read
   once serves every request in the step. Batching is the throughput lever,
   and the kernels are built for every row count from 1 to 1024.
3. **Keep communication off the critical path.** A 16-rank collective costs
   tens to hundreds of microseconds, and a model issues about 90 of them per
   token. They must be cheap, overlapped with compute, or both.

## Models compile to firmware

A model is not interpreted at runtime. One model-description JSON names each
pipeline stage, its exact GPU target, and the ordered firmware modules each
program runs ([`SPEC.md`](SPEC.md)).

**Module library.** A firmware module is a content-addressed link unit: an
object file or archive, identified by its SHA-256. It is validated once
against its numerical and behavioural contract, then stored read-only. The
same bytes are never re-validated. Changed code produces different bytes,
and therefore a new identity that must pass validation again.

**Compilation.** The deployment compiler resolves every module ID. It then
generates a C orchestrator for each stage that calls the modules directly,
in order, and links each stage into one driver shared object. At runtime
there is:

- no graph interpreter;
- no per-request kernel selection;
- no module lookup;
- no fallback search.

**Module ABI.** The interface is five symbols: initialize, execute, admit,
snapshot, destroy. The runtime asks a module for admission decisions,
dispatch slots and snapshots. It never learns the module's KV layout, MoE
queue structure, graph topology or sampler. A module may specialize freely
for its exact checkpoint, shapes, precision, GPU and placement.

## One serving policy for every model

Common code owns everything that is not the model's mathematics:

- admission and continuous batching;
- cache transactions and sequence ownership;
- cancellation, release and reset;
- collective ordering and progress;
- engine lifecycle.

A driver supplies geometry, state layout and math through narrow callbacks.
This is the "qsort pattern": one optimized algorithm, parameterized by small
hooks and an opaque context.

The contract is enforced rather than advisory
([`sparkpipe_invariants.md`](sparkpipe_invariants.md)):

- **Required means required.** A missing required operation fails and names
  itself. No capability bit, environment variable or stub can opt a driver
  out of batching, JIT cache, prefix reuse or state restore.
- **No silent fallbacks.** An unsupported shape, topology or precision is an
  error, not a slower path.
- **No hidden clamps.** A power-of-two kernel bucket is never an admission
  rule. Diagnostic clamps live under `#ifdef DEBUG` and report themselves.

**Adding a model.** Every driver runs on the common stage runner
(`inference/runner/`). A new driver supplies only its math, as a stage model
behind the runner's model interface
(`include/sparkpipe/spark_stage_runner_model.h`), plus its serving glue and
packer ([`docs/MODULE_MAP.md`](docs/MODULE_MAP.md)). The recipe generator
derives TP and PP placements from the model's authoritative contract in
`model_contracts/`.

## Weights: stage packs and a residency daemon

**Stage packs.** Weights are repackaged once, offline, into immutable stage
packs. A pack is already sharded for one topology and already in the exact
byte layout the kernels read, and it is identified by SHA-256 at every
storage tier.

SparkPipe never quantizes a model itself. Packs carry either the
publisher's own precision (BF16, FP8, MXFP4, NVFP4) or a vetted community
release with pinned provenance. The packer only converts formats, slices by
topology and lays out scale planes; its codecs come from one shared table.

**weightd.** One persistent daemon per node, `weightd`, owns the weight
arenas:

- It loads a pack once and verifies its hash and geometry once.
- It exports the memory through CUDA VMM shareable handles, mapped read-only
  into consumers. The pointer a kernel reads is the daemon's memory, with no
  copy.
- It attaches by identity: model, revision, topology, pack hash, geometry
  fingerprint and ABI version. A code-only redeploy restarts the serving
  process and re-attaches in milliseconds, with the weights untouched.
- It loads routed experts lazily. The arena reserves the full expert address
  space; experts are acquired on demand under leases and reclaimed within a
  declared budget. A missing or corrupt expert manifest is an error, never a
  reason to load the whole pack eagerly.
- Captured decode graphs work with lazy residency. A graph records which
  kernel arguments are expert pointers and has them patched when leases
  move, so a graph never forces the whole expert set to stay resident.
  Pinning every expert is a separate resident mode, selected explicitly and
  reported with its memory cost. GLM graphs currently require that mode
  (invariants I28-I30, [`TECHDEBT.md`](TECHDEBT.md)): since `78c2c21`, GLM
  5.3 Flash graph serving pins the whole expert pack on every rank, and it
  will until graphs are relocatable
  ([`docs/WEIGHTD_DESIGN.md`](docs/WEIGHTD_DESIGN.md#glm-graph-residency-today)).

**Storage tiers.** Each node has a 4 TB internal NVMe and at least 4 TB
external:

- internal: hot KV (2.5 TB), active model shards (1 TB), system (0.5 TB);
- external: direct rank-local model storage, with the remainder pooled
  across the fleet as a striped model store.

Every tier caches the same content-addressed pack bytes. A nonresident model
is promoted to serving in at most one minute. Promotion installs rank-local
shards, binds stable pointers, prewarms kernels and graphs, constructs
communicators and then publishes readiness atomically. It never converts or
guesses a weight layout. Activating a model changes an execution plan and a
residency assignment, not the public endpoint, and eviction keeps resumable
KV and model artifacts in the storage tiers.

## The mesh

**Wiring.** Each node's `weightd` also owns the RDMA mesh. At boot it wires
queue pairs to all fifteen peers and keeps them for the life of the node. A
serving process attaches to a mesh that is already up, so a deploy never
re-exchanges connection records. One all-to-all mesh contains every
topology: tree, ring, pipeline, and TP groups of 4, 8 or 16.

**Data path.** Collectives work on fixed registered slots. A GPU kernel
writes its contribution into a mapped slot. The NIC moves it to each peer's
receive slot, and the consumer kernel reduces in place. There is no payload
copy through the CPU and no allocation or registration on the token path.

**Routes.** Each publication carries a route with named bitfields:

- a peer mask;
- a slice size;
- a mode: `FULL` sends the whole payload to each peer; `SCATTER` sends peer
  `p` only its slice; `GATHER` sends the local slice to every peer.

These modes compose into the collective algorithms, which are chosen per
call from degree, datatype, payload bytes and a measured crossover profile:

| Regime | Algorithm | Wire bytes per rank |
| --- | --- | --- |
| Latency-bound (B1–B8 decode) | one-round direct all-to-all | 15 × payload |
| Bandwidth-bound sums | reduce-scatter + all-gather over slice routes | 1.875 × payload |
| Prefill, large batches | pair sum over the direct link, then an 8-way switched exchange | 3.5 × payload |

**Invariants.** Every algorithm sums peers in the same fixed order, so
switching algorithm never changes a result bit. A collective carries its
full logical batch count separately from its execution rows, so splitting a
batch cannot select the wrong algorithm.

**End state.** Two micro-batches alternate, so one computes while the other
communicates. A GB10 SM cannot ring the CX-7 doorbell (the 2026-10-01 probe:
no write lands, no completion, no fault), and GPU-built work requests with a
CPU doorbell measured within about 1 us of today's relay per round, so the
CPU keeps ringing the NIC. Both rails carry traffic at once: the switch for
all-to-all reach, the direct pair link for the first reduction level.

## Batching

**Continuous batching.** The batch engine admits arrivals and retires
completions at every step, at any occupancy. It caps decode submissions in
flight at the pipeline depth: 1 for fanout TP, `stage_count` for PP. A
request that becomes ready during a step joins the next wave instead of
running as a separate one-row chain.

**Resident decode chains.** A decode submission runs up to eight steps on
the resident stages. Each step's tokens feed the next step on the device, and
cache blocks completed inside the chain are published from it, so the host
protocol between submissions is paid once per chain rather than once per
token.

**Expert-grouped MoE.** Routed rows are sorted by expert, and each expert's
weights are read once per step for all of its rows. At 288 experts with
top-8 routing, B64 already touches 240 experts. From there, weight traffic
per step is roughly constant, and throughput grows with rows until compute
becomes the limit
([`docs/EXPERT_GROUPED_SCHEDULING.md`](docs/EXPERT_GROUPED_SCHEDULING.md)).

**Kernels by row count.** Every kernel family has a variant for each row
regime:

- skinny kernels (a warp per output neuron, eight 16-byte loads in flight
  per lane) for 1–8 rows;
- per-expert grouped kernels up to 16 rows per expert;
- row-blocked shared-tile kernels, in which one coalesced weight tile serves
  16 rows;
- tensor-core GEMM above that.

Where the arithmetic allows, the variants accumulate in the same order and
are bitwise equal to each other, so the batch a request lands in does not
change its tokens. Decode graphs are captured for each row count. A frame
that no graph covers enqueues the same launches in one pass and checks its
collective rounds once, at the end of the frame.

**The harness.** `tools/glm5_next_batch_roofline.cu` runs the real kernels
on one GPU at any batch and context. It prints per-phase time against the
memory bound, so the next bottleneck can be read off directly.

## KV cache and model state

**What a cache entry holds.** An entry is the complete computation state:

- every KV tensor;
- sparse-attention indexer keys;
- recurrent state (KDA, GDN) and convolution windows;
- position and continuity fields.

Restoring an entry must produce the same tokens as never having evicted it.

**Paging.** Logical identity and physical residency are separate. A
sequence's pages are resolved to physical pages at submission, pinned during
use, and shared copy-on-write when they belong to an immutable prefix.

**Prefix reuse.** Prefix identity is a chained content hash, so a block
matches only if its whole prefix matches. Store keys add model and
cache-layout fingerprints, so a changed model or layout can never read old
KV. Publication is transactional: prepare, commit and abort, with
generations checked at every transition.

**Tiers.** KV device memory belongs to weightd, which keeps each engine's
pool across an engine restart. Pages leaving the device park in a per-engine
spill file on the KV NVMe, and published prefixes are saved to a persistent
snapshot store on the same drive, which serves restores after eviction or a
restart; restores run on a worker off the admission path. There is no host
DRAM tier ([`TECHDEBT.md`](TECHDEBT.md), KV tiers).

**Sharding.** The fleet stores 1/N of the cache on each of N nodes:

- Attention heads and recurrent state are split by head.
- State that every head shares, such as an MLA latent or a DSA indexer key,
  is split by context. Each rank scores and attends over its own tokens, and
  the partial results are merged exactly.
- One node-level pool holds each node's share for all resident drivers.

## Parallelism and placement

**Placement.** Tensor, pipeline and hybrid parallelism share one topology
layer. It owns partitioning, collective identity, credits and progress; the
model hooks supply only stage math and boundary shapes. Placement is chosen
per model, hardware and workload by measured tokens/s and latency, not fixed
in advance. On sixteen Sparks the candidates include:

- TP16, which spreads every layer's bytes over all sixteen nodes and pays a
  16-rank collective per reduction;
- TP4 × PP4 (stage `s` on ranks `4s` to `4s+3`), in which each TP group holds
  two complete direct pairs;
- PP16 and other TP × PP splits, which trade per-token latency for fewer
  collectives and more concurrent micro-batches.

Each candidate is qualified separately before it serves. Each catalog entry
pairs a checkpoint with a topology, a resource envelope and qualification
receipts for one hardware configuration.

**Package identity.** One generic resident process runs on each rank. A
model package binds one exact adapter, driver, stage pack, weight format, KV
contract, topology and hardware profile, and startup rejects any identity
mismatch. Batch width, speculation mode and collective algorithm change per
dispatch without reloading weights, drivers, KV, communicators or resident
processes.

**Execution order.** Submissions run in the same order on every rank.
Collective sequence numbers are therefore identical across the fleet, and a
reordering cannot deadlock the mesh.

**Several models.** Models can be resident at the same time:

- Each serving engine holds a mesh lane.
- Weight arenas are shared by content identity, so a model never loads twice
  on a node.
- A model that is loading does not stop one that is serving.
- Co-resident models run bounded, gang-scheduled quanta, and priorities and
  deadlines apply across models. No collective of one model is interleaved
  inside another model's committed kernel sequence.
- Short high-priority quanta bound interactive latency, and agent work fills
  the remaining compute, memory and network capacity.

## Speculative decoding

Speculation goes through one seam per serving adapter. Each family has one
source mask, `SPARK_<FAMILY>_SPECULATORS`, that selects its draft sources:
local drafts the model makes itself (MTP heads, DSpark) and remote drafts
from a drafter host through the DFT3 draft bridge (DFlash2, n-gram and suffix
sources). One model-neutral policy engine does all acceptance accounting: the
longest accepted path plus one bonus token, with a chain treated as a
degenerate tree.

A model customizes exactly three things:

- a model contract: geometry, draft limits and hidden-tap layers;
- a draft function, or the hidden-tap rows a remote drafter needs;
- a fold/rollback for its own recurrent and paged state.

**Sources.** A new method arrives as a new source behind the seam, not as
edits to every family. The end state, not yet built, is a tree verifier that
checks several drafters' candidates in one target pass, and multi-drafter
composition that races decorrelated drafters and retires the ones that stop
paying for themselves. Today every served acceptance is a chain
([`docs/SPECULATION_UNIFIED_DESIGN.md`](docs/SPECULATION_UNIFIED_DESIGN.md)).

**Guarantees.** Verification pins every emitted token to the target model's
multi-row verify output. Batched rows are not yet bitwise equal to B1
([`TECHDEBT.md`](TECHDEBT.md), Dynamic batching), so a speculative run can
differ from a B1 run without speculation. Every built speculation path can
be selected individually, and none can be compiled out.

## Determinism and evidence

**Determinism.** Greedy decoding is bitwise reproducible from run to run:

- reductions use fixed orders;
- collectives sum peers in a fixed order;
- kernel variants agree bitwise wherever the arithmetic allows.

Sampling is Gumbel-max on the vocab-parallel argmax. Every rank perturbs
its logits with noise from a counter-based RNG keyed by (request seed,
position, global token id), and the existing cross-rank max picks the
sample, so sampling adds no collective. The noise does not depend on which
rank or batch row computes it, so a sampled completion replays like a greedy
one and can be audited off-node against logged (driver, contract, request)
identities.

**Evidence domains.** Each of these is separate:

- source identity;
- host tests;
- exact-target CUDA compilation;
- transport measurements;
- numerical agreement with the reference implementation;
- end-to-end service behaviour;
- performance.

A result qualifies only its own domain. A driver gets two verdicts,
functional and performance, and a correct-but-slow driver is recorded as
exactly that. Hardware tests run on an exact committed revision, and every
receipt records source, build, configuration and pack hashes.

## Hardware independence

**Kernels.** Kernels are written as strided loops over `THREADS`, so most
of them are correct single-threaded CPU programs when `THREADS == 1`. Two
host shims run the real kernels in CI without a GPU:

- a single-thread shim, for arithmetic and indexing;
- a threaded shim, for kernels that depend on cooperation between threads.
  It runs one thread per CUDA thread, with real barriers and warp shuffles.

**Device layer.** Below the module boundary, the device layer models memory
as `(address, space, residency)`. The spaces are device-private,
coherent-shared, host-pinned, host-pageable and file-backed. Copies are
elided when both ends are coherent (GB10) and staged when they are not
(discrete GPUs). Weights, KV pages and collective buffers are all memory
objects with residency transitions.

**Topology model.** Topology is described as ranks inside islands joined by
link classes: shared memory, peer memory, fabric RDMA and TCP. Collectives
decompose hierarchically from that link matrix. The backends are:

- CUDA (production);
- host (CI oracle);
- Metal (Mac Studio);
- ROCm.

Scheduling and cache policy contain no CUDA, NCCL or Metal assumptions
([`docs/INFERENCE_OS_DESIGN.md`](docs/INFERENCE_OS_DESIGN.md)).

## Hardware

| Deployment | Use |
| --- | --- |
| 4 Sparks | TP4 models; entry system |
| 8 Sparks | TP8 or TP4 × PP2 |
| 16 Sparks | TP16, TP4 × PP4, PP16 or other splits, chosen by measurement |
| 1 to 8 Mac Studios (M5 Ultra, 256 GB) | one replica per Studio, TP inside a Thunderbolt 5 island of four, or TP4 × PP2 across two islands |
| Sparks with Mac Studios | one catalog: Sparks prefill and Studios decode, or PP2 across both |

**Fabric.** Every Spark uses two data rails:

- a 100 Gb/s port on a CRS804 switch;
- a nominal 200 Gb/s port cabled directly to `rank XOR 1`.

The GB10 PCIe path limits the direct link to about 100 Gb/s of useful
payload, so the two rails are equal-rate. Neither is a fallback for the
other. Management networks never carry inference traffic.

**Mac Studios.** Studios link to each other over Thunderbolt 5 RDMA, in two
fully meshed islands of four, and to the Sparks over a TCP bridge. No
tensor-parallel collective crosses hardware classes: a mixed plan hands over
the KV cache after prefill, or each token's residual stream at a pipeline
stage boundary. The Studios need the Metal backend. See
[`docs/HARDWARE_TOPOLOGY.md`](docs/HARDWARE_TOPOLOGY.md#mac-studio-pool).

**Topology generation.** One deployment JSON generates the pinned
interfaces, peers and topology tables for every rank. Startup rejects
missing, duplicated or misrouted rails.

**Office hardware.** The fleet runs on desks and ordinary office power.
Capacity grows by adding nodes, storage and generated deployment plans,
without changing the API, package identities, scheduler semantics or
evidence rules.

**Hub.** One host outside the Sparks publishes release roots, collects node
heartbeats and serves the model API. In the reference fleet it is an x86
workstation with an RTX 5090, so the API is built for x86 separately from
the aarch64 Spark binaries. Host roles and the release procedure are in
[`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md).

## Models

The product set and the owner's direction are in
[`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md). Sixteen models serve today,
all on the common TP16 stage runner (`inference/runner/`). Each model's math
lives under `model-families/<family>/` and its serving adapter under
`modules/`. One model holds the fleet at a time. A request names its model;
if another model is loaded, the request waits while the swap controller
(`tools/serving/model_swap.py`) swaps the named one in, and that model's
cached prefixes come back from the KV snapshot store.

| Served name | Model | Family | Context | Swap in | Decode, B1 | Of ideal floor | Prefill, 7-8K prompt | COMPSEC-17 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `glm-5.3` | GLM 5.3 Full | `glm52` | 1M | 38 s | 28.9 tok/s | 27% | 1,162 tok/s | 15 |
| `glm-5.3-flash` | GLM 5.3 Flash | `glm5_next` | 128K | 24 s | 55.0 tok/s | 23% | 876 tok/s | 13 |
| `kimi-k3` | Kimi K3 | `k3` | 1M | 46 s | 16.2 tok/s | 51% | 590 tok/s | 17 |
| `deepseek-v4.1-flash` | DeepSeek V4.1 Flash | `dsv41_flash` | 1M | 24 s | 29.5 tok/s | – | 182 tok/s | 16 |
| `mimo-v2.6-pro` | MiMo V2.6 Pro | `mimo26` | 32K | 24 s | 19.0 tok/s ‡ | – | 55 tok/s ‡ | 17 |
| `mimo-v2.6-flash` | MiMo V2.6 Flash | `mimo26` | 32K | 19 s | 40.9 tok/s ‡ | – | 157 tok/s ‡ | 16 |
| `hy4-preview` | Hunyuan HY4 preview | `hy4` | 32K | 28 s | 14.2 tok/s | 16% | 53 tok/s | 17 |
| `gemma-4-31b` | Gemma 4 31B | `gemma4` | 32K | 17 s | 34.4 tok/s | 48% | 834 tok/s | 16 |
| `gemma-4-26b-a4b` | Gemma 4 26B A4B | `gemma4` | 256K | 16 s | 89.5 tok/s | 16% | 1,848 tok/s | 16 |
| `ling-3.0-flash` | Ling 3.0 Flash | `ling` | 256K | 20 s | 80.4 tok/s | 19% | 1,514 tok/s | 13 |
| `ling-3.0-flash-fin` | Ling 3.0 Flash, finance fine-tune | `ling` | 256K | 20 s | 79.2 tok/s | 18% | 1,550 tok/s | 13 |
| `laguna-s-2.1` | Laguna S 2.1 | `laguna` | 32K | 20 s | 51.0 tok/s | – | 773 tok/s | 13 |
| `muse-glimmer-30b` | Muse Glimmer 30B | `muse_glimmer` | 128K | 16 s | 37.9 tok/s | 51% | 1,186 tok/s | 17 |
| `qwen3.8-max` | Qwen 3.8 Max (internal) | `qwen38` | 32K | 35 s | 10.4 tok/s | – | 35 tok/s | 5 † |
| `qwen3.8-27b` | Qwen 3.8 27B (internal) | `qwen38` | 256K | 19 s | 57.4 tok/s | 35% | 1,852 tok/s | 14 |
| `qwen3.8-flash` | Qwen 3.8 Flash (internal) | `qwen38` | 256K | 20 s | 45.0 tok/s | – | 358 tok/s | 15 |

All on sixteen Sparks at TP16, with the GPU clocks capped at 2200 MHz and
no speculation:

- **Decode, B1** is one request, greedy, generating 256 tokens (case
  `b1_o256` of `tools/api_serving_perf.py`), on a quiet fleet. Most rows
  come from the qualification of release `91c0e6be3` on 2026-10-12, with
  old and new builds measured in one hold. Newer runs replace four rows:
  K3 on 2026-10-14 (release `1d89e0377`), the two GLM models on
  2026-10-18, and DeepSeek V4.1 Flash on 2026-10-13 after its decode fix.
- **Of ideal floor** compares B1 with the ideal memory floor: each rank
  reads 1/16 of the active weights once per token at 273 GB/s. A dash means
  the floor is not computed, or the split of active parameters between
  formats is not verified. The arithmetic is in
  [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md#all-served-models).
- **Prefill** is one cold prompt of about 7,400 tokens (case `ttft6000`;
  the count depends on the tokenizer): prompt tokens over the time to the
  first token. K3 and GLM 5.3 Full were measured at 8K, on 2026-10-13 and
  2026-10-15.
- **COMPSEC-17** is the 17-question security set of the release
  qualification, from release `91c0e6be3`; MiMo at its bring-up on
  2026-10-13. † Qwen 3.8 Max always thinks; it ran with a 2,048-token
  budget.
- **Swap in** is the median time from a swap request until the model
  serves, over the swaps the controller logged on 2026-10-13 and 2026-10-14.
  It includes stopping the previous model. The slowest single swap was K3's,
  at 79 s.
- **Context** is the window the chat layer serves
  (`deployment/hub/chat_frontend.json`), from the lane records in
  `deployment/lanes.json`; `tests/test_lane_records.py` keeps this column equal
  to them.
- ‡ MiMo was measured only in the release qualification of 2026-10-14,
  while a CPU tenant kept about 16 cores busy on every node, so its numbers
  are lower bounds. The same load cut K3 from 16.2 to 10.0 tok/s.
- On the six fixed release prompts every model produced the same tokens as
  the build it replaced. GLM 5.3 Full's logprob golden moved slightly: 706 of
  708 top-1 tokens agree, and the one flip is a near tie.
- The Qwen models are internal only. They serve on the LAN door but not on
  sparkpipe.ai, because of their license terms.

Serving is not a readiness claim. Hy4 has not yet matched a reference run
([`TECHDEBT.md`](TECHDEBT.md), Production qualification), and each driver's
acceptance status is in [`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md)
and [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md).

## Serving API

The API is OpenAI-compatible:

- `/v1/completions`, `/v1/chat/completions` and `/v1/models`;
- text or token-ID prompts;
- a native tokenizer with no Python in the serving process;
- streamed responses.

Callers name a model, a priority and a deadline. SparkPipe chooses
placement, batch width, microbatch geometry, speculation and collective
algorithm.

A LiteLLM front door routes several islands behind one endpoint
([`docs/LITELLM_FRONTEND.md`](docs/LITELLM_FRONTEND.md)). Memory, tools,
policy and user interfaces belong above the API.

## Fleet-hours

HYOR AI Tech Ltd. rents whole SparkPipe fleets to businesses by the hour, from Seoul. Sales open on 1 December 2026; the terms are published for review on [sparkpipe.ai](https://sparkpipe.ai).

- **A fleet-hour** is one entire fleet for sixty minutes. During those hours it serves only the customer's requests, with the model the customer chooses from the [Fleet List](https://sparkpipe.ai/fleets.html).
- **Hourly terms:** prepaid hours at the hourly price, with no commitment and no business check. A customer holds up to 24 hours at a time, or 720 after a business check. A two-year customer may take an hourly reservation until 6 hours before it starts.
- **The two-year agreement:** every hour of a two-year term at a lower price, after a business check, negotiated clause by clause. The models the customer names stay available for the term.
- **Payment and records:** accounts are identified by an Ethereum address, and acceptance and reservations are messages signed with its key. Payments are prepaid in USDC through a payment processor, with Korean VAT; two-year customers may also wire USD from abroad. Every payment and reservation is a line in an append-only, hash-chained ledger.
- **Responsibility:** we provide capacity only. The customer decides what runs, answers for its users and anyone it passes hours to, and carries any AI-law duties toward its own users. Our system has no feature for transferring hours.

The provider network once planned here, reselling other installations' idle capacity, is deferred ([`docs/MARKETPLACE_PLAN.md`](docs/MARKETPLACE_PLAN.md)).

## Repository

| Path | Contents |
| --- | --- |
| `node/` | `model_residentd` (per-rank serving process), `weightd` (weights and mesh), `model_api` (HTTP server), batch client |
| `runtime/` | batch engine, serving adapter template, stage-module lifecycle, weightd client, module library and compiler |
| `src/` | status codes, hashing, admission, driver loader, orchestrator, speculation policy and seam |
| `scheduler/` | the batch client's continuous-batch queue and topology-switch tables |
| `cache/` | paged KV cache, prefix cache, page store, pager, NVMe tier |
| `ring/transport/` | RDMA transport, device collectives, fabric topology |
| `inference/kernels/` | shared CUDA kernels: GEMM/MMA/TMA, skinny and row-blocked GEMV, attention, routing, top-k, norms, speculation |
| `inference/llms/` | older per-checkpoint device code (K3, MiMo 2.5, Qwen 3.6) |
| `modules/` | per-family resident decode stages (firmware modules and serving adapters) |
| `common/` | GLM-lineage stage module and CUDA tree, GDN stage kernels, KV frame and geometry headers; the GLM files are pending moves in `tests/test_dry_law.py` |
| `model-families/`, `model_contracts/` | model geometry headers and authoritative checkpoint contracts |
| `include/sparkpipe/` | public ABIs and family templates |
| `schema/` | model-description and hardware-topology JSON schemas |
| `text/` | tokenizer |
| `deployment/`, `examples/`, `config/` | release assembly and topology tables, example descriptions and deployments, LiteLLM configuration |
| `qualification/` | retained receipts: transport, numerical, evaluation and performance runs |
| `validation/` | reference implementations for numerical checks |
| `tools/` | packers, recipe generator, harnesses, fleet tooling |
| `tests/` | host, contract, host-shim kernel and GPU tests |
| `experiments/` | research code outside the code-size ratchet; no serving path links it |
| `site/` | sparkpipe.ai: home page, Fleet List, terms (generated from `site/terms/` by `tools/site_terms.py`), privacy policy and checkout preview |

Common runtime code does not choose a model family, codec, topology, batch
width or fallback implementation by name. A deployment package binds those
decisions through exact model, hardware and release contracts.

## Documentation

Each document is the authority for one thing:

- this file: the system SparkPipe is built to be;
- [`SPEC.md`](SPEC.md): the firmware, module library and compiler contract;
- [`sparkpipe_invariants.md`](sparkpipe_invariants.md): the rules every
  driver must satisfy;
- [`TECHDEBT.md`](TECHDEBT.md): unfinished work against this description;
- [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md): measurements, kept
  separate from projections;
- [`docs/FLEET_RELEASE_RUNBOOK.md`](docs/FLEET_RELEASE_RUNBOOK.md): host
  roles, releases and fleet operations;
- [`docs/README.md`](docs/README.md): index of maintained technical
  references.

Production readiness is evaluated per exact model checkpoint and deployment.
It requires matching source and package identity, host and CUDA gates,
physical route evidence, numerical correctness, end-to-end service results
and retained receipts from a merged-main release.

Superseded designs and experiment logs are kept under
[`docs/archive/`](docs/archive/) as history, not authority;
[`docs/ARCHIVE_INDEX.md`](docs/ARCHIVE_INDEX.md) gives the reason each was
archived.

## Build and test

```sh
make clean
make -j1 all
make test
sh tools/gates.sh
```

`make test` runs without a GPU. On a host with a CUDA toolkit it does not
pass yet: one serving adapter links the TP transport without the mesh
launchers it calls ([`TECHDEBT.md`](TECHDEBT.md), Production
qualification). GPU tests and harnesses have their own targets, for example
`make test-glm5-next-rows-kernels` and `make bench-glm5-next-batch`.
Hardware qualification requires CUDA, transport, numerical and service
receipts from an exact committed revision. Host or simulator results cannot
stand in for them.

## License

SparkPipe is licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 HYOR AI Tech Ltd. Third-party files kept in the tree and their
licenses are listed in [`NOTICE`](NOTICE).

Model weights, tokenizers and the stage packs built from them are not covered
by the Apache License. They keep their publishers' licenses, some of which
add conditions for large commercial deployments.
