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

**Adding a model.** Model families share code through
[family templates](docs/FAMILY_TEMPLATES.md). A template moves into common
code only if every family compiles it to identical host code and SASS. A new
driver starts from one parameter header, whose compile errors list every
value it must supply
([`docs/COMMON_MODULE_ARCHITECTURE.md`](docs/COMMON_MODULE_ARCHITECTURE.md)).
The recipe generator derives TP and PP placements from the model's
authoritative contract in `model_contracts/`.

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

**Storage tiers.** Each node has a 4 TB internal NVMe and at least 4 TB
external:

- internal: hot KV (2.5 TB), active model shards (1 TB), system (0.5 TB);
- external: direct rank-local model storage, with the remainder pooled
  across the fleet as a striped model store.

Every tier caches the same content-addressed pack bytes. A nonresident model
is promoted to serving in at most one minute.

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

**End state.** Kernels ring the NIC themselves (GPU-initiated RDMA), with no
CPU relay per round. Two micro-batches alternate, so one computes while the
other communicates. Both rails carry traffic at once: the switch for
all-to-all reach, the direct pair link for the first reduction level.

## Batching

**Continuous batching.** The batch engine admits arrivals and retires
completions at every step, at any occupancy. It caps decode submissions in
flight at the pipeline depth: 1 for fanout TP, `stage_count` for PP. A
request that becomes ready during a step joins the next wave instead of
running as a separate one-row chain.

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
change its tokens. Decode graphs are captured for each row count.

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

**Tiers.** Pages move between GPU memory, host memory, the 2.5 TB NVMe tier
and an optional external store. The pager parks and restores whole lanes
rather than paging per token, because attention reads the entire context on
every step. Under pressure, admission queues work instead of thrashing.

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
per model and per hardware:

- TP16 is the latency topology.
- TP4 × PP4 is the capacity topology, with each TP group containing two
  complete direct pairs.

Each catalog entry pairs a checkpoint with a topology, a resource envelope
and qualification receipts for one hardware configuration.

**Execution order.** Submissions run in the same order on every rank.
Collective sequence numbers are therefore identical across the fleet, and a
reordering cannot deadlock the mesh.

**Several models.** Models can be resident at the same time:

- Each serving engine holds a mesh lane.
- Weight arenas are shared by content identity, so a model never loads twice
  on a node.
- A model that is loading does not stop one that is serving.
- Co-resident models run bounded, gang-scheduled quanta, and priorities and
  deadlines apply across models.

## Speculative decoding

Speculation is a provider plugged into the serving adapter. The adapter
sees one lifecycle slot, and one model-neutral engine does all acceptance
accounting: the longest accepted path plus one bonus token, with a chain
treated as a degenerate tree.

A model customizes exactly three things:

- a geometry descriptor;
- a draft function;
- a fold/rollback for its own recurrent and paged state.

**Providers.** Supported providers include MTP heads, DFlash and DSpark, and
new methods arrive as new provider modules. A tree verifier checks several
drafters' candidates in one target pass. A tournament provider races
decorrelated drafters and retires the ones that stop paying for themselves.

**Guarantees.** Verification pins every emitted token to the target model,
so speculation changes throughput, never output. Every built speculation
path can be selected individually, and none can be compiled out.

## Determinism and evidence

**Determinism.** Greedy decoding is bitwise reproducible from run to run:

- reductions use fixed orders;
- collectives sum peers in a fixed order;
- kernel variants agree bitwise wherever the arithmetic allows.

Sampling uses a counter-based RNG keyed by the request seed, so a sampled
completion can be replayed off-node and audited against logged
(driver, contract, request) identities.

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
| 16 Sparks | TP16 latency plans; TP4 × PP4 capacity plans |
| DGX Station (1×, 2×, 4×, 8×) | standalone, or added to a Spark fabric under the same contracts |

**Fabric.** Every Spark uses two data rails:

- a 100 Gb/s port on a CRS804 switch;
- a nominal 200 Gb/s port cabled directly to `rank XOR 1`.

The GB10 PCIe path limits the direct link to about 100 Gb/s of useful
payload, so the two rails are equal-rate. Neither is a fallback for the
other. Management networks never carry inference traffic.

**Topology generation.** One deployment JSON generates the pinned
interfaces, peers and topology tables for every rank. Startup rejects
missing, duplicated or misrouted rails.

## Models

Drivers exist in `modules/` for these families:

- DeepSeek V4 Flash, V4 Pro and V4.1 Flash;
- GLM 5.3 Flash (`glm5_next`, the current optimization focus) and GLM 5.3
  Full (`glm52` module);
- Kimi K3;
- Qwen 3.8 Max, Qwen 3.8 27B and Qwen4 Flash;
- MiniMax;
- MiMo 2.5 and 2.6 (Flash, Pro);
- Gemma 4;
- Ling;
- Laguna;
- Hunyuan HY4;
- Muse Glimmer.

A driver in the tree is not a readiness claim. Each exact checkpoint needs
its own contract, pack, numerical result, transport profile and service
receipt. Status is in [`docs/DRIVER_ACCEPTANCE.md`](docs/DRIVER_ACCEPTANCE.md)
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

## Repository

| Path | Contents |
| --- | --- |
| `node/` | `model_residentd` (per-rank serving process), `weightd` (weights and mesh), `model_api`, batch client |
| `runtime/` | batch engine, serving adapter template, stage-module lifecycle, weightd client, module library and compiler |
| `cache/` | paged KV cache, prefix cache, page store, pager, NVMe tier |
| `ring/transport/` | RDMA transport, device collectives, fabric topology |
| `inference/kernels/` | shared CUDA kernels: GEMM/MMA/TMA, skinny and row-blocked GEMV, attention, routing, top-k, norms, speculation |
| `modules/` | per-family resident decode stages (firmware modules and serving adapters) |
| `model-families/`, `model_contracts/` | model geometry headers and authoritative checkpoint contracts |
| `include/sparkpipe/` | public ABIs and family templates |
| `tools/` | packers, recipe generator, harnesses, fleet tooling |
| `tests/` | host, contract, host-shim kernel and GPU tests |

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): deployment ladder, fabric and
  storage contracts.
- [`SPEC.md`](SPEC.md): firmware, module library and compiler contract.
- [`sparkpipe_invariants.md`](sparkpipe_invariants.md): the rules every
  driver must satisfy.
- [`TECHDEBT.md`](TECHDEBT.md): unfinished work against this description.
- [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md): measurements, kept
  separate from projections.
- [`docs/README.md`](docs/README.md): index of maintained technical
  references.

Superseded designs and experiment logs are kept under
[`docs/archive/`](docs/archive/) as history, not authority.

## Build and test

```sh
make clean
make -j1 all
make test
sh tools/gates.sh
```

`make test` runs without a GPU. GPU tests and harnesses have their own
targets, for example `make test-glm5-next-rows-kernels` and
`make bench-glm5-next-batch`. Hardware qualification requires CUDA,
transport, numerical and service receipts from an exact committed revision.
Host or simulator results cannot stand in for them.
