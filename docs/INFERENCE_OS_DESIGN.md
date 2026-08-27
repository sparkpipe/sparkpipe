# The Inference OS — design notes for true hardware independence (2026-08-27)

Response to the sharpest critique of the HAL plan: a device API that
only wraps launch/streams is a *driver*, not an *operating system*. The
deep coupling is memory — unified vs split address spaces, DMA paths,
file-backed weights, tiered KV. This document thinks it through as an
OS and states the memory model the device API must carry.

## The OS framing, taken seriously

An operating system is: stable ABIs over unstable hardware, a scheduler
that multiplexes a scarce resource, a memory hierarchy manager, and a
storage hierarchy manager — with POLICY in the processes and MECHANISM
in the kernel. Mapped honestly to what exists:

| OS subsystem | SparkPipe analog | Status |
|---|---|---|
| Syscall ABI / process model | serving adapter ABI, module ABI, dlopen chain | exists, device-opaque |
| Scheduler | batch engine, B* admission (compute-bound quantum), expert-grouped batching | in flight (spark3 lane) |
| Virtual memory / paging | KV page store, prefix cache, JIT-KV NVMe overflow | exists for KV only |
| Physical memory manager | cudaMalloc/cudaMallocHost/managed calls scattered per family | **missing** |
| Block I/O / DMA engine | open-coded cudaMemcpyAsync, memlink lanes for network | **missing as a layer** |
| Filesystem / storage tiers | pack format v2, warm/ceph + cold archive, verifier | exists (model-scoped) |
| Driver discovery | hardware probes (topology/kernel/transport, dlopen'd) | probe pattern exists |
| Device drivers / HAL | spark_device.h (Phase 1B) | not started |
| IPC / network stack | collectives (two-backend), hidden transport, memlink, fabric topology | exists — flat peers, config-picked backend; link classes + island model are the addendum below |

The two missing rows are both memory. That is the critique, and it is
correct: today "a pointer" in this tree means five different things
depending on which family allocated it and on which hardware it runs.

## The memory space model

Memory is not a pointer. Memory is a tuple:
**(address, space, residency, access set, provenance)**.

Five spaces, chosen because every target backend expresses exactly these
distinctions:

```
DEVICE_PRIVATE   cudaMalloc          Metal .private     hipMalloc      host: malloc
                 (device pages; on UM hardware still valid + prefetchable)
SHARED_COHERENT  cudaMallocManaged   Metal .shared      HSA fine-grain host: malloc
                 (one address, both processors coherent — GB10's nature)
HOST_PINNED      cudaMallocHost /    (DMA-eligible host staging)
                 cudaHostRegister
HOST_PAGEABLE    malloc              (never a DMA source on discrete GPUs)
FILE_BACKED      mmap of a pack      (device visibility via register or staging)
```

Operations on spaces (the mechanism set — all policy stays in modules):

- `alloc(kind, bytes)` / `free` — every buffer is born with a space tag
- `register(ptr, len)` / `unregister` — make EXTERNAL memory device-visible
  (cudaHostRegister on cuda; RDMA-registration where transport needs it;
  no-op on host/metal-shared). K3's chunked pack registration is this.
- `map_file(fd, off, len, kind)` — the pack path: `mmap+register` on
  UM hardware (GB10: weights become device-visible with zero copies),
  `staged upload` on discrete (weights stream into DEVICE_PRIVATE), plain
  `mmap` on host. ONE call, three backend behaviors — today this logic
  is private to the K3 loader and must become shared.
- `copy(dst, src, n, stream)` — the DMA engine, SPACE-AWARE: elided when
  both ends are SHARED_COHERENT, staged through HOST_PINNED when crossing
  into DEVICE_PRIVATE on split hardware, memcpy on host. Open-coded
  cudaMemcpy with assumed spaces is exactly what breaks on discrete.
- `make_resident(ptr, len)` / `evict` — the migration policy hook:
  prefetch advice under UM, explicit DMA on discrete, no-op on host.
  The JIT-KV pager and a future weight pager are the SAME operation
  applied to different objects: KV is swap, weights are demand-paged
  executables. One mechanism, two policies.
- `access(ptr) -> space set` — lets dispatch pick fast paths (zero-copy
  eligibility) instead of #ifdefs.

The unification this buys: **KV pages, pack weights, activations, and
collective buffers are all memory objects with residency transitions.**
The page store (swap), the pack loader (demand paging), memlink
(network memory), and staging copies (block I/O) stop being four
subsystems with four vocabularies and become one memory manager.

## Backend matrix (what each port actually decides)

| | GB10 today | discrete CUDA | host oracle | Metal | ROCm |
|---|---|---|---|---|---|
| default weight path | FILE_BACKED→register (zero-copy) | staged→DEVICE_PRIVATE | mmap | mmap or shared buffer | staged or HSA |
| CPU↔GPU copy | often elidable (coherent) | always DMA via pinned | memcpy | often elidable | APU: elide / discrete: DMA |
| KV overflow | NVMe tier (swap) | NVMe tier | N/A | file tier | NVMe tier |
| UM usage | primary | avoid (policy) | — | native | APU-only |
| peer access | fabric via transport | NVLink P2P where probe says | — | — | XGMI |

The hardware probes (topology/kernel/transport) are the driver-discovery
layer that fills this matrix at runtime; MODULE_TARGET's tuple is the
compile-time half.

## Migration order (memory first, deliberately)

The original Phase 1B ordering (streams → launch → memory) is wrong for
this tree: memory is where families diverge most, and memory mistakes
are the ones that silently work on GB10 and die on the first discrete
port. Revised:

1. **M1 — buffer handles + alloc/free/copy.** `SparkDeviceBuffer{ptr,
   space, bytes}` and a space-aware copy. Lands inside the DRY adapter
   template (Phase 1A) so it wraps ONE lifecycle, not seven — one new
   call site per family, not fifty.
2. **M2 — register + map_file.** Absorb K3's chunked registration and
   the family pack loaders' mapping. This is where GB10's zero-copy
   weight trick becomes a backend behavior instead of a K3 secret.
3. **M3 — residency/evict.** Absorb JIT-KV tier transitions; unify with
   weight residency. The batch engine's backpressure signal becomes
   memory-pressure signaling (the OOM-avoidance path an OS gives).
4. **M4 — streams/events/launch.** The classic HAL surface, last,
   because it is the most mechanical.

## The comms subsystem and the two-layer compute resource (2026-08-27 addendum)

Comms is its own subsystem, not a memory footnote — but the two are
linked: a link with MEMORY semantics decomposes comms operations INTO
memory operations (peer copies, zero-copy mappings), while a message
semantics link runs them over a transport. The four-GPU-server case
forces the model: a server is NOT one compute resource, and it is not
four independent ones either.

**RANK and ISLAND — the two layers:**

- **Rank** = one GPU (or compute unit) + its local memory spaces. The
  unit of parallelism: owns memory objects, KV pages, kernels execute
  against its spaces. A spark today is exactly one rank.
- **Island** = a set of ranks + the internal fabric between them,
  characterized by a LINK MATRIX. The unit of deployment topology and
  collective algorithm choice. A 4x-GPU server is one island of four
  ranks; a GB10 spark is one island of one rank (the degenerate case —
  but the model must SAY that, never assume rank == node).

**Link classes** (what the matrix holds; each declares semantics,
permitted one-sided ops, latency/bandwidth class, ordering):

```
INTRA_DEVICE_SHARED  same address space — no comms op exists at all
P2P_MEMORY           NVLink/switch: memory semantics — peer space is
                     mappable, DMA peer-to-peer copy, zero-copy eligible
FABRIC_RDMA          our 100Gbps fabric: message semantics + RDMA
                     windows (memlink); host-staged or device-direct
NETWORK_TCP          fallback: host-staged copies over sockets
```

**The placement law falls out:** TP is a collective every layer (tight
loop) → it wants the highest-bandwidth lowest-latency links → ranks of
one island first. PP is point-to-point between adjacent stages (loose) →
across islands. Hence a server "as one pipeline stage" is just a stage
spanning multiple ranks within one island — TP4 intra-server, then PP
across servers. The same law explains our current fleet: 16 islands of
1 rank, all-fabric links → TP4xPP4 or TP16 across the fabric, exactly
what we run; nothing about the model changes, it finally has a name.

**Collective selection moves from config to topology.** Today the
deployment picks a backend by hand (host TCP tier vs device tier with
nccl/hidden_transport). The design: probes feed the island link matrix;
the collective engine decomposes an all-reduce hierarchically (reduce
over P2P links intra-island, then over fabric links between islands)
from the matrix, with the deployment recipe as override. Comms ops are
INTENT (all_reduce, send_recv, broadcast); the backend decomposes into
memory ops where the link class permits and transport ops otherwise —
that is the precise sense in which comms "relates to memory."

**Why not three layers** (server/rack/DC): islands compose. A rack is
islands + a switch topology — already expressible in the fabric
topology descriptor (rails, switches). Deeper nesting would encode
physical proximity the link matrix already measures.

**Migration (extends the memory M-ladder):**
M5 — island-aware deployment descriptors: rank gains island membership;
the hand-written flat `peers: [host:port]` list becomes DERIVED from the
topology (same-island first). No behavior change on the 16-spark fleet.
M6 — link-matrix probes (P2P detect, RDMA window probes — the hardware
transport probe pattern exists) feeding backend selection.
M7 — hierarchical collectives: the two-tier all-reduce; NCCL and
hidden_transport become per-link-class engines instead of config-picked
wholes.

**What exists today:** two-backend collective switch, memlink lanes,
hidden transport, the fabric topology descriptor (rails/switches/MTU),
transport/kernel/topology probes. **Missing:** rank != node awareness,
link classes, derived peer sets, hierarchical collective decomposition.

## Lane discipline NOW (no infrastructure required)

While M1-M4 wait on the DRY template, three rules hold today and are in
the lane contract: every allocation names its space kind in a one-line
comment (device-private / pinned / coherent / file-backed); any NEW
allocation kind or new CUDA API category in a family goes into the
integration-request inventory; cross-space copies never assume pointer
identity (no open-coded cudaMemcpy between "a host pointer" and "a
device pointer" without the comment saying which spaces). Zero-cost to
follow, and it makes the M1-M4 inventory complete when extraction runs.

## What this deliberately does NOT do

No hardware autodetection magic replacing explicit deployment recipes
(an OS still has fstab); no per-family conditional code (capability
branching lives in the backend + probes, never in modules); no attempt
to make host-oracle numbers predict device performance (numerics gates
only, as the shim already states). The goal is: one memory model, five
backends, policy in the model modules, mechanism in the runtime.
