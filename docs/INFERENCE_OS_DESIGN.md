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
| IPC / network stack | collectives (two-backend), hidden transport, memlink | exists, CUDA-flavored |

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
