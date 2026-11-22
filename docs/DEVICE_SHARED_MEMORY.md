# Device shared memory

Processes share device memory through the device layer
([`include/sparkpipe/spark_device.h`](../include/sparkpipe/spark_device.h)).
weightd allocates weight arenas and the KV pool, and the engines map them.
One process creates exportable memory and passes it to another as a file
descriptor. Each process reserves its own address range and maps the memory
into it.

The device layer and one backend, chosen at build time, are compiled into
`sparkpipe_weightd`, `weightd_warm` and `weightdctl`, so they ship without a
separate device library.
- `make` builds them with the CUDA backend.
- `make SPARKPIPE_DEVICE=nvrm nvrm-weightd` builds them with the nvrm
  backend as `build/nvrm/sparkpipe_weightd`, `build/nvrm/weightd_warm` and
  `build/nvrm/weightdctl`.

Engines reach the same entry points through the device library their
adapter loads. The RDMA transport links that library too.

## Entry points

```c
SparkStatus SparkDeviceMemoryGranularity(uint32_t granularity, uint64_t *bytes);
SparkStatus SparkDeviceMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceMemoryCreateHost(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceMemoryRelease(SparkDeviceMemoryHandle memory);
SparkStatus SparkDeviceMemoryExport(SparkDeviceMemoryHandle memory, int *fd);
SparkStatus SparkDeviceMemoryImport(SparkDeviceMemoryHandle *memory, int fd, const char *owner);
SparkStatus SparkDeviceAddressReserve(SparkDeviceAddress *address, uint64_t bytes, const char *owner);
SparkStatus SparkDeviceAddressRelease(SparkDeviceAddress address, uint64_t bytes);
SparkStatus SparkDeviceMemoryMap(SparkDeviceAddress address, uint64_t bytes, SparkDeviceMemoryHandle memory, uint64_t offset, uint32_t access);
SparkStatus SparkDeviceMemoryUnmap(SparkDeviceAddress address, uint64_t bytes);
SparkStatus SparkDeviceThreadInstance(SparkDeviceInstance *instance);
SparkStatus SparkDeviceThreadBind(SparkDeviceInstance instance);
```

**Sizes.** `SparkDeviceMemoryGranularity` gives the minimum
(`SPARK_DEVICE_GRANULARITY_MINIMUM`) or recommended
(`SPARK_DEVICE_GRANULARITY_RECOMMENDED`) granularity, each a power of two.
Allocation sizes, reservation sizes, mapping sizes, addresses and offsets are
multiples of the minimum.

**Memory.** `SparkDeviceMemoryCreate` allocates device memory that can be
exported. It is not mapped anywhere until `SparkDeviceMemoryMap` maps it.
- `SparkDeviceMemoryExport` returns a new file descriptor for it.
- `SparkDeviceMemoryImport` turns a descriptor from another process into a
  handle; the caller still owns and closes the descriptor.

**Host memory.** `SparkDeviceMemoryCreateHost` allocates pinned system memory
near the device that exports, imports, maps and releases like device memory.
- It exists only where the backend sets
  `SPARK_DEVICE_PROPERTY_HOST_MEMORY_SHARE`; elsewhere the device layer refuses
  it, before any backend sees the call, with `SPARK_STATUS_UNSUPPORTED` and
  `the <backend> backend has no exportable host memory`.
- Its size is a multiple of the minimum granularity.
- A process that holds host memory for others (weightd's held packs, see
  [WEIGHTD_DESIGN.md](WEIGHTD_DESIGN.md)) maps it with host access to fill it;
  an importer maps it device-read-only and copies from it with the copy
  engines, with no CPU copy.

**Addresses and mappings.** `SparkDeviceAddressReserve` reserves a range at
an address the backend chooses; nothing is requested at a fixed address.
- `SparkDeviceMemoryMap` maps `bytes` of the memory, from `offset`, at an
  address inside one reservation.
- `access` grants `SPARK_DEVICE_ACCESS_DEVICE_READ` or
  `SPARK_DEVICE_ACCESS_HOST_READ`, or both. `SPARK_DEVICE_ACCESS_DEVICE_WRITE`
  requires device read and `SPARK_DEVICE_ACCESS_HOST_WRITE` requires host read
  (`host write access needs host read access`).
- Host access needs the host memory share property and a handle that
  `SparkDeviceMemoryCreateHost` returned in this process. Any other handle,
  imported host memory included, and any handle on a backend without the
  property, is an invalid argument: `host access maps only host memory`. The
  backend keeps the handles it created and never asks the driver about a
  handle it does not know.
- A failed map leaves nothing mapped.
- `SparkDeviceMemoryUnmap` takes one or more whole, adjacent mappings. It
  refuses part of a mapping.
- `SparkDeviceAddressRelease` returns an unmapped reservation.

**Lifetime.** Memory lives while any handle, mapping or exported descriptor
refers to it, in any process. Releasing a handle while its memory is mapped
keeps the memory and its contents.

**Copies.** A mapped range enters the buffer API with `SparkDeviceAdopt` as
`SPARK_DEVICE_SPACE_DEVICE_PRIVATE` with no host pointer. Copies, fills and
kernel bindings then use it like any other buffer. `SparkDeviceHostOf` on
it fails.
- The adopted range may cover parts of its reservation that are not mapped
  yet, as a pool that maps chunks as it grows does; only the mapped parts
  may be touched.
- The host and nvrm backends, which keep their reservations, refuse a range
  that is neither inside one reservation nor across adjacent mappings. CUDA's
  adopt takes any device range.

weightd copies pack bytes and the arena epoch into its own mappings with
`SparkWeightdDeviceWrite`
([`include/sparkpipe/spark_weightd_device.h`](../include/sparkpipe/spark_weightd_device.h)).
When no buffer holds the destination, it adopts exactly the range it writes,
copies, and releases the view. A failed write fails the load or the eviction
that made it, the epoch's included.

## Coherence

Shared memory has no CPU view. No entry point returns a host pointer for it.
On GB10 the backend's memory is GPU-cached and not coherent with the CPU,
while GPU engines are coherent with each other through the shared L2. So:

1. **CPU data moves only by copy.** It reaches shared memory and leaves it
   only through copy-engine copies (`SparkDeviceCopy`, `SparkDeviceCopySync`,
   `SparkGpuMemcpy`).
2. **The completion point is defined.** A synchronous copy returns after its
   writes have landed. `SparkDeviceCopy` is complete after
   `SparkDeviceQueueWait` on its queue.
3. **Readers start after it.** A reader in another process starts only after
   the writer's completion point, signalled by the writer's own protocol (for
   weightd, the arena epoch, copied after the data).

GPU work in different processes needs nothing beyond that ordering. Every
mapping of a memory object uses the same caching attribute, and the L2 is
physically addressed. A caller that ever needs the CPU to read or write
shared memory directly first gets explicit flush and invalidate entry
points; until then the CPU cannot touch it.

## Thread binding

`SparkDeviceThreadInstance` returns the device-layer instance the calling
thread is bound to, and fails when it is bound to none.
`SparkDeviceThreadBind` binds the calling thread to an instance that
`SparkDeviceThreadInstance` returned, and refuses any other value. A worker
thread binds to its creator's instance before it touches device memory.

## Backends

| Backend | Shared memory | Thread binding |
| --- | --- | --- |
| CUDA | the driver's virtual memory management, reached through `cudaGetDriverEntryPointByVersion`, so the device library links only the CUDA runtime. Pinned device allocations exported as POSIX file descriptors. Host memory is a `CU_MEM_LOCATION_TYPE_HOST_NUMA` allocation at the device's host NUMA node, offered when the device reports `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`; host access adds a host-NUMA access descriptor. Mappings take offset 0 only, as the driver requires | an instance is the device's primary context |
| host | POSIX shared memory objects. Reservations are `PROT_NONE` anonymous mappings and maps are `MAP_FIXED` mappings of the object. It adopts device-private ranges inside one of its reservations or across its mappings | one instance per process |
| Metal | refused with `SPARK_STATUS_UNSUPPORTED` | one instance per process |
| nvrm | no host memory share yet (a system-memory object exported through the same NV0000 path, with a CPU mapping when host access is asked for, is the runtime track's next item). Device memory: module system memory, GPU-cached, in 2 MiB pages, with no CPU mapping. Export puts the object in a fresh `/dev/nvidiactl` descriptor (`NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD`); import duplicates it under this process's device (`..._IMPORT_OBJECT_FROM_FD`) and reads its size back. Reservations come from the backend's own device address range, and maps are an RM mapping plus a unified-memory external mapping at the address, with the offset asked for. The backend keeps its handle until the last unmap and the release. Both granularities are 2 MiB | one instance per process |

## Tests

[`tests/test_device_shared_memory.c`](../tests/test_device_shared_memory.c)
runs in `make test` against the host backend. It covers:

- refusals before initialization and for bad arguments;
- a read-write mapping filled and read back by copies;
- no host view of the adopted range;
- export, then release of the handle while mapped;
- another process, started by `exec`, importing the memory, reading every
  byte through a read-only mapping and writing through a read-write one,
  with the exporter then reading those writes;
- two allocations mapped next to each other, copied across, refused a
  partial unmap, and unmapped by one call;
- a second thread binding to the instance, and refusing a value the device
  layer did not return.

[`tests/test_device_contract.c`](../tests/test_device_contract.c) checks host
memory on every backend: without the property, the refusal and its text; with
it, creation in whole granules, a CPU-only mapping filled by the holder, export,
import, a device-read-only mapping the device reads back, and the access
refusals above.

The nvrm build of the same test (`build/test_device_shared_memory_nvrm`) and
[`tests/nvrm/shared_coherence.cu`](../tests/nvrm/shared_coherence.cu) run in
`make test-nvrm-gpu`. The coherence test runs three 1 GiB cases across two
processes, and each case also runs as a mutant that drops its completion
point; every mutant must be caught as a mismatch:

1. the exporter fills the memory from pageable memory by copy, and a kernel in
   the importer checksums every byte through a read-only mapping;
2. 64 kernels in the importer write through a read-write mapping, and the
   exporter reads it back by copy;
3. the importer writes by `SparkDeviceCopy` through an adopted range and
   waits on its queue, and the exporter reads it back by copy (the KV pool's
   path).

It also checks that no adopted range has a host view, and runs a thousand
rounds of the whole lifetime: create, map, export, release while mapped,
import, map read-only, unmap and release. Every round keeps its bytes and
reserves the same address again.

The leak check compares exact counts taken before and after the rounds:

- the nvrm backend's driver objects, shared memories, reservations and
  mappings (`SparkNvrmResourceCountsRead`);
- the process's pinned and locked pages (`VmPin`, `VmLck`);
- the process's open descriptors.

All of them must end where they started. A handle kept past the rounds
must show in those counts until it is released, which shows that a
leaking round would be caught. The host's `MemAvailable`, with 256 MiB
of slack, is only a secondary check: other processes and the page cache
move system memory counters, so they cannot carry the check alone.

The writer in case 2 is 64 grids, which catches the mutant whichever engine
the reader's copy runs on. When nvrm copies still ran on the graphics
channel, a reader's copy queued behind another process's single 30 ms grid
even without the completion point. Copies now run on their own copy engine
([NVRM_BACKEND.md](NVRM_BACKEND.md), queues).

[`tests/test_device_shared_memory_cuda.py`](../tests/test_device_shared_memory_cuda.py)
runs the same binary against the CUDA backend on a CUDA host.
