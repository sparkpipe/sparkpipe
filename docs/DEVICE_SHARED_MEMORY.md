# Device shared memory

Processes share device memory through the device layer
([`include/sparkpipe/spark_device.h`](../include/sparkpipe/spark_device.h)).
weightd allocates weight arenas and the KV pool, and the engines map them.
One process creates exportable memory and passes it to another as a file
descriptor. Each process reserves its own address range and maps the memory
into it.

## Entry points

```c
SparkStatus SparkDeviceMemoryGranularity(uint32_t granularity, uint64_t *bytes);
SparkStatus SparkDeviceMemoryCreate(SparkDeviceMemoryHandle *memory, uint64_t bytes, const char *owner);
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

**Addresses and mappings.** `SparkDeviceAddressReserve` reserves a range at
an address the backend chooses; nothing is requested at a fixed address.
- `SparkDeviceMemoryMap` maps `bytes` of the memory, from `offset`, at an
  address inside one reservation.
- `access` is `SPARK_DEVICE_ACCESS_DEVICE_READ`, optionally with
  `SPARK_DEVICE_ACCESS_DEVICE_WRITE`. Host access bits are refused.
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
| CUDA | the driver's virtual memory management, reached through `cudaGetDriverEntryPointByVersion`, so the device library links only the CUDA runtime. Pinned device allocations exported as POSIX file descriptors. Mappings take offset 0 only, as the driver requires | an instance is the device's primary context |
| host | POSIX shared memory objects. Reservations are `PROT_NONE` anonymous mappings and maps are `MAP_FIXED` mappings of the object. It adopts device-private ranges of its own mappings | one instance per process |
| Metal | refused with `SPARK_STATUS_UNSUPPORTED` | one instance per process |
| nvrm | module system memory, GPU-cached, in 2 MiB pages, with no CPU mapping. Export puts the object in a fresh `/dev/nvidiactl` descriptor (`NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD`); import duplicates it under this process's device (`..._IMPORT_OBJECT_FROM_FD`) and reads its size back. Reservations come from the backend's own device address range, and maps are an RM mapping plus a unified-memory external mapping at the address, with the offset asked for. The backend keeps its handle until the last unmap and the release. Both granularities are 2 MiB | one instance per process |

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
reserves the same address again, and memory and descriptors end where
they started.

The writer in case 2 is 64 grids, which catches the mutant whichever engine
the reader's copy runs on. When nvrm copies still ran on the graphics
channel, a reader's copy queued behind another process's single 30 ms grid
even without the completion point. Copies now run on their own copy engine
([NVRM_BACKEND.md](NVRM_BACKEND.md), queues).

[`tests/test_device_shared_memory_cuda.py`](../tests/test_device_shared_memory_cuda.py)
runs the same binary against the CUDA backend on a CUDA host.
