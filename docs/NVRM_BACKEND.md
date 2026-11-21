# nvrm device backend

`runtime/device/nvrm/` is a device backend that drives an NVIDIA GPU through
the open kernel module's device files (`/dev/nvidiactl`, `/dev/nvidia<N>`,
`/dev/nvidia-uvm`) and its resource-manager ioctls, with no vendor
user-space library. It implements the backend interface in
`runtime/device/device_backend.h` that the host, CUDA and Metal backends
implement, so code above `include/sparkpipe/spark_device.h` runs on it
unchanged. What it does not do yet is listed in
[`TECHDEBT.md`](../TECHDEBT.md#hardware-independence).

## Build flavours

`SPARKPIPE_DEVICE` selects the build flavour: `cuda` (the default, the build
as it has always been) or `nvrm`. Any other value stops the build. The nvrm
device library, `build/runtime_libs/libsparkpipe_device_nvrm.so`, builds in
both flavours (`make nvrm-device`, and as part of `make all`) from
`runtime/device/device.c` and `runtime/device/nvrm/*.c` with the include
paths below; it links no vendor library and takes no toolkit include path.

`make SPARKPIPE_DEVICE=nvrm` accepts only the goals ported to it,
`nvrm-device`, `test-nvrm` and `clean`, and stops naming any other goal:
the engine, its modules and tools still call the CUDA runtime, so an nvrm
build of them would not be one. The list grows as targets move over.

Every backend entry the backend does not implement yet fails with
`SPARK_STATUS_UNSUPPORTED` through `SPARK_FAIL`, naming the operation and
the module release; no entry reports success for work it did not do.
`tests/test_device_nvrm.c` checks that on a host without the module's
control device the backend refuses to open, names itself, and leaves the
device closed; `tests/test_device_flavours.py` checks the flavour switch,
the refused goals, that the default build is the cuda flavour, and the
library's dynamic dependencies.

## One module version

The resource-manager ioctl structures change between module releases, so the
backend is built for exactly one release of the open kernel module and
refuses any other when it opens the device. The pinned release is
**580.178.04**: `runtime/device/nvrm/nvrm_version.h` names it, and the
generated headers below come from it. A module update is a regeneration, a
rebuild and a requalification of the backend, like any other release.

## Generated headers

The structures, ioctl numbers, class numbers, methods and QMD layout come
from two pinned upstream trees, copied by `tools/nvrm_headers.py`:

| Upstream | Pin | Provides | Output |
| --- | --- | --- | --- |
| [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | release 580.178.04, commit `c8e699821c23` | the ioctl numbers and structures, the UVM ioctls, the resource-manager SDK (`nvos.h`, status codes, allocation and control parameters), class numbers, and the GPFIFO entry and host method formats | `runtime/device/nvrm/vendor/open-gpu-kernel-modules/` |
| [open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc) | commit `9fdf5c406200` | the Blackwell compute methods (`clcec0.h`), its QMD version 5 layout (`clcec0qmd.h`) and the copy-engine methods (`clcab5.h`) | `runtime/device/nvrm/vendor/open-gpu-doc/` |

The tool copies the headers it lists and every header they include, keeps
each one's path relative to its include root, removes the comments and
writes the matching `NOTICE` entries. Only nvrm translation units compile
with these include paths:

```
-Iruntime/device/nvrm/vendor/open-gpu-kernel-modules -Iruntime/device/nvrm/vendor/open-gpu-doc
```

To regenerate, clone both trees at their pins into a new directory outside
the repository and run the tool from the repository root; `--check` renders
into memory and fails when the tree differs:

```
git clone --depth 1 --branch 580.178.04 https://github.com/NVIDIA/open-gpu-kernel-modules DIR/open-gpu-kernel-modules
git clone https://github.com/NVIDIA/open-gpu-doc DIR/open-gpu-doc
git -C DIR/open-gpu-doc checkout 9fdf5c4062007929d9f4e6cbad9c9771fe61b880
python3 -I tools/nvrm_headers.py --module DIR/open-gpu-kernel-modules --doc DIR/open-gpu-doc --check
```

The tool refuses a checkout at any other commit. `tests/test_nvrm_headers.py`
runs in `make test` without the upstream trees: it checks the tool's comment
removal, include resolution and `NOTICE` rewriting on scratch trees, and
that the headers in the tree, their `NOTICE` entries and
`nvrm_version.h` all name the pins.

## Kernel images

The backend runs the cubins nvcc builds for the release target. They
are ELF images, and `runtime/device/nvrm/nvrm_cubin.c` reads them into a
kernel table and one loadable image:

- **Image.** The allocated `PROGBITS` sections (code and constant banks) are laid out in
  section order at their own alignment, at least 128 bytes. Allocated
  `NOBITS` sections (a module's zero-initialized `__device__` variables)
  get zeroed space in the same layout, except the per-block
  `.nv.shared.<name>` sections, which only size shared memory. Nothing else
  is loaded. A loaded section that asks for a fixed address is refused.
  `tests/test_nvrm_module_state.c` (`make test-nvrm-gpu`) checks on the GPU
  that a module's `__device__` variable starts zeroed and keeps its value
  across launches, and that its `__constant__` table reads back.
- **Kernels.** A kernel is a `.nv.info.<name>` section that carries a
  parameter-bank record. Its code is `.text.<name>`. Its static shared memory is the size of
  `.nv.shared.<name>`, which includes the 1 KiB reserved per block. Its
  register count and minimum stack come from the function records in
  `.nv.info`, matched to the kernel through the symbol table. Its constant
  banks are `.nv.constant<N>.<name>`, or `.nv.constant<N>` when every kernel
  shares the bank, and at most 8 of them are allowed.
- **Parameters.** They start at the offset in the parameter-bank record and
  run to the end of `.nv.constant0.<name>`. On Blackwell they start at
  0x380, after the 224 driver words.
- **Relocations.** These are the 64-bit address and the low and high 32-bit address
  fields. They target loaded sections only and resolve against loaded
  symbols with the addend. `SparkNvrmCubinRelocate` writes them for a given
  load address.

Anything else (an unknown relocation, a record cut short, a missing
register count, parameters outside bank 0) fails with
`SPARK_STATUS_PARSE_ERROR` or `SPARK_STATUS_UNSUPPORTED`, naming the
cause, against `library_load`.

Every cubin ships with a manifest beside it (`<name>.manifest` next to
`<name>.cubin`), written by the same compile
(`tools/nvrm_cubin_fixtures.py` for the fixtures). It names the cubin by
size and sha256 and lists, per kernel:

- **The compiler's resource report:** registers, barriers, stack frame,
  spills and static shared memory, from `ptxas -v`.
- **The parameter table:** each parameter's PTX type, alignment, size and
  offset, in declaration order. It comes from the `.param` declarations of
  the PTX that nvcc fed ptxas in that compile (kept with `-keep`), laid out
  as the PTX ISA defines kernel parameters.
- **Launch bounds and cluster directives:** `.maxntid`, `.reqntid`,
  `.minnctapersm`, `.maxnreg`, `.reqnctapercluster`, `.explicitcluster` and
  `.maxclusterrank`.

`SparkNvrmProgramLoadFile` loads a cubin only with its manifest, and
`nvrm_manifest.c` checks the two against each other: the digest and
size, the same kernels, and each kernel's registers, stack and parameter
table ending exactly where its constant bank does. A mismatch fails with
`SPARK_STATUS_TARGET_MISMATCH` against `library_load`. A launch takes its
barrier count from the manifest, and at least one.

## Launch descriptors

`runtime/device/nvrm/nvrm_qmd.c` packs a Blackwell compute launch
descriptor (QMD version 5, 96 words) from a launch shape, using the bit
ranges of `clcec0qmd.h`:

- **Shape.** The grid and the group, the registers and the barriers.
- **Memory.** The shared memory size and the SM shared-memory configuration
  (32, 64 or 100 KiB), and the per-thread local memory.
- **Bindings.** The program address and its prefetch, up to 8 constant
  banks, and an optional release of a 64-bit payload (two words) or a
  timestamp (four words) to a semaphore.

It refuses a shape the hardware cannot run (no groups, more than 1024
threads, misaligned addresses, more shared memory than a configuration
offers) and any value that does not fit its field, naming the field.
`SparkNvrmBank0Build` writes constant bank 0: the shared and local memory
windows and the last driver word, then the argument block at the kernel's
parameter offset, refusing an argument block of any other size.

## Host tests

The parser and the packer are tested in `make test` without a GPU:

- **`tests/test_nvrm_cubin.c`.** It parses
  `tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin`, release-built from
  `tests/nvrm/fixture_kernels.cu` (first-light, parameter-layout, static
  shared, stack, 4104-byte and 32,764-byte parameter and 2000-byte name
  kernels, plus two RMS-norm kernels from
  `inference/kernels/norm.cuh`). Each kernel's registers, stack and shared
  memory must equal ptxas's own report in the fixture manifest, and its
  parameter bytes the C layout of its signature. A synthetic image then
  covers relocations and every refusal.
- **`tests/test_nvrm_qmd.c`.** It reads every packed field back with an
  independent bit reader, pins several descriptor words, and covers the
  refusals and constant bank 0.
- **`tests/test_nvrm_fixtures.py`.** It checks that the fixtures match their
  source and the release build's nvcc flags.

Rebuild the fixtures on a host with the release nvcc:

```
python3 -I tools/nvrm_cubin_fixtures.py --nvcc /usr/local/cuda/bin/nvcc
python3 -I tools/nvrm_cubin_fixtures.py --nvcc /usr/local/cuda/bin/nvcc --check
```

## Opening the device

`runtime/device/nvrm/nvrm_rm.c` and `nvrm_gpu.c` open a GPU through the
module's device files:

1. **Version check.** Open `/dev/nvidiactl` and check the module version
   strictly (`NV_ESC_CHECK_VERSION_STR`). A mismatch fails with
   `SPARK_STATUS_TARGET_MISMATCH`, naming the loaded version and the one
   the backend is built for.
2. **Device.** Allocate a root client and pick the GPU from the module's
   card list. Open and register `/dev/nvidia<minor>` before asking for the
   GPU's identity: the first open of a GPU's device file initializes it,
   and until then the module knows no GPU by that id (on a host without a
   persistence daemon nothing else has opened it). Then allocate the device
   and subdevice. `tests/test_nvrm_open_order.c` (Linux,
   `make test-nvrm`) stands in for the module with link-time wrappers of
   `open` and `ioctl` that refuse the identity query until the device file
   is open and registered, and fails on the old order.
3. **Classes.** Require the Blackwell compute B, GPFIFO A, copy B and Hopper
   usermode classes from the device's class list, and read the GPC, TPC,
   SM and warp counts and the SM version.
4. **Address space.** Allocate a virtual memory object, map the usermode
   page (the doorbell), and allocate a GPU address space owned by unified
   memory. Register the GPU and the address space with `/dev/nvidia-uvm`.
5. **Channel.** Allocate a channel group and a context share. Build one
   compute channel from a 1024-entry GPFIFO ring and its USERD page, an
   error notifier, a 1 MiB push buffer and a fence semaphore. Read its
   work-submit token, register it with unified memory and schedule the
   group. Bind the compute class with its shared and local memory windows.

Every RM status is turned into a `SparkStatus` naming the RM status code
(`NV_ERR_NO_MEMORY` becomes `CAPACITY_EXCEEDED`, an unknown class
`UNSUPPORTED`, and so on).

At open the backend reserves one range of process addresses,
64 GiB to 1 TiB (`nvrm_address.c`), and hands out GPU addresses from it
first-fit, with released spans merged and used again. Everything the GPU
reaches lives in this range, so push buffers stay below 2^40 and a GPU
address that the CPU also maps is the same number on both sides. A range
another mapping already holds is refused rather than replaced.

Memory comes in five kinds:

- **Host memory.** Anonymous pages the backend maps itself at the GPU
  address and registers by OS descriptor. It is CPU-cached and coherent.
- **System memory.** Allocated by the module, GPU-cached, with 2 MiB
  pages from 8 MiB up. It is not coherent with the CPU: the copy engine
  and the SMs see each other's writes through the GPU's L2, but the CPU's
  view misses writes still held there. A coherent mapping (CPU-cached and
  snooped) fixes that but cost 38% of read and 29% of copy bandwidth on
  GB10, so device memory stays non-coherent and the device layer never
  gives it a host view.
- **Uncached system memory.** Allocated by the module, GPU-uncached, for
  notifiers.
- **User memory.** Pages the caller already has (`SparkDeviceRegister`),
  registered by OS descriptor from the page that holds the first byte.
  The module pins every page when the descriptor is made.
- **Read-only user memory.** The same for a file mapping
  (`SparkDeviceMapFile`): pinned without write access, and mapped into
  the GPU read-only.

`nvrm_submit.c` writes push-buffer methods, GPFIFO entries, GPPut and the
doorbell, host semaphore releases and acquires, and waits on a semaphore
value with a timeout. `nvrm_channel.c` builds channels and provisions the
per-thread local memory a program needs. `nvrm_copy.c` queues copy-engine
copies and fills. `nvrm_launch.c` loads a cubin into host memory,
relocates it, and queues a launch: the QMD and constant bank 0 in a
256-byte aligned slot, then `SEND_PCAS_A` and the scheduling method.

`tests/test_nvrm_submit.c` checks, without a GPU, that methods, ring
entries, GPPut, the doorbell, push-buffer reuse, ordered copies and
fills, semaphore releases and launch slots are encoded as the class
headers say, that the address map hands out, reuses and merges spans,
and that each misuse is refused.

## Submission and launch shape

A channel's methods are appended to its push buffer and go to the GPU
only when `SparkNvrmSubmit` writes a GPFIFO entry: a submission carries
every method pushed since the previous one. Every submission ends with a
release of the channel's 64-bit fence, its own or one the submission
adds, and the channel remembers each submission's span of the push
buffer with its fence value. The push buffer is a ring: when the next
method does not fit before the end, the unsubmitted methods move to the
start, after the backend has waited for the GPU to finish the
submissions they would overwrite and no longer. The GPFIFO ring is
reused the same way.

Binding the compute class (subchannel 1), the copy class (subchannel 4)
and the shared and local memory windows is submitted and synchronized
when a channel is made. A launch binds the current local memory on its
own channel when that channel has not seen it yet. Growing the local
memory allocates a larger one and keeps the old one until the device
closes, because launches on other channels may still use it.

Compiled kernels read their launch shape (`blockDim`, `gridDim`) from
driver words in constant bank 0. The launcher fills the words the
compiled code reads, group x, y, z at words 216-218 and grid x, y, z at
words 220-222, and leaves the rest of the driver area as it is. The
offsets belong to the compiler, not the hardware, so
`tests/test_nvrm_launch_dimensions.c` (`make test-nvrm-gpu`) pins them: it
fills the driver words with sentinels, has a kernel store its shape, and
requires each value to come from the word the launcher fills, then
checks 11 launch shapes. Rerun it for every toolkit or module change.

A grid's release semaphore carries a 64-bit payload only with the QMD's
`RELEASE_PAYLOAD64B` bit set; without it the upper word is dropped.

## Queues, copies and fills

A queue is one channel: one GPFIFO ring with compute on subchannel 1 and
the copy engine on subchannel 4, and the channel's fence as the queue's
timeline. Each copy, fill or launch is one submission that the engine
doing the work ends by releasing the next fence value: the copy engine
with its own 64-bit semaphore release after the transfers, a grid with
its QMD release. Before work on another engine than the previous
operation's, the queue waits until the fence reaches that operation's
value. Back-to-back copy-engine work needs no wait, because each
transfer is non-pipelined.

Copies and fills go in lines of at most 2^31 bytes. A fill writes the
byte from the copy engine's remap constant: four bytes per element over
the aligned middle, one byte per element over an unaligned head and
tail.

`device_nvrm.c` maps the device layer onto this:

| Space | nvrm memory | Host view | Device address |
|---|---|---|---|
| device-private | system memory, zero-filled by the copy engine | none | yes |
| shared coherent | host memory | same address | yes |
| host pinned | host memory | same address | yes |
| host pageable | heap | yes | none |
| registered | user memory | the caller's pointer | yes |
| file-backed | read-only user memory over a read-only mapping | yes | yes |

Events are 32-byte marks carved from a pooled host allocation. Recording
one is a queue operation that releases a timestamped copy of the event's
next generation 16 bytes into the mark, then the queue's fence, then an
unsigned 64-bit maximum of the generation into the mark's first word. The
event completes when that word reaches the generation, by which time its
queue has passed the record; a later record that finishes first is never
undone by an earlier one. A queue waits on an event by acquiring the
mark's first word at or past the generation, so waits between queues
need no host round trip. Elapsed time is the difference of the two
marks' GPU timestamps.

Each queue has a host worker thread for host work. A host step releases
a reached gate the worker polls, acquires a done gate the worker writes
after running the function, then releases the queue's fence, so the
queue holds behind the host work for as long as it takes (the contract
holds one for 6 s). Host callbacks run there, and so do copies and fills
that touch pageable memory: in queue order, through an 8 MiB pinned
bounce buffer per queue when the device side has no host view. Because
one device-layer call can take several submissions, the backend maps
device-layer sequence numbers to fence values through a ring.

Kernel libraries are cubins with their manifests
(`SparkDeviceLibraryLoad` on the `.cubin` path). `SparkDeviceKernelCreate`
finds a kernel by name and reports its thread limit (the PTX bound, else
1024), its static shared memory and its parameter layout
(`SparkDeviceKernelInfo.parameters`, from the manifest). A launch passes
the kernel's whole parameter block, packed by its C parameter types, as
the launch's constants; the block must be exactly the kernel's parameter
size, and bindings are refused. A parameter block may be up to 32,764
bytes, CUDA's own kernel parameter limit on compute capability 7.0 and
later, which the backend reports as `max_constant_bytes`. Each launch's
descriptor and bank 0 (512 bytes plus the kernel's bank 0 rounded up to
256) go in a slot from a per-queue 4 MiB byte ring; the space is reused
once the launch that held it has completed. A kernel name may be up to
4096 bytes, so mangled template names are found whole; the manifest keeps
each name on the heap. The backend sets the `SPARK_DEVICE_PROPERTY_KERNEL_LIBRARIES` and
`SPARK_DEVICE_PROPERTY_PARAMETER_BLOCKS` property bits (Metal sets the
first), and the device contract branches on them: its parameter-block
branch loads the fixture cubin,
checks the reported layout, refuses bindings and a wrong-sized block, and
reads back a launch. It also creates, launches and destroys a kernel with
a 2000-byte name, launches a 4104-byte and a 32,764-byte parameter block
and reads every word back, and refuses a block 4 bytes over the limit.
`SPARK_DEVICE_PROPERTY_SHARED_ADDRESS` is always set. The backend sets
`SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY` only when the module reports the GPU as
an SoC: bit `NV0000_CTRL_GPU_ID_INFO_SOC` of the identity's `gpuFlags`,
from `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2`. GB10 reports it. A discrete GPU
does not, so on one the backend makes no claim of host-coherent memory,
as the CUDA backend does from its integrated flag. The backend adopts no
foreign memory or queue, and a queue it owns is never captured into a
graph.

Every channel registers with unified memory before anything is
submitted on it, and the backend never schedules a single channel: the
module refuses to schedule a graphics channel in a unified-memory address
space whose context has not been bound by that registration ("Cannot
schedule externally-owned channel ... with unbound allocations"). The
channel group is scheduled once, after its first channel is registered;
queue channels join the scheduled group afterwards.

`make test-nvrm-gpu` reads the kernel log before and after the run
(`tools/nvrm_kernel_log.sh`, with `dmesg` or `sudo -n dmesg`) and fails if
the module logged any `NVRM:` line meanwhile that reports an Xid, an
assertion, a refusal ("Cannot ..."), an error, a failure or a fault, or if
the log cannot be read.

The GPU tests that run under `make test-nvrm-gpu`, besides the
launch-shape probe and first light:

- **`tests/test_nvrm_memory.c`** opens the device through the device
  layer and checks every space, address reuse, zero-filled fresh memory,
  copies and unaligned fills through every space and registered memory,
  a copy and a fill of more than 2^31 bytes, a mapped file read from an
  unaligned offset, pageable copies through the bounce buffer and pageable
  fills, 64 queues opened and closed in turn, two queues copying at once,
  and reopening after a shutdown.
- **`tests/test_device_contract.c`** (`build/test_device_contract_nvrm`),
  the backend-neutral device contract the host, Metal and CUDA backends
  pass: spaces, copies, events, waits between queues, host callbacks, busy
  buffers and events, holds longer than a GPU watchdog, 20000 event
  record-and-destroy rounds, timing, file mappings and registration.
- **`tests/test_nvrm_queue_order.c`** is the ordering contract: a
  copy-engine fill then a kernel reading it, a kernel writing then a
  copy-engine copy of it, 32 rounds each, then 32 back-to-back sets of
  fill, kernel and copy with no host wait, and a copy on a second queue
  that waits on the first queue's mark after a kernel there. It reads
  results back through the copy engine into host memory, and starts the
  fence just below 2^32 so every engine's release crosses it.

On GB10 both pass. With the copies moved to a second channel and no wait
between the channels, a copy after a kernel reads stale data in 32 of 32
rounds and every back-to-back set is wrong, so the test sees misordering.
Within one channel the hardware orders an engine switch by itself (the
test passes with the queue's wait removed). The queue keeps the wait,
which is what the class headers document.

## Kernel registration

`runtime/device/nvrm/nvrm_register.c` defines the registration entry points
that nvcc's host stubs call, so an nvcc-compiled module links against the
nvrm library with `-cudart none` and no NVIDIA user-space library. The set
is what an object compiled that way leaves undefined: `__cudaRegisterFatBinary`,
`__cudaRegisterFatBinaryEnd`, `__cudaUnregisterFatBinary`,
`__cudaRegisterFunction`, `__cudaRegisterVar`, `__cudaPopCallConfiguration`,
`__cudaGetKernel` and `__cudaLaunchKernel`. Their declarations are the CUDA
headers' (`crt/host_runtime.h`, `crt/device_functions.h`).
`__cudaRegisterManagedVar`, `__cudaInitModule`,
`__cudaPushCallConfiguration`, `__cudaLaunchKernel_ptsz` and
`cudaLaunchKernel` are defined too.

- **Modules.** The first fat binary a module registers finds that module from
  its caller's address. On Linux this uses `dl_iterate_phdr`, and the main
  program is `/proc/self/exe`; elsewhere it uses `dladdr`. The return
  address is read in a frame with no large locals: a function with a 4 KiB
  path buffer gets a stack-probe call at entry, and that call overwrites the
  link register first. The module's kernels live in `<module path>.kernels/`.
  Every `.cubin` there needs its `.manifest`, and every manifest its cubin.
  Other files, such as compiler leftovers from `--keep`, are ignored. The
  manifests are parsed on the spot (host work only, safe in a constructor).
  A kernel name in two cubins of one module is refused. So is a missing
  directory, an unpaired file, a directory with no cubin, or a manifest
  that does not parse.
- **Failures are sticky.** A module that fails prints one line naming its
  path and the cause. It keeps the failure, and every launch of its kernels
  returns it. Later fat binaries of the same module reuse the first scan.
- **Functions and variables.** `__cudaRegisterFunction` maps the host stub's
  address to the kernel's mangled name. A name no cubin of the module lists
  is refused, naming it. `__cudaRegisterVar` records the host shadow's
  address and the symbol's name. Any host access to it
  (`SparkNvrmRegisteredVariable`) is refused, naming the symbol: the engine
  makes no symbol calls. A managed variable fails its module, because the
  hosted runtime provides no managed memory.
- **Lazy kernels.** The device is not initialized while constructors run, so
  `SparkNvrmRegisteredKernel(host, &kernel)` loads the kernel's cubin as a
  device library and creates the kernel on first use, then caches it for
  the process. `SparkNvrmRegisteredRelease` destroys the cached kernels and
  unloads their libraries, so `SparkDeviceShutdown` finds none live; the
  next launch creates them again. Unregistering a module's last fat binary
  releases its kernels and forgets its entries.
- **`<<<>>>` aborts.** The `<<<>>>` entry points, the runtime-API
  `cudaLaunchKernel` and `__cudaInitModule` abort, naming the kernel where
  they know it. Every launch goes through `LM_LAUNCH`. A push-configuration
  failure that merely returned an error would make `<<<>>>` skip its launch
  without a word.

`tests/test_nvrm_register.c` covers each refusal, the name and variable
lookups, a fat binary registered from the test program itself, and
unregistering; each abort is checked in a child process. Its
`build/test_nvrm_register_gpu` build (in `make test-nvrm-gpu`) also launches
a registered kernel. It checks that the kernel is cached, that a
2000-byte-named kernel resolves, and that after the release the device
shuts down with no live library.

## First light and memory bandwidth

Two tools run on a GPU host with the pinned module:

- **`build/nvrm_first_light`** (`make nvrm-first-light`) opens the GPU and
  prints its identity. It launches the fixture's first-light kernel 16
  times (4 groups of 128 threads), waits for both the grid's release and
  a timestamped channel release, and checks every output word. It
  reports the host submit-to-completion time and the GPU time around the
  grid.
- **`build/nvrm_memory_bandwidth`** (`make nvrm-memory-bandwidth`) measures
  the stream read and copy kernels over each memory kind between two
  timestamped releases. `build/cuda_memory_bandwidth` measures the same
  kernels over `cudaMalloc` memory on the CUDA flavour, for the
  comparison.

Measured on GB10 (1 GiB buffers, 5 interleaved rounds of 20 launches,
median of medians): module-allocated system memory, GPU cached, 2 MiB
pages, reads at 262.5 GB/s and copies at 228.4 GB/s, within 0.5% of
`cudaMalloc` (261.3 and 228.3 GB/s). Host memory registered by OS
descriptor in 4 KiB pages copies at par but reads 18% slower, so device
memory comes from the module with 2 MiB pages. Remeasured with queues in
place (5 interleaved rounds, median): system memory reads at 262.5 and
copies at 228.2 GB/s against `cudaMalloc`'s 262.1 and 228.1.

