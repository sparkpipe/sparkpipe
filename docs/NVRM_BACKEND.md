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
  parameter-bank record, or that has a `.text.<name>` section of its own: a
  kernel without parameters has no parameter-bank record, and its
  parameters are taken to start at the end of bank 0, 0 bytes long. Its
  code is `.text.<name>`. Its static shared memory is the size of
  `.nv.shared.<name>`, which includes the 1 KiB reserved per block. Its
  register count and minimum stack come from the function records in
  `.nv.info`, matched to the kernel through the symbol table. The
  minimum-stack record (0x12) is the whole call chain's stack: for a kernel
  that calls a function with a 4200-byte frame, ptxas reports a 0-byte frame
  and 4200 bytes of cumulative stack, and the record holds 4200. Its constant
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

- **The compiler's resource report:** registers, barriers, the kernel's
  own stack frame, spills, static shared memory and, whenever ptxas reports
  it, the cumulative stack of the kernel's call chain (`cumulative C` at the
  end of the kernel line), from `ptxas -v`. A frame line counts only under
  the "Function properties" block that names the kernel; a callee's block
  is not the kernel's. A log that reports one kernel twice for the target
  with different numbers, or a cumulative stack below the kernel's frame,
  is refused. `tools/nvrm_kernel_manifest.py` writes the manifest, and
  `tools/nvrm_cubin_fixtures.py` loads its code by path for the fixtures, so
  both use one parser.
- **The parameter table:** each parameter's PTX type, alignment, size and
  offset, in declaration order. It comes from the `.param` declarations of
  the PTX that nvcc fed ptxas in that compile (kept with `-keep`), laid out
  as the PTX ISA defines kernel parameters.
- **Launch bounds and cluster directives:** `.maxntid`, `.reqntid`,
  `.minnctapersm`, `.maxnreg`, `.reqnctapercluster`, `.explicitcluster` and
  `.maxclusterrank`.

`SparkNvrmProgramLoadFile` loads a cubin only with its manifest, and
`nvrm_manifest.c` checks the two against each other: the digest and
size, the same kernels, and each kernel's registers, stack need (the
cumulative stack when reported, else the frame) and parameter table ending
exactly where its constant bank does. A mismatch fails with
`SPARK_STATUS_TARGET_MISMATCH` against `library_load`. A launch takes its
barrier count from the manifest, and at least one.

Loading a module provisions per-thread local memory for its largest stack
need plus 0x240 bytes, and a launch whose need is above what was provisioned
is refused. A kernel's own frame works at any size tried (4400 bytes, every
thread of a grid that fills the GPU). A kernel whose callees keep stack
frames of their own, which is any kernel whose stack need exceeds its own
frame, is refused at kernel creation and at launch with
`SPARK_STATUS_UNSUPPORTED`, naming the kernel. On GB10 such a call faults
with an out-of-range local address whatever the per-thread size, up to
16 KiB, while an indirect call to a frameless callee runs. What the hardware
needs for callee stacks is not known from an allowed source yet.

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
  shared, stack, 4400-byte frame, call-chain stack, 4104-byte and
  32,764-byte parameter and 152-character mangled name
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

A queue is two channels, each with its own GPFIFO ring and fence:

- **Graphics channel.** It sits in the process's graphics channel group,
  with compute on subchannel 1. Launches, host steps and event records
  and waits run here.
- **Copy channel.** It sits in the process's copy channel group, on the
  first copy engine whose capabilities do not mark it as the graphics
  engine's own (`NV2080_CTRL_CMD_CE_GET_CAPS_V2`, the GRCE bit), with the
  copy class on subchannel 4. Copies and fills run here.

The copy engine is separate hardware with its own runlist, so a copy
overlaps graphics work, including a running grid in another process. On
the graphics channel's own copy subchannel, the copy waited for another
process's grid to end: 46 ms behind a 50 ms grid, against 1.1 ms alone.
The copy group needs no context share, and the module refuses a copy
channel that has one, but each copy channel is still registered with
unified memory. A GPU with no copy engine outside the graphics engine is refused
at initialization.

Each copy, fill or launch is one submission that the engine doing the work
ends by releasing its channel's next fence value: the copy engine with its
own 64-bit semaphore release after the transfers, a grid with its QMD
release. The queue stays totally ordered:

- **Across channels.** When an operation goes to the other channel than
  the queue's previous one, it first acquires that channel's fence at the
  previous operation's value, unless the host has already seen it there.
- **Within the graphics channel,** a change of engine waits for the
  channel's own fence.
- **Within the copy channel,** back-to-back copy-engine work needs no wait,
  because each transfer is non-pipelined.
- **Completion.** Each device-layer sequence records the channel and value
  of its last operation. A sequence is complete when that channel's fence
  reaches the value, which by the ordering means every earlier operation
  is complete too.

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

Every host wait on the GPU goes through one waiter
(`runtime/device/nvrm/nvrm_wait.c`). This covers queue and event waits,
the fence waits behind ring space and synchronization, and the host
worker's wait for its reached gate.
- **Spin.** The waiter spins on the value for 100 us, reading the clock
  every 64 polls. A launch's whole round trip on GB10 is about 6 us, so
  short waits finish inside the spin.
- **Sleep.** Past the budget it sleeps 2 us, doubling to an 8 us
  ceiling. A long wait then finishes at most about 11 us after the value
  lands: 8 us of sleep plus about 3 us to wake. While sleeping, the
  waiting thread costs about a fifth of a core, where a spin would cost
  all of it.
- **Timer slack.** While it sleeps, the waiting thread's timer slack is
  1 us, and the thread's own slack is restored when the wait ends. Under
  Linux's default 50 us slack, a 20 us sleep lasts about 72 us.
- **Timeout.** A timeout is checked at the same points and a sleep never
  runs past it. A zero timeout checks the value once.

All of these are constants, not settings.

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
once the launch that held it has completed.

Kernel names are at most 64 characters, so the runtime stores and compares
only short names:

- **A name that fits is its own short name.** A longer one, which in
  practice is a C++ mangled template symbol, becomes its base identifier
  (the last source name of an Itanium nested name, otherwise the name
  itself), truncated, then `_` and the first 12 hex digits of sha256 over
  the full name. `SparkNvrmShortName` implements the rule, and
  `tools/nvrm_kernel_manifest.py` implements the same one.
- **The manifest records it.** Each kernel line carries `short <name>`, and
  the tool refuses two kernels of one unit with one short name. The
  runtime checks every short name against its own computation for the
  symbol, so the tool and the runtime cannot drift apart.
- **The cubin parser** uses the full ELF names only while it parses, then
  keeps the short names.
- **Registration** hashes the `deviceName` nvcc passes into the same short
  name.
- **Creating a kernel.** `SparkDeviceKernelCreate` refuses a name longer
  than 64, naming its length, unless the backend maps it to a kernel of
  the library: on nvrm, the full mangled symbol of one of its kernels. That
  name is used for the lookup and never stored.

The backend sets the `SPARK_DEVICE_PROPERTY_KERNEL_LIBRARIES` and
`SPARK_DEVICE_PROPERTY_PARAMETER_BLOCKS` property bits (Metal sets the
first), and the device contract branches on them: its parameter-block
branch loads the fixture cubin,
checks the reported layout, refuses bindings and a wrong-sized block, and
reads back a launch. It also creates, launches and destroys a kernel with
its 152-character mangled symbol, refuses a 65-character name that is no kernel's symbol, launches a 4104-byte and a 32,764-byte parameter block
and reads every word back, and refuses a block 4 bytes over the limit.
`SPARK_DEVICE_PROPERTY_SHARED_ADDRESS` is always set. The backend sets
`SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY` only when the module reports the GPU as
an SoC: bit `NV0000_CTRL_GPU_ID_INFO_SOC` of the identity's `gpuFlags`,
from `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2`. GB10 reports it. A discrete GPU
does not, so on one the backend makes no claim of host-coherent memory,
as the CUDA backend does from its integrated flag. The backend adopts no
foreign memory or queue. Adopting one of the device's own live queues works
on every backend: the device layer finds the handle in its own list of
queues by comparing pointers, never by reading through the handle, and
hands back that queue as a borrow. The caller gets the same record, so the
owner's sequence numbers and capture state apply. A release while borrows
remain only returns a borrow, so the owner and the adopter may release in
either order, and the last release destroys the queue. Any other handle, a
freed queue's included, goes to the backend and is refused. This lets a
runner adopt its execution stream the same way in the nvrm flavour, where
the stream is a device queue, and in the CUDA flavour, where it is a CUDA
stream. A queue the backend owns is never captured into a graph.

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
the log cannot be read. It also fails when `/dev/nvidiactl` is missing,
before building anything, and when any of its test binaries prints a
`SKIP` line: a GPU gate that skipped a test is not a pass
(`tests/test_nvrm_gpu_gate.py`).

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
- **`tests/nvrm/queue_channels.cu`** (`build/test_nvrm_queue_channels`) checks
  the order across a queue's two channels through `spark_gpu.h`. It runs 32
  rounds on 64 MiB buffers, with no host wait inside a round: a kernel then a
  copy-engine readback, a copy-engine fill then a kernel reading it, and an
  upload then a kernel. Each kernel holds for 1 ms before it writes. With the
  cross-channel wait removed, every transition reads most words out of order;
  the device contract's 400 KB kernel, copy and kernel round did not catch it.
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

## Shared memory

The device layer's shared memory entry points
([DEVICE_SHARED_MEMORY.md](DEVICE_SHARED_MEMORY.md)) map onto the module
like this:

- **Create.** Module system memory in 2 MiB pages, GPU-cached, allocated
  without the CPU mapping the device's own system memory gets
  (`SparkNvrmSystemAllocate`).
- **Export.** Opens a fresh `/dev/nvidiactl` descriptor and attaches the
  object to it with `NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD`. The
  module keeps the exported object alive even after the exporting client
  is gone.
- **Import.** `NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD` duplicates the
  object under this process's device, and the module picks the handle.
  The size comes back from `NV0041_CTRL_CMD_GET_SURFACE_INFO` and must be a
  whole number of 2 MiB pages.
- **Reserve.** A span of the backend's own device address range, 2 MiB
  aligned. The CPU side of that range stays reserved with no access.
- **Map** (`SparkNvrmGpuMap`). A unified-memory external range, an RM
  mapping of the object at the address with the offset and length asked
  for, then the unified-memory external mapping, read-only or read-write.
  Each step that fails undoes the steps before it, so a failed map leaves
  nothing mapped.
- **Unmap** (`SparkNvrmGpuUnmap`). Frees the external range and the RM
  mapping of each whole mapping in the span.

The backend counts each object's mappings and frees the object only after
its last unmap and its release. Unified memory also duplicates the object
for each external mapping, so a mapping never outlives its memory. One
mutex guards the shared tables, and another guards the address allocator,
because weightd maps and unmaps from worker threads without the device
mutex. Adopting a device-private range inside one or more adjacent
mappings makes a buffer with no host pointer, for copies, fills and kernel
bindings.

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
- **Staged roots.** A lane loads each module by its staged name, so its
  kernels directory must sit beside it under that name.
  `tools/stage_lane.sh` and `tools/k3_lane.sh` copy every firmware entry
  from one table, and `stage-local ROOT` builds the same layout in a local
  directory. `tools/staged_kernels_audit.py ROOT` reads the section headers
  of every ELF file in a root. Each file with a `.nv_fatbin` or
  `__nv_relfatbin` section must have complete cubin and manifest pairs
  beside it under its own name. A kernels directory with no module of its
  name is refused too. The CUDA compile gate stages every lane root from
  the build and audits it. Only the serving adapter carries device code
  there: the hidden transport is host C and has no fat binary.
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
152-character mangled kernel resolves, and that after the release the device
shuts down with no live library.

## Hosted launches

A build with `-DSPARKPIPE_DEVICE_NVRM` takes the nvrm fronts:
`sparkpipe/spark_gpu.h` includes `runtime/device/nvrm/spark_gpu_nvrm.h`, and
`runtime/launch.h` includes `runtime/device/nvrm/launch_nvrm.h`. Without the
macro both stay on the CUDA fronts. Their implementation is
`runtime/device/nvrm/spark_gpu_nvrm.c`, in the nvrm library.

- **Streams.** `SparkGpuStream` is a `SparkDeviceQueue *`. The null stream
  is a process default queue, created on first use
  (`SparkNvrmGpuDefaultStream`).
- **Errors.** `SparkGpuError` takes CUDA's numbers for the
  `SPARK_GPU_SUCCESS` and `SPARK_GPU_ERROR_*` names the call sites use. Each
  thread keeps its last error. A failed call records it,
  `SparkGpuPeekAtLastError` reads it without clearing, `SparkGpuGetLastError`
  reads and clears it, and a later successful call leaves it as it was. The
  detail is in `SparkDeviceLastError()`.
- **`LM_LAUNCH`** keeps its form: `LM_LAUNCH((kernel), grid, block, shared,
  stream, args...)`, which yields no value. `LmLaunchKernel` deduces the
  parameter types from the kernel alone and takes each argument as its
  parameter's type, so arguments convert as in a call, as `<<<>>>` does: a
  bit-field by value, a literal 0 to a null pointer, an arithmetic value to
  the parameter's type. The kernel must name one function, because an
  overloaded name cannot be resolved without a call. It lays the
  parameters out at their alignment and size.
  `tests/nvrm/launch_arguments.cu` (`make test-nvrm-launch`, run by the
  CUDA compile gate) records the packed block in place of a launch and
  checks each conversion and offset, with no GPU. `SparkNvrmLaunchPacked` finds
  the registered kernel and checks the layout against the kernel's PTX
  parameter table: the count, each offset and size, and the total. A
  mismatch is refused, naming the parameter. It then launches with the
  packed block as the launch's constants.
- **Not supported yet.** `LM_LAUNCH_DEPENDENT` and `LmLaunchClusterKernel`
  are refused with `SPARK_GPU_ERROR_NOT_SUPPORTED`, naming the kernel,
  until the launch descriptor carries programmatic dependence and clusters
  (Phase 3).
- **Shared memory.** `LmKernelSharedMemoryOptIn` grants what the device has
  and refuses more.
- **Release.** `SparkNvrmGpuRelease` destroys the default queue and the
  registered kernels before `SparkDeviceShutdown`.

The rest of the front maps onto the device layer:

- **Initialization.** The first call initializes the device, unless the
  device layer already has. `SparkGpuSetDevice` picks the ordinal before
  that; once the device is in use, only that ordinal is accepted.
- **Memory.**
  - `SparkGpuMalloc` allocates device-private memory.
  - `SparkGpuMallocHost` and `SparkGpuHostAlloc` allocate pinned host
    memory, and `SparkGpuMallocManaged` allocates shared coherent memory.
    Their host and device addresses are the same, so
    `SparkGpuHostGetDevicePointer` returns the host pointer after checking
    that the memory belongs to the front.
  - `SparkGpuHostRegister` registers host memory through the device layer.
    The front keeps the host range so that copies and device-pointer
    lookups can find it by host pointer.
  - Freeing waits for the device first, as `cudaFree` does.
- **Copies and fills.**
  - A copy between memory the front knows goes on the stream's queue.
  - Pageable host memory the device layer does not know is staged through
    an 8 MiB pinned buffer, one chunk at a time. The call returns once the
    host side may be reused: the source of a copy to the device, or the
    filled destination of a copy to the host.
  - `SparkGpuMemcpy` and `SparkGpuMemset` run on the default queue and wait
    for it.
  - 2D copies and fills run row by row, or as one operation when the
    pitches equal the width.
- **Streams.**
  - Streams must be non-blocking. A blocking stream is refused, because
    the hosted runtime has no legacy default stream to synchronize it with.
  - Stream priorities are not implemented yet. `SparkGpuDeviceGetStreamPriorityRange`,
    and a stream created with a priority other than the default, return
    `SPARK_GPU_ERROR_NOT_SUPPORTED`. Ignoring the priority would run every
    lane at one priority without saying so.
  - `SparkGpuStreamQuery` and `SparkGpuEventQuery` return
    `SPARK_GPU_ERROR_NOT_READY` without recording it as the last error.
- **Events.** Events are device-layer events, and their elapsed time is
  converted to milliseconds.
- **Not supported yet.** These return `SPARK_GPU_ERROR_NOT_SUPPORTED`,
  naming the call:
  - stream capture and every graph call, until command lists exist
    (Phase 3);
  - every device attribute but the multiprocessor count;
  - `SparkGpuMemGetInfo` on a discrete GPU.
- **Free memory on a unified GPU.** `SparkGpuMemGetInfo` reports `MemFree`
  and `MemTotal` from `/proc/meminfo` at the moment of the call, each capped
  by the device budget.
  - The device budget is `SparkDeviceConfiguration.memory_budget_bytes` when
    set, else physical memory (`_SC_PHYS_PAGES` times `_SC_PAGESIZE`). The
    front's lazy initialization sets none.
  - `MemFree`, not `MemAvailable`, is what the CUDA flavour reports on GB10.
    On spark1 its free figure equalled `MemFree` read just before and just
    after the call, as ours does, and both totals were `MemTotal`.
  - `MemAvailable` would add the reclaimable page cache: 17.4 GB on spark1
    at the time.
- **Occupancy.** `SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor` takes
  the fewest blocks allowed by each per-SM limit:
  - **Warps.** 48, from the module (`NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM`).
  - **Registers.** 64K per SM, from the Blackwell Tuning Guide, allocated per
    warp in units of 256. The module answers 0 on GB10 for its
    register-bank indices.
  - **Blocks.** 24 per SM, which is what the CUDA flavour reports on GB10,
    and 24 one-warp blocks are what both flavours keep resident. The tuning
    guide's 32 for compute capability 12.0 does not hold on this device.
  - **Shared memory.** 100 KB per SM, from the programming guide's 12.x
    carveouts, in units of 128 bytes, counting the 1 KB reserved per block.

  The launcher sizes each grid's target shared-memory carveout for that
  occupancy. The minimum carveout fits one block; the target fits every
  block the other limits allow. The old rule targeted the carveout that
  fits one block, which kept fewer blocks resident than CUDA does: 3
  instead of 10 at 8 KB per block. `make nvrm-occupancy-ab` builds one
  source in both flavours and fails on any difference, both in the
  occupancy reported (169 cases) and in the residency a spinning kernel
  measures per SM (16 cases). It needs the CUDA runtime for its CUDA half,
  so it is not part of `make test-nvrm-gpu`.
- **Function attributes.** `SparkGpuFuncGetAttributes` reports a registered
  kernel's thread limit, registers, static shared memory and stack.

`tests/test_nvrm_gpu_api.c` (in `make test-nvrm-gpu`) runs every call on
the GPU. It covers every space, a 20 MiB pageable round trip through the
staging buffer, registered memory, pitched 2D copies and fills, events
across two streams with a host function, the refusals, device queries,
and the release and shutdown.

`tests/nvrm/hosted_launch.cu` is compiled with `nvcc -cudart none
-DSPARKPIPE_DEVICE_NVRM` and `-keep`. Its kernel directory gets the cubin
and a manifest written by `tools/nvrm_kernel_manifest.py`, and it links
with the C++ compiler against the nvrm library alone. It covers:

- a signature mixing a byte, a pointer, a 16-bit value, a struct, a float
  and a signed byte, given an `int`, a pointer, an `int`, the struct, a
  float, an `int` and a pointer;
- a template kernel on a created queue;
- a kernel without arguments;
- Peek and Get, and a refused launch;
- the dependent and cluster refusals;
- a disagreeing layout, an unregistered function, and the opt-in limit.

`build/test_nvrm_hosted_launch` is part of `make test-nvrm-gpu`, so the GPU
gate needs `nvcc` on the host. Without it the gate fails, naming `nvcc`.

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

