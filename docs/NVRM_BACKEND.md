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
  section order at their own alignment, at least 128 bytes; nothing else
  is loaded. A loaded section that asks for a fixed address is refused.
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
  shared, stack and large-parameter kernels, plus two RMS-norm kernels from
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
   and subdevice.
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

Memory comes in three kinds, all mapped into the GPU at the same address
the CPU sees, below 2^40:

- **Host memory.** Anonymous pages the backend maps itself and
  registers by OS descriptor.
- **System memory.** Allocated by the module, GPU-cached, with 2 MiB
  pages from 8 MiB up.
- **Uncached system memory.** Allocated by the module, GPU-uncached, for
  notifiers.

`nvrm_channel.c` writes push-buffer methods, GPFIFO entries, GPPut and the
doorbell, host semaphore releases (with an optional GPU timestamp), and
waits on a semaphore value with a timeout. It provisions the
per-thread local memory a program needs. `nvrm_launch.c` loads a cubin
into host memory, relocates it, and queues a launch: the QMD and constant
bank 0 in a 256-byte aligned slot, then `SEND_PCAS_A` and the scheduling
method.

`tests/test_nvrm_submit.c` checks, without a GPU, that methods, ring
entries, GPPut, the doorbell, semaphore releases and launch slots are
encoded as the class headers say, and that each misuse is refused.

## Submission and launch shape

A channel's methods are appended to its push buffer and go to the GPU
only when `SparkNvrmSubmit` writes a GPFIFO entry: a submission carries
every method pushed since the previous one. Binding the compute class,
its shared and local memory windows and the local memory with its
per-TPC size are submitted (and synchronized) when they are set, and
persist for later launches. `SparkNvrmPushRewind` reuses the push buffer
once everything in it has been submitted and the caller knows the GPU
has consumed it.

Compiled kernels read their launch shape (`blockDim`, `gridDim`) from
driver words in constant bank 0. The launcher fills the words the
compiled code reads, group x, y, z at words 216-218 and grid x, y, z at
words 220-222, and leaves the rest of the driver area as it is. The
offsets belong to the compiler, not the hardware, so
`tests/test_nvrm_launch_dimensions.c` (`make test-nvrm-gpu`) pins them: it
fills the driver words with sentinels, has a kernel store its shape, and
requires each value to come from the word the launcher fills, then
checks 11 launch shapes. Rerun it for every toolkit or module change.

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
memory comes from the module with 2 MiB pages.

