# nvrm device backend

`runtime/device/nvrm/` is a device backend that drives an NVIDIA GPU through
the open kernel module's device files (`/dev/nvidiactl`, `/dev/nvidia<N>`,
`/dev/nvidia-uvm`) and its resource-manager ioctls, with no vendor
user-space library. It implements the backend interface in
`runtime/device/device_backend.h` that the host, CUDA and Metal backends
implement, so code above `include/sparkpipe/spark_device.h` runs on it
unchanged. What it does not do yet is listed in
[`TECHDEBT.md`](../TECHDEBT.md#hardware-independence).

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
