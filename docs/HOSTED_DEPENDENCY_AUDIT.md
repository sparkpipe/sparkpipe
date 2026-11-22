# Hosted dependency audit

Rented nodes run SparkPipe's own device backend over NVIDIA's open kernel
module, on a node image that carries no NVIDIA user-space software.
[`tools/hosted_dependency_audit.py`](../tools/hosted_dependency_audit.py)
checks that property for a release root, a node image and a running node.
It is the release gate for hosted release roots and images: a root or image
that fails it is not deployed to a rented node.

```bash
python3 tools/hosted_dependency_audit.py root RELEASE_ROOT
python3 tools/hosted_dependency_audit.py image --rootfs ROOTFS --release-root opt/sparkpipe/release
python3 tools/hosted_dependency_audit.py image --dpkg-status ROOTFS/var/lib/dpkg/status
sudo python3 tools/hosted_dependency_audit.py processes
```

Each failure prints one `FAIL <kind> <path>: <detail>` line, then a verdict
line. `--json` (before the mode) prints the same report as one JSON
document. The exit code is 0 for a pass, 1 for any finding and 2 when the
target does not exist. A target with nothing to audit (an empty root, an
empty process table, a status file without packages) fails.

## What fails

**Libraries.** NVIDIA's user-space libraries, by soname: `libcuda`,
`libcudart`, `libnvidia-*` (NVML, the PTX JIT, the container library and the
rest of the driver's user space), `libGLX_nvidia` and the other GL vendor
libraries, `libnvrtc`, `libnvJitLink`, `libnvptxcompiler`, `libnvvm`,
`libnvfatbin`, cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, cuDSS, cuTENSOR,
cuFile, NPP, nvJPEG, cuDNN, NCCL, NVSHMEM, CUPTI and the profiler host
libraries, NVTX, TensorRT, nvCOMP, DCGM and the CUDA injection libraries,
plus the CUDA builds of PyTorch (`libtorch_cuda`, `libc10_cuda`,
`libcaffe2_nvrtc`) and Triton. `LIBRARY_STEMS` in the tool is the list.

**Tools.** `nvidia-smi`, `nvidia-persistenced`, `nvidia-debugdump`, the MPS
control and server, `nvidia-bug-report.sh`, the container toolkit's
binaries (`nvidia-ctk`, `nvidia-cdi-hook`, `nvidia-container-cli`,
`nvidia-container-runtime` and its hook), `nvidia-settings`,
`nvidia-xconfig`, `nvidia-powerd`, `nvidia-sleep.sh`, the fabric manager,
DCGM (`nv-hostengine`, `dcgmi`) and IMEX.

| Mode | Kind | Fails when |
|---|---|---|
| root | `needed` | an ELF file's `DT_NEEDED` entry names one of the libraries |
| root, image | `soname` | an ELF file declares one of the libraries as its `DT_SONAME` |
| root | `library-string` | a file carries a loadable name of one of the libraries (a `dlopen` string, or the one a statically linked runtime opens) |
| root | `tool-string` | a file names one of the tools (a script or binary that runs it) |
| root, image | `library-file`, `tool-file` | a file or link is named as one of the libraries or tools, or a link (or any link it leads to) points at one by name |
| root, image | `toolkit-file` | a file is named as a CUDA toolkit compiler, debugger, profiler or runtime header (`nvcc`, `ptxas`, `nvlink`, `fatbinary`, `cuobjdump`, `nvdisasm`, `cicc`, `cuda-gdb`, `compute-sanitizer`, `nsys`, `ncu`, `cuda.h`, `cuda_runtime.h` and the rest of `TOOLKIT_FILES`; the kernel's UAPI header `include/linux/cuda.h`, for the Apple CUDA microcontroller, is not one) |
| root, image | `runpath` | an ELF file's `DT_RPATH` or `DT_RUNPATH` names a CUDA toolkit directory |
| root, image | `pytorch-cuda` | a `torch/version.py` declares a CUDA version |
| root, image | `triton` | a Triton package is present |
| root, image | `unparsed`, `unreadable` | an ELF file's headers cannot be parsed, or a file cannot be read |
| image | `needed` | a binary or library in the image needs one of the libraries |
| image | `package` | an installed package, or one removed with its configuration files left, is NVIDIA user space (below) |
| image | `apt-source` | an APT source names an NVIDIA, Mellanox or NGC repository, or a `canonical-nvidia` archive |
| image | `package-database`, `missing` | the image has no `var/lib/dpkg/status`, or a named `--release-root` is not in it |
| processes | `mapped` | a process's `/proc/<pid>/maps` maps one of the libraries or tools |
| processes | `unreadable` | a process's maps cannot be read (run as root); a process that exits during the scan is skipped |

A file named as one of the libraries or tools is reported by its name; the
audit does not read its contents. A symbolic link is followed hop by hop,
and fails when any target names one of the libraries or tools. In a release
root a relative target is followed while it stays inside the root, and an
absolute target is checked by its name only. In an image an absolute target
resolves inside the image, never on the host running the audit. An ELF file that declares one of the
libraries as its soname is reported from its dynamic section and is not
scanned further. Files under `lib/firmware`, `lib/modules` and `usr/src` in
an image are checked by name only.

**Packages.** A package fails when it is installed, or removed with its
configuration files left (the detail shows the state; purge it instead).
It fails when it is:

- a driver user-space library: `libnvidia-*`, and the `lib` packages of
  the CUDA libraries above (`libcublas-*`, `libnvjitlink-*`, `libcudnn*`,
  `libnccl*` and the rest of `PACKAGE_STEMS`);
- any other `nvidia-*` package: the utilities with `nvidia-smi`, the
  compute utilities, `nvidia-kernel-common`, the driver metapackages,
  `nvidia-modprobe` from NVIDIA's repository, the container toolkit, the
  DGX OS `nvidia-spark-*`, `nvidia-system-*` and `nvidia-oem-config-*`
  packages; also `xserver-xorg-video-nvidia-*`, Nsight, the HPC SDK,
  TensorRT and DCGM;
- the CUDA toolkit: `cuda` and every `cuda-*` package;
- the closed kernel module: `linux-modules-nvidia-<branch>-<kernel>` and
  `linux-objects-nvidia-<branch>-<kernel>` without `-open`, and
  `nvidia-dkms-<branch>` and `nvidia-kernel-source-<branch>` without
  `-open`;
- DGX OS: `dgx-*` and `dgxstation-*`;
- DOCA host software: `doca-*`, `mlnx-*`, `ofed-*`, MFT, XLIO, DPCP, HCOLL,
  SHARP, ClusterKit and `ibutils2`;
- a vendor rebuild of any package: a version containing `nvidia`, `mlnx` or
  `doca`, such as DOCA's `rdma-core` or DGX OS's themed Ubuntu packages.

**Allowed.** Canonical's kernel packages (`linux-*`), including its builds
of the open kernel module (`linux-modules-nvidia-<branch>-open-*`) and the
`nvidia-fs` module; the open module's sources
(`nvidia-kernel-source-<branch>-open`, `nvidia-dkms-<branch>-open`); GPU
firmware, including the GSP firmware (`nvidia-firmware-<branch>-<version>`,
`linux-firmware`); Ubuntu's `rdma-core`, `ibverbs-providers` and
`mstflint`; and SparkPipe's own files.

## Release gate

A release root built today still carries CUDA-linked files: the serving
adapters, weightd and the rest. Until each one moves to SparkPipe's own
device backend, the gate accepts exactly those findings and nothing else.
[`tools/hosted_release_gate.py`](../tools/hosted_release_gate.py) runs the
`root` audit over a staged release root and compares every finding with the
debt list,
[`tools/hosted_release_debt.json`](../tools/hosted_release_debt.json):

```bash
make hosted-release-gate
make hosted-release-gate HOSTED_RELEASE_GATE_FLAGS=--complete
HOSTED=1 scripts/release_build.sh REV
python3 tools/hosted_release_gate.py findings build/hosted_release_root
python3 tools/hosted_release_gate.py shrink BASE
```

`make hosted-release-gate` builds the files a release ships, copies them to
`build/hosted_release_root` under their repository paths and checks that
root. The default set is every file a lane root or hub API directory ships:

- the node files `HOSTED_RELEASE_NODE_FILES`, which equal the kit's `NODE`
  list;
- the hub API files `HOSTED_RELEASE_HUB_FILES`;
- every CUDA serving adapter, with the `<adapter>.kernels` directory of
  cubins and kernel manifests its build installs beside it, which lane setup
  ships with it (`HOSTED_RELEASE_KERNELS`);
- `tools/fleet_node_agent.sh`.

`HOSTED_RELEASE_FILES` replaces the set and `HOSTED_RELEASE_ROOT` the
directory. A replaced set still stages the kernels directory of every
adapter it names. `--stage` takes exact paths only: a file, or a directory
whose entries are all regular files or directories. It refuses symbolic
links and a directory with no files.

The debt list groups files by the reason they need an NVIDIA library. Each
file is named by its exact path, with no patterns or directories, and each
finding by its kind and detail, for example `needed libcuda.so.1`. The gate
fails:

- `new`: a finding the list does not name;
- `stale`: a listed finding that the shipped file no longer has, which must
  be removed from the list;
- `missing`, with `--complete` only: a listed file the root does not ship.

The files `HOSTED_RELEASE_HOST_ONLY_FILES` names never appear on the list:

- `sparkpipe_model_api`;
- `sparkpipe_tokenize_prompt`;
- `sparkpipe_model_batch`;
- `weightd_receipt`;
- the node agent.

They have no device code, and no link line names a CUDA library for them.
The CUDA flavour gives `-lcuda` and `-lcudart` only to the targets that call
those libraries. The global `LDFLAGS` carry neither.

The list can only shrink:

- the CUDA compile job in CI builds the complete set in the CUDA 13
  container and runs the gate with `--complete`, so a new dependency fails
  and so does a cleared one still on the list;
- the CI lanes job runs `shrink` against the pull request's base, which fails
  on any file or finding the base list did not have;
- the receipts of the compile job include `hosted_release_findings.json`,
  the findings in the debt list's format.

`HOSTED=1 scripts/release_build.sh` runs the gate on the build host over the
node files and adapters, and on the hub over the API files and adapters,
before it writes the firmware directory. A refusal stops the kit with
nothing written. Without `HOSTED=1` the kit builds as before.

## Tests

[`tests/test_hosted_dependency_audit.py`](../tests/test_hosted_dependency_audit.py)
runs in `make test`. Its ELF fixtures are libraries and programs it builds
with `cc` from a few lines of C: a stand-in library with a soname such as
`libcuda.so.1`, a program linked against it, a program that `dlopen`s a
library by name. Each check runs on a root, image or process table that
carries the dependency and on a clean one, and fails on the first and passes
on the second. A live-process case loads a stand-in library and audits the
running process. The ELF cases need a `cc` that links ELF, so they skip on
macOS; the package, maps, name, script and Python-package cases run
everywhere.

[`tests/test_hosted_release_gate.py`](../tests/test_hosted_release_gate.py)
also runs in `make test`. It checks four things.

- **The debt list.** It is valid and names only files a release ships. The
  validation refuses patterns, loose or duplicated paths, unsorted entries,
  short reasons and finding kinds that are never debt.
- **Link flags.** With a stand-in CUDA toolkit, the global link flags name
  no CUDA library. The host-only programs' link lines name none either,
  while weightd's tools keep `-lcuda`.
- **The gate.** It passes a listed finding and fails new, stale and missing
  ones. It stages exact paths only, and a directory only when it holds
  regular files and no links. `shrink` fails on any added file or
  finding in a scratch Git repository. The gate stages each shipped adapter
  with its kernels directory. The release kit's `NODE` list equals
  `HOSTED_RELEASE_NODE_FILES`, and its hosted mode gates both builds before
  it writes the firmware directory.
- **The host-only build**, on Linux. It builds the host-only programs from a
  copy of the sources with the stand-in toolkit and `-Wl,--no-as-needed`.
  The audit must find nothing in them, so a CUDA library on any of their
  link lines fails even where `--as-needed` would drop it. A control build
  of `weightd_warm` must show `libcuda.so.1` and `libcudart.so.13`.
