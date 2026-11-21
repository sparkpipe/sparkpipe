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
| root, image | `toolkit-file` | a file is named as a CUDA toolkit compiler, debugger, profiler or runtime header (`nvcc`, `ptxas`, `nvlink`, `fatbinary`, `cuobjdump`, `nvdisasm`, `cicc`, `cuda-gdb`, `compute-sanitizer`, `nsys`, `ncu`, `cuda.h`, `cuda_runtime.h` and the rest of `TOOLKIT_FILES`) |
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
