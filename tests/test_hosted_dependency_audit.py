import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "hosted_dependency_audit.py"
SPEC = importlib.util.spec_from_file_location("hosted_dependency_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RULE_LIBRARIES = (
    "libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1",
    "libnvidia-nvvm.so.4", "libnvidia-gpucomp.so.580.159.03", "libnvrtc.so.13", "libnvrtc-builtins.so.13.0",
    "libnvJitLink.so.13", "libcublas.so.13", "libcublasLt.so.13", "libcufft.so.12", "libcurand.so.10",
    "libcusparse.so.12", "libcusolver.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", "libnccl.so.2",
    "libcupti.so.13", "libnvperf_host.so", "libnvToolsExt.so.1", "libnvtx3interop.so.1", "libcufile.so.0",
    "libnvjpeg.so.13", "libnppc.so.13", "libnvidia-container.so.1", "libGLX_nvidia.so.0", "libtorch_cuda.so",
    "libc10_cuda.so",
)
ALLOWED_LIBRARIES = (
    "libibverbs.so.1", "libmlx5.so.1", "librdmacm.so.1", "libc.so.6", "libm.so.6", "libstdc++.so.6",
    "libOpenCL.so.1", "libnvme.so.1", "libnvpair.so.3", "libnuma.so.1", "libcurl.so.4", "libcups.so.2",
    "libsparkpipe_core.so",
)
CLEAN_PACKAGES = (
    ("base-files", "13ubuntu10.3"), ("rdma-core", "50.0-2ubuntu0.2"), ("ibverbs-providers", "50.0-2ubuntu0.2"),
    ("libibverbs1", "50.0-2ubuntu0.2"), ("mstflint", "4.26.0+1-2ubuntu3"), ("libnvme1t64", "1.8-3ubuntu1"),
    ("linux-image-6.17.0-1026-nvidia", "6.17.0-1026.26"), ("linux-modules-6.17.0-1026-nvidia", "6.17.0-1026.26"),
    ("linux-modules-nvidia-580-open-6.17.0-1026-nvidia", "6.17.0-1026.26"),
    ("linux-modules-nvidia-580-server-open-6.17.0-1026-nvidia", "6.17.0-1026.26"),
    ("linux-signatures-nvidia-6.17.0-1026-nvidia", "6.17.0-1026.26"), ("linux-firmware", "20240318.git3b128b60-0ubuntu2"),
    ("nvidia-firmware-580-580.159.03", "580.159.03-0ubuntu0.24.04.1"), ("nvidia-kernel-source-580-open", "580.159.03-0ubuntu0.24.04.1"),
)
FORBIDDEN_PACKAGES = (
    ("libnvidia-compute-580", "580.159.03-0ubuntu0.24.04.1"), ("libnvidia-egl-wayland1", "1:1.1.21-1ubuntu1"),
    ("nvidia-utils-580", "580.159.03-0ubuntu0.24.04.1"), ("nvidia-compute-utils-580", "580.159.03-0ubuntu0.24.04.1"),
    ("nvidia-kernel-common-580", "580.159.03-0ubuntu0.24.04.1"), ("nvidia-driver-580-open", "580.159.03-0ubuntu0.24.04.1"),
    ("xserver-xorg-video-nvidia-580", "580.159.03-0ubuntu0.24.04.1"), ("nvidia-modprobe", "610.43.02-1ubuntu1"),
    ("cuda-toolkit-13-0", "13.0.3-1"), ("cuda-cudart-13-0", "13.0.96-1"), ("libcublas-13-0", "13.1.1.3-1"),
    ("libnvjitlink-13-0", "13.0.88-1"), ("libcudnn9-cuda-13", "9.13.0.50-1"), ("libnccl2", "2.28.3-1+cuda13.0"),
    ("nsight-systems-2025.3.2", "2025.3.2.474-253236389321v0"), ("nvidia-container-toolkit", "1.19.1-1"),
    ("libnvidia-container1", "1.19.1-1"), ("dgx-dashboard", "0.29.0"), ("dgx-release", "7.5.0"),
    ("dgxstation-grub", "25.02-1"), ("nvidia-spark-ota-check", "1.0.23-1"), ("nvidia-system-core", "2404.26.01-1"),
    ("nvidia-mlnx-tools", "23.04"), ("doca-host", "3.1.0-091000-25.07-ubuntu2404"), ("mlnx-ofed-kernel-dkms", "25.07.OFED.25.07.0.9.7.1-1"),
    ("mft", "4.33.0-169"), ("rdma-core", "2507mlnx58-1.2507097"), ("yaru-theme-gtk", "24.04.2-nvidia"),
    ("linux-modules-nvidia-580-6.17.0-1026-nvidia", "6.17.0-1026.26"),
    ("linux-objects-nvidia-580-server-6.17.0-1026-nvidia", "6.17.0-1026.26"),
    ("nvidia-dkms-580", "580.159.03-0ubuntu0.24.04.1"), ("nvidia-kernel-source-580", "580.159.03-0ubuntu0.24.04.1"),
)
LOADER_SOURCE = """#include <dlfcn.h>
#include <stdio.h>
int main(int count, char **values)
{
    void *library = count == 2 ? dlopen(values[1], RTLD_NOW) : 0;
    if (!library) return 2;
    puts("ready");
    fflush(stdout);
    return getchar() == EOF ? 0 : 1;
}
"""


def compiler_links_elf():
    compiler = shutil.which("cc")
    if compiler is None:
        return False
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory, "probe.c")
        source.write_text("int spark_probe(void) { return 0; }\n")
        output = Path(directory, "libprobe.so")
        built = subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(output)], capture_output=True)
        return built.returncode == 0 and output.read_bytes()[:4] == b"\x7fELF"


ELF_HOST = compiler_links_elf()
needs_elf = unittest.skipUnless(ELF_HOST, "the host cc does not link ELF objects")


def run_audit(*arguments):
    completed = subprocess.run([sys.executable, str(SCRIPT), "--json", *arguments], capture_output=True, text=True)
    document = json.loads(completed.stdout) if completed.stdout.strip() else None
    return completed.returncode, document


def kinds(document):
    return {(finding["kind"], finding["path"], finding["detail"]) for finding in document["findings"]}


def write(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_bytes(data)
    path.chmod(mode)
    return path


def compile_library(directory, soname, source="int spark_fixture_symbol(void) { return 1; }\n", filename=None):
    source_path = write(Path(directory, f"{soname}.c"), source)
    output = Path(directory, filename or soname)
    subprocess.run(["cc", "-shared", "-fPIC", f"-Wl,-soname,{soname}", str(source_path), "-o", str(output)], check=True)
    return output


def compile_program(output, source, libraries=(), flags=()):
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = write(output.with_suffix(".c"), source)
    command = ["cc", str(source_path), "-o", str(output), "-Wl,--no-as-needed", *flags, *map(str, libraries), "-ldl"]
    subprocess.run(command, check=True)
    source_path.unlink()
    return output


def dpkg_status(packages, states=None):
    states = states or {}
    return "\n".join(
        f"Package: {name}\nStatus: {states.get(name, 'install ok installed')}\nArchitecture: arm64\nVersion: {version}\n"
        for name, version in packages)


class ClassifierTest(unittest.TestCase):
    def test_rule_libraries_are_forbidden_and_platform_libraries_are_not(self):
        self.assertEqual([name for name in RULE_LIBRARIES if not MODULE.forbidden_library(name)], [])
        self.assertEqual([name for name in ALLOWED_LIBRARIES if MODULE.forbidden_library(name)], [])

    def test_clean_packages_pass_and_rule_packages_fail(self):
        self.assertEqual([package for package in CLEAN_PACKAGES if MODULE.package_reason(*package)], [])
        self.assertEqual([package for package in FORBIDDEN_PACKAGES if not MODULE.package_reason(*package)], [])


class ReleaseRootTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.build = self.directory / "build"
        self.build.mkdir()

    def root(self, name):
        path = self.directory / name
        path.mkdir()
        return path

    @needs_elf
    def test_needed_entries_on_rule_libraries_fail_and_platform_needs_pass(self):
        libraries = [compile_library(self.build, soname) for soname in RULE_LIBRARIES]
        dirty = self.root("dirty")
        compile_program(dirty / "bin" / "engine", "int main(void) { return 0; }\n", libraries)
        code, document = run_audit("root", str(dirty))
        self.assertEqual(code, 1)
        self.assertEqual({detail for kind, path, detail in kinds(document) if kind == "needed"}, set(RULE_LIBRARIES))
        clean_library = compile_library(self.build, "libspark_clean.so.1")
        clean = self.root("clean")
        compile_program(clean / "bin" / "engine", "#include <math.h>\nint main(int c, char **v) { return (int)cos((double)c); }\n",
                        [clean_library], ["-lm"])
        (clean / "lib").mkdir()
        shutil.copy(clean_library, clean / "lib" / "libspark_clean.so.1")
        code, document = run_audit("root", str(clean))
        self.assertEqual((code, document["findings"]), (0, []))

    @needs_elf
    def test_each_rule_library_alone_fails_a_root(self):
        for soname in ("libcuda.so.1", "libcudart.so.13", "libnvidia-ml.so.1", "libnccl.so.2", "libcudnn.so.9"):
            with self.subTest(soname=soname):
                library = compile_library(self.build, soname)
                root = self.directory / soname.replace(".", "_")
                compile_program(root / "bin" / "engine", "int main(void) { return 0; }\n", [library])
                code, document = run_audit("root", str(root))
                self.assertEqual((code, kinds(document)), (1, {("needed", "bin/engine", soname)}))

    @needs_elf
    def test_library_declaring_a_rule_soname_fails_under_any_file_name(self):
        dirty = self.root("dirty")
        compile_library(dirty, "libcuda.so.1", filename="libgpu.so")
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {("soname", "libgpu.so", "libcuda.so.1")}))
        clean = self.root("clean")
        compile_library(clean, "libspark_gpu.so.1", filename="libgpu.so")
        self.assertEqual(run_audit("root", str(clean))[0], 0)

    @needs_elf
    def test_innocently_named_link_to_a_rule_library_outside_the_root_fails(self):
        outside = compile_library(self.build, "libcuda.so.1")
        clean_outside = compile_library(self.build, "libspark_clean.so.1")
        needed = compile_library(self.build, "libdev.so")
        for name, target, expected in (("dirty", outside, 1), ("clean", clean_outside, 0)):
            with self.subTest(root=name):
                root = self.root(name)
                compile_program(root / "bin" / "engine", "int main(void) { return 0; }\n", [needed])
                (root / "lib" / "runtime_libs").mkdir(parents=True)
                (root / "lib" / "runtime_libs" / "libdev.so").symlink_to(target)
                code, document = run_audit("root", str(root))
                self.assertEqual(code, expected)
                self.assertEqual(kinds(document), {("library-file", "lib/runtime_libs/libdev.so", f"link to {outside}")} if expected else set())

    def test_link_chains_are_followed_to_a_rule_library_name(self):
        dirty = self.root("dirty")
        (dirty / "lib").mkdir()
        (dirty / "vendor").mkdir()
        (dirty / "lib" / "libdev.so").symlink_to("../vendor/libgpu.so")
        (dirty / "vendor" / "libgpu.so").symlink_to("libnvidia-ml.so.1")
        (dirty / "lib" / "libabsolute.so").symlink_to("/usr/lib/aarch64-linux-gnu/libcudart.so.13")
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {
            ("library-file", "lib/libdev.so", "link to libnvidia-ml.so.1"),
            ("library-file", "vendor/libgpu.so", "link to libnvidia-ml.so.1"),
            ("library-file", "lib/libabsolute.so", "link to /usr/lib/aarch64-linux-gnu/libcudart.so.13")}))
        clean = self.root("clean")
        write(clean / "lib" / "libspark.so.1", b"spark")
        (clean / "lib" / "libspark.so").symlink_to("libspark.so.1")
        (clean / "lib" / "libverbs.so").symlink_to("/usr/lib/aarch64-linux-gnu/libibverbs.so.1")
        code, document = run_audit("root", str(clean))
        self.assertEqual((code, document["findings"]), (0, []))
        looped = self.root("looped")
        (looped / "a.so").symlink_to("b.so")
        (looped / "b.so").symlink_to("a.so")
        code, document = run_audit("root", str(looped))
        self.assertEqual(code, 1)
        self.assertEqual({kind for kind, path, detail in kinds(document)}, {"unreadable"})

    @needs_elf
    def test_dlopen_string_of_a_rule_library_fails(self):
        dirty = self.root("dirty")
        compile_program(dirty / "bin" / "telemetry",
                        '#include <dlfcn.h>\nint main(void) { return dlopen("libnvidia-ml.so.1", RTLD_NOW) == 0; }\n')
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {("library-string", "bin/telemetry", "libnvidia-ml.so.1")}))
        clean = self.root("clean")
        compile_program(clean / "bin" / "telemetry",
                        '#include <dlfcn.h>\nint main(void) { return dlopen("libibverbs.so.1", RTLD_NOW) == 0; }\n')
        code, document = run_audit("root", str(clean))
        self.assertEqual((code, document["findings"]), (0, []))

    @needs_elf
    def test_runpath_into_a_toolkit_fails(self):
        dirty = self.root("dirty")
        compile_program(dirty / "bin" / "engine", "int main(void) { return 0; }\n", flags=["-Wl,-rpath,/usr/local/cuda-13.0/lib64"])
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {("runpath", "bin/engine", "/usr/local/cuda-13.0/lib64")}))
        clean = self.root("clean")
        compile_program(clean / "bin" / "engine", "int main(void) { return 0; }\n", flags=["-Wl,-rpath,$ORIGIN/../lib"])
        self.assertEqual(run_audit("root", str(clean))[0], 0)

    @needs_elf
    def test_truncated_elf_fails(self):
        root = self.root("dirty")
        program = compile_program(self.build / "engine", "int main(void) { return 0; }\n")
        write(root / "bin" / "engine", program.read_bytes()[:200])
        code, document = run_audit("root", str(root))
        self.assertEqual(code, 1)
        self.assertEqual({(kind, path) for kind, path, detail in kinds(document)}, {("unparsed", "bin/engine")})

    def test_unparseable_elf_header_fails(self):
        root = self.root("dirty")
        write(root / "lib" / "module.so", b"\x7fELF\x02\x01\x01" + bytes(20))
        code, document = run_audit("root", str(root))
        self.assertEqual(code, 1)
        self.assertEqual({(kind, path) for kind, path, detail in kinds(document)}, {("unparsed", "lib/module.so")})

    def test_rule_library_files_are_reported_by_name_and_never_opened(self):
        dirty = self.root("dirty")
        write(dirty / "lib" / "libcuda.so.1", b"\x7fELF")
        write(dirty / "lib" / "libnvidia-ml.so.580.159.03", b"\x7fELF")
        write(dirty / "lib" / "libcudart_static.a", b"!<arch>\n")
        write(dirty / "lib" / "libspark.so.1", b"spark")
        opened = []
        original = MODULE.audit_file
        with mock.patch.object(MODULE, "audit_file", side_effect=lambda report, relative, path, scan: (opened.append(relative), original(report, relative, path, scan))):
            report = MODULE.audit_root(str(dirty))
        self.assertEqual(opened, ["lib/libspark.so.1"])
        self.assertEqual({(kind, path) for path, kind, detail in report.found},
                         {("library-file", "lib/libcuda.so.1"), ("library-file", "lib/libnvidia-ml.so.580.159.03"),
                          ("library-file", "lib/libcudart_static.a")})

    def test_tools_and_scripts_that_call_them_fail(self):
        dirty = self.root("dirty")
        write(dirty / "bin" / "agent.sh", "#!/bin/sh\nnvidia-smi -lgc 0,2200\n", 0o755)
        write(dirty / "bin" / "nvidia-smi", b"\x00", 0o755)
        write(dirty / "bin" / "nvcc", b"\x00", 0o755)
        write(dirty / "bin" / "ptxas", b"\x00", 0o755)
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {
            ("tool-string", "bin/agent.sh", "nvidia-smi"), ("tool-file", "bin/nvidia-smi", "nvidia-smi"),
            ("toolkit-file", "bin/nvcc", "nvcc"), ("toolkit-file", "bin/ptxas", "ptxas")}))
        clean = self.root("clean")
        write(clean / "bin" / "agent.sh", "#!/bin/sh\nibv_devinfo -d mlx5_0\ncat /sys/class/drm/card0/device/uevent\n", 0o755)
        code, document = run_audit("root", str(clean))
        self.assertEqual((code, document["findings"]), (0, []))

    def test_cuda_pytorch_and_triton_fail_and_cpu_pytorch_passes(self):
        dirty = self.root("dirty")
        write(dirty / "python" / "torch" / "version.py", "from typing import Optional\ncuda: Optional[str] = '13.0'\n")
        write(dirty / "packages" / "torch" / "version.py", "cuda = '12.8'\n")
        write(dirty / "python" / "triton" / "__init__.py", "__version__ = '3.5.0'\n")
        code, document = run_audit("root", str(dirty))
        self.assertEqual((code, kinds(document)), (1, {
            ("pytorch-cuda", "python/torch/version.py", "cuda = '13.0'"), ("pytorch-cuda", "packages/torch/version.py", "cuda = '12.8'"),
            ("triton", "python/triton/__init__.py", "Triton package")}))
        clean = self.root("clean")
        write(clean / "python" / "torch" / "version.py", "from typing import Optional\ncuda: Optional[str] = None\n")
        self.assertEqual(run_audit("root", str(clean))[0], 0)

    def test_empty_root_fails_and_missing_root_is_a_usage_error(self):
        code, document = run_audit("root", str(self.root("empty")))
        self.assertEqual((code, kinds(document)), (1, {("empty", ".", "nothing to audit")}))
        self.assertEqual(run_audit("root", str(self.directory / "absent")), (2, None))


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def process(self, proc, pid, comm, paths):
        lines = "".join(f"7f00{index:04x}000-7f00{index:04x}fff r-xp 00000000 103:02 {index} {path}\n" for index, path in enumerate(paths))
        write(proc / str(pid) / "maps", lines + "7ffd0000-7ffd1000 rw-p 00000000 00:00 0 [stack]\n")
        write(proc / str(pid) / "comm", comm + "\n")

    def test_process_mapping_a_rule_library_fails(self):
        proc = self.directory / "proc"
        self.process(proc, 100, "weightd", ["/usr/lib/aarch64-linux-gnu/libc.so.6", "/usr/lib/aarch64-linux-gnu/libibverbs.so.1"])
        code, document = run_audit("processes", "--proc", str(proc))
        self.assertEqual((code, document["findings"], document["scanned"]), (0, [], 1))
        self.process(proc, 200, "residentd", ["/usr/lib/aarch64-linux-gnu/libcuda.so.580.159.03 (deleted)",
                                              "/usr/lib/aarch64-linux-gnu/libc.so.6"])
        self.process(proc, 300, "nvidia-smi", ["/usr/bin/nvidia-smi"])
        code, document = run_audit("processes", "--proc", str(proc))
        self.assertEqual((code, kinds(document)), (1, {
            ("mapped", "200", "residentd maps /usr/lib/aarch64-linux-gnu/libcuda.so.580.159.03 (deleted)"),
            ("mapped", "300", "nvidia-smi maps /usr/bin/nvidia-smi")}))

    def test_unreadable_maps_and_empty_tables_fail(self):
        proc = self.directory / "proc"
        (proc / "400" / "maps").mkdir(parents=True)
        code, document = run_audit("processes", "--proc", str(proc))
        self.assertEqual(code, 1)
        self.assertEqual({(kind, path) for kind, path, detail in kinds(document)}, {("unreadable", "400"), ("empty", str(proc))})
        empty = self.directory / "empty"
        empty.mkdir()
        self.assertEqual(run_audit("processes", "--proc", str(empty))[0], 1)
        self.assertEqual(run_audit("processes", "--proc", str(self.directory / "absent")), (2, None))

    @needs_elf
    @unittest.skipUnless(os.path.isdir("/proc/self/map_files") or os.path.isfile("/proc/self/maps"), "no /proc maps on this host")
    def test_live_process_that_loads_a_rule_library_fails(self):
        loader = compile_program(self.directory / "loader", LOADER_SOURCE)
        for soname, expected in (("libcuda.so.1", 1), ("libspark_clean.so.1", 0)):
            with self.subTest(soname=soname):
                library = compile_library(self.directory, soname)
                child = subprocess.Popen([str(loader), str(library)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                    code, document = run_audit("processes", "--pid", str(child.pid))
                finally:
                    child.communicate("")
                self.assertEqual(code, expected)
                details = {detail for kind, path, detail in kinds(document) if kind == "mapped"}
                self.assertEqual(details, {f"loader maps {library}"} if expected else set())


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def rootfs(self, name, packages=CLEAN_PACKAGES):
        rootfs = self.directory / name
        write(rootfs / "var" / "lib" / "dpkg" / "status", dpkg_status(packages))
        write(rootfs / "etc" / "apt" / "sources.list.d" / "ubuntu.sources",
              "Types: deb\nURIs: http://ports.ubuntu.com/ubuntu-ports/\nSuites: noble noble-updates\nComponents: main restricted\n")
        write(rootfs / "lib" / "firmware" / "nvidia" / "580.159.03" / "gsp_ga10x.bin", b"\x7fELF")
        write(rootfs / "lib" / "modules" / "6.17.0-1026-nvidia" / "updates" / "nvidia.ko", b"\x7fELF")
        write(rootfs / "etc" / "udev" / "rules.d" / "71-sparkpipe-gpu.rules", 'KERNEL=="nvidia_uvm", RUN+="/usr/lib/sparkpipe/gpu-device-nodes"\n')
        write(rootfs / "usr" / "bin" / "true", b"\x00", 0o755)
        return rootfs

    def test_status_file_with_any_rule_package_fails(self):
        clean = write(self.directory / "status", dpkg_status(CLEAN_PACKAGES))
        code, document = run_audit("image", "--dpkg-status", str(clean))
        self.assertEqual((code, document["findings"], document["scanned"]), (0, [], len(CLEAN_PACKAGES)))
        for name, version in FORBIDDEN_PACKAGES:
            with self.subTest(package=name):
                packages = [package for package in CLEAN_PACKAGES if package[0] != name] + [(name, version)]
                status = write(self.directory / f"status-{name}", dpkg_status(packages))
                code, document = run_audit("image", "--dpkg-status", str(status))
                self.assertEqual(code, 1)
                self.assertEqual({(kind, path) for kind, path, detail in kinds(document)}, {("package", name)})

    def test_removed_package_with_config_files_fails_and_purged_package_passes(self):
        packages = CLEAN_PACKAGES + (("nvidia-utils-580", "580.159.03-0ubuntu0.24.04.1"),)
        removed = write(self.directory / "removed", dpkg_status(packages, {"nvidia-utils-580": "deinstall ok config-files"}))
        self.assertEqual(run_audit("image", "--dpkg-status", str(removed))[0], 1)
        purged = write(self.directory / "purged", dpkg_status(packages, {"nvidia-utils-580": "purge ok not-installed"}))
        self.assertEqual(run_audit("image", "--dpkg-status", str(purged))[0], 0)

    def test_clean_rootfs_with_open_module_and_firmware_passes(self):
        rootfs = self.rootfs("clean")
        opened = []
        original = MODULE.audit_file
        with mock.patch.object(MODULE, "audit_file", side_effect=lambda report, relative, path, scan: (opened.append(relative), original(report, relative, path, scan))):
            report = MODULE.audit_image(str(rootfs), None, [])
        self.assertEqual(report.findings(), [])
        self.assertEqual([path for path in opened if path.startswith(("lib/firmware/", "lib/modules/"))], [])

    def test_rootfs_carrying_user_space_files_or_vendor_sources_fails(self):
        cases = {
            "library": ("usr/lib/aarch64-linux-gnu/libcuda.so.1", b"\x7fELF", ("library-file", "libcuda.so.1")),
            "tool": ("usr/bin/nvidia-smi", b"\x00", ("tool-file", "nvidia-smi")),
            "toolkit": ("usr/local/cuda-13.0/bin/nvcc", b"\x00", ("toolkit-file", "nvcc")),
            "source": ("etc/apt/sources.list.d/cuda.sources", b"URIs: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/\n",
                       ("apt-source", "URIs: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/")),
        }
        for name, (relative, data, (kind, detail)) in cases.items():
            with self.subTest(case=name):
                rootfs = self.rootfs(name)
                write(rootfs / relative, data)
                code, document = run_audit("image", "--rootfs", str(rootfs))
                self.assertEqual((code, kinds(document)), (1, {(kind, relative, detail)}))

    def test_absolute_links_in_an_image_resolve_inside_the_image(self):
        rootfs = self.rootfs("links")
        write(rootfs / "opt" / "vendor" / "libgpu.so.1", b"spark")
        (rootfs / "opt" / "vendor" / "libgpu.so").symlink_to("/usr/lib/aarch64-linux-gnu/libcuda.so.1")
        (rootfs / "usr" / "lib").mkdir()
        (rootfs / "usr" / "lib" / "libdev.so").symlink_to("/opt/vendor/libgpu.so")
        code, document = run_audit("image", "--rootfs", str(rootfs))
        self.assertEqual((code, kinds(document)), (1, {
            ("library-file", "opt/vendor/libgpu.so", "link to /usr/lib/aarch64-linux-gnu/libcuda.so.1"),
            ("library-file", "usr/lib/libdev.so", "link to /usr/lib/aarch64-linux-gnu/libcuda.so.1")}))
        (rootfs / "opt" / "vendor" / "libgpu.so").unlink()
        (rootfs / "opt" / "vendor" / "libgpu.so").symlink_to("libgpu.so.1")
        code, document = run_audit("image", "--rootfs", str(rootfs))
        self.assertEqual((code, document["findings"]), (0, []))

    def test_rootfs_without_package_database_fails(self):
        rootfs = self.rootfs("bare")
        (rootfs / "var" / "lib" / "dpkg" / "status").unlink()
        code, document = run_audit("image", "--rootfs", str(rootfs))
        self.assertEqual((code, kinds(document)), (1, {("package-database", "var/lib/dpkg/status", "missing")}))

    def test_release_root_inside_the_image_is_audited_as_a_release_root(self):
        rootfs = self.rootfs("release")
        write(rootfs / "opt" / "sparkpipe" / "release" / "bin" / "agent.sh", "#!/bin/sh\nexec ibv_devinfo\n", 0o755)
        code, document = run_audit("image", "--rootfs", str(rootfs), "--release-root", "opt/sparkpipe/release")
        self.assertEqual((code, document["findings"]), (0, []))
        write(rootfs / "opt" / "sparkpipe" / "release" / "bin" / "agent.sh", "#!/bin/sh\nnvidia-smi -q\n", 0o755)
        code, document = run_audit("image", "--rootfs", str(rootfs), "--release-root", "opt/sparkpipe/release")
        self.assertEqual((code, kinds(document)), (1, {("tool-string", "opt/sparkpipe/release/bin/agent.sh", "nvidia-smi")}))
        code, document = run_audit("image", "--rootfs", str(rootfs), "--release-root", "opt/sparkpipe/absent")
        self.assertEqual((code, kinds(document)), (1, {("missing", "opt/sparkpipe/absent", "release root is not a directory")}))

    @needs_elf
    def test_rootfs_binary_needing_a_rule_library_fails(self):
        library = compile_library(self.directory, "libcuda.so.1")
        rootfs = self.rootfs("needed")
        compile_program(rootfs / "usr" / "bin" / "gpu-probe", "int main(void) { return 0; }\n", [library])
        code, document = run_audit("image", "--rootfs", str(rootfs))
        self.assertEqual((code, kinds(document)), (1, {("needed", "usr/bin/gpu-probe", "libcuda.so.1")}))


if __name__ == "__main__":
    unittest.main()
