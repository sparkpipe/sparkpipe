import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "nvrm_cubin_fixtures.py"
ARCH = "sm_121a"


def load_tool():
    spec = importlib.util.spec_from_file_location("nvrm_cubin_fixtures", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_nvcc_flags():
    run = subprocess.run(["make", "--no-print-directory", "-s", "-f", "Makefile", "-f", "-", "nvrm-print-nvcc-flags"], cwd=ROOT,
                         input="nvrm-print-nvcc-flags:\n\t@echo $(NVCCFLAGS)\n", capture_output=True, text=True)
    return run.stdout.split() if run.returncode == 0 else None


def fixture_archs(tool):
    return sorted(path.name[len("fixture_kernels."):-len(".manifest")] for path in (ROOT / tool.FIXTURES).glob("fixture_kernels.*.manifest"))


def fixture_cases(tool, arch):
    failures = []
    cubin = ROOT / tool.FIXTURES / f"fixture_kernels.{arch}.cubin"
    manifest = ROOT / tool.FIXTURES / f"fixture_kernels.{arch}.manifest"
    if not cubin.is_file() or not manifest.is_file():
        return [f"{tool.FIXTURES}: the {arch} cubin or its manifest is missing"]
    lines = manifest.read_text().splitlines()
    fields = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines if " " in line and not line.startswith("kernel ")}
    blob = cubin.read_bytes()
    if fields.get("cubin") != f"{hashlib.sha256(blob).hexdigest()} {len(blob)}":
        failures.append(f"the {arch} manifest does not describe the cubin in the tree; rebuild both with tools/nvrm_cubin_fixtures.py")
    source = hashlib.sha256((ROOT / tool.SOURCE).read_bytes()).hexdigest()
    if fields.get("source") != f"{tool.SOURCE} {source}":
        failures.append(f"{tool.SOURCE} changed since the {arch} fixtures were built; rebuild them with tools/nvrm_cubin_fixtures.py")
    flags = tool.release_flags(arch)
    if fields.get("flags", "").split()[:len(flags)] != flags:
        failures.append(f"the {arch} fixtures were not built with the release flags {' '.join(flags)}")
    makefile = release_nvcc_flags()
    if arch == ARCH and makefile != flags:
        failures.append(f"the release build compiles with {makefile}, the fixture tool with {flags}")
    if not any(line.startswith("kernel ") for line in lines):
        failures.append(f"the {arch} manifest lists no kernel")
    return failures


def main():
    tool = load_tool()
    archs = fixture_archs(tool)
    failures = [] if ARCH in archs else [f"{tool.FIXTURES}: the {ARCH} fixtures of the release build are missing"]
    for arch in archs:
        failures += fixture_cases(tool, arch)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print(f"\nPASS the {', '.join(archs)} cubin fixtures are fresh, built with the release flags, and described by ptxas's own report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
