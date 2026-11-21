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


def report_cases(tool):
    failures = []
    text = ("ptxas info    : 0 bytes gmem\n"
            "ptxas info    : Compiling entry function 'a' for 'sm_121a'\n"
            "ptxas info    : Function properties for a\n"
            "    16 bytes stack frame, 4 bytes spill stores, 8 bytes spill loads\n"
            "ptxas info    : Used 30 registers, used 1 barriers, 2048 bytes smem\n"
            "ptxas info    : Compiling entry function 'b' for 'sm_121a'\n"
            "ptxas info    : Function properties for b\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            "ptxas info    : Used 8 registers, used 0 barriers\n")
    got = tool.report(text, ARCH)
    if got != {"a": {"stack": 16, "spill_stores": 4, "spill_loads": 8, "registers": 30, "barriers": 1, "smem": 2048},
               "b": {"stack": 0, "spill_stores": 0, "spill_loads": 0, "registers": 8, "barriers": 0, "smem": 0}}:
        failures.append(f"the ptxas report parses to {got}")
    for label, broken in (("another arch", text.replace("for 'sm_121a'", "for 'sm_120a'", 1)),
                          ("a missing resource line", text.replace("ptxas info    : Used 8 registers, used 0 barriers\n", "")),
                          ("no entry function", "ptxas info    : 0 bytes gmem\n")):
        try:
            tool.report(broken, ARCH)
            failures.append(f"a report with {label} was accepted")
        except tool.FixtureError:
            pass
    return failures


def fixture_cases(tool):
    failures = []
    cubin = ROOT / tool.FIXTURES / f"fixture_kernels.{ARCH}.cubin"
    manifest = ROOT / tool.FIXTURES / f"fixture_kernels.{ARCH}.manifest"
    if not cubin.is_file() or not manifest.is_file():
        return [f"{tool.FIXTURES}: the {ARCH} cubin or its manifest is missing"]
    lines = manifest.read_text().splitlines()
    fields = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines if " " in line and not line.startswith("kernel ")}
    blob = cubin.read_bytes()
    if fields.get("cubin") != f"{hashlib.sha256(blob).hexdigest()} {len(blob)}":
        failures.append("the manifest does not describe the cubin in the tree; rebuild both with tools/nvrm_cubin_fixtures.py")
    source = hashlib.sha256((ROOT / tool.SOURCE).read_bytes()).hexdigest()
    if fields.get("source") != f"{tool.SOURCE} {source}":
        failures.append(f"{tool.SOURCE} changed since the fixtures were built; rebuild them with tools/nvrm_cubin_fixtures.py")
    flags = tool.release_flags(ARCH)
    if fields.get("flags", "").split()[:len(flags)] != flags:
        failures.append(f"the fixtures were not built with the release flags {' '.join(flags)}")
    makefile = release_nvcc_flags()
    if makefile != flags:
        failures.append(f"the release build compiles with {makefile}, the fixture tool with {flags}")
    if not any(line.startswith("kernel ") for line in lines):
        failures.append("the manifest lists no kernel")
    return failures


def main():
    tool = load_tool()
    failures = report_cases(tool) + fixture_cases(tool)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print(f"\nPASS the {ARCH} cubin fixtures are fresh, built with the release flags, and described by ptxas's own report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
