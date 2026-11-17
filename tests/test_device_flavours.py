import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_EXT = "dylib" if sys.platform == "darwin" else "so"
NVRM_LIBRARY = f"build/runtime_libs/libsparkpipe_device_nvrm.{LIBRARY_EXT}"
VENDOR_LIBRARY = re.compile(r"lib(?:cuda|cudart|nvidia|nvrtc|nvJitLink|cublas|cublasLt|cudnn|nccl|cupti|nvToolsExt|nvml)\b")
VENDOR_FLAG = re.compile(r"-l(?:cuda|cudart|nvidia\S*|nvrtc|cublas\S*|nccl|nvidia-ml)\b|cuda[^ ]*/include|cuda[^ ]*/lib|\bnvcc\b")


def make(*arguments):
    return subprocess.run(["make", "--no-print-directory", *arguments], cwd=ROOT, capture_output=True, text=True)


def refusal_cases():
    failures = []
    for value in ("rocm", "cuda nvrm", ""):
        run = make("-n", f"SPARKPIPE_DEVICE={value}", "nvrm-device")
        if run.returncode == 0 or "is not a device flavour; the flavours are cuda nvrm" not in run.stderr:
            failures.append(f"SPARKPIPE_DEVICE={value!r} was accepted: {run.stderr.strip()[-160:]}")
    for goals in ((), ("all",), ("test",), ("nvrm-device", "tools")):
        run = make("-n", "SPARKPIPE_DEVICE=nvrm", *goals)
        unported = " ".join(goal for goal in (goals or ("all",)) if goal not in ("nvrm-device", "test-nvrm", "clean"))
        if run.returncode == 0 or f"; {unported} still call the CUDA runtime" not in run.stderr:
            failures.append(f"the nvrm flavour accepted the unported goals {unported}: {run.stderr.strip()[-160:]}")
    return failures


def goal_list_cases():
    failures = []
    lines = (ROOT / "Makefile").read_text().splitlines()
    checks = [index for index, line in enumerate(lines) if line.startswith("NVRM_UNPORTED_GOALS :=")]
    if len(checks) != 1:
        return [f"the Makefile has {len(checks)} unported-goal checks, not one"]
    goals = []
    for index, line in enumerate(lines):
        if not line.startswith("NVRM_FLAVOUR_GOALS") or line == "NVRM_FLAVOUR_GOALS :=":
            continue
        added = re.fullmatch(r"NVRM_FLAVOUR_GOALS \+= (\S+)", line)
        if added is None:
            failures.append(f"line {index + 1} names nvrm goals other than one goal beside its rule: {line}")
            continue
        goal = added.group(1)
        goals.append(goal)
        if index > checks[0]:
            failures.append(f"{goal} is added after the unported-goal check, which then refuses it")
        if not any(following.startswith(f"{goal}:") for following in lines[index + 1:index + 3]):
            failures.append(f"{goal} is not added beside its own rule")
    if "nvrm-device" not in goals or "clean" not in goals:
        failures.append(f"the nvrm goals lack nvrm-device or clean: {goals}")
    run = make("-n", "SPARKPIPE_DEVICE=nvrm", *goals)
    if run.returncode != 0 or "still call the CUDA runtime" in run.stderr:
        failures.append(f"the nvrm flavour refused its own goals {' '.join(goals)}: {run.stderr.strip()[-200:]}")
    return failures


def nvrm_build_cases():
    failures = []
    run = make("-n", "-B", "SPARKPIPE_DEVICE=nvrm", "nvrm-device")
    lines = [line for line in run.stdout.splitlines() if NVRM_LIBRARY in line and "-o" in line]
    if run.returncode != 0 or len(lines) != 1:
        failures.append(f"the nvrm flavour does not build {NVRM_LIBRARY}: {run.stderr.strip()[-160:]}")
    for line in lines:
        found = VENDOR_FLAG.findall(line)
        if found:
            failures.append(f"the nvrm library build names a vendor library or toolkit path: {found}")
    default = make("-n", "-B", "all")
    explicit = make("-n", "-B", "SPARKPIPE_DEVICE=cuda", "all")
    if default.returncode != 0 or default.stdout != explicit.stdout:
        failures.append("the default build is not the cuda flavour")
    if not any(NVRM_LIBRARY in line and "-o" in line for line in default.stdout.splitlines()):
        failures.append("the cuda flavour's all does not compile the nvrm library")
    return failures


def dependency_cases():
    built = make("-s", "nvrm-device")
    if built.returncode != 0:
        return [f"make nvrm-device failed: {(built.stdout + built.stderr).strip()[-400:]}"]
    library = ROOT / NVRM_LIBRARY
    if sys.platform == "darwin":
        tool = ["otool", "-L", str(library)]
    else:
        tool = ["readelf", "-d", str(library)]
    if shutil.which(tool[0]) is None:
        return [f"{tool[0]} is missing; it lists the nvrm library's dynamic dependencies"]
    listed = subprocess.run(tool, capture_output=True, text=True)
    if listed.returncode != 0:
        return [f"{' '.join(tool)} failed: {listed.stderr.strip()}"]
    needed = [line.strip() for line in listed.stdout.splitlines()[1:] if line.strip()]
    if not sys.platform == "darwin":
        needed = [line for line in needed if "(NEEDED)" in line]
    if not needed:
        return [f"{' '.join(tool)} listed no dependencies"]
    found = [line for line in needed if VENDOR_LIBRARY.search(line)]
    return [f"the nvrm library depends on {line}" for line in found]


def main():
    failures = refusal_cases() + goal_list_cases() + nvrm_build_cases() + dependency_cases()
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nPASS SPARKPIPE_DEVICE is cuda by default, refuses other values and unported nvrm goals, accepts each nvrm goal added beside its "
          "rule, and the nvrm library links no vendor library")
    return 0


if __name__ == "__main__":
    sys.exit(main())
