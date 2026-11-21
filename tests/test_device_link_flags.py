"""The CUDA link flags belong to the CUDA device flavour.

Every binary takes LDFLAGS, so a CUDA library in the global LDFLAGS reaches
programs that use no GPU and every binary of the nvrm flavour. The CUDA
driver and runtime link flags are set only when SPARKPIPE_DEVICE=cuda. This
test points CUDA_HOME at a scratch toolkit (a cuda_runtime.h header and
nothing else), so the toolkit counts as installed on any host, and reads the
expanded link variables of each flavour: the nvrm flavour names no vendor
library in LDFLAGS, SPARKPIPE_CUDA_DRIVER_LINK or SPARKPIPE_CUDA_RUNTIME_LINK,
and the cuda flavour still links the driver everywhere and the runtime where
a rule asks for it.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIABLES = ("LDFLAGS", "SPARKPIPE_CUDA_DRIVER_LINK", "SPARKPIPE_CUDA_RUNTIME_LINK")
VENDOR = re.compile(r"-l(?:cuda|cudart|nvidia\S*|nvrtc|cublas\S*|nccl|nvidia-ml)\b|cuda[^ ]*/lib")
PRINTER = "".join("$(info LINK %s=$(%s))\n" % (name, name) for name in VARIABLES)


def link_variables(flavour, toolkit, goal):
    with tempfile.TemporaryDirectory(prefix="link-flags-") as scratch:
        printer = Path(scratch) / "print.mk"
        printer.write_text(PRINTER)
        run = subprocess.run(["make", "--no-print-directory", "-n", "-f", "Makefile", "-f", str(printer),
                              "CUDA_HOME=" + str(toolkit), "SPARKPIPE_DEVICE=" + flavour, goal],
                             cwd=ROOT, capture_output=True, text=True)
    values = {}
    for line in run.stdout.splitlines():
        match = re.match(r"LINK (\w+)=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    if run.returncode != 0 or set(values) != set(VARIABLES):
        raise SystemExit("make -n %s for SPARKPIPE_DEVICE=%s failed: %s" % (goal, flavour, run.stderr.strip()[-300:]))
    return values


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="toolkit-") as scratch:
        toolkit = Path(scratch)
        (toolkit / "include").mkdir()
        (toolkit / "include" / "cuda_runtime.h").write_text("")
        nvrm = link_variables("nvrm", toolkit, "nvrm-device")
        cuda = link_variables("cuda", toolkit, "nvrm-device")
    for name, value in sorted(nvrm.items()):
        found = VENDOR.findall(value)
        if found:
            failures.append("SPARKPIPE_DEVICE=nvrm: %s names %s" % (name, " ".join(found)))
    lib = str(toolkit / "lib64")
    if cuda["SPARKPIPE_CUDA_DRIVER_LINK"] != "-L%s -lcuda" % lib:
        failures.append("SPARKPIPE_DEVICE=cuda: SPARKPIPE_CUDA_DRIVER_LINK is %r" % cuda["SPARKPIPE_CUDA_DRIVER_LINK"])
    if cuda["SPARKPIPE_CUDA_RUNTIME_LINK"] != "-L%s -lcudart" % lib:
        failures.append("SPARKPIPE_DEVICE=cuda: SPARKPIPE_CUDA_RUNTIME_LINK is %r" % cuda["SPARKPIPE_CUDA_RUNTIME_LINK"])
    if "-lcuda" not in cuda["LDFLAGS"].split():
        failures.append("SPARKPIPE_DEVICE=cuda: LDFLAGS no longer links the CUDA driver: %r" % cuda["LDFLAGS"])
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("PASS the CUDA link flags are set only in the cuda device flavour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
