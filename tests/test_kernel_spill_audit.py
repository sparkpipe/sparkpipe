"""Host checks of tools/kernel_spill_audit.py on cuobjdump resource listings.

A function whose name contains a --kernel name must be refused when it spills
to local memory or keeps a stack frame, and accepted when it does neither;
functions with other names are not checked; and the CUDA sm_121a compile gate
runs the audit on every serving adapter and CUDA program for the grouped expert
GEMV.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("kernel_spill_audit", ROOT / "tools" / "kernel_spill_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

LISTING = """
Resource usage:
 Common:
  GLOBAL:0
 Function _Z21LmSkinnyGroupedKernelI7LmNvfp4Lj32ELj16ELj2EEv17LmSkinnyArguments:
  REG:255 STACK:7000 SHARED:8324 LOCAL:7000 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z21LmSkinnyGroupedKernelI7LmNvfp4Lj32ELj4ELj4EEv17LmSkinnyArguments:
  REG:128 STACK:0 SHARED:4228 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z21LmSkinnyGroupedKernelI12LmBf16FormatLj32ELj8ELj4EEv17LmSkinnyArguments:
  REG:96 STACK:8 SHARED:8324 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z14OtherSpillingKernelPf:
  REG:255 STACK:64 SHARED:0 LOCAL:64 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def main():
    failures = []
    found = audit.resources(LISTING)
    refused, checked = audit.refusals(found, ["LmSkinnyGroupedKernel"])
    names = [name for name, _, _, _ in refused]
    if checked != 3:
        failures.append("the audit checked %d grouped kernels, not 3" % checked)
    if not any("Lj16ELj2E" in name for name in names):
        failures.append("a grouped kernel that spills to local memory was not refused")
    if not any("LmBf16Format" in name for name in names):
        failures.append("a grouped kernel with a stack frame was not refused")
    if any("Lj4ELj4E" in name for name in names):
        failures.append("a grouped kernel that neither spills nor keeps a stack frame was refused")
    if any("OtherSpillingKernel" in name for name in names):
        failures.append("a kernel with another name was checked")
    gate = (ROOT / "tools" / "cuda13_sm121a_compile_gate.sh").read_text()
    if "tools/kernel_spill_audit.py\" --kernel LmSkinnyGroupedKernel" not in gate:
        failures.append("the sm_121a compile gate does not run the spill audit on the grouped expert GEMV")
    for failure in failures:
        print("FAIL " + failure)
    if not failures:
        print("PASS kernel spill audit: spilling grouped expert GEMVs are refused")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
