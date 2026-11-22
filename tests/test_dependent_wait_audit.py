"""Host checks of tools/dependent_wait_audit.py on SASS listings.

A function that waits for the kernel it depends on (ACQBULK) must not store to
global memory, or touch it with an atomic or a reduction, before its first
wait: the audit must refuse a plain, predicated or bulk store, an atomic and a
reduction there, and accept loads, L2 prefetches, shared-memory stores, bulk
copies into shared memory and every write after the wait. A function that
never waits is not checked, a kernel on the allowed list is checked but not
refused, mangled names are reduced to the kernel's own name, and the CUDA
sm_121a compile gate runs the audit on every serving adapter and CUDA program
it builds.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("dependent_wait_audit", ROOT / "tools" / "dependent_wait_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def listing(name, body):
    lines = [f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM121"']
    lines += [f"        /*{index * 16:04x}*/                   {instruction} ;                 /* 0x0000000000000000 */"
              for index, instruction in enumerate(body)]
    return "\n".join(lines) + "\n"


def refused(name, body):
    found, checked = audit.refusals(audit.functions(listing(name, body)))
    return [text for _, text in found], checked


def check(failures):
    prologue = ["LDC R1, c[0x0][0x37c]", "S2R R7, SR_TID.X", "LDG.E R0, desc[UR4][R2.64]", "UBLKPF.L2 [UR4], UR6", "STS [R3], R0",
                "UBLKCP.S.G [UR8], [UR4], UR6"]
    tail = ["ACQBULK", "PREEXIT", "STG.E desc[UR4][R4.64], R9", "REDG.E.ADD.STRONG.GPU desc[UR4][R6.64], R11", "EXIT"]
    found, checked = refused("_Z5CleanPKfPf", prologue + tail)
    if found or checked != 1:
        failures.append("loads, prefetches, shared stores and writes after the wait were refused or not checked: %s" % found)
    for write in ("STG.E desc[UR4][R4.64], R9", "@P0 STG.E.64 desc[UR4][R4.64], R8", "ST.E desc[UR4][R4.64], R9",
                  "ATOMG.E.ADD.STRONG.GPU PT, R3, desc[UR4][R2.64], R5", "@P0 REDG.E.ADD.STRONG.GPU desc[UR4][R6.64], R11",
                  "UBLKCP.G.S [UR4], [UR6], UR7", "UBLKRED.G.S.ADD [UR4], [UR6], UR7", "UTMASTG.2D [UR4], [UR8]"):
        found, _ = refused("_Z4WritePKfPf", prologue[:3] + [write] + tail)
        if len(found) != 1 or write.split()[-1] not in found[0]:
            failures.append("a write before the wait was not refused: %s" % write)
    found, checked = refused("_Z7NoWaitPf", ["STG.E desc[UR4][R4.64], R9", "EXIT"])
    if found or checked != 0:
        failures.append("a function that never waits was checked")
    allowed = next(iter(audit.ALLOWED))
    mangled = "_Z%d%sPKhmmP23SparkTpMeshRoundControl" % (len(allowed), allowed)
    found, checked = refused(mangled, ["STG.E desc[UR4][R4.64], R9"] + tail)
    if found or checked != 1:
        failures.append("an allowed kernel was refused or not checked")
    if audit.stem("_Z19D41RmsNormQdqKernelILj256EEvPKt") != "D41RmsNormQdqKernel" or audit.stem("_ZN3foo3barEv") != "foo":
        failures.append("mangled names are not reduced to the kernel's name")
    gate = (ROOT / "tools" / "cuda13_sm121a_compile_gate.sh").read_text()
    if "tools/dependent_wait_audit.py" not in gate:
        failures.append("the sm_121a compile gate does not run the dependent wait audit")


def main():
    failures = []
    check(failures)
    for failure in failures:
        print("FAIL " + failure)
    if not failures:
        print("PASS dependent wait audit: writes before a dependent wait are refused")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
