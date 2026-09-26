#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "kernel_codegen_diff.py"
FAKE_CUOBJDUMP = """#!{python}
import sys
sass, res = open(sys.argv[2]).read().split("=====RES=====\\n")
sys.stdout.write(sass if sys.argv[1] == "-sass" else res)
"""


def cubin(directory, name, kernels):
    sass = "".join("\t\tFunction : %s\n" % kernel + "".join("        /*%04x*/ %s ;  /* 0x%016x */\n" % (16 * index, instruction, 7919 * index + len(name)) for index, instruction in enumerate(body)) for kernel, body, _ in kernels)
    res = "".join(" Function %s:\n  REG:%d STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:900\n" % (kernel, registers) for kernel, _, registers in kernels)
    path = directory / name
    path.write_text(sass + "=====RES=====\n" + res)
    return str(path)


def run(directory, base, head, *extra):
    result = subprocess.run([sys.executable, str(TOOL), base, head, *extra], capture_output=True, text=True, env={"PATH": str(directory)})
    return result.returncode, result.stdout


def check(condition, message, output):
    if not condition:
        print("FAIL %s\n%s" % (message, output))
        sys.exit(1)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fake = directory / "cuobjdump"
        fake.write_text(FAKE_CUOBJDUMP.format(python=sys.executable))
        fake.chmod(0o755)
        gemm = ["LDG.E R2, desc[UR4][R2.64]", "MOV R4, 0x1f0", "LDC R5, c[0x4][0x10]", "FFMA R6, R2, R4, R6", "EXIT"]
        head_kernel = ["LDG.E R2, desc[UR4][R2.64]", "HMMA.16816.F32 R4, R8, R12, R4", "EXIT"]
        base = cubin(directory, "base.cubin", [("_Z6LmGemmv", gemm, 117), ("_Z8LmSkinnyv", head_kernel, 64)])
        relocated = cubin(directory, "relocated.cubin", [("_Z6LmGemmv", [line.replace("0x1f0", "0x2a8").replace("0x10]", "0x58]") for line in gemm], 117), ("_Z8LmSkinnyv", head_kernel, 64)])
        code, output = run(directory, base, relocated)
        check(code == 0 and "kernels 2 identical 2 unexpected 0" in output, "relocated immediates and constant-bank symbols compare equal", output)
        changed = cubin(directory, "changed.cubin", [("_Z6LmGemmv", gemm[:3] + ["FFMA R6, R4, R2, R6"] + gemm[4:], 60), ("_Z8LmSkinnyv", head_kernel, 64)])
        code, output = run(directory, base, changed)
        check(code == 1 and "UNEXPECTED changed REG 117 STACK 0 SHARED 0 LOCAL 0 -> REG 60 STACK 0 SHARED 0 LOCAL 0 _Z6LmGemmv" in output, "a changed kernel fails without --allow and names its register change", output)
        code, output = run(directory, base, changed, "--allow", "LmSkinny")
        check(code == 1 and "unexpected 1" in output, "a change outside the allow pattern fails", output)
        code, output = run(directory, base, changed, "--allow", "LmGemm")
        check(code == 0 and "allowed changed" in output, "a change inside the allow pattern passes", output)
        added = cubin(directory, "added.cubin", [("_Z6LmGemmv", gemm, 117), ("_Z8LmSkinnyv", head_kernel, 64), ("_Z5LmNewv", ["EXIT"], 8)])
        code, output = run(directory, base, added)
        check(code == 1 and "UNEXPECTED added" in output, "a new kernel counts as a change", output)
        empty = cubin(directory, "empty.cubin", [])
        code, output = run(directory, base, empty)
        check(code == 1 and "no kernels parsed from %s" % empty in output, "a cubin that yields no kernels fails instead of comparing nothing", output)
    print("PASS kernel codegen diff: relocations normalize, changes outside ALLOW fail with or without --allow, empty dumps fail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
