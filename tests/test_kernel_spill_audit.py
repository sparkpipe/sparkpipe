"""Host checks of tools/kernel_spill_audit.py on cuobjdump resource listings.

A kernel of the skinny, cell, stream GEMM or attention family that keeps a
stack frame or local memory must be refused, except an attention kernel whose
PENDING entry matches its frame exactly; a larger frame, a smaller one, and an
entry whose kernel keeps no frame or is gone must be refused too. A frame in
any other family is reported and not refused. With --kernel names, a frame in
any function whose name contains one of them is refused and other functions
are not checked; an empty name checks every function. The tool's PENDING list
names only attention kernels, and the CUDA sm_121a compile gate runs the audit
on every serving adapter and CUDA program.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("kernel_spill_audit", ROOT / "tools" / "kernel_spill_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def mangled(name, suffix="Ev"):
    return f"_Z{len(name)}{name}{suffix}"


SKINNY = mangled("LmSkinnyGroupedKernel", "I7LmNvfp4Lj32ELj16ELj2EEv17LmSkinnyArguments")
SKINNY_CLEAN = mangled("LmSkinnyGroupedKernel", "I7LmNvfp4Lj32ELj4ELj4EEv17LmSkinnyArguments")
CELL = mangled("LmCellMmaKernel", "ILj8EEvPKv")
STREAM = mangled("LmStreamGemmKernel", "I7LmMxfp8Lj64ELj128ELj128ELj32ELb0EEv21LmStreamGemmArguments")
ATTEND_NEW = mangled("D41AttendKernel", "ILj512ELj2EEv13LmKvShardView")
ATTEND_SAME = mangled("LmLatentShardPrefillKernel", "ILj512ELj64ELj1EEv13LmKvShardView")
ATTEND_GREW = mangled("LmGqaShardTiledKernel", "ILj2ELj256ELj256EEv13LmKvShardView")
ATTEND_FELL = mangled("LmAttentionBlockPartialKernel", "ILj192ELj128EEvPKt")
ATTEND_CLEAN = mangled("LmLatentAttentionDecodeSplitKernel", "ILj256EEvPKt")
ATTEND_GONE = mangled("D41WindowFoldWholeKernel", "EvPf")
GATHER = mangled("SparkTpGatherRanksKernel", "EPt18SparkTpRankSourcesjj")
STATIC_GEMM = "_ZL12LmGemmKernelILj64EEv15LmGemmArguments"
NESTED = "_ZN5outer18LmStreamGemmKernelILj16EEEvv"

LISTING = f"""
Resource usage:
 Common:
  GLOBAL:0
 Function {SKINNY}:
  REG:255 STACK:7000 SHARED:8324 LOCAL:7000 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {SKINNY_CLEAN}:
  REG:128 STACK:0 SHARED:4228 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {CELL}:
  REG:96 STACK:8 SHARED:8324 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {STREAM}:
  REG:254 STACK:128 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {ATTEND_NEW}:
  REG:168 STACK:168 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {ATTEND_SAME}:
  REG:168 STACK:72 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {ATTEND_GREW}:
  REG:255 STACK:16 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {ATTEND_FELL}:
  REG:168 STACK:40 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {ATTEND_CLEAN}:
  REG:109 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function {GATHER}:
  REG:38 STACK:128 SHARED:0 LOCAL:0 CONSTANT[0]:1352 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

PENDING = {ATTEND_SAME: 72, ATTEND_GREW: 8, ATTEND_FELL: 56, ATTEND_CLEAN: 128, ATTEND_GONE: 64}


def main():
    failures = []
    found = [("adapter.so",) + item for item in audit.resources(LISTING)]
    if len(found) != 10:
        failures.append("the listing parsed into %d functions, not 10" % len(found))
    expected_families = {SKINNY: "skinny", CELL: "cell", STREAM: "stream GEMM", ATTEND_NEW: "attention", ATTEND_SAME: "attention",
                         ATTEND_GREW: "attention", ATTEND_FELL: "attention", ATTEND_CLEAN: "attention", ATTEND_GONE: "attention",
                         GATHER: "collective", STATIC_GEMM: "tensor-map GEMM", NESTED: "stream GEMM"}
    for name, label in expected_families.items():
        if audit.family(name) != label:
            failures.append("%s is in family %s, not %s" % (name, audit.family(name), label))
    if audit.family(mangled("LmRouteBuildKernel")) != "other":
        failures.append("a routing kernel was put in a named family")
    refused, reported = audit.judge(audit.frames(found), PENDING)
    refused_names = {name for name, _ in refused}
    reported_names = {name for name, _ in reported}
    verdicts = {name: verdict for name, verdict in refused + reported}
    for name, why in ((SKINNY, "a skinny kernel with a frame"), (CELL, "a cell kernel with a frame"), (STREAM, "a stream GEMM with a frame"),
                      (ATTEND_NEW, "an attention kernel with a frame and no PENDING entry")):
        if name not in refused_names:
            failures.append(why + " was not refused")
    if ATTEND_SAME in refused_names or ATTEND_SAME not in reported_names:
        failures.append("an attention frame that matches its PENDING entry was refused or not reported")
    if ATTEND_GREW not in refused_names or "grew" not in verdicts.get(ATTEND_GREW, ""):
        failures.append("an attention frame larger than its PENDING entry was not refused")
    if ATTEND_FELL not in refused_names or "lower" not in verdicts.get(ATTEND_FELL, ""):
        failures.append("an attention frame smaller than its PENDING entry was not refused until the entry is lowered")
    if ATTEND_CLEAN not in refused_names or "remove" not in verdicts.get(ATTEND_CLEAN, ""):
        failures.append("a PENDING entry whose kernel keeps no frame was not refused")
    if ATTEND_GONE not in refused_names or "remove" not in verdicts.get(ATTEND_GONE, ""):
        failures.append("a PENDING entry whose kernel is in no artifact was not refused")
    if GATHER in refused_names or GATHER not in reported_names:
        failures.append("a frame outside the hot-path families was refused or not reported")
    if SKINNY_CLEAN in refused_names | reported_names:
        failures.append("a kernel without a frame was listed")
    plain = [item for item in audit.resources(LISTING)]
    named, checked = audit.refusals(plain, ["LmSkinnyGroupedKernel"])
    if checked != 2 or [name for name, _, _, _ in named] != [SKINNY]:
        failures.append("--kernel LmSkinnyGroupedKernel did not check the two grouped kernels and refuse only the one with a frame")
    named, checked = audit.refusals(plain, [""])
    if checked != len(plain) or {name for name, _, _, _ in named} != {SKINNY, CELL, STREAM, ATTEND_NEW, ATTEND_SAME, ATTEND_GREW, ATTEND_FELL, GATHER}:
        failures.append("--kernel '' did not check every function and refuse every frame")
    for name, size in audit.PENDING.items():
        if audit.family(name) != audit.PENDING_FAMILY or size <= 0:
            failures.append("the tool's PENDING entry %s is not an attention kernel with a frame" % name)
    gate = (ROOT / "tools" / "cuda13_sm121a_compile_gate.sh").read_text()
    if 'tools/kernel_spill_audit.py" "${spill_audited[@]}"' not in gate or '"${adapters[@]}" "${cuda_programs[@]}"' not in gate:
        failures.append("the sm_121a compile gate does not run the spill audit on every serving adapter and CUDA program")
    for failure in failures:
        print("FAIL " + failure)
    if not failures:
        print("PASS kernel spill audit: frames in the skinny, cell, stream GEMM and attention families are refused apart from exact "
              "attention PENDING entries, PENDING can only shrink, other frames are reported, and --kernel refuses every frame in the functions it names")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
