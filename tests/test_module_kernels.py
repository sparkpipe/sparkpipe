"""Every serving adapter installs its cubins and kernel manifests beside it.

The nvrm backend loads a module's kernels from <module>.kernels/: one
<tu>.cubin and one <tu>.manifest per .cu translation unit, from the compile
that built the module. This test checks, without a GPU or nvcc, that every
CUDA serving adapter is built through the nvcc_module recipe that keeps those
outputs and installs them without adding ptxas options to the build, and
removes the module when a step fails; that tools/module_kernels.py pairs each
translation unit with exactly its kept PTX and cubin, takes the ptxas report
from a ptxas run that reproduces the kept cubin and refuses one that does not,
passes the compile's -Xptxas options to that run, removes every other
intermediate, accounts for the kept device-link image nvcc embeds without
installing it, refuses a module whose installed cubins differ from the
embedded ones and relocatable device code, and leaves nothing behind when it
fails; and that lane staging copies the directory with the adapter.
"""
import importlib.util
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("module_kernels", ROOT / "tools" / "module_kernels.py")
module_kernels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module_kernels)


def script(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def check_makefile():
    failures = []
    text = (ROOT / "Makefile").read_text().replace("\\\n", " ")
    adapters = re.findall(r"^([A-Z0-9_]+_SERVING_ADAPTER)\s*:=", text, re.M)
    adapters = [name for name in adapters if not name.endswith("_METAL_SERVING_ADAPTER")]
    for name in adapters:
        rule = re.search(r"^\$\(%s\):[^\n]*\n((?:\t[^\n]*\n)+)" % name, text, re.M)
        if rule is None:
            define = re.search(r"\$\(%s\)" % name, text)
            if define is None:
                failures.append("%s has no rule" % name)
            continue
        recipe = rule.group(1)
        if "$(NVCC) $(NVCCFLAGS)" in recipe:
            failures.append("%s compiles with nvcc outside nvcc_module, so its kernels are not installed" % name)
    raw = [line for line in text.splitlines() if "spark-gated artifact" in line and "$(call nvcc_module," not in line]
    for line in raw:
        failures.append("an adapter recipe does not build through nvcc_module: %s" % line.strip()[:120])
    flags = re.search(r"^MODULE_KERNELS_FLAGS = (.*)$", text, re.M)
    define = re.search(r"^define nvcc_module\n(.*?)\nendef", text, re.M | re.S)
    if flags is None or flags.group(1).strip() != "--keep --keep-dir $@.kernels" or define is None:
        failures.append("the nvcc_module recipe does not keep the compile's outputs with --keep alone")
    elif "-Xptxas" in define.group(1) or "rm -rf $@ $@.kernels" not in define.group(1) or "|| { rm -f $@; exit 1; }" not in define.group(1):
        failures.append("the nvcc_module recipe adds ptxas options to the build or leaves the module behind when a step fails")
    stage = (ROOT / "tools" / "stage_lane.sh").read_text()
    if "$ADAPTER_FILE.kernels" not in stage:
        failures.append("tools/stage_lane.sh does not copy the adapter's kernels directory")
    return failures


def check_install():
    failures = []
    with tempfile.TemporaryDirectory(prefix="module-kernels-") as scratch:
        root = Path(scratch)
        module = root / "libmodel.so"
        module.write_bytes(b"module")
        kernels = root / "libmodel.so.kernels"

        def keep():
            if kernels.exists():
                for path in kernels.iterdir():
                    path.unlink()
            kernels.mkdir(exist_ok=True)
            for stem in ("stage", "model"):
                (kernels / (stem + ".ptx")).write_text(".entry k()\n{\n}\n")
                (kernels / (stem + ".cubin")).write_bytes(b"cubin-" + stem.encode())
                (kernels / (stem + ".cpp1.ii")).write_text("intermediate")
            (kernels / "libmodel_dlink.cubin").write_bytes(b"link-image")

        keep()
        calls = root / "ptxas.calls"
        ptxas = script(root / "ptxas", 'echo "$@" >> %s; for last; do :; done; ptx=""; out=""; verbose=0; previous=""\n'
                       'for word in "$@"; do [ "$previous" = -o ] && out="$word"; [ "$word" = -v ] && verbose=1; case "$word" in *.ptx) ptx="$word";; esac; previous="$word"; done\n'
                       'cp "${ptx%%.ptx}.cubin" "$out"; [ $verbose = 1 ] && echo "ptxas info    : Used 8 registers, used 0 barriers" >&2; exit 0\n' % calls)
        wrong = script(root / "wrong_ptxas", 'previous=""; for word in "$@"; do [ "$previous" = -o ] && echo other > "$word"; previous="$word"; done; exit 0\n')
        manifest_tool = script(root / "manifest.sh", 'while [ $# -gt 0 ]; do [ "$1" = --output ] && out="$2"; [ "$1" = --ptxas-log ] && log="$2"; shift; done; cat "$log" > "$out"\n')
        extracted = root / "extracted"
        extracted.mkdir()
        (extracted / "a.cubin").write_bytes(b"cubin-stage")
        (extracted / "b.cubin").write_bytes(b"cubin-model")
        (extracted / "c.cubin").write_bytes(b"link-image")
        cuobjdump = script(root / "cuobjdump", 'cp %s/*.cubin . \n' % extracted)
        manifest = ["/bin/sh", str(manifest_tool)]
        sources = ["a/stage.cu", "b/model.cu"]
        count = module_kernels.install(module, "nvcc", "-O3 -Xptxas=-dlcm=cg", "sm_121a", sources, str(cuobjdump), str(ptxas), manifest)
        names = sorted(path.name for path in kernels.iterdir())
        if count != 2 or names != ["model.cubin", "model.manifest", "stage.cubin", "stage.manifest"]:
            failures.append("install left %s after %d units" % (names, count))
        elif "Used 8 registers" not in (kernels / "stage.manifest").read_text():
            failures.append("the manifest tool did not get the ptxas report of the replay run")
        lines = calls.read_text().splitlines() if calls.exists() else []
        if len(lines) != 4 or any(not line.startswith("-arch=sm_121a -m64 -dlcm=cg ") for line in lines) or sum("-v" in line.split() for line in lines) != 2:
            failures.append("ptxas did not run twice per unit with the compile's options and -v on the second run: %s" % lines)
        keep()
        try:
            module_kernels.install(module, "nvcc", "-O3", "sm_121a", sources, str(cuobjdump), str(wrong), manifest)
            failures.append("install accepted a ptxas run that does not reproduce the kept cubin")
        except module_kernels.KernelsError as error:
            if "does not reproduce the kept cubin" not in str(error):
                failures.append("a ptxas replay mismatch failed for another reason: %s" % error)
        keep()
        (kernels / "model.cubin").unlink()
        (kernels / "model.sm_121a.cubin").write_bytes(b"cubin-model")
        try:
            module_kernels.install(module, "nvcc", "-O3", "sm_121a", sources, str(cuobjdump), str(ptxas), manifest)
            failures.append("install accepted a unit without its kept <tu>.cubin")
        except module_kernels.KernelsError as error:
            if "nvcc kept no model.cubin" not in str(error):
                failures.append("a missing kept cubin failed for another reason: %s" % error)
        keep()
        (extracted / "b.cubin").write_bytes(b"other")
        try:
            module_kernels.install(module, "nvcc", "-O3", "sm_121a", sources, str(cuobjdump), str(ptxas), manifest)
            failures.append("install accepted installed cubins that differ from the embedded ones")
        except module_kernels.KernelsError as error:
            if "not the 3 embedded" not in str(error):
                failures.append("a cubin mismatch failed for another reason: %s" % error)
        (extracted / "b.cubin").write_bytes(b"cubin-model")
        keep()
        (kernels / "libmodel_dlink.cubin").unlink()
        try:
            module_kernels.install(module, "nvcc", "-O3", "sm_121a", sources, str(cuobjdump), str(ptxas), manifest)
            failures.append("install accepted an embedded cubin that is neither installed nor the kept device-link image")
        except module_kernels.KernelsError as error:
            if "device-link image (0) are not the 3 embedded" not in str(error):
                failures.append("an unaccounted embedded cubin failed for another reason: %s" % error)
        keep()
        try:
            module_kernels.install(module, "nvcc", "-O3 -rdc=true", "sm_121a", sources, str(cuobjdump), str(ptxas), manifest)
            failures.append("install accepted relocatable device code")
        except module_kernels.KernelsError as error:
            if "relocatable device code" not in str(error):
                failures.append("relocatable device code failed for another reason: %s" % error)
        if module_kernels.ptxas_options("-O3 -Xptxas=-dlcm=cg,-O2 -Xptxas -warn-spills --ptxas-options=--allow-expensive-optimizations=true") != [
                "-dlcm=cg", "-O2", "-warn-spills", "--allow-expensive-optimizations=true"]:
            failures.append("the compile's -Xptxas options are not all passed to the replay")
        try:
            module_kernels.stems(["a/stage.cu", "b/stage.cu"])
            failures.append("two translation units with one name were accepted")
        except module_kernels.KernelsError:
            pass
        empty = root / "empty.so"
        empty.write_bytes(b"")
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "module_kernels.py"), "--module", str(empty), "--nvcc", "nvcc",
                                 "--flags=-O3", "--arch", "sm_121a", "x.cu"], capture_output=True, text=True)
        if result.returncode != 1 or "FAIL" not in result.stderr or os.path.exists(str(empty) + ".kernels"):
            failures.append("a module without a keep directory did not fail cleanly: %s" % result.stderr.strip())
    return failures


def main():
    failures = check_makefile() + check_install()
    for failure in failures:
        print("  FAIL %s" % failure)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("PASS every CUDA serving adapter installs its cubins and kernel manifests beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
