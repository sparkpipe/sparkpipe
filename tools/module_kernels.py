#!/usr/bin/env python3
"""Install a module's cubins and kernel manifests beside it.

A module that links nvcc-compiled code (a serving adapter or an executable)
is built with `--keep --keep-dir <module>.kernels`, which leaves the module's
device code byte for byte as it is without them. This step turns that keep
directory into what the nvrm backend loads: for every .cu translation unit of
the module, <tu>.cubin (the compile's own cubin) and <tu>.manifest, written by
tools/nvrm_kernel_manifest.py from the same compile's PTX and cubin and a
ptxas report. Every other intermediate is removed.

nvcc keeps <tu>.ptx and <tu>.cubin for the compile's one code target. The
report comes from ptxas run again on the kept PTX with the compile's ptxas
options (-arch, -m64 and any -Xptxas options in the flags). That run must
reproduce the kept cubin byte for byte, so the report describes the shipped
code; the same run with -v gives the report. The module's embedded cubins
must be exactly the installed ones plus the device-link image nvcc keeps as
<module stem>_dlink.cubin, which holds no kernels and is not installed: the
step extracts the module's cubins with cuobjdump and fails unless the two
sets of digests are equal. Relocatable device code is refused, since its
device-link image would carry the code.
Anything missing, different or not reproduced fails, and the build then
removes the module and its directory, so a module never ships without its
pairs or with stale ones.

Usage:
    python3 -I tools/module_kernels.py --module PATH --nvcc NVCC --flags="FLAGS" \\
        --arch sm_121a SOURCE.cu...
"""
import argparse
import hashlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_TOOL = ROOT / "tools" / "nvrm_kernel_manifest.py"


class KernelsError(Exception):
    pass


def kept_file(directory, name):
    path = directory / name
    if not path.is_file():
        stem = name.rsplit(".", 1)[0]
        found = sorted(entry.name for entry in directory.iterdir() if entry.name.startswith(stem + "."))
        raise KernelsError("%s: nvcc kept no %s (found %s)" % (directory, name, found or "nothing for that unit"))
    return path


def stems(sources):
    names = [Path(source).name[:-len(".cu")] for source in sources]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise KernelsError("translation units share a name in one module: %s" % ", ".join(duplicated))
    return names


def ptxas_options(flags):
    options = []
    words = shlex.split(flags)
    index = 0
    while index < len(words):
        word = words[index]
        value = None
        for spelling in ("-Xptxas", "--ptxas-options"):
            if word == spelling and index + 1 < len(words):
                index += 1
                value = words[index]
            elif word.startswith(spelling + "="):
                value = word[len(spelling) + 1:]
        if value is not None:
            options += [option for option in value.split(",") if option]
        index += 1
    return options


def ptxas_report(ptxas, arch, options, ptx, cubin, scratch):
    command = [ptxas, "-arch=" + arch, "-m64"] + options
    replay = scratch / (cubin.stem + ".replay.cubin")
    run = subprocess.run(command + [str(ptx), "-o", str(replay)], capture_output=True, text=True)
    if run.returncode != 0:
        raise KernelsError("%s: ptxas cannot compile the kept PTX again: %s" % (ptx, (run.stdout + run.stderr).strip()))
    if replay.read_bytes() != cubin.read_bytes():
        raise KernelsError("%s: %s does not reproduce the kept cubin, so its report would not describe the shipped code" % (cubin, " ".join(command)))
    run = subprocess.run(command + ["-v", str(ptx), "-o", str(scratch / (cubin.stem + ".report.cubin"))], capture_output=True, text=True)
    if run.returncode != 0:
        raise KernelsError("%s: ptxas -v failed on the kept PTX: %s" % (ptx, (run.stdout + run.stderr).strip()))
    log = scratch / (cubin.stem + ".ptxas.log")
    log.write_text(run.stderr + run.stdout)
    return log


def embedded_digests(cuobjdump, module):
    with tempfile.TemporaryDirectory(prefix="module-kernels-") as scratch:
        run = subprocess.run([cuobjdump, "-xelf", "all", str(Path(module).resolve())], cwd=scratch, capture_output=True, text=True)
        if run.returncode != 0:
            raise KernelsError("cuobjdump cannot extract the cubins of %s: %s" % (module, run.stderr.strip()))
        return sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(scratch).iterdir())


RELOCATABLE = ("-rdc=true", "--relocatable-device-code=true", "-dc", "--device-c", "-dlink", "--device-link")


def install(module, nvcc, flags, arch, sources, cuobjdump, ptxas, manifest_command=None):
    manifest_command = manifest_command or [sys.executable, "-I", str(MANIFEST_TOOL)]
    directory = Path(str(module) + ".kernels")
    if not directory.is_dir():
        raise KernelsError("%s is missing; build the module with --keep --keep-dir %s" % (directory, directory))
    relocatable = [word for word in shlex.split(flags) if word in RELOCATABLE]
    if relocatable:
        raise KernelsError("%s: relocatable device code (%s) puts the code in the device-link image; the nvrm backend loads one cubin per translation unit" % (module, " ".join(relocatable)))
    options = ptxas_options(flags)
    keep = set()
    with tempfile.TemporaryDirectory(prefix="module-kernels-") as scratch:
        for source, stem in zip(sources, stems(sources)):
            ptx = kept_file(directory, stem + ".ptx")
            cubin = kept_file(directory, stem + ".cubin")
            log = ptxas_report(ptxas, arch, options, ptx, cubin, Path(scratch))
            manifest = directory / (stem + ".manifest")
            run = subprocess.run(manifest_command + ["--cubin", str(cubin), "--ptx", str(ptx), "--ptxas-log", str(log),
                                  "--nvcc", nvcc, "--flags=" + flags, "--source", source, "--arch", arch, "--output", str(manifest)],
                                 capture_output=True, text=True)
            if run.returncode != 0:
                raise KernelsError("%s: %s" % (source, (run.stdout + run.stderr).strip()))
            keep.update((cubin.name, manifest.name))
    installed = [hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in keep if name.endswith(".cubin")]
    link = directory / (Path(module).name.rsplit(".", 1)[0] + "_dlink.cubin")
    linked = [hashlib.sha256(link.read_bytes()).hexdigest()] if link.is_file() else []
    for path in directory.iterdir():
        if path.name not in keep:
            path.unlink() if path.is_file() else shutil.rmtree(path)
    embedded = embedded_digests(cuobjdump, module)
    if sorted(installed + linked) != embedded:
        raise KernelsError("%s: the installed cubins (%d) and the kept device-link image (%d) are not the %d embedded in the module" % (
            directory, len(installed), len(linked), len(embedded)))
    return len(installed)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cuobjdump", default=None)
    parser.add_argument("--ptxas", default=None)
    parser.add_argument("sources", nargs="+")
    options = parser.parse_args(argv)
    toolkit = Path(shutil.which(options.nvcc) or options.nvcc).resolve().parent
    cuobjdump = options.cuobjdump or str(toolkit / "cuobjdump")
    ptxas = options.ptxas or str(toolkit / "ptxas")
    try:
        count = install(Path(options.module), options.nvcc, options.flags, options.arch, options.sources, cuobjdump, ptxas)
    except (KernelsError, OSError) as error:
        print("FAIL %s" % error, file=sys.stderr)
        shutil.rmtree(str(options.module) + ".kernels", ignore_errors=True)
        return 1
    print("%s.kernels: %d translation units, each cubin reproduced by ptxas and equal to the one embedded" % (options.module, count))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
