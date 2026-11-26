import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from codegen import compile_program
from cubin_elf import build_cubin, build_manifest
from frontend import parse_skl


def compile_skl_source(src_text, source_path="input.skl", arch="sm121"):
    prog = parse_skl(src_text)
    kernels = compile_program(prog)
    cubin = build_cubin(kernels, arch=arch)
    manifest = build_manifest(kernels, cubin, source_path, src_text.encode("utf-8"), arch=arch)
    return kernels, cubin, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sklc")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--arch", default="sm121", choices=["sm121", "sm121a", "sm120", "sm120a"])
    ap.add_argument("--rename", action="append", default=[])
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(argv)

    renames = {}
    for spec in args.rename:
        if "=" not in spec:
            raise ValueError(f"invalid --rename spec (expected SRC=DST): {spec!r}")
        src_name, dst_name = spec.split("=", 1)
        if not src_name or not dst_name:
            raise ValueError(f"invalid --rename spec (expected SRC=DST): {spec!r}")
        renames[src_name] = dst_name
    only_set = set(args.only)

    all_kernels = []
    combined_src = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        combined_src.append(text)
        prog = parse_skl(text)
        if only_set:
            prog["kernels"] = [k for k in prog["kernels"] if k["name"] in only_set]
        for k in prog["kernels"]:
            if k["name"] in renames:
                k["name"] = renames[k["name"]]
        all_kernels.extend(compile_program(prog))
    if not all_kernels:
        raise ValueError("no kernels selected to compile")

    cubin = build_cubin(all_kernels, arch=args.arch)
    with open(args.output, "wb") as f:
        f.write(cubin)

    if args.manifest:
        src_label = args.inputs[0] if len(args.inputs) == 1 else ",".join(args.inputs)
        manifest = build_manifest(
            all_kernels,
            cubin,
            src_label,
            "\n".join(combined_src).encode("utf-8"),
            arch=args.arch,
        )
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write(manifest)

    if args.stats:
        for k in all_kernels:
            print(
                f"{k.name}: insns={len(k.insns)} regs={k.registers} "
                f"block={k.max_threads} smem={k.smem_bytes} barriers={k.barriers}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
