import argparse
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKLC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "sklc")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SKLC_DIR not in sys.path:
    sys.path.append(SKLC_DIR)

from cubin_elf import build_manifest
from skl2_frontend import Skl2CompileError, load_unit
from skl2_lower import build_lockfile, compile_unit, resolve_models, verify_lockfile


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    is_lock_cmd = bool(raw_argv and raw_argv[0] == "lock")
    if is_lock_cmd:
        raw_argv = raw_argv[1:]
    ap = argparse.ArgumentParser(prog="skl2c")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--arch", default="sm121", choices=["sm121", "sm121a", "sm120", "sm120a"])
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--rename", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--emit-model-manifest", default=None)
    ap.add_argument("--verify-lock", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args(raw_argv)

    renames = {}
    for spec in args.rename:
        if "=" not in spec:
            raise SystemExit(f"invalid --rename spec (expected SRC=DST): {spec!r}")
        src, dst = spec.split("=", 1)
        if not src or not dst:
            raise SystemExit(f"invalid --rename spec (expected SRC=DST): {spec!r}")
        renames[src] = dst

    repo_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    search_roots = [os.path.join(repo_root, "inference", "skl2")]

    try:
        unit = load_unit(args.inputs, search_roots=search_roots)
        if args.verify_lock:
            with open(args.verify_lock, "r", encoding="utf-8") as f:
                verify_lockfile(unit, json.load(f))
        if is_lock_cmd:
            lock_doc = build_lockfile(unit, args.model if args.model and "all" not in args.model else None)
            out_txt = json.dumps(lock_doc, indent=2, sort_keys=True) + "\n"
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(out_txt)
            else:
                print(out_txt, end="")
            return 0
        if args.model or args.emit_model_manifest:
            plan = resolve_models(unit, args.model if args.model and "all" not in args.model else None)
            out_txt = json.dumps(plan, indent=2, sort_keys=True) + "\n"
            if args.emit_model_manifest:
                with open(args.emit_model_manifest, "w", encoding="utf-8") as f:
                    f.write(out_txt)
            else:
                print(out_txt, end="")
            if not args.output and not args.check:
                return 0

        images, cubin = compile_unit(
            unit,
            arch=args.arch,
            only_set=set(args.only) if args.only else None,
            renames=renames,
            allow_unused_lib=bool(args.only),
        )
    except Skl2CompileError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "wb") as f:
            f.write(cubin)
    if args.manifest:
        combined = []
        for fp in unit["files"]:
            with open(fp, "rb") as f:
                combined.append(f.read())
        src_label = args.inputs[0] if len(args.inputs) == 1 else ",".join(args.inputs)
        manifest = build_manifest(images, cubin, src_label, b"\n".join(combined), arch=args.arch)
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write(manifest)
    if args.stats or args.check:
        sha = hashlib.sha256(cubin).hexdigest()[:12]
        print(f"skl2c: {len(images)} kernels compiled for {args.arch} ({len(cubin)}B, sha256={sha})")
        for k in images:
            print(f"  {k.name}: insns={len(k.insns)} regs={k.registers} block={k.max_threads} smem={k.smem_bytes} barriers={k.barriers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
