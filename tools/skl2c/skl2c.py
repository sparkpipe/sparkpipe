import argparse
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKLC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "sklc")
if SCRIPT_DIR not in sys.path: sys.path.insert(0, SCRIPT_DIR)
if SKLC_DIR not in sys.path: sys.path.append(SKLC_DIR)

from cubin_elf import build_manifest
from skl2_frontend import Skl2CompileError, load_unit
from skl2_lint import audit_repo
from skl2_lower import build_lockfile, compile_unit, emit_c_header, emit_embedded_c_header, emit_policy_header, resolve_models, verify_lockfile


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    is_lock = bool(raw and raw[0] == "lock")
    if is_lock: raw = raw[1:]
    ap = argparse.ArgumentParser(prog="skl2c")
    ap.add_argument("inputs", nargs="*")
    for flg, kw in (
        (("-o", "--output"), {"default": None}), (("--manifest",), {"default": None}),
        (("--arch",), {"default": "sm121", "choices": ["sm121", "sm121a", "sm_121", "sm_121a", "sm120", "sm120a", "sm_120", "sm_120a"]}),
        (("--only",), {"action": "append", "default": []}), (("--rename",), {"action": "append", "default": []}),
        (("--model",), {"action": "append", "default": []}), (("--emit-model-manifest",), {"default": None}),
        (("--emit-c-header",), {"default": None}), (("--emit-embedded-h",), {"default": None}), (("--emit-policy-h",), {"default": None}),
        (("--prefix",), {"default": "Skl2Embedded"}), (("--audit-repo",), {"default": None}),
        (("--verify-lock",), {"default": None}), (("--check",), {"action": "store_true"}), (("--stats",), {"action": "store_true"}),
    ):
        ap.add_argument(*flg, **kw)
    args = ap.parse_args(raw)
    args.arch = args.arch.replace("sm_", "sm")
    if args.audit_repo:
        print(json.dumps(audit_repo(args.audit_repo), indent=2, sort_keys=True))
        return 0
    if not args.inputs:
        ap.error("the following arguments are required: inputs")

    renames = {}
    for spec in args.rename:
        if "=" not in spec or not all(spec.split("=", 1)):
            raise SystemExit(f"invalid --rename spec (expected SRC=DST): {spec!r}")
        src, dst = spec.split("=", 1)
        renames[src] = dst

    search_roots = [os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "inference", "skl2")]
    model_plan = None
    try:
        unit = load_unit(args.inputs, search_roots=search_roots)
        if args.verify_lock:
            with open(args.verify_lock, "r", encoding="utf-8") as f: verify_lockfile(unit, json.load(f))
        if args.emit_policy_h:
            with open(args.emit_policy_h, "w", encoding="utf-8") as f: f.write(emit_policy_header(unit))
            if not args.output and not args.check and not args.emit_c_header and not args.emit_embedded_h and not args.model: return 0
        if is_lock or args.model or args.emit_model_manifest:
            doc = build_lockfile(unit, args.model if args.model and "all" not in args.model else None) if is_lock else resolve_models(unit, args.model if args.model and "all" not in args.model else None)
            if not is_lock: model_plan = doc
            out_txt, dst_p = json.dumps(doc, indent=2, sort_keys=True) + "\n", (args.output if is_lock else args.emit_model_manifest)
            if dst_p:
                with open(dst_p, "w", encoding="utf-8") as f: f.write(out_txt)
            elif not args.emit_c_header and not args.emit_embedded_h and not args.output: print(out_txt, end="")
            if is_lock or (not args.output and not args.check and not args.emit_c_header and not args.emit_embedded_h): return 0
        only_set = set(args.only) if args.only else None
        if not only_set and model_plan and not args.emit_c_header:
            avail = {k["name"] for k in unit["kernels"]} | {i["name"] for i in unit["instantiations"]}
            picked = {k for m in model_plan["models"].values() for k in m.get("compiled_kernels", []) if k in avail}
            if picked: only_set = picked
        images, cubin = compile_unit(unit, arch=args.arch, only_set=only_set, renames=renames, allow_unused_lib=bool(only_set))
    except Skl2CompileError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "wb") as f: f.write(cubin)
    if args.emit_c_header:
        with open(args.emit_c_header, "w", encoding="utf-8") as f: f.write(emit_c_header(images, model_plan=model_plan, arch=args.arch, unit=unit))
    if args.manifest or args.emit_embedded_h:
        combined = [open(fp, "rb").read() for fp in unit["files"]]
        src_label = args.inputs[0] if len(args.inputs) == 1 else ",".join(args.inputs)
        manifest_txt = build_manifest(images, cubin, src_label, b"\n".join(combined), arch=args.arch)
        if args.manifest:
            with open(args.manifest, "w", encoding="utf-8") as f: f.write(manifest_txt)
        if args.emit_embedded_h:
            with open(args.emit_embedded_h, "w", encoding="utf-8") as f: f.write(emit_embedded_c_header(images, cubin, manifest_txt, prefix=args.prefix))
    if args.stats or args.check:
        print(f"skl2c: {len(images)} kernels compiled for {args.arch} ({len(cubin)}B, sha256={hashlib.sha256(cubin).hexdigest()[:12]})")
        for k in images:
            print(f"  {k.name}: insns={len(k.insns)} regs={k.registers} block={k.max_threads} smem={k.smem_bytes} barriers={k.barriers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
