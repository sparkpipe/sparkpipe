import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKL2C_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "skl2c")
for _d in (SKL2C_DIR, SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from codegen import compile_program
from cubin_elf import build_cubin, build_manifest
from skl2_frontend import parse_skl2_text
from skl2_lower import lower_unit
import skl2c


def compile_skl_source(src_text, source_path="input.skl2", arch="sm121"):
    prog = lower_unit(parse_skl2_text(src_text, source_path), allow_unused_lib=True)
    kernels = compile_program(prog, strict_reg_budget=True)
    cubin = build_cubin(kernels, arch=arch)
    manifest = build_manifest(kernels, cubin, source_path, src_text.encode("utf-8"), arch=arch)
    return kernels, cubin, manifest


def main(argv=None):
    return skl2c.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
