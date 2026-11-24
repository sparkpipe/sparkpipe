#!/usr/bin/env python3
"""Keep bulk copies into shared memory free of CUDA driver routines.

On sm_121a, CUDA 13.0's ptxas compiles cp.async.bulk and cp.async.bulk.tensor loads whose destination is
.shared::cluster into calls to routines the CUDA driver resolves when it loads the module
(__cuda_syscall_cp_async_bulk_unicast and __cuda_syscall_cp_async_bulk_tensor_{2d,3d}_tile_unicast). The nvrm
backend loads kernel images without the CUDA driver, so it refuses any image that calls them. The .shared::cta
destination (PTX ISA 8.6) compiles to the copy itself and moves the same bytes into the issuing CTA's shared
memory. No engine copy multicasts or targets another CTA of a cluster, so every such load names .shared::cta.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("inference", "model-families", "modules", "runtime")
SUFFIXES = {".cu", ".cuh", ".h", ".c"}
CLUSTER_LOAD = re.compile(r"cp\.async\.bulk(?:\.tensor\.[1-5]d)?\.shared::cluster\b")


def offenders(text):
    return [number for number, line in enumerate(text.splitlines(), 1) if CLUSTER_LOAD.search(line)]


def main():
    contaminated = 'asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];");'
    clean = 'asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];");'
    if offenders(contaminated) != [1] or offenders(clean) != [] or offenders('asm volatile("cp.async.bulk.shared::cluster.global.mbarrier");') != [1]:
        print("FAIL bulk copy forms: the gate does not recognise a .shared::cluster load")
        return 1
    found = []
    checked = 0
    for root in ROOTS:
        for path in sorted((ROOT / root).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            checked += 1
            for number in offenders(path.read_text(errors="replace")):
                found.append("%s:%d" % (path.relative_to(ROOT), number))
    if found:
        print("FAIL bulk copy forms: these loads target .shared::cluster, which sm_121a compiles into CUDA driver calls the nvrm backend cannot load; use .shared::cta")
        for site in found:
            print("  " + site)
        return 1
    print("PASS bulk copies into shared memory name .shared::cta (%d files checked)" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
