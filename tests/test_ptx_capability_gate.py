#!/usr/bin/env python3
"""Assemble the PTX forms this codebase relies on, for the target it ships to.

There is no GPU and no nvcc in most working containers, so nothing here has ever
checked a CUDA-level assumption. ptxas alone is enough to answer one useful
question: does the target actually support the instruction we are about to build
on? That is exactly the assumption that has been carried in prose - "TMA and
clusters ARE available, tcgen05 is NOT" - without anything verifying it.

This gate does not check kernel correctness. It checks instruction availability
and encoding, which is the part that silently changes between architectures.
"""

import os
import shutil
import subprocess
import sys
import tempfile

TARGET = "sm_121a"
PTX_VERSION = "8.8"

# name -> (body, must_assemble)
PROBES = {
    "cp.async.ca": (
        "\tcp.async.ca.shared.global [tile], [%rd2], 16;", True),
    "cp.async.cg": (
        "\tcp.async.cg.shared.global [tile], [%rd2], 16;", True),
    "cp.async.bulk": (
        "\tcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [tile], [%rd2], 1024, [bar];", True),
    "cp.async.bulk.tensor.1d": (
        "\tcp.async.bulk.tensor.1d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [tile], [%rd2, {%r1}], [bar];", True),
    "cp.async.bulk.tensor.2d": (
        "\tcp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [tile], [%rd2, {%r1, %r2}], [bar];", True),
    "cp.async.bulk.tensor.3d": (
        "\tcp.async.bulk.tensor.3d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [tile], [%rd2, {%r1, %r2, %r1}], [bar];", True),
    # The exact forms spark_lm_async_copy.cuh embeds. If one stops assembling,
    # that header is broken and this fails before anyone reaches a GPU.
    "async_copy.ca.4": (
        "\tcp.async.ca.shared.global [tile], [%rd2], 4;", True),
    "async_copy.ca.8": (
        "\tcp.async.ca.shared.global [tile], [%rd2], 8;", True),
    "async_copy.bounded": (
        "\tcp.async.ca.shared.global [tile], [%rd2], 16, %r1;", True),
    "async_copy.cg.bounded": (
        "\tcp.async.cg.shared.global [tile], [%rd2], 16, %r1;", True),
    "async_copy.commit": (
        "\tcp.async.commit_group;", True),
    "async_copy.wait_group": (
        "\tcp.async.wait_group 1;", True),
    "async_copy.wait_all": (
        "\tcp.async.wait_all;", True),
    # .cg exists only at 16 bytes. The header's static_assert enforces this;
    # if the hardware ever relaxes it, that assert is over-strict.
    "async_copy.cg.4.rejected": (
        "\tcp.async.cg.shared.global [tile], [%rd2], 4;", False),
    "mbarrier.expect_tx": (
        "\tmbarrier.expect_tx.shared::cta.b64 [bar], 1024;", True),
    "cluster.barrier": (
        "\tbarrier.cluster.arrive;\n\tbarrier.cluster.wait;", True),
    "cluster.mapa": (
        "\tmov.u32 %r3, %cluster_ctarank;\n"
        "\tmapa.shared::cluster.u32 %r4, %r3, 0;", True),
    "mma.m16n8k16.bf16": (
        "\tmma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    # Documented as unavailable. If one of these starts assembling the target
    # has changed and the kernel strategy should be revisited.
    "tcgen05.alloc": (
        "\ttcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [bar], 32;", False),
    "wgmma.fence": (
        "\twgmma.fence.sync.aligned;", False),
}

MODULE = """.version {version}
.target {target}
.address_size 64

.visible .entry spark_probe(.param .u64 a)
{{
\t.reg .b32 %r<8>;
\t.reg .b64 %rd<8>;
\t.reg .f32 %f<8>;
\t.reg .pred %p<2>;
\t.shared .align 128 .b8 tile[16384];
\t.shared .align 8 .b64 bar[1];
\tld.param.u64 %rd1, [a];
\tcvta.to.global.u64 %rd2, %rd1;
\tmov.u32 %r1, 0;
\tmov.u32 %r2, 0;
{body}
\tret;
}}
"""


def find_ptxas():
    found = shutil.which("ptxas")
    if found:
        return found
    for root in (sys.prefix, "/usr/local/lib", "/usr/lib"):
        for base, _dirs, files in os.walk(root):
            if "ptxas" in files and "cuda_nvcc" in base:
                return os.path.join(base, "ptxas")
    return None


def assembles(ptxas, body, workdir):
    path = os.path.join(workdir, "probe.ptx")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(MODULE.format(version=PTX_VERSION, target=TARGET, body=body))
    result = subprocess.run(
        [ptxas, "-arch=" + TARGET, path, "-o", os.path.join(workdir, "probe.cubin")],
        capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip().split("\n")[0]


def main():
    ptxas = find_ptxas()
    if ptxas is None:
        print("SKIP ptx capability gate: no ptxas on this machine")
        return 0
    failures = []
    with tempfile.TemporaryDirectory() as workdir:
        for name, (body, required) in sorted(PROBES.items()):
            ok, message = assembles(ptxas, body, workdir)
            if ok == required:
                print("  {:<26} {}".format(name, "available" if ok else "unavailable, as documented"))
                continue
            if required:
                failures.append("{} must assemble for {} but did not: {}".format(name, TARGET, message))
            else:
                failures.append("{} now assembles for {}; the target gained a capability "
                                "and the kernel strategy should be revisited".format(name, TARGET))
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS ptx capability gate ({} on {})".format(len(PROBES), TARGET))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
