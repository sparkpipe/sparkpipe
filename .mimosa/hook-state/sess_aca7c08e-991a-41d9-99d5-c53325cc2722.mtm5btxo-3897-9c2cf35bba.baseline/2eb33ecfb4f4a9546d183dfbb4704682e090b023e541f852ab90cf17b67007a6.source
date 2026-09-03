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
    # -- Narrow-precision MMA. The quantization ladder rests on these being
    # native rather than emulated by a dequant pass. The .kind:: modifier is
    # mandatory on this family; without it ptxas reports
    # ".kind::f8f6f4 modifier required", which reads like absence and is not.
    "mma.m16n8k32.e4m3": (
        "\tmma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    "mma.f8f6f4.e2m1": (
        "\tmma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    # Mixed operand widths: fp8 activations against fp4 weights in one mma.
    "mma.f8f6f4.e4m3xe2m1": (
        "\tmma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    # DSV4 production atom: independently block-scaled MXFP8 activations and
    # MXFP4 weights.  The unscaled mixed-width probe above is not sufficient:
    # it neither checks the UE8M0 scale operands nor scale_vec::1X grammar.
    "mma.mxf8f6f4.e4m3xe2m1": (
        "\tmma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale"
        ".scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4},"
        " {%r5,%r6}, {%f1,%f2,%f3,%f4}, %r7, {0, 0}, %r8, {0, 0};", True),
    # Shared expert and dense FP8 projections use the same block-scaled family
    # with E4M3 on both operands.
    "mma.mxf8f6f4.e4m3xe4m3": (
        "\tmma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale"
        ".scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4},"
        " {%r5,%r6}, {%f1,%f2,%f3,%f4}, %r7, {0, 0}, %r8, {0, 0};", True),
    # NVFP4: e2m1 data, ue4m3 scale, one scale per 16 elements (scale_vec::4X at
    # k64). Matches SPARK_GLM52_MODEL_NVFP4_GROUP_SIZE.
    "mma.mxf4nvf4.ue4m3.4X": (
        "\tmma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col"
        ".f32.e2m1.e2m1.f32.ue4m3 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6},"
        " {%f1,%f2,%f3,%f4}, %r7, {0, 0}, %r8, {0, 0};", True),
    # MXFP4: e2m1 data, ue8m0 scale, one scale per 32 elements (scale_vec::2X).
    # Matches SPARK_GLM52_MODEL_MXFP4_GROUP_SIZE.
    "mma.mxf4.ue8m0.2X": (
        "\tmma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col"
        ".f32.e2m1.e2m1.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6},"
        " {%f1,%f2,%f3,%f4}, %r7, {0, 0}, %r8, {0, 0};", True),
    # CUTLASS emits scale_vec::4X with ue8m0 for VS=16 under
    # CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED. That is an sm_100 capability and
    # ptxas rejects it here, so the SM120 collective's VS=16 emission cannot be
    # copied onto this target. On sm_121a NVFP4 is 4X with ue4m3. If this ever
    # starts assembling the target gained a capability and the NVFP4 kernel
    # should be revisited.
    "mma.mxf4nvf4.ue8m0.4X.rejected": (
        "\tmma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col"
        ".f32.e2m1.e2m1.f32.ue8m0 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6},"
        " {%f1,%f2,%f3,%f4}, %r7, {0, 0}, %r8, {0, 0};", False),
    # Integer atoms. INT8 and INT4 are native; there is no s6 and no s7 mma of
    # any shape, which is why kernels/formats/ has fp6 and not int6 - six bits
    # exists here as E3M2/E2M3, and a 7-bit scheme would have to unpack to 8 and
    # spend exactly the storage saving it was chosen for.
    "mma.m16n8k32.s8": (
        "\tmma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " {%r1,%r2,%r3,%r4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r1,%r2,%r3,%r4};", True),
    "mma.m16n8k64.s4": (
        "\tmma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"
        " {%r1,%r2,%r3,%r4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%r1,%r2,%r3,%r4};", True),
    "mma.f8f6f4.e3m2": (
        "\tmma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    "mma.f8f6f4.e2m3": (
        "\tmma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32"
        " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4};", True),
    "cvt.e2m1x2.f32": (
        "\tcvt.rn.satfinite.e2m1x2.f32 %rb1, %f1, %f2;", True),
    "cvt.ue8m0x2.f32": (
        "\tcvt.rz.satfinite.ue8m0x2.f32 %rs1, %f1, %f2;", True),
    # -- The forms spark_lm_tma.cuh embeds. TMA plus mbarrier is the structure
    # CUTLASS's SM120 collective uses; if one stops assembling that header is
    # broken and this fails before anyone reaches a GPU.
    "tma.load.2d": (
        "\tcp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [tile], [%rd2, {%r1, %r2}], [bar];", True),
    "tma.load.3d": (
        "\tcp.async.bulk.tensor.3d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [tile], [%rd2, {%r1, %r2, %r3}], [bar];", True),
    "tma.store.2d": (
        "\tcp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%rd2, {%r1, %r2}], [tile];", True),
    "tma.store.commit": ("\tcp.async.bulk.commit_group;", True),
    "tma.store.wait_read": ("\tcp.async.bulk.wait_group.read 0;", True),
    "mbarrier.init": ("\tmbarrier.init.shared::cta.b64 [bar], 128;", True),
    "mbarrier.inval": ("\tmbarrier.inval.shared::cta.b64 [bar];", True),
    "mbarrier.arrive.expect_tx": (
        "\tmbarrier.arrive.expect_tx.shared::cta.b64 %rd3, [bar], 16384;", True),
    "mbarrier.try_wait.parity": (
        "\tmbarrier.try_wait.parity.shared::cta.b64 %p1, [bar], %r1;", True),
    "fence.proxy.async": ("\tfence.proxy.async.shared::cta;", True),
    "elect.sync": ("\telect.sync %r4|%p1, 0xffffffff;", True),
    # ldmatrix produces the exact register layout mma.sync consumes, which is
    # why operand fragments are loaded with it rather than by hand-derived index
    # arithmetic - the one part of an mma kernel a wrong implementation renders
    # silently incorrect while still assembling.
    "ldmatrix.x4": (
        "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4,%r5,%r6,%r7}, [tile];", True),
    "ldmatrix.x2": (
        "\tldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r4,%r5}, [tile];", True),
    "stmatrix.x4": (
        "\tstmatrix.sync.aligned.m8n8.x4.shared.b16 [tile], {%r4,%r5,%r6,%r7};", True),
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
\t.reg .b32 %r<16>;
\t.reg .b64 %rd<8>;
\t.reg .f32 %f<8>;
\t.reg .b16 %rs<8>;
\t.reg .b8  %rb<8>;
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
