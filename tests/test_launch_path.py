"""Every kernel launch in production code goes through LM_LAUNCH.

A device backend supplies its launcher behind the LM_LAUNCH family of names in
runtime/launch.h, so a triple-chevron launch in a production source root is a
launch that backend cannot see. tools/launch_sites.py finds those launches and
rewrites them; this test fails while any remain, and checks on scratch text that
the rewrite keeps the kernel, the configuration, the arguments and every line
break, writes out an omitted shared size and stream, and keeps a parenthesized
kernel expression whole.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("launch_sites", ROOT / "tools" / "launch_sites.py")
launch_sites = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch_sites)

REWRITES = (
    (
        "\tLmScaleKernel<256u, true><<<rows, 256u, 0, stream>>>(input, output, rows);\n",
        "\tLM_LAUNCH((LmScaleKernel<256u, true>), rows, 256u, 0, stream, input, output, rows);\n",
    ),
    (
        "\tLmCopyKernel<<<(uint32_t)blocks,LM_THREADS,shared,stream>>>(*args);\n",
        "\tLM_LAUNCH((LmCopyKernel),(uint32_t)blocks,LM_THREADS,shared,stream,*args);\n",
    ),
    (
        "\tLmTableKernel<<<1, PAIRS>>>(table, theta);\n",
        "\tLM_LAUNCH((LmTableKernel), 1, PAIRS, 0, 0, table, theta);\n",
    ),
    (
        "\tLmGatherKernel<<<(count + 255u) / 256u,\n\t\t256u, 0, (cudaStream_t)queue>>>(\n\t\tsources, destination);\n",
        "\tLM_LAUNCH((LmGatherKernel), (count + 255u) / 256u,\n\t\t256u, 0, (cudaStream_t)queue,\n\t\tsources, destination);\n",
    ),
    (
        "    LmTileKernel<\n        A,\n        B>\n        <<<plan.grid, plan.threads, shared, stream>>>(\n            args);\n",
        "    LM_LAUNCH((LmTileKernel<\n        A,\n        B>),\n        plan.grid, plan.threads, shared, stream,\n            args);\n",
    ),
    (
        "\t(wide != 0u ? LmWideKernel : LmNarrowKernel)<<<dim3(rows, heads), 64u, 0, stream>>>(x);\n",
        "\tLM_LAUNCH(((wide != 0u ? LmWideKernel : LmNarrowKernel)), dim3(rows, heads), 64u, 0, stream, x);\n",
    ),
    (
        "\tif ( rows != 0u )\n\t\tdetail::LmRowsKernel<4u><<<blocks < 1024u ? blocks : 1024u, 128u, 0, stream>>>(a, \"<<<\");\n",
        "\tif ( rows != 0u )\n\t\tLM_LAUNCH((detail::LmRowsKernel<4u>), blocks < 1024u ? blocks : 1024u, 128u, 0, stream, a, \"<<<\");\n",
    ),
)

REFUSALS = (
    "\tLmEmptyKernel<<<1, 32, 0, stream>>>();\n",
    "\tLmBrokenKernel<<<1>>>(x);\n",
    "\tLmOpenKernel<<<1, 32, 0, stream>>>(x;\n",
)


def check_rewrites():
    failures = []
    for source, expected in REWRITES:
        try:
            result, count = launch_sites.rewrite_text(source)
        except launch_sites.SiteError as error:
            failures.append("rewrite refused %r: %s" % (source, error))
            continue
        if count != 1 or result != expected:
            failures.append("rewrite of %r gave %r, expected %r" % (source, result, expected))
        if result.count("\n") != source.count("\n"):
            failures.append("rewrite of %r moved lines" % source)
    for source in REFUSALS:
        try:
            result, _ = launch_sites.rewrite_text(source)
        except launch_sites.SiteError:
            continue
        failures.append("rewrite accepted malformed launch %r as %r" % (source, result))
    return failures


def check_scan():
    failures = []
    with tempfile.TemporaryDirectory(prefix="launch-path-") as scratch:
        root = Path(scratch)
        kernel = root / "inference/kernels/control.cuh"
        kernel.parent.mkdir(parents=True)
        kernel.write_text("static void Run(float *x, cudaStream_t stream)\n{\n\tLmControlKernel<<<1, 32, 0, stream>>>(x);\n}\n")
        header = root / launch_sites.LAUNCH_HEADER
        header.parent.mkdir(parents=True)
        header.write_text("#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \\\n\tkernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__)\n")
        found = launch_sites.scan(root)
        if list(found) != ["inference/kernels/control.cuh"] or [line for line, _ in found["inference/kernels/control.cuh"]] != [3]:
            failures.append("scan of a scratch tree found %r" % found)
        kernel.write_text(launch_sites.rewrite_text(kernel.read_text())[0])
        if launch_sites.scan(root):
            failures.append("scan still finds a launch after the rewrite")
    return failures


def main():
    failures = check_rewrites() + check_scan()
    found = launch_sites.scan(ROOT)
    for relative, sites in sorted(found.items()):
        if isinstance(sites, launch_sites.SiteError):
            failures.append("%s: %s" % (relative, sites))
            continue
        for line, _ in sites:
            failures.append("%s:%d: kernel launch outside LM_LAUNCH" % (relative, line))
    for failure in failures:
        print("  FAIL %s" % failure)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("PASS every production kernel launch goes through LM_LAUNCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
