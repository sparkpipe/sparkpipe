#!/usr/bin/env python3
"""Every GEMM K extent must be a whole number of tiles, for every format.

LmGemmKernel computes k_tiles = input_dimension / TILE_K with an integer
division, and its stagers bound rows and neurons but never K. A trailing
partial K tile is dropped from the dot product. Wrong output, no crash.

The tile geometry static_asserts cannot reach this: they are compile-time and
input_dimension is a runtime argument. runtime/launch.h rejects it at plan
time, which turns silent wrongness into a returned error, but an error at plan
time is still a model that does not run. This gate is the one that fails on a
laptop instead.

The stricter number is INT7's, which tiles at 256 rather than 128 - and INT7
is the format the tree is built around. A width that is a clean multiple of
128 can still be wrong under INT7. GLM52_QK_NOPE_DIM and MIMO25_HEAD_DIM are
both 192 and would produce zero tiles at TILE_K=256; neither reaches a GEMM
today, and this gate is what notices if one starts to.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tile_depths():
    depths = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "inference/kernels/formats/*.cuh"))):
        text = open(path).read()
        match = re.search(r"kTileK\s*=\s*(\d+)u", text)
        if match:
            depths[os.path.basename(path)[:-4]] = int(match.group(1))
    return depths


def config_defines(path):
    values = {}
    for name, value in re.findall(r"#define\s+(\w+)\s+([^\n/]+)", open(path).read()):
        values[name] = value.strip()
    return values


def evaluate(expression, values):
    text = expression
    for _ in range(16):
        names = [t for t in set(re.findall(r"[A-Za-z_]\w*", text)) if t != "u"]
        if not names:
            break
        progressed = False
        for name in names:
            if name in values:
                text = re.sub(r"\b" + name + r"\b", "(" + values[name] + ")", text)
                progressed = True
        if not progressed:
            return None
    text = re.sub(r"(\d)u\b", r"\1", text)
    try:
        return int(eval(text))
    except Exception:
        return None


def k_extents():
    found = []
    for layer in sorted(glob.glob(os.path.join(ROOT, "inference/llms/*/layer.cuh"))):
        model = os.path.basename(os.path.dirname(layer))
        values = config_defines(os.path.join(os.path.dirname(layer), "config.h"))
        # widths derived in layer.cuh count too: QWEN38_27B_Q_DIM and
        # QWEN38_27B_GDN_V_DIM are K extents and live there, not in config.h
        values.update(config_defines(layer))
        source = re.sub(r"\s+", " ", open(layer).read())
        for match in re.finditer(r"LmGemmLaunch<[^(]*\(([^;]*?)\);", source):
            arguments = [a.strip() for a in match.group(1).split(",")]
            if len(arguments) < 10:
                continue
            expression = arguments[7]
            extent = evaluate(expression, values)
            found.append((model, expression, extent))
    return found


def main():
    depths = tile_depths()
    extents = k_extents()
    if not depths or not extents:
        print("found no formats or no GEMM call sites; the parser is broken")
        return 1
    worst = max(depths.values())
    failures = []
    unresolved = []
    for model, expression, extent in extents:
        if extent is None:
            unresolved.append((model, expression))
            continue
        for name, depth in sorted(depths.items()):
            if extent % depth:
                failures.append((model, expression, extent, name, depth))
    print(f"formats {len(depths)}  strictest tile depth {worst}")
    print(f"GEMM K extents checked {len(extents)}")
    if unresolved:
        # A model may factor its projections into a helper, which hides the
        # widths from this parse. That is fine ONLY if it asserts them at
        # compile time instead - kimi_k3 does, and writing those assertions is
        # what found that its 128-wide decay bottleneck is narrower than an INT7
        # tile and would have computed zero tiles.
        still_open = []
        for model, expression in unresolved:
            layer = os.path.join(ROOT, "inference", "llms", model, "layer.cuh")
            asserted = (os.path.exists(layer) and
                        open(layer).read().count("static_assert") >= 4)
            if not asserted:
                still_open.append((model, expression))
        if still_open:
            print(f"\n{len(still_open)} K extent(s) neither resolvable nor asserted:")
            for model, expression in still_open:
                print(f"  {model}: {expression}")
            return 1
        print(f"{len(unresolved)} extent(s) hidden behind a helper, "
              "asserted at compile time instead")
    if failures:
        print(f"\n{len(failures)} K extent(s) are not a whole number of tiles:")
        for model, expression, extent, name, depth in failures:
            print(f"  {model}: {expression} = {extent}, "
                  f"{name} tiles at {depth}, remainder {extent % depth}")
        return 1
    print(f"\nevery K extent is a whole number of tiles for every format")
    return 0


if __name__ == "__main__":
    sys.exit(main())
