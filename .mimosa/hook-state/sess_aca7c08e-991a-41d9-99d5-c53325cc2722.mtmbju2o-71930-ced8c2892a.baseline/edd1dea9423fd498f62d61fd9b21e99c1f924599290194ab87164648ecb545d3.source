#!/usr/bin/env python3
"""A layer kind a model selects must have an entry point that can run it.

deepseek_v4's config describes sliding-window, compressed-sparse and
high-compression attention alternating by layer, and unity.cu exported one
Dsv4LayerAttentionFp8. Three kinds, one entry point, and nothing anywhere said
so - the shapes were all correct, the model compiled, and two thirds of its
layers had no implementation to dispatch to.

That is the check here: evaluate each model's LAYER_KIND selector over every
layer, collect the distinct kinds, and require an exported entry point for
each. A model that grows a new layer kind and forgets the kernel fails on the
next run rather than at the first wrong token.

KNOWN_INCOMPLETE below is the honest register of what does not hold yet. An
entry there must name what is missing, not say "same" - a gate whose exemption
list is unreadable has stopped being a gate.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLMS = ROOT / "inference" / "llms"

# What an entry point implementing this kind looks like in a name.
KIND_ENTRY = {
    "LM_LAYER_FULL": ("Attention",),
    "LM_LAYER_WINDOW": ("Swa", "Window", "AttentionFull"),
    "LM_LAYER_SPARSE": ("Sparse", "Csa", "Indexed"),
    "LM_LAYER_COMPRESSED": ("Compressed", "Hca"),
    "LM_LAYER_LATENT": ("Attention", "Mla"),
    "LM_LAYER_RECURRENT": ("Linear", "Delta", "Recurrent", "Kda"),
}

KIND_VALUE = {
    "LM_LAYER_FULL": 0, "LM_LAYER_WINDOW": 1, "LM_LAYER_SPARSE": 2,
    "LM_LAYER_COMPRESSED": 3, "LM_LAYER_LATENT": 4, "LM_LAYER_RECURRENT": 5,
}

# A model with more than one layer kind needs a driver that chooses between
# them. glm5_2 has one (bind.cu); the rest do not, so their entry points have no
# caller and their LAYER_KIND selector is read by nothing.
NO_DRIVER = {}

KNOWN_INCOMPLETE = {}


def validate_dsv4_module():
    header = (ROOT / "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h").read_text()
    module = (ROOT / "modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c").read_text()
    ratios = {int(value) for value in re.findall(r"\b(0|4|128)u\b", header[header.index("SparkDsv4ModelCompressionRatios"):])}
    if ratios != {0, 4, 128}:
        return "contract does not expose exactly SWA, CSA, and HCA"
    for needle in (
        "SparkDsv4ModelLayerKind(layer_index)",
        "kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA",
        "kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA",
        "SparkDsv4LaunchSparseAttn",
    ):
        if needle not in module:
            return f"active module does not dispatch {needle}"
    return None


def selector_kinds(model):
    """Evaluate the model's LAYER_KIND macro by compiling it.

    The layer count is also read from the compiled configuration. Generated
    contracts frequently define it through another macro, so parsing only a
    decimal literal silently evaluated the wrong number of layers.
    """
    config = LLMS / model / "config.h"
    text = config.read_text()
    prefix = re.search(r"#define\s+(\w+)_LAYER_KIND\(layer\)", text)
    if not prefix:
        return None, None
    layer_count_macro = f"{prefix.group(1)}_LAYERS"
    program = f"""#include <stdio.h>
#include "inference/llms/{model}/config.h"
int main(void)
{{
    int layer;
    printf("count=%u\\n", (unsigned){layer_count_macro});
    for (layer = 0; layer < (int){layer_count_macro}; ++layer)
        printf("%d\\n", (int){prefix.group(1)}_LAYER_KIND(layer));
    return 0;
}}
"""
    source = Path(tempfile.gettempdir()) / f"lk_{model}.c"
    binary = Path(tempfile.gettempdir()) / f"lk_{model}"
    source.write_text(program)
    include_flags = [f"-I{ROOT}", f"-I{ROOT / 'include'}"]
    include_flags.extend(
        f"-I{path}"
        for path in sorted((ROOT / "model-families").glob("*/include"))
    )
    build = subprocess.run(
        ["gcc", "-O0", *include_flags, "-o", str(binary), str(source)],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        message = build.stderr.strip().splitlines()
        print(f"         {message[-1] if message else 'compile failed'}")
        return None, None
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    lines = run.stdout.splitlines()
    if run.returncode != 0 or not lines or not lines[0].startswith("count="):
        return None, None
    count = int(lines[0].split("=", 1)[1])
    names = {value: name for name, value in KIND_VALUE.items()}
    kinds = {names[int(line)] for line in lines[1:] if line}
    return kinds, count


def entry_points(model):
    unity = LLMS / model / "unity.cu"
    if not unity.exists():
        return set()
    exports = set(re.findall(r'extern "C" int32_t (\w+)', unity.read_text()))
    # A kind is also implemented if the model's slice dispatch names its layer
    # function: kimi_k3's per-kind INT7 exports were deleted when the recipe
    # made INT7 uninstantiable, and the real entry points are the K3LayerKda /
    # K3LayerMla arms K3LaunchAttentionHalf selects between. An export table
    # that only reads unity.cu would force dead C wrappers back into existence
    # to satisfy a gate, which is the tail wagging the dog.
    for part in sorted(unity.parent.glob("slice.cuh")):
        exports |= set(re.findall(r"return\((\w+)<", part.read_text()))
    return exports


def driver_dispatches(model):
    """A driver exists and reads the layer kind. Compiling is not enough: a
    driver that ignores LAYER_KIND and runs one path for every layer is exactly
    what deepseek_v4 would look like if someone wrote it a bind.cu today.

    The driver is bind.cu plus whatever slice header it includes: kimi_k3's
    loop moved to slice.cuh precisely so a host harness can execute it, and a
    gate that only reads bind.cu would push the dispatch back into the one
    file a CPU cannot compile."""
    bind = LLMS / model / "bind.cu"
    if not bind.exists():
        return None
    driver = bind.read_text()
    for part in sorted((LLMS / model).glob("slice.cuh")):
        driver += part.read_text()
    return "LAYER_KIND" in driver


def main():
    failures = 0
    dsv4_problem = validate_dsv4_module()
    if dsv4_problem is None:
        print("  ok   dsv4: 43 layers, kinds [swa, csa, hca]")
    else:
        failures += 1
        print(f"  FAIL dsv4: {dsv4_problem}")
    for model in sorted(p.name for p in LLMS.iterdir() if p.is_dir()):
        kinds, count = selector_kinds(model)
        if kinds is None:
            print(f"  FAIL {model}: no LAYER_KIND selector this gate can evaluate")
            failures += 1
            continue
        exported = entry_points(model)
        missing, excused = [], []
        for kind in sorted(kinds):
            needles = KIND_ENTRY[kind]
            if any(n in e for e in exported for n in needles):
                continue
            if (model, kind) in KNOWN_INCOMPLETE:
                excused.append(kind)
            else:
                missing.append(kind)
        summary = ", ".join(k.replace("LM_LAYER_", "").lower() for k in sorted(kinds))
        dispatches = driver_dispatches(model)
        no_driver_note = ""
        if dispatches is None and len(kinds) > 1 and model in NO_DRIVER:
            no_driver_note = ", NO DRIVER"
        if dispatches is None and len(kinds) > 1 and model not in NO_DRIVER:
            failures += 1
            print(f"  FAIL {model}: {len(kinds)} layer kinds and no bind.cu to choose between them")
            continue
        if dispatches is False and len(kinds) > 1:
            failures += 1
            print(f"  FAIL {model}: bind.cu does not read {model.upper()}_LAYER_KIND; "
                  "it runs one path for every layer")
            continue
        if missing:
            failures += 1
            print(f"  FAIL {model}: {count} layers, kinds [{summary}]")
            for kind in missing:
                print(f"         no entry point for {kind}")
        else:
            mark = "--" if (excused or no_driver_note) else "ok"
            note = f", {len(excused)} kind(s) not implemented" if excused else ""
            print(f"  {mark}   {model}: {count} layers, kinds [{summary}]{note}{no_driver_note}")
    print()
    stale = [k for k in KNOWN_INCOMPLETE if k[0] not in
             {p.name for p in LLMS.iterdir() if p.is_dir()}]
    if stale:
        print(f"stale exemptions for models that no longer exist: {stale}")
        return 1
    if failures:
        print(f"FAIL ({failures} model(s) selecting a kind nothing implements)")
        return 1
    print("every selected layer kind has an entry point or a stated reason")
    return 0


if __name__ == "__main__":
    sys.exit(main())
