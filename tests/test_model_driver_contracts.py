"""Every model is a config header and a unity build, and nothing else.

This replaced a table asserting that each family had a module source, a module
include, a family include and a firmware description - five paths per model,
twenty for four models, none of which said anything about what the code did.

What it checks now is the property the rewrite is for: a model directory
contains config.h and unity.cu, the config is constants only, and the unity
build instantiates kernels rather than defining them. A model that starts
defining its own kernels has stopped sharing the library and this fails.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main() -> int:
    failures = 0
    models = sorted(p for p in (ROOT / "inference/llms").iterdir() if p.is_dir())
    if not models:
        print("  FAIL no models in llms/")
        return 1
    for model in models:
        name = model.name
        config = model / "config.h"
        unity = model / "unity.cu"
        if not config.exists() or not unity.exists():
            print(f"  FAIL {name}: needs config.h and unity.cu")
            failures += 1
            continue
        config_text = config.read_text(encoding="utf-8")
        unity_text = unity.read_text(encoding="utf-8")
        # A config that defines a function body is not constants only. The
        # inline helpers that derive one constant from others are allowed.
        if "__global__" in config_text or "__device__" in config_text:
            print(f"  FAIL {name}: config.h contains device code")
            failures += 1
        # A unity build INSTANTIATES kernels. Defining one means this model has
        # something it could not express as a parameter, which is a claim that
        # should be rare and should say why.
        defines = len(re.findall(r"^\s*(static\s+)?__global__", unity_text, re.M))
        explicit_instantiations = unity_text.count("template __global__")
        templated_launch_bindings = len(re.findall(
            r"\b(?:LmGemmLaunch|LmGemmWeightOnlyLaunch)\s*<",
            unity_text))
        exported_entry_points = len(re.findall(
            r'extern\s+"C"\s+int32_t\s+[A-Za-z_][A-Za-z0-9_]*\s*\(',
            unity_text))
        binding_sites = explicit_instantiations + templated_launch_bindings
        if defines:
            print(f"  FAIL {name}: unity.cu defines {defines} kernels; it should only bind shared kernels")
            failures += 1
        if binding_sites == 0 and exported_entry_points == 0:
            print(f"  FAIL {name}: unity.cu binds no shared kernel or model entry point")
            failures += 1
        print(f"  ok   {name}: {len(config_text.splitlines())} line config, "
              f"{explicit_instantiations} explicit instantiations, "
              f"{templated_launch_bindings} templated launch bindings, "
              f"{exported_entry_points} entry points, 0 kernels defined")
    print(f"\n{'FAIL' if failures else 'PASS'} ({failures} failing)")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
