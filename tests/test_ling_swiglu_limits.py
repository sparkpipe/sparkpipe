#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACTS = [REPOSITORY / "model_contracts/ling_authoritative.json", REPOSITORY / "model_contracts/lingfin_authoritative.json"]
PROGRAM = r"""
#include <stdio.h>
#include "sparkpipe/spark_ling_model.h"
#include "sparkpipe/spark_ling_swiglu_limits.h"
int main(void)
{
	unsigned layer;
	for (layer = 0u; layer < SPARK_LING_MODEL_LAYER_COUNT; layer++)
		printf("%u %g %g\n", layer, (double)SPARK_LING_MODEL_MOE_ROUTED_SWIGLU_LIMIT(layer), (double)SPARK_LING_MODEL_MOE_SHARED_SWIGLU_LIMIT(layer));
	return 0;
}
"""


def driver_limits():
    with tempfile.TemporaryDirectory() as scratch:
        source = os.path.join(scratch, "limits.c")
        binary = os.path.join(scratch, "limits")
        Path(source).write_text(PROGRAM)
        subprocess.run(["cc", "-std=c11", "-I", str(REPOSITORY / "model-families/ling/include"), "-I", str(REPOSITORY / "include"), "-I", str(REPOSITORY / "model-families/common/include"), source, "-o", binary], check=True)
        output = subprocess.run([binary], check=True, capture_output=True, text=True).stdout
    routed, shared = [], []
    for line in output.splitlines():
        _, routed_limit, shared_limit = line.split()
        routed.append(float(routed_limit))
        shared.append(float(shared_limit))
    return routed, shared


def main():
    failures = []
    routed, shared = driver_limits()
    for path in CONTRACTS:
        document = json.loads(path.read_text())
        moe = document["moe"]
        if [float(value) for value in moe["expert_swiglu_limit_list"]] != routed:
            failures.append(f"{path.name}: routed limits differ from SPARK_LING_MODEL_MOE_ROUTED_SWIGLU_LIMIT")
        if [float(value) for value in moe["share_expert_swiglu_limit_list"]] != shared:
            failures.append(f"{path.name}: shared limits differ from SPARK_LING_MODEL_MOE_SHARED_SWIGLU_LIMIT")
    limited = [layer for layer in range(len(routed)) if routed[layer] or shared[layer]]
    if limited != list(range(34, 42)):
        failures.append(f"limited layers {limited} are not 34..41")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print(f"PASS ling swiglu limits: driver macros equal both contracts on {len(routed)} layers, limited layers 34..41")
    return 0


if __name__ == "__main__":
    sys.exit(main())
