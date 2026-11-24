#!/usr/bin/env python3
"""Host check that CI's sm_121a compile gate runs the release build hosts' CUDA toolkit.

The compile gate's audits (the grouped expert GEMV spill audit among them) judge compiled code, and ptxas allocates
registers differently from one toolkit release to the next: CUDA 13.3.1 spilled grouped variants that CUDA 13.0, the
release build hosts' toolkit, keeps in registers. So the gate must check what ships. The CUDA version of the
compile-sm121a job's container image (pinned by digest) must be the one the gate requires (release_toolkit in
tools/cuda13_sm121a_compile_gate.sh), and the gate must refuse any other nvcc release. A toolkit upgrade is then one
change that moves both, reruns the audits and the variant timings, and is reviewed as such.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "cuda13-sm121a-compile.yml"
GATE = ROOT / "tools" / "cuda13_sm121a_compile_gate.sh"


def main():
    failures = []
    workflow = WORKFLOW.read_text()
    job = workflow[workflow.find("  compile-sm121a:"):]
    image = re.search(r"image: nvidia/cuda:(\d+)\.(\d+)\.\d+-devel-[a-z0-9.]+@sha256:[0-9a-f]{64}", job)
    if image is None:
        failures.append("the compile-sm121a job does not run a digest-pinned nvidia/cuda devel image")
    gate = GATE.read_text()
    required = re.search(r'^release_toolkit="(\d+)\.(\d+)"$', gate, re.M)
    if required is None:
        failures.append("the compile gate names no release_toolkit")
    if 'grep -Fq "release ${release_toolkit}," <<<"${nvcc_version}"' not in gate:
        failures.append("the compile gate does not refuse an nvcc release other than release_toolkit")
    if image is not None and required is not None and image.groups() != required.groups():
        failures.append("CI's image is CUDA %s.%s but the gate requires the release toolkit %s.%s" % (image.group(1), image.group(2), required.group(1),
                                                                                                    required.group(2)))
    for failure in failures:
        print("FAIL " + failure)
    if not failures:
        print("PASS CI's sm_121a compile gate runs CUDA %s.%s, the release toolkit" % required.groups())
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
