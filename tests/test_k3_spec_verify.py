"""The K3 speculation policy core stays greedy-prefix exact and the adaptive
draft depth follows the acceptance rule the gb10-vllm reference pins (target =
floor(mean accepted + 1.5) clamped to [floor, ceiling], step down direct, step
up one observation at a time, defaults window 8 floor 4).

Scenarios (driven through tests/host_cuda/k3_spec_verify_host.c):

  resolve: greedy prefix acceptance over the verify row stream, the bonus is
           the target prediction of the last accepted row, oversized and null
           inputs fail closed
  adaptive: the reference depth rule - ceiling hold, sustained rejection
            parking at the floor, one-depth-per-observation recovery, and the
            floor never breached
  engine_cycle: a full draft-verify loop on the K3 engine - the plan carries
            depth+1 rows, CommitVerify interleaves drafts and bonuses in
            order, and the engine's decided depth adapts from observed
            acceptance
  dcp: the decode-context-parallel range split tiles the context exactly
       (remainder to the leading ranks, empty tails past the context) and the
       two-range online-softmax merge reproduces the direct value
"""
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "k3_spec_verify_host"
        build = subprocess.run(
            shlex.split(os.environ.get("CC", "cc")) +
            ["-std=c11", "-O1", "-I", str(ROOT),
             str(ROOT / "tests" / "host_cuda" / "k3_spec_verify_host.c"),
             "-o", str(binary), "-lm"],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr[:400])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0 or "k3_spec_verify_host PASS" not in run.stdout:
        print("FAIL spec verify host:")
        print(run.stdout[-800:])
        print(run.stderr[-400:])
        return 1
    print("PASS k3 spec verify: greedy prefix resolve, adaptive depth rule, "
          "engine draft-verify cycle, dcp range split + merge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
