#!/usr/bin/env python3
"""Check that device API queries never break a stream capture on the CUDA backend.

A stage runner records its decode step into a CUDA graph on the stream it handed the device API as
an adopted queue, with a thread-local capture. Another thread that queries that stream while the
capture runs (a KV copy worker checking the stamp a fill left on a shared buffer, for example)
invalidates the capture, and the runner's next launch fails. The negative control shows a raw
stream query from another thread doing exactly that. With the queue marked for the length of the
capture, queries, timed waits and fills on a buffer the adopted queue stamped, from another thread,
must report the adopted queue busy without touching its stream, the capture must end whole and its
graph run, and the queue must answer from its stream again once the mark clears. Queues the device
API created itself refuse the mark.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_device_capture_cuda"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_device_capture_cuda (no CUDA toolkit on this host)")
        return 0
    if shutil.which("nvidia-smi") is None or subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0:
        print("SKIP test_device_capture_cuda (no CUDA device on this host)")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS device capture (cuda backend)" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
