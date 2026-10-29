"""Run the weight-stream library loopback test (four ranks on one RoCE device: pair-relay broadcasts and expert rotation steps, every byte checked) on a host with an active RoCE v2 port."""
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CC = os.environ.get("CC", "cc")
GID_INDEX = 3


def active_device():
    base = pathlib.Path("/sys/class/infiniband")
    if not base.is_dir():
        return None
    for device in sorted(base.iterdir()):
        port = device / "ports" / "1"
        try:
            state = (port / "state").read_text()
            gid = (port / "gids" / str(GID_INDEX)).read_text().strip()
        except OSError:
            continue
        if "ACTIVE" in state and gid.replace(":", "").strip("0"):
            return device.name
    return None


device = active_device()
if device is None:
    print("SKIP test_weight_stream_loopback (needs an active RoCE v2 port with a GID at index 3)")
    raise SystemExit(0)
if shutil.which(CC) is None:
    raise SystemExit("FAIL test_weight_stream_loopback: a RoCE port is present but no C compiler")
with tempfile.TemporaryDirectory(prefix="weight-stream-loopback-") as directory:
    binary = str(pathlib.Path(directory) / "probe")
    subprocess.run([CC, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-pthread", "-D_GNU_SOURCE", "-Iinclude",
                    "tests/test_weight_stream_loopback.c", "ring/transport/weight_stream.c", "-libverbs", "-o", binary], cwd=ROOT, check=True)
    subprocess.run([binary, device, str(GID_INDEX)], check=True, timeout=300)
