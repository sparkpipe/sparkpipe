#!/usr/bin/env python3
"""Regression test for systemd automount handling in fsck health checks."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "devcycle" / "sparkpipe_fsck_health.sh"


def write_mock(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sparkpipe-fsck-test-") as temporary:
        root = Path(temporary)
        mock_bin = root / "bin"
        out_dir = root / "out"
        data_path = root / "data"
        root_device = root / "root-device"
        data_device = root / "data-device"
        fstab = root / "fstab"
        log = root / "commands.log"
        mock_bin.mkdir()
        data_path.mkdir()
        root_device.touch()
        data_device.touch()
        fstab.write_text(
            f"{data_device} {data_path} ext4 nofail,x-systemd.automount 0 2\n",
            encoding="utf-8",
        )
        write_mock(mock_bin, "tune2fs", "printf 'Filesystem state: clean\\nFilesystem errors: continue\\n'")
        write_mock(mock_bin, "timeout", "shift; exec \"$@\"")
        write_mock(mock_bin, "e2fsck", "exit 0")
        write_mock(mock_bin, "findmnt", "printf 'autofs\\n'")
        write_mock(mock_bin, "systemd-escape", "printf 'test-data\\n'")
        write_mock(
            mock_bin,
            "systemctl",
            "printf 'systemctl %s\\n' \"$*\" >> \"$SPARKPIPE_TEST_LOG\"\n"
            "case \"$1\" in is-active|stop|start) exit 0;; *) exit 1;; esac",
        )
        write_mock(mock_bin, "mountpoint", "exit 1")
        write_mock(mock_bin, "fsck", "printf 'fsck %s\\n' \"$*\" >> \"$SPARKPIPE_TEST_LOG\"; exit 0")
        write_mock(mock_bin, "mount", "printf 'UNEXPECTED mount %s\\n' \"$*\" >> \"$SPARKPIPE_TEST_LOG\"; exit 99")
        write_mock(mock_bin, "umount", "printf 'UNEXPECTED umount %s\\n' \"$*\" >> \"$SPARKPIPE_TEST_LOG\"; exit 99")
        write_mock(mock_bin, "logger", "exit 0")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{mock_bin}:{environment['PATH']}",
                "SPARKPIPE_E2FSCK_TIMEOUT": "5",
                "SPARKPIPE_FSCK_OUT_DIR": str(out_dir),
                "SPARKPIPE_FSTAB_PATH": str(fstab),
                "SPARKPIPE_ROOT_DEV": str(root_device),
                "SPARKPIPE_TEST_LOG": str(log),
            }
        )
        result = subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"health script failed: {result.stderr}")
        commands = log.read_text(encoding="utf-8").splitlines()
        expected = [
            "systemctl is-active --quiet test-data.automount",
            "systemctl stop test-data.automount",
            "systemctl is-active --quiet test-data.mount",
            "systemctl stop test-data.mount",
            f"fsck -f -y {data_device}",
            "systemctl start test-data.automount",
        ]
        if commands != expected:
            raise RuntimeError(f"unexpected command sequence: {commands!r}")
        receipt = json.loads((out_dir / "last.json").read_text(encoding="utf-8"))
        if receipt["data_volumes"][0]["result"] != "ok":
            raise RuntimeError(f"unexpected receipt: {receipt!r}")
    print("sparkpipe_fsck_health_automount: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
