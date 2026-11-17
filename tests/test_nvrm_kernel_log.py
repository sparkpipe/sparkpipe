#!/usr/bin/env python3
"""tools/nvrm_kernel_log.sh marks the kernel log by its own last stamp.

The kernel stamps its log lines with a clock that runs behind
/proc/uptime: on a GB10 node an Xid line was stamped 0.23 s before an uptime
mark taken before the faulting process started. A mark taken from
/proc/uptime therefore hid the lines of a fault that happened just after
it. The rule now marks with the last line's own stamp and reports every
later line, so this test drives it with a fake dmesg whose stamps lie
before any uptime the host could have: check must still see the fault,
xids must list each Xid code once, and an unreadable log must fail.
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "tools" / "nvrm_kernel_log.sh"

BEFORE = """[    1.000000] Booting Linux
[   12.500000] NVRM: loading NVIDIA UNIX Open Kernel Module
[   20.250000] libceph: mon0 socket closed
"""
FAULTS = """[   20.250100] NVRM: Xid (PCI:000f:01:00): 13, Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Illegal Instruction Encoding
[   20.250101] NVRM: Xid (PCI:000f:01:00): 13, pid=7, name=fault_reports, Graphics Exception: channel 0x00000002
[   20.900000] NVRM: Xid (PCI:000f:01:00): 31, pid=8, name=fault_reports, channel 0x00000002. MMU Fault: ENGINE GRAPHICS GPC0 faulted @ 0x7f00_00001000
"""
QUIET = """[   20.300000] libceph: mon2 socket closed
"""


def fake_tools(directory, log, readable):
    bin_dir = Path(directory) / "bin"
    bin_dir.mkdir()
    dmesg = bin_dir / "dmesg"
    dmesg.write_text("#!/bin/sh\n" + (f"cat '{log}'\n" if readable else "exit 1\n"))
    sudo = bin_dir / "sudo"
    sudo.write_text("#!/bin/sh\nexit 1\n")
    for tool in (dmesg, sudo):
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def rule(bin_dir, *arguments):
    environment = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin")
    return subprocess.run(["sh", str(RULE), *arguments], capture_output=True, text=True, env=environment)


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "kernel.log"
        log.write_text(BEFORE)
        bin_dir = fake_tools(directory, log, True)
        mark = rule(bin_dir, "mark")
        if mark.returncode != 0 or mark.stdout.strip() != "20.250000":
            failures.append(f"mark must be the last line's stamp, 20.250000: {mark.stdout.strip()} {mark.stderr.strip()}")
        stamp = mark.stdout.strip()
        if rule(bin_dir, "check", stamp).returncode != 0:
            failures.append("check must pass when nothing was logged after the mark")
        log.write_text(BEFORE + QUIET)
        if rule(bin_dir, "check", stamp).returncode != 0:
            failures.append("check must pass when only non-NVRM lines follow the mark")
        log.write_text(BEFORE + FAULTS + QUIET)
        check = rule(bin_dir, "check", stamp)
        if check.returncode == 0 or "Xid (PCI:000f:01:00): 13" not in check.stdout or "): 31," not in check.stdout:
            failures.append(f"check must fail on Xid lines stamped after the mark: {check.stdout.strip()[-300:]}")
        xids = rule(bin_dir, "xids", stamp)
        if xids.returncode != 0 or xids.stdout.split() != ["13", "31"]:
            failures.append(f"xids must list each Xid code after the mark once, in order: {xids.stdout.strip()}")
        later = rule(bin_dir, "xids", "20.250101")
        if later.returncode != 0 or later.stdout.split() != ["31"]:
            failures.append(f"xids must leave out lines at or before the mark: {later.stdout.strip()}")
        log.write_text("")
        empty = rule(bin_dir, "mark")
        if empty.returncode != 0 or empty.stdout.strip() != "0.000000":
            failures.append(f"an empty log marks 0: {empty.stdout.strip()}")
    with tempfile.TemporaryDirectory() as directory:
        bin_dir = fake_tools(directory, Path(directory) / "missing.log", False)
        for arguments in (("mark",), ("check", "0"), ("xids", "0")):
            run = rule(bin_dir, *arguments)
            if run.returncode == 0 or "cannot read the kernel log" not in run.stderr:
                failures.append(f"{' '.join(arguments)} must fail when the log cannot be read: {run.stdout.strip()} {run.stderr.strip()}")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("PASS the kernel-log rule marks by the log's own stamps, reports later faults and lists their Xid codes")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
