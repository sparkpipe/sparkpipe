#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TRIGGERS = (
    ("smmu-cmd-sync-timeout", re.compile(r"arm-smmu-v3 \S+: CMD_SYNC timeout")),
    ("smmu-failure", re.compile(r"arm-smmu-v3 \S+: (?:CMDQ error|unexpected global error reported|device has entered Service Failure Mode)")),
    ("mlx5-pci-error", re.compile(r"mlx5_core \S+: (?:mlx5_pci_err_detected|PCIe Bus Error: severity=Uncorrectable)")),
    ("pcie-fatal", re.compile(r"\[Hardware Error\]:.*event severity: fatal")),
)
AER_UNCORRECTABLE = re.compile(r"AER: (?:Multiple )?Uncorrectable \((Non-Fatal|Fatal)\) error message received from (\S+)")
NIC_VENDOR = "0x15b3"
KERNEL_FACILITY = 0
UNCLEAN_BOOT_LIMIT = 3
LOOP_WINDOW_SECONDS = 6 * 3600
FOLLOW_UP_SECONDS = 0.03
STATE_PATH = Path("/var/lib/spark-dma-guard/boots.json")
BINARY_PATH = Path("/usr/local/sbin/spark-dma-guard")
SERVICE_PATH = Path("/etc/systemd/system/spark-dma-guard.service")
TIMER_PATH = Path("/etc/systemd/system/spark-dma-guard.timer")


def parse_record(record: bytes) -> tuple[int, int, str] | None:
    head, separator, rest = record.partition(b";")
    if not separator:
        return None
    fields = head.split(b",")
    try:
        priority = int(fields[0])
        sequence = int(fields[1])
    except (IndexError, ValueError):
        return None
    text = rest.split(b"\n", 1)[0].decode("utf-8", "replace")
    return priority >> 3, sequence, text


def pci_vendor(bdf: str) -> str | None:
    try:
        return Path("/sys/bus/pci/devices", bdf, "vendor").read_text().strip()
    except OSError:
        return None


def match_trigger(facility: int, text: str, accept_user: bool, vendor_of=pci_vendor) -> str | None:
    if facility != KERNEL_FACILITY and not accept_user:
        return None
    for name, pattern in TRIGGERS:
        if pattern.search(text):
            return name
    aer = AER_UNCORRECTABLE.search(text)
    if aer is None:
        return None
    if aer.group(1) == "Fatal":
        return "pcie-aer-fatal"
    return "pcie-aer-nic-uncorrectable" if vendor_of(aer.group(2)) == NIC_VENDOR else None


def current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def load_boots(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    boots = data.get("boots") if isinstance(data, dict) else None
    return [b for b in boots if isinstance(b, dict) and "id" in b] if isinstance(boots, list) else []


def save_boots(path: Path, boots: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".boots.")
    with os.fdopen(handle, "w") as stream:
        json.dump({"boots": boots[-20:]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def register_boot(path: Path, boot_id: str, limit: int, now: int | None = None) -> bool:
    now = int(time.time()) if now is None else now
    boots = [b for b in load_boots(path) if b["id"] != boot_id]
    recent = boots[-limit:]
    looping = len(recent) == limit and all(not b.get("clean") for b in recent) and now - int(recent[0].get("started", 0)) < LOOP_WINDOW_SECONDS
    armed = not looping
    boots.append({"id": boot_id, "clean": False, "armed": armed, "started": now})
    save_boots(path, boots)
    return armed


def mark_clean(path: Path, boot_id: str) -> None:
    boots = load_boots(path)
    for boot in boots:
        if boot["id"] == boot_id:
            boot["clean"] = True
    save_boots(path, boots)


def notify(target: tuple[str, int] | None, message: str) -> None:
    if target is None:
        return
    payload = message.encode("utf-8", "replace")[:1400]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.settimeout(0.2)
            for _ in range(3):
                udp.sendto(payload, target)
    except OSError:
        pass


def parse_target(text: str | None) -> tuple[str, int] | None:
    if not text:
        return None
    host, _, port = text.rpartition(":")
    return socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_DGRAM)[0][4]


def follow_up(descriptor: int, seconds: float) -> list[str]:
    lines = []
    deadline = time.monotonic() + seconds
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    fcntl.fcntl(descriptor, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        while time.monotonic() < deadline and len(lines) < 24:
            try:
                record = os.read(descriptor, 8192)
            except BlockingIOError:
                time.sleep(0.002)
                continue
            except OSError as error:
                if error.errno == errno.EPIPE:
                    continue
                break
            parsed = parse_record(record)
            if parsed is not None and parsed[0] == KERNEL_FACILITY:
                lines.append(parsed[2])
    finally:
        fcntl.fcntl(descriptor, fcntl.F_SETFL, flags)
    return lines


def lock_memory() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    return 0 if libc.mlockall(3) == 0 else ctypes.get_errno()


def run(arguments: argparse.Namespace) -> int:
    target = parse_target(arguments.notify)
    host = socket.gethostname()
    boot_id = current_boot_id()
    state = Path(arguments.state)
    sysrq = None if arguments.dry_run else os.open("/proc/sysrq-trigger", os.O_WRONLY)
    locked = lock_memory()
    if locked != 0:
        notify(target, f"SPARK-DMA-GUARD host={host} boot={boot_id} event=mlockall-failed errno={locked}")
    try:
        armed = register_boot(state, boot_id, UNCLEAN_BOOT_LIMIT)
    except OSError as error:
        armed = False
        notify(target, f"SPARK-DMA-GUARD host={host} boot={boot_id} event=state-unwritable error={error}")
    mode = "dry-run" if arguments.dry_run else ("armed" if armed else "watch-only")
    banner = f"SPARK-DMA-GUARD host={host} boot={boot_id} event=start mode={mode}"
    if not armed:
        banner += f" reason=last-{UNCLEAN_BOOT_LIMIT}-boots-unclean-or-state-unwritable"
    print(banner, flush=True)
    notify(target, banner)

    def stop(signum, frame):
        try:
            mark_clean(state, boot_id)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    descriptor = os.open(arguments.kmsg, os.O_RDONLY)
    fired = False
    while True:
        try:
            record = os.read(descriptor, 8192)
        except OSError as error:
            if error.errno == errno.EPIPE:
                continue
            raise
        if not record:
            time.sleep(0.05)
            continue
        parsed = parse_record(record)
        if parsed is None:
            continue
        facility, _, text = parsed
        reason = match_trigger(facility, text, arguments.accept_user)
        if reason is None or fired:
            continue
        message = f"SPARK-DMA-GUARD host={host} boot={boot_id} event=trigger reason={reason} mode={mode} line={text[:400]}"
        notify(target, message)
        for detail in follow_up(descriptor, FOLLOW_UP_SECONDS):
            notify(target, f"SPARK-DMA-GUARD host={host} boot={boot_id} event=detail line={detail[:400]}")
        if mode == "armed":
            time.sleep(0.1)
            os.write(sysrq, b"b")
        print(message, flush=True)
        fired = mode != "dry-run"


def scan(arguments: argparse.Namespace) -> int:
    descriptor = os.open(arguments.kmsg, os.O_RDONLY | os.O_NONBLOCK)
    matches = 0
    while True:
        try:
            record = os.read(descriptor, 8192)
        except BlockingIOError:
            break
        except OSError as error:
            if error.errno == errno.EPIPE:
                continue
            raise
        parsed = parse_record(record)
        if parsed is None:
            continue
        facility, _, text = parsed
        reason = match_trigger(facility, text, False)
        if reason is not None:
            matches += 1
            print(f"MATCH reason={reason} line={text[:400]}")
    print(f"scan host={socket.gethostname()} matches={matches}")
    return 1 if matches else 0


def listen(arguments: argparse.Namespace) -> int:
    log = Path(arguments.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind(("0.0.0.0", arguments.port))
        while True:
            payload, (address, _) = udp.recvfrom(2048)
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            with log.open("a") as stream:
                stream.write(f"{stamp} {address} {payload.decode('utf-8', 'replace').strip()}\n")


def unit_text(notify_target: str) -> tuple[str, str]:
    service = (
        "[Unit]\n"
        "Description=Reboot at once when device DMA can no longer be trusted\n"
        "After=multi-user.target ssh.service network-online.target\n"
        "StartLimitIntervalSec=600\n"
        "StartLimitBurst=5\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={BINARY_PATH} run --notify {notify_target}\n"
        "Restart=on-failure\n"
        "RestartSec=10\n"
    )
    timer = (
        "[Unit]\n"
        "Description=Arm the DMA guard after boot\n\n"
        "[Timer]\n"
        "OnBootSec=60\n"
        "Unit=spark-dma-guard.service\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    return service, timer


def install(arguments: argparse.Namespace) -> int:
    if os.geteuid() != 0:
        print("install needs root", file=sys.stderr)
        return 2
    host, port = parse_target(arguments.notify)
    service, timer = unit_text(f"{host}:{port}")
    if Path(__file__).resolve() != BINARY_PATH.resolve():
        shutil.copyfile(Path(__file__).resolve(), BINARY_PATH)
    BINARY_PATH.chmod(0o755)
    SERVICE_PATH.write_text(service)
    TIMER_PATH.write_text(timer)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "spark-dma-guard.timer"], check=True)
    subprocess.run(["systemctl", "restart", "spark-dma-guard.service"], check=True)
    time.sleep(3)
    active = subprocess.run(["systemctl", "is-active", "spark-dma-guard.service"], capture_output=True, text=True).stdout.strip()
    restarts = subprocess.run(["systemctl", "show", "-p", "NRestarts", "--value", "spark-dma-guard.service"], capture_output=True, text=True).stdout.strip()
    print(f"installed {BINARY_PATH} service={active} restarts={restarts}")
    return 0 if active == "active" and restarts == "0" else 1


def self_test(arguments: argparse.Namespace) -> int:
    failures = []
    vendors = {"0000:01:00.1": "0x15b3", "0002:01:00.0": "0x15b3", "0004:01:00.0": "0x144d"}
    vendor_of = vendors.get
    crash_lines = (
        "pcieport 0000:00:00.0: AER: Uncorrectable (Non-Fatal) error message received from 0000:01:00.1",
        "pcieport 0004:00:00.0: AER: Uncorrectable (Fatal) error message received from 0004:01:00.0",
        "pcieport 0002:00:00.0: AER: Multiple Uncorrectable (Non-Fatal) error message received from 0002:01:00.0",
        "mlx5_core 0000:01:00.1: PCIe Bus Error: severity=Uncorrectable (Non-Fatal), type=Transaction Layer, (Requester ID)",
        "mlx5_core 0000:01:00.0: mlx5_pci_err_detected Device state = 1 health sensors: 0 pci_status: 1. Enter, pci channel state = 1",
        "arm-smmu-v3 arm-smmu-v3.0.auto: CMD_SYNC timeout at 0x0000b1b7 [hwprod 0x0000b1c0, hwcons 0x0000b1b7]",
        "arm-smmu-v3 arm-smmu-v3.0.auto: CMDQ error (cons 0x0100b1b7): ATC invalidate timeout",
        "arm-smmu-v3 arm-smmu-v3.0.auto: unexpected global error reported (0x00000001), this could be serious",
        "{1}[Hardware Error]: event severity: fatal",
    )
    quiet_lines = (
        "pcieport 0000:00:00.0: AER: enabled with IRQ 329",
        "pcieport 0000:00:00.0: AER: Corrected error message received from 0000:01:00.1",
        "pcieport 0004:00:00.0: AER: Uncorrectable (Non-Fatal) error message received from 0004:01:00.0",
        "arm-smmu-v3 arm-smmu-v3.0.auto: allocated 65536 entries for cmdq",
        "arm-smmu-v3 arm-smmu-v3.2.auto: event 0x10 received:",
        "NVRM: Xid (PCI:000f:01:00): 31, pid=1, name=sparkpipe_model, channel 0x2, intr 0. MMU Fault: ENGINE GRAPHICS",
        "mlx5_core 0000:01:00.1: Port module event: module 1, Cable plugged",
        "{1}[Hardware Error]: event severity: corrected",
    )
    for line in crash_lines:
        if match_trigger(KERNEL_FACILITY, line, False, vendor_of) is None:
            failures.append(f"missed: {line}")
        if match_trigger(1, line, False, vendor_of) is not None:
            failures.append(f"user-space copy triggered: {line}")
    for line in quiet_lines:
        if match_trigger(KERNEL_FACILITY, line, False, vendor_of) is not None:
            failures.append(f"false trigger: {line}")
    if parse_record(b"3,1234,5678,-;arm-smmu-v3 arm-smmu-v3.0.auto: CMD_SYNC timeout at 0x1\n SUBSYSTEM=platform\n") != (0, 1234, "arm-smmu-v3 arm-smmu-v3.0.auto: CMD_SYNC timeout at 0x1"):
        failures.append("kmsg record parse")
    if parse_record(b"14,7,1,-;user line\n")[0] != 1:
        failures.append("facility parse")
    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "boots.json"
        armed = [register_boot(state, f"boot{n}", UNCLEAN_BOOT_LIMIT, 1000 + 60 * n) for n in range(5)]
        if armed != [True, True, True, False, False]:
            failures.append(f"unclean boot arming {armed}")
        if "boot4" not in [b["id"] for b in load_boots(state)]:
            failures.append("boot record not persisted")
        mark_clean(state, "boot4")
        if not register_boot(state, "boot5", UNCLEAN_BOOT_LIMIT, 1400):
            failures.append("a clean boot did not re-arm")
        if register_boot(state, "boot5", UNCLEAN_BOOT_LIMIT, 1400) is not True:
            failures.append("a restart within one boot changed arming")
        spread = Path(directory) / "spread.json"
        spaced = [register_boot(spread, f"day{n}", UNCLEAN_BOOT_LIMIT, n * 86400) for n in range(5)]
        if spaced != [True] * 5:
            failures.append(f"unclean boots days apart disarmed the guard {spaced}")
    for failure in failures:
        print(f"FAIL {failure}")
    print("PASS spark DMA guard self-test" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--notify")
    run_parser.add_argument("--state", default=str(STATE_PATH))
    run_parser.add_argument("--kmsg", default="/dev/kmsg")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--accept-user", action="store_true")
    listen_parser = commands.add_parser("listen")
    listen_parser.add_argument("--port", type=int, default=5514)
    listen_parser.add_argument("--log", required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--notify", required=True)
    scan_parser = commands.add_parser("scan")
    scan_parser.add_argument("--kmsg", default="/dev/kmsg")
    commands.add_parser("self-test")
    arguments = parser.parse_args()
    handlers = {"run": run, "listen": listen, "install": install, "scan": scan, "self-test": self_test}
    return handlers[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
