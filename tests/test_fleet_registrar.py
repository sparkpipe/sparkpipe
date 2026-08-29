#!/usr/bin/env python3
"""Fleet registrar unit tests (docs/FLEET_STARTUP_PROTOCOL.md phases 1+1b).

Loopback multi-instance: the launch table is N copies of 127.0.0.1, one
registrar process per rank, distinct ports via --port-base. Runs the real
binary (build/sparkpipe_registrar) the wave tools deploy.

Covers:
  1. GO path — 16 fake ranks converge, all exit 0, deterministic leader.
  2. Missing-node path — one registrar never starts; the others FAIL LOUD
     naming the missing rank in the diff, exit nonzero.
  3. Partial-view path — a fake peer announces a stable partial view;
     the real registrars time out with PARTIAL-VIEW naming it.
  4. Subset path — --expect-subset over a 16-host table converges on the
     subset alone (GO) while the rest of the table is absent.
  5. /proc-gated cleanslate paths (Linux only):
     a. stale daemon holds GO: a live matching daemon blocks readiness,
     b. TERM clears it: the registrar TERMs it by cwd, stale clears, GO,
     c. TERM-immune: a TERM-ignoring daemon ends as STALE-IMMUNE: rank->pid
        in the FAIL-LOUD diff, exit nonzero, and the registrar never KILLs.

Run: python3 tests/test_fleet_registrar.py
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
BINARY = REPOSITORY / "build" / "sparkpipe_registrar"
HOSTS_16 = ",".join(["127.0.0.1"] * 16)
PORT_BASE = 25480
HAVE_PROC = os.path.isdir("/proc")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status} {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


class Registrar:
    def __init__(self, rank: int, hosts: str, port_base: int, log_path: Path,
                 extra: list[str]) -> None:
        self.rank = rank
        self.log_path = log_path
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(BINARY), "--rank", str(rank), "--hosts", hosts,
             "--port-base", str(port_base), *extra],
            stdout=self.log, stderr=subprocess.STDOUT,
            cwd=str(tempfile.gettempdir()))
        self.returncode: int | None = None

    def wait(self, timeout: float) -> int | None:
        try:
            self.returncode = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.returncode = None
        self.log.close()
        return self.returncode

    def text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def launchfleet(ranks: range, port_base: int, workdir: Path,
                extra: list[str] | None = None,
                members: int = 16) -> list[Registrar]:
    extra = extra or []
    hosts = ",".join(["127.0.0.1"] * members)
    return [Registrar(r, hosts, port_base, workdir / f"reg{r}.log", extra)
            for r in ranks]


def test_go_path(workdir: Path) -> None:
    port = PORT_BASE + 0
    started = time.monotonic()
    fleet = launchfleet(range(16), port, workdir / "go")
    codes = [r.wait(15.0) for r in fleet]
    elapsed = time.monotonic() - started
    check("go: all 16 exit 0", codes == [0] * 16, f"codes={codes}")
    leader_lines = [r.text() for r in fleet if r.rank == 0][0]
    check("go: rank 0 is the deterministic leader with 3/3 levels",
          "GO rank=0 leader=0 levels=3/3" in leader_lines, leader_lines)
    check("go: members relayed GO",
          all("relay_go" in r.text() for r in fleet))
    check("go: converged well under the 10s acceptance bound",
          elapsed < 10.0, f"elapsed={elapsed:.2f}s")


def test_missing_node(workdir: Path) -> None:
    port = PORT_BASE + 100
    # rank 7's registrar never starts.
    fleet = launchfleet([r for r in range(16) if r != 7], port,
                        workdir / "missing", ["--timeout-ms", "2500"])
    codes = [r.wait(20.0) for r in fleet]
    check("missing: every survivor exits nonzero",
          all(code not in (0, None) for code in codes), f"codes={codes}")
    rank0 = fleet[0].text()
    check("missing: diff names the missing rank",
          "MISSING: [7]" in rank0, rank0[-600:])
    check("missing: partial views of survivors are reported",
          "PARTIAL-VIEW:" in rank0 and "0->[" in rank0, rank0[-600:])
    check("missing: FAIL line is loud (REGISTRAR FAIL)",
          "REGISTRAR FAIL rank=0" in rank0 and "registrar exit rank=0 status=1"
          in rank0, rank0[-400:])


def test_partial_view(workdir: Path) -> None:
    port = PORT_BASE + 200
    workdir = workdir / "partial"
    workdir.mkdir(parents=True, exist_ok=True)
    # Fake rank 15: alive, reachable, announces a STABLE partial view {15}.
    fake = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fake.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    fake.bind(("127.0.0.1", port + 15))
    fake.listen(16)
    fake.settimeout(0.2)
    import threading

    stop = threading.Event()

    def fake_peer() -> None:
        while not stop.is_set():
            try:
                conn, _ = fake.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                conn.settimeout(1.0)
                try:
                    conn.recv(4096)
                    # view mask bit 15 = 0x8000: stable partial view {15}.
                    conn.sendall(b"SPGA 15 0000000000008000 0 0\n")
                except OSError:
                    pass

    thread = threading.Thread(target=fake_peer, daemon=True)
    thread.start()
    fleet = launchfleet(range(15), port, workdir,
                        ["--timeout-ms", "2500"])
    codes = [r.wait(20.0) for r in fleet]
    stop.set()
    thread.join(timeout=2)
    fake.close()
    check("partial: every real registrar exits nonzero",
          all(code not in (0, None) for code in codes), f"codes={codes}")
    rank0 = fleet[0].text()
    check("partial: diff names rank 15's partial view",
          "15->[15]" in rank0, rank0[-600:])
    check("partial: nothing is MISSING (rank 15 was heard; its VIEW is the problem)",
          "MISSING: [" not in rank0, rank0[-600:])


def test_subset(workdir: Path) -> None:
    port = PORT_BASE + 300
    # Only ranks 5, 9, 14 of the 16-slot table run registrars; the subset
    # must converge on the subset (leader 5) with no GO from absent ranks.
    subset = "5,9,14"
    fleet = [Registrar(r, HOSTS_16, port, workdir / f"subset/sub{r}.log",
                       ["--expect-subset", subset, "--timeout-ms", "8000"])
             for r in (5, 9, 14)]
    codes = [r.wait(15.0) for r in fleet]
    check("subset: all subset members exit 0", codes == [0] * 3,
          f"codes={codes}")
    leader_text = fleet[0].text()
    check("subset: leader is the lowest subset rank (5)",
          "GO rank=5 leader=5 levels=3/3" in leader_text, leader_text)
    check("subset: expected set is the subset, not the table",
          "expected=[5,9,14]" in leader_text, leader_text.split("\n")[0])


def _spawn_helper(workdir: Path, immune: bool) -> tuple[int, subprocess.Popen, str]:
    script = workdir / ("immune_daemon.py" if immune else "stale_daemon.py")
    ready = workdir / "daemon_ready"
    if ready.exists():
        ready.unlink()
    if immune:
        script.write_text(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "open(%r, 'w').write('ready')\n" % str(ready) +
            "while True:\n"
            "    time.sleep(0.2)\n")
    else:
        script.write_text("import time\ntime.sleep(300)\n")
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(workdir))
    if immune:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.05)
        if not ready.exists():
            raise RuntimeError("immune daemon never installed its handler")
    exe = os.path.realpath(sys.executable).rsplit("/", 1)[-1]
    return proc.pid, proc, exe


def test_cleanslate(workdir: Path) -> None:
    if not HAVE_PROC:
        print("SKIP cleanslate paths (no /proc on this host; run on a node)")
        return
    base = PORT_BASE + 400

    # (a)+(b) stale daemon holds GO, TERM clears it, GO then fires.
    port = base + 10
    stale_dir = workdir / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "fleet").mkdir(parents=True, exist_ok=True)
    pid, proc, exe = _spawn_helper(stale_dir, immune=False)
    fleet = launchfleet(range(3), port, stale_dir / "fleet", [
        "--timeout-ms", "15000", "--term-wait-ms", "10000",
        "--deployment-cwd", str(stale_dir), "--daemon-name", exe],
        members=3)
    try:
        codes = [r.wait(30.0) for r in fleet]
        texts = [r.text() for r in fleet]
        # On loopback every registrar shares one node; whichever scanned
        # first owns the TERM. The fleet log set must show the TERM
        # lifecycle for the right pid, and no GO may precede it.
        termer = next((t for t in texts
                       if f"pid={pid}" in t and "stale_term" in t), None)
        check("cleanslate: stale daemon was found and TERMed by cwd",
              termer is not None, "\n".join(texts)[:900])
        check("cleanslate: stale daemon held GO until cleared",
              termer is not None
              and "GO rank=" not in termer.split("stale_term")[0],
              "\n".join(texts)[:900])
        check("cleanslate: TERM cleared it and GO fired",
              codes == [0] * 3
              and termer is not None
              and f"stale_clear rank=" in termer
              and any("GO rank=" in t for t in texts),
              f"codes={codes}")
        alive = proc.poll() is None
        check("cleanslate: TERMed daemon is gone", not alive)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # (c) TERM-immune daemon: never KILLed, reported in the fail-loud diff.
    port = base + 20
    immune_dir = workdir / "immune"
    immune_dir.mkdir(parents=True, exist_ok=True)
    (immune_dir / "fleet").mkdir(parents=True, exist_ok=True)
    pid, proc, exe = _spawn_helper(immune_dir, immune=True)
    fleet = launchfleet(range(3), port, immune_dir / "fleet", [
        "--timeout-ms", "4000", "--term-wait-ms", "800",
        "--deployment-cwd", str(immune_dir), "--daemon-name", exe],
        members=3)
    try:
        codes = [r.wait(20.0) for r in fleet]
        rank0 = fleet[0].text()
        check("immune: registrar exits nonzero",
              all(code not in (0, None) for code in codes), f"codes={codes}")
        check("immune: fail-loud diff reports STALE-IMMUNE: rank->pid",
              f"STALE-IMMUNE:" in rank0 and f"0->{pid}" in rank0, rank0[-600:])
        check("immune: immune detection is logged",
              f"stale_immune rank=0 pid={pid}" in rank0, rank0)
        still_alive = proc.poll() is None
        check("immune: registrar never KILLed the daemon (still alive)",
              still_alive, "immune daemon died — the registrar KILLed")
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
            proc.wait()


def main() -> int:
    if not BINARY.exists():
        print(f"FAIL {BINARY} missing; run: make build/sparkpipe_registrar")
        return 1
    workdir = Path(tempfile.mkdtemp(prefix="sparkpipe_registrar_test."))
    (workdir / "go").mkdir()
    (workdir / "missing").mkdir()
    (workdir / "partial").mkdir(parents=True)
    (workdir / "subset").mkdir()
    test_go_path(workdir)
    test_missing_node(workdir)
    test_partial_view(workdir)
    test_subset(workdir)
    test_cleanslate(workdir)
    print(f"\n{len(failures)} failure(s)" + (f": {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
