#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "hardware" / "spark_pmtu_characterize.c"
QUALIFIER = ROOT / "tools" / "spark_hardware_qualify.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return result


def unused_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as descriptor:
        descriptor.bind(("127.0.0.1", 0))
        return int(descriptor.getsockname()[1])


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        executable = directory / "spark_pmtu_characterize"
        receipt = directory / "receipt.json"
        compile_result = run([
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Itools/hardware",
            str(SOURCE),
            "-o",
            str(executable),
        ])
        require(compile_result.returncode == 0, compile_result.stderr)

        port = unused_udp_port()
        server = subprocess.Popen(
            [
                str(executable),
                "--server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--idle-timeout-seconds",
                "10",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            require(server.stdout is not None, "server stdout unavailable")
            ready_line = server.stdout.readline().strip()
            require(ready_line == f"READY port={port}", f"unexpected ready line: {ready_line}")
            client_result = run([
                str(executable),
                "--question",
                "NET-PMTU-001",
                "--peer-address",
                "127.0.0.1",
                "--port",
                str(port),
                "--minimum-payload-bytes",
                "512",
                "--maximum-payload-bytes",
                "4096",
                "--iterations",
                "3",
                "--source-package-sha256",
                "ab" * 32,
                "--run-id",
                "pmtu-loopback",
                "--topology",
                "ring-test",
                "--node",
                "spark0",
                "--peer",
                "spark1",
                "--candidate",
                "udp_df_binary_search",
                "--output",
                str(receipt),
            ])
            require(client_result.returncode == 0, client_result.stderr)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5.0)

        document = json.loads(receipt.read_text(encoding="utf-8"))
        answer = document["answers"][0]
        observation = answer["observations"][0]
        metrics = observation["metrics"]
        require(answer["status"] == "measured", "PMTU receipt is not measured")
        require(observation["parameters"]["candidate"] == "udp_df_binary_search",
                "PMTU candidate drift")
        require(512 <= metrics["maximum_payload_bytes"] <= 4096,
                "PMTU payload is outside requested bounds")
        require(metrics["sample_count"] == 3, "PMTU sample count mismatch")
        require(metrics["integrity_pass"] is True, "PMTU integrity failed")
        validate_result = run([sys.executable, str(QUALIFIER), "validate-receipt", str(receipt)])
        require(validate_result.returncode == 0, validate_result.stderr)

        invalid_result = run([
            str(executable),
            "--question",
            "NET-PMTU-001",
            "--peer-address",
            "127.0.0.1",
            "--port",
            str(port),
            "--minimum-payload-bytes",
            "512",
            "--maximum-payload-bytes",
            "65484",
            "--iterations",
            "1",
            "--source-package-sha256",
            "ab" * 32,
            "--run-id",
            "pmtu-boundary",
            "--topology",
            "ring-test",
            "--node",
            "spark0",
            "--peer",
            "spark1",
            "--candidate",
            "udp_df_binary_search",
        ])
        require(invalid_result.returncode == 2,
                "PMTU parser accepted an application payload beyond the IPv4 UDP limit")

    print("PASS PMTU integrity, loopback discovery, receipt, and payload boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
