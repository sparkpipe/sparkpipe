#!/usr/bin/env python3
"""Run two overlapping lazy consumers against an owned daemon."""
import argparse
import pathlib
import selectors
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", required=True, type=pathlib.Path)
    parser.add_argument("--probe", required=True, type=pathlib.Path)
    args = parser.parse_args()
    processes = []
    with tempfile.TemporaryDirectory(prefix="weightd-pair-") as directory:
        root = pathlib.Path(directory)
        pack, socket = root / "pack", root / "socket"
        probe, daemon = str(args.probe.resolve()), str(args.daemon.resolve())
        subprocess.run([probe, "prepare", str(pack)], check=True, timeout=10)
        with (root / "daemon.log").open("w+") as log:
            try:
                server = subprocess.Popen([daemon, "--socket", str(socket), "--device-bytes-max", str(8 * 1024 * 1024)], stdout=log, stderr=log)
                processes.append(server)
                deadline = time.monotonic() + 10
                while not socket.is_socket():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("daemon did not become ready")
                    time.sleep(0.02)
                with selectors.DefaultSelector() as selector:
                    clients = []
                    for first in (0, 1):
                        client = subprocess.Popen([probe, str(socket), str(pack), "consumer", str(first)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        processes.append(client)
                        clients.append(client)
                        selector.register(client.stdout, selectors.EVENT_READ, client)
                    deadline = time.monotonic() + 10
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError("consumers did not hold both leases before deadline")
                        for key, _ in selector.select(remaining):
                            line = key.fileobj.readline()
                            if line.strip() != "READY":
                                raise RuntimeError(f"consumer failed before barrier: {line!r}")
                            selector.unregister(key.fileobj)
                    for client in clients:
                        client.stdin.write("G")
                        client.stdin.flush()
                    for client in clients:
                        output, _ = client.communicate(timeout=10)
                        if client.returncode != 0 or "PASS consumer-local lazy reads" not in output:
                            raise RuntimeError(f"consumer failure: {output}")
                server.terminate()
                if server.wait(timeout=10) != 0:
                    raise RuntimeError("daemon shutdown failed")
                print("PASS two simultaneous lazy consumers, overlapping sets {0,1}/{1,2}, pool 3 of 4 chunks")
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                log.seek(0)
                print(log.read(), end="")


if __name__ == "__main__":
    main()
