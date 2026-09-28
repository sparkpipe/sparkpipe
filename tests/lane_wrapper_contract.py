import os
import socket
import stat
import subprocess
import uuid
from pathlib import Path

FLEET_WEIGHTD_SOCKET = "/tmp/spark_weightd.sock"
STALE_SHARED_UNIT_SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"
WORLD = 16


def foreign_rank():
    host = socket.gethostname().split(".")[0]
    return next(rank for rank in range(WORLD) if f"spark{rank:x}" != host)


def fleet_socket_is_live():
    try:
        return stat.S_ISSOCK(os.stat(FLEET_WEIGHTD_SOCKET).st_mode)
    except OSError:
        return False


class LaneWrapper:
    def __init__(self, script, prefix):
        self.script = Path(script)
        self.prefix = prefix
        self.attempt = uuid.uuid4().hex
        self.runtime_root = Path("/tmp/sparkqueue-" + self.attempt)

    def base(self, socket_path):
        return {
            "SPARK_QUEUE_ATTEMPT": self.attempt,
            "SPARK_QUEUE_RUNTIME_ROOT": str(self.runtime_root),
            "SPARK_QUEUE_RANK": str(foreign_rank()),
            "SPARK_QUEUE_SIZE": str(WORLD),
            self.prefix + "WEIGHTD_SOCKET": str(socket_path),
        }

    def run(self, overrides):
        environment = {key: value for key, value in os.environ.items()
                       if not (key.startswith("SPARK_QUEUE_")
                               or key.startswith(self.prefix)
                               or key == "SPARK_WEIGHTD_SOCKET")}
        environment.update(overrides)
        return subprocess.run(["bash", str(self.script)], env=environment,
                              capture_output=True, text=True)

    def queue_cases(self, base, reserved):
        without_attempt = {key: value for key, value in base.items()
                           if key != "SPARK_QUEUE_ATTEMPT"}
        return [
            ("missing attempt id", without_attempt,
             "authoritative spark queue"),
            ("malformed attempt id", dict(base, SPARK_QUEUE_ATTEMPT="z" * 32,
                                          SPARK_QUEUE_PORTS=reserved),
             "bad attempt id"),
            ("short attempt id", dict(base, SPARK_QUEUE_ATTEMPT="abc",
                                      SPARK_QUEUE_PORTS=reserved),
             "bad attempt id length"),
            ("job namespace mismatch",
             dict(base, SPARK_QUEUE_RUNTIME_ROOT="/tmp/elsewhere",
                  SPARK_QUEUE_PORTS=reserved),
             "unexpected job namespace"),
            ("queue size mismatch", dict(base, SPARK_QUEUE_SIZE="8",
                                         SPARK_QUEUE_PORTS=reserved),
             f"SPARK_QUEUE_SIZE must be {WORLD}"),
            ("rank out of range", dict(base, SPARK_QUEUE_RANK=str(WORLD),
                                       SPARK_QUEUE_PORTS=reserved),
             f"SPARK_QUEUE_RANK must be 0..{WORLD - 1}"),
            ("no reserved ports", dict(base), "queue reserved no ports"),
            ("ports outside the lane blocks",
             dict(base, SPARK_QUEUE_PORTS="7000:7001"),
             "not inside a queue-reserved range"),
            ("dead weightd socket", dict(base, SPARK_QUEUE_PORTS=reserved),
             f"shared weightd socket {base[self.prefix + 'WEIGHTD_SOCKET']} "
             "is not a live socket"),
        ]

    def check_cases(self, cases, failures):
        for name, overrides, expected in cases:
            result = self.run(overrides)
            if result.returncode == 0 or expected not in result.stderr:
                failures.append(
                    f"{self.script.name} case '{name}': rc={result.returncode}, "
                    f"expected '{expected}' in stderr: {result.stderr.strip()}")
            if self.runtime_root.exists():
                failures.append(f"{self.script.name} case '{name}' created "
                                f"{self.runtime_root} before failing")

    def default_socket_result(self, base, reserved):
        overrides = dict(base, SPARK_QUEUE_PORTS=reserved)
        del overrides[self.prefix + "WEIGHTD_SOCKET"]
        return self.run(overrides)

    def check_default_socket(self, base, reserved, live_expected, failures):
        result = self.default_socket_result(base, reserved)
        dead = (f"shared weightd socket {FLEET_WEIGHTD_SOCKET} "
                "is not a live socket")
        if fleet_socket_is_live():
            reached = result.returncode != 0 and dead not in result.stderr
            if live_expected is not None:
                reached = reached and live_expected in result.stderr
        else:
            reached = result.returncode != 0 and dead in result.stderr
        if not reached:
            failures.append(f"{self.script.name} must default to the fleet "
                            f"weightd {FLEET_WEIGHTD_SOCKET}: "
                            f"rc={result.returncode} {result.stderr.strip()}")
        if self.runtime_root.exists():
            failures.append(f"{self.script.name} default-socket run created "
                            f"{self.runtime_root}")


class LiveSocket:
    def __init__(self, directory):
        self.path = Path(directory) / "weightd.sock"
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def __enter__(self):
        self.listener.bind(str(self.path))
        self.listener.listen(1)
        return self.path

    def __exit__(self, *exception):
        self.listener.close()
        return False
