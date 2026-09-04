#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    repository = Path(__file__).resolve().parents[1]
    manager = repository / "build" / "sparkpipe_release_manager"
    subprocess.run(
        ["make", "build/sparkpipe_release_manager"],
        check=True,
        cwd=repository,
    )
    with tempfile.TemporaryDirectory(
            prefix="sparkpipe_release_agent_",dir="/private/tmp") as directory:
        root = Path(directory)
        release = root / "release"
        daemon = release / "bin" / "exit_immediately"
        marker = root / "starts"
        daemon.parent.mkdir(parents=True)
        daemon.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$$\" >> "
            '"$SPARKPIPE_TEST_MARKER"\nexit 1\n',
            encoding="utf-8",
        )
        daemon.chmod(0o755)
        manifest = {
            "schema_version": 1,
            "generation": 1,
            "release_id": "release-agent-reap-test",
            "git_commit": "0" * 40,
            "install_root": str(root / "install"),
            "state_root": str(root / "state"),
            "rank_count": 1,
            "max_active_sequence_count": 1,
            "poll_interval_ms": 10,
            "stop_grace_ms": 50,
            "files": [{
                "path": "bin/exit_immediately",
                "bytes": daemon.stat().st_size,
                "sha256": hashlib.sha256(daemon.read_bytes()).hexdigest(),
                "executable": True,
            }],
            "roles": [{
                "name": "test",
                "selector": "rank",
                "command": "bin/exit_immediately",
                "argv": [],
                "env": [f"SPARKPIPE_TEST_MARKER={marker}"],
            }],
        }
        (release / "sparkpipe.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        agent_log = root / "agent.log"
        with agent_log.open("wb") as log_stream:
            process = subprocess.Popen(
                [
                    str(manager), "agent",
                    "--release-dir", str(release),
                    "--state-dir", str(root / "agent-state"),
                    "--install-dir", str(root / "install"),
                    "--host", "spark0",
                    "--rank", "0",
                    "--role", "test",
                    "--poll-ms", "10",
                ],
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 3.0
                starts = []
                while time.monotonic() < deadline:
                    if marker.exists():
                        starts = marker.read_text(encoding="utf-8").splitlines()
                    if len(set(starts)) >= 2:
                        break
                    time.sleep(0.01)
            finally:
                process.terminate()
                process.wait(timeout=3)
        diagnostics = []
        for path in sorted(root.rglob("*")):
            diagnostics.append(str(path.relative_to(root)))
            if path.name.endswith(".log"):
                diagnostics.append(path.read_text(encoding="utf-8"))
        assert len(set(starts)) >= 2, (
            "release agent did not reap and restart its exited child:\n" +
            agent_log.read_text(encoding="utf-8") + "\n" +
            "\n".join(diagnostics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
