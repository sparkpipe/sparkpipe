#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


COMMIT = "a" * 40
REQUIRED_FILES = (
    "bin/sparkpipe_model_residentd",
    "lib/model_serving_adapter.so",
    "lib/model_driver.so",
    "lib/hidden_transport.so",
    "config/model_resident.json",
)


def run_tool(tool,*arguments,check=True):
    return subprocess.run(
        ["python3",str(tool),*arguments],
        check=check,
        capture_output=not check,
        text=True,
    )


def make_template(root,roles=None):
    template = root / "template"
    for relative in REQUIRED_FILES:
        path = template / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(("old:" + relative).encode())
    manifest = {
        "schema_version": 1,
        "release_id": "old",
        "generation": 1,
        "git_commit": "0" * 40,
        "install_root": "/home/{host}/sparkpipe_runtime",
        "state_root": "/home/{host}/sparkpipe_state",
        "rank_count": 13,
        "max_active_sequence_count": 128,
        "poll_interval_ms": 1000,
        "stop_grace_ms": 5000,
        "files": [
            {"path": relative,"bytes": 0,"sha256": "0" * 64}
            for relative in REQUIRED_FILES
        ],
        "roles": roles if roles is not None else [
            {
                "name": "model_resident",
                "selector": "rank",
                "command": "bin/sparkpipe_model_residentd",
                "argv": [
                    "--deployment",
                    "{install_root}/config/model_resident.json",
                    "--rank-index",
                    "{rank}",
                ],
                "env": [
                    "LD_LIBRARY_PATH={install_root}/lib",
                    "KEEP=1",
                ],
            }
        ],
    }
    (template / "sparkpipe.json").write_text(
        json.dumps(manifest),encoding="utf-8")
    return template


def main():
    repository = Path(__file__).resolve().parents[1]
    tool = repository / "tools" / "sparkpipe_release_assemble.py"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        template = make_template(root)
        replacement = root / "model_driver.so"
        replacement.write_bytes(b"exact-model-driver")
        output = root / "output"
        run_tool(
            tool,
            "--template",str(template),
            "--output",str(output),
            "--release-id","dsv4-flash-production",
            "--git-commit",COMMIT,
            "--role-env-unset","model_resident=KEEP",
            "--role-env","model_resident=CUDA_MODULE_LOADING=EAGER",
            "--replace","lib/model_driver.so=" + str(replacement),
        )
        result = json.loads(
            (output / "sparkpipe.json").read_text(encoding="utf-8"))
        assert result["release_id"] == "dsv4-flash-production"
        assert result["git_commit"] == COMMIT
        assert result["generation"] > 20260000000000
        assert len(result["roles"]) == 1
        role = result["roles"][0]
        assert role["name"] == "model_resident"
        assert role["command"] == "bin/sparkpipe_model_residentd"
        assert "KEEP=1" not in role["env"]
        assert "CUDA_MODULE_LOADING=EAGER" in role["env"]
        assert "SPARKPIPE_RELEASE_ID=dsv4-flash-production" in role["env"]
        assert "SPARKPIPE_RELEASE_GIT_COMMIT=" + COMMIT in role["env"]
        assert any(value.startswith("SPARKPIPE_RELEASE_GENERATION=")
                   for value in role["env"])
        driver_entry = next(entry for entry in result["files"]
                            if entry["path"] == "lib/model_driver.so")
        assert driver_entry["bytes"] == len(b"exact-model-driver")
        assert driver_entry["sha256"] == hashlib.sha256(
            b"exact-model-driver").hexdigest()
        assert (output / "lib" / "model_driver.so").read_bytes() == (
            b"exact-model-driver")
        assert list(root.glob("output.assembling.*")) == []

        duplicate = run_tool(
            tool,
            "--template",str(template),
            "--output",str(root / "duplicate"),
            "--release-id","duplicate",
            "--git-commit",COMMIT,
            "--replace","lib/model_driver.so=" + str(replacement),
            "--replace","lib/model_driver.so=" + str(replacement),
            check=False,
        )
        assert duplicate.returncode != 0
        assert "replacement occurs more than once" in duplicate.stderr
        assert list(root.glob("duplicate.assembling.*")) == []

        bad_commit = run_tool(
            tool,
            "--template",str(template),
            "--output",str(root / "bad-commit"),
            "--release-id","bad-commit",
            "--git-commit","abc123",
            check=False,
        )
        assert bad_commit.returncode != 0
        assert "exact lowercase 40-hex commit" in bad_commit.stderr

        old_roles = [
            {"name":"spark0_gateway","command":"bin/old","argv":[]},
            {"name":"ring_cuda_residentd","command":"bin/old","argv":[]},
            {"name":"ring_rank_daemon","command":"bin/old","argv":[]},
        ]
        old_template = make_template(root / "old",old_roles)
        old_result = run_tool(
            tool,
            "--template",str(old_template),
            "--output",str(root / "old-output"),
            "--release-id","old",
            "--git-commit",COMMIT,
            check=False,
        )
        assert old_result.returncode != 0
        assert "only the model_resident role" in old_result.stderr
        assert list(root.glob("old-output.assembling.*")) == []

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
