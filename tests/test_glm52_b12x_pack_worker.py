#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace


def load_worker_module():
    repo_root = Path(__file__).resolve().parents[1]
    worker_path = repo_root / "tools" / "glm52_b12x_pack_worker.py"
    spec = importlib.util.spec_from_file_location("glm52_b12x_pack_worker", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load B12x pack worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_worker_module()
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        queue_dir = root / "queue"
        model_dir = root / "model"
        aot_manifest = root / "aot.json"
        output_dir = root / "packs"
        model_dir.mkdir()
        aot_manifest.write_text("{}", encoding="utf-8")
        submit_args = SimpleNamespace(
            aot_manifest=str(aot_manifest),
            job_id="stage-11-18",
            layers="11,12,13",
            local_jobs=16,
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            queue_dir=queue_dir,
            verify_reused_sha256=False,
        )
        with redirect_stdout(StringIO()):
            assert module.command_submit(submit_args) == 0
        depth = module.queue_depth(queue_dir, 16)
        assert depth["pending"] == 1
        assert depth["running"] == 0
        assert depth["available"] == 16
        status_path = queue_dir / "status" / "stage-11-18.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["state"] == "pending"
        assert status["artifact_transfer"] == "none_control_plane_only"
        job = json.loads((queue_dir / "pending" / "stage-11-18.json").read_text(encoding="utf-8"))
        command = module.pack_command(Path("packer.py"), job)
        assert "--jobs" in command
        assert "16" in command
        assert "--model-dir" in command
        assert "--output-dir" in command
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
