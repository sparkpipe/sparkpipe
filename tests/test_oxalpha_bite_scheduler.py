#!/usr/bin/env python3

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oxalpha_bite_scheduler", ROOT / "tools" / "oxalpha_bite_scheduler.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bite(task_id, path, priority=0):
    return {
        "id": task_id,
        "pert_id": "FND-004",
        "biggulp": "foundation",
        "status": "ready",
        "development_phase": "production",
        "action_kind": "production_code",
        "title": task_id,
        "objective": "produce one tested result",
        "dependencies": [],
        "write_set": [path],
        "test_commands": ["git diff --check"],
        "priority": priority,
    }


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        state_root = root / "states"
        state_root.mkdir()
        catalog = {
            "bites": [
                bite("BITE-A", "tools/a.py", 3),
                bite("BITE-B", "tools/b.py", 2),
                bite("BITE-C", "tools/a.py", 1),
                {
                    **bite("SPARK-A", "tools/c.py", 9),
                    "action_kind": "hardware_experiment",
                },
                bite("EXTERNAL-A", "/private/tmp/not-a-repo-patch.py", 8),
            ]
        }
        pert = {"tasks": [{"id": "FND-004"}]}
        queue = {
            "ready": ["foundation"], "waiting": [], "completed": [],
            "running": [], "blocked": [],
        }
        plan = MODULE.build_plan(
            catalog, pert, state_root, queue, target=2,
            stale_seconds=3600, now=1000,
        )
        assert [item["task"]["id"] for item in plan["selected"]] == ["BITE-A", "BITE-B"]
        assert plan["coverage"] == {
            "pert_tasks": 1,
            "bites": 5,
            "covered_pert_tasks": 1,
            "uncovered_pert_tasks": 0,
            "dependency_ready_gaps": [],
        }
        assert {item["id"]: item["reason"] for item in plan["skipped"]} == {
            "BITE-C": "target_full",
            "SPARK-A": "external_hardware_experiment",
            "EXTERNAL-A": "external_write_set",
        }
        dispatcher = root / "dispatcher.json"
        MODULE.launch(
            plan, root / "runner.py", root / "tasks", dispatcher, root,
            dry_run=True,
        )
        compact = json.loads(dispatcher.read_text())["plan"]
        assert "skipped" not in compact
        assert compact["skipped_count"] == 3
        assert compact["skipped_by_reason"] == {
            "external_hardware_experiment": 1,
            "external_write_set": 1,
            "target_full": 1,
        }
        active = state_root / "LIVE"
        write(active / "state.json", {
            "phase": "provider_stream_active", "runner_pid": 99999999,
            "updated_at": 995,
        })
        write(active / "task.json", {"write_set": ["tools/b.py"]})
        plan = MODULE.build_plan(
            catalog, pert, state_root, queue, target=3,
            stale_seconds=10, now=1000,
        )
        assert plan["active"] == []
        assert [item["task"]["id"] for item in plan["selected"]] == ["BITE-A", "BITE-B"]
        original_process_state = MODULE.process_state
        original_waitpid = MODULE.os.waitpid
        original_kill = MODULE.os.kill
        MODULE.os.waitpid = lambda pid, _flags: (pid, 0)
        MODULE.os.kill = lambda _pid, _signal: None
        assert MODULE.process_state(12345) == "absent"
        MODULE.os.waitpid = original_waitpid
        MODULE.os.kill = original_kill
        MODULE.process_state = lambda _pid: "unknown"
        plan = MODULE.build_plan(
            catalog, pert, state_root, queue, target=3,
            stale_seconds=10, now=1000,
        )
        assert plan["active"] == ["LIVE"]
        MODULE.process_state = original_process_state
        failed = state_root / "PROVIDER-FAILED"
        write(failed / "state.json", {
            "phase": "failed", "updated_at": 900,
            "error": "RaceFailure: implementation exhausted 12 provider failures",
        })
        (failed / "provider-events.jsonl").write_text(json.dumps({
            "event": "race_failed",
            "failures": [{"status": 401}, {"status": 404}],
        }) + "\n")
        assert MODULE.task_state(
            state_root, "PROVIDER-FAILED", 1000, 10,
            provider_config_mtime=899,
        )[0] == "terminal"
        assert MODULE.task_state(
            state_root, "PROVIDER-FAILED", 1000, 10,
            provider_config_mtime=901,
        )[0] == "stale"
        external = state_root / "EXTERNAL-LIVE"
        write(external / "state.json", {
            "phase": "provider_stream_active", "updated_at": 995,
        })
        write(external / "task.json", {"write_set": ["tools/b.py"]})
        plan = MODULE.build_plan(
            catalog, pert, state_root, queue, target=3,
            stale_seconds=10, now=1000,
        )
        assert plan["active"] == ["EXTERNAL-LIVE"]
        assert plan["orphan_active"] == ["EXTERNAL-LIVE"]
        assert [item["task"]["id"] for item in plan["selected"]] == ["BITE-A"]
        stale = state_root / "BITE-B"
        write(stale / "state.json", {
            "phase": "provider_stream_active", "runner_pid": 99999999,
            "updated_at": 1,
        })
        plan = MODULE.build_plan(
            catalog, pert, state_root, queue, target=3,
            stale_seconds=10, now=1000,
        )
        selected = {item["task"]["id"]: item["restart"] for item in plan["selected"]}
        assert selected == {"BITE-A": False}
        atom = state_root / "ATOM"
        write(atom / "state.json", {"phase": "integrated"})
        write(atom / "task.json", {"pert_id": "PARENT"})
        dependency_bites = [{
            "id": "ATOM", "pert_id": "PARENT", "completes_pert": False,
        }, {
            "id": "COMPLETE", "pert_id": "PARENT", "completes_pert": True,
        }]
        satisfied = MODULE.satisfied_dependency_ids(state_root, dependency_bites)
        assert "ATOM" in satisfied
        assert "PARENT" not in satisfied
        complete = state_root / "COMPLETE"
        write(complete / "state.json", {"phase": "integrated"})
        write(complete / "task.json", {"pert_id": "PARENT"})
        assert "PARENT" in MODULE.satisfied_dependency_ids(
            state_root, dependency_bites
        )
    print("oxalpha bite scheduler tests: PASS")


if __name__ == "__main__":
    main()
