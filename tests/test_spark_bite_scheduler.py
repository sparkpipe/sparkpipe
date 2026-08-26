#!/usr/bin/env python3

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_bite_scheduler", ROOT / "tools" / "spark_bite_scheduler.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pert = root / "pert.json"
        catalog = root / "catalog.json"
        queue = root / "queue.json"
        tasks = root / "tasks"
        write(pert, {"tasks": [{"id": "P1"}, {"id": "P2"}]})
        common = {
            "status": "ready", "development_phase": "exploratory",
            "objective": "answer one measured question",
            "expected_value": "unblocks the next correction",
            "required_data": ["raw output", "exact decision"],
        }
        write(catalog, {"bites": [
            dict(common, id="H1", pert_id="P1", biggulp="model-a",
                 action_kind="hardware_experiment", priority=9,
                 hardware={"nodes": ["spark1"], "resources": ["gpu"],
                           "runner_role": "model-launcher"},
                 spark_queue={"job_id": "J1", "result_path": str(root / "J1.json")}),
            dict(common, id="B1", pert_id="P2", biggulp="model-b",
                 action_kind="spark_model_benchmark", priority=8,
                 nodes=["spark2"], resources=["gpu"], role="benchmarker",
                 result_path=str(root / "B1.json")),
            dict(common, id="BAD", pert_id="P1", biggulp="model-a",
                 action_kind="hardware_experiment", priority=7),
            dict(common, id="B2", pert_id="P2", biggulp="model-b",
                 status="dependency_blocked", dependencies=["H1"],
                 action_kind="spark_model_benchmark", priority=6,
                 nodes=["spark3"], resources=["gpu"], role="benchmarker",
                 result_path=str(root / "B2.json")),
        ]})
        result = MODULE.run_once(catalog, pert, queue, root / "states", tasks)
        assert result["submitted"] == ["J1", "B1"]
        assert result["incomplete_contracts"] == [{
            "id": "BAD", "reason": "missing_nodes,resources,result,role"
        }]
        state = json.loads(queue.read_text())
        assert [job["job_id"] for job in state["jobs"]] == ["J1", "B1"]
        assert state["jobs"][0]["role"] == "model_launcher"
        assert json.loads((tasks / "H1.json").read_text())["id"] == "H1"
        again = MODULE.run_once(catalog, pert, queue, root / "states", tasks)
        assert again["submitted"] == []
        assert again["existing"] == ["J1", "B1"]
        assert len(json.loads(queue.read_text())["jobs"]) == 2
        state = json.loads(queue.read_text())
        state["jobs"][0]["state"] = "succeeded"
        queue.write_text(json.dumps(state))
        unlocked = MODULE.run_once(catalog, pert, queue, root / "states", tasks)
        assert unlocked["submitted"] == ["B2"]
    print("spark bite scheduler tests: PASS")


if __name__ == "__main__":
    main()
