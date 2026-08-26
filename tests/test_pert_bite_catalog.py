#!/usr/bin/env python3

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pert_bite_catalog", ROOT / "tools" / "pert_bite_catalog.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    pert = json.loads((ROOT / "orchestration" / "program_pert.json").read_text())
    active = json.loads((ROOT / "orchestration" / "active_bites.json").read_text())
    platform = json.loads((ROOT / "orchestration" / "platform_tasks.json").read_text())
    catalog = MODULE.build_catalog(pert, active, platform, ROOT)
    assert MODULE.validate_catalog(catalog, pert) == []
    summary = catalog["summary"]
    assert summary["pert_nodes"] == 423
    assert summary["fully_decomposed_nodes"] == 423
    assert summary["total_bites"] > 1200
    assert summary["ready_bites"] >= 100
    assert summary["dependency_ready_pert_nodes"] == sum(
        node["dependency_ready"] for node in catalog["nodes"]
    )
    assert summary["runnable_code_bites"] <= summary["ready_bites"]
    assert all(
        node["dependency_ready"] == bool(node["ready_bites"])
        for node in catalog["nodes"]
    )
    for logical in ("model-dsv4-flash", "model-glm52", "model-qwen38-27b"):
        assert summary["model_ready_bites"][logical] >= 6
    assert summary["model_ready_bites"]["model-minimax-h3"] >= 1
    node_ids = {node["pert_id"] for node in catalog["nodes"]}
    assert node_ids == {task["id"] for task in pert["tasks"]}
    generated = [
        bite for bite in catalog["bites"]
        if bite.get("catalog_origin") == "generated_full_pert"
    ]
    completion = [bite for bite in generated if bite.get("completes_pert")]
    assert len(completion) == len(pert["tasks"])
    assert {bite["pert_id"] for bite in completion} == {
        task["id"] for task in pert["tasks"]
    }
    for bite in completion:
        if bite["action_kind"] == "external_gate":
            continue
        assert bite["action_kind"] == "evidence_capture"
        assert len(bite["write_set"]) == 1
        assert bite["test_commands"]
    assert all(bite["time_budget_minutes"] <= 45 for bite in generated)
    assert all(
        bite.get("development_phase") == "exploratory"
        for bite in generated
        if bite["id"].startswith(("MOD-D4F-", "MOD-GLM-", "MOD-Q27-"))
        and bite["id"].rsplit("-", 1)[-1].startswith("B")
        and bite["status"] == "ready"
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "catalog.json"
        output.write_text(MODULE.canonical_json(catalog))
        assert json.loads(output.read_text()) == catalog
    print("PERT bite catalog tests: PASS")


if __name__ == "__main__":
    main()
