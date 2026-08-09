#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "generate_model_resident_deployment.py"
SPECIFICATION = (
    ROOT / "examples" / "deployments" /
    "dsv4_flash_pp13_host_rdma.spec.json")
DEPLOYMENT = (
    ROOT / "examples" / "deployments" / "dsv4_flash_pp13_host_rdma.json")
STAGE_CONFIGURATION = (
    ROOT / "examples" / "deployments" / "dsv4_flash_stage.json")


def load_module():
    specification = importlib.util.spec_from_file_location("deployment_generator", TOOL)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expect_failure(module, specification, message):
    try:
        module.build_deployment(specification)
    except module.DeploymentError:
        return
    raise AssertionError(message)


def main() -> int:
    module = load_module()
    specification = module.load_specification(SPECIFICATION)
    rendered = module.render_deployment(module.build_deployment(specification))
    assert rendered == DEPLOYMENT.read_text(encoding="utf-8")
    deployment = module.build_deployment(specification)
    assert deployment["runtime_limits"] == {
        "max_inflight_submissions": 13,
        "max_active_sequences": 128,
        "max_input_rows": 128,
        "resident_sequence_capacity": 1024,
    }
    assert deployment["adapter"]["shared_object_path"] == (
        "lib/model_serving_adapter.so")
    assert deployment["transport"]["shared_object_path"] == (
        "lib/hidden_transport.so")
    assert {node["adapter_configuration_path"]
            for node in deployment["nodes"]} == {
                "config/dsv4_flash_stage.json"}
    stage_configuration = module.load_specification(STAGE_CONFIGURATION)
    assert stage_configuration == {
        "schema_version": 3,
        "model_revision":
            "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "stage_pack_path": "packs/dsv4_flash_stage.spstage",
        "max_sequence_positions": 8192,
        "cuda_graph_count": 64,
    }
    source = TOOL.read_text(encoding="utf-8").lower()
    for forbidden in ("glm", "dsv", "codec", "int8", "fp8", "mxfp4"):
        assert forbidden not in source
    alternate = copy.deepcopy(specification)
    alternate["topology"]["node_target"] = "cuda.test.package.selected.target"
    alternate["adapter"]["shared_object_path"] = "lib/alternate_adapter.so"
    deployment = module.build_deployment(alternate)
    assert all(node["node_target"] == "cuda.test.package.selected.target"
               for node in deployment["nodes"])
    assert deployment["nodes"][12]["adapter_configuration_path"] == (
        "config/dsv4_flash_stage.json")
    invalid = copy.deepcopy(specification)
    invalid["runtime_limits"]["max_active_sequences"] = 129
    expect_failure(module, invalid, "inconsistent capacities accepted")
    invalid = copy.deepcopy(specification)
    invalid["topology"]["stage_indices"][12] = 11
    expect_failure(module, invalid, "duplicate stage accepted")
    invalid = copy.deepcopy(specification)
    invalid["topology"]["runtime_root_template"] = "/tmp/{unknown}"
    expect_failure(module, invalid, "unknown placeholder accepted")
    invalid = copy.deepcopy(specification)
    invalid["expert_weight_codec"] = "int8"
    expect_failure(module, invalid, "model-specific root member accepted")
    print("PASS strict model-neutral deployment generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
