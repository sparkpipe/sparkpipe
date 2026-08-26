#!/usr/bin/env python3
"""Verify that generated GLM52 TP8 launch files support exact-32K tests."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "glm52_gen_deployment.py"
HOSTS = [
    "spark8", "spark9", "sparka", "sparkb",
    "sparkc", "sparkd", "sparke", "sparkf",
]
MODEL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
PACK_TEMPLATE = "packs/glm52_tp8_rank%02d.fp8.glms52sp"
COLLECTIVE_BASE = 63620
TRANSPORT_BASE = 60700


def generated_tree(output_dir):
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), str(output_dir)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    tree = {}
    for host in HOSTS:
        config_dir = output_dir / host / "config"
        stage = json.loads(
            (config_dir / "glm52_stage.json").read_text(encoding="utf-8")
        )
        resident = json.loads(
            (config_dir / "model_resident.json").read_text(encoding="utf-8")
        )
        tree[host] = (stage, resident)
    return tree


def test_exact_32k_positions(tree):
    assert len(tree) == len(HOSTS)
    for host in HOSTS:
        positions = tree[host][0]["max_sequence_positions"]
        assert isinstance(positions, int) and not isinstance(positions, bool), host
        assert positions == 32768, host


def test_unchanged_deployment_contract(tree):
    for rank, host in enumerate(HOSTS):
        stage, resident = tree[host]
        assert stage["schema_version"] == 3, host
        assert stage["model_revision"] == MODEL_REVISION, host
        assert stage["expert_weight_codec"] == "fp8", host
        assert stage["tp_degree"] == len(HOSTS), host
        assert stage["tp_rank"] == rank, host
        assert stage["stage_pack_path"] == PACK_TEMPLATE % rank, host
        collective = stage["tp_collective"]
        assert collective["backend"] == "hidden_transport", host
        assert collective["peer_hosts"] == HOSTS, host
        assert collective["peer_ports"] == [
            COLLECTIVE_BASE + peer for peer in range(len(HOSTS))
        ], host
        nodes = resident["nodes"]
        assert len(nodes) == len(HOSTS), host
        node = nodes[rank]
        assert node["rank_index"] == rank, host
        assert node["stage_index"] == rank, host
        assert node["transport_host"] == host, host
        assert resident["transport"]["control_port_base"] == TRANSPORT_BASE, host


def main():
    with tempfile.TemporaryDirectory(prefix="glm52-deploy-test-") as temporary:
        tree = generated_tree(Path(temporary))
        test_exact_32k_positions(tree)
        test_unchanged_deployment_contract(tree)
    print("PASS (GLM52 TP8 exact-32K deployment contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
