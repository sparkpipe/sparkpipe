#!/usr/bin/env python3
"""Stated-identity contracts for the weightd mesh (audit A-0036).

The mesh rank, the verbs interface, and the sgid index arrive as explicit
launch parameters from the deployment; no hostname tail may ever decide
identity, and no interface name or sgid index may be baked into the source.
Also compiles the daemon translation units with the tree's syntax stubs.
"""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MESH = (ROOT / "node/weightd_mesh.c").read_text(encoding="utf-8")
DAEMON = (ROOT / "node/weightd.c").read_text(encoding="utf-8")
AGENT = (ROOT / "tools/fleet_node_agent.sh").read_text(encoding="utf-8")

assert "RankFromHost" not in MESH
assert "gethostname" not in MESH
assert "hostname" not in MESH
assert '"rocep1s0f1"' not in MESH
assert "sgid_index = 3" not in MESH
assert "SparkWeightdMeshInit(uint32_t rank" in MESH
assert "--mesh-rank" in DAEMON
assert "--mesh-interface" in DAEMON
assert "--mesh-sgid-index" in DAEMON
assert "--mesh-rank" in AGENT
assert "--mesh-interface" in AGENT
assert "--mesh-sgid-index" in AGENT
assert "RANK=$((16#${HOST#spark}))" not in AGENT

with tempfile.TemporaryDirectory(prefix="weightd-mesh-src-") as build:
    for source in ("node/weightd.c", "node/weightd_mesh.c"):
        subprocess.run(
            ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-D_GNU_SOURCE", "-pthread", "-c",
             "-Itests/rdma_syntax_stub", "-Iinclude", source,
             "-o", str(Path(build) / (Path(source).stem + ".o"))],
            cwd=ROOT,
            check=True,
        )
