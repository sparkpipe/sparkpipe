#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "rtx5090_spec_node.py"
SPEC = importlib.util.spec_from_file_location("rtx5090_spec_node",MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import rtx5090_spec_node")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PROFILE = ROOT / "deployment/rtx5090_speculation/node.example.json"


class SpecNodeProfileTest(unittest.TestCase):
    def test_profile_is_auxiliary_and_exposes_runtime_boundary(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        self.assertIsNone(profile["spark_rank"])
        self.assertEqual(profile["runtime"]["state"],"infrastructure_only")
        self.assertEqual(profile["runtime"]["transport"],"not_implemented")
        self.assertEqual(profile["management"]["wired"]["route_metric"],100)
        self.assertEqual(profile["management"]["wifi"]["route_metric"],300)
        self.assertEqual(profile["storage"]["drafters_path"],"/srv/drafters")
        self.assertEqual(len(profile["hostname_aliases"]["spark_nodes"]),16)

    def test_private_link_is_isolated_point_to_point(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        netplan = MODULE.render_netplan(profile)
        command = MODULE.render_sparkf_nmcli(profile)
        self.assertIn("198.51.100.2/30",netplan)
        self.assertIn("mtu: 9000",netplan)
        self.assertNotIn("gateway",netplan)
        self.assertIn("198.51.100.1/30",command)
        self.assertIn("ipv4.never-default",command)
        self.assertIn("yes",command)

    def test_emergency_ssh_is_key_only_and_separate(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        config = MODULE.render_emergency_sshd(profile)
        self.assertIn("Port 22022",config)
        self.assertIn("PermitRootLogin no",config)
        self.assertIn("PasswordAuthentication no",config)
        self.assertIn("AllowUsers recovery",config)

    def test_rejects_spark_rank_assignment(self) -> None:
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        profile["spark_rank"] = 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile),encoding="utf-8")
            with self.assertRaises(MODULE.SpecNodeError):
                MODULE.load_profile(path)

    def test_live_gate_rejects_wrong_link_speed(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        outputs = [
            "rtx5090",
            "NVIDIA GeForce RTX 5090, 595.84",
            "Dual MIT/GPL",
            "SecureBoot enabled",
            "wired0 192.0.2.4\ndefault dev wired0 metric 100\ndefault dev wifi0 metric 300",
            "SSID: REPLACE_WITH_SITE_SSID\nwifi0 192.0.2.68",
            "/srv/drafters ext4\n653G\nspec:spec",
            "198.51.100.2/30\n1000",
            "enabled\nactive",
            "rtx5090",
            "2 packets transmitted, 2 received",
            "198.51.100.1/30\n10000",
            "2 packets transmitted, 2 received",
            "rtx5090",
        ]
        with mock.patch.object(MODULE,"_run",side_effect=outputs):
            with self.assertRaisesRegex(MODULE.SpecNodeError,"rtx_link"):
                MODULE.verify_live(profile)


if __name__ == "__main__":
    unittest.main()
