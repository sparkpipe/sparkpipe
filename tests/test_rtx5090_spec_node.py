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
NODES = [
    "spark0","spark1","spark2","spark3","spark4","spark5","spark6","spark7",
    "spark8","spark9","sparka","sparkb","sparkc","sparkd","sparke","sparkf",
]
HEALTHY = {
    "hostname": "rtx5090",
    "nvidia-smi": "NVIDIA GeForce RTX 5090, 595.84",
    "modinfo": "Dual MIT/GPL",
    "mokutil": "SecureBoot enabled",
    "ip -4 -o address show dev wired0": "wired0 192.0.2.4\ndefault dev wired0 metric 100\ndefault dev wifi0 metric 300",
    "iw dev wifi0": "SSID: REPLACE_WITH_SITE_SSID\nwifi0 192.0.2.68",
    "findmnt": "/srv/drafters ext4\n653G\nspec:spec",
    "ip -4 -o address show dev direct0": "4: direct0 inet 198.51.100.2/30 scope global direct0\n10000",
    "ip -4 route show 203.0.113.0/24": "203.0.113.0/24 via 198.51.100.1 dev direct0",
    "systemctl is-enabled tailscaled": "enabled\nactive",
    "systemctl --user is-active g53-api": "active",
    "ls current": "\n".join(f"{node}.json" for node in NODES),
    "ping": "2 packets transmitted, 2 received",
    "ip -4 -o address show dev direct1": "10: direct1 inet 198.51.100.1/30 scope global direct1\n10000",
    "sysctl -n net.ipv4.ip_forward": "1",
    "systemctl is-active sparkpipe-hub-route.service": "active\n198.51.100.2 via 203.0.113.25 dev fabric0 src 203.0.113.10",
    "curl": "200",
}


def fake_run(overrides: dict[str,str]):
    outputs = dict(HEALTHY,**overrides)
    def run(argv: list[str]) -> str:
        command = argv[-1]
        matches = [key for key in outputs if command.startswith(key)]
        if len(matches) != 1:
            raise AssertionError(f"unexpected command {command!r}")
        return(outputs[matches[0]])
    return(run)


def write_profile(directory: str,change) -> Path:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    change(profile)
    path = Path(directory) / "profile.json"
    path.write_text(json.dumps(profile),encoding="utf-8")
    return(path)


class HubNodeProfileTest(unittest.TestCase):
    def test_profile_is_the_fleet_hub_without_a_rank(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        self.assertEqual(profile["role"],"hub")
        self.assertIsNone(profile["spark_rank"])
        self.assertEqual(profile["hub"]["release_http_port"],8802)
        self.assertEqual(profile["hub"]["api_port"],8433)
        self.assertEqual(profile["hub"]["api_unit"],"g53-api")
        self.assertEqual(profile["hub"]["heartbeat_directory"],"current")
        self.assertEqual(profile["fleet_link"]["speed_mbps"],10000)
        self.assertEqual(profile["runtime"],{"state":"infrastructure_only","transport":"not_implemented"})
        self.assertEqual(profile["hostname_aliases"]["spark_nodes"],NODES)

    def test_fleet_link_routes_the_fabric_through_sparkf(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        netplan = MODULE.render_netplan(profile)
        self.assertIn("addresses: [198.51.100.2/30]",netplan)
        self.assertIn("mtu: 9000",netplan)
        self.assertIn("- to: 203.0.113.0/24\n          via: 198.51.100.1\n",netplan)
        self.assertNotIn("default",netplan)
        self.assertNotIn("gateway",netplan)
        command = MODULE.render_sparkf_nmcli(profile)
        self.assertEqual(command[4],"ds4-speculation-link")
        self.assertEqual(command[command.index("ipv4.addresses") + 1],"198.51.100.1/30")
        self.assertEqual(command[command.index("ipv4.never-default") + 1],"yes")
        self.assertEqual(command[command.index("ipv4.gateway") + 1],"")
        self.assertEqual(MODULE.render_sparkf_forwarding(),"net.ipv4.ip_forward = 1\n")
        unit = MODULE.render_hub_route_unit(profile)
        self.assertIn("ExecStart=/usr/sbin/ip route replace 198.51.100.0/30 via 203.0.113.25 dev fabric0\n",unit)
        self.assertIn("ExecStop=/usr/sbin/ip route del 198.51.100.0/30 via 203.0.113.25 dev fabric0\n",unit)
        self.assertIn("After=ds4-switched-fabric.service network-online.target\n",unit)

    def test_emergency_ssh_is_key_only_and_separate(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        config = MODULE.render_emergency_sshd(profile)
        self.assertIn("Port 22022",config)
        self.assertIn("PermitRootLogin no",config)
        self.assertIn("PasswordAuthentication no",config)
        self.assertIn("AllowUsers recovery",config)

    def test_rejects_invalid_profiles(self) -> None:
        cases = {
            "spark rank": lambda profile: profile.update(spark_rank=16),
            "speculation-only role": lambda profile: profile.update(role="speculation"),
            "claimed draft transport": lambda profile: profile["runtime"].update(transport="rpc"),
            "1 GbE link": lambda profile: profile["fleet_link"].update(speed_mbps=1000),
            "fabric address outside fabric": lambda profile: profile["fleet_link"]["fabric"].update(sparkf_address="192.0.2.25"),
            "fabric overlapping link": lambda profile: profile["fleet_link"]["fabric"].update(network="198.51.100.0/24",sparkf_address="198.51.100.25"),
            "release port equal to API port": lambda profile: profile["hub"].update(release_http_port=8433),
            "wider link": lambda profile: profile["fleet_link"]["rtx5090"].update(address="198.51.100.2/29"),
        }
        for name,change in cases.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(MODULE.SpecNodeError):
                    MODULE.load_profile(write_profile(directory,change))

    def test_live_gate_accepts_a_healthy_hub(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        with mock.patch.object(MODULE,"_run",side_effect=fake_run({})):
            results = MODULE.verify_live(profile)
        self.assertEqual(results["spark_release"],"200")

    def test_live_gate_rejects_each_broken_hub_contract(self) -> None:
        profile = MODULE.load_profile(PROFILE)
        cases = {
            "rtx_link speed": {"ip -4 -o address show dev direct0": "4: direct0 inet 198.51.100.2/30\n1000"},
            "sparkf_link speed": {"ip -4 -o address show dev direct1": "10: direct1 inet 198.51.100.1/30\n100000"},
            "rtx_fabric_route": {"ip -4 route show 203.0.113.0/24": ""},
            "does not forward": {"sysctl -n net.ipv4.ip_forward": "0"},
            "spark_hub_route": {"systemctl is-active sparkpipe-hub-route.service": "inactive\n198.51.100.2 via 192.168.50.1 dev wan0"},
            "release endpoint": {"curl": "404"},
            "hub API unit": {"systemctl --user is-active g53-api": "inactive"},
            "heartbeats missing for sparkc": {"ls current": "\n".join(f"{node}.json" for node in NODES if node != "sparkc")},
        }
        for message,overrides in cases.items():
            with self.subTest(message=message):
                with mock.patch.object(MODULE,"_run",side_effect=fake_run(overrides)):
                    with self.assertRaisesRegex(MODULE.SpecNodeError,message):
                        MODULE.verify_live(profile)


if __name__ == "__main__":
    unittest.main()
