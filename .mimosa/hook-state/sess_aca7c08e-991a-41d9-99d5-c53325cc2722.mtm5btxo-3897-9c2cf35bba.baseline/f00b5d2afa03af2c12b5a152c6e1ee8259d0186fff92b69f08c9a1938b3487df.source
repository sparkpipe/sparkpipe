import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "devcycle" / "ds4_parallel_pxe_rescue.py"
SPEC = importlib.util.spec_from_file_location("ds4_parallel_pxe_rescue",SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParallelPxeRescueTest(unittest.TestCase):
    CONFIG = {
        "format":"ds4-parallel-pxe-rescue-v2",
        "interface":"enP7s7",
        "recovery_public_key":"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA test",
        "root_device":"/dev/nvme0n1p2",
        "server_ip":"192.168.50.128",
        "source_commit":"abc123",
    }

    def test_dnsmasq_serves_any_arm64_client(self) -> None:
        result = MODULE.dnsmasq_config(self.CONFIG)
        self.assertNotIn("dhcp-host=",result)
        self.assertNotIn("dhcp-ignore=",result)
        self.assertIn("option:client-arch,11",result)
        self.assertIn("tag:efi-arm64,shimaa64.efi",result)
        self.assertIn("tftp-root=/var/lib/ds4-pxe-rescue",result)

    def test_grub_only_boots_to_login_with_fabric_bypassed(self) -> None:
        result = MODULE.grub_config(self.CONFIG)
        self.assertIn("root=/dev/nvme0n1p2",result)
        self.assertIn("ip=:::::enP7s7:dhcp",result)
        self.assertIn("systemd.unit=multi-user.target",result)
        self.assertIn("root=/dev/nvme0n1p2 rw",result)
        self.assertIn("systemd.mask=ds4-switched-fabric.service",result)
        self.assertIn("systemd.mask=ds4-direct-pair-fabric.service",result)
        self.assertNotIn("ds4_spark_brickproof",result)
        self.assertNotIn("10.20.0.",result)

    def test_grub_config_covers_the_embedded_network_prefix(self) -> None:
        self.assertEqual(MODULE.GRUB_CONFIG_NAMES,("grub.cfg","grub/grub.cfg"))
        self.assertEqual(MODULE.EFFECTIVE_GRUB_CONFIG,MODULE.STATE_DIR / "grub" / "grub.cfg")

    def test_controller_stages_the_committed_sparkpipe_tool(self) -> None:
        self.assertEqual(MODULE.ROOT,Path(__file__).resolve().parents[1])
        self.assertEqual(
            MODULE.REPOSITORY_SCRIPT,
            "tools/devcycle/ds4_parallel_pxe_rescue.py",
        )

    def test_staged_remote_installer_starts_outside_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "ds4_parallel_pxe_rescue.py"
            shutil.copyfile(SCRIPT,staged)
            result = subprocess.run(
                [sys.executable,str(staged),"--help"],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode,0,result.stderr)

    def test_service_is_manual_and_bounded(self) -> None:
        result = MODULE.service_config()
        self.assertNotIn("[Install]",result)
        self.assertIn("TimeoutStartSec=30",result)
        self.assertIn("TimeoutStopSec=15",result)
        self.assertIn("--remote-preflight",result)
        self.assertIn("--remote-firewall-open",result)
        self.assertIn("--remote-firewall-close",result)

    def test_config_validation_rejects_bad_inputs(self) -> None:
        for field,value in (
            ("interface","enP7s7;reboot"),
            ("recovery_public_key","not-a-key"),
            ("server_ip","not-an-ip"),
            ("root_device","../../etc/passwd"),
        ):
            payload = dict(self.CONFIG)
            payload[field] = value
            with self.assertRaises(MODULE.PxeRescueError):
                MODULE.validated_config(payload)

    def test_rescue_initramfs_repairs_root_key_before_switch_root(self) -> None:
        self.assertIn("scripts/local-bottom/ds4-recovery-key",MODULE.RECOVERY_INITRAMFS_HOOK)
        self.assertIn("/root/root/.ssh/authorized_keys",MODULE.RECOVERY_LOCAL_BOTTOM)
        self.assertIn("mount -o remount,rw /root",MODULE.RECOVERY_LOCAL_BOTTOM)
        self.assertIn('cat "$key_file" > "$temporary"',MODULE.RECOVERY_LOCAL_BOTTOM)

    def test_invalid_drop_handle_is_found(self) -> None:
        rules = [
            {"handle":1,"expr":[{"accept":None}]},
            {"handle":7,"expr":[
                {"match":{"left":{"ct":{"key":"state"}},"right":"invalid"}},
                {"drop":None},
            ]},
        ]
        self.assertEqual(MODULE.invalid_drop_handle(rules),7)

    def test_probe_node_parser_rejects_an_empty_set(self) -> None:
        self.assertEqual(MODULE.parse_nodes("spark0, spark2"),("spark0","spark2"))
        with self.assertRaises(MODULE.PxeRescueError):
            MODULE.parse_nodes(" , ")

    def test_legacy_exact_dhcp_rule_is_detected(self) -> None:
        rule = {"handle":9,"expr":[
            {"match":{"left":{"meta":{"key":"iifname"}},"right":"enP7s7"}},
            {"match":{"left":{"payload":{"protocol":"ether","field":"saddr"}},"right":"02:00:00:00:00:01"}},
            {"match":{"left":{"payload":{"protocol":"udp","field":"sport"}},"right":68}},
            {"match":{"left":{"payload":{"protocol":"udp","field":"dport"}},"right":67}},
            {"accept":None},
        ]}
        self.assertTrue(MODULE.legacy_exact_dhcp_rule(rule,"enP7s7"))
        self.assertFalse(MODULE.legacy_exact_dhcp_rule(rule,"wlP9s9"))


if __name__ == "__main__":
    unittest.main()
