import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "devcycle" / "ds4_spark_brickproof.py"
FABRIC = ROOT / "tools" / "devcycle" / "fleet" / "ds4_switched_fabric_apply.sh"
SPEC = importlib.util.spec_from_file_location("ds4_spark_brickproof_contract",SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

FACTORY_NETWORK_MANAGER = """[main]
plugins=ifupdown,keyfile

[keyfile]
unmanaged-devices=interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1
"""
FLEET_NETWORK_MANAGER = """[main]
plugins=ifupdown,keyfile

[keyfile]
unmanaged-devices=interface-name:enp1s0f0np0;interface-name:enp1s0f1np1;interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1;interface-name:enP2p1;interface-name:enP2p2

[device-ds4-uplink-wired]
match-device=interface-name:enP7s7
keep-configuration=no
"""


def fabric_functions() -> str:
    return(FABRIC.read_text().split('\nif [ "$(id -u)" -ne 0 ]; then',1)[0])


class NodeBaselineContractTest(unittest.TestCase):
    def test_emergency_sshd_admits_root_and_the_emergency_account_with_keys_only(self) -> None:
        lines = MODULE.EMERGENCY_SSH_CONFIG.splitlines()
        self.assertIn("AllowUsers root sparkemerg",lines)
        self.assertIn("Port 2222",lines)
        self.assertIn("PasswordAuthentication no",lines)
        self.assertIn("UsePAM no",lines)

    def test_earlyoom_policy_and_runtime_check_use_one_argument_string(self) -> None:
        self.assertEqual(MODULE.EARLYOOM_ARGS,"-m 3 -s 100 -r 5 --avoid sshd|systemd|init")
        self.assertEqual(MODULE.EARLYOOM_DEFAULT,f'EARLYOOM_ARGS="{MODULE.EARLYOOM_ARGS}"\n')

    def test_factory_boot_units_are_masked_including_the_boot_splash_wait(self) -> None:
        for unit in ("plymouth-quit-wait.service","srp_daemon.service","srptools.service","nvidia-spark-run-apt-upgrade-once.service","cloud-init.service"):
            self.assertIn(unit,MODULE.OPTIONAL_BOOT_UNITS)
        commands = []
        with mock.patch.object(MODULE,"has_configured_remote_storage",return_value=[]), \
                mock.patch.object(MODULE,"atomic_write"), \
                mock.patch.object(MODULE,"run",side_effect=lambda argv,**_kwargs: commands.append(argv)):
            MODULE.decouple_optional_boot_work()
        self.assertEqual([argv[2] for argv in commands if argv[:2] == ["systemctl","mask"]],list(MODULE.OPTIONAL_BOOT_UNITS))
        with mock.patch.object(MODULE,"has_configured_remote_storage",return_value=["iscsi"]):
            with self.assertRaises(MODULE.BrickproofError):
                MODULE.decouple_optional_boot_work()

    def test_every_asset_has_a_repository_source_and_every_unit_runs_an_installed_program(self) -> None:
        sources = {**MODULE.ASSET_SOURCES,**MODULE.LINKED_RANK_ASSET_SOURCES}
        for destination,(source,mode) in sources.items():
            with self.subTest(destination=destination):
                self.assertTrue((ROOT / source).is_file(),source)
                self.assertIn(mode,(0o644,0o755))
                if destination.startswith("/usr/local/"):
                    self.assertEqual(mode,0o755)
        installed = set(sources)
        for destination,(source,_mode) in sources.items():
            if not destination.endswith((".service",".conf")):
                continue
            for program in re.findall(r"^Exec\w*=-?(/usr/local/\S+)",(ROOT / source).read_text(),re.MULTILINE):
                with self.subTest(unit=destination,program=program):
                    self.assertIn(program,installed)

    def test_hub_route_unit_follows_the_hub_gateway_rank(self) -> None:
        for rank,linked in ((0,True),(7,True),(14,True),(15,False)):
            with self.subTest(rank=rank),mock.patch.object(MODULE,"run") as run,mock.patch.object(MODULE,"disable_persistent_unit") as disable:
                MODULE.enable_policy_services(rank)
                enabled = [call.args[0] for call in run.call_args_list if call.args and "sparkpipe-hub-route.service" in call.args[0]]
                self.assertEqual(enabled,[["systemctl","enable","sparkpipe-hub-route.service"]] if linked else [])
                self.assertEqual(mock.call("sparkpipe-hub-route.service") in disable.call_args_list,not linked)

    def test_network_manager_parsing_reports_managed_fabric_and_the_assume_race(self) -> None:
        self.assertEqual(set(MODULE.FABRIC_INTERFACES) - MODULE.unmanaged_interfaces(FACTORY_NETWORK_MANAGER),{"enp1s0f0np0","enp1s0f1np1"})
        self.assertFalse(MODULE.uplink_assume_disabled(FACTORY_NETWORK_MANAGER))
        self.assertEqual(set(MODULE.FABRIC_INTERFACES) - MODULE.unmanaged_interfaces(FLEET_NETWORK_MANAGER),set())
        self.assertTrue(MODULE.uplink_assume_disabled(FLEET_NETWORK_MANAGER))
        other = FLEET_NETWORK_MANAGER.replace("interface-name:enP7s7","interface-name:wlP9s9")
        self.assertFalse(MODULE.uplink_assume_disabled(other))

    def test_hosts_audit_compares_entries_not_block_order(self) -> None:
        managed = "\n".join((MODULE.HOSTS_BLOCK_BEGIN,*MODULE.fleet_alias_lines(),MODULE.HOSTS_BLOCK_END))
        node = f"127.0.0.1\tlocalhost\n\n{managed}\n# DS4 RTX5090 SPECULATION NODE\n192.168.50.4 rtx5090\n"
        canonical = MODULE.canonical_hosts(node,{"spark3"})
        self.assertNotEqual(node,canonical)
        self.assertEqual(MODULE.hosts_entries(node),MODULE.hosts_entries(canonical))
        stale = node + "127.0.0.1 thinkstationpgx-1449\n"
        self.assertNotEqual(MODULE.hosts_entries(stale),MODULE.hosts_entries(canonical))

    def test_asset_drift_reports_changed_content_mode_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            same = root / "same"
            same.write_text("a\n")
            same.chmod(0o644)
            changed = root / "changed"
            changed.write_text("b\n")
            changed.chmod(0o644)
            mode = root / "mode"
            mode.write_text("c\n")
            mode.chmod(0o600)
            assets = {
                str(same):{"text":"a\n","mode":0o644},
                str(changed):{"text":"B\n","mode":0o644},
                str(mode):{"text":"c\n","mode":0o644},
                str(root / "missing"):{"text":"d\n","mode":0o644},
            }
            self.assertEqual(MODULE.asset_drift(assets),sorted([str(changed),str(mode),str(root / "missing")]))

    def test_stale_snap_mount_units_are_found_and_retired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snaps = root / "snaps"
            snaps.mkdir()
            present = snaps / "core22_2956.snap"
            present.write_text("")
            systemd = root / "systemd"
            (systemd / "snapd.mounts.target.wants").mkdir(parents=True)
            (systemd / "snap-core22-2956.mount").write_text(f"[Mount]\nWhat={present}\n")
            stale = systemd / "snap-core22-2438.mount"
            stale.write_text("[Mount]\nWhat=/var/lib/snapd/snaps/core22_2438.snap\n")
            link = systemd / "snapd.mounts.target.wants" / stale.name
            link.symlink_to(stale)
            self.assertEqual(MODULE.stale_snap_units(systemd),["snap-core22-2438.mount"])
            with mock.patch.object(MODULE,"run"):
                retired = MODULE.retire_stale_snap_units(systemd,root / "backup")
            self.assertEqual(retired,["snap-core22-2438.mount"])
            self.assertFalse(stale.exists())
            self.assertFalse(link.is_symlink())
            self.assertTrue((systemd / "snap-core22-2956.mount").exists())

    def test_apply_sets_network_manager_after_assets_and_arms_the_guard_last(self) -> None:
        for rank,node in ((3,"spark3"),(15,"sparkf")):
            with self.subTest(node=node),tempfile.TemporaryDirectory() as directory:
                payload = Path(directory) / "payload.json"
                payload.write_text(json.dumps({
                    "assets":{"/a":{"text":"a\n","mode":0o644}},
                    "linked_rank_assets":{"/route":{"text":"r\n","mode":0o644}},
                    "fleet_public_key":"ssh-ed25519 AAAA test",
                    "recovery_network":{"node_id":node},
                    "source_commit":"abc",
                }))
                calls = []
                names = (
                    "install_packages","install_required_kernel_modules","install_node_identity","atomic_write",
                    "install_sshd_config","install_recovery_keys","install_grub_policy","install_efi_boot_order",
                    "rebuild_initramfs","decouple_optional_boot_work","install_ceph_policy","retire_stale_snap_units",
                )
                patches = [mock.patch.object(MODULE,name,side_effect=lambda *args,_name=name,**_kwargs: calls.append((_name,args)) or []) for name in names]
                patches.append(mock.patch.object(MODULE,"install_assets",side_effect=lambda assets: calls.append(("install_assets",tuple(assets)))))
                patches.append(mock.patch.object(MODULE,"enable_policy_services",side_effect=lambda value: calls.append(("enable_policy_services",(value,)))))
                patches.append(mock.patch.object(MODULE,"run",side_effect=lambda argv,**_kwargs: calls.append(("run",tuple(argv)))))
                patches.append(mock.patch.object(MODULE,"command",return_value="0"))
                patches.append(mock.patch.object(MODULE.os,"geteuid",return_value=0))
                for patch in patches:
                    patch.start()
                try:
                    MODULE.remote_apply(payload)
                finally:
                    for patch in patches:
                        patch.stop()
                order = [name if name != "run" else " ".join(args) for name,args in calls]
                installs = [args for name,args in calls if name == "install_assets"]
                self.assertEqual(installs,[("/a",),("/route",)] if rank != 15 else [("/a",)])
                network = order.index(f"{MODULE.NETWORK_MANAGER_RUNTIME} --network-manager")
                self.assertLess(max(index for index,name in enumerate(order) if name == "install_assets"),network)
                guard = order.index(f"{MODULE.DMA_GUARD_RUNTIME} install --notify {MODULE.HUB_NOTIFY}")
                self.assertLess(order.index("enable_policy_services"),guard)
                self.assertLess(order.index("install_ceph_policy"),order.index("enable_policy_services"))
                self.assertIn(("enable_policy_services",(rank,)),calls)

    def test_switched_fabric_network_manager_mode_writes_the_fleet_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conf = Path(directory) / "conf.d"
            backups = Path(directory) / "backups"
            conf.mkdir()
            (conf / "ds4-200g-unmanaged.conf").write_text("[keyfile]\nunmanaged-devices=interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1\n")
            (conf / "99-ds4-ring-unmanaged.conf").write_text("[keyfile]\nunmanaged-devices=interface-name:enp1s0f0np0\n")
            harness = (fabric_functions().replace("/etc/NetworkManager/conf.d",str(conf)).replace("/var/backups/ds4-fleet",str(backups))
                       + f'\nnmcli() {{ echo "NMCLI $*" >> "{directory}/nmcli.log"; }}\nfabric_devices() {{ printf "%s\\n" enp1s0f0np0; }}\nconfigure_unmanaged_fabric\n')
            result = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(sorted(path.name for path in conf.iterdir()),["90-ds4-switched-fabric-unmanaged.conf","95-ds4-uplink-no-assume.conf"])
            merged = "\n".join((conf / name).read_text() for name in sorted(os.listdir(conf)))
            self.assertEqual(set(MODULE.FABRIC_INTERFACES) - MODULE.unmanaged_interfaces(merged),set())
            self.assertTrue(MODULE.uplink_assume_disabled(merged))
            self.assertEqual(len(list(backups.rglob("ds4-200g-unmanaged.conf"))),1)
            self.assertIn("NMCLI device set enp1s0f0np0 managed no",(Path(directory) / "nmcli.log").read_text())
        script = FABRIC.read_text()
        mode = script.split('if [ "${1:-}" = "--network-manager" ]; then',1)[1].split("\nfi\n",1)[0]
        self.assertEqual([line.strip() for line in mode.strip().splitlines()],["configure_unmanaged_fabric","exit 0"])

    def test_switched_fabric_routes_the_ceph_mds_only_through_an_on_link_gateway(self) -> None:
        cases = (
            ("10.20.0.2 dev enP7s7 src 10.20.0.13 uid 1000 \\    cache","10.10.100.2_via_10.20.0.2",True),
            ("10.20.0.2 via 192.168.50.1 dev wlP9s9 src 192.168.50.143 uid 1000 \\    cache","skipped",False),
        )
        for route,expected,replaced in cases:
            with self.subTest(route=route):
                harness = (fabric_functions()
                           + f'\nip() {{ if [ "$1 $2 $3 $4" = "-4 -o route get" ]; then echo "{route}"; else echo "IP $*" >&2; fi; }}\n'
                           + 'out=$(configure_ceph_mds_route); echo "OUT=$out"\n')
                result = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(f"OUT={expected}",result.stdout)
                self.assertEqual("IP route replace 10.10.100.2/32 via 10.20.0.2 dev enP7s7" in result.stderr,replaced)
        harness = fabric_functions() + '\nip() { if [ "$2" = "-o" ]; then echo "10.20.0.2 dev enP7s7 src 10.20.0.13"; else return 2; fi; }\nconfigure_ceph_mds_route\n'
        failed = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
        self.assertNotEqual(failed.returncode,0)


if __name__ == "__main__":
    unittest.main()
