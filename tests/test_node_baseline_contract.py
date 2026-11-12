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
        self.assertEqual(MODULE.EARLYOOM_ARGS,"-m 2 -s 100 -r 5 --avoid sshd|systemd|init")
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
                    "node_layout_roots":["extnvme"],
                    "recovery_network":{"node_id":node},
                    "source_commit":"abc",
                }))
                calls = []
                names = (
                    "install_packages","install_required_kernel_modules","install_node_identity","atomic_write",
                    "install_sshd_config","install_recovery_keys","install_grub_policy","install_efi_boot_order",
                    "rebuild_initramfs","decouple_optional_boot_work","install_ceph_policy","retire_stale_snap_units",
                    "install_node_layout","heal_time_sync",
                )
                patches = [mock.patch.object(MODULE,name,side_effect=lambda *args,_name=name,**_kwargs: calls.append((_name,args)) or []) for name in names]
                patches.append(mock.patch.object(MODULE,"install_assets",side_effect=lambda assets: calls.append(("install_assets",tuple(assets)))))
                patches.append(mock.patch.object(MODULE,"enable_policy_services",side_effect=lambda value,node_id: calls.append(("enable_policy_services",(value,node_id)))))
                patches.append(mock.patch.object(MODULE,"run",side_effect=lambda argv,**_kwargs: calls.append(("run",tuple(argv)))))
                patches.append(mock.patch.object(MODULE,"command",return_value="0"))
                patches.append(mock.patch.object(MODULE.os,"geteuid",return_value=0))
                ruleset = Path(directory) / "spark-firewall.nft"
                ruleset.write_text("table inet spark_guard {}\n")
                patches.append(mock.patch.object(MODULE,"FIREWALL_RULESET",ruleset))
                patches.append(mock.patch.object(MODULE,"reset_vanished_failed_units",return_value=[]))
                for patch in patches:
                    patch.start()
                try:
                    MODULE.remote_apply(json.loads(payload.read_text()))
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
                self.assertIn(("install_node_layout",(node,["extnvme"])),calls)
                self.assertLess(max(index for index,name in enumerate(order) if name == "install_assets"),order.index("heal_time_sync"))
                self.assertIn(("enable_policy_services",(rank,node)),calls)

    def test_node_numbers_split_the_fleet_from_staged_nodes(self) -> None:
        for node,number,staged in (("spark0",0,False),("sparkf",15,False),("spark10",16,True),("spark1f",31,True),("spark20",32,True),("SPARK12",18,True)):
            with self.subTest(node=node):
                self.assertEqual(MODULE.node_number(node),number)
                self.assertEqual(MODULE.node_is_staged(node),staged)
        for invalid in ("spark01","sparkg","spark100","node10","spark"):
            with self.subTest(invalid=invalid),self.assertRaises(MODULE.BrickproofError):
                MODULE.node_number(invalid)
        self.assertEqual(MODULE.recovery_network_for_node("spark11"),{"node_id":"spark11"})
        with self.assertRaises(MODULE.BrickproofError):
            MODULE.spark_rank("spark11")

    def test_staged_grub_keeps_recovery_settings_without_a_management_address(self) -> None:
        source = 'GRUB_DEFAULT="x"\nGRUB_CMDLINE_LINUX_DEFAULT="quiet ip=10.20.0.27::10.20.0.1:255.255.255.0:spark11:enP7s7:none console=ttyS0"\n'
        once = MODULE.canonical_grub(source,{"node_id":"spark11"})
        tokens = MODULE.shell_assignment(once,"GRUB_CMDLINE_LINUX_DEFAULT").split()
        self.assertEqual(MODULE.shell_assignment(once,"GRUB_DEFAULT"),"0")
        self.assertFalse(any(token.startswith("ip=") for token in tokens))
        for token in ("console=tty0","console=ttyS0,921600","fsck.mode=skip","fsck.repair=no"):
            self.assertIn(token,tokens)
        self.assertEqual(MODULE.canonical_grub(once,{"node_id":"spark11"}),once)

    def test_staged_hosts_resolve_the_node_name_on_loopback_and_carry_no_fleet_aliases(self) -> None:
        factory = "127.0.0.1\tlocalhost\n127.0.0.1       spark11\n::1     ip6-localhost ip6-loopback\n"
        once = MODULE.canonical_hosts(factory,{"spark11","cube-48d2"},MODULE.node_alias_lines("spark11"))
        entries = MODULE.hosts_entries(once)
        self.assertIn(("127.0.1.1",("spark11",)),entries)
        self.assertIn(("127.0.0.1",("localhost",)),entries)
        self.assertNotIn(("127.0.0.1",("spark11",)),entries)
        self.assertFalse(MODULE.fleet_aliases() & {alias for _address,aliases in entries for alias in aliases})
        self.assertEqual(MODULE.canonical_hosts(once,{"spark11"},MODULE.node_alias_lines("spark11")),once)
        self.assertEqual(MODULE.node_alias_lines("spark3"),MODULE.fleet_alias_lines())

    def test_staged_assets_are_the_network_independent_subset(self) -> None:
        sources = dict(MODULE.ASSET_SOURCES)
        sources.update({f"/etc/systemd/system/{unit}.d/10-boot-timeout.conf":"" for unit in MODULE.BOOT_TIMEOUT_SOURCES})
        for path in MODULE.STAGED_ASSET_DESTINATIONS:
            with self.subTest(path=path):
                self.assertIn(path,sources)
                self.assertNotRegex(path,r"ceph|osd|fabric|roce|hub-route|timesyncd|resolved")
        assets = {path:{"text":"x\n","mode":0o644} for path in sources}
        self.assertEqual(sorted(MODULE.staged_assets(assets)),sorted(MODULE.STAGED_ASSET_DESTINATIONS))
        del assets[MODULE.DMA_GUARD_RUNTIME]
        with self.assertRaises(MODULE.BrickproofError):
            MODULE.staged_assets(assets)

    def test_staged_services_skip_fabric_hub_ceph_and_hub_time(self) -> None:
        with mock.patch.object(MODULE,"run") as run,mock.patch.object(MODULE,"disable_persistent_unit") as disable:
            MODULE.enable_policy_services(None,"spark11")
        commands = [" ".join(call.args[0]) for call in run.call_args_list]
        disabled = [call.args[0] for call in disable.call_args_list]
        for absent in ("fabric","hub-route","osd-guard","ceph","roce-qos","timesyncd"):
            with self.subTest(absent=absent):
                self.assertFalse([line for line in commands + disabled if absent in line])
        for present in ("systemctl enable spark-firewall.service","systemctl restart earlyoom.service","systemctl restart ssh-emergency.service",
                        "loginctl enable-linger spark11","systemctl set-default multi-user.target","systemctl enable --now fstrim.timer"):
            with self.subTest(present=present):
                self.assertIn(present,commands)

    def staged_apply(self,fsids: list[str]) -> list[tuple[str,tuple[object,...]]]:
        with tempfile.TemporaryDirectory() as directory:
            assets = {path:{"text":"s\n","mode":0o644} for path in MODULE.STAGED_ASSET_DESTINATIONS}
            assets["/fleet-only"] = {"text":"f\n","mode":0o644}
            payload = {"assets":assets,"linked_rank_assets":{"/route":{"text":"r\n","mode":0o644}},"fleet_public_key":"ssh-ed25519 AAAA test",
                       "node_layout_roots":["extnvme"],"recovery_network":{"node_id":"spark11"},"source_commit":"abc","apt_proxy":"socks5h://127.0.0.1:1080"}
            calls = []
            names = (
                "install_packages","install_required_kernel_modules","install_node_identity","atomic_write",
                "install_sshd_config","install_recovery_keys","install_grub_policy","install_efi_boot_order",
                "rebuild_initramfs","decouple_optional_boot_work","install_ceph_policy","retire_stale_snap_units",
                "install_node_layout","heal_time_sync",
            )
            patches = [mock.patch.object(MODULE,name,side_effect=lambda *args,_name=name,**_kwargs: calls.append((_name,args)) or []) for name in names]
            patches.append(mock.patch.object(MODULE,"install_assets",side_effect=lambda assets: calls.append(("install_assets",tuple(sorted(assets))))))
            patches.append(mock.patch.object(MODULE,"enable_policy_services",side_effect=lambda value,node_id: calls.append(("enable_policy_services",(value,node_id)))))
            patches.append(mock.patch.object(MODULE,"run",side_effect=lambda argv,**_kwargs: calls.append(("run",tuple(argv)))))
            patches.append(mock.patch.object(MODULE,"command",return_value="0"))
            patches.append(mock.patch.object(MODULE,"ceph_fsids",return_value=fsids))
            patches.append(mock.patch.object(MODULE.os,"geteuid",return_value=0))
            ruleset = Path(directory) / "spark-firewall.nft"
            ruleset.write_text("table inet spark_guard {}\n")
            patches.append(mock.patch.object(MODULE,"FIREWALL_RULESET",ruleset))
            patches.append(mock.patch.object(MODULE,"reset_vanished_failed_units",return_value=[]))
            for patch in patches:
                patch.start()
            try:
                result = MODULE.remote_apply(payload)
            finally:
                for patch in patches:
                    patch.stop()
            self.assertEqual(result["profile"],"staged")
            return(calls)

    def test_staged_apply_installs_only_network_independent_policy(self) -> None:
        calls = self.staged_apply([])
        names = [name for name,_args in calls]
        for absent in ("install_efi_boot_order","install_ceph_policy","heal_time_sync"):
            self.assertNotIn(absent,names)
        self.assertNotIn(("run",(MODULE.NETWORK_MANAGER_RUNTIME,"--network-manager")),calls)
        self.assertEqual([args for name,args in calls if name == "install_assets"],[tuple(sorted(MODULE.STAGED_ASSET_DESTINATIONS))])
        self.assertIn(("install_packages",("socks5h://127.0.0.1:1080",)),calls)
        self.assertIn(("enable_policy_services",(None,"spark11")),calls)
        self.assertIn(("run",(MODULE.DMA_GUARD_RUNTIME,"install","--notify",MODULE.HUB_NOTIFY)),calls)
        self.assertLess(names.index("enable_policy_services"),calls.index(("run",(MODULE.DMA_GUARD_RUNTIME,"install","--notify",MODULE.HUB_NOTIFY))))

    def test_staged_apply_refuses_a_node_that_holds_a_ceph_cluster(self) -> None:
        with self.assertRaises(MODULE.BrickproofError):
            self.staged_apply([MODULE.CEPH_CLUSTER_FSID])

    def test_switched_fabric_network_manager_mode_writes_the_fleet_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            conf = Path(directory) / "conf.d"
            backups = Path(directory) / "backups"
            conf.mkdir()
            (conf / "ds4-200g-unmanaged.conf").write_text("[keyfile]\nunmanaged-devices=interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1\n")
            (conf / "99-ds4-ring-unmanaged.conf").write_text("[keyfile]\nunmanaged-devices=interface-name:enp1s0f0np0\n")
            harness = (fabric_functions().replace("/etc/NetworkManager/conf.d",str(conf)).replace("/var/backups/ds4-fleet",str(backups))
                       + f'\nnmcli() {{ echo "NMCLI $*" >> "{directory}/nmcli.log"; }}\nfabric_devices() {{ printf "%s\\n" enp1s0f0np0; }}\nconfigure_unmanaged_fabric keep-addresses\n')
            result = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(sorted(path.name for path in conf.iterdir()),["90-ds4-switched-fabric-unmanaged.conf","95-ds4-uplink-no-assume.conf"])
            merged = "\n".join((conf / name).read_text() for name in sorted(os.listdir(conf)))
            self.assertEqual(set(MODULE.FABRIC_INTERFACES) - MODULE.unmanaged_interfaces(merged),set())
            self.assertTrue(MODULE.uplink_assume_disabled(merged))
            self.assertEqual(len(list(backups.rglob("ds4-200g-unmanaged.conf"))),1)
            calls = (Path(directory) / "nmcli.log").read_text()
            self.assertIn("NMCLI device set enp1s0f0np0 managed no",calls)
            self.assertNotIn("disconnect",calls)
        script = FABRIC.read_text()
        mode = script.split('if [ "${1:-}" = "--network-manager" ]; then',1)[1].split("\nfi\n",1)[0]
        self.assertEqual([line.strip() for line in mode.strip().splitlines()],["configure_unmanaged_fabric keep-addresses","exit 0"])
        install = script.split('if [ "${1:-}" = "--install" ]; then',1)[1].split("\nfi\n",1)[0]
        self.assertIn("    configure_unmanaged_fabric\n",install)

    def test_switched_fabric_boot_run_only_brings_up_and_addresses_the_port(self) -> None:
        harness = (fabric_functions()
                   + '\nip() { echo "IP $*"; }\nethtool() { echo "ETHTOOL $*"; }\n'
                   + 'disable_cx7_hotplug_power_saving() { :; }\nnode_rank() { echo 3; }\nselect_fabric_device() { echo enp1s0f1np1; }\n'
                   + 'apply_switched_fabric\n')
        result = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        calls = [line for line in result.stdout.splitlines() if line.startswith(("IP ","ETHTOOL "))]
        self.assertEqual(calls,[
            "IP link set dev enp1s0f1np1 up",
            "ETHTOOL -G enp1s0f1np1 rx 1024 tx 1024",
            "ETHTOOL -K enp1s0f1np1 tx-tcp-mangleid-segmentation off",
            "IP link set dev enp1s0f1np1 mtu 9000",
            "IP address replace 10.10.100.13/24 dev enp1s0f1np1"])

    def test_node_layout_installs_every_missing_root_owned_by_the_node_user(self) -> None:
        self.assertEqual(MODULE.node_layout_roots((ROOT / "tools" / "spark_node_layout.json").read_text()),["extnvme","kvcache","sparkdata","srcdata"])
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "spark3" / "kvcache").mkdir(parents=True)
            (home / "spark3" / "kvcache" / "keep").write_text("x")
            owner = (home / "spark3").stat()
            created = MODULE.install_node_layout("spark3",["extnvme","kvcache","sparkdata","srcdata"],home)
            self.assertEqual(created,[str(home / "spark3" / name) for name in ("extnvme","sparkdata","srcdata")])
            for name in ("extnvme","sparkdata","srcdata"):
                info = (home / "spark3" / name).stat()
                self.assertEqual((info.st_mode & 0o777,info.st_uid,info.st_gid),(0o775,owner.st_uid,owner.st_gid))
            self.assertEqual((home / "spark3" / "kvcache" / "keep").read_text(),"x")
            self.assertEqual(MODULE.install_node_layout("spark3",["extnvme","kvcache","sparkdata","srcdata"],home),[])
            for roots in ([],["../etc"],["a/b"],"extnvme"):
                with self.subTest(roots=roots),self.assertRaises(MODULE.BrickproofError):
                    MODULE.install_node_layout("spark3",roots,home)

    def test_time_sync_follows_only_the_hub_and_apply_heals_a_fallback(self) -> None:
        config = (ROOT / "tools" / "devcycle" / "fleet" / "timesyncd-sparkpipe-hub.conf").read_text()
        servers = [line.split("=",1)[1].split() for line in config.splitlines() if line.startswith("NTP=")]
        self.assertEqual(servers,[list(MODULE.NTP_SERVERS)])
        for server,restarted in (("ntp.ubuntu.com",True),("",True),("192.168.50.4",False),("10.10.250.2",False)):
            with self.subTest(server=server),mock.patch.object(MODULE,"command",return_value=server),mock.patch.object(MODULE,"run") as run:
                self.assertEqual(MODULE.heal_time_sync(),restarted)
                self.assertEqual(run.call_args_list,[mock.call(["systemctl","restart","systemd-timesyncd.service"])] if restarted else [])

    def test_emergency_probe_falls_back_to_the_node_lan_address(self) -> None:
        targets = []
        def attempt(argv: list[str],**_kwargs: object) -> object:
            targets.append(argv[-2])
            return(mock.Mock(returncode=0 if argv[-2] == "root@192.168.50.143" else 255,stdout=b"",stderr=b"timed out"))
        with mock.patch.object(MODULE,"run",side_effect=attempt):
            self.assertEqual(MODULE.emergency_probe("sparkf",Path("/k")),"ok")
        self.assertEqual(targets,["root@sparkf","root@sparkf.local","root@192.168.50.143"])
        targets.clear()
        with mock.patch.object(MODULE,"run",side_effect=lambda argv,**_k: (targets.append(argv[-2]),mock.Mock(returncode=255,stdout=b"",stderr=b"refused"))[1]):
            self.assertEqual(MODULE.emergency_probe("spark3",Path("/k")),"refused")
        self.assertEqual(targets,["root@spark3","root@spark3.local","root@192.168.50.131"])

if __name__ == "__main__":
    unittest.main()
