import importlib.util
import json
import re
import base64
import shlex
import subprocess
import sys
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "devcycle" / "ds4_spark_brickproof.py"
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ds4_spark_brickproof",SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BrickproofTest(unittest.TestCase):
    RECOVERY = {
        "address":"10.20.0.12",
        "gateway":"10.20.0.1",
        "interface":"enP7s7",
        "netmask":"255.255.255.0",
        "node_id":"spark2",
    }

    def test_grub_policy_preserves_recovery_network_and_serial(self) -> None:
        source = 'GRUB_DEFAULT="ds4-fastboot"\nGRUB_CMDLINE_LINUX_DEFAULT="quiet ip=10.20.0.12::10.20.0.1:255.255.255.0:spark2:enP7s7:none console=tty0 console=ttyS0,921600n8"\n'
        result = MODULE.canonical_grub(source,self.RECOVERY)
        self.assertEqual(MODULE.shell_assignment(result,"GRUB_DEFAULT"),"0")
        tokens = MODULE.shell_assignment(result,"GRUB_CMDLINE_LINUX_DEFAULT").split()
        self.assertIn("fsck.mode=skip",tokens)
        self.assertIn("fsck.repair=no",tokens)
        self.assertTrue(any(token.startswith("ip=") for token in tokens))
        self.assertTrue(any(token.startswith("console=ttyS0") for token in tokens))

    def test_grub_policy_repairs_unrecoverable_entry(self) -> None:
        result = MODULE.canonical_grub('GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"\n',self.RECOVERY)
        tokens = MODULE.shell_assignment(result,"GRUB_CMDLINE_LINUX_DEFAULT").split()
        self.assertIn("console=tty0",tokens)
        self.assertIn("console=ttyS0,921600",tokens)
        self.assertIn(MODULE.recovery_ip_token(self.RECOVERY),tokens)

    def test_grub_policy_is_idempotent(self) -> None:
        source = 'GRUB_DEFAULT=0\nGRUB_CMDLINE_LINUX_DEFAULT="ip=dhcp console=ttyS0 fsck.mode=force fsck.repair=yes"\n'
        once = MODULE.canonical_grub(source,self.RECOVERY)
        self.assertEqual(MODULE.canonical_grub(once,self.RECOVERY),once)
        tokens = MODULE.shell_assignment(once,"GRUB_CMDLINE_LINUX_DEFAULT").split()
        self.assertEqual(tokens.count("fsck.mode=skip"),1)
        self.assertEqual(tokens.count("fsck.repair=no"),1)

    def test_recovery_network_uses_canonical_fleet_mapping(self) -> None:
        self.assertEqual(MODULE.recovery_network_for_node("spark8"),{
            "address":"10.20.0.18",
            "gateway":"10.20.0.1",
            "interface":"enP7s7",
            "netmask":"255.255.255.0",
            "node_id":"spark8",
        })

    def test_default_nodes_cover_the_complete_fleet(self) -> None:
        self.assertEqual(MODULE.DEFAULT_NODES,tuple(f"spark{rank:x}" for rank in range(16)))

    def test_node_rank_is_single_hex_suffix(self) -> None:
        self.assertEqual(MODULE.spark_rank("sparkf"),15)
        for invalid in ("spark10","sparkg","node0"):
            with self.assertRaises(MODULE.BrickproofError):
                MODULE.spark_rank(invalid)

    @staticmethod
    def public_key_blob(seed: int) -> str:
        key_type = b"ssh-ed25519"
        payload = struct.pack(">I",len(key_type)) + key_type + bytes([seed]) * 32
        return(base64.b64encode(payload).decode("ascii"))

    def test_public_key_repair_recovers_split_keys_and_deduplicates(self) -> None:
        first = self.public_key_blob(1)
        second = self.public_key_blob(2)
        required = f"ssh-ed25519 {self.public_key_blob(3)} recovery"
        corrupt = f"ssh-ed25519 {first} ceph-anssh-ed25519\n{second}\nceph-b\nssh-ed25519 {first} duplicate\n"
        once = MODULE.canonical_authorized_keys(corrupt,required)
        twice = MODULE.canonical_authorized_keys(once,required)
        self.assertEqual(once,twice)
        self.assertEqual(once.splitlines(),[
            f"ssh-ed25519 {first} ceph-a",
            f"ssh-ed25519 {second}",
            required,
        ])

    def test_public_key_repair_preserves_options_and_ignores_comments(self) -> None:
        restricted_blob = self.public_key_blob(4)
        comment_blob = self.public_key_blob(5)
        required = f"ssh-ed25519 {self.public_key_blob(6)} recovery"
        restricted = f'from="10.20.0.0/24",no-agent-forwarding ssh-ed25519 {restricted_blob} ceph'
        source = f"# ssh-ed25519 {comment_blob} disabled\n{restricted}\n"
        result = MODULE.canonical_authorized_keys(source,required)
        self.assertEqual(result.splitlines(),[
            restricted,
            required,
        ])

    def test_efi_boot_order_moves_pxe_before_ubuntu(self) -> None:
        source = """BootCurrent: 0007
BootOrder: 0007,0005
Boot0005* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller
Boot0007* ubuntu HD(1,GPT,abc)
"""
        self.assertEqual(MODULE.desired_efi_boot_order(source),["0005","0007"])

    def test_efi_boot_order_moves_pxe_before_dgx_os(self) -> None:
        source = """BootCurrent: 0000
BootOrder: 0000,0002,0003
Boot0000* DGX OS HD(1,GPT,abc)
Boot0002* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller
Boot0003* UEFI:CD/DVD Drive
"""
        self.assertEqual(MODULE.desired_efi_boot_order(source),["0002","0000","0003"])

    def test_efi_boot_order_requires_pxe_ipv4(self) -> None:
        source = """BootOrder: 0001,0002
Boot0001* ubuntu HD(1,GPT,abc)
Boot0002* UEFI:CD/DVD Drive
"""
        with self.assertRaises(MODULE.BrickproofError):
            MODULE.desired_efi_boot_order(source)

    def test_efi_boot_order_rejects_ambiguous_linux_entries(self) -> None:
        source = """BootOrder: 0001,0002
Boot0001* ubuntu HD(1,GPT,abc)
Boot0002* ubuntu HD(1,GPT,def)
"""
        with self.assertRaises(MODULE.BrickproofError):
            MODULE.desired_efi_boot_order(source)

    def test_atomic_write_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.conf"
            self.assertTrue(MODULE.atomic_write(path,"value\n",0o600))
            self.assertFalse(MODULE.atomic_write(path,"value\n",0o600))
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_fabric_services_are_bounded_and_not_boot_critical(self) -> None:
        for name in ("ds4-switched-fabric","ds4-rail2-fabric"):
            service = (ROOT / "tools" / "devcycle" / "fleet" / f"{name}.service").read_text()
            timer = (ROOT / "tools" / "devcycle" / "fleet" / f"{name}.timer").read_text()
            self.assertIn("TimeoutStartSec=60",service)
            self.assertNotIn("Before=network-online.target",service)
            self.assertNotIn("WantedBy=multi-user.target",service)
            self.assertIn("WantedBy=timers.target",timer)

    def test_ceph_package_units_are_masked_and_cephadm_targets_are_kept(self) -> None:
        for name in ("ds4-optional-storage.service","ds4-optional-storage.timer"):
            self.assertFalse((ROOT / "tools" / "devcycle" / "fleet" / name).exists())
            self.assertNotIn(f"/etc/systemd/system/{name}",MODULE.ASSET_SOURCES)
        for name in ("ds4-switched-fabric.service.conf","ds4-rail2-fabric.service.conf"):
            self.assertFalse((ROOT / "tools" / "devcycle" / "boot-unblock" / name).exists())
        self.assertNotIn("ceph.target",MODULE.CEPH_MASK_UNITS)
        self.assertIn("ceph-crash.service",MODULE.CEPH_MASK_UNITS)
        self.assertIn("ceph-osd@.service",MODULE.CEPH_MASK_UNITS)
        self.assertNotIn(
            "ceph-b52b3459-74b2-428d-b944-1bb691b263c7@.service",
            MODULE.BOOT_TIMEOUT_SOURCES,
        )

    def test_ceph_policy_keeps_cephadm_daemons_and_disables_units_without_state(self) -> None:
        fsid = "cd8715a1-47b9-4e4a-8d3c-2cddb02077d1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            systemd = root / "systemd"
            state = root / "state"
            logs = root / "log"
            wants = systemd / f"ceph-{fsid}.target.wants"
            wants.mkdir(parents=True)
            (systemd / f"ceph-{fsid}.target").write_text("[Unit]\n")
            for daemon in ("osd.8","mgr.spark8.abc","mds.ds4warm.spark8.x","crash"):
                (state / fsid / daemon).mkdir(parents=True)
            (state / "79e60319-97d7-11f1-b71f-48210b969ab9").mkdir(parents=True)
            for instance in ("osd.8","crash"):
                (wants / f"ceph-{fsid}@{instance}.service").symlink_to(systemd / f"ceph-{fsid}@.service")
            retired = systemd / "ceph-osd-retry.service"
            retired.write_text("[Service]\n")
            commands = []
            disabled = []

            def record(argv: list[str],**_kwargs: object) -> object:
                commands.append(argv)
                return(mock.Mock(returncode=0))

            with mock.patch.object(MODULE,"run",side_effect=record), \
                    mock.patch.object(MODULE,"disable_persistent_unit",side_effect=disabled.append), \
                    mock.patch.object(MODULE,"retired_paths",return_value=[retired]), \
                    mock.patch.object(MODULE.os,"chown") as chown:
                changes = MODULE.install_ceph_policy(systemd,state,logs,root / "marker",root / "backup")
            self.assertIn(f"ceph-{fsid}@crash.service",disabled)
            self.assertNotIn(f"ceph-{fsid}@osd.8.service",disabled)
            enable = [argv for argv in commands if argv[:2] == ["systemctl","enable"]]
            self.assertEqual(enable,[["systemctl","enable","ceph.target",f"ceph-{fsid}.target",f"ceph-{fsid}@mgr.spark8.abc.service",f"ceph-{fsid}@osd.8.service"]])
            self.assertIn(["systemctl","reset-failed","ceph-osd-retry.service",*MODULE.OBSOLETE_CEPH_STARTUP_UNITS],commands)
            self.assertEqual((logs / fsid).stat().st_mode & 0o777,0o770)
            self.assertFalse((logs / "79e60319-97d7-11f1-b71f-48210b969ab9").exists())
            chown.assert_called_once_with(logs / fsid,MODULE.CEPH_UID,MODULE.CEPH_UID)
            for unit in MODULE.CEPH_MASK_UNITS:
                self.assertEqual((systemd / unit).readlink(),Path("/dev/null"))
            self.assertFalse(retired.exists())
            self.assertTrue((root / "backup" / retired.relative_to("/")).exists())
            self.assertIn(f"disabled stray ceph-{fsid}@crash.service",changes)

    def test_ceph_clusters_are_directories_with_a_target_or_daemons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            systemd = root / "systemd"
            state = root / "state"
            systemd.mkdir()
            (state / "79e60319-97d7-11f1-b71f-48210b969ab9").mkdir(parents=True)
            (state / "cd8715a1-47b9-4e4a-8d3c-2cddb02077d1" / "osd.3").mkdir(parents=True)
            (state / "541e9482-0000-0000-0000-000000000000").mkdir()
            (systemd / "ceph-541e9482-0000-0000-0000-000000000000.target").write_text("[Unit]\n")
            (state / "bootstrap-osd").mkdir()
            self.assertEqual(MODULE.ceph_fsids(state,systemd),["541e9482-0000-0000-0000-000000000000","cd8715a1-47b9-4e4a-8d3c-2cddb02077d1"])

    def test_ceph_stray_units_come_from_enablement_links(self) -> None:
        fsid = "cd8715a1-47b9-4e4a-8d3c-2cddb02077d1"
        with tempfile.TemporaryDirectory() as directory:
            systemd = Path(directory)
            wants = systemd / f"ceph-{fsid}.target.wants"
            wants.mkdir()
            for instance in ("osd.3","crash"):
                (wants / f"ceph-{fsid}@{instance}.service").symlink_to(systemd / f"ceph-{fsid}@.service")
            other = systemd / "ceph-541e9482-0000-0000-0000-000000000000.target.wants"
            other.mkdir()
            (other / "ceph-541e9482-0000-0000-0000-000000000000@osd.1.service").symlink_to(systemd / "x")
            enabled = MODULE.ceph_enabled_instances(fsid,systemd)
            self.assertEqual(enabled,["crash","osd.3"])
            self.assertEqual(MODULE.ceph_stray_units(fsid,["osd.3"],enabled),[f"ceph-{fsid}@crash.service"])

    def test_ceph_unit_mask_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = root / "ceph-osd.target"
            unit.write_text("[Unit]\n")
            MODULE.mask_unit_file("ceph-osd.target",root)
            self.assertTrue(unit.is_symlink())
            self.assertEqual(unit.readlink(),Path("/dev/null"))
            MODULE.mask_unit_file("ceph-osd.target",root)
            self.assertEqual(unit.readlink(),Path("/dev/null"))

    def test_remove_path_handles_dropin_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ceph-osd@.service.d"
            path.mkdir()
            (path / "override.conf").write_text("[Service]\n")
            MODULE.remove_path(path)
            self.assertFalse(path.exists())

    def test_ceph_logrotate_compatibility_is_processed_first(self) -> None:
        source = (ROOT / "tools" / "devcycle" / "fleet" / "00-sparkpipe-cephadm").read_text()
        self.assertIn("/var/log/ceph/cephadm.log",source)
        self.assertIn("ignoreduplicates",source)
        self.assertIn("/etc/logrotate.d/00-sparkpipe-cephadm",MODULE.ASSET_SOURCES)

    def test_switched_fabric_runtime_only_applies_local_fabric(self) -> None:
        source = (ROOT / "tools" / "devcycle" / "fleet" / "ds4_switched_fabric_apply.sh").read_text()
        runtime = source.rsplit("\nfi\n",1)[1].strip()
        self.assertEqual(runtime,"apply_switched_fabric")
        install = source.split('if [ "${1:-}" = "--install" ]; then',1)[1].split("\nfi\n",1)[0]
        self.assertIn("configure_management_link",install)
        self.assertIn("retire_legacy_mac_mounts",install)

    def test_apply_restarts_emergency_ssh_and_always_rebuilds_initramfs(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn('run(["systemctl","restart","ssh-emergency.service"])',source)
        self.assertIn('run(["systemctl","reset-failed","user@0.service"],check=False)',source)
        self.assertIn('run(["update-initramfs",mode,"-k",release],timeout=600)',source)
        self.assertIn('line.endswith("/.ssh/authorized_keys")',source)

    def test_deployment_bytes_are_loaded_from_the_recorded_git_ref(self) -> None:
        source = SCRIPT.read_text()
        self.assertNotIn("LOCAL_ASSET_SOURCES",source)
        self.assertIn('remote_script = git_show(repo,ref,SELF_SOURCE)',source)
        self.assertIn('controller does not match {ref}:{SELF_SOURCE}',source)

    def test_remote_action_streams_script_and_payload_over_stdin(self) -> None:
        script = "def remote_audit(payload):\n    return({'node_seen':payload['recovery_network']['node_id'],'value':payload['value']})\n"
        calls = []
        def fake_ssh(host,*argv,input_bytes=None,timeout=120,check=True):
            remote_argv = shlex.split(" ".join(argv))
            calls.append((host,remote_argv))
            result = subprocess.run([sys.executable,*remote_argv[3:]],input=input_bytes,capture_output=True,check=True)
            return(subprocess.CompletedProcess(remote_argv,0,result.stdout,result.stderr))
        with mock.patch.object(MODULE,"ssh",side_effect=fake_ssh):
            document = MODULE.remote_action("spark2","audit",{"remote_script":script,"value":7})
        self.assertEqual(document,{"node":"spark2","node_seen":"spark2","value":7})
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1][:5],["sudo","-n","python3","-I","-c"])
        self.assertEqual(calls[0][1][6:],["audit"])
        self.assertFalse(any("/tmp" in argument for argument in calls[0][1]))
        with mock.patch.object(MODULE,"ssh") as ssh, self.assertRaises(MODULE.BrickproofError):
            MODULE.remote_action("spark2","remove",{"remote_script":script})
        ssh.assert_not_called()

    def test_remote_bootstrap_loads_the_real_script_without_running_its_controller(self) -> None:
        probe = "\ndef remote_probe(payload):\n    return({'module':BrickproofError.__module__,'payload':payload,'path':sys.path[0]})\n"
        request = json.dumps({"payload":{"value":3},"script":SCRIPT.read_text() + probe})
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "json.py").write_text("raise SystemExit('planted json imported')\n")
            result = subprocess.run([sys.executable,"-I","-c",MODULE.REMOTE_BOOTSTRAP,"probe"],input=request.encode("utf-8"),capture_output=True,cwd=directory,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr.decode("utf-8",errors="replace"))
        document = json.loads(result.stdout)
        self.assertEqual(document["module"],"ds4_spark_brickproof")
        self.assertEqual(document["payload"],{"value":3})
        self.assertNotEqual(document["path"],"")

    def test_kernel_image_and_modules_are_required_packages(self) -> None:
        self.assertEqual(MODULE.required_packages("6.17.0-test"),(
            "earlyoom",
            "dropbear-initramfs",
            "efibootmgr",
            *MODULE.FLEET_TOOL_PACKAGES,
            "linux-image-6.17.0-test",
            "linux-modules-6.17.0-test",
        ))
        self.assertTrue({"iperf3","fio","ninja-build","python3.12-dev"} <= set(MODULE.FLEET_TOOL_PACKAGES))

    def test_boot_kernel_release_comes_from_canonical_vmlinuz_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "vmlinuz-6.17.0-test"
            link = root / "vmlinuz"
            image.touch()
            link.symlink_to(image.name)
            self.assertEqual(MODULE.kernel_release_from_vmlinuz(link),"6.17.0-test")

    def test_boot_kernel_release_rejects_ambiguous_image_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "kernel"
            image.touch()
            with self.assertRaises(MODULE.BrickproofError):
                MODULE.kernel_release_from_vmlinuz(image)

    def test_hosts_remove_vendor_and_node_loopback_aliases(self) -> None:
        source = "127.0.0.1 localhost local-api\n127.0.0.1 aitopatom-test spark2\n10.10.100.12 spark2 spark2-fabric\n"
        result = MODULE.canonical_hosts(source,{"aitopatom-test","spark2"})
        self.assertIn("127.0.0.1\tlocalhost local-api\n",result)
        self.assertNotIn("aitopatom-test",result)
        self.assertEqual(result.count("10.10.100.12 spark2 spark2-fabric"),1)
        self.assertIn("10.20.0.12 spark2-mgmt",result)

    def test_hosts_canonicalization_is_idempotent(self) -> None:
        source = "127.0.0.1 localhost localhost.localdomain\n::1 localhost ip6-localhost\n"
        once = MODULE.canonical_hosts(source,{"spark2"})
        self.assertEqual(MODULE.canonical_hosts(once,{"spark2"}),once)

    def test_hosts_replace_both_legacy_alias_blocks(self) -> None:
        source = """127.0.0.1 localhost
# DS4 Spark aliases BEGIN
10.20.0.10 spark0
10.10.100.10 spark0-200g spark0-ring
# DS4 Spark aliases END
# DS4 switched Spark aliases BEGIN
10.10.100.10 spark0 spark0-fabric
10.20.0.10 spark0-mgmt spark0-10g
# DS4 switched Spark aliases END
"""
        result = MODULE.canonical_hosts(source,{"spark0"})
        self.assertNotIn("DS4 Spark aliases",result)
        self.assertNotIn("DS4 switched Spark aliases",result)
        self.assertNotIn("spark0-ring",result)
        self.assertNotIn("spark0-200g",result)
        self.assertNotIn("spark0-10g",result)
        self.assertEqual(result.count(MODULE.HOSTS_BLOCK_BEGIN),1)
        self.assertEqual(result.count("10.10.100.10 spark0 spark0-fabric"),1)
        self.assertEqual(result.count("10.20.0.10 spark0-mgmt"),1)

    def test_hosts_reject_unterminated_managed_block(self) -> None:
        with self.assertRaises(MODULE.BrickproofError):
            MODULE.canonical_hosts("# DS4 Spark aliases BEGIN\n10.20.0.10 spark0\n",{"spark0"})

    def test_network_kernel_modules_have_one_canonical_load_list(self) -> None:
        self.assertEqual(MODULE.REQUIRED_KERNEL_MODULES,("sch_fq_codel",))
        self.assertEqual(MODULE.KERNEL_MODULES_LOAD,"sch_fq_codel\n")

    def test_fleet_audit_rejects_degraded_maintenance(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn('["systemctl","unmask","fstrim.service"]',source)
        self.assertIn('["systemctl","--failed","--no-legend","--plain"]',source)
        self.assertIn('["logrotate","--debug","/etc/logrotate.conf"]',source)

    def test_fabric_netdev_rings_are_sized_for_roce(self) -> None:
        fleet = ROOT / "tools" / "devcycle" / "fleet"
        for name in ("ds4_rail2_fabric_apply.sh","ds4_switched_fabric_apply.sh"):
            with self.subTest(script=name):
                rings = re.findall(r'ethtool -G "\$\{[A-Z_]+\}" rx (\d+) tx (\d+)',(fleet / name).read_text())
                self.assertEqual(rings,[("1024","1024")])
        door = (ROOT / "tools" / "devcycle" / "spark_fabric_door.sh").read_text()
        self.assertIn('rings="${1:-1024}"',door)

    def test_policy_services_enable_rail2_on_every_rank_and_retire_the_pair_units(self) -> None:
        for rank in (0,2,14,15):
            with self.subTest(rank=rank),mock.patch.object(MODULE,"run") as run,mock.patch.object(MODULE,"disable_persistent_unit") as disable:
                MODULE.enable_policy_services(rank)
                calls = [call.args[0] for call in run.call_args_list if call.args]
                self.assertIn(["systemctl","enable","--now","ds4-rail2-fabric.timer"],calls)
                for unit in ("ds4-direct-pair-fabric.timer","ds4-direct-pair-fabric.service"):
                    self.assertIn(["systemctl","disable","--now",unit],calls)
                    self.assertIn(mock.call(unit),disable.call_args_list)
                self.assertFalse([call for call in calls if "enable" in call and any("direct-pair" in part for part in call)])

    def test_retired_paths_cover_every_pair_fabric_file(self) -> None:
        root = Path("/etc/systemd/system")
        retired = set(MODULE.retired_paths(root))
        for path in (root / "ds4-direct-pair-fabric.service",root / "ds4-direct-pair-fabric.timer",
                     root / "ds4-direct-pair-fabric.service.d",root / "ds4-direct-pair-fabric-p2.service",
                     Path("/usr/local/sbin/ds4-direct-pair-fabric-apply")):
            self.assertIn(path,retired)
        self.assertFalse([path for path in MODULE.ASSET_SOURCES if "direct-pair" in path])

    def rail2_harness(self,directory: str,rank: object,addresses: str,queue_pairs: str = "",autoneg: str = "on") -> subprocess.CompletedProcess:
        script = (ROOT / "tools" / "devcycle" / "fleet" / "ds4_rail2_fabric_apply.sh").read_text()
        functions = script.split('\nif [ "$(id -u)" -ne 0 ]; then',1)[0]
        sysfs = Path(directory) / "net"
        for device,rdma in (("rail0","roceRail"),("idle0","roceIdle")):
            (sysfs / device / "device" / "infiniband" / rdma).mkdir(parents=True,exist_ok=True)
        (Path(directory) / "rank").write_text(f"{rank}\n")
        stubs = f"""
ip() {{ echo "IP $*"; if [ "$1 $2" = "-4 -o" ]; then printf '%s' {shlex.quote(addresses)}; fi; }}
ethtool() {{ echo "ETHTOOL $*"; case "$*" in rail0) printf '\\tAuto-negotiation: {autoneg}\\n';; "--show-fec rail0") echo "Supported/Configured FEC encodings: Auto";; esac; }}
sysctl() {{ echo "SYSCTL $*"; }}
rdma() {{ echo "link $4 lqpn 1 type GSI state RTS comm [ib_core]"; printf '%s' {shlex.quote(queue_pairs)}; }}
mlnx_qos() {{ echo "MLNX_QOS $*" >&2; echo "Priority trust state: dscp"; }}
dcb() {{ echo "DCB $*" >&2; echo "prio-pfc 0:off 1:off 2:off 3:on 4:off 5:off 6:on 7:off"; echo "prio-buffer 0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:1"; }}
"""
        harness = (f"export DS4_NODE_RANK_PATH={directory}/rank DS4_SYSFS_NET={sysfs} DS4_RAIL2_DEVICE=rail0 DS4_RAIL2_IDLE_DEVICE=idle0\n"
                   + functions + stubs + "apply_rail2\n")
        return(subprocess.run(["bash","-c",harness],capture_output=True,text=True))

    def test_rail2_apply_sets_the_rank_address_and_one_shared_lossless_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.rail2_harness(directory,"05","x x x 10.10.201.5/31\nx x x 10.10.201.5/24\n")
            self.assertEqual(result.returncode,0,result.stderr)
            out = result.stdout
            self.assertIn("IP -4 addr flush dev idle0 scope global",out)
            self.assertIn("IP link set dev idle0 down",out)
            self.assertIn("IP address replace 10.10.201.5/24 dev rail0",out)
            self.assertIn("IP address del 10.10.201.5/31 dev rail0",out)
            self.assertNotIn("IP address del 10.10.201.5/24",out)
            self.assertIn("ETHTOOL -A rail0 rx off tx off",out)
            self.assertNotIn("autoneg off",out)
            qos = [line for line in result.stderr.splitlines() if line.startswith("MLNX_QOS -i rail0 --")]
            self.assertEqual(qos,["MLNX_QOS -i rail0 --trust dscp","MLNX_QOS -i rail0 --pfc 0,0,0,1,0,0,1,0",
                                  "MLNX_QOS -i rail0 --prio2buffer 1,1,1,1,1,1,1,1","MLNX_QOS -i rail0 --buffer_size 0,459360,0,0,0,0,0,0"])
            self.assertTrue(out.endswith("rail2_fabric rank=5 device=rail0 address=10.10.201.5/24 mtu=9000 idle=idle0 forced_link=no\n"),out)

    def test_rail2_apply_keeps_addresses_that_live_queue_pairs_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.rail2_harness(directory,4,"x x x 10.10.201.4/31\nx x x 10.10.201.4/24\n",
                                        queue_pairs="link roceRail/1 lqpn 534 rqpn 535 type RC state RTS comm sparkpipe_weigh\n")
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn("IP address del",result.stdout)
            self.assertIn("kept 10.10.201.4/31 on rail0; 1 live queue pairs hold its RoCE GID",result.stderr)

    def test_rail2_apply_forces_the_link_only_on_listed_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for rank,autoneg,forced in ((2,"on",True),(2,"off",False),(3,"on",False)):
                with self.subTest(rank=rank,autoneg=autoneg):
                    result = self.rail2_harness(directory,rank,f"x x x 10.10.201.{rank}/24\n",autoneg=autoneg)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual("ETHTOOL -s rail0 autoneg off speed 100000 duplex full" in result.stdout,forced,result.stdout)
                    self.assertEqual("ETHTOOL --set-fec rail0 encoding rs" in result.stdout,rank == 2,result.stdout)
                    self.assertIn(f"forced_link={'yes' if rank == 2 else 'no'}",result.stdout)

    def test_rail2_apply_fails_when_the_lossless_state_does_not_take(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            patched = (ROOT / "tools" / "devcycle" / "fleet" / "ds4_rail2_fabric_apply.sh").read_text().split('\nif [ "$(id -u)" -ne 0 ]; then',1)[0]
            sysfs = Path(directory) / "net"
            (sysfs / "rail0").mkdir(parents=True)
            (Path(directory) / "rank").write_text("6\n")
            harness = (f"set -euo pipefail\nexport DS4_NODE_RANK_PATH={directory}/rank DS4_SYSFS_NET={sysfs} DS4_RAIL2_DEVICE=rail0 DS4_RAIL2_IDLE_DEVICE=idle0\n"
                       + patched + '\nip() { :; }\nethtool() { :; }\nsysctl() { :; }\nmlnx_qos() { echo "Priority trust state: dscp"; }\n'
                       + 'dcb() { echo "prio-pfc 3:on 6:on"; echo "prio-buffer 0:0 1:0 2:0 3:1 4:0 5:0 6:0 7:0"; }\napply_rail2\n')
            result = subprocess.run(["bash","-c",harness],capture_output=True,text=True)
            self.assertEqual(result.returncode,4,result.stdout + result.stderr)
            self.assertIn("lossless state did not take on rail0",result.stderr)

    def test_rail2_audit_reports_each_drift(self) -> None:
        healthy = {
            "ip -4 -o addr show dev enP2p1s0f0np0 scope global":"7: enP2p1s0f0np0    inet 10.10.201.6/24 brd 10.10.201.255 scope global enP2p1s0f0np0",
            "mlnx_qos -i enP2p1s0f0np0":"Priority trust state: dscp",
            "dcb pfc show dev enP2p1s0f0np0 prio-pfc":"prio-pfc 0:off 1:off 2:off 3:on 4:off 5:off 6:on 7:off",
            "dcb buffer show dev enP2p1s0f0np0 prio-buffer":"prio-buffer 0:1 1:1 2:1 3:1 4:1 5:1 6:1 7:1",
        }
        cases = (
            ({},"down",[]),
            ({"ip -4 -o addr show dev enP2p1s0f0np0 scope global":"7: enP2p1s0f0np0    inet 10.10.201.6/31 scope global enP2p1s0f0np0"},"down",["rail2-address:10.10.201.6/31"]),
            ({},"up",["rail2-idle-function:up"]),
            ({"mlnx_qos -i enP2p1s0f0np0":"Priority trust state: pcp"},"down",["rail2-trust"]),
            ({"dcb pfc show dev enP2p1s0f0np0 prio-pfc":"prio-pfc 3:off 6:on"},"down",["rail2-pfc"]),
            ({"dcb pfc show dev enP2p1s0f0np0 prio-pfc":"prio-pfc 3:on 6:off"},"down",["rail2-pfc"]),
            ({"dcb pfc show dev enP2p1s0f0np0 prio-pfc":"prio-pfc 3:on 4:on 6:on"},"down",["rail2-pfc"]),
            ({"dcb buffer show dev enP2p1s0f0np0 prio-buffer":"prio-buffer 0:0 1:0 2:0 3:1 4:0 5:0 6:0 7:0"},"down",["rail2-prio-buffer"]),
        )
        for overrides,idle,expected in cases:
            commands = {**healthy,**overrides}
            with self.subTest(expected=expected),\
                    mock.patch.object(MODULE,"command",side_effect=lambda argv,**_: commands.get(" ".join(argv),"")),\
                    mock.patch.object(MODULE,"read_optional",return_value=f"{idle}\n"):
                self.assertEqual(MODULE.audit_rail2(6)[0],expected)

    def test_fleet_audit_pins_physical_interfaces(self) -> None:
        self.assertEqual(MODULE.SWITCHED_INTERFACE,"enp1s0f1np1")
        self.assertEqual(MODULE.SWITCHED_SPEED_MBIT,"100000")
        self.assertEqual(MODULE.RAIL2_INTERFACE,"enP2p1s0f0np0")
        self.assertEqual(MODULE.RAIL2_IDLE_INTERFACE,"enp1s0f0np0")
        self.assertEqual(MODULE.RAIL2_SPEED_MBIT,"100000")
        source = SCRIPT.read_text()
        self.assertIn('"nvidia-smi","--query-gpu=name,temperature.gpu"',source)
        self.assertIn('["tailscale","status","--json"]',source)
        self.assertIn('layout-root-not-writable:',source)

    def test_persistent_unit_links_only_returns_dependency_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit = root / "ds4-test.service"
            unit.write_text("[Service]\nExecStart=/bin/true\n")
            wants = root / "multi-user.target.wants"
            wants.mkdir()
            link = wants / unit.name
            link.symlink_to(unit)
            self.assertEqual(MODULE.persistent_unit_links(unit.name,root),[link])


if __name__ == "__main__":
    unittest.main()
