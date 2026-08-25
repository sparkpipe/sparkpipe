import importlib.util
import base64
from pathlib import Path
import struct
import tempfile
import unittest


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
        for name in ("ds4-switched-fabric","ds4-direct-pair-fabric"):
            service = (ROOT / "tools" / "devcycle" / "fleet" / f"{name}.service").read_text()
            timer = (ROOT / "tools" / "devcycle" / "fleet" / f"{name}.timer").read_text()
            self.assertIn("TimeoutStartSec=60",service)
            self.assertNotIn("Before=network-online.target",service)
            self.assertNotIn("WantedBy=multi-user.target",service)
            self.assertIn("WantedBy=timers.target",timer)

    def test_optional_ceph_requires_explicit_selection(self) -> None:
        service = (ROOT / "tools" / "devcycle" / "fleet" / "ds4-optional-storage.service").read_text()
        self.assertIn("ConditionPathExists=/etc/sparkpipe/enable-ceph-warm-storage",service)

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
        self.assertIn('run(["update-initramfs",mode,"-k",release],timeout=600)',source)
        self.assertIn('line.endswith("/.ssh/authorized_keys")',source)

    def test_deployment_bytes_are_loaded_from_the_recorded_git_ref(self) -> None:
        source = SCRIPT.read_text()
        self.assertNotIn("LOCAL_ASSET_SOURCES",source)
        self.assertIn('remote_script = git_show(repo,ref,SELF_SOURCE)',source)
        self.assertIn('controller does not match {ref}:{SELF_SOURCE}',source)
        self.assertIn('script_data = str(staged_payload.pop("remote_script")).encode("utf-8")',source)

    def test_kernel_image_and_modules_are_required_packages(self) -> None:
        self.assertEqual(MODULE.required_packages("6.17.0-test"),(
            "earlyoom",
            "dropbear-initramfs",
            "efibootmgr",
            "linux-image-6.17.0-test",
            "linux-modules-6.17.0-test",
        ))

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

    def test_fleet_audit_pins_physical_interfaces(self) -> None:
        self.assertEqual(MODULE.SWITCHED_INTERFACE,"enp1s0f1np1")
        self.assertEqual(MODULE.SWITCHED_SPEED_MBIT,"100000")
        self.assertEqual(MODULE.DIRECT_INTERFACE,"enp1s0f0np0")
        self.assertEqual(MODULE.DIRECT_SPEED_MBIT,"200000")
        source = SCRIPT.read_text()
        self.assertIn('"nvidia-smi","--query-gpu=name,temperature.gpu"',source)
        self.assertIn('["tailscale","status","--json"]',source)
        self.assertIn('extnvme-not-mounted-rw',source)

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
