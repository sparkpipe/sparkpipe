import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "devcycle" / "ds4_spark_brickproof.py"
SPEC = importlib.util.spec_from_file_location("ds4_spark_brickproof_audit",SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

FSID = MODULE.CEPH_CLUSTER_FSID
PUBLIC_KEY = (ROOT / "tools" / "devcycle" / "sparkpipe_fleet_root.pub").read_text().strip()
FLEET_NETWORK_MANAGER = """[keyfile]
unmanaged-devices=interface-name:enp1s0f0np0;interface-name:enp1s0f1np1;interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1
[device-ds4-uplink-wired]
match-device=interface-name:enP7s7
keep-configuration=no
"""
FACTORY_NETWORK_MANAGER = """[keyfile]
unmanaged-devices=interface-name:enP2p1s0f0np0;interface-name:enP2p1s0f1np1
"""
EFI = """BootCurrent: 0004
BootOrder: 0002,0004
Boot0002* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller\tPcieRoot(0x7)
Boot0004* ubuntu\tHD(1,GPT,x)/File(\\EFI\\ubuntu\\shimaa64.efi)
"""


class Node:
    def __init__(self,node: str = "spark3") -> None:
        self.node = node
        self.rank = MODULE.spark_rank(node)
        self.commands = {
            "sysctl -n vm.swappiness":"10","sysctl -n kernel.panic":"60","sysctl -n kernel.softlockup_panic":"1",
            "sysctl -n kernel.hung_task_panic":"1","sysctl -n net.core.default_qdisc":"fq_codel",
            "pgrep -af earlyoom":"812 /usr/bin/earlyoom -m 3 -s 100 -r 5 --avoid sshd|systemd|init",
            "modinfo -F filename sch_fq_codel":"/lib/modules/x/sch_fq_codel.ko","readlink -f /boot/initrd.img":"/boot/initrd.img-x",
            "efibootmgr -v":EFI,"ss -ltn":"LISTEN 0 128 0.0.0.0:2222 0.0.0.0:*","findmnt -n -o OPTIONS /":"rw,relatime",
            "hostname -s":node,"uname -r":"6.17.0-1026-nvidia","systemctl is-system-running":"running",
            "systemctl --failed --no-legend --plain":"","systemctl get-default":"multi-user.target",
            "journalctl -b -u spark-dma-guard.service -o cat --no-pager":f"SPARK-DMA-GUARD host={node} event=start mode=armed",
            "NetworkManager --print-config":FLEET_NETWORK_MANAGER,
            "nmcli -t -f NAME,DEVICE connection show --active":"ds4-uplink-wired:enP7s7",
            "ip -4 route show default":"default via 192.168.50.1 dev enP7s7 proto static metric 10",
            "timedatectl show-timesync -p ServerName --value":"192.168.50.4","timedatectl show -p NTPSynchronized --value":"yes",
            "systemctl show -p RuntimeWatchdogUSec --value":"10min","systemctl show -p RebootWatchdogUSec --value":"5min",
            f"loginctl show-user {node} -p Linger --value":"yes",
        }
        self.unit_states = {unit:"masked" for unit in MODULE.OPTIONAL_BOOT_UNITS}
        self.unit_states.update({unit:"enabled" for unit in (
            "serial-getty@ttyS0.service","sparkpipe-fsck-health.timer","ds4-switched-fabric.timer","ds4-direct-pair-fabric.timer",
            "ssh.service","spark-dma-guard.timer","spark-osd-guard-scan.timer","restart-resolved.service","ds4-ceph-warm-mount.service",
            "sparkpipe-hub-route.service","spark-firewall.service","sparkpipe-roce-qos.service",f"ceph-{FSID}@osd.3.service",
        )})
        self.unit_states.update({"fstrim.service":"static","fstrim.timer":"enabled","ds4-switched-fabric.service":"static","ds4-direct-pair-fabric.service":"static","sparkpipe-fsck-health.service":"static"})
        self.memtotal = 127600808
        self.drift: list[str] = []
        self.enabled_instances = ["osd.3"]
        self.missing_packages: set[str] = set()
        self.firewall = True

    def command(self,argv: list[str],timeout: int = 120,check: bool = True) -> str:
        key = " ".join(argv)
        if argv[0] == "lsinitramfs":
            return("root/.ssh/authorized_keys")
        return(self.commands.get(key,""))

    def run(self,argv: list[str],**_kwargs: object) -> object:
        return(mock.Mock(returncode=0,stdout=b"",stderr=b""))

    def service_value(self,unit: str,name: str) -> str:
        if name == "ActiveState":
            return("active")
        if unit == "user-1000.slice":
            return({"MemoryHigh":str(100 * 1024**3),"MemoryMax":str(108 * 1024**3),"MemorySwapMax":"0"}[name])
        if name == "TimeoutStartUSec":
            return("15s")
        return("")

    def read_optional(self,path: Path) -> str:
        text = str(path)
        if text == "/etc/modules-load.d/90-ds4-network.conf":
            return(MODULE.KERNEL_MODULES_LOAD)
        if text in ("/root/.ssh/authorized_keys","/etc/dropbear/initramfs/authorized_keys"):
            return(PUBLIC_KEY + "\n")
        if text == "/etc/default/grub":
            token = MODULE.recovery_ip_token(MODULE.recovery_network_for_node(self.node))
            return(f'GRUB_DEFAULT=0\nGRUB_CMDLINE_LINUX_DEFAULT="quiet console=tty0 console=ttyS0,921600 {token} fsck.mode=skip fsck.repair=no"\n')
        if text == "/etc/hosts":
            return(MODULE.canonical_hosts("127.0.0.1\tlocalhost\n",{self.node}))
        if text == "/proc/meminfo":
            return(f"MemTotal:       {self.memtotal} kB\n")
        if text == "/etc/ds4-node-rank":
            return(f"{self.rank}\n")
        if text.startswith("/etc/systemd/system/ds4-"):
            return("[Service]\nTimeoutStartSec=60\n")
        return("")

    def audit(self) -> set[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            payload.write_text(json.dumps({
                "assets":{"/a":{"text":"a\n","mode":0o644}},
                "linked_rank_assets":{"/route":{"text":"r\n","mode":0o644}},
                "fleet_public_key":PUBLIC_KEY,
                "recovery_network":MODULE.recovery_network_for_node(self.node),
                "source_commit":"abc",
            }))
            logs = root / "log"
            (logs / FSID).mkdir(parents=True)
            ruleset = root / "spark-firewall.nft"
            if self.firewall:
                ruleset.write_text("table inet spark_guard {}\n")
            states = {unit:"masked" for unit in MODULE.CEPH_MASK_UNITS}
            states.update({"ceph.target":"enabled",f"ceph-{FSID}.target":"enabled"})
            patches = (
                mock.patch.object(MODULE,"command",side_effect=self.command),
                mock.patch.object(MODULE,"run",side_effect=self.run),
                mock.patch.object(MODULE,"service_value",side_effect=self.service_value),
                mock.patch.object(MODULE,"is_enabled",side_effect=lambda unit: self.unit_states.get(unit,"disabled")),
                mock.patch.object(MODULE,"read_optional",side_effect=self.read_optional),
                mock.patch.object(MODULE,"package_installed",side_effect=lambda name: name not in self.missing_packages),
                mock.patch.object(MODULE,"package_verification_lines",return_value=()),
                mock.patch.object(MODULE,"boot_kernel_release",return_value="6.17.0-1026-nvidia"),
                mock.patch.object(MODULE,"audit_fleet_hardware",return_value=([],{})),
                mock.patch.object(MODULE,"ceph_unit_file_states",return_value=states),
                mock.patch.object(MODULE,"ceph_fsids",return_value=[FSID]),
                mock.patch.object(MODULE,"ceph_daemons",return_value=["osd.3"]),
                mock.patch.object(MODULE,"ceph_enabled_instances",return_value=self.enabled_instances),
                mock.patch.object(MODULE,"asset_drift",side_effect=lambda assets: [path for path in self.drift if path in assets]),
                mock.patch.object(MODULE,"stale_snap_units",return_value=[]),
                mock.patch.object(MODULE,"retired_paths",return_value=[root / "absent"]),
                mock.patch.object(MODULE,"CEPH_LOG_ROOT",logs),
                mock.patch.object(MODULE,"FIREWALL_RULESET",ruleset),
            )
            for patch in patches:
                patch.start()
            try:
                return(set(MODULE.remote_audit(payload)["failures"]))
            finally:
                for patch in patches:
                    patch.stop()


class RemoteAuditTest(unittest.TestCase):
    def assert_adds(self,change,expected: set[str],node: str = "spark3") -> None:
        healthy = Node(node).audit()
        broken = Node(node)
        change(broken)
        self.assertEqual(broken.audit() - healthy,expected)

    def test_each_baseline_rule_reports_exactly_its_own_failure(self) -> None:
        cases = (
            (lambda n: n.commands.__setitem__("systemctl get-default","graphical.target"),{"default-target:graphical.target"}),
            (lambda n: n.commands.__setitem__("journalctl -b -u spark-dma-guard.service -o cat --no-pager","event=start mode=watch-only"),{"dma-guard:active:watch-only"}),
            (lambda n: n.commands.__setitem__("NetworkManager --print-config",FACTORY_NETWORK_MANAGER),{"fabric-managed:enp1s0f0np0,enp1s0f1np1","uplink-assume-enabled"}),
            (lambda n: n.commands.__setitem__("ip -4 route show default","default via 192.168.50.1 dev enP7s7 proto static metric 100"),{"uplink-default-route"}),
            (lambda n: n.commands.__setitem__("nmcli -t -f NAME,DEVICE connection show --active","netplan-enP7s7:enP7s7"),{"uplink-profile-inactive"}),
            (lambda n: setattr(n,"memtotal",125505916),{"firmware-carveout:125505916"}),
            (lambda n: n.commands.__setitem__("timedatectl show -p NTPSynchronized --value","no"),{"ntp:192.168.50.4:no"}),
            (lambda n: n.commands.__setitem__("loginctl show-user spark3 -p Linger --value","no"),{"linger:no"}),
            (lambda n: n.commands.__setitem__("systemctl show -p RuntimeWatchdogUSec --value","0"),{"systemd-watchdog:0:5min"}),
            (lambda n: n.commands.__setitem__("sysctl -n vm.swappiness","60"),{"sysctl:vm.swappiness=60"}),
            (lambda n: n.missing_packages.add("iperf3"),{"missing-package:iperf3"}),
            (lambda n: setattr(n,"drift",["/a"]),{"asset-drift:/a"}),
            (lambda n: setattr(n,"drift",["/route"]),{"asset-drift:/route"}),
            (lambda n: setattr(n,"enabled_instances",["osd.3","crash"]),{f"ceph-stray-unit:ceph-{FSID}@crash.service"}),
            (lambda n: n.unit_states.__setitem__("plymouth-quit-wait.service","static"),{"optional-boot-unit:plymouth-quit-wait.service=static"}),
            (lambda n: n.unit_states.__setitem__("sparkpipe-roce-qos.service","disabled"),{"sparkpipe-roce-qos.service:disabled:active"}),
        )
        for change,expected in cases:
            with self.subTest(expected=sorted(expected)):
                self.assert_adds(change,expected)

    def test_a_missing_firewall_ruleset_is_reported(self) -> None:
        healthy = Node().audit()
        broken = Node()
        broken.firewall = False
        added = broken.audit() - healthy
        self.assertEqual(len(added),1)
        self.assertTrue(next(iter(added)).startswith("firewall-ruleset-missing:"))

    def test_ntp_accepts_either_hub_address(self) -> None:
        self.assert_adds(lambda n: n.commands.__setitem__("timedatectl show-timesync -p ServerName --value","10.10.250.2"),set())
        self.assert_adds(lambda n: n.commands.__setitem__("timedatectl show-timesync -p ServerName --value","ntp.ubuntu.com"),{"ntp:ntp.ubuntu.com:yes"})

    def test_hub_gateway_rank_skips_the_wired_uplink_and_the_hub_route(self) -> None:
        def detach(node: Node) -> None:
            node.commands["ip -4 route show default"] = "default via 192.168.50.1 dev wlP9s9 proto static metric 100"
            node.commands["nmcli -t -f NAME,DEVICE connection show --active"] = "ds4-speculation-link:enP7s7"
            node.drift = ["/route"]
            node.unit_states["sparkpipe-hub-route.service"] = "not-found"
        self.assert_adds(detach,set(),node="sparkf")

    def test_guard_mode_falls_back_to_the_state_file_after_journal_vacuum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "boots.json"
            state.write_text(json.dumps({"boots":[{"id":"old","armed":False},{"id":"current","armed":True}]}))
            with mock.patch.object(MODULE,"command",return_value=""), \
                    mock.patch.object(MODULE,"read_optional",return_value="current\n"):
                self.assertEqual(MODULE.dma_guard_mode(state),"armed")
            state.write_text(json.dumps({"boots":[{"id":"current","armed":False}]}))
            with mock.patch.object(MODULE,"command",return_value=""), \
                    mock.patch.object(MODULE,"read_optional",return_value="current\n"):
                self.assertEqual(MODULE.dma_guard_mode(state),"")


class ControllerTest(unittest.TestCase):
    def arguments(self,directory: str,action: str,nodes: str) -> object:
        identity = Path(directory) / "identity"
        identity.write_text("key\n")
        return(mock.Mock(action=action,nodes=nodes,canary="spark0",jobs=1,sparkpipe_repo=Path(directory),source_ref="HEAD",recovery_identity=identity))

    def test_apply_stops_after_the_first_failure_and_keeps_a_receipt(self) -> None:
        applied = []

        def apply_one(host: str,_payload: object,_identity: object) -> dict[str,object]:
            applied.append(host)
            if host == "spark1":
                raise MODULE.BrickproofError("spark1 failed post-apply audit")
            return({"node":host,"failures":[]})

        with tempfile.TemporaryDirectory() as directory:
            receipts = []
            with mock.patch.object(MODULE,"build_payload",return_value={}), \
                    mock.patch.object(MODULE,"apply_one",side_effect=apply_one), \
                    mock.patch.object(MODULE,"write_receipt",side_effect=lambda action,documents: receipts.append(documents) or Path(directory) / "r.json"):
                rc = MODULE.controller_main(self.arguments(directory,"apply","spark0,spark1,spark2,spark3"))
        self.assertEqual(rc,1)
        self.assertEqual(applied[:2],["spark0","spark1"])
        self.assertNotIn("spark3",applied)
        nodes = {item["node"]:item["failures"] for item in receipts[0]}
        self.assertEqual(nodes["spark0"],[])
        self.assertTrue(nodes["spark1"][0].startswith("controller-error:"))
        self.assertTrue(all(failures and failures[0].startswith("skipped") for node,failures in nodes.items() if node not in ("spark0","spark1")))

    def test_audit_records_an_unreachable_node_and_audits_the_rest(self) -> None:
        def remote_action(host: str,_action: str,_payload: object) -> dict[str,object]:
            if host == "spark2":
                raise MODULE.BrickproofError("ssh: connect to host spark2 port 22: timed out")
            return({"node":host,"failures":[]})

        with tempfile.TemporaryDirectory() as directory:
            receipts = []
            with mock.patch.object(MODULE,"build_payload",return_value={}), \
                    mock.patch.object(MODULE,"remote_action",side_effect=remote_action), \
                    mock.patch.object(MODULE,"emergency_probe",return_value="ok"), \
                    mock.patch.object(MODULE,"write_receipt",side_effect=lambda action,documents: receipts.append(documents) or Path(directory) / "r.json"):
                rc = MODULE.controller_main(self.arguments(directory,"audit","spark1,spark2,spark3"))
        self.assertEqual(rc,1)
        nodes = {item["node"]:item["failures"] for item in receipts[0]}
        self.assertEqual(nodes["spark1"],[])
        self.assertEqual(nodes["spark3"],[])
        self.assertTrue(nodes["spark2"][0].startswith("controller-error:"))

    def test_emergency_probe_falls_back_to_the_local_name(self) -> None:
        targets = []

        def run(argv: list[str],**_kwargs: object) -> object:
            targets.append(argv[-2])
            return(mock.Mock(returncode=0 if argv[-2] == "root@sparkf.local" else 255,stdout=b"",stderr=b"refused"))

        with mock.patch.object(MODULE,"run",side_effect=run):
            self.assertEqual(MODULE.emergency_probe("sparkf",Path("/k")),"ok")
        self.assertEqual(targets,["root@sparkf","root@sparkf.local"])

    def test_child_processes_get_no_terminal_input(self) -> None:
        with mock.patch.object(MODULE.subprocess,"run",return_value=mock.Mock(returncode=0,stdout=b"",stderr=b"")) as call:
            MODULE.run(["true"])
            self.assertIs(call.call_args.kwargs["stdin"],MODULE.subprocess.DEVNULL)
            MODULE.run(["tee","x"],input_bytes=b"data")
            self.assertIsNone(call.call_args.kwargs["stdin"])

    def test_route_metric_must_be_exactly_the_uplink_metric(self) -> None:
        self.assertTrue(MODULE.uplink_default_route_present(["default via 192.168.50.1 dev enP7s7 proto static metric 10"]))
        for metric in ("100","101","1000"):
            self.assertFalse(MODULE.uplink_default_route_present([f"default via 192.168.50.1 dev enP7s7 proto static metric {metric}"]))

    def test_vanished_failed_units_are_reset(self) -> None:
        listing = "snap-core22-2438.mount not-found failed failed snap-core22-2438.mount\nceph-x@osd.3.service loaded failed failed Ceph\n"
        with mock.patch.object(MODULE,"command",return_value=listing),mock.patch.object(MODULE,"run") as run:
            self.assertEqual(MODULE.reset_vanished_failed_units(),["snap-core22-2438.mount"])
        run.assert_called_once_with(["systemctl","reset-failed","snap-core22-2438.mount"],check=False)


if __name__ == "__main__":
    unittest.main()
