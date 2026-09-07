from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROXY_SCRIPT = ROOT / "tools" / "spark_ssh_proxy.py"
INSTALL_SCRIPT = ROOT / "tools" / "install_spark_ssh_failover.py"
PROFILE = ROOT / "deployment" / "rtx5090_speculation" / "spark_ssh_failover.example.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name,path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return(module)


proxy = load_module("spark_ssh_proxy",PROXY_SCRIPT)
installer = load_module("install_spark_ssh_failover",INSTALL_SCRIPT)


class SparkSshFailoverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = proxy.load_profile(PROFILE)
        self.node = self.profile.nodes["spark2"]

    def test_profile_covers_all_sixteen_nodes(self) -> None:
        expected = {
            "spark0","spark1","spark2","spark3","spark4","spark5","spark6",
            "spark7","spark8","spark9","sparka","sparkb","sparkc","sparkd",
            "sparke","sparkf",
        }
        self.assertEqual(set(self.profile.nodes),expected)
        self.assertEqual(self.profile.ring_bastions,("spark8","spark9"))
        self.assertEqual(self.profile.wifi_bastions,("spark8","spark9"))

    def test_profile_rejects_duplicate_node(self) -> None:
        value = json.loads(PROFILE.read_text(encoding="utf-8"))
        value["nodes"].append(dict(value["nodes"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(value),encoding="utf-8")
            with self.assertRaises(proxy.RouteFailure):
                proxy.load_profile(path)

    def test_auto_uses_direct_channel_first(self) -> None:
        direct = proxy.OpenChannel("10g","direct",b"SSH-2.0-test\r\n")
        with mock.patch.object(proxy,"_open_tcp_channel",return_value=direct) as opened:
            with mock.patch.object(proxy,"_open_process_channel") as process_opened:
                channel,attempts = proxy.open_channel(self.profile,self.node,"auto",22)
        self.assertIs(channel,direct)
        self.assertEqual(attempts,[])
        opened.assert_called_once_with(self.profile,self.node,22)
        process_opened.assert_not_called()

    def test_sparkf_skips_removed_management_link(self) -> None:
        node = self.profile.nodes["sparkf"]
        ring = proxy.OpenChannel("200g","spark8",b"SSH-2.0-test\r\n")
        with mock.patch.object(proxy,"_open_tcp_channel") as direct:
            with mock.patch.object(proxy,"_open_process_channel",return_value=ring) as opened:
                channel,attempts = proxy.open_channel(self.profile,node,"auto",22)
        self.assertIs(channel,ring)
        self.assertEqual(attempts,[])
        direct.assert_not_called()
        self.assertIn("198.51.100.25:22",opened.call_args.args[0])

    def test_auto_uses_ring_after_direct_failure(self) -> None:
        ring = proxy.OpenChannel("200g","spark8",b"SSH-2.0-test\r\n")
        with mock.patch.object(
            proxy,
            "_open_tcp_channel",
            side_effect=proxy.RouteFailure("direct down"),
        ):
            with mock.patch.object(proxy,"_open_process_channel",return_value=ring) as opened:
                channel,attempts = proxy.open_channel(self.profile,self.node,"auto",22)
        self.assertIs(channel,ring)
        self.assertEqual(attempts,["direct down"])
        command = opened.call_args.args[0]
        self.assertIn("-W",command)
        self.assertEqual(command[-1],"spark8-10g")
        self.assertIn("198.51.100.12:22",command)

    def test_auto_uses_direct_wifi_after_both_ring_bastions_fail(self) -> None:
        wifi = proxy.OpenChannel("wifi","wifi",b"SSH-2.0-test\r\n")
        commands = []

        def open_process(command,route,endpoint,timeout):
            commands.append(command)
            if "-W" in command:
                raise proxy.RouteFailure("ring down")
            return(wifi)

        with mock.patch.object(
            proxy,
            "_open_tcp_channel",
            side_effect=proxy.RouteFailure("direct down"),
        ):
            with mock.patch.object(proxy,"_open_process_channel",side_effect=open_process):
                channel,attempts = proxy.open_channel(self.profile,self.node,"auto",22)
        self.assertIs(channel,wifi)
        self.assertEqual(len(attempts),3)
        self.assertEqual(commands[-1][-5:],[f"{self.profile.emergency.user}@2001:db8:2::2","exec","/usr/bin/nc","127.0.0.1","22"])
        self.assertIn("-6",commands[-1])

    def test_wifi_can_reach_non_wifi_node_through_wifi_bastion(self) -> None:
        node = self.profile.nodes["spark0"]
        wifi = proxy.OpenChannel("wifi","wifi-bastion",b"SSH-2.0-test\r\n")
        commands = []

        def open_process(command,route,endpoint,timeout):
            commands.append(command)
            return(wifi)

        with mock.patch.object(proxy,"_open_process_channel",side_effect=open_process):
            channel,attempts = proxy.open_channel(self.profile,node,"wifi",22)
        self.assertIs(channel,wifi)
        self.assertEqual(attempts,[])
        self.assertEqual(commands[-1][-2:],["198.51.100.10","22"])
        self.assertIn("recovery@spark8-wifi.example.invalid",commands[-1])

    def test_ssh_banner_reader_preserves_banner(self) -> None:
        reader,writer = socket.socketpair()
        thread = threading.Thread(target=lambda: writer.sendall(b"notice\r\nSSH-2.0-test\r\n"))
        thread.start()
        try:
            self.assertEqual(
                proxy._read_ssh_preface(reader.fileno(),1.0),
                b"notice\r\nSSH-2.0-test\r\n",
            )
        finally:
            reader.close()
            writer.close()
            thread.join()

    def test_generated_config_has_transparent_and_forced_routes(self) -> None:
        text = installer.generate_config(
            self.profile,
            Path("/opt/ds4/proxy"),
            Path("/opt/ds4/profile.json"),
        )
        self.assertIn("Host spark2\n",text)
        self.assertIn("--node %n --port %p --route auto",text)
        self.assertIn("Host spark2-10g\n",text)
        self.assertIn("Host spark2-200g\n",text)
        self.assertIn("--node spark2 --port %p --route 200g",text)
        self.assertIn("Host spark2-wifi\n",text)
        self.assertIn("Host spark2-emergency\n",text)
        self.assertIn("Host sparkf\n",text)
        sparkf_10g = text.split("Host sparkf-10g\n",1)[1].split("\n\n",1)[0]
        self.assertIn("--node sparkf --port %p --route auto",sparkf_10g)

    def test_managed_block_replacement_preserves_unrelated_config(self) -> None:
        old = "\n".join([
            "Host before",
            "    HostName before",
            installer.BEGIN,
            "old",
            installer.END,
            "Host after",
            "    HostName after",
            "",
        ])
        updated = installer.replace_managed_block(old,"NEW\n")
        self.assertEqual(updated.count(installer.BEGIN),0)
        self.assertIn("Host before",updated)
        self.assertIn("NEW\nHost after",updated)

    def test_dry_run_does_not_write_install_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_root = root / "install"
            ssh_config = root / ".ssh" / "config"
            stdout = io.StringIO()
            with mock.patch("sys.stdout",stdout):
                code = installer.main([
                    "--profile",str(PROFILE),
                    "--install-root",str(install_root),
                    "--ssh-config",str(ssh_config),
                    "--dry-run",
                ])
            self.assertEqual(code,0)
            self.assertFalse(install_root.exists())
            self.assertFalse(ssh_config.exists())
            self.assertIn("Host spark0",stdout.getvalue())

    def test_install_writes_parseable_config_and_persistent_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = json.loads(PROFILE.read_text(encoding="utf-8"))
            value["route_log"] = str(root / "state" / "routes.log")
            value["spark_known_hosts"] = str(root / ".ssh" / "spark_known_hosts")
            value["emergency"]["known_hosts"] = str(root / ".ssh" / "emergency_known_hosts")
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(value),encoding="utf-8")
            install_root = root / "install"
            ssh_config = root / ".ssh" / "config"
            stdout = io.StringIO()
            with mock.patch("sys.stdout",stdout):
                code = installer.main([
                    "--profile",str(profile_path),
                    "--install-root",str(install_root),
                    "--ssh-config",str(ssh_config),
                ])
            self.assertEqual(code,0)
            self.assertTrue((install_root / "spark_ssh_proxy.py").is_file())
            self.assertTrue((install_root / "spark_ssh_failover.json").is_file())
            self.assertTrue(ssh_config.is_file())
            self.assertIn("Host sparkc-wifi",ssh_config.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
