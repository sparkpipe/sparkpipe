import os
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT = (ROOT / "tools/fleet_node_agent.sh").read_text()
SCRIPT = (ROOT / "tools/fabric/sparkpipe_roce_qos.sh").read_text()
UNIT = (ROOT / "tools/fabric/sparkpipe-roce-qos.service").read_text()
TRUST = ROOT / "tools/fabric/sparkpipe_dscp_trust.sh"
DCB_STUB = """#!/bin/sh
state="$DCB_STATE"
[ "$1" = -N ] && shift
case "$1 $2" in
    "app show") printf 'dscp-prio '; tr '\\n' ' ' < "$state"; echo ;;
    "app replace") shift 5; [ "$DCB_FAIL" = 1 ] && exit 1; printf '%s\\n' "$@" > "$state" ;;
    "app flush") : > "$state" ;;
    *) exit 3 ;;
esac
"""


def dcb_map(command, key):
    match = re.search(rf"^{command} .*?{key} ((?:(?:all|[0-9]+):[0-9a-z]+ ?)+)", SCRIPT, re.M)
    assert match, (command, key)
    values = {}
    for item in match.group(1).split():
        index, value = item.split(":")
        for slot in (range(8) if index == "all" else [int(index)]):
            values[slot] = value
    return [values[slot] for slot in range(8)]


class RoceQosContract(unittest.TestCase):
    def test_traffic_class_lands_in_the_pfc_priority(self):
        traffic_class = int(re.search(r"^MESH_TRAFFIC_CLASS=(\d+)$", AGENT, re.M).group(1))
        dscp = traffic_class >> 2
        priority = dscp >> 3
        self.assertEqual(dscp, 26)
        pfc = dcb_map(r'dcb pfc set dev "\$interface"', "prio-pfc")
        self.assertEqual([index for index, value in enumerate(pfc) if value == "on"], [priority])
        buffers = [int(value) for value in dcb_map(r'dcb buffer set dev "\$interface"', "prio-buffer")]
        sizes = [int(value) for value in dcb_map(r'dcb buffer set dev "\$interface"', "buffer-size")]
        self.assertTrue(all(buffers[index] == 0 for index in range(8) if index != priority))
        self.assertNotEqual(buffers[priority], 0)
        self.assertGreater(sizes[buffers[priority]], 0)
        self.assertGreater(sizes[0], 0)
        self.assertIn(f'"{priority}:on"', SCRIPT)
        self.assertIn(f'"{priority}:{buffers[priority]}"', SCRIPT)

    def test_script_trusts_dscp_and_disables_global_pause(self):
        self.assertIn('/usr/local/sbin/sparkpipe-dscp-trust set "$interface"', SCRIPT)
        self.assertIn('/usr/local/sbin/sparkpipe-dscp-trust check "$interface"', SCRIPT)
        self.assertNotIn("mlnx_qos", SCRIPT)
        self.assertIn('ethtool -A "$interface" rx off tx off', SCRIPT)
        self.assertIn("set -eu", SCRIPT)
        interface = re.search(r"^interface=(\S+)$", SCRIPT, re.M).group(1)
        self.assertEqual(interface, re.search(r'^MESH_INTERFACE="\$\{SPARK_MESH_INTERFACE:-rocep1s0f1\}"$', AGENT, re.M) and "enp1s0f1np1")

    def test_unit_runs_the_script_once_per_link(self):
        self.assertIn("ExecStart=/usr/local/sbin/sparkpipe_roce_qos.sh", UNIT)
        self.assertIn("Type=oneshot", UNIT)
        self.assertIn("RemainAfterExit=yes", UNIT)
        self.assertIn("WantedBy=sys-subsystem-net-devices-enp1s0f1np1.device", UNIT)

    def test_agent_refuses_the_lossless_class_without_the_unit(self):
        gate = AGENT.index("systemctl is-active -q sparkpipe-roce-qos")
        self.assertLess(gate, AGENT.index('--mesh-traffic-class "$MESH_TRAFFIC_CLASS"'))
        self.assertIn("dependent startup blocked", AGENT[gate:gate + 400])
        self.assertIn('sparkpipe-dscp-trust check "$mesh_netdev"', AGENT)


class DscpTrustTest(unittest.TestCase):
    def setUp(self):
        self.directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.directory)]))
        stubs = self.directory / "bin"
        stubs.mkdir()
        (stubs / "dcb").write_text(DCB_STUB)
        (stubs / "dcb").chmod(0o755)
        self.state = self.directory / "state"
        self.state.write_text("")
        self.environment = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "DCB_STATE": str(self.state), "DCB_FAIL": "0"}

    def trust(self, action, device="rail0"):
        return subprocess.run(["sh", str(TRUST), action, device], capture_output=True, text=True, env=self.environment)

    def test_set_programs_the_dscp_shift_three_map_and_check_confirms_it(self):
        self.assertEqual(self.trust("check").returncode, 1)
        result = self.trust("set")
        self.assertEqual((result.returncode, result.stdout), (0, "rail0 trusts DSCP\n"), result.stderr)
        mapping = dict(line.split(":") for line in self.state.read_text().split())
        self.assertEqual(len(mapping), 64)
        self.assertEqual(mapping["26"], "3")
        self.assertTrue(all(int(priority) == int(dscp) >> 3 for dscp, priority in mapping.items()))
        self.assertEqual(self.trust("check").returncode, 0)

    def test_a_partial_or_wrong_map_is_not_dscp_trust_and_clear_returns_to_pcp(self):
        self.state.write_text("".join(f"{dscp}:{dscp >> 3}\n" for dscp in range(63)))
        self.assertEqual(self.trust("check").returncode, 1)
        self.state.write_text("".join(f"{dscp}:{0 if dscp == 26 else dscp >> 3}\n" for dscp in range(64)))
        self.assertEqual(self.trust("check").returncode, 1)
        self.assertEqual(self.trust("set").returncode, 0)
        result = self.trust("clear")
        self.assertEqual((result.returncode, result.stdout), (0, "rail0 trusts PCP\n"))
        self.assertEqual(self.trust("check").returncode, 1)

    def test_a_failed_set_fails_and_bad_usage_is_refused(self):
        self.environment["DCB_FAIL"] = "1"
        self.assertNotEqual(self.trust("set").returncode, 0)
        self.assertEqual(self.trust("enable").returncode, 2)
        self.assertEqual(subprocess.run(["sh", str(TRUST), "set"], capture_output=True, env=self.environment).returncode, 2)


if __name__ == "__main__":
    unittest.main()
