import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT = (ROOT / "tools/fleet_node_agent.sh").read_text()
SCRIPT = (ROOT / "tools/fabric/sparkpipe_roce_qos.sh").read_text()
UNIT = (ROOT / "tools/fabric/sparkpipe-roce-qos.service").read_text()


def vector(flag):
    match = re.search(flag + r" ([0-9,]+)", SCRIPT)
    assert match, flag
    return [int(value) for value in match.group(1).split(",")]


class RoceQosContract(unittest.TestCase):
    def test_traffic_class_lands_in_the_pfc_priority(self):
        traffic_class = int(re.search(r"^MESH_TRAFFIC_CLASS=(\d+)$", AGENT, re.M).group(1))
        priority = (traffic_class >> 2) >> 3
        self.assertEqual(traffic_class >> 2, 26)
        pfc = vector("--pfc")
        self.assertEqual([index for index, value in enumerate(pfc) if value], [priority])
        buffers = vector("--prio2buffer")
        sizes = vector("--buffer_size")
        self.assertTrue(all(buffers[index] == 0 for index in range(8) if index != priority))
        self.assertNotEqual(buffers[priority], 0)
        self.assertGreater(sizes[buffers[priority]], 0)
        self.assertGreater(sizes[0], 0)
        self.assertIn(f'"{priority}:on"', SCRIPT)
        self.assertIn(f'"{priority}:{buffers[priority]}"', SCRIPT)

    def test_script_trusts_dscp_and_disables_global_pause(self):
        self.assertIn("--trust dscp", SCRIPT)
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


if __name__ == "__main__":
    unittest.main()
