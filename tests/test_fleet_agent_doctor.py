#!/usr/bin/env python3
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools/fleet_node_agent.sh").read_text()


def function(name):
    return name + "() {" + SOURCE.split("\n" + name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


HARNESS = r'''
set -u
declare -A DOCTOR_SINCE DOCTOR_NEXT
MESH_INTERFACE=rocep1s0f1
MESH_PAIR_INTERFACE=rocep1s0f0
AGENT_STARTED=0
NOW=0
date() { if [ "$1" = +%s ]; then echo "$NOW"; else echo 00:00:00; fi; }
sleep() { :; }
ibv_devinfo() { case "$2" in rocep1s0f1) echo "  state: $SWITCH_STATE";; rocep1s0f0) echo "  state: $PAIR_STATE";; esac; }
ibdev2netdev() { printf 'rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)\nrocep1s0f1 port 1 ==> enp1s0f1np1 (Up)\n'; }
ip() {
    case "$*" in
        "-o link show dev enp1s0f0np0") echo "4: enp1s0f0np0: <$PAIR_FLAGS> mtu 9000";;
        "-o link show dev enp1s0f1np1") echo "5: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000";;
        "link set enp1s0f0np0 down"|"link set enp1s0f1np1 down") echo "FLAP $3 down";;
        "link set enp1s0f0np0 up"|"link set enp1s0f1np1 up") echo "FLAP $3 up";;
    esac
}
sudo() { shift; "$@"; }
cat() { case "$1" in /sys/class/net/enp1s0f0np0/carrier) echo "$PAIR_CARRIER";; /sys/class/net/enp1s0f1np1/carrier) echo 1;; esac; }
''' + function("doctor_port") + function("node_doctor") + r'''
for t in $TIMES; do
    NOW=$t
    echo "T $t"
    node_doctor 2>/dev/null
done
'''


def run(times, pair_state="PORT_DOWN", pair_flags="NO-CARRIER,BROADCAST,MULTICAST,UP", pair_carrier="0", switch_state="PORT_ACTIVE", started=0):
    env = {"TIMES": " ".join(str(t) for t in times), "PAIR_STATE": pair_state, "PAIR_FLAGS": pair_flags,
           "PAIR_CARRIER": pair_carrier, "SWITCH_STATE": switch_state, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    script = HARNESS.replace("AGENT_STARTED=0", f"AGENT_STARTED={started}")
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    flaps, now = {}, None
    for line in result.stdout.splitlines():
        if line.startswith("T "):
            now = int(line[2:])
        elif line.startswith("FLAP"):
            flaps.setdefault(now, []).append(line)
    return flaps


class FleetAgentDoctor(unittest.TestCase):
    def test_a_pair_partner_reboot_never_flaps_the_active_switch_port(self):
        flaps = run([1000, 1030, 1061, 1100, 1362])
        self.assertTrue(all("enp1s0f1np1" not in " ".join(lines) for lines in flaps.values()), flaps)

    def test_a_dead_port_is_flapped_once_after_a_minute_then_at_most_every_five_minutes(self):
        flaps = run([1000, 1030, 1061, 1100, 1300, 1362])
        self.assertEqual(sorted(flaps), [1061, 1362], flaps)
        self.assertEqual(flaps[1061], ["FLAP enp1s0f0np0 down", "FLAP enp1s0f0np0 up"])

    def test_ports_that_are_admin_down_or_have_carrier_are_left_alone(self):
        self.assertEqual(run([1000, 1100, 1500], pair_flags="BROADCAST,MULTICAST"), {})
        self.assertEqual(run([1000, 1100, 1500], pair_carrier="1"), {})

    def test_no_flap_in_the_first_five_minutes_of_the_agent(self):
        self.assertEqual(run([1000, 1100, 1200], started=1000), {})
        self.assertEqual(sorted(run([1000, 1100, 1300], started=1000)), [1300])

    def test_a_port_that_recovers_resets_the_hold_down(self):
        script_times = [1000, 1030]
        self.assertEqual(run(script_times), {})
        self.assertEqual(run([1000], pair_state="PORT_ACTIVE"), {})


if __name__ == "__main__":
    unittest.main()
