import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools","telemetry"))
import importlib.util
import sys
import unittest
import unittest.mock
from pathlib import Path


SCRIPT = Path("tools/telemetry/spark_telemetry_stop.py")


def load_module():
    spec = importlib.util.spec_from_file_location("spark_telemetry_stop", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["spark_telemetry_stop"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SparkTelemetryStopTest(unittest.TestCase):
    def test_matches_only_python_monitor(self) -> None:
        mod = load_module()
        self.assertTrue(
            mod.is_monitor_command(
                "python3 /home/spark0/bin/spark_node_telemetry_monitor.py --out-dir /tmp/ds4",
                "/tmp/ds4",
            )
        )
        self.assertFalse(
            mod.is_monitor_command(
                "bash -c python3 /home/spark0/bin/spark_node_telemetry_monitor.py --out-dir /tmp/ds4",
                "/tmp/ds4",
            )
        )

    def test_rejects_wrong_out_dir_and_stop_script(self) -> None:
        mod = load_module()
        self.assertFalse(
            mod.is_monitor_command(
                "python3 /home/spark0/bin/spark_node_telemetry_monitor.py --out-dir /tmp/other",
                "/tmp/ds4",
            )
        )
        self.assertFalse(
            mod.is_monitor_command(
                "python3 /home/spark0/bin/spark_telemetry_stop.py spark_node_telemetry_monitor.py",
                "",
            )
        )


    def test_out_dir_is_matched_by_argument_not_substring(self) -> None:
        mod = load_module()
        monitor = "python3 /home/spark0/bin/spark_node_telemetry_monitor.py"
        self.assertFalse(mod.is_monitor_command(f"{monitor} --out-dir /tmp/ds4_telemetry", "/tmp/ds4"))
        self.assertTrue(mod.is_monitor_command(f"{monitor} --out-dir=/tmp/ds4", "/tmp/ds4"))
        self.assertTrue(mod.is_monitor_command(f"{monitor} --node spark0 --interval 5", mod.telemetry.TELEMETRY_DIR))
        self.assertFalse(mod.is_monitor_command(f"{monitor} --node spark0 --interval 5", "/tmp/elsewhere"))
        self.assertFalse(mod.is_monitor_command("python3 /home/spark0/bin/spark_telemetry_start.sh spark_node_telemetry_monitor.py", ""))

    def test_an_active_unit_is_stopped_before_any_signal(self) -> None:
        mod = load_module()
        calls = []

        def run(argv, check=False):
            calls.append(argv)
            if argv[2] == "stop":
                return mod.subprocess.CompletedProcess(argv, 0)
            active = sum(1 for call in calls if call[2:3] == ["stop"]) == 0
            return mod.subprocess.CompletedProcess(argv, 0 if active else 3)
        with unittest.mock.patch.object(mod.subprocess, "run", side_effect=run), \
                unittest.mock.patch.object(mod, "process_table", return_value=[]), \
                unittest.mock.patch.object(mod.sys, "argv", ["spark_telemetry_stop.py"]):
            self.assertEqual(mod.main(), 0)
        self.assertIn(["systemctl", "--user", "stop", mod.UNIT], calls)


if __name__ == "__main__":
    unittest.main()
