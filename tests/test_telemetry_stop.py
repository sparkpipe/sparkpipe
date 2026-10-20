import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools","telemetry"))
import importlib.util
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
