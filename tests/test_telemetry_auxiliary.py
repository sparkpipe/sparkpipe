import os
import sys

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tools","telemetry"))
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import spark_telemetry_collect as collect
import spark_telemetry_dashboard as dashboard


class AuxiliaryTelemetryTest(unittest.TestCase):
    def test_inventory_label_preserves_reported_hostname(self):
        text = "unix_ts,node,hostname,cpu_util_pct\n123,spark7,lenovo7,1.5\n"
        rows = collect.read_rows(collect.labelled_csv(text,"lenovo7"))
        self.assertEqual(rows[0]["node"],"lenovo7")
        self.assertEqual(rows[0]["hostname"],"lenovo7")
        self.assertEqual(rows[0]["cpu_util_pct"],"1.5")

    def test_spark_labels_and_empty_input(self):
        text = "unix_ts,node,hostname\n123,spark8,spark8\n"
        self.assertEqual(collect.read_rows(collect.labelled_csv(text,"spark8"))[0]["node"],"spark8")
        self.assertEqual(collect.labelled_csv("","lenovo8"),"")

    def test_auxiliary_history_names_reject_path_traversal(self):
        for node in ("spark7","spark8","spark9","lenovo7","lenovo8","lenovo9"):
            self.assertTrue(dashboard.valid_node_name(node))
        for node in ("../lenovo7","lenovo7/../../etc/passwd","lenovo10",""):
            self.assertFalse(dashboard.valid_node_name(node))

    def test_explicit_ssh_config_is_passed_as_one_argument(self):
        with patch.object(collect.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="",stderr="")) as run:
            collect.fetch_node("lenovo8","/tmp/telemetry",8,10,ssh_config="/tmp/config with spaces")
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3],["ssh","-F","/tmp/config with spaces"])


if __name__ == "__main__":
    unittest.main()
