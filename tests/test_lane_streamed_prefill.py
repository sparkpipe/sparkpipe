#!/usr/bin/env python3
"""A lane record's streamed prefill reaches the rank adapters and the API limits.

Each rank names the pack of its rank ^ 1 partner under its own home, and the
API deployment carries the quantum and the minimum prompt.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lane_config
import stage_lane


class LaneStreamedPrefill(unittest.TestCase):
    def test_a_streamed_record_renders_partner_packs_and_limits(self):
        lane = lane_config.records()["kimi-k3.1-pro"]
        streamed = lane["streamed_prefill"]
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "lane"
            lane_config.render("kimi-k3.1-pro", directory)
            limits = json.loads((directory / "model_resident.json").read_text())["runtime_limits"]
            self.assertEqual(limits["streamed_prefill_quantum"], streamed["quantum"])
            self.assertEqual(limits["streamed_prefill_min_tokens"], streamed["min_tokens"])
            for rank, host in enumerate(stage_lane.HOSTS):
                config = json.loads((directory / f"adapter.{host}.json").read_text())["streamed_prefill"]
                self.assertEqual(config["partner_pack"], f"/home/{host}/{streamed['partner_pack'].format(partner=rank ^ 1)}")
                self.assertEqual(config["node_rows"], streamed["node_rows"])
                self.assertNotIn("quantum", config)
                self.assertNotIn("min_tokens", config)


if __name__ == "__main__":
    unittest.main()
