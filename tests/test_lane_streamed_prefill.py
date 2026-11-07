#!/usr/bin/env python3
"""A lane record's streamed prefill reaches the rank adapters and the API limits.

Each rank names the pack of its rank ^ 1 partner under its own home, and the
API deployment carries the minimum prompt.
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
    def test_the_fill_lane_renders_beside_the_decode_lane(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "lane"
            lane_config.render("kimi-k3", directory)
            streamed = stage_lane.lane_records()["lanes"]["kimi-k3"]["fill"]["streamed_prefill"]
            self.assertNotIn("streamed_prefill_min_tokens", json.loads((directory / "model_resident.json").read_text())["runtime_limits"])
            self.assertEqual(json.loads((directory / "model_resident.fill.json").read_text())["runtime_limits"]["streamed_prefill_min_tokens"], streamed["min_tokens"])
            self.assertIn("config/adapter.fill.json", (directory / "deployment.fill.json").read_text())
            self.assertNotIn("config/adapter.fill.json", (directory / "deployment.json").read_text())
            for rank, host in enumerate(stage_lane.HOSTS):
                self.assertNotIn("streamed_prefill", json.loads((directory / f"adapter.{host}.json").read_text()))
                config = json.loads((directory / f"adapter.fill.{host}.json").read_text())["streamed_prefill"]
                self.assertEqual(config["partner_pack"], f"/home/{host}/{streamed['partner_pack'].format(partner=rank ^ 1)}")
                self.assertEqual(config["node_rows"], streamed["node_rows"])
                self.assertNotIn("min_tokens", config)


if __name__ == "__main__":
    unittest.main()
