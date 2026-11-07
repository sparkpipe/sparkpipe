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
    def test_a_streamed_record_renders_partner_packs_and_limits(self):
        streamed = {"node_rows": 2048, "dense_buffers": 1, "expert_slots": 2, "min_tokens": 16384,
                    "partner_pack": "sparkdata/k3.mxfp4.tp16/partner/k3.stage0.rank{partner:02d}.pack",
                    "fabric_device": "rocep1s0f1", "pair_device": "roceP2p1s0f0"}
        document = json.loads(json.dumps(stage_lane.lane_records()))
        document["lanes"]["kimi-k3"]["streamed_prefill"] = streamed
        stage_lane.lane_records = lambda: document
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch) / "lane"
            lane_config.render("kimi-k3", directory)
            limits = json.loads((directory / "model_resident.json").read_text())["runtime_limits"]
            self.assertEqual(limits["streamed_prefill_min_tokens"], streamed["min_tokens"])
            for rank, host in enumerate(stage_lane.HOSTS):
                config = json.loads((directory / f"adapter.{host}.json").read_text())["streamed_prefill"]
                self.assertEqual(config["partner_pack"], f"/home/{host}/{streamed['partner_pack'].format(partner=rank ^ 1)}")
                self.assertEqual(config["node_rows"], streamed["node_rows"])
                self.assertNotIn("min_tokens", config)


if __name__ == "__main__":
    unittest.main()
