#!/usr/bin/env python3
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fill_cutoff


class FillCutoff(unittest.TestCase):
    def test_streaming_must_pay_for_its_reload_and_beat_waves_by_the_gain(self):
        model = {"reload_seconds": 100.0, "wave_seconds": [[1000, 1.0], [1000000, 1000.0]], "streamed_seconds": [[1000, 1.0], [1000000, 100.0]]}
        tokens = fill_cutoff.cutoff(model, 0.10)
        self.assertIsNotNone(tokens)
        self.assertLessEqual(fill_cutoff.seconds_at(model["streamed_seconds"], tokens) + 100.0, 0.9 * fill_cutoff.seconds_at(model["wave_seconds"], tokens))
        before = tokens - fill_cutoff.STEP_TOKENS
        self.assertGreater(fill_cutoff.seconds_at(model["streamed_seconds"], before) + 100.0, 0.9 * fill_cutoff.seconds_at(model["wave_seconds"], before))

    def test_streaming_under_ten_percent_faster_is_never_chosen(self):
        model = {"reload_seconds": 0.0, "wave_seconds": [[1000, 1.0], [1000000, 1000.0]], "streamed_seconds": [[1000, 0.95], [1000000, 950.0]]}
        self.assertIsNone(fill_cutoff.cutoff(model, 0.10))

    def test_every_streamed_record_starts_at_its_cutoff(self):
        costs = json.loads(fill_cutoff.COSTS.read_text())
        lanes = json.loads((ROOT / "deployment" / "lanes.json").read_text())["lanes"]
        for name, lane in lanes.items():
            if "fill" in lane:
                self.assertIn(name, costs["models"], f"{name} streams without measured costs")
                self.assertEqual(lane["fill"]["streamed_prefill"]["min_tokens"], fill_cutoff.cutoff(costs["models"][name], costs["required_gain"]), name)


if __name__ == "__main__":
    unittest.main()
