#!/usr/bin/env python3
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fill_plan

PLAN = {"pass_floor_seconds": 100.0, "seconds_per_token": 0.0, "attention_seconds_per_token_pair": 0.0, "row_bytes": 1000000, "ring_bytes": 0,
        "dense_bytes": 0, "free_bytes": 1000000000, "expert_bytes": 90000000000, "reload_bytes_per_second": 1000000000, "restart_seconds": 0.0,
        "row_align": 256, "max_passes": 4}


class FillPlan(unittest.TestCase):
    def test_ram_is_bought_while_a_saved_pass_outweighs_its_reload(self):
        option = fill_plan.best(PLAN, 655360)
        self.assertEqual(option["passes"], 1)
        self.assertEqual(option["node_rows"], 40960)
        self.assertEqual(option["reload_s"], 40.0)

    def test_ram_is_not_bought_when_its_reload_costs_more_than_the_pass(self):
        plan = dict(PLAN, reload_bytes_per_second=100000000)
        option = fill_plan.best(plan, 655360)
        self.assertGreater(option["passes"], 1)
        self.assertLess(option["evicted_gb"], 39.0)

    def test_a_fill_that_fits_beside_the_experts_pays_no_reload(self):
        option = fill_plan.best(PLAN, 16000)
        self.assertEqual(option["evicted_gb"], 0.0)
        self.assertEqual(option["reload_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
