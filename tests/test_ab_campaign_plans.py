import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_fleet
import ab_plan

AB = ROOT / "qualification" / "ab"
PLANS = sorted(AB.glob("*/PLAN.json"))
SPECS = sorted((AB / "fleet").glob("*.json"))


def by_role(plan, role):
    return [arm["arm_id"] for arm in plan["arms"] if arm["role"] == role]


class CampaignPlans(unittest.TestCase):
    def test_every_committed_plan_is_frozen_and_verifies(self):
        self.assertTrue(PLANS)
        for path in PLANS:
            plan = ab_plan.load(path)
            self.assertEqual(plan["campaign"], path.parent.name, path)

    def test_an_edited_plan_is_refused(self):
        for path in PLANS:
            plan = json.loads(path.read_text())
            plan["margins"]["E"]["kl_ratio_vs_anchor_upper"] = 2.0
            edited = path.parent / "PLAN.edited.json"
            try:
                edited.write_text(json.dumps(plan))
                with self.assertRaises(ab_plan.PlanError):
                    ab_plan.load(edited)
            finally:
                edited.unlink()

    def test_campaigns_that_share_a_reference_share_its_frame(self):
        plans = [ab_plan.load(path) for path in PLANS]
        for plan in plans:
            for other in plans:
                if by_role(plan, "reference") != by_role(other, "reference"):
                    continue
                self.assertEqual(plan["firmware_commit"], other["firmware_commit"])
                self.assertEqual(plan["tokenizer_sha256"], other["tokenizer_sha256"])
                self.assertEqual(plan["margins"], other["margins"])
                self.assertEqual(plan["statistics"], other["statistics"])
                self.assertEqual(plan["backstops"], other["backstops"])
                self.assertEqual(by_role(plan, "anchor"), by_role(other, "anchor"))
                for name in set(plan["corpora"]) & set(other["corpora"]):
                    self.assertEqual(plan["corpora"][name], other["corpora"][name])

    def test_every_e_arm_is_anchored_and_scored_on_a_plan_corpus(self):
        for path in PLANS:
            plan = ab_plan.load(path)
            for arm in plan["arms"]:
                if arm.get("axis") == "E" and arm["role"] == "arm":
                    self.assertIn(arm["anchor"], by_role(plan, "anchor"), path)
                    self.assertIn(arm["corpus"], plan["corpora"], path)


class FleetSpecs(unittest.TestCase):
    def test_every_fleet_spec_loads(self):
        self.assertTrue(SPECS)
        for path in SPECS:
            ab_fleet.load_spec(path)

    def test_every_arm_pack_dir_is_protected(self):
        for path in SPECS:
            spec = ab_fleet.load_spec(path)
            for name, arm in spec["arms"].items():
                self.assertIn(f"sparkdata/{arm['pack_root']}", spec["protected_paths"], f"{path.name} {name}")

    def test_every_slot_fits_the_full_fleet_window_budget(self):
        budget = 86 - ab_fleet.agent_headroom_gib()
        for path in SPECS:
            spec = ab_fleet.load_spec(path)
            for slot in spec["slots"]:
                for evicted in ((), (spec["production"]["arena"],)):
                    needs = ab_fleet.arm_needs(spec, ab_fleet.slot_arms(slot, True), evicted)
                    self.assertLessEqual(sum(need for _, need in needs), budget, f"{path.name} slot {slot['name']} evicted={evicted}")

if __name__ == "__main__":
    unittest.main()
