#!/usr/bin/env python3
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_arm
import ab_dry_run
import ab_receipt

CORPUS = {"corpus_sha256": ab_dry_run.hexid("corpus"), "tokenizer_sha256": ab_dry_run.hexid("tok"), "documents": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
PLAN = ab_dry_run.hexid("plan")
PROBE = ab_dry_run.hexid("probe")


def pair(expert=("nvfp4", "nvfp4nv", "community"), flags=None):
    reference_arm = ab_dry_run.arm_descriptor("demo", "S1", "bf16", "bf16", "publisher")
    arm = ab_dry_run.arm_descriptor("demo", "S1", *expert)
    reference = ab_dry_run.receipt(reference_arm, "ref", ab_dry_run.hexid("m", "ref"), ab_dry_run.hexid("tok"), None, CORPUS, PLAN, {"EXPERT_CODEC": "bf16"})
    candidate = ab_dry_run.receipt(arm, "arm", ab_dry_run.hexid("m", "arm"), ab_dry_run.hexid("tok"), PROBE, CORPUS, PLAN, flags or {"EXPERT_CODEC": expert[0]})
    return reference, candidate


def rearm(receipt, mutate):
    receipt = copy.deepcopy(receipt)
    mutate(receipt["arm"])
    receipt["arm"]["arm_id"] = ab_arm.expected_id(receipt["arm"])
    receipt["arm_digest"] = ab_arm.arm_digest(receipt["arm"])
    receipt["ready_event"]["arm_digest"] = receipt["arm_digest"]
    receipt["ready_event"]["pack_set_sha256"] = ab_arm.pack_set_sha256(receipt["arm"])
    receipt["packs"]["pack_sha256"] = receipt["arm"]["pack_sha256"]
    return receipt


class ReceiptTest(unittest.TestCase):
    def test_comparable_pair(self):
        reference, arm = pair()
        self.assertEqual(ab_receipt.compare(reference, arm, "E"), [])

    def test_spine_digest_mismatch_is_refused(self):
        reference, arm = pair()
        arm = rearm(arm, lambda a: a["spine"]["spine_digest"].__setitem__(2, ab_dry_run.hexid("other-spine")))
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any("spine digests differ on 1 ranks" in problem for problem in problems), problems)
        self.assertTrue(any("differ on ['spine', 'E']" in problem for problem in problems), problems)

    def test_two_axis_difference_is_refused(self):
        reference, arm = pair()

        def kv(a):
            a["kv"].update(latent="fp8", group=128, mode="sim")
        arm = rearm(arm, kv)
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any("['E', 'K']" in problem for problem in problems), problems)

    def test_declared_axis_must_match(self):
        reference, arm = pair()
        problems = ab_receipt.compare(reference, arm, "K")
        self.assertTrue(any("declares exactly [K]" in problem for problem in problems), problems)

    def test_commit_mismatch_is_refused(self):
        reference, arm = pair()
        arm["source_commit"] = "2" * 40
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any(problem.startswith("C1 source commit") for problem in problems), problems)

    def test_build_flags_outside_the_axis_are_refused(self):
        reference, arm = pair(flags={"EXPERT_CODEC": "nvfp4", "KV_QUANT_SIM": "1"})
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any("KV_QUANT_SIM" in problem for problem in problems), problems)

    def test_cached_prompt_tokens_are_refused(self):
        reference, arm = pair()
        arm["requests"]["cached_prompt_tokens"][1] = 64
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "cached_prompt_tokens > 0"):
            ab_receipt.compare(reference, arm, "E")

    def test_snapshot_directory_outside_the_root_is_refused(self):
        reference, arm = pair()
        for path in ("/home/spark/release/glm/kv", "/home/spark/ab/arm/../ref/kv", "/home/spark/ab/arm"):
            broken = copy.deepcopy(arm)
            broken["cache"]["kv_snapshot_directory"] = path
            with self.assertRaisesRegex(ab_receipt.ReceiptError, "not under the arm's own root|absolute, normalized"):
                ab_receipt.validate(broken)

    def test_arm_root_must_be_a_private_absolute_directory(self):
        reference, arm = pair()
        for root, snapshot, message in (("run", "run/kv", "absolute, normalized"), ("/home/spark/ab/arm/", "/home/spark/ab/arm/kv", "absolute, normalized"),
                                        ("~/ab/arm", "~/ab/arm/kv", "absolute, normalized"), ("/home/spark", "/home/spark/release/kv", "not an arm's own directory"),
                                        ("/", "/home/spark/release/kv", "not an arm's own directory")):
            broken = copy.deepcopy(arm)
            broken["cache"]["arm_root"] = root
            broken["cache"]["kv_snapshot_directory"] = snapshot
            with self.assertRaisesRegex(ab_receipt.ReceiptError, message):
                ab_receipt.validate(broken)

    def test_default_weightd_lane_is_refused(self):
        reference, arm = pair()
        arm["weightd"]["lane"] = 0
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "weightd.lane 0"):
            ab_receipt.validate(arm)

    def test_headroom_floor_and_memory_coverage(self):
        reference, arm = pair()
        broken = copy.deepcopy(arm)
        broken["memory"]["nodes"][1]["mem_available_after_gib"] = 19.5
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "headroom floor on \\['node1'\\]"):
            ab_receipt.validate(broken)
        broken = copy.deepcopy(arm)
        del broken["memory"]["nodes"][2]
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "no MemAvailable record for rank nodes \\['node2'\\]"):
            ab_receipt.validate(broken)
        broken = copy.deepcopy(arm)
        broken["memory"]["nodes"][0]["mem_available_before_gib"] = "80"
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "must be a number"):
            ab_receipt.validate(broken)
        arm["memory"]["nodes"][1]["mem_available_after_gib"] = 20
        ab_receipt.validate(arm)

    def test_ready_event_must_name_the_arm(self):
        reference, arm = pair()
        arm["ready_event"]["arm_digest"] = reference["arm_digest"]
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "engine ran another arm"):
            ab_receipt.validate(arm)

    def test_topology_and_wave_changes_are_refused(self):
        reference, arm = pair()
        arm["topology"]["rank_nodes"] = list(reversed(arm["topology"]["rank_nodes"]))
        arm["wave"]["max_prefill_rows_per_submission"] = 128
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any(problem.startswith("C2 topology rank_nodes") for problem in problems), problems)
        self.assertTrue(any(problem.startswith("C3 wave") for problem in problems), problems)

    def test_schema_refuses_unknown_members(self):
        reference, _ = pair()
        reference["extra"] = 1
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "not a receipt member"):
            ab_receipt.validate(reference)

    def test_aa_gate(self):
        reference, _ = pair()
        second = copy.deepcopy(reference)
        second["run_label"] = "ref-aa"
        second["cache"]["arm_root"] = "/home/spark/ab/ref-aa"
        second["cache"]["kv_snapshot_directory"] = "/home/spark/ab/ref-aa/kv"
        second["inputs"]["document_order"] = "permuted"
        self.assertEqual(ab_receipt.aa(reference, second)["status"], "BIT-IDENTICAL")
        second["dumps"]["merged_sha256"] = ab_dry_run.hexid("different")
        self.assertEqual(ab_receipt.aa(reference, second)["status"], "DIVERGED")
        with self.assertRaises(ab_receipt.ReceiptError):
            ab_receipt.aa(reference, reference)


if __name__ == "__main__":
    unittest.main(verbosity=1)
