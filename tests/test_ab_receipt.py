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


def k_pair():
    reference, _ = pair()
    base = ab_dry_run.receipt(ab_dry_run.arm_descriptor("demo", "S1", "fp8", "fp8", "publisher"), "k-ref", ab_dry_run.hexid("m", "k-ref"),
                              ab_dry_run.hexid("tok"), None, CORPUS, PLAN, {"EXPERT_CODEC": "fp8"})

    def kv(a):
        a["kv"].update(latent="fp8", group=128, mode="sim")
    arm = rearm(base, kv)
    arm["run_label"] = "k-arm"
    arm["cache"] = {**arm["cache"], "arm_root": "/home/spark/ab/k-arm", "kv_snapshot_directory": "/home/spark/ab/k-arm/kv-snapshots"}
    arm["inputs"]["probe_sha256"] = PROBE
    arm["build"]["flags"] = {"EXPERT_CODEC": "fp8", "KV_QUANT_SIM": "1"}
    return base, arm


class CompareCoverageTest(unittest.TestCase):
    def refused(self, mutate, needle, axis="E", pairing=pair):
        reference, arm = pairing()
        mutate(reference, arm)
        problems = ab_receipt.compare(reference, arm, axis)
        self.assertTrue(any(needle in problem for problem in problems), (needle, problems))

    def test_k_pair_is_comparable(self):
        reference, arm = k_pair()
        self.assertEqual(ab_receipt.compare(reference, arm, "K"), [])

    def test_every_contract_refusal_fires(self):
        cases = [
            (lambda r, a: a["build"].__setitem__("contract_sha256", ab_dry_run.hexid("other-contract")), "C1 build contract_sha256", "E"),
            (lambda r, a: r["cache"].__setitem__("kv_snapshot_directory", a["cache"]["kv_snapshot_directory"]) or r["cache"].__setitem__("arm_root", a["cache"]["arm_root"]),
             "C4 both runs share one kv_snapshot_directory", "E"),
            (lambda r, a: a["execution"].__setitem__("mode", "eager"), "C5 execution mode", "E"),
            (lambda r, a: a["execution"].__setitem__("dropin_sha256", ab_dry_run.hexid("other-dropin")), "C5 execution dropin_sha256", "E"),
            (lambda r, a: a["weightd"].__setitem__("residency", "lazy"), "C5 weightd residency", "E"),
            (lambda r, a: a["inputs"].__setitem__("corpus_tokens_sha256", ab_dry_run.hexid("other-corpus")), "C6 input corpus_tokens_sha256", "E"),
            (lambda r, a: a["inputs"].__setitem__("corpus_index_sha256", ab_dry_run.hexid("other-index")), "C6 input corpus_index_sha256", "E"),
            (lambda r, a: a["inputs"].__setitem__("tokenizer_sha256", ab_dry_run.hexid("other-tokenizer")), "C6 input tokenizer_sha256", "E"),
            (lambda r, a: a["inputs"].__setitem__("corpus", "OTHER"), "C6 input corpus", "E"),
            (lambda r, a: a["inputs"].__setitem__("probe_sha256", None), "C6 the arm run did not read the reference probe file", "E"),
            (lambda r, a: a.__setitem__("plan_sha256", ab_dry_run.hexid("other-plan")), "C6 plan sha differs", "E"),
        ]
        for mutate, needle, axis in cases:
            self.refused(mutate, needle, axis)
        reference, arm = pair()
        arm["build"]["model_revision"] = "other@1"
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "model_revision differs from the arm revision"):
            ab_receipt.compare(reference, arm, "E")

    def test_model_revision_outside_a_spine_comparison(self):
        reference, arm = pair()
        arm = rearm(arm, lambda a: a.__setitem__("revision", "org/Other@9"))
        arm["build"]["model_revision"] = "org/Other@9"
        problems = ab_receipt.compare(reference, arm, "E")
        self.assertTrue(any("C1 build model_revision differs outside a declared spine comparison" in problem for problem in problems), problems)

    def test_k_arm_on_other_pack_bytes_is_refused(self):
        def repack(r, a):
            other = rearm(a, lambda d: d["pack_sha256"].__setitem__(0, ab_dry_run.hexid("other-pack")))
            a.clear()
            a.update(other)
        self.refused(repack, "a K arm runs on its base arm's pack bytes", "K", k_pair)

    def test_plan_commit_is_enforced(self):
        reference, arm = pair()
        problems = ab_receipt.compare(reference, arm, "E", {"firmware_commit": "3" * 40, "plan_sha256": PLAN})
        self.assertTrue(any("not on the plan's pinned firmware commit" in problem for problem in problems), problems)
        problems = ab_receipt.compare(reference, arm, "E", {"firmware_commit": reference["source_commit"], "plan_sha256": ab_dry_run.hexid("other-plan")})
        self.assertTrue(any("C6 plan sha differs" in problem for problem in problems), problems)

    def test_validate_refusals(self):
        def set_path(receipt, path, value):
            node = receipt
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value

        cases = [
            (("format",), "sparkpipe-ab-receipt-v0", "must be 'sparkpipe-ab-receipt-v1'"),
            (("weightd", "residency"), "mapped", "must be one of"),
            (("topology", "tp"), "4", "must be of type"),
            (("source_commit",), "ABC", "does not match"),
            (("inputs", "corpus"), "", "is too short"),
            (("wave", "output_token_budget"), 0, "is below 1"),
            (("topology", "kv_shard"), 2, "is above 1"),
            (("arm", "format"), "sparkpipe-quant-arm-v0", "receipt.arm:"),
            (("ready_event", "pack_set_sha256"), ab_dry_run.hexid("other-pack-set"), "ready event pack_set_sha256 differs"),
            (("packs", "pack_sha256"), [ab_dry_run.hexid("other-pack", r) for r in range(ab_dry_run.RANKS)], "receipt pack_sha256 differs"),
            (("topology", "kv_shard"), 0, "receipt topology differs"),
            (("dumps", "per_rank_sha256"), [ab_dry_run.hexid("dump")], "dumps.per_rank_sha256 has 1 entries"),
            (("requests", "count"), 2, "one value per request"),
        ]
        for path, value, needle in cases:
            _, arm = pair()
            set_path(arm, path, value)
            with self.assertRaisesRegex(ab_receipt.ReceiptError, needle, msg=str(path)):
                ab_receipt.validate(arm)
        _, arm = pair()
        del arm["window"]
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "receipt.window is required"):
            ab_receipt.validate(arm)
        reference, arm = pair()
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "axis must be one of"):
            ab_receipt.compare(reference, arm, "X")

    def test_aa_refusals(self):
        reference, arm = pair()
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "runs one arm twice"):
            ab_receipt.aa(reference, arm)
        second = copy.deepcopy(reference)
        second["cache"]["arm_root"] = "/home/spark/ab/ref-aa"
        second["cache"]["kv_snapshot_directory"] = "/home/spark/ab/ref-aa/kv"
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "two distinct runs"):
            ab_receipt.aa(reference, second)
        second = copy.deepcopy(reference)
        second["run_label"] = "ref-aa"
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "two fresh snapshot directories"):
            ab_receipt.aa(reference, second)

    def test_validate_refuses_a_foreign_arm_digest(self):
        reference, arm = pair()
        arm["arm_digest"] = reference["arm_digest"]
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "is not the digest of its arm"):
            ab_receipt.validate(arm)

    def test_aa_checks_generated_tokens(self):
        reference, _ = pair()
        second = copy.deepcopy(reference)
        second["run_label"] = "ref-aa"
        second["cache"]["arm_root"] = "/home/spark/ab/ref-aa"
        second["cache"]["kv_snapshot_directory"] = "/home/spark/ab/ref-aa/kv"
        second["requests"]["generated_token_ids_sha256"] = ab_dry_run.hexid("other-tokens")
        self.assertEqual(ab_receipt.aa(reference, second)["status"], "DIVERGED")

    def test_dumpcheck(self):
        reference, _ = pair()
        off = copy.deepcopy(reference)
        off["run_label"] = "ref-off"
        off["dumps"]["score_dump_on"] = False
        self.assertTrue(ab_receipt.dumpcheck(reference, off))
        off["requests"]["generated_token_ids_sha256"] = ab_dry_run.hexid("other-tokens")
        self.assertFalse(ab_receipt.dumpcheck(reference, off))
        with self.assertRaisesRegex(ab_receipt.ReceiptError, "dump on and one with it off"):
            ab_receipt.dumpcheck(reference, reference)


if __name__ == "__main__":
    unittest.main(verbosity=1)
