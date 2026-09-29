import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import draftd_mtp_g8 as g8


def case(kind, index, fed, target):
    return {"id": f"{kind}{index}", "kind": kind, "fed": fed, "target": target}


def build(real_count, real_miss, impl_count, impl_miss):
    cases, reference, device = [], {}, {}
    for kind, count, miss in (("real-tap", real_count, real_miss), ("impl", impl_count, impl_miss)):
        for i in range(count):
            c = case(kind, i, [1], [2])
            cases.append(c)
            reference[c["id"]] = [{"top1": 2, "margin": 1.0, "head_in_sha": "a"}]
            device[c["id"]] = [{"top1": 3 if i < miss else 2, "margin": 1.0, "head_in_sha": "a"}]
    return g8.compare(cases, reference, device, 0.0625)


class G8VerdictTest(unittest.TestCase):
    def test_pass_needs_real_positions_at_the_floor(self):
        self.assertEqual(g8.g8_verdict(build(1000, 10, 0, 0), True)[0], "PASS")

    def test_real_agreement_below_the_floor_fails(self):
        self.assertEqual(g8.g8_verdict(build(1000, 11, 0, 0), True)[0], "FAIL")

    def test_implementation_positions_do_not_count_toward_the_gate(self):
        verdict, reason = g8.g8_verdict(build(77, 0, 5000, 0), True)
        self.assertEqual(verdict, "PENDING")
        self.assertIn("real_positions=77", reason)

    def test_implementation_positions_do_not_dilute_real_misses(self):
        self.assertEqual(g8.g8_verdict(build(1000, 20, 5000, 0), True)[0], "FAIL")

    def test_run_to_run_difference_fails(self):
        self.assertEqual(g8.g8_verdict(build(1000, 0, 0, 0), False)[0], "FAIL")

    def test_floor_is_not_a_flag(self):
        with self.assertRaises(SystemExit):
            g8.main(["--checkpoint", "x", "--header", "x", "--fixtures", "x", "--streams", "x",
                     "--output", "x", "--floor", "0.5"])


if __name__ == "__main__":
    unittest.main()
