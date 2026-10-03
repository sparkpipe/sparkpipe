import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dsv4_pro_merge_stagepacks as merge  # noqa: E402

flash = merge.flash
CONTRACT = flash.load_contract(Path(__file__).resolve().parents[1] / "model_contracts" / "dsv4_pro.json")
LAYERS = int(CONTRACT["model"]["layer_count"])
WINDOWS = ((0, 16), (16, 15), (31, 15), (46, 15))


def directory(first, count):
    entries, _ = flash.make_directory(flash.build_records(CONTRACT, first, count))
    return [flash.ENTRY_STRUCT.unpack(flash.pack_entry(entry)) for entry in entries]


def full_directory():
    return directory(0, LAYERS)


class PlanMergeTest(unittest.TestCase):
    def test_windows_rebuild_the_full_directory_in_order(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS]
        full = full_directory()
        order, plan = merge.plan_merge(windows, full, LAYERS)
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual([(entry[0], entry[1]) for entry, _ in plan],
                         [(entry[0], entry[1]) for entry in full])

    def test_window_tensor_counts_match_the_stage_receipts(self):
        counts = [len(directory(first, count)) for first, count in WINDOWS]
        self.assertEqual(counts[:3], [572, 549, 543])

    def test_only_globals_and_draft_layers_repeat(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS]
        _, plan = merge.plan_merge(windows, full_directory(), LAYERS)
        repeated = [entry for entry, candidates in plan if len(candidates) > 1]
        self.assertTrue(repeated)
        for entry in repeated:
            self.assertTrue(merge.is_replicated(entry[0], entry[1]))
        embedding = [candidates for entry, candidates in plan if entry[0] == flash.KIND_EMBEDDING]
        self.assertEqual([index for index, _ in embedding[0]], [0, 3])

    def test_unordered_inputs_are_sorted_by_first_layer(self):
        windows = [(first, count, directory(first, count)) for first, count in reversed(WINDOWS)]
        order, plan = merge.plan_merge(windows, full_directory(), LAYERS)
        self.assertEqual(order, [3, 2, 1, 0])
        self.assertEqual(len(plan), len(full_directory()))

    def test_gap_is_refused(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS[:2] + WINDOWS[3:]]
        with self.assertRaisesRegex(merge.MergeFailure, "non-contiguous"):
            merge.plan_merge(windows, full_directory(), LAYERS)

    def test_short_coverage_is_refused(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS[:3]]
        with self.assertRaisesRegex(merge.MergeFailure, "cover 46 layers"):
            merge.plan_merge(windows, full_directory(), LAYERS)

    def test_repeated_regular_layer_is_refused(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS]
        extra = [entry for entry in windows[1][2] if entry[1] == 16][0]
        windows[0][2].append(extra)
        with self.assertRaisesRegex(merge.MergeFailure, "appears in 2 windows"):
            merge.plan_merge(windows, full_directory(), LAYERS)

    def test_shape_mismatch_is_refused(self):
        windows = [(first, count, directory(first, count)) for first, count in WINDOWS]
        entry = windows[2][2][5]
        windows[2][2][5] = entry[:3] + (entry[3] + 1,) + entry[4:]
        with self.assertRaisesRegex(merge.MergeFailure, "shape differs"):
            merge.plan_merge(windows, full_directory(), LAYERS)

    def test_kv_codec_override_follows_the_windows(self):
        self.assertEqual(merge.contract_codecs(CONTRACT, "fp8_e4m3")[2], flash.CODEC_IDS["fp8_e4m3"])
        self.assertEqual(merge.contract_codecs(CONTRACT, None)[2], flash.CODEC_IDS["bf16"])


if __name__ == "__main__":
    unittest.main()
