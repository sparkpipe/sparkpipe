"""Directory conformance for the standalone DSV4 Pro rank verifier.

The verifier re-derives a rank's expected directory from the contract
alone; these checks pin that planning against the build receipts of the
deployed TP4xPP4 packs (tensor counts) and exercise the directory
failure modes on a synthetic pack layout, no GPU or pack file needed.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dsv4_pro_rank_pack_verify as verify  # noqa: E402

CONTRACT = (Path(__file__).resolve().parents[1]
            / "model_contracts" / "dsv4_pro.json")

TP4PP4_RECEIPT_COUNTS = {
    0: 572, 3: 572, 4: 549, 8: 543, 12: 554, 15: 554,
}


def _expected(rank, tp_degree, pp_stages):
    stage, tp = divmod(rank, tp_degree)
    return verify.build_expected(tp, stage, tp_degree, pp_stages,
                                 json.loads(CONTRACT.read_text()))


def _synthetic_directory(expected):
    cursor = verify.HEADER.size + verify.ENTRY.size * len(expected)
    entries = []
    for (kind, layer), plan in sorted(expected.items()):
        payload = verify.tp16.payload_bytes(
            plan["weight"], plan["rows"], plan["columns"])
        offset = cursor
        cursor += payload
        scale_offset = 0
        if plan["scale_bytes"]:
            scale_offset = cursor
            cursor += plan["scale_bytes"]
        entries.append(verify.ENTRY.pack(
            kind, layer, plan["weight"], plan["rows"], plan["columns"],
            0, offset, scale_offset))
    slice_first, slice_count = verify.tp16.layer_slice(1, 0)
    header = verify.HEADER.pack(
        verify.MAGIC, 4, verify.HEADER.size, verify.ENTRY.size, 1,
        4, 3, 1, len(entries), slice_first, slice_count,
        verify.PRO_LAYERS, verify.PRO_HIDDEN, verify.PRO_VOCAB,
        verify.PRO_EXPERTS, verify.PRO_MTP_PACKED,
        verify.HEADER.size, cursor)
    return verify.HEADER.unpack(header), \
        [verify.ENTRY.unpack(e) for e in entries], cursor


class RankPackVerify(unittest.TestCase):
    def test_planning_matches_deployed_tp4pp4_receipts(self):
        for rank, count in TP4PP4_RECEIPT_COUNTS.items():
            with self.subTest(rank=rank):
                self.assertEqual(len(_expected(rank, 4, 4)), count)

    def test_tp16_plans_full_directory_on_every_rank(self):
        for rank in (0, 7, 15):
            with self.subTest(rank=rank):
                expected = _expected(rank, 16, 1)
                self.assertEqual(len(expected), 1975)
                self.assertEqual(
                    expected[(verify.tp16.KIND_WQ_A, 0)]["rows"], 96)
                draft_key = (verify.tp16.KIND_EXPERTS_W1,
                             verify.tp16.MTP_LAYER_FIRST)
                self.assertEqual(expected[draft_key]["rows"],
                                 verify.PRO_EXPERTS * 3072)

    def test_conforming_directory_passes(self):
        expected = _expected(5, 16, 1)
        header, entries, size = _synthetic_directory(expected)
        verify.verify_directory(header, entries, expected, size, 5, 16, 1)

    def test_directory_failures_are_named(self):
        expected = _expected(5, 16, 1)
        header, entries, size = _synthetic_directory(expected)

        def layer_slice(hdr, ent):
            hdr[9] = 3
            return hdr, ent, size

        def dims_drift(hdr, ent):
            ent[0][3] += 1
            return hdr, ent, size

        def missing_entry(hdr, ent):
            return hdr, ent[:10] + ent[11:], size

        def payload_out_of_bounds(hdr, ent):
            ent[3][6] = size + 1
            return hdr, ent, size

        def header_size_disagrees(hdr, ent):
            hdr[17] = size - 1
            return hdr, ent, size

        cases = (
            (layer_slice, "layer slice"),
            (dims_drift, "dims mismatch"),
            (missing_entry, "tensor count"),
            (payload_out_of_bounds, "payload bounds exceed file"),
            (header_size_disagrees, "size fields"),
        )
        for mutate, reason in cases:
            with self.subTest(case=mutate.__name__):
                hdr, ent, file_bytes = mutate(
                    list(header), [list(entry) for entry in entries])
                with self.assertRaises(verify.VerifyFailure) as caught:
                    verify.verify_directory(
                        tuple(hdr), [tuple(entry) for entry in ent],
                        expected, file_bytes, 5, 16, 1)
                self.assertIn(reason, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
