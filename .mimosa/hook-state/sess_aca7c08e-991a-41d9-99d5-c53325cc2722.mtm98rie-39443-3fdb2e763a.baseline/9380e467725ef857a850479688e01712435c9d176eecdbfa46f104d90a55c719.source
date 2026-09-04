"""Directory conformance for the standalone DSV4 Pro rank verifier.

The verifier re-derives a rank's expected directory from the contract
alone; these checks pin that planning against the build receipts of the
deployed TP4xPP4 packs (tensor counts) and exercise the directory
failure modes on a synthetic pack layout, no GPU or pack file needed.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dsv4_pro_rank_pack_verify as verify  # noqa: E402

CONTRACT = (Path(__file__).resolve().parents[1]
            / "model_contracts" / "dsv4_pro.json")

# World rank -> tensor count, from the deployed TP4xPP4 build receipts
# (docs/AGENT_LANE_BRIEFS/reports/dsv4pro-packs-2026-08-27.md, P2).
TP4PP4_RECEIPT_COUNTS = {
    0: 572, 3: 572, 4: 549, 8: 543, 12: 554, 15: 554,
}


def _expected(rank, tp_degree, pp_stages):
    stage, tp = divmod(rank, tp_degree)
    return verify.build_expected(tp, stage, tp_degree, pp_stages,
                                 json.loads(CONTRACT.read_text()))


def _synthetic_directory(expected):
    """Fabricate a conforming header + entry list from a planned directory."""
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
    return list(verify.HEADER.unpack(header)), \
        [verify.ENTRY.unpack(e) for e in entries], cursor


def test_planning_matches_deployed_tp4pp4_receipts():
    for rank, count in TP4PP4_RECEIPT_COUNTS.items():
        assert len(_expected(rank, 4, 4)) == count, \
            f"tp4pp4 rank {rank}: planned {len(_expected(rank, 4, 4))}, receipt said {count}"


def test_tp16_plans_full_directory_on_every_rank():
    for rank in (0, 7, 15):
        expected = _expected(rank, 16, 1)
        assert len(expected) == 1975
        # sharded dims actually shrink 16-way: wq_a rows 1536 -> 96
        assert expected[(verify.tp16.KIND_WQ_A, 0)]["rows"] == 96
        # draft block stays replicated full-width on every rank
        draft_key = (verify.tp16.KIND_EXPERTS_W1, verify.tp16.MTP_LAYER_FIRST)
        plan = expected[draft_key]
        assert plan["rows"] == verify.PRO_EXPERTS * 3072


def test_conforming_directory_passes():
    expected = _expected(5, 16, 1)
    header, entries, size = _synthetic_directory(expected)
    verify.verify_directory(header, entries, expected, size, 5, 16, 1)


def test_directory_failures_are_named():
    expected = _expected(5, 16, 1)
    header, entries, size = _synthetic_directory(expected)

    def fresh():
        return ([list(h) for h in [header]], [list(e) for e in entries])

    hdr, ent = fresh()
    hdr[0][9] = 3  # wrong first layer for rank 5's stage
    try:
        verify.verify_directory(hdr[0], ent, expected, size, 5, 16, 1)
        raise AssertionError("bad layer slice accepted")
    except verify.VerifyFailure as error:
        assert "layer slice" in str(error)

    hdr, ent = fresh()
    ent[0][3] += 1  # dims drift on one entry
    try:
        verify.verify_directory(hdr[0], ent, expected, size, 5, 16, 1)
        raise AssertionError("dims drift accepted")
    except verify.VerifyFailure as error:
        assert "dims mismatch" in str(error)

    hdr, ent = fresh()
    dropped = ent[10]
    rest = ent[:10] + ent[11:]  # entry missing -> count mismatch
    try:
        verify.verify_directory(hdr[0], rest, expected, size, 5, 16, 1)
        raise AssertionError("missing entry accepted")
    except verify.VerifyFailure as error:
        assert "tensor count" in str(error)

    hdr, ent = fresh()
    ent[3][6] = size + 1  # payload out of file bounds
    try:
        verify.verify_directory(hdr[0], ent, expected, size, 5, 16, 1)
        raise AssertionError("out-of-bounds payload accepted")
    except verify.VerifyFailure as error:
        assert "bounds" in str(error)

    hdr, ent = fresh()
    hdr[0][17] = size - 1  # header file-bytes field disagrees with disk
    try:
        verify.verify_directory(hdr[0], ent, expected, size - 1, 5, 16, 1)
        raise AssertionError("size disagreement accepted")
    except verify.VerifyFailure as error:
        assert "size fields" in str(error)
