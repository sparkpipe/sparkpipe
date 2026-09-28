"""gemma4 --verify-existing rank-list semantics on a synthetic 26B stage.

The placement proof must plan the requested rank (not always rank 0) and,
for the MoE geometry, prove rank identity through the experts manifest:
every record's expert id must sit in the verified rank's span and the
truncated ck128 payload digests must match the pack bytes. The geometry is
shrunk (same kind plan, same layout math) so the ck128 walk stays fast.
"""

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gemma4_stagepack as G

TOOL = ROOT / "tools" / "gemma4_stagepack.py"
GEOMETRY = "26b-a4b"
TP_DEGREE = 4
FIRST_LAYER, LAYER_COUNT = 16, 7
SMALL = dict(G.GEOMETRY[GEOMETRY])
SMALL.update(hidden=256, vocab=1024, dense_inter=128, expert_inter=32)


def shrink_geometry():
    G.GEOMETRY[GEOMETRY] = dict(SMALL)


def build_stage_pack(pack_path: Path, manifest_path: Path, tp_rank: int) -> None:
    geometry = G.GEOMETRY[GEOMETRY]
    plan = G.build_inventory(geometry, TP_DEGREE, tp_rank, FIRST_LAYER, LAYER_COUNT)
    for entry in plan:
        elements = entry["rows"] * entry["columns"]
        entry["payload_bytes"] = elements * (4 if entry["weight_format"] == G.WEIGHT_F32 else 2)
        entry["scale_bytes"] = 0
    plan, file_bytes, payload_bytes = G.place(plan)
    directory_offset = G.HEADER_BYTES
    fields = G.header_fields(geometry, plan, FIRST_LAYER, LAYER_COUNT, file_bytes,
                             directory_offset)
    directory = b"".join(G.ENTRY_STRUCT.pack(
        entry["kind"],
        G.GLOBAL_LAYER if entry["layer"] == G.GLOBAL_LAYER else entry["layer"],
        entry["weight_format"], entry["rows"], entry["columns"], 0,
        entry["payload_offset"], entry["payload_bytes"], 0, 0)
        for entry in plan)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    with pack_path.open("wb") as pack:
        pack.truncate(file_bytes)
        pack.seek(0)
        pack.write(G.HEADER_STRUCT.pack(*fields))
        pack.seek(directory_offset)
        pack.write(directory)
    G.write_experts_manifest(None, geometry, plan, pack_path, manifest_path,
                             FIRST_LAYER, LAYER_COUNT, TP_DEGREE, tp_rank)


RUNNER = (
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.insert(0, r'{tools}')\n"
    "import gemma4_stagepack as G\n"
    "G.GEOMETRY['{geo}'].update(hidden=256, vocab=1024, dense_inter=128,"
    " expert_inter=32)\n"
    "sys.exit(G.main())\n"
).format(tools=str(ROOT / "tools"), geo=GEOMETRY)


def run_verify(pack: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, "-c", RUNNER, "--output", str(pack), "--model",
         GEOMETRY, "--checkpoint", "/nonexistent", "--tp-degree",
         str(TP_DEGREE), "--tp-rank", "0", "--first-layer", str(FIRST_LAYER),
         "--layer-count", str(LAYER_COUNT), "--verify-existing", *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_receipt(pack: Path) -> dict:
    receipt = Path(str(pack) + ".verify-receipt.json")
    require(receipt.is_file(), f"no verify receipt beside {pack}")
    return json.loads(receipt.read_text())


def test_own_rank_verify_passes_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "stage2.gemma4sp"
        build_stage_pack(pack, Path(str(pack).rsplit(".", 1)[0] + ".experts"), 0)
        code, output = run_verify(pack, "--verify-ranks", "0")
        require(code == 0, output)
        require("proof=True" in output and "manifest_ok=True" in output, output)
        receipt = verify_receipt(pack)
        require(receipt["experts_manifest_ok"] is True, receipt)
        require(receipt["placement_proof"]["passed"] is True, receipt)


def test_wrong_rank_fails_on_expert_span():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "stage2.gemma4sp"
        build_stage_pack(pack, Path(str(pack).rsplit(".", 1)[0] + ".experts"), 0)
        code, output = run_verify(pack, "--verify-ranks", "1")
        require(code == 1, output)
        require("proof=True manifest_ok=False" in output,
                f"the placement proof holds and only the manifest fails: {output}")
        problems = verify_receipt(pack)["experts_manifest_problems"]
        require(any("outside the rank 1 span" in problem for problem in problems),
                f"expected an expert-span problem for rank 1: {problems}")


def test_rank_zero_plan_regression_is_gone():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "stage2.gemma4sp"
        build_stage_pack(pack, Path(str(pack).rsplit(".", 1)[0] + ".experts"), 3)
        code, output = run_verify(pack, "--verify-ranks", "3")
        require(code == 0, output)
        receipt = verify_receipt(pack)
        require(receipt["verify_ranks"] == [3], receipt["verify_ranks"])
        require(receipt["experts_manifest_problems"] == [], receipt["experts_manifest_problems"])


def test_directory_corruption_fails():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "stage2.gemma4sp"
        build_stage_pack(pack, Path(str(pack).rsplit(".", 1)[0] + ".experts"), 0)
        with pack.open("r+b") as handle:
            handle.seek(G.HEADER_BYTES + 3 * G.ENTRY_BYTES + 24)
            fields = struct.unpack("<I", handle.read(4))
            handle.seek(G.HEADER_BYTES + 3 * G.ENTRY_BYTES + 24)
            handle.write(struct.pack("<I", fields[0] + 8))
        code, output = run_verify(pack, "--verify-ranks", "0")
        require(code == 1, output)
        require("proof=False manifest_ok=True" in output,
                f"the corrupted directory fails the placement proof: {output}")
        proof = verify_receipt(pack)["placement_proof"]
        require(proof["passed"] is False and proof["ranks"]["0"]["mismatched_entries"] > 0, proof)


if __name__ == "__main__":
    shrink_geometry()
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
