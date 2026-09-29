#!/usr/bin/env python3
import hashlib
import io
import json
import random
import struct
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm52_expert_graft as front
import glm5_next_expert_graft as engine
import pack_spine_sha
from glm52_model_contract import load_model_contract

LAYOUT = front.GLM52_LAYOUT
HEADER_BYTES = 264
ENTRY_BYTES = 64
GLOBAL = 0xFFFFFFFF
BF16, FP8 = 1, 5
PAYLOAD_BF16, PAYLOAD_PACKED = 1, 4
SCALE_NONE, SCALE_F32 = 0, 1
EXPERT_UP_GATE, EXPERT_DOWN = 22, 23
SPINE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
EXPERT_REVISION = "935644c05e76fc198714f4cca449fd8b970ff6d7"
U2_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
U2_CONTRACT = "2ed1dad883ec799c736dc61c82404e73bd787dde3e0749e1db09a253be470bd7"


def entries(codec, layers=(3, 4), groups=4, rows=6, columns=256):
    rows_out = [(3, 3, PAYLOAD_BF16, BF16, SCALE_NONE, 1, 1, 64, 128, 0),
                (20, 3, PAYLOAD_BF16, BF16, SCALE_NONE, 1, 4, 64, 512, 0)]
    for layer in layers:
        for kind in (EXPERT_UP_GATE, EXPERT_DOWN):
            if codec == BF16:
                rows_out.append((kind, layer, PAYLOAD_BF16, BF16, SCALE_NONE, groups, rows, columns,
                                 groups * rows * columns * 2, 0))
            else:
                rows_out.append((kind, layer, PAYLOAD_PACKED, FP8, SCALE_F32, groups, rows, columns,
                                 groups * rows * columns, groups * rows * (columns // 128) * 4))
        rows_out.append((24, layer, PAYLOAD_BF16, BF16, SCALE_NONE, 1, 2, 64, 256, 0))
    rows_out.append((0, GLOBAL, PAYLOAD_BF16, BF16, SCALE_NONE, 1, 2, 32, 128, 0))
    return rows_out


def write_pack(path, rows_in, seed, codec, revision, contract, stage=(1, 0), tp=(16, 5),
               geometry=LAYOUT["geometry"], magic=LAYOUT["magic"], source_config=b"\x07" * 32,
               recipe=b"\x08" * 32):
    rng = random.Random(seed)
    directory_offset = engine.align(HEADER_BYTES)
    cursor = directory_offset + len(rows_in) * ENTRY_BYTES
    rows = []
    for entry in rows_in:
        payload_offset = engine.align(cursor)
        cursor = payload_offset + entry[8]
        scale_offset = 0
        if entry[9]:
            scale_offset = engine.align(cursor)
            cursor = scale_offset + entry[9]
        rows.append(entry[:8] + (payload_offset, entry[8], scale_offset, entry[9]))
    hidden, vocab, experts, layers = geometry
    header = struct.pack("<20I2Q65s32s32s32s", magic, LAYOUT["format_version"], HEADER_BYTES, ENTRY_BYTES,
                         1, 0, len(rows), stage[0], stage[1], 0, layers, layers, hidden, vocab, experts,
                         BF16, codec, BF16, tp[0], tp[1], directory_offset, cursor,
                         revision.encode().ljust(65, b"\0"), bytes.fromhex(contract), source_config, recipe)
    header += bytes(HEADER_BYTES - len(header))
    data = bytearray(cursor)
    data[:HEADER_BYTES] = header
    for index, row in enumerate(rows):
        struct.pack_into("<8I4Q", data, directory_offset + index * ENTRY_BYTES, *row)
        for offset, count in ((row[8], row[9]), (row[10], row[11])):
            data[offset:offset + count] = bytes(rng.getrandbits(8) for _ in range(count))
    path.write_bytes(bytes(data))


def regions(path):
    pack = engine.RankPack(path, LAYOUT)
    data = path.read_bytes()
    return pack, {(e["kind"], e["layer"]): (data[e["payload_offset"]:e["payload_offset"] + e["payload_bytes"]],
                                            data[e["scale_offset"]:e["scale_offset"] + e["scale_bytes"]])
                  for e in pack.entries}


def run_cli(arguments):
    out, err = io.StringIO(), io.StringIO()
    with patch.object(sys, "argv", ["glm52_expert_graft.py", *arguments]), redirect_stdout(out), redirect_stderr(err):
        try:
            code = front.main()
        except SystemExit as exit_error:
            code = exit_error.code
    return code, out.getvalue() + err.getvalue()


def c_header_offsets():
    source = ('#include <stddef.h>\n#include <stdio.h>\n#include "spark_glm52_stagepack_format.h"\n'
              'int main(void){printf("%zu %zu %zu %zu %zu\\n",offsetof(SparkGlm52StagePackHeader,model_revision),'
              'offsetof(SparkGlm52StagePackHeader,contract_sha256),offsetof(SparkGlm52StagePackHeader,pack_recipe_sha256),'
              'offsetof(SparkGlm52StagePackHeader,stage_count),sizeof(SparkGlm52StagePackHeader));return 0;}\n')
    with tempfile.TemporaryDirectory() as directory:
        program = Path(directory) / "offsets.c"
        program.write_text(source)
        binary = Path(directory) / "offsets"
        subprocess.run(["cc", "-std=c11", "-Iinclude", "-Imodel-families/common/include", "-Imodel-families/glm52/include",
                        "-Imodules/glm52_resident_decode_stage/source", str(program), "-o", str(binary)],
                       cwd=ROOT, check=True, capture_output=True)
        return [int(value) for value in subprocess.run([str(binary)], check=True, capture_output=True,
                                                       text=True).stdout.split()]


def main():
    failures = []
    contract = load_model_contract(ROOT)
    geometry = (contract["hidden_dimension"], contract["output_vocab_count"], contract["moe_expert_count"],
                contract["layer_count"])
    if LAYOUT["geometry"] != geometry or LAYOUT["magic"] != pack_spine_sha.LAYOUTS["glm52"]["magic"]:
        failures.append(f"glm52 graft layout {LAYOUT['geometry']} {LAYOUT['magic']:#x} is not the contract's {geometry}")
    offsets = c_header_offsets()
    if offsets != [engine.REVISION_OFFSET, engine.CONTRACT_OFFSET, engine.RECIPE_OFFSET, engine.STAGE_OFFSET, HEADER_BYTES]:
        failures.append(f"graft header offsets differ from SparkGlm52StagePackHeader: {offsets}")
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        (base / "spine").mkdir()
        (base / "experts").mkdir()
        (base / "u2").mkdir()
        spine_path, expert_path = base / "spine/rank5.glm52sp", base / "experts/rank5.glm52sp"
        write_pack(spine_path, entries(BF16), 1, BF16, SPINE_REVISION, "575bd854" + "0" * 56, stage=(16, 5))
        write_pack(expert_path, entries(FP8), 2, FP8, EXPERT_REVISION, "6d9751b3" + "0" * 56,
                   source_config=b"\x09" * 32, recipe=b"\x0a" * 32)
        common = ["--spine-pack", str(spine_path), "--expert-pack", str(expert_path), "--expert-codec", "fp8",
                  "--tp-degree", "16", "--tp-rank", "5", "--model-revision", U2_REVISION]
        code, text = run_cli(common + ["--output", str(base / "u2/a.glm52sp"), "--contract-sha256", U2_CONTRACT])
        if code == 0 or "stage_count" not in text:
            failures.append(f"a stage 16/5 spine grafted without --restage: rc={code}")
        code, text = run_cli(common + ["--output", str(base / "u2/b.glm52sp")])
        if code == 0:
            failures.append("a glm52 graft ran without --contract-sha256")
        code, text = run_cli(common + ["--output", str(base / "u2/c.glm52sp"), "--contract-sha256", "XYZ", "--restage"])
        if code == 0 or "64 lowercase hex" not in text:
            failures.append("a malformed contract digest was accepted")
        code, text = run_cli(common + ["--output", str(base / "u2/d.glm52sp"), "--contract-sha256", "G" * 64, "--restage"])
        if code == 0 or "64 lowercase hex" not in text:
            failures.append("a 64-character non-hex contract digest was accepted")
        code, text = run_cli(common + ["--output", str(base / "u2/e.glm52sp"), "--contract-sha256", U2_CONTRACT.upper(),
                                       "--restage"])
        if code == 0 or "64 lowercase hex" not in text:
            failures.append("an uppercase contract digest was accepted")
        flagged = base / "experts/flags.glm52sp"
        write_pack(flagged, entries(FP8), 2, FP8, EXPERT_REVISION, "6d9751b3" + "0" * 56)
        flagged_raw = bytearray(flagged.read_bytes())
        struct.pack_into("<I", flagged_raw, 5 * 4, 1)
        flagged.write_bytes(bytes(flagged_raw))
        code, text = run_cli(["--spine-pack", str(spine_path), "--expert-pack", str(flagged), "--expert-codec", "fp8",
                              "--tp-degree", "16", "--tp-rank", "5", "--model-revision", U2_REVISION,
                              "--contract-sha256", U2_CONTRACT, "--restage", "--output", str(base / "u2/h.glm52sp")])
        if code == 0 or "header flags" not in text:
            failures.append("--restage waived a header field other than stage_count/stage_index")
        output = base / "u2/rank5.glm52sp"
        code, text = run_cli(common + ["--output", str(output), "--contract-sha256", U2_CONTRACT, "--restage",
                                       "--arm", "U2"])
        if code != 0 or "GRAFT-PASS" not in text:
            failures.append(f"U2 graft failed: rc={code} {text[-300:]}")
            return report(failures)
        raw = output.read_bytes()
        words = struct.unpack_from("<20I", raw, 0)
        revision = raw[96:161].split(b"\0")[0].decode()
        receipt = json.loads(Path(str(output) + ".receipt.json").read_text())
        spine_raw = spine_path.read_bytes()
        if words[0] != LAYOUT["magic"] or words[16] != FP8 or revision != U2_REVISION:
            failures.append(f"output header magic/codec/revision {words[0]:#x}/{words[16]}/{revision}")
        if raw[161:193].hex() != U2_CONTRACT:
            failures.append("output header contract is not the U2 contract")
        if raw[193:225] != spine_raw[193:225]:
            failures.append("output source_config_sha256 is not the spine pack's")
        if raw[225:257] == bytes(32) or raw[225:257].hex() != receipt["pack_recipe_sha256"] or raw[225:257] == spine_raw[225:257]:
            failures.append("output pack_recipe_sha256 is not the graft recipe")
        if (words[7], words[8]) != (1, 0) or receipt["stage"] != {"count": 1, "index": 0, "spine_header": [16, 5]}:
            failures.append(f"restaged output stage {words[7]}/{words[8]} receipt {receipt.get('stage')}")
        if receipt["kind"] != LAYOUT["receipt_kind"] or receipt["contract_sha256"] != U2_CONTRACT:
            failures.append(f"receipt kind/contract {receipt['kind']} {receipt.get('contract_sha256')}")
        _, got = regions(output)
        _, spine_regions = regions(spine_path)
        _, expert_regions = regions(expert_path)
        for key, planes in got.items():
            want = expert_regions[key] if key[0] in (EXPERT_UP_GATE, EXPERT_DOWN) else spine_regions[key]
            if planes != want:
                failures.append(f"entry kind={key[0]} layer={key[1]:#x} bytes differ from its source")
        spine_digest = pack_spine_sha.spine_digest(spine_path, "glm52")["spine_digest"]
        if receipt["spine_digest"] != spine_digest or pack_spine_sha.spine_digest(output, "glm52")["spine_digest"] != spine_digest:
            failures.append("spine digest differs from pack_spine_sha.py --layout glm52 of the spine pack")
        code, text = run_cli(["--spine-digest", str(output)])
        if code != 0 or not text.startswith(spine_digest):
            failures.append("--spine-digest does not print the pack_spine_sha digest")
        again = base / "u2/again.glm52sp"
        run_cli(common + ["--output", str(again), "--contract-sha256", U2_CONTRACT, "--restage", "--arm", "U2"])
        if again.read_bytes() != raw:
            failures.append("two identical grafts produced different packs")
        wrong = base / "experts/geometry.glm52sp"
        write_pack(wrong, entries(FP8), 2, FP8, EXPERT_REVISION, "6d9751b3" + "0" * 56,
                   geometry=(4096,) + LAYOUT["geometry"][1:])
        code, text = run_cli(["--spine-pack", str(spine_path), "--expert-pack", str(wrong), "--expert-codec", "fp8",
                              "--tp-degree", "16", "--tp-rank", "5", "--model-revision", U2_REVISION,
                              "--contract-sha256", U2_CONTRACT, "--restage", "--output", str(base / "u2/g.glm52sp")])
        if code == 0 or "model geometry" not in text:
            failures.append("an expert pack with another model's geometry was grafted")
        flash = base / "experts/flash.glm52sp"
        write_pack(flash, entries(FP8), 2, FP8, EXPERT_REVISION, "6d9751b3" + "0" * 56, magic=0x33584C47)
        code, text = run_cli(["--spine-pack", str(spine_path), "--expert-pack", str(flash), "--expert-codec", "fp8",
                              "--tp-degree", "16", "--tp-rank", "5", "--model-revision", U2_REVISION,
                              "--contract-sha256", U2_CONTRACT, "--restage", "--output", str(base / "u2/f.glm52sp")])
        if code == 0 or "not a v3 glm52 rank pack" not in text:
            failures.append("a glm5_next pack was accepted as a glm52 expert pack")
        try:
            engine.graft(spine_path, expert_path, base / "u2/n.glm52sp", 16, 5, "fp8", U2_REVISION,
                         contract_sha256=U2_CONTRACT)
            failures.append("the glm5_next layout accepted a contract stamp")
        except engine.PackFailure:
            pass
        leftovers = sorted(path.name for path in (base / "u2").iterdir() if path.name.endswith(".partial"))
        if leftovers:
            failures.append(f"refused grafts left partial files: {leftovers}")
    return report(failures)


def report(failures):
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS glm52 expert graft: header offsets match the C format, U2 contract/revision/recipe stamped, "
          "restage explicit, spine and expert bytes verbatim, spine digest equals pack_spine_sha, refusals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
