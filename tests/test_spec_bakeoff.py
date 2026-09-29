#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_bakeoff as bakeoff  # noqa: E402
import spec_roofline  # noqa: E402

W2_OFF = [("prose", 0, 336, 7.5423), ("prose", 1, 511, 11.7269), ("prose", 2, 511, 11.7788), ("code", 0, 511, 11.7835), ("code", 1, 511, 11.8097),
          ("code", 2, 255, 5.778), ("repetitive", 0, 40, 0.979), ("repetitive", 1, 511, 11.6789), ("repetitive", 2, 428, 9.7159)]
W2_MTP3 = [("prose", 0, 336, 9.4609), ("prose", 1, 511, 14.6714), ("prose", 2, 511, 14.8034), ("code", 0, 511, 15.0421), ("code", 1, 511, 15.173),
           ("code", 2, 255, 7.281), ("repetitive", 0, 40, 1.289), ("repetitive", 1, 511, 14.8978), ("repetitive", 2, 428, 12.451)]
W2_ORACLE_REPLAY = {"exact": True, "expected_tokens": 256, "output_tokens": 256, "first_mismatch": None, "ttft_s": 0.078, "decode_tok_s": 84.32124197328015}
W2_MTP3_LOG = ["VERIFY-FRAME slot=2 position=38 budget=8 produced=8 rounds=7 accepted=0 steps=1 | frames=1 plain=5 rounds=7 proposed=8 accepted=0 steps=1 tokens=8",
               "VERIFY-POSITIONS p1=0/7 p2=0/0 p3=0/0 p4=0/0 p5=0/0 p6=0/0 p7=0/0",
               "VERIFY-FRAME slot=1 position=3620 budget=8 produced=8 rounds=7 accepted=0 steps=1 | frames=458 plain=5 rounds=3161 proposed=3171 accepted=2 steps=458 tokens=3621",
               "VERIFY-POSITIONS p1=2/3161 p2=0/2 p3=0/0 p4=0/0 p5=0/0 p6=0/0 p7=0/0"]


def check_arms() -> None:
    arm = bakeoff.parse_arm("glmflash/mtp:l45@fleet/chain-k3/B1")
    assert arm.model == "glmflash" and arm.method == "mtp" and arm.drafter == "l45" and arm.placement == "fleet" and arm.batch == 1
    assert arm.id == "glmflash/mtp:l45@fleet/chain-k3/B1" and "/" not in arm.slug
    assert bakeoff.parse_arm("glmflash/dspark:rh-flash-preview@rtx5090/chain-k7/B1").placement == "rtx5090"
    assert bakeoff.parse_arm("k3/off@fleet/chain-k0/B8").method == "off"
    assert bakeoff.parse_arm("glmfull/mtp+lookup:l78@fleet/adaptive/B2").method == "mtp+lookup"
    assert bakeoff.parse_arm("glmflash/tree:mtp-suffix@rtx5090/trie-mtp+suffix-r8/B1").shape == "trie-mtp+suffix-r8"
    for bad in ("glmflash/mtp@fleet/chain-k3/B1", "glmflash/lookup:x@fleet/chain-k3/B1", "glmflash/off@fleet/chain-k3/B1", "glmflash/lookup@fleet/chain-k0/B1",
                "glmflash/lookup@moon/chain-k3/B1", "glmflash/lookup@fleet/chain-k3/B3", "glmflash/warp@fleet/chain-k3/B1", "glmflash lookup"):
        try:
            bakeoff.parse_arm(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")
    assert bakeoff.baseline_arms("glmflash", 1)[:2] == ["glmflash/off@fleet/chain-k0/B1"] * 2


def run_json(rows, ids_shift: int = 0, drop_ids: bool = False) -> dict:
    results = []
    for content_class, index, tokens, seconds in rows:
        ids = [((index + 1) * 1000 + position + ids_shift) % 154880 for position in range(tokens + 1)]
        results.append({"class": content_class, "index": index, "decode_tokens": tokens, "decode_s": seconds, "decode_tok_s": tokens / seconds,
                        "ttft_s": 0.3, "wall_s": seconds + 0.3, "text": "x" * tokens, "token_ids": None if drop_ids else ids, "output_tokens": tokens + 1})
    return {"label": "t", "max_tokens": 512, "results": results}


def refused(argv) -> bool:
    try:
        bakeoff.main(argv)
    except SystemExit as error:
        return error.code not in (0, None)
    except ValueError:
        return True
    return False


def roofline_for(tok_s: float, rows: int = 1, tau: float = 1.0) -> str:
    registry = spec_roofline.load_registry()
    return spec_roofline.roofline(registry, "glmflash", 1, rows, 1000.0 * tau / tok_s, tau, 6.5)["line"]


def check_roofline_lines() -> None:
    registry = spec_roofline.load_registry()
    line = spec_roofline.roofline(registry, "glmflash", 1, 1, 25.2, 1.0, 4.7)
    assert 41.0 < line["memory_pct"] < 43.0 and 93.0 < line["ceiling_tok_s"] < 97.0, line
    assert 75.0 < line["transport_latency_pct"] < 82.0 and line["transport_bw_pct"] < 2.0, line
    assert spec_roofline.is_roofline_line(line["line"])
    wave = spec_roofline.roofline(registry, "glmflash", 1, 8, 62.5, 8.0, 19.7)
    assert 36.0 < wave["memory_pct"] < 44.0 and 320.0 < wave["ceiling_tok_s"] < 370.0, wave
    assert wave["line"].startswith("roofline @B=1,rows=8: memory") and spec_roofline.is_roofline_line(wave["line"])
    unseparated = spec_roofline.roofline(registry, "glmfull", 1, 1, 40.0, 1.0)
    assert "transport not separated" in unseparated["line"] and spec_roofline.is_roofline_line(unseparated["line"])
    assert 47.0 < unseparated["memory_pct"] < 52.0
    assert not spec_roofline.is_roofline_line("memory 42% compute 1%")
    assert not spec_roofline.is_roofline_line("roofline @B=1: memory 42% (ceiling 95 tok/s) | compute 1%")
    assert spec_roofline.main(["--model", "glmflash", "--tok-s", "43.7", "--collective-ms", "6.5"]) == 0


def ingest(root: Path, arm: str, window: str, run: dict | None = None, roofline: str | None = None, log: list[str] | None = None, repeat: int = 1, **extra) -> None:
    argv = ["ingest", "--arm", arm, "--window", window, "--out-root", str(root), "--repeat", str(repeat)]
    root.mkdir(parents=True, exist_ok=True)
    if run is not None:
        path = root / f"{arm.replace('/', '_')}.{repeat}.run.json"
        path.write_text(json.dumps(run))
        argv += ["--run", str(path)]
    if roofline is not None:
        argv += ["--roofline", roofline]
    if log is not None:
        path = root / f"{arm.replace('/', '_')}.{repeat}.log"
        path.write_text("\n".join(log) + "\n")
        argv += ["--log", str(path)]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert bakeoff.main(argv) == 0, argv


def report(root: Path) -> tuple[int, dict]:
    out = root / "ledger.json"
    try:
        code = bakeoff.main(["report", "--receipts", str(root / "receipts"), "--out", str(out), "--markdown", str(root / "ledger.md")])
    except SystemExit as error:
        code = error.code
    return code, json.loads(out.read_text())


def check_report(directory: Path) -> None:
    receipts = directory / "receipts"
    window = "20260929T1039Z-w2"
    off_line = roofline_for(43.74)
    ingest(receipts, "glmflash/off@fleet/chain-k0/B1", window, run_json(W2_OFF), off_line, firmware="a597ff0")
    ingest(receipts, "glmflash/off@fleet/chain-k0/B1", window, run_json(W2_OFF), off_line, repeat=2)
    mtp_line = roofline_for(34.88)
    ingest(receipts, "glmflash/mtp:l45@fleet/chain-k3/B1", window, run_json(W2_MTP3), mtp_line, W2_MTP3_LOG, admission_id="SPECULATOR-LICENSE-ADMISSION-20260828/glm53flash-mtp", drafter_sha256="ab" * 32, memavailable_min_gib=36.0)
    code, ledger = report(directory)
    assert code == 0
    prose = next(row for row in ledger["ledger"]["rows"] if row["class"] == "prose")
    assert prose["no_spec_tok_s"] == 43.74 and prose["spec_tok_s"] == 34.88 and abs(prose["ratio"] - 0.7974) < 1e-3
    assert next(row for row in ledger["ledger"]["rows"] if row["class"] == "code")["no_spec_tok_s"] == 43.48
    assert prose["acceptance_per_position"][0] == 2 / 3161 and prose["acceptance_per_position"][2] is None
    assert prose["gates"]["G1_exactness"] == "PASS" and prose["gates"]["G2_determinism"] == "not evaluated (one run)"
    assert prose["gates"]["G4_regression_floor"].startswith("not eligible") and prose["gates"]["G5_memory"] == "PASS" and prose["gates"]["G7_licence"] == "PASS"
    assert prose["gates"]["G8_drafter_correctness"] == "not evaluated" and prose["gates"]["G3_offline_online"] == "not evaluated"
    assert ledger["ledger"]["windows"][f"glmflash/{window}/B1"]["G2_off_determinism"] == "PASS"
    assert {row["class"]: row["no_spec_tok_s"] for row in ledger["ledger"]["no_spec"] if row["repeat"] == 1} == {"prose": 43.74, "code": 43.48, "repetitive": 43.76}
    verdict = ledger["verdicts"]["glmflash/mtp:l45@fleet/chain-k3/B1"]
    assert not verdict["eligible"] and 0.78 < verdict["geometric_mean_ratio"] < 0.80
    text = (directory / "ledger.md").read_text()
    assert "| spec tok/s | no-spec tok/s |" in text and "34.88 | 43.74" in text and "roofline @B=1" in text
    ingest(receipts, "glmflash/lookup@fleet/chain-k7/B1", window, run_json(W2_MTP3, ids_shift=1), roofline_for(30.0))
    code, ledger = report(directory)
    assert code == 1
    broken = [row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/lookup")]
    assert broken and all(row["gates"]["G1_exactness"] == "FAIL" and row["evidence_only"] for row in broken)
    assert len(broken[0]["gates"]["G1_mismatches"]) == 9 and not ledger["verdicts"]["glmflash/lookup@fleet/chain-k7/B1"]["eligible"]
    one_token = run_json(W2_MTP3)
    one_token["results"][4]["token_ids"][100] += 1
    ingest(receipts, "glmflash/suffix@rtx5090/chain-k5/B1", window, one_token, roofline_for(50.0))
    code, ledger = report(directory)
    suffix = [row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/suffix")]
    assert suffix[0]["gates"]["G1_exactness"] == "FAIL" and suffix[0]["gates"]["G1_mismatches"] == ["code[1]"]
    ingest(receipts, "glmflash/ngram3@rtx5090/chain-k3/B1", window, run_json(W2_MTP3))
    code, ledger = report(directory)
    assert any(item["arm"].startswith("glmflash/ngram3") and item["reason"] == "no roofline line" for item in ledger["ledger"]["refused"])
    assert not any(row["arm"].startswith("glmflash/ngram3") for row in ledger["ledger"]["rows"])
    ingest(receipts, "glmflash/lookup@fleet/chain-k3/B1", "20260929T2359Z-orphan", run_json(W2_MTP3), roofline_for(40.0))
    code, ledger = report(directory)
    assert any(item["window"] == "20260929T2359Z-orphan" and "no matched off" in item["reason"] for item in ledger["ledger"]["refused"])
    ingest(receipts, "glmflash/dspark:rh@rtx5090/chain-k7/B1", window, run_json(W2_MTP3), roofline_for(60.0))
    code, ledger = report(directory)
    dspark = next(row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/dspark"))
    assert dspark["gates"]["G7_licence"] == "FAIL" and not ledger["verdicts"]["glmflash/dspark:rh@rtx5090/chain-k7/B1"]["eligible"]
    assert refused(["ingest", "--arm", "glmflash/lookup@fleet/chain-k3/B1", "--window", window, "--out-root", str(receipts), "--run", write_run(directory, run_json(W2_OFF, drop_ids=True))])
    assert refused(["ingest", "--arm", "glmflash/lookup@fleet/chain-k3/B1", "--window", window, "--out-root", str(receipts), "--run", write_run(directory, run_json(W2_OFF)), "--roofline", "memory 40%"])
    assert refused(["ingest", "--arm", "glmflash/nope@fleet/chain-k3/B1", "--window", window, "--out-root", str(receipts), "--run", write_run(directory, run_json(W2_OFF))])


def check_matching(directory: Path) -> None:
    receipts = directory / "receipts"
    window = "20260929T2000Z-sb1"
    fast = [(content_class, index, tokens, seconds / 6) for content_class, index, tokens, seconds in W2_OFF]
    ingest(receipts, "glmflash/off@fleet/chain-k0/B1", window, run_json(W2_OFF), roofline_for(43.74))
    ingest(receipts, "glmflash/off@fleet/chain-k0/B8", window, run_json(fast), roofline_for(262.0))
    ingest(receipts, "glmflash/lookup@rtx5090/chain-k3/B8", window, run_json(fast), roofline_for(262.0))
    ingest(receipts, "glmflash/suffix@rtx5090/chain-k3/B2", window, run_json(W2_OFF), roofline_for(43.74))
    code, ledger = report(directory)
    batched = [row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/lookup")]
    assert code == 0 and batched and all(row["ratio"] == 1.0 and row["no_spec_tok_s"] > 200 for row in batched), batched
    assert any(item["arm"].startswith("glmflash/suffix") and "B2" in item["reason"] for item in ledger["ledger"]["refused"])
    changed = run_json(W2_OFF)
    changed["results"][2]["token_ids"][7] += 1
    ingest(receipts, "glmflash/lookup@rtx5090/chain-k7/B1", window, run_json(W2_OFF), roofline_for(43.74))
    ingest(receipts, "glmflash/lookup@rtx5090/chain-k7/B1", window, changed, roofline_for(43.74), repeat=2)
    code, ledger = report(directory)
    repeated = next(row for row in ledger["ledger"]["rows"] if row["arm"] == "glmflash/lookup@rtx5090/chain-k7/B1")
    assert code == 1 and repeated["gates"]["G1_exactness"] == "FAIL" and repeated["gates"]["G1_mismatches"] == ["prose[2]"]
    extra = run_json(W2_OFF + [("chat", 0, 64, 1.5)])
    ingest(receipts, "glmflash/ngram3@rtx5090/chain-k3/B1", window, extra, roofline_for(43.74), memavailable_min_gib=12.0)
    code, ledger = report(directory)
    ngram = next(row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/ngram3"))
    assert ngram["gates"]["G1_exactness"] == "FAIL" and ngram["gates"]["G1_unmatched"] == ["chat[0]"] and ngram["gates"]["G5_memory"] == "FAIL"
    assert not ledger["verdicts"]["glmflash/ngram3@rtx5090/chain-k3/B1"]["eligible"]


def write_run(directory: Path, run: dict) -> str:
    path = directory / f"run{len(list(directory.iterdir()))}.json"
    path.write_text(json.dumps(run))
    return str(path)


def check_oracle_anchor(directory: Path) -> None:
    receipts = directory / "receipts"
    window = "20260929T1039Z-w2"
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    output = [(index * 31 + 7) % 154880 for index in range(256)]
    expect = directory / "spec_oracle.u32"
    expect.write_bytes(struct.pack(f"<II{len(prompt) + len(output)}I", len(prompt), len(prompt) + len(output), *prompt, *output))
    replay = directory / "oracle.replay.json"
    replay.write_text(json.dumps(W2_ORACLE_REPLAY))
    off_line = roofline_for(43.74)
    ingest(receipts, "glmflash/off@fleet/chain-k0/B1", window, run_json(W2_OFF), off_line)
    assert bakeoff.main(["ingest", "--arm", "glmflash/off@fleet/chain-k0/B1", "--window", window, "--out-root", str(receipts), "--repeat", "3", "--expect", str(expect), "--roofline", off_line]) == 0
    oracle_line = roofline_for(84.32, 8, 8.0)
    assert bakeoff.main(["ingest", "--arm", "glmflash/oracle@fleet/chain-k7/B1", "--window", window, "--out-root", str(receipts), "--replay-json", str(replay), "--expect", str(expect),
                         "--roofline", oracle_line, "--log", write_log(directory, ["VERIFY-POSITIONS p1=28/28 p2=28/28 p3=28/28 p4=28/28 p5=28/28 p6=28/28 p7=27/27"])]) == 0
    code, ledger = report(directory)
    anchor = next(row for row in ledger["ledger"]["rows"] if row["arm"].startswith("glmflash/oracle") and row["class"] == "anchor")
    assert anchor["spec_tok_s"] == 84.32 and anchor["no_spec_tok_s"] is None and anchor["gates"]["G1_exactness"] == "PASS"
    assert ledger["ledger"]["windows"][f"glmflash/{window}/B1"]["G2_off_determinism"] == "not evaluated (one off run)"
    other = directory / "other_oracle.u32"
    other.write_bytes(struct.pack(f"<II{len(prompt) + len(output)}I", len(prompt), len(prompt) + len(output), *prompt, *output[:-1], output[-1] + 1))
    assert bakeoff.main(["ingest", "--arm", "glmflash/off@fleet/chain-k0/B1", "--window", window, "--out-root", str(receipts), "--repeat", "4", "--expect", str(other), "--roofline", off_line]) == 0
    code, ledger = report(directory)
    assert ledger["ledger"]["windows"][f"glmflash/{window}/B1"]["G2_off_determinism"] == "FAIL: anchor[0]" and code == 1
    assert anchor["acceptance_per_position"] == [1.0] * 7
    bad = dict(W2_ORACLE_REPLAY, exact=False)
    replay.write_text(json.dumps(bad))
    assert refused(["ingest", "--arm", "glmflash/oracle@fleet/chain-k7/B1", "--window", window, "--out-root", str(receipts), "--replay-json", str(replay), "--expect", str(expect), "--roofline", oracle_line])


def write_log(directory: Path, lines: list[str]) -> str:
    path = directory / f"log{len(list(directory.iterdir()))}.log"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def check_plan(directory: Path) -> None:
    out = directory / "plan.json"
    assert bakeoff.main(["plan", "--model", "glmflash", "--arms", "mtp:l45", "lookup", "--placements", "rtx5090,offline", "--shapes", "chain-k3,adaptive", "--batches", "1,8", "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    arms = [row["arm"] for row in plan["arms"]]
    assert arms[:4] == bakeoff.baseline_arms("glmflash", 1) and arms.count("glmflash/off@fleet/chain-k0/B1") == 2
    assert "glmflash/mtp:l45@rtx5090/adaptive/B8" in arms and "glmflash/lookup@offline/chain-k3/B1" in arms
    assert [row["repeat"] for row in plan["arms"][:2]] == [1, 2] and plan["classes"][:3] == ["prose", "code", "repetitive"]
    assert refused(["plan", "--model", "glmflash", "--arms", "mtp:l45", "--placements", "fleet", "--out", str(out)])
    assert bakeoff.main(["plan", "--model", "glmflash", "--arms", "suffix", "--out", str(out)]) == 0
    drafted = [row for row in json.loads(out.read_text())["arms"] if row["method"] not in bakeoff.FLEET_BASELINES]
    assert drafted and all(row["placement"] == "rtx5090" for row in drafted)


def main() -> int:
    check_arms()
    check_roofline_lines()
    with tempfile.TemporaryDirectory() as scratch:
        check_report(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        check_oracle_anchor(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        check_matching(Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        check_plan(Path(scratch))
    print("PASS spec_bakeoff: arm ids parse strictly, the W2 off/mtp3/oracle numbers (43.74, 34.88, 84.32, VERIFY-POSITIONS) re-report from receipts, "
          "one changed token in any run fails G1 and the report, off arms match by batch, drafter arms plan on the rtx5090, text without ids is refused, spec rows without a matched off arm or a roofline line are refused, "
          "and roofline lines follow ROOFLINE_REPORTING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
