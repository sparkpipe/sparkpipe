"""Shared oracle for the score-dump harness: run it, merge its rank files, check against numpy."""
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import score_merge
import score_kl_partial

TP = 16
ROWS = 40
HIDDEN = 64
WIDTH = 80
DOCUMENT_ROWS = 10
SEED = 20260929
TOLERANCE = 1.0e-5


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def corpus_from_tokens(tokens, path):
    with open(path, "w", encoding="utf-8") as handle:
        for doc in range(len(tokens) // DOCUMENT_ROWS):
            chunk = tokens[doc * DOCUMENT_ROWS:(doc + 1) * DOCUMENT_ROWS]
            handle.write(json.dumps({"doc": f"d{doc}", "tokens": [int(t) for t in chunk]}) + "\n")


def run_harness(binary, directory, probe="-", tier2="-"):
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), str(directory), str(TP), str(ROWS), str(HIDDEN), str(WIDTH), str(DOCUMENT_ROWS),
                    str(SEED), str(probe), str(tier2)], check=True, timeout=1800)
    return sorted(directory.glob("score.r*.bin"))


def numpy_logits(directory):
    hidden = bf16_to_f32((directory / "hidden.bf16").read_bytes()).reshape(ROWS, HIDDEN).astype(np.float64)
    head = bf16_to_f32((directory / "head.bf16").read_bytes()).reshape(TP * WIDTH, HIDDEN).astype(np.float64)
    return hidden @ head.T


def expected_top(logits):
    ids = np.arange(logits.shape[1], dtype=np.int64)
    return np.stack([ids[np.lexsort((ids, -row))][:score_merge.TOP_K] for row in logits])


def check_top_ids(merged_ids, reference_logits):
    expected = expected_top(reference_logits)
    for row in range(len(merged_ids)):
        got = merged_ids[row].astype(np.int64)
        if np.array_equal(got, expected[row]):
            continue
        values_got = reference_logits[row][got]
        values_expected = reference_logits[row][expected[row]]
        if np.max(np.abs(values_got - values_expected)) > TOLERANCE:
            raise AssertionError(f"row {row}: top-k ids differ beyond a numerical tie")


def check_pipeline(binary, work):
    work = pathlib.Path(work)
    first = run_harness(binary, work / "pass1")
    tokens = np.frombuffer((work / "pass1" / "tokens.u32").read_bytes(), dtype="<u4")
    corpus = work / "corpus.jsonl"
    corpus_from_tokens(tokens, corpus)
    assert score_merge.main(["targets", str(corpus), "--out", str(work / "targets.bin")]) == 0
    assert score_merge.main(["tier2", str(corpus), "--count", "7", "--seed", str(SEED), "--out", str(work / "tier2.bin")]) == 0
    reference = numpy_logits(work / "pass1")
    log_z = np.log(np.sum(np.exp(reference - reference.max(axis=1, keepdims=True)), axis=1)) + reference.max(axis=1)
    dumps = [score_merge.read_rank_file(p) for p in first]
    dumps, merged, _ = score_merge.merge(dumps)
    assert np.max(np.abs(merged["log_z"] - log_z)) <= TOLERANCE, np.max(np.abs(merged["log_z"] - log_z))
    got = merged["top_logits"].astype(np.float64) - merged["log_z"][:, None]
    want = np.take_along_axis(reference, merged["top_ids"].astype(np.int64), axis=1) - log_z[:, None]
    assert np.max(np.abs(got - want)) <= TOLERANCE
    check_top_ids(merged["top_ids"], reference)

    second = run_harness(binary, work / "pass2", work / "targets.bin", work / "tier2.bin")
    merged_path = work / "ref.merged"
    assert score_merge.main(["merge", *map(str, second), "--probe", str(work / "targets.bin"),
                             "--out", str(merged_path), "--probe-out", str(work / "probe2.bin")]) == 0
    again = work / "ref.again"
    assert score_merge.main(["merge", *map(str, reversed(second)), "--probe", str(work / "targets.bin"),
                             "--out", str(again)]) == 0
    assert merged_path.read_bytes() == again.read_bytes(), "merge is not byte-reproducible"
    _, merged2, probes2 = score_merge.read_merged(merged_path)
    for doc in range(ROWS // DOCUMENT_ROWS):
        for position in range(DOCUMENT_ROWS - 1):
            row = doc * DOCUMENT_ROWS + position
            target = int(tokens[row + 1])
            ids = probes2[row]["id"].tolist()
            assert target in ids, (row, target, ids)
            logit = float(probes2[row]["logit"][ids.index(target)])
            assert abs((logit - merged2["log_z"][row]) - (reference[row, target] - log_z[row])) <= TOLERANCE
    tier2_rows = int(((merged2["flags"] & score_merge.ROW_TIER2) != 0).sum())
    assert tier2_rows == 7, tier2_rows

    third = run_harness(binary, work / "pass3", work / "probe2.bin", work / "tier2.bin")
    arm_path = work / "arm.merged"
    assert score_merge.main(["merge", *map(str, third), "--probe", str(work / "probe2.bin"), "--out", str(arm_path)]) == 0
    report = work / "compare.json"
    assert score_kl_partial.main(["compare", str(merged_path), str(arm_path), str(corpus), "--out", str(report)]) == 0
    summary = json.loads(report.read_text())["summary"]
    assert summary["kl_b_mean"] == 0.0 and summary["dlogp_mean"] == 0.0 and summary["decisive_flips"] == 0, summary
    partials = []
    for rank in range(TP):
        out = work / f"partial.r{rank:02d}.bin"
        assert score_kl_partial.main(["partial", "--ref-tier2", str(work / "pass2" / f"tier2.r{rank:02d}.bin"),
                                      "--arm-tier2", str(work / "pass3" / f"tier2.r{rank:02d}.bin"),
                                      "--ref-merged", str(merged_path), "--arm-merged", str(arm_path),
                                      "--out", str(out)]) == 0
        partials.append(str(out))
    exact = work / "exact.json"
    assert score_kl_partial.main(["combine", *reversed(partials), "--out", str(exact)]) == 0
    exact_report = json.loads(exact.read_text())
    assert exact_report["kl_mean"] == 0.0 and len(exact_report["rows"]) == 7
    assert exact_report["max_mass_error"] <= 1.0e-9
    return {"rows": ROWS, "tp": TP, "log_z_max_error": float(np.max(np.abs(merged["log_z"] - log_z)))}
