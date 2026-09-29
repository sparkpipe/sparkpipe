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

SMALL = {"tp": 16, "rows": 40, "hidden": 64, "width": 80, "document_rows": 10, "tier2": 7}
SEED = 20260929


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def corpus_from_tokens(tokens, path, document_rows):
    with open(path, "w", encoding="utf-8") as handle:
        for doc in range(len(tokens) // document_rows):
            chunk = tokens[doc * document_rows:(doc + 1) * document_rows]
            handle.write(json.dumps({"doc": f"d{doc}", "tokens": [int(t) for t in chunk]}) + "\n")


def run_harness(binary, shape, directory, probe="-", tier2="-", fail_rank=None):
    directory.mkdir(parents=True, exist_ok=True)
    extra = [] if fail_rank is None else [str(fail_rank)]
    subprocess.run([str(binary), str(directory), str(shape["tp"]), str(shape["rows"]), str(shape["hidden"]),
                    str(shape["width"]), str(shape["document_rows"]), str(SEED), str(probe), str(tier2), *extra],
                   check=True, timeout=1800)
    return sorted(directory.glob("score.r*.bin"))


def load_inputs(directory, shape):
    rows, hidden_size = shape["rows"], shape["hidden"]
    hidden = bf16_to_f32((directory / "hidden.bf16").read_bytes()).reshape(rows, hidden_size)
    head = bf16_to_f32((directory / "head.bf16").read_bytes()).reshape(shape["tp"] * shape["width"], hidden_size)
    return hidden, head


def exact_logits(hidden, head):
    return np.concatenate([hidden.astype(np.float64) @ head[start:start + 8192].astype(np.float64).T
                           for start in range(0, head.shape[0], 8192)], axis=1)


def kernel_order_logits(hidden, head, lanes):
    rows, dimension = hidden.shape
    steps = dimension // lanes
    assert steps * lanes == dimension
    lane_rows = hidden.reshape(rows, steps, lanes)
    out = np.empty((rows, head.shape[0]), dtype=np.float32)
    for start in range(0, head.shape[0], 4096):
        weight = head[start:start + 4096].reshape(-1, steps, lanes)
        accumulator = np.zeros((rows, weight.shape[0], lanes), dtype=np.float32)
        for step in range(steps):
            accumulator += lane_rows[:, None, step, :] * weight[None, :, step, :]
        offset = lanes // 2
        while offset:
            accumulator = accumulator + accumulator[..., np.arange(lanes) ^ offset]
            offset //= 2
        out[:, start:start + weight.shape[0]] = accumulator[..., 0]
    return out


def log_partition(logits):
    values = logits.astype(np.float64)
    peak = values.max(axis=1)
    return peak + np.log(np.sum(np.exp(values - peak[:, None]), axis=1))


def expected_top(logits):
    ids = np.arange(logits.shape[1], dtype=np.int64)
    return np.stack([ids[np.lexsort((ids, -row.astype(np.float64)))][:score_merge.TOP_K] for row in logits])


def check_pipeline(binary, work, shape=SMALL, lanes=1):
    work = pathlib.Path(work)
    tp, rows, document_rows = shape["tp"], shape["rows"], shape["document_rows"]
    first = run_harness(binary, shape, work / "pass1")
    tokens = np.frombuffer((work / "pass1" / "tokens.u32").read_bytes(), dtype="<u4")
    corpus = work / "corpus.jsonl"
    corpus_from_tokens(tokens, corpus, document_rows)
    assert score_merge.main(["targets", str(corpus), "--out", str(work / "targets.bin")]) == 0
    assert score_merge.main(["tier2", str(corpus), "--count", str(shape["tier2"]), "--seed", str(SEED),
                             "--out", str(work / "tier2.bin")]) == 0
    hidden, head = load_inputs(work / "pass1", shape)
    reference = kernel_order_logits(hidden, head, lanes)
    log_z = log_partition(reference)
    accumulation = float(np.max(np.abs(reference.astype(np.float64) - exact_logits(hidden, head))))
    dumps = [score_merge.read_rank_file(p) for p in first]
    dumps, merged, _ = score_merge.merge(dumps)
    assert np.max(np.abs(merged["log_z"] - log_z)) <= 1.0e-9 * np.max(np.abs(log_z)), np.max(np.abs(merged["log_z"] - log_z))
    assert np.array_equal(merged["top_ids"].astype(np.int64), expected_top(reference)), "global top-k differs from numpy"
    assert np.array_equal(merged["top_logits"], np.take_along_axis(reference, merged["top_ids"].astype(np.int64), axis=1)), \
        "top-k logits are not bit-identical to the numpy kernel-order logits"

    second = run_harness(binary, shape, work / "pass2", work / "targets.bin", work / "tier2.bin")
    merged_path = work / "ref.merged"
    assert score_merge.main(["merge", *map(str, second), "--probe", str(work / "targets.bin"),
                             "--out", str(merged_path), "--probe-out", str(work / "probe2.bin")]) == 0
    again = work / "ref.again"
    assert score_merge.main(["merge", *map(str, reversed(second)), "--probe", str(work / "targets.bin"),
                             "--out", str(again)]) == 0
    assert merged_path.read_bytes() == again.read_bytes(), "merge is not byte-reproducible"
    _, merged2, probes2 = score_merge.read_merged(merged_path)
    for doc in range(rows // document_rows):
        for position in range(document_rows - 1):
            row = doc * document_rows + position
            target = int(tokens[row + 1])
            ids = probes2[row]["id"].tolist()
            assert target in ids, (row, target, ids)
            assert probes2[row]["logit"][ids.index(target)] == reference[row, target], (row, target)
    tier2_rows = int(((merged2["flags"] & score_merge.ROW_TIER2) != 0).sum())
    assert tier2_rows == shape["tier2"], tier2_rows
    tier2_logits = {}
    for path in sorted((work / "pass2").glob("tier2.r*.bin")):
        _, part = score_kl_partial.read_tier2(path)
        for identity, values in part.items():
            tier2_logits.setdefault(identity, []).append(values)
    for index in np.nonzero(merged2["flags"] & score_merge.ROW_TIER2)[0]:
        identity = (int(merged2["key"][index]), int(merged2["position"][index]))
        kernel = np.concatenate(tier2_logits[identity])
        assert np.array_equal(kernel.astype(np.float32), reference[index]), f"Tier-2 row {index} logits differ from numpy"
        ids = np.arange(kernel.size)
        exact = ids[np.lexsort((ids, -kernel))][:score_merge.TOP_K]
        assert np.array_equal(merged2["top_ids"][index], exact), index

    third = run_harness(binary, shape, work / "pass3", work / "probe2.bin", work / "tier2.bin")
    arm_path = work / "arm.merged"
    assert score_merge.main(["merge", *map(str, third), "--probe", str(work / "probe2.bin"), "--out", str(arm_path)]) == 0
    report = work / "compare.json"
    assert score_kl_partial.main(["compare", str(merged_path), str(arm_path), str(corpus), "--out", str(report)]) == 0
    summary = json.loads(report.read_text())["summary"]
    assert summary["kl_b_mean"] == 0.0 and summary["dlogp_mean"] == 0.0 and summary["decisive_flips"] == 0, summary
    partials = []
    for rank in range(tp):
        out = work / f"partial.r{rank:02d}.bin"
        assert score_kl_partial.main(["partial", "--ref-tier2", str(work / "pass2" / f"tier2.r{rank:02d}.bin"),
                                      "--arm-tier2", str(work / "pass3" / f"tier2.r{rank:02d}.bin"),
                                      "--ref-merged", str(merged_path), "--arm-merged", str(arm_path),
                                      "--out", str(out)]) == 0
        partials.append(str(out))
    exact = work / "exact.json"
    assert score_kl_partial.main(["combine", *reversed(partials), "--out", str(exact)]) == 0
    exact_report = json.loads(exact.read_text())
    assert exact_report["kl_mean"] == 0.0 and len(exact_report["rows"]) == shape["tier2"]
    assert exact_report["max_mass_error"] <= 1.0e-9
    failed = run_harness(binary, shape, work / "failed", fail_rank=tp // 2)
    try:
        score_merge.read_rank_file(failed[tp // 2])
    except score_merge.DumpError:
        pass
    else:
        raise AssertionError("a rank file marked failed was written with an end record")
    return {"rows": rows, "tp": tp, "lanes": lanes, "log_z_max_error": float(np.max(np.abs(merged["log_z"] - log_z))),
            "fp32_accumulation_vs_float64": accumulation}
