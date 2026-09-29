"""Score-dump export into the A/B formats: corpus order, per-document segments, permuted-run byte identity, exact KL rows.

Synthetic engine runs write real rank files (score_merge.write_rank_file) and
Tier-2 shards over a small vocabulary. Row logits are a function of the row's
prefix key, so documents that share a prefix score identical rows. The full
W1 score flow is exercised: corpus export, pass-1 targets, reference merge with
the pass-2 probe file, a permuted reference repeat, an arm run on the pass-2
probes, node-local exact-KL partials and their combination, and the A/B
comparison over the exports against float64 KL from the synthetic logits.
"""
import hashlib
import json
import pathlib
import struct
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ab_dry_run
import ab_dump
import ab_score_compare
import score_export
import score_kl_partial
import score_merge

TP = 4
WIDTH = 80
VOCAB = TP * WIDTH
WAVE = 8
TIER2_ROWS = 12
DOCS = [
    [5, 9, 13, 21, 34, 55, 89, 144, 233, 17, 4, 8, 15, 16, 23, 42, 7, 3, 99, 100],
    [5, 9, 13, 2, 71, 82, 93, 104, 115, 126, 137, 148, 159, 160, 171, 182, 193, 204, 215, 226],
    [300, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [300, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266],
    [5, 9, 13, 2, 71, 82, 93, 104, 115],
]
FOREIGN = [77, 78, 79, 80, 81, 82]


def write_corpus(directory):
    data = b"".join(struct.pack(f"<{len(d)}I", *d) for d in DOCS)
    (directory / "CT-test.tokens.u32").write_bytes(data)
    entries = []
    offset = 0
    for index, tokens in enumerate(DOCS):
        raw = struct.pack(f"<{len(tokens)}I", *tokens)
        entries.append({"id": f"doc-{index}", "stratum": "code" if index % 2 else "prose", "offset": offset,
                        "length": len(tokens), "tokens_sha256": hashlib.sha256(raw).hexdigest()})
        offset += len(tokens)
    index = {"format": "sparkpipe-ab-corpus-v1", "name": "CT-test", "corpus_sha256": hashlib.sha256(data).hexdigest(),
             "tokenizer_sha256": "ab" * 32, "documents": entries, "total_tokens": offset}
    path = directory / "CT-test.index.json"
    path.write_text(json.dumps(index, indent=1, sort_keys=True))
    return path, index


def row_logits(key, arm):
    generator = np.random.default_rng([int(key) & 0xFFFFFFFF, int(key) >> 32])
    logits = generator.standard_normal(VOCAB) * 3.0
    if arm:
        logits = logits + np.random.default_rng([int(key) & 0xFFFFFFFF, 7]).standard_normal(VOCAB) * 0.05
    return logits.astype(np.float32)


def local_stats(shard, base):
    order = np.lexsort((np.arange(shard.size), -shard.astype(np.float64)))[:score_merge.TOP_K]
    maximum = shard.max()
    return maximum, float(np.sum(np.exp(shard.astype(np.float64) - np.float64(maximum)))), order + base, shard[order]


def engine_run(directory, order, arm=False, probe_path=None, tier2_path=None, foreign_first=False, arm_digest=bytes(32), drop=None,
               nonfinite=None, bent_probe=None):
    directory.mkdir(parents=True)
    probe_table, probe_sha, _ = score_merge.read_table(probe_path, score_merge.PROBE_MAGIC) if probe_path else ({}, bytes(32), 0)
    tier2_table, tier2_sha, _ = score_merge.read_table(tier2_path, score_merge.TIER2_MAGIC) if tier2_path else ({}, bytes(32), 0)
    sequences = [DOCS[d] for d in order]
    sequences = ([FOREIGN] if foreign_first else []) + sequences
    records = []
    for tokens in sequences:
        keys = score_merge.row_keys(tokens)
        for position, key in enumerate(keys):
            records.append((key, position, tokens[position]))
    if drop is not None:
        del records[drop]
    rows_per_rank = [np.zeros(len(records), dtype=score_merge.ROW_DTYPE) for _ in range(TP)]
    probes_per_rank = [[] for _ in range(TP)]
    tier2_records = [[] for _ in range(TP)]
    for index, (key, position, token) in enumerate(records):
        logits = row_logits(key, arm)
        flags = score_merge.ROW_KEY_VALID
        wanted = probe_table.get((key, position))
        if wanted is not None:
            flags |= score_merge.ROW_PROBED
        if (key, position) in tier2_table:
            flags |= score_merge.ROW_TIER2
        for rank in range(TP):
            begin = rank * WIDTH
            maximum, total, ids, values = local_stats(logits[begin:begin + WIDTH], begin)
            row = rows_per_rank[rank][index]
            row["key"] = key
            row["position"] = position
            row["wave_ordinal"] = index // WAVE
            row["row_in_wave"] = index % WAVE
            row["input_token"] = token
            row["served_token"] = int(np.argmax(logits))
            row["flags"] = flags
            row["local_max"] = maximum
            row["local_sum_exp"] = total
            row["top_ids"] = ids
            row["top_logits"] = values
            owned = sorted(i for i in (wanted or []) if begin <= i < begin + WIDTH)
            probe = np.zeros(len(owned), dtype=score_merge.PROBE_DTYPE)
            probe["id"] = owned
            probe["logit"] = logits[owned]
            if index == bent_probe and rank == 0 and owned:
                probe["logit"][0] = logits[owned[0]] + np.float32(0.5)
            if index == nonfinite and rank == 0:
                row["flags"] = flags | score_merge.ROW_NONFINITE
            probes_per_rank[rank].append(probe)
            if flags & score_merge.ROW_TIER2:
                tier2_records[rank].append(struct.pack("<QII", key, position, WIDTH) + logits[begin:begin + WIDTH].astype("<f4").tobytes())
    for rank in range(TP):
        header = {"tp_rank": rank, "tp_degree": TP, "shard_begin": rank * WIDTH, "shard_end": (rank + 1) * WIDTH,
                  "vocabulary": VOCAB, "hidden_dimension": 64, "tier2": 1 if tier2_path else 0,
                  "arm_digest": arm_digest, "probe_sha256": probe_sha, "tier2_sha256": tier2_sha}
        score_merge.write_rank_file(directory / f"score.r{rank:02d}.bin", header, rows_per_rank[rank], probes_per_rank[rank])
        if tier2_path:
            with open(directory / f"tier2.r{rank:02d}.bin", "wb") as handle:
                handle.write(score_merge.HEADER.pack(score_merge.TIER2_OUT_MAGIC, score_merge.VERSION, score_merge.HEADER.size, rank, TP,
                                                     rank * WIDTH, (rank + 1) * WIDTH, VOCAB, 64, score_merge.TOP_K, 1, arm_digest,
                                                     probe_sha, tier2_sha))
                handle.write(b"".join(tier2_records[rank]))
    return sorted(directory.glob("score.r*.bin"))


def merge(ranks, out, probe=None, probe_out=None):
    args = ["merge", *map(str, ranks), "--out", str(out)]
    if probe:
        args += ["--probe", str(probe)]
    if probe_out:
        args += ["--probe-out", str(probe_out)]
    assert score_merge.main(args) == 0
    return out


def export(merged, index_path, arm_path, out, reference=None):
    args = ["dump", str(merged), "--corpus", str(index_path), "--arm", str(arm_path), "--out", str(out)]
    if reference:
        args += ["--reference", str(reference)]
    return score_export.main(args)


def expected_bucket_kl(key):
    p = row_logits(key, False).astype(np.float64)
    q = row_logits(key, True).astype(np.float64)
    lp = p - np.logaddexp.reduce(p)
    lq = q - np.logaddexp.reduce(q)
    ids = np.lexsort((np.arange(VOCAB), -p))[:score_merge.TOP_K]
    kl = float(np.sum(np.exp(lp[ids]) * (lp[ids] - lq[ids])))
    rest = np.setdiff1d(np.arange(VOCAB), ids)
    p_tail = np.logaddexp.reduce(lp[rest])
    q_tail = np.logaddexp.reduce(lq[rest])
    exact = float(np.sum(np.exp(lp) * (lp - lq)))
    return kl + float(np.exp(p_tail) * (p_tail - q_tail)), exact


def write_arm(directory, label):
    path = directory / f"{label}.arm.json"
    path.write_text(json.dumps(ab_dry_run.arm_descriptor("flash", "S1", "fp8", label, "publisher")))
    return path


def refused(args, needle):
    captured = tempfile.TemporaryFile(mode="w+")
    previous = sys.stderr
    sys.stderr = captured
    try:
        code = score_export.main(args)
    finally:
        sys.stderr = previous
    captured.seek(0)
    text = captured.read()
    assert code == 2 and needle in text, (args[0], needle, code, text)


def main():
    with tempfile.TemporaryDirectory(prefix="score-export-") as name:
        work = pathlib.Path(name)
        index_path, index = write_corpus(work)
        ref_arm = write_arm(work, "fp8")
        other_arm = write_arm(work, "fp8b")
        assert score_export.main(["corpus", str(index_path), "--out", str(work / "corpus.jsonl")]) == 0
        corpus_lines = [json.loads(line) for line in (work / "corpus.jsonl").read_text().splitlines()]
        assert [item["tokens"] for item in corpus_lines] == DOCS and corpus_lines[3]["doc"] == "doc-3"
        refused(["corpus", str(index_path), "--out", str(work / "corpus.jsonl")], "exists")
        assert score_merge.main(["targets", str(work / "corpus.jsonl"), "--out", str(work / "targets.bin")]) == 0
        assert score_merge.main(["tier2", str(work / "corpus.jsonl"), "--count", str(TIER2_ROWS), "--seed", "20260929",
                                 "--out", str(work / "tier2.bin")]) == 0

        natural = list(range(len(DOCS)))
        permuted = [3, 5, 0, 4, 2, 1]
        ref = merge(engine_run(work / "ref", natural, probe_path=work / "targets.bin", tier2_path=work / "tier2.bin"),
                    work / "ref.merged", work / "targets.bin", work / "probe2.bin")
        aa = merge(engine_run(work / "aa", permuted, probe_path=work / "targets.bin", tier2_path=work / "tier2.bin", foreign_first=True),
                   work / "aa.merged", work / "targets.bin")
        assert ref.read_bytes() != aa.read_bytes(), "the permuted run must differ as a raw merged dump"
        assert export(ref, index_path, ref_arm, work / "ref.npz") == 0
        assert export(aa, index_path, ref_arm, work / "aa.npz") == 0
        assert (work / "ref.npz").read_bytes() == (work / "aa.npz").read_bytes(), "permuted A/A export is not byte-identical"
        reference = ab_dump.read(work / "ref.npz")
        assert reference["header"]["probe_sha256"] is None
        assert reference["header"]["dump_arm_digest"] == "0" * 64
        assert len(reference["doc"]) == sum(len(d) - 1 for d in DOCS)
        assert reference["pos"][:3].tolist() == [1, 2, 3] and reference["target"][:3].tolist() == DOCS[0][1:4]
        first_doc3 = sum(len(d) - 1 for d in DOCS[:3])
        assert reference["doc"][first_doc3] == 3 and reference["target"][first_doc3] == DOCS[3][1]

        arm_ranks = engine_run(work / "arm", natural, arm=True, probe_path=work / "probe2.bin", tier2_path=work / "tier2.bin")
        arm = merge(arm_ranks, work / "arm.merged", work / "probe2.bin")
        assert export(arm, index_path, other_arm, work / "arm.npz", work / "ref.npz") == 0
        arm_dump = ab_dump.read(work / "arm.npz")
        assert arm_dump["header"]["probe_sha256"] == ab_dump.probe_sha256(reference["top_ids"])

        partials = []
        for rank in range(TP):
            out = work / f"partial.r{rank:02d}.bin"
            assert score_kl_partial.main(["partial", "--ref-tier2", str(work / "ref" / f"tier2.r{rank:02d}.bin"),
                                          "--arm-tier2", str(work / "arm" / f"tier2.r{rank:02d}.bin"),
                                          "--ref-merged", str(ref), "--arm-merged", str(arm), "--out", str(out)]) == 0
            partials.append(str(out))
        assert score_kl_partial.main(["combine", *partials, "--out", str(work / "exact.json")]) == 0
        assert score_export.main(["exact", str(work / "exact.json"), "--corpus", str(index_path), "--reference", str(work / "ref.npz"),
                                  "--arm", str(work / "arm.npz"), "--out", str(work / "exact.npz")]) == 0
        exact = ab_dump.read_exact(work / "exact.npz")
        assert exact["rows"].size == TIER2_ROWS and np.all(np.diff(exact["rows"].astype(np.int64)) > 0)
        occurrences = {}
        row = 0
        for d, tokens in enumerate(DOCS):
            for p, key in enumerate(score_merge.row_keys(tokens)[:-1]):
                occurrences.setdefault((key, p), []).append(row)
                row += 1
        chosen = sorted(occurrences[(int(e["key"]), int(e["position"]))][0] for e in json.loads((work / "exact.json").read_text())["rows"])
        assert exact["rows"].tolist() == chosen
        shared = [r for r in exact["rows"].tolist() if any(len(v) > 1 and r in v for v in occurrences.values())]
        assert shared, "the Tier-2 draw must include a row shared by two documents"

        result = ab_score_compare.compare(json.loads(index_path.read_text()), reference, arm_dump, None, exact,
                                          [0, 8, 64], 1.0e-3, 1.0)
        keys = [score_merge.row_keys(tokens) for tokens in DOCS]
        wanted = np.array([expected_bucket_kl(keys[d][p])[0] for d in range(len(DOCS)) for p in range(len(DOCS[d]) - 1)])
        metrics = ab_score_compare.row_metrics(reference, arm_dump, 1.0e-3, 1.0)
        error = float(np.max(np.abs(metrics["kl"] - wanted)))
        assert error <= 1.0e-9, error
        dnll_want = []
        for d in range(len(DOCS)):
            for p in range(len(DOCS[d]) - 1):
                lp = row_logits(keys[d][p], False).astype(np.float64)
                lq = row_logits(keys[d][p], True).astype(np.float64)
                y = DOCS[d][p + 1]
                dnll_want.append((np.logaddexp.reduce(lq) - lq[y]) - (np.logaddexp.reduce(lp) - lp[y]))
        dnll_error = float(np.max(np.abs(metrics["dnll"] - np.array(dnll_want))))
        assert dnll_error <= 1.0e-9, dnll_error
        exact_want = []
        for row in exact["rows"].tolist():
            d = int(reference["doc"][row])
            p = int(reference["pos"][row]) - 1
            exact_want.append(expected_bucket_kl(keys[d][p])[1])
        exact_error = float(np.max(np.abs(exact["kl"] - np.array(exact_want))))
        assert exact_error <= 1.0e-9, exact_error
        assert result["bucket_vs_exact"]["violations"] == 0 and result["rows"] == len(wanted)

        unprobed = merge(engine_run(work / "unprobed", natural), work / "unprobed.merged")
        refused(["dump", str(unprobed), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "unprobed.npz")],
                "no logit for corpus target")
        refused(["dump", str(merge(engine_run(work / "arm1", natural, arm=True, probe_path=work / "targets.bin"), work / "arm1.merged",
                                   work / "targets.bin")), "--corpus", str(index_path), "--arm", str(other_arm),
                 "--reference", str(work / "ref.npz"), "--out", str(work / "arm1.npz")], "no logit for reference top id")
        refused(["dump", str(arm), "--corpus", str(index_path), "--arm", str(other_arm), "--reference", str(work / "arm.npz"),
                 "--out", str(work / "arm2.npz")], "is an arm export")
        refused(["dump", str(ref), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "ref.npz")], "exists")
        missing = merge(engine_run(work / "missing", [0, 1, 2, 4, 5], probe_path=work / "targets.bin"), work / "missing.merged", work / "targets.bin")
        refused(["dump", str(missing), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "m.npz")], "no dump segment")
        twice = merge(engine_run(work / "twice", [0, 1, 2, 3, 4, 5, 2], probe_path=work / "targets.bin"), work / "twice.merged", work / "targets.bin")
        refused(["dump", str(twice), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "t.npz")], "scored twice")
        gap = merge(engine_run(work / "gap", natural, probe_path=work / "targets.bin", drop=25), work / "gap.merged", work / "targets.bin")
        refused(["dump", str(gap), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "g.npz")], "does not continue")
        signed = merge(engine_run(work / "signed", natural, probe_path=work / "targets.bin", arm_digest=b"\x01" * 32),
                       work / "signed.merged", work / "targets.bin")
        refused(["dump", str(signed), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "s.npz")], "rank files carry arm_digest")
        bent = merge(engine_run(work / "bent", natural, arm=True, probe_path=work / "probe2.bin", bent_probe=3), work / "bent.merged",
                     work / "probe2.bin")
        refused(["dump", str(bent), "--corpus", str(index_path), "--arm", str(other_arm), "--reference", str(work / "ref.npz"),
                 "--out", str(work / "b.npz")], "disagree")
        nan = merge(engine_run(work / "nan", natural, probe_path=work / "targets.bin", nonfinite=4), work / "nan.merged", work / "targets.bin")
        refused(["dump", str(nan), "--corpus", str(index_path), "--arm", str(ref_arm), "--out", str(work / "n.npz")], "non-finite")
        truncated = json.loads((work / "exact.json").read_text())
        truncated["max_mass_error"] = 1.0e-3
        (work / "bad-exact.json").write_text(json.dumps(truncated))
        refused(["exact", str(work / "bad-exact.json"), "--corpus", str(index_path), "--reference", str(work / "ref.npz"),
                 "--arm", str(work / "arm.npz"), "--out", str(work / "bad.npz")], "mass error")
        print(f"PASS test_score_export rows={len(wanted)} bucket_kl_max_error={error:.3g} exact_kl_max_error={exact_error:.3g} "
              f"tier2_rows={exact['rows'].size}")


if __name__ == "__main__":
    main()
