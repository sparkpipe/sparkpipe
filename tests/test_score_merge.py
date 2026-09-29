"""Score-dump merge contract: exact rank-order log-sum-exp, exact global top-k, lowest-id ties, byte reproducibility.

Part one merges synthetic 16-shard rank files against a float64 full-vocabulary
log-softmax. Part two runs the real head_score kernels and writer on the CPU
shim and checks the merged dump against numpy on the same hidden rows.
"""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

from host_cuda_compiler import host_cuda_cxx
import score_dump_oracle
import score_merge
import score_kl_partial

ROOT = pathlib.Path(__file__).resolve().parents[1]
TP = 16
WIDTH = 96
ROWS = 48


def synthetic_logits(seed):
    generator = np.random.default_rng(seed)
    logits = (generator.standard_normal((ROWS, TP * WIDTH)) * 3.0).astype(np.float32)
    logits[:, 7] = logits[:, 3]
    logits[:, WIDTH + 2] = logits[:, 3]
    logits[:, TP * WIDTH - 1] = logits[:, 3]
    logits[3, [3, 7, WIDTH + 2, TP * WIDTH - 1]] = np.float32(30.0)
    logits[0, :] = 1.25
    logits[1, 5 * WIDTH:6 * WIDTH] += 40.0
    logits[2, :] = np.float32(-80.0)
    logits[2, 11 * WIDTH + 4] = np.float32(60.0)
    return logits


def local_stats(shard, base):
    order = np.lexsort((np.arange(shard.size), -shard.astype(np.float64)))[:score_merge.TOP_K]
    maximum = shard.max()
    return maximum, float(np.sum(np.exp(shard.astype(np.float64) - np.float64(maximum)))), order + base, shard[order]


def write_ranks(directory, logits, probe_ids=None):
    paths = []
    for rank in range(TP):
        rows = np.zeros(ROWS, dtype=score_merge.ROW_DTYPE)
        probes = []
        begin = rank * WIDTH
        for row in range(ROWS):
            maximum, total, ids, values = local_stats(logits[row, begin:begin + WIDTH], begin)
            rows[row]["key"] = 1000 + row
            rows[row]["position"] = row
            rows[row]["wave_ordinal"] = row // 4
            rows[row]["row_in_wave"] = row % 4
            rows[row]["input_token"] = row
            rows[row]["served_token"] = score_merge.NO_TOKEN
            rows[row]["flags"] = score_merge.ROW_KEY_VALID
            rows[row]["local_max"] = maximum
            rows[row]["local_sum_exp"] = total
            rows[row]["top_ids"] = ids
            rows[row]["top_logits"] = values
            owned = [i for i in (probe_ids[row] if probe_ids else []) if begin <= i < begin + WIDTH]
            probe = np.zeros(len(owned), dtype=score_merge.PROBE_DTYPE)
            probe["id"] = owned
            probe["logit"] = logits[row, owned]
            probes.append(probe)
        path = directory / f"score.r{rank:02d}.bin"
        score_merge.write_rank_file(path, {"tp_rank": rank, "tp_degree": TP, "shard_begin": begin,
                                           "shard_end": begin + WIDTH, "vocabulary": TP * WIDTH,
                                           "hidden_dimension": 64}, rows, probes)
        paths.append(path)
    return paths


def full_log_softmax(logits):
    values = logits.astype(np.float64)
    peak = values.max(axis=1, keepdims=True)
    return values - (peak + np.log(np.sum(np.exp(values - peak), axis=1, keepdims=True)))


def test_synthetic_merge(work):
    logits = synthetic_logits(1)
    paths = write_ranks(work, logits)
    dumps, merged, _ = score_merge.merge([score_merge.read_rank_file(p) for p in paths])
    exact = full_log_softmax(logits)
    got = merged["top_logits"].astype(np.float64) - merged["log_z"][:, None]
    want = np.take_along_axis(exact, merged["top_ids"].astype(np.int64), axis=1)
    error = float(np.max(np.abs(got - want)))
    assert error <= 1.0e-5, error
    ids = np.arange(TP * WIDTH)
    for row in range(ROWS):
        expected = ids[np.lexsort((ids, -logits[row].astype(np.float64)))][:score_merge.TOP_K]
        assert np.array_equal(merged["top_ids"][row], expected), row
    assert merged["top_ids"][0].tolist() == list(range(score_merge.TOP_K))
    assert merged["top_ids"][3][:4].tolist() == [3, 7, WIDTH + 2, TP * WIDTH - 1]
    first = work / "a.merged"
    second = work / "b.merged"
    score_merge.write_merged(first, dumps, merged, [np.zeros(0, dtype=score_merge.PROBE_DTYPE)] * ROWS)
    shuffled = [paths[i] for i in np.random.default_rng(3).permutation(TP)]
    dumps2, merged2, probes2 = score_merge.merge([score_merge.read_rank_file(p) for p in shuffled])
    score_merge.write_merged(second, dumps2, merged2, probes2)
    assert first.read_bytes() == second.read_bytes()
    return error


def test_refusals(work):
    logits = synthetic_logits(2)
    paths = write_ranks(work, logits)
    dumps = [score_merge.read_rank_file(p) for p in paths]
    for mutate, label in (
            (lambda d: d.pop(), "missing rank"),
            (lambda d: d[4]["rows"].__setitem__("key", d[4]["rows"]["key"] + 1), "row identity"),
            (lambda d: d[2]["header"].__setitem__("shard_begin", 1), "shard gap"),
            (lambda d: d[9]["header"].__setitem__("arm_digest", b"\x01" * 32), "arm digest")):
        copy = [dict(item, header=dict(item["header"]), rows=item["rows"].copy()) for item in dumps]
        mutate(copy)
        try:
            score_merge.merge(copy)
        except score_merge.DumpError:
            continue
        raise AssertionError(f"merge accepted a {label} mismatch")
    truncated = work / "trunc.bin"
    truncated.write_bytes(paths[0].read_bytes()[:-score_merge.END.size])
    try:
        score_merge.read_rank_file(truncated)
    except score_merge.DumpError:
        pass
    else:
        raise AssertionError("an incomplete rank file was accepted")


def test_bucketed_kl_bound(work):
    generator = np.random.default_rng(7)
    for _ in range(200):
        lp = generator.standard_normal(500) * 2.0
        lq = lp + generator.standard_normal(500) * 0.3
        lp -= np.logaddexp.reduce(lp)
        lq -= np.logaddexp.reduce(lq)
        top = np.argsort(-lp)[:score_merge.TOP_K]
        bucketed, _ = score_kl_partial.bucketed_kl(lp[top], lq[top])
        exact = float(np.sum(np.exp(lp) * (lp - lq)))
        assert -1.0e-12 <= bucketed <= exact + 1.0e-12, (bucketed, exact)
    same, _ = score_kl_partial.bucketed_kl(lp[top], lp[top])
    assert same == 0.0


def corpus_rows(generator, documents, length):
    tokens = generator.integers(0, TP * WIDTH, size=(documents, length))
    identities = []
    for doc in range(documents):
        keys = score_merge.row_keys(tokens[doc].tolist())
        identities.extend((keys[position], position) for position in range(length))
    return tokens, identities


def write_arm(directory, logits, identities, probe_rows, tier2):
    directory.mkdir(parents=True, exist_ok=True)
    paths = write_ranks(directory, logits, probe_rows)
    for rank, path in enumerate(paths):
        dump = score_merge.read_rank_file(path)
        rows = dump["rows"].copy()
        rows["key"] = [key for key, _ in identities]
        rows["position"] = [position for _, position in identities]
        rows["flags"] = [score_merge.ROW_KEY_VALID | score_merge.ROW_PROBED | (score_merge.ROW_TIER2 if identity in tier2 else 0)
                         for identity in identities]
        header = dict(dump["header"], tier2=1)
        score_merge.write_rank_file(path, header, rows, dump["probes"])
        begin = rank * WIDTH
        with open(directory / f"tier2.r{rank:02d}.bin", "wb") as handle:
            handle.write(score_merge.HEADER.pack(score_merge.TIER2_OUT_MAGIC, score_merge.VERSION, score_merge.HEADER.size,
                                                 rank, TP, begin, begin + WIDTH, TP * WIDTH, 64, score_merge.TOP_K, 1,
                                                 bytes(32), bytes(32), bytes(32)))
            for row, identity in enumerate(identities):
                if identity in tier2:
                    handle.write(struct.pack("<QII", identity[0], identity[1], WIDTH))
                    handle.write(logits[row, begin:begin + WIDTH].astype("<f4").tobytes())
    merged_path = directory / "merged"
    dumps, merged, probes = score_merge.merge([score_merge.read_rank_file(p) for p in paths])
    score_merge.write_merged(merged_path, dumps, merged, probes)
    return merged_path


def test_arm_metrics(work):
    generator = np.random.default_rng(11)
    documents, length = 4, ROWS // 4
    tokens, identities = corpus_rows(generator, documents, length)
    reference = synthetic_logits(4)
    arm = (reference + generator.standard_normal(reference.shape).astype(np.float32) * np.float32(0.4)).astype(np.float32)
    reference[6, 17] = np.float32(35.0)
    reference[6, 9 * WIDTH + 1] = np.float32(34.9995)
    arm[5, :] = reference[5, :]
    arm[5, int(np.argmax(reference[5]))] -= np.float32(9.0)
    reference_top = [np.lexsort((np.arange(TP * WIDTH), -reference[row].astype(np.float64)))[:score_merge.TOP_K].tolist()
                     for row in range(ROWS)]
    targets = [int(tokens[row // length][row % length + 1]) if row % length + 1 < length else int(tokens[row // length][0])
               for row in range(ROWS)]
    probe_rows = [sorted(set(reference_top[row]) | {targets[row]}) for row in range(ROWS)]
    tier2 = set(identities[::3])
    ref_path = write_arm(work / "ref", reference, identities, probe_rows, tier2)
    arm_path = write_arm(work / "arm", arm, identities, probe_rows, tier2)
    corpus = work / "corpus.jsonl"
    with open(corpus, "w", encoding="utf-8") as handle:
        for doc in range(documents):
            handle.write(json.dumps({"doc": f"d{doc}", "tokens": tokens[doc].tolist()}) + "\n")
    result = score_kl_partial.compare(ref_path, arm_path, corpus)
    lp = full_log_softmax(reference)
    lq = full_log_softmax(arm)
    exact = np.sum(np.exp(lp) * (lp - lq), axis=1)
    rows = [doc * length + position for doc in range(documents) for position in range(length - 1)]
    got = {name: np.concatenate([np.array(d[name], dtype=float) for d in result]) for name in ("kl_b", "dlogp", "top1_agree", "decisive_flip", "near_tie")}
    ids = np.arange(TP * WIDTH)
    for slot, row in enumerate(rows):
        top = reference_top[row]
        p_top, q_top = lp[row, top], lq[row, top]
        p_tail, q_tail = -np.expm1(np.logaddexp.reduce(p_top)), -np.expm1(np.logaddexp.reduce(q_top))
        want = float(np.sum(np.exp(p_top) * (p_top - q_top)) + (p_tail * np.log(p_tail / q_tail) if p_tail > 0 else 0.0))
        assert abs(got["kl_b"][slot] - want) <= 1.0e-9, (row, got["kl_b"][slot], want)
        assert got["kl_b"][slot] <= exact[row] + 1.0e-9, (row, got["kl_b"][slot], exact[row])
        y = int(tokens[row // length][row % length + 1])
        assert abs(got["dlogp"][slot] - (lq[row, y] - lp[row, y])) <= 1.0e-9, row
        ref_first = ids[np.lexsort((ids, -reference[row].astype(np.float64)))][:2]
        arm_first = ids[np.lexsort((ids, -arm[row].astype(np.float64)))][0]
        gap = float(reference[row, ref_first[0]]) - float(reference[row, ref_first[1]])
        assert bool(got["top1_agree"][slot]) == (arm_first == ref_first[0]), row
        assert bool(got["near_tie"][slot]) == (gap < score_merge.NEAR_TIE_NATS), row
        assert bool(got["decisive_flip"][slot]) == (arm_first != ref_first[0] and gap > 1.0), row
    assert got["decisive_flip"].sum() >= 1 and got["top1_agree"].sum() < len(rows) and got["near_tie"][rows.index(6)]
    assert got["dlogp"].min() < 0.0 < got["dlogp"].max()
    partials = []
    for rank in range(TP):
        header, part = score_kl_partial.partial(work / "ref" / f"tier2.r{rank:02d}.bin", work / "arm" / f"tier2.r{rank:02d}.bin", ref_path, arm_path)
        out = work / f"partial.r{rank:02d}.bin"
        score_kl_partial.write_partial(out, dict(header, arm_digest=bytes(32)), part)
        partials.append(out)
    keys, kl, mass = score_kl_partial.combine(list(reversed(partials)))
    by_identity = {identity: row for row, identity in enumerate(identities)}
    for (key, position), value in zip(keys.tolist(), kl):
        row = by_identity[(key, position)]
        assert abs(value - exact[row]) <= 1.0e-9 * max(1.0, exact[row]), (row, value, exact[row])
    assert len(kl) == len(tier2) and float(np.max(np.abs(mass - 1.0))) <= 1.0e-12
    truncated = work / "arm" / "tier2.r03.bin"
    truncated.write_bytes(truncated.read_bytes()[:-8])
    try:
        score_kl_partial.partial(work / "ref" / "tier2.r03.bin", truncated, ref_path, arm_path)
    except score_kl_partial.CompareError:
        pass
    else:
        raise AssertionError("a truncated Tier-2 dump was accepted")
    record = 16 + 4 * WIDTH
    for side in ("ref", "arm"):
        whole = work / side / "tier2.r04.bin"
        whole.write_bytes(whole.read_bytes()[:-record])
    try:
        score_kl_partial.partial(work / "ref" / "tier2.r04.bin", work / "arm" / "tier2.r04.bin", ref_path, arm_path)
    except score_kl_partial.CompareError:
        pass
    else:
        raise AssertionError("a Tier-2 dump missing a whole row was accepted")
    _, merged, _ = score_merge.read_merged(arm_path)
    doubled = np.concatenate([merged, merged[:1]])
    doubled[-1]["log_z"] += 1.0
    assert score_merge.row_index(doubled)[(int(merged["key"][0]), int(merged["position"][0]))] == 0
    return {"kl_b_mean": float(got["kl_b"].mean()), "exact_kl_mean": float(kl.mean()), "flips": int(got["decisive_flip"].sum())}


def test_host_kernel_pipeline(work):
    binary = work / "head_score_host"
    objects = []
    for source in ("src/spark_score_dump.c", "src/spark_sha256.c"):
        target = work / (pathlib.Path(source).stem + ".o")
        subprocess.run([host_cuda_cxx(), "-x", "c", "-std=c11", "-O1", "-D_POSIX_C_SOURCE=200809L", "-D_DARWIN_C_SOURCE", "-Iinclude", "-c", source, "-o", str(target)],
                       cwd=ROOT, check=True)
        objects.append(str(target))
    subprocess.run([host_cuda_cxx(), "-std=c++17", "-O1", "-I.", "-Itests/host_cuda", "-Iinclude",
                    "-x", "c++", "tests/host_cuda/head_score_host.cu", "-x", "none", *objects, "-o", str(binary)],
                   cwd=ROOT, check=True)
    return score_dump_oracle.check_pipeline(binary, work / "pipeline")


def main():
    with tempfile.TemporaryDirectory(prefix="score-merge-") as directory:
        work = pathlib.Path(directory)
        results = {}
        for name, test in (("synthetic_merge", test_synthetic_merge), ("refusals", test_refusals),
                           ("bucketed_kl_bound", test_bucketed_kl_bound), ("arm_metrics", test_arm_metrics),
                           ("host_kernel_pipeline", test_host_kernel_pipeline)):
            sub = work / name
            sub.mkdir()
            results[name] = test(sub)
            print(f"PASS {name} {results[name] if results[name] is not None else ''}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
