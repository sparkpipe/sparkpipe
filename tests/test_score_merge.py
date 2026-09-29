"""Score-dump merge contract: exact rank-order log-sum-exp, exact global top-k, lowest-id ties, byte reproducibility.

Part one merges synthetic 16-shard rank files against a float64 full-vocabulary
log-softmax. Part two runs the real head_score kernels and writer on the CPU
shim and checks the merged dump against numpy on the same hidden rows.
"""
import pathlib
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


def test_host_kernel_pipeline(work):
    binary = work / "head_score_host"
    objects = []
    for source in ("src/spark_score_dump.c", "src/spark_sha256.c"):
        target = work / (pathlib.Path(source).stem + ".o")
        subprocess.run(["cc", "-std=c11", "-O1", "-D_POSIX_C_SOURCE=200809L", "-Iinclude", "-c", source, "-o", str(target)],
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
                           ("bucketed_kl_bound", test_bucketed_kl_bound), ("host_kernel_pipeline", test_host_kernel_pipeline)):
            sub = work / name
            sub.mkdir()
            results[name] = test(sub)
            print(f"PASS {name} {results[name] if results[name] is not None else ''}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
