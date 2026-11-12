#!/usr/bin/env python3
"""Teacher-forced next-token A/B through the serving API, and the variant gate built on it.

score    sends one greedy request per scored position of a frozen corpus
         (sparkpipe-ab-corpus-v1): the document's tokens before the position as
         prompt_token_ids, max_tokens 1, the top 20 log-probs, no speculation.
         Each request names the last 64-token boundary before its end as a
         cache point, so the next position of the same document resumes there.
         A rerun appends only the missing positions. A gate run asks one
         request at a time: a row's log-probs depend slightly on the rows
         that share its prefill wave (docs/VARIANT_GATE.md).
join     writes a one-document corpus: the documents of an index back to back,
         for scoring positions deep in a long context.
compare  pairs two score files position by position: top-1 agreement, a lower
         bound on KL(reference || arm), the corpus token's NLL change and
         next-token accuracy, each with a bootstrap bound whose unit is the
         document (or the run of positions, for a joined corpus). Where a
         model's next token ends the text the API lists no log-probs: the
         position counts for top-1 agreement (both stop, or not) and is
         left out of KL and NLL.
gate     decides a variant from its comparisons (docs/VARIANT_GATE.md).

usage:
  ab_api_score.py score --endpoint URL --index I --tokens T --out ROWS.jsonl
      [--starts 256 | --starts FIRST:LAST:STEP] [--stride 16] [--count 64]
      [--documents N] [--concurrency N] [--timeout S]
  ab_api_score.py join --index I --tokens T --out-index I2 --out-tokens T2 [--length N]
  ab_api_score.py compare REFERENCE.jsonl ARM.jsonl [--unit document|run] [--out CMP.json]
  ab_api_score.py gate --corpus CMP.json --repeat CMP.json [--depth CMP.json]
      [--suite NAME=REFERENCE_SUMMARY.json:ARM_SUMMARY.json ...] [--out VERDICT.json]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOP = 20
BLOCK = 64
CORPUS_FORMAT = "sparkpipe-ab-corpus-v1"
COMPARISON_FORMAT = "sparkpipe-ab-api-comparison-v1"
VERDICT_FORMAT = "sparkpipe-variant-gate-v1"
SEED = 20260929
REPLICATES = 10000
LEVEL = 0.95
NEAR_TIE_NATS = 0.001
DNLL_REL_PCT_ABS = 0.5
MCNEMAR_ALPHA = 0.05
DEPTH_BINS = (0, 16384, 65536, 131072)


class ScoreError(ValueError):
    pass


def load_corpus(index_path, tokens_path):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index.get("format") != CORPUS_FORMAT:
        raise ScoreError(f"{index_path}: not a {CORPUS_FORMAT} index")
    data = Path(tokens_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != index["corpus_sha256"]:
        raise ScoreError(f"{tokens_path}: SHA-256 differs from the index's corpus_sha256")
    tokens = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    documents = []
    for record in index["documents"]:
        body = tokens[record["offset"]:record["offset"] + record["length"]]
        raw = b"".join(token.to_bytes(4, "little") for token in body)
        if len(body) != record["length"] or hashlib.sha256(raw).hexdigest() != record["tokens_sha256"]:
            raise ScoreError(f"{record['id']}: tokens differ from the index")
        documents.append({"id": record["id"], "stratum": record["stratum"], "tokens": body})
    return index, documents


def parse_starts(text):
    starts = []
    for part in text.split(","):
        fields = [int(field) for field in part.split(":")]
        if len(fields) == 1:
            starts.append(fields[0])
        elif len(fields) == 3 and fields[2] > 0 and fields[1] >= fields[0]:
            starts += list(range(fields[0], fields[1] + 1, fields[2]))
        else:
            raise ScoreError(f"--starts takes positions or FIRST:LAST:STEP ranges, not {part!r}")
    if not starts or starts != sorted(set(starts)) or starts[0] < 1:
        raise ScoreError("--starts must be distinct ascending positions of at least 1")
    return starts


def positions(length, starts, stride, count):
    if stride < 1 or count < 1:
        raise ScoreError("--stride and --count are at least 1")
    chosen = [(start, start + stride * step) for start in starts for step in range(count)]
    if chosen[-1][1] >= length:
        raise ScoreError(f"position {chosen[-1][1]} does not fit a {length}-token document")
    if len({position for _, position in chosen}) != len(chosen):
        raise ScoreError("the runs of positions overlap: space --starts at least --stride x --count apart")
    return chosen


def ask(endpoint, prompt, timeout):
    body = {"prompt_token_ids": prompt, "max_tokens": 1, "temperature": 0, "logprobs": TOP, "speculation": False}
    point = (len(prompt) - 1) // BLOCK * BLOCK
    if point > 0:
        body["cache_points"] = [point]
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score(args):
    _, documents = load_corpus(args.index, args.tokens)
    if args.documents:
        documents = documents[:args.documents]
    starts = parse_starts(args.starts)
    out = Path(args.out)
    done = set(read(out)) if out.exists() else set()
    lock = threading.Lock()
    started = time.time()
    scored = [0]

    def one(document):
        rows = 0
        for start, position in positions(len(document["tokens"]), starts, args.stride, args.count):
            if (document["id"], position) in done:
                continue
            reply = ask(args.endpoint, document["tokens"][:position], args.timeout)
            lists = reply.get("token_logprobs") or []
            stopped = not lists and reply.get("tokens") == [] and (reply.get("choices") or [{}])[0].get("finish_reason") == "stop"
            if not stopped and (len(lists) != 1 or not lists[0]):
                raise ScoreError(f"{document['id']}@{position}: the reply has no logprobs: {json.dumps(reply)[:300]}")
            row = {"document": document["id"], "stratum": document["stratum"], "start": start, "position": position,
                   "truth": document["tokens"][position], "top": None if stopped else lists[0], "usage": reply.get("usage")}
            with lock:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                scored[0] += 1
            rows += 1
        with lock:
            print(f"{document['id']} done, {scored[0]} positions in {time.time() - started:.0f} s", file=sys.stderr, flush=True)
        return rows

    with out.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in [pool.submit(one, document) for document in documents]:
            future.result()
    print(f"{out}: {scored[0]} new positions, {len(done) + scored[0]} in all, {time.time() - started:.0f} s")
    return 0


def join(args):
    index, documents = load_corpus(args.index, args.tokens)
    body = [token for document in documents for token in document["tokens"]]
    if args.length:
        if args.length > len(body):
            raise ScoreError(f"the corpus has {len(body)} tokens, fewer than --length {args.length}")
        body = body[:args.length]
    raw = b"".join(token.to_bytes(4, "little") for token in body)
    digest = hashlib.sha256(raw).hexdigest()
    record = {"id": f"joined:{index['corpus_sha256'][:16]}", "stratum": "joined", "offset": 0, "length": len(body), "tokens_sha256": digest,
              "source": {"corpus_sha256": index["corpus_sha256"], "documents": len(documents)}}
    out = dict(index, name=f"{index['name']}-joined", documents=[record], skipped=[], strata={"joined": 1}, corpus_sha256=digest,
               total_tokens=len(body), document_tokens=len(body),
               notes=list(index.get("notes", [])) + [f"the documents of {index['name']} ({index['corpus_sha256']}) back to back, cut to {len(body)} tokens"])
    Path(args.out_tokens).write_bytes(raw)
    Path(args.out_index).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out_index}: one document of {len(body)} tokens, corpus_sha256 {digest}")
    return 0


def read(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["document"], row["position"])
        if key in rows:
            raise ScoreError(f"{path}: {key} appears twice")
        rows[key] = row
    return rows


def distribution(top):
    table = {}
    for token, logprob in top:
        if token not in table:
            table[token] = logprob
    return table


def bounded_kl(reference, arm):
    shared = [token for token in reference if token in arm and reference[token] is not None and arm[token] is not None]
    p = [math.exp(reference[token]) for token in shared]
    q = [math.exp(arm[token]) for token in shared]
    p.append(max(0.0, 1.0 - sum(p)))
    q.append(max(0.0, 1.0 - sum(q)))
    total = 0.0
    for left, right in zip(p, q):
        if left > 0.0:
            total += left * math.log(left / max(right, 1e-30))
    return max(0.0, total)


def top_gap(table):
    values = sorted((value for value in table.values() if value is not None), reverse=True)
    return values[0] - values[1] if len(values) > 1 else math.inf


def pair(reference, arm):
    stopped = (reference["top"] is None, arm["top"] is None)
    if any(stopped):
        return {"near_tie": False, "agree": all(stopped), "kl": None, "reference_right": False, "arm_right": False, "truth_reference_only": False,
                "truth_arm_only": False, "nll_reference": None, "nll_arm": None, "identical": all(stopped), "logprob_delta_max": 0.0,
                "stopped": "both" if all(stopped) else "reference" if stopped[0] else "arm"}
    p, q = distribution(reference["top"]), distribution(arm["top"])
    truth = reference["truth"]
    listed_reference = p.get(truth) is not None
    listed_arm = q.get(truth) is not None
    shared = [token for token in p if token in q and p[token] is not None and q[token] is not None]
    return {
        "near_tie": top_gap(p) < NEAR_TIE_NATS,
        "agree": reference["top"][0][0] == arm["top"][0][0],
        "kl": bounded_kl(p, q),
        "reference_right": reference["top"][0][0] == truth,
        "arm_right": arm["top"][0][0] == truth,
        "truth_reference_only": listed_reference and not listed_arm,
        "truth_arm_only": listed_arm and not listed_reference,
        "nll_reference": -p[truth] if listed_reference and listed_arm else None,
        "nll_arm": -q[truth] if listed_reference and listed_arm else None,
        "identical": reference["top"] == arm["top"],
        "logprob_delta_max": max((abs(p[token] - q[token]) for token in shared), default=0.0),
        "stopped": None,
    }


def bound(columns, scale, seed, replicates):
    import ab_stats
    result = ab_stats.bounds(ab_stats.ratio_statistic, columns, seed, replicates, LEVEL)
    return {"estimate": scale * result["estimate"], "lower": scale * result["lower"], "upper": scale * result["upper"]}


def point(numerator, denominator, scale):
    return {"estimate": scale * sum(numerator) / sum(denominator) if sum(denominator) else None, "lower": None, "upper": None}


def summarize(pairs, seed, replicates):
    import ab_stats
    units = {}
    for row in pairs:
        units.setdefault(row["unit"], []).append(row)
    names = sorted(units)

    def column(function):
        return [float(sum(function(row) for row in units[name])) for name in names]

    counted = column(lambda row: 0 if row["near_tie"] else 1)
    agreed = column(lambda row: 1 if row["agree"] and not row["near_tie"] else 0)
    rows = column(lambda row: 0 if row["kl"] is None else 1)
    kl = column(lambda row: row["kl"] or 0.0)
    truth = column(lambda row: 0 if row["nll_reference"] is None else 1)
    nll_reference = column(lambda row: row["nll_reference"] or 0.0)
    nll_delta = column(lambda row: 0.0 if row["nll_reference"] is None else row["nll_arm"] - row["nll_reference"])
    only_reference = sum(1 for row in pairs if row["reference_right"] and not row["arm_right"])
    only_arm = sum(1 for row in pairs if row["arm_right"] and not row["reference_right"])
    out = {
        "positions": len(pairs),
        "units": len(names),
        "identical_lists": sum(1 for row in pairs if row["identical"]),
        "logprob_delta_max": max(row["logprob_delta_max"] for row in pairs),
        "near_tie_positions": sum(1 for row in pairs if row["near_tie"]),
        "stopped_by_both": sum(1 for row in pairs if row["stopped"] == "both"),
        "stopped_by_reference_only": sum(1 for row in pairs if row["stopped"] == "reference"),
        "stopped_by_arm_only": sum(1 for row in pairs if row["stopped"] == "arm"),
        "truth_listed_by_both": int(sum(truth)),
        "truth_listed_by_reference_only": sum(1 for row in pairs if row["truth_reference_only"]),
        "truth_listed_by_arm_only": sum(1 for row in pairs if row["truth_arm_only"]),
        "reference_nll": sum(nll_reference) / sum(truth) if sum(truth) else None,
        "reference_accuracy_pct": 100.0 * sum(1 for row in pairs if row["reference_right"]) / len(pairs),
        "arm_accuracy_pct": 100.0 * sum(1 for row in pairs if row["arm_right"]) / len(pairs),
        "accuracy_only_reference": only_reference,
        "accuracy_only_arm": only_arm,
        "accuracy_mcnemar_p": ab_stats.mcnemar_exact(only_reference, only_arm),
    }
    measure = bound if len(names) > 1 else lambda columns, scale, seed, replicates: point(columns[0], columns[1], scale)
    out["top1_agreement_pct"] = measure([agreed, counted], 100.0, seed, replicates) if sum(counted) else point(agreed, counted, 100.0)
    out["kl_lower_bound"] = measure([kl, rows], 1.0, seed, replicates) if sum(rows) else point(kl, rows, 1.0)
    out["dnll_rel_pct"] = measure([nll_delta, nll_reference], 100.0, seed, replicates) if sum(nll_reference) > 0.0 else point(nll_delta, nll_reference, 100.0)
    return out


def depth_bin(start):
    floor = max(edge for edge in DEPTH_BINS if edge <= start)
    return f"from {floor}"


def compare(args):
    reference, arm = read(args.reference), read(args.arm)
    if set(reference) != set(arm):
        raise ScoreError(f"the two runs score different positions: {len(set(reference) ^ set(arm))} differ")
    pairs = []
    for key in sorted(reference):
        left, right = reference[key], arm[key]
        if left["truth"] != right["truth"]:
            raise ScoreError(f"{key}: the runs disagree on the corpus token")
        row = pair(left, right)
        start = left.get("start", 0)
        row.update(stratum=left["stratum"], start=start, unit=key[0] if args.unit == "document" else f"{key[0]}@{start}")
        pairs.append(row)
    report = {"format": COMPARISON_FORMAT, "reference": str(args.reference), "arm": str(args.arm), "unit": args.unit, "seed": args.seed,
              "replicates": args.replicates, "level": LEVEL, "near_tie_nats": NEAR_TIE_NATS,
              "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              "arm_sha256": hashlib.sha256(Path(args.arm).read_bytes()).hexdigest(),
              "overall": summarize(pairs, args.seed, args.replicates)}
    groups = sorted({row["stratum"] for row in pairs}) if args.unit == "document" else sorted({depth_bin(row["start"]) for row in pairs}, key=lambda name: int(name.split()[1]))
    name = "strata" if args.unit == "document" else "depth"
    report[name] = {}
    for group in groups:
        members = [row for row in pairs if (row["stratum"] if args.unit == "document" else depth_bin(row["start"])) == group]
        report[name][group] = summarize(members, args.seed, args.replicates)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def load_comparison(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("format") != COMPARISON_FORMAT:
        raise ScoreError(f"{path}: not a {COMPARISON_FORMAT} comparison")
    return report


def nll_check(name, report):
    change = report["overall"]["dnll_rel_pct"]
    if change["lower"] is None:
        return {"check": name, "passed": False, "detail": "no bootstrap bound: the comparison needs at least two units that list the corpus token in both runs"}
    inside = -DNLL_REL_PCT_ABS < change["lower"] and change["upper"] < DNLL_REL_PCT_ABS
    return {"check": name, "passed": inside, "dnll_rel_pct": change, "margin_pct": DNLL_REL_PCT_ABS,
            "detail": f"the corpus token's NLL changes by {change['estimate']:+.4f}% [{change['lower']:+.4f}, {change['upper']:+.4f}] against a margin of +/-{DNLL_REL_PCT_ABS}%"}


def accuracy_check(name, report):
    overall = report["overall"]
    worse = overall["accuracy_only_reference"] > overall["accuracy_only_arm"] and overall["accuracy_mcnemar_p"] < MCNEMAR_ALPHA
    return {"check": name, "passed": not worse, "only_reference": overall["accuracy_only_reference"], "only_arm": overall["accuracy_only_arm"],
            "mcnemar_p": overall["accuracy_mcnemar_p"],
            "detail": f"next-token accuracy {overall['reference_accuracy_pct']:.3f}% against {overall['arm_accuracy_pct']:.3f}%: {overall['accuracy_only_reference']} positions only the reference predicts, "
                      f"{overall['accuracy_only_arm']} only the variant, exact McNemar p {overall['accuracy_mcnemar_p']:.4f}"}


def suite_check(spec):
    import ab_stats
    name, _, paths = spec.partition("=")
    left, _, right = paths.partition(":")
    if not name or not left or not right:
        raise ScoreError(f"--suite takes NAME=REFERENCE_SUMMARY.json:ARM_SUMMARY.json, not {spec!r}")
    runs = []
    for path in (left, right):
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        results = {record["id"]: record for record in summary["results"]}
        if len(results) != len(summary["results"]):
            raise ScoreError(f"{path}: a case appears twice")
        failed = sorted(case for case, record in results.items() if record.get("status") != "ok")
        if failed:
            raise ScoreError(f"{path}: {len(failed)} cases did not complete ({', '.join(failed[:4])}); a suite with errors decides nothing")
        runs.append(results)
    if set(runs[0]) != set(runs[1]):
        raise ScoreError(f"suite {name}: the two runs hold different cases")
    lost = sum(1 for case in runs[0] if runs[0][case]["passed"] and not runs[1][case]["passed"])
    gained = sum(1 for case in runs[0] if runs[1][case]["passed"] and not runs[0][case]["passed"])
    p = ab_stats.mcnemar_exact(lost, gained)
    passed = [sum(1 for record in run.values() if record["passed"]) for run in runs]
    return {"check": f"suite {name}", "passed": not (lost > gained and p < MCNEMAR_ALPHA), "cases": len(runs[0]), "reference_passed": passed[0], "arm_passed": passed[1],
            "only_reference": lost, "only_arm": gained, "mcnemar_p": p,
            "detail": f"{passed[0]} of {len(runs[0])} against {passed[1]}: {lost} cases only the reference answers, {gained} only the variant, exact McNemar p {p:.4f}"}


def gate(args):
    corpus, repeat = load_comparison(args.corpus), load_comparison(args.repeat)
    checks = []
    same = repeat["overall"]["identical_lists"] == repeat["overall"]["positions"]
    checks.append({"check": "repeat", "passed": same, "positions": repeat["overall"]["positions"], "identical_lists": repeat["overall"]["identical_lists"],
                   "logprob_delta_max": repeat["overall"]["logprob_delta_max"],
                   "detail": f"{repeat['overall']['identical_lists']} of {repeat['overall']['positions']} repeated positions return the same tokens and log-probs"
                             + ("" if same else f"; the largest log-prob difference is {repeat['overall']['logprob_delta_max']:.3g}")})
    checks.append(nll_check("corpus NLL", corpus))
    checks.append(accuracy_check("corpus next-token accuracy", corpus))
    if args.depth:
        depth = load_comparison(args.depth)
        if depth["unit"] != "run":
            raise ScoreError(f"{args.depth}: the depth comparison is made with --unit run")
        checks.append(nll_check("depth NLL", depth))
        checks.append(accuracy_check("depth next-token accuracy", depth))
    checks += [suite_check(spec) for spec in args.suite]
    verdict = {"format": VERDICT_FORMAT, "passed": all(check["passed"] for check in checks), "checks": checks,
               "criteria": {"dnll_rel_pct_abs": DNLL_REL_PCT_ABS, "mcnemar_alpha": MCNEMAR_ALPHA, "bound_level": LEVEL, "near_tie_nats": NEAR_TIE_NATS},
               "reported": {"top1_agreement_pct": corpus["overall"]["top1_agreement_pct"], "kl_lower_bound": corpus["overall"]["kl_lower_bound"]},
               "inputs": {"corpus": str(args.corpus), "repeat": str(args.repeat), "depth": str(args.depth) if args.depth else None, "suites": list(args.suite)}}
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for check in checks:
        print(("PASS " if check["passed"] else "FAIL ") + f"{check['check']}: {check['detail']}")
    print("VARIANT GATE " + ("PASS" if verdict["passed"] else "FAIL"))
    return 0 if verdict["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("score")
    command.add_argument("--endpoint", required=True)
    command.add_argument("--index", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--starts", default="256")
    command.add_argument("--stride", type=int, default=16)
    command.add_argument("--count", type=int, default=64)
    command.add_argument("--documents", type=int, default=0)
    command.add_argument("--concurrency", type=int, default=1)
    command.add_argument("--timeout", type=float, default=1800.0)
    command = sub.add_parser("join")
    command.add_argument("--index", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--out-index", required=True)
    command.add_argument("--out-tokens", required=True)
    command.add_argument("--length", type=int, default=0)
    command = sub.add_parser("compare")
    command.add_argument("reference")
    command.add_argument("arm")
    command.add_argument("--unit", choices=("document", "run"), default="document")
    command.add_argument("--out")
    command.add_argument("--seed", type=int, default=SEED)
    command.add_argument("--replicates", type=int, default=REPLICATES)
    command = sub.add_parser("gate")
    command.add_argument("--corpus", required=True)
    command.add_argument("--repeat", required=True)
    command.add_argument("--depth")
    command.add_argument("--suite", action="append", default=[])
    command.add_argument("--out")
    args = parser.parse_args()
    if args.command == "score" and args.concurrency < 1:
        parser.error("--concurrency is at least 1")
    try:
        return {"score": score, "join": join, "compare": compare, "gate": gate}[args.command](args)
    except ScoreError as error:
        print(f"ab_api_score: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
