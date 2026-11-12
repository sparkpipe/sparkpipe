#!/usr/bin/env python3
import hashlib
import http.server
import json
import math
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/ab_api_score.py"
sys.path.insert(0, str(ROOT / "tools"))
import ab_api_score  # noqa: E402
import ab_stats  # noqa: E402

VOCABULARY = 40
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def logits(prompt, shift):
    seed = int.from_bytes(hashlib.sha256(json.dumps(prompt).encode()).digest()[:8], "little")
    values = []
    for token in range(VOCABULARY):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % 2 ** 64
        values.append((seed >> 40) / 2 ** 22 + (shift if token % 7 == 3 else 0.0))
    return values


def reply(prompt, shift):
    values = logits(prompt, shift)
    lse = math.log(sum(math.exp(value) for value in values))
    ranked = sorted(range(VOCABULARY), key=lambda token: (-values[token], token))[:ab_api_score.TOP]
    listed = [[token, round(values[token] - lse, 6)] for token in ranked]
    return {"tokens": [ranked[0]], "token_logprobs": [[listed[0]] + listed], "usage": {"prompt_tokens": len(prompt)}}


class Api:
    def __init__(self, shift):
        self.requests = []
        api = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.requests.append(body)
                data = json.dumps(reply(body["prompt_token_ids"], shift)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *arguments):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def corpus(directory, documents, length):
    body, records = bytearray(), []
    for index in range(documents):
        tokens = [(index * 131 + position * 17 + position * position) % VOCABULARY for position in range(length)]
        raw = b"".join(token.to_bytes(4, "little") for token in tokens)
        records.append({"id": f"doc:{index}", "stratum": "code" if index % 2 else "prose", "offset": len(body) // 4, "length": length,
                        "tokens_sha256": hashlib.sha256(raw).hexdigest()})
        body += raw
    index = {"format": "sparkpipe-ab-corpus-v1", "name": "T", "documents": records, "skipped": [], "strata": {"code": documents // 2, "prose": documents - documents // 2},
             "corpus_sha256": hashlib.sha256(bytes(body)).hexdigest(), "total_tokens": len(body) // 4, "document_tokens": length, "notes": []}
    (directory / "T.index.json").write_text(json.dumps(index))
    (directory / "T.tokens.u32").write_bytes(bytes(body))
    return directory / "T.index.json", directory / "T.tokens.u32"


def run(*arguments):
    return subprocess.run([sys.executable, str(TOOL), *map(str, arguments)], capture_output=True, text=True)


def rows(path):
    return {(row["document"], row["position"]): row for row in map(json.loads, Path(path).read_text().splitlines())}


def score_cases(work):
    index, tokens = corpus(work, 6, 200)
    api = Api(0.0)
    try:
        common = ("score", "--endpoint", api.endpoint, "--index", index, "--tokens", tokens, "--starts", "70", "--stride", "3", "--count", "5")
        first = run(*common, "--out", work / "a.jsonl")
        scored = rows(work / "a.jsonl")
        check(first.returncode == 0 and len(scored) == 30 and len(api.requests) == 30, "score: one request for every position of every document")
        check(all(request["max_tokens"] == 1 and request["temperature"] == 0 and request["logprobs"] == 20 and request["speculation"] is False for request in api.requests),
              "score: every request is one greedy token with the top 20 log-probs and no speculation")
        check(sorted({len(request["prompt_token_ids"]) for request in api.requests}) == [70, 73, 76, 79, 82] and all(request["cache_points"] == [64] for request in api.requests),
              "score: a prompt is the document's tokens before the position, and its cache point is the last 64-token boundary before its end")
        document = json.loads(index.read_text())["documents"][2]
        data = tokens.read_bytes()
        truth = int.from_bytes(data[4 * (document["offset"] + 76):4 * (document["offset"] + 77)], "little")
        check(scored[("doc:2", 76)]["truth"] == truth and scored[("doc:2", 76)]["start"] == 70 and scored[("doc:2", 76)]["stratum"] == "prose",
              "score: a row carries the corpus token at its position, its run's start and its stratum")
        again = run(*common, "--out", work / "a.jsonl")
        check(again.returncode == 0 and len(api.requests) == 30 and "0 new positions" in again.stdout, "score: a rerun over a finished file asks nothing")
        lines = (work / "a.jsonl").read_text().splitlines()
        (work / "a.jsonl").write_text("\n".join(lines[:11]) + "\n")
        resumed = run(*common, "--out", work / "a.jsonl")
        check(resumed.returncode == 0 and len(api.requests) == 49 and rows(work / "a.jsonl") == scored, "score: a rerun over a cut file asks only the missing positions")
        parallel = run(*common, "--out", work / "c.jsonl", "--concurrency", "4")
        check(parallel.returncode == 0 and rows(work / "c.jsonl") == scored, "score: four documents at a time give the same rows as one")
        short = run("score", "--endpoint", api.endpoint, "--index", index, "--tokens", tokens, "--starts", "190", "--stride", "3", "--count", "5", "--out", work / "s.jsonl")
        check(short.returncode == 2 and "does not fit" in short.stderr, "score: positions past a document's end are refused")
        overlap = run("score", "--endpoint", api.endpoint, "--index", index, "--tokens", tokens, "--starts", "70,76", "--stride", "3", "--count", "5", "--out", work / "o.jsonl")
        check(overlap.returncode == 2 and "overlap" in overlap.stderr, "score: runs of positions that overlap are refused")
        tokens.write_bytes(data[:-4] + b"\x01\x00\x00\x00")
        changed = run(*common, "--out", work / "x.jsonl")
        check(changed.returncode == 2 and "corpus_sha256" in changed.stderr, "score: a token file that differs from its index is refused")
        tokens.write_bytes(data)
    finally:
        api.close()
    return index, tokens


def row(document, position, truth, top, start=0, stratum="prose"):
    return {"document": document, "stratum": stratum, "start": start, "position": position, "truth": truth, "top": [top[0]] + top, "usage": None}


def write(path, records):
    Path(path).write_text("".join(json.dumps(record) + "\n" for record in records))


def log(value):
    return math.log(value)


def compare_cases(work):
    reference = [
        row("d0", 5, 7, [[7, log(0.5)], [8, log(0.3)], [9, log(0.1)]]),
        row("d0", 6, 8, [[7, log(0.6)], [8, log(0.2)]]),
        row("d1", 5, 9, [[7, log(0.4)], [9, log(0.3999)], [8, log(0.1)]]),
        row("d1", 6, 3, [[4, log(0.7)], [3, log(0.2)]]),
    ]
    arm = [
        row("d0", 5, 7, [[7, log(0.4)], [8, log(0.4)], [9, log(0.1)]]),
        row("d0", 6, 8, [[8, log(0.5)], [7, log(0.4)]]),
        row("d1", 5, 9, [[9, log(0.5)], [7, log(0.3)], [8, log(0.1)]]),
        row("d1", 6, 3, [[4, log(0.7)], [5, log(0.2)]]),
    ]
    write(work / "r.jsonl", reference)
    write(work / "m.jsonl", arm)
    done = run("compare", work / "r.jsonl", work / "m.jsonl", "--out", work / "cmp.json", "--replicates", "200")
    report = json.loads((work / "cmp.json").read_text())
    overall = report["overall"]
    check(done.returncode == 0 and overall["positions"] == 4 and overall["units"] == 2 and overall["identical_lists"] == 0, "compare: four positions in two documents, none identical")
    check(overall["near_tie_positions"] == 1 and abs(overall["top1_agreement_pct"]["estimate"] - 100.0 * 2 / 3) < 1e-9,
          "compare: a position whose two best reference tokens are within 0.001 nats is left out of top-1 agreement")
    kl = [0.5 * log(0.5 / 0.4) + 0.3 * log(0.3 / 0.4) + 0.1 * log(1.0) + 0.1 * log(1.0),
          0.6 * log(0.6 / 0.4) + 0.2 * log(0.2 / 0.5) + 0.2 * log(0.2 / 0.1),
          0.4 * log(0.4 / 0.3) + 0.3999 * log(0.3999 / 0.5) + 0.1 * log(1.0) + 0.1001 * log(0.1001 / 0.1),
          0.7 * log(1.0) + 0.3 * log(1.0)]
    check(abs(overall["kl_lower_bound"]["estimate"] - sum(kl) / 4) < 1e-9, "compare: KL is taken over the tokens both runs list plus one bucket for the rest")
    delta = (log(0.5) - log(0.4)) + (log(0.2) - log(0.5)) + (log(0.3999) - log(0.5))
    base = -log(0.5) - log(0.2) - log(0.3999)
    check(overall["truth_listed_by_both"] == 3 and overall["truth_listed_by_reference_only"] == 1 and overall["truth_listed_by_arm_only"] == 0
          and abs(overall["dnll_rel_pct"]["estimate"] - 100.0 * delta / base) < 1e-9 and abs(overall["reference_nll"] - base / 3) < 1e-12,
          "compare: the NLL change is over the positions whose corpus token both runs list, relative to the reference's NLL there")
    check(overall["accuracy_only_reference"] == 0 and overall["accuracy_only_arm"] == 2 and overall["accuracy_mcnemar_p"] == ab_stats.mcnemar_exact(0, 2)
          and overall["reference_accuracy_pct"] == 25.0 and overall["arm_accuracy_pct"] == 75.0, "compare: next-token accuracy pairs the two runs by position")
    check(set(report["strata"]) == {"prose"} and report["unit"] == "document", "compare: strata are reported apart")
    same = run("compare", work / "r.jsonl", work / "r.jsonl", "--out", work / "same.json", "--replicates", "200")
    report = json.loads((work / "same.json").read_text())["overall"]
    check(same.returncode == 0 and report["identical_lists"] == 4 and report["kl_lower_bound"]["estimate"] == 0.0 and report["dnll_rel_pct"]["estimate"] == 0.0
          and report["logprob_delta_max"] == 0.0 and report["dnll_rel_pct"]["upper"] == 0.0, "compare: a run against itself is identical, with zero KL and zero NLL change")
    write(work / "missing.jsonl", arm[:3])
    check(run("compare", work / "r.jsonl", work / "missing.jsonl").returncode == 2, "compare: runs that score different positions are refused")
    write(work / "other.jsonl", arm[:3] + [dict(arm[3], truth=4)])
    check(run("compare", work / "r.jsonl", work / "other.jsonl").returncode == 2, "compare: runs that disagree on a corpus token are refused")


def shifted(count, factor, flip=()):
    reference, arm = [], []
    for document in range(count):
        for position in range(8):
            p = 0.30 + 0.05 * ((document * 3 + position) % 7)
            q = p ** factor
            truth = 11
            other = 12 if (document, position) not in flip else 11
            reference.append(row(f"d{document}", position, truth, [[11, log(p)], [12, log((1 - p) / 2)]]))
            best = [[11, log(q)], [12, log((1 - q) / 2)]] if other == 12 else [[12, log(q)], [11, log((1 - q) / 2)]]
            arm.append(row(f"d{document}", position, truth, best))
    return reference, arm


def suite(path, passed):
    Path(path).write_text(json.dumps({"format": "sparkpipe-ds4-eval92-phase-v1", "results": [{"id": f"case-{index}", "status": "ok", "passed": flag} for index, flag in enumerate(passed)]}))


def gate_cases(work):
    reference, close = shifted(12, 1.001)
    _, far = shifted(12, 1.03)
    write(work / "g-ref.jsonl", reference)
    write(work / "g-close.jsonl", close)
    write(work / "g-far.jsonl", far)
    for name, arm in (("close", "g-close.jsonl"), ("far", "g-far.jsonl"), ("repeat", "g-close.jsonl")):
        base = "g-close.jsonl" if name == "repeat" else "g-ref.jsonl"
        done = run("compare", work / base, work / arm, "--out", work / f"g-{name}.json", "--replicates", "400")
        check(done.returncode == 0, f"gate: the {name} comparison is written")
    close_report = json.loads((work / "g-close.json").read_text())["overall"]["dnll_rel_pct"]
    far_report = json.loads((work / "g-far.json").read_text())["overall"]["dnll_rel_pct"]
    check(abs(close_report["estimate"] - 0.1) < 1e-6 and close_report["lower"] <= close_report["estimate"] <= close_report["upper"]
          and abs(far_report["estimate"] - 3.0) < 1e-6, "gate: a model whose log-probs are scaled by 1.001 or 1.03 shows an NLL change of 0.1% or 3%")
    passing = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--out", work / "verdict.json")
    verdict = json.loads((work / "verdict.json").read_text())
    check(passing.returncode == 0 and verdict["passed"] and "VARIANT GATE PASS" in passing.stdout and [item["check"] for item in verdict["checks"]] == ["repeat", "corpus NLL", "corpus next-token accuracy"]
          and verdict["criteria"] == {"dnll_rel_pct_abs": 0.5, "mcnemar_alpha": 0.05, "bound_level": 0.95, "near_tie_nats": 0.001},
          "gate: a variant whose repeat is identical and whose NLL bound is inside 0.5% passes, and the verdict records the criteria")
    failing = run("gate", "--corpus", work / "g-far.json", "--repeat", work / "g-repeat.json")
    check(failing.returncode == 1 and "FAIL corpus NLL" in failing.stdout and "VARIANT GATE FAIL" in failing.stdout, "gate: an NLL bound outside 0.5% fails")
    unstable = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-close.json")
    check(unstable.returncode == 1 and "FAIL repeat" in unstable.stdout, "gate: a repeat that returns other log-probs fails")
    _, flipped = shifted(12, 1.001, flip={(document, 0) for document in range(9)})
    write(work / "g-flip.jsonl", flipped)
    run("compare", work / "g-ref.jsonl", work / "g-flip.jsonl", "--out", work / "g-flip.json", "--replicates", "400")
    worse = run("gate", "--corpus", work / "g-flip.json", "--repeat", work / "g-repeat.json")
    check(worse.returncode == 1 and "FAIL corpus next-token accuracy" in worse.stdout, "gate: nine positions only the reference predicts and none the other way fail next-token accuracy")
    suite(work / "s-ref.json", [True] * 20)
    suite(work / "s-near.json", [False] * 3 + [True] * 17)
    suite(work / "s-lost.json", [False] * 8 + [True] * 12)
    near = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--suite", f"tasks={work / 's-ref.json'}:{work / 's-near.json'}")
    check(near.returncode == 0 and "PASS suite tasks: 20 of 20 against 17" in near.stdout, "gate: three cases lost of twenty is no significant regression")
    lost = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--suite", f"tasks={work / 's-ref.json'}:{work / 's-lost.json'}")
    check(lost.returncode == 1 and "FAIL suite tasks" in lost.stdout, "gate: eight cases lost and none gained is a regression")
    Path(work / "s-error.json").write_text(json.dumps({"results": [{"id": "case-0", "status": "error", "passed": False}]}))
    broken = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--suite", f"tasks={work / 's-ref.json'}:{work / 's-error.json'}")
    check(broken.returncode == 2 and "did not complete" in broken.stderr, "gate: a suite run with a failed case decides nothing")


def depth_cases(work, index, tokens):
    joined = run("join", "--index", index, "--tokens", tokens, "--out-index", work / "J.index.json", "--out-tokens", work / "J.tokens.u32", "--length", "1100")
    document = json.loads((work / "J.index.json").read_text())
    check(joined.returncode == 0 and len(document["documents"]) == 1 and document["documents"][0]["length"] == 1100
          and (work / "J.tokens.u32").read_bytes() == tokens.read_bytes()[:4400] and document["corpus_sha256"] == hashlib.sha256(tokens.read_bytes()[:4400]).hexdigest(),
          "join: the documents back to back become one document, cut to the length asked for")
    first, second = Api(0.0), Api(0.02)
    try:
        for name, api in (("ja", first), ("jb", second)):
            done = run("score", "--endpoint", api.endpoint, "--index", work / "J.index.json", "--tokens", work / "J.tokens.u32", "--starts", "100:1000:300", "--stride", "4", "--count", "6",
                       "--out", work / f"{name}.jsonl")
            check(done.returncode == 0 and len(rows(work / f"{name}.jsonl")) == 24, f"depth: run {name} scores four runs of six positions from a range of starts")
        check(sorted({request["cache_points"][0] for request in first.requests}) == [64, 384, 640, 704, 960], "depth: deep prompts name the last 64-token boundary before their end")
    finally:
        first.close()
        second.close()
    done = run("compare", work / "ja.jsonl", work / "jb.jsonl", "--unit", "run", "--out", work / "depth.json", "--replicates", "200")
    report = json.loads((work / "depth.json").read_text())
    check(done.returncode == 0 and report["overall"]["units"] == 4 and list(report["depth"]) == ["from 0"] and report["overall"]["dnll_rel_pct"]["lower"] is not None,
          "depth: with --unit run each run of positions is a bootstrap unit and the report is by depth")
    refused = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--depth", work / "g-close.json")
    check(refused.returncode == 2 and "--unit run" in refused.stderr, "depth: the gate refuses a depth comparison whose unit is the document")
    accepted = run("gate", "--corpus", work / "g-close.json", "--repeat", work / "g-repeat.json", "--depth", work / "depth.json")
    check("depth NLL" in accepted.stdout and "depth next-token accuracy" in accepted.stdout, "depth: the gate holds the depth comparison to the same NLL margin")


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    index, tokens = score_cases(work)
    compare_cases(work)
    gate_cases(work)
    depth_cases(work, index, tokens)
print(("FAIL " if failures else "PASS ") + f"ab_api_score: {len(failures)} failures")
sys.exit(1 if failures else 0)
