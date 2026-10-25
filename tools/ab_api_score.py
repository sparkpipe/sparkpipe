#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

TOP = 20


def load_corpus(index_path, tokens_path):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index.get("format") != "sparkpipe-ab-corpus-v1":
        raise SystemExit(f"{index_path}: not a sparkpipe-ab-corpus-v1 index")
    data = Path(tokens_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != index["corpus_sha256"]:
        raise SystemExit(f"{tokens_path}: SHA-256 differs from the index's corpus_sha256")
    tokens = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    documents = []
    for record in index["documents"]:
        body = tokens[record["offset"]:record["offset"] + record["length"]]
        raw = b"".join(t.to_bytes(4, "little") for t in body)
        if len(body) != record["length"] or hashlib.sha256(raw).hexdigest() != record["tokens_sha256"]:
            raise SystemExit(f"{record['id']}: tokens differ from the index")
        documents.append({"id": record["id"], "stratum": record["stratum"], "tokens": body})
    return index, documents


def positions(length, first, stride, count):
    chosen = [first + stride * i for i in range(count)]
    if not chosen or chosen[-1] >= length or first < 1:
        raise SystemExit(f"positions {first}+{stride}*{count} do not fit a {length}-token document")
    return chosen


def ask(endpoint, prompt, timeout):
    body = json.dumps({"prompt_token_ids": prompt, "max_tokens": 1, "temperature": 0, "logprobs": TOP, "speculation": False}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score(args):
    _, documents = load_corpus(args.index, args.tokens)
    if args.documents:
        documents = documents[:args.documents]
    done = set()
    out = Path(args.out)
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["document"], row["position"]))
    started = time.time()
    scored = 0
    with out.open("a", encoding="utf-8") as stream:
        for document in documents:
            for position in positions(len(document["tokens"]), args.first, args.stride, args.count):
                if (document["id"], position) in done:
                    continue
                reply = ask(args.endpoint, document["tokens"][:position], args.timeout)
                lists = reply.get("token_logprobs") or []
                if len(lists) != 1 or not lists[0]:
                    raise SystemExit(f"{document['id']}@{position}: the reply has no logprobs: {json.dumps(reply)[:300]}")
                row = {
                    "document": document["id"],
                    "stratum": document["stratum"],
                    "position": position,
                    "truth": document["tokens"][position],
                    "top": lists[0],
                    "usage": reply.get("usage"),
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                scored += 1
            print(f"{document['id']} done, {scored} positions in {time.time() - started:.0f} s", file=sys.stderr)
    return 0


def read(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["document"], row["position"])
        if key in rows:
            raise SystemExit(f"{path}: {key} appears twice")
        rows[key] = row
    return rows


def distribution(top):
    table = {}
    for token, logprob in top:
        if token not in table:
            table[token] = math.exp(logprob) if logprob is not None else 0.0
    return table


def bounded_kl(reference, arm):
    shared = [t for t in reference if t in arm]
    p = [reference[t] for t in shared]
    q = [arm[t] for t in shared]
    p.append(max(0.0, 1.0 - sum(p)))
    q.append(max(0.0, 1.0 - sum(q)))
    total = 0.0
    for a, b in zip(p, q):
        if a > 0.0:
            total += a * math.log(a / max(b, 1e-30))
    return max(0.0, total)


def mcnemar(only_reference, only_arm):
    n = only_reference + only_arm
    if n == 0:
        return 1.0
    k = min(only_reference, only_arm)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def summarize(pairs, seed):
    by_document = {}
    for row in pairs:
        by_document.setdefault(row["document"], []).append(row)
    def means(groups):
        rows = [r for g in groups for r in g]
        n = len(rows)
        return (sum(r["agree"] for r in rows) / n, sum(r["kl"] for r in rows) / n)
    names = sorted(by_document)
    generator = random.Random(seed)
    agree, kl = [], []
    for _ in range(1000):
        sample = [by_document[generator.choice(names)] for _ in names]
        a, k = means(sample)
        agree.append(a)
        kl.append(k)
    agree.sort()
    kl.sort()
    a, k = means(by_document.values())
    reference_right = sum(r["reference_right"] for r in pairs)
    arm_right = sum(r["arm_right"] for r in pairs)
    only_reference = sum(1 for r in pairs if r["reference_right"] and not r["arm_right"])
    only_arm = sum(1 for r in pairs if r["arm_right"] and not r["reference_right"])
    both = [r for r in pairs if r["truth_delta"] is not None]
    return {
        "positions": len(pairs),
        "documents": len(names),
        "top1_agreement": a,
        "top1_agreement_ci95": [agree[25], agree[974]],
        "kl_lower_bound": k,
        "kl_lower_bound_ci95": [kl[25], kl[974]],
        "reference_accuracy": reference_right / len(pairs),
        "arm_accuracy": arm_right / len(pairs),
        "accuracy_only_reference": only_reference,
        "accuracy_only_arm": only_arm,
        "accuracy_mcnemar_p": mcnemar(only_reference, only_arm),
        "truth_logprob_positions": len(both),
        "truth_logprob_arm_minus_reference": sum(r["truth_delta"] for r in both) / len(both) if both else None,
    }


def compare(args):
    reference = read(args.reference)
    arm = read(args.arm)
    if set(reference) != set(arm):
        raise SystemExit(f"the two runs score different positions: {len(set(reference) ^ set(arm))} differ")
    pairs = []
    identical = 0
    for key in sorted(reference):
        r, a = reference[key], arm[key]
        if r["truth"] != a["truth"]:
            raise SystemExit(f"{key}: the runs disagree on the true token")
        identical += 1 if r["top"] == a["top"] else 0
        p, q = distribution(r["top"]), distribution(a["top"])
        truth = r["truth"]
        pairs.append({
            "document": key[0],
            "stratum": r["stratum"],
            "agree": 1 if r["top"][0][0] == a["top"][0][0] else 0,
            "kl": bounded_kl(p, q),
            "reference_right": r["top"][0][0] == truth,
            "arm_right": a["top"][0][0] == truth,
            "truth_delta": math.log(q[truth]) - math.log(p[truth]) if truth in p and truth in q and p[truth] > 0.0 and q[truth] > 0.0 else None,
        })
    report = {"reference": args.reference, "arm": args.arm, "identical_lists": identical, "overall": summarize(pairs, args.seed), "strata": {}}
    for stratum in sorted({p["stratum"] for p in pairs}):
        report["strata"][stratum] = summarize([p for p in pairs if p["stratum"] == stratum], args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Teacher-forced next-token A/B through the serving API: each scored position is one greedy request with max_tokens 1 and the top 20 log-probs, sent one at a time.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score a frozen corpus against one arm's endpoint; reruns append only the missing positions")
    s.add_argument("--endpoint", required=True)
    s.add_argument("--index", required=True)
    s.add_argument("--tokens", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--first", type=int, default=256)
    s.add_argument("--stride", type=int, default=16)
    s.add_argument("--count", type=int, default=64)
    s.add_argument("--documents", type=int, default=0)
    s.add_argument("--timeout", type=float, default=600.0)
    c = sub.add_parser("compare", help="pair two score files: top-1 agreement, a lower bound on KL(reference || arm) over the shared listed tokens plus one bucket for the rest, next-token accuracy with an exact McNemar test, and the true token's log-prob change where both lists hold it")
    c.add_argument("reference")
    c.add_argument("arm")
    c.add_argument("--out")
    c.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    return score(args) if args.command == "score" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
