#!/usr/bin/env python3
import json
import sys


def positions(entry):
    if "token_logprobs" in entry:
        rows = entry["token_logprobs"] or []
        return [row[0][0] for row in rows], rows
    return entry["tokens"], entry["logprobs"] or []


def alternatives(row):
    return {token: logprob for token, logprob in row[1:]}


def margin(row):
    values = sorted((logprob for _, logprob in row[1:]), reverse=True)
    return values[0] - values[1] if len(values) > 1 else float("inf")


def load(path):
    data = json.load(open(path))
    for key in ("results", "b1"):
        if isinstance(data.get(key), dict):
            return data[key]
    return data


def main():
    old = load(sys.argv[1])
    new = load(sys.argv[2])
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    report = {"prompts": {}, "compared": 0, "agree": 0, "max_delta": 0.0, "large": []}
    for name in old:
        if name not in new:
            report["large"].append(f"{name}: missing in the new capture")
            continue
        old_tokens, old_rows = positions(old[name])
        new_tokens, new_rows = positions(new[name])
        length = min(len(old_tokens), len(new_tokens))
        flip = next((index for index in range(length) if old_tokens[index] != new_tokens[index]), None)
        compared = length if flip is None else flip + 1
        delta = 0.0
        served = 0.0
        for index in range(compared):
            a = alternatives(old_rows[index])
            b = alternatives(new_rows[index])
            for token in set(a) & set(b):
                delta = max(delta, abs(a[token] - b[token]))
            if index != flip:
                served = max(served, abs(old_rows[index][0][1] - new_rows[index][0][1]))
        item = {"old_len": len(old_tokens), "new_len": len(new_tokens), "compared": compared,
                "agree": compared - (0 if flip is None else 1), "first_flip": flip,
                "identical": old_tokens == new_tokens and old_rows == new_rows,
                "max_delta": round(delta, 6), "served_max_delta": round(served, 6)}
        if flip is not None:
            item["old_margin_at_flip"] = round(margin(old_rows[flip]), 4)
            item["new_token_old_rank"] = next((rank for rank, (token, _) in enumerate(old_rows[flip][1:], 1)
                                               if token == new_tokens[flip]), None)
            if item["old_margin_at_flip"] >= limit:
                report["large"].append(f"{name}: flip at {flip} with old top-1/top-2 margin {item['old_margin_at_flip']}")
        if flip is None and len(old_tokens) != len(new_tokens):
            report["large"].append(f"{name}: lengths {len(old_tokens)} vs {len(new_tokens)} with no token flip")
        if served >= limit:
            report["large"].append(f"{name}: served-token logprob delta {served:.4f}")
        report["prompts"][name] = item
        report["compared"] += compared
        report["agree"] += item["agree"]
        report["max_delta"] = max(report["max_delta"], round(delta, 6))
        report["served_max_delta"] = max(report.get("served_max_delta", 0.0), round(served, 6))
    report["identical"] = all(item["identical"] for item in report["prompts"].values()) and not report["large"]
    for name, item in report["prompts"].items():
        flip = "" if item["first_flip"] is None else f" flip@{item['first_flip']} margin {item['old_margin_at_flip']} rank {item['new_token_old_rank']}"
        print(f"{name}: {'IDENTICAL' if item['identical'] else 'differs'} agree {item['agree']}/{item['compared']} len {item['old_len']}->{item['new_len']} served_max_delta {item['served_max_delta']} topk_max_delta {item['max_delta']}{flip}")
    print(f"TOTAL top-1 agreement {report['agree']}/{report['compared']} served_max_delta {report.get('served_max_delta', 0.0)} topk_max_delta {report['max_delta']} large {len(report['large'])}")
    for line in report["large"]:
        print("LARGE " + line)
    if len(sys.argv) > 4:
        json.dump(report, open(sys.argv[4], "w"), indent=1)
    return 2 if report["large"] else 0


if __name__ == "__main__":
    sys.exit(main())
