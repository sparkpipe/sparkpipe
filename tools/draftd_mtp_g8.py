import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import (_E4M3_LUT, bf16_round_f32, bf16_to_f32,
                                 read_fixture)

FINAL_LAYER_KEY = "pos{p:04d}_layer{layer:04d}_streams"


def hc_mean(streams, hc, hidden):
    return bf16_round_f32(streams.reshape(hc, hidden).mean(axis=0))


def fixture_cases(paths, final_layer, hc, hidden, depth):
    cases, pool = [], []
    for path in paths:
        _, arrays = read_fixture(path)
        tokens = [int(t) for t in arrays["prompt_token_ids"]] + [int(t) for t in arrays["generated_token_ids"]]
        layers = sorted({int(k.split("_layer")[1][:4]) for k in arrays if k.endswith("_streams")})
        for p in range(len(tokens)):
            for layer in layers:
                streams = bf16_to_f32(arrays[FINAL_LAYER_KEY.format(p=p, layer=layer)])
                rows = streams.reshape(hc, hidden)
                if layer == final_layer and p + 2 < len(tokens):
                    steps = min(depth, len(tokens) - p - 2)
                    cases.append({"id": f"{os.path.basename(path)}:p{p}", "kind": "real-tap",
                                  "hidden": hc_mean(streams, hc, hidden),
                                  "fed": tokens[p + 1:p + 1 + steps], "target": tokens[p + 2:p + 2 + steps]})
                pool.append((f"{os.path.basename(path)}:p{p}:L{layer}:mean", hc_mean(streams, hc, hidden)))
                for s in range(hc):
                    pool.append((f"{os.path.basename(path)}:p{p}:L{layer}:s{s}", bf16_round_f32(rows[s])))
    return cases, pool


def stream_cases(pool, streams, count, depth, seed):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(count):
        tokens = streams[i % len(streams)]
        start = int(rng.integers(0, len(tokens) - depth - 1))
        source, hidden = pool[i % len(pool)]
        cases.append({"id": f"impl{i}:{source}:t{start}", "kind": "impl", "hidden": hidden,
                      "fed": tokens[start:start + depth], "target": tokens[start + 1:start + 1 + depth]})
    return cases


def load_streams(path):
    data = json.load(open(path))
    return [[int(t) for t in r["token_ids"]] for r in data["results"] if len(r["token_ids"]) > 16]


def _reference_worker(arguments):
    checkpoint, header, cases = arguments
    import glm53flash_mtp_reference as reference
    engine, config = reference.load_engine(checkpoint, header, True)
    memo = {}
    tensor = engine.tensor

    def cached_tensor(name):
        if name not in memo:
            memo[name] = tensor(name)
        return memo[name]

    def expert_weight(name):
        raw = engine.raw(name)
        scale = engine.raw(name + "_scale_inv").astype(np.float32)
        rows, cols = raw.shape
        codes = _E4M3_LUT[raw].astype(np.float32).reshape(rows // 128, 128, cols // 128, 128)
        return (codes * scale[:, None, :, None]).reshape(rows, cols)

    engine.tensor = cached_tensor
    engine.expert_weight = expert_weight
    mtp = reference.MtpReference(engine, int(config["num_hidden_layers"]))
    out = []
    for case in cases:
        cache = []
        hidden = case["hidden"]
        steps = []
        for token in case["fed"]:
            hidden_in = hidden
            hidden, head_in = mtp.step(hidden, int(token), "embed_hidden", cache)
            steps.append({"hidden_in": hidden_in.astype(np.float32), "head_in": head_in.astype(np.float32),
                          "slots": np.stack([np.asarray(row, dtype=np.float32) if row.dtype != np.uint16
                                             else bf16_to_f32(row) for row in cache])})
        out.append((case["id"], steps))
    return out


def head_top2(checkpoint, columns, chunk=8192):
    import glm53flash_mtp_reference as reference
    from t1_reference_common import Safetensors
    st = Safetensors(checkpoint)
    entry = st.entry("lm_head.weight")
    matrix = np.stack(columns, axis=1).astype(np.float32)
    count = matrix.shape[1]
    best = np.full(count, -np.inf, dtype=np.float32)
    second = np.full(count, -np.inf, dtype=np.float32)
    best_token = np.full(count, -1, dtype=np.int64)
    for start in range(0, entry["shape"][0], chunk):
        rows = st.raw_rows("lm_head.weight", start, min(chunk, entry["shape"][0] - start))
        scores = bf16_to_f32(rows) @ matrix
        order = np.argsort(-scores, axis=0, kind="stable")[:2]
        top = scores[order[0], np.arange(count)]
        nxt = scores[order[1], np.arange(count)]
        better = top > best
        second = np.where(better, np.maximum(best, nxt), np.maximum(second, top))
        best_token[better] = start + order[0][better]
        best = np.where(better, top, best)
    return best_token, best - second


def run_reference(checkpoint, header, cases, workers):
    chunks = [(checkpoint, header, cases[i::workers]) for i in range(workers)]
    steps = {}
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for part in pool.map(_reference_worker, chunks):
            for case_id, rows in part:
                steps[case_id] = rows
    flat, keys = [], []
    for case in cases:
        for step, row in enumerate(steps[case["id"]]):
            flat.append(row["head_in"])
            keys.append((case["id"], step))
    tokens, margins = head_top2(checkpoint, flat)
    result = {}
    for (case_id, step), token, margin in zip(keys, tokens, margins):
        row = steps[case_id][step]
        result.setdefault(case_id, []).append({"top1": int(token), "margin": float(margin),
                                               "hidden_in": row["hidden_in"], "slots": row["slots"],
                                               "head_in_sha": hashlib.sha256(row["head_in"].tobytes()).hexdigest()[:16]})
    return result


def run_device(drafter, cases, forced=None):
    import torch
    result = {}
    for case in cases:
        cache = torch.zeros((drafter.max_chain, drafter.latent), dtype=torch.float32, device=drafter.device)
        hidden = torch.as_tensor(case["hidden"], device=drafter.device)
        rows = []
        for step, token in enumerate(case["fed"]):
            if forced is not None:
                ref = forced[case["id"]][step]
                hidden = torch.as_tensor(ref["hidden_in"], device=drafter.device)
                if step:
                    cache[:step] = torch.as_tensor(ref["slots"][:step], device=drafter.device)
            hidden, head_in, best = drafter.step(hidden, torch.as_tensor([int(token)], device=drafter.device), cache, step)
            top2 = torch.topk(drafter.logits, 2).values
            rows.append({"top1": int(best), "margin": float(top2[0] - top2[1]),
                         "head_in_sha": hashlib.sha256(head_in.cpu().numpy().tobytes()).hexdigest()[:16]})
        result[case["id"]] = rows
    return result


def compare(cases, reference, device, near_tie):
    stats = {"real-tap": {"positions": 0, "agree": 0, "hit_ref": 0, "hit_dev": 0},
             "impl": {"positions": 0, "agree": 0, "hit_ref": 0, "hit_dev": 0}}
    per_depth = {}
    disagreements = []
    for case in cases:
        s = stats[case["kind"]]
        for step, (r, d) in enumerate(zip(reference[case["id"]], device[case["id"]])):
            s["positions"] += 1
            same = r["top1"] == d["top1"]
            s["agree"] += int(same)
            s["hit_ref"] += int(r["top1"] == case["target"][step])
            s["hit_dev"] += int(d["top1"] == case["target"][step])
            depth = per_depth.setdefault(f"{case['kind']}/d{step + 1}",
                                         {"positions": 0, "agree": 0, "hit_ref": 0, "hit_dev": 0})
            depth["positions"] += 1
            depth["agree"] += int(same)
            depth["hit_ref"] += int(r["top1"] == case["target"][step])
            depth["hit_dev"] += int(d["top1"] == case["target"][step])
            s["head_in_equal"] = s.get("head_in_equal", 0) + int(r["head_in_sha"] == d["head_in_sha"])
            if not same:
                disagreements.append({"case": case["id"], "step": step, "ref": r["top1"], "dev": d["top1"],
                                      "ref_margin": r["margin"], "dev_margin": d["margin"],
                                      "near_tie": r["margin"] <= near_tie})
    for s in stats.values():
        s["agreement"] = s["agree"] / s["positions"] if s["positions"] else None
    total = sum(s["positions"] for s in stats.values())
    agree = sum(s["agree"] for s in stats.values())
    return {"positions": total, "agree": agree, "agreement": agree / total if total else None,
            "classes": stats, "per_depth": per_depth, "disagreements": disagreements,
            "non_tie_disagreements": sum(1 for d in disagreements if not d["near_tie"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description="G8: draftd MTP top-1 against the numpy MTP reference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--fixtures", nargs="+", required=True)
    parser.add_argument("--streams", required=True, help="bench result json with results[].token_ids")
    parser.add_argument("--impl-chains", type=int, default=150)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--near-tie", type=float, default=0.0625)
    parser.add_argument("--floor", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = json.load(open(os.path.join(args.checkpoint, "config.json")))
    config = config.get("text_config", config)
    hc, hidden = int(config["hc_mult"]), int(config["hidden_size"])
    real, pool = fixture_cases(args.fixtures, int(config["num_hidden_layers"]) - 1, hc, hidden, args.depth)
    cases = real + stream_cases(pool, load_streams(args.streams), args.impl_chains, args.depth, args.seed)
    started = time.time()
    reference = run_reference(args.checkpoint, args.header, cases, args.workers)
    reference_s = time.time() - started
    from draftd_glm53flash_mtp import Glm53FlashMtpDrafter
    drafter = Glm53FlashMtpDrafter(args.checkpoint)
    started = time.time()
    forced = run_device(drafter, cases, forced=reference)
    device_s = time.time() - started
    drafter.k.wide = 1
    forced_wide = run_device(drafter, cases, forced=reference)
    drafter.k.wide = 0
    free = run_device(drafter, cases)
    free_again = run_device(drafter, cases)
    report = compare(cases, reference, forced, args.near_tie)
    free_report = compare(cases, reference, free, args.near_tie)
    wide_report = compare(cases, reference, forced_wide, args.near_tie)
    order_report = compare(cases, forced_wide, forced, args.near_tie)
    report["accumulation_control"] = {
        "reference_vs_device_f64": {k: wide_report[k] for k in ("agree", "agreement", "non_tie_disagreements")},
        "device_f64_vs_device_f32": {k: order_report[k] for k in ("agree", "agreement", "non_tie_disagreements")}}
    deterministic = all(a["head_in_sha"] == b["head_in_sha"] and a["top1"] == b["top1"]
                        for cid in free for a, b in zip(free[cid], free_again[cid]))
    report.update({"checkpoint_sha256": json.load(open(os.path.join(args.checkpoint, "SUBSET.json")))["file_sha256"],
                   "depth": args.depth, "device_s": device_s, "reference_s": reference_s,
                   "device_bytes": drafter.device_bytes(), "run_to_run_identical": deterministic,
                   "free_chain": {k: free_report[k] for k in ("positions", "agree", "agreement", "per_depth",
                                                             "non_tie_disagreements")}})
    verdict = "PASS" if report["agreement"] >= args.floor and deterministic else "FAIL"
    report["verdict"] = verdict
    json.dump(report, open(args.output, "w"), indent=1)
    real_stats = report["classes"]["real-tap"]
    d1 = report["per_depth"].get("real-tap/d1", {})
    print(f"G8-MTP {verdict} identical-input positions={report['positions']} agree={report['agree']} "
          f"agreement={report['agreement']:.4f} non_tie_disagreements={report['non_tie_disagreements']} "
          f"real_tap={real_stats['agree']}/{real_stats['positions']} real_tap_d1_hits ref={d1.get('hit_ref')} "
          f"dev={d1.get('hit_dev')}/{d1.get('positions')} free_chain_agreement={free_report['agreement']:.4f} "
          f"run_to_run_identical={deterministic} control ref_vs_f64={wide_report['agreement']:.4f} "
          f"f64_vs_f32={order_report['agreement']:.4f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
