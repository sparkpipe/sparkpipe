#!/usr/bin/env python3
"""Per-position accuracy metrics from merged score dumps (reference P, arm Q).

compare  ref.merged arm.merged corpus.jsonl -> per-document metrics JSON:
  bucketed KL KL_b(P||Q) over the reference global top-k probe set plus one tail
  bucket (a lower bound on the true KL), reference tail mass, top-1 agreement
  with reference near ties (top-2 gap < 1e-3 nats) excluded and counted,
  decisive flips (reference margin > 1 nat), and dlogp = logp_Q(y) - logp_P(y)
  for the corpus next token y (dNLL = -dlogp).
partial  node-local exact KL partial of one rank's Tier-2 shard logits:
  sum_{v in shard} p_v (lp_v - LSE_P - lq_v + LSE_Q), plus the shard P mass.
combine  partial.r00 .. partial.rNN -> exact full-vocabulary KL per Tier-2 row,
  summed in rank order.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_merge

DECISIVE_NATS = 1.0
TAIL_ROUNDING = 1.0e-12
PARTIAL_MAGIC = b"SPKLPRT1"
PARTIAL_HEADER = struct.Struct("<8sIIQ32s32s")
PARTIAL_DTYPE = np.dtype([("key", "<u8"), ("position", "<u4"), ("pad", "<u4"), ("partial", "<f8"), ("mass", "<f8")])


class CompareError(Exception):
    pass


def logits_at(merged, probes, index, ids):
    table = {int(i): float(v) for i, v in zip(merged["top_ids"][index], merged["top_logits"][index])}
    for item in probes[index]:
        value = float(item["logit"])
        previous = table.get(int(item["id"]))
        if previous is not None and previous != value:
            raise CompareError(f"row {index}: probe and top-k logits disagree for id {int(item['id'])}")
        table[int(item["id"])] = value
    missing = [i for i in ids if i not in table]
    if missing:
        raise CompareError(f"row {index}: logits missing for ids {missing[:4]} (arm run needs the pass-2 probe file)")
    return np.array([table[i] for i in ids], dtype=np.float64)


def tail_mass(log_probabilities):
    total = np.logaddexp.reduce(log_probabilities)
    tail = -np.expm1(total)
    if tail < -TAIL_ROUNDING:
        raise CompareError(f"bucket mass exceeds one by {-tail}")
    return max(tail, 0.0)


def bucketed_kl(lp, lq):
    p = np.exp(lp)
    kl = float(np.sum(p * (lp - lq)))
    p_tail = tail_mass(lp)
    q_tail = tail_mass(lq)
    if p_tail > 0.0:
        if q_tail <= 0.0:
            return float("inf"), p_tail
        kl += p_tail * (np.log(p_tail) - np.log(q_tail))
    return kl, p_tail


def compare(reference_path, arm_path, corpus_path):
    ref_header, ref, ref_probes = score_merge.read_merged(reference_path)
    arm_header, arm, arm_probes = score_merge.read_merged(arm_path)
    if ref_header["vocabulary"] != arm_header["vocabulary"]:
        raise CompareError("reference and arm vocabularies differ")
    ref_rows = score_merge.row_index(ref)
    arm_rows = score_merge.row_index(arm)
    documents = []
    for doc, tokens in score_merge.read_corpus(corpus_path):
        keys = score_merge.row_keys(tokens)
        result = {"doc": doc, "positions": [], "kl_b": [], "tail": [], "top1_agree": [], "near_tie": [],
                  "decisive_flip": [], "dlogp": []}
        for position in range(len(tokens) - 1):
            identity = (keys[position], position)
            if identity not in ref_rows or identity not in arm_rows:
                raise CompareError(f"{doc}@{position}: row missing from {'reference' if identity not in ref_rows else 'arm'} dump")
            r = ref_rows[identity]
            a = arm_rows[identity]
            probe_ids = [int(v) for v in ref["top_ids"][r]]
            target = tokens[position + 1]
            lp = logits_at(ref, ref_probes, r, probe_ids) - ref["log_z"][r]
            lq = logits_at(arm, arm_probes, a, probe_ids) - arm["log_z"][a]
            kl, tail = bucketed_kl(lp, lq)
            gap = float(ref["top_logits"][r][0]) - float(ref["top_logits"][r][1])
            agree = int(arm["top_ids"][a][0]) == int(ref["top_ids"][r][0])
            near = gap < score_merge.NEAR_TIE_NATS
            ref_target = logits_at(ref, ref_probes, r, [target])[0] - ref["log_z"][r]
            arm_target = logits_at(arm, arm_probes, a, [target])[0] - arm["log_z"][a]
            result["positions"].append(position)
            result["kl_b"].append(kl)
            result["tail"].append(tail)
            result["top1_agree"].append(bool(agree))
            result["near_tie"].append(bool(near))
            result["decisive_flip"].append(bool(not agree and gap > DECISIVE_NATS))
            result["dlogp"].append(float(arm_target - ref_target))
        documents.append(result)
    return documents


def summarize(documents):
    kl = np.concatenate([np.array(d["kl_b"]) for d in documents]) if documents else np.zeros(0)
    tail = np.concatenate([np.array(d["tail"]) for d in documents]) if documents else np.zeros(0)
    near = np.concatenate([np.array(d["near_tie"], dtype=bool) for d in documents]) if documents else np.zeros(0, dtype=bool)
    agree = np.concatenate([np.array(d["top1_agree"], dtype=bool) for d in documents]) if documents else np.zeros(0, dtype=bool)
    flips = np.concatenate([np.array(d["decisive_flip"], dtype=bool) for d in documents]) if documents else np.zeros(0, dtype=bool)
    dlogp = np.concatenate([np.array(d["dlogp"]) for d in documents]) if documents else np.zeros(0)
    scored = ~near
    return {"positions": int(kl.size), "documents": len(documents),
            "kl_b_mean": float(kl.mean()) if kl.size else None,
            "tail_mass_p99": float(np.quantile(tail, 0.99)) if tail.size else None,
            "near_ties": int(near.sum()),
            "top1_agreement": float(agree[scored].mean()) if scored.any() else None,
            "decisive_flips": int(flips.sum()),
            "dlogp_mean": float(dlogp.mean()) if dlogp.size else None}


def read_tier2(path):
    data = Path(path).read_bytes()
    fields = score_merge.HEADER.unpack_from(data, 0)
    header = dict(zip(("magic", "version", "header_bytes", "tp_rank", "tp_degree", "shard_begin", "shard_end",
                       "vocabulary", "hidden_dimension", "top_k", "tier2", "arm_digest", "probe_sha256",
                       "tier2_sha256"), fields))
    if header["magic"] != score_merge.TIER2_OUT_MAGIC or header["version"] != score_merge.VERSION:
        raise CompareError(f"{path}: not a Tier-2 dump")
    width = header["shard_end"] - header["shard_begin"]
    record = 16 + 4 * width
    if (len(data) - score_merge.HEADER.size) % record != 0:
        raise CompareError(f"{path}: truncated Tier-2 dump")
    rows = {}
    offset = score_merge.HEADER.size
    while offset < len(data):
        key, position, used = struct.unpack_from("<QII", data, offset)
        if used != width:
            raise CompareError(f"{path}: Tier-2 row width {used} != shard width {width}")
        logits = np.frombuffer(data, dtype="<f4", count=width, offset=offset + 16).astype(np.float64)
        previous = rows.setdefault((key, position), logits)
        if previous is not logits and not np.array_equal(previous, logits):
            raise CompareError(f"{path}: Tier-2 row {(key, position)} appears twice with different logits")
        offset += record
    return header, rows


def tier2_identities(merged):
    wanted = score_merge.ROW_KEY_VALID | score_merge.ROW_TIER2
    return sorted({(int(merged["key"][i]), int(merged["position"][i])) for i in range(len(merged))
                   if merged["flags"][i] & wanted == wanted})


def partial(ref_tier2, arm_tier2, ref_merged, arm_merged):
    ref_header, ref_rows = read_tier2(ref_tier2)
    arm_header, arm_rows = read_tier2(arm_tier2)
    for name in ("tp_rank", "tp_degree", "shard_begin", "shard_end", "tier2_sha256"):
        if ref_header[name] != arm_header[name]:
            raise CompareError(f"Tier-2 header {name} differs between reference and arm")
    _, ref, _ = score_merge.read_merged(ref_merged)
    _, arm, _ = score_merge.read_merged(arm_merged)
    ref_index = score_merge.row_index(ref)
    arm_index = score_merge.row_index(arm)
    identities = sorted(ref_rows)
    if identities != sorted(arm_rows):
        raise CompareError("reference and arm Tier-2 row sets differ")
    if identities != tier2_identities(ref) or identities != tier2_identities(arm):
        raise CompareError("Tier-2 dump rows differ from the merged rows flagged Tier-2 (incomplete Tier-2 dump)")
    missing = [identity for identity in identities if identity not in ref_index or identity not in arm_index]
    if missing:
        raise CompareError(f"Tier-2 rows {missing[:4]} have no finite merged row")
    out = np.zeros(len(identities), dtype=PARTIAL_DTYPE)
    for index, identity in enumerate(identities):
        lp = ref_rows[identity] - ref["log_z"][ref_index[identity]]
        lq = arm_rows[identity] - arm["log_z"][arm_index[identity]]
        p = np.exp(lp)
        out[index] = (identity[0], identity[1], 0, float(np.sum(p * (lp - lq))), float(np.sum(p)))
    return ref_header, out


def write_partial(path, header, rows):
    with open(path, "wb") as handle:
        handle.write(PARTIAL_HEADER.pack(PARTIAL_MAGIC, header["tp_rank"], header["tp_degree"], len(rows),
                                         header["tier2_sha256"], header["arm_digest"]))
        handle.write(rows.tobytes())


def read_partial(path):
    data = Path(path).read_bytes()
    magic, rank, degree, count, tier2_sha, arm = PARTIAL_HEADER.unpack_from(data, 0)
    if magic != PARTIAL_MAGIC or len(data) != PARTIAL_HEADER.size + count * PARTIAL_DTYPE.itemsize:
        raise CompareError(f"{path}: not a KL partial file")
    rows = np.frombuffer(data, dtype=PARTIAL_DTYPE, count=count, offset=PARTIAL_HEADER.size)
    return {"rank": rank, "degree": degree, "tier2_sha256": tier2_sha, "rows": rows}


def combine(paths):
    parts = sorted((read_partial(p) for p in paths), key=lambda item: item["rank"])
    degree = parts[0]["degree"]
    if [p["rank"] for p in parts] != list(range(degree)):
        raise CompareError("partials must be exactly ranks 0..tp_degree-1")
    for part in parts:
        if part["tier2_sha256"] != parts[0]["tier2_sha256"] or not np.array_equal(part["rows"][["key", "position"]], parts[0]["rows"][["key", "position"]]):
            raise CompareError(f"rank {part['rank']}: Tier-2 rows differ from rank 0")
    kl = np.zeros(len(parts[0]["rows"]), dtype=np.float64)
    mass = np.zeros(len(parts[0]["rows"]), dtype=np.float64)
    for part in parts:
        kl = kl + part["rows"]["partial"]
        mass = mass + part["rows"]["mass"]
    if mass.size and np.max(np.abs(mass - 1.0)) > 1.0e-6:
        raise CompareError(f"reference probability mass off by {float(np.max(np.abs(mass - 1.0)))}")
    return parts[0]["rows"][["key", "position"]], kl, mass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("reference")
    cmp_parser.add_argument("arm")
    cmp_parser.add_argument("corpus")
    cmp_parser.add_argument("--out", required=True)
    partial_parser = sub.add_parser("partial")
    partial_parser.add_argument("--ref-tier2", required=True)
    partial_parser.add_argument("--arm-tier2", required=True)
    partial_parser.add_argument("--ref-merged", required=True)
    partial_parser.add_argument("--arm-merged", required=True)
    partial_parser.add_argument("--out", required=True)
    combine_parser = sub.add_parser("combine")
    combine_parser.add_argument("partials", nargs="+")
    combine_parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "compare":
            documents = compare(arguments.reference, arguments.arm, arguments.corpus)
            report = {"summary": summarize(documents), "documents": documents}
            Path(arguments.out).write_text(json.dumps(report, sort_keys=True))
            print(json.dumps(report["summary"], sort_keys=True))
        elif arguments.command == "partial":
            header, rows = partial(arguments.ref_tier2, arguments.arm_tier2, arguments.ref_merged, arguments.arm_merged)
            write_partial(arguments.out, header, rows)
            print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(Path(arguments.out).read_bytes()).hexdigest()}))
        else:
            identities, kl, mass = combine(arguments.partials)
            report = {"rows": [{"key": int(i["key"]), "position": int(i["position"]), "kl": float(k)}
                               for i, k in zip(identities, kl)],
                      "kl_mean": float(kl.mean()) if kl.size else None,
                      "max_mass_error": float(np.max(np.abs(mass - 1.0))) if mass.size else None}
            Path(arguments.out).write_text(json.dumps(report, sort_keys=True))
            print(json.dumps({k: report[k] for k in ("kl_mean", "max_mass_error")} | {"count": len(report["rows"])}))
    except (CompareError, score_merge.DumpError) as error:
        print(f"score_kl_partial: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
