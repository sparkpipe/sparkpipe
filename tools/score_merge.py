#!/usr/bin/env python3
"""Merge per-rank teacher-forced score dumps and build probe and Tier-2 row files.

Formats are defined by include/sparkpipe/spark_score_dump.h. Every rank file is
read in rank order, checked for a matching header, complete trailer and
identical row identity, then reduced exactly:
  M = max_r m_r, log Z = M + log sum_r s_r exp(m_r - M) summed in rank order,
  global top-k = top-k of the union of the local top-k lists (value desc, id asc).
The merged file is byte-reproducible for identical inputs.

Subcommands:
  targets  corpus.jsonl -> pass-1 probe file (next-token targets only)
  tier2    corpus.jsonl -> Tier-2 row list (deterministic seeded selection)
  merge    score.rNN.bin... -> merged dump, optional pass-2 probe file
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

VERSION = 1
TOP_K = 64
KEY_SEED = 0xCBF29CE484222325
KEY_PRIME = 0x00000100000001B3
MASK64 = (1 << 64) - 1
NO_TOKEN = 0xFFFFFFFF
ROW_KEY_VALID = 0x1
ROW_PROBED = 0x2
ROW_TIER2 = 0x4
ROW_NONFINITE = 0x8
RECORD_ROW = 0x31574F52
RECORD_END = 0x31444E45
NEAR_TIE_NATS = 1.0e-3

ROWS_MAGIC = b"SPSCORE1"
TIER2_OUT_MAGIC = b"SPSCORT2"
PROBE_MAGIC = b"SPPROBE1"
TIER2_MAGIC = b"SPTIER21"
MERGED_MAGIC = b"SPMERGE1"

HEADER = struct.Struct("<8s10I32s32s32s")
TABLE_HEADER = struct.Struct("<8sIIQ")
END = struct.Struct("<IIQQQQQQQ")
ROW_DTYPE = np.dtype([
    ("record_kind", "<u4"), ("flags", "<u4"), ("key", "<u8"), ("wave_ordinal", "<u8"),
    ("position", "<u4"), ("row_in_wave", "<u4"), ("input_token", "<u4"), ("served_token", "<u4"),
    ("probe_count", "<u4"), ("local_max", "<f4"), ("local_sum_exp", "<f8"),
    ("top_ids", "<u4", (TOP_K,)), ("top_logits", "<f4", (TOP_K,)),
])
PROBE_DTYPE = np.dtype([("id", "<u4"), ("logit", "<f4")])
MERGED_HEADER = struct.Struct("<8sIIIIQ32s32s32s32s")
MERGED_DTYPE = np.dtype([
    ("key", "<u8"), ("position", "<u4"), ("input_token", "<u4"), ("served_token", "<u4"),
    ("flags", "<u4"), ("log_z", "<f8"), ("top_ids", "<u4", (TOP_K,)),
    ("top_logits", "<f4", (TOP_K,)), ("probe_count", "<u4"),
])
assert ROW_DTYPE.itemsize == 568 and HEADER.size == 144 and END.size == 64


class DumpError(Exception):
    pass


def key_next(previous, token):
    value = previous
    for index in range(4):
        value ^= (token >> (8 * index)) & 0xFF
        value = (value * KEY_PRIME) & MASK64
    return value


def row_keys(tokens):
    keys = []
    value = KEY_SEED
    for token in tokens:
        value = key_next(value, int(token))
        keys.append(value)
    return keys


def read_corpus(path):
    documents = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                documents.append((str(item["doc"]), [int(t) for t in item["tokens"]]))
    if len({doc for doc, _ in documents}) != len(documents):
        raise DumpError("corpus document ids are not unique")
    return documents


def write_table(path, magic, entries, id_width):
    ordered = sorted(entries.items())
    with open(path, "wb") as handle:
        handle.write(TABLE_HEADER.pack(magic, VERSION, id_width, len(ordered)))
        for (key, position), ids in ordered:
            ids = sorted(ids)
            if len(ids) > id_width:
                raise DumpError("probe entry wider than the table")
            handle.write(struct.pack("<QII", key, position, len(ids)))
            handle.write(struct.pack("<%dI" % id_width, *(ids + [NO_TOKEN] * (id_width - len(ids)))))


def read_table(path, magic):
    data = Path(path).read_bytes()
    found, version, width, count = TABLE_HEADER.unpack_from(data, 0)
    if found != magic or version != VERSION:
        raise DumpError(f"{path}: not a {magic.decode()} v{VERSION} table")
    entry_bytes = 16 + 4 * width
    if len(data) != TABLE_HEADER.size + count * entry_bytes:
        raise DumpError(f"{path}: size does not match {count} entries")
    table = {}
    previous = None
    for index in range(count):
        offset = TABLE_HEADER.size + index * entry_bytes
        key, position, used = struct.unpack_from("<QII", data, offset)
        if used > width or (previous is not None and (key, position) <= previous):
            raise DumpError(f"{path}: entry {index} is malformed or out of order")
        previous = (key, position)
        table[(key, position)] = list(struct.unpack_from("<%dI" % used, data, offset + 16))
    return table, hashlib.sha256(data).digest(), width


def targets_table(documents):
    entries = {}
    for _, tokens in documents:
        keys = row_keys(tokens)
        for position in range(len(tokens) - 1):
            entries.setdefault((keys[position], position), set()).add(tokens[position + 1])
    return entries


def tier2_table(documents, count, seed):
    candidates = []
    for _, tokens in documents:
        keys = row_keys(tokens)
        for position in range(len(tokens) - 1):
            candidates.append((keys[position], position))
    candidates = sorted(set(candidates))
    ranked = sorted(candidates, key=lambda item: hashlib.sha256(
        struct.pack("<QQI", seed, item[0], item[1])).digest())
    return {item: set() for item in ranked[:count]}


def read_rank_file(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size + END.size:
        raise DumpError(f"{path}: truncated")
    fields = HEADER.unpack_from(data, 0)
    header = dict(zip(("magic", "version", "header_bytes", "tp_rank", "tp_degree", "shard_begin",
                       "shard_end", "vocabulary", "hidden_dimension", "top_k", "tier2",
                       "arm_digest", "probe_sha256", "tier2_sha256"), fields))
    if header["magic"] != ROWS_MAGIC or header["version"] != VERSION or header["header_bytes"] != HEADER.size or header["top_k"] != TOP_K:
        raise DumpError(f"{path}: bad header")
    offsets = []
    offset = HEADER.size
    while True:
        if offset + 4 > len(data):
            raise DumpError(f"{path}: missing end record (incomplete dump)")
        kind = struct.unpack_from("<I", data, offset)[0]
        if kind == RECORD_END:
            break
        if kind != RECORD_ROW or offset + ROW_DTYPE.itemsize > len(data):
            raise DumpError(f"{path}: bad record at byte {offset}")
        probes = struct.unpack_from("<I", data, offset + 40)[0]
        offsets.append(offset)
        offset += ROW_DTYPE.itemsize + PROBE_DTYPE.itemsize * probes
    if offset + END.size != len(data):
        raise DumpError(f"{path}: trailing bytes after end record")
    end = dict(zip(("record_kind", "reserved", "row_count", "wave_count", "keyless_row_count",
                    "probed_row_count", "tier2_row_count", "nonfinite_row_count", "skipped_wave_count"),
                   END.unpack_from(data, offset)))
    if end["row_count"] != len(offsets):
        raise DumpError(f"{path}: end record counts {end['row_count']} rows, file has {len(offsets)}")
    rows = np.empty(len(offsets), dtype=ROW_DTYPE)
    probes = []
    for index, start in enumerate(offsets):
        rows[index] = np.frombuffer(data, dtype=ROW_DTYPE, count=1, offset=start)[0]
        count = int(rows[index]["probe_count"])
        probes.append(np.frombuffer(data, dtype=PROBE_DTYPE, count=count, offset=start + ROW_DTYPE.itemsize).copy())
    return {"path": str(path), "header": header, "rows": rows, "probes": probes, "end": end,
            "sha256": hashlib.sha256(data).digest()}


def write_rank_file(path, header, rows, probes, skipped_waves=0):
    with open(path, "wb") as handle:
        handle.write(HEADER.pack(ROWS_MAGIC, VERSION, HEADER.size, header["tp_rank"], header["tp_degree"],
                                 header["shard_begin"], header["shard_end"], header["vocabulary"],
                                 header["hidden_dimension"], TOP_K, header.get("tier2", 0),
                                 header.get("arm_digest", bytes(32)), header.get("probe_sha256", bytes(32)),
                                 header.get("tier2_sha256", bytes(32))))
        for index in range(len(rows)):
            row = rows[index:index + 1].copy()
            row["record_kind"] = RECORD_ROW
            row["probe_count"] = probes[index].size
            handle.write(row.tobytes())
            handle.write(probes[index].astype(PROBE_DTYPE).tobytes())
        flags = rows["flags"]
        handle.write(END.pack(RECORD_END, 0, len(rows), len(set(rows["wave_ordinal"].tolist())),
                              int(((flags & ROW_KEY_VALID) == 0).sum()), int(((flags & ROW_PROBED) != 0).sum()),
                              int(((flags & ROW_TIER2) != 0).sum()), int(((flags & ROW_NONFINITE) != 0).sum()),
                              skipped_waves))


IDENTITY_FIELDS = ("key", "wave_ordinal", "position", "row_in_wave", "input_token", "served_token")


def check_ranks(dumps):
    dumps = sorted(dumps, key=lambda item: item["header"]["tp_rank"])
    degree = dumps[0]["header"]["tp_degree"]
    if [d["header"]["tp_rank"] for d in dumps] != list(range(degree)) or len(dumps) != degree:
        raise DumpError("rank files must be exactly ranks 0..tp_degree-1 once each")
    shared = ("tp_degree", "vocabulary", "hidden_dimension", "tier2", "arm_digest", "probe_sha256", "tier2_sha256")
    cursor = 0
    for dump in dumps:
        header = dump["header"]
        for name in shared:
            if header[name] != dumps[0]["header"][name]:
                raise DumpError(f"{dump['path']}: header {name} differs from rank 0")
        if header["shard_begin"] != cursor or header["shard_end"] <= cursor:
            raise DumpError(f"{dump['path']}: shard [{header['shard_begin']},{header['shard_end']}) is not contiguous")
        cursor = header["shard_end"]
        if len(dump["rows"]) != len(dumps[0]["rows"]):
            raise DumpError(f"{dump['path']}: row count differs from rank 0")
        for name in IDENTITY_FIELDS:
            if not np.array_equal(dump["rows"][name], dumps[0]["rows"][name]):
                raise DumpError(f"{dump['path']}: row {name} differs from rank 0")
        mask = ROW_KEY_VALID | ROW_PROBED | ROW_TIER2
        if not np.array_equal(dump["rows"]["flags"] & mask, dumps[0]["rows"]["flags"] & mask):
            raise DumpError(f"{dump['path']}: row flags differ from rank 0")
    if cursor != dumps[0]["header"]["vocabulary"]:
        raise DumpError("shards do not cover the vocabulary")
    return dumps


def merge(dumps, probe_table=None):
    dumps = check_ranks(dumps)
    rows = len(dumps[0]["rows"])
    base = dumps[0]["rows"]
    maxima = np.stack([d["rows"]["local_max"].astype(np.float64) for d in dumps])
    sums = np.stack([d["rows"]["local_sum_exp"] for d in dumps])
    nonfinite = np.zeros(rows, dtype=bool)
    for dump in dumps:
        nonfinite |= (dump["rows"]["flags"] & ROW_NONFINITE) != 0
    peak = maxima.max(axis=0)
    total = np.zeros(rows, dtype=np.float64)
    for rank in range(len(dumps)):
        total = total + sums[rank] * np.exp(maxima[rank] - peak)
    with np.errstate(divide="ignore"):
        log_z = peak + np.log(total)
    ids = np.concatenate([d["rows"]["top_ids"] for d in dumps], axis=1)
    logits = np.concatenate([d["rows"]["top_logits"] for d in dumps], axis=1)
    order = np.lexsort((ids, -logits.astype(np.float64)), axis=1)[:, :TOP_K]
    top_ids = np.take_along_axis(ids, order, axis=1)
    top_logits = np.take_along_axis(logits, order, axis=1)
    merged = np.zeros(rows, dtype=MERGED_DTYPE)
    merged["key"] = base["key"]
    merged["position"] = base["position"]
    merged["input_token"] = base["input_token"]
    merged["served_token"] = base["served_token"]
    merged["flags"] = (base["flags"] & (ROW_KEY_VALID | ROW_PROBED | ROW_TIER2)) | np.where(nonfinite, ROW_NONFINITE, 0).astype(np.uint32)
    merged["log_z"] = np.where(nonfinite, np.nan, log_z)
    merged["top_ids"] = np.where(nonfinite[:, None], NO_TOKEN, top_ids)
    merged["top_logits"] = np.where(nonfinite[:, None], 0.0, top_logits)
    probes = []
    for index in range(rows):
        parts = [d["probes"][index] for d in dumps]
        for dump, part in zip(dumps, parts):
            header = dump["header"]
            if part.size and ((part["id"] < header["shard_begin"]).any() or (part["id"] >= header["shard_end"]).any()):
                raise DumpError(f"{dump['path']}: row {index} probe id outside the rank shard")
        joined = np.concatenate(parts) if parts else np.zeros(0, dtype=PROBE_DTYPE)
        joined = joined[np.argsort(joined["id"], kind="stable")]
        if joined.size and (np.diff(joined["id"].astype(np.int64)) == 0).any():
            raise DumpError(f"row {index}: duplicate probe id across ranks")
        if probe_table is not None and base["flags"][index] & ROW_PROBED:
            expected = sorted(probe_table.get((int(base["key"][index]), int(base["position"][index])), []))
            if joined["id"].tolist() != expected:
                raise DumpError(f"row {index}: probe ids do not match the probe file entry")
        probes.append(joined)
    merged["probe_count"] = [p.size for p in probes]
    return dumps, merged, probes


def write_merged(path, dumps, merged, probes):
    header = dumps[0]["header"]
    rank_digest = hashlib.sha256(b"".join(d["sha256"] for d in dumps)).digest()
    with open(path, "wb") as handle:
        handle.write(MERGED_HEADER.pack(MERGED_MAGIC, VERSION, header["tp_degree"], header["vocabulary"], TOP_K,
                                        len(merged), header["arm_digest"], header["probe_sha256"],
                                        header["tier2_sha256"], rank_digest))
        for index in range(len(merged)):
            handle.write(merged[index:index + 1].tobytes())
            handle.write(probes[index].tobytes())


def read_merged(path):
    data = Path(path).read_bytes()
    fields = MERGED_HEADER.unpack_from(data, 0)
    if fields[0] != MERGED_MAGIC or fields[1] != VERSION or fields[4] != TOP_K:
        raise DumpError(f"{path}: not a merged dump")
    count = fields[5]
    merged = np.empty(count, dtype=MERGED_DTYPE)
    probes = []
    offset = MERGED_HEADER.size
    for index in range(count):
        merged[index] = np.frombuffer(data, dtype=MERGED_DTYPE, count=1, offset=offset)[0]
        offset += MERGED_DTYPE.itemsize
        used = int(merged[index]["probe_count"])
        probes.append(np.frombuffer(data, dtype=PROBE_DTYPE, count=used, offset=offset).copy())
        offset += PROBE_DTYPE.itemsize * used
    if offset != len(data):
        raise DumpError(f"{path}: trailing bytes")
    header = dict(zip(("magic", "version", "tp_degree", "vocabulary", "top_k", "row_count", "arm_digest",
                       "probe_sha256", "tier2_sha256", "rank_digest"), fields))
    return header, merged, probes


def row_index(merged):
    table = {}
    for index in range(len(merged)):
        if merged["flags"][index] & ROW_KEY_VALID and not merged["flags"][index] & ROW_NONFINITE:
            table.setdefault((int(merged["key"][index]), int(merged["position"][index])), index)
    return table


def pass2_entries(merged, probe_table):
    entries = {}
    for index in range(len(merged)):
        flags = int(merged["flags"][index])
        if not flags & ROW_KEY_VALID or flags & ROW_NONFINITE:
            continue
        key = (int(merged["key"][index]), int(merged["position"][index]))
        ids = entries.setdefault(key, set(probe_table.get(key, [])) if probe_table else set())
        ids.update(int(v) for v in merged["top_ids"][index])
    return entries


def served_check(merged):
    agree = mismatch = near_tie = unserved = 0
    for index in range(len(merged)):
        served = int(merged["served_token"][index])
        if served == NO_TOKEN or merged["flags"][index] & ROW_NONFINITE:
            unserved += 1
            continue
        if int(merged["top_ids"][index][0]) == served:
            agree += 1
        elif float(merged["top_logits"][index][0]) - float(merged["top_logits"][index][1]) < NEAR_TIE_NATS:
            near_tie += 1
        else:
            mismatch += 1
    return {"served_agree": agree, "served_near_tie": near_tie, "served_mismatch": mismatch, "unserved": unserved}


def command_targets(arguments):
    entries = targets_table(read_corpus(arguments.corpus))
    width = max((len(v) for v in entries.values()), default=0)
    write_table(arguments.out, PROBE_MAGIC, entries, max(width, 1))
    print(json.dumps({"entries": len(entries), "id_width": max(width, 1)}))


def command_tier2(arguments):
    entries = tier2_table(read_corpus(arguments.corpus), arguments.count, arguments.seed)
    write_table(arguments.out, TIER2_MAGIC, entries, 0)
    print(json.dumps({"entries": len(entries)}))


def command_merge(arguments):
    probe_table = None
    probe_sha = None
    if arguments.probe:
        probe_table, probe_sha, _ = read_table(arguments.probe, PROBE_MAGIC)
    dumps = [read_rank_file(path) for path in arguments.ranks]
    dumps, merged, probes = merge(dumps, probe_table)
    if probe_sha is not None and dumps[0]["header"]["probe_sha256"] != probe_sha:
        raise DumpError("dump was written against a different probe file")
    write_merged(arguments.out, dumps, merged, probes)
    summary = {"rows": len(merged), "merged_sha256": hashlib.sha256(Path(arguments.out).read_bytes()).hexdigest(),
               "keyless_rows": int(dumps[0]["end"]["keyless_row_count"]),
               "skipped_waves": int(dumps[0]["end"]["skipped_wave_count"]),
               "nonfinite_rows": int(((merged["flags"] & ROW_NONFINITE) != 0).sum())}
    summary.update(served_check(merged))
    if arguments.probe_out:
        entries = pass2_entries(merged, probe_table)
        width = max((len(v) for v in entries.values()), default=1)
        write_table(arguments.probe_out, PROBE_MAGIC, entries, width)
        summary["probe_out_entries"] = len(entries)
        summary["probe_out_sha256"] = hashlib.sha256(Path(arguments.probe_out).read_bytes()).hexdigest()
    print(json.dumps(summary, sort_keys=True))
    if arguments.require_served_match and summary["served_mismatch"] != 0:
        raise DumpError(f"{summary['served_mismatch']} rows: merged argmax differs from the served token beyond a near tie")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    targets = sub.add_parser("targets")
    targets.add_argument("corpus")
    targets.add_argument("--out", required=True)
    tier2 = sub.add_parser("tier2")
    tier2.add_argument("corpus")
    tier2.add_argument("--count", type=int, required=True)
    tier2.add_argument("--seed", type=int, required=True)
    tier2.add_argument("--out", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("ranks", nargs="+")
    merge_parser.add_argument("--out", required=True)
    merge_parser.add_argument("--probe")
    merge_parser.add_argument("--probe-out")
    merge_parser.add_argument("--require-served-match", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        {"targets": command_targets, "tier2": command_tier2, "merge": command_merge}[arguments.command](arguments)
    except DumpError as error:
        print(f"score_merge: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
