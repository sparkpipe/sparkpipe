#!/usr/bin/env python3
"""Export teacher-forced score dumps into the A/B analysis formats.

tools/score_merge.py keys rows by (prefix key, position) and keeps them in the
order the engine scored them. tools/ab_dump.py and tools/ab_score_compare.py
read one row per corpus (document, predicted position) in corpus order. This
tool is the bridge between the two.

corpus  CORPUS.index.json --out CORPUS.jsonl
        sparkpipe-ab-corpus-v1 (index + tokens.u32, both SHA-checked) to the JSON
        lines {"doc", "tokens"} that score_merge.py targets|tier2 read.
dump    MERGED --corpus CORPUS.index.json --arm ARM.json [--reference REF.npz] --out OUT.npz
        score_merge.py merged dump to sparkpipe-score-merged-v1. Rows run
        document by document in corpus order, positions 1..len-1; row (doc, pos)
        scores corpus token pos from the prefix tokens[0..pos-1]. Each document's
        rows come from the dump segment that scored that document (a run of
        positions 0, 1, 2, ... in dump order), so documents that share a prefix
        keep their own rows and a run submitted in another document order exports
        the same bytes. Segments that belong to no corpus document (for example
        generation requests) are counted and skipped; a corpus document that is
        missing, scored twice or not scored contiguously is refused.
        Without --reference the run defines the probes: probe_logits equals
        top_logits and probe_sha256 is null. With --reference the run is an arm:
        probe_logits are its logits at the reference's top ids, read from its
        pass-2 probes, and probe_sha256 names the reference's top ids.
        The header arm_digest is the digest of ARM.json. The rank files' own
        arm_digest must be that digest or all zero (firmware that does not carry
        the member yet); it is recorded as dump_arm_digest.
exact   COMBINED.json --corpus CORPUS.index.json --reference REF.npz --arm ARM.npz --out EXACT.npz
        score_kl_partial.py combine output to sparkpipe-score-exact-kl-v1, whose
        rows index the exported dumps; a Tier-2 row shared by documents with a
        common prefix is attributed to its first corpus row.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_arm
import ab_corpus_build
import ab_dump
import score_merge

MASS_TOLERANCE = 1.0e-6


class ExportError(Exception):
    pass


def load_corpus(index_path):
    try:
        index, documents = ab_corpus_build.load_documents(Path(index_path))
    except ab_corpus_build.CorpusError as error:
        raise ExportError(str(error)) from error
    if index.get("format") != ab_corpus_build.FORMAT:
        raise ExportError(f"{index_path}: format {index.get('format')!r} is not {ab_corpus_build.FORMAT}")
    for entry, tokens in zip(index["documents"], documents):
        if len(tokens) < 2:
            raise ExportError(f"{entry['id']}: a document needs at least two tokens to score one position")
    return index, [[int(t) for t in tokens] for tokens in documents]


def document_segments(merged):
    spans = []
    start = None
    for row in range(len(merged)):
        position = int(merged["position"][row])
        if position == 0:
            if start is not None:
                spans.append((start, row))
            start = row
        elif start is None or position != int(merged["position"][row - 1]) + 1:
            raise ExportError(f"dump row {row}: position {position} does not continue a document segment "
                              f"(documents must be scored one request at a time: SPARK_MODEL_BATCH_SEQUENTIAL=1)")
    if start is not None:
        spans.append((start, len(merged)))
    return spans


def assign_documents(merged, documents):
    chains = [score_merge.row_keys(tokens) for tokens in documents]
    terminal = {}
    for doc, keys in enumerate(chains):
        identity = (keys[-2], len(keys) - 2)
        if identity in terminal:
            raise ExportError(f"documents {terminal[identity]} and {doc} share every scored prefix")
        terminal[identity] = doc
    owner = [None] * len(documents)
    foreign = 0
    for start, end in document_segments(merged):
        found = None
        for row in range(end - 1, start - 1, -1):
            found = terminal.get((int(merged["key"][row]), int(merged["position"][row])))
            if found is not None:
                break
        if found is None:
            foreign += 1
            continue
        need = len(chains[found]) - 1
        if end - start < need:
            raise ExportError(f"document {found}: its segment at dump row {start} has {end - start} rows, {need} needed")
        keys = merged["key"][start:start + need]
        if not np.array_equal(keys, np.array(chains[found][:need], dtype=np.uint64)):
            raise ExportError(f"document {found}: the segment at dump row {start} does not follow the document's prefix keys")
        flags = merged["flags"][start:start + need]
        if np.any((flags & score_merge.ROW_KEY_VALID) == 0):
            raise ExportError(f"document {found}: keyless rows (prefix reuse was on or the history is unknown)")
        if np.any((flags & score_merge.ROW_NONFINITE) != 0):
            raise ExportError(f"document {found}: {int(np.sum((flags & score_merge.ROW_NONFINITE) != 0))} non-finite rows")
        if owner[found] is not None:
            raise ExportError(f"document {found} was scored twice (dump rows {owner[found]} and {start})")
        owner[found] = start
    missing = [doc for doc, start in enumerate(owner) if start is None]
    if missing:
        raise ExportError(f"{len(missing)} corpus documents have no dump segment, first {missing[:4]}")
    return owner, foreign


def row_logits(merged, probes, row):
    ids = np.concatenate([merged["top_ids"][row].astype(np.int64), probes[row]["id"].astype(np.int64)])
    values = np.concatenate([merged["top_logits"][row], probes[row]["logit"]]).astype(np.float32)
    order = np.argsort(ids, kind="stable")
    ids, values = ids[order], values[order]
    same = ids[1:] == ids[:-1]
    if np.any(same & (values[1:] != values[:-1])):
        raise ExportError(f"dump row {row}: probe and top-k logits disagree for one id")
    keep = np.concatenate([[True], ~same])
    return ids[keep], values[keep]


def lookup(ids, values, wanted, row, what):
    wanted = np.atleast_1d(np.asarray(wanted, dtype=np.int64))
    at = np.minimum(np.searchsorted(ids, wanted), ids.size - 1)
    found = ids[at] == wanted
    if not np.all(found):
        raise ExportError(f"dump row {row}: no logit for {what} {int(wanted[~found][0])}")
    return values[at]


def fresh(out):
    if Path(out).exists():
        raise ExportError(f"{out} exists; exports are written once")


def export_dump(merged_path, index_path, arm_path, reference_path, out):
    fresh(out)
    index, documents = load_corpus(index_path)
    try:
        digest = ab_arm.arm_digest(ab_arm.load(arm_path))
    except ab_arm.ArmError as error:
        raise ExportError(f"{arm_path}: {error}") from error
    header, merged, probes = score_merge.read_merged(merged_path)
    dump_digest = header["arm_digest"].hex()
    if dump_digest not in (digest, "0" * 64):
        raise ExportError(f"rank files carry arm_digest {dump_digest}, not the digest of {arm_path} ({digest})")
    owner, foreign = assign_documents(merged, documents)
    lengths = [len(tokens) - 1 for tokens in documents]
    source = np.concatenate([np.arange(owner[d], owner[d] + lengths[d]) for d in range(len(documents))])
    doc = np.repeat(np.arange(len(documents), dtype=np.uint32), lengths)
    pos = np.concatenate([np.arange(1, n + 1, dtype=np.uint32) for n in lengths])
    target = np.concatenate([np.asarray(tokens[1:], dtype=np.uint32) for tokens in documents])
    reference = None
    if reference_path is not None:
        reference = ab_dump.read(reference_path)
        if reference["header"].get("probe_sha256") is not None:
            raise ExportError(f"{reference_path} is an arm export, not the run that defines the probes")
        for key in ("corpus_sha256", "tokenizer_sha256"):
            if reference["header"].get(key) != index[key]:
                raise ExportError(f"{reference_path}: {key} is not the corpus index's")
        for key, values in (("doc", doc), ("pos", pos), ("target", target)):
            if not np.array_equal(reference[key], values):
                raise ExportError(f"{reference_path}: rows differ in {key}")
    target_logit = np.empty(len(source), dtype=np.float32)
    probe_logits = np.empty((len(source), score_merge.TOP_K), dtype=np.float32)
    for out_row, row in enumerate(source.tolist()):
        ids, values = row_logits(merged, probes, row)
        target_logit[out_row] = lookup(ids, values, int(target[out_row]), row, "corpus target")[0]
        if reference is not None:
            probe_logits[out_row] = lookup(ids, values, reference["top_ids"][out_row].astype(np.int64), row, "reference top id")
    arrays = {"doc": doc, "pos": pos, "target": target, "lse": merged["log_z"][source].astype(np.float64),
              "target_logit": target_logit, "top_ids": merged["top_ids"][source], "top_logits": merged["top_logits"][source],
              "probe_logits": merged["top_logits"][source] if reference is None else probe_logits}
    out_header = {"arm_digest": digest, "corpus_sha256": index["corpus_sha256"],
                  "tokenizer_sha256": index["tokenizer_sha256"], "dump_arm_digest": dump_digest,
                  "probe_sha256": None if reference is None else ab_dump.probe_sha256(reference["top_ids"])}
    sha = ab_dump.write(out, out_header, arrays)
    return {"rows": int(len(source)), "documents": len(documents), "foreign_segments": foreign,
            "dump_rows": int(len(merged)), "sha256": sha, "arm_digest": digest, "dump_arm_digest": dump_digest,
            "role": "reference" if reference is None else "arm"}


def export_exact(combined_path, index_path, reference_path, arm_path, out):
    fresh(out)
    index, documents = load_corpus(index_path)
    reference = ab_dump.read(reference_path)
    arm = ab_dump.read(arm_path)
    for name, dump in (("reference", reference), ("arm", arm)):
        if dump["header"].get("corpus_sha256") != index["corpus_sha256"]:
            raise ExportError(f"{name} export is not for this corpus")
    for key in ("doc", "pos", "target"):
        if not np.array_equal(reference[key], arm[key]):
            raise ExportError(f"reference and arm exports differ in {key}")
    first = {}
    row = 0
    for tokens in documents:
        keys = score_merge.row_keys(tokens)
        for position in range(len(tokens) - 1):
            first.setdefault((keys[position], position), row)
            row += 1
    if row != len(reference["doc"]):
        raise ExportError(f"reference export has {len(reference['doc'])} rows, the corpus scores {row}")
    report = json.loads(Path(combined_path).read_text())
    error = report.get("max_mass_error")
    if error is None or error > MASS_TOLERANCE:
        raise ExportError(f"combined exact KL has reference mass error {error}; refusing")
    rows = []
    values = []
    for entry in report["rows"]:
        identity = (int(entry["key"]), int(entry["position"]))
        if identity not in first:
            raise ExportError(f"Tier-2 row key {identity[0]:#x} position {identity[1]} is not a corpus row")
        rows.append(first[identity])
        values.append(float(entry["kl"]))
    order = np.argsort(np.asarray(rows, dtype=np.int64), kind="stable")
    rows = np.asarray(rows, dtype=np.int64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    if rows.size and np.any(rows[1:] == rows[:-1]):
        raise ExportError("combined exact KL lists one corpus row twice")
    sha = ab_dump.write_exact(out, {"reference_arm_digest": reference["header"]["arm_digest"],
                                    "arm_digest": arm["header"]["arm_digest"],
                                    "corpus_sha256": index["corpus_sha256"]}, rows, values)
    return {"rows": int(rows.size), "kl_mean": float(values.mean()) if values.size else None, "sha256": sha}


def export_corpus(index_path, out):
    index, documents = load_corpus(index_path)
    with open(out, "x", encoding="utf-8") as handle:
        for entry, tokens in zip(index["documents"], documents):
            handle.write(json.dumps({"doc": entry["id"], "tokens": tokens}) + "\n")
    return {"documents": len(documents), "scored_rows": sum(len(t) - 1 for t in documents),
            "corpus_sha256": index["corpus_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("index")
    corpus.add_argument("--out", required=True)
    dump = sub.add_parser("dump")
    dump.add_argument("merged")
    dump.add_argument("--corpus", required=True)
    dump.add_argument("--arm", required=True)
    dump.add_argument("--reference")
    dump.add_argument("--out", required=True)
    exact = sub.add_parser("exact")
    exact.add_argument("combined")
    exact.add_argument("--corpus", required=True)
    exact.add_argument("--reference", required=True)
    exact.add_argument("--arm", required=True)
    exact.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "corpus":
            summary = export_corpus(arguments.index, arguments.out)
        elif arguments.command == "dump":
            summary = export_dump(arguments.merged, arguments.corpus, arguments.arm, arguments.reference, arguments.out)
        else:
            summary = export_exact(arguments.combined, arguments.corpus, arguments.reference, arguments.arm, arguments.out)
    except (ExportError, score_merge.DumpError, ab_dump.DumpError, FileExistsError) as error:
        print(f"score_export: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
