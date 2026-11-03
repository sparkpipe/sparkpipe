#!/usr/bin/env python3
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import spark_named_pack  # noqa: E402
from spark_named_pack import PayloadWriter, source_identity  # noqa: E402

TENSORS = (("a", bytes(range(200)), (200,)), ("b", bytes(range(255, 0, -1)) * 3, (3, 255)), ("c", b"\x07" * 129, (129,)))


def write(writer, names):
    for name, data, shape in TENSORS:
        if name not in names:
            continue
        if name == "b":
            writer.add_parts(name, [data[:300], data[300:]], "u8", shape)
        else:
            writer.add(name, data, "u8", shape)


def interrupted(path, identity):
    writer = PayloadWriter(path, identity)
    write(writer, ("a", "b"))
    writer.handle.write(b"\xee" * 77)
    writer.handle.flush()
    writer.handle.close()
    writer.journal.close()


def refused(path, identity):
    sizes = (os.path.getsize(path), os.path.getsize(str(path) + ".journal"))
    try:
        PayloadWriter(path, identity, resume=True)
    except ValueError:
        return sizes == (os.path.getsize(path), os.path.getsize(str(path) + ".journal"))
    return False


def main():
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        source, other = scratch / "source", scratch / "other"
        for directory, layers in ((source, 2), (other, 3)):
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
        contract = scratch / "contract.json"
        contract.write_text("{}")
        identity = source_identity(source, "rank0", files=(contract,))

        reference = scratch / "reference.payload"
        writer = PayloadWriter(reference, identity)
        write(writer, ("a", "b", "c"))
        reference_manifest = dict(writer.manifest)
        writer.close()

        resumed = scratch / "resumed.payload"
        interrupted(resumed, identity)
        writer = PayloadWriter(resumed, identity, resume=True)
        if sorted(writer.manifest) != ["a", "b"]:
            failures.append(f"a resume must hold exactly the journaled tensors, held {sorted(writer.manifest)}")
        write(writer, ("a", "b", "c"))
        if writer.manifest != reference_manifest:
            failures.append("a resumed pack's manifest must equal a straight-through pack's")
        writer.close()
        if resumed.read_bytes() != reference.read_bytes():
            failures.append("a resumed payload must equal a straight-through payload byte for byte (the unjournaled tail is cut)")
        if os.path.exists(str(resumed) + ".journal"):
            failures.append("a closed writer leaves no journal")

        changed = [source_identity(other, "rank0", files=(contract,)), source_identity(source, "rank1", files=(contract,))]
        contract.write_text('{"layout":2}')
        changed.append(source_identity(source, "rank0", files=(contract,)))
        contract.write_text("{}")
        if source_identity(source, "rank0", files=(contract,)) != identity:
            failures.append("the same source, parts and files must give the same identity")
        for index, foreign in enumerate(changed):
            if foreign == identity:
                failures.append(f"identity {index}: another config, rank or contract must give another identity")
                continue
            path = scratch / f"foreign{index}.payload"
            interrupted(path, identity)
            if not refused(path, foreign):
                failures.append(f"identity {index}: a resume from another source must be refused and must not touch the payload or journal")

        short = scratch / "short.payload"
        interrupted(short, identity)
        end = json.loads(Path(str(short) + ".journal").read_text().splitlines()[-1])["end"]
        os.truncate(short, end - 1)
        if not refused(short, identity):
            failures.append("a payload shorter than its journal's last end must be refused, never zero-extended")

        headless = scratch / "headless.payload"
        interrupted(headless, identity)
        journal = Path(str(headless) + ".journal")
        journal.write_text("".join(journal.read_text().splitlines(keepends=True)[1:]))
        if not refused(headless, identity):
            failures.append("a journal without a source identity must be refused")

        try:
            PayloadWriter(scratch / "anonymous.payload", "")
            failures.append("a writer without a source identity must be refused")
        except ValueError:
            pass

        events = []
        original = spark_named_pack.os.fsync
        durable = scratch / "durable.payload"
        writer = PayloadWriter(durable, identity)

        def record(fd):
            events.append((fd, os.path.getsize(durable), len(Path(writer.journal_path).read_text().splitlines())))
            original(fd)

        payload_fd, journal_fd = writer.handle.fileno(), writer.journal.fileno()
        spark_named_pack.os.fsync = record
        try:
            write(writer, ("a", "b"))
        finally:
            spark_named_pack.os.fsync = original
        end = writer.offset
        writer.close()
        if events != [(payload_fd, 200, 1), (journal_fd, 200, 2), (payload_fd, end, 2), (journal_fd, end, 3)]:
            failures.append(f"each tensor's bytes must be fsynced before its journal record is written, and the record fsynced after: {events}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS pack resume journal: a resumed payload equals a straight-through one; a resume from another config, rank or contract, "
          "onto a payload shorter than its journal, or from a journal without an identity is refused untouched; bytes are fsynced before their record")


if __name__ == "__main__":
    main()
