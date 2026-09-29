#!/usr/bin/env python3
"""The Python tap-dump reader parses what the C writer produces, and refuses damaged dumps."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import spec_tap_dump

WRITER = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_speculation_tap.h"
int main(int argc,char **argv)
{
	SparkSpeculationTapSet set;
	SparkSpeculationTapDump dump;
	SparkSpeculationTapRecord record;
	uint8_t payload[2u * 8u * 2u];
	uint32_t index,byte;
	if ( argc != 3 || SparkSpeculationTapSetParse("mean:3,7",9u,4u,8u,&set) != SPARK_STATUS_OK || set.record_bytes != sizeof(payload) )
		return(2);
	if ( SparkSpeculationTapDumpOpen(&dump,argv[1],&set,"model-x",77u,15u,strtoull(argv[2],0,10)) != SPARK_STATUS_OK )
		return(3);
	for (index=0u; index<5u; index++)
	{
		memset(&record,0,sizeof(record));
		record.sequence_id = index < 3u ? 11u : 12u;
		record.position = index < 3u ? 40u + index : index == 3u ? 10u : 12u;
		record.token_id = 1000u + index;
		record.next_token_id = 2000u + index;
		record.flags = index == 0u ? SPARK_SPECULATION_TAP_FLAG_PREFILL : index == 4u ? SPARK_SPECULATION_TAP_FLAG_DECODE : SPARK_SPECULATION_TAP_FLAG_VERIFY;
		record.serial = index + 1u;
		for (byte=0u; byte<sizeof(payload); byte++)
			payload[byte] = (uint8_t)(index * 16u + byte);
		(void)SparkSpeculationTapDumpAppend(&dump,&record,payload);
	}
	return(SparkSpeculationTapDumpClose(&dump) == SPARK_STATUS_OK ? 0 : 4);
}
'''


def build(directory):
    source, binary = Path(directory) / "writer.c", Path(directory) / "writer"
    source.write_text(WRITER)
    subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O1", "-Iinclude", str(source), "src/spark_speculation_tap.c",
                    "src/spark_status.c", "-o", str(binary)], cwd=ROOT, check=True)
    return binary


def sptd_bytes(records, flags=2):
    header = bytearray(128)
    struct.pack_into("<4I", header, 0, 0x44545053, 1, 128, 40)
    struct.pack_into("<8I", header, 16, 1, 2, 4, 4, 9, 4, 8, 1)
    struct.pack_into("<8I", header, 48, 3, 8, 0, 0, 0, 0, 0, 0)
    struct.pack_into("<2Q2IQ", header, 80, 0xabc, 9, 15, flags, len(records))
    header[112:115] = b"glm"
    body = b"".join(struct.pack("<QQIIIIQ", *fields) + rows for fields, rows in records)
    return bytes(header) + body


def check_export():
    def row(position, tap):
        return bytes((position * 8 + tap * 64 + byte) % 256 for byte in range(8))
    records = []
    for position in range(6):
        flags = 1 if position < 4 else 2
        records.append(((5, position, 100 + position, 101 + position, flags, 0, position + 1), row(position, 0) + row(position, 1)))
    records.append(((6, 3, 7, 8, 2, 0, 7), row(3, 0) + row(3, 1)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "taps.sptd"
        path.write_bytes(sptd_bytes(records))
        classes = Path(directory) / "classes.json"
        classes.write_text(json.dumps({"5": "code"}))
        out = Path(directory) / "offline"
        assert spec_tap_dump.main(["export-offline", str(path), str(out), "--layer", "8", "--classes", str(classes), "--model", "glmflash", "--firmware", "f00"]) == 0
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["format"] == "spark-tapdump-1" and manifest["hidden_dimension"] == 4 and manifest["taps"][0]["name"] == "L8.mean", manifest
        assert manifest["streams"] == [{"name": "seq5", "class": "code", "tokens": "seq5.u32", "prompt_tokens": 4, "total_tokens": 7, "taps": "seq5.L8.mean.bf16"}], manifest["streams"]
        assert "6" in manifest["notes"]["skipped"], manifest["notes"]
        data = (out / "seq5.u32").read_bytes()
        assert struct.unpack("<II7I", data) == (4, 7, 100, 101, 102, 103, 104, 105, 106), data
        rows = (out / "seq5.L8.mean.bf16").read_bytes()
        assert rows == b"".join(row(position, 1) for position in range(6)) + bytes(8), rows
        for line in (out / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, name
        assert spec_tap_dump.main(["export-offline", str(path), str(Path(directory) / "none"), "--layer", "4", "--model", "glmflash", "--firmware", "f00"]) == 1


def main():
    with tempfile.TemporaryDirectory() as directory:
        binary = build(directory)
        path = Path(directory) / "taps.sptd"
        subprocess.run([str(binary), str(path), "1000000"], check=True, stderr=subprocess.DEVNULL)
        report = spec_tap_dump.summarize(str(path))
        header = report["header"]
        assert header["reduction"] == "mean" and header["tap_count"] == 2 and header["layers"] == [3, 7], header
        assert header["row_elements"] == 8 and header["row_bytes"] == 16 and header["record_bytes"] == 32, header
        assert header["tp_rank"] == 15 and header["engine_generation"] == 77 and header["model_tag"] == "model-x", header
        assert header["closed"] and not header["truncated"] and header["records"] == 5 and report["records"] == 5, report
        assert report["flags"] == {"prefill": 1, "verify": 3, "decode": 1}, report["flags"]
        assert report["sequences"]["11"] == {"records": 3, "first": 40, "next": 43, "gaps": 0, "repeats": 0}, report["sequences"]
        assert report["problems"] == [], report["problems"]
        records = [record for _, record in spec_tap_dump.iter_records(str(path))]
        assert [record["position"] for record in records] == [40, 41, 42, 10, 12]
        assert report["sequences"]["12"] == {"records": 2, "first": 10, "next": 13, "gaps": 1, "repeats": 0}, report["sequences"]
        assert [record["token_id"] for record in records] == [1000, 1001, 1002, 1003, 1004]
        assert [record["next_token_id"] for record in records] == [2000, 2001, 2002, 2003, 2004]
        rows = spec_tap_dump.tap_rows(header, records[1]["rows"])
        assert rows[0] == bytes(16 + byte for byte in range(16)) and rows[1] == bytes(32 + byte for byte in range(16)), rows
        assert spec_tap_dump.main(["verify", str(path)]) == 0

        small = Path(directory) / "small.sptd"
        subprocess.run([str(binary), str(small), str(128 + 2 * 72)], check=True, stderr=subprocess.DEVNULL)
        report = spec_tap_dump.summarize(str(small))
        assert report["header"]["truncated"] and report["records"] == 2, report

        subprocess.run([str(binary), str(path), "1000000"], check=False, stderr=subprocess.DEVNULL)
        assert spec_tap_dump.summarize(str(path))["records"] == 5, "the writer never overwrites an existing dump"

        cut = Path(directory) / "cut.sptd"
        cut.write_bytes(path.read_bytes()[:-5])
        try:
            spec_tap_dump.summarize(str(cut))
            raise AssertionError("a trailing partial record must be refused")
        except spec_tap_dump.TapDumpError:
            pass
        assert spec_tap_dump.main(["verify", str(cut)]) == 1

        raw = bytearray(path.read_bytes())
        raw[100] = 0
        open_dump = Path(directory) / "open.sptd"
        open_dump.write_bytes(bytes(raw))
        assert spec_tap_dump.summarize(str(open_dump))["problems"] == ["dump was not closed (engine did not shut down cleanly)"]
        assert spec_tap_dump.main(["verify", str(open_dump)]) == 1

        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        bad = Path(directory) / "bad.sptd"
        bad.write_bytes(bytes(raw))
        assert spec_tap_dump.main(["verify", str(bad)]) == 1
    check_export()
    print("PASS tap dump: the reader parses the C writer's header, records and per-tap rows, reports truncation and position gaps, and refuses partial, unclosed or foreign dumps")


if __name__ == "__main__":
    main()
