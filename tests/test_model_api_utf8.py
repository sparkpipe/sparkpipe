#!/usr/bin/env python3
from pathlib import Path
import json
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#define main SparkModelApiMain
#include "node/model_api.c"
#undef main

static void TestPrintHex(const char *bytes, size_t length)
{
	size_t index;
	for (index = 0u; index < length; index++)
		fprintf(stdout,"%02x",(unsigned)(unsigned char)bytes[index]);
	fputc('\n',stdout);
}

int main(void)
{
	static char line[1u << 16],raw[1u << 15],pending[1u << 15],escaped[1u << 18],part[1u << 18];
	while (fgets(line,sizeof(line),stdin) != 0)
	{
		char *cuts = strchr(line,'|'),*cursor;
		uint32_t length = 0u,pending_bytes = 0u,start = 0u,ready;
		size_t escaped_length = 0u,part_length;
		unsigned value;
		assert(cuts != 0);
		*cuts++ = '\0';
		for (cursor = line; cursor[0] != '\0' && cursor[1] != '\0'; cursor += 2)
		{
			assert(sscanf(cursor,"%2x",&value) == 1);
			raw[length++] = (char)value;
		}
		escaped[0] = '\0';
		if (*cuts == '\n' || *cuts == '\0')
		{
			assert(append_json_escaped(escaped,sizeof(escaped),&escaped_length,raw,length));
			TestPrintHex(escaped,escaped_length);
			continue;
		}
		for (cursor = cuts; ; )
		{
			uint32_t cut = (uint32_t)strtoul(cursor,&cursor,10),flush = *cursor != ',';
			memcpy(pending + pending_bytes,raw + start,cut - start);
			pending_bytes += cut - start;
			start = cut;
			ready = flush != 0u ? pending_bytes : api_utf8_complete(pending,pending_bytes);
			part_length = 0u;
			assert(append_json_escaped(part,sizeof(part),&part_length,pending,ready));
			memcpy(escaped + escaped_length,part,part_length);
			escaped_length += part_length;
			memmove(pending,pending + ready,pending_bytes - ready);
			pending_bytes -= ready;
			if (flush != 0u)
				break;
			cursor++;
		}
		assert(start == length && pending_bytes == 0u);
		TestPrintHex(escaped,escaped_length);
	}
	return(0);
}
'''

KOREAN = "서울은 대한민국의 수도이며, 한강이 도시를 가로지른다. 🇰🇷 Ünïcödé — “quotes” \u0007\t\n"


def cases(rng):
    pieces = [bytes([b]) for b in (0x00, 0x1f, 0x22, 0x5c, 0x7f, 0x80, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xed, 0xef, 0xf0, 0xf4, 0xf5, 0xff)]
    pieces += [b"\xe0\xa0\x80", b"\xed\x9f\xbf", b"\xed\xa0\x80", b"\xf0\x90\x80\x80", b"\xf4\x8f\xbf\xbf", b"\xf4\x90\x80\x80", b"\xe0\x80", b"\xf0\x80\x80\x80", b"A"]
    body = KOREAN.encode()
    for _ in range(4000):
        if rng.random() < 0.5:
            data = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 24)))
        else:
            a, b = sorted(rng.randint(0, len(body)) for _ in range(2))
            data = body[a:b] + rng.choice(pieces) + body[rng.randint(0, len(body)):][:rng.randint(0, 9)]
        yield data, sorted(set(rng.randint(0, len(data)) for _ in range(rng.randint(0, 6))) - {len(data)}) + [len(data)] if rng.random() < 0.5 else None


def main():
    rng = random.Random(1026)
    inputs = list(cases(rng))
    with tempfile.TemporaryDirectory(prefix="api-utf8-") as directory:
        source = Path(directory) / "probe.c"
        binary = Path(directory) / "probe"
        source.write_text(HARNESS)
        subprocess.run([
            "cc", "-std=c11", "-O1", "-D_GNU_SOURCE", "-pthread",
            "-I.", "-Iinclude", "-Isrc", "-Imodel-families/common/include",
            str(source), "build/libsparkpipe_runtime.a",
            "build/libsparkpipe_model_common.a", "build/libsparkpipe_core.a",
            "-ldl", "-lpthread", "-o", str(binary),
        ], cwd=ROOT, check=True)
        request = "".join(data.hex() + "|" + (",".join(map(str, cuts)) if cuts else "") + "\n" for data, cuts in inputs)
        result = subprocess.run([str(binary)], cwd=ROOT, input=request, timeout=60, capture_output=True, text=True)
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            result.check_returncode()
    lines = result.stdout.splitlines()
    assert len(lines) == len(inputs), (len(lines), len(inputs))
    streamed = 0
    for (data, cuts), line in zip(inputs, lines):
        out = bytes.fromhex(line)
        text = out.decode("utf-8")
        assert json.loads('"' + text + '"') == data.decode("utf-8", "replace"), (data.hex(), cuts, line)
        streamed += cuts is not None
    assert streamed > 1000 and len(inputs) - streamed > 1000
    print(f"PASS {len(inputs)} byte strings ({streamed} split into stream chunks) come out of the API as valid UTF-8 JSON strings equal to a UTF-8 decode that replaces each maximal invalid subpart with U+FFFD")


if __name__ == "__main__":
    main()
