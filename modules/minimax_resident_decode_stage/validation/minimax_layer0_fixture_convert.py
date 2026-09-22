#!/usr/bin/env python3
import struct
import sys
import zlib

sys.path.insert(0, ".")
from t1_reference_common import read_fixture

FIXTURE_NAME = sys.argv[1]
CAPTURE_POSITIONS = int(sys.argv[2])
OUTPUT = sys.argv[3]

meta, arrays = read_fixture(FIXTURE_NAME)
prompt = arrays["prompt_token_ids"].tolist()
generated = arrays["generated_token_ids"].tolist()
tokens = prompt + generated
if len(tokens) < CAPTURE_POSITIONS:
    raise SystemExit(f"fixture {FIXTURE_NAME} carries {len(tokens)} tokens, need {CAPTURE_POSITIONS}")
tokens = tokens[:CAPTURE_POSITIONS]
with open(OUTPUT + ".positions.bin", "wb") as out:
    out.write(struct.pack("<I", CAPTURE_POSITIONS))
    out.write(struct.pack(f"<{CAPTURE_POSITIONS}I", *tokens))
with open(OUTPUT + ".expected.bin", "wb") as out:
    out.write(struct.pack("<I", CAPTURE_POSITIONS))
    for position in range(CAPTURE_POSITIONS):
        name = f"pos{position:04d}_layer0000_streams"
        if name not in arrays:
            raise SystemExit(f"fixture {FIXTURE_NAME} missing {name}")
        streams = arrays[name].reshape(-1)
        if streams.shape[0] != 5120:
            raise SystemExit(f"fixture {FIXTURE_NAME} {name} width {streams.shape[0]}")
        out.write(streams.astype("<u2").tobytes())
print(f"minimax_layer0_fixture_convert {FIXTURE_NAME} positions={CAPTURE_POSITIONS} tokens={tokens} wrote {OUTPUT}.positions.bin {OUTPUT}.expected.bin")
