#!/usr/bin/env python3
"""Remove the spurious blank line inside the generated contract-sha branches."""
from pathlib import Path

BS = chr(92)
p = Path("modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c")
lines = p.read_text().split(chr(10))
out = []
fixed = 0
i = 0
while i < len(lines):
    cur = lines[i]
    if (
        out
        and cur == ""
        and out[-1].endswith("SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 " + BS)
        and i + 1 < len(lines)
        and lines[i + 1].startswith(TAB := chr(9))
        and "SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B" in lines[i + 1]
    ):
        fixed += 1
        i += 1
        continue
    out.append(cur)
    i += 1
assert fixed == 2, "expected exactly 2 fixes, got %d" % fixed
p.write_text(chr(10).join(out))
print("continuation blank lines removed:", fixed)
