#!/usr/bin/env python3
"""Extend the lean DSV4 tree's canonical bucket ladder with buckets 4 and 16.

Mechanical, additive-only delta on ef8fa302ad8f545ee8bfad20c63329d79b76f72c
(all five handoff source pins verified):
  1. generate dense firmware descriptions b4/b16 via the repo's own generator
  2. register their contract sha256 defines in spark_dsv4_model.h
  3. add matching branches to the serving adapter contract ladder
  4. widen the native TP-width admission whitelist to include 4 and 16
No kernel, scheduler, or arithmetic source is touched.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "tools")
import generate_dsv4_contracts as g

NL = chr(10)
CONT = chr(92) + NL
TAB = chr(9)

contract = json.loads(Path(g.CONTRACTS["flash"][0]).read_text())
g.validate_contract("flash", contract)

b1 = g.render_flash_model_description(contract, 1)
b1_sha = hashlib.sha256(b1.encode()).hexdigest()
assert b1_sha == "dd6e9fd0c238c9ae28c16d5276a2d49f112097c0682e142b0b86bfa7eb5b334c", b1_sha
print("B1 canonical regeneration: MATCH", flush=True)

descs = {}
for b in (4, 16):
    d = g.render_flash_model_description(contract, b)
    Path("examples/model_descriptions/dsv4_resident_decode_stage_firmware_b%d.json" % b).write_text(d)
    descs[b] = hashlib.sha256(d.encode()).hexdigest()
    print("b%d description written sha=%s" % (b, descs[b]), flush=True)

hp = Path("model-families/dsv4/include/sparkpipe/spark_dsv4_model.h")
h = hp.read_text()
assert "SHA256_B16" not in h and "SHA256_B4" not in h
anchor = "#define SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B1 \"dd6e9fd0c238c9ae28c16d5276a2d49f112097c0682e142b0b86bfa7eb5b334c\""
assert anchor in h, "header anchor missing"
add = "".join(NL + "#define SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B%d \"%s\"" % (b, descs[b]) for b in (4, 16))
hp.write_text(h.replace(anchor, anchor + add))
print("header defines added", flush=True)

ap = Path("modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c")
a = ap.read_text()
key = "#elif SPARK_BATCH_BUCKET == 1024u"
assert key in a and "SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B16" not in a, "adapter state unexpected"
branches = ""
for b in (4, 16):
    branches += (
        "#elif SPARK_BATCH_BUCKET == %du" % b + NL +
        "#define SPARK_DSV4_SERVING_MODEL_CONTRACT_SHA256 " + CONT + NL +
        TAB + "SPARK_DSV4_MODEL_DESCRIPTION_SHA256_B%d" % b + NL
    )
ap.write_text(a.replace(key, branches + key))
print("adapter ladder extended", flush=True)

fp = Path("modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_resident_decode_stage_firmware.h")
f = fp.read_text()
old = "return(width == 1u || width == 8u || width == 1024u ? 1u : 0u);"
assert old in f, "whitelist line missing"
fp.write_text(f.replace(old, "return(width == 1u || width == 4u || width == 8u || width == 16u || width == 1024u ? 1u : 0u);"))
print("width whitelist extended", flush=True)
print("ALL PATCHES OK", flush=True)
