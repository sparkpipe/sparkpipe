#!/usr/bin/env python3
"""In-place patch of GLM52 stage packs: BF16 weight codec 0 -> 1.

The C enum SparkWeightCodec has SPARK_WEIGHT_CODEC_BF16 == 1, but the first
packer wrote 0 for BF16 (header linear/kv codec fields and every BF16 entry).
Patch header offsets 60 (linear) and 68 (kv) and every directory entry whose
weight_codec == 0, in place. Expert codec 5 (FP8_E4M3) must already match.
"""
import struct
import sys


def patch(path):
    with open(path, "r+b") as f:
        h = f.read(264)
        (magic, ver, hb, eb, abi, flags, count, sc, si, fl, lc, tlc,
         hid, voc, exp, lin, expc, kv) = struct.unpack_from("<18I", h, 0)
        assert magic == 0x32534C47 and ver == 3, "bad header: %s" % path
        assert expc == 5, "unexpected expert codec %d: %s" % (expc, path)
        dir_off = struct.unpack_from("<Q", h, 80)[0]
        if lin != 1:
            f.seek(60)
            f.write(struct.pack("<I", 1))
        if kv != 1:
            f.seek(68)
            f.write(struct.pack("<I", 1))
        patched = 0
        for i in range(count):
            off = dir_off + i * 64
            f.seek(off + 12)
            (wc,) = struct.unpack("<I", f.read(4))
            if wc == 0:
                f.seek(off + 12)
                f.write(struct.pack("<I", 1))
                patched += 1
            elif wc != 5:
                raise RuntimeError("unexpected codec %d at entry %d of %s" % (wc, i, path))
        print("%s: tensors=%d header linear=%d kv=%d patched_entries=%d" % (path, count, lin, kv, patched))


for p in sys.argv[1:]:
    patch(p)
