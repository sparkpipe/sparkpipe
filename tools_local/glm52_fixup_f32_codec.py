#!/usr/bin/env python3
"""Set weight_codec=0 (NONE) for every F32-payload entry in GLM52 packs.

PAYLOAD_F32 entries (ROUTER_CORRECTION) carry no weight codec; the C module
expects SPARK_WEIGHT_CODEC_NONE == 0 there. Earlier blanket BF16-codec patch
set them to 1; undo for payload_type == 2 only.
"""
import struct
import sys


def fixup(path):
    with open(path, "r+b") as f:
        h = f.read(264)
        (magic, ver, hb, eb, abi, flags, count) = struct.unpack_from("<7I", h, 0)
        assert magic == 0x32534C47 and ver == 3, "bad header: %s" % path
        dir_off = struct.unpack_from("<Q", h, 80)[0]
        fixed = 0
        for i in range(count):
            off = dir_off + i * 64
            f.seek(off + 8)
            (pt,) = struct.unpack("<I", f.read(4))
            if pt == 2:
                f.seek(off + 12)
                (wc,) = struct.unpack("<I", f.read(4))
                if wc != 0:
                    f.seek(off + 12)
                    f.write(struct.pack("<I", 0))
                    fixed += 1
        print("%s: f32 entries reverted to codec 0: %d" % (path, fixed))


for p in sys.argv[1:]:
    fixup(p)
