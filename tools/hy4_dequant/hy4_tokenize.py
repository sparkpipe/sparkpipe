#!/usr/bin/env python3
"""hy4 lane: GGUF-BPE tokenizer CLI (encode text -> ids, decode ids -> text).

The same byte-level BPE verified against round-trips on real GGUF vocab
(see tick-7 receipt). Reads the tokenizer KVs straight from the rank-00
GGUF header (first ~5 MB). No ceph, no warm: node-local file only.

Usage:
  hy4_tokenize.py encode "some text"          # prints ids (space separated)
  hy4_tokenize.py decode 1 2 3                # prints text
  hy4_tokenize.py bos                         # prints the bos token id
"""
import struct
import sys

GGUF = "/home/spark2/hy4-allranks/rank-00/model-ud-iq1m-tp16-rank-00.gguf"


def load_tokenizer():
    f = open(GGUF, "rb")
    def r(fmt):
        d = f.read(struct.calcsize(fmt))
        assert len(d) == struct.calcsize(fmt)
        return struct.unpack(fmt, d)
    def rstring():
        (n,) = r("<Q")
        return f.read(n).decode("utf-8")
    def rvalue(vt):
        if vt == 8:
            return rstring()
        if vt == 9:
            (et,) = r("<I")
            (n,) = r("<Q")
            return [rvalue(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}.get(vt)
        assert fmt, vt
        return r(fmt)[0]
    r("<II")
    (tensor_count, kv_count) = r("<QQ")
    kv = {}
    for _ in range(kv_count):
        key = rstring()
        (vt,) = r("<I")
        kv[key] = rvalue(vt)
    return kv


def main():
    kv = load_tokenizer()
    tokens = kv["tokenizer.ggml.tokens"]
    merges = [tuple(m.split(" ")) for m in kv["tokenizer.ggml.merges"]]
    bos = kv.get("tokenizer.ggml.bos_token_id")
    index = {t: i for i, t in enumerate(tokens)}
    ranks = {pair: i for i, pair in enumerate(merges)}

    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1)))
    n = 0
    char_to_byte = {}
    for b in range(256):
        if b in bs:
            char_to_byte[chr(b)] = chr(b)
        else:
            char_to_byte[chr(256 + n)] = chr(b)
            n += 1

    def bpe_word(word):
        parts = list(word)
        while len(parts) > 1:
            best = min((ranks.get((parts[i], parts[i + 1]), 1 << 30)
                        for i in range(len(parts) - 1)), default=1 << 30)
            if best == 1 << 30:
                break
            for i in range(len(parts) - 1):
                if ranks.get((parts[i], parts[i + 1])) == best:
                    parts[i:i + 2] = [parts[i] + parts[i + 1]]
                    break
        return parts

    def encode(text, prepend_bos=True):
        ids = [bos] if (prepend_bos and bos is not None) else []
        words = text.split(" ")
        for j, w in enumerate(words):
            marked = ("\u0120" + w) if j else w
            for piece in bpe_word(marked):
                ids.append(index[piece])
        return ids

    def decode(ids):
        out = []
        for i in ids:
            tok = tokens[i]
            out.append("".join(char_to_byte.get(ch, ch) for ch in tok))
        return "".join(out).replace("\u0120", " ")

    cmd = sys.argv[1]
    if cmd == "encode":
        print(" ".join(str(i) for i in encode(sys.argv[2])))
    elif cmd == "decode":
        ids = [int(v) for v in sys.argv[2:]]
        print(decode(ids))
    elif cmd == "bos":
        print(bos)
    else:
        raise SystemExit("unknown cmd")


if __name__ == "__main__":
    main()
