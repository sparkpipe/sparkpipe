#!/usr/bin/env python3
"""hy4 lane: GGUF-BPE tokenizer CLI (encode text -> ids, decode ids -> text).

Byte-level BPE with the hyv4 pretokenizer ported from llama.cpp
(llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_HYV4: three sequential regex splits,
then byte-level). Verified against llama.cpp tokenizations of the same
GGUF:
  "The quick brown fox"                  -> [802, 5466, 19405, 63357]
  "-p The quick brown fox -n 4 --temp 0" -> [2707, 499, 5466, 19405, 63357,
                                             516, 77, 220, 19, 2411, 22093,
                                             220, 15]
hyv4 emits NO BOS. Reads the tokenizer KVs straight from the rank-00 GGUF
header (first ~5 MB). No ceph, no warm: node-local file only. Needs the
`regex` module (RE2-compatible \\p classes): run with the spark2 venv
python or any interpreter that has it.

Usage:
  hy4_tokenize.py encode "some text"          # prints ids (space separated)
  hy4_tokenize.py decode 1 2 3                # prints text
  hy4_tokenize.py bos                         # prints the bos token id
  hy4_tokenize.py selftest                    # llama-vector verification
"""
import struct
import sys

GGUF = "/home/spark2/hy4-allranks/rank-00/model-ud-iq1m-tp16-rank-00.gguf"

try:
    import regex as _rx
except ImportError:
    _rx = None

HYV4_SPLITS = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
]

SELFTEST_CASES = [
    ("The quick brown fox", [802, 5466, 19405, 63357]),
    ("-p The quick brown fox -n 4 --temp 0",
     [2707, 499, 5466, 19405, 63357, 516, 77, 220, 19, 2411, 22093, 220, 15]),
]


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


def byte_encoder():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def hyv4_fragments(text):
    if _rx is None:
        raise SystemExit("the `regex` module is required (spark2 venv python)")
    frags = [text]
    for pat in HYV4_SPLITS:
        rx = _rx.compile(pat)
        nxt = []
        for frag in frags:
            pos = 0
            for m in rx.finditer(frag):
                if m.start() > pos:
                    nxt.append(frag[pos:m.start()])
                nxt.append(frag[m.start():m.end()])
                pos = m.end()
            if pos < len(frag):
                nxt.append(frag[pos:])
        frags = nxt
    return frags


def main():
    kv = load_tokenizer()
    tokens = kv["tokenizer.ggml.tokens"]
    merges = [tuple(m.split(" ")) for m in kv["tokenizer.ggml.merges"]]
    bos = kv.get("tokenizer.ggml.bos_token_id")
    index = {t: i for i, t in enumerate(tokens)}
    ranks = {pair: i for i, pair in enumerate(merges)}
    be = byte_encoder()
    char_to_byte = {v: chr(k) for k, v in be.items()}

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

    def encode(text, prepend_bos=False):
        ids = [bos] if (prepend_bos and bos is not None) else []
        for frag in hyv4_fragments(text):
            if not frag:
                continue
            word = "".join(be[b] for b in frag.encode("utf-8"))
            for piece in bpe_word(word):
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
    elif cmd == "selftest":
        ok = True
        for text, want in SELFTEST_CASES:
            got = encode(text)
            passed = got == want
            ok &= passed
            print("PASS" if passed else "FAIL", repr(text), got)
        for text, want in SELFTEST_CASES:
            rt = decode(encode(text))
            passed = rt == text
            ok &= passed
            print("PASS" if passed else "FAIL", "roundtrip", repr(text))
        raise SystemExit(0 if ok else 1)
    else:
        raise SystemExit("unknown cmd")


if __name__ == "__main__":
    main()
