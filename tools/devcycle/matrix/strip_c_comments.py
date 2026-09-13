#!/usr/bin/env python3
"""Strip C comments while preserving line numbers and string/char literals.
Comment characters become spaces; newlines inside block comments survive, so
every symbol stays on its original line."""

BS = chr(92)
DQ = chr(34)
SQ = chr(39)
NL = chr(10)
SLASH = chr(47)
STAR = chr(42)

def strip_comments(text):
    out = []
    i = 0
    n = len(text)
    mode = "code"
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if mode == "code":
            if c == SLASH and nxt == SLASH:
                mode = "line_comment"
                i += 2
                continue
            if c == SLASH and nxt == STAR:
                mode = "block_comment"
                out.append("  ")
                i += 2
                continue
            if c == DQ:
                mode = "string"
            elif c == SQ:
                mode = "char"
            out.append(c)
            i += 1
        elif mode == "line_comment":
            if c == NL:
                mode = "code"
                out.append(NL)
            i += 1
        elif mode == "block_comment":
            if c == STAR and nxt == SLASH:
                mode = "code"
                out.append("  ")
                i += 2
                continue
            out.append(c if c == NL else " ")
            i += 1
        elif mode == "string":
            out.append(c)
            if c == BS:
                out.append(nxt)
                i += 2
                continue
            if c == DQ:
                mode = "code"
            i += 1
        else:  # char
            out.append(c)
            if c == BS:
                out.append(nxt)
                i += 2
                continue
            if c == SQ:
                mode = "code"
            i += 1
    return "".join(out)

def main():
    import sys
    for path in sys.argv[1:]:
        original = open(path).read()
        stripped = strip_comments(original)
        cleaned = NL.join(line.rstrip() for line in stripped.split(NL))
        if cleaned and not cleaned.endswith(NL):
            cleaned += NL
        open(path, "w").write(cleaned)
        print(path + ": " + str(len(original) - len(cleaned)) + " comment bytes removed")

if __name__ == "__main__":
    main()
