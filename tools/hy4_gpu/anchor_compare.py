#!/usr/bin/env python3
import math
import os
import struct
import sys


def load(p):
    n = os.path.getsize(p)
    with open(p, "rb") as f:
        return struct.unpack("<%df" % (n // 4), f.read())


def main():
    if len(sys.argv) != 4:
        print("usage: anchor_compare.py <cpu_prefix> <gpu_prefix> <n_tokens>")
        return 2
    cpu_pfx, gpu_pfx, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
    layers = 78
    rows = []
    for il in range(layers):
        for t in range(ntok):
            cf = "%s_L%d_t%d" % (cpu_pfx, il, t)
            gf = "%s_L%d_t%d" % (gpu_pfx, il, t)
            for p in (cf, gf):
                if not os.path.exists(p):
                    print("MISSING %s" % p)
                    return 1
            a, b = load(cf), load(gf)
            if len(a) != len(b) or not a:
                print("SIZE %s %d %d" % (cf, len(a), len(b)))
                return 1
            md = 0.0
            mi = -1
            mr = 0.0
            for i in range(len(a)):
                x = a[i]
                y = b[i]
                if x != x or y != y or x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")):
                    print("NONFINITE %s elem %d gpu %.6e cpu %.6e" % (cf, i, y, x))
                    return 1
                d = abs(x - y)
                if d > md:
                    md = d
                    mi = i
                ax = abs(x)
                if ax > mr:
                    mr = ax
            rel = md / mr if mr > 0 else 0.0
            rows.append((il, t, md, rel, mr, mi))
    print("layer tok  max|d|      rel_max     max|ref|    argmax  vs_prev")
    prev = {}
    worst = 0.0
    for il, t, md, rel, mr, mi in rows:
        p = prev.get(t, 0.0)
        ratio = md / p if p > 0 else 0.0
        print("L%02d   t%d  %.3e  %.3e  %.3e  %6d  x%.2f" % (il, t, md, rel, mr, mi, ratio))
        prev[t] = md
        if md > worst:
            worst = md
    print("WORST %.3e over %d slabs" % (worst, len(rows)))
    return 0


sys.exit(main())
