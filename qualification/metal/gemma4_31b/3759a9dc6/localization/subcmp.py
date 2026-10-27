import sys
import numpy as np
L, H, P = 60, 5376, 1099
d = sys.argv[1]
m = np.memmap(f"{d}/design.metal.hidden", dtype="<u2", mode="r").reshape(2 * L, -1, H)[:, :P]
a = np.memmap(f"{d}/design.ref_a.hidden", dtype="<u2", mode="r").reshape(2 * L, -1, H)[:, :P]
b = np.memmap(f"{d}/design.ref_b.hidden", dtype="<u2", mode="r").reshape(2 * L, -1, H)[:, :P]
f32 = lambda x: (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)
bands = ((0, 550), (550, 1024), (1024, P))
print("checkpoint | ratio/share per band [0,550) [550,1024) [1024,1099)")
for c in range(2 * L):
    layer, part = c // 2, ("attn" if c % 2 == 0 else "ffn ")
    if layer < 30 and layer % 6 != 5 and layer not in (0, 1):
        continue
    A = f32(a[c]); M = f32(m[c]); B = f32(b[c])
    na = np.linalg.norm(A, axis=-1) + 1e-30
    em = np.linalg.norm(M - A, axis=-1) / na
    eb = np.linalg.norm(B - A, axis=-1) / na
    cells = []
    for lo, hi in bands:
        cells.append("%.2f/%.2f" % (np.median(em[lo:hi]) / np.median(eb[lo:hi]), (em[lo:hi] > eb[lo:hi]).mean()))
    kind = "full " if layer % 6 == 5 else "slide"
    print("%2d %s %s | %s" % (layer, kind, part, "  ".join(cells)))
