import sys
import numpy as np
L, H, P, split = 60, 5376, 1099, 550
d = sys.argv[1]
m = np.memmap(f"{d}/design.metal.hidden", dtype="<u2", mode="r").reshape(L, -1, H)[:, :P]
a = np.memmap(f"{d}/design.ref_a.hidden", dtype="<u2", mode="r").reshape(L, -1, H)[:, :P]
b = np.memmap(f"{d}/design.ref_b.hidden", dtype="<u2", mode="r").reshape(L, -1, H)[:, :P]
f32 = lambda x: (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)
print("layer kind | prefill: metal b ratio share | decode: metal b ratio share")
for layer in range(L):
    A = f32(a[layer]); M = f32(m[layer]); B = f32(b[layer])
    na = np.linalg.norm(A, axis=-1) + 1e-30
    em = np.linalg.norm(M - A, axis=-1) / na
    eb = np.linalg.norm(B - A, axis=-1) / na
    kind = "full " if layer % 6 == 5 else "slide"
    pm, pb, dm, db = np.median(em[:split]), np.median(eb[:split]), np.median(em[split:]), np.median(eb[split:])
    print("%2d %s | %.2e %.2e %.2f %.2f | %.2e %.2e %.2f %.2f" % (layer, kind, pm, pb, pm / pb, (em[:split] > eb[:split]).mean(), dm, db, dm / db,
          (em[split:] > eb[split:]).mean()))
