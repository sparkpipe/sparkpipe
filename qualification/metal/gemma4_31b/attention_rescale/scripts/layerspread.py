import sys
import numpy as np
L, H, P = 60, 5376, 1099
bands = [("0-549", 0, 550), ("550-1023", 550, 1024), ("1024+", 1024, P)]
d = sys.argv[1]
names = sys.argv[2:]
load = lambda path: np.memmap(path, dtype="<u2", mode="r").reshape(L, -1, H)[:, :P]
f32 = lambda x: (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)
a = load(f"{d}/design.ref_a.hidden")
b = load(f"{d}/design.ref_b.hidden")
runs = {name: load(f"{d}/design.{name}.hidden") for name in names}
header = "layer kind | " + " | ".join(f"{band}: " + " ".join(f"{name}" for name in names) for band, _, _ in bands)
print("per layer: median of |x - a| / |a| over the band's positions, as a ratio to b's, then the share of positions where x is further from a than b is")
print(header)
for layer in range(L):
    A = f32(a[layer])
    na = np.linalg.norm(A, axis=-1) + 1e-30
    eb = np.linalg.norm(f32(b[layer]) - A, axis=-1) / na
    errors = {name: np.linalg.norm(f32(run[layer]) - A, axis=-1) / na for name, run in runs.items()}
    kind = "full " if layer % 6 == 5 else "slide"
    cells = []
    for band, lo, hi in bands:
        cells.append(" ".join("%.2f/%.2f" % (np.median(errors[name][lo:hi]) / np.median(eb[lo:hi]), (errors[name][lo:hi] > eb[lo:hi]).mean()) for name in names))
    print("%2d %s | %s" % (layer, kind, " | ".join(cells)))
