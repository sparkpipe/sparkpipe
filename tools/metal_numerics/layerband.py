import sys
import numpy as np
d, prompt, L, H, P, split = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
candidates = sys.argv[7].split(",")
draws = sys.argv[8].split(",") if len(sys.argv) > 8 else ["b", "c", "d"]
load = lambda path: np.memmap(path, dtype="<u2", mode="r").reshape(L, -1, H)[:, :P]
f32 = lambda x: (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)
a = load(f"{d}/{prompt}.ref_a.hidden")
refs = {x: load(f"{d}/{prompt}.ref_{x}.hidden") for x in draws}
runs = {x: load(f"{d}/{prompt}.{x}.hidden") for x in candidates}
bands = [("prefill", 0, split), ("decode", split, P)] if 0 < split < P else [("all", 0, P)]
print("per layer: median over positions of |x - a| / |a|, as a ratio to the largest median among draws %s; then the share of positions where x is further from a than every draw (a fourth draw would be 0.25)" % ",".join(draws))
print("layer | " + " | ".join(f"{name}: band(min..max draw) " + " ".join(candidates) for name, _, _ in bands))
first = {}
for layer in range(L):
    A = f32(a[layer])
    na = np.linalg.norm(A, axis=-1) + 1e-30
    err = {x: np.linalg.norm(f32(r[layer]) - A, axis=-1) / na for x, r in {**refs, **runs}.items()}
    cells = []
    for name, lo, hi in bands:
        draw_medians = [np.median(err[x][lo:hi]) for x in draws]
        top = max(draw_medians)
        worst = np.max(np.stack([err[x][lo:hi] for x in draws]), axis=0)
        parts = ["%.2f..%.2f" % (min(draw_medians) / top, 1.0)]
        for x in candidates:
            ratio = np.median(err[x][lo:hi]) / top
            share = (err[x][lo:hi] > worst).mean()
            parts.append("%.2f/%.2f" % (ratio, share))
            if ratio > 1.0 and (name, x) not in first:
                first[(name, x)] = layer
        cells.append(" ".join(parts))
    print("%2d | %s" % (layer, " | ".join(cells)), flush=True)
print("first layer above the draws' band (median ratio > 1):", {f"{k[0]} {k[1]}": v for k, v in sorted(first.items())})
