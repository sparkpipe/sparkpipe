import sys
import numpy as np
V, P = 262144, 1099
d = sys.argv[1]
def lsm(x):
    x = x.astype(np.float64)
    mx = x.max(axis=-1, keepdims=True)
    return x - mx - np.log(np.exp(x - mx).sum(axis=-1, keepdims=True))
la = np.memmap(f"{d}/design.ref_a.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P]
lb = np.memmap(f"{d}/design.ref_b.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P]
names = sys.argv[2:]
runs = {name: np.memmap(f"{d}/design.{name}.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P] for name in names}
kl = {name: np.empty(P) for name in names + ["b"]}
for i in range(0, P, 64):
    A = lsm(la[i:i+64]); pa = np.exp(A)
    kl["b"][i:i+64] = (pa * (A - lsm(lb[i:i+64]))).sum(-1)
    for name, r in runs.items():
        kl[name][i:i+64] = (pa * (A - lsm(r[i:i+64]))).sum(-1)
fk = kl["b"].mean()
for name in names:
    above = kl[name] > kl["b"]
    share = above.mean()
    blocks = [(above[i:i+128].sum(), len(above[i:i+128])) for i in range(0, P, 128)]
    n = len(blocks)
    err = np.sqrt(n / (n - 1) * sum((a - share * c) ** 2 for a, c in blocks)) / P
    print("%-6s design vs a, floor b: KL %.4f = %.2fx floor, paired share %.3f = %+.1f sigma" % (name, kl[name].mean(), kl[name].mean() / fk, share, (share - 0.5) / err))
