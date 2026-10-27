import sys
import numpy as np
V, P, split = 262144, 1099, 550
d = sys.argv[1]
def lsm(x):
    x = x.astype(np.float64)
    mx = x.max(axis=-1, keepdims=True)
    return x - mx - np.log(np.exp(x - mx).sum(axis=-1, keepdims=True))
la = np.memmap(f"{d}/design.ref_a.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P]
lb = np.memmap(f"{d}/design.ref_b.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P]
runs = {name: np.memmap(f"{d}/design.{name}.logits", dtype=np.float32, mode="r").reshape(-1, V)[:P] for name in sys.argv[2:]}
kl = {name: np.empty(P) for name in list(runs) + ["b"]}
for i in range(0, P, 64):
    A = lsm(la[i:i+64]); pa = np.exp(A)
    kl["b"][i:i+64] = (pa * (A - lsm(lb[i:i+64]))).sum(-1)
    for name, r in runs.items():
        kl[name][i:i+64] = (pa * (A - lsm(r[i:i+64]))).sum(-1)
for name in runs:
    above = kl[name] > kl["b"]
    print(name, "share all %.3f prefill %.3f decode<1024 %.3f decode>=1024 %.3f | mean kl prefill %.4f decode %.4f (b %.4f %.4f)" % (
        above.mean(), above[:split].mean(), above[split:1024].mean(), above[1024:].mean(),
        kl[name][:split].mean(), kl[name][split:].mean(), kl["b"][:split].mean(), kl["b"][split:].mean()))
