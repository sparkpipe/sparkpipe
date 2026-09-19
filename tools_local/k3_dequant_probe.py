import sys
import time

sys.path.insert(0, "/home/spark7/lane-k3-t1/tools")
import numpy as np

from t1_reference_common import Safetensors
from t1_reference_k3 import _E2M1

st = Safetensors("/home/spark7/lane-k3-t1/kimi-k3-local")
base = "language_model.model.layers.1.block_sparse_moe.experts.0."


def t(label, fn):
    t0 = time.time()
    r = fn()
    print(label, f"{(time.time() - t0) * 1000:.0f} ms", flush=True)
    return r


p1 = t("read payload", lambda: st.raw_rows(base + "w1.weight_packed", 0, 3072))
s1 = t("read scale", lambda: st.raw_rows(base + "w1.weight_scale", 0, 3072))
packed = t("astype int32", lambda: p1.astype(np.int32))
nib = np.empty((3072, 3584), dtype=np.int32)
t("nib lo", lambda: nib.__setitem__((slice(None), slice(0, None, 2)), packed & 0xF))
t("nib hi", lambda: nib.__setitem__((slice(None), slice(1, None, 2)), packed >> 4))
w = t("lut gather", lambda: _E2M1[nib])
w = w.reshape(3072, 96, 32)
sc = t("scale", lambda: np.ldexp(np.float32(1), s1.astype(np.int32) - 127))
prod = t("mul", lambda: w * sc[:, :, None])
out = t("reshape", lambda: prod.reshape(3072, 3584))
print("w sample", out[0, :4], flush=True)
print("PROBE DONE", flush=True)
