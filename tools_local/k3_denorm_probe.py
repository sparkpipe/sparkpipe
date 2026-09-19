import sys
import time

sys.path.insert(0, "/home/spark7/lane-k3-t1/tools")
import numpy as np

from t1_reference_common import Safetensors
from t1_reference_k3 import Mxfp4Scratch

st = Safetensors("/home/spark7/lane-k3-t1/kimi-k3-local")


def probe(layer, expert):
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}."
    t0 = time.time()
    p1 = st.raw_rows(base + "w1.weight_packed", 0, 3072)
    s1 = st.raw_rows(base + "w1.weight_scale", 0, 3072)
    read_ms = (time.time() - t0) * 1000
    codes = s1.astype(np.int32) - 127
    t0 = time.time()
    scratch = Mxfp4Scratch(3072, 3584)
    w = scratch.dequant(p1, s1, 3072, 3584)
    dequant_ms = (time.time() - t0) * 1000
    denorm = int((np.abs(w) < np.float32(1.1754944e-38)).sum())
    nonzero = int((w != 0).sum())
    print(f"layer {layer} expert {expert}: read {read_ms:.0f}ms "
          f"dequant {dequant_ms:.0f}ms code_min {codes.min()} "
          f"code_max {codes.max()} denormals {denorm}/{nonzero}",
          flush=True)
    t0 = time.time()
    y = w @ np.ones(3584, dtype=np.float32)
    print(f"  gemv {(time.time() - t0) * 1000:.0f}ms y0 {float(y[0]):.3e}",
          flush=True)


probe(1, 0)
probe(2, 0)
probe(2, 1)
print("PROBE DONE", flush=True)
