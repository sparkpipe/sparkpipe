import json
import os
import sys
import time

sys.path.insert(0, "/home/spark7/lane-k3-t1/tools")
import numpy as np

from t1_reference_common import parse_llm_defines
from t1_reference_k3 import K3Engine

CKPT = "/home/spark7/lane-k3-t1/kimi-k3-local"
HDR = "/home/spark7/lane-k3-t1/model-families/k3/include/sparkpipe/llm_defines.h"

defines = parse_llm_defines(HDR)
config = json.load(open(os.path.join(CKPT, "config.json")))["text_config"]
engine = K3Engine(CKPT, defines, config)
print("engine ready", flush=True)

x = engine.embed(1008)
prefix = "language_model.model.layers.0."
t0 = time.time()
normed = bf16n = None
import t1_reference_common as C
normed = C.bf16_round_f32(C.rmsnorm(
    x, engine.tensor(prefix + "input_layernorm.weight"), engine.eps))
print(f"layer0 norm {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
q = engine.linear(normed, prefix + "self_attn.q_proj")
print(f"layer0 q_proj linear {time.time() - t0:.2f}s", flush=True)

state = {"state": np.zeros((engine.kda_heads, engine.kd, engine.kd),
                           dtype=np.float32),
         "wq": np.zeros((engine.kda_dim, engine.conv), dtype=np.uint16),
         "wk": np.zeros((engine.kda_dim, engine.conv), dtype=np.uint16),
         "wv": np.zeros((engine.kda_dim, engine.conv), dtype=np.uint16)}
t0 = time.time()
attn = engine.kda_attention(prefix, normed, state)
print(f"layer0 full kda_attention {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
attn = engine.kda_attention(prefix, normed, state)
print(f"layer0 kda_attention again {time.time() - t0:.2f}s", flush=True)

p1 = "language_model.model.layers.1."
t0 = time.time()
moe_in = C.bf16_round_f32(engine.tensor(
    p1 + "block_sparse_moe.routed_expert_down_proj.weight") @ normed)
print(f"layer1 routed_down {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
plane = engine.routed_expert(p1, 0, moe_in)
print(f"layer1 one routed expert {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
plane = engine.routed_expert(p1, 1, moe_in)
print(f"layer1 second routed expert {time.time() - t0:.2f}s", flush=True)
m4 = "language_model.model.layers.4."
t0 = time.time()
normed4 = C.bf16_round_f32(C.rmsnorm(
    x, engine.tensor(m4 + "input_layernorm.weight"), engine.eps))
mla = engine.mla_attention(m4, normed4, [])
print(f"layer4 first mla {time.time() - t0:.2f}s", flush=True)
print("PROBE DONE", flush=True)
