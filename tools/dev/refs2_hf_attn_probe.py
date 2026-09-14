import sys, json
sys.path.insert(0, "/home/spark5/refs2/tools")
import numpy as np
from t1_reference_common import parse_llm_defines, rmsnorm, bf16_round_f32, f32_to_bf16_u16, bf16_to_f32
import importlib
module = importlib.import_module("t1_reference_gemma4")
config = json.load(open("/home/spark5/refs2_ckpt/gemma-4-31b-it/config.json"))["text_config"]
defines = parse_llm_defines("/home/spark5/refs2/model-families/gemma4/include/sparkpipe/llm_defines.h")
engine = module.ENGINE_CLASS("/home/spark5/refs2_ckpt/gemma-4-31b-it", defines, config)
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/home/spark5/refs2_ckpt/gemma-4-31b-it",
    torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()
ids = [818, 5279, 529, 7001, 563]
cap = {}
layer0 = model.model.language_model.layers[0]
def hook(name):
    def f(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap[name] = t.detach().float()[0, 4].numpy()
    return f
layer0.self_attn.q_proj.register_forward_hook(hook("q_proj"))
layer0.self_attn.q_norm.register_forward_hook(hook("q_norm"))
layer0.self_attn.k_proj.register_forward_hook(hook("k_proj"))
layer0.self_attn.v_norm.register_forward_hook(hook("v_norm"))
layer0.self_attn.k_norm.register_forward_hook(hook("k_norm"))
layer0.self_attn.register_forward_hook(hook("attn_final"))
with torch.no_grad():
    model(input_ids=torch.tensor([ids]))
p = "model.language_model.layers.0.self_attn."
h = engine.embed(ids[4])
x = bf16_round_f32(rmsnorm(h, engine.tensor("model.language_model.layers.0.input_layernorm.weight"), engine.eps))
q = engine.linear(x, p + "q_proj")
print("q_proj maxrel", round(float(np.abs(q - cap["q_proj"]).max() / (np.abs(cap["q_proj"]).max() + 1e-9)), 5))
q = engine.head_rms(q, engine.tensor(p + "q_norm.weight"), 32, 256)
print("q_norm maxrel", round(float(np.abs(q.reshape(32, 256) - cap["q_norm"]).max() / (np.abs(cap["q_norm"]).max() + 1e-9)), 5))
k_raw = engine.linear(x, p + "k_proj")
v = engine.head_rms(k_raw, None, 16, 256, scaled=False)
print("v_norm maxrel", round(float(np.abs(v - cap["v_norm"].reshape(-1)).max() / (np.abs(cap["v_norm"]).max() + 1e-9)), 5))
k = engine.head_rms(k_raw, engine.tensor(p + "k_norm.weight"), 16, 256)
print("k_norm maxrel", round(float(np.abs(k - cap["k_norm"].reshape(-1)).max() / (np.abs(cap["k_norm"]).max() + 1e-9)), 5))
cos, sin = engine.cos_sin("sliding", 4)
q = engine.rope(q.reshape(32, 256), cos, sin)
k = engine.rope(k.reshape(16, 256), cos, sin)
cache = []
cache.append((f32_to_bf16_u16(k.reshape(-1)), f32_to_bf16_u16(v.reshape(-1))))
keys = bf16_to_f32(np.stack([r[0] for r in cache])).reshape(1, 16, 256)
values = bf16_to_f32(np.stack([r[1] for r in cache])).reshape(1, 16, 256)
out = np.empty((32, 256), dtype=np.float32)
for hh in range(32):
    kvh = hh // 2
    sc = bf16_round_f32((keys[:, kvh, :] @ q[hh]) * np.float32(engine.qk_scale))
    w = np.exp(sc - sc.max())
    w = w / w.sum()
    out[hh] = bf16_round_f32(bf16_round_f32(w) @ values[:, kvh, :])
o = engine.linear(out.reshape(-1), p + "o_proj")
print("attn_final maxrel", round(float(np.abs(o - cap["attn_final"]).max() / (np.abs(cap["attn_final"]).max() + 1e-9)), 5))
