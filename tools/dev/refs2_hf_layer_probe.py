import sys, json
sys.path.insert(0, "/home/spark5/refs2/tools")
import numpy as np
from t1_reference_common import parse_llm_defines
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
with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
hf = [h[0, 4].float().numpy() for h in out.hidden_states]
for pos, tok in enumerate(ids):
    engine.decode_step(tok, pos, {}, {}, {})
worst = 0.0
worst_layer = -1
for L in range(60):
    mine = engine.layer_trace[L]
    ref = hf[L + 1]
    rel = float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-9))
    if rel > worst:
        worst = rel
        worst_layer = L
    if rel > 0.02 or L < 2 or L > 57:
        print("layer", L, "maxrel", round(rel, 4), flush=True)
print("WORST layer", worst_layer, "maxrel", round(worst, 4))
