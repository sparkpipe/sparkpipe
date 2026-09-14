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
for L in (57, 58, 59):
    mine = engine.layer_trace[L]
    ref = hf[L + 1]
    print("layer", L, "mine norm", round(float(np.sqrt((mine*mine).mean())), 5),
          "hf norm", round(float(np.sqrt((ref*ref).mean())), 5),
          "maxrel", round(float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-9)), 4))
    print("  mine[:5]", [round(float(v), 4) for v in mine[:5]])
    print("  hf  [:5]", [round(float(v), 4) for v in ref[:5]])
logits_hf = out.logits[0, 4].float().numpy()
norm = engine.layer_trace[59]
import t1_reference_common as tc
mine_norm = tc.bf16_round_f32(tc.rmsnorm(norm, engine.tensor("model.language_model.norm.weight"), engine.eps))
emb = engine.tensor("model.language_model.embed_tokens.weight")
scores = bf = bf16_to_f32(emb) @ mine_norm
top5 = np.argsort(-scores)[:5]
print("mine top5:", top5.tolist(), [round(float(scores[i]), 3) for i in top5])
hf_top5 = np.argsort(-logits_hf)[:5]
print("hf   top5:", hf_top5.tolist(), [round(float(logits_hf[i]), 3) for i in hf_top5])
soft = np.tanh(logits_hf / 30.0) * 30.0
hf_soft_top5 = np.argsort(-soft)[:5]
print("hf softcap top5:", hf_soft_top5.tolist(), [round(float(soft[i]), 3) for i in hf_soft_top5])
