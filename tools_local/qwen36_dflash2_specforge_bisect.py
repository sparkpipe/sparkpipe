#!/usr/bin/env python3
"""Layer-bisect: specforge's DFlashDraftModel vs our torch forward on the
reference input dumps. Per-layer output hooks on both sides; the first
diverging layer localizes the scoring-path difference."""
import glob
import json
import re

import torch
from safetensors import safe_open
from transformers import Qwen3Config
from specforge.modeling.draft.dflash import DFlashDraftModel

DRAFTER = "/home/spark3/sparkdata/qwen38-dflash2-drafter"
H = 5120
BLOCK = 8
MASK_ID = 248070
DEV = "cpu"

# --- load specforge model ---
cfg_dict = json.load(open(f"{DRAFTER}/config.json"))
d = cfg_dict.pop("dflash_config")
cfg_dict.update({k: d[k] for k in ("block_size", "mask_token_id", "target_layer_ids", "conv_kernel_size", "conv_group_size")})
cfg = Qwen3Config(**cfg_dict)
cfg.torch_dtype = torch.float32
model = DFlashDraftModel(cfg)
state = {}
with safe_open(f"{DRAFTER}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k)
# force-remap unexpected: specforge param tree vs checkpoint naming
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())
remapped = {}
for k in ckpt_keys - model_keys:
    # try inserting under model.* prefix or other common remaps
    for cand in (f"model.{k}", k.replace("fc.", "fc."), k):
        if cand in model_keys:
            remapped[cand] = state[k]
            break
sd = {k: v for k, v in state.items() if k in model_keys}
sd.update(remapped)
res = model.load_state_dict(sd, strict=False)
print("missing:", [k for k in res.missing_keys if "rotary" not in k][:5])
print("unmapped ckpt keys:", len([k for k in res.unexpected_keys]))
model.eval()

# --- load our forward (from the parity harness) ---
import importlib.util
spec = importlib.util.spec_from_file_location("dp", "/tmp/qwen38_27b_dflash2_vllm_input_parity.py")
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)
dp.DEV = DEV
dp.ROPE_MODE = "neox"

# shared weights for our forward
w = dp.load(f"{DRAFTER}/model.safetensors")
emb_w = dp.load("/home/spark3/sparkdata/Qwen3.8-27B-local/model-00003-of-00018.safetensors", ["embed_tokens.weight"])["model.language_model.embed_tokens.weight"]

# --- dumps ---
dumps = {}
for f in glob.glob("/tmp/vllmin_*.pt"):
    n = int(re.search(r"vllmin_(\d+)", f).group(1))
    dumps[n] = torch.load(f, weights_only=False, map_location="cpu")

# --- hooks: capture specforge per-layer outputs ---
sf_captures = {}
def mk_hook(i):
    def hook(mod, args, output):
        if isinstance(output, tuple):
            sf_captures[i] = output[0].detach()
        else:
            sf_captures[i] = output.detach()
    return hook
for i, layer in enumerate(model.layers):
    layer.register_forward_hook(mk_hook(i))

# our per-layer captures: wrap our layer()
our_captures = {}
orig_layer = dp.layer
def wrapped_layer(x, ctx, lw, q_pos, kv_pos, kv_store=None, layer_i=0, ctx_pos=None):
    out = orig_layer(x, ctx, lw, q_pos, kv_pos, kv_store, layer_i, ctx_pos)
    our_captures[layer_i] = out.detach().float()
    return out
dp.layer = wrapped_layer

# --- compare one round ---
cur = dumps[3]  # a mid-run round
nxt = dumps[4]
hidden = cur["target_hidden_states"].to(DEV, torch.bfloat16).float()
pos = cur["target_positions"].to(DEV).view(-1)
bonus = int(cur["next_token_ids"][0])
their_walked = [int(t) for t in nxt["target_token_ids"].view(-1).tolist()]
their_drafts = their_walked[1:8]
nr = int(cur["num_rejected"].view(-1)[0]) if cur.get("num_rejected") is not None else 0
vis = max(1, hidden.shape[0] - nr)
pmax = int(pos[:vis].max())

# --- specforge forward ---
with torch.no_grad():
    noise_ids = torch.tensor([[bonus] + [MASK_ID] * (BLOCK - 1)], dtype=torch.long, device=DEV)
    noise_emb = emb_w.to(DEV).float()[noise_ids]  # [1, 8, H]
    # the dump's target_hidden_states = fc(raw_aux) ALREADY COMBINED; specforge's
    # forward applies fc+hidden_norm internally - bypass both with Identity and
    # feed the pre-normed context (rms(combined, hidden_norm_w)) ourselves
    ctx_pre = dp.rms(hidden[:vis].to(torch.bfloat16), w["hidden_norm.weight"]).float()
    import torch.nn as _nn
    model.fc = _nn.Identity()
    model.hidden_norm = _nn.Identity()
    full_pos = torch.cat([pos[:vis], torch.arange(pmax + 1, pmax + 1 + BLOCK, device=DEV)]).unsqueeze(0)
    out = model(
        position_ids=full_pos,
        noise_embedding=noise_emb,
        target_hidden=ctx_pre.unsqueeze(0),
        past_key_values=None,
    )
    sf_hidden = out  # [1, 8, H] after norm

# --- our forward ---
with torch.no_grad():
    ctx = dp.rms(hidden[:vis].to(torch.bfloat16), w["hidden_norm.weight"]).float()
    ids = torch.tensor([bonus] + [MASK_ID] * (BLOCK - 1), device=DEV)
    x = emb_w.to(DEV)[ids].float()
    q_pos = torch.arange(pmax + 1, pmax + 1 + BLOCK, device=DEV)
    kv_pos = torch.cat([pos[:vis], q_pos])
    x = x.to(torch.bfloat16)
    ctx = ctx.to(torch.bfloat16)
    for layer_i in range(dp.LAYERS):
        lw = {k.split(f"layers.{layer_i}.")[1]: v for k, v in w.items() if f"layers.{layer_i}." in k}
        x = dp.layer(x, ctx, lw, q_pos, kv_pos, None, layer_i)
        x = x.to(torch.bfloat16)
    our_hidden = dp.rms(x.to(torch.bfloat16), w["norm.weight"]).float()

# --- per-layer compare ---
print(f"\nround: bonus={bonus} vis={vis} pmax={pmax}")
print(f"their drafts: {their_drafts}")
for i in range(5):
    sf = sf_captures[i].squeeze(0)      # [8, H]
    ou = our_captures[i]                 # [8, H]
    rel = ((sf - ou).norm() / (sf.norm() + 1e-9)).item()
    cos = torch.nn.functional.cosine_similarity(sf.flatten(), ou.flatten(), dim=0).item()
    print(f"layer {i}: rel-L2={rel:.6f} cos={cos:.6f} sf_norm={sf.norm():.1f} our_norm={ou.norm():.1f}")
fin_rel = ((sf_hidden.squeeze(0) - our_hidden).norm() / (sf_hidden.norm() + 1e-9)).item()
print(f"final: rel-L2={fin_rel:.6f}")
