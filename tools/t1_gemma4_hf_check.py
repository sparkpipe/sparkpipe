import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import parse_llm_defines  # noqa: E402


def build_engine(arguments):
    module = importlib.import_module("t1_reference_gemma4")
    config = json.load(open(os.path.join(arguments.checkpoint, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    defines = parse_llm_defines(arguments.header)
    return module.ENGINE_CLASS(arguments.checkpoint, defines, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt", default="capital_of_france")
    parser.add_argument("--positions", type=int, default=4)
    arguments = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    engine = build_engine(arguments)
    prompts = json.load(open(arguments.prompts))["prompts"]
    spec = next(p for p in prompts if p["name"] == arguments.prompt)
    token_ids = [int(t) for t in spec["prompt_token_ids"]][:arguments.positions]
    model = AutoModelForCausalLM.from_pretrained(
        arguments.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
    model.eval()
    worst_rel = 0.0
    agree = True
    with torch.no_grad():
        for position, token in enumerate(token_ids):
            out = model(input_ids=torch.tensor([[token]]), use_cache=True,
                        output_hidden_states=True)
            logits = out.logits[0, -1]
            hidden = out.hidden_states[-1][0, -1]
            print(json.dumps({"position": position,
                              "hf_last_hidden_norm":
                                  float(hidden.float().pow(2).mean().sqrt())}))
            hf_top = int(torch.argmax(logits))
            hf_score = float(logits[hf_top])
            streams = engine.decode_step(token, position, {}, {}, {})
            token_id, score = engine.logits(streams)
            top = torch.topk(logits, 8)
            inside = float(top.values[torch.where(top.indices == token_id)[0][0]]) \
                if token_id in top.indices.tolist() else float("-inf")
            scale = max(abs(hf_score), abs(inside), 1e-9)
            rel = abs(hf_score - inside) / scale
            worst_rel = max(worst_rel, rel)
            agree = agree and token_id == hf_top
            print(json.dumps({"position": position, "engine_token": token_id,
                              "hf_token": hf_top, "engine_score": score,
                              "hf_top1": hf_score,
                              "top8_rel": rel if inside != float("-inf") else None}))
    print(f"RESULT: {'PASS' if agree else 'TOP1-DIVERGE'} worst top8 rel "
          f"{worst_rel:.6g}")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
