#!/usr/bin/env python3
"""Greedy reference tokens for gemma-4 from the publisher's transformers
implementation, independent of the SparkPipe driver.

Each prompt is rendered with the checkpoint's own chat template (thinking
off, generation prompt on) and decoded greedily in bf16 on the CPU. The
model is built without weight initialization (its parameters are untouched
lazy allocations, its buffers are computed as the publisher computes them)
and every checkpoint tensor is then bound zero-copy to a read-only memory map of its safetensors file, so the weights
are reclaimable page cache rather than anonymous memory and the run fits
beside a serving lane under a cgroup memory bound. The output JSON records
the rendered prompt, its token ids, the greedy ids, the decoded text and the
top-1/top-2 logit margin per step (a margin near zero marks a step where a
bf16 engine may legally pick the other token).

Run:
  python3 tools/gemma4_hf_reference.py --checkpoint /path/gemma-4-31b-it \
      --prompts qualification/gemma4/reference_prompts.json \
      --max-new-tokens 32 --threads 8 --output reference.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import platform
import struct
import sys
import time
import warnings
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_tensors(checkpoint: Path, torch) -> dict:
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
              "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    tensors = {}
    for file_name in sorted(set(index["weight_map"].values())):
        handle = open(checkpoint / file_name, "rb")
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        header_bytes = struct.unpack("<Q", mapped[:8])[0]
        header = json.loads(mapped[8:8 + header_bytes])
        base = 8 + header_bytes
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            dtype = dtypes[meta["dtype"]]
            flat = torch.frombuffer(mapped, dtype=dtype, count=(end - begin) // torch.empty(0, dtype=dtype).element_size(), offset=base + begin)
            tensors[name] = flat.view(meta["shape"])
    return tensors


def bind_weights(model, tensors: dict) -> dict:
    expected = set(model.state_dict().keys())
    bound = {}
    for name in expected:
        if name in tensors:
            bound[name] = tensors[name]
    missing = sorted(expected - set(bound))
    model.load_state_dict(bound, strict=False, assign=True)
    model.tie_weights()
    unbound = sorted(name for name in expected - set(bound) if name != "lm_head.weight")
    if model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        unbound.append("lm_head.weight (not tied to the bound embedding)")
    return {"missing": missing, "unbound": unbound,
            "unused": sorted(set(tensors) - set(bound))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--bind-only", action="store_true")
    arguments = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    warnings.filterwarnings("ignore", message=".*not writable.*")
    torch.set_num_threads(arguments.threads)
    started = time.time()
    config = AutoConfig.from_pretrained(arguments.checkpoint)
    from transformers.initialization import no_init_weights
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    binding = bind_weights(model, mapped_tensors(arguments.checkpoint, torch))
    print(json.dumps({"model_class": type(model).__name__,
                      "missing": binding["missing"][:20],
                      "unbound": binding["unbound"][:20],
                      "unused": binding["unused"][:20],
                      "unused_count": len(binding["unused"])}), flush=True)
    if binding["unbound"]:
        raise SystemExit("model weights left unbound to the checkpoint")
    for name, buffer in model.named_buffers():
        if buffer.is_meta:
            raise SystemExit(f"buffer {name} left on the meta device")
    if arguments.bind_only:
        return 0
    model.eval()
    load_seconds = time.time() - started
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint)
    prompts = json.loads(arguments.prompts.read_text())["prompts"]
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt["text"]}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = tokenizer(rendered, return_tensors="pt",
                              add_special_tokens=False).input_ids
        step_started = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False, max_new_tokens=arguments.max_new_tokens,
                output_scores=True, return_dict_in_generate=True,
                top_k=None, top_p=None, temperature=None)
        new_ids = generated.sequences[0, input_ids.shape[1]:].tolist()
        margins = []
        for scores in generated.scores:
            top = torch.topk(scores[0].float(), 2)
            margins.append({"top1": int(top.indices[0]), "top2": int(top.indices[1]),
                            "margin": float(top.values[0] - top.values[1])})
        results.append({
            "name": prompt["name"],
            "text": prompt["text"],
            "rendered_prompt": rendered,
            "prompt_token_ids": input_ids[0].tolist(),
            "greedy_token_ids": new_ids,
            "greedy_text": tokenizer.decode(new_ids, skip_special_tokens=False),
            "step_margins": margins,
            "seconds": time.time() - step_started,
        })
        print(json.dumps({"name": prompt["name"], "greedy_token_ids": new_ids,
                          "text": results[-1]["greedy_text"],
                          "seconds": results[-1]["seconds"]}), flush=True)
    receipt = {
        "tool": "tools/gemma4_hf_reference.py",
        "checkpoint": str(arguments.checkpoint),
        "model_class": type(model).__name__,
        "config_sha256": sha256_file(arguments.checkpoint / "config.json"),
        "index_sha256": sha256_file(arguments.checkpoint / "model.safetensors.index.json"),
        "chat_template_sha256": sha256_file(arguments.checkpoint / "chat_template.jinja"),
        "tokenizer_sha256": sha256_file(arguments.checkpoint / "tokenizer.json"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "host": platform.node(),
        "dtype": "bfloat16",
        "attention": "eager",
        "decoding": "greedy",
        "threads": arguments.threads,
        "max_new_tokens": arguments.max_new_tokens,
        "load_seconds": load_seconds,
        "unbound_checkpoint_tensors": len(binding["unused"]),
        "results": results,
    }
    arguments.output.write_text(json.dumps(receipt, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
