#!/usr/bin/env python3
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

import torch

MODELING_SHA256 = "c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_rope_parameters(config, device=None, seq_len=None):
    base = config.rope_theta
    partial = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def install_transformers_compat():
    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters


def load_publisher_modules(checkpoint):
    install_transformers_compat()
    modeling = os.path.join(checkpoint, "modeling_bailing_moe_v3.py")
    actual = sha256_file(modeling)
    if actual != MODELING_SHA256:
        raise SystemExit(f"modeling sha256 {actual} != pinned {MODELING_SHA256}")
    package_root = tempfile.mkdtemp(prefix="ling_hf_ref_")
    package = os.path.join(package_root, "ling_publisher")
    os.makedirs(package)
    open(os.path.join(package, "__init__.py"), "w").close()
    for name in ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"):
        shutil.copy(os.path.join(checkpoint, name), os.path.join(package, name))
    sys.path.insert(0, package_root)
    import ling_publisher.configuration_bailing_moe_v3 as configuration
    import ling_publisher.modeling_bailing_moe_v3 as modeling_module
    return configuration, modeling_module


DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


class ShardReader:
    def __init__(self, checkpoint):
        with open(os.path.join(checkpoint, "model.safetensors.index.json")) as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self.checkpoint = checkpoint
        self.headers = {}
        self.descriptors = {}
        self.bytes_read = 0

    def header(self, shard):
        if shard not in self.headers:
            descriptor = os.open(os.path.join(self.checkpoint, shard), os.O_RDONLY)
            length = int.from_bytes(os.pread(descriptor, 8, 0), "little")
            self.headers[shard] = (json.loads(os.pread(descriptor, length, 8)), 8 + length)
            self.descriptors[shard] = descriptor
        return self.headers[shard]

    def get(self, name):
        shard = self.weight_map[name]
        header, base = self.header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        descriptor = self.descriptors[shard]
        data = bytearray(end - begin)
        view = memoryview(data)
        done = 0
        while done < len(data):
            chunk = os.preadv(descriptor, [view[done:]], base + begin + done)
            if chunk <= 0:
                raise SystemExit(f"{name}: short read at {done} of {len(data)}")
            done += chunk
        os.posix_fadvise(descriptor, base + begin, end - begin, os.POSIX_FADV_DONTNEED)
        self.bytes_read += len(data)
        return torch.frombuffer(data, dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])


SWIGLU_LIMIT_MODES = ("modeling", "serving")


def swiglu(gate, up, limit):
    activated = torch.nn.functional.silu(gate)
    if limit is None:
        return activated * up
    return activated.clamp(max=limit) * up.clamp(min=-limit, max=limit)


def layer_limit(limits, layer):
    if limits is None or not 0 <= layer < len(limits) or not limits[layer]:
        return None
    return float(limits[layer])


class LazyExpert(torch.nn.Module):
    def __init__(self, gate, up, down, limit):
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.limit = limit

    def forward(self, x):
        hidden = swiglu(torch.nn.functional.linear(x, self.gate), torch.nn.functional.linear(x, self.up), self.limit)
        return torch.nn.functional.linear(hidden, self.down)


class LimitedSharedExpert(torch.nn.Module):
    def __init__(self, mlp, limit):
        super().__init__()
        self.mlp = mlp
        self.limit = limit

    def forward(self, x):
        return self.mlp.down_proj(swiglu(self.mlp.gate_proj(x), self.mlp.up_proj(x), self.limit))


class ExpertCache:
    def __init__(self, reader, device, capacity, limits):
        self.reader = reader
        self.device = device
        self.capacity = capacity
        self.limits = limits
        self.entries = collections.OrderedDict()
        self.misses = 0
        self.hits = 0

    def fetch(self, layer, expert):
        key = (layer, expert)
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        tensors = [self.reader.get(prefix + part + ".weight").to(self.device) for part in ("gate_proj", "up_proj", "down_proj")]
        module = LazyExpert(*tensors, layer_limit(self.limits, layer))
        self.entries[key] = module
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return module


class LazyExperts(torch.nn.Module):
    def __init__(self, cache, layer, count):
        super().__init__()
        self.cache = cache
        self.layer = layer
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.cache.fetch(self.layer, int(index))


def place_tensor(model, name, tensor):
    path = name.split(".")
    module = model
    for part in path[:-1]:
        module = getattr(module, part)
    leaf = path[-1]
    if leaf in module._parameters:
        expected = module._parameters[leaf]
        if tuple(expected.shape) != tuple(tensor.shape):
            raise SystemExit(f"{name}: checkpoint shape {tuple(tensor.shape)} != module shape {tuple(expected.shape)}")
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor
    else:
        raise SystemExit(f"{name}: no parameter or buffer in the publisher module")


def build_model(checkpoint, device, expert_capacity, swiglu_limits):
    configuration, modeling_module = load_publisher_modules(checkpoint)
    config = configuration.BailingMoeV3Config.from_pretrained(checkpoint)
    config.num_nextn_predict_layers = 0
    if config.rope_scaling is not None and config.rope_scaling.get("rope_type") != "default":
        raise SystemExit(f"unexpected rope scaling {config.rope_scaling}")
    config.rope_scaling = None
    if config.rope_theta != 6000000:
        raise SystemExit(f"rope_theta {config.rope_theta} lost in config normalisation")
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = modeling_module.BailingMoeV3ForCausalLM(config)
    reader = ShardReader(checkpoint)
    serving = swiglu_limits == "serving"
    expert_limits = getattr(config, "expert_swiglu_limit_list", None) if serving else None
    shared_limits = getattr(config, "share_expert_swiglu_limit_list", None) if serving else None
    cache = ExpertCache(reader, device, expert_capacity, expert_limits)
    limited = []
    for layer_index, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, modeling_module.BailingMoeV3SparseMoeBlock):
            layer.mlp.experts = LazyExperts(cache, layer_index, config.num_experts)
            if layer_limit(expert_limits, layer_index) is not None or layer_limit(shared_limits, layer_index) is not None:
                limited.append(layer_index)
    placed = 0
    for name in reader.weight_map:
        if ".mlp.experts." in name:
            continue
        if name.startswith(f"model.layers.{config.num_hidden_layers}."):
            continue
        place_tensor(model, name, reader.get(name).to(device))
        placed += 1
    for layer_index in limited:
        block = model.model.layers[layer_index].mlp
        limit = layer_limit(shared_limits, layer_index)
        if limit is not None:
            if getattr(block, "shared_experts", None) is None:
                raise SystemExit(f"layer {layer_index}: shared expert limit {limit} without a shared expert")
            block.shared_experts = LimitedSharedExpert(block.shared_experts, limit)
    model.model.rotary_emb = modeling_module.BailingMoeV3RotaryEmbedding(config=config, device=device)
    missing = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if missing:
        raise SystemExit(f"unplaced tensors: {missing[:8]} (+{max(0, len(missing) - 8)})")
    model.eval()
    return model, config, cache, placed, limited


@torch.no_grad()
def greedy(model, prompt_ids, new_tokens, device, top):
    past = None
    ids = list(prompt_ids)
    feed = torch.tensor([ids], device=device)
    position = 0
    steps = []
    for _ in range(new_tokens):
        length = feed.shape[1]
        positions = torch.arange(position, position + length, device=device)
        output = model(input_ids=feed, past_key_values=past, use_cache=True,
                       cache_position=positions, position_ids=positions.unsqueeze(0))
        past = output.past_key_values
        logits = output.logits[0, -1].float()
        values, indices = torch.topk(logits, top)
        token = int(indices[0])
        steps.append({"token": token, "top": [[int(i), float(v)] for i, v in zip(indices.tolist(), values.tolist())]})
        position += length
        ids.append(token)
        feed = torch.tensor([[token]], device=device)
    return ids[len(prompt_ids):], steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--new-tokens", type=int, required=True)
    parser.add_argument("--expert-cache", type=int, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--swiglu-limits", choices=SWIGLU_LIMIT_MODES, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(os.path.join(args.checkpoint, "tokenizer.json"))
    with open(args.prompts) as handle:
        prompts = json.load(handle)["prompts"]
    started = time.time()
    model, config, cache, placed, limited = build_model(args.checkpoint, device, args.expert_cache, args.swiglu_limits)
    print(f"placed {placed} non-expert tensors in {time.time() - started:.1f}s; swiglu limits {args.swiglu_limits} on layers {limited}", flush=True)
    results = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt["text"], add_special_tokens=False).ids
        if "prompt_token_ids" in prompt and prompt["prompt_token_ids"] != prompt_ids:
            raise SystemExit(f"{prompt['name']}: tokenizer ids {prompt_ids} != pinned {prompt['prompt_token_ids']}")
        began = time.time()
        tokens, steps = greedy(model, prompt_ids, args.new_tokens, device, args.top)
        text = tokenizer.decode(tokens)
        print(f"{prompt['name']}: {tokens} {text!r} {time.time() - began:.1f}s", flush=True)
        results.append({"name": prompt["name"], "text": prompt["text"], "prompt_token_ids": prompt_ids,
                        "tokens": tokens, "completion": text, "steps": steps})
    receipt = {
        "engine": "publisher modeling_bailing_moe_v3.py with fla kernels, lazy per-expert loading",
        "swiglu_limits": args.swiglu_limits,
        "swiglu_limited_layers": limited,
        "checkpoint": args.checkpoint,
        "config_sha256": sha256_file(os.path.join(args.checkpoint, "config.json")),
        "index_sha256": sha256_file(os.path.join(args.checkpoint, "model.safetensors.index.json")),
        "modeling_sha256": MODELING_SHA256,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "fla": __import__("fla").__version__,
        "new_tokens": args.new_tokens,
        "expert_loads": cache.misses,
        "expert_hits": cache.hits,
        "results": results,
    }
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=1)
    print(f"wrote {args.out} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
