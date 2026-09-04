#!/usr/bin/env python3
import argparse
import os

import torch
from transformers import AutoModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-id", action="append", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def capture_layer_input(captures, layer_index):
    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        captures[(layer_index, "layer_input")] = hidden.detach().clone()
    return capture


def capture_output(captures, layer_index, phase_name):
    def capture(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captures[(layer_index, phase_name)] = value.detach().clone()
    return capture


def capture_expert_routing(captures, layer_index):
    def capture(module, args, kwargs):
        captures[(layer_index, "topk_indices")] = args[1].detach().clone()
        captures[(layer_index, "topk_weights")] = args[2].detach().clone()
    return capture


def register_hooks(model, captures):
    hooks = []
    for layer_index, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_pre_hook(
            capture_layer_input(captures, layer_index),
            with_kwargs=True))
        hooks.append(layer.self_attn.register_forward_hook(
            capture_output(
                captures,
                layer_index,
                "attention_projected")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            capture_output(
                captures,
                layer_index,
                "post_attention_normalized")))
        hooks.append(layer.mlp.register_forward_hook(
            capture_output(
                captures,
                layer_index,
                "moe_output")))
        if hasattr(layer.mlp, "experts"):
            hooks.append(layer.mlp.experts.register_forward_pre_hook(
                capture_expert_routing(captures, layer_index),
                with_kwargs=True))
            hooks.append(layer.mlp.experts.register_forward_hook(
                capture_output(
                    captures,
                    layer_index,
                    "routed_moe_output")))
            hooks.append(layer.mlp.shared_experts.register_forward_hook(
                capture_output(
                    captures,
                    layer_index,
                    "shared_moe_output")))
        hooks.append(layer.register_forward_hook(
            capture_output(
                captures,
                layer_index,
                "layer_output")))
    return hooks


def write_bf16(path, tensor):
    data = tensor.to(torch.bfloat16).cpu().contiguous().view(torch.uint16)
    with open(path, "wb") as output_file:
        output_file.write(data.numpy().tobytes())


def write_tensor(path, tensor, dtype):
    data = tensor.to(dtype).cpu().contiguous()
    with open(path, "wb") as output_file:
        output_file.write(data.numpy().tobytes())


def token_row(tensor, token_index):
    if tensor.ndim == 2:
        return tensor[token_index]
    return tensor[0, token_index]


def write_layer(output_dir, captures, layer_index, token_count):
    layer_input = captures[(layer_index, "layer_input")]
    attention = captures[(layer_index, "attention_projected")]
    captures[(layer_index, "post_attention")] = layer_input + attention
    write_bf16(
        os.path.join(output_dir, f"after_layer_{layer_index}.bf16"),
        captures[(layer_index, "layer_output")][0])
    for token_index in range(token_count):
        phase_names = [
            "attention_projected",
            "post_attention",
            "post_attention_normalized",
            "moe_output"]
        if (layer_index, "routed_moe_output") in captures:
            phase_names.extend(["routed_moe_output", "shared_moe_output"])
        for phase_name in phase_names:
            write_bf16(
                os.path.join(
                    output_dir,
                    f"token_{token_index:04d}_layer_{layer_index:04d}_{phase_name}.bf16"),
                token_row(captures[(layer_index, phase_name)], token_index))
        if (layer_index, "topk_indices") in captures:
            write_tensor(
                os.path.join(
                    output_dir,
                    f"token_{token_index:04d}_layer_{layer_index:04d}_topk_expert_ids.i32"),
                token_row(captures[(layer_index, "topk_indices")], token_index),
                torch.int32)
            write_tensor(
                os.path.join(
                    output_dir,
                    f"token_{token_index:04d}_layer_{layer_index:04d}_topk_weights.f32"),
                token_row(captures[(layer_index, "topk_weights")], token_index),
                torch.float32)


def main():
    arguments = parse_arguments()
    os.makedirs(arguments.output_dir, exist_ok=True)
    model = AutoModel.from_pretrained(
        arguments.model,
        dtype=torch.bfloat16,
        device_map=arguments.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True)
    captures = {}
    hooks = register_hooks(model, captures)
    token_ids = torch.tensor(
        [arguments.token_id],
        dtype=torch.long,
        device=arguments.device)
    model.eval()
    with torch.inference_mode():
        model(input_ids=token_ids, use_cache=False, return_dict=True)
    for hook in hooks:
        hook.remove()
    for layer_index in range(len(model.layers)):
        write_layer(
            arguments.output_dir,
            captures,
            layer_index,
            len(arguments.token_id))
    print(
        f"glm52_transformers_stage_reference layers={len(model.layers)} "
        f"tokens={len(arguments.token_id)} output={arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
