from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from glm52_model_contract import load_model_contract


MODEL_CONTRACT = load_model_contract()
HIDDEN_DIMENSION = MODEL_CONTRACT["hidden_dimension"]
INTERMEDIATE_DIMENSION = MODEL_CONTRACT["moe_intermediate_dimension"]
EXPERT_COUNT = MODEL_CONTRACT["moe_expert_count"]
TOP_K = MODEL_CONTRACT["moe_top_k"]
W1_COMPONENT_COUNT = MODEL_CONTRACT["moe_w1_component_count"]


class PackFailure(RuntimeError):
    pass


def import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise PackFailure("torch is required to build GLM-5.2 resident packs") from error
    return torch


class SafetensorReader:
    def __init__(self, model_dir: Path) -> None:
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise PackFailure("safetensors is required to build GLM-5.2 resident packs") from error
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise PackFailure(f"missing safetensors index: {index_path}")
        self.model_dir = model_dir
        self.safe_open = safe_open
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.handles: Dict[str, Any] = {}

    def tensor(self, name: str) -> Any:
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        if shard not in self.handles:
            path = self.model_dir / shard
            if not path.is_file():
                raise PackFailure(f"missing safetensors shard: {path}")
            self.handles[shard] = self.safe_open(str(path), framework="pt", device="cpu")
        handle = self.handles[shard]
        if name not in handle.keys():
            raise PackFailure(f"missing tensor in shard {shard}: {name}")
        return handle.get_tensor(name)

    def close(self) -> None:
        self.handles.clear()


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tensor_name(layer: int, expert: int, projection: str, suffix: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"


def parse_layers(value: str) -> List[int]:
    layers: List[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise PackFailure("no layers selected")
    if len(set(layers)) != len(layers):
        raise PackFailure("duplicate layer selection")
    return layers


def tp_shard_range(dimension: int, tp_degree: int, tp_rank: int, block: int = 1) -> Tuple[int, int]:
    """This rank's [start, start+count) slice of a dimension split across a TP
    group, with count required to be a whole number of quantization blocks so
    scale tensors slice on block boundaries. Fails closed on any misalignment
    rather than producing a shard whose scales cannot be represented."""
    if tp_degree < 1 or tp_rank < 0 or tp_rank >= tp_degree:
        raise PackFailure(f"invalid tp shard {tp_rank}/{tp_degree}")
    if dimension % tp_degree != 0:
        raise PackFailure(f"dimension {dimension} not divisible by tp degree {tp_degree}")
    count = dimension // tp_degree
    if block > 1 and count % block != 0:
        raise PackFailure(
            f"tp shard extent {count} not aligned to quantization block {block}")
    return tp_rank * count, count
