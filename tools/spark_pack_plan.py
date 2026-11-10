"""Plan builder for codec-ABI stage packs cut from memory-mapped checkpoints.

MappedCheckpoint reads safetensors payloads through np.memmap views and
decodes spine tensors to bf16 once, keeping a byte-capped LRU so a lockstep
all-rank writer decodes each tensor a single time: BF16 passes through, F32
narrows round-to-nearest-even, F8_E4M3 dequantises through its 128 x 128 F32
weight_scale_inv blocks and packed NVFP4 decodes through its per-16 E4M3 block
scales and F32 global. Routed experts are never decoded: their payload and
scale windows are copied byte for byte.

PlanBuilder turns tensors into directory entries with lazy producers. A
tensor-parallel rank takes whole rows or whole columns of a sharded tensor;
fused tensors shard every row section separately so each rank keeps whole
heads of every projection; the latent kv_b projection splits into per-head
transposed keys and values; routed experts take one row window (gate and up)
or column window (down) of every expert.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from spark_pack_codecs import (CODEC_BF16, CODEC_NONE, CODEC_NVFP4, E4M3_FINITE_LUT, E4M3_LUT, FP8_BLOCK,
                               NVFP4_GROUP, PAYLOAD_BF16, PAYLOAD_F32, PAYLOAD_PACKED_WEIGHT, SCALE_F32, SCALE_NONE,
                               SCALE_UE4M3_F32_GLOBAL, bf16_to_f32, f32_to_bf16, fp8_block_dequant_bf16,
                               nvfp4_to_bf16)
from spark_pack_common import PackFailure, SafetensorsSource, tp_shard_range
from spark_pack_wire import Entry


def to_bytes(array) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def matrix_shape(shape) -> Tuple[int, int]:
    shape = tuple(shape)
    if len(shape) == 3 and shape[1] == 1:
        shape = (shape[0], shape[2])
    if len(shape) == 1:
        shape = (1, shape[0])
    if len(shape) != 2:
        raise PackFailure(f"expected a 2-D tensor, got {shape}")
    return shape


class MappedCheckpoint(SafetensorsSource):
    spine_dtypes = ("BF16", "F32", "F8_E4M3", "U8")
    quiet_nan = False

    def __init__(self, model_dir: Path, cache_byte_cap: int, config_required: bool = True):
        super().__init__(Path(model_dir), config_required=config_required)
        self._mmaps = {}
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_bytes = 0
        self._cache_byte_cap = cache_byte_cap

    def map_name(self, name: str) -> str:
        return name

    def has(self, name: str) -> bool:
        return self.map_name(name) in self.weight_map

    def mmap(self, shard: str) -> np.ndarray:
        if shard not in self._mmaps:
            path = self.root / shard
            self._mmaps[shard] = np.memmap(path, dtype=np.uint8, mode="r", shape=(path.stat().st_size,))
        return self._mmaps[shard]

    def meta(self, name: str) -> Tuple[str, Tuple[int, ...], str]:
        name = self.map_name(name)
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"missing tensor in index: {name}")
        entry = self.shard_header(shard).get(name)
        if entry is None:
            raise PackFailure(f"tensor {name} not in shard {shard}")
        return entry["dtype"], tuple(entry["shape"]), shard

    def raw(self, name: str) -> np.ndarray:
        self.meta(name)
        shard, meta, offset = self.resolve(self.map_name(name))
        length = meta["data_offsets"][1] - meta["data_offsets"][0]
        view = self.mmap(shard)[offset:offset + length]
        if view.shape[0] != length:
            raise PackFailure(f"short payload for {name}")
        return view

    def remember(self, name: str, matrix: np.ndarray) -> np.ndarray:
        self._cache[name] = matrix
        self._cache_bytes += matrix.nbytes
        while self._cache_bytes > self._cache_byte_cap and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
        return matrix

    def spine_bf16(self, name: str) -> np.ndarray:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        dtype, shape, _ = self.meta(name)
        if dtype not in self.spine_dtypes:
            raise PackFailure(f"{name}: unexpected spine dtype {dtype}")
        rows, columns = matrix_shape(shape)
        if dtype == "BF16":
            matrix = self.raw(name).view(np.uint16).reshape(rows, columns)
        elif dtype == "F32":
            matrix = f32_to_bf16(self.raw(name).view(np.float32).reshape(rows, columns))
        elif dtype == "F8_E4M3":
            matrix = self.fp8_spine(name, rows, columns)
        else:
            matrix = self.nvfp4_spine(name, rows, columns)
        return self.remember(name, matrix)

    def fp8_spine(self, name: str, rows: int, columns: int) -> np.ndarray:
        scale_name = name + "_scale_inv"
        scale_dtype, scale_shape, _ = self.meta(scale_name)
        if scale_dtype != "F32":
            raise PackFailure(f"{scale_name}: dtype {scale_dtype}, expected F32")
        expected = ((rows + FP8_BLOCK - 1) // FP8_BLOCK, (columns + FP8_BLOCK - 1) // FP8_BLOCK)
        if scale_shape != expected:
            raise PackFailure(f"{scale_name}: shape {scale_shape}, expected {expected}")
        codes = self.raw(name).reshape(rows, columns)
        scale = self.raw(scale_name).view(np.float32).reshape(scale_shape)
        return fp8_block_dequant_bf16(codes, scale, E4M3_LUT, self.quiet_nan)

    def nvfp4_spine(self, name: str, rows: int, columns: int) -> np.ndarray:
        scale_name = name + "_scale"
        _, scale_shape, _ = self.meta(scale_name)
        width = columns * 2
        if tuple(scale_shape) != (rows, width // NVFP4_GROUP):
            raise PackFailure(f"{scale_name}: shape {scale_shape}, expected {(rows, width // NVFP4_GROUP)}")
        global_name = name + "_scale_2"
        self.meta(global_name)
        global_scale = float(self.raw(global_name).view(np.float32)[0])
        packed = self.raw(name).reshape(rows, columns)
        blocks = E4M3_FINITE_LUT[self.raw(scale_name).view(np.uint8).reshape(rows, width // NVFP4_GROUP)]
        return nvfp4_to_bf16(packed, blocks, global_scale, scale_first=True)

    def f32_vector(self, name: str) -> np.ndarray:
        dtype = self.meta(name)[0]
        if dtype == "F32":
            return self.raw(name).view(np.float32).copy()
        if dtype == "BF16":
            return bf16_to_f32(self.raw(name).view(np.uint16))
        raise PackFailure(f"{name}: f32 upcast from {dtype} unsupported")

    def spine_f32(self, name: str) -> np.ndarray:
        return self.f32_vector(name).reshape(matrix_shape(self.meta(name)[1]))

    def window(self, name: str, element, r0: int, r1: int, c0: int, c1: int) -> bytes:
        _, shape, _ = self.meta(name)
        if len(shape) != 2:
            raise PackFailure(f"{name}: expected a 2-D tensor, got {shape}")
        return to_bytes(self.raw(name).view(element).reshape(shape[0], shape[1])[r0:r1, c0:c1])

    def expert_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        dtype = self.meta(name)[0]
        if dtype not in ("BF16", "F8_E4M3"):
            raise PackFailure(f"{name}: expected F8_E4M3, got {dtype}")
        return self.window(name, np.uint16 if dtype == "BF16" else np.uint8, r0, r1, c0, c1)

    def expert_scale(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        if self.meta(name)[0] == "BF16":
            return b""
        scale_name = name + "_scale_inv"
        _, scale_shape, _ = self.meta(scale_name)
        scale = self.raw(scale_name).view(np.float32).reshape(scale_shape)
        return to_bytes(np.repeat(scale, FP8_BLOCK, axis=0)[r0:r1, c0 // FP8_BLOCK:c1 // FP8_BLOCK])

    def nvfp4_payload(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        packed = name + "_packed" if self.has(name + "_packed") else name
        dtype, shape, _ = self.meta(packed)
        if dtype != "U8" or len(shape) != 2:
            raise PackFailure(f"{packed}: expected 2-D U8 (packed e2m1), got {dtype} {shape}")
        if c0 % 2 != 0 or c1 % 2 != 0:
            raise PackFailure(f"{packed}: nvfp4 column slice [{c0}:{c1}] not nibble aligned")
        return self.window(packed, np.uint8, r0, r1, c0 // 2, c1 // 2)

    def nvfp4_block_scale(self, name: str, r0: int, r1: int, c0: int, c1: int) -> bytes:
        plane = name + "_scale"
        dtype, shape, _ = self.meta(plane)
        if dtype != "F8_E4M3" or len(shape) != 2:
            raise PackFailure(f"{plane}: expected 2-D F8_E4M3, got {dtype} {shape}")
        return self.window(plane, np.uint8, r0, r1, c0 // NVFP4_GROUP, c1 // NVFP4_GROUP)

    def nvfp4_weight_global(self, name: str) -> bytes:
        for global_name in (name + "_global_scale", name + "_scale_2"):
            if self.has(global_name):
                dtype, shape, _ = self.meta(global_name)
                if dtype != "F32" or tuple(shape) not in ((), (1,)):
                    raise PackFailure(f"{global_name}: expected scalar F32, got {dtype} {shape}")
                return to_bytes(self.raw(global_name).view(np.float32))
        raise PackFailure(f"{name}: no nvfp4 weight global scale in the source")

    def close(self) -> None:
        self._mmaps.clear()
        self._cache.clear()
        self._cache_bytes = 0


class PlanItem:
    def __init__(self, entry: Entry, produce_payload: Optional[Callable[[], Iterator[bytes]]],
                 produce_scale: Optional[Callable[[], Iterator[bytes]]] = None,
                 payload_bytes: Optional[int] = None, scale_bytes: Optional[int] = None,
                 sources: Optional[List[str]] = None):
        self.entry = entry
        self.produce_payload = produce_payload
        self.produce_scale = produce_scale
        if payload_bytes is not None:
            entry.payload_bytes = payload_bytes
        if scale_bytes is not None:
            entry.scale_bytes = scale_bytes
        self.sources = sources or []
        self.expert_digests: Optional[List[bytes]] = None
        self.expert_scale_digests: Optional[List[bytes]] = None


class PlanBuilder:
    def __init__(self, source, tp_degree: int, tp_rank: int):
        self.source = source
        self.tp_degree = tp_degree
        self.tp_rank = tp_rank
        self.plan: List[PlanItem] = []

    def slice_range(self, extent: int) -> Tuple[int, int]:
        return tp_shard_range(extent, self.tp_degree, self.tp_rank)

    def track(self, *names: str) -> None:
        pass

    def add_item(self, entry: Entry, produce_payload, produce_scale=None, payload_bytes: Optional[int] = None,
                 scale_bytes: Optional[int] = None, sources: Sequence[str] = ()) -> PlanItem:
        item = PlanItem(entry, produce_payload, produce_scale, payload_bytes, scale_bytes, list(sources))
        self.plan.append(item)
        return item

    def add_bf16(self, kind: int, layer: int, name: str, shard: str = "", dtypes=None, shape=None) -> PlanItem:
        dtype, source_shape, _ = self.source.meta(name)
        rows, columns = matrix_shape(source_shape)
        if shape is not None and (rows, columns) != tuple(shape):
            raise PackFailure(f"{name}: shape {(rows, columns)}, expected {tuple(shape)}")
        if dtypes is not None and dtype not in dtypes:
            raise PackFailure(f"{name}: spine dtype {dtype}")
        if dtype == "U8":
            columns *= 2
        sharded = shard if self.tp_degree > 1 else ""
        start = 0
        if sharded == "rows":
            start, rows = self.slice_range(rows)
        elif sharded == "cols":
            start, columns = self.slice_range(columns)
        expected = rows * columns * 2
        source = self.source

        def produce() -> Iterator[bytes]:
            matrix = source.spine_bf16(name)
            if sharded == "rows":
                matrix = matrix[start:start + rows, :]
            elif sharded == "cols":
                matrix = matrix[:, start:start + columns]
            blob = to_bytes(matrix)
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.track(name)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows, columns)
        return self.add_item(entry, produce, payload_bytes=expected, sources=[name])

    def add_f32(self, kind: int, layer: int, name: str, dtypes=None) -> PlanItem:
        dtype, shape, _ = self.source.meta(name)
        if dtypes is not None and dtype not in dtypes:
            raise PackFailure(f"{name}: dtype {dtype}, expected {'/'.join(dtypes)}")
        rows, columns = matrix_shape(shape)
        expected = rows * columns * 4
        source = self.source

        def produce() -> Iterator[bytes]:
            blob = to_bytes(source.spine_f32(name))
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.track(name)
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, rows, columns)
        return self.add_item(entry, produce, payload_bytes=expected, sources=[name])

    def add_f32_vector(self, kind: int, layer: int, name: str, shard: bool = False) -> PlanItem:
        count = int(np.prod(self.source.meta(name)[1], dtype=np.int64))
        start = 0
        sharded = shard and self.tp_degree > 1
        if sharded:
            start, count = self.slice_range(count)
        expected = count * 4
        source = self.source

        def produce() -> Iterator[bytes]:
            vector = source.f32_vector(name).reshape(-1)
            blob = to_bytes(vector[start:start + count] if sharded else vector)
            if len(blob) != expected:
                raise PackFailure(f"{name}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.track(name)
        entry = Entry(kind, layer, PAYLOAD_F32, CODEC_NONE, SCALE_NONE, 1, 1, count)
        return self.add_item(entry, produce, payload_bytes=expected, sources=[name])

    def add_fused_rows(self, kind: int, layer: int, names: Sequence[str], section_rows: Sequence[int],
                       columns: int, shard: bool = False, sources: Optional[Sequence[str]] = None) -> PlanItem:
        slices = [self.slice_range(rows) for rows in section_rows] if shard and self.tp_degree > 1 else None
        rows_out = sum(count for _, count in slices) if slices else sum(section_rows)
        expected = rows_out * columns * 2
        source = self.source

        def produce() -> Iterator[bytes]:
            matrices = [source.spine_bf16(name) for name in names]
            if slices:
                matrices = [matrix[start:start + count, :] for matrix, (start, count) in zip(matrices, slices)]
            blob = to_bytes(np.concatenate(matrices, axis=0))
            if len(blob) != expected:
                raise PackFailure(f"fused {list(names)}: {len(blob)} bytes, planned {expected}")
            yield blob

        self.track(*names)
        entry = Entry(kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, 1, rows_out, columns)
        return self.add_item(entry, produce, payload_bytes=expected,
                             sources=list(names) if sources is None else list(sources))

    def add_kv_b(self, key_kind: int, value_kind: int, layer: int, name: str, heads: int, nope: int,
                 value_dim: int, latent: int, dtypes=None) -> None:
        dtype, shape, _ = self.source.meta(name)
        if dtypes is not None and dtype not in dtypes:
            raise PackFailure(f"{name}: dtype {dtype}, expected {'/'.join(dtypes)}")
        if tuple(shape) != (heads * (nope + value_dim), latent):
            raise PackFailure(f"kv_b shape {tuple(shape)} != ({heads * (nope + value_dim)}, {latent})")
        key_bytes = heads * latent * nope * 2
        value_bytes = heads * value_dim * latent * 2
        source = self.source

        def per_head() -> np.ndarray:
            return source.spine_bf16(name).reshape(heads, nope + value_dim, latent)

        def produce_key() -> Iterator[bytes]:
            blob = to_bytes(per_head()[:, :nope, :].transpose(0, 2, 1))
            if len(blob) != key_bytes:
                raise PackFailure(f"{name}: key transpose size")
            yield blob

        def produce_value() -> Iterator[bytes]:
            blob = to_bytes(per_head()[:, nope:, :])
            if len(blob) != value_bytes:
                raise PackFailure(f"{name}: value size")
            yield blob

        self.track(name)
        self.add_item(Entry(key_kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, heads, latent, nope),
                      produce_key, payload_bytes=key_bytes, sources=[name])
        self.add_item(Entry(value_kind, layer, PAYLOAD_BF16, CODEC_BF16, SCALE_NONE, heads, value_dim, latent),
                      produce_value, payload_bytes=value_bytes, sources=[name])

    def add_routed_experts(self, kind: int, layer: int, expert_names: Sequence[Sequence[str]], rows: int,
                           columns: int, row_window: Tuple[int, int], column_window: Tuple[int, int],
                           codec: int, bf16_payload_type: int = PAYLOAD_PACKED_WEIGHT,
                           check_shapes: bool = False, payload_method: str = "expert_payload") -> PlanItem:
        r0, r1 = row_window
        c0, c1 = column_window
        shard_rows, shard_columns = r1 - r0, c1 - c0
        experts = len(expert_names)
        projections = len(expert_names[0]) if expert_names else 0
        names = [name for group in expert_names for name in group]
        source = self.source

        def check(name: str, width: int, dtype: Optional[str]) -> None:
            got_dtype, shape = source.meta(name)[:2]
            if check_shapes and tuple(shape) != (rows, width):
                raise PackFailure(f"{name} shape {tuple(shape)} != ({rows}, {width})")
            if dtype is not None and got_dtype != dtype:
                raise PackFailure(f"{name}: expected {dtype}, got {got_dtype}")

        if codec == CODEC_NVFP4:
            entry = Entry(kind, layer, PAYLOAD_PACKED_WEIGHT, CODEC_NVFP4, SCALE_UE4M3_F32_GLOBAL, experts,
                          projections * shard_rows, shard_columns)
            payload_bytes = experts * projections * shard_rows * (shard_columns // 2)
            scale_bytes = experts * 4 + experts * projections * shard_rows * (shard_columns // NVFP4_GROUP)

            def produce_payload() -> Iterator[bytes]:
                for name in names:
                    if check_shapes:
                        check(name, columns // 2, None)
                    yield source.nvfp4_payload(name, r0, r1, c0, c1)

            def produce_scale() -> Iterator[bytes]:
                for group in expert_names:
                    scales = [source.nvfp4_weight_global(name) for name in group]
                    if any(scale != scales[0] for scale in scales[1:]):
                        raise PackFailure(f"layer {layer}: the weight global scales of {list(group)} disagree; "
                                          f"the fused-entry global would be inexact")
                    yield scales[0]
                for name in names:
                    yield source.nvfp4_block_scale(name, r0, r1, c0, c1)
        else:
            bf16 = codec == CODEC_BF16
            entry = Entry(kind, layer, bf16_payload_type if bf16 else PAYLOAD_PACKED_WEIGHT, codec,
                          SCALE_NONE if bf16 else SCALE_F32, experts, projections * shard_rows, shard_columns)
            payload_bytes = experts * projections * shard_rows * shard_columns * (2 if bf16 else 1)
            scale_bytes = 0 if bf16 else experts * projections * shard_rows * (shard_columns // FP8_BLOCK) * 4
            dtype = "BF16" if bf16 else "F8_E4M3"

            def produce_payload() -> Iterator[bytes]:
                for name in names:
                    check(name, columns, dtype)
                    yield getattr(source, payload_method)(name, r0, r1, c0, c1)

            def produce_scale() -> Iterator[bytes]:
                for name in names if not bf16 else ():
                    yield source.expert_scale(name, r0, r1, c0, c1)
        return self.add_item(entry, produce_payload, produce_scale, payload_bytes, scale_bytes, names)
