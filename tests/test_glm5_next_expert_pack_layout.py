#!/usr/bin/env python3
"""Check the packer's expert slabs against separate driver payload/scale planes."""
import io
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack


class Source:
    weight_map = {"model.language_model.layers.3.mlp.experts.0.up_proj.weight": "fixture"}

    def __init__(self, bf16):
        self.bf16 = bf16

    def meta(self, name):
        return ("BF16" if self.bf16 else "F8_E4M3", ())

    def value(self, name):
        expert = int(name.split(".experts.")[1].split(".")[0])
        return 10 + expert * 10 + (1 if ".gate_proj." in name else 2 if ".down_proj." in name else 0)

    def expert_payload(self, name, r0, r1, c0, c1):
        return bytes([self.value(name)]) * ((r1 - r0) * (c1 - c0) * (2 if self.bf16 else 1))

    def expert_scale(self, name, r0, r1, c0, c1):
        assert not self.bf16
        return bytes([self.value(name) + 100]) * ((r1 - r0) * ((c1 - c0) // 128) * 4)


def main():
    pack.EXPERTS, pack.HIDDEN, pack.EXPERT_INTER = 2, 128, 2048
    for bf16 in (False, True):
        for degree in (1, 4, 16):
            for rank in range(degree):
                builder = pack.Packer(Source(bf16), degree, rank, 3, 1, False, False, False)
                builder.add_experts(3)
                for item in builder.plan:
                    entry = item.entry
                    weights = b"".join(item.produce_payload())
                    scales = b"".join(item.produce_scale())
                    assert len(weights) == entry.payload_bytes > 0
                    assert len(scales) == entry.scale_bytes
                    # Interpret the same separate per-expert planes the driver uses.
                    for expert in range(2):
                        per = len(weights) // 2
                        slab = weights[expert * per:(expert + 1) * per]
                        base = 10 + expert * 10
                        if entry.kind == pack.K_EXPERT_DOWN:
                            expected = bytes([base + 2]) * per
                        elif degree == 1:
                            expected = bytes([base]) * (per // 2) + bytes([base + 1]) * (per // 2)
                        else:
                            expected = bytes([base + int(rank >= degree // 2)]) * per
                        assert slab == expected
                        if not bf16:
                            scale_per = len(scales) // 2
                            expected_scale = bytes([value + 100 for value in expected[::128]]) * 4
                            # Same value order; F32 scale fixtures use four identical bytes.
                            if degree == 1 and entry.kind == pack.K_EXPERT_UP_GATE:
                                expected_scale = bytes([base + 100]) * (scale_per // 2) + bytes([base + 101]) * (scale_per // 2)
                            assert scales[expert * scale_per:(expert + 1) * scale_per] == expected_scale
                    output = io.BytesIO()
                    pack.emit_region(output, 0, len(weights), iter([weights]))
                    pack.emit_region(output, len(weights), len(scales), iter([scales]))
                    assert output.getvalue() == weights + scales
                if degree == 4 and rank == 0:
                    builder.build = lambda: None
                    header = dict(stage_count=1, stage_index=0, first_layer=3, layer_count=1, flags=0)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "pack.sp"
                        pack.emit(builder, path, header)
                        original = path.read_bytes()
                        try:
                            pack.emit(builder, path, header)
                        except pack.PackFailure:
                            pass
                        else:
                            raise AssertionError("existing artifact overwritten")
                        assert path.read_bytes() == original
                        path.unlink()
                        builder.plan[0].produce_payload = lambda: iter([b"x"])
                        try:
                            pack.emit(builder, path, header)
                        except pack.PackFailure:
                            pass
                        else:
                            raise AssertionError("partial pack published")
                        assert not path.exists() and list(Path(directory).iterdir()) == []
    for chunks, expected in (([b"1234"], 3), ([b"12"], 3)):
        try:
            pack.emit_region(io.BytesIO(), 0, expected, iter(chunks))
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("incorrect region size accepted")
    print("PASS expert planes: FP8/BF16, TP1/TP4/TP16 all ranks, region bounds")


if __name__ == "__main__":
    main()
