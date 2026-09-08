#!/usr/bin/env python3
"""Check the packer's expert slabs against separate driver payload/scale planes."""
import io
import struct
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack
import glm5_next_pack_verify as verify


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
    for stage, first, count in ((0, 0, 12), (1, 12, 11), (2, 23, 11), (3, 34, 11)):
        pack.validate_stage(4, stage, first, count, stage == 0, stage == 3, False)
    for values in ((0, 0, 0, 45, False, False, False),
                   (4, 4, 0, 12, False, False, False),
                   (4, 1, 12, 34, False, False, False),
                   (4, 1, 12, 11, True, False, False),
                   (4, 0, 0, 12, False, True, False),
                   (4, 0, 0, 12, False, False, True)):
        try:
            pack.validate_stage(*values)
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("invalid pipeline metadata accepted")
    assert pack.stage_pack_name(16, 2, 1, 0) == "glm5_next_stage.tp16.rank2.g5nsp"
    assert len({pack.stage_pack_name(4, rank, 4, stage)
                for rank in range(4) for stage in range(4)}) == 16
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
                    header = dict(stage_count=4, stage_index=1, first_layer=3, layer_count=1, flags=0)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "pack.sp"
                        pack.emit(builder, path, header)
                        original = path.read_bytes()
                        assert struct.unpack_from("<4I", original, 7 * 4) == (4, 1, 3, 1)
                        source = type("VerifySource", (), {"close": lambda self: None})()
                        argv = ["verify", "--pack", str(path), "--source", "fixture",
                                "--tp-degree", "4", "--tp-rank", "0", "--stage-count", "4",
                                "--stage-index", "1", "--first-layer", "3", "--layer-count", "1",
                                "--expected-bytes", str(len(original)), "--all-tensors"]
                        with patch.object(sys, "argv", argv), patch.object(verify, "SourceReader", return_value=source), patch.object(verify, "Packer", return_value=builder) as factory:
                            assert verify.main() == 0
                            factory.assert_called_once_with(source, 4, 0, 3, 1, False, False, False)
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
