import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack
import glm5_next_pack_verify as verify

HEADER_EXPERT_CODEC_OFFSET = 16 * 4
PREFIX = "model.language_model.layers.{}.mlp.experts.{}.{}_proj.weight"


class Source:
    def __init__(self, layer_dtypes):
        self.layer_dtypes = layer_dtypes
        self.weight_map = {}
        for layer, dtype in layer_dtypes.items():
            for expert in range(pack.EXPERTS):
                for projection in ("up", "gate", "down"):
                    name = PREFIX.format(layer, expert, projection)
                    self.weight_map[name] = "fixture"
                    if dtype == "U8":
                        self.weight_map[name + "_scale"] = "fixture"
                        self.weight_map[name + "_scale_2"] = "fixture"

    def meta(self, name):
        layer = int(name.split(".layers.")[1].split(".")[0])
        if name.endswith("_scale"):
            return ("F8_E4M3", (1, 1), "fixture")
        if name.endswith("_scale_2"):
            return ("F32", (), "fixture")
        return (self.layer_dtypes[layer], (1, 1), "fixture")

    def expert_payload(self, name, r0, r1, c0, c1):
        width = 2 if self.meta(name)[0] == "BF16" else 1
        return b"\x11" * ((r1 - r0) * (c1 - c0) * width)

    def expert_scale(self, name, r0, r1, c0, c1):
        if self.meta(name)[0] == "BF16":
            return b""
        return b"\x22" * ((r1 - r0) * ((c1 - c0) // 128) * 4)

    def nvfp4_payload(self, name, r0, r1, c0, c1):
        return b"\x33" * ((r1 - r0) * (c1 - c0) // 2)

    def nvfp4_block_scale(self, name, r0, r1, c0, c1):
        return b"\x44" * ((r1 - r0) * (c1 - c0) // 16)

    def nvfp4_weight_global(self, name):
        return struct.pack("<f", 0.5)

    def close(self):
        pass


def build(source, layers, requested=None, degree=4, rank=1):
    builder = pack.Packer(source, degree, rank, 3, 1, False, False, False, requested)
    for layer in layers:
        builder.add_experts(layer)
    builder.build = lambda: None
    return builder


def emit_header_codec(builder, directory):
    path = Path(directory) / "pack.sp"
    pack.emit(builder, path, dict(stage_count=1, stage_index=0, first_layer=3,
                                  layer_count=1, flags=0))
    data = path.read_bytes()
    path.unlink()
    return struct.unpack_from("<I", data, HEADER_EXPERT_CODEC_OFFSET)[0], data


def expect_refusal(builder, directory, needle):
    try:
        emit_header_codec(builder, directory)
    except pack.PackFailure as error:
        assert needle in str(error), str(error)
    else:
        raise AssertionError(f"accepted a pack that must be refused ({needle})")
    assert list(Path(directory).iterdir()) == [], "refused pack left a file behind"


def main():
    with patch.multiple(pack, EXPERTS=2, HIDDEN=128, EXPERT_INTER=2048), \
            tempfile.TemporaryDirectory() as directory:
        for dtype, codec in (("BF16", pack.CODEC_BF16), ("F8_E4M3", pack.CODEC_FP8),
                             ("U8", pack.CODEC_NVFP4)):
            builder = build(Source({3: dtype}), [3])
            entry_codecs = {item.entry.weight_codec for item in builder.plan}
            assert entry_codecs == {codec}, (dtype, entry_codecs)
            header_codec, _ = emit_header_codec(builder, directory)
            assert header_codec == codec, (dtype, header_codec)
            header_codec, _ = emit_header_codec(build(Source({3: dtype}), [3], codec), directory)
            assert header_codec == codec
            wrong = pack.CODEC_FP8 if codec != pack.CODEC_FP8 else pack.CODEC_BF16
            expect_refusal(build(Source({3: dtype}), [3], wrong), directory, "source-driven")

        mixed = Source({3: "U8", 45: "BF16"})
        builder = build(mixed, [3, 45])
        assert [item.entry.weight_codec for item in builder.plan] == [
            pack.CODEC_NVFP4, pack.CODEC_NVFP4, pack.CODEC_BF16, pack.CODEC_BF16]
        expect_refusal(builder, directory, "mixed routed-expert codecs")

        mtp_only = build(Source({3: "U8", 45: "BF16"}), [45])
        assert emit_header_codec(mtp_only, directory)[0] == pack.CODEC_BF16

        empty = pack.Packer(Source({}), 4, 1, 0, 3, False, False, False)
        empty.build = lambda: None
        expect_refusal(empty, directory, "no routed-expert entries")
        empty = pack.Packer(Source({}), 4, 1, 0, 3, False, False, False, pack.CODEC_NVFP4)
        empty.build = lambda: None
        assert emit_header_codec(empty, directory)[0] == pack.CODEC_NVFP4

        try:
            build(Source({3: "F32"}), [3])
        except pack.PackFailure as error:
            assert "has no pack codec" in str(error)
        else:
            raise AssertionError("unknown routed-expert dtype accepted")

        builder = build(Source({3: "BF16"}), [3])
        _, data = emit_header_codec(builder, directory)
        stale = bytearray(data)
        struct.pack_into("<I", stale, HEADER_EXPERT_CODEC_OFFSET, pack.CODEC_FP8)
        path = Path(directory) / "stale.sp"
        path.write_bytes(bytes(stale))
        source = type("VerifySource", (), {"close": lambda self: None})()
        argv = ["verify", "--pack", str(path), "--source", "fixture", "--tp-degree", "4",
                "--tp-rank", "1", "--expected-bytes", str(len(stale)), "--first-layer", "3",
                "--layer-count", "1", "--skip-spot"]
        for extra, passes in (([], False), (["--accept-header-expert-codec", "5"], True),
                              (["--accept-header-expert-codec", "6"], False)):
            with patch.object(sys, "argv", argv + extra), \
                    patch.object(verify, "SourceReader", return_value=source), \
                    patch.object(verify, "Packer", return_value=builder), \
                    patch.object(verify, "check_stage_header", lambda header, args: None):
                try:
                    result = verify.main()
                except SystemExit as stop:
                    result = stop.code
            assert (result == 0) == passes, (extra, result)
        path.write_bytes(data)
        with patch.object(sys, "argv", argv), \
                patch.object(verify, "SourceReader", return_value=source), \
                patch.object(verify, "Packer", return_value=builder), \
                patch.object(verify, "check_stage_header", lambda header, args: None):
            assert verify.main() == 0
        path.unlink()

        builder = build(Source({3: "U8"}), [3])
        router = pack.Entry(pack.K_ROUTER, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE,
                            1, 1, 4)
        router.payload_bytes = 8
        builder.plan.insert(0, pack.PlanItem(router, lambda: iter([b"\x55" * 8])))
        _, data = emit_header_codec(builder, directory)
        directory_offset = struct.unpack_from("<Q", data, 80)[0]
        for entry_index, scoped_passes in ((0, True), (1, False)):
            changed = bytearray(data)
            at = directory_offset + entry_index * pack.ENTRY_BYTES + 6 * 4
            rows, columns = struct.unpack_from("<2I", changed, at)
            reshaped = (2, columns // 2) if rows == 1 else (rows // 2, columns * 2)
            struct.pack_into("<2I", changed, at, *reshaped)
            path.write_bytes(bytes(changed))
            for extra, passes in ((["--expert-layers", "3"], scoped_passes), (["--skip-spot"], False)):
                argv = ["verify", "--pack", str(path), "--source", "fixture", "--tp-degree", "4",
                        "--tp-rank", "1", "--expected-bytes", str(len(changed)), "--first-layer", "3",
                        "--layer-count", "1", *extra]
                with patch.object(sys, "argv", argv), \
                        patch.object(verify, "SourceReader", return_value=source), \
                        patch.object(verify, "Packer", return_value=builder), \
                        patch.object(verify, "check_stage_header", lambda header, args: None):
                    try:
                        result = verify.main()
                    except SystemExit as stop:
                        result = stop.code
                assert (result == 0) == passes, (entry_index, extra, result)
            path.unlink()
    print("PASS header expert codec: bf16/fp8/nvfp4 source-driven, explicit match, "
          "mixed and mismatched codecs refused, no-expert pack needs an explicit codec, "
          "pack_verify refuses a stale header unless the exact value is accepted, --expert-layers "
          "scopes the plan diff to routed experts")


if __name__ == "__main__":
    main()
