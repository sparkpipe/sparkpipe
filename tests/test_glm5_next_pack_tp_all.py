import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack
from test_glm5_next_pack_header_codec import Source

REVISION = "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"
HEADER = dict(stage_count=1, stage_index=0, first_layer=3, layer_count=1, flags=0)


class CountingSource(Source):
    def __init__(self, layer_dtypes):
        super().__init__(layer_dtypes)
        self.reads = 0
        self.order = []

    def marked(self, name, r0, r1, c0, c1, blob):
        self.reads += 1
        rank = r0 // (r1 - r0) if ".down_proj." not in name else c0 // (c1 - c0)
        self.order.append((".down_proj." in name, rank))
        return bytes([rank + 1]) * len(blob)

    def expert_payload(self, name, r0, r1, c0, c1):
        return self.marked(name, r0, r1, c0, c1, super().expert_payload(name, r0, r1, c0, c1))

    def nvfp4_payload(self, name, r0, r1, c0, c1):
        return self.marked(name, r0, r1, c0, c1, super().nvfp4_payload(name, r0, r1, c0, c1))


def packers(source, degree):
    built = []
    for rank in range(degree):
        builder = pack.Packer(source, degree, rank, 3, 1, False, False, False)
        builder.build = lambda builder=builder: builder.add_experts(3) if not builder.plan else None
        built.append(builder)
    return built


def main():
    with patch.multiple(pack, EXPERTS=2, HIDDEN=128, EXPERT_INTER=2048), \
            tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for dtype in ("BF16", "F8_E4M3", "U8"):
            degree = 4
            single = []
            for builder in packers(CountingSource({3: dtype}), degree):
                path = directory / f"single-{dtype}-{builder.tp_rank}.sp"
                pack.emit(builder, path, HEADER, REVISION)
                single.append(path.read_bytes())
            paths = [directory / f"all-{dtype}-{rank}.sp" for rank in range(degree)]
            counting = CountingSource({3: dtype})
            sizes = pack.emit_ranks(packers(counting, degree), paths, HEADER, REVISION)
            assert counting.reads == degree * 3 * pack.EXPERTS, counting.reads
            assert counting.order == sorted(counting.order), counting.order
            assert [rank for down, rank in counting.order if down] == \
                [rank for rank in range(degree) for _ in range(pack.EXPERTS)]
            for rank, path in enumerate(paths):
                data = path.read_bytes()
                assert data == single[rank], (dtype, rank)
                assert len(data) == sizes[rank]
                assert data[96:96 + len(REVISION)] == REVISION.encode()
                assert struct.unpack_from("<2I", data, 18 * 4) == (degree, rank)
            assert len({path.read_bytes()[4096:] for path in paths}) == degree
            dropped = [directory / f"dropped-{dtype}-{rank}.sp" for rank in range(degree)]
            if hasattr(os, "posix_fadvise"):
                pack.emit_ranks(packers(CountingSource({3: dtype}), degree), dropped, HEADER,
                                REVISION, drop_output_cache=True, sync_bytes=4096)
                assert [path.read_bytes() for path in dropped] == single
            else:
                try:
                    pack.emit_ranks(packers(CountingSource({3: dtype}), degree), dropped, HEADER,
                                    REVISION, drop_output_cache=True)
                except pack.PackFailure as error:
                    assert "posix_fadvise" in str(error)
                else:
                    raise AssertionError("cache drop silently skipped")
            try:
                pack.emit_ranks(packers(CountingSource({3: dtype}), degree), paths, HEADER, REVISION)
            except pack.PackFailure as error:
                assert "already exists" in str(error)
            else:
                raise AssertionError("existing rank pack overwritten")
        mixed = packers(CountingSource({3: "BF16"}), 2)
        mixed[1] = pack.Packer(CountingSource({3: "BF16"}), 2, 1, 3, 1, False, False, False)
        mixed[1].build = lambda: None
        outputs = [directory / "mixed-0.sp", directory / "mixed-1.sp"]
        try:
            pack.emit_ranks(mixed, outputs, HEADER, REVISION)
        except pack.PackFailure as error:
            assert "plan order differs" in str(error)
        else:
            raise AssertionError("rank plans with different entries accepted")
        broken = packers(CountingSource({3: "BF16"}), 2)
        broken[1].build()
        broken[1].plan[1].produce_payload = lambda: iter([b"x"])
        outputs = [directory / "broken-0.sp", directory / "broken-1.sp"]
        try:
            pack.emit_ranks(broken, outputs, HEADER, REVISION)
        except pack.PackFailure:
            pass
        else:
            raise AssertionError("short region accepted")
        assert not any(path.name.startswith(("mixed", "broken")) for path in directory.iterdir())
    print("PASS single-pass tp-all: entry-major order (entry i of every rank before entry i+1), "
          "every rank byte-identical to its own emit, revision and "
          "rank in the header, existing outputs kept, mismatched plans and short regions leave no file")


if __name__ == "__main__":
    main()
