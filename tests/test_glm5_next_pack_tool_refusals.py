import io
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glm5_next_resident_stagepack as pack
import glm5_next_pack_verify as verify
import glm5_next_expert_graft as graft
import test_glm5_next_expert_graft as graft_fixture
import test_glm5_next_pack_header_codec as codec_fixture

REVISION = graft_fixture.REVISION


def expect_graft_refused(spine, expert, output, needle):
    try:
        graft.graft(spine, expert, output, 16, 3, "fp8", REVISION, 4096)
    except pack.PackFailure as error:
        assert needle in str(error), (needle, str(error))
    else:
        raise AssertionError(f"graft wrote {output} beside a source pack")
    leftovers = [p.name for p in output.parent.iterdir() if p.name.startswith(output.name)]
    assert leftovers == [], leftovers


def graft_output_directory_refusals(directory):
    sources = directory / "sources"
    production = directory / "production"
    arm = directory / "arm"
    for folder in (sources, production, arm):
        folder.mkdir()
    spine = sources / "spine-bf16.sp"
    fp8 = production / "experts-fp8.sp"
    graft_fixture.write_pack(spine, graft_fixture.entries_for(pack.CODEC_BF16, (3, 4)), 1)
    graft_fixture.write_pack(fp8, graft_fixture.entries_for(pack.CODEC_FP8, (3, 4)), 2)
    before = sorted(p.name for p in production.iterdir())
    expect_graft_refused(spine, fp8, production / "arm.sp", "is the expert pack's directory")
    expect_graft_refused(spine, fp8, sources / "arm.sp", "is the spine pack's directory")
    alias = directory / "production-alias"
    os.symlink(production, alias)
    expect_graft_refused(spine, fp8, alias / "arm.sp", "is the expert pack's directory")
    linked_expert = arm / "linked-experts-fp8.sp"
    os.symlink(fp8, linked_expert)
    expect_graft_refused(spine, linked_expert, production / "arm.sp", "is the expert pack's directory")
    assert sorted(p.name for p in production.iterdir()) == before
    receipt = graft.graft(spine, fp8, arm / "arm.sp", 16, 3, "fp8", REVISION, 4096)
    assert receipt["output"]["path"] == str(arm / "arm.sp")
    assert sorted(p.name for p in production.iterdir()) == before


def run_verify(argv, builder):
    source = type("VerifySource", (), {"close": lambda self: None})()
    stdout = io.StringIO()
    with patch.object(sys, "argv", argv), redirect_stdout(stdout), \
            patch.object(verify, "SourceReader", return_value=source), \
            patch.object(verify, "Packer", return_value=builder), \
            patch.object(verify, "check_stage_header", lambda header, args: None):
        try:
            result = verify.main()
        except SystemExit as stop:
            result = stop.code
    passes = [line for line in stdout.getvalue().splitlines() if line.startswith("VERIFY-PASS")]
    return result, passes


def verify_pass_names_waivers(directory):
    with patch.multiple(pack, EXPERTS=2, HIDDEN=128, EXPERT_INTER=2048):
        builder = codec_fixture.build(codec_fixture.Source({3: "BF16"}), [3])
        _, data = codec_fixture.emit_header_codec(builder, str(directory))
        path = directory / "verify.sp"
        base = ["verify", "--pack", str(path), "--source", "fixture", "--tp-degree", "4",
                "--tp-rank", "1", "--expected-bytes", str(len(data)), "--first-layer", "3",
                "--layer-count", "1"]
        path.write_bytes(data)
        for mode in (["--skip-spot"], ["--expert-layers", "3"]):
            result, passes = run_verify(base + mode, builder)
            assert result == 0 and len(passes) == 1 and "WAIVED" not in passes[0], (mode, passes)
        stale = bytearray(data)
        struct.pack_into("<I", stale, codec_fixture.HEADER_EXPERT_CODEC_OFFSET, pack.CODEC_FP8)
        path.write_bytes(bytes(stale))
        for mode in (["--skip-spot"], ["--expert-layers", "3"]):
            result, passes = run_verify(base + mode + ["--accept-header-expert-codec", "5"], builder)
            assert result == 0 and len(passes) == 1, (mode, result, passes)
            assert "WAIVED[header-expert-codec-accepted=5!=1]" in passes[0], passes[0]

        builder = codec_fixture.build(codec_fixture.Source({3: "U8"}), [3])
        router = pack.Entry(pack.K_ROUTER, 3, pack.PAYLOAD_BF16, pack.CODEC_BF16, pack.SCALE_NONE,
                            1, 1, 4)
        router.payload_bytes = 8
        builder.plan.insert(0, pack.PlanItem(router, lambda: iter([b"\x55" * 8])))
        _, data = codec_fixture.emit_header_codec(builder, str(directory))
        directory_offset = struct.unpack_from("<Q", data, 80)[0]
        changed = bytearray(data)
        at = directory_offset + 6 * 4
        rows, columns = struct.unpack_from("<2I", changed, at)
        struct.pack_into("<2I", changed, at, 2, columns // 2)
        path.write_bytes(bytes(changed))
        base[base.index("--expected-bytes") + 1] = str(len(changed))
        result, passes = run_verify(base + ["--expert-layers", "3"], builder)
        assert result == 0 and len(passes) == 1, (result, passes)
        assert "WAIVED[non-expert-plan-mismatches-out-of-scope=1]" in passes[0], passes[0]
        path.unlink()


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "graft").mkdir()
        (root / "verify").mkdir()
        graft_output_directory_refusals(root / "graft")
        verify_pass_names_waivers(root / "verify")
    print("PASS pack tool refusals: graft never writes beside a source pack (direct, symlinked "
          "directory, symlinked source), and VERIFY-PASS names every accepted waiver")


if __name__ == "__main__":
    main()
