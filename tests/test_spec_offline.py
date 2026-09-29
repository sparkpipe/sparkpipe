#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from spec_offline import acceptance, chain, cost, drafters, tapdump, tree  # noqa: E402
from spec_offline.streams import Stream, read_u32, write_u32  # noqa: E402
import spec_offline.t0 as t0  # noqa: E402
import spec_roofline  # noqa: E402

VOCAB = spec_roofline.load_registry()["models"]["glmflash"]["vocab"]


def build_replay(directory: Path) -> Path:
    binary = directory / "replay"
    subprocess.run(["cc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-I.", "-Iinclude", "-Imodel-families/glm5_next/include",
                    "-Imodel-families/common/include", "tools/glm5_next_spec_replay.c", "src/spark_speculation_lookup_draft.c",
                    "src/spark_speculation_drafter_mix.c", "src/spark_speculation_policy.c", "src/spark_status.c", "-o", str(binary)], cwd=ROOT, check=True)
    return binary


def c_replay(binary: Path, files: list[Path], *options: str) -> list[dict]:
    output = subprocess.run([str(binary), *options, "--", *map(str, files)], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in output.splitlines()]


def synthetic_streams(directory: Path) -> list[Path]:
    prompt = [(index * 37 + 5) % 997 for index in range(40)]
    cycle = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    repeating = [cycle[index % len(cycle)] for index in range(300)]
    noisy = [(index * 7919 + 13) % 50021 for index in range(300)]
    mixed = [cycle[index % 11] if (index // 23) % 2 == 0 else (index * 104729 + 7) % 90001 for index in range(400)]
    long_prompt = [(index * 31 + 3) % 3001 for index in range(200)]
    files = []
    for name, prefix, output in (("repetitive_0", prompt, repeating), ("prose_0", prompt, noisy), ("mixed_0", prompt, mixed), ("code_0", long_prompt, repeating[:150] + noisy[:150])):
        path = directory / f"{name}.u32"
        write_u32(path, prefix, output)
        files.append(path)
    return files


def check_chain_parity(binary: Path, files: list[Path]) -> None:
    for options in (["--drafter", "lookup"], ["--drafter", "synthetic:1000", "--fixed-depth"], ["--drafter", "synthetic:0"],
                    ["--drafter", "synthetic:700"], ["--drafter", "synthetic:700", "--rows", "4", "--frame", "32"], ["--drafter", "lookup", "--rows", "3", "--frame", "5", "--block", "16"]):
        expected = c_replay(binary, files, *options)
        drafter = options[1]
        rows = int(options[options.index("--rows") + 1]) if "--rows" in options else 8
        frame = int(options[options.index("--frame") + 1]) if "--frame" in options else 8
        block = int(options[options.index("--block") + 1]) if "--block" in options else 64
        for path, reference in zip(files, expected):
            stream = read_u32(path)
            counts = chain.simulate_chain(stream, drafter, VOCAB, rows=rows, frame=frame, block=block, fixed_depth="--fixed-depth" in options).as_dict()
            for key in ("frames", "verify_frames", "plain_frames", "plain_frame_tokens", "rounds", "proposed", "accepted", "plain_steps", "rows"):
                assert counts[key] == reference[key], f"{path.name} {options}: {key} python {counts[key]} vs C {reference[key]}"
            assert counts["tokens"] == reference["generated"]
    twice = [chain.simulate_chain(read_u32(files[2]), "synthetic:700", VOCAB).as_dict() for _ in range(2)]
    assert twice[0] == twice[1]
    for path in files:
        stream = read_u32(path)
        for drafter in ("lookup", "synthetic:500"):
            assert tuple(chain.simulate_chain(stream, drafter, VOCAB).committed) == stream.tokens


def check_regime_rules() -> None:
    assert chain.graph_regime(1, 64) == chain.REGIME_UNSPLIT and chain.graph_regime(64, 64) == chain.REGIME_SPLIT and chain.graph_regime(2049, 64) == chain.REGIME_SELECTED
    assert chain.rows_fit(60, 8, 64) == 3 and chain.rows_fit(63, 8, 64) == 8 and chain.rows_fit(2045, 8, 64) == 3
    assert chain.verify_depth(8, 0, 8, 100, 64) == 7 and chain.verify_depth(8, 6, 8, 100, 64) == 1 and chain.verify_depth(8, 7, 8, 100, 64) == 0
    assert chain.verify_depth(8, 0, 8, 60, 64) == 2
    assert chain.depth_cap_next(7, 7, 7, 7) == 7 and chain.depth_cap_next(2, 2, 2, 7) == 4 and chain.depth_cap_next(4, 4, 4, 7) == 7 and chain.depth_cap_next(7, 7, 2, 7) == 3
    assert chain.resolve_chain([5, 6, 7], [5, 6, 8, 9]) == (2, 3) and chain.resolve_chain([1], [2, 3]) == (0, 1)


def check_tree_resolver() -> None:
    rng = np.random.default_rng(3)
    for _ in range(300):
        count = int(rng.integers(1, 8))
        tokens = [int(t) for t in rng.integers(0, 4, count)]
        parents = [tree.ROOT if index == 0 else int(rng.integers(-1, index)) for index in range(count)]
        truth = [int(t) for t in rng.integers(0, 4, count + 1)]
        verifier = []
        for index in range(count + 1):
            depth = 0
            node = index - 1
            path = []
            while node != tree.ROOT:
                path.append(tokens[node])
                node = parents[node]
            depth = len(path)
            verifier.append(truth[depth])
        nodes = [tree.Node(token, parent, 0) for token, parent in zip(tokens, parents)]
        best, committed, _, best_node = tree.resolve_tree(nodes, truth)
        assert tree.path_tokens(nodes, best_node) == truth[:best]
        rows_best, rows_committed = tree.resolve_tree_rows(tokens, parents, verifier)
        assert best == rows_best and committed == rows_committed, (tokens, parents, truth)
    shape = tree.parse_shape("tree-top2d2-r8")
    assert shape.width == 2 and shape.depth == 2 and shape.rows == 8
    assert tree.parse_shape("chain-k7").rows == 8 and tree.parse_shape("trie-mtp+suffix-r8").members == ("mtp", "suffix")
    for bad in ("chain-k32", "tree-top1d2-r8", "trie-a+a-r4", "bush-3"):
        try:
            tree.parse_shape(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def check_round_replay(files: list[Path]) -> None:
    for path in files:
        stream = read_u32(path)
        for spec in ("oracle", "adversary", "lookup", "suffix", "ngram3", "synthetic:600"):
            for shape_text in ("chain-k1", "chain-k7", "tree-top2d2-r8"):
                result = acceptance.round_replay(stream, drafters.make_drafter(spec, VOCAB), None, tree.parse_shape(shape_text))
                assert result.stream_exact, f"{spec} {shape_text} on {path.name} changed the committed stream"
                assert result.committed == stream.length - stream.prompt - 1
                if spec == "oracle" and shape_text == "chain-k7":
                    assert result.tokens_per_round() > 7.5
                if spec == "adversary":
                    assert result.accepted == 0 and result.tokens_per_round() == 1.0
        oracle_tree = acceptance.round_replay(stream, drafters.make_drafter("oracle", VOCAB), None, tree.parse_shape("tree-top2d2-r8"))
        oracle_chain = acceptance.round_replay(stream, drafters.make_drafter("oracle", VOCAB), None, tree.parse_shape("chain-k5"))
        assert oracle_tree.tokens_per_round() <= oracle_chain.tokens_per_round() + 1e-9
        members = [drafters.make_drafter("synthetic:0", VOCAB), drafters.make_drafter("oracle", VOCAB)]
        trie = acceptance.round_replay(stream, members[0], None, tree.parse_shape("trie-synthetic0+oracle-r8"), members)
        assert trie.stream_exact and trie.tokens_per_round() > 3.0
        honest = acceptance.resolve_tree
        acceptance.resolve_tree = lambda nodes, truth: (1, 2, 1, 0)
        try:
            lying = acceptance.round_replay(stream, drafters.make_drafter("adversary", VOCAB), None, tree.parse_shape("chain-k1"))
        finally:
            acceptance.resolve_tree = honest
        assert not lying.stream_exact, "a resolver that accepts a wrong draft token must break the committed-stream identity"
    repetitive = read_u32(files[0])
    table = acceptance.position_table(repetitive, drafters.make_drafter("lookup", VOCAB), None)
    rates = table.acceptance()
    assert rates[0] is not None and rates[0] > 0.9 and table.tau(7) > 6.0
    noisy = acceptance.position_table(read_u32(files[1]), drafters.make_drafter("lookup", VOCAB), None)
    assert noisy.accepted[0] == 0
    synthetic = acceptance.position_table(read_u32(files[1]), drafters.make_drafter("synthetic:800", VOCAB), None)
    assert 0.7 < synthetic.acceptance()[0] < 0.9 and synthetic.acceptance()[3] < synthetic.acceptance()[0]
    twice = [acceptance.position_table(read_u32(files[2]), drafters.make_drafter("suffix", VOCAB), None).as_dict() for _ in range(2)]
    assert twice[0] == twice[1]


def check_tap_dump(directory: Path) -> None:
    hidden = 64
    vocab = 1000
    prompt = [3, 5, 7, 11, 13]
    output = [(index * 17 + 1) % vocab for index in range(120)]
    tokens = prompt + output
    taps = np.zeros((len(tokens), hidden), dtype=np.float32)
    schedule = [position % 3 != 2 for position in range(len(tokens))]
    for position in range(len(tokens) - 1):
        target = tokens[position + 1] if schedule[position] else (tokens[position + 1] + 1) % vocab
        taps[position] = drafters.probe_tap_for_token(target, vocab, position, hidden)
    dump = directory / "dump"
    tapdump.write_dump(dump, "testmodel", "deadbeef", "hc_mean", 45, hidden, [("prose_0", "prose", prompt, output, taps)], {"synthetic": True})
    manifest, streams = tapdump.read_dump(dump)
    assert manifest["format"] == tapdump.FORMAT and streams[0].taps.shape == (len(tokens), hidden)
    assert np.allclose(streams[0].taps, tapdump.bf16_to_f32(tapdump.f32_to_bf16(taps)))
    stream = streams[0].stream
    drafter = drafters.make_drafter("tap:probe", vocab=vocab)
    table = acceptance.position_table(stream, drafter, streams[0].taps)
    expected = sum(1 for anchor in acceptance.anchors(stream) if schedule[anchor])
    assert table.accepted[0] == expected and table.reached[0] == table.positions, (table.accepted, expected)
    assert expected - 1 <= table.reached[1] <= expected and table.accepted[1] == 0 and abs(table.tau(3) - (1.0 + expected / table.positions)) < 1e-9
    exact = acceptance.round_replay(stream, drafter, streams[0].taps, tree.parse_shape("chain-k3"))
    assert exact.stream_exact and 0 < exact.accepted <= expected and exact.committed == stream.length - stream.prompt - 1
    (dump / "prose_0.hc_mean.bf16").write_bytes(b"\0\0" + (dump / "prose_0.hc_mean.bf16").read_bytes()[2:])
    try:
        tapdump.read_dump(dump)
    except ValueError as error:
        assert "SHA256SUMS" in str(error)
    else:
        raise AssertionError("a tampered tap file was accepted")
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "acc.json"
        tapdump.write_dump(dump, "testmodel", "deadbeef", "hc_mean", 45, hidden, [("prose_0", "prose", prompt, output, taps)])
        assert t0.main(["acceptance", "--dump", str(dump), "--model", "glmflash", "--vocab", str(vocab), "--drafters", "oracle,tap:probe", "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["drafters"]["oracle"]["classes"]["prose"]["acceptance_per_position"][0] == 1.0
        assert report["drafters"]["tap:probe"]["classes"]["prose"]["accepted"][0] == expected
        agree = Path(scratch) / "agree.json"
        code = t0.main(["agreement", "--dump", str(dump), "--candidate", "tap:probe", "--reference", "oracle", "--vocab", str(vocab), "--min-positions", "10", "--out", str(agree)])
        assert code == 1 and json.loads(agree.read_text())["verdict"] == "FAIL"
        code = t0.main(["agreement", "--dump", str(dump), "--candidate", "oracle", "--reference", "oracle", "--min-positions", "10", "--out", str(agree)])
        assert code == 0


def rank_log(rows_ms: dict[int, list[float]], slot_base: int = 0) -> list[str]:
    lines = ["noise", "GRAPH-VERIFY-TABLE slot=1 rows_max=8 status=0 capture_ms=100"]
    for rows, values in rows_ms.items():
        for value in values:
            lines.append(f"CHAIN slot={slot_base} stage=0 layer=0 rows={rows} mi=728 hi=8")
            lines.append(f"GRAPH-REPLAY-TIME slot={slot_base} wall_ns={int(value * 1e6)} stream_status=0")
    lines.append(f"CHAIN slot={slot_base} stage=0 layer=0 rows=8 mi=1 hi=1")
    lines.append(f"GRAPH-REPLAY-TIME slot={slot_base} wall_ns=999000000 stream_status=7")
    return lines


def check_cost_fit(directory: Path) -> None:
    samples = cost.parse_rank_log(rank_log({1: [20.0, 21.0], 8: [50.3]}))
    assert samples == [(1, 20.0), (1, 21.0), (8, 50.3)]
    oracle = directory / "logs-oracle"
    adversary = directory / "logs-adversary"
    oracle.mkdir()
    adversary.mkdir()
    for rank in range(16):
        jitter = 0.01 * rank
        seven = {7: [47.6]} if rank < 8 else {}
        (oracle / f"residentd.rank{rank}.log").write_text("\n".join(rank_log({1: [20.0 + jitter] * 8 + [49.0], 8: [50.3 + jitter] * 8 + [76.0], **seven})) + "\n")
        (adversary / f"residentd.rank{rank}.log").write_text("\n".join(rank_log({1: [19.8 + jitter] * 8, 2: [23.6 + jitter] * 20})) + "\n")
    model = cost.fit_verify_cost({"oracle": oracle, "adversary": adversary}, min_samples=16, extra_points=[{"rows": 64, "ms": 400.0, "label": "B8 x K8 no-spec"}])
    assert model["samples"]["1"]["measured"] and model["samples"]["8"]["measured"] and not model["samples"]["7"]["measured"]
    assert abs(model["samples"]["1"]["engine_median_ms"] - 20.0) < 0.2 and abs(model["samples"]["8"]["engine_median_ms"] - 50.3) < 0.2
    assert abs(model["samples"]["2"]["engine_median_ms"] - 23.6) < 0.2
    assert model["samples"]["8"]["max_ms"] == 76.08 or model["samples"]["8"]["max_ms"] > 75.0
    fit = model["fit"]
    assert 3.5 < fit["per_row_ms"] < 5.0 and 18.0 < fit["intercept_ms"] < 22.0
    ms, source = cost.verify_ms(model, 7)
    assert source == "measured" and abs(ms - 50.3) < 0.2
    ms, source = cost.verify_ms(model, 3)
    assert source == "fitted" and 30.0 < ms < 38.0
    ms, source = cost.verify_ms(model, 7, 8)
    assert source == "B8 x K8 no-spec" and ms == 400.0
    ms, source = cost.verify_ms(model, 15)
    assert source == "extrapolated"
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "cost.json"
        assert t0.main(["cost-fit", "--arm-logs", f"oracle={oracle}", f"adversary={adversary}", "--out", str(out)]) == 0
        acceptance_path = Path(scratch) / "acc.json"
        streams = Path(scratch) / "streams"
        streams.mkdir()
        synthetic_streams(streams)
        assert t0.main(["acceptance", "--streams", str(streams), "--drafters", "oracle,lookup,synthetic:800", "--out", str(acceptance_path)]) == 0
        predicted = Path(scratch) / "pred.json"
        assert t0.main(["predict", "--acceptance", str(acceptance_path), "--cost", str(out), "--model", "glmflash", "--out", str(predicted)]) == 0
        report = json.loads(predicted.read_text())
        assert report["no_spec"]["roofline"].startswith("roofline @B=1: memory")
        oracle_all = report["drafters"]["oracle"]["all"]
        assert oracle_all["best_k"] == 7 and oracle_all["best_spec_tok_s"] > 2.5 * oracle_all["no_spec_tok_s"]
        lookup_prose = report["drafters"]["lookup"]["prose"]
        assert lookup_prose["best_spec_tok_s"] < lookup_prose["no_spec_tok_s"]
        for depth in oracle_all["depths"].values():
            assert depth["roofline"].startswith("roofline @B=1,rows=")
        rounds_out = Path(scratch) / "rounds.json"
        assert t0.main(["rounds", "--streams", str(streams), "--drafter", "lookup", "--shapes", "chain-k3,tree-top2d1-r4,trie-lookup+suffix-r6", "--out", str(rounds_out)]) == 0
        rounds = json.loads(rounds_out.read_text())
        assert all(cell["stream_exact"] for shape in rounds["shapes"].values() for cell in shape["classes"].values())
        assert rounds["shapes"]["chain-k3"]["classes"]["repetitive"]["tokens_per_round"] > 2.5


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        binary = build_replay(directory)
        files = synthetic_streams(directory)
        check_regime_rules()
        check_chain_parity(binary, files)
        check_tree_resolver()
        check_round_replay(files)
        check_tap_dump(directory)
        check_cost_fit(directory)
    print("PASS spec_offline: chain simulation equals glm5_next_spec_replay on lookup and synthetic streams, tree accept equals the row resolver, "
          "round replays keep the committed stream identical to the no-spec stream, tap dumps round-trip and refuse tampering, "
          "the verify-cost fit recovers the per-row medians and predictions carry roofline lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
