#!/usr/bin/env python3

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "glm52_dspark_trace_quality.py"
SPEC = importlib.util.spec_from_file_location("glm52_dspark_trace_quality", TOOL)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def trace(
    accepted: int,
    draft: list[int],
    confidence: list[int],
    verifier: list[int],
) -> str:
    return (
        "dspark_trace verify request=1 sequence=2 "
        f"proposed={len(draft)} accepted={accepted} "
        f"committed={accepted + 1} fallback={verifier[accepted]} "
        f"draft_ids={','.join(str(value) for value in draft)} "
        f"confidence_milli={','.join(str(value) for value in confidence)} "
        f"verifier_ids={','.join(str(value) for value in verifier)}"
    )


def main() -> None:
    lines = [
        trace(
            3,
            [10, 11, 12, 13, 14, 15, 16],
            [900, 800, 700, 600, 500, 400, 300],
            [10, 11, 12, 99, 14, 15, 16, 17],
        ),
        trace(
            7,
            [20, 21, 22, 23, 24, 25, 26],
            [950, 900, 850, 800, 750, 700, 650],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ),
        "unrelated log line",
    ]
    summary = QUALITY.summarize(lines)
    assert summary["verify_count"] == 2
    assert summary["accepted_draft_tokens"] == 10
    assert summary["committed_tokens"] == 12
    assert summary["mean_accepted_draft_tokens"] == 5.0
    assert summary["mean_accept_length"] == 6.0
    assert summary["full_accept_rate"] == 0.5
    assert summary["accepted_prefix_histogram"] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert summary["prefix_acceptance_rate_by_position"] == [
        1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
    assert summary["token_match_rate_by_position"] == [
        1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    try:
        QUALITY.summarize([
            trace(
                2,
                [10, 11, 12],
                [900, 800, 700],
                [10, 99, 12, 13],
            )
        ])
    except QUALITY.QualityFailure:
        pass
    else:
        raise AssertionError("inconsistent accepted prefix was not rejected")


if __name__ == "__main__":
    main()
