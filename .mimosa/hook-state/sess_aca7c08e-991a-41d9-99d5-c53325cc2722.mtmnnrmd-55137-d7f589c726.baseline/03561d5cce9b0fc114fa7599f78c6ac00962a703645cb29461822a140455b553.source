#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


TRACE_RE = re.compile(
    r"dspark_trace verify "
    r"request=(?P<request>\d+) "
    r"sequence=(?P<sequence>\d+) "
    r"proposed=(?P<proposed>\d+) "
    r"accepted=(?P<accepted>\d+) "
    r"committed=(?P<committed>\d+) "
    r"fallback=(?P<fallback>\d+) "
    r"draft_ids=(?P<draft>[0-9,]*) "
    r"confidence_milli=(?P<confidence>[0-9,]*) "
    r"verifier_ids=(?P<verifier>[0-9,]*)"
)
SCHEMA = "sparkpipe.glm52.dspark.trace_quality.v1"
MAX_DRAFT_TOKENS = 7


class QualityFailure(RuntimeError):
    pass


def parse_numbers(text: str) -> list[int]:
    if not text:
        return []
    return [int(value) for value in text.split(",")]


def parse_trace_line(line: str) -> dict[str, Any] | None:
    match = TRACE_RE.search(line)
    if match is None:
        return None
    record: dict[str, Any] = {
        "request_id": int(match.group("request")),
        "sequence_id": int(match.group("sequence")),
        "proposed": int(match.group("proposed")),
        "accepted": int(match.group("accepted")),
        "committed": int(match.group("committed")),
        "fallback_token_id": int(match.group("fallback")),
        "draft_token_ids": parse_numbers(match.group("draft")),
        "confidence_milli": parse_numbers(match.group("confidence")),
        "verifier_token_ids": parse_numbers(match.group("verifier")),
    }
    validate_record(record)
    return record


def first_mismatch(draft: list[int], verifier: list[int]) -> int:
    for index, token_id in enumerate(draft):
        if token_id != verifier[index]:
            return index
    return len(draft)


def validate_record(record: dict[str, Any]) -> None:
    proposed = record["proposed"]
    accepted = record["accepted"]
    committed = record["committed"]
    draft = record["draft_token_ids"]
    confidence = record["confidence_milli"]
    verifier = record["verifier_token_ids"]
    if proposed <= 0 or proposed > MAX_DRAFT_TOKENS:
        raise QualityFailure(f"invalid proposed token count: {proposed}")
    if len(draft) != proposed or len(confidence) != proposed:
        raise QualityFailure("draft or confidence count does not match proposed")
    if len(verifier) < proposed or len(verifier) > proposed + 1:
        raise QualityFailure("verifier token count is not proposed or proposed+1")
    if accepted != first_mismatch(draft, verifier):
        raise QualityFailure("accepted count does not match the token prefix")
    if committed != accepted + 1:
        raise QualityFailure("committed count is not accepted+1")
    if any(value < 0 or value > 1000 for value in confidence):
        raise QualityFailure("confidence_milli must be in 0..1000")


def empty_summary() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "verify_count": 0,
        "proposed_draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "committed_tokens": 0,
        "fully_accepted_verifies": 0,
        "accepted_prefix_histogram": [0] * (MAX_DRAFT_TOKENS + 1),
        "position_observations": [0] * MAX_DRAFT_TOKENS,
        "position_token_matches": [0] * MAX_DRAFT_TOKENS,
        "position_prefix_accepts": [0] * MAX_DRAFT_TOKENS,
        "confidence_observations": 0,
        "confidence_absolute_error_sum": 0.0,
        "confidence_squared_error_sum": 0.0,
    }


def add_record(summary: dict[str, Any], record: dict[str, Any]) -> None:
    proposed = record["proposed"]
    accepted = record["accepted"]
    draft = record["draft_token_ids"]
    verifier = record["verifier_token_ids"]
    confidence = record["confidence_milli"]
    summary["verify_count"] += 1
    summary["proposed_draft_tokens"] += proposed
    summary["accepted_draft_tokens"] += accepted
    summary["committed_tokens"] += record["committed"]
    summary["fully_accepted_verifies"] += int(accepted == proposed)
    summary["accepted_prefix_histogram"][accepted] += 1
    for index in range(proposed):
        matched = int(draft[index] == verifier[index])
        prefix_accepted = int(accepted > index)
        predicted = confidence[index] / 1000.0
        error = predicted - prefix_accepted
        summary["position_observations"][index] += 1
        summary["position_token_matches"][index] += matched
        summary["position_prefix_accepts"][index] += prefix_accepted
        summary["confidence_observations"] += 1
        summary["confidence_absolute_error_sum"] += abs(error)
        summary["confidence_squared_error_sum"] += error * error


def ratios(numerators: list[int], denominators: list[int]) -> list[float | None]:
    return [
        numerator / denominator if denominator else None
        for numerator, denominator in zip(numerators, denominators)
    ]


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    count = summary["verify_count"]
    confidence_count = summary.pop("confidence_observations")
    absolute_error = summary.pop("confidence_absolute_error_sum")
    squared_error = summary.pop("confidence_squared_error_sum")
    summary["mean_proposed_draft_tokens"] = (
        summary["proposed_draft_tokens"] / count if count else None)
    summary["mean_accepted_draft_tokens"] = (
        summary["accepted_draft_tokens"] / count if count else None)
    summary["mean_accept_length"] = (
        summary["committed_tokens"] / count if count else None)
    summary["full_accept_rate"] = (
        summary["fully_accepted_verifies"] / count if count else None)
    summary["token_match_rate_by_position"] = ratios(
        summary["position_token_matches"],
        summary["position_observations"],
    )
    summary["prefix_acceptance_rate_by_position"] = ratios(
        summary["position_prefix_accepts"],
        summary["position_observations"],
    )
    summary["confidence_mean_absolute_error"] = (
        absolute_error / confidence_count if confidence_count else None)
    summary["confidence_brier_score"] = (
        squared_error / confidence_count if confidence_count else None)
    return summary


def summarize(lines: Iterable[str]) -> dict[str, Any]:
    summary = empty_summary()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = parse_trace_line(line)
        except QualityFailure as error:
            raise QualityFailure(f"line {line_number}: {error}") from error
        if record is not None:
            add_record(summary, record)
    return finalize_summary(summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize DSpark draft prediction quality from trace logs")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", default="")
    arguments = parser.parse_args()
    try:
        lines = (
            line
            for path_text in arguments.logs
            for line in Path(path_text).read_text(
                encoding="utf-8", errors="replace").splitlines()
        )
        summary = summarize(lines)
    except (OSError, QualityFailure) as error:
        print(f"dspark_quality_error: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
