#!/usr/bin/env python3
"""Validate and compare normalized ds4-eval response archives.

The archive format is intentionally strict. Configuration or provenance drift
must be made explicit instead of being hidden by fallback parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_CASES = 92
EXPECTED_FAMILIES = {
    "GPQA Diamond": 25,
    "SuperGPQA": 25,
    "AIME2025": 25,
    "COMPSEC": 17,
}


class ArchiveError(ValueError):
    """The run is incomplete, malformed, or not comparison-compatible."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArchiveError(f"missing required file: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"invalid JSON file {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveError(message)


def family_name(source: str) -> str:
    if source.startswith("GPQA Diamond"):
        return "GPQA Diamond"
    return source


def is_letter_boundary(character: str) -> bool:
    return not character.isalpha()


def find_last_answer_marker(text: str) -> int:
    """Mirror ds4_eval.c find_last_answer_marker()."""
    lowered = text.lower()
    last = -1
    start = 0
    while True:
        index = lowered.find("answer", start)
        if index < 0:
            break
        before = text[index - 1] if index > 0 else " "
        after_index = index + len("answer")
        after = text[after_index] if after_index < len(text) else "\0"
        if is_letter_boundary(before) and is_letter_boundary(after):
            cursor = after_index
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ":":
                last = index
        start = index + 1
    if last >= 0:
        return last
    return lowered.find("answer")


def mc_letter_is_negated(text: str, start: int, letter: int) -> bool:
    cursor = letter
    while cursor > start:
        character = text[cursor - 1]
        if character == "\n":
            return False
        if character in " \t,;":
            cursor -= 1
        else:
            break
    word_end = cursor
    while cursor > start and (
        text[cursor - 1].isalpha() or text[cursor - 1] == "'"
    ):
        cursor -= 1
    word = text[cursor:word_end].lower()
    if not word or len(word) >= 16:
        return False
    if len(word) >= 3 and word.endswith("n't"):
        return True
    if word in {
        "not",
        "except",
        "excluding",
        "exclude",
        "excludes",
        "eliminate",
        "eliminates",
        "eliminated",
        "reject",
        "rejects",
        "rejected",
        "rejecting",
    }:
        return True
    if word == "out":
        prior = cursor
        while prior > start and text[prior - 1] in " \t":
            prior -= 1
        prior_end = prior
        while prior > start and text[prior - 1].isalpha():
            prior -= 1
        return text[prior:prior_end].lower() in {"rule", "rules", "ruled"}
    return False


def find_answer_letter(text: str, choice_count: int) -> str:
    max_answer = chr(ord("A") + choice_count - 1)
    marker = find_last_answer_marker(text)
    if marker >= 0:
        end = min(marker + 96, len(text))
        for index in range(marker, end):
            candidate = text[index].upper()
            if not ("A" <= candidate <= max_answer):
                continue
            before = text[index - 1] if index > 0 else " "
            after = text[index + 1] if index + 1 < len(text) else "\0"
            if not (
                is_letter_boundary(before) and is_letter_boundary(after)
            ):
                continue
            if after == "'":
                continue
            if after in " \t":
                cursor = index + 1
                while cursor < len(text) and text[cursor] in " \t":
                    cursor += 1
                if cursor < len(text) and text[cursor].islower():
                    continue
            if mc_letter_is_negated(text, marker, index):
                continue
            return candidate
    for index in range(len(text) - 1, -1, -1):
        candidate = text[index].upper()
        if not ("A" <= candidate <= max_answer):
            continue
        before = text[index - 1] if index > 0 else " "
        after = text[index + 1] if index + 1 < len(text) else "\0"
        if is_letter_boundary(before) and is_letter_boundary(after):
            return candidate
    return "?"


def first_unsigned_integer(text: str) -> str:
    match = re.search(r"\d+", text)
    return str(int(match.group(0))) if match else "?"


def find_integer_answer(text: str) -> str:
    marker = find_last_answer_marker(text)
    if marker >= 0:
        answer_line = text[marker : marker + 160].split("\n", 1)[0]
        last_equals = answer_line.rfind("=")
        if last_equals >= 0:
            candidate = first_unsigned_integer(answer_line[last_equals + 1 :])
            if candidate != "?":
                return candidate
        candidate = first_unsigned_integer(answer_line)
        if candidate != "?":
            return candidate
    integers = re.findall(r"\d+", text)
    return str(int(integers[-1])) if integers else "?"


def normalize_line_spec(text: str) -> str:
    pieces = [
        re.sub(r"\s+", "", match.group(0))
        for match in re.finditer(r"\d+\s*(?:-\s*\d+)?", text)
    ]
    return ",".join(pieces) if pieces else "?"


def parse_line_set(spec: str) -> set[int]:
    values: set[int] = set()
    for match in re.finditer(r"(\d+)(?:\s*-\s*(\d+))?", spec):
        left = int(match.group(1))
        right = int(match.group(2) or left)
        if left > right:
            left, right = right, left
        values.update(range(max(left, 0), min(right, 255) + 1))
    return values


def extract_answer(
    case: dict[str, Any], content: str, reasoning: str
) -> tuple[str, str]:
    surface = content.strip()
    text_source = "content"
    if not surface:
        surface = reasoning.strip()
        text_source = "reasoning_fallback"
    if "</think>" in surface:
        surface = surface.split("</think>", 1)[1]
    choices = case.get("choices")
    require(isinstance(choices, list), "case choices must be an array")
    if choices:
        return find_answer_letter(surface, len(choices)), text_source
    if case.get("source") == "COMPSEC":
        marker = find_last_answer_marker(surface)
        if marker >= 0:
            answer_line = surface[marker : marker + 160].split("\n", 1)[0]
            normalized = normalize_line_spec(answer_line)
            if normalized != "?":
                return normalized, text_source
    return find_integer_answer(surface), text_source


def answer_matches(case: dict[str, Any], extracted: str) -> bool:
    expected = str(case.get("answer") or "")
    if case.get("choices"):
        return extracted == expected
    if case.get("source") == "COMPSEC":
        expected_set = parse_line_set(expected)
        extracted_set = parse_line_set(extracted)
        return bool(extracted_set) and extracted_set.issubset(expected_set)
    return first_unsigned_integer(extracted) == first_unsigned_integer(expected)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def exact_output_sha(content: str, reasoning: str) -> str:
    value = json.dumps(
        {"content": content, "reasoning": reasoning},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256_text(value)


@dataclass(frozen=True)
class CaseResult:
    index: int
    case_id: str
    family: str
    passed: bool
    extracted: str
    expected: str
    finish_reason: str
    completion_tokens: int
    prompt_sha256: str
    output_sha256: str


@dataclass(frozen=True)
class Run:
    root: Path
    source_commit: str
    source_sha256: str
    request: dict[str, Any]
    results: tuple[CaseResult, ...]
    response_stream_sha256: str

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def by_family(self) -> dict[str, int]:
        return {
            family: sum(
                result.passed
                for result in self.results
                if result.family == family
            )
            for family in EXPECTED_FAMILIES
        }


def response_stream_sha256(response_paths: list[Path], run_root: Path) -> str:
    lines = bytearray()
    for path in sorted(response_paths, key=lambda value: value.name):
        relative = path.relative_to(run_root).as_posix()
        digest = sha256_bytes(path.read_bytes())
        lines.extend(f"{digest}  {relative}\n".encode("ascii"))
    return sha256_bytes(bytes(lines))


def load_run(root: Path) -> Run:
    root = root.resolve()
    summary = read_json(root / "summary.json")
    cases_document = read_json(root / "cases.json")
    integrity = read_json(root / "INTEGRITY.json")

    require(isinstance(summary, dict), f"{root}: summary must be an object")
    require(
        isinstance(cases_document, dict),
        f"{root}: cases manifest must be an object",
    )
    require(
        isinstance(integrity, dict)
        and integrity.get("format") == "ds4-eval-run-integrity-v1",
        f"{root}: invalid integrity manifest",
    )
    file_sha256 = integrity.get("file_sha256")
    require(
        isinstance(file_sha256, dict),
        f"{root}: integrity file hashes must be an object",
    )
    for name in ("REPORT.md", "cases.json", "summary.json"):
        expected_sha = file_sha256.get(name)
        require(
            isinstance(expected_sha, str) and len(expected_sha) == 64,
            f"{root}: missing integrity hash for {name}",
        )
        require(
            sha256_bytes((root / name).read_bytes()) == expected_sha,
            f"{root}: integrity hash mismatch for {name}",
        )
    cases = cases_document.get("cases")
    require(isinstance(cases, list), f"{root}: cases must be an array")
    require(
        len(cases) == EXPECTED_CASES,
        f"{root}: expected {EXPECTED_CASES} cases, found {len(cases)}",
    )

    cases_by_index: dict[int, dict[str, Any]] = {}
    for case in cases:
        require(isinstance(case, dict), f"{root}: case must be an object")
        index = case.get("index")
        case_id = case.get("id")
        require(
            isinstance(index, int) and 1 <= index <= EXPECTED_CASES,
            f"{root}: invalid case index {index!r}",
        )
        require(
            isinstance(case_id, str) and case_id,
            f"{root}: case {index} has no ID",
        )
        require(index not in cases_by_index, f"{root}: duplicate index {index}")
        cases_by_index[index] = case
    require(
        set(cases_by_index) == set(range(1, EXPECTED_CASES + 1)),
        f"{root}: case indices must be exactly 1..{EXPECTED_CASES}",
    )
    case_ids = [str(case["id"]) for case in cases_by_index.values()]
    require(
        len(case_ids) == len(set(case_ids)),
        f"{root}: duplicate case ID",
    )
    case_families = Counter(
        family_name(str(case.get("source") or ""))
        for case in cases_by_index.values()
    )
    require(
        case_families == Counter(EXPECTED_FAMILIES),
        f"{root}: unexpected family distribution {dict(case_families)}",
    )

    results_summary = summary.get("results")
    require(
        isinstance(results_summary, list),
        f"{root}: summary results must be an array",
    )
    require(
        len(results_summary) == EXPECTED_CASES,
        f"{root}: expected {EXPECTED_CASES} summary results, "
        f"found {len(results_summary)}",
    )
    require(
        summary.get("completed") == EXPECTED_CASES,
        f"{root}: summary is not complete",
    )
    require(summary.get("errors") == 0, f"{root}: summary contains errors")

    manifest_commit = cases_document.get("source_commit")
    manifest_sha = cases_document.get("ds4_eval_c_sha256")
    require(
        isinstance(manifest_commit, str) and len(manifest_commit) == 40,
        f"{root}: invalid source commit",
    )
    require(
        isinstance(manifest_sha, str) and len(manifest_sha) == 64,
        f"{root}: invalid ds4_eval.c SHA-256",
    )
    manifest_system_prompt = cases_document.get("system_prompt")
    require(
        isinstance(manifest_system_prompt, str) and manifest_system_prompt,
        f"{root}: system prompt is missing",
    )
    request = summary.get("request")
    require(isinstance(request, dict), f"{root}: request must be an object")
    for key in ("endpoint", "model", "max_tokens", "temperature", "stream"):
        require(key in request, f"{root}: request setting {key!r} is missing")
    summary_source = summary.get("source")
    require(
        isinstance(summary_source, dict)
        and summary_source.get("commit") == manifest_commit
        and summary_source.get("ds4_eval_c_sha256") == manifest_sha,
        f"{root}: summary source provenance mismatch",
    )

    seen_indices: set[int] = set()
    seen_response_ids: set[str] = set()
    response_paths: list[Path] = []
    results: list[CaseResult] = []
    for item in results_summary:
        require(isinstance(item, dict), f"{root}: result must be an object")
        index = item.get("index")
        require(
            isinstance(index, int) and index in cases_by_index,
            f"{root}: invalid result index {index!r}",
        )
        require(index not in seen_indices, f"{root}: duplicate result {index}")
        seen_indices.add(index)
        case = cases_by_index[index]
        require(
            item.get("id") == case.get("id"),
            f"{root}: result {index} case ID mismatch",
        )
        response_name = item.get("response_file")
        require(
            isinstance(response_name, str) and response_name,
            f"{root}: result {index} has no response file",
        )
        response_path = root / response_name
        require(
            response_path.parent == root / "responses",
            f"{root}: result {index} response path escapes responses/",
        )
        response_paths.append(response_path)
        document = read_json(response_path)
        require(
            isinstance(document, dict),
            f"{response_path}: response must be an object",
        )
        require(
            document.get("status") == "ok",
            f"{response_path}: response status is not ok",
        )
        archived_case = document.get("case")
        require(
            isinstance(archived_case, dict)
            and archived_case.get("index") == index
            and archived_case.get("id") == case.get("id"),
            f"{response_path}: archived case identity mismatch",
        )
        for key in (
            "source",
            "id",
            "domain",
            "title",
            "question",
            "answer",
            "choices",
        ):
            require(
                archived_case.get(key) == case.get(key),
                f"{response_path}: archived case field {key!r} mismatch",
            )
        source = document.get("source")
        require(
            isinstance(source, dict)
            and source.get("commit") == manifest_commit
            and source.get("ds4_eval_c_sha256") == manifest_sha,
            f"{response_path}: source provenance mismatch",
        )
        prompt = document.get("prompt")
        require(
            isinstance(prompt, dict)
            and isinstance(prompt.get("system"), str)
            and isinstance(prompt.get("user"), str)
            and isinstance(prompt.get("sha256"), str),
            f"{response_path}: malformed prompt record",
        )
        calculated_prompt_sha = sha256_text(
            prompt["system"] + "\n" + prompt["user"]
        )
        require(
            prompt["sha256"] == calculated_prompt_sha,
            f"{response_path}: prompt SHA-256 mismatch",
        )
        require(
            prompt["system"] == manifest_system_prompt
            and prompt["user"] == case.get("rendered_prompt"),
            f"{response_path}: prompt text differs from cases manifest",
        )
        archived_request = document.get("request")
        require(
            isinstance(archived_request, dict),
            f"{response_path}: malformed request record",
        )
        for key, value in request.items():
            require(
                archived_request.get(key) == value,
                f"{response_path}: request setting {key!r} differs "
                "from summary",
            )
        api_response = document.get("response")
        require(
            isinstance(api_response, dict),
            f"{response_path}: malformed response record",
        )
        content = api_response.get("content")
        reasoning = api_response.get("reasoning")
        response_id = api_response.get("response_id")
        finish_reason = api_response.get("finish_reason")
        require(
            isinstance(content, str) and isinstance(reasoning, str),
            f"{response_path}: content and reasoning must be strings",
        )
        require(
            bool(content or reasoning),
            f"{response_path}: response text is empty",
        )
        require(
            isinstance(response_id, str) and response_id,
            f"{response_path}: response ID is missing",
        )
        require(
            response_id not in seen_response_ids,
            f"{root}: duplicate response ID {response_id}",
        )
        seen_response_ids.add(response_id)
        require(
            isinstance(finish_reason, str) and finish_reason,
            f"{response_path}: finish reason is missing",
        )
        require(
            api_response.get("response_model") == request.get("model"),
            f"{response_path}: returned model differs from requested model",
        )
        score = document.get("score")
        require(
            isinstance(score, dict)
            and isinstance(score.get("passed"), bool)
            and isinstance(score.get("expected"), str)
            and isinstance(score.get("extracted"), str),
            f"{response_path}: malformed score record",
        )
        extracted, text_source = extract_answer(case, content, reasoning)
        expected = str(case.get("answer") or "")
        passed = answer_matches(case, extracted)
        require(
            score.get("expected") == expected
            and score.get("extracted") == extracted
            and score.get("passed") == passed
            and score.get("text_source") == text_source,
            f"{response_path}: stored score differs from independent regrade",
        )
        require(
            item.get("passed") == passed
            and item.get("expected") == expected
            and item.get("extracted") == extracted,
            f"{response_path}: independently regraded score differs "
            "from summary",
        )
        usage = api_response.get("usage")
        require(
            isinstance(usage, dict)
            and isinstance(usage.get("completion_tokens"), int),
            f"{response_path}: completion-token usage is missing",
        )
        results.append(
            CaseResult(
                index=index,
                case_id=str(case["id"]),
                family=family_name(str(case.get("source") or "")),
                passed=score["passed"],
                extracted=score["extracted"],
                expected=score["expected"],
                finish_reason=finish_reason,
                completion_tokens=usage["completion_tokens"],
                prompt_sha256=prompt["sha256"],
                output_sha256=exact_output_sha(content, reasoning),
            )
        )

    require(
        seen_indices == set(range(1, EXPECTED_CASES + 1)),
        f"{root}: result indices must be exactly 1..{EXPECTED_CASES}",
    )
    actual_response_paths = sorted((root / "responses").glob("*.json"))
    require(
        {path.resolve() for path in response_paths}
        == {path.resolve() for path in actual_response_paths},
        f"{root}: response directory and summary references differ",
    )
    require(
        summary.get("passed") == sum(result.passed for result in results),
        f"{root}: aggregate pass count mismatch",
    )
    calculated_by_family = {
        family: {
            "completed": total,
            "passed": sum(
                result.passed
                for result in results
                if result.family == family
            ),
        }
        for family, total in EXPECTED_FAMILIES.items()
    }
    require(
        summary.get("by_source") == calculated_by_family,
        f"{root}: per-family summary mismatch",
    )
    stream_sha256 = response_stream_sha256(response_paths, root)
    require(
        integrity.get("response_count") == EXPECTED_CASES
        and integrity.get("response_stream_sha256") == stream_sha256,
        f"{root}: response-set integrity mismatch",
    )

    return Run(
        root=root,
        source_commit=manifest_commit,
        source_sha256=manifest_sha,
        request=request,
        results=tuple(sorted(results, key=lambda result: result.index)),
        response_stream_sha256=stream_sha256,
    )


def print_run(run: Run) -> None:
    print(f"run: {run.root}")
    print(f"source commit: {run.source_commit}")
    print(f"ds4_eval.c sha256: {run.source_sha256}")
    print(f"score: {run.passed}/{len(run.results)}")
    for family, total in EXPECTED_FAMILIES.items():
        print(f"  {family}: {run.by_family[family]}/{total}")
    print(f"response stream sha256: {run.response_stream_sha256}")


def compare_runs(baseline: Run, candidate: Run) -> None:
    require(
        baseline.source_commit == candidate.source_commit,
        "source commit differs between runs",
    )
    require(
        baseline.source_sha256 == candidate.source_sha256,
        "ds4_eval.c SHA-256 differs between runs",
    )
    for setting in ("max_tokens", "temperature"):
        require(
            baseline.request.get(setting) == candidate.request.get(setting),
            f"request setting {setting!r} differs between runs",
        )
    require(
        baseline.request.get("seed") == candidate.request.get("seed"),
        "request setting 'seed' differs between runs",
    )
    baseline_by_index = {result.index: result for result in baseline.results}
    candidate_by_index = {result.index: result for result in candidate.results}
    require(
        baseline_by_index.keys() == candidate_by_index.keys(),
        "case index sets differ between runs",
    )

    for index, base in baseline_by_index.items():
        other = candidate_by_index[index]
        require(
            base.case_id == other.case_id,
            f"case {index}: ID differs between runs",
        )
        require(
            base.prompt_sha256 == other.prompt_sha256,
            f"case {index}: prompt differs between runs",
        )
        require(
            base.expected == other.expected,
            f"case {index}: expected answer differs between runs",
        )

    print("\ncomparison:")
    delta = candidate.passed - baseline.passed
    print(
        f"  overall: {baseline.passed} -> {candidate.passed} "
        f"({delta:+d})"
    )
    for family, total in EXPECTED_FAMILIES.items():
        before = baseline.by_family[family]
        after = candidate.by_family[family]
        print(f"  {family}: {before}/{total} -> {after}/{total} "
              f"({after - before:+d})")

    regressions: list[int] = []
    improvements: list[int] = []
    answer_changes: list[int] = []
    finish_changes: list[int] = []
    token_changes: list[int] = []
    output_changes: list[tuple[int, str, str]] = []
    exact_matches = 0
    for index, base in baseline_by_index.items():
        other = candidate_by_index[index]
        if base.passed and not other.passed:
            regressions.append(index)
        if not base.passed and other.passed:
            improvements.append(index)
        if base.extracted != other.extracted:
            answer_changes.append(index)
        if base.finish_reason != other.finish_reason:
            finish_changes.append(index)
        if base.completion_tokens != other.completion_tokens:
            token_changes.append(index)
        if base.output_sha256 == other.output_sha256:
            exact_matches += 1
        else:
            output_changes.append(
                (index, base.output_sha256, other.output_sha256)
            )

    def indices(values: list[int]) -> str:
        return ",".join(str(value) for value in values) if values else "none"

    print(f"  pass-to-fail: {indices(regressions)}")
    print(f"  fail-to-pass: {indices(improvements)}")
    print(f"  extracted-answer changes: {indices(answer_changes)}")
    print(f"  finish-reason changes: {indices(finish_changes)}")
    print(f"  completion-token changes: {indices(token_changes)}")
    print(f"  exact content+reasoning matches: {exact_matches}/{EXPECTED_CASES}")
    if output_changes:
        print("  exact output changes:")
        for index, before, after in output_changes:
            print(f"    {index}: {before} -> {after}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one normalized ds4-eval run or compare two."
    )
    parser.add_argument(
        "run",
        nargs="+",
        type=Path,
        help="one run directory to validate, or baseline and candidate",
    )
    args = parser.parse_args()
    if len(args.run) not in (1, 2):
        parser.error("provide exactly one or two run directories")
    return args


def main() -> int:
    args = parse_args()
    try:
        runs = [load_run(path) for path in args.run]
        for run in runs:
            print_run(run)
        if len(runs) == 2:
            compare_runs(runs[0], runs[1])
    except ArchiveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
