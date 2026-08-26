#!/usr/bin/env python3
"""Executable host-only contract for POST /v1/chat/completions, stream=false."""

import json
import pathlib
import unittest


FIXTURES = (
    pathlib.Path(__file__).resolve().parent
    / "fixtures" / "openai_compat" / "chat_non_streaming"
)


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def require_integer(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")


def validate_chat_completion(payload):
    """Accept exactly one non-streaming chat.completion envelope, else raise."""
    if not isinstance(payload, dict):
        raise ValueError("envelope must be a JSON object")
    if payload.get("stream"):
        raise ValueError("stream=true is unsupported by the non-streaming contract")
    kind = payload.get("object")
    if kind == "chat.completion.chunk":
        raise ValueError("streamed chunks are unsupported by the non-streaming contract")
    if kind != "chat.completion":
        raise ValueError(f"object must be 'chat.completion', got {kind!r}")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"model must be a non-empty string, got {model!r}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(f"exactly one choice is required, got {choices!r}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("choices[0] must be a JSON object")
    require_integer(choice.get("index"), "choices[0].index")
    if choice["index"] != 0:
        raise ValueError(f"choices[0].index must be 0, got {choice['index']!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("choices[0].message must be a JSON object")
    if message.get("role") != "assistant":
        raise ValueError(
            f"choices[0].message.role must be 'assistant', got {message.get('role')!r}"
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(
            f"choices[0].message.content must be a string, got {content!r}"
        )
    if choice.get("finish_reason") != "stop":
        raise ValueError(
            f"choices[0].finish_reason must be 'stop', "
            f"got {choice.get('finish_reason')!r}"
        )
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage must be a JSON object")
    for counter in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require_integer(usage.get(counter), f"usage.{counter}")
    return payload


class ChatNonStreamingContractTest(unittest.TestCase):
    def test_positive_fixture_is_valid(self):
        payload = load_fixture("positive_chat_completion.json")
        self.assertIs(validate_chat_completion(payload), payload)

    def test_negative_missing_content_is_rejected(self):
        payload = load_fixture("negative_missing_content.json")
        with self.assertRaisesRegex(ValueError, "choices\\[0\\]\\.message\\.content"):
            validate_chat_completion(payload)

    def test_negative_streamed_chunk_is_rejected_as_unsupported(self):
        payload = load_fixture("negative_streamed_chunk.json")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_chat_completion(payload)


if __name__ == "__main__":
    unittest.main()
