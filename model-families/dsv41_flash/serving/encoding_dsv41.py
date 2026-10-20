import importlib.util
from dataclasses import dataclass
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("dsv41_publisher_encoding", Path(__file__).with_name("encoding.py"))
_ENCODING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ENCODING)


@dataclass(frozen=True)
class EncodeSegment:
    text: str
    allow_special: bool = True


def build_chat_segments(messages, tools=None, add_generation_prompt=True, thinking=True, effort="high", **_):
    if not add_generation_prompt:
        raise ValueError("deepseek-v4.1-flash renders generation prompts only")
    messages = [dict(message) for message in messages]
    if tools:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = list(tools)
    text = _ENCODING.encode_messages(messages, thinking_mode="thinking" if thinking else "chat",
                                     reasoning_effort=effort if thinking else None)
    return [EncodeSegment(text, True)]
