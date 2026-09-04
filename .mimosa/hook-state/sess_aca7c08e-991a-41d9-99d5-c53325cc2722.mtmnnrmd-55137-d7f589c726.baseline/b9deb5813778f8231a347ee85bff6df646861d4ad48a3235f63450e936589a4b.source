#!/usr/bin/env python3

import argparse
import importlib.util
import pathlib
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sparkpipe_api_stress", ROOT / "tools" / "sparkpipe_api_stress.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StreamingResponse:
    def __init__(self):
        self.lines = iter([
            b"event: token\n",
            b'data: {"text":"ok"}\n',
            b"\n",
            b"event: done\n",
            b"data: {}\n",
            b"\n",
        ])

    def readline(self):
        return next(self.lines,b"")

    def read(self,count):
        raise AssertionError("streaming response used buffered read")


def arguments():
    return argparse.Namespace(
        concurrency=64,
        launch_spacing_ms=0.0,
        model="glm-5.2",
        requests=64,
        stream=True,
        temperature=0.0,
        max_completion_tokens=16,
    )


def main():
    chunks,body,first_byte_s = MODULE.read_response_chunks(
        StreamingResponse(),True,MODULE.time.monotonic())
    assert body.startswith(b"event: token\n")
    assert first_byte_s is not None
    token_events,done_events,text,events,ttft_s = MODULE.parse_sse_events(chunks)
    assert token_events == 1
    assert done_events == 1
    assert text == "ok"
    assert len(events) == 2
    assert ttft_s is not None
    barrier, worker_count = MODULE.make_start_barrier(arguments())
    assert barrier is not None
    assert barrier.parties == 65
    assert worker_count == 64
    spaced = arguments()
    spaced.launch_spacing_ms = 1.0
    barrier, worker_count = MODULE.make_start_barrier(spaced)
    assert barrier is None
    assert worker_count == 0
    chat = MODULE.build_request_body(
        arguments(),
        urllib.parse.urlparse("http://spark0:18080/v1/chat/completions"),
        "hello",
    )
    assert chat["messages"] == [{"role": "user", "content": "hello"}]
    assert chat["max_completion_tokens"] == 16
    assert "prompt" not in chat
    completion = MODULE.build_request_body(
        arguments(),
        urllib.parse.urlparse("http://spark0:18080/v1/completions"),
        "hello",
    )
    assert completion["prompt"] == "hello"
    assert completion["max_tokens"] == 16
    assert "messages" not in completion
    try:
        MODULE.build_request_body(
            arguments(), urllib.parse.urlparse("http://spark0:18080/other"), "hello"
        )
    except ValueError:
        return
    raise AssertionError("unsupported endpoint was accepted")


if __name__ == "__main__":
    main()
