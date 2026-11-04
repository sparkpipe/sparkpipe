#!/usr/bin/env python3
import importlib.util
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("chat_replay", ROOT / "tools" / "serving" / "chat_replay.py")
chat_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat_replay)
STOP = 9


def main():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    chain = chat_replay.Chain("model", [{"type": "function", "function": {"name": "read"}}], {"effort": "high"})
    user = {"role": "user", "content": "hi"}
    turn = {"role": "assistant", "content": "  Checking.\n", "reasoning_content": "think", "tool_calls": [{"id": "t1", "function": {"name": "read", "arguments": '{"path": "a"}'}}]}
    same = {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read", "arguments": {"path": "a"}}}]}
    check(chain.digests([user, turn]) == chain.digests([user, same]),
          "an assistant turn is identified by its text and tool calls, whatever the client does with whitespace, reasoning or argument encoding")
    check(chain.digests([user, turn])[-1] != chain.digests([dict(user, content="hi!"), turn])[-1], "a changed earlier message changes the digest of every later turn")
    check(chat_replay.Chain("model", None, {}).digests([user]) != chat_replay.Chain("model", None, {"effort": "low"}).digests([user]),
          "the tools and template arguments are part of the digest")
    check(chat_replay.Chain("model", None, {}).digests([{"role": "assistant", "content": None}]) ==
          chat_replay.Chain("model", None, {}).digests([{"role": "assistant", "content": [{"type": "text", "text": ""}]}]),
          "an empty assistant text is the same as a null or empty text part list")
    check(chat_replay.splice([1, 2, 3, STOP], [STOP, 5, 6], {STOP}) == ([1, 2, 3], [STOP, 5, 6]),
          "a reply's stop token is taken from the template's turn end, not doubled")
    check(chat_replay.splice([1, 2, 3], [3, 4], {STOP}) == ([1, 2, 3], [4]), "a template turn end that repeats the reply's last tokens is not doubled")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state" / "replay.sqlite"
        store = chat_replay.ReplayStore(path, budget_bytes=1 << 20)
        digests = [bytes([index]) * 32 for index in range(4)]
        first = store.put("a", digests[0], "model", None, 0, [1, 2, 3, STOP])
        second = store.put("a", digests[1], "model", first, 3, [4, 5, STOP])
        third = store.put("a", digests[2], "model", second, 5, [6, 7])
        check(list(store.context(third)) == [1, 2, 3, 4, 5, 6, 7], "a turn's context is its parent's kept tokens plus its own")
        check(store.find("a", digests[2]) == third and store.find("b", digests[2]) is None, "a turn is found only in its own account")
        check(oct(path.parent.stat().st_mode & 0o777) == "0o700", "the store directory is private to the chat layer")
        store.put_response("a", "resp_1", None, "model", [{"role": "user", "content": "one"}])
        store.put_response("a", "resp_2", "resp_1", "model", [{"role": "assistant", "content": "two"}])
        check(store.response_messages("a", "resp_2") == [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
              "a stored response rebuilds its whole conversation through its parents")
        check(store.response_messages("b", "resp_2") is None, "another account cannot refer to a stored response")
        store.set_aliases("a", {"claude-*": "x"})
        store.db.close()
        store = chat_replay.ReplayStore(path, budget_bytes=1 << 20)
        check(list(store.context(third)) == [1, 2, 3, 4, 5, 6, 7] and store.aliases("a") == {"claude-*": "x"},
              "turns, responses and aliases survive a restart")

        store.observe_prefill("model", 1000, 400, 4000)
        check(store.prefill_rate("model", 1000) is None, "a mostly cached request is not a cold prefill measurement")
        store.observe_prefill("model", 1000, 1000, 4000)
        store.observe_prefill("model", 1000, 1000, 2000)
        rate = store.prefill_rate("model", 1100)
        check(rate is not None and 2.0 < rate < 4.0, f"cold prefills at a length give a smoothed rate for nearby lengths: {rate}")
        check(store.prefill_rate("model", 100000) is None, "no rate is guessed for a length far from any measurement")

        small = chat_replay.ReplayStore(None, budget_bytes=4 * 35)
        root = small.put("a", digests[0], "model", None, 0, list(range(10)))
        old = small.put("a", digests[1], "model", None, 0, list(range(100, 110)))
        time.sleep(0.01)
        child = small.put("a", digests[2], "model", root, 10, list(range(10, 20)))
        time.sleep(0.01)
        small.put("a", digests[3], "model", child, 20, list(range(20, 30)))
        check(small.find("a", digests[1]) is None and small.find("a", digests[0]) is not None and small.find("a", digests[2]) is not None,
              "over budget the least recently used turn goes, and a chain in use keeps its ancestors")
        small.clear("a")
        check(small.turns() == 0 and small.bytes == 0, "clearing an account drops its turns")
        mixed = chat_replay.ReplayStore(None)
        mixed.put("a", digests[0], "model", None, 0, [1])
        mixed.put("b", digests[0], "model", None, 0, [2])
        mixed.set_aliases("b", {"x": "y"})
        mixed.clear("b")
        check(mixed.turns("a") == 1 and mixed.turns("b") == 0 and mixed.aliases("b") is None, "clearing one account leaves the others")
    for failure in failures:
        print("FAIL test_chat_replay_store: " + failure)
    if failures:
        return 1
    print("PASS test_chat_replay_store: turns are addressed by a digest of the conversation that ignores whitespace, reasoning and argument encoding, chain "
          "to their parents, survive a restart with responses and aliases, are scoped per account, evict least recently used without breaking a chain in "
          "use, and splice without doubling the turn end; cold prefills give a measured rate near their length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
