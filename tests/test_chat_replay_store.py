#!/usr/bin/env python3
import importlib.util
import tempfile
import time
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "serving" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat_replay = load("chat_replay")
working_state = load("working_state")
STOP = 9
MODEL = "model"


def mode(path):
    return oct(path.stat().st_mode & 0o777)


def main():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    turn = {"role": "assistant", "content": "  Checking.\n", "reasoning_content": "think", "tool_calls": [{"id": "t1", "function": {"name": "read", "arguments": '{"path": "a"}'}}]}
    same = {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read", "arguments": {"path": "a"}}}]}
    view = chat_replay.view_digest(MODEL, turn)
    check(view == chat_replay.view_digest(MODEL, same), "a turn's view ignores whitespace around the text, reasoning and how the arguments are encoded")
    check(view != chat_replay.view_digest(MODEL, dict(same, content="Checking!")) and view != chat_replay.view_digest("other", same),
          "a changed reply text or another model changes the view")
    check(chat_replay.view_digest(MODEL, {"role": "assistant", "content": None}) == chat_replay.view_digest(MODEL, {"role": "assistant", "content": [{"type": "text", "text": ""}]}),
          "an empty assistant text is the same as a null or empty text part list")
    check(chat_replay.splice([1, 2, 3, STOP], [STOP, 5, 6], {STOP}) == ([1, 2, 3], [STOP, 5, 6]),
          "a reply's stop token is taken from the template's turn end, not doubled")
    check(chat_replay.splice([1, 2, 3], [3, 4], {STOP}) == ([1, 2, 3], [4]), "a template turn end that repeats the reply's last tokens is not doubled")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "state"
        root.mkdir(mode=0o755)
        store = chat_replay.ReplayStore(root, budget_bytes=1 << 20)
        generated = [101, 102, 103, STOP]
        turn_id = store.put("account:acme", MODEL, view, generated, ["sp1.abc.0", "toolu_1", "t1", "bad handle!"])
        check(list(store.find("account:acme", MODEL, ["toolu_1"], view)) == generated, "a turn is found by any handle it was given")
        check(list(store.find("account:acme", MODEL, [], view)) == generated, "a turn without a known handle is found by its reply, as chat completions sends it back")
        check(store.find("account:acme", MODEL, ["toolu_1"], chat_replay.view_digest(MODEL, dict(same, content="edited"))) is None,
              "a handle whose turn the client edited does not replay the stored tokens")
        check(store.find("account:other", MODEL, ["toolu_1"], view) is None and store.find("own", MODEL, ["toolu_1"], view) is None,
              "a turn is found only in its own account's store")
        check(store.find("account:acme", "other", ["toolu_1"], view) is None, "a turn replays only for the model that generated it")
        blob = store.scopes["account:acme"].db.execute("SELECT tokens FROM turns WHERE id = ?", (turn_id,)).fetchone()[0]
        check(array("I", blob).tolist() == generated, "the store keeps exactly the generated token ids, nothing of the prompt")
        files = sorted(path.name for path in root.iterdir())
        check("account-acme.sqlite" in files and chat_replay.PREFILL_FILE in files, f"each account has its own store file: {files}")
        check(mode(root) == "0o700", f"the store directory is private even when something else made it: {mode(root)}")
        check(all(mode(path) == "0o600" for path in root.iterdir()), f"every store file, with its -wal and -shm, is private: {[(path.name, mode(path)) for path in root.iterdir()]}")

        store.put_response("account:acme", "resp_1", None, MODEL, [{"role": "user", "content": "one"}])
        store.put_response("account:acme", "resp_2", "resp_1", MODEL, [{"role": "assistant", "content": "two"}])
        check(store.response_messages("account:acme", "resp_2") == [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
              "a stored response rebuilds its whole conversation through its parents")
        check(store.response_messages("account:other", "resp_2") is None, "another account cannot refer to a stored response")
        before = store.bytes
        store.put_response("account:acme", "resp_3", None, MODEL, [{"role": "user", "content": "x" * 1000}])
        check(store.bytes >= before + 1000, "stored responses count against the store's budget")
        store.set_aliases("account:acme", {"claude-*": "x"})
        store.close()
        store = chat_replay.ReplayStore(root, budget_bytes=1 << 20)
        check(list(store.find("account:acme", MODEL, ["sp1.abc.0"], view)) == generated and store.aliases("account:acme") == {"claude-*": "x"} and
              store.response_messages("account:acme", "resp_2") is not None, "turns, responses and aliases survive a restart")

        store.observe_prefill(MODEL, 1000, 400, 4000)
        check(store.prefill_rate(MODEL, 1000) is None, "a mostly cached request is not a cold prefill measurement")
        store.observe_prefill(MODEL, 1000, 1000, 4000)
        store.observe_prefill(MODEL, 1000, 1000, 2000)
        rate = store.prefill_rate(MODEL, 1100)
        check(rate is not None and 2.0 < rate < 4.0, f"cold prefills at a length give a smoothed rate for nearby lengths: {rate}")
        check(store.prefill_rate(MODEL, 100000) is None, "no rate is guessed for a length far from any measurement")

        marker = [0x5A5A5A5A] * 64
        store.put("own", MODEL, chat_replay.view_digest(MODEL, {"role": "assistant", "content": "keep"}), [7, 8], [])
        old = store.put("own", MODEL, chat_replay.view_digest(MODEL, {"role": "assistant", "content": "old"}), marker, ["toolu_old"])
        store.scopes["own"].db.execute("UPDATE turns SET used = ? WHERE id = ?", (time.time() - 100, old))
        for table in ("turns", "responses", "aliases"):
            store.scopes["account:acme"].db.execute(f"UPDATE {table} SET used = ?", (time.time() - 100,))
        wiped = time.time() - 50
        dropped = store.expire(lambda used: working_state.expired(used, wiped, 0, time.time()))
        raw = b"".join(path.read_bytes() for path in root.iterdir() if path.name.startswith("own"))
        check(dropped >= 1 and store.find("own", MODEL, ["toolu_old"], chat_replay.view_digest(MODEL, {"role": "assistant", "content": "old"})) is None,
              "a turn last used before the fleet's current user started is deleted")
        check(array("I", marker).tobytes() not in raw, "a deleted turn's tokens are overwritten in the store file, not left in free pages")
        check("account:acme" not in store.scopes and not (root / "account-acme.sqlite").exists() and not (root / "account-acme.sqlite-wal").exists(),
              "an account whose whole store expired has its files removed")
        store.put("own", MODEL, view, [1], [])
        store.scopes["own"].db.execute("UPDATE turns SET used = ?", (time.time() - 2 * 86400,))
        store.expire(lambda used: working_state.expired(used, 0, time.time() + 3600, time.time()))
        check(store.turns("own") > 0, "state stays while its account still holds the fleet, even after 24 hours without use")
        store.expire(lambda used: working_state.expired(used, 0, time.time() - 86401, time.time()))
        check(store.turns("own") == 0 and not (root / "own.sqlite").exists(), "state goes 24 hours after the later of its last use and the last hour held")

        store.put("key:k_1", MODEL, view, [1, 2], [])
        store.put("account:b", MODEL, view, [3, 4], [])
        store.clear("key:k_1")
        check(not (root / "key-k_1.sqlite").exists() and store.turns("account:b") == 1, "clearing one account removes its store and leaves the others")
        store.clear()
        check(store.turns() == 0 and not any(chat_replay.file_scope(path.name) for path in root.iterdir()), "clearing everything removes every account's store")
        store.close()

    small = chat_replay.ReplayStore(None, budget_bytes=4 * 25)
    views = [chat_replay.view_digest(MODEL, {"role": "assistant", "content": str(index)}) for index in range(4)]
    small.put("account:a", MODEL, views[0], list(range(10)), [])
    time.sleep(0.01)
    small.put("account:b", MODEL, views[1], list(range(10)), [])
    time.sleep(0.01)
    small.find("account:a", MODEL, [], views[0])
    small.scopes["account:a"].db.execute("UPDATE turns SET used = ?", (time.time(),))
    time.sleep(0.01)
    small.put("account:a", MODEL, views[2], list(range(10)), [])
    check(small.find("account:b", MODEL, [], views[1]) is None and small.find("account:a", MODEL, [], views[0]) is not None,
          "over budget the least recently used turn goes, whichever account holds it")
    check(small.bytes <= small.budget, f"the store stays within its budget: {small.bytes}")
    try:
        small.put("../escape", MODEL, views[3], [1], [])
        check(False, "a scope that is not an account, a key or own is refused")
    except ValueError:
        pass
    small.close()

    for failure in failures:
        print("FAIL test_chat_replay_store: " + failure)
    if failures:
        return 1
    print("PASS test_chat_replay_store: a turn keeps only its generated tokens, is found by its handles or its reply and never when the client edited it; "
          "each account has its own private store file; responses count against the budget; turns, responses and aliases survive a restart; the "
          "working-state rule deletes state used before the current user started or 24 hours after the last use and hour held, overwriting it and "
          "removing empty stores; eviction is least recently used across accounts; cold prefills give a measured rate near their length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
