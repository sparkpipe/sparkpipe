#!/usr/bin/env python3
"""The memory bench through the chat layer. --setup-only imports and compacts each history once and saves the setups, while /health counts the
memory jobs in flight; an --isolated run then answers on those setups without importing again, takes every answer alone, once the engine has no
request in flight and the chat layer no memory job queued or running, and follows each with a turn in the same chat that finds the first turn in
the engine's cache. The summary joins the chat log: each answer's time to first token, the jobs that prepared its history, with the compaction's
job under the same chat as its answer, and the jobs after it. It lists the questions each arm missed and which of them the full history answered."""
import asyncio
import contextlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load("chat_harness_fixture", ROOT / "tests" / "chat_harness_fixture.py")
bench = load("chat_memory_bench", ROOT / "tools" / "chat_memory_bench.py")
SECRET = "b" * 48
ROWS = [{"question_id": "q-paris", "question_type": "single-session-user", "question": "Where did I go in May?", "answer": "Paris",
         "question_date": "2023/06/01", "haystack_dates": ["2023/05/01", "2023/05/02"],
         "haystack_sessions": [[{"role": "user", "content": "I went to Paris in May."}, {"role": "assistant", "content": "Nice, Paris in May."}],
                               [{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "It says hello world."}]]},
        {"question_id": "q-file", "question_type": "single-session-user", "question": "What did a.txt say?", "answer": "hello world",
         "question_date": "2023/06/02", "haystack_dates": ["2023/05/03", "2023/05/04"],
         "haystack_sessions": [[{"role": "user", "content": "Read a.txt"}, {"role": "assistant", "content": "It says hello world."}],
                               [{"role": "user", "content": "The weather in Paris is sunny today."}, {"role": "assistant", "content": "Lovely."}]]}]
OPEN = "<|im_start|>assistant\n<think>"
MEASURED = ("Answer briefly, from what was said in this conversation.<|im_end|>\n" + OPEN, bench.FOLLOW_UP + "<|im_end|>\n" + OPEN)


def respond(prompt):
    if "You compact conversations." in prompt:
        return "</think>" + json.dumps({"keep": [], "summary": "The user went to Paris in May (e1) and read a.txt (e2)."})
    if "Is the model response correct?" in prompt:
        expected, given = (prompt.split(label)[1].split("\n")[0].lower() for label in ("Correct Answer: ", "Model Response: "))
        return "</think>" + ("yes" if expected in given else "no")
    if prompt.endswith(MEASURED):
        if "<summary>" in prompt:
            return "</think>Paris."
        return "</think>" + ("Hello world." if prompt.rfind("What did a.txt say?") > prompt.rfind("Where did I go in May?") else "Paris.")
    return None


async def health(harness):
    async with harness.session.get(harness.base + "/health") as response:
        return (await response.json())["models"]["alpha"]


async def busy(harness):
    for _ in range(500):
        state = await health(harness)
        if (state["memory_jobs"]["queued"] or state["memory_jobs"]["running"]) and state["engine"]["live_requests"]:
            return state
        await asyncio.sleep(0.01)
    return state


def check_summary(check):
    records = [{"id": question, "kind": "single-session-user", "arm": arm, "correct": correct, "prompt_tokens": 100, "cached_tokens": 0, "seconds": 1.0}
               for question, arm, correct in [("q1", "full", True), ("q1", "compaction", True), ("q1", "memory", False),
                                              ("q2", "full", False), ("q2", "compaction", False), ("q2", "memory", True),
                                              ("q3", "full", True), ("q3", "compaction", False), ("q3", "memory", True), ("q4", "full", True)]]
    found = bench.misses(records)
    check(found == {"questions": 3, "missed": {"full": ["q2"], "compaction": ["q2", "q3"], "memory": ["q1"]},
                    "compaction_not_full": ["q3"], "full_not_compaction": [], "memory_not_full": ["q1"], "full_not_memory": ["q2"]},
          f"misses compare each arm with the full history over the questions every arm answered: {found}")
    arms = bench.summary(records)
    check(arms["memory"]["all"]["missed"] == ["q1"] and arms["full"]["all"]["missed"] == ["q2"],
          f"each arm lists the questions it missed: {arms['memory']['all']['missed']} {arms['full']['all']['missed']}")


async def run(tmp, check):
    secret = tmp / "memory.secret"
    secret.write_text(SECRET)
    layer = tmp / "layer"
    layer.mkdir()
    (layer / "config.json").write_text(json.dumps({"chat_memory": {"jwt_secret_file": str(secret)}}))
    data = tmp / "longmemeval.json"
    data.write_text(json.dumps(ROWS))
    out = tmp / "out"
    store = tmp / "memory-store"
    memory = {"store": str(store), "jwt_secret_file": str(secret), "idle_seconds": 0.01, "budgets": fixture.MEMORY_BUDGETS, "background_with_foreground": True}
    log = tmp / "chat.log"
    bench.IDLE_POLL_SECONDS = 0.02
    argv = ["--dataset", "longmemeval", "--data", str(data), "--model", "alpha", "--per-type", "2", "--ids", "q-paris,q-file", "--parallel", "2",
            "--effort", "low", "--layer", str(layer), "--out", str(out)]
    async with fixture.Harness(tmp, extra={"chat_memory": memory}) as harness:
        engine = harness.engine
        engine.responder = respond
        engine.delay = 0.03
        argv += ["--url", harness.base]
        with log.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            engine.hold = asyncio.Event()
            setups = asyncio.ensure_future(asyncio.to_thread(bench.main, argv + ["--setup-only"]))
            held = await busy(harness)
            engine.hold.set()
            engine.hold = None
            await setups
            check(held["memory_jobs"]["running"] + held["memory_jobs"]["queued"] >= 1 and held["engine"]["live_requests"] >= 1,
                  f"/health counts the memory jobs in flight beside the engine's live requests: {held}")
            idle = await health(harness)
            check(idle["memory_jobs"] == {"queued": 0, "running": 0}, f"and none once the setups finish: {idle['memory_jobs']}")
            saved = [json.loads(line) for line in (out / "setup.jsonl").read_text().splitlines()]
            check(sorted(entry["arm"] for entry in saved) == ["compaction", "compaction", "memory", "memory"] and not (out / "records.jsonl").read_text(),
                  f"--setup-only saves one import and one compaction per history and answers nothing: {[entry['arm'] for entry in saved]}")
            chats = sorted(path.name for path in store.rglob("*.jsonl"))
            first = len(engine.requests)

            await asyncio.to_thread(bench.main, argv + ["--isolated"])
        records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
        check(len(records) == 12 and {(r["id"], r["arm"], r["turn"]) for r in records} ==
              {(question, arm, turn) for question in ("q-paris", "q-file") for arm in ("full", "compaction", "memory") for turn in (1, 2)},
              f"--isolated answers each question once per arm and follows each answer up: {[(r['id'], r['arm'], r['turn']) for r in records]}")
        check(sorted(path.name for path in store.rglob("*.jsonl")) == chats and len((out / "setup.jsonl").read_text().splitlines()) == 4,
              "it answers on the saved setups and imports nothing again")
        firsts = [r for r in records if r["turn"] == 1 and r["arm"] != "full"]
        check(all(r[bench.SETUP[r["arm"]] + "_seconds"] > 0 and not r[bench.SETUP[r["arm"]] + "_reused"] for r in firsts),
              f"each history's first answer carries the setup it was prepared with: {[(r['arm'], r.get('import_seconds', r.get('compact_seconds'))) for r in firsts]}")
        measured = [index for index in range(first, len(engine.requests)) if harness.prompt_text(index).endswith(MEASURED)]
        check(len(measured) == 12 and all(engine.arrivals[index] == 0 for index in measured),
              f"every measured request reaches the engine alone: {[(index, engine.arrivals[index]) for index in measured]}")
        check(sum("<summary>" in harness.prompt_text(index) for index in measured) == 4, "the compaction arm answers from its summary")
        turns = {(r["id"], r["arm"], r["turn"]): r for r in records}
        warm = [(key, turns[key]["cached_tokens"], turns[key[:2] + (1,)]["prompt_tokens"]) for key in turns if key[2] == 2]
        check(all(cached >= prompt - 1 for _, cached, prompt in warm), f"each follow-up finds its first turn in the engine's cache: {warm}")

        bench.main(argv + ["--summarize", "--chat-log", str(log)])
        report = json.loads((out / "summary.json").read_text())
        memory_arm, compaction_arm = report["arms"]["memory"], report["arms"]["compaction"]
        setups = {arm: round(sum(r[bench.SETUP[arm] + "_seconds"] for r in firsts if r["arm"] == arm) / 2, 1) for arm in bench.SETUP}
        check(all(report["arms"][arm]["all"]["setup_seconds_per_history"] == setups[arm] for arm in bench.SETUP)
              and report["arms"]["full"]["all"]["setup_seconds_per_history"] is None,
              f"each arm's setup time is the mean over the histories it prepared: {[report['arms'][arm]['all']['setup_seconds_per_history'] for arm in report['arms']]} {setups}")
        check("prewarm" in (memory_arm["all"].get("setup_jobs_per_history") or {}),
              f"the summary joins the import's jobs to the history's first answer: {memory_arm['all'].get('setup_jobs_per_history')}")
        check("compact" in (compaction_arm["all"].get("setup_jobs_per_history") or {}),
              f"and the compaction's job, logged under the same chat as its answer: {compaction_arm['all'].get('setup_jobs_per_history')}")
        check("prewarm" in (memory_arm["all"].get("jobs_after_each_answer") or {}),
              f"and the memory jobs after each answer: {memory_arm['all'].get('jobs_after_each_answer')}")
        check(all(report["arms"][arm]["all"]["mean_ttft_seconds"] is not None and report["arms"][arm]["follow_up"]["questions"] == 2
                  for arm in ("full", "compaction", "memory")), f"every arm has its answers' and follow-ups' times to first token: {report['arms']}")
        check(report["misses"] == {"questions": 2, "missed": {"full": [], "compaction": ["q-file"], "memory": []}, "compaction_not_full": ["q-file"],
                                   "full_not_compaction": [], "memory_not_full": [], "full_not_memory": []}, f"and the misses: {report['misses']}")


def main():
    checks = fixture.Checks("test_chat_memory_bench")
    check_summary(checks)
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(run(Path(tmp), checks))
    checks.finish("the memory bench prepares histories once, answers each alone with a warm follow-up and joins its chat log into misses, times and jobs")


if __name__ == "__main__":
    main()
