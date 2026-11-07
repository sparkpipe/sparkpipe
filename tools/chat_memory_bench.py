"""Measure chat memory on LongMemEval and LoCoMo through the chat layer: the full history, one-summary compaction and memory, on one model."""
import argparse
import collections
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_memory_qualify as qualify

LONGMEMEVAL_SOURCE = "xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py at 9e0b455f4ef0, MIT"
JUDGE_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
              "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
              "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. ")
JUDGE_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE = {
    "single-session-user": JUDGE_BASE + JUDGE_TAIL,
    "single-session-assistant": JUDGE_BASE + JUDGE_TAIL,
    "multi-session": JUDGE_BASE + JUDGE_TAIL,
    "temporal-reasoning": JUDGE_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    + JUDGE_TAIL,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct "
    "answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as "
    "correct as long as the updated answer is the required answer." + JUDGE_TAIL,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes "
    "if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly "
    "identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked "
    "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only.",
}
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal-reasoning", 3: "open-domain", 4: "single-hop"}
SYSTEM = "You are a helpful assistant. Each session of this conversation starts with its date."
ASK = "Current date: {}\nQuestion: {}\nAnswer briefly, from what was said in this conversation."
FOLLOW_UP = "Thank you. Repeat your answer in one short sentence."
SETUP = {"memory": "import", "compaction": "compact"}
JOB_FIELDS = ("jobs", "new_prefill_tokens", "output_tokens", "seconds")
IDLE_POLL_SECONDS = 0.5
IDLE_QUIET_POLLS = 2
IDLE_TIMEOUT_SECONDS = 3600


def session_messages(sessions):
    messages = []
    for date, turns in sessions:
        if turns and turns[0][0] == "assistant":
            turns = [("user", "")] + list(turns)
        for index, (role, text) in enumerate(turns):
            content = f"[Session date: {date}]\n{text}".rstrip() if index == 0 else text
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})
        if messages and messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": "(no reply)"})
    return messages


def longmemeval(path, per_type, seed):
    by_type = collections.defaultdict(list)
    for row in json.loads(Path(path).read_text()):
        kind = "abstention" if row["question_id"].endswith("_abs") else row["question_type"]
        sessions = [(date, [(turn["role"], turn["content"]) for turn in session]) for date, session in zip(row["haystack_dates"], row["haystack_sessions"])]
        by_type[kind].append({"id": row["question_id"], "kind": kind, "judge": kind, "question": row["question"], "answer": str(row["answer"]),
                              "date": row["question_date"], "history": session_messages(sessions)})
    rng = random.Random(seed)
    chosen = [rng.sample(by_type[kind], min(per_type, len(by_type[kind]))) for kind in sorted(by_type)]
    return [group[index] for index in range(max(map(len, chosen))) for group in chosen if index < len(group)]


def locomo(path, conversations, per_conversation, seed):
    rng, items = random.Random(seed), []
    for conversation in json.loads(Path(path).read_text())[:conversations]:
        talk = conversation["conversation"]
        speakers = {talk["speaker_a"]: "user", talk["speaker_b"]: "assistant"}
        numbers = sorted(int(key.split("_")[1]) for key in talk if key.startswith("session_") and not key.endswith("date_time"))
        sessions = [(talk[f"session_{n}_date_time"], [(speakers[turn["speaker"]], f"{turn['speaker']}: {turn['text']}") for turn in talk[f"session_{n}"]])
                    for n in numbers]
        history = session_messages(sessions)
        asked = [qa for qa in conversation["qa"] if qa.get("category") in LOCOMO_CATEGORIES and "answer" in qa]
        for index, qa in enumerate(rng.sample(asked, min(per_conversation, len(asked)))):
            items.append({"id": f"{conversation['sample_id']}-{index}", "kind": LOCOMO_CATEGORIES[qa["category"]], "judge": "single-session-user",
                          "question": qa["question"], "answer": str(qa["answer"]), "date": sessions[-1][0], "history": history})
    return items


def post(url, body, headers):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=dict(headers, **{"Content-Type": "application/json"}))
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            return json.load(response), time.monotonic() - started
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code} from {url}: {error.read().decode(errors='replace')[:2000]}")


def identity(secret, user, chat=None):
    chat = chat or str(uuid.uuid4())
    return chat, {"X-OpenWebUI-User-Jwt": qualify.token(secret, user), "X-OpenWebUI-Chat-Id": chat, "X-OpenWebUI-Task": ""}


def load_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


class Prepared:
    def __init__(self, path, records):
        self.path, self.lock, self.locks, self.entries = path, threading.Lock(), {}, {}
        self.claimed = {(record["arm"], record["history"]) for record in records if record["arm"] in SETUP and "history" in record
                        and record.get("turn", 1) == 1 and not record.get(SETUP[record["arm"]] + "_reused", True)}
        for entry in load_records(path):
            self.entries[(entry["arm"], entry["history"])] = entry

    def get(self, key, build, claim):
        with self.lock:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self.entries:
                entry = dict(build(), arm=key[0], history=key[1])
                with self.lock:
                    self.entries[key] = entry
                    with self.path.open("a") as handle:
                        handle.write(json.dumps(entry) + "\n")
        with self.lock:
            first = claim and key not in self.claimed
            if claim:
                self.claimed.add(key)
            return self.entries[key], first


def turn_headers(headers, parent):
    turn = dict(headers, **{"X-OpenWebUI-User-Message-Id": str(uuid.uuid4()), "X-OpenWebUI-Message-Id": str(uuid.uuid4())})
    if parent:
        turn["X-OpenWebUI-User-Message-Parent-Id"] = parent
    return turn


def conversation(item, memory, remembered):
    return [{"role": "system", "content": SYSTEM if remembered else SYSTEM + memory.MEMORY_OFF}] + item["history"]


def setup(args, item, arm, secret, memory, prepared, claim=False):
    user = f"bench-{args.model}"

    def imported():
        chat, headers = identity(secret, user)
        result, seconds = post(args.url + "/v1/sparkpipe/memory/import",
                               {"model": args.model, "messages": conversation(item, memory, True), "batch": args.batch,
                                "reasoning_effort": args.effort}, headers)
        return {"chat": chat, "import": result, "seconds": seconds}

    def compacted():
        chat, headers = identity(secret, user)
        _, _, _, seconds = qualify.ask(args.url + "/v1/chat/completions", args.model,
                                       conversation(item, memory, False) + [{"role": "user", "content": "/compact full"}], turn_headers(headers, None), args.effort)
        return {"chat": chat, "seconds": seconds}
    return prepared.get((arm, memory.exchange_key(item["history"])), imported if arm == "memory" else compacted, claim)


def answer(args, item, arm, secret, memory, prepared, before=None):
    question = {"role": "user", "content": ASK.format(item["date"], item["question"])}
    record = {"id": item["id"], "kind": item["kind"], "arm": arm, "turn": 1 if before is None else 2, "history": memory.exchange_key(item["history"])}
    messages = conversation(item, memory, arm == "memory") + [question]
    headers, parent = {}, None
    if arm in SETUP:
        entry, first = setup(args, item, arm, secret, memory, prepared, claim=before is None)
        if before is None:
            record.update({SETUP[arm] + "_seconds": entry["seconds"] if first else 0.0, SETUP[arm] + "_reused": not first},
                          **({"import": entry["import"]} if first and "import" in entry else {}))
        record["chat"] = entry["chat"]
        _, headers = identity(secret, f"bench-{args.model}", entry["chat"])
        if arm == "memory":
            parent = "import-reply-" + memory.exchange_key(memory.exchanges_of(item["history"])[1][-1])[:24]
    if before is not None:
        messages += [{"role": "assistant", "content": before["reply"]}, {"role": "user", "content": FOLLOW_UP}]
        parent = before.get("message")
    if headers:
        headers = turn_headers(headers, parent)
        record["message"] = headers["X-OpenWebUI-Message-Id"]
    record["reply"], record["prompt_tokens"], record["cached_tokens"], record["seconds"] = qualify.ask(args.url + "/v1/chat/completions", args.model,
                                                                                                      messages, headers, args.effort)
    return record


def judge(args, item, record):
    template = JUDGE[item["judge"]]
    prompt = template.format(item["question"], item["answer"], record["reply"])
    verdict, _, _, _ = qualify.ask(args.url + "/v1/chat/completions", args.judge_model, [{"role": "user", "content": prompt}], {}, args.effort)
    record["correct"] = verdict.strip().lower().startswith("yes")
    return record


def wait_idle(args):
    deadline, quiet = time.monotonic() + IDLE_TIMEOUT_SECONDS, 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url + "/health", timeout=30) as response:
                state = json.load(response)["models"][args.model]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
            state = {}
        if state.get("status") == 200 and (state.get("engine") or {}).get("live_requests") is None:
            raise SystemExit(f"{args.url}/health reports no live_requests for {args.model}, so the bench cannot tell when the engine is idle")
        jobs = state.get("memory_jobs") or {}
        idle = state.get("status") == 200 and state["engine"]["live_requests"] == 0 and not jobs.get("queued") and not jobs.get("running")
        quiet = quiet + 1 if idle else 0
        if quiet >= IDLE_QUIET_POLLS:
            return
        time.sleep(IDLE_POLL_SECONDS)
    raise SystemExit(f"the engine and the chat layer's memory jobs did not go idle within {IDLE_TIMEOUT_SECONDS} s")


def tally(events):
    kinds = {}
    for event in events:
        row = kinds.setdefault(event.get("kind"), dict.fromkeys(JOB_FIELDS, 0))
        row["jobs"] += 1
        row["new_prefill_tokens"] += (event.get("prompt_tokens") or 0) - (event.get("cached_tokens") or 0)
        row["output_tokens"] += event.get("completion_tokens") or 0
        row["seconds"] = round(row["seconds"] + (event.get("seconds") or 0.0), 3)
    return kinds


def join_log(chat_log, records):
    answers, asked, jobs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for position, event in enumerate(qualify.events(chat_log, 0, '"event": "')):
        if event.get("event") == "chat":
            answers[(event.get("prompt_tokens"), event.get("cached_tokens"))].append((position, event))
            asked[event.get("chat")].append(position)
        elif event.get("event") == "memory_job":
            jobs[event.get("chat")].append((position, event))
    for record in records:
        found = answers.get((record["prompt_tokens"], record["cached_tokens"]))
        if not found:
            continue
        position, event = found.pop(0)
        record.update(ttft_seconds=event.get("ttft_s"), decode_tokens_per_second=event.get("decode_tok_s"),
                      engine_rounds=event.get("engine_rounds", 1), new_prefill_tokens=event.get("engine_prefill_tokens"))
        chat = event.get("chat")
        if chat is None:
            continue
        later = min((other for other in asked[chat] if other > position), default=math.inf)
        if not any(other < position for other in asked[chat]) and not record.get(SETUP.get(record["arm"], "") + "_reused", True):
            record["setup_jobs"] = tally(job for at, job in jobs[chat] if at < position)
        record["after_jobs"] = tally(job for at, job in jobs[chat] if position < at < later)
    return records


def new_prefill(record):
    joined = record.get("new_prefill_tokens")
    return record["prompt_tokens"] - record["cached_tokens"] if joined is None else joined


def timing(rows):
    ttfts = [row["ttft_seconds"] for row in rows if row.get("ttft_seconds") is not None]
    prefill = [new_prefill(row) for row in rows]
    return {"questions": len(rows), "mean_prompt_tokens": round(statistics.mean(row["prompt_tokens"] for row in rows)),
            "mean_new_prefill_tokens": round(statistics.mean(prefill)), "median_new_prefill_tokens": round(statistics.median(prefill)),
            "answers_with_memory_rounds": sum(1 for row in rows if (row.get("engine_rounds") or 1) > 1),
            "mean_seconds": round(statistics.mean(row["seconds"] for row in rows), 2),
            "mean_ttft_seconds": round(statistics.mean(ttfts), 2) if ttfts else None,
            "median_ttft_seconds": round(statistics.median(ttfts), 2) if ttfts else None}


def per_answer(tallies):
    totals = [dict(found, total={field: sum(row[field] for row in found.values()) for field in JOB_FIELDS}) for found in tallies]
    return {kind: {field: round(statistics.mean(found.get(kind, dict.fromkeys(JOB_FIELDS, 0))[field] for found in totals), 1) for field in JOB_FIELDS}
            for kind in sorted({kind for found in totals for kind in found})} if totals else None


def stats(rows):
    setups = [row[SETUP[row["arm"]] + "_seconds"] for row in rows if row["arm"] in SETUP and not row.get(SETUP[row["arm"]] + "_reused", True)]
    result = dict(timing(rows), accuracy=round(sum(row["correct"] for row in rows) / len(rows), 4), missed=sorted(row["id"] for row in rows if not row["correct"]),
                  setup_seconds_per_history=round(statistics.mean(setups), 1) if setups else None)
    for name, field in (("setup_jobs_per_history", "setup_jobs"), ("jobs_after_each_answer", "after_jobs")):
        found = per_answer([row[field] for row in rows if row.get(field)])
        if found:
            result[name] = found
    return result


def summary(records):
    report = {}
    for arm in dict.fromkeys(record["arm"] for record in records):
        answers = [record for record in records if record["arm"] == arm and record.get("turn", 1) == 1]
        if not answers:
            continue
        kinds = collections.defaultdict(list)
        for record in answers:
            kinds[record["kind"]].append(record)
        report[arm] = dict({kind: stats(rows) for kind, rows in sorted(kinds.items())}, all=stats(answers))
        follow_ups = [record for record in records if record["arm"] == arm and record.get("turn", 1) == 2]
        if follow_ups:
            report[arm]["follow_up"] = dict(timing(follow_ups), jobs_after_each_answer=per_answer([row["after_jobs"] for row in follow_ups if row.get("after_jobs")]))
    return report


def misses(records):
    answers = {(record["id"], record["arm"]): record for record in records if record.get("turn", 1) == 1}
    arms = list(dict.fromkeys(arm for _, arm in answers))
    complete = sorted({question for question, _ in answers if all((question, arm) in answers for arm in arms)})
    missed = {arm: [question for question in complete if not answers[(question, arm)]["correct"]] for arm in arms}
    report = {"questions": len(complete), "missed": missed}
    for arm in arms:
        if arm != "full" and "full" in missed:
            report[arm + "_not_full"] = sorted(set(missed[arm]) - set(missed["full"]))
            report["full_not_" + arm] = sorted(set(missed["full"]) - set(missed[arm]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", help="the judge; defaults to the model under test")
    parser.add_argument("--arms", default="full,compaction,memory")
    parser.add_argument("--per-type", type=int, default=10, help="LongMemEval questions per question type, taken one type at a time so a partial run stays balanced")
    parser.add_argument("--conversations", type=int, default=10, help="LoCoMo conversations")
    parser.add_argument("--per-conversation", type=int, default=20, help="LoCoMo questions per conversation")
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--ids", help="comma-separated question ids from the drawn sample, to answer only those")
    parser.add_argument("--batch", type=int, default=8, help="exchanges per curator call when the memory arm imports a history")
    parser.add_argument("--parallel", type=int, default=1, help="questions in flight at once; a history is imported or compacted once and shared")
    parser.add_argument("--setup-only", action="store_true", help="only import and compact the histories, in parallel, so a later run answers on them")
    parser.add_argument("--isolated", action="store_true",
                        help="prepare every history first, then take each answer alone once the engine and the chat layer's memory jobs are idle, "
                             "each followed by a follow-up turn in the same chat")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--url", default="http://127.0.0.1:8433")
    parser.add_argument("--layer", type=Path, default=Path.home() / "sparkpipe-chat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chat-log", type=Path, help="the chat layer's log, to join each answer's time to first token and its memory jobs into the summary")
    parser.add_argument("--summarize", action="store_true", help="only summarize the records already saved in --out")
    args = parser.parse_args(argv)
    args.judge_model = args.judge_model or args.model
    if args.summarize:
        records = load_records(args.out / "records.jsonl")
        write_summary(args, join_log(args.chat_log, records) if args.chat_log else records)
        return
    memory = qualify.load("chat_memory", Path(__file__).resolve().parent / "serving" / "chat_memory.py")
    config = json.loads((args.layer / "config.json").read_text())
    secret = Path(config["chat_memory"]["jwt_secret_file"]).read_text().strip()
    items = longmemeval(args.data, args.per_type, args.seed) if args.dataset == "longmemeval" else \
        locomo(args.data, args.conversations, args.per_conversation, args.seed)
    if args.ids:
        wanted = set(args.ids.split(","))
        if wanted - {item["id"] for item in items}:
            raise SystemExit(f"--ids names questions the drawn sample does not hold: {sorted(wanted - {item['id'] for item in items})}")
        items = [item for item in items if item["id"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    saved, arms = args.out / "records.jsonl", args.arms.split(",")
    records = load_records(saved)
    done = {(record["id"], record["arm"], record.get("turn", 1)): record for record in records}
    written, prepared = threading.Lock(), Prepared(args.out / "setup.jsonl", records)
    with saved.open("a") as handle:
        def keep(record):
            with written:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                done[(record["id"], record["arm"], record["turn"])] = record
            print(json.dumps({k: record.get(k) for k in ("id", "kind", "arm", "turn", "correct", "prompt_tokens", "cached_tokens", "seconds")}), flush=True)

        if args.setup_only or args.isolated:
            with ThreadPoolExecutor(args.parallel) as pool:
                for future in [pool.submit(setup, args, item, arm, secret, memory, prepared) for item in items for arm in arms if arm in SETUP]:
                    future.result()
            if args.setup_only:
                return
        if args.isolated:
            for item in items:
                for arm in arms:
                    for turn in (1, 2):
                        if (item["id"], arm, turn) not in done:
                            wait_idle(args)
                            record = answer(args, item, arm, secret, memory, prepared, done[(item["id"], arm, 1)] if turn == 2 else None)
                            keep(judge(args, item, record) if turn == 1 else record)
        else:
            def run(item):
                for arm in arms:
                    if (item["id"], arm, 1) not in done:
                        keep(judge(args, item, answer(args, item, arm, secret, memory, prepared)))
            with ThreadPoolExecutor(args.parallel) as pool:
                for future in [pool.submit(run, item) for item in items]:
                    future.result()
    write_summary(args, join_log(args.chat_log, records) if args.chat_log else records)


def write_summary(args, records):
    report = {"dataset": args.dataset, "model": args.model, "judge": args.judge_model, "judge_prompts": LONGMEMEVAL_SOURCE, "seed": args.seed,
              "arms": summary(records), "misses": misses(records)}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"arms": report["arms"], "misses": report["misses"]}, indent=2))


if __name__ == "__main__":
    main()
