"""Measure chat memory on LongMemEval and LoCoMo through the chat layer: the full history, one-summary compaction and memory, on one model."""
import argparse
import collections
import json
import random
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


class Prepared:
    def __init__(self):
        self.lock = threading.Lock()
        self.locks = {}
        self.entries = {}

    def get(self, key, build):
        with self.lock:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            built = key not in self.entries
            if built:
                self.entries[key] = build()
            return self.entries[key], built


def turn_headers(headers, parent):
    turn = dict(headers, **{"X-OpenWebUI-User-Message-Id": str(uuid.uuid4()), "X-OpenWebUI-Message-Id": str(uuid.uuid4())})
    if parent:
        turn["X-OpenWebUI-User-Message-Parent-Id"] = parent
    return turn


def answer(args, item, arm, secret, memory, prepared):
    question = {"role": "user", "content": ASK.format(item["date"], item["question"])}
    record = {"id": item["id"], "kind": item["kind"], "arm": arm}
    user = f"bench-{args.model}"
    history = memory.exchange_key(item["history"])
    if arm == "full":
        messages = [{"role": "system", "content": SYSTEM + memory.MEMORY_OFF}] + item["history"] + [question]
        record["reply"], record["prompt_tokens"], record["cached_tokens"], record["seconds"] = qualify.ask(args.url + "/v1/chat/completions", args.model,
                                                                                                          messages, {}, args.effort)
        return record
    if arm == "memory":
        def imported():
            chat, headers = identity(secret, user)
            result, seconds = post(args.url + "/v1/sparkpipe/memory/import",
                                   {"model": args.model, "messages": [{"role": "system", "content": SYSTEM}] + item["history"], "batch": args.batch,
                                    "reasoning_effort": args.effort}, headers)
            return {"chat": chat, "import": result, "seconds": seconds}
        entry, built = prepared.get(("memory", history), imported)
        record.update(import_seconds=entry["seconds"] if built else 0.0, import_reused=not built, **({"import": entry["import"]} if built else {}))
        _, headers = identity(secret, user, entry["chat"])
        last = memory.exchanges_of(item["history"])[1][-1]
        messages = [{"role": "system", "content": SYSTEM}] + item["history"] + [question]
        reply = qualify.ask(args.url + "/v1/chat/completions", args.model, messages, turn_headers(headers, "import-reply-" + memory.exchange_key(last)[:24]),
                            args.effort)
    else:
        base = [{"role": "system", "content": SYSTEM + memory.MEMORY_OFF}] + item["history"]

        def compacted():
            chat, headers = identity(secret, user)
            _, _, _, seconds = qualify.ask(args.url + "/v1/chat/completions", args.model, base + [{"role": "user", "content": "/compact full"}],
                                           turn_headers(headers, None), args.effort)
            return {"chat": chat, "seconds": seconds}
        entry, built = prepared.get(("compaction", history), compacted)
        record.update(compact_seconds=entry["seconds"] if built else 0.0, compact_reused=not built)
        _, headers = identity(secret, user, entry["chat"])
        reply = qualify.ask(args.url + "/v1/chat/completions", args.model, base + [question], turn_headers(headers, None), args.effort)
    record["reply"], record["prompt_tokens"], record["cached_tokens"], record["seconds"] = reply
    return record


def judge(args, item, record):
    template = JUDGE[item["judge"]]
    prompt = template.format(item["question"], item["answer"], record["reply"])
    verdict, _, _, _ = qualify.ask(args.url + "/v1/chat/completions", args.judge_model, [{"role": "user", "content": prompt}], {}, args.effort)
    record["correct"] = verdict.strip().lower().startswith("yes")
    return record


def first_tokens(chat_log, records):
    seen = collections.defaultdict(list)
    for line in chat_log.read_text(errors="replace").splitlines():
        if '"event": "chat"' in line:
            event = json.loads(line[line.index("{"):])
            seen[(event.get("prompt_tokens"), event.get("cached_tokens"))].append(event)
    for record in records:
        found = seen.get((record["prompt_tokens"], record["cached_tokens"]))
        if found:
            event = found.pop(0)
            record.update(ttft_seconds=event.get("ttft_s"), decode_tokens_per_second=event.get("decode_tok_s"),
                          engine_rounds=event.get("engine_rounds", 1), new_prefill_tokens=event.get("engine_prefill_tokens"))
    return records


def new_prefill(record):
    joined = record.get("new_prefill_tokens")
    return record["prompt_tokens"] - record["cached_tokens"] if joined is None else joined


def summary(records):
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        for key in (record["kind"], "all"):
            table[record["arm"]][key].append(record)
    report = {}
    for arm, kinds in table.items():
        report[arm] = {kind: {"questions": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
                              "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
                              "mean_new_prefill_tokens": round(sum(new_prefill(r) for r in rows) / len(rows)),
                              "answers_with_memory_rounds": sum(1 for r in rows if (r.get("engine_rounds") or 1) > 1),
                              "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
                              "mean_ttft_seconds": (round(sum(r["ttft_seconds"] for r in rows if r.get("ttft_seconds") is not None)
                                                          / max(1, sum(1 for r in rows if r.get("ttft_seconds") is not None)), 2)
                                                    if any(r.get("ttft_seconds") is not None for r in rows) else None),
                              "setup_seconds": round(sum(r.get("import_seconds", 0.0) + r.get("compact_seconds", 0.0) for r in rows), 1)}
                       for kind, rows in kinds.items()}
    return report


def main():
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
    parser.add_argument("--batch", type=int, default=8, help="exchanges per curator call when the memory arm imports a history")
    parser.add_argument("--parallel", type=int, default=1, help="questions in flight at once; a history is imported or compacted once and shared")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--url", default="http://127.0.0.1:8433")
    parser.add_argument("--layer", type=Path, default=Path.home() / "sparkpipe-chat")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chat-log", type=Path, help="the chat layer's log, to join each answer's time to first token into the summary")
    parser.add_argument("--summarize", action="store_true", help="only summarize the records already saved in --out")
    args = parser.parse_args()
    args.judge_model = args.judge_model or args.model
    if args.summarize:
        records = [json.loads(line) for line in (args.out / "records.jsonl").read_text().splitlines() if line.strip()]
        write_summary(args, first_tokens(args.chat_log, records) if args.chat_log else records)
        return
    memory = qualify.load("chat_memory", Path(__file__).resolve().parent / "serving" / "chat_memory.py")
    config = json.loads((args.layer / "config.json").read_text())
    secret = Path(config["chat_memory"]["jwt_secret_file"]).read_text().strip()
    items = longmemeval(args.data, args.per_type, args.seed) if args.dataset == "longmemeval" else \
        locomo(args.data, args.conversations, args.per_conversation, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / "records.jsonl"
    records = [json.loads(line) for line in saved.read_text().splitlines() if line.strip()] if saved.exists() else []
    done = {(record["id"], record["arm"]) for record in records}
    written, prepared = threading.Lock(), Prepared()
    with saved.open("a") as handle:
        def run(item):
            for arm in args.arms.split(","):
                if (item["id"], arm) in done:
                    continue
                record = judge(args, item, answer(args, item, arm, secret, memory, prepared))
                with written:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                print(json.dumps({k: record.get(k) for k in ("id", "kind", "arm", "correct", "prompt_tokens", "cached_tokens", "seconds")}), flush=True)
        with ThreadPoolExecutor(args.parallel) as pool:
            for future in [pool.submit(run, item) for item in items]:
                future.result()
    write_summary(args, first_tokens(args.chat_log, records) if args.chat_log else records)


def write_summary(args, records):
    report = {"dataset": args.dataset, "model": args.model, "judge": args.judge_model, "judge_prompts": LONGMEMEVAL_SOURCE, "seed": args.seed,
              "arms": summary(records)}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["arms"], indent=2))


if __name__ == "__main__":
    main()
