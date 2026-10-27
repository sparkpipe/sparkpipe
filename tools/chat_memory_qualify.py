"""Qualify one model's chat memory through the production chat layer, as Open WebUI drives it."""
import argparse
import base64
import hashlib
import hmac
import importlib.util
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PEOPLE = ["Priya Natarajan", "Tomas Ekberg", "Lucia Ferrante", "Kwame Mensah", "Ingrid Solberg", "Haruto Sato", "Amara Okafor", "Elif Kaya"]
PLACES = ["Halcyon", "Brightwater", "Tamarind", "Quillon", "Sablefin", "Corvane", "Ostrel", "Vantorre", "Kestrin", "Silverfall", "Thornbury", "Merriden"]
VENDORS = ["Northwind Supply", "Bluepeak Metals", "Orbis Freight", "Cedarline", "Vexley & Sons", "Harmon Labs", "Stellar Parts", "Grayson Co"]
WORDS = ["copper", "lantern", "meadow", "falcon", "harbor", "quartz", "willow", "ember", "cobalt", "saffron", "juniper", "granite"]
FILLERS = ["What's a quick way to keep basil fresh?", "Explain what a hash map is in two sentences.", "Any tips for a first half marathon?",
           "Why is the sky blue?", "How do I convert Celsius to Fahrenheit?", "What's the difference between a list and a tuple in Python?",
           "Suggest a name for a grey cat.", "How long should I steep green tea?", "What does HTTP 429 mean?", "How do I undo my last git commit but keep the changes?",
           "Explain compound interest simply.", "What's the capital of Australia?"]


def note(rng, words):
    sentences = []
    while sum(len(sentence.split()) for sentence in sentences) < words:
        size = rng.randint(9, 22)
        sentence = " ".join(rng.choice(WORDS + PLACES + ["the", "a", "of", "and", "to", "in", "for", "with", "team", "report", "plan", "week", "review", "budget", "draft", "client", "schedule", "update"]) for _ in range(size))
        sentences.append(sentence[0].upper() + sentence[1:] + ".")
    return " ".join(sentences)


def make_chat(seed, note_words):
    rng = random.Random(seed)
    person, project, item = rng.choice(PEOPLE), rng.choice(PLACES), rng.choice(["standing desk", "server rack", "forklift lease", "solar panel"])
    room, code, date = rng.randint(200, 899), f"{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}", f"{rng.randint(3, 27)} November"
    vendors, prices = rng.sample(VENDORS, 4), [rng.randint(180, 990) * 10 for _ in range(4)]
    options = rng.sample(["Blue-Green Cutover", "Log Shipping Mirror", "Snapshot and Replay", "Dual-Write Bridge", "Read Replica Promotion"], 3)
    budget, corrected = rng.randint(20, 60) * 1000, rng.randint(61, 99) * 1000
    codeword = rng.choice(WORDS) + str(rng.randint(10, 99))
    facts = [
        (f"Here is a memo from {person}: the {project} quarterly review moves to {date}, charges go to budget code {code}, and we meet in room {room} on the fourth floor. Summarise it in one line.",
         f"Which room did {person}'s memo give for the {project} review? Reply with the number only.", [str(room)], []),
        ("Here are the " + item + " quotes we got: " + "; ".join(f"{v} {p} EUR" for v, p in zip(vendors, prices)) + ". Which is cheapest?",
         f"What price did {vendors[2]} quote for the {item}? Reply with the number only.", [str(prices[2]), f"{prices[2]:,}"], []),
        (f"For moving the {project} database we are weighing option A {options[0]}, option B {options[1]} and option C {options[2]}. Which carries the least risk?",
         f"For the {project} database move, what was option B called? Reply with its name only.", [options[1]], []),
        (f"For the record, the {project} budget is {budget} euros.", None, None, None),
        (f"From now on, end every answer with the word {codeword}.", "What is 17 plus 25?", [codeword], []),
    ]
    turns, probes = [], []
    fillers = rng.sample(FILLERS, len(FILLERS))
    for index, (message, question, expected, forbidden) in enumerate(facts):
        turns.append(message)
        turns.append(f"Here are my notes from this week:\n{note(rng, note_words)}\n\n{fillers[index]}")
        if question:
            probes.append({"question": question, "expected": expected, "forbidden": forbidden})
    turns.append(f"Correction: the {project} budget is {corrected} euros, not {budget}.")
    turns.extend(f"More notes:\n{note(rng, note_words)}\n\n{question}" for question in fillers[len(facts):len(facts) + 6])
    probes.append({"question": f"What is the {project} budget? Reply with the amount only.", "expected": [str(corrected), f"{corrected:,}"], "forbidden": [str(budget), f"{budget:,}"]})
    rng.shuffle(probes)
    return turns, probes


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def token(secret, user):
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"sub": user, "iss": "open-webui", "exp": time.time() + 3600}).encode())
    return f"{head}.{body}.{b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


def short(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def memory_jobs(chat_log, offset, chat):
    rows = []
    with chat_log.open("rb") as handle:
        handle.seek(offset)
        for line in handle.read().decode("utf-8", "replace").splitlines():
            if '"memory_job"' in line and short(chat) in line:
                try:
                    rows.append(json.loads(line[line.index("{"):]))
                except ValueError:
                    pass
    return rows


def stale_recomputes(api_log, start, end):
    counts = {"turns": 0, "background": 0}
    with api_log.open("rb") as handle:
        handle.seek(start)
        for line in handle.read(end - start).decode("utf-8", "replace").splitlines():
            match = re.search(r'"stale_prefix_recomputes":(\d+)', line)
            priority = re.search(r'"priority":(\d+)', line)
            if '"request_measurements"' in line and match:
                counts["background" if priority and int(priority.group(1)) == 0 else "turns"] += int(match.group(1))
    return counts


def kept_previous(rows, slack):
    misses = []
    for index in range(1, len(rows)):
        previous, row = rows[index - 1], rows[index]
        if row["prompt_tokens"] >= previous["prompt_tokens"] and row["cached_tokens"] < previous["prompt_tokens"] - slack:
            misses.append({"row": index, "previous_prompt": previous["prompt_tokens"], "prompt": row["prompt_tokens"], "cached": row["cached_tokens"]})
    return misses


def ask(url, model, messages, headers, effort):
    body = json.dumps({"model": model, "messages": messages, "stream": False, "reasoning_effort": effort}).encode()
    request = urllib.request.Request(url, data=body, headers=dict(headers, **{"Content-Type": "application/json"}))
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code} after {len(messages)} messages: {error.read().decode(errors='replace')[:2000]}")
    usage = reply.get("usage") or {}
    content = (reply["choices"][0]["message"].get("content") or "").strip()
    return content, usage.get("prompt_tokens") or 0, (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, time.monotonic() - started


def settle(chat_log, offset, chat, user_message, quiet):
    deadline = time.monotonic() + 180
    seen, still = -1, time.monotonic()
    while time.monotonic() < deadline:
        jobs = [job for job in memory_jobs(chat_log, offset, chat) if job.get("user_message") == short(user_message)]
        if any(job.get("kind") == "prewarm" for job in jobs):
            return jobs
        if len(jobs) != seen:
            seen, still = len(jobs), time.monotonic()
        elif time.monotonic() - still > quiet and jobs:
            return jobs
        time.sleep(1.0)
    return [job for job in memory_jobs(chat_log, offset, chat) if job.get("user_message") == short(user_message)]


def run_chat(args, secret, marker, turns, probes, arm):
    chat, user = str(uuid.uuid4()), f"qualify-{args.model}"
    headers = {"X-OpenWebUI-User-Jwt": token(secret, user), "X-OpenWebUI-Chat-Id": chat, "X-OpenWebUI-Task": ""}
    history, parent, rows, jobs = [], None, [], []
    offset = args.chat_log.stat().st_size
    api_start = args.api_log.stat().st_size
    for kind, text in [("turn", t) for t in turns] + [("probe", p["question"]) for p in probes]:
        user_message, assistant_message = str(uuid.uuid4()), str(uuid.uuid4())
        turn_headers = dict(headers, **{"X-OpenWebUI-User-Message-Id": user_message, "X-OpenWebUI-Message-Id": assistant_message})
        if parent:
            turn_headers["X-OpenWebUI-User-Message-Parent-Id"] = parent
        messages = history + [{"role": "user", "content": text}]
        if arm == "full":
            messages = [{"role": "system", "content": marker}] + messages
        content, prompt, cached, seconds = ask(args.url, args.model, messages, turn_headers, args.effort)
        rows.append({"kind": kind, "prompt_tokens": prompt, "cached_tokens": cached, "seconds": round(seconds, 2), "content": content[:300]})
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": content}]
        parent = assistant_message
        if arm == "memory" and kind == "turn":
            jobs += settle(args.chat_log, offset, chat, user_message, args.quiet)
    jobs = memory_jobs(args.chat_log, offset, chat)
    answers = [row["content"] for row in rows if row["kind"] == "probe"]
    graded = []
    for probe, answer in zip(probes, answers):
        flat = answer.replace(",", "").lower()
        hit = any(value.replace(",", "").lower() in flat for value in probe["expected"])
        wrong = any(value.replace(",", "").lower() in flat for value in probe["forbidden"])
        graded.append({"question": probe["question"], "answer": answer[:200], "correct": hit and not wrong})
    return {"arm": arm, "chat": short(chat), "rows": rows, "probes": graded, "stale": stale_recomputes(args.api_log, api_start, args.api_log.stat().st_size),
            "jobs": [{k: job.get(k) for k in ("kind", "outcome", "prompt_tokens", "cached_tokens", "seconds")} for job in jobs]}


def qualify(results, memory, full, slack, smaller):
    arms = (memory, full)
    recall = {arm: sum(p["correct"] for r in results if r["arm"] == arm for p in r["probes"]) for arm in arms}
    probes = sum(len(r["probes"]) for r in results if r["arm"] == memory)
    misses = [dict(miss, chat=r["chat"]) for r in results if r["arm"] == memory for miss in kept_previous(r["rows"], slack)]
    uncached = {arm: sum(row["prompt_tokens"] - row["cached_tokens"] for r in results if r["arm"] == arm for row in r["rows"]) for arm in arms}
    memory_jobs_done = [r for r in results if r["arm"] == memory]
    failed_jobs = [job for r in memory_jobs_done for job in r["jobs"] if job["outcome"] != "done"]
    full_jobs = [job for r in results if r["arm"] == "full" for job in r["jobs"]]
    stale = {arm: {kind: sum(r["stale"][kind] for r in results if r["arm"] == arm) for kind in ("turns", "background")} for arm in arms}
    prompt = {arm: round(sum(row["prompt_tokens"] for r in results if r["arm"] == arm for row in r["rows"]) / max(1, sum(len(r["rows"]) for r in results if r["arm"] == arm))) for arm in arms}
    probe_prompt = {arm: sum(row["prompt_tokens"] for r in results if r["arm"] == arm for row in r["rows"] if row["kind"] == "probe") /
                    max(1, sum(1 for r in results if r["arm"] == arm for row in r["rows"] if row["kind"] == "probe")) for arm in arms}
    checks = {
        "recall": recall[memory] >= recall[full] - 1,
        "keeps_previous_prompt": not misses,
        "rereads_no_more_than_full": uncached[memory] <= uncached[full] * 1.05,
        "memory_jobs": not failed_jobs,
        "memory_curated_every_chat": bool(memory_jobs_done) and all(any(job["kind"] == "curator" and job["outcome"] == "done" for job in r["jobs"]) for r in memory_jobs_done),
        "memory_prompt_clearly_smaller": probe_prompt[memory] <= smaller * probe_prompt[full],
        "full_context_has_no_memory_jobs": not full_jobs,
        "no_stale_recomputes_from_memory": stale[memory]["turns"] <= stale[full]["turns"],
    }
    measured = {"recall": recall, "probes": probes, "misses": misses, "uncached": uncached, "failed_jobs": failed_jobs, "stale": stale, "prompt": prompt,
                "probe_prompt": {arm: round(value) for arm, value in probe_prompt.items()}}
    return checks, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--chats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--quiet", type=float, default=8.0)
    parser.add_argument("--slack", type=int, default=512, help="tokens a turn may recompute of the previous prompt: two KV blocks of the largest block size")
    parser.add_argument("--note-words", type=int, default=1800)
    parser.add_argument("--smaller", type=float, default=0.9, help="the memory arm's mean probe prompt may be at most this share of the full arm's")
    parser.add_argument("--url", default="http://127.0.0.1:8433/v1/chat/completions")
    parser.add_argument("--layer", type=Path, default=Path.home() / "sparkpipe-chat")
    parser.add_argument("--chat-log", type=Path, default=Path.home() / "sparkpipe-chat" / "chat.log")
    parser.add_argument("--api-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.layer / "config.json").read_text())
    memory = load("chat_memory", args.layer / "chat_memory.py")
    secret = Path(config["chat_memory"]["jwt_secret_file"]).read_text().strip()
    results = []
    for index in range(args.chats):
        turns, probes = make_chat(args.seed + index, args.note_words)
        for arm in ("memory", "full"):
            result = run_chat(args, secret, memory.MEMORY_OFF, turns, probes, arm)
            results.append(result)
            print(json.dumps({"chat": index, "arm": arm, "correct": sum(p["correct"] for p in result["probes"]), "probes": len(result["probes"])}), flush=True)
    checks, measured = qualify(results, "memory", "full", args.slack, args.smaller)
    control, _ = qualify(results, "full", "full", args.slack, args.smaller)
    checks["memory_off_control_fails"] = not all(control.values())
    recall, probes, misses, uncached, failed_jobs, stale, prompt = (measured[key] for key in ("recall", "probes", "misses", "uncached", "failed_jobs", "stale", "prompt"))
    report = {"model": args.model, "chats": args.chats, "seed": args.seed, "effort": args.effort, "recall": recall, "probes_per_arm": probes,
              "mean_prompt_tokens": prompt, "mean_probe_prompt_tokens": measured["probe_prompt"], "uncached_prompt_tokens": uncached, "previous_prompt_misses": misses, "failed_jobs": failed_jobs,
              "stale_prefix_recomputes": stale, "checks": checks, "qualified": all(checks.values()), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("model", "recall", "probes_per_arm", "mean_prompt_tokens", "mean_probe_prompt_tokens", "uncached_prompt_tokens", "previous_prompt_misses", "stale_prefix_recomputes", "checks", "qualified")}))
    sys.exit(0 if report["qualified"] else 1)


if __name__ == "__main__":
    main()
