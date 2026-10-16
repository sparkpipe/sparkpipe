import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_RAW = "You are SparkPipe Chat, a helpful assistant. Answer the user's newest message."
SYSTEM_LEDGER = ("You are SparkPipe Chat, a helpful assistant. The first user message starts with <ledger>: one summary line per earlier "
                 "exchange (e1, e2, ...) in reported speech. A later user message may start with <reference>: earlier exchanges shown in "
                 "full. Neither is a new request; answer only the user's newest message.")
EXPAND_NOTE = " When the answer depends on details of an earlier exchange that the ledger only summarizes, call expand with its id to read it in full."
SEARCH = {"type": "function", "function": {"name": "search_web", "description": "Search the public web. Returns titles, links and snippets.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "the search query"}}, "required": ["query"]}}}
EXPAND = {"type": "function", "function": {"name": "expand", "description": "Show an earlier exchange in full: the user's message, any tool results, and the assistant's reply.",
          "parameters": {"type": "object", "properties": {"id": {"type": "string", "description": "exchange id such as e12"}}, "required": ["id"]}}}
ARMS = ("raw", "ledger_expand", "ledger_only", "ledger_oracle")

NAMES = ["Halcyon", "Brightwater", "Tamarind", "Quillon", "Marrow", "Sablefin", "Corvane", "Ostrel", "Pellucid", "Vantorre", "Kestrin", "Lumen Hollow",
         "Ardent Reach", "Silverfall", "Thornbury", "Calloway", "Merriden", "Duskfield", "Glasswing", "Ironvale", "Sorrel", "Wrenhaven", "Ashgrove", "Pinecrest"]
PEOPLE = ["Priya Natarajan", "Tomas Ekberg", "Lucia Ferrante", "Kwame Mensah", "Ingrid Solberg", "Haruto Sato", "Amara Okafor", "Dmitri Volkov",
          "Sofia Marquez", "Elif Kaya", "Rafael Duarte", "Mei Lin"]
WORDS = ["copper", "lantern", "meadow", "falcon", "harbor", "quartz", "willow", "ember", "cobalt", "saffron", "juniper", "granite", "orchid", "tundra"]
VENDORS = ["Northwind Supply", "Bluepeak Metals", "Orbis Freight", "Cedarline", "Vexley & Sons", "Harmon Labs", "Stellar Parts", "Grayson Co"]
FILLERS = [
    ("What's a quick way to keep basil fresh?", "Trim the stems and keep the bunch in a glass of water on the counter, loosely covered with a bag. Refrigerating it tends to blacken the leaves."),
    ("Can you explain what a hash map is in two sentences?", "A hash map stores key-value pairs and uses a hash function to turn each key into an array index. That makes lookups, inserts and deletes take constant time on average."),
    ("Any tips for a first half marathon?", "Build mileage gradually, about ten percent a week, and include one long run weekly. Practise your race-day breakfast and pacing on long runs so nothing is new on the day."),
    ("Why is the sky blue?", "Air molecules scatter short blue wavelengths of sunlight much more than red ones, an effect called Rayleigh scattering, so blue light reaches your eyes from every direction."),
    ("How do I convert Celsius to Fahrenheit?", "Multiply the Celsius value by 9/5 and add 32. For example, 20 degrees Celsius is 68 degrees Fahrenheit."),
    ("What's the difference between a list and a tuple in Python?", "Lists are mutable, so you can change their contents; tuples are immutable. Tuples are hashable when their items are, so they can be dictionary keys."),
    ("Suggest a name for a grey cat.", "How about Smoke, Pebble or Ash? If you want something playful, Earl Grey works too."),
    ("How long should I steep green tea?", "About two to three minutes in water around 75 to 80 degrees Celsius; hotter water or longer steeping makes it bitter."),
    ("What does HTTP 429 mean?", "429 Too Many Requests: the server is rate limiting you. Back off and retry after the time in the Retry-After header if it is present."),
    ("Is it better to stretch before or after running?", "Do dynamic moves like leg swings before running and save longer static stretches for afterwards, when your muscles are warm."),
    ("Recommend a classic novel for a long flight.", "The Count of Monte Cristo is long, fast-moving and easy to dip in and out of, which suits a flight well."),
    ("How do I undo my last git commit but keep the changes?", "Run git reset --soft HEAD~1. The commit is removed and its changes stay staged."),
    ("What's a good stretch for a stiff neck?", "Slowly tilt your ear towards your shoulder and hold for twenty to thirty seconds on each side, keeping your shoulders relaxed."),
    ("Explain compound interest simply.", "You earn interest on your interest: each period's interest is added to the balance, so the next period's interest is calculated on a larger amount."),
    ("What's the capital of Australia?", "Canberra. Sydney is the largest city, but Canberra was chosen as a compromise between Sydney and Melbourne."),
    ("How many cups are in a litre?", "About 4.2 US cups, or exactly 4 metric cups of 250 millilitres."),
]


def make_chat(seed):
    rng = random.Random(seed)
    name, project, person, codeword = rng.choice(NAMES), rng.choice(NAMES), rng.choice(PEOPLE), rng.choice(WORDS) + str(rng.randint(10, 99))
    n = rng.randint(17, 24)
    exchanges = [None] * n
    facts = {}

    def put(index, exchange):
        exchanges[index] = exchange

    order = ["code", "drafts", "options", "document", "tool", "instruction", "table", "correction"]
    rng.shuffle(order)
    slots = sorted(rng.sample(range(0, n - 7), 8))
    corr_second = None
    for pattern, slot in zip(order, slots):
        if pattern == "code":
            n_retry, base = rng.randint(3, 9), rng.choice([0.25, 0.5, 0.75, 1.5, 2.5])
            fn = name.lower().replace(" ", "_")
            u = f"My retry helper fetch_{fn} loops one time too many. Can you fix it?\n```python\ndef fetch_{fn}(client):\n    for attempt in range(MAX_RETRIES + 1):\n        try:\n            return client.get('/{fn}')\n        except TimeoutError:\n            time.sleep(BACKOFF_BASE * 2 ** attempt)\n```"
            a = (f"The loop should run MAX_RETRIES times, not MAX_RETRIES + 1. I also pinned the constants so the behaviour is explicit:\n```python\nMAX_RETRIES = {n_retry}\nBACKOFF_BASE = {base}\n\n"
                 f"def fetch_{fn}(client):\n    for attempt in range(MAX_RETRIES):\n        try:\n            return client.get('/{fn}')\n        except TimeoutError:\n            time.sleep(BACKOFF_BASE * 2 ** attempt)\n    raise TimeoutError('fetch_{fn} gave up')\n```")
            facts[pattern] = (slot, f"In the fix for fetch_{fn}, what value did you set BACKOFF_BASE to? Reply with the number only.", [str(base)], [])
            put(slot, {"u": u, "a": a})
        elif pattern == "drafts":
            topic = rng.choice(NAMES)
            titles = [f"{rng.choice(['Meet', 'Introducing', 'Say Hello to', 'Inside'])} {topic} {rng.choice(['2.0', 'Studio', 'Cloud', 'Go'])}",
                      f"{topic}: {rng.choice(['Faster Than', 'Built for', 'Made for', 'Ready for'])} {rng.choice(['Teams', 'Makers', 'Night Owls', 'Small Shops'])}",
                      f"Why We Rebuilt {topic} {rng.choice(['From Scratch', 'in Rust', 'for Everyone', 'This Year'])}"]
            u = f"Give me three title drafts for the {topic} launch post."
            a = "Here are three drafts:\n1. " + titles[0] + "\n2. " + titles[1] + "\n3. " + titles[2] + "\nThe second is the most direct; the third works if you want a story-led post."
            facts[pattern] = (slot, f"What was the second title draft for the {topic} launch post? Reply with the title only.", [titles[1]], [])
            put(slot, {"u": u, "a": a})
        elif pattern == "options":
            problem = f"moving the {rng.choice(NAMES)} database off the old server"
            opts = rng.sample(["Blue-Green Cutover", "Log Shipping Mirror", "Snapshot and Replay", "Dual-Write Bridge", "Read Replica Promotion", "Weekend Freeze"], 3)
            u = f"What are my options for {problem}?"
            a = (f"Three realistic options:\nOption A, {opts[0]}: lowest risk, needs a short write freeze.\nOption B, {opts[1]}: no downtime, but more moving parts to monitor.\n"
                 f"Option C, {opts[2]}: simplest to run, longest downtime.\nI would start from option A unless downtime is unacceptable.")
            facts[pattern] = (slot, f"For {problem}, what was option B called? Reply with its name only.", [opts[1]], [])
            put(slot, {"u": u, "a": a})
        elif pattern == "document":
            room, code, date = rng.randint(200, 899), f"{rng.choice('ABCDEFGH')}{rng.randint(1000, 9999)}", f"{rng.randint(3, 27)} November"
            memo = (f"From: {person}\nSubject: {project} quarterly review\n\nThe {project} quarterly review moves to {date}. Please bring the updated cost sheet. "
                    f"Charges for the session go to budget code {code}. We will meet in room {room} on the fourth floor; the projector there needs the HDMI adapter. "
                    f"Lunch is provided. Send agenda items to me by the Friday before.")
            u = f"Here is a memo from {person}:\n---\n{memo}\n---\nCan you summarise it in one line?"
            a = f"{person} says the {project} quarterly review moves to {date}; bring the updated cost sheet, charge it to the project budget code, and send agenda items by the Friday before."
            facts[pattern] = (slot, f"Which room number did {person}'s memo give for the {project} review? Reply with the number only.", [str(room)], [])
            put(slot, {"u": u, "a": a})
        elif pattern == "tool":
            thing = f"{rng.choice(NAMES)} {rng.choice(['subscriber count', 'annual output in tonnes', 'median order value in euros', 'peak daily visitors'])}"
            value = f"{rng.randint(11, 98)},{rng.randint(100, 999)}"
            query = f"latest {thing}"
            result = json.dumps([{"title": f"{thing.title()} - annual report", "link": "https://example.org/report", "snippet": f"The latest figure for {thing} is {value}, up from last year."}])
            u = f"Can you look up the latest {thing}?"
            a = f"I found it in the annual report summary; the figure is up from last year, and the report has a regional breakdown if you need it."
            facts[pattern] = (slot, f"What figure did your search find for the {thing}? Reply with the figure only.", [value, value.replace(",", "")], [])
            put(slot, {"u": u, "a": a, "tool": {"name": "search_web", "arguments": {"query": query}, "result": result}})
        elif pattern == "instruction":
            u = f"From now on, end every answer with the word {codeword}."
            a = f"Understood. I will end every answer with that word. {codeword}"
            facts[pattern] = (slot, "What is 17 plus 25?", [codeword], [])
            put(slot, {"u": u, "a": a, "codeword": codeword})
        elif pattern == "table":
            item = rng.choice(["standing desk", "server rack", "forklift lease", "office chair", "solar panel"])
            vendors = rng.sample(VENDORS, 4)
            prices = [rng.randint(180, 990) * 10 for _ in vendors]
            table = "| Vendor | Price (EUR) |\n|---|---|\n" + "".join(f"| {v} | {p:,} |\n" for v, p in zip(vendors, prices))
            u = f"Put the {item} quotes we got into a table."
            a = f"Here are the {item} quotes:\n{table}The cheapest is {vendors[prices.index(min(prices))]}."
            facts[pattern] = (slot, f"In the {item} quote table, what price did {vendors[2]} quote? Reply with the price only.", [f"{prices[2]:,}", str(prices[2])], [])
            put(slot, {"u": u, "a": a})
        elif pattern == "correction":
            b1, b2 = rng.randint(20, 60) * 1000, rng.randint(61, 99) * 1000
            put(slot, {"u": f"For the record, the {project} budget is {b1:,} euros.", "a": f"Noted: the {project} budget is {b1:,} euros."})
            corr_second = (slot, b1, b2)
            facts[pattern] = (slot, f"What is the {project} budget? Reply with the amount only.", [f"{b2:,}", str(b2)], [f"{b1:,}", str(b1)])
    if corr_second:
        slot, b1, b2 = corr_second
        free = [i for i in range(slot + 1, n - 3) if exchanges[i] is None]
        second = free[0] if free else None
        if second is not None:
            put(second, {"u": f"Correction: the {project} budget is {b2:,} euros, not {b1:,}.", "a": f"Updated: the {project} budget is now {b2:,} euros."})
            facts["correction"] = (second,) + facts["correction"][1:]
    fillers = rng.sample(FILLERS, len(FILLERS))
    k = 0
    for i in range(n):
        if exchanges[i] is None:
            u, a = fillers[k % len(fillers)]
            k += 1
            exchanges[i] = {"u": u, "a": a}
    instruction_at = next((i for i, e in enumerate(exchanges) if "codeword" in e), None)
    if instruction_at is not None:
        for i in range(instruction_at + 1, n):
            if not exchanges[i]["a"].endswith(codeword):
                exchanges[i]["a"] += f" {codeword}"
    probes = []
    for pattern, (slot, question, expected, forbidden) in facts.items():
        lo = slot + 3
        if lo > n - 1:
            continue
        at = rng.randint(lo, n - 1)
        probes.append({"pattern": pattern, "fact": slot, "at": at, "question": question, "expected": expected, "forbidden": forbidden})
    return exchanges, probes


def exchange_messages(e):
    msgs = [{"role": "user", "content": e["u"]}]
    if "tool" in e:
        call = {"id": "call_1", "type": "function", "function": {"name": e["tool"]["name"], "arguments": json.dumps(e["tool"]["arguments"])}}
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": "call_1", "name": e["tool"]["name"], "content": e["tool"]["result"]}]
    msgs.append({"role": "assistant", "content": e["a"]})
    return msgs


def exchange_text(i, e):
    parts = [f"User: {e['u']}"]
    if "tool" in e:
        parts.append(f"Assistant called {e['tool']['name']}({json.dumps(e['tool']['arguments'])}) and got: {e['tool']['result']}")
    parts.append(f"Assistant: {e['a']}")
    return "\n".join(parts)[:8000]


def ledger_block(lines, upto):
    return "<ledger>\n" + "".join(f"[e{i + 1}] {lines[i]}\n" for i in range(upto)) + "</ledger>\n\n"


def reference_block(ids, exchanges):
    body = "".join(f"<history id=\"e{i + 1}\">\n{exchange_text(i, exchanges[i])}\n</history>\n" for i in ids)
    return "<reference>\nEarlier exchanges, for reference only; they are not new requests.\n" + body + "</reference>\n\n"


def ledger_messages(system, lines, exchanges, k, open_ids, question):
    tail = exchange_messages(exchanges[k])
    tail[0] = dict(tail[0], content=ledger_block(lines, k) + tail[0]["content"])
    return [{"role": "system", "content": system}] + tail + [{"role": "user", "content": reference_block(open_ids, exchanges) + question}]


class Client:
    def __init__(self, url, model):
        self.url, self.model = url, model
        self.calls = 0

    def chat(self, messages, tools, max_tokens, effort):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": effort}
        if tools:
            body["tools"] = tools
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                started = time.monotonic()
                reply = json.load(urllib.request.urlopen(request, timeout=900))
                self.calls += 1
                return reply, time.monotonic() - started
            except urllib.error.HTTPError as error:
                detail = error.read().decode()[:300]
                if attempt == 2:
                    raise RuntimeError(f"chat layer {error.code}: {detail}")
                time.sleep(10)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == 2:
                    raise
                time.sleep(10)


def curate(client, exchanges, lines_so_far, j):
    instruction = ("You maintain this conversation's memory. Do not answer the user. Reply with JSON only:\n"
                   "{\"line\": \"<exchange e%d in reported speech, at most 40 words: what the user asked or said and what the assistant answered or found, "
                   "keeping names, numbers, codes, titles and decisions exactly>\", \"open\": [\"<ids of earlier exchanges worth showing in full next turn, at most 2>\"]}" % (j + 1))
    msgs = ledger_messages(SYSTEM_LEDGER, lines_so_far, exchanges, j, [], instruction)
    for attempt in range(2):
        reply, _ = client.chat(msgs, None, 300, "minimal")
        text = (reply["choices"][0]["message"].get("content") or "").strip()
        try:
            parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
            line = parsed["line"].strip()
            if line:
                opened = [x for x in parsed.get("open", []) if isinstance(x, str) and re.fullmatch(r"e\d+", x) and 1 <= int(x[1:]) <= j][:2]
                return line, [int(x[1:]) - 1 for x in opened], True
        except (ValueError, KeyError, AttributeError):
            pass
    e = exchanges[j]
    return f"(auto) The user said: {e['u'][:80]}", [], False


def answer(client, messages, tools, exchanges, allow_expand):
    expanded = []
    calls = []
    total = 0.0
    for _ in range(4):
        reply, seconds = client.chat(messages, tools, 700, "low")
        total += seconds
        message = reply["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content") or "", expanded, calls, total, reply.get("usage", {})
        messages = messages + [{"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls}]
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": name, "args": args})
            if name == "expand" and allow_expand:
                ident = str(args.get("id", ""))
                match = re.fullmatch(r"e(\d+)", ident)
                if match and 1 <= int(match.group(1)) <= len(exchanges):
                    expanded.append(int(match.group(1)) - 1)
                    result = exchange_text(int(match.group(1)) - 1, exchanges[int(match.group(1)) - 1])
                else:
                    result = f"No exchange {ident!r}."
            else:
                result = "This tool is unavailable in this conversation."
            messages = messages + [{"role": "tool", "tool_call_id": call.get("id", "call"), "name": name, "content": result}]
    return "", expanded, calls, total, {}


def normalize(text):
    return re.sub(r"[^a-z0-9.]+", "", text.lower().replace(",", ""))


def correct(probe, text):
    t = normalize(text)
    hit = any(normalize(x) in t for x in probe["expected"])
    wrong = any(normalize(x) in t for x in probe["forbidden"])
    return hit and not wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8433/v1/chat/completions")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--chats", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model)
    rows = []
    with (out / "answers.jsonl").open("a") as log, (out / "ledgers.jsonl").open("a") as ledger_log:
        for chat in range(args.chats):
            exchanges, probes = make_chat(1000 + chat)
            lines, opens, valid = [], [], []
            for j in range(len(exchanges)):
                line, opened, ok = curate(client, exchanges, lines, j)
                lines.append(line)
                opens.append(opened)
                valid.append(ok)
            ledger_log.write(json.dumps({"chat": chat, "lines": lines, "opens": opens, "valid": valid}) + "\n")
            ledger_log.flush()
            for p_index, probe in enumerate(probes):
                k = probe["at"]
                needed = probe["fact"]
                for arm in ARMS:
                    if arm == "raw":
                        msgs = [{"role": "system", "content": SYSTEM_RAW}]
                        for e in exchanges[:k + 1]:
                            msgs += exchange_messages(e)
                        msgs.append({"role": "user", "content": probe["question"]})
                        tools, allow = [SEARCH], False
                    else:
                        open_ids = [needed] if arm == "ledger_oracle" else [i for i in opens[k] if i != k]
                        system = SYSTEM_LEDGER + (EXPAND_NOTE if arm == "ledger_expand" else "")
                        msgs = ledger_messages(system, lines, exchanges, k, open_ids, probe["question"])
                        tools, allow = ([SEARCH, EXPAND], True) if arm == "ledger_expand" else ([SEARCH], False)
                    text, expanded, calls, seconds, usage = answer(client, msgs, tools, exchanges, allow)
                    row = {"chat": chat, "probe": p_index, "pattern": probe["pattern"], "arm": arm, "at": k, "fact": needed,
                           "visible": needed == k or (arm != "raw" and needed in (opens[k] if arm != "ledger_oracle" else [needed])),
                           "fact_in_line": any(normalize(x) in normalize(lines[needed]) for x in probe["expected"]),
                           "correct": correct(probe, text), "answer": text[-300:], "expanded": expanded, "calls": calls,
                           "seconds": round(seconds, 2), "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
            done = [r for r in rows if r["chat"] == chat]
            print(json.dumps({"chat": chat, "exchanges": len(exchanges), "probes": len(probes),
                              "accuracy": {arm: round(sum(r["correct"] for r in done if r["arm"] == arm) / max(1, sum(1 for r in done if r["arm"] == arm)), 3) for arm in ARMS},
                              "curator_valid": f"{sum(valid)}/{len(valid)}", "calls": client.calls}), flush=True)
    summarize(rows, out)


def summarize(rows, out):
    rng = random.Random(7)
    probes = sorted({(r["chat"], r["probe"]) for r in rows})
    by = {(r["chat"], r["probe"], r["arm"]): r for r in rows}
    summary = {"probes": len(probes), "arms": {}}
    for arm in ARMS:
        hits = [by[(c, p, arm)]["correct"] for c, p in probes if (c, p, arm) in by]
        entry = {"accuracy": round(sum(hits) / max(1, len(hits)), 3), "n": len(hits)}
        if arm != "raw":
            diffs = [int(by[(c, p, arm)]["correct"]) - int(by[(c, p, "raw")]["correct"]) for c, p in probes if (c, p, arm) in by and (c, p, "raw") in by]
            boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
            entry["minus_raw"] = round(sum(diffs) / len(diffs), 3)
            entry["minus_raw_95ci"] = [round(boots[50], 3), round(boots[1949], 3)]
        patterns = {}
        for r in rows:
            if r["arm"] == arm:
                patterns.setdefault(r["pattern"], []).append(r["correct"])
        entry["by_pattern"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(patterns.items())}
        summary["arms"][arm] = entry
    hidden = [r for r in rows if r["arm"] == "ledger_expand" and not r["visible"]]
    summary["expand"] = {"probes_with_fact_hidden": len(hidden),
                         "expanded_the_fact": sum(1 for r in hidden if r["fact"] in r["expanded"]),
                         "expanded_anything": sum(1 for r in hidden if r["expanded"]),
                         "correct_when_hidden": sum(1 for r in hidden if r["correct"]),
                         "correct_when_hidden_and_fact_not_in_line": sum(1 for r in hidden if r["correct"] and not r["fact_in_line"]),
                         "hidden_and_fact_not_in_line": sum(1 for r in hidden if not r["fact_in_line"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        out = Path(sys.argv[2])
        summarize([json.loads(l) for l in (out / "answers.jsonl").open()], out)
    else:
        main()
