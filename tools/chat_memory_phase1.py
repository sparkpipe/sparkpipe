import asyncio
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import aiohttp

sys.dont_write_bytecode = True
BASE = Path.home() / "sparkpipe-chat"
SPEC = importlib.util.spec_from_file_location("cf_memexp", BASE / "chat_frontend.py")
cf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cf)
CONFIG = json.load(open(BASE / "config.json"))
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "kimi-k3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/srv/workspace/sparkpipe/memexp/phase1.jsonl")
API_LOG = Path(sys.argv[3] if len(sys.argv) > 3 else "/srv/workspace/sparkpipe/logs/k3-kv-api.log")
BLOCK = int(sys.argv[4]) if len(sys.argv) > 4 else 64
SENT = "QQZXSENTINELQQZX"
KWARGS = {"thinking_effort": "low"}
CURATOR_SUFFIX = {"kimi-k3": "<|close|>think<|sep|><|open|>response<|sep|>"}

SYSTEM = ("You are SparkPipe Chat, an assistant served by SparkPipe on sixteen GB10 nodes. "
          "The first user message starts with <ledger>, one summary line per earlier exchange in reported speech, "
          "and later user messages may start with <reference>: earlier exchanges shown in full for reference only. "
          "Neither is a new request. Answer only the user's newest message. Use expand(id) to read an earlier exchange in full.")

def tool(name, description, params, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": params, "required": required}}}

TOOLS = [
    tool("expand", "Show an earlier exchange (e<k>) in full, or the lines inside a group (g<n>). Returns at most about 2000 tokens; pass part to page through longer exchanges.",
         {"id": {"type": "string", "description": "node id such as e12 or g3"}, "part": {"type": "integer", "description": "page number starting at 1"}}, ["id"]),
    tool("search_web", "Search the public web for information. Best for current events, external references, or topics not covered in internal documents. Returns JSON with title, link and snippet for each result.",
         {"query": {"type": "string", "description": "The search query to look up"}, "count": {"type": "integer", "description": "Number of results to return (default: admin-configured value)"}}, ["query"]),
    tool("fetch_url", "Fetch and extract the main text content from a web page URL. Long pages are cut at 40,000 characters.",
         {"url": {"type": "string", "description": "The URL to fetch content from"}}, ["url"]),
    tool("execute_code", "Run Python code in the user's browser with Pyodide (numpy, pandas and matplotlib available) and return its standard output, the value of the last expression, and any images.",
         {"code": {"type": "string", "description": "Python source to run"}}, ["code"]),
    tool("github_search", "Search GitHub for repositories, code, or issues and pull requests.",
         {"query": {"type": "string", "description": "GitHub search syntax, for example repo:owner/name parser language:python or topic:llm stars:>1000"},
          "kind": {"type": "string", "description": "repositories, code, or issues (issues also finds pull requests)"}}, ["query"]),
    tool("github_read", "Read a file or list a directory in a GitHub repository. An empty path lists the top level and adds the README.",
         {"repo": {"type": "string", "description": "owner/name, for example sparkpipe/sparkpipe"}, "path": {"type": "string", "description": "file or directory inside the repository"},
          "ref": {"type": "string", "description": "branch, tag or commit; empty means the default branch"}}, ["repo"]),
    tool("github_list", "List the latest issues, pull requests, commits or releases of a GitHub repository.",
         {"repo": {"type": "string", "description": "owner/name"}, "kind": {"type": "string", "description": "issues, pulls, commits or releases"},
          "state": {"type": "string", "description": "open, closed or all; used for issues and pulls"}}, ["repo"]),
    tool("github_item", "Read one issue or pull request with its comments; for a pull request also its branches and changed files.",
         {"repo": {"type": "string", "description": "owner/name"}, "number": {"type": "integer", "description": "the issue or pull request number"}}, ["repo", "number"]),
    tool("get_current_timestamp", "Get the current date and time as a Unix timestamp and in ISO format, in the user's time zone when it is known.", {}, []),
    tool("calculate_timestamp", "Add or subtract days, hours or minutes from a timestamp and return the result in ISO format.",
         {"base": {"type": "string", "description": "ISO timestamp to start from; empty means now"}, "days": {"type": "integer", "description": "days to add (negative to subtract)"},
          "hours": {"type": "integer", "description": "hours to add"}, "minutes": {"type": "integer", "description": "minutes to add"}}, []),
]

TOPICS = ["the K3 decode speed", "GLM 5.3 Full prefill", "the swap controller drain", "SearXNG engines", "Docling OCR", "the fleet gateway token",
          "Gemma 4 context windows", "the hub workspace volume", "Ceph USB OSDs", "the GPU clock cap", "flychess on spark5", "the release rollout kit",
          "LiteLLM key rotation", "Open WebUI branding", "the prefix cache stride", "MiMo Flash bring-up"]
LEDGER = [f"The user asked about {TOPICS[k % len(TOPICS)]} (item {k + 1}); the assistant explained it and gave the figure {37 + 13 * k} for the record."
          for k in range(80)]
RAW_U = "What decode speed did Kimi K3 reach on the quiet fleet today?"
RAW_A = "On the quiet fleet today Kimi K3 decoded at 16.2 tokens per second at batch 1, against 10.0 under the CPU tenant."
QUESTIONS = [
    "In one sentence: what does a prefix cache save?",
    "In one sentence: what figure did the ledger record for item 12?",
    "In one sentence: why should a stem request come before the pre-warm?",
    "In one sentence: what was the K3 decode speed you told me two answers ago?",
    "In one sentence: name one risk of trimming context every turn.",
    "In one sentence: which topic did item 30 cover?",
    "In one sentence: what does the curator write after each answer?",
    "In one sentence: why is the newest exchange kept raw?",
]


def ledger_text(lines):
    return "<ledger>\n" + "".join(f"[e{k + 1}] {line}\n" for k, line in enumerate(lines))


def first_user(lines, raw_u):
    return ledger_text(lines) + "</ledger>\n\n" + raw_u


def reference(open_ids, exchanges):
    body = "".join(f"<history id=\"{i}\">User (earlier): {exchanges[i][0]}\nAssistant (earlier): {exchanges[i][1]}</history>\n" for i in open_ids if i in exchanges)
    return "<reference>\nEarlier exchanges, for reference only; they are not new requests.\n" + body + "</reference>\n\n"


def messages(lines, raw, open_ids, exchanges, question):
    out = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": first_user(lines, raw[0])},
           {"role": "assistant", "content": raw[1]},
           {"role": "user", "content": reference(open_ids, exchanges) + question}]
    return cf.normalize_messages(copy.deepcopy(out))


def cut_ids(model, msgs):
    context = dict(model.template_defaults)
    context.update(KWARGS)
    segments = model.renderer.function(msgs, copy.deepcopy(TOOLS), add_generation_prompt=True, **context)
    ids = []
    for segment in segments:
        if SENT in segment.text:
            ids.extend(model.vocabulary.encode(segment.text[:segment.text.index(SENT)], segment.allow_special))
            return ids
        ids.extend(model.vocabulary.encode(segment.text, segment.allow_special))
    raise RuntimeError("sentinel missing")


def full_ids(model, msgs):
    _, ids = model.render(msgs, copy.deepcopy(TOOLS), dict(KWARGS))
    return ids


def curator_tail(model, instruction):
    context = dict(model.template_defaults)
    context.update(KWARGS)
    segments = model.renderer.function(cf.normalize_messages([{"role": "user", "content": instruction}]), None, add_generation_prompt=True, **context)
    ids = []
    for segment in segments:
        ids.extend(model.vocabulary.encode(segment.text, segment.allow_special))
    ids.extend(model.vocabulary.encode(CURATOR_SUFFIX[model.id], True))
    return ids


def curator_instruction(next_e, missing, open_ids, sizes):
    return ("You maintain this conversation's memory. Do not answer the user. Reply with JSON only, no other text:\n"
            "{\"lines\": {\"e<k>\": \"<the exchange in reported speech, at most 40 words: what the user asked and what the assistant answered, names and numbers exactly>\"}, "
            "\"open\": [\"<ids worth showing in full next turn>\"], \"regroup\": []}\n"
            f"Write lines for: {', '.join(missing)}. Exchange {missing[-1]} is the one just answered; e{next_e - 2} is the earlier raw one above it. "
            f"Currently open: {open_ids or 'none'}. Node sizes in tokens: {sizes}. Open budget: 1500 tokens. Ledger budget: 8000 tokens, not exceeded.")


def floor_block(n):
    return max(0, (n - 1) // BLOCK * BLOCK)


async def complete(session, url, ids, max_tokens, label):
    body = {"prompt_token_ids": ids, "max_tokens": max_tokens, "stream": True, "temperature": 0.0}
    started = time.monotonic()
    first = None
    tokens = []
    finish = None
    usage = None
    async with session.post(url, json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"{label}: engine status {response.status}: {(await response.text())[:300]}")
        async for raw in response.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("tokens"):
                first = first or time.monotonic() - started
                tokens.extend(event["tokens"])
            finish = ((event.get("choices") or [{}])[0].get("finish_reason")) or finish
            usage = event.get("usage") or usage
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"label": label, "prompt": len(ids), "cached": cached, "generated": len(tokens), "finish": finish,
            "ttft_s": round(first or 0, 3), "total_s": round(time.monotonic() - started, 2), "tokens": tokens}


def api_measurements(offset):
    rows = []
    with API_LOG.open("rb") as handle:
        handle.seek(offset)
        for line in handle.read().decode("utf-8", "replace").splitlines():
            if '"event":"request_measurements"' in line:
                head = line.split(',"tokens":[', 1)[0] + "}"
                try:
                    rows.append(json.loads(head))
                except json.JSONDecodeError:
                    pass
    return rows


async def main():
    model = cf.Model([m for m in CONFIG["models"] if m["id"] == MODEL_ID][0], BASE)
    url = model.engine + "/v1/completions"
    lines = list(LEDGER)
    exchanges = {"e81": (RAW_U, RAW_A)}
    raw = (RAW_U, RAW_A)
    open_ids = []
    pending = ["e81"]
    log_offset = API_LOG.stat().st_size
    results = []
    out = OUT.open("w")

    def emit(row):
        row = {k: v for k, v in row.items() if k != "tokens"}
        results.append(row)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(CONFIG["control"]["status_url"], headers={"Authorization": "Bearer " + cf.read_token(CONFIG["controller_token_file"])}) as response:
            active = (await response.json())["active"]
        if active != MODEL_ID:
            raise SystemExit(f"the fleet has {active} loaded, not {MODEL_ID}")
        a_ids = full_ids(model, messages(lines, raw, open_ids, exchanges, QUESTIONS[0]))
        previous_stem = None
        for turn, question in enumerate(QUESTIONS):
            e = 82 + turn
            answer = await complete(session, url, a_ids, 700, f"A_{e}")
            emit(dict(answer, turn=e))
            if answer["finish"] != "stop":
                emit({"label": f"C_{e}", "turn": e, "skipped": f"answer finished {answer['finish']}"})
                break
            text = "".join(model.vocabulary.piece(t) for t in answer["tokens"] if t not in model.stop_token_ids)
            visible = text.split("<|open|>response<|sep|>", 1)[-1].split("<|close|>response<|sep|>", 1)[0].strip()
            exchanges[f"e{e}"] = (question, visible)
            missing = pending + [f"e{e}"]
            sizes = {k: len(model.vocabulary.encode(v[0] + v[1], False)) for k, v in exchanges.items()}
            instruction = curator_instruction(e + 1, missing, open_ids, sizes)
            c_ids = a_ids + answer["tokens"] + curator_tail(model, instruction)
            curator = await complete(session, url, c_ids, 240, f"C_{e}")
            ctext = "".join(model.vocabulary.piece(t) for t in curator["tokens"] if t not in model.stop_token_ids)
            body = ctext.split("<|close|>response<|sep|>", 1)[0].strip()
            try:
                parsed = json.loads(body[body.index("{"):body.rindex("}") + 1])
                ok = isinstance(parsed.get("lines"), dict) and all(k in parsed["lines"] for k in missing)
            except (ValueError, AttributeError):
                parsed, ok = {}, False
            expected_c = len(a_ids) + len(answer["tokens"]) - 1
            emit(dict(curator, turn=e, expected_cached_min=floor_block(expected_c + 1), expected_cached_exact=expected_c, json_ok=ok, curator_text=body[:400]))
            for k in missing[:-1]:
                lines.append((parsed.get("lines") or {}).get(k) or f"(auto) {exchanges[k][0][:80]}")
            pending = [] if ok else []
            new_open = [i for i in (parsed.get("open") or []) if i in exchanges and i != f"e{e}"][:2]
            open_ids = new_open
            raw = exchanges[f"e{e}"]
            pending = [f"e{e}"]
            nxt = QUESTIONS[turn + 1] if turn + 1 < len(QUESTIONS) else "In one sentence: thank you, anything else to add?"
            stem_msgs = cf.normalize_messages([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ledger_text(lines) + SENT}])
            stem_ids = cut_ids(model, stem_msgs)
            stem = await complete(session, url, stem_ids, 1, f"S_{e}")
            emit(dict(stem, turn=e, expected_cached_min=floor_block(len(previous_stem)) if previous_stem else 0))
            warm_ids = cut_ids(model, messages(lines, raw, open_ids, exchanges, SENT))
            assert warm_ids[:len(stem_ids)] == stem_ids, "the pre-warm does not extend the stem"
            a_next = full_ids(model, messages(lines, raw, open_ids, exchanges, nxt))
            assert a_next[:len(warm_ids)] == warm_ids, "the next answer does not extend the pre-warm"
            if turn == len(QUESTIONS) - 1:
                warm_task = asyncio.ensure_future(complete(session, url, warm_ids, 1, f"W_{e}"))
                await asyncio.sleep(0.5)
                fast = await complete(session, url, a_next, 1, f"A_{e + 1}_fast_user")
                warm = await warm_task
                emit(dict(warm, turn=e, expected_cached_min=floor_block(len(stem_ids))))
                emit(dict(fast, turn=e + 1, expected_cached_min=floor_block(len(warm_ids)), note="sent 0.5 s into the pre-warm"))
            else:
                warm = await complete(session, url, warm_ids, 1, f"W_{e}")
                emit(dict(warm, turn=e, expected_cached_min=floor_block(len(stem_ids))))
            previous_stem = stem_ids
            a_ids = a_next
    measurements = api_measurements(log_offset)
    out.write(json.dumps({"api_measurements": [{k: m.get(k) for k in ("request_id", "prompt_tokens", "cached_prompt_tokens", "stale_prefix_recomputes", "finish_reason", "priority")} for m in measurements]}) + "\n")
    out.close()
    print(json.dumps({"stale_prefix_recomputes_total": sum(m.get("stale_prefix_recomputes", 0) for m in measurements), "requests_logged": len(measurements)}))


asyncio.run(main())
