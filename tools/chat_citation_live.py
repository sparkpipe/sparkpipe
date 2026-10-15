import argparse
import base64
import hashlib
import hmac
import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

URLS = ["https://docs.example.org/release-4.2", "https://docs.example.org/release-4.3"]
TOOLS = [{"type": "function", "function": {"name": "fetch_url", "description": "Fetch a web page and return its text.",
          "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "the page URL"}}, "required": ["url"]}}}]


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwt(secret, sub):
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    body = b64(json.dumps({"sub": sub, "iss": "open-webui", "iat": now, "exp": now + 300}).encode())
    return f"{head}.{body}.{b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


def page(url, salt):
    rng = random.Random(url + salt)
    version = url.rsplit("-", 1)[1]
    lines = [f"Release notes for version {version}.", "Navigation: Home | Docs | Blog | Pricing | Contact"]
    for i in range(60):
        lines.append(f"Change {i + 1}: component {rng.choice(['parser', 'scheduler', 'cache', 'router'])} gained {rng.randint(2, 40)} percent "
                     f"throughput after fix #{rng.randint(1000, 9999)}; latency moved from {rng.randint(10, 90)} ms to {rng.randint(5, 60)} ms.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--head-file", required=True, help="the citation wrapper head text, read from the client deployment")
    parser.add_argument("--secret-file")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--salt", default="")
    parser.add_argument("--no-chat-id", action="store_true", help="send no Open WebUI ids, as a non-Open WebUI client would")
    args = parser.parse_args()
    head = Path(args.head_file).read_text()
    secret = Path(args.secret_file).read_text().strip() if args.secret_file else None
    question = f"(Run {args.salt}.) Fetch {URLS[0]} and then {URLS[1]} with fetch_url, one call at a time, and tell me which change number improved throughput the most in each release."
    messages = [{"role": "system", "content": "You are SparkPipe Chat, a helpful assistant."}, {"role": "user", "content": question}]
    chat, user_message, message = "citelive-" + uuid.uuid4().hex[:12], str(uuid.uuid4()), str(uuid.uuid4())
    fetched = []
    for round_index in range(1, 6):
        headers = {"Content-Type": "application/json"}
        if not args.no_chat_id:
            headers.update({"X-OpenWebUI-Chat-Id": chat, "X-OpenWebUI-User-Message-Id": user_message, "X-OpenWebUI-Message-Id": message, "X-OpenWebUI-Task": ""})
            if secret:
                headers["X-OpenWebUI-User-Jwt"] = jwt(secret, "citelive-user")
        body = {"model": args.model, "messages": messages, "tools": TOOLS, "max_tokens": 1500, "temperature": 0, "reasoning_effort": "low"}
        started = time.monotonic()
        reply = json.load(urllib.request.urlopen(urllib.request.Request(args.url, data=json.dumps(body).encode(), headers=headers), timeout=900))
        usage = reply.get("usage", {})
        choice = reply["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        print(json.dumps({"round": round_index, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "completion_tokens": usage.get("completion_tokens"), "seconds": round(time.monotonic() - started, 2), "finish": choice["finish_reason"],
                          "calls": [json.loads(c["function"]["arguments"]).get("url") for c in calls], "answer": (choice["message"].get("content") or "")[-160:]}), flush=True)
        if not calls:
            break
        assistant = {"role": "assistant", "content": (choice["message"].get("content") or "").strip(),
                     "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": json.dumps(json.loads(c["function"]["arguments"]), separators=(",", ":"))}} for c in calls]}
        messages = messages + [assistant] + [{"role": "tool", "tool_call_id": c["id"], "content": page(json.loads(c["function"]["arguments"]).get("url", ""), args.salt)} for c in calls]
        for c in calls:
            url = json.loads(c["function"]["arguments"]).get("url", "")
            if url not in fetched:
                fetched.append(url)
        markers = "\n".join(f'<source id="{i + 1}" name="{url}" resource-id="{url}"></source>' for i, url in enumerate(fetched))
        user_index = max(i for i, m in enumerate(messages) if m["role"] == "user")
        messages[user_index] = dict(messages[user_index], content=head + "<context>\n" + markers + "\n</context>\n" + "\n" + question)


if __name__ == "__main__":
    main()
