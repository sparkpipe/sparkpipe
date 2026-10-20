import argparse
import base64
import hashlib
import hmac
import json
import time
import urllib.request
import uuid
from pathlib import Path

FACTS = [
    "Remember this for later: the Ostrel project's budget code is QX-7731 and its review is in room 412.",
    "Give me three short title ideas for the Halcyon launch post.",
    "In one sentence, why is the sky blue?",
    "What is 17 times 23? Answer with the number only.",
    "Name one benefit of a prefix cache, in one sentence.",
    "Suggest a name for a grey cat, one word.",
    "What was the second title idea you gave for the Halcyon launch post? Reply with the title only.",
    "Which room is the Ostrel review in, and what is its budget code? Reply in one short sentence.",
]


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def jwt(secret, sub):
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    body = b64(json.dumps({"sub": sub, "iss": "open-webui", "iat": now, "exp": now + 300}).encode())
    return f"{head}.{body}.{b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--think", type=float, default=0.0)
    parser.add_argument("--salt", default="")
    parser.add_argument("--questions", help="a file with one question per line instead of the built-in conversation")
    args = parser.parse_args()
    questions = [line.strip() for line in Path(args.questions).read_text().splitlines() if line.strip()] if args.questions else FACTS
    secret = Path(args.secret_file).read_text().strip()
    chat = "memlive-" + uuid.uuid4().hex[:12]
    own = Path(args.store) / hashlib.sha256(b"memlive-user").hexdigest()[:32] / (hashlib.sha256(chat.encode()).hexdigest()[:32] + ".jsonl")
    history = [{"role": "system", "content": "You are SparkPipe Chat, a helpful assistant."}]
    parent = None
    out = Path(args.out).open("w")
    for turn, question in enumerate(questions, start=1):
        user_message, message = str(uuid.uuid4()), str(uuid.uuid4())
        headers = {"Content-Type": "application/json", "X-OpenWebUI-User-Jwt": jwt(secret, "memlive-user"), "X-OpenWebUI-Chat-Id": chat,
                   "X-OpenWebUI-User-Message-Id": user_message, "X-OpenWebUI-Message-Id": message, "X-OpenWebUI-Task": ""}
        if parent:
            headers["X-OpenWebUI-User-Message-Parent-Id"] = parent
        if turn == 1 and args.salt:
            question = f"(Run {args.salt}.) " + question
        body = {"model": args.model, "messages": history + [{"role": "user", "content": question}], "max_tokens": 700, "temperature": 0, "reasoning_effort": "low"}
        started = time.monotonic()
        reply = json.load(urllib.request.urlopen(urllib.request.Request(args.url, data=json.dumps(body).encode(), headers=headers), timeout=900))
        seconds = time.monotonic() - started
        answer = reply["choices"][0]["message"].get("content") or ""
        usage = reply.get("usage", {})
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.strip()}]
        parent = message
        waited = 0.0
        while waited < 90:
            records = [json.loads(line) for line in own.read_text().splitlines() if line.strip()] if own.is_file() else []
            if any(r.get("type") == "pending" and r.get("message") == message for r in records):
                break
            time.sleep(1)
            waited += 1
        time.sleep(args.think)
        row = {"turn": turn, "question": question, "answer": answer[-200:], "seconds": round(seconds, 2), "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "line_wait_s": waited}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
    state = [json.loads(line) for line in own.read_text().splitlines() if line.strip()]
    print(json.dumps({"lines": [r["line"] for r in state if r.get("type") == "line"], "pending": len([r for r in state if r.get("type") == "pending"]),
                      "expansions": len([r for r in state if r.get("type") == "expansion"]), "layouts": [r.get("lined") for r in state if r.get("type") == "layout"]}, indent=1))


if __name__ == "__main__":
    main()
