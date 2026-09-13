#!/usr/bin/env python3
"""COMPSEC-17 eval through a SparkPipe residentd endpoint (spark3 qwen38.fp8.tp1).

Encodes everything learned 2026-08-25:
  - canonical 17-question MC set (fewshot-prompted) built into ONE sparkbatch
  - waits for an idle endpoint (no concurrent model_batch) + stability window
    before firing - the shared box is contended and mid-run daemon swaps happen
  - decodes completions with the OFFICIAL Qwen/Qwen3.8-27B tokenizer.json
    (cached under /tmp/compsec17/) - never guess letters from raw token ids
  - grades with qualification/ds4_eval/compare_runs.find_answer_letter
    (ds4_eval.c convention: last "answer:" marker, negation-aware,
    reverse-scan fallback)

Usage:
  run_compsec17.py --host spark3 --runtime-root /home/spark3/sparkdata/qwen38.fp8.tp1
  run_compsec17.py --grade-only OUTFILE            # rescore existing batch .out
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSET_DIR = "/tmp/compsec17"
TOKENIZER_URLS = [
    "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json",
    "https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/resolve/main/tokenizer.json",
]

QS = [
 ("Which HTTP cookie attribute prevents client-side JavaScript from reading the cookie?","Secure","HttpOnly","SameSite","Partitioned","B"),
 ("A web app builds SQL by concatenating user input into query strings. Which vulnerability does this enable?","Cross-site scripting","SQL injection","CSRF","Open redirect","B"),
 ("Which port does HTTPS use by default?","21","22","443","8080","C"),
 ("A TLS server certificate binds an identity to what?","A username","A public key","A password","A MAC address","B"),
 ("What does AES stand for?","Advanced Encryption Standard","Applied Encryption Scheme","Authenticated Encrypted Session","Asymmetric Encryption Standard","A"),
 ("Which attack causes a victims browser to send an authenticated request to a different site without their knowledge?","XSS","CSRF","SSRF","DNS rebinding","B"),
 ("Which combination most directly mitigates stack-smashing return redirection?","Longer passwords","Stack canaries with ASLR and NX","Host firewalls","Access logging","B"),
 ("How should user passwords be stored server-side?","MD5","SHA-1","Argon2 or bcrypt with a per-password salt","Base64 encoding","C"),
 ("Gaining root from a standard user account is called what?","Horizontal privilege escalation","Vertical privilege escalation","Lateral movement","Persistence","B"),
 ("What is a JWT signature verified with?","The servers secret or public key","The users password","Nothing; payloads are trusted","The token itself","A"),
 ("Which of these is a symmetric cipher?","RSA","ECC","ChaCha20","DSA","C"),
 ("DNS cache poisoning primarily corrupts what?","Routers","Resolver caches","Firewalls","DHCP leases","B"),
 ("Which protocol issues time-limited tickets so passwords are never sent to services?","OAuth 1.0","Kerberos","SAML","NTLMv1","B"),
 ("What does XSS primarily let an attacker do?","Run script in the victims browser context","Read kernel memory","Crack password hashes","Spoof ARP entries","A"),
 ("The principle of least privilege means what?","Grant only the rights required for a task","Give every user admin rights","Rotate all passwords hourly","Deny all network access","A"),
 ("Which pair is a classic phishing indicator?","An extended-validation certificate","A look-alike domain with urgent language","An HTTPS padlock","A short email body","B"),
 ("Why does two-factor authentication improve security?","It encrypts network traffic","Compromising one factor alone is insufficient","It makes passwords longer","It speeds up logins","B"),
]
FEWSHOT = ("Question: Which HTTP method is idempotent and safe for repeated retries?\n"
           "A) POST\nB) GET\nC) PATCH\nD) CONNECT\nAnswer: B\n\n")

class Tok:
    def __init__(self, path):
        tk = json.load(open(path))
        self.ranks = {}
        for i, m in enumerate(tk["model"].get("merges", [])):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = i
        self.vocab = dict(tk["model"]["vocab"])
        self.inv = {v: k for k, v in self.vocab.items()}
        self.added = {t["id"]: t["content"] for t in tk.get("added_tokens", [])}
        bs = list(range(33,127))+list(range(161,173))+list(range(174,256))
        b2u = {}; n = 0
        for b in range(256):
            if b in bs: b2u[b] = chr(b)
            else: b2u[b] = chr(256+n); n += 1
        self.u2b = {c: b for b, c in b2u.items()}
        self.b2u = dict(b2u)
    def _bpe(self, piece):
        w = [chr(c) if 33 <= c < 127 else "" for c in piece]
        w = []
        for c in piece.decode("utf-8"):
            w.append(c)
        while len(w) > 1:
            best = None; bi = -1
            for p in range(len(w)-1):
                r = self.ranks.get((w[p], w[p+1]))
                if r is not None and (best is None or r < best):
                    best = r; bi = p
            if bi < 0: break
            w = w[:bi]+[w[bi]+w[bi+1]]+w[bi+2:]
        return w
    def enc(self, text):
        ids = []
        for piece in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+", text):
            mapped = "".join(self.b2u.get(b, "\uFFFD") for b in piece.encode())
            for tok in self._bpe(mapped.encode()):
                if tok in self.vocab: ids.append(self.vocab[tok])
        return ids
    def dec(self, ids):
        out = bytearray()
        for i in ids:
            if i in self.added: out += str(self.added[i]).encode(); continue
            for ch in self.inv.get(i, ""): out.append(self.u2b.get(ch, 63))
        return out.decode("utf-8", errors="replace")

def get_tokenizer():
    p = os.path.join(ASSET_DIR, "tokenizer_dl.json")
    if not os.path.exists(p):
        os.makedirs(ASSET_DIR, exist_ok=True)
        import urllib.request
        last = None
        for u in TOKENIZER_URLS:
            try:
                urllib.request.urlretrieve(u, p); last = None; break
            except Exception as e: last = e
        if last: raise SystemExit(f"cannot fetch tokenizer: {last}")
    return Tok(p)

def build_batch(tok):
    reqs, key = [], {}
    for i, (q, a, b, c, d, ans) in enumerate(QS):
        text = FEWSHOT + f"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"
        ids = tok.enc(text)
        assert tok.dec(ids) == text, f"roundtrip fail q{i}"
        rid = 920000 + i
        reqs.append({"request_id": rid, "sequence_id": rid, "priority": 0,
                     "output_token_budget": 8, "prompt_token_ids": ids})
        key[str(rid)] = ans
    batch = {"schema_version": 1, "connect_timeout_ms": 60000, "request_capacity": 32,
             "max_context_tokens": 4096, "max_prefill_rows_per_submission": 8,
             "maximum_messages_per_rank_per_progress": 16,
             "maximum_new_submissions_per_progress": 8, "stop_token_ids": [],
             "requests": reqs}
    json.dump(batch, open(os.path.join(ASSET_DIR, "batch_compsec17.json"), "w"))
    json.dump({"key": key}, open(os.path.join(ASSET_DIR, "key.json"), "w"))
    return len(reqs), max(len(r["prompt_token_ids"]) for r in reqs)

def wait_idle(host, timeout_s=900):
    def sh(cmd): return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                                       capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        busy = sh("pgrep -c -f '[m]odel_batch' || true")
        up = sh("ss -ltn 2>/dev/null | grep -c ':17480 ' || true")
        if busy == "0" and up != "0":
            time.sleep(45)  # stability window: deploys settle, half-up states clear
            if sh("pgrep -c -f '[m]odel_batch' || true") == "0":
                return True
        time.sleep(15)
    return False

def grade(out_path, tok):
    evs = [json.loads(l) for l in open(out_path) if l.strip()]
    seqs = {}
    meta = {}
    for e in evs:
        if e.get("event") == "token":
            seqs.setdefault(e["request_id"], []).append((e["token_index"], e["token_id"]))
        if e.get("event") == "ready":
            meta["adapter_id"] = e.get("adapter_id"); meta["model_revision"] = e.get("model_revision")
    sys.path.insert(0, os.path.join(REPO, "qualification", "ds4_eval"))
    from compare_runs import find_answer_letter
    key = {920000+i: QS[i][5] for i in range(len(QS))}
    rows, score = [], 0
    for rid in sorted(seqs):
        txt = tok.dec([t for _, t in sorted(seqs[rid])])
        got = find_answer_letter(txt, 4)
        ok = got == key[rid]; score += ok
        rows.append({"request_id": rid, "got": got, "expected": key[rid], "pass": ok, "text": txt})
    return {"score": score, "total": len(QS), "meta": meta, "items": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--runtime-root", default="/home/spark3/sparkdata/qwen38.tp1")
    ap.add_argument("--grade-only", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "receipt_compsec17.json"))
    args = ap.parse_args()
    tok = get_tokenizer()
    if args.grade_only:
        result = grade(args.grade_only, tok)
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"SCORE: {result['score']}/{result['total']}  ({args.grade_only})")
        for r in result["items"]:
            print(f"  {r['request_id']}: got={r['got']} exp={r['expected']} {'PASS' if r['pass'] else 'MISS'} | {r['text'][:50]!r}")
        return
    n, maxlen = build_batch(tok)
    print(f"batch built: {n} requests, max prompt {maxlen} tokens")
    if args.host:
        if not wait_idle(args.host): raise SystemExit("endpoint never settled")
        print("endpoint idle; firing")
        batch = os.path.join(ASSET_DIR, "batch_compsec17.json")
        outp = os.path.join(ASSET_DIR, "compsec17_run.out")
        cmd = f"cd {args.runtime_root} && timeout 900 ./bin/sparkpipe_model_batch --deployment config/model_resident.json --runtime-root $PWD --batch {batch} > {outp} 2> {outp}.err"
        rc = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, cmd]).returncode
        if rc != 0: raise SystemExit(f"batch failed rc={rc}; see {outp}.err")
        err = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, f"tail -1 {outp}.err"],
                             capture_output=True, text=True).stdout.strip()
        print("daemon-side:", err)
        data = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, f"cat {outp}"],
                              capture_output=True, text=True).stdout
        local = os.path.join(HERE, "compsec17_run.out.jsonl")
        open(local, "w").write(data)
        result = grade(local, tok)
    else:
        raise SystemExit("local endpoint mode not supported; pass --host")
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"SCORE: {result['score']}/{result['total']}")

if __name__ == "__main__":
    main()
