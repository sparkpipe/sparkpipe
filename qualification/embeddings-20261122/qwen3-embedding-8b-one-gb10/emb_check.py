import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ours", default="http://127.0.0.1:8491")
parser.add_argument("--reference", default="http://127.0.0.1:8477")
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", default="corpus,first,agreement,ranking,invariance,speed")
parser.add_argument("--max-tokens", type=int, default=1900)
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()
steps = set(args.steps.split(","))
args.out.mkdir(parents=True, exist_ok=True)
INSTRUCT = "Instruct: Given a question about a software system, retrieve the passage that answers it\nQuery: "


def post(url, body, timeout=1800):
    request = urllib.request.Request(url + "/v1/embeddings", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        answer = json.loads(response.read())
    return answer, time.monotonic() - started


def vectors_of(answer):
    rows = sorted(answer["data"], key=lambda item: item["index"])
    return np.stack([np.frombuffer(base64.b64decode(row["embedding"]), dtype="<f4") for row in rows])


def embed(url, inputs, batch):
    out, tokens, seconds = [], 0, 0.0
    for first in range(0, len(inputs), batch):
        answer, took = post(url, {"input": inputs[first:first + batch], "encoding_format": "base64"})
        out.append(vectors_of(answer))
        tokens += answer["usage"]["prompt_tokens"]
        seconds += took
    return np.concatenate(out), tokens, seconds


def corpus():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    paragraphs = []
    for path in sorted(list((args.repo / "docs").glob("*.md")) + [args.repo / "README.md", args.repo / "SPEC.md", args.repo / "sparkpipe_invariants.md"]):
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if len(block) >= 40 and not block.startswith("|") and not block.startswith("```"):
                paragraphs.append(block)
    rng = np.random.default_rng(20261122)
    order = rng.permutation(len(paragraphs))
    paragraphs = [paragraphs[index] for index in order]
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True).ids)
    short, medium, long_ = [], [], []
    cursor = 0
    for block in paragraphs:
        sentence = re.split(r"(?<=[.:;])\s", block)[0][:200]
        if len(short) < 64 and 4 <= count(sentence) <= 40 and sentence not in short:
            short.append(sentence)
        elif len(medium) < 96 and 50 <= count(block) <= 400 and block not in medium:
            medium.append(block)
    while len(long_) < 48 and cursor + 12 < len(paragraphs):
        size = int(rng.integers(4, 12))
        joined = "\n\n".join(paragraphs[cursor:cursor + size])
        cursor += size
        if 450 <= count(joined) <= args.max_tokens:
            long_.append(joined)
    other = [
        "스파크파이프는 고객이 플릿 전체를 시간 단위로 빌려 쓰는 추론 시스템입니다. 모델 가중치는 각 노드에 미리 올라가 있고, 요청은 열여섯 개의 랭크가 함께 계산합니다.",
        "这个系统把一个模型分成十六个部分，每台机器只保存其中一部分权重，所有机器通过高速网络共同完成每一步计算。",
        "このシステムはプロンプトの接頭辞をキャッシュし、同じ会話の続きを計算するときは新しいトークンだけを処理します。",
        "Le moteur garde en mémoire les préfixes déjà calculés, de sorte qu'une conversation reprend là où elle s'était arrêtée.",
        "Der Speicher merkt sich, was in früheren Gesprächen gesagt wurde, und ordnet die Erinnerungen nach ihrer Bedeutung.",
        "def cosine(a, b):\n    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))",
        "static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue) { Frame frame; return Begin(state, step, queue, &frame); }",
        "SELECT account, sum(fleet_hours) FROM bookings WHERE started >= '2026-11-01' GROUP BY account ORDER BY 2 DESC LIMIT 10;",
        "El usuario pregunta cuánto tarda en cargarse un modelo de un billón de parámetros y si la caché sobrevive a un reinicio.",
        "Пользователь просит объяснить, почему первый токен приходит позже, когда префикс не найден в кэше.",
        "a", "Yes.", "42", "RDMA", "weightd", "What is the capital of France?",
    ]
    questions = [
        "How does the engine reuse a prompt prefix it has already computed?", "What happens to a request when a rank disconnects?",
        "Where are the model weights kept on each node?", "How is a decode step verified when speculation is on?",
        "What does the weight daemon own?", "How does a customer delete their cached data?", "Which precision do the Qwen models serve at?",
        "How many sequences can one lane hold at once?", "What is the role of the hub?", "How are files copied between machines?",
        "How does the chat layer reach the memory program?", "What is a stage pack?", "How is a release rolled out to the fleet?",
        "What limits the size of the device KV pool?", "How are tool calls parsed from the model's output?", "What is the mesh used for?",
        "When does a prefill run as more than one wave?", "What does the swap controller do?", "How is a tokenizer loaded by the API?",
        "What is batch invariance and why does it matter?", "How does the API answer when the engine is starting?", "What is a lane record?",
        "How is sampling made reproducible?", "What does the node agent restart?", "How does the snapshot store survive a restart?",
        "What is the purpose of the DMA guard?", "How are drafts produced for Kimi K3?", "Which calls does the public API allow?",
        "How does a deployment declare its chat template?", "What is tensor parallel rank?", "How are experts leased from weightd?",
        "What does a completion carry back to the batch engine?",
    ]
    queries = [INSTRUCT + question for question in questions]
    texts = short + medium + long_ + other + queries
    groups = {"short": len(short), "medium": len(medium), "long": len(long_), "other": len(other), "queries": len(queries)}
    ids = [tokenizer.encode(text, add_special_tokens=True).ids for text in texts]
    (args.out / "corpus.json").write_text(json.dumps({"texts": texts, "ids": ids, "groups": groups}, ensure_ascii=False))
    lengths = [len(row) for row in ids]
    print(json.dumps({"step": "corpus", "texts": len(texts), "groups": groups, "tokens": sum(lengths), "min": min(lengths), "max": max(lengths),
                      "median": int(np.median(lengths))}))


def load():
    return json.loads((args.out / "corpus.json").read_text())


def first():
    data = load()
    pick = [0, len(data["texts"]) - 1]
    ours = embed(args.ours, [data["texts"][index] for index in pick], 2)[0]
    reference = embed(args.reference, [data["texts"][index] for index in pick], 2)[0]
    cos = [float(np.dot(ours[row], reference[row])) for row in range(len(pick))]
    print(json.dumps({"step": "first", "shape": list(ours.shape), "norms": [float(np.linalg.norm(row)) for row in ours], "cosine_to_reference": cos,
                      "max_abs_difference": float(np.abs(ours - reference).max()), "ours_head": [float(value) for value in ours[0][:6]],
                      "reference_head": [float(value) for value in reference[0][:6]]}))


def agreement():
    data = load()
    texts, ids = data["texts"], data["ids"]
    ours, our_tokens, our_seconds = embed(args.ours, texts, args.batch)
    np.save(args.out / "ours.npy", ours)
    reference, reference_tokens, reference_seconds = embed(args.reference, texts, args.batch)
    np.save(args.out / "reference.npy", reference)
    by_ids = embed(args.ours, ids, args.batch)[0]
    cosine = np.sum(ours * reference, axis=1)
    difference = np.abs(ours - reference)
    bounds, start, per_group = data["groups"], 0, {}
    for name, size in bounds.items():
        part = cosine[start:start + size]
        per_group[name] = {"texts": size, "min": float(part.min()), "mean": float(part.mean())}
        start += size
    print(json.dumps({"step": "agreement", "texts": len(texts), "tokens_ours": our_tokens, "tokens_reference": reference_tokens,
                      "tokenizer_text_equals_ids": bool(np.array_equal(ours, by_ids)), "rows_differing_text_vs_ids": int(np.any(ours != by_ids, axis=1).sum()),
                      "cosine": {"min": float(cosine.min()), "p01": float(np.quantile(cosine, 0.01)), "median": float(np.median(cosine)), "mean": float(cosine.mean()), "max": float(cosine.max())},
                      "worst_text": int(cosine.argmin()), "worst_tokens": len(ids[int(cosine.argmin())]), "max_abs_difference": float(difference.max()),
                      "mean_abs_difference": float(difference.mean()), "norm_ours": [float(np.linalg.norm(ours, axis=1).min()), float(np.linalg.norm(ours, axis=1).max())],
                      "per_group": per_group, "seconds_ours": round(our_seconds, 3), "seconds_reference": round(reference_seconds, 3)}))


def ranking():
    data = load()
    ours, reference = np.load(args.out / "ours.npy"), np.load(args.out / "reference.npy")
    queries = data["groups"]["queries"]
    documents = len(data["texts"]) - queries
    score_ours = ours[documents:] @ ours[:documents].T
    score_reference = reference[documents:] @ reference[:documents].T
    order_ours, order_reference = np.argsort(-score_ours, axis=1), np.argsort(-score_reference, axis=1)
    delta = np.abs(score_ours - score_reference)
    result = {"step": "ranking", "queries": queries, "documents": documents, "score_difference": {"max": float(delta.max()), "mean": float(delta.mean()), "p99": float(np.quantile(delta, 0.99))}}
    for depth in (1, 5, 10, 20):
        same_order = float(np.mean([np.array_equal(order_ours[row][:depth], order_reference[row][:depth]) for row in range(queries)]))
        overlap = float(np.mean([len(set(order_ours[row][:depth]) & set(order_reference[row][:depth])) / depth for row in range(queries)]))
        result[f"top{depth}"] = {"same_order": same_order, "overlap": overlap}
    spearman = []
    for row in range(queries):
        rank_ours, rank_reference = np.argsort(np.argsort(-score_ours[row])), np.argsort(np.argsort(-score_reference[row]))
        spearman.append(float(np.corrcoef(rank_ours, rank_reference)[0, 1]))
    result["spearman"] = {"min": min(spearman), "mean": float(np.mean(spearman))}
    sorted_reference = -np.sort(-score_reference, axis=1)
    gaps = sorted_reference[:, :-1] - sorted_reference[:, 1:]
    result["reference_gaps"] = {"top1_min": float(gaps[:, 0].min()), "top1_median": float(np.median(gaps[:, 0])), "top10_median": float(np.median(gaps[:, :10])),
                                "all_median": float(np.median(gaps)), "top10_below_twice_max_difference": float(np.mean(gaps[:, :10] < 2.0 * delta.max())),
                                "all_below_twice_max_difference": float(np.mean(gaps < 2.0 * delta.max()))}
    swaps = 0
    for row in range(queries):
        for position in range(min(10, documents) - 1):
            first_doc, second_doc = order_reference[row][position], order_reference[row][position + 1]
            if score_ours[row][first_doc] < score_ours[row][second_doc]:
                swaps += 1
    result["adjacent_swaps_in_top10"] = swaps
    print(json.dumps(result))


def invariance():
    data = load()
    texts = data["texts"]
    pick = [0, 3, 70, 71, 170, 171, len(texts) - 1, len(texts) - 2]
    alone = np.concatenate([embed(args.ours, [texts[index]], 1)[0] for index in pick])
    again = np.concatenate([embed(args.ours, [texts[index]], 1)[0] for index in pick])
    together = embed(args.ours, [texts[index] for index in pick], len(pick))[0]
    crowd = list(range(40, 40 + 24))
    crowded = embed(args.ours, [texts[index] for index in pick + crowd], len(pick) + len(crowd))[0][:len(pick)]
    fresh = ["invariance probe %d: " % index + texts[index] for index in pick]
    fresh_alone = np.concatenate([embed(args.ours, [text], 1)[0] for text in fresh])
    fresh_together = embed(args.ours, fresh, len(fresh))[0]
    reference_alone = np.concatenate([embed(args.reference, [texts[index]], 1)[0] for index in pick])
    reference_together = embed(args.reference, [texts[index] for index in pick + crowd], len(pick) + len(crowd))[0][:len(pick)]
    compare = lambda a, b: {"rows_bit_equal": int(np.all(a == b, axis=1).sum()), "rows": len(a), "min_cosine": float(np.sum(a * b, axis=1).min()), "max_abs_difference": float(np.abs(a - b).max())}
    print(json.dumps({"step": "invariance", "ours_repeat_alone": compare(alone, again), "ours_alone_vs_batch_of_8": compare(alone, together),
                      "ours_alone_vs_batch_of_32": compare(alone, crowded), "ours_uncached_alone_vs_batch": compare(fresh_alone, fresh_together),
                      "reference_alone_vs_batch_of_32": compare(reference_alone, reference_together)}))


def speed():
    data = load()
    texts, ids = data["texts"], data["ids"]
    tokens = sum(len(row) for row in ids)
    salt = str(int(time.time()))
    salted = lambda label: [f"{label} {salt} {index}. " + text for index, text in enumerate(texts)]
    results = {"step": "speed", "texts": len(texts), "tokens": tokens}
    for name, url, batches in (("ours", args.ours, (1, 16, 64, 256)), ("reference", args.reference, (1, 16, 64, 256))):
        for batch in batches:
            sample = salted(f"{name} b{batch}")
            if batch == 1:
                sample = sample[:48]
            _, counted, seconds = embed(url, sample, batch)
            results[f"{name}_batch{batch}"] = {"texts": len(sample), "tokens": counted, "seconds": round(seconds, 3), "texts_per_second": round(len(sample) / seconds, 2),
                                               "tokens_per_second": round(counted / seconds, 1)}
    short = [f"short {salt} {index}: " + text for index, text in enumerate(texts[:64])]
    for name, url in (("ours", args.ours), ("reference", args.reference)):
        _, counted, seconds = embed(url, short, 64)
        results[f"{name}_short64"] = {"texts": len(short), "tokens": counted, "seconds": round(seconds, 3), "texts_per_second": round(len(short) / seconds, 2),
                                      "tokens_per_second": round(counted / seconds, 1)}
    print(json.dumps(results))


for name, function in (("corpus", corpus), ("first", first), ("agreement", agreement), ("ranking", ranking), ("invariance", invariance), ("speed", speed)):
    if name in steps:
        function()
        sys.stdout.flush()
