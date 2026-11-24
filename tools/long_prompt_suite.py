#!/usr/bin/env python3
"""Long-prompt task suite: two lanes compared on prompts that take their long-prompt path.

build    writes prompts for RULER's retrieval and tracking tasks with several questions a prompt:
         multikey needles in a text haystack (mk), multivalue needles (mv), a haystack made only of
         uuid key-value needles (mk3) and variable chains in noise (vt). Each prompt opens with its own
         task id, so no two prompts share a cached prefix.
run      sends each prompt alone through the chat endpoint: temperature 0, no speculation, the top
         log-probs of every generated token. A rerun skips the prompts its output already answered; with
         --deadline it starts a prompt only when its estimated time fits before the deadline.
score    scores one run question by question.
compare  pairs two runs question by question: each side's score, the questions only one side gets,
         the change with a bootstrap bound whose unit is the prompt, by family and length, and
         where each pair of answers first differs.

usage:
  long_prompt_suite.py build --tokenizer DIR --haystack TEXT --lengths 131000,458000 --prompts 2,2 --seed S --out SUITE.jsonl
  long_prompt_suite.py run --endpoint URL --model NAME --suite SUITE.jsonl --out RUN.jsonl [--only ID,...]
      [--deadline EPOCH --estimate LENGTH=SECONDS,...]
  long_prompt_suite.py score --suite SUITE.jsonl RUN.jsonl
  long_prompt_suite.py compare --suite SUITE.jsonl REFERENCE.jsonl ARM.jsonl [--out CMP.json]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SUITE_FORMAT = "sparkpipe-long-prompt-suite-v1"
SEED = 20260929
REPLICATES = 10000
LEVEL = 0.95
TEMPLATE_TOKENS = 24
QUESTIONS = 8
MK_NEEDLES = 32
MV_KEYS = 8
MV_VALUES = 4
VT_CHAINS = 4
VT_HOPS = 4
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
FAMILIES = ("mk", "mv", "mk3", "vt")
MAX_TOKENS = {"mk": 192, "mv": 384, "mk3": 448, "vt": 192}
WORDS = (
    "apricot badger balloon banjo barnacle beetle biscuit blizzard bramble bucket buffalo cactus camel canoe "
    "caramel carousel cashew cello chestnut chimney cinnamon clover cobbler coconut comet coral cornet cougar "
    "cricket crocus crumpet cucumber dahlia dandelion dolphin donkey dragonfly dumpling eggplant elephant falcon "
    "ferret fiddle flamingo flannel fossil foxglove gazelle gecko geranium ginger giraffe goblet gondola gorilla "
    "granite griffin guava hamster harmonica harpoon hazelnut hedgehog heron hibiscus hippo honeydew hyacinth "
    "iceberg iguana jackal jaguar jasmine jellyfish juniper kangaroo kettle kiwi koala lagoon lantern lavender "
    "lemur leopard lilac llama lobster locket lynx macaw magnolia mallard mandolin mango marmot marigold meadow "
    "meerkat melon mongoose moose mulberry narwhal nectarine nutmeg ocelot octopus okapi olive orchid ostrich "
    "otter oyster paddle pagoda panther papaya parsnip peacock pebble pelican penguin peony pepper petunia pickle "
    "pigeon pineapple pistachio platypus plum porcupine possum pretzel primrose puffin pumpkin quail quince "
    "raccoon radish raven reindeer rhubarb robin rosemary saffron salamander sardine scallop seahorse sequoia "
    "sparrow squash squirrel starling stingray strudel sunflower swan tamarind tangerine tapir teapot thimble "
    "thistle thrush tortoise toucan trombone trumpet tulip turnip tuxedo ukulele umbrella urchin vanilla "
    "violet vulture walnut walrus wasabi weasel whistle wombat yak zebra zucchini"
).split()


def tiktoken_pattern(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pat_str" for target in node.targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "join"
                    and isinstance(value.func.value, ast.Constant) and len(value.args) == 1 and isinstance(value.args[0], (ast.List, ast.Tuple))):
                return value.func.value.value.join(element.value for element in value.args[0].elts)
    raise SystemExit("the tokenizer source names no pat_str")


def load_tokenizer(directory):
    path = Path(directory)
    if (path / "tokenizer.json").exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        digest = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        return (lambda text: tokenizer.encode(text, add_special_tokens=False).ids), tokenizer.decode, digest
    sources = sorted(path.glob("tokenization_*.py"))
    if not (path / "tiktoken.model").exists() or not sources:
        raise SystemExit(f"{path}: neither tokenizer.json nor tiktoken.model with a tokenization_*.py")
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    encoding = tiktoken.Encoding(name=path.name, pat_str=tiktoken_pattern(sources[0].read_text()),
                                 mergeable_ranks=load_tiktoken_bpe(str(path / "tiktoken.model")), special_tokens={})
    digest = hashlib.sha256((path / "tiktoken.model").read_bytes()).hexdigest()
    return (lambda text: encoding.encode(text, disallowed_special=())), encoding.decode, digest


def counter(encode):
    return lambda text: len(encode(text))


def magic_number(rng, used):
    while True:
        value = str(rng.randrange(1000000, 10000000))
        if value not in used:
            used.add(value)
            return value


def variable_name(rng, used):
    while True:
        name = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if name not in used:
            used.add(name)
            return name


def random_uuid(rng):
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def text_haystack(rng, haystack_ids, decode, tokens):
    start = rng.randrange(len(haystack_ids))
    taken, position = [], start
    while len(taken) < tokens:
        piece = haystack_ids[position:position + tokens - len(taken)]
        taken.extend(piece)
        position = 0
    return decode(taken)


def insert_lines(rng, text, needles):
    breaks = [match.end() for match in re.finditer("\n", text)]
    if len(breaks) < len(needles):
        raise SystemExit("the haystack has fewer lines than needles")
    spots = sorted(rng.sample(breaks, len(needles)))
    pieces, last, depths = [], 0, []
    for spot, needle in zip(spots, needles):
        pieces.append(text[last:spot])
        pieces.append(needle + "\n")
        depths.append(spot / len(text))
        last = spot
    pieces.append(text[last:])
    return "".join(pieces), depths


def numbered(lines):
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))


def build_mk(rng, length, count, haystack_ids, decode, multivalue):
    used = set()
    keys = rng.sample(WORDS, MV_KEYS if multivalue else MK_NEEDLES)
    needles, gold = [], {}
    for key in keys:
        values = [magic_number(rng, used) for _ in range(MV_VALUES if multivalue else 1)]
        gold[key] = values
        needles.extend(f"One of the special magic numbers for {key} is: {value}." for value in values)
    rng.shuffle(needles)
    asked = keys if multivalue else rng.sample(keys, QUESTIONS)
    intro = "Some special magic numbers are hidden within the following text. Make sure to memorize them. I will quiz you about the numbers afterwards.\n"
    if multivalue:
        question = ("\nWhat are all the special magic numbers for the following words mentioned in the provided text? Each word has four.\n"
                    + numbered(asked) + "\nAnswer with one line per word, in the form `1. word: number, number, number, number`, and nothing else.")
    else:
        question = ("\nWhat are the special magic numbers for the following words mentioned in the provided text?\n"
                    + numbered(asked) + "\nAnswer with one line per word, in the form `1. word: number`, and nothing else.")
    fixed = count(intro + question + "\n".join(needles))
    text = text_haystack(rng, haystack_ids, decode, length - fixed - TEMPLATE_TOKENS - 16)
    context, depths = insert_lines(rng, text, needles)
    depth_of = {}
    for needle, depth in zip(needles, depths):
        depth_of.setdefault(needle.split(" for ", 1)[1].split(" is: ", 1)[0], []).append(depth)
    items = [{"n": index, "key": key, "gold": gold[key], "depth": round(sum(depth_of[key]) / len(depth_of[key]), 4)}
             for index, key in enumerate(asked, 1)]
    return intro, context, question, items


def build_mk3(rng, length, count):
    intro = "Some special magic uuids are hidden within the following text. Make sure to memorize them. I will quiz you about the uuids afterwards.\n"
    question_head = "\nWhat are the special magic uuids for the following uuids mentioned in the provided text?\n"
    tail = "\nAnswer with one line per uuid asked, in the form `1. uuid`, giving its special magic uuid, and nothing else."
    budget = length - TEMPLATE_TOKENS - 16 - count(intro + question_head + tail) - QUESTIONS * 26
    lines, pairs, total = [], [], 0
    while total < budget:
        key, value = random_uuid(rng), random_uuid(rng)
        line = f"One of the special magic uuids for {key} is: {value}.\n"
        total += count(line)
        lines.append(line)
        pairs.append((key, value))
    chosen = sorted(rng.sample(range(len(pairs)), QUESTIONS))
    rng.shuffle(chosen)
    asked = [pairs[index] for index in chosen]
    question = question_head + numbered(key for key, _ in asked) + tail
    items = [{"n": number, "key": key, "gold": [value], "depth": round(index / len(pairs), 4)}
             for number, ((key, value), index) in enumerate(zip(asked, chosen), 1)]
    return intro, "".join(lines), question, items


def build_vt(rng, length, count):
    used_names, used_values = set(), set()
    chains = []
    for _ in range(VT_CHAINS):
        while True:
            value = str(rng.randrange(10000, 100000))
            if value not in used_values:
                used_values.add(value)
                break
        names = [variable_name(rng, used_names) for _ in range(VT_HOPS + 1)]
        statements = [f"VAR {names[0]} = {value}"] + [f"VAR {names[index]} = VAR {names[index - 1]}" for index in range(1, len(names))]
        chains.append((value, names, statements))
    intro = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    question = ("\nQuestion: Find all variables that are assigned each of the following values in the text above, directly or through other variables.\n"
                + numbered(value for value, _, _ in chains)
                + "\nAnswer with one line per value, in the form `1. NAME, NAME, ...`, listing every variable assigned that value, and nothing else.")
    statements = [statement for _, _, chain in chains for statement in chain]
    fixed = count(intro + question + " ".join(statements) + "." * len(statements))
    repeats = max(len(statements) + 1, (length - TEMPLATE_TOKENS - 16 - fixed) // (count(NOISE * 3) - count(NOISE * 2)))
    owners = [chain_index for chain_index in range(VT_CHAINS) for _ in range(VT_HOPS + 1)]
    rng.shuffle(owners)
    by_chain = {chain_index: [] for chain_index in range(VT_CHAINS)}
    for slot, owner in zip(sorted(rng.sample(range(1, repeats), len(statements))), owners):
        by_chain[owner].append(slot)
    slots = {slot: statement for chain_index, (_, _, chain) in enumerate(chains) for statement, slot in zip(chain, by_chain[chain_index])}
    pieces = []
    for index in range(repeats):
        if index in slots:
            pieces.append(slots[index] + ". ")
        pieces.append(NOISE)
    items = [{"n": number, "key": value, "gold": names, "depth": round(sum(by_chain[number - 1]) / len(by_chain[number - 1]) / repeats, 4)}
             for number, (value, names, _) in enumerate(chains, 1)]
    return intro, "".join(pieces), question, items


def build(args):
    encode, decode, tokenizer_sha256 = load_tokenizer(args.tokenizer)
    count = counter(encode)
    haystack_bytes = Path(args.haystack).read_bytes()
    haystack_ids = encode(haystack_bytes.decode("utf-8"))
    lengths = [int(value) for value in args.lengths.split(",")]
    prompts = [int(value) for value in args.prompts.split(",")]
    if len(prompts) == 1:
        prompts = prompts * len(lengths)
    if len(prompts) != len(lengths):
        raise SystemExit("--prompts takes one count or one per length")
    families = args.families.split(",")
    rng = random.Random(args.seed)
    records = []
    for length, per_family in zip(lengths, prompts):
        for copy in range(per_family):
            for family in families:
                task = f"{rng.getrandbits(48):012x}"
                prompt_rng = random.Random(f"{args.seed}:{family}:{length}:{copy}")
                if family in ("mk", "mv"):
                    intro, context, question, items = build_mk(prompt_rng, length, count, haystack_ids, decode, family == "mv")
                elif family == "mk3":
                    intro, context, question, items = build_mk3(prompt_rng, length, count)
                elif family == "vt":
                    intro, context, question, items = build_vt(prompt_rng, length, count)
                else:
                    raise SystemExit(f"unknown family {family}")
                content = f"Task {task}.\n" + intro + context + question
                tokens = count(content)
                record = {"format": SUITE_FORMAT, "id": f"{family}-{length // 1000}k-{copy}", "family": family, "length": length,
                          "copy": copy, "task": task, "content_tokens": tokens, "max_tokens": MAX_TOKENS[family],
                          "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "items": items,
                          "messages": [{"role": "user", "content": content}]}
                records.append(record)
                print(f"{record['id']}: {tokens} content tokens, {len(items)} questions", flush=True)
    manifest = {"format": SUITE_FORMAT, "seed": args.seed, "lengths": lengths, "prompts": prompts, "families": families,
                "tokenizer_sha256": tokenizer_sha256, "haystack": Path(args.haystack).name,
                "haystack_sha256": hashlib.sha256(haystack_bytes).hexdigest(), "haystack_tokens": len(haystack_ids),
                "prompt_ids": [record["id"] for record in records]}
    with open(args.out, "w") as out:
        for record in records:
            out.write(json.dumps(dict(record, manifest=manifest)) + "\n")
    return 0


def load_suite(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {record["id"]: record for record in records}


def read_runs(path):
    if not Path(path).exists():
        return {}
    return {row["id"]: row for row in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def estimate_table(text):
    table = sorted((int(length), float(seconds)) for length, seconds in (part.split("=") for part in text.split(",") if part))
    def estimate(length):
        for limit, seconds in table:
            if length <= limit:
                return seconds
        return table[-1][1] if table else 0.0
    return estimate


def run(args):
    suite = load_suite(args.suite)
    done = {prompt_id for prompt_id, row in read_runs(args.out).items() if row.get("failure") is None}
    order = args.only.split(",") if args.only else list(suite)
    estimate = estimate_table(args.estimate or "")
    for prompt_id in order:
        if prompt_id in done:
            continue
        record = suite[prompt_id]
        if args.deadline and time.time() + estimate(record["length"]) > args.deadline:
            print(json.dumps({"id": prompt_id, "skipped": "deadline"}), flush=True)
            continue
        body = {"model": args.model, "messages": record["messages"], "temperature": 0, "max_tokens": record["max_tokens"],
                "speculation": False, "logprobs": args.logprobs}
        request = urllib.request.Request(args.endpoint.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                reply = json.loads(response.read())
            failure = None
        except Exception as error:
            reply, failure = {}, f"{type(error).__name__}: {error}"
        wall = time.time() - started
        choice = (reply.get("choices") or [{}])[0]
        usage = reply.get("usage") or {}
        row = {"id": prompt_id, "family": record["family"], "length": record["length"], "content_sha256": record["content_sha256"],
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "wall_s": round(wall, 3), "failure": failure,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish_reason": choice.get("finish_reason"), "content": (choice.get("message") or {}).get("content"),
               "tokens": reply.get("tokens"), "token_logprobs": reply.get("token_logprobs")}
        with open(args.out, "a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps({key: row[key] for key in ("id", "utc", "wall_s", "prompt_tokens", "cached_tokens", "completion_tokens", "finish_reason", "failure")}), flush=True)
        if failure is None and args.show:
            print((row["content"] or "")[:400], flush=True)
    return 0


LINE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*(\d+)\s*\**\s*[.):]\s*\**\s*(.*)$")


def answer_lines(content):
    lines = {}
    for line in (content or "").splitlines():
        match = LINE.match(line)
        if match:
            lines.setdefault(int(match.group(1)), match.group(2))
    if not lines:
        plain = [line for line in (content or "").splitlines() if line.strip()]
        lines = {index: line for index, line in enumerate(plain, 1)}
    return lines


def item_fraction(family, item, line):
    if family in ("mk", "mv"):
        found = set(re.findall(r"\d+", line))
    elif family == "mk3":
        found = {match.lower() for match in re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", line)}
    else:
        found = set(re.findall(r"\b[A-Z]{5}\b", line))
    gold = [value.lower() if family == "mk3" else value for value in item["gold"]]
    return sum(1 for value in gold if value in found) / len(gold)


def score_rows(suite, runs):
    rows = []
    for prompt_id, record in suite.items():
        run_row = runs.get(prompt_id)
        if run_row is None:
            continue
        lines = answer_lines(run_row.get("content"))
        for item in record["items"]:
            fraction = item_fraction(record["family"], item, lines.get(item["n"], "")) if run_row.get("failure") is None else 0.0
            rows.append({"item": f"{prompt_id}:{item['n']}", "prompt": prompt_id, "family": record["family"], "length": record["length"],
                         "depth": item["depth"], "fraction": fraction, "right": fraction == 1.0})
    return rows


def score(args):
    suite = load_suite(args.suite)
    rows = score_rows(suite, read_runs(args.run))
    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["length"]), []).append(row)
    for (family, length), members in sorted(groups.items()):
        right = sum(1 for row in members if row["right"])
        print(f"{family:4s} {length:>8d}: {right}/{len(members)} right, mean fraction {sum(row['fraction'] for row in members) / len(members):.3f}")
    right = sum(1 for row in rows if row["right"])
    print(f"all: {right}/{len(rows)} right")
    return 0


def divergence(reference, arm):
    left, right = reference.get("tokens") or [], arm.get("tokens") or []
    shared = 0
    while shared < min(len(left), len(right)) and left[shared] == right[shared]:
        shared += 1
    out = {"identical": reference.get("content") == arm.get("content") and left == right, "shared_tokens": shared,
           "reference_tokens": len(left), "arm_tokens": len(right)}
    reference_lp, arm_lp = reference.get("token_logprobs") or [], arm.get("token_logprobs") or []
    deltas = []
    for index in range(min(shared, len(reference_lp), len(arm_lp))):
        first, second = reference_lp[index][0][1], arm_lp[index][0][1]
        if first is not None and second is not None:
            deltas.append(second - first)
    out["compared_tokens"] = len(deltas)
    out["chosen_logprob_delta_sum"] = sum(deltas)
    out["chosen_logprob_delta_abs_sum"] = sum(abs(delta) for delta in deltas)
    out["chosen_logprob_delta_mean"] = sum(abs(delta) for delta in deltas) / len(deltas) if deltas else None
    out["chosen_logprob_delta_signed_mean"] = sum(deltas) / len(deltas) if deltas else None
    out["chosen_logprob_delta_max"] = max(abs(delta) for delta in deltas) if deltas else None
    out["identical_logprobs"] = out["identical"] and reference_lp == arm_lp
    if shared < min(len(left), len(right)) and shared < len(reference_lp):
        alternatives = [entry for entry in reference_lp[shared] if entry[1] is not None]
        listed = {token: logprob for token, logprob in alternatives}
        if right[shared] in listed:
            out["divergence_margin"] = round(alternatives[0][1] - listed[right[shared]], 6)
        else:
            out["divergence_margin"] = None
    return out


def compare(args):
    import ab_stats
    suite = load_suite(args.suite)
    reference_runs, arm_runs = read_runs(args.reference), read_runs(args.arm)
    common = [prompt_id for prompt_id in suite if prompt_id in reference_runs and prompt_id in arm_runs
              and reference_runs[prompt_id].get("failure") is None and arm_runs[prompt_id].get("failure") is None]
    subset = {prompt_id: suite[prompt_id] for prompt_id in common}
    reference_rows = {row["item"]: row for row in score_rows(subset, reference_runs)}
    arm_rows = {row["item"]: row for row in score_rows(subset, arm_runs)}

    def summary(items):
        prompts = sorted({reference_rows[item]["prompt"] for item in items})
        only_reference = sum(1 for item in items if reference_rows[item]["right"] and not arm_rows[item]["right"])
        only_arm = sum(1 for item in items if arm_rows[item]["right"] and not reference_rows[item]["right"])
        out = {"prompts": len(prompts), "questions": len(items),
               "reference_right": sum(1 for item in items if reference_rows[item]["right"]),
               "arm_right": sum(1 for item in items if arm_rows[item]["right"]),
               "reference_fraction": sum(reference_rows[item]["fraction"] for item in items) / len(items),
               "arm_fraction": sum(arm_rows[item]["fraction"] for item in items) / len(items),
               "only_reference": only_reference, "only_arm": only_arm,
               "mcnemar_p": ab_stats.mcnemar_exact(only_reference, only_arm),
               "changed_fraction": sum(1 for item in items if reference_rows[item]["fraction"] != arm_rows[item]["fraction"])}
        if len(prompts) > 1:
            for name, key in (("change_right_pct", "right"), ("change_fraction_pct", "fraction")):
                delta = [sum(float(arm_rows[item][key]) - float(reference_rows[item][key]) for item in items if reference_rows[item]["prompt"] == prompt) for prompt in prompts]
                size = [float(sum(1 for item in items if reference_rows[item]["prompt"] == prompt)) for prompt in prompts]
                drawn = ab_stats.bounds(ab_stats.ratio_statistic, [delta, size], args.seed, args.replicates, LEVEL)
                out[name] = {key: 100.0 * drawn[key] for key in ("estimate", "lower", "upper")}
        return out

    items = sorted(reference_rows)
    report = {"reference": args.reference, "arm": args.arm, "all": summary(items), "by_family_length": {}, "by_length": {}, "by_family": {},
              "prompts": {}}
    for family in sorted({row["family"] for row in reference_rows.values()}):
        report["by_family"][family] = summary([item for item in items if reference_rows[item]["family"] == family])
    for length in sorted({row["length"] for row in reference_rows.values()}):
        report["by_length"][str(length)] = summary([item for item in items if reference_rows[item]["length"] == length])
        for family in sorted({row["family"] for row in reference_rows.values()}):
            chosen = [item for item in items if reference_rows[item]["length"] == length and reference_rows[item]["family"] == family]
            if chosen:
                report["by_family_length"][f"{family}-{length}"] = summary(chosen)
    for prompt_id in common:
        row = divergence(reference_runs[prompt_id], arm_runs[prompt_id])
        row.update({"reference_wall_s": reference_runs[prompt_id]["wall_s"], "arm_wall_s": arm_runs[prompt_id]["wall_s"],
                    "prompt_tokens": reference_runs[prompt_id].get("prompt_tokens"),
                    "same_prompt_tokens": reference_runs[prompt_id].get("prompt_tokens") == arm_runs[prompt_id].get("prompt_tokens"),
                    "reference_right": sum(1 for item in items if reference_rows[item]["prompt"] == prompt_id and reference_rows[item]["right"]),
                    "arm_right": sum(1 for item in items if arm_rows[item]["prompt"] == prompt_id and arm_rows[item]["right"]),
                    "questions": sum(1 for item in items if reference_rows[item]["prompt"] == prompt_id)})
        report["prompts"][prompt_id] = row
    report["items"] = [{"item": item, "reference": reference_rows[item]["fraction"], "arm": arm_rows[item]["fraction"]}
                       for item in items if reference_rows[item]["fraction"] != arm_rows[item]["fraction"]]
    whole = report["all"]
    print(f"{whole['questions']} questions in {whole['prompts']} prompts: reference {whole['reference_right']} right, arm {whole['arm_right']}; "
          f"only reference {whole['only_reference']}, only arm {whole['only_arm']} (exact McNemar p {whole['mcnemar_p']:.4f})")
    for name, block in (("family", report["by_family"]), ("length", report["by_length"]), ("family-length", report["by_family_length"])):
        for key, value in block.items():
            print(f"  {name} {key}: {value['reference_right']}/{value['questions']} vs {value['arm_right']}/{value['questions']}, "
                  f"fraction {value['reference_fraction']:.3f} vs {value['arm_fraction']:.3f}, only reference {value['only_reference']}, only arm {value['only_arm']}")
    rows = list(report["prompts"].values())
    compared = sum(row["compared_tokens"] for row in rows)
    report["logprobs"] = {"compared_tokens": compared,
                          "chosen_logprob_delta_mean": sum(row["chosen_logprob_delta_abs_sum"] for row in rows) / compared if compared else None,
                          "chosen_logprob_delta_signed_mean": sum(row["chosen_logprob_delta_sum"] for row in rows) / compared if compared else None,
                          "chosen_logprob_delta_max": max((row["chosen_logprob_delta_max"] for row in rows if row["chosen_logprob_delta_max"] is not None), default=None)}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    identical = sum(1 for row in rows if row["identical"])
    print(f"  answers identical in {identical} of {len(rows)} prompts; over {compared} answer tokens both sides generated alike, "
          f"the chosen token's log-prob moves {report['logprobs']['chosen_logprob_delta_mean'] or 0.0:.4f} on average "
          f"({report['logprobs']['chosen_logprob_delta_signed_mean'] or 0.0:+.4f} signed, arm minus reference), at most {report['logprobs']['chosen_logprob_delta_max'] or 0.0:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build")
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--haystack", required=True)
    command.add_argument("--lengths", required=True)
    command.add_argument("--prompts", default="1")
    command.add_argument("--families", default=",".join(FAMILIES))
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--out", required=True)
    command = commands.add_parser("run")
    command.add_argument("--endpoint", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--suite", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--only")
    command.add_argument("--deadline", type=float)
    command.add_argument("--estimate")
    command.add_argument("--logprobs", type=int, default=5)
    command.add_argument("--timeout", type=float, default=3600.0)
    command.add_argument("--show", action="store_true")
    command = commands.add_parser("score")
    command.add_argument("--suite", required=True)
    command.add_argument("run")
    command = commands.add_parser("compare")
    command.add_argument("--suite", required=True)
    command.add_argument("reference")
    command.add_argument("arm")
    command.add_argument("--out")
    command.add_argument("--seed", type=int, default=SEED)
    command.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args()
    return {"build": build, "run": run, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
