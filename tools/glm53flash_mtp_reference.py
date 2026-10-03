import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import (bf16_round_f32, bf16_to_f32, f32_to_bf16_u16,
                                 parse_llm_defines, read_fixture, rmsnorm)
import t1_reference_glm53flash as flash

MTP_PREFIX = "model.language_model.layers.{layer}."
ORDERS = ("embed_hidden", "hidden_embed")
HIDDEN_TAPS = ("final_norm", "hc_mean", "streams")
CONTEXTS = ("sequence", "chain")


def load_engine(checkpoint, header):
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    config = config.get("text_config", config)
    engine = flash.Glm53FlashEngine(checkpoint, parse_llm_defines(header), config)
    return engine, config


def fixture_rows(path, final_layer):
    _, arrays = read_fixture(path)
    tokens = [int(t) for t in arrays["prompt_token_ids"]] + [int(t) for t in arrays["generated_token_ids"]]
    streams = []
    for position in range(len(tokens)):
        name = f"pos{position:04d}_layer{final_layer:04d}_streams"
        if name not in arrays:
            raise ValueError(f"{path}: {name} missing; the fixture must capture the final layer")
        streams.append(bf16_to_f32(arrays[name]))
    return tokens, streams, arrays


def index_pool_scores(query, head_weight, keys, gates, ape, context, kpool):
    heads, dim = query.shape
    pools = context // kpool
    pooled = pools * kpool
    logits = gates[:pooled].reshape(pools, kpool, dim).astype(np.float32) + ape[None].astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    mix = np.exp(logits)
    mix = mix / mix.sum(axis=1, keepdims=True)
    pool_key = (mix * keys[:pooled].reshape(pools, kpool, dim)).sum(axis=1)
    scores = np.maximum((query.astype(np.float32) @ pool_key.T) * np.float32(dim ** -0.5), 0.0)
    return (scores * (head_weight.astype(np.float32) * np.float32(heads ** -0.5))[:, None]).sum(axis=0)


def index_positions(query, head_weight, keys, gates, ape, context, topk, kpool):
    if context <= topk:
        return np.arange(context)
    scores = index_pool_scores(query, head_weight, keys, gates, ape, context, kpool)
    chosen = np.argsort(-scores, kind="stable")[:topk // kpool]
    pooled = (context // kpool) * kpool
    positions = (chosen[:, None] * kpool + np.arange(kpool)[None]).reshape(-1)
    return np.sort(np.concatenate([positions, np.arange(pooled, context)]))


class MtpReference:
    def __init__(self, engine, layer):
        self.engine = engine
        self.prefix = MTP_PREFIX.format(layer=layer)
        self.final_norm = engine.tensor("model.language_model.norm.weight")
        self.enorm = engine.tensor(self.prefix + "enorm.weight")
        self.hnorm = engine.tensor(self.prefix + "hnorm.weight")
        self.eh_proj = engine.tensor(self.prefix + "eh_proj.weight")
        self.input_norm = engine.tensor(self.prefix + "input_layernorm.weight")
        self.post_norm = engine.tensor(self.prefix + "post_attention_layernorm.weight")
        self.head_norm = engine.tensor(self.prefix + "shared_head.norm.weight")

    def hidden_tap(self, streams, tap):
        engine = self.engine
        mean = bf16_round_f32(streams.reshape(engine.hc, engine.hidden).mean(axis=0))
        if tap == "hc_mean":
            return mean
        return bf16_round_f32(rmsnorm(mean, self.final_norm, engine.eps))

    def streams_step(self, streams, token, order):
        engine = self.engine
        rows = streams.reshape(engine.hc, engine.hidden)
        outputs = [self.step(rows[index], token, order, [])[0] for index in range(engine.hc)]
        x = bf16_round_f32(np.mean(outputs, axis=0))
        return x, bf16_round_f32(rmsnorm(x, self.head_norm, engine.eps))

    def step(self, hidden, token, order, cache):
        engine = self.engine
        embed = engine.embed(token)
        e = bf16_round_f32(rmsnorm(embed, self.enorm, engine.eps))
        h = bf16_round_f32(rmsnorm(hidden, self.hnorm, engine.eps))
        joined = np.concatenate([e, h] if order == "embed_hidden" else [h, e])
        x = bf16_round_f32(self.eh_proj @ joined)
        attention = engine.dsa_attention(self.prefix, bf16_round_f32(rmsnorm(x, self.input_norm, engine.eps)), cache)
        x = bf16_round_f32(x + attention)
        mlp = engine.sparse_mlp(self.prefix, bf16_round_f32(rmsnorm(x, self.post_norm, engine.eps)), [])
        x = bf16_round_f32(x + mlp)
        return x, bf16_round_f32(rmsnorm(x, self.head_norm, engine.eps))


def head_argmax(engine, columns, targets, chunk=4096):
    lm = engine.st.entry("lm_head.weight")
    matrix = np.stack(columns, axis=1).astype(np.float32)
    count = matrix.shape[1]
    best = np.full(count, -np.inf, dtype=np.float32)
    best_token = np.full(count, -1, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    target_score = np.zeros(count, dtype=np.float64)
    peak = np.full(count, -np.inf, dtype=np.float64)
    total = np.zeros(count, dtype=np.float64)
    for start in range(0, lm["shape"][0], chunk):
        rows = engine.st.raw_rows("lm_head.weight", start, min(chunk, lm["shape"][0] - start))
        scores = bf16_to_f32(rows) @ matrix
        local = np.argmax(scores, axis=0)
        value = scores[local, np.arange(count)]
        better = value > best
        best[better] = value[better]
        best_token[better] = start + local[better]
        high = np.maximum(peak, value.astype(np.float64))
        total = total * np.exp(peak - high) + np.exp(scores.astype(np.float64) - high).sum(axis=0)
        peak = high
        inside = (target >= start) & (target < start + scores.shape[0])
        target_score[inside] = scores[target[inside] - start, np.nonzero(inside)[0]]
    logprob = target_score - peak - np.log(total)
    return best_token.tolist(), logprob.tolist()


def decoded_rows(engine, path, count, final_layer):
    tokens = [int(t) for t in np.fromfile(path, dtype=np.uint32)[:count]]
    states = {}
    caches = {}
    streams = []
    top1 = []
    for position, token in enumerate(tokens):
        held = {}

        def snap(layer_index, layer_streams):
            if layer_index == final_layer:
                held["streams"] = bf16_to_f32(f32_to_bf16_u16(layer_streams.reshape(-1)))

        engine.decode_step(token, position, states, caches, {}, snap)
        streams.append(held["streams"])
        top1.append(engine.logits(held["streams"].reshape(engine.hc, engine.hidden))[0])
        print(json.dumps({"sequence": os.path.basename(path), "position": position, "top1": int(top1[-1])}), flush=True)
    return tokens, streams, top1


def run(arguments):
    engine, config = load_engine(arguments.checkpoint, arguments.header)
    layer = int(config["num_hidden_layers"])
    if int(config.get("num_nextn_predict_layers", 0)) < 1:
        raise ValueError("the checkpoint declares no MTP layer")
    mtp = MtpReference(engine, layer)
    variants = [(o, t, c) for o in arguments.orders.split(",") for t in HIDDEN_TAPS for c in CONTEXTS if not (t == "streams" and c == "sequence")]
    report = {"layer": layer, "fixtures": {}, "variants": {}}
    columns = []
    slots = []
    sources = []
    for path in arguments.fixtures:
        tokens, streams, arrays = fixture_rows(path, layer - 1)
        prompt = len(arrays["prompt_token_ids"])
        check = [int(arrays[f"pos{p:04d}_head_top1_token"][0]) == tokens[p + 1] for p in range(prompt - 1, len(tokens) - 1)]
        sources.append((os.path.basename(path), tokens, streams, {"tokens": len(tokens), "head_top1_matches_stream": all(check)}))
    for spec in arguments.sequence:
        path, count, prompt = spec.split(":")
        tokens, streams, top1 = decoded_rows(engine, path, int(count), layer - 1)
        check = [top1[p] == tokens[p + 1] for p in range(int(prompt) - 1, len(tokens) - 1)]
        sources.append((os.path.basename(path), tokens, streams, {"tokens": len(tokens), "head_top1_matches_stream": all(check),
                                                                    "head_top1_match_count": sum(check), "head_top1_checks": len(check)}))
    for name, tokens, streams, summary in sources:
        report["fixtures"][name] = summary
        for variant in variants:
            order, tap, context = variant
            cache = []
            for position in range(len(tokens) - 2):
                if context == "chain":
                    cache = []
                if tap == "streams":
                    _, head_in = mtp.streams_step(streams[position], tokens[position + 1], order)
                else:
                    _, head_in = mtp.step(mtp.hidden_tap(streams[position], tap), tokens[position + 1], order, cache)
                columns.append(head_in)
                slots.append((variant, name, position, tokens[position + 2]))
                print(json.dumps({"fixture": name, "variant": "/".join(variant), "position": position}), flush=True)
    predicted, logprobs = head_argmax(engine, columns, [slot[3] for slot in slots])
    for (variant, name, position, want), got, logprob in zip(slots, predicted, logprobs):
        key = "/".join(variant)
        row = report["variants"].setdefault(key, {"hits": 0, "checks": 0, "rows": []})
        row["checks"] += 1
        row["hits"] += int(got == want)
        row["rows"].append({"fixture": name, "position": position, "want": want, "got": int(got), "target_logprob": logprob})
    for key, row in report["variants"].items():
        row["acceptance"] = row["hits"] / row["checks"] if row["checks"] else None
        row["mean_target_logprob"] = float(np.mean([r["target_logprob"] for r in row["rows"]])) if row["rows"] else None
        print(f"MTP-REFERENCE {key} {row['hits']}/{row['checks']} mean_target_logprob {row['mean_target_logprob']:.3f}", flush=True)
    with open(arguments.output, "w") as fh:
        json.dump(report, fh, indent=1)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--orders", default=",".join(ORDERS))
    parser.add_argument("--sequence", action="append", default=[], help="tokens.u32:count:prompt_tokens, decoded teacher-forced")
    parser.add_argument("fixtures", nargs="*")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
