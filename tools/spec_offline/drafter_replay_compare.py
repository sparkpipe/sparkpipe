#!/usr/bin/env python3
"""Replay a tap recording through a latent block drafter as block_draftd.py serves it, and compare with a drafter module.

    drafter_replay_compare.py --checkpoint DIR --target SHARD --recording FILE --module-drafts FILE [--rule SEED,INVTEMP,NUCLEUS,TOP_K]
                              [--context-bytes N] [--depth N] [--limit N] [--embed NAME --head NAME --norm NAME]

`build/drafter_module_replay` drives a drafter module through the recording and writes one line per request. This
script replays the same request schedule through the Python drafter with block_draftd.py's context store (LatentContext:
owners, rewinds, donor prefixes, token checks) and its eager forward and heads: before each decode, verify or chain
frame at position p > 0, a request at p with the frame's first token as the anchor; then the frame is written. It
prints how many requests both sides answered alike, how many drafts are equal token for token, and each side's mean
accepted length against the recording's committed stream (the last token written at each position).
"""
import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from block_drafter import open_drafter  # noqa: E402
from context_retention import evict_contexts  # noqa: E402
from latent_block_drafter import LatentContext  # noqa: E402
from tapstream import frames  # noqa: E402

ANSWERING_FLAGS = 2 | 4 | 8


def committed_streams(recording):
    streams = {}
    for _, frame in frames(recording):
        key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
        stream = streams.setdefault(key, {})
        for offset, token in enumerate(frame["tokens"]):
            stream[frame["first_position"] + offset] = token
    return streams


def module_answers(path):
    answers = []
    for line in open(path):
        if not line.startswith("draft "):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        drafts = [int(item.split(":")[0]) for item in fields["drafts"].split(",") if item]
        answers.append(((int(fields["control"]), int(fields["request"]), int(fields["sequence"])), int(fields["position"]), fields["answer"], drafts))
    return answers


def accepted(drafts, stream, position):
    count = 0
    for offset, token in enumerate(drafts):
        if stream.get(position + 1 + offset) != token:
            break
        count += 1
    return count


class Replay:
    def __init__(self, drafter, context_bytes, rule, depth):
        self.drafter, self.context_bytes, self.depth = drafter, context_bytes, depth
        self.contexts = OrderedDict()
        self.tokens = {}
        self.draft_seconds = []
        d = drafter
        self.columns = None
        self.rule = None
        if rule[1] > 0.0:
            seed = rule[0] - (1 << 64) if rule[0] >= (1 << 63) else rule[0]
            self.rule = (torch.tensor(seed, device=d.device), torch.tensor(np.float32(rule[1]), device=d.device),
                         torch.tensor(np.float32(rule[2]), device=d.device), torch.tensor(rule[3], device=d.device))

    def context_for(self, sequence, owner, first):
        context = self.contexts.get(sequence)
        if context is not None and first != 0 and getattr(context, "owner", None) != owner:
            self.contexts[("retired", sequence, getattr(context, "owner", None))] = self.contexts.pop(sequence)
            context = None
        if context is None or first == 0:
            context = LatentContext(self.drafter)
            context.owner = owner
            self.contexts[sequence] = context
        self.contexts.move_to_end(sequence)
        return context

    def committed(self, key, position, anchor):
        stream = self.tokens.setdefault(key, np.zeros(0, dtype=np.uint32))
        if stream.shape[0] <= position:
            stream = np.concatenate((stream, np.zeros(position + 1 - stream.shape[0], dtype=np.uint32)))
        stream[position] = anchor
        self.tokens[key] = stream
        return stream[:position + 1]

    def draft(self, sequence, position, anchor, committed):
        d = self.drafter
        context = self.contexts.get(sequence)
        if context is None or not context.covers(position):
            return "await", []
        if not context.complete(position):
            for other_sequence, other in reversed(self.contexts.items()):
                if other_sequence != sequence:
                    context.copy_prefix(other, committed, position)
                if context.complete(position):
                    break
        if not context.matches(committed, position):
            return "no-context", []
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad():
            rows, _ = context.contexts(position)
            y = d.forward_block(anchor, position, rows)
            if self.rule is None:
                tokens = d.propose(y, anchor)[0]
            else:
                tokens = d.propose(y, anchor, self.rule, torch.tensor(position, device=d.device), torch.tensor(True, device=d.device))[0]
            drafts = tokens.tolist()[:min(self.depth, d.block)]
        self.draft_seconds.append(time.monotonic() - started)
        return "drafts", drafts

    def write(self, hello, frame, data):
        key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
        owner = (frame["control_generation"], frame["request_id"])
        d = self.drafter
        if self.columns is None:
            self.columns = [hello["entries"].index(name) for name in d.tap_names] if hasattr(d, "tap_names") else \
                [hello["entries"].index(f"o{layer}") for layer in d.layer_ids]
        rows = torch.from_numpy(data.reshape(frame["rows"], len(hello["entries"]), hello["hidden"]).astype(np.int16)).view(torch.bfloat16)
        taps = rows.to(d.device)[:, self.columns].reshape(frame["rows"], -1)
        context = self.context_for(frame["sequence_id"], owner, frame["first_position"])
        context.write(frame["first_position"], np.asarray(frame["tokens"], dtype=np.uint32), taps)
        stream = self.tokens.get(key)
        end = frame["first_position"] + frame["rows"]
        if frame["first_position"] == 0 or stream is None:
            stream = np.zeros(0, dtype=np.uint32)
        if stream.shape[0] < end:
            stream = np.concatenate((stream, np.zeros(end - stream.shape[0], dtype=np.uint32)))
        stream[frame["first_position"]:end] = frame["tokens"]
        self.tokens[key] = stream
        evict_contexts(self.contexts, self.context_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--module-drafts", required=True)
    parser.add_argument("--rule", default="0,0,0,0")
    parser.add_argument("--context-bytes", type=int, default=8 << 30)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    parser.add_argument("--head", default="language_model.lm_head.weight")
    parser.add_argument("--norm", default="language_model.model.norm.weight")
    options = parser.parse_args()
    seed, inverse_temperature, nucleus, top_k = options.rule.split(",")
    rule = (int(seed), float(inverse_temperature), float(nucleus), int(top_k))
    drafter = open_drafter(options.checkpoint, options.target, {"embed": options.embed, "head": options.head, "norm": options.norm}, "cuda")
    replay = Replay(drafter, options.context_bytes, rule, options.depth)
    module = module_answers(options.module_drafts)
    streams = committed_streams(options.recording)
    python = []
    with open(options.recording, "rb") as handle:
        for hello, frame in frames(options.recording):
            if options.limit and len(python) >= options.limit:
                break
            key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
            if frame["first_position"] > 0 and frame["flags"] & ANSWERING_FLAGS:
                committed = replay.committed(key, frame["first_position"], frame["tokens"][0])
                answer, drafts = replay.draft(frame["sequence_id"], frame["first_position"], frame["tokens"][0], committed)
                python.append((key, frame["first_position"], answer, drafts))
            handle.seek(frame["offset"])
            data = np.frombuffer(handle.read(frame["rows"] * frame["row_bytes"]), dtype=np.uint16)
            replay.write(hello, frame, data)
    if len(module) != len(python):
        print(f"FAIL the module answered {len(module)} requests and the Python replay {len(python)}")
        return 1
    same_answer = sum(1 for a, b in zip(module, python) if a[:3] == b[:3])
    drafted = [(a, b) for a, b in zip(module, python) if a[2] == "drafts" and b[2] == "drafts"]
    same_drafts = sum(1 for a, b in drafted if a[3] == b[3])
    module_accepted = [accepted(a[3], streams[a[0]], a[1]) for a, _ in drafted]
    python_accepted = [accepted(b[3], streams[b[0]], b[1]) for _, b in drafted]
    answers = {}
    for item in module:
        answers[item[2]] = answers.get(item[2], 0) + 1
    print(f"requests {len(module)}: answers equal {same_answer} ({answers}); both drafted {len(drafted)}, drafts equal {same_drafts};"
          f" Python draft {1e3 * np.median(replay.draft_seconds) if replay.draft_seconds else 0.0:.2f} ms median (eager, without CUDA graphs)")
    if drafted:
        print(f"mean accepted length against the recording: module {np.mean(module_accepted):.4f} python {np.mean(python_accepted):.4f};"
              f" accepted length distribution module {np.bincount(module_accepted, minlength=drafter.block + 1).tolist()}"
              f" python {np.bincount(python_accepted, minlength=drafter.block + 1).tolist()}")
    for a, b in zip(module, python):
        if a[:3] != b[:3] or a[3] != b[3]:
            print(f"first difference: sequence {a[0][2]} position {a[1]}: module {a[2]} {a[3]} python {b[2]} {b[3]}")
            break
    return 0 if same_answer == len(module) else 1


if __name__ == "__main__":
    sys.exit(main())
