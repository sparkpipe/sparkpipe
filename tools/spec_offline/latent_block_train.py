#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import SafeTensors, load_rows, prefix_donor, request_set, requests, rms  # noqa: E402
from latent_block_drafter import refuse_unsupported, rope_tables, rotate_pairs  # noqa: E402
from tapstream import FORMS, FRAME, HELLO, frames  # noqa: E402

CONTEXT_PROJECTOR = ("fc.weight", "hidden_norm.weight")
UNUSED = ("embed_tokens.weight",)
RECEIPT_FORMAT = "sparkpipe-drafter-training-v1"
FEATURE_CHUNK = 4096


class LatentBlockTrainer:
    def __init__(self, directory, shard, names, device):
        self.device = device
        self.directory = Path(directory)
        with open(self.directory / "config.json") as handle:
            self.config = c = json.load(handle)
        refuse_unsupported(c)
        dflash = c["dflash_config"]
        weights = SafeTensors(self.directory / "model.safetensors")
        self.names = [name for name in weights.header if name != "__metadata__" and name not in UNUSED]
        self.params = {name: weights.load(name, device).float().requires_grad_(name not in CONTEXT_PROJECTOR) for name in self.names}
        target = SafeTensors(shard)
        self.embed = target.load(names["embed"], device)
        self.head = target.load(names["head"], device)
        self.eps = c["rms_norm_eps"]
        self.heads, self.nope, self.rope_dim, self.value_dim = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        self.kv_rank = c["kv_lora_rank"]
        self.block = dflash["block_size"]
        self.layer_count = c["num_hidden_layers"]
        self.layer_ids = list(dflash["target_layer_ids"])
        self.mask = dflash["mask_token_id"]
        self.group = dflash["conv_group_size"]
        self.taps = dflash["conv_kernel_size"]
        self.top_k = dflash["selector_top_k"]
        self.windows = [c["sliding_window"] if kind == "sliding_attention" else None for kind in c["layer_types"]]
        self.inverse, self.rotary_gain, softmax_gain = rope_tables(c, self.rope_dim, device)
        self.scale = softmax_gain / math.sqrt(self.nope + self.rope_dim)

    def trainable(self):
        return [tensor for tensor in self.params.values() if tensor.requires_grad]

    def weight(self, name):
        return self.params[name]

    def layer(self, index, name):
        return self.params[f"layers.{index}.{name}"]

    @staticmethod
    def project(x, weight):
        return x @ weight.to(torch.bfloat16).t()

    def features(self, taps):
        with torch.no_grad():
            parts = [rms(self.project(taps[begin:begin + FEATURE_CHUNK].to(self.device, torch.bfloat16), self.weight("fc.weight")).float(),
                         self.weight("hidden_norm.weight"), self.eps).to(torch.bfloat16) for begin in range(0, taps.shape[0], FEATURE_CHUNK)]
        return torch.cat(parts, 0)

    def latent_rows(self, index, x, positions):
        latent = self.project(x, self.layer(index, "self_attn.kv_a_proj_with_mqa.weight")).float()
        compressed = rms(latent[..., :self.kv_rank], self.layer(index, "self_attn.kv_a_layernorm.weight"), self.eps)
        rotated = rotate_pairs(latent[..., self.kv_rank:], positions, self.inverse, self.rotary_gain)
        return torch.cat((compressed, rotated), -1).to(torch.bfloat16)

    def conv(self, z, base, delta):
        coefficients = base.float()[None, None] + delta.repeat_interleave(self.group, -1)
        out = coefficients[:, :, 0] * z
        for tap in range(1, base.shape[0]):
            out = out + coefficients[:, :, tap] * torch.cat((torch.zeros_like(z[:, :tap]), z[:, :-tap]), 1)
        return out

    def conv_pair(self, index, name, x):
        blocks, rows = x.shape[:2]
        coefficients = self.project(x, self.layer(index, name + ".kernel_projection.weight")).float().view(blocks, rows, 2, self.taps, -1)
        return self.conv(x.float(), self.layer(index, name + ".base_kernel")[0], coefficients[:, :, 0]).to(torch.bfloat16), coefficients[:, :, 1]

    def mlp(self, index, x):
        gate = self.project(x, self.layer(index, "mlp.gate_proj.weight")).float()
        up = self.project(x, self.layer(index, "mlp.up_proj.weight")).float()
        return self.project((torch.nn.functional.silu(gate) * up).to(torch.bfloat16), self.layer(index, "mlp.down_proj.weight")).float()

    def attend(self, index, absorbed, query_rope, own, context, anchor):
        span = context.shape[0]
        where = torch.arange(span, device=self.device)
        visible = where[None, :] < anchor[:, None]
        if self.windows[index] is not None:
            visible = visible & (where[None, :] >= anchor[:, None] - (self.windows[index] - self.block))
        rows = context.float()
        scores = (torch.einsum("bshr,pr->bshp", absorbed, rows[:, :self.kv_rank]) + torch.einsum("bshd,pd->bshp", query_rope, rows[:, self.kv_rank:])) * self.scale
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        own = own.float()
        local = (torch.einsum("bshr,btr->bsht", absorbed, own[..., :self.kv_rank]) + torch.einsum("bshd,btd->bsht", query_rope, own[..., self.kv_rank:])) * self.scale
        weights = torch.softmax(torch.cat((scores, local), -1), -1)
        return torch.einsum("bshp,pr->bshr", weights[..., :span], rows[:, :self.kv_rank]) + torch.einsum("bsht,btr->bshr", weights[..., span:], own[..., :self.kv_rank])

    def layer_forward(self, index, residual, positions, context, anchor):
        blocks, rows = residual.shape[:2]
        x = rms(residual, self.layer(index, "input_layernorm.weight"), self.eps).to(torch.bfloat16)
        prepared, finish = self.conv_pair(index, "attention_conv", x)
        flat = prepared.reshape(blocks * rows, -1)
        query = self.project(rms(self.project(flat, self.layer(index, "self_attn.q_a_proj.weight")).float(), self.layer(index, "self_attn.q_a_layernorm.weight"), self.eps).to(torch.bfloat16),
                             self.layer(index, "self_attn.q_b_proj.weight")).view(blocks * rows, self.heads, self.nope + self.rope_dim)
        kv_b = self.layer(index, "self_attn.kv_b_proj.weight").view(self.heads, self.nope + self.value_dim, self.kv_rank)
        absorbed = torch.einsum("ihn,hnr->ihr", query[..., :self.nope].float(), kv_b[:, :self.nope].float()).view(blocks, rows, self.heads, self.kv_rank)
        query_rope = rotate_pairs(query[..., self.nope:], positions, self.inverse, self.rotary_gain).view(blocks, rows, self.heads, self.rope_dim)
        own = self.latent_rows(index, flat, positions).view(blocks, rows, -1)
        latent = self.attend(index, absorbed, query_rope, own, context, anchor)
        values = torch.einsum("bshr,hvr->bshv", latent, kv_b[:, self.nope:].float()).reshape(blocks, rows, self.heads * self.value_dim)
        attended = self.project(values.to(torch.bfloat16), self.layer(index, "self_attn.o_proj.weight")).float()
        residual = residual + self.conv(attended, self.layer(index, "attention_conv.base_kernel")[1], finish)
        x = rms(residual, self.layer(index, "post_attention_layernorm.weight"), self.eps).to(torch.bfloat16)
        prepared, finish = self.conv_pair(index, "mlp_conv", x)
        return residual + self.conv(self.mlp(index, prepared), self.layer(index, "mlp_conv.base_kernel")[1], finish)

    def blocks(self, features, tokens, anchors):
        anchor = torch.as_tensor(anchors, device=self.device)
        offsets = torch.arange(self.block, device=self.device)
        positions = (anchor[:, None] + offsets).reshape(-1)
        block_tokens = torch.full((len(anchors), self.block), self.mask, dtype=torch.long, device=self.device)
        block_tokens[:, 0] = tokens[anchor]
        residual = self.embed[block_tokens].float()
        span = int(anchor.max())
        context_positions = torch.arange(span, device=self.device)
        for index in range(self.layer_count):
            residual = self.layer_forward(index, residual, positions, self.latent_rows(index, features[:span], context_positions), anchor)
        return rms(residual, self.weight("norm.weight"), self.eps).to(torch.bfloat16)

    def loss(self, y, tokens, anchors, decay):
        anchor = torch.as_tensor(anchors, device=self.device)
        steps = torch.arange(1, self.block, device=self.device)
        where = anchor[:, None] + steps
        valid = where < tokens.shape[0]
        labels = tokens[where.clamp(max=tokens.shape[0] - 1)]
        previous = tokens[(where - 1).clamp(max=tokens.shape[0] - 1)]
        rows = y[:, 1:]
        logits = (rows @ self.head.t()).float()
        full = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view(labels.shape)
        unary, candidates = torch.topk(logits, self.top_k, -1)
        total = torch.logsumexp(logits, -1)
        hidden = self.project(rows, self.weight("candidate_selector.hidden_projection.weight")).float()
        edges = unary + torch.einsum("bsr,bsr,bscr->bsc", self.weight("candidate_selector.predecessor_codebook")[previous].float(), hidden,
                                     self.weight("candidate_selector.successor_codebook")[candidates].float())
        tail = total + torch.log1p(-torch.exp(torch.logsumexp(unary, -1) - total).clamp(max=1.0 - 1e-7))
        normalizer = torch.logaddexp(torch.logsumexp(edges, -1), tail)
        hit = candidates == labels[..., None]
        found = hit.any(-1)
        truth = hit.float().argmax(-1)
        chosen = edges.gather(-1, truth[..., None]).squeeze(-1)
        lattice = torch.where(found, normalizer - chosen, torch.zeros_like(chosen))
        weight = torch.exp(-(steps - 1).float() / decay)[None, :] * valid
        loss = ((full + lattice) * weight).sum() / weight.sum()
        with torch.no_grad():
            top1 = ((logits.argmax(-1) == labels) & valid).sum() / valid.sum()
            walk = (edges.argmax(-1) == truth) & found & valid
            prefix = torch.cumprod(walk.to(torch.int32), -1).sum(-1).float().mean()
        return loss, {"loss": float(loss.detach()), "head_top1": float(top1), "teacher_forced_prefix": float(prefix)}


def sequences(paths, chosen):
    hello, streams = requests(paths, None)
    out = []
    for key, item in sorted(streams.items()):
        if chosen and key[1] not in chosen:
            continue
        end = max(item["positions"]) + 1
        start = end - 1
        while start - 1 in item["positions"]:
            start -= 1
        records = [item["positions"][p] for p in range(start, end)]
        if start > 0:
            donor = prefix_donor(key, item, start, streams)
            if donor is None:
                print(f"request {key[1]} starts at {start} and no earlier request lends its prefix: skipped", flush=True)
                continue
            records = [donor["positions"][p] for p in range(start)] + records
        out.append((key[1], records))
    return hello, out


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint(trainer, directory, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    header, offset = {}, 0
    for name in trainer.names:
        tensor = trainer.params[name]
        header[name] = {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [offset, offset + tensor.numel() * 2]}
        offset += tensor.numel() * 2
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = directory / "model.safetensors.partial"
    digest = hashlib.sha256()
    with open(partial, "wb") as handle:
        for chunk in (struct.pack("<Q", len(encoded)), encoded):
            handle.write(chunk)
            digest.update(chunk)
        for name in trainer.names:
            data = trainer.params[name].detach().to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tobytes()
            handle.write(data)
            digest.update(data)
    partial.rename(directory / "model.safetensors")
    shutil.copyfile(trainer.directory / "config.json", directory / "config.json")
    receipt = dict(receipt, model_sha256=digest.hexdigest(), tensors=len(trainer.names))
    (directory / "TRAINING-RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def ranks():
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1 and not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return world, rank


def average_gradients(tensors, world):
    for tensor in tensors:
        torch.distributed.all_reduce(tensor.grad)
        tensor.grad.div_(world)


def check_replicas(tensors, world):
    total = torch.stack([tensor.detach().double().sum() for tensor in tensors]).sum().view(1)
    gathered = [torch.zeros_like(total) for _ in range(world)]
    torch.distributed.all_gather(gathered, total)
    if any(not torch.equal(other, total) for other in gathered):
        raise SystemExit(f"replicas diverged: weight sums {[float(value) for value in gathered]}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    world, rank = ranks()
    trainer = LatentBlockTrainer(args.drafter, args.target_shard, {"embed": args.embed, "head": args.head}, device)
    hello, data = sequences(args.taps, request_set(args.requests))
    if len(data) < world:
        raise SystemExit(f"{len(data)} requests cannot feed {world} ranks")
    data = data[rank::world]
    wanted = [f"o{layer}" for layer in trainer.layer_ids]
    missing = [name for name in wanted if name not in hello["entries"]]
    if missing:
        raise SystemExit(f"tap stream lacks {missing}; it has {hello['entries']}")
    columns = [hello["entries"].index(name) for name in wanted]
    prepared = []
    for request, records in data:
        if len(records) < trainer.block + 2:
            continue
        tokens = torch.tensor([record[2] for record in records], dtype=torch.long, device=device)
        prepared.append((request, tokens, trainer.features(load_rows(hello, records, columns).reshape(len(records), -1))))
    if not prepared:
        raise SystemExit(f"rank {rank} has no request to train on")
    positions = sum(len(tokens) for _, tokens, _ in prepared)
    print(f"TRAIN-DATA rank {rank}/{world} requests {len(prepared)} positions {positions} trainable {sum(t.numel() for t in trainer.trainable())}", flush=True)
    requests_trained = [request for request, _, _ in prepared]
    if world > 1:
        counts = [None] * world
        torch.distributed.all_gather_object(counts, (positions, requests_trained))
        positions, requests_trained = sum(count for count, _ in counts), sorted(request for _, owned in counts for request in owned)
    optimizer = torch.optim.AdamW(trainer.trainable(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(args.seed * 1000003 + rank)
    weights = [len(tokens) for _, tokens, _ in prepared]
    history = []
    started = time.monotonic()
    rows = 0
    for step in range(args.steps):
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * warm * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        _, tokens, features = rng.choices(prepared, weights=weights)[0]
        anchors = sorted(rng.sample(range(1, len(tokens) - 1), min(args.anchors, len(tokens) - 2)))
        loss, metrics = trainer.loss(trainer.blocks(features, tokens, anchors), tokens, anchors, args.decay)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            average_gradients(trainer.trainable(), world)
        norm = float(torch.nn.utils.clip_grad_norm_(trainer.trainable(), args.clip))
        optimizer.step()
        rows += len(anchors) * trainer.block
        metrics.update(step=step, lr=optimizer.param_groups[0]["lr"], grad_norm=norm)
        history.append(metrics)
        if rank == 0 and (step % args.log_every == 0 or step == args.steps - 1):
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
            print(f"TRAIN-STEP {step} loss {metrics['loss']:.4f} head_top1 {metrics['head_top1']:.3f} prefix {metrics['teacher_forced_prefix']:.2f} "
                  f"grad {norm:.2f} lr {metrics['lr']:.2e} block_rows_per_s {rows / elapsed:.0f} peak_gib {peak:.1f}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if world > 1:
        check_replicas(trainer.trainable(), world)
        if rank != 0:
            torch.distributed.barrier()
            return
    receipt = {"format": RECEIPT_FORMAT, "base": {"directory": str(trainer.directory), "model_sha256": sha256_file(trainer.directory / "model.safetensors")},
               "target_shard": Path(args.target_shard).name, "taps": [Path(path).name for path in args.taps], "tap_entries": wanted,
               "target_model": hello["model"], "pack_identity": hello["pack_identity"], "world": world,
               "requests": requests_trained, "positions": positions, "frozen": list(CONTEXT_PROJECTOR),
               "steps": args.steps, "anchors": args.anchors, "lr": args.lr, "warmup": args.warmup, "decay": args.decay, "clip": args.clip, "seed": args.seed,
               "seconds": round(seconds, 1), "block_rows_per_second": round(rows / seconds, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None,
               "first": history[0], "last": history[-1]}
    receipt = write_checkpoint(trainer, args.out, receipt)
    print("TRAIN-DONE " + json.dumps({k: receipt[k] for k in ("model_sha256", "seconds", "block_rows_per_second", "peak_gib", "first", "last")}), flush=True)
    if world > 1:
        torch.distributed.barrier()


def entry_names(hello_fields):
    count, blob = hello_fields[9], hello_fields[13]
    entries = [struct.unpack_from("<HH", blob, 4 * index) for index in range(count)]
    return entries, [FORMS[form] + (str(layer) if form == 1 else "") for form, layer in entries]


def select(paths, names, chosen, out):
    with open(paths[0], "rb") as handle:
        fields = list(HELLO.unpack(handle.read(HELLO.size)))
    entries, available = entry_names(fields)
    missing = [name for name in names if name not in available]
    if missing:
        raise SystemExit(f"tap stream lacks {missing}; it has {available}")
    columns = [available.index(name) for name in names]
    hidden, count = fields[6], fields[9]
    fields[9] = len(columns)
    fields[13] = b"".join(struct.pack("<HH", *entries[column]) for column in columns).ljust(len(fields[13]), b"\0")
    out.write(HELLO.pack(*fields))
    fingerprint = fields[12]
    written = 0
    for path in paths:
        with open(path, "rb") as handle:
            other = HELLO.unpack(handle.read(HELLO.size))
        if entry_names(other)[1] != available or other[12] != fingerprint:
            raise SystemExit(f"{path} carries another tap set than {paths[0]}")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        for _, frame in frames(path):
            if chosen and frame["request_id"] not in chosen:
                continue
            rows = frame["rows"]
            payload = data[frame["offset"] // 2: frame["offset"] // 2 + rows * count * hidden].reshape(rows, count, hidden)[:, columns]
            out.write(FRAME.pack(fields[0], 1, 2, FRAME.size, fingerprint, frame["control_generation"], frame["request_id"], frame["request_generation"],
                                 frame["sequence_id"], frame["first_position"], frame["serial"], rows, frame["flags"], frame["tail_token"], len(columns),
                                 rows * (4 + len(columns) * hidden * 2)))
            out.write(struct.pack(f"<{rows}I", *frame["tokens"]))
            out.write(np.ascontiguousarray(payload).tobytes())
            written += rows
    return written


def main():
    parser = argparse.ArgumentParser(description="fine-tune a latent block drafter on recorded target taps")
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train", help="fine-tune from a checkpoint and write the new checkpoint with its receipt")
    trainer.add_argument("--drafter", required=True)
    trainer.add_argument("--target-shard", required=True)
    trainer.add_argument("--embed", default="language_model.model.embed_tokens.weight")
    trainer.add_argument("--head", default="language_model.lm_head.weight")
    trainer.add_argument("--taps", nargs="+", required=True)
    trainer.add_argument("--requests", default="", help="request ids to train on, as a comma list of ids and ranges; all when empty")
    trainer.add_argument("--steps", type=int, default=300)
    trainer.add_argument("--anchors", type=int, default=48, help="blocks per step, drawn from one request")
    trainer.add_argument("--lr", type=float, default=2e-5)
    trainer.add_argument("--warmup", type=int, default=20)
    trainer.add_argument("--decay", type=float, default=4.0, help="block position k weighs exp(-(k-1)/decay)")
    trainer.add_argument("--clip", type=float, default=1.0)
    trainer.add_argument("--seed", type=int, default=0)
    trainer.add_argument("--log-every", type=int, default=10)
    trainer.add_argument("--out", required=True)
    chooser = commands.add_parser("select", help="write a tap stream holding only some entries and requests")
    chooser.add_argument("--taps", nargs="+", required=True)
    chooser.add_argument("--entries", required=True, help="comma list, for example o19,o37,final")
    chooser.add_argument("--requests", default="")
    chooser.add_argument("--out", required=True, help="a path, or - for standard output")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
        return
    names = [name for name in args.entries.split(",") if name]
    if args.out == "-":
        rows = select(args.taps, names, request_set(args.requests), sys.stdout.buffer)
    else:
        with open(args.out, "wb") as handle:
            rows = select(args.taps, names, request_set(args.requests), handle)
    print(f"SELECT-DONE rows {rows}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
