import argparse
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "spec_offline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from latent_block_drafter import LatentBlockDrafter
    from latent_block_train import CONTEXT_PROJECTOR, LatentBlockTrainer, select, sequences, train, write_checkpoint
    from block_drafter import SafeTensors
    from drafter_contract import drafter_tensors
    from tapstream import FRAME, HELLO, MAGIC, frames, read_hello
    from test_latent_block_drafter import BLOCK, HIDDEN, TARGET_LAYERS, VOCAB, tiny_checkpoint

NAMES = {"embed": "embed", "head": "head", "norm": "norm"}
FINGERPRINT = 0x5eed


def write_stream(path, entries, requests):
    blob = b"".join(struct.pack("<HH", 3 if name == "final" else 1, 0 if name == "final" else int(name[1:])) for name in entries).ljust(512, b"\0")
    with open(path, "wb") as handle:
        handle.write(HELLO.pack(MAGIC, 1, 1, HELLO.size, b"tiny", bytes(32), HIDDEN, 12, 1, len(entries), 0, 1, FINGERPRINT, blob))
        serial = 0
        for request, first, tokens, rows, flags in requests:
            for begin in range(0, len(tokens), 16):
                part = tokens[begin:begin + 16]
                payload = rows[begin:begin + 16].to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
                handle.write(FRAME.pack(MAGIC, 1, 2, FRAME.size, FINGERPRINT, 0, request, 0, request, first + begin, serial, len(part), flags, 0, len(entries),
                                        len(part) * (4 + len(entries) * HIDDEN * 2)))
                handle.write(struct.pack(f"<{len(part)}I", *part))
                handle.write(payload)
                serial += 1


def periodic(count, period=5):
    return [index % period for index in range(count)]


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class LatentBlockTrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = self.tmp.name
        tiny_checkpoint(self.directory)
        self.target = os.path.join(self.directory, "target.safetensors")
        generator = torch.Generator().manual_seed(11)
        self.count = 48
        self.taps = (torch.randn(self.count, HIDDEN * len(TARGET_LAYERS), generator=generator) * 0.5).to(torch.bfloat16)
        self.tokens = torch.randint(0, VOCAB - 1, (self.count,), generator=generator)

    def tearDown(self):
        self.tmp.cleanup()

    def drafter(self, directory=None):
        return LatentBlockDrafter(directory or self.directory, self.target, NAMES, "cpu")

    def test_training_forward_equals_the_serving_block_for_every_anchor_and_window(self):
        trainer = LatentBlockTrainer(self.directory, self.target, NAMES, "cpu")
        drafter = self.drafter()
        anchors = [1, 6, 14, 30, 47]
        trained = trainer.blocks(trainer.features(self.taps), self.tokens, anchors)
        rows, _ = drafter.context(self.taps, torch.arange(self.count))
        for index, anchor in enumerate(anchors):
            served = drafter.forward_block(int(self.tokens[anchor]), anchor, [layer[drafter.visible(i, anchor):anchor] for i, layer in enumerate(rows)])
            error = (trained[index].float() - served.float()).norm() / served.float().norm()
            self.assertLess(float(error), 1e-2, f"anchor {anchor}")

    def test_the_context_projector_stays_frozen_and_every_other_tensor_learns(self):
        trainer = LatentBlockTrainer(self.directory, self.target, NAMES, "cpu")
        anchors = [3, 9, 20, 40]
        loss, metrics = trainer.loss(trainer.blocks(trainer.features(self.taps), self.tokens, anchors), self.tokens, anchors, 4.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, tensor in trainer.params.items():
            if name in CONTEXT_PROJECTOR:
                self.assertIsNone(tensor.grad, name)
            else:
                self.assertIsNotNone(tensor.grad, name)
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0, name)
        self.assertEqual({name: list(tensor.shape) for name, tensor in trainer.params.items()}, drafter_tensors(trainer.config))
        self.assertEqual(set(metrics), {"loss", "head_top1", "teacher_forced_prefix"})

    def accepted(self, drafter, tokens, anchors):
        rows, _ = drafter.context(self.taps, torch.arange(self.count))
        total = 0
        for anchor in anchors:
            y = drafter.forward_block(int(tokens[anchor]), anchor, [layer[drafter.visible(i, anchor):anchor] for i, layer in enumerate(rows)])
            drafts, _ = drafter.sequential(y, None, int(tokens[anchor]), range(1, BLOCK))
            for offset, token in enumerate(drafts):
                if anchor + 1 + offset >= len(tokens) or token != int(tokens[anchor + 1 + offset]):
                    break
                total += 1
        return total / len(anchors)

    def test_training_on_a_stream_raises_acceptance_and_the_checkpoint_serves(self):
        tokens = torch.tensor(periodic(self.count))
        anchors = list(range(8, self.count - BLOCK, 3))
        before = self.accepted(self.drafter(), tokens, anchors)
        trainer = LatentBlockTrainer(self.directory, self.target, NAMES, "cpu")
        features = trainer.features(self.taps)
        optimizer = torch.optim.AdamW(trainer.trainable(), lr=1e-2, weight_decay=0.0)
        losses = []
        for step in range(60):
            chosen = [int(a) for a in torch.randperm(self.count - 2, generator=torch.Generator().manual_seed(step))[:12] + 1]
            loss, _ = trainer.loss(trainer.blocks(features, tokens, sorted(chosen)), tokens, sorted(chosen), 4.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(sum(losses[-5:]) / 5, 0.5 * sum(losses[:5]) / 5)
        out = os.path.join(self.directory, "trained")
        receipt = write_checkpoint(trainer, out, {"format": "test"})
        after = self.accepted(self.drafter(out), tokens, anchors)
        self.assertGreaterEqual(after, before + 1.0)
        self.assertGreaterEqual(after, 2.0)
        with open(os.path.join(out, "model.safetensors"), "rb") as handle:
            self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), receipt["model_sha256"])
        written = SafeTensors(os.path.join(out, "model.safetensors"))
        self.assertEqual(set(written.header), set(trainer.names))
        self.assertTrue(all(entry["dtype"] == "BF16" for entry in written.header.values()))
        for name in CONTEXT_PROJECTOR:
            self.assertTrue(torch.equal(written.load(name, "cpu"), SafeTensors(os.path.join(self.directory, "model.safetensors")).load(name, "cpu")))

    def test_select_keeps_only_the_named_entries_and_requests(self):
        entries = ["o2", "o3", "o7", "final"]
        rows = [torch.randn(20, len(entries), HIDDEN) for _ in range(2)]
        source = os.path.join(self.directory, "source.spt2")
        write_stream(source, entries, [(1, 0, list(range(20)), rows[0], 1), (2, 0, list(range(5, 25)), rows[1], 2)])
        out = io.BytesIO()
        self.assertEqual(select([source], ["o7", "o3"], {2}, out), 20)
        path = os.path.join(self.directory, "selected.spt2")
        with open(path, "wb") as handle:
            handle.write(out.getvalue())
        with open(path, "rb") as handle:
            hello = read_hello(handle)
        self.assertEqual(hello["entries"], ["o7", "o3"])
        self.assertEqual(hello["fingerprint"], FINGERPRINT)
        seen = []
        for _, frame in frames(path):
            self.assertEqual(frame["request_id"], 2)
            seen.extend(frame["tokens"])
            with open(path, "rb") as handle:
                handle.seek(frame["offset"])
                data = torch.frombuffer(bytearray(handle.read(frame["rows"] * 2 * HIDDEN * 2)), dtype=torch.bfloat16).view(frame["rows"], 2, HIDDEN)
            first = frame["first_position"]
            expected = rows[1][first:first + frame["rows"]][:, [2, 1]].to(torch.bfloat16)
            self.assertTrue(torch.equal(data, expected))
        self.assertEqual(seen, list(range(5, 25)))
        with self.assertRaises(SystemExit):
            select([source], ["o9"], set(), io.BytesIO())

    def test_sequences_borrow_a_prefix_only_from_an_earlier_request_that_holds_it(self):
        entries = ["o3", "o7"]
        rows = torch.randn(30, 2, HIDDEN)
        tokens = list(range(30))
        source = os.path.join(self.directory, "prefix.spt2")
        write_stream(source, entries, [(1, 0, tokens, rows, 1), (2, 12, tokens[12:], rows[12:], 1), (3, 25, [9] * 5, rows[25:], 1)])
        _, data = sequences([source], set())
        lengths = {request: len(records) for request, records in data}
        self.assertEqual(lengths, {1: 30, 2: 30})
        late = dict(data)[2]
        self.assertEqual([record[2] for record in late], tokens)

    def test_train_writes_a_checkpoint_and_a_receipt_naming_its_base_and_data(self):
        entries = ["o3", "o7", "final"]
        generator = torch.Generator().manual_seed(3)
        streams = [(request, 0, periodic(40), torch.randn(40, len(entries), HIDDEN, generator=generator), 1) for request in (7, 8, 9)]
        source = os.path.join(self.directory, "train.spt2")
        write_stream(source, entries, streams)
        out = os.path.join(self.directory, "run")
        args = argparse.Namespace(drafter=self.directory, target_shard=self.target, embed="embed", head="head", taps=[source], requests="7-8",
                                  steps=6, anchors=8, lr=1e-3, warmup=2, decay=4.0, clip=1.0, seed=1, log_every=100, out=out)
        train(args)
        with open(os.path.join(out, "TRAINING-RECEIPT.json")) as handle:
            receipt = json.load(handle)
        with open(os.path.join(self.directory, "model.safetensors"), "rb") as handle:
            base = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(receipt["base"]["model_sha256"], base)
        self.assertEqual(receipt["requests"], [7, 8])
        self.assertEqual(receipt["positions"], 80)
        self.assertEqual(receipt["tap_entries"], [f"o{layer}" for layer in TARGET_LAYERS])
        self.assertEqual(receipt["frozen"], list(CONTEXT_PROJECTOR))
        self.assertEqual((receipt["target_model"], receipt["pack_identity"]), ("tiny", "00" * 32))
        self.assertTrue(os.path.exists(os.path.join(out, "config.json")))
        self.drafter(out)

    def test_two_ranks_train_one_replica_on_their_own_shards(self):
        entries = ["o3", "o7"]
        generator = torch.Generator().manual_seed(4)
        source = os.path.join(self.directory, "ranks.spt2")
        write_stream(source, entries, [(request, 0, periodic(30 + request), torch.randn(30 + request, 2, HIDDEN, generator=generator), 1) for request in range(1, 5)])
        out = os.path.join(self.directory, "ranks")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                   os.path.join(ROOT, "tools", "spec_offline", "latent_block_train.py"), "train", "--drafter", self.directory, "--target-shard", self.target,
                   "--embed", "embed", "--head", "head", "--taps", source, "--steps", "4", "--anchors", "6", "--lr", "1e-3", "--warmup", "1", "--out", out]
        done = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-2000:])
        with open(os.path.join(out, "TRAINING-RECEIPT.json")) as handle:
            receipt = json.load(handle)
        self.assertEqual((receipt["world"], receipt["requests"], receipt["positions"]), (2, [1, 2, 3, 4], sum(30 + request for request in range(1, 5))))
        self.drafter(out)


if __name__ == "__main__":
    unittest.main()
