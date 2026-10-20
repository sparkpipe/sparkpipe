import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "spec_offline"))

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False


@unittest.skipUnless(HAVE_CUDA, "needs a CUDA device")
class KeyedDraftCaptureTest(unittest.TestCase):
    def capture(self, function, *inputs):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                function(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function(*inputs)
        return graph, output

    def test_coupled_choice_replays_in_a_cuda_graph_like_eager(self):
        from block_drafter import coupled_choice
        generator = torch.Generator(device="cuda").manual_seed(5)
        logits = torch.randn(1000, device="cuda", generator=generator) * 3
        seed, position = torch.zeros((), dtype=torch.long, device="cuda"), torch.zeros((), dtype=torch.long, device="cuda")
        rates = torch.tensor([1.0, 0.95], device="cuda")
        top_k = torch.zeros((), dtype=torch.long, device="cuda")

        def choose(logits, seed, position, rates, top_k):
            return coupled_choice(logits, (seed, rates[0], rates[1], top_k), position)
        graph, output = self.capture(choose, logits, seed, position, rates, top_k)
        for trial in range(16):
            logits.copy_(torch.randn(1000, device="cuda", generator=generator) * 3)
            seed.fill_(4242 + 7 * trial)
            position.fill_(1000 + trial)
            rates.copy_(torch.tensor([0.5 + 0.1 * (trial % 6), 0.8 + 0.01 * trial], device="cuda"))
            top_k.fill_(trial % 3 * 20)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(output), int(choose(logits, seed, position, rates, top_k)))

    def test_lattice_choice_replays_in_a_cuda_graph_like_eager(self):
        from latent_block_drafter import coupled_candidates
        generator = torch.Generator(device="cuda").manual_seed(9)
        row = torch.randn(16, device="cuda", generator=generator)
        tokens = torch.randperm(163840, device="cuda", generator=generator)[:16]
        seed, position = torch.zeros((), dtype=torch.long, device="cuda"), torch.zeros((), dtype=torch.long, device="cuda")
        rates = torch.tensor([1.0, 0.95], device="cuda")
        top_k = torch.zeros((), dtype=torch.long, device="cuda")

        def choose(row, tokens, seed, position, rates, top_k):
            return coupled_candidates(row, tokens, (seed, rates[0], rates[1], top_k), position)
        graph, output = self.capture(choose, row, tokens, seed, position, rates, top_k)
        for trial in range(16):
            row.copy_(torch.randn(16, device="cuda", generator=generator) * 2)
            seed.fill_(77 + trial)
            position.fill_(5000 + trial)
            top_k.fill_(trial % 4 * 3)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(output), int(choose(row, tokens, seed, position, rates, top_k)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
