import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False

HIDDEN, HEADS, QLORA, LATENT, NOPE, VDIM = 128, 2, 128, 64, 64, 64
EXPERTS, TOPK, INTER, VOCAB = 4, 2, 128, 512
IHEADS, IDIM, ITOPK, KPOOL = 2, 32, 8, 4
LAYER = 1


def bf16_bits(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def write_checkpoint(root, seed=3):
    rng = np.random.default_rng(seed)
    tensors = {}

    def bf16(name, shape, scale=0.08, offset=0.0):
        tensors[name] = ("BF16", bf16_bits(offset + scale * rng.standard_normal(shape)))

    def fp8(name, shape):
        codes = rng.integers(0x20, 0x58, size=shape, dtype=np.uint8) | (rng.integers(0, 2, size=shape, dtype=np.uint8) << 7)
        tensors[name] = ("F8_E4M3", codes)
        tensors[name + "_scale_inv"] = ("F32", rng.uniform(0.002, 0.004, size=(shape[0] // 128, shape[1] // 128)).astype(np.float32))

    p = f"model.language_model.layers.{LAYER}."
    bf16("model.language_model.embed_tokens.weight", (VOCAB, HIDDEN), 1.0)
    bf16("lm_head.weight", (VOCAB, HIDDEN), 0.2)
    bf16("model.language_model.norm.weight", (HIDDEN,), 0.1, 1.0)
    for name in ("enorm", "hnorm", "input_layernorm", "post_attention_layernorm", "shared_head.norm"):
        bf16(p + name + ".weight", (HIDDEN,), 0.1, 1.0)
    bf16(p + "eh_proj.weight", (HIDDEN, 2 * HIDDEN))
    bf16(p + "self_attn.q_a_proj.weight", (QLORA, HIDDEN))
    bf16(p + "self_attn.q_a_layernorm.weight", (QLORA,), 0.1, 1.0)
    bf16(p + "self_attn.q_b_proj.weight", (HEADS * NOPE, QLORA))
    bf16(p + "self_attn.kv_a_proj_with_mqa.weight", (LATENT, HIDDEN))
    bf16(p + "self_attn.kv_a_layernorm.weight", (LATENT,), 0.1, 1.0)
    bf16(p + "self_attn.kv_b_proj.weight", (HEADS * (NOPE + VDIM), LATENT))
    bf16(p + "self_attn.o_proj.weight", (HIDDEN, HEADS * VDIM))
    i = p + "self_attn.indexer."
    bf16(i + "wq_b.weight", (IHEADS * IDIM, QLORA), 0.2)
    bf16(i + "wk.weight", (IDIM, HIDDEN), 0.2)
    bf16(i + "k_norm.weight", (IDIM,), 0.1, 1.0)
    bf16(i + "k_norm.bias", (IDIM,), 0.1)
    bf16(i + "weights_proj.weight", (IHEADS, HIDDEN), 0.2)
    bf16(i + "index_kpool_compress_gate.weight", (IDIM, HIDDEN), 0.2)
    bf16(i + "index_kpool_compress_ape", (KPOOL, IDIM), 0.5)
    bf16(p + "mlp.gate.weight", (EXPERTS, HIDDEN), 0.2)
    tensors[p + "mlp.gate.e_score_correction_bias"] = ("F32", (0.01 * rng.standard_normal(EXPERTS)).astype(np.float32))
    for e in range(EXPERTS):
        fp8(p + f"mlp.experts.{e}.gate_proj.weight", (INTER, HIDDEN))
        fp8(p + f"mlp.experts.{e}.up_proj.weight", (INTER, HIDDEN))
        fp8(p + f"mlp.experts.{e}.down_proj.weight", (HIDDEN, INTER))
    bf16(p + "mlp.shared_experts.gate_proj.weight", (INTER, HIDDEN))
    bf16(p + "mlp.shared_experts.up_proj.weight", (INTER, HIDDEN))
    bf16(p + "mlp.shared_experts.down_proj.weight", (HIDDEN, INTER))
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(os.path.join(root, "model.safetensors"), "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for raw in blobs:
            fh.write(raw)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": {name: "model.safetensors" for name in tensors}}, fh)
    config = {"num_hidden_layers": LAYER, "hidden_size": HIDDEN, "rms_norm_eps": 1e-5, "num_attention_heads": HEADS,
              "kv_lora_rank": LATENT, "qk_nope_head_dim": NOPE, "v_head_dim": VDIM, "num_experts_per_tok": TOPK,
              "n_routed_experts": EXPERTS, "norm_topk_prob": True, "routed_scaling_factor": 2.5, "swiglu_limit": 7.0,
              "index_n_heads": IHEADS, "index_head_dim": IDIM, "index_topk": ITOPK, "index_kpool": KPOOL,
              "index_kpool_compress": True, "index_kpool_always_select_tail": True}
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(config, fh)


@unittest.skipUnless(HAVE_CUDA, "needs a CUDA device")
class DraftdMtpSequenceKvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from draftd_glm53flash_mtp import Glm53FlashMtpDrafter
        cls.tmp = tempfile.TemporaryDirectory()
        write_checkpoint(cls.tmp.name)
        cls.drafter = Glm53FlashMtpDrafter(cls.tmp.name, max_chain=7)
        cls.hc_drafter = Glm53FlashMtpDrafter(cls.tmp.name, max_chain=7, tap="hc_mean")
        rng = np.random.default_rng(5)
        cls.taps = torch.as_tensor((bf16_bits(rng.standard_normal((64, HIDDEN))).astype(np.uint32) << 16).view(np.float32)).cuda()
        cls.tokens = torch.as_tensor(rng.integers(0, VOCAB, size=65), dtype=torch.int64).cuda()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def kv(self, drafter=None, capacity=80):
        from draftd_glm53flash_mtp import MtpSequenceKv
        return MtpSequenceKv(drafter or self.drafter, capacity)

    def test_gemm_rows_equals_gemv_bitwise(self):
        d = self.drafter
        x = self.taps[:9].contiguous()
        rows = torch.empty((9, d.index_wk.shape[0]), dtype=torch.float32, device="cuda")
        d.k.gemm_rows_bf16(d.index_wk, x, rows)
        for r in range(9):
            one = torch.empty(d.index_wk.shape[0], dtype=torch.float32, device="cuda")
            d.k.gemv_bf16(d.index_wk, x[r].contiguous(), one)
            self.assertTrue(torch.equal(one, rows[r]), f"row {r}")

    def test_final_norm_tap_matches_numpy(self):
        from t1_reference_common import bf16_round_f32, rmsnorm
        row = self.taps[3].cpu().numpy()
        want = bf16_round_f32(rmsnorm(row, self.drafter.final_norm.cpu().numpy(), 1e-5))
        got = self.drafter.tap_hidden(self.taps[3:4])[0].cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=0, atol=float(np.abs(want).max()) * 2 ** -7)
        self.assertTrue(torch.equal(self.hc_drafter.tap_hidden(self.taps[3:4])[0], self.taps[3]))

    def test_prefill_commit_equals_rowwise_commit(self):
        a, b = self.kv(), self.kv()
        self.drafter.commit(a, self.taps[:20], self.tokens[1:21], 0)
        for p in range(20):
            self.drafter.commit(b, self.taps[p:p + 1], self.tokens[p + 1:p + 2], p)
        self.assertEqual(a.length, 20)
        self.assertTrue(torch.equal(a.latent[:20], b.latent[:20]))
        self.assertTrue(torch.equal(a.index_key[:20], b.index_key[:20]))
        self.assertTrue(torch.equal(a.index_gate[:20], b.index_gate[:20]))
        self.assertTrue(torch.equal(self.drafter.draft(a, 3), self.drafter.draft(b, 3)))

    def test_commit_rejects_gaps_and_overflow(self):
        kv = self.kv(capacity=8)
        with self.assertRaises(ValueError):
            self.drafter.commit(kv, self.taps[:2], self.tokens[1:3], 1)
        with self.assertRaises(IndexError):
            self.drafter.commit(kv, self.taps[:9], self.tokens[1:10], 0)
        self.drafter.commit(kv, self.taps[:6], self.tokens[1:7], 0)
        with self.assertRaises(IndexError):
            self.drafter.draft(kv, 4)

    def test_sequence_draft_matches_teacher_forced_full_cache(self):
        d = self.drafter
        anchor = 6
        kv = self.kv()
        d.commit(kv, self.taps[:anchor], self.tokens[1:anchor + 1], 0)
        got = d.draft(kv, 1)
        cache = torch.zeros((16, d.latent), dtype=torch.float32, device="cuda")
        hidden = d.tap_hidden(self.taps[:anchor])
        for p in range(anchor - 1):
            d.step(hidden[p], self.tokens[p + 1:p + 2], cache, p)
        _, _, best = d.step(hidden[anchor - 1], self.tokens[anchor:anchor + 1], cache, anchor - 1)
        self.assertEqual(int(got[0]), int(best))
        np.testing.assert_allclose(kv.latent[:anchor].cpu().numpy(), cache[:anchor].cpu().numpy(), rtol=0, atol=1e-2)

    def test_speculative_rows_never_touch_committed_rows(self):
        d = self.drafter
        kv = self.kv()
        d.commit(kv, self.taps[:10], self.tokens[1:11], 0)
        before = kv.latent[:10].clone(), kv.index_key[:10].clone(), kv.index_gate[:10].clone()
        first = d.draft(kv, 5)
        self.assertEqual(kv.length, 10)
        self.assertTrue(torch.equal(kv.latent[:10], before[0]))
        self.assertTrue(torch.equal(kv.index_key[:10], before[1]))
        self.assertTrue(torch.equal(kv.index_gate[:10], before[2]))
        self.assertTrue(torch.equal(first, d.draft(kv, 5)), "a draft is a function of the committed rows only")
        fresh = self.kv()
        d.commit(fresh, self.taps[:12], self.tokens[1:13], 0)
        d.commit(kv, self.taps[10:12], self.tokens[11:13], 10)
        self.assertTrue(torch.equal(kv.latent[:12], fresh.latent[:12]))
        self.assertTrue(torch.equal(d.draft(kv, 4), d.draft(fresh, 4)))

    def test_truncate_rolls_back(self):
        d = self.drafter
        kv, ref = self.kv(), self.kv()
        d.commit(kv, self.taps[:14], self.tokens[1:15], 0)
        kv.truncate(9)
        d.commit(kv, self.taps[9:11], self.tokens[10:12], 9)
        d.commit(ref, self.taps[:11], self.tokens[1:12], 0)
        self.assertTrue(torch.equal(d.draft(kv, 3), d.draft(ref, 3)))
        with self.assertRaises(ValueError):
            kv.truncate(12)

    def test_index_selection_matches_numpy_reference(self):
        import glm53flash_mtp_reference as reference
        d = self.drafter
        for context in (ITOPK + 1, 23, 40, 57):
            kv = self.kv()
            d.commit(kv, self.taps[:context], self.tokens[1:context + 1], 0)
            x = d.tap_hidden(self.taps[context - 1:context])[0]
            q_norm = self.taps[context % 7][:QLORA].contiguous()
            got = d.index_select(q_norm, x, kv, context).cpu().numpy()
            query = d.linear_rows(d.index_wq, q_norm.reshape(1, -1)).view(IHEADS, IDIM).cpu().numpy()
            head = d.linear_rows(d.index_head_w, x.reshape(1, -1)).reshape(-1).cpu().numpy()
            want = reference.index_positions(query, head, kv.index_key[:context].cpu().numpy(), kv.index_gate[:context].cpu().numpy(),
                                             d.index_ape.cpu().numpy(), context, ITOPK, KPOOL)
            self.assertEqual(list(got), list(want), f"context {context}")
            self.assertEqual(len(got), ITOPK + context % KPOOL)

    def test_long_context_draft_uses_the_indexer(self):
        d = self.drafter
        kv = self.kv()
        d.commit(kv, self.taps[:40], self.tokens[1:41], 0)
        calls = []
        original = d.index_select

        def spy(*args):
            calls.append(args[-1])
            return original(*args)

        d.index_select = spy
        try:
            drafts = d.draft(kv, 3)
        finally:
            d.index_select = original
        self.assertEqual(calls, [40, 41, 42])
        self.assertEqual(drafts.numel(), 3)
        short = self.kv()
        d.commit(short, self.taps[:ITOPK - 1], self.tokens[1:ITOPK], 0)
        calls.clear()
        d.index_select = spy
        try:
            d.draft(short, 3)
        finally:
            d.index_select = original
        self.assertEqual(calls, [ITOPK + 1], "context <= index_topk attends densely; index_topk + 1 selects")


if __name__ == "__main__":
    unittest.main()
