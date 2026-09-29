import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draftd_mtp_g8 as g8
from draftd_glm53flash_mtp import Glm53FlashMtpDrafter

E4M3_MAX = 448.0
SLACK = 1e-3


class CertifiedFp8Head:
    def __init__(self, drafter):
        self.d = drafter
        head = drafter.lm_head
        rows, cols = head.shape
        self.rows, self.cols = rows, cols
        self.codes = torch.empty((1, rows, cols), dtype=torch.uint8, device=drafter.device)
        self.scale = torch.empty((1, rows // 128, cols // 128), dtype=torch.float32, device=drafter.device)
        self.block_error = torch.empty((rows, cols // 128), dtype=torch.float32, device=drafter.device)
        self.row_l2 = torch.empty(rows, dtype=torch.float32, device=drafter.device)
        chunk = 8192
        for first in range(0, rows, chunk):
            last = min(rows, first + chunk)
            blocks = head[first:last].view((last - first) // 128, 128, cols // 128, 128).to(torch.float32)
            amax = blocks.abs().amax(dim=(1, 3))
            scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
            codes = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
            delta = codes.to(torch.float32) * scale[:, None, :, None] - blocks
            self.row_l2[first:last] = delta.pow(2).sum(dim=(2, 3)).reshape(last - first).sqrt() * (1.0 + SLACK)
            error = delta.abs().sum(dim=3)
            self.codes[0, first:last] = codes.view(torch.uint8).reshape(last - first, cols)
            self.scale[0, first // 128:last // 128] = scale
            self.block_error[first:last] = error.reshape(last - first, cols // 128) * (1.0 + SLACK)
        self.ids = torch.zeros(1, dtype=torch.int64, device=drafter.device)
        self.screen = torch.empty((1, rows), dtype=torch.float32, device=drafter.device)

    def device_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.codes, self.scale, self.block_error))

    def argmax(self, head_in):
        self.d.k.gemv_fp8_block(self.codes, self.scale, self.ids, head_in, self.screen)
        screen = self.screen[0]
        xmax = head_in.abs().view(self.cols // 128, 128).amax(dim=1)
        bound = torch.minimum(self.block_error @ xmax, self.row_l2 * torch.linalg.vector_norm(head_in))
        floor = torch.max(screen - bound)
        candidates = torch.nonzero(screen + bound >= floor).flatten()
        rows = self.d.lm_head[candidates].contiguous()
        exact = torch.empty(rows.shape[0], dtype=torch.float32, device=self.d.device)
        self.d.k.gemv_bf16(rows, head_in, exact)
        best = candidates[torch.argmax(exact)]
        return int(torch.argmax(screen)), int(best), int(candidates.numel())


def time_ms(fn, repeats=100):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main(argv=None):
    parser = argparse.ArgumentParser(description="draftd MTP head options: bf16, fp8 screen, certified fp8 screen")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixtures", nargs="+", required=True)
    parser.add_argument("--streams", required=True)
    parser.add_argument("--impl-chains", type=int, default=150)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    drafter = Glm53FlashMtpDrafter(args.checkpoint)
    head = CertifiedFp8Head(drafter)
    config = drafter.config
    real, pool = g8.fixture_cases(args.fixtures, drafter.layer - 1, int(config["hc_mult"]), drafter.hidden, args.depth)
    cases = real + g8.stream_cases(pool, g8.load_streams(args.streams), args.impl_chains, args.depth, args.seed)
    positions = fp8_agree = certified_agree = 0
    candidate_counts = []
    for case in cases:
        cache = torch.zeros((drafter.max_chain, drafter.latent), dtype=torch.float32, device=drafter.device)
        hidden = torch.as_tensor(case["hidden"], device=drafter.device)
        for step, token in enumerate(case["fed"]):
            hidden, head_in, best = drafter.step(hidden, torch.as_tensor([int(token)], device=drafter.device), cache, step)
            fp8_top, certified_top, count = head.argmax(head_in)
            positions += 1
            fp8_agree += int(fp8_top == int(best))
            certified_agree += int(certified_top == int(best))
            candidate_counts.append(count)
    head_in = torch.as_tensor(real[0]["hidden"], device=drafter.device)
    bf16_ms = time_ms(lambda: drafter.k.gemv_bf16(drafter.lm_head, head_in, drafter.logits))
    screen_ms = time_ms(lambda: drafter.k.gemv_fp8_block(head.codes, head.scale, head.ids, head_in, head.screen))
    certified_ms = time_ms(lambda: head.argmax(head_in), repeats=20)
    counts = np.asarray(candidate_counts)
    report = {"positions": positions, "fp8_screen_top1_agreement": fp8_agree / positions,
              "certified_top1_agreement": certified_agree / positions,
              "candidates_p50": float(np.percentile(counts, 50)), "candidates_p99": float(np.percentile(counts, 99)),
              "candidates_max": int(counts.max()), "bf16_head_bytes": drafter.lm_head.numel() * 2,
              "fp8_head_bytes": head.device_bytes(), "bf16_head_ms": bf16_ms, "fp8_screen_ms": screen_ms,
              "certified_eager_ms": certified_ms}
    json.dump(report, open(args.output, "w"), indent=1)
    print(f"DRAFTD-HEAD positions={positions} fp8_top1_agreement={report['fp8_screen_top1_agreement']:.4f} "
          f"certified_top1_agreement={report['certified_top1_agreement']:.4f} candidates p50={report['candidates_p50']:.0f} "
          f"p99={report['candidates_p99']:.0f} max={report['candidates_max']} bf16_head={bf16_ms:.3f}ms "
          f"fp8_screen={screen_ms:.3f}ms certified_eager={certified_ms:.3f}ms fp8_bytes={report['fp8_head_bytes']}")
    return 0 if report["certified_top1_agreement"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
