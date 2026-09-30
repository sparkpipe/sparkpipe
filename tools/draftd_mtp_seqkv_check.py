import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import bf16_to_f32

FLOOR = 0.99


def reference_rows(checkpoint, header, sources):
    import glm53flash_mtp_reference as reference
    engine, config = reference.load_engine(checkpoint, header)
    mtp = reference.MtpReference(engine, int(config["num_hidden_layers"]))
    out = {}
    for name, tokens, streams in sources:
        cache, columns = [], []
        for position in range(len(tokens) - 2):
            _, head_in = mtp.step(mtp.hidden_tap(streams[position], "final_norm"), tokens[position + 1], "embed_hidden", cache)
            columns.append(head_in)
        predicted, logprobs = reference.head_argmax(engine, columns, tokens[2:])
        out[name] = {"top1": [int(t) for t in predicted], "logprob": [float(v) for v in logprobs],
                     "latent": bf16_to_f32(np.stack(cache)) if cache else np.zeros((0, 0), dtype=np.float32)}
    return out


def device_rows(drafter, sources):
    import torch
    from draftd_glm53flash_mtp import MtpSequenceKv
    out = {}
    for name, tokens, streams in sources:
        hc = np.stack([s.reshape(drafter.hc_streams, drafter.hidden).mean(axis=0) for s in streams]).astype(np.float32)
        hc = torch.as_tensor(hc, device=drafter.device).to(torch.bfloat16).to(torch.float32)
        ids = torch.as_tensor(tokens, dtype=torch.int64, device=drafter.device)
        kv = MtpSequenceKv(drafter, len(tokens) + 8)
        top1, margins = [], []
        for position in range(len(tokens) - 2):
            drafter.commit(kv, hc[position:position + 1], ids[position + 1:position + 2], position)
            top1.append(int(drafter.draft(kv, 1)[0]))
            best = torch.topk(drafter.logits, 2).values
            margins.append(float(best[0] - best[1]))
        out[name] = {"top1": top1, "margin": margins, "latent": kv.latent[:kv.length].cpu().numpy()}
    return out


def main():
    parser = argparse.ArgumentParser(description="draftd per-sequence MTP KV (post-final-norm tap) vs the numpy MTP reference in sequence context, depth 1, identical inputs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--hc-streams", type=int, required=True, help="hyper-connection stream count of the fixtures")
    parser.add_argument("--output", required=True)
    parser.add_argument("fixtures", nargs="+")
    args = parser.parse_args()
    import glm53flash_mtp_reference as reference
    from draftd_glm53flash_mtp import Glm53FlashMtpDrafter
    config = json.load(open(os.path.join(args.checkpoint, "config.json")))
    config = config.get("text_config", config)
    layer = int(config["num_hidden_layers"])
    sources = []
    for path in args.fixtures:
        tokens, streams, _ = reference.fixture_rows(path, layer - 1)
        sources.append((os.path.basename(path), tokens, streams))
    started = time.time()
    drafter = Glm53FlashMtpDrafter(args.checkpoint, tap="final_norm")
    drafter.hc_streams = args.hc_streams
    first = device_rows(drafter, sources)
    second = device_rows(drafter, sources)
    device_s = time.time() - started
    started = time.time()
    ref = reference_rows(args.checkpoint, args.header, sources)
    reference_s = time.time() - started
    report = {"positions": 0, "agree": 0, "hits_reference": 0, "hits_device": 0, "fixtures": {},
              "run_to_run_identical": all(first[n]["top1"] == second[n]["top1"] for n in first),
              "device_s": round(device_s, 1), "reference_s": round(reference_s, 1), "tap": "final_norm", "context": "sequence"}
    for name, tokens, _ in sources:
        r, d = ref[name], first[name]
        want = tokens[2:]
        agree = sum(int(a == b) for a, b in zip(r["top1"], d["top1"]))
        diff = float(np.abs(r["latent"] - d["latent"]).max()) if r["latent"].size else 0.0
        scale = float(np.abs(r["latent"]).max()) if r["latent"].size else 0.0
        report["fixtures"][name] = {"positions": len(want), "agree": agree,
                                    "hits_reference": sum(int(a == b) for a, b in zip(r["top1"], want)),
                                    "hits_device": sum(int(a == b) for a, b in zip(d["top1"], want)),
                                    "latent_max_abs_diff": diff, "latent_max_abs": scale,
                                    "disagreements": [{"position": i, "reference": a, "device": b, "device_margin": d["margin"][i]}
                                                      for i, (a, b) in enumerate(zip(r["top1"], d["top1"])) if a != b]}
        for key in ("positions", "agree", "hits_reference", "hits_device"):
            report[key] += report["fixtures"][name][key]
    report["agreement"] = report["agree"] / report["positions"] if report["positions"] else None
    report["verdict"] = "PASS" if report["agreement"] is not None and report["agreement"] >= FLOOR and report["run_to_run_identical"] else "FAIL"
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"MTP-SEQKV-CHECK {report['verdict']} agree {report['agree']}/{report['positions']} hits reference {report['hits_reference']} device {report['hits_device']} "
          f"run_to_run {report['run_to_run_identical']} " + " ".join(f"{n}:latent_diff={v['latent_max_abs_diff']:.4g}/{v['latent_max_abs']:.3g}" for n, v in report["fixtures"].items()))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
