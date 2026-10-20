#!/usr/bin/env python3
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "sample_host.cu"
VOCAB = 300
ROWS = 3


def order(logits):
    return sorted(range(len(logits)), key=lambda t: (-logits[t], t))


def kept_set(logits, temperature, top_p, top_k):
    ranked = order(logits)
    if top_k and top_k < len(logits):
        ranked = ranked[:top_k]
    if top_p >= 1.0:
        return ranked, None
    top = logits[ranked[0]]
    weights = [math.exp((logits[t] - top) / temperature) for t in ranked]
    total = sum(weights)
    running = 0.0
    for index, weight in enumerate(weights):
        before = running
        running += weight
        if running >= top_p * total:
            margin = min(top_p * total - before, running - top_p * total) / total
            return ranked[:index + 1], margin
    return ranked, 0.0


def main():
    with tempfile.TemporaryDirectory() as directory:
        binary = Path(directory) / "sample_host"
        build = subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}/tests/host_cuda",
                                f"-I{ROOT}", "-x", "c++", str(SOURCE), "-o", str(binary)], capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr[:2000])
            return 1
        output = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout
    failures = []
    logits = [[0.0] * VOCAB for _ in range(ROWS)]
    cases, counts, logprobs, shard_equal, forced, forced_lp = {}, {}, {}, {}, {}, {}
    for line in output.splitlines():
        part = line.split()
        if not part:
            continue
        if part[0] == "FAIL":
            failures.append(line)
        elif part[0] == "LOGIT":
            logits[int(part[1])][int(part[2])] = float(part[3])
        elif part[0] == "CASE":
            cases[part[1]] = (float(part[2]), float(part[3]), int(part[4]), int(part[5]), int(part[6]))
        elif part[0] == "COUNT":
            counts.setdefault(part[1], {})[(int(part[2]), int(part[3]))] = int(part[4])
        elif part[0] == "LP":
            logprobs.setdefault(part[1], {})[(int(part[2]), int(part[3]))] = (int(part[4]), float(part[5]))
        elif part[0] == "SHARD_EQUAL":
            shard_equal[part[1]] = part[2] == "1"
        elif part[0] == "FORCED":
            forced[int(part[1])] = (int(part[2]), int(part[3]))
        elif part[0] == "FORCED_LP":
            forced_lp[(int(part[1]), int(part[2]))] = (int(part[3]), float(part[4]))
    if "DONE" not in output:
        failures.append("harness did not finish")
    for name, (temperature, top_p, top_k, logprob_count, samples) in cases.items():
        if not shard_equal.get(name):
            failures.append(f"{name}: one shard and four shards gave different tokens, logits or logprobs")
        for row in range(ROWS):
            row_logits = logits[row]
            top = max(row_logits)
            lse = top + math.log(sum(math.exp(v - top) for v in row_logits))
            row_counts = {t: c for (r, t), c in counts.get(name, {}).items() if r == row}
            if sum(row_counts.values()) != samples:
                failures.append(f"{name} row {row}: {sum(row_counts.values())} samples, expected {samples}")
                continue
            if temperature == 0.0:
                best = order(row_logits)[0]
                if row_counts != {best: 1}:
                    failures.append(f"{name} row {row}: greedy picked {row_counts}, expected token {best} (ties go to the lower id)")
            else:
                kept, margin = kept_set(row_logits, temperature, top_p, top_k)
                if margin is not None and margin < 1e-4:
                    failures.append(f"{name} row {row}: the test's nucleus boundary is ambiguous (margin {margin})")
                outside = [t for t in row_counts if t not in kept]
                if outside:
                    failures.append(f"{name} row {row}: sampled tokens outside the kept set: {outside[:5]}")
                kept_top = max(row_logits[t] for t in kept)
                weights = {t: math.exp((row_logits[t] - kept_top) / temperature) for t in kept}
                total = sum(weights.values())
                chi = 0.0
                for t in kept:
                    expected = samples * weights[t] / total
                    observed = row_counts.get(t, 0)
                    if expected >= 5.0:
                        chi += (observed - expected) ** 2 / expected
                    elif observed > expected + 12.0:
                        failures.append(f"{name} row {row}: token {t} drawn {observed} times, expected {expected:.2f}")
                dof = max(1, sum(1 for t in kept if samples * weights[t] / total >= 5.0) - 1)
                if chi > dof + 6.0 * math.sqrt(2.0 * dof) + 10.0:
                    failures.append(f"{name} row {row}: frequencies disagree with the renormalized distribution (chi2 {chi:.1f}, dof {dof})")
            if logprob_count:
                reported = logprobs.get(name, {})
                chosen_token, chosen_lp = reported[(row, 0)]
                if abs(chosen_lp - (row_logits[chosen_token] - lse)) > 2e-5:
                    failures.append(f"{name} row {row}: chosen logprob {chosen_lp} != log_softmax {row_logits[chosen_token] - lse}")
                expected_top = order(row_logits)[:logprob_count - 1]
                for index, token in enumerate(expected_top):
                    got_token, got_lp = reported[(row, index + 1)]
                    if got_token != token or abs(got_lp - (row_logits[token] - lse)) > 2e-5:
                        failures.append(f"{name} row {row}: top logprob {index} is ({got_token}, {got_lp}), expected ({token}, {row_logits[token] - lse})")
                        break
    if len(forced) != ROWS:
        failures.append("forced-greedy case missing")
    for row, (wanted, got) in sorted(forced.items()):
        row_logits = logits[row]
        top = max(row_logits)
        lse = top + math.log(sum(math.exp(v - top) for v in row_logits))
        if row < 2 and got != wanted:
            failures.append(f"forced greedy row {row}: emitted {got}, the caller's greedy token is {wanted}")
        if row == 2 and got not in order(row_logits)[:5]:
            failures.append(f"forced greedy row {row}: a sampled row took {got}, outside its top-5")
        chosen = forced_lp.get((row, 0))
        if chosen is None or chosen[0] != got or abs(chosen[1] - (row_logits[got] - lse)) > 2e-5:
            failures.append(f"forced greedy row {row}: chosen logprob {chosen} is not at the remapped row or is not log_softmax of {got}")
        for index, token in enumerate(order(row_logits)[:3]):
            entry = forced_lp.get((row, index + 1))
            if entry is None or entry[0] != token or abs(entry[1] - (row_logits[token] - lse)) > 2e-5:
                failures.append(f"forced greedy row {row}: top logprob {index} is {entry}, expected token {token}")
                break
    for name in ("greedy_logprobs", "temperature", "top_k", "top_p", "top_k_top_p", "logprobs_max"):
        if name not in cases:
            failures.append(f"case {name} missing")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print(f"PASS sampling kernel: greedy ties, temperature, top-k, top-p and both match the renormalized distribution over {cases['top_k'][4]} draws, "
          "chosen and top logprobs equal log_softmax, one shard and four shards agree bit for bit, "
          "a caller's greedy token is kept and logprobs land on the remapped rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
