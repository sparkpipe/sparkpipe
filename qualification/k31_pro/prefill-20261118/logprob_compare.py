import json
import math
import sys


def load(path):
    data = json.load(open(path))
    return data["tokens"], data["token_logprobs"]


def main(reference_path, candidate_path):
    reference_tokens, reference = load(reference_path)
    candidate_tokens, candidate = load(candidate_path)
    shared = 0
    while shared < min(len(reference_tokens), len(candidate_tokens)) and reference_tokens[shared] == candidate_tokens[shared]:
        shared += 1
    deltas, same_top = [], 0
    for position in range(shared):
        deltas.append(abs(candidate[position][0][1] - reference[position][0][1]))
        same_top += [entry[0] for entry in reference[position][1:]] == [entry[0] for entry in candidate[position][1:]]
    probability = [abs(math.exp(candidate[position][0][1]) - math.exp(reference[position][0][1])) for position in range(shared)]
    print(f"generated {len(reference_tokens)} and {len(candidate_tokens)} tokens; the first {shared} are identical")
    if shared:
        worst = max(range(shared), key=lambda position: deltas[position])
        print(f"over those: the chosen token's logprob moves {sum(deltas) / shared:.4f} on average, at most {deltas[worst]:.4f} (position {worst}); "
              f"its probability moves at most {max(probability):.4f}; the top-5 tokens and their order agree at {same_top} of {shared} positions")
    if shared < min(len(reference_tokens), len(candidate_tokens)):
        def top(entries):
            return ", ".join(f"{token} {logprob:.3f}" for token, logprob in entries[1:3])
        print(f"position {shared}: reference picks {reference_tokens[shared]} (top 2: {top(reference[shared])}), "
              f"candidate picks {candidate_tokens[shared]} (top 2: {top(candidate[shared])})")
        margin = reference[shared][1][1] - reference[shared][2][1]
        print(f"the reference's margin between its first and second token there is {margin:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
