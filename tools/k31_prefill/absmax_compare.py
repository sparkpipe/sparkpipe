import re
import sys


def absmax(path):
    passes = {}
    for line in open(path):
        match = re.search(r"streamed prefill pass (\d+) rank \d+ hidden absmax by layer \(-1 = embedding\):(.*)$", line)
        if match:
            passes[int(match.group(1))] = {int(layer): float(value) for layer, value in re.findall(r"(-?\d+):([0-9.eE+-]+)", match.group(2))}
    return passes


def main(reference_path, candidate_path):
    reference, candidate = absmax(reference_path), absmax(candidate_path)
    worst = 0.0
    for pass_index in sorted(reference):
        cells = []
        for layer in sorted(reference[pass_index]):
            if layer not in candidate.get(pass_index, {}):
                continue
            ref, got = reference[pass_index][layer], candidate[pass_index][layer]
            change = (got - ref) / ref if ref != 0.0 else 0.0
            worst = max(worst, abs(change))
            cells.append(f"{layer}:{ref:g}->{got:g} ({change * 100:+.1f}%)")
        print(f"pass {pass_index}: " + " ".join(cells))
    print(f"largest change of a layer's hidden absmax: {worst * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
