"""Row hash comparator: joins traces by row identity and names the first divergence."""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "row_hash_compare.py"
HEADER = "wave\tstep\tlayer\tsite\trow\tsequence\tposition\thash"
SITES = ("q_a", "attn_out", "hidden_mlp")


def trace(wave, batched, changes=(), drop=(), duplicate=False):
    lines = [HEADER]
    for step in range(2):
        for layer in range(3):
            for site in SITES:
                for sequence in range(9):
                    key = (step, layer, site, sequence)
                    if key in drop:
                        continue
                    value = (step * 1000003 + layer * 7919 + SITES.index(site) * 104729 + sequence * 31) ^ (0xABCDEF if key in changes else 0)
                    row = sequence if batched else 0
                    lines.append(f"{wave}\t{step}\t{layer}\t{site}\t{row}\t{sequence}\t{100 + sequence + step}\t{value:016x}")
                    if duplicate and key == (0, 0, "q_a", 0):
                        lines.append(lines[-1])
    return "\n".join(lines) + "\n"


def run(directory, reference, candidate):
    left = pathlib.Path(directory) / "reference.tsv"
    right = pathlib.Path(directory) / "candidate.tsv"
    left.write_text(reference)
    right.write_text(candidate)
    result = subprocess.run([sys.executable, str(TOOL), str(left), str(right)], capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout


def fields(output, prefix):
    for line in output.splitlines():
        if line.startswith(prefix + " "):
            return dict(item.split("=", 1) for item in line.split()[1:])
    raise AssertionError(f"{prefix} missing from:\n{output}")


with tempfile.TemporaryDirectory(prefix="row-hash-compare-") as directory:
    code, output = run(directory, trace("serial", False), trace("b9", True))
    assert code == 0, output
    assert fields(output, "ROWHASH-EQUAL") == {"entries": str(2 * 3 * 3 * 9), "sites": "3"}, output

    injected = {(1, 2, "hidden_mlp", 8), (1, 1, "attn_out", 4), (1, 2, "q_a", 2)}
    code, output = run(directory, trace("serial", False), trace("b9", True, changes=injected))
    assert code == 1, output
    first = fields(output, "ROWHASH-FIRST-DIVERGENCE")
    assert (first["step"], first["layer"], first["site"], first["sequence"]) == ("1", "1", "attn_out", "4"), output
    assert (first["reference_row"], first["candidate_row"], first["candidate_wave"], first["diverged"]) == ("0", "4", "b9", "3"), output
    for line in output.splitlines():
        if line.startswith("ROWHASH-SITE "):
            site = dict(item.split("=", 1) for item in line.split()[1:])
            expected = {"q_a": ("1", "2"), "attn_out": ("1", "1"), "hidden_mlp": ("1", "2")}[site["site"]]
            assert (site["diverged"], site["first_layer"]) == expected, output

    code, output = run(directory, trace("serial", False), trace("b9", True, drop={(0, 1, "q_a", 3)}))
    assert code == 2 and "first_missing_in=candidate" in output and "sequence=3" in output, output

    code, output = run(directory, trace("serial", False), trace("b9", True, duplicate=True))
    assert code == 2 and "duplicate entry" in output, output

    code, output = run(directory, "wave\tlayer\n", trace("b9", True))
    assert code == 2 and "ROWHASH-ERROR" in output, output

print("PASS row hash comparator: equal traces pass, the first injected divergence is named by step/layer/site/sequence with both rows, and missing, duplicate or malformed traces fail with status 2")
