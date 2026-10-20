import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNLINKED = {
    "scheduler/topology_switch.c": {"topology_switch"},
}
UNLINKED_GROUP = "unlinked_components"
RULE = re.compile(r"^(?![\t#])([^\s:=][^:=]*?):(?!=)", re.MULTILINE)


def makefile_rules(text):
    headers = list(RULE.finditer(text))
    rules = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        for target in header.group(1).split():
            rules[target] = text[header.start():end]
    return rules


def product_text(rules):
    parts = [(ROOT / "sources.mk").read_text(), (ROOT / "runtime/weightd_sources.mk").read_text()]
    parts += [path.read_text() for path in sorted(ROOT.glob("modules/*/Makefile*"))]
    parts += [path.read_text() for path in sorted(ROOT.glob("modules/*.mk"))]
    parts += [body for target, body in rules.items() if not target.startswith("build/test_")]
    return "\n".join(parts)


def reliability_groups():
    source = (ROOT / "tools/test_serving_reliability_host.py").read_text()
    block = source.split("GROUPS = {", 1)[1].split("\n}", 1)[0]
    return {group: set(names.split()) for group, names in re.findall(r'"(\w+)":\s*"([^"]*)"', block)}


def main():
    failures = []
    rules = makefile_rules((ROOT / "Makefile").read_text())
    product = product_text(rules)
    for source, tests in UNLINKED.items():
        if not (ROOT / source).is_file():
            failures.append(f"{source} is gone; remove it and its tests from UNLINKED")
        if source in product or Path(source).name in product:
            failures.append(f"{source} is now built into a product target; remove it from UNLINKED and move {sorted(tests)} into a production reliability group")
    expected = set().union(*UNLINKED.values())
    compiling = {target[len("build/test_"):] for target, body in rules.items()
                 if target.startswith("build/test_") and any(source in body for source in UNLINKED)}
    if compiling != expected:
        failures.append(f"tests compiling unlinked sources {sorted(compiling)} differ from UNLINKED {sorted(expected)}")
    groups = reliability_groups()
    if groups.get(UNLINKED_GROUP) != expected:
        failures.append(f"reliability group {UNLINKED_GROUP} is {sorted(groups.get(UNLINKED_GROUP, set()))}, expected {sorted(expected)}")
    for group, names in groups.items():
        if group != UNLINKED_GROUP and names & expected:
            failures.append(f"reliability group {group} counts unlinked-component tests {sorted(names & expected)} as serving evidence")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS {len(UNLINKED)} unlinked components: tests {sorted(expected)} are reported apart from serving evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
