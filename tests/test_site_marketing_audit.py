#!/usr/bin/env python3

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STATUS = (ROOT / "PERFORMANCE_STATUS.md").read_text()
RELEASES = ROOT / "qualification" / "releases"
MARKETING = ("index.html", "index-ko.html", "fleets.html", "fleets-ko.html")
TWINS = (("index.html", "index-ko.html"), ("fleets.html", "fleets-ko.html"))


def page(name):
    return (SITE / name).read_text()


def pricing():
    text = (SITE / "pricing.js").read_text()
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def model_rows(text):
    rows = {}
    for match in re.finditer(r'<span class="mono">([a-z0-9.\-]+)</span></td>(.*?)</tr>', text):
        rows[match.group(1)] = re.findall(r'<td class="num">([^<]*)</td>', match.group(2))
    return rows


def check_model_speeds(failures):
    models = {m["id"]: m for m in pricing()["models"]}
    for name in ("fleets.html", "fleets-ko.html"):
        rows = model_rows(page(name))
        for mid, model in models.items():
            if mid not in rows:
                failures.append(f"{name}: no row for {mid}")
                continue
            decode, prefill = (cell.replace("¹", "").replace("tok/s", "").strip() for cell in rows[mid][1:3])
            if model["decode"] is None:
                if (decode, prefill) != ("–", "–"):
                    failures.append(f"{name}: {mid} shows {decode}/{prefill} but pricing.js has no receipted speed")
                continue
            want = (f"{model['decode']:.1f}", f"{model['prefill']:,}")
            if (decode, prefill) != want:
                failures.append(f"{name}: {mid} shows {decode}/{prefill}, pricing.js has {want}")
            line = next((row for row in STATUS.splitlines() if row.startswith(f"| `{mid}` |")), "")
            if f"{want[0]} tok/s" not in line or f"{want[1]} tok/s" not in line:
                failures.append(f"{mid}: {want} has no row in PERFORMANCE_STATUS.md")


def section(text, ident):
    start = text.index(f'<section id="{ident}">')
    return text[start:text.index("</section>", start)]


def check_precision_and_limits(failures):
    receipted = {m["name"] for m in pricing()["models"] if m["decode"] is not None}
    for name in ("fleets.html", "fleets-ko.html"):
        text = page(name)
        if re.search(r"\d tok/s", section(text, "precision")):
            failures.append(f"{name}: the precision comparison quotes a speed that no published receipt covers")
        for model, cells in re.findall(r'<tr><td><b>([^<]+)</b></td><td>[^<]*</td>((?:<td class="num">[^<]*</td>)+)</tr>', section(text, "precision-choice")):
            speeds = re.findall(r'<td class="num">([^<]*)</td>', cells)[:2]
            if model not in receipted and speeds != ["–", "–"]:
                failures.append(f"{name}: the precision table shows {model} at {speeds}, but the model table has no receipted speed for it")
        rows = re.findall(r'<tr><td><b>([^<]+)</b></td><td class="num">([^<]*)</td><td class="num">([^<]*)</td></tr>', section(text, "limits"))
        if not rows:
            failures.append(f"{name}: no rows in the hardware limit table")
        for model, decode, prefill in rows:
            if model not in receipted and (decode, prefill) != ("–", "–"):
                failures.append(f"{name}: the hardware limit table shows {model} at {decode}/{prefill} of the floors, but the model table has no receipted speed for it")


def check_cited_releases(failures):
    for path in sorted(SITE.glob("*.html")):
        for build in re.findall(r"(?:build|release|릴리스|빌드) ([0-9a-f]{9})\b", path.read_text()):
            if not (RELEASES / build).is_dir():
                failures.append(f"{path.name}: cites build {build} with no qualification/releases/{build}/")


def check_ranges(failures):
    for name in ("index.html", "index-ko.html"):
        for low, high in re.findall(r"(\d+(?:\.\d+)?)–(\d+(?:\.\d+)?) ?(?:s\b|초)", page(name)):
            if f"{low}-{high} s" not in STATUS:
                failures.append(f"{name}: {low}–{high} s is not a measured range in PERFORMANCE_STATUS.md")
        for minutes in re.findall(r"<strong>(\d+\.\d+) ?(?:min|분)</strong>", page(name)):
            if f"{minutes} minutes" not in STATUS:
                failures.append(f"{name}: {minutes} min has no receipt in PERFORMANCE_STATUS.md")


def check_bindings(failures):
    if "maintenance: { hours: String(B.maintenanceHoursPerWeek) }" not in (SITE / "prices.js").read_text():
        failures.append("prices.js does not bind data-maintenance to maintenanceHoursPerWeek")
    for name in MARKETING:
        if 'data-hourfleet="hours"' not in page(name):
            failures.append(f"{name}: the weekly Hour Fleet cap per account is not shown")
    for name, word in (("index.html", "consecutive"), ("index-ko.html", "연속")):
        row = re.search(r'<tr><th scope="row">(?:Downtime|장애)</th><td>(.*?)</td></tr>', page(name))
        if not row or word not in row.group(1) or 'data-maintenance="hours"' not in row.group(1):
            failures.append(f"{name}: the Downtime row does not follow clause 4.4 or the published maintenance allowance")
    for name, words in (("security.html", ("15 consecutive minutes", "every booked hour it touches")), ("security-ko.html", ("15분 넘게 연속된 장애", "걸친 모든 예약 시간"))):
        clause = next((item for item in re.findall(r"<li>(.*?)</li>", page(name), re.S) if "4.4" in item), "")
        if not all(word in clause for word in words):
            failures.append(f"{name}: the clause 4.4 credit does not say that downtime of more than 15 consecutive minutes credits every booked hour it touches")


def check_cells(failures):
    for name in ("fleets.html", "fleets-ko.html"):
        for cell in re.findall(r'<td class="num">([^<]*)</td>', page(name)):
            if not re.search(r"\d|–", cell):
                failures.append(f"{name}: number cell holds {cell!r}, not a measured value or –")
    for left, right in TWINS:
        a, b = model_rows(page(left)), model_rows(page(right))
        if a != b:
            failures.append(f"{right}: model rows differ from {left}")


class Head(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta, self.links, self.headings, self.images, self.title = {}, [], [], [], ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta":
            key = values.get("property") or values.get("name")
            if key:
                self.meta.setdefault(key, []).append(values.get("content", ""))
        if tag == "link":
            self.links.append(values)
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        if tag == "img":
            self.images.append(values)
        self.in_title = tag == "title"

    def handle_endtag(self, tag):
        self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def check_heads(failures):
    for name in MARKETING:
        head = Head()
        head.feed(page(name))
        url = "https://sparkpipe.ai/" + name
        if not head.title.strip():
            failures.append(f"{name}: no title")
        for key in ("description", "og:title", "og:description", "og:url", "og:image", "og:locale", "og:locale:alternate"):
            if not any(value.strip() for value in head.meta.get(key, [])):
                failures.append(f"{name}: no {key}")
        if head.meta.get("og:url") != [url]:
            failures.append(f"{name}: og:url is not {url}")
        if not any(link.get("rel") == "canonical" and link.get("href") == url for link in head.links):
            failures.append(f"{name}: no self-referencing canonical {url}")
        if not any(link.get("rel") == "alternate" and link.get("hreflang") == "x-default" for link in head.links):
            failures.append(f"{name}: no x-default alternate")
        for image in head.images:
            if "alt" not in image:
                failures.append(f"{name}: img {image.get('src')} has no alt")
        for before, after in zip(head.headings, head.headings[1:]):
            if after > before + 1:
                failures.append(f"{name}: heading level jumps from h{before} to h{after}")
                break


def check_phone_nav(failures):
    for name in MARKETING:
        footer = page(name)[page(name).rfind("<footer"):]
        for target in ("availability", "fleets"):
            if not re.search(rf'href="{target}(?:-ko)?\.html"', footer):
                failures.append(f"{name}: the footer has no {target} link for phone widths")


def main():
    failures = []
    for check in (check_model_speeds, check_precision_and_limits, check_cited_releases, check_ranges, check_bindings, check_cells, check_heads, check_phone_nav):
        check(failures)
    if failures:
        print("\n".join(failures))
        return 1
    print("marketing pages: speeds match pricing.js and PERFORMANCE_STATUS.md, the precision and hardware limit tables show no speed the model table lacks, the security page credits downtime as clause 4.4 does, cited releases have receipts, prices bind the published settings, heads carry title, description, Open Graph, canonical and x-default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
