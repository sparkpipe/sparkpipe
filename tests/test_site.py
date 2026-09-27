#!/usr/bin/env python3

from html.parser import HTMLParser
from pathlib import Path


SITE = Path(__file__).resolve().parents[1] / "site"
EXTERNAL = ("https://", "http://", "mailto:")
ALLOWED_STYLES = ("https://fonts.googleapis.com/",)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.scripts, self.styles = set(), [], [], []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        for name in ("href", "src"):
            if values.get(name) and not (tag == "use" or values[name].startswith(EXTERNAL)):
                self.links.append(values[name].split("?")[0])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        if tag == "link" and values.get("rel") == "stylesheet":
            self.styles.append(values["href"])


def cell_count(row_html):
    import re
    count = 0
    for tag in re.findall(r"<t[dh]\b[^>]*>", row_html):
        span = re.search(r'colspan="(\d+)"', tag)
        count += int(span.group(1)) if span else 1
    return count


def check_tables(name, page, failures):
    import re
    for table in re.findall(r"<table.*?</table>", page.source, re.S):
        header = re.search(r"<tr>(.*?)</tr>", table, re.S)
        if not header:
            continue
        expected = cell_count(header.group(1))
        for row in re.findall(r"<tr>(.*?)</tr>", table, re.S)[1:]:
            if cell_count(row) != expected:
                failures.append(f"{name}: table row has {cell_count(row)} cells, header has {expected}")


def parse(path):
    source = path.read_text(encoding="utf-8")
    page = Page()
    page.source = source
    page.feed(source)
    return page


def main():
    pages = {path.name: parse(path) for path in sorted(SITE.glob("*.html"))}
    failures = []
    for name, page in pages.items():
        for link in page.links:
            target, _, fragment = link.partition("#")
            if target and not (SITE / target.lstrip("/")).is_file():
                failures.append(f"{name}: {link} names no file in site/")
            elif fragment and fragment not in pages.get(target or name, page).ids:
                failures.append(f"{name}: {link} names no element")
        failures += [f"{name}: external script {src}" for src in page.scripts]
        failures += [f"{name}: stylesheet {href}" for href in page.styles if href.startswith(EXTERNAL) and not href.startswith(ALLOWED_STYLES)]
        check_tables(name, page, failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures or not pages:
        return 1
    print(f"site: {len(pages)} pages, every local link and anchor resolves, no external scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
