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


def parse(path):
    page = Page()
    page.feed(path.read_text(encoding="utf-8"))
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
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures or not pages:
        return 1
    print(f"site: {len(pages)} pages, every local link and anchor resolves, no external scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
