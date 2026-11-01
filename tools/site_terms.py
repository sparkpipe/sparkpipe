import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site" / "terms"
OUT = ROOT / "site"

ESC = {"\\[": "\x01", "\\]": "\x02", "\\_": "\x03", "\\*": "\x04", "\\|": "\x05"}
UNESC = {"\x01": "[", "\x02": "]", "\x03": "_", "\x04": "*", "\x05": "|"}


def inline(text):
    for k, v in ESC.items():
        text = text.replace(k, v)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    for k, v in UNESC.items():
        text = text.replace(k, v)
    return text


def split_row(line):
    line = line.strip()
    for k, v in ESC.items():
        line = line.replace(k, v)
    return [c.strip() for c in line.strip("|").split("|")]


def table(lines):
    head = split_row(lines[0])
    aligns = []
    for spec in split_row(lines[1]):
        aligns.append("right" if spec.endswith(":") and not spec.startswith(":") else "")
    out = ['<div class="scroll"><table>', "<tr>" + "".join(f"<th{' class=\"num\"' if a else ''}>{inline(c)}</th>" for c, a in zip(head, aligns)) + "</tr>"]
    for line in lines[2:]:
        cells = split_row(line)
        cells += [""] * (len(head) - len(cells))
        out.append("<tr>" + "".join(f"<td{' class=\"num\"' if a else ''}>{inline(c)}</td>" for c, a in zip(cells, aligns)) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


PART = re.compile(r"^(?:Part (\d+)\.|제(\d+)부)\s")
NUMBERED = re.compile(r"^(?:제)?(\d+)[.조]")


def heading_id(head, n):
    part = PART.match(head)
    if part:
        return f"part{part.group(1) or part.group(2)}"
    m = NUMBERED.match(head)
    if m:
        return f"s{m.group(1)}"
    return re.sub(r"[^a-z0-9]+", "-", head.lower()).strip("-") or f"h{n}"


def body(md, ids=None):
    lines = md.splitlines()
    title = lines[0][2:].strip()
    lines = lines[1:]
    blocks, cur = [], []
    for line in lines + [""]:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    out, open_section, found, parts = [], False, [], False
    for b in blocks:
        first = b[0]
        if first.startswith("## "):
            if open_section:
                out.append("</section>")
            head = first[3:].strip()
            own = heading_id(head, len(found) + 1)
            if ids is None:
                sid = own
            elif len(found) >= len(ids) or ((PART.match(head) or NUMBERED.match(head)) and own != ids[len(found)]):
                raise ValueError(f"heading {head!r} has no twin with the same number")
            else:
                sid = ids[len(found)]
            found.append(sid)
            part = PART.match(head)
            parts = parts or bool(part)
            level = 3 if parts and not part else 2
            kind = ' class="part"' if part else ""
            out.append(f'<section id="{sid}"{kind}>\n<h{level}>{inline(head)}</h{level}>')
            open_section = True
            rest = b[1:]
            if rest:
                out.append(f"<p>{inline(' '.join(rest))}</p>")
        elif first.startswith("|"):
            out.append(table(b))
        elif first.startswith("- "):
            items, item = [], None
            for l in b:
                if l.startswith("- "):
                    if item is not None:
                        items.append(item)
                    item = l[2:]
                else:
                    item += " " + l.strip()
            items.append(item)
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
        else:
            text = " ".join(l.strip() for l in b)
            m = re.match(r"^(\d+)\.(\d+)\.", text)
            pid = f' id="c{m.group(1)}-{m.group(2)}"' if m else ""
            out.append(f"<p{pid}>{inline(text)}</p>")
    if open_section:
        out.append("</section>")
    if ids is not None and len(found) != len(ids):
        raise ValueError(f"{len(found)} headings, but the twin has {len(ids)}")
    return title, "\n".join(out), found


STYLE = """<style>
.policy { padding-block: 48px 64px; display: grid; gap: 28px; }
.policy h1 { font-size: clamp(32px, 5vw, 56px); }
.policy h2 { font-size: clamp(22px, 2.6vw, 30px); margin-top: 12px; }
.policy .part h2 { font-size: clamp(28px, 3.6vw, 42px); margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--rule); }
.policy h3 { font-size: clamp(20px, 2.2vw, 26px); margin-top: 8px; }
.policy section { display: grid; gap: 12px; max-width: 76ch; }
.policy p, .policy li { color: var(--steel); font: 400 16px/1.6 var(--text); }
.policy ul { margin: 0; padding-left: 1.2em; display: grid; gap: 8px; }
.policy b { color: var(--ink); font-weight: 650; }
.policy table { border-collapse: collapse; width: 100%; font: 400 15px/1.5 var(--text); }
.policy th, .policy td { text-align: left; vertical-align: top; padding: 10px 12px; border-top: 1px solid var(--rule); color: var(--steel); }
.policy th { color: var(--ink); font-weight: 650; }
.policy th.num, .policy td.num { text-align: right; font-variant-numeric: tabular-nums; }
.policy .scroll { overflow-x: auto; max-width: 76ch; }
.policy .scroll table { min-width: 520px; }
.policy .notice { max-width: 76ch; border: 1px solid var(--rule); border-left: 3px solid var(--ink); padding: 14px 16px; }
.policy .notice p { color: var(--ink); }
.effective { font: 500 12px/1.6 var(--mono); color: var(--faint); }
</style>"""

PAGES = [
    {
        "src": "agreement-en.md", "out": "agreement.html", "lang": "en", "tag": "terms",
        "title": "SparkPipe Customer Agreement", "other": "agreement-ko.html",
        "desc": "Draft SparkPipe Customer Agreement for hour-credits, demo hours, weeks on a pinned fleet and the waitlist, published for review.",
        "lead": "This Agreement is accepted online while signed in to the account, and we record the hashes of its exact texts. The Korean text prevails (clause 12.3).",
    },
    {
        "src": "agreement-ko.md", "out": "agreement-ko.html", "lang": "ko", "tag": "약관",
        "title": "SparkPipe 고객 계약", "other": "agreement.html",
        "desc": "SparkPipe 시간 크레딧, 데모 시간, 고정 플릿의 주 예약과 대기 명단에 관한 고객 계약 검토용 공개 문안.",
        "lead": "본 계약은 계정에 로그인한 상태에서 온라인으로 동의되며, 공급자는 그 정확한 문안의 해시를 기록한다. 국문본이 우선한다(제12.3조).",
    },
]

NOTICE = {
    "en": "<b>Published for review.</b> Demo hours can be booked from 1 December 2026 and weeks on a pinned fleet from 4 January 2027. This text may still change. Send comments to <a href=\"mailto:hyor@sparkpipe.ai\">hyor@sparkpipe.ai</a>.",
    "ko": "<b>검토용으로 공개한 문안입니다.</b> 데모 시간은 2026년 12월 1일부터, 고정 플릿의 주는 2027년 1월 4일부터 예약할 수 있습니다. 문안은 바뀔 수 있습니다. 의견은 <a href=\"mailto:hyor@sparkpipe.ai\">hyor@sparkpipe.ai</a>로 보내 주십시오.",
}
VERSION = {"en": "Version of 1 November 2026", "ko": "2026년 11월 1일 판"}
NAV = {
    "en": ('<a href="terms.html">All terms</a>', '<a href="privacy.html">Privacy</a>', '<a href="index.html">Home</a>'),
    "ko": ('<a href="terms-ko.html">약관 목록</a>', '<a href="privacy-ko.html">개인정보처리방침</a>', '<a href="index-ko.html">홈</a>'),
}
HOME = {"en": "index.html", "ko": "index-ko.html"}
COMPANY = {"en": "https://hyor.io/", "ko": "https://hyor.io/index-ko.html"}
GROUP = {"en": "Language / 언어", "ko": "언어 / Language"}


def toggle(p):
    en, ko = (p["out"], p["other"]) if p["lang"] == "en" else (p["other"], p["out"])
    current = {"en": ' aria-current="true"', "ko": ""} if p["lang"] == "en" else {"en": "", "ko": ' aria-current="true"'}
    attr = {"en": ' lang="en"' if p["lang"] == "ko" else "", "ko": ' lang="ko"' if p["lang"] == "en" else ""}
    return (f'<div class="lang" role="group" aria-label="{GROUP[p["lang"]]}">'
            f'<a href="{en}?lang=en" hreflang="en"{attr["en"]}{current["en"]}>EN</a>'
            f'<a href="{ko}?lang=ko" hreflang="ko"{attr["ko"]}{current["ko"]}>한국어</a></div>')


def alternates(p):
    en, ko = (p["out"], p["other"]) if p["lang"] == "en" else (p["other"], p["out"])
    return (f'<link rel="alternate" hreflang="en" href="https://sparkpipe.ai/{en}">\n'
            f'<link rel="alternate" hreflang="ko" href="https://sparkpipe.ai/{ko}">\n'
            f'<link rel="alternate" hreflang="x-default" href="https://sparkpipe.ai/{en}">')


def twin_ids(p):
    if p["lang"] == "en":
        return None
    twin = next(q for q in PAGES if q["out"] == p["other"])
    return body((SRC / twin["src"]).read_text(encoding="utf-8"))[2]


def page(p):
    md = (SRC / p["src"]).read_text(encoding="utf-8")
    try:
        title, content, _ = body(md, twin_ids(p))
    except ValueError as error:
        raise ValueError(f"{p['src']}: {error}") from None
    nav = "".join(NAV[p["lang"]])
    return f"""<!doctype html>
<html lang="{p['lang']}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(p['title'])}</title>
<meta name="description" content="{html.escape(p['desc'])}">
<meta name="theme-color" content="#0a0c10">
{alternates(p)}
<link rel="icon" href="/favicon.ico" sizes="32x32">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="stylesheet" href="site.css?v=91e4a0e1">
{STYLE}
</head>
<body>
<header class="nav">
  <div class="wrap">
    <a class="brand" href="{HOME[p['lang']]}"><picture class="logo"><img src="hyor-ai-logo-reversed.svg" width="144" height="20" alt="HYOR AI"></picture><img class="seal" src="hyor-seal.svg" width="28" height="28" alt="HYOR AI"><span class="product">SparkPipe</span><span class="tag">{p['tag']}</span></a>
    <nav class="nav-links" aria-label="Sections">
      {nav}
    </nav>
    <div class="nav-actions">
      {toggle(p)}
    </div>
  </div>
</header>

<main class="wrap policy">
<section>
<h1>{inline(title)}</h1>
<p class="effective">{VERSION[p['lang']]}</p>
<p>{inline(p['lead'])}</p>
</section>
<div class="notice"><p>{NOTICE[p['lang']]}</p></div>
{content}
</main>

<footer class="foot">
  <div class="wrap">
    <a class="maker" href="{COMPANY[p['lang']]}"><img src="hyor-seal.svg" width="20" height="20" alt="">© 2026 HYOR AI Tech Ltd</a>
    <nav aria-label="Footer">{nav}<a href="mailto:hyor@sparkpipe.ai">hyor@sparkpipe.ai</a></nav>
  </div>
</footer>
</body>
</html>
"""


def pages():
    return {p["out"]: page(p) for p in PAGES}


def main():
    for name, text in pages().items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
