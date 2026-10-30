#!/usr/bin/env python3

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "docs" / "brand" / "hyor-ai"
SITE = ROOT / "site"
COMPANY = ROOT / "deployment" / "vps" / "hyor.io"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

INK = "#111522"
RED = "#D63A26"
WHITE = "#FFFFFF"
BLACK = "#000000"

HYOR = (
    '<path d="M0 0h22v39h40V0h22v100H62V61H22v39H0z"/>'
    '<path transform="translate(102 0)" d="M0 0h26l20 33.1L66 0h26L57 58v42H35V58z"/>'
    '<path transform="translate(198 0)" d="M0 50a51 51 0 1 0 102 0a51 51 0 1 0 -102 0z M23 50a28 28 0 1 1 56 0a28 28 0 1 1 -56 0z"/>'
    '<g transform="translate(318 0)"><rect width="22" height="100"/>'
    '<path d="M0 0H52A29 29 0 0 1 52 58H0Z M22 22V36H52A7 7 0 0 0 52 22Z"/><path d="M30 36H56L90 100H64Z"/></g>'
    '<g transform="translate(458 0)"><path d="M0 100L38 0H58L96 100H72.7L65.1 80H30.9L23.3 100Z M48 35L38.5 60H57.5Z"/>'
    '<rect x="116" width="22" height="100"/></g>'
)
NETWORK = (
    '<rect x="52" y="15" width="16" height="6"/><circle cx="52" cy="18" r="5"/><circle cx="68" cy="18" r="5"/>'
    '<rect x="29" y="30" width="62" height="6"/><circle cx="29" cy="33" r="6"/><circle cx="91" cy="33" r="6"/>'
    '<path d="M43 59a17 17 0 1 0 34 0a17 17 0 1 0 -34 0z M53 59a7 7 0 1 1 14 0a7 7 0 1 1 -14 0z"/>'
    '<rect x="39" y="84" width="6" height="17"/><circle cx="42" cy="84" r="5.5"/>'
    '<rect x="75" y="84" width="6" height="17"/><circle cx="78" cy="84" r="5.5"/>'
    '<rect x="19" y="98" width="82" height="6"/><circle cx="19" cy="101" r="6"/><circle cx="101" cy="101" r="6"/>'
)
SOLID = (
    '<rect x="49" y="13" width="22" height="10"/><rect x="24" y="28" width="72" height="10"/>'
    '<path d="M43 59a17 17 0 1 0 34 0a17 17 0 1 0 -34 0z M53 59a7 7 0 1 1 14 0a7 7 0 1 1 -14 0z"/>'
    '<rect x="37" y="81" width="10" height="16"/><rect x="73" y="81" width="10" height="16"/>'
    '<rect x="14" y="96" width="92" height="10"/>'
)
CLEAR = (-10, -11, 752, 122)
TIGHT = (0, -1, 732, 102)
SQUARE = (0, 0, 120, 120)


def seal(fill, glyph=NETWORK, carve=WHITE, uid="s"):
    if carve is None:
        return (
            f'<defs><mask id="{uid}"><rect width="120" height="120" fill="#fff"/><g fill="#000">{glyph}</g></mask></defs>'
            f'<rect width="120" height="120" rx="10" fill="{fill}" mask="url(#{uid})"/>'
        )
    return f'<rect width="120" height="120" rx="10" fill="{fill}"/><g fill="{carve}">{glyph}</g>'


def doc(box, body):
    x, y, w, h = box
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}" width="{w}" height="{h}">{body}</svg>\n'


def logo(box, type_fill, seal_fill, carve):
    return doc(box, f'<g fill="{type_fill}">{HYOR}</g><g transform="translate(632 0) scale(0.83333)">{seal(seal_fill, carve=carve)}</g>')


SVGS = {
    BRAND / "logo/hyor-ai-logo-color.svg": logo(CLEAR, INK, RED, WHITE),
    BRAND / "logo/hyor-ai-logo-reversed.svg": logo(CLEAR, WHITE, RED, WHITE),
    BRAND / "logo/hyor-ai-logo-black.svg": logo(CLEAR, BLACK, BLACK, None),
    BRAND / "logo/hyor-ai-logo-white.svg": logo(CLEAR, WHITE, WHITE, None),
    BRAND / "seal/hyor-seal.svg": doc(SQUARE, seal(RED)),
    BRAND / "seal/hyor-seal-black.svg": doc(SQUARE, seal(BLACK, carve=None)),
    BRAND / "seal/hyor-seal-white.svg": doc(SQUARE, seal(WHITE, carve=None)),
    BRAND / "favicon/favicon.svg": doc(SQUARE, seal(RED, glyph=SOLID)),
    SITE / "hyor-ai-logo.svg": logo(TIGHT, INK, RED, WHITE),
    SITE / "hyor-ai-logo-reversed.svg": logo(TIGHT, WHITE, RED, WHITE),
    SITE / "hyor-seal.svg": doc(SQUARE, seal(RED)),
    SITE / "favicon.svg": doc(SQUARE, seal(RED, glyph=SOLID)),
    COMPANY / "hyor-ai-logo.svg": logo(TIGHT, INK, RED, WHITE),
    COMPANY / "hyor-ai-logo-reversed.svg": logo(TIGHT, WHITE, RED, WHITE),
    COMPANY / "hyor-seal.svg": doc(SQUARE, seal(RED)),
    COMPANY / "favicon.svg": doc(SQUARE, seal(RED, glyph=SOLID)),
}

PNGS = [
    ("logo/hyor-ai-logo-color.svg", "logo/hyor-ai-logo-color.png", 2400),
    ("logo/hyor-ai-logo-reversed.svg", "logo/hyor-ai-logo-reversed.png", 2400),
    ("logo/hyor-ai-logo-black.svg", "logo/hyor-ai-logo-black.png", 2400),
    ("logo/hyor-ai-logo-white.svg", "logo/hyor-ai-logo-white.png", 2400),
    ("seal/hyor-seal.svg", "seal/hyor-seal-1024.png", 1024),
    ("seal/hyor-seal.svg", "seal/hyor-seal-512.png", 512),
    ("seal/hyor-seal.svg", "seal/apple-touch-icon-180.png", 180),
    ("seal/hyor-seal-black.svg", "seal/hyor-seal-black-1024.png", 1024),
    ("seal/hyor-seal-white.svg", "seal/hyor-seal-white-1024.png", 1024),
    ("favicon/favicon.svg", "favicon/favicon-48.png", 48),
    ("favicon/favicon.svg", "favicon/favicon-32.png", 32),
    ("favicon/favicon.svg", "favicon/favicon-16.png", 16),
]


def render(src, dst, width, work):
    box = SVGS[src].split('viewBox="')[1].split('"')[0].split()
    height = round(width * float(box[3]) / float(box[2]))
    page = Path(work) / "render.html"
    page.write_text(
        "<html><head><style>html,body{margin:0;background:transparent}</style></head>"
        f'<body><img src="{src.as_uri()}" style="display:block;width:{width}px;height:{height}px"></body></html>'
    )
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--default-background-color=00000000",
         "--force-device-scale-factor=1", f"--window-size={width},{height}", f"--screenshot={dst}", page.as_uri()],
        check=True, capture_output=True,
    )


def ico(path, sizes):
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for size in sizes:
        blob = (BRAND / f"favicon/favicon-{size}.png").read_bytes()
        entries += struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(blob), offset + len(data))
        data += blob
    path.write_bytes(head + entries + data)


def main():
    for path, text in SVGS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    with tempfile.TemporaryDirectory() as work:
        for src, dst, width in PNGS:
            render(BRAND / src, BRAND / dst, width, work)
    ico(BRAND / "favicon/favicon.ico", (16, 32, 48))
    for site in (SITE, COMPANY):
        ico(site / "favicon.ico", (16, 32, 48))
        (site / "apple-touch-icon.png").write_bytes((BRAND / "seal/apple-touch-icon-180.png").read_bytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
