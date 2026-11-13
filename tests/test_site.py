#!/usr/bin/env python3

import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
LANG_CONF = ROOT / "deployment" / "vps" / "sparkpipe-lang.conf"
COMPANY = ROOT / "deployment" / "vps" / "hyor.io"
COMPANY_CONF = ROOT / "deployment" / "vps" / "hyor-io.conf"
EXTERNAL = ("https://", "http://", "mailto:")
ALLOWED_STYLES = ("https://fonts.googleapis.com/",)
CHAT_APP_PAGES = {"fleet.html": {"/", "/static/favicon.svg", "/static/favicon.png"}}
SSL_CONF = ROOT / "deployment" / "vps" / "sparkpipe-ai-ssl.conf"
PLAIN_CONF = ROOT / "deployment" / "vps" / "sparkpipe-ai.conf"
CHAT_LOCAL_CONF = ROOT / "deployment" / "vps" / "sparkpipe-chat-local.conf"
TOSS_SDK = "https://js.tosspayments.com/v2/standard"
PAYMENT_PAGES = ("account.html", "buy.html")
CACHED = (".svg", ".png", ".ico")


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


def check_terms(failures):
    import importlib.util
    spec = importlib.util.spec_from_file_location("site_terms", SITE.parent / "tools" / "site_terms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, text in module.pages().items():
        path = SITE / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            failures.append(f"{name}: differs from site/terms/; run python3 tools/site_terms.py")


def twins():
    import re
    conf = LANG_CONF.read_text(encoding="utf-8")
    names = ["index"] if re.search(r"RewriteRule \^/\(\?:index\\\.html\)\?\$ /index-ko\.html ", conf) else []
    for group in re.findall(r"RewriteRule \^/\(([a-z|-]+)\)\\\.html\$ /\$1-ko\.html ", conf):
        names += group.split("|")
    return {f"{name}.html": f"{name}-ko.html" for name in names}


def check_languages(pages, failures):
    import re
    pairs = twins()
    if "index.html" not in pairs:
        failures.append("sparkpipe-lang.conf: the home page has no Korean redirect")
    korean = {name for name in pages if name.endswith("-ko.html") and 'http-equiv="refresh"' not in pages[name].source}
    for name in sorted(korean - set(pairs.values())):
        failures.append(f"{name}: Korean page without a redirect in {LANG_CONF.name}")
    for en, ko in pairs.items():
        for name in (en, ko):
            if name not in pages:
                failures.append(f"{LANG_CONF.name}: {name} names no file in site/")
                continue
            check_twin(name, pages[name].source, en, ko, "sparkpipe.ai", failures)


def check_twin(name, source, en, ko, host, failures):
    import re
    lang = "ko" if name == ko else "en"
    if f'<html lang="{lang}">' not in source:
        failures.append(f"{name}: html lang is not {lang}")
    for code, target in (("en", en), ("ko", ko)):
        if f'<link rel="alternate" hreflang="{code}" href="https://{host}/{target}">' not in source:
            failures.append(f"{name}: no hreflang {code} alternate for {target}")
    toggle = re.search(r'<div class="lang" role="group"[^>]*>(.*?)</div>', source, re.S)
    if not toggle:
        failures.append(f"{name}: no language switch")
        return
    links = re.findall(r'<a href="([^"]+)"([^>]*)>', toggle.group(1))
    if [href for href, _ in links] != [f"{en}?lang=en", f"{ko}?lang=ko"]:
        failures.append(f"{name}: language switch links {[href for href, _ in links]}")
    current = [href for href, attrs in links if 'aria-current="true"' in attrs]
    if current != [f"{name}?lang={lang}"]:
        failures.append(f"{name}: language switch marks {current} as current")


def check_links(name, page, pages, root, failures, skip=frozenset()):
    for link in page.links:
        target, _, fragment = link.partition("#")
        if target in skip:
            continue
        if target and not (root / target.lstrip("/")).is_file():
            failures.append(f"{name}: {link} names no file in {root.relative_to(ROOT)}/")
        elif fragment and fragment not in pages.get(target or name, page).ids:
            failures.append(f"{name}: {link} names no element")
    failures += [f"{name}: external script {src}" for src in page.scripts if src.startswith(EXTERNAL) and not (src == TOSS_SDK and root == SITE and name in PAYMENT_PAGES)]
    failures += [f"{name}: stylesheet {href}" for href in page.styles if href.startswith(EXTERNAL) and not href.startswith(ALLOWED_STYLES)]


def check_company(failures):
    pages = {path.name: parse(path) for path in sorted(COMPANY.glob("*.html"))}
    if sorted(pages) != ["index-ko.html", "index.html"]:
        failures.append(f"hyor.io: pages {sorted(pages)}, expected index.html and index-ko.html")
        return 0
    if "RewriteRule ^/(?:index\\.html)?$ /index-ko.html [R=302,L]" not in COMPANY_CONF.read_text(encoding="utf-8"):
        failures.append(f"{COMPANY_CONF.name}: the home page has no Korean redirect")
    for name, page in pages.items():
        check_twin(name, page.source, "index.html", "index-ko.html", "hyor.io", failures)
        check_links(name, page, pages, COMPANY, failures)
        failures += [f"{name}: script {src}, but the hyor.io policy allows none" for src in page.scripts]
        if "<script" in page.source:
            failures.append(f"{name}: inline script, but the hyor.io policy allows none")
    return len(pages)


def check_brand(failures):
    import importlib.util
    spec = importlib.util.spec_from_file_location("brand_logo", ROOT / "tools" / "brand_logo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path, text in module.SVGS.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            failures.append(f"{path.relative_to(ROOT)}: differs from tools/brand_logo.py; run python3 tools/brand_logo.py")


def check_kr_networks(failures):
    import importlib.util
    spec = importlib.util.spec_from_file_location("site_kr_networks", ROOT / "tools" / "site_kr_networks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = ["2|apnic|20261024|5|19830613|20261023|+1000",
             "apnic|KR|ipv4|1.11.0.0|65536|20100101|allocated",
             "apnic|KR|ipv4|27.100.0.128|128|20100101|assigned",
             "apnic|KR|ipv4|61.32.0.0|1024|20000101|available",
             "apnic|JP|ipv4|1.0.16.0|4096|20110412|allocated",
             "apnic|KR|ipv6|2001:220::|32|19990917|allocated"]
    keys = module.prefixes(lines)
    if len(keys) != 257 or keys[0] != "1.11.0" or keys[255] != "1.11.255" or keys[256] != "27.100.0":
        failures.append(f"site_kr_networks: wrong /24 networks {len(keys)} {keys[:1]} {keys[-2:]}")


def check_payment_pages(pages, failures):
    loading = sorted(name for name, page in pages.items() if TOSS_SDK in page.scripts)
    if loading != list(PAYMENT_PAGES):
        failures.append(f"the Toss Payments SDK loads on {loading}, but only the payment pages {list(PAYMENT_PAGES)} take payment")
    conf = SSL_CONF.read_text(encoding="utf-8")
    blocks = re.findall(r'<LocationMatch "\^/\(([a-z|-]+)\)\\\.html\$">(.*?)</LocationMatch>', conf, re.S)
    allowed = [(sorted(f"{name}.html" for name in names.split("|")), body) for names, body in blocks if "tosspayments" in body]
    if len(allowed) != 1 or allowed[0][0] != list(PAYMENT_PAGES) or "script-src 'self' 'unsafe-inline' https://*.tosspayments.com https://*.toss.im;" not in allowed[0][1]:
        failures.append(f"{SSL_CONF.name}: the Content-Security-Policy must let exactly the payment pages {list(PAYMENT_PAGES)} load the Toss Payments SDK")
    if "tosspayments" in re.sub(r"<LocationMatch.*?</LocationMatch>", "", conf, flags=re.S):
        failures.append(f"{SSL_CONF.name}: the Toss Payments hosts are allowed beyond the payment pages")


def check_fresh(failures):
    conf = SSL_CONF.read_text(encoding="utf-8")
    fresh = [re.compile(pattern) for pattern, body in re.findall(r'<FilesMatch "(.*?)">(.*?)</FilesMatch>', conf, re.S) if 'Header set Cache-Control "no-cache"' in body]
    stale = sorted(str(path.relative_to(SITE)) for path in SITE.rglob("*") if path.is_file() and path.suffix not in CACHED and not any(p.search(path.name) for p in fresh))
    if stale:
        failures.append(f"{SSL_CONF.name}: a browser may keep these after a deploy and run a page against old published settings: {stale}")


def check_transport(failures):
    plain = PLAIN_CONF.read_text(encoding="utf-8")
    redirect = "RewriteRule ^ https://sparkpipe.ai%{REQUEST_URI} [END,NE,R=permanent]"
    for line in ("<VirtualHost *:80>", "ServerName sparkpipe.ai", "ServerAlias www.sparkpipe.ai", "RewriteCond %{REQUEST_URI} !^/\\.well-known/acme-challenge/", redirect, "ServerTokens Prod", "ServerSignature Off"):
        if line not in plain:
            failures.append(f"{PLAIN_CONF.name}: port 80 must send sparkpipe.ai and www to HTTPS and hide the server version; missing {line!r}")
    if re.search(r"^\s*(Include|ProxyPass|Alias)\b", plain, re.M):
        failures.append(f"{PLAIN_CONF.name}: port 80 serves content instead of only redirecting")
    if f"sudo install -m 644 deployment/vps/{PLAIN_CONF.name} /etc/apache2/sites-available/000-default.conf" not in (ROOT / "docs" / "PAYMENTS.md").read_text(encoding="utf-8"):
        failures.append(f"docs/PAYMENTS.md: the install steps never install {PLAIN_CONF.name}")
    for conf, root in ((SSL_CONF, "/var/www/html"), (COMPANY_CONF, "/var/www/hyor"), (PLAIN_CONF, "/var/www/html")):
        text = conf.read_text(encoding="utf-8")
        if not re.search(rf"<Directory {re.escape(root)}>\s*Options -Indexes\s*</Directory>", text):
            failures.append(f"{conf.name}: {root} must not list directories")
        if conf is PLAIN_CONF:
            continue
        secure = re.findall(r"<VirtualHost \*:443>(.*?)</VirtualHost>", text, re.S)
        if not secure or any('Header always set Strict-Transport-Security "max-age=63072000; includeSubDomains"' not in body for body in secure):
            failures.append(f"{conf.name}: every HTTPS virtual host must send Strict-Transport-Security")
    local = CHAT_LOCAL_CONF.read_text(encoding="utf-8")
    block = re.search(r'<LocationMatch "\^/local\(/\|\$\)">(.*?)</LocationMatch>', local, re.S)
    body = block.group(1) if block else ""
    for header in ("Content-Security-Policy", "Permissions-Policy", "X-Frame-Options", "X-Content-Type-Options"):
        if f"Header always set {header} " not in body:
            failures.append(f"{CHAT_LOCAL_CONF.name}: /local/ is served without {header}")
    csp = re.search(r'Content-Security-Policy "([^"]*)"', body)
    for directive in ("default-src 'self'", "connect-src 'self'", "object-src 'none'", "frame-ancestors 'none'"):
        if not csp or directive not in csp.group(1).split("; "):
            failures.append(f"{CHAT_LOCAL_CONF.name}: the /local/ Content-Security-Policy lacks {directive}")


def check_mail(failures):
    doc = (ROOT / "docs" / "VPS_MAIL.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| `(/[^`]+)` \|", doc, re.M)
    installed = set(re.findall(r"^sudo install -m \d+ deployment/vps/(\S+) (\S+)$", doc, re.M))
    missing = [name for name, target in rows if (name, target) not in installed]
    if not rows or missing:
        failures.append(f"docs/VPS_MAIL.md: the deploy steps never install {missing}")
    for name, _ in rows:
        if not (ROOT / "deployment" / "vps" / name).is_file():
            failures.append(f"docs/VPS_MAIL.md: deployment/vps/{name} does not exist")
    main = (ROOT / "deployment" / "vps" / "postfix-main.cf").read_text(encoding="utf-8")
    if not re.search(r"^smtp_tls_security_level = may$", main, re.M):
        failures.append("postfix-main.cf: outbound mail must use opportunistic TLS, or sign-in codes never reach a mail server without STARTTLS")
    hook = (ROOT / "deployment" / "vps" / "certbot-reload-mail-services.sh").read_text(encoding="utf-8")
    if re.search(r"^set -[a-z]*e", hook, re.M) or "systemctl reload postfix || status=1" not in hook or "systemctl reload dovecot || status=1" not in hook:
        failures.append("certbot-reload-mail-services.sh: a failed Postfix reload must not skip the Dovecot reload")
    for domain in ("sparkpipe.ai", "hyor.io"):
        if f"| `{domain}` | TXT | `v=spf1 ip4:27.102.120.99 -all` |" not in doc or f"rua=mailto:dmarc@{domain}` |" not in doc:
            failures.append(f"docs/VPS_MAIL.md: the expected SPF and DMARC records of {domain} are not written down")


def check_models(failures):
    text = (SITE / "pricing.js").read_text()
    listed = [(m["name"], m["id"]) for m in json.loads(text[text.index("{"):text.rindex("}") + 1])["models"]]
    rows = re.findall(r'<tr><td><b>([^<]+)</b><br><span class="mono">([^<]+)</span></td>.*?<td>(Offered[^<]*|Not offered)</td></tr>', (SITE / "fleets.html").read_text())
    offered = [(name, ident) for name, ident, state in rows if state.startswith("Offered")]
    if not offered or listed != offered:
        failures.append(f"the models pricing.js lists differ from the Fleet List's offered models: {listed} {offered}")


def main():
    pages = {path.name: parse(path) for path in sorted(SITE.glob("*.html"))}
    failures = []
    check_terms(failures)
    check_languages(pages, failures)
    check_kr_networks(failures)
    check_brand(failures)
    check_models(failures)
    check_payment_pages(pages, failures)
    check_fresh(failures)
    check_transport(failures)
    check_mail(failures)
    for name, page in pages.items():
        check_links(name, page, pages, SITE, failures, CHAT_APP_PAGES.get(name, frozenset()))
        check_tables(name, page, failures)
    company = check_company(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures or not pages:
        return 1
    print(f"site: {len(pages)} pages, every local link and anchor resolves, no external scripts beyond the Toss Payments SDK on the payment pages and their Content-Security-Policy, the contract pages match site/terms/, every page with a Korean twin has the language switch and redirect, every page, script, style and contract text is revalidated on each load")
    print(f"hyor.io: {company} pages with the language switch and redirect, no scripts; the logo files match tools/brand_logo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
