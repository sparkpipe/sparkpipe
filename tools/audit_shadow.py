#!/usr/bin/env python3
"""audit_shadow.py — silent-override detector for SparkPipe.

Detects module-private copies that silently shadow common implementations:
  class 1  duplicate-signature copies of common-header functions
  class 2  header-name shadowing (include-path binds a family-local header)
  class 3  feature drift between a private copy and its common counterpart
  class 4  link-resolution duplicates (nm over built objects)
  class 5  #ifndef guard violations, silent #ifndef/env defaults

Usage:
  python3 tools/audit_shadow.py [--repo PATH] [--json PATH] [--report PATH]
      [--gate] [--allowlist PATH] [--objects-glob PATTERN]

Gate mode exits 2 when any class 1/2/3 finding is absent from the allowlist.
Allowlist lines: <class>|<repo-relative-path>|<symbol-or-include>
"""

import argparse
import datetime
import fnmatch
import json
import os
import re
import subprocess
import sys

KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "else", "do",
    "defined", "static_assert",
}

EXCLUDED_SYMBOLS = {"main"}

FAMILY_BASES = [
    "qwen38_max", "qwen38max", "qwen38_27b", "qwen4_flash", "qwen38",
    "glm5_next", "glm53flash", "glm53", "glm52", "dsv41_flash",
    "dsv4", "gemma4", "muse_glimmer", "mimo25", "laguna", "hy4",
    "ling", "k3",
]

C_KEYWORDS_BODY = {
    "if", "for", "while", "switch", "return", "else", "do", "case",
    "break", "continue", "sizeof", "const", "static", "inline", "void",
    "unsigned", "signed", "struct", "enum", "union", "typedef", "extern",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "size_t", "float", "double", "char", "int",
    "long", "short", "volatile",
}

GUARD_RE = re.compile(r"SPARKPIPE_[A-Z0-9_]+_H\b")


def variant_tokens():
    variants = []
    for base in FAMILY_BASES:
        parts = base.split("_")
        snake = base
        upper = "_".join(p.upper() for p in parts)
        pascal = "".join(p[:1].upper() + p[1:] for p in parts)
        camel = pascal[:1].lower() + pascal[1:]
        for v in (snake, upper, pascal, camel):
            if v not in variants:
                variants.append(v)
    variants.sort(key=len, reverse=True)
    compiled = []
    for v in variants:
        if "_" in v or (v.islower() and len(v) > 2):
            rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")
        elif v == v.upper():
            compiled.append((re.compile(r"(?<![0-9A-Z_])" + re.escape(v) + r"(?![a-z0-9_])"), "<fam>"))
            rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])")
        else:
            rx = re.compile(r"(?<![0-9A-Z_])" + re.escape(v) + r"(?![a-z0-9_])")
        compiled.append((rx, "<fam>"))
    return compiled


FAMILY_RES = variant_tokens()


def normalize(text):
    for rx, repl in FAMILY_RES:
        text = rx.sub(repl, text)
    return text


def norm_name(name):
    n = normalize(name)
    if n == "<fam>" or len(n.replace("<fam>", "")) < 3:
        return ""
    return n


def weak_name(name):
    n = norm_name(name)
    if not n:
        return ""
    w = n.replace("<fam>", "")
    if len(w) < 5:
        return ""
    return w


def strip_comments_strings(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append("  ")
                i += 2
        elif c == '"' or c == "'":
            quote = c
            out.append(quote)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out.append("  ")
                    i += 2
                else:
                    out.append("\n" if text[i] == "\n" else " ")
                    i += 1
            if i < n:
                out.append(quote)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def line_of_offset(text, offset):
    return text.count("\n", 0, offset) + 1


def match_paren(text, open_idx):
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def match_brace(text, open_idx):
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


DEF_HEAD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def extract_definitions(clean, path):
    defs = []
    for m in DEF_HEAD_RE.finditer(clean):
        name = m.group(1)
        start = m.start()
        line_start = clean.rfind("\n", 0, start) + 1
        prefix = clean[line_start:start]
        if name in KEYWORDS:
            continue
        open_idx = m.end() - 1
        close_idx = match_paren(clean, open_idx)
        if close_idx < 0:
            continue
        after = clean[close_idx + 1:].lstrip()
        if not after.startswith("{"):
            continue
        body_open = close_idx + 1 + (len(clean[close_idx + 1:]) - len(after))
        body_close = match_brace(clean, body_open)
        if body_close < 0:
            continue
        kind = "function"
        if "define" in prefix and "#" in prefix:
            kind = "macro"
        if "typedef" in prefix:
            continue
        params = clean[open_idx + 1:close_idx]
        body = clean[body_open:body_close + 1]
        defs.append({
            "name": name,
            "kind": kind,
            "file": path,
            "line": line_of_offset(clean, line_start),
            "end_line": line_of_offset(clean, body_close),
            "params": params.strip(),
            "body": body,
        })
    return defs


def body_identifiers(body):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body))


def git_blame_last(repo, path, start, end):
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "blame", "-l", "--date", "short",
             "-L", "%d,%d" % (start, end), "--", path],
            capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return None
        newest = None
        for line in proc.stdout.splitlines():
            parts = line.split(" ", 1)
            sha = parts[0].lstrip("^")
            if sha.startswith("0000000"):
                continue
            md = re.search(r"(\d{4}-\d{2}-\d{2})", line)
            if md is None:
                continue
            date = datetime.date.fromisoformat(md.group(1))
            if newest is None or date > newest["date"]:
                newest = {"sha": sha, "date": date}
        return newest
    except Exception:
        return None


def scope_of(path):
    p = path.replace(os.sep, "/")
    if p.startswith("include/sparkpipe/") or p.startswith("src/") or p.startswith("runtime/"):
        return "common"
    if p.startswith("model-families/common/"):
        return "common"
    if p.startswith("model-families/"):
        return "family:" + p.split("/")[1]
    if p.startswith("modules/"):
        return "module:" + p.split("/")[1]
    if p.startswith("ring/"):
        return "ring"
    if p.startswith("node/"):
        return "node"
    if p.startswith("tools/"):
        return "tools"
    if p.startswith("inference/"):
        return "inference"
    return "other"


SOURCE_EXTS = (".h", ".c", ".cuh", ".cu", ".cc")


def to_rel(repo, path):
    return os.path.relpath(path, repo).replace(os.sep, "/")


def collect_sources(repo, scopes):
    sources = []
    for scope in scopes:
        base = os.path.join(repo, scope)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "runs", "cache")]
            for f in files:
                if f.endswith(SOURCE_EXTS):
                    sources.append(to_rel(repo, os.path.join(root, f)))
    return sorted(set(sources))


def load_defs(repo, files):
    all_defs = []
    for rel in files:
        try:
            with open(os.path.join(repo, rel), "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        clean = strip_comments_strings(raw)
        for d in extract_definitions(clean, rel):
            d["scope"] = scope_of(d["file"])
            all_defs.append(d)
    return all_defs


COMMON_SCOPES = ["include", "model-families/common", "runtime", "ring"]
PRIVATE_SCOPES = ["modules", "model-families", "tools", "node"]


def private_only(files):
    keep = []
    for f in files:
        s = scope_of(f)
        if s.startswith("family:") or s.startswith("module:") or s in ("tools", "node"):
            keep.append(f)
    return keep


def detector_duplicates(common_defs, private_defs):
    common_by_key = {}
    common_by_weak = {}
    for d in common_defs:
        if d["kind"] != "function":
            continue
        k = norm_name(d["name"])
        if not k:
            continue
        common_by_key.setdefault(k, []).append(d)
        w = weak_name(d["name"])
        if w:
            common_by_weak.setdefault(w, []).append(d)
    findings = []
    matched = set()
    for d in private_defs:
        if d["kind"] != "function":
            continue
        k = norm_name(d["name"])
        if not k or k in EXCLUDED_SYMBOLS:
            continue
        weak = weak_name(d["name"])
        if k in common_by_key:
            primary = min(common_by_key[k], key=lambda c: (c["file"], c["line"]))
            matched.add((d["file"], d["line"]))
            findings.append({
                "class": 1,
                "symbol": d["name"],
                "normalized": k,
                "match": "exact-normalized",
                "private": {"file": d["file"], "line": d["line"], "scope": d["scope"]},
                "common": {"file": primary["file"], "line": primary["line"], "scope": primary["scope"]},
                "common_alternates": [
                    {"file": c["file"], "line": c["line"]} for c in common_by_key[k] if c is not primary
                ],
            })
        elif weak and weak in common_by_weak:
            primary = min(common_by_weak[weak], key=lambda c: (c["file"], c["line"]))
            matched.add((d["file"], d["line"]))
            findings.append({
                "class": 1,
                "symbol": d["name"],
                "normalized": k,
                "match": "family-prefixed",
                "private": {"file": d["file"], "line": d["line"], "scope": d["scope"]},
                "common": {"file": primary["file"], "line": primary["line"], "scope": primary["scope"]},
                "common_alternates": [
                    {"file": c["file"], "line": c["line"]} for c in common_by_weak[weak] if c is not primary
                ],
            })
    findings.extend(detector_family_twins(private_defs, matched))
    return findings


def detector_family_twins(private_defs, matched):
    groups = {}
    for d in private_defs:
        if d["kind"] != "function" or (d["file"], d["line"]) in matched:
            continue
        k = norm_name(d["name"])
        if not k or k in EXCLUDED_SYMBOLS:
            continue
        if "<fam>" not in k:
            continue
        groups.setdefault(k, []).append(d)
    findings = []
    for k, defs in sorted(groups.items()):
        scopes = {d["scope"] for d in defs}
        if len(scopes) < 2:
            continue
        primary = min(defs, key=lambda d: (d["file"], d["line"]))
        findings.append({
            "class": 6,
            "symbol": primary["name"],
            "normalized": k,
            "private": {"file": primary["file"], "line": primary["line"], "scope": primary["scope"]},
            "common": None,
            "twins": [
                {"file": d["file"], "line": d["line"], "scope": d["scope"],
                 "symbol": d["name"], "end_line": d["end_line"]}
                for d in defs
            ],
        })
    return findings


def param_list(params):
    parts = []
    depth = 0
    cur = ""
    for ch in params:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [normalize(" ".join(p.split())) for p in parts if p.strip()]


def drift_fingerprint(common_def, private_def):
    cp = param_list(common_def["params"])
    pp = param_list(private_def["params"])
    missing = [p for p in cp if p not in pp]
    extra = [p for p in pp if p not in cp]
    ci = {normalize(t) for t in body_identifiers(common_def["body"])}
    pi = {normalize(t) for t in body_identifiers(private_def["body"])}
    ci -= C_KEYWORDS_BODY
    pi -= C_KEYWORDS_BODY
    common_only = sorted(t for t in ci - pi if len(t) > 3 and "<fam>" not in t)
    union = ci | pi
    similarity = (len(ci & pi) / len(union)) if union else 1.0
    return {
        "params_common": cp,
        "params_private": pp,
        "missing_params": missing,
        "extra_params": extra,
        "common_only_tokens": common_only[:12],
        "similarity": round(similarity, 3),
    }


def attach_drift(findings, defs_by_loc, repo):
    for f in findings:
        if f.get("class") == 6:
            twin_bodies = []
            for t in f["twins"]:
                d = defs_by_loc.get((t["file"], t["line"]))
                if d is None:
                    continue
                twin_bodies.append((t, d))
            if len(twin_bodies) >= 2:
                base_t, base_d = twin_bodies[0]
                base_consts = set(re.findall(r"[0-9]+\.[0-9]+f|[0-9]+u?ull|0x[0-9a-fA-F]+u?ll?", base_d["body"]))
                divergences = []
                for t, d in twin_bodies[1:]:
                    consts = set(re.findall(r"[0-9]+\.[0-9]+f|[0-9]+u?ull|0x[0-9a-fA-F]+u?ll?", d["body"]))
                    only_base = sorted(base_consts - consts)
                    only_other = sorted(consts - base_consts)
                    sim = drift_fingerprint(base_d, d)["similarity"]
                    if only_base or only_other:
                        divergences.append({
                            "file": t["file"],
                            "vs": t["symbol"],
                            "constants_only_in_first": only_base[:10],
                            "constants_only_in_this": only_other[:10],
                            "similarity": sim,
                        })
                f["twin_constant_divergences"] = divergences
            continue
        ckey = (f["common"]["file"], f["common"]["line"])
        pkey = (f["private"]["file"], f["private"]["line"])
        c = defs_by_loc.get(ckey)
        p = defs_by_loc.get(pkey)
        if c is None or p is None:
            continue
        f["drift"] = drift_fingerprint(c, p)
        cb = git_blame_last(repo, c["file"], c["line"], c["end_line"])
        pb = git_blame_last(repo, p["file"], p["line"], p["end_line"])
        f["common_last_change"] = {"sha": cb["sha"], "date": str(cb["date"])} if cb else None
        f["private_last_change"] = {"sha": pb["sha"], "date": str(pb["date"])} if pb else None
        if cb and pb:
            f["stale_days"] = (cb["date"] - pb["date"]).days
    return findings


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)


def build_header_index(repo):
    index = {}
    flat = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "runs", "cache")]
        for f in files:
            if f.endswith((".h", ".cuh")):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, repo).replace(os.sep, "/")
                flat.append(rel)
                index.setdefault(f, []).append(rel)
                tail = tuple(rel.split("/")[-2:])
                index.setdefault("/".join(tail), []).append(rel)
    return index, flat


def parse_include_dirs(repo, module_dir):
    mk = os.path.join(repo, module_dir, "Makefile")
    dirs = []
    if os.path.isfile(mk):
        with open(mk, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in re.finditer(r"-I(\S+)", text):
            d = m.group(1)
            d = d.replace("../../", "")
            if d.startswith("/"):
                continue
            dirs.append(d)
    seen = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return seen


def detector_header_shadowing(repo):
    index, flat = build_header_index(repo)
    findings = []
    module_dirs = sorted(
        os.path.join("modules", d) for d in os.listdir(os.path.join(repo, "modules"))
        if os.path.isdir(os.path.join(repo, "modules", d)))
    module_inc_dirs = {m: parse_include_dirs(repo, m) for m in module_dirs}
    scan_files = collect_sources(
        repo, ["modules", "model-families", "include", "runtime", "ring", "tools", "node", "inference"])
    for src in scan_files:
        try:
            with open(os.path.join(repo, src), "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        module_dir = None
        for m in module_dirs:
            if src.startswith(m + "/"):
                module_dir = m
                break
        for m in INCLUDE_RE.finditer(text):
            inc = m.group(1)
            line = line_of_offset(text, m.start())
            cands = []
            for rel_path in index.get(inc, []):
                if rel_path != src and rel_path.endswith("/" + inc):
                    cands.append(rel_path)
            if not cands and "/" in inc:
                cands = [r for r in flat if r != src and r.endswith("/" + inc)]
            if not cands and "/" not in inc:
                base = os.path.basename(inc)
                for rel_path in index.get(base, []):
                    if rel_path != src:
                        cands.append(rel_path)
            cands = sorted(set(cands))
            if len(cands) < 2:
                continue
            cand_scopes = [scope_of(c) for c in cands]
            family_scopes = {s for s in cand_scopes if s.startswith("family:")}
            common_cands = [c for c, s in zip(cands, cand_scopes) if s == "common"]
            local_cands = [c for c, s in zip(cands, cand_scopes) if s != "common"]
            winner = None
            src_dir = os.path.dirname(src)
            if os.path.join(src_dir, inc).replace(os.sep, "/") in cands:
                winner = os.path.join(src_dir, inc).replace(os.sep, "/")
            elif module_dir:
                for d in module_inc_dirs[module_dir]:
                    for c in cands:
                        if c.startswith(d + "/") and (inc in c or "/" not in inc):
                            winner = c
                            break
                    if winner:
                        break
            if not winner:
                winner = cands[0]
            common_shadow = bool(common_cands) and winner not in common_cands
            cross_family = len(family_scopes) > 1
            if not common_shadow and not cross_family:
                continue
            findings.append({
                "class": 2,
                "include": inc,
                "site": {"file": src, "line": line, "scope": scope_of(src)},
                "winner": winner,
                "winner_basis": "quote-dir" if winner == os.path.join(src_dir, inc).replace(os.sep, "/") else ("-I order" if module_dir else "alphabetical"),
                "shadowed_common": common_cands,
                "candidates": cands,
                "kind": "common-shadowed" if common_shadow else "cross-family",
                "module": module_dir,
            })
    seen = set()
    unique = []
    for f in findings:
        key = (f["site"]["file"], f["site"]["line"], f["winner"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def detector_guards_and_env(repo):
    findings = []
    for rel in collect_sources(repo, COMMON_SCOPES + PRIVATE_SCOPES + ["inference", "src", "ring"]):
        if not rel.endswith((".h", ".cuh")):
            continue
        try:
            with open(os.path.join(repo, rel), "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "#pragma once" in text:
            continue
        head = text[:512]
        if re.search(r"^\s*#\s*ifndef\s+SPARKPIPE_\w+_H\b", head, re.M) and \
                re.search(r"^\s*#\s*define\s+SPARKPIPE_\w+_H\b", head, re.M):
            findings.append({
                "class": 5,
                "kind": "header-guard",
                "file": rel,
                "line": 1,
                "detail": "ifndef guard instead of #pragma once",
            })
    env_seen = set()
    for rel in collect_sources(repo, COMMON_SCOPES + PRIVATE_SCOPES + ["inference", "src"]):
        try:
            with open(os.path.join(repo, rel), "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for m in re.finditer(r'getenv\(\s*("([A-Z0-9_]+)"|([A-Za-z_][A-Za-z0-9_]*))\s*\)', text):
            if m.group(2):
                var = m.group(2)
                indirect = False
            else:
                var = m.group(3)
                indirect = True
            stmt_end = text.find(";", m.end())
            stmt = text[m.start():stmt_end if stmt_end > 0 else m.end() + 200]
            has_default = re.search(r":\s*[^:;]+|else|null", stmt, re.I)
            silent = has_default is not None and not re.search(r"warn|printf|fprintf|log", stmt, re.I)
            line = line_of_offset(text, m.start())
            key = (rel, var, line)
            if key in env_seen:
                continue
            env_seen.add(key)
            findings.append({
                "class": 5,
                "kind": "env-default",
                "file": rel,
                "line": line,
                "var": var,
                "indirect": indirect,
                "silent": silent,
                "statement": " ".join(stmt.split())[:200],
            })
    return findings


def detector_link(repo, objects_glob):
    findings = []
    patterns = []
    if objects_glob:
        patterns.append(objects_glob)
    else:
        patterns.extend(["build/**/*.o", "build/**/*.a", "build/**/*.so"])
    objects = []
    for pat in patterns:
        objects.extend(collect_glob(repo, pat))
    objects = sorted(set(objects))
    if not objects:
        return findings, False
    sym_defs = {}
    for obj in objects:
        proc = subprocess.run(["nm", "-A", obj], capture_output=True, text=True, timeout=120)
        for line in proc.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) < 3:
                continue
            sym_type = parts[1]
            sym = parts[2].strip()
            if sym_type not in ("T", "W", "D") or not sym:
                continue
            rel = os.path.relpath(obj, repo).replace(os.sep, "/")
            sym_defs.setdefault(sym, set()).add(rel)
    for sym, objs in sorted(sym_defs.items()):
        scopes = {}
        for o in objs:
            scopes[scope_of(o)] = o
        common = [f for s, f in sorted(scopes.items()) if s == "common"]
        private = [f for s, f in sorted(scopes.items())
                   if s.startswith("module:") or s.startswith("family:") or s in ("tools", "node")]
        if common and private:
            findings.append({
                "class": 4,
                "symbol": sym,
                "common_objects": common,
                "private_objects": private,
            })
    return findings, True


def collect_glob(repo, pattern):
    import glob
    hits = glob.glob(os.path.join(repo, pattern), recursive=True)
    return [h for h in hits if os.path.isfile(h)]


RISK_TOKENS = [
    (["abort", "cancel", "deadline", "timeout"], "hang class (missing abort/deadline path)"),
    (["slot", "tail", "parity", "doorbell", "publish"], "deadlock class (missing publish tail/stamp)"),
    (["diag", "telemetry", "epoch", "seq"], "undebuggable class (missing diag/epoch)"),
    (["format", "stamp", "layout"], "data-corruption class (format stamp drift)"),
]

SUSPICIOUS_TOKENS = (
    "abort", "cancel", "deadline", "timeout", "diag", "telemetry",
    "epoch", "seq", "stamp", "guard", "error_word", "slot", "tail",
    "parity", "doorbell", "format",
)


def is_suspicious_drift(drift):
    for p in drift.get("missing_params", []):
        low = p.lower()
        if any(t in low for t in SUSPICIOUS_TOKENS):
            return True
    for t in drift.get("common_only_tokens", []):
        low = t.lower()
        if any(s in low for s in SUSPICIOUS_TOKENS):
            return True
    return drift.get("similarity", 1.0) < 0.5


def risk_of(f):
    tags = []
    text = json.dumps(f.get("drift", {})) + f.get("symbol", "") + f.get("include", "")
    low = text.lower()
    for tokens, label in RISK_TOKENS:
        if any(t in low for t in tokens):
            tags.append(label)
    if f.get("drift", {}).get("missing_params"):
        tags.append("stale-signature class (older parameter list)")
    return tags


def allowlist_load(path):
    entries = set()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    entries.add((parts[0].strip(), parts[1].strip(),
                                 parts[2].strip() if len(parts) > 2 else ""))
    return entries


def allowlist_key(f):
    if f["class"] == 1:
        return ("1", f["private"]["file"], f["symbol"])
    if f["class"] == 2:
        return ("2", f["site"]["file"], f["include"])
    if f["class"] == 3:
        return ("3", f["private"]["file"], f["symbol"])
    if f["class"] == 4:
        return ("4", (f.get("private_objects") or ["?"])[0], f["symbol"])
    if f["class"] == 6:
        return ("6", f["private"]["file"], f["symbol"])
    return ("5", f.get("file", ""), f.get("kind", ""))


def main():
    ap = argparse.ArgumentParser(prog="audit_shadow")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--json")
    ap.add_argument("--report")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--allowlist", default=None)
    ap.add_argument("--objects-glob", default=None)
    ap.add_argument("--no-blame", action="store_true")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    common_files = collect_sources(repo, ["include", "model-families/common"])
    private_files = private_only(collect_sources(
        repo, ["modules", "model-families", "tools", "node"]))
    common_defs = load_defs(repo, common_files)
    private_defs = load_defs(repo, private_files)
    defs_by_loc = {}
    for d in common_defs + private_defs:
        defs_by_loc[(d["file"], d["line"])] = d

    dup = detector_duplicates(common_defs, private_defs)
    if not args.no_blame:
        dup = attach_drift(dup, defs_by_loc, repo)
    for f in dup:
        f["risk"] = risk_of(f)
        if f.get("drift") and is_suspicious_drift(f["drift"]):
            f["class"] = 3
    shadow = detector_header_shadowing(repo)
    guards_env = detector_guards_and_env(repo)
    link, built = detector_link(repo, args.objects_glob)

    findings = dup + shadow + link + guards_env
    counts = {}
    for f in findings:
        counts[f["class"]] = counts.get(f["class"], 0) + 1

    allow = allowlist_load(args.allowlist or os.path.join(repo, "tools", "audit_shadow_allowlist.txt"))
    gate_fail = [f for f in findings
                 if f["class"] in (1, 2, 3, 6) and allowlist_key(f) not in allow]

    payload = {
        "repo": repo,
        "generated": str(datetime.date.today()),
        "common_defs": len(common_defs),
        "private_defs": len(private_defs),
        "link_check_built": built,
        "counts": counts,
        "gate_failures": len(gate_fail),
        "findings": findings,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
    if args.report:
        write_report(args.report, payload, gate_fail)
    print("audit_shadow: class1=%d class2=%d class3=%d class4=%d class5=%d "
          "class6=%d gate_failures=%d link_built=%s" % (
              counts.get(1, 0), counts.get(2, 0), counts.get(3, 0),
              counts.get(4, 0), counts.get(5, 0), counts.get(6, 0),
              len(gate_fail), built))
    if args.gate and gate_fail:
        for f in gate_fail[:40]:
            print("GATE %s|%s|%s" % (allowlist_key(f)[0], allowlist_key(f)[1],
                                     allowlist_key(f)[2]))
        sys.exit(2)


def write_report(path, payload, gate_fail):
    lines = []
    lines.append("# audit-shadow findings (generated %s)" % payload["generated"])
    lines.append("")
    lines.append("common defs scanned: %d; private defs scanned: %d; "
                 "link check ran over objects: %s" % (
                     payload["common_defs"], payload["private_defs"],
                     payload["link_check_built"]))
    lines.append("")
    lines.append("counts by class: %s" % json.dumps(payload["counts"], sort_keys=True))
    lines.append("")
    for f in payload["findings"]:
        c = f["class"]
        if c == 1:
            lines.append("C1 %s:%s %s <- common %s:%s stale=%s risk=%s" % (
                f["private"]["file"], f["private"]["line"], f["symbol"],
                f["common"]["file"], f["common"]["line"],
                f.get("stale_days"), ",".join(f.get("risk", []))))
        elif c == 2:
            lines.append("C2 %s:%s include <%s> binds %s shadowing %s" % (
                f["site"]["file"], f["site"]["line"], f["include"],
                f["winner"], ",".join(f["shadowed_common"])))
        elif c == 3:
            d = f.get("drift", {})
            lines.append("C3 %s:%s %s missing=%s common_only=%s sim=%s" % (
                f["private"]["file"], f["private"]["line"], f["symbol"],
                d.get("missing_params"), d.get("common_only_tokens"),
                d.get("similarity")))
        elif c == 6:
            for div in f.get("twin_constant_divergences", []):
                lines.append("C6 twin %s:%s %s vs %s:%s const_first=%s const_this=%s sim=%s" % (
                    f["private"]["file"], f["private"]["line"], f["symbol"],
                    div["file"], div.get("vs", "?"),
                    div.get("constants_only_in_first"),
                    div.get("constants_only_in_this"), div.get("similarity")))
            if not f.get("twin_constant_divergences"):
                lines.append("C6 twin %s:%s %s scopes=%s identical-constants" % (
                    f["private"]["file"], f["private"]["line"], f["symbol"],
                    ",".join(sorted({t["scope"] for t in f["twins"]}))))
        elif c == 4:
            lines.append("C4 symbol %s defined in common %s AND private %s" % (
                f["symbol"], f["common_objects"], f["private_objects"]))
        else:
            if f["kind"] == "header-guard":
                lines.append("C5 guard %s" % f["file"])
            else:
                lines.append("C5 env %s:%s %s silent=%s stmt=%s" % (
                    f["file"], f["line"], f.get("var"), f.get("silent"),
                    f.get("statement", "")))
    lines.append("")
    lines.append("gate failures (class 1/2/3 not allowlisted): %d" % len(gate_fail))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
