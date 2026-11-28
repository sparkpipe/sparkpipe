import os
import re

TOKEN_SPEC = [
    ("SKIP", r"[ \t\r\n]+"),
    ("COMMENT_LINE", r"//[^\n]*"),
    ("COMMENT_BLOCK", r"/\*[\s\S]*?\*/"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("FLOAT", r"\d+\.\d+(?:[eE][+-]?\d+)?f?|\d+[eE][+-]?\d+f?"),
    ("HEX", r"0[xX][0-9a-fA-F]+u?"),
    ("INT", r"\d+u?"),
    ("DOTDOT", r"\.\."),
    ("FATARROW", r"=>"),
    ("ARROW", r"->"),
    ("COLON2", r"::"),
    ("OP2", r"<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|&=|\|="),
    ("OP1", r"[+\-*/%&|^~!<>?:=.,;()\[\]{}@]"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
]
TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))
KEYWORDS = {
    "import", "use", "const", "codec", "epilogue", "concept", "primitive",
    "variant", "policy", "when", "default", "macro", "template",
    "instantiate", "as", "model", "extends", "override", "fuse", "kernel",
    "private", "global", "shared", "ureg", "reg", "if", "else", "constexpr",
    "for", "pipeline", "unroll", "return", "true", "false",
}
TYPES = {"u8", "u16", "u32", "i32", "u64", "f32", "f32x2", "f32x4", "u32x2", "u32x4", "u32x8"}


class Skl2CompileError(Exception):
    def __init__(self, code, message, file="<input>", line=1):
        self.code, self.file, self.line, self.raw_message = code, file, line, message
        super().__init__(f"{file}:{line}: error[{code}]: {message}")


class Token:
    __slots__ = ("kind", "val", "line")
    def __init__(self, kind, val, line):
        self.kind, self.val, self.line = kind, val, line


def tokenize(src, file="<input>"):
    tokens, line, pos, n = [], 1, 0, len(src)
    while pos < n:
        m = TOKEN_RE.match(src, pos)
        if not m:
            snippet = src[pos:pos + 24].replace("\n", "\\n")
            raise Skl2CompileError("E_SYNTAX", f"unexpected token: {snippet!r}", file, line)
        kind, val, tok_line = m.lastgroup, m.group(), line
        line += val.count("\n")
        pos = m.end()
        if kind in ("SKIP", "COMMENT_LINE", "COMMENT_BLOCK"):
            continue
        if kind == "IDENT":
            kind = "KW" if val in KEYWORDS else ("TYPE" if val in TYPES else "IDENT")
        tokens.append(Token(kind, val, tok_line))
    tokens.append(Token("EOF", "", line))
    return tokens


def _parse_versioned_ref(p):
    name = p.eat().val
    while p.match("OP1", ".") or p.match("COLON2"):
        name = f"{name}.{p.eat().val}"
    return name, (p.eat().val if p.match("OP1", "@") else "latest")


def _parse_kv_value(p):
    t = p.cur()
    if t.kind == "STRING": return p.eat("STRING").val[1:-1]
    if t.kind == "KW" and t.val in ("true", "false"): return p.eat("KW").val == "true"
    if t.kind == "OP1" and t.val == "[":
        p.eat("OP1", "[")
        items = []
        if not p.is_tok("OP1", "]"):
            while True:
                items.append(_parse_kv_value(p))
                if not p.match("OP1", ","): break
        p.eat("OP1", "]")
        return items
    if t.kind == "OP1" and t.val == "{": return _parse_kv_block(p)
    if t.kind in ("INT", "HEX", "FLOAT") or (t.kind == "OP1" and t.val == "-"):
        return eval_meta_expr(p.parse_expr(), {})
    if t.kind in ("IDENT", "TYPE", "KW"):
        name = p.eat().val
        while p.match("OP1", ".") or p.match("COLON2"):
            name = f"{name}.{p.eat().val}"
        if p.match("OP1", "("):
            args = []
            if not p.is_tok("OP1", ")"):
                while True:
                    args.append(_parse_kv_value(p))
                    if not p.match("OP1", ","): break
            p.eat("OP1", ")")
            return {"call": name, "args": args}
        return name
    raise Skl2CompileError("E_SYNTAX", f"unexpected value token {t.val!r}", p.file, t.line)


def _parse_policy_block(p):
    p.eat("OP1", "{")
    rules, default_v = [], None
    while not p.is_tok("OP1", "}"):
        if p.match("KW", "when"):
            cond = p.parse_expr()
            p.eat("FATARROW")
            target = p.eat().val
            p.eat("OP1", ";")
            rules.append({"cond": cond, "target": target})
        elif p.match("KW", "default"):
            p.eat("FATARROW")
            default_v = p.eat().val
            p.eat("OP1", ";")
        else:
            raise Skl2CompileError("E_SYNTAX", f"expected 'when' or 'default' in policy, got {p.cur().val!r}", p.file, p.cur().line)
    p.eat("OP1", "}")
    return {"rules": rules, "default": default_v}


def _parse_kv_block(p):
    p.eat("OP1", "{")
    out = {}
    while not p.is_tok("OP1", "}"):
        if p.match("KW", "use"):
            ref, ver = _parse_versioned_ref(p)
            p.eat("OP1", ";")
            out.setdefault("uses", []).append({"ref": ref, "version": ver})
        elif p.match("KW", "variant"):
            vname = p.eat().val
            out.setdefault("variants", {})[vname] = _parse_kv_block(p)
        elif p.match("KW", "policy"):
            out["policy"] = _parse_policy_block(p)
        elif p.match("KW", "fuse"):
            p.eat("OP1", "{")
            fusions = out.setdefault("fusions", [])
            while not p.is_tok("OP1", "}"):
                lhs = [p.eat().val]
                while p.match("OP1", "+"): lhs.append(p.eat().val)
                p.eat("ARROW")
                rhs = p.eat().val
                p.eat("OP1", ";")
                fusions.append((lhs, rhs))
            p.eat("OP1", "}")
        elif p.match("KW", "override"):
            out.setdefault("overrides", {}).update(_parse_kv_block(p))
        else:
            k = p.eat().val
            if p.match("OP1", "="):
                out[k] = _parse_kv_value(p)
                p.eat("OP1", ";")
            elif p.is_tok("OP1", "{"):
                out[k] = _parse_kv_block(p)
            else:
                raise Skl2CompileError("E_SYNTAX", f"expected '=' or '{{' after {k!r}", p.file, p.cur().line)
    p.eat("OP1", "}")
    return out


class Parser:
    PREC = [
        ("||",), ("&&",), ("|",), ("^",), ("&",),
        ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"), ("*", "/"),
    ]

    def __init__(self, tokens, file="<input>"):
        self.tokens, self.file, self.idx = tokens, file, 0

    def cur(self): return self.tokens[self.idx]
    def is_tok(self, kind, val=None):
        t = self.cur()
        return t.kind == kind and (val is None or t.val == val)

    def eat(self, kind=None, val=None):
        t = self.cur()
        if kind is not None and t.kind != kind:
            raise Skl2CompileError("E_SYNTAX", f"expected {kind}, got {t.kind} ({t.val!r})", self.file, t.line)
        if val is not None and t.val != val:
            raise Skl2CompileError("E_SYNTAX", f"expected {val!r}, got {t.val!r}", self.file, t.line)
        self.idx += 1
        return t

    def match(self, kind, val=None):
        if not self.is_tok(kind, val): return False
        self.idx += 1
        return True

    def parse_file(self):
        unit = {
            "file": self.file, "imports": [], "uses": [], "consts": {}, "codecs": {},
            "epilogues": {}, "concepts": {}, "primitives": {}, "macros": {},
            "templates": {}, "instantiations": [], "kernels": [], "models": {},
        }
        while self.cur().kind != "EOF":
            attrs, t = self.parse_attrs(), self.cur()
            if t.kind != "KW":
                raise Skl2CompileError("E_SYNTAX", f"unexpected top-level token {t.val!r}", self.file, t.line)
            kw = t.val
            if kw == "import":
                self.eat("KW", "import")
                unit["imports"].append((self.eat("STRING").val[1:-1], t.line))
                self.eat("OP1", ";")
            elif kw == "use":
                self.eat("KW", "use")
                ref, ver = _parse_versioned_ref(self)
                self.eat("OP1", ";")
                unit["uses"].append({"ref": ref, "version": ver, "file": self.file, "line": t.line})
            elif kw == "const":
                self.eat("KW", "const")
                name = self.eat("IDENT").val
                self.eat("OP1", "=")
                unit["consts"][name] = (self.parse_expr(), self.file, t.line)
                self.eat("OP1", ";")
            elif kw in ("codec", "epilogue", "concept"):
                self.eat("KW", kw)
                name = self.eat().val
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit[kw + "s"][name] = body
            elif kw == "primitive":
                self.eat("KW", "primitive")
                pname, pver = _parse_versioned_ref(self)
                pver = "v1" if pver == "latest" else pver
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit["primitives"].setdefault(pname, {})[pver] = {"name": pname, "version": pver, "spec": body, "file": self.file, "line": t.line}
            elif kw == "model":
                self.eat("KW", "model")
                name, mver = _parse_versioned_ref(self)
                mver = "v1" if mver == "latest" else mver
                parent, parent_ver = _parse_versioned_ref(self) if self.match("KW", "extends") else (None, "latest")
                unit["models"].setdefault(name, {})[mver] = {
                    "name": name, "version": mver, "extends": parent, "extends_version": parent_ver,
                    "spec": _parse_kv_block(self), "file": self.file, "line": t.line,
                }
            elif kw == "macro":
                m = self.parse_macro(attrs)
                unit["macros"][m["name"]] = m
            elif kw == "template":
                self.eat("KW", "template")
                k = self.parse_kernel(attrs, is_template=True)
                unit["templates"][k["name"]] = k
            elif kw == "instantiate":
                unit["instantiations"].append(self.parse_instantiate())
            elif kw in ("kernel", "private"):
                is_priv = self.match("KW", "private")
                unit["kernels"].append(self.parse_kernel(attrs, is_template=False, is_private=is_priv))
            else:
                raise Skl2CompileError("E_SYNTAX", f"unexpected top-level token {kw!r}", self.file, t.line)
        return unit

    def parse_attrs(self):
        attrs = {}
        while self.match("OP1", "@"):
            name = self.eat().val
            if self.match("OP1", "("):
                args = []
                if not self.is_tok("OP1", ")"):
                    while True:
                        args.append(self.eat().val)
                        if not self.match("OP1", ","): break
                self.eat("OP1", ")")
                attrs[name] = args
            else:
                attrs[name] = True
        return attrs

    def parse_macro(self, attrs):
        t = self.eat("KW", "macro")
        name = self.eat("IDENT").val
        self.eat("OP1", "(")
        params = []
        if not self.is_tok("OP1", ")"):
            while True:
                params.append(self.eat("IDENT").val)
                if not self.match("OP1", ","): break
        self.eat("OP1", ")")
        return {"name": name, "params": params, "body": self.parse_block(), "attrs": attrs, "file": self.file, "line": t.line}

    def parse_instantiate(self):
        t = self.eat("KW", "instantiate")
        tpl, targs = self.eat("IDENT").val, []
        if self.match("OP1", "<"):
            if not self.is_tok("OP1", ">"):
                while True:
                    targs.append(self.parse_bin(7))
                    if not self.match("OP1", ","): break
            self.eat("OP1", ">")
        self.eat("KW", "as")
        alias = self.eat("IDENT").val
        self.eat("OP1", ";")
        return {"template": tpl, "args": targs, "name": alias, "file": self.file, "line": t.line}

    def parse_kernel(self, attrs, is_template=False, is_private=False):
        t = self.eat("KW", "kernel")
        self.eat("OP1", "[")
        block_expr = self.parse_expr()
        self.eat("OP1", ",")
        regs_expr = self.parse_expr()
        self.eat("OP1", "]")
        name, tparams = self.eat("IDENT").val, []
        if is_template:
            self.eat("OP1", "<")
            while True:
                tparams.append(self.eat("IDENT").val)
                if not self.match("OP1", ","): break
            self.eat("OP1", ">")
        self.eat("OP1", "(")
        params = []
        if not self.is_tok("OP1", ")"):
            while True:
                cond_expr = None
                if self.match("KW", "if"):
                    self.eat("OP1", "(")
                    cond_expr = self.parse_expr()
                    self.eat("OP1", ")")
                if self.is_tok("IDENT", "blob"):
                    self.eat("IDENT", "blob")
                    self.eat("OP1", "[")
                    bbytes = self.parse_expr()
                    self.eat("OP1", ",")
                    balign = self.parse_expr()
                    self.eat("OP1", "]")
                    params.append({"name": self.eat("IDENT").val, "type": "blob", "is_ptr": False, "blob_bytes": bbytes, "blob_align": balign, "cond": cond_expr})
                else:
                    is_global = self.match("KW", "global")
                    ptype, is_ptr, pname = self.eat("TYPE").val, self.match("OP1", "*"), self.eat("IDENT").val
                    params.append({"name": pname, "type": ptype, "is_ptr": is_ptr or is_global, "cond": cond_expr})
                if not self.match("OP1", ","): break
        self.eat("OP1", ")")
        return {
            "name": name, "tparams": tparams, "block": block_expr, "regs": regs_expr, "params": params,
            "body": self.parse_block(), "attrs": attrs, "private": is_private, "file": self.file, "line": t.line,
        }

    def parse_block(self):
        self.eat("OP1", "{")
        stmts = []
        while not self.is_tok("OP1", "}"):
            stmts.append(self.parse_stmt())
        self.eat("OP1", "}")
        return stmts

    def parse_stmt(self):
        t, ln = self.cur(), self.cur().line
        if self.match("KW", "shared"):
            vtype, name = self.eat("TYPE").val, self.eat("IDENT").val
            self.eat("OP1", "[")
            sz = self.parse_expr()
            self.eat("OP1", "]")
            self.eat("OP1", ";")
            return ("shared", vtype, name, sz, ln)
        if t.kind == "KW" and t.val in ("reg", "ureg"):
            storage, vtype, name = self.eat("KW").val, self.eat("TYPE").val, self.eat("IDENT").val
            if self.match("OP1", "["):
                sz = self.parse_expr()
                self.eat("OP1", "]")
                self.eat("OP1", ";")
                return ("reg_array", storage, vtype, name, sz, ln)
            init = self.parse_expr() if self.match("OP1", "=") else None
            self.eat("OP1", ";")
            return ("decl", storage, vtype, name, init, ln)
        if self.match("KW", "return"):
            self.eat("OP1", ";")
            return ("return", ln)
        if self.is_tok("KW", "if"): return self.parse_if()
        if self.match("KW", "unroll"):
            self.eat("OP1", "(")
            var = self.eat("IDENT").val
            self.eat("OP1", "=")
            s = self.parse_expr()
            self.eat("DOTDOT")
            e = self.parse_expr()
            self.eat("OP1", ")")
            return ("unroll", var, s, e, self.parse_block(), ln)
        if t.kind == "KW" and t.val in ("for", "pipeline"):
            is_pipe = self.match("KW", "pipeline")
            self.eat("KW", "for")
            self.eat("OP1", "(")
            storage = self.eat("KW").val if (self.cur().kind == "KW" and self.cur().val in ("reg", "ureg")) else "reg"
            vtype, var = self.eat("TYPE").val, self.eat("IDENT").val
            self.eat("OP1", "=")
            init = self.parse_expr()
            self.eat("OP1", ";")
            cond = self.parse_expr()
            self.eat("OP1", ";")
            lhs, op, rhs = self.parse_expr(), self.eat().val, self.parse_expr()
            self.eat("OP1", ")")
            return ("pipeline_for" if is_pipe else "for", storage, vtype, var, init, cond, ("assign", op, lhs, rhs, ln), self.parse_block(), ln)
        lhs = self.parse_expr()
        if self.cur().kind in ("OP1", "OP2") and self.cur().val in ("=", "+=", "-=", "*=", "&=", "|="):
            op, rhs = self.eat().val, self.parse_expr()
            self.eat("OP1", ";")
            return ("assign", op, lhs, rhs, ln)
        self.eat("OP1", ";")
        return ("expr", lhs, ln)

    def parse_if(self):
        t, is_ce = self.eat("KW", "if"), self.match("KW", "constexpr")
        self.eat("OP1", "(")
        cond = self.parse_expr()
        self.eat("OP1", ")")
        then_b = self.parse_block()
        else_b = ([self.parse_if()] if self.is_tok("KW", "if") else self.parse_block()) if self.match("KW", "else") else []
        return ("if_constexpr" if is_ce else "if", cond, then_b, else_b, t.line)

    def parse_expr(self):
        c = self.parse_bin(0)
        if self.match("OP1", "?"):
            t = self.parse_expr()
            self.eat("OP1", ":")
            return ("ternary", c, t, self.parse_expr())
        return c

    def parse_bin(self, lvl):
        if lvl >= len(self.PREC): return self.parse_unary()
        node, ops = self.parse_bin(lvl + 1), self.PREC[lvl]
        while self.cur().kind in ("OP1", "OP2") and self.cur().val in ops:
            node = ("bin", self.eat().val, node, self.parse_bin(lvl + 1))
        return node

    def parse_unary(self):
        if self.cur().kind == "OP1" and self.cur().val in ("-", "!", "~"):
            return ("un", self.eat().val, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.match("OP1", "("):
                args = []
                if not self.is_tok("OP1", ")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.match("OP1", ","): break
                self.eat("OP1", ")")
                if node[0] != "ident":
                    raise Skl2CompileError("E_SYNTAX", "call target must be identifier", self.file, self.cur().line)
                node = ("call", node[1], args)
            elif self.match("OP1", "["):
                idx = self.parse_expr()
                self.eat("OP1", "]")
                node = ("index", node, idx)
            elif self.match("OP1", "."):
                node = ("field", node, self.eat().val)
            else:
                return node

    def parse_primary(self):
        t = self.cur()
        if t.kind == "INT": self.eat(); return ("int", int(t.val.rstrip("u")))
        if t.kind == "HEX": self.eat(); return ("int", int(t.val.rstrip("u"), 16))
        if t.kind == "FLOAT": self.eat(); return ("float", float(t.val.rstrip("f")))
        if t.kind == "STRING": self.eat(); return ("str", t.val[1:-1])
        if t.kind == "KW" and t.val in ("true", "false"): self.eat(); return ("int", 1 if t.val == "true" else 0)
        if t.kind == "IDENT": self.eat(); return ("ident", t.val)
        if self.match("OP1", "("):
            e = self.parse_expr()
            self.eat("OP1", ")")
            return e
        raise Skl2CompileError("E_SYNTAX", f"unexpected token in expression: {t.val!r}", self.file, t.line)


BIN_META = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if isinstance(a, float) or isinstance(b, float) else a // b,
    "<<": lambda a, b: int(a) << int(b), ">>": lambda a, b: int(a) >> int(b),
    "&": lambda a, b: int(a) & int(b), "|": lambda a, b: int(a) | int(b), "^": lambda a, b: int(a) ^ int(b),
    "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b), "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b), ">=": lambda a, b: int(a >= b),
    "&&": lambda a, b: int(bool(a and b)), "||": lambda a, b: int(bool(a or b)),
}


def eval_meta_expr(expr, env):
    k = expr[0]
    if k in ("int", "float", "str"): return expr[1]
    if k == "ident":
        if expr[1] in env: return env[expr[1]]
        raise ValueError(f"not const: {expr[1]}")
    if k == "field":
        base = eval_meta_expr(expr[1], env)
        if isinstance(base, dict) and expr[2] in base: return base[expr[2]]
        raise ValueError(f"missing field {expr[2]}")
    if k == "un":
        v = eval_meta_expr(expr[2], env)
        return {"-": -v, "~": ~int(v), "!": 0 if v else 1}[expr[1]]
    if k == "bin":
        return BIN_META[expr[1]](eval_meta_expr(expr[2], env), eval_meta_expr(expr[3], env))
    if k == "ternary":
        return eval_meta_expr(expr[2] if eval_meta_expr(expr[1], env) else expr[3], env)
    raise ValueError(f"non-const expr: {k}")


def try_eval_meta(expr, env):
    try: return eval_meta_expr(expr, env)
    except Exception: return None


def load_unit(paths, search_roots=()):
    merged = {
        "files": [], "uses": [], "consts": {}, "codecs": {}, "epilogues": {}, "concepts": {},
        "primitives": {}, "macros": {}, "templates": {}, "instantiations": [], "kernels": [], "models": {},
    }
    visited = set()

    def _load(path, from_dir=None, from_file="<cli>", from_line=1):
        candidates = ([os.path.join(from_dir, path)] if from_dir else []) + [path] + [os.path.join(r, path) for r in search_roots]
        resolved = next((os.path.abspath(c) for c in candidates if os.path.isfile(c)), None)
        if resolved is None:
            raise Skl2CompileError("E_IMPORT", f"cannot resolve import {path!r}", from_file, from_line)
        if resolved in visited: return
        visited.add(resolved)
        with open(resolved, "r", encoding="utf-8") as f:
            u = Parser(tokenize(f.read(), resolved), resolved).parse_file()
        for imp_path, imp_line in u["imports"]:
            _load(imp_path, os.path.dirname(resolved), resolved, imp_line)
        merged["files"].append(resolved)
        merged["uses"].extend(u["uses"])
        for sec in ("consts", "codecs", "epilogues", "concepts", "macros", "templates"):
            merged[sec].update(u[sec])
        for sec in ("primitives", "models"):
            for k, vers in u[sec].items():
                merged[sec].setdefault(k, {}).update(vers)
        merged["instantiations"].extend(u["instantiations"])
        merged["kernels"].extend(u["kernels"])

    for p in paths: _load(p)
    return merged


def parse_skl2_text(src, file="<input>"):
    u = Parser(tokenize(src, file), file).parse_file()
    u["files"] = [file]
    return u
