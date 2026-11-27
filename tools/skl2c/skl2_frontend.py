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
        self.code = code
        self.file = file
        self.line = line
        self.raw_message = message
        super().__init__(f"{file}:{line}: error[{code}]: {message}")


class Token:
    __slots__ = ("kind", "val", "line")

    def __init__(self, kind, val, line):
        self.kind = kind
        self.val = val
        self.line = line


def tokenize(src, file="<input>"):
    tokens = []
    line = 1
    pos = 0
    n = len(src)
    while pos < n:
        m = TOKEN_RE.match(src, pos)
        if not m:
            snippet = src[pos:pos + 24].replace("\n", "\\n")
            raise Skl2CompileError("E_SYNTAX", f"unexpected token: {snippet!r}", file, line)
        kind = m.lastgroup
        val = m.group()
        tok_line = line
        line += val.count("\n")
        pos = m.end()
        if kind in ("SKIP", "COMMENT_LINE", "COMMENT_BLOCK"):
            continue
        if kind == "IDENT" and val in KEYWORDS:
            kind = "KW"
        elif kind == "IDENT" and val in TYPES:
            kind = "TYPE"
        tokens.append(Token(kind, val, tok_line))
    tokens.append(Token("EOF", "", line))
    return tokens


def _parse_kv_value(p):
    t = p.cur()
    if t.kind == "STRING":
        return p.eat("STRING").val[1:-1]
    if t.kind == "KW" and t.val in ("true", "false"):
        return p.eat("KW").val == "true"
    if t.kind == "OP1" and t.val == "[":
        p.eat("OP1", "[")
        items = []
        if not (p.cur().kind == "OP1" and p.cur().val == "]"):
            while True:
                items.append(_parse_kv_value(p))
                if not p.match("OP1", ","):
                    break
        p.eat("OP1", "]")
        return items
    if t.kind == "OP1" and t.val == "{":
        return _parse_kv_block(p)
    if t.kind in ("INT", "HEX", "FLOAT") or (t.kind == "OP1" and t.val == "-"):
        expr = p.parse_expr()
        return eval_meta_expr(expr, {})
    if t.kind in ("IDENT", "TYPE", "KW"):
        name = p.eat().val
        while p.match("OP1", ".") or p.match("COLON2"):
            name = f"{name}.{p.eat().val}"
        if p.match("OP1", "("):
            args = []
            if not (p.cur().kind == "OP1" and p.cur().val == ")"):
                while True:
                    args.append(_parse_kv_value(p))
                    if not p.match("OP1", ","):
                        break
            p.eat("OP1", ")")
            return {"call": name, "args": args}
        return name
    raise Skl2CompileError("E_SYNTAX", f"unexpected value token {t.val!r}", p.file, t.line)


def _parse_policy_block(p):
    p.eat("OP1", "{")
    rules = []
    default_v = None
    while not (p.cur().kind == "OP1" and p.cur().val == "}"):
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
    while not (p.cur().kind == "OP1" and p.cur().val == "}"):
        if p.cur().kind == "KW" and p.cur().val == "use":
            p.eat("KW", "use")
            ref, ver = _parse_versioned_ref(p)
            p.eat("OP1", ";")
            out.setdefault("uses", []).append({"ref": ref, "version": ver})
            continue
        if p.cur().kind == "KW" and p.cur().val == "variant":
            p.eat("KW", "variant")
            vname = p.eat().val
            out.setdefault("variants", {})[vname] = _parse_kv_block(p)
            continue
        if p.cur().kind == "KW" and p.cur().val == "policy":
            p.eat("KW", "policy")
            out["policy"] = _parse_policy_block(p)
            continue
        if p.cur().kind == "KW" and p.cur().val == "fuse":
            p.eat("KW", "fuse")
            p.eat("OP1", "{")
            fusions = out.setdefault("fusions", [])
            while not (p.cur().kind == "OP1" and p.cur().val == "}"):
                lhs = [p.eat().val]
                while p.match("OP1", "+"):
                    lhs.append(p.eat().val)
                p.eat("ARROW")
                rhs = p.eat().val
                p.eat("OP1", ";")
                fusions.append((lhs, rhs))
            p.eat("OP1", "}")
            continue
        if p.cur().kind == "KW" and p.cur().val == "override":
            p.eat("KW", "override")
            ov = _parse_kv_block(p)
            out.setdefault("overrides", {}).update(ov)
            continue
        k = p.eat().val
        if p.match("OP1", "="):
            out[k] = _parse_kv_value(p)
            p.eat("OP1", ";")
        elif p.cur().kind == "OP1" and p.cur().val == "{":
            out[k] = _parse_kv_block(p)
        else:
            raise Skl2CompileError("E_SYNTAX", f"expected '=' or '{{' after {k!r}", p.file, p.cur().line)
    p.eat("OP1", "}")
    return out


def _parse_versioned_ref(p):
    name = p.eat().val
    while p.match("OP1", ".") or p.match("COLON2"):
        name = f"{name}.{p.eat().val}"
    ver = "latest"
    if p.match("OP1", "@"):
        ver = p.eat().val
    return name, ver


class Parser:
    def __init__(self, tokens, file="<input>"):
        self.tokens = tokens
        self.file = file
        self.idx = 0

    def cur(self):
        return self.tokens[self.idx]

    def eat(self, kind=None, val=None):
        t = self.cur()
        if kind is not None and t.kind != kind:
            raise Skl2CompileError("E_SYNTAX", f"expected {kind}, got {t.kind} ({t.val!r})", self.file, t.line)
        if val is not None and t.val != val:
            raise Skl2CompileError("E_SYNTAX", f"expected {val!r}, got {t.val!r}", self.file, t.line)
        self.idx += 1
        return t

    def match(self, kind, val=None):
        t = self.cur()
        if t.kind != kind or (val is not None and t.val != val):
            return False
        self.idx += 1
        return True

    def parse_file(self):
        unit = {
            "file": self.file,
            "imports": [],
            "uses": [],
            "consts": {},
            "codecs": {},
            "epilogues": {},
            "concepts": {},
            "primitives": {},
            "macros": {},
            "templates": {},
            "instantiations": [],
            "kernels": [],
            "models": {},
        }
        while self.cur().kind != "EOF":
            attrs = self.parse_attrs()
            t = self.cur()
            if t.kind == "KW" and t.val == "import":
                self.eat("KW", "import")
                path = self.eat("STRING").val[1:-1]
                self.eat("OP1", ";")
                unit["imports"].append((path, t.line))
            elif t.kind == "KW" and t.val == "use":
                self.eat("KW", "use")
                ref, ver = _parse_versioned_ref(self)
                self.eat("OP1", ";")
                unit["uses"].append({"ref": ref, "version": ver, "file": self.file, "line": t.line})
            elif t.kind == "KW" and t.val == "const":
                self.eat("KW", "const")
                name = self.eat("IDENT").val
                self.eat("OP1", "=")
                expr = self.parse_expr()
                self.eat("OP1", ";")
                unit["consts"][name] = (expr, self.file, t.line)
            elif t.kind == "KW" and t.val == "codec":
                self.eat("KW", "codec")
                name = self.eat("IDENT").val
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit["codecs"][name] = body
            elif t.kind == "KW" and t.val == "epilogue":
                self.eat("KW", "epilogue")
                name = self.eat("IDENT").val
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit["epilogues"][name] = body
            elif t.kind == "KW" and t.val == "concept":
                self.eat("KW", "concept")
                name = self.eat().val
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit["concepts"][name] = body
            elif t.kind == "KW" and t.val == "primitive":
                self.eat("KW", "primitive")
                pname, pver = _parse_versioned_ref(self)
                if pver == "latest":
                    pver = "v1"
                body = _parse_kv_block(self)
                body["_file"], body["_line"] = self.file, t.line
                unit["primitives"].setdefault(pname, {})[pver] = {
                    "name": pname,
                    "version": pver,
                    "spec": body,
                    "file": self.file,
                    "line": t.line,
                }
            elif t.kind == "KW" and t.val == "model":
                self.eat("KW", "model")
                name, mver = _parse_versioned_ref(self)
                if mver == "latest":
                    mver = "v1"
                parent = None
                parent_ver = "latest"
                if self.match("KW", "extends"):
                    parent, parent_ver = _parse_versioned_ref(self)
                body = _parse_kv_block(self)
                unit["models"].setdefault(name, {})[mver] = {
                    "name": name,
                    "version": mver,
                    "extends": parent,
                    "extends_version": parent_ver,
                    "spec": body,
                    "file": self.file,
                    "line": t.line,
                }
            elif t.kind == "KW" and t.val == "macro":
                m = self.parse_macro(attrs)
                unit["macros"][m["name"]] = m
            elif t.kind == "KW" and t.val == "template":
                self.eat("KW", "template")
                k = self.parse_kernel(attrs, is_template=True)
                unit["templates"][k["name"]] = k
            elif t.kind == "KW" and t.val == "instantiate":
                unit["instantiations"].append(self.parse_instantiate())
            elif t.kind == "KW" and t.val in ("kernel", "private"):
                is_priv = self.match("KW", "private")
                k = self.parse_kernel(attrs, is_template=False, is_private=is_priv)
                unit["kernels"].append(k)
            else:
                raise Skl2CompileError("E_SYNTAX", f"unexpected top-level token {t.val!r}", self.file, t.line)
        return unit

    def parse_attrs(self):
        attrs = {}
        while self.match("OP1", "@"):
            name = self.eat().val
            if self.match("OP1", "("):
                args = []
                if not (self.cur().kind == "OP1" and self.cur().val == ")"):
                    while True:
                        args.append(self.eat().val)
                        if not self.match("OP1", ","):
                            break
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
        if not (self.cur().kind == "OP1" and self.cur().val == ")"):
            while True:
                params.append(self.eat("IDENT").val)
                if not self.match("OP1", ","):
                    break
        self.eat("OP1", ")")
        body = self.parse_block()
        return {
            "name": name,
            "params": params,
            "body": body,
            "attrs": attrs,
            "file": self.file,
            "line": t.line,
        }

    def parse_instantiate(self):
        t = self.eat("KW", "instantiate")
        tpl = self.eat("IDENT").val
        targs = []
        if self.match("OP1", "<"):
            if not (self.cur().kind == "OP1" and self.cur().val == ">"):
                while True:
                    targs.append(self.parse_bin(7))
                    if not self.match("OP1", ","):
                        break
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
        name = self.eat("IDENT").val
        tparams = []
        if is_template:
            self.eat("OP1", "<")
            while True:
                tparams.append(self.eat("IDENT").val)
                if not self.match("OP1", ","):
                    break
            self.eat("OP1", ">")
        self.eat("OP1", "(")
        params = []
        if not (self.cur().kind == "OP1" and self.cur().val == ")"):
            while True:
                cond_expr = None
                if self.match("KW", "if"):
                    self.eat("OP1", "(")
                    cond_expr = self.parse_expr()
                    self.eat("OP1", ")")
                if self.cur().kind == "IDENT" and self.cur().val == "blob":
                    self.eat("IDENT", "blob")
                    self.eat("OP1", "[")
                    bbytes = self.parse_expr()
                    self.eat("OP1", ",")
                    balign = self.parse_expr()
                    self.eat("OP1", "]")
                    pname = self.eat("IDENT").val
                    params.append({
                        "name": pname, "type": "blob", "is_ptr": False,
                        "blob_bytes": bbytes, "blob_align": balign, "cond": cond_expr,
                    })
                else:
                    is_global = self.match("KW", "global")
                    ptype = self.eat("TYPE").val
                    is_ptr = self.match("OP1", "*")
                    pname = self.eat("IDENT").val
                    params.append({
                        "name": pname, "type": ptype, "is_ptr": is_ptr or is_global, "cond": cond_expr,
                    })
                if not self.match("OP1", ","):
                    break
        self.eat("OP1", ")")
        body = self.parse_block()
        return {
            "name": name,
            "tparams": tparams,
            "block": block_expr,
            "regs": regs_expr,
            "params": params,
            "body": body,
            "attrs": attrs,
            "private": is_private,
            "file": self.file,
            "line": t.line,
        }

    def parse_block(self):
        self.eat("OP1", "{")
        stmts = []
        while not (self.cur().kind == "OP1" and self.cur().val == "}"):
            stmts.append(self.parse_stmt())
        self.eat("OP1", "}")
        return stmts

    def parse_stmt(self):
        t = self.cur()
        ln = t.line
        if t.kind == "KW" and t.val == "shared":
            self.eat("KW", "shared")
            vtype = self.eat("TYPE").val
            name = self.eat("IDENT").val
            self.eat("OP1", "[")
            sz = self.parse_expr()
            self.eat("OP1", "]")
            self.eat("OP1", ";")
            return ("shared", vtype, name, sz, ln)
        if t.kind == "KW" and t.val in ("reg", "ureg"):
            storage = self.eat("KW").val
            vtype = self.eat("TYPE").val
            name = self.eat("IDENT").val
            init = self.parse_expr() if self.match("OP1", "=") else None
            self.eat("OP1", ";")
            return ("decl", storage, vtype, name, init, ln)
        if t.kind == "KW" and t.val == "return":
            self.eat("KW", "return")
            self.eat("OP1", ";")
            return ("return", ln)
        if t.kind == "KW" and t.val == "if":
            return self.parse_if()
        if t.kind == "KW" and t.val == "unroll":
            self.eat("KW", "unroll")
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
            vtype = self.eat("TYPE").val
            var = self.eat("IDENT").val
            self.eat("OP1", "=")
            init = self.parse_expr()
            self.eat("OP1", ";")
            cond = self.parse_expr()
            self.eat("OP1", ";")
            lhs = self.parse_expr()
            op = self.eat().val
            rhs = self.parse_expr()
            self.eat("OP1", ")")
            return ("pipeline_for" if is_pipe else "for", storage, vtype, var, init, cond, ("assign", op, lhs, rhs, ln), self.parse_block(), ln)
        lhs = self.parse_expr()
        if self.cur().kind in ("OP1", "OP2") and self.cur().val in ("=", "+=", "-=", "*=", "&=", "|="):
            op = self.eat().val
            rhs = self.parse_expr()
            self.eat("OP1", ";")
            return ("assign", op, lhs, rhs, ln)
        self.eat("OP1", ";")
        return ("expr", lhs, ln)

    def parse_if(self):
        t = self.eat("KW", "if")
        is_ce = self.match("KW", "constexpr")
        self.eat("OP1", "(")
        cond = self.parse_expr()
        self.eat("OP1", ")")
        then_b = self.parse_block()
        else_b = []
        if self.match("KW", "else"):
            else_b = [self.parse_if()] if (self.cur().kind == "KW" and self.cur().val == "if") else self.parse_block()
        return ("if_constexpr" if is_ce else "if", cond, then_b, else_b, t.line)

    def parse_expr(self):
        return self.parse_ternary()

    def parse_ternary(self):
        c = self.parse_bin(0)
        if self.match("OP1", "?"):
            t = self.parse_expr()
            self.eat("OP1", ":")
            return ("ternary", c, t, self.parse_ternary())
        return c

    PREC = [
        ("||",), ("&&",), ("|",), ("^",), ("&",),
        ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"), ("*", "/"),
    ]

    def parse_bin(self, lvl):
        if lvl >= len(self.PREC):
            return self.parse_unary()
        node = self.parse_bin(lvl + 1)
        ops = self.PREC[lvl]
        while self.cur().kind in ("OP1", "OP2") and self.cur().val in ops:
            op = self.eat().val
            node = ("bin", op, node, self.parse_bin(lvl + 1))
        return node

    def parse_unary(self):
        if self.cur().kind == "OP1" and self.cur().val in ("-", "!", "~"):
            op = self.eat().val
            return ("un", op, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.match("OP1", "("):
                args = []
                if not (self.cur().kind == "OP1" and self.cur().val == ")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.match("OP1", ","):
                            break
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
                break
        return node

    def parse_primary(self):
        t = self.cur()
        if t.kind == "INT":
            self.eat()
            return ("int", int(t.val.rstrip("u")))
        if t.kind == "HEX":
            self.eat()
            return ("int", int(t.val.rstrip("u"), 16))
        if t.kind == "FLOAT":
            self.eat()
            return ("float", float(t.val.rstrip("f")))
        if t.kind == "STRING":
            self.eat()
            return ("str", t.val[1:-1])
        if t.kind == "KW" and t.val in ("true", "false"):
            self.eat()
            return ("int", 1 if t.val == "true" else 0)
        if t.kind == "IDENT":
            self.eat()
            return ("ident", t.val)
        if self.match("OP1", "("):
            e = self.parse_expr()
            self.eat("OP1", ")")
            return e
        raise Skl2CompileError("E_SYNTAX", f"unexpected token in expression: {t.val!r}", self.file, t.line)


def eval_meta_expr(expr, env):
    k = expr[0]
    if k in ("int", "float", "str"):
        return expr[1]
    if k == "ident":
        if expr[1] in env:
            return env[expr[1]]
        raise ValueError(f"not const: {expr[1]}")
    if k == "field":
        base = eval_meta_expr(expr[1], env)
        if isinstance(base, dict) and expr[2] in base:
            return base[expr[2]]
        raise ValueError(f"missing field {expr[2]}")
    if k == "un":
        v = eval_meta_expr(expr[2], env)
        return {"-": -v, "~": ~int(v), "!": 0 if v else 1}[expr[1]]
    if k == "bin":
        op, a, b = expr[1], eval_meta_expr(expr[2], env), eval_meta_expr(expr[3], env)
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b if isinstance(a, float) or isinstance(b, float) else a // b
        if op == "<<": return int(a) << int(b)
        if op == ">>": return int(a) >> int(b)
        if op == "&": return int(a) & int(b)
        if op == "|": return int(a) | int(b)
        if op == "^": return int(a) ^ int(b)
        if op == "==": return 1 if a == b else 0
        if op == "!=": return 1 if a != b else 0
        if op == "<": return 1 if a < b else 0
        if op == "<=": return 1 if a <= b else 0
        if op == ">": return 1 if a > b else 0
        if op == ">=": return 1 if a >= b else 0
        if op == "&&": return 1 if (a and b) else 0
        if op == "||": return 1 if (a or b) else 0
    if k == "ternary":
        return eval_meta_expr(expr[2] if eval_meta_expr(expr[1], env) else expr[3], env)
    raise ValueError(f"non-const expr: {k}")


def try_eval_meta(expr, env):
    try:
        return eval_meta_expr(expr, env)
    except Exception:
        return None


def load_unit(paths, search_roots=()):
    merged = {
        "files": [],
        "uses": [],
        "consts": {},
        "codecs": {},
        "epilogues": {},
        "concepts": {},
        "primitives": {},
        "macros": {},
        "templates": {},
        "instantiations": [],
        "kernels": [],
        "models": {},
    }
    visited = set()

    def _load(path, from_dir=None, from_file="<cli>", from_line=1):
        candidates = [path]
        if from_dir:
            candidates.insert(0, os.path.join(from_dir, path))
        for r in search_roots:
            candidates.append(os.path.join(r, path))
        resolved = None
        for c in candidates:
            if os.path.isfile(c):
                resolved = os.path.abspath(c)
                break
        if resolved is None:
            raise Skl2CompileError("E_IMPORT", f"cannot resolve import {path!r}", from_file, from_line)
        if resolved in visited:
            return
        visited.add(resolved)
        with open(resolved, "r", encoding="utf-8") as f:
            src = f.read()
        u = Parser(tokenize(src, resolved), resolved).parse_file()
        for imp_path, imp_line in u["imports"]:
            _load(imp_path, os.path.dirname(resolved), resolved, imp_line)
        merged["files"].append(resolved)
        merged["uses"].extend(u["uses"])
        for sec in ("consts", "codecs", "epilogues", "concepts", "macros", "templates"):
            merged[sec].update(u[sec])
        for pname, vers in u["primitives"].items():
            merged["primitives"].setdefault(pname, {}).update(vers)
        for mname, vers in u["models"].items():
            merged["models"].setdefault(mname, {}).update(vers)
        merged["instantiations"].extend(u["instantiations"])
        merged["kernels"].extend(u["kernels"])

    for p in paths:
        _load(p)
    return merged


def parse_skl2_text(src, file="<input>"):
    u = Parser(tokenize(src, file), file).parse_file()
    u["files"] = [file]
    return u
