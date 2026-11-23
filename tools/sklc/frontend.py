import re

TOKEN_SPEC = [
    ("SKIP", r"[ \t\r\n]+"),
    ("COMMENT_LINE", r"//[^\n]*"),
    ("COMMENT_BLOCK", r"/\*[\s\S]*?\*/"),
    ("FLOAT", r"\d+\.\d+(?:[eE][+-]?\d+)?f?|\d+[eE][+-]?\d+f?"),
    ("HEX", r"0[xX][0-9a-fA-F]+u?"),
    ("INT", r"\d+u?"),
    ("DOTDOT", r"\.\."),
    ("OP2", r"<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|&=|\|="),
    ("OP1", r"[+\-*/%&|^~!<>?:=.,;()\[\]{}]"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
]

TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))

KEYWORDS = {
    "const",
    "kernel",
    "global",
    "shared",
    "ureg",
    "reg",
    "if",
    "else",
    "for",
    "pipeline",
    "unroll",
    "return",
}

TYPES = {
    "u8",
    "u16",
    "u32",
    "i32",
    "u64",
    "f32",
    "f32x2",
    "f32x4",
    "u32x2",
    "u32x4",
    "u32x8",
}


class Token:
    def __init__(self, kind, val, line):
        self.kind = kind
        self.val = val
        self.line = line


def tokenize(src):
    tokens = []
    line = 1
    pos = 0
    n = len(src)
    while pos < n:
        m = TOKEN_RE.match(src, pos)
        if not m:
            snippet = src[pos:pos + 24].replace("\n", "\\n")
            raise SyntaxError(f"unexpected token at line {line}: {snippet!r}")
        kind = m.lastgroup
        val = m.group()
        line += val.count("\n")
        pos = m.end()
        if kind in ("SKIP", "COMMENT_LINE", "COMMENT_BLOCK"):
            continue
        if kind == "IDENT" and val in KEYWORDS:
            kind = "KW"
        elif kind == "IDENT" and val in TYPES:
            kind = "TYPE"
        tokens.append(Token(kind, val, line))
    tokens.append(Token("EOF", "", line))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.idx = 0

    def cur(self):
        return self.tokens[self.idx]

    def peek(self, off=1):
        p = self.idx + off
        if p < len(self.tokens):
            return self.tokens[p]
        return self.tokens[-1]

    def eat(self, kind=None, val=None):
        t = self.cur()
        if kind is not None and t.kind != kind:
            raise SyntaxError(f"line {t.line}: expected {kind}, got {t.kind} ({t.val!r})")
        if val is not None and t.val != val:
            raise SyntaxError(f"line {t.line}: expected {val!r}, got {t.val!r}")
        self.idx += 1
        return t

    def match(self, kind, val=None):
        t = self.cur()
        if t.kind != kind:
            return False
        if val is not None and t.val != val:
            return False
        self.idx += 1
        return True

    def parse_program(self):
        consts = {}
        kernels = []
        while self.cur().kind != "EOF":
            if self.cur().kind == "KW" and self.cur().val == "const":
                self.eat("KW", "const")
                name = self.eat("IDENT").val
                self.eat("OP1", "=")
                expr = self.parse_expr()
                self.eat("OP1", ";")
                consts[name] = eval_const_expr(expr, consts)
            elif self.cur().kind == "KW" and self.cur().val == "kernel":
                kernels.append(self.parse_kernel(consts))
            else:
                t = self.cur()
                raise SyntaxError(f"line {t.line}: unexpected top-level token {t.val!r}")
        return {"consts": consts, "kernels": kernels}

    def parse_kernel(self, consts):
        self.eat("KW", "kernel")
        self.eat("OP1", "[")
        block_expr = self.parse_expr()
        self.eat("OP1", ",")
        regs_expr = self.parse_expr()
        self.eat("OP1", "]")
        name = self.eat("IDENT").val
        self.eat("OP1", "(")
        params = []
        if not (self.cur().kind == "OP1" and self.cur().val == ")"):
            while True:
                is_global = self.match("KW", "global")
                ptype = self.eat("TYPE").val
                is_ptr = self.match("OP1", "*")
                pname = self.eat("IDENT").val
                params.append({
                    "name": pname,
                    "type": ptype,
                    "is_ptr": is_ptr or is_global,
                })
                if not self.match("OP1", ","):
                    break
        self.eat("OP1", ")")
        body = self.parse_block()
        return {
            "name": name,
            "block": int(eval_const_expr(block_expr, consts)),
            "regs": int(eval_const_expr(regs_expr, consts)),
            "params": params,
            "body": body,
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
        if t.kind == "KW" and t.val == "shared":
            self.eat("KW", "shared")
            vtype = self.eat("TYPE").val
            name = self.eat("IDENT").val
            self.eat("OP1", "[")
            size_expr = self.parse_expr()
            self.eat("OP1", "]")
            self.eat("OP1", ";")
            return ("shared", vtype, name, size_expr)
        if t.kind == "KW" and t.val in ("reg", "ureg"):
            storage = self.eat("KW").val
            vtype = self.eat("TYPE").val
            name = self.eat("IDENT").val
            init = None
            if self.match("OP1", "="):
                init = self.parse_expr()
            self.eat("OP1", ";")
            return ("decl", storage, vtype, name, init)
        if t.kind == "KW" and t.val == "return":
            self.eat("KW", "return")
            self.eat("OP1", ";")
            return ("return",)
        if t.kind == "KW" and t.val == "if":
            return self.parse_if()
        if t.kind == "KW" and t.val == "unroll":
            self.eat("KW", "unroll")
            self.eat("OP1", "(")
            var = self.eat("IDENT").val
            self.eat("OP1", "=")
            start = self.parse_expr()
            self.eat("DOTDOT")
            end = self.parse_expr()
            self.eat("OP1", ")")
            body = self.parse_block()
            return ("unroll", var, start, end, body)
        if t.kind == "KW" and t.val in ("for", "pipeline"):
            is_pipe = False
            if t.val == "pipeline":
                self.eat("KW", "pipeline")
                is_pipe = True
            self.eat("KW", "for")
            self.eat("OP1", "(")
            storage = "reg"
            if self.cur().kind == "KW" and self.cur().val in ("reg", "ureg"):
                storage = self.eat("KW").val
            vtype = self.eat("TYPE").val
            var = self.eat("IDENT").val
            self.eat("OP1", "=")
            init = self.parse_expr()
            self.eat("OP1", ";")
            cond = self.parse_expr()
            self.eat("OP1", ";")
            step = self.parse_for_step()
            self.eat("OP1", ")")
            body = self.parse_block()
            return ("pipeline_for" if is_pipe else "for", storage, vtype, var, init, cond, step, body)
        lhs = self.parse_expr()
        if self.cur().kind in ("OP1", "OP2") and self.cur().val in ("=", "+=", "-=", "*=", "&=", "|="):
            op = self.eat().val
            rhs = self.parse_expr()
            self.eat("OP1", ";")
            return ("assign", op, lhs, rhs)
        self.eat("OP1", ";")
        return ("expr", lhs)

    def parse_for_step(self):
        lhs = self.parse_expr()
        op = self.eat().val
        rhs = self.parse_expr()
        return ("assign", op, lhs, rhs)

    def parse_if(self):
        self.eat("KW", "if")
        self.eat("OP1", "(")
        cond = self.parse_expr()
        self.eat("OP1", ")")
        then_body = self.parse_block()
        else_body = []
        if self.match("KW", "else"):
            if self.cur().kind == "KW" and self.cur().val == "if":
                else_body = [self.parse_if()]
            else:
                else_body = self.parse_block()
        return ("if", cond, then_body, else_body)

    def parse_expr(self):
        return self.parse_ternary()

    def parse_ternary(self):
        cond = self.parse_or()
        if self.match("OP1", "?"):
            t_expr = self.parse_expr()
            self.eat("OP1", ":")
            f_expr = self.parse_ternary()
            return ("ternary", cond, t_expr, f_expr)
        return cond

    def parse_or(self):
        node = self.parse_and()
        while self.match("OP2", "||"):
            node = ("bin", "||", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_bitor()
        while self.match("OP2", "&&"):
            node = ("bin", "&&", node, self.parse_bitor())
        return node

    def parse_bitor(self):
        node = self.parse_bitxor()
        while self.match("OP1", "|"):
            node = ("bin", "|", node, self.parse_bitxor())
        return node

    def parse_bitxor(self):
        node = self.parse_bitand()
        while self.match("OP1", "^"):
            node = ("bin", "^", node, self.parse_bitand())
        return node

    def parse_bitand(self):
        node = self.parse_eq()
        while self.match("OP1", "&"):
            node = ("bin", "&", node, self.parse_eq())
        return node

    def parse_eq(self):
        node = self.parse_rel()
        while self.cur().kind == "OP2" and self.cur().val in ("==", "!="):
            op = self.eat().val
            node = ("bin", op, node, self.parse_rel())
        return node

    def parse_rel(self):
        node = self.parse_shift()
        while (self.cur().kind == "OP1" and self.cur().val in ("<", ">")) or (
            self.cur().kind == "OP2" and self.cur().val in ("<=", ">=")
        ):
            op = self.eat().val
            node = ("bin", op, node, self.parse_shift())
        return node

    def parse_shift(self):
        node = self.parse_add()
        while self.cur().kind == "OP2" and self.cur().val in ("<<", ">>"):
            op = self.eat().val
            node = ("bin", op, node, self.parse_add())
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.cur().kind == "OP1" and self.cur().val in ("+", "-"):
            op = self.eat().val
            node = ("bin", op, node, self.parse_mul())
        return node

    def parse_mul(self):
        node = self.parse_unary()
        while self.cur().kind == "OP1" and self.cur().val in ("*", "/"):
            op = self.eat().val
            node = ("bin", op, node, self.parse_unary())
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
                    raise SyntaxError("call target must be identifier")
                node = ("call", node[1], args)
            elif self.match("OP1", "["):
                idx = self.parse_expr()
                self.eat("OP1", "]")
                node = ("index", node, idx)
            elif self.match("OP1", "."):
                field = self.eat("IDENT").val
                node = ("field", node, field)
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
        if t.kind == "IDENT":
            self.eat()
            return ("ident", t.val)
        if self.match("OP1", "("):
            expr = self.parse_expr()
            self.eat("OP1", ")")
            return expr
        raise SyntaxError(f"line {t.line}: unexpected token in expression: {t.val!r}")


def try_eval_const(expr, env):
    try:
        return eval_const_expr(expr, env)
    except Exception:
        return None


def eval_const_expr(expr, env):
    kind = expr[0]
    if kind == "int":
        return expr[1]
    if kind == "float":
        return expr[1]
    if kind == "ident":
        if expr[1] in env:
            return env[expr[1]]
        raise ValueError(f"not a const: {expr[1]}")
    if kind == "un":
        v = eval_const_expr(expr[2], env)
        if expr[1] == "-":
            return -v
        if expr[1] == "~":
            return ~int(v)
        if expr[1] == "!":
            return 0 if v else 1
    if kind == "bin":
        op = expr[1]
        a = eval_const_expr(expr[2], env)
        b = eval_const_expr(expr[3], env)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if isinstance(a, float) or isinstance(b, float) else a // b
        if op == "<<":
            return int(a) << int(b)
        if op == ">>":
            return int(a) >> int(b)
        if op == "&":
            return int(a) & int(b)
        if op == "|":
            return int(a) | int(b)
        if op == "^":
            return int(a) ^ int(b)
        if op == "==":
            return 1 if a == b else 0
        if op == "!=":
            return 1 if a != b else 0
        if op == "<":
            return 1 if a < b else 0
        if op == "<=":
            return 1 if a <= b else 0
        if op == ">":
            return 1 if a > b else 0
        if op == ">=":
            return 1 if a >= b else 0
        if op == "&&":
            return 1 if (a and b) else 0
        if op == "||":
            return 1 if (a or b) else 0
    if kind == "ternary":
        c = eval_const_expr(expr[1], env)
        return eval_const_expr(expr[2] if c else expr[3], env)
    raise ValueError(f"non-const expr: {kind}")


def parse_skl(src):
    tokens = tokenize(src)
    return Parser(tokens).parse_program()
