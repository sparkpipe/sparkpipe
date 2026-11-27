import hashlib
from skl2_frontend import Skl2CompileError

MAX_CCN = 16
MAX_NESTING = 4
MAX_KERNEL_STMTS = 80
DUP_WINDOW = 10

STORE_CALLS = {
    "stg256", "stg128", "stg128_stream", "stg128_sys",
    "stg64", "stg32", "stg32_stream", "stg32_sys", "stg16", "stg8",
}
PDL_WAIT_CALLS = {"pdl_wait", "acqbulk"}
PDL_RELEASE_CALLS = {"pdl_release", "preexit"}


def _count_stmts(stmts):
    total = 0
    for st in stmts:
        total += 1
        k = st[0]
        if k in ("if", "if_constexpr"):
            total += _count_stmts(st[2]) + _count_stmts(st[3])
        elif k == "unroll":
            total += _count_stmts(st[4])
        elif k in ("for", "pipeline_for"):
            total += _count_stmts(st[7])
    return total


def _expr_ccn(expr):
    if not isinstance(expr, tuple):
        return 0
    k = expr[0]
    if k == "ternary":
        return 1 + _expr_ccn(expr[1]) + _expr_ccn(expr[2]) + _expr_ccn(expr[3])
    if k == "bin":
        add = 1 if expr[1] in ("&&", "||") else 0
        return add + _expr_ccn(expr[2]) + _expr_ccn(expr[3])
    if k == "un":
        return _expr_ccn(expr[2])
    if k == "call":
        return sum(_expr_ccn(a) for a in expr[2])
    if k == "index":
        return _expr_ccn(expr[1]) + _expr_ccn(expr[2])
    if k == "field":
        return _expr_ccn(expr[1])
    return 0


def _ccn_and_depth(stmts, depth=0):
    ccn = 0
    max_d = depth
    for st in stmts:
        k = st[0]
        if k in ("if", "if_constexpr"):
            ccn += 1 + _expr_ccn(st[1])
            c1, d1 = _ccn_and_depth(st[2], depth + 1)
            ccn += c1
            max_d = max(max_d, d1)
            if st[3]:
                if len(st[3]) == 1 and st[3][0][0] in ("if", "if_constexpr"):
                    c2, d2 = _ccn_and_depth(st[3], depth)
                else:
                    c2, d2 = _ccn_and_depth(st[3], depth + 1)
                ccn += c2
                max_d = max(max_d, d2)
        elif k in ("for", "pipeline_for"):
            ccn += 1 + _expr_ccn(st[5])
            c1, d1 = _ccn_and_depth(st[7], depth + 1)
            ccn += c1
            max_d = max(max_d, d1)
        elif k == "unroll":
            c1, d1 = _ccn_and_depth(st[4], depth + 1)
            ccn += c1
            max_d = max(max_d, d1)
        elif k == "decl" and st[4] is not None:
            ccn += _expr_ccn(st[4])
        elif k == "assign":
            ccn += _expr_ccn(st[2]) + _expr_ccn(st[3])
        elif k == "expr":
            ccn += _expr_ccn(st[1])
    return ccn, max_d


def _norm_expr(expr, vmap):
    if not isinstance(expr, tuple):
        return repr(expr)
    k = expr[0]
    if k in ("int", "float", "str"):
        return f"{k}:{expr[1]}"
    if k == "ident":
        nm = expr[1]
        if nm not in vmap:
            vmap[nm] = f"${len(vmap)}"
        return vmap[nm]
    if k == "un":
        return f"({expr[1]}{_norm_expr(expr[2], vmap)})"
    if k == "bin":
        return f"({_norm_expr(expr[2], vmap)}{expr[1]}{_norm_expr(expr[3], vmap)})"
    if k == "ternary":
        return f"({_norm_expr(expr[1], vmap)}?{_norm_expr(expr[2], vmap)}:{_norm_expr(expr[3], vmap)})"
    if k == "call":
        args = ",".join(_norm_expr(a, vmap) for a in expr[2])
        return f"{expr[1]}({args})"
    if k == "index":
        return f"{_norm_expr(expr[1], vmap)}[{_norm_expr(expr[2], vmap)}]"
    if k == "field":
        return f"{_norm_expr(expr[1], vmap)}.{expr[2]}"
    return repr(expr)


def _norm_stmt(st, vmap):
    k = st[0]
    if k == "shared":
        if st[2] not in vmap:
            vmap[st[2]] = f"${len(vmap)}"
        return f"shared {st[1]} {vmap[st[2]]}[{_norm_expr(st[3], vmap)}]"
    if k == "decl":
        init = _norm_expr(st[4], vmap) if st[4] is not None else ""
        if st[3] not in vmap:
            vmap[st[3]] = f"${len(vmap)}"
        return f"{st[1]} {st[2]} {vmap[st[3]]}={init}"
    if k == "return":
        return "ret"
    if k == "assign":
        return f"{_norm_expr(st[2], vmap)}{st[1]}{_norm_expr(st[3], vmap)}"
    if k == "expr":
        return _norm_expr(st[1], vmap)
    if k in ("if", "if_constexpr"):
        c = _norm_expr(st[1], vmap)
        t = ";".join(_norm_stmt(x, vmap) for x in st[2])
        e = ";".join(_norm_stmt(x, vmap) for x in st[3])
        return f"{k}({c}){{{t}}}else{{{e}}}"
    if k == "unroll":
        if st[1] not in vmap:
            vmap[st[1]] = f"${len(vmap)}"
        b = ";".join(_norm_stmt(x, vmap) for x in st[4])
        return f"unroll({vmap[st[1]]}={_norm_expr(st[2], vmap)}..{_norm_expr(st[3], vmap)}){{{b}}}"
    if k in ("for", "pipeline_for"):
        if st[3] not in vmap:
            vmap[st[3]] = f"${len(vmap)}"
        b = ";".join(_norm_stmt(x, vmap) for x in st[7])
        return f"{k}({st[2]} {vmap[st[3]]}={_norm_expr(st[4], vmap)};{_norm_expr(st[5], vmap)};{_norm_stmt(st[6], vmap)}){{{b}}}"
    return repr(st)


def _flat_stmts(stmts):
    out = []
    for st in stmts:
        out.append(st)
        k = st[0]
        if k in ("if", "if_constexpr"):
            out.extend(_flat_stmts(st[2]))
            out.extend(_flat_stmts(st[3]))
        elif k == "unroll":
            out.extend(_flat_stmts(st[4]))
        elif k in ("for", "pipeline_for"):
            out.extend(_flat_stmts(st[7]))
    return out


def _check_dup_subtrees(items):
    seen = {}
    for owner_name, file, stmts in items:
        flat = _flat_stmts(stmts)
        if len(flat) < DUP_WINDOW:
            continue
        for i in range(0, len(flat) - DUP_WINDOW + 1):
            window = flat[i:i + DUP_WINDOW]
            vmap = {}
            canon = "\n".join(_norm_stmt(s, vmap) for s in window)
            digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
            ln = window[0][-1]
            if digest in seen:
                p_owner, p_file, p_line, p_idx = seen[digest]
                if p_owner != owner_name or abs(i - p_idx) >= DUP_WINDOW:
                    raise Skl2CompileError(
                        "E_DUP_SUBTREE",
                        f"duplicated {DUP_WINDOW}-statement subtree in {owner_name} matches "
                        f"{p_owner} at {p_file}:{p_line}; extract a macro or template",
                        file,
                        ln,
                    )
            else:
                seen[digest] = (owner_name, file, ln, i)


def _collect_refs_and_calls(stmts, refs, calls):
    def walk_e(e):
        if not isinstance(e, tuple):
            return
        k = e[0]
        if k == "ident":
            refs.add(e[1])
        elif k == "call":
            calls.add(e[1])
            for a in e[2]:
                walk_e(a)
        elif k == "un":
            walk_e(e[2])
        elif k == "bin":
            walk_e(e[2])
            walk_e(e[3])
        elif k == "ternary":
            walk_e(e[1])
            walk_e(e[2])
            walk_e(e[3])
        elif k == "index":
            walk_e(e[1])
            walk_e(e[2])
        elif k == "field":
            walk_e(e[1])

    for st in stmts:
        k = st[0]
        if k == "shared":
            walk_e(st[3])
        elif k == "decl":
            if st[4] is not None:
                walk_e(st[4])
        elif k == "assign":
            if st[1] != "=":
                walk_e(st[2])
            elif st[2][0] != "ident":
                walk_e(st[2])
            walk_e(st[3])
        elif k == "expr":
            walk_e(st[1])
        elif k in ("if", "if_constexpr"):
            walk_e(st[1])
            _collect_refs_and_calls(st[2], refs, calls)
            _collect_refs_and_calls(st[3], refs, calls)
        elif k == "unroll":
            walk_e(st[2])
            walk_e(st[3])
            _collect_refs_and_calls(st[4], refs, calls)
        elif k in ("for", "pipeline_for"):
            walk_e(st[4])
            walk_e(st[5])
            _collect_refs_and_calls([st[6]], refs, calls)
            _collect_refs_and_calls(st[7], refs, calls)


def _check_dead_locals_and_unreachable(stmts, owner, file, macro_params=()):
    refs, _ = set(), set()
    _collect_refs_and_calls(stmts, refs, set())

    def walk(block):
        terminated = False
        for st in block:
            ln = st[-1]
            if terminated:
                raise Skl2CompileError("E_DEAD_CODE", f"unreachable statement after return in {owner}", file, ln)
            k = st[0]
            if k == "return":
                terminated = True
            elif k == "shared":
                if st[2] not in refs:
                    raise Skl2CompileError("E_DEAD_CODE", f"unused shared array {st[2]!r} in {owner}", file, ln)
            elif k == "decl":
                if st[3] not in refs and st[3] not in macro_params:
                    raise Skl2CompileError("E_DEAD_CODE", f"unused local register {st[3]!r} in {owner}", file, ln)
            elif k in ("if", "if_constexpr"):
                t_ret = walk(st[2])
                e_ret = walk(st[3]) if st[3] else False
                if t_ret and e_ret:
                    terminated = True
            elif k == "unroll":
                walk(st[4])
            elif k in ("for", "pipeline_for"):
                walk(st[7])
        return terminated

    walk(stmts)


def _expr_calls(expr, out):
    if not isinstance(expr, tuple):
        return
    k = expr[0]
    if k == "call":
        out.append(expr[1])
        for a in expr[2]:
            _expr_calls(a, out)
    elif k == "un":
        _expr_calls(expr[2], out)
    elif k == "bin":
        _expr_calls(expr[2], out)
        _expr_calls(expr[3], out)
    elif k == "ternary":
        _expr_calls(expr[1], out)
        _expr_calls(expr[2], out)
        _expr_calls(expr[3], out)
    elif k == "index":
        _expr_calls(expr[1], out)
        _expr_calls(expr[2], out)
    elif k == "field":
        _expr_calls(expr[1], out)


def _check_pdl(kernel, macros):
    is_pdl = bool(kernel.get("attrs", {}).get("pdl"))
    all_calls = set()
    _collect_refs_and_calls(kernel["body"], set(), all_calls)
    if not is_pdl and not (all_calls & (PDL_WAIT_CALLS | PDL_RELEASE_CALLS)):
        return
    file = kernel["file"]
    name = kernel["name"]

    def step_stmts(stmts, acq, rel):
        for st in stmts:
            ln = st[-1]
            k = st[0]
            if k == "expr" and st[1][0] == "call" and st[1][1] in macros:
                m = macros[st[1][1]]
                acq, rel, exited = step_stmts(m["body"], acq, rel)
                if exited:
                    return acq, rel, True
                continue
            calls = []
            if k == "expr":
                _expr_calls(st[1], calls)
            elif k == "decl" and st[4] is not None:
                _expr_calls(st[4], calls)
            elif k == "assign":
                _expr_calls(st[3], calls)
            for c in calls:
                if c in PDL_WAIT_CALLS:
                    acq = True
                elif c in PDL_RELEASE_CALLS:
                    rel = True
                elif c in STORE_CALLS and not acq:
                    raise Skl2CompileError(
                        "E_PDL_HAZARD",
                        f"PDL kernel {name} performs global write {c}() before acqbulk()/pdl_wait()",
                        file,
                        ln,
                    )
            if k == "return":
                if not rel:
                    raise Skl2CompileError(
                        "E_PDL_HAZARD",
                        f"PDL kernel {name} returns without calling preexit()/pdl_release()",
                        file,
                        ln,
                    )
                return acq, rel, True
            if k in ("if", "if_constexpr"):
                a1, r1, e1 = step_stmts(st[2], acq, rel)
                if st[3]:
                    a2, r2, e2 = step_stmts(st[3], acq, rel)
                    if e1 and e2:
                        return a1 and a2, r1 and r2, True
                    if e1:
                        acq, rel = a2, r2
                    elif e2:
                        acq, rel = a1, r1
                    else:
                        acq, rel = a1 and a2, r1 and r2
                else:
                    if not e1:
                        acq, rel = acq and a1, rel and r1
            elif k in ("unroll",):
                acq, rel, exited = step_stmts(st[4], acq, rel)
                if exited:
                    return acq, rel, True
            elif k in ("for", "pipeline_for"):
                step_stmts(st[7], acq, rel)
        return acq, rel, False

    _, rel_end, exited = step_stmts(kernel["body"], False, False)
    if not exited and not rel_end:
        raise Skl2CompileError(
            "E_PDL_HAZARD",
            f"PDL kernel {name} exits without calling preexit()/pdl_release()",
            file,
            kernel["line"],
        )


def check_anti_slop(unit, allow_unused_lib=False):
    dup_items = []
    all_calls = set()
    used_templates = {inst["template"] for inst in unit["instantiations"]}

    for mname, m in unit["macros"].items():
        ccn, depth = _ccn_and_depth(m["body"])
        ccn += 1
        if ccn > MAX_CCN or depth > MAX_NESTING:
            raise Skl2CompileError(
                "E_COMPLEXITY",
                f"macro {mname} has CCN={ccn} (max {MAX_CCN}) and nesting={depth} (max {MAX_NESTING})",
                m["file"],
                m["line"],
            )
        _check_dead_locals_and_unreachable(m["body"], f"macro {mname}", m["file"], set(m["params"]))
        dup_items.append((f"macro {mname}", m["file"], m["body"]))

    all_kobjs = list(unit["templates"].values()) + list(unit["kernels"])
    for k in all_kobjs:
        name = k["name"]
        scount = _count_stmts(k["body"])
        if scount > MAX_KERNEL_STMTS:
            raise Skl2CompileError(
                "E_KERNEL_TOO_LONG",
                f"kernel {name} has {scount} statements (max {MAX_KERNEL_STMTS}); compose with macro/codec/epilogue",
                k["file"],
                k["line"],
            )
        ccn, depth = _ccn_and_depth(k["body"])
        ccn += 1
        if ccn > MAX_CCN or depth > MAX_NESTING:
            raise Skl2CompileError(
                "E_COMPLEXITY",
                f"kernel {name} has CCN={ccn} (max {MAX_CCN}) and nesting={depth} (max {MAX_NESTING})",
                k["file"],
                k["line"],
            )
        _check_dead_locals_and_unreachable(k["body"], f"kernel {name}", k["file"])
        _check_pdl(k, unit["macros"])
        _collect_refs_and_calls(k["body"], set(), all_calls)
        dup_items.append((f"kernel {name}", k["file"], k["body"]))

    for m in unit["macros"].values():
        _collect_refs_and_calls(m["body"], set(), all_calls)

    _check_dup_subtrees(dup_items)

    if not allow_unused_lib:
        for mname, m in unit["macros"].items():
            if mname not in all_calls and not m.get("attrs", {}).get("export"):
                raise Skl2CompileError("E_DEAD_CODE", f"unused macro {mname!r}", m["file"], m["line"])
        for tname, tpl in unit["templates"].items():
            if tname not in used_templates and not tpl.get("attrs", {}).get("export"):
                raise Skl2CompileError("E_DEAD_CODE", f"uninstantiated template kernel {tname!r}", tpl["file"], tpl["line"])
        for k in unit["kernels"]:
            if k.get("private"):
                raise Skl2CompileError("E_DEAD_CODE", f"uninstantiated private kernel {k['name']!r}", k["file"], k["line"])
