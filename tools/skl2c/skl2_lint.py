import hashlib
from skl2_frontend import Skl2CompileError

MAX_CCN, MAX_NESTING, MAX_KERNEL_STMTS, DUP_WINDOW = 16, 4, 80, 10
STORE_CALLS = {
    "stg256", "stg128", "stg128_stream", "stg128_sys",
    "stg64", "stg32", "stg32_stream", "stg32_sys", "stg16", "stg8",
    "atomg_add_u32", "atomg_exch_u32",
}
PDL_WAIT_CALLS, PDL_RELEASE_CALLS = {"pdl_wait", "acqbulk"}, {"pdl_release", "preexit"}


def _sub_blocks(st):
    k = st[0]
    if k in ("if", "if_constexpr"): return (st[2], st[3])
    if k == "unroll": return (st[4],)
    if k in ("for", "pipeline_for"): return (st[7],)
    return ()


def _count_stmts(stmts):
    return sum(1 + sum(_count_stmts(b) for b in _sub_blocks(st)) for st in stmts)


def _expr_ccn(e):
    if not isinstance(e, tuple): return 0
    k = e[0]
    if k == "ternary": return 1 + _expr_ccn(e[1]) + _expr_ccn(e[2]) + _expr_ccn(e[3])
    if k == "bin": return (1 if e[1] in ("&&", "||") else 0) + _expr_ccn(e[2]) + _expr_ccn(e[3])
    if k == "call": return sum(_expr_ccn(a) for a in e[2])
    if k == "index": return _expr_ccn(e[1]) + _expr_ccn(e[2])
    if k in ("un", "field"): return _expr_ccn(e[2 if k == "un" else 1])
    return 0


def _ccn_and_depth(stmts, depth=0):
    ccn, max_d = 0, depth
    for st in stmts:
        k = st[0]
        if k in ("if", "if_constexpr"):
            c1, d1 = _ccn_and_depth(st[2], depth + 1)
            ccn += 1 + _expr_ccn(st[1]) + c1
            max_d = max(max_d, d1)
            if st[3]:
                ed = depth if (len(st[3]) == 1 and st[3][0][0] in ("if", "if_constexpr")) else depth + 1
                c2, d2 = _ccn_and_depth(st[3], ed)
                ccn += c2
                max_d = max(max_d, d2)
        elif k in ("for", "pipeline_for"):
            c1, d1 = _ccn_and_depth(st[7], depth + 1)
            ccn += 1 + _expr_ccn(st[5]) + c1
            max_d = max(max_d, d1)
        elif k == "unroll":
            c1, d1 = _ccn_and_depth(st[4], depth + 1)
            ccn += c1
            max_d = max(max_d, d1)
        elif k == "decl" and st[4] is not None: ccn += _expr_ccn(st[4])
        elif k == "assign": ccn += _expr_ccn(st[2]) + _expr_ccn(st[3])
        elif k == "expr": ccn += _expr_ccn(st[1])
    return ccn, max_d


def _norm_expr(e, vmap):
    if not isinstance(e, tuple): return repr(e)
    k = e[0]
    if k in ("int", "float", "str"): return f"{k}:{e[1]}"
    if k == "ident": return vmap.setdefault(e[1], f"${len(vmap)}")
    if k == "un": return f"({e[1]}{_norm_expr(e[2], vmap)})"
    if k == "bin": return f"({_norm_expr(e[2], vmap)}{e[1]}{_norm_expr(e[3], vmap)})"
    if k == "ternary": return f"({_norm_expr(e[1], vmap)}?{_norm_expr(e[2], vmap)}:{_norm_expr(e[3], vmap)})"
    if k == "call": return f"{e[1]}({','.join(_norm_expr(a, vmap) for a in e[2])})"
    if k == "index": return f"{_norm_expr(e[1], vmap)}[{_norm_expr(e[2], vmap)}]"
    if k == "field": return f"{_norm_expr(e[1], vmap)}.{e[2]}"
    return repr(e)


def _norm_stmt(st, vmap):
    k = st[0]
    if k == "shared": return f"shared {st[1]} {vmap.setdefault(st[2], f'${len(vmap)}')}[{_norm_expr(st[3], vmap)}]"
    if k == "reg_array": return f"{st[1]} {st[2]} {vmap.setdefault(st[3], f'${len(vmap)}')}[{_norm_expr(st[4], vmap)}]"
    if k == "decl":
        init = _norm_expr(st[4], vmap) if st[4] is not None else ""
        return f"{st[1]} {st[2]} {vmap.setdefault(st[3], f'${len(vmap)}')}={init}"
    if k == "return": return "ret"
    if k == "assign": return f"{_norm_expr(st[2], vmap)}{st[1]}{_norm_expr(st[3], vmap)}"
    if k == "expr": return _norm_expr(st[1], vmap)
    if k in ("if", "if_constexpr"):
        return f"{k}({_norm_expr(st[1], vmap)}){{{';'.join(_norm_stmt(x, vmap) for x in st[2])}}}else{{{';'.join(_norm_stmt(x, vmap) for x in st[3])}}}"
    if k == "unroll":
        v = vmap.setdefault(st[1], f"${len(vmap)}")
        return f"unroll({v}={_norm_expr(st[2], vmap)}..{_norm_expr(st[3], vmap)}){{{';'.join(_norm_stmt(x, vmap) for x in st[4])}}}"
    if k in ("for", "pipeline_for"):
        v = vmap.setdefault(st[3], f"${len(vmap)}")
        return f"{k}({st[2]} {v}={_norm_expr(st[4], vmap)};{_norm_expr(st[5], vmap)};{_norm_stmt(st[6], vmap)}){{{';'.join(_norm_stmt(x, vmap) for x in st[7])}}}"
    return repr(st)


def _flat_stmts(stmts):
    out = []
    for st in stmts:
        out.append(st)
        for b in _sub_blocks(st): out.extend(_flat_stmts(b))
    return out


def _check_dup_subtrees(items):
    seen = {}
    for owner, file, stmts in items:
        flat = _flat_stmts(stmts)
        for i in range(max(0, len(flat) - DUP_WINDOW + 1)):
            win, vmap = flat[i:i + DUP_WINDOW], {}
            digest = hashlib.sha256("\n".join(_norm_stmt(s, vmap) for s in win).encode("utf-8")).hexdigest()[:16]
            if digest in seen:
                po, pf, pl, pi = seen[digest]
                if po != owner or abs(i - pi) >= DUP_WINDOW:
                    raise Skl2CompileError("E_DUP_SUBTREE", f"duplicated {DUP_WINDOW}-statement subtree in {owner} matches {po} at {pf}:{pl}; extract a macro or template", file, win[0][-1])
            else:
                seen[digest] = (owner, file, win[0][-1], i)


def _walk_expr(e, refs=None, calls=None):
    if not isinstance(e, tuple): return
    k = e[0]
    if k == "ident" and refs is not None: refs.add(e[1])
    elif k == "call":
        if calls is not None: (calls.append if isinstance(calls, list) else calls.add)(e[1])
        for a in e[2]: _walk_expr(a, refs, calls)
    elif k in ("un", "field"): _walk_expr(e[2 if k == "un" else 1], refs, calls)
    elif k in ("bin", "index"): _walk_expr(e[2 if k == "bin" else 1], refs, calls); _walk_expr(e[3 if k == "bin" else 2], refs, calls)
    elif k == "ternary":
        for a in e[1:4]: _walk_expr(a, refs, calls)


def _collect_refs_and_calls(stmts, refs, calls):
    for st in stmts:
        k = st[0]
        if k in ("shared", "reg_array"): _walk_expr(st[3 if k == "shared" else 4], refs, calls)
        elif k == "decl" and st[4] is not None: _walk_expr(st[4], refs, calls)
        elif k == "assign":
            if st[1] != "=" or st[2][0] != "ident": _walk_expr(st[2], refs, calls)
            _walk_expr(st[3], refs, calls)
        elif k == "expr": _walk_expr(st[1], refs, calls)
        elif k in ("if", "if_constexpr"):
            _walk_expr(st[1], refs, calls); _collect_refs_and_calls(st[2], refs, calls); _collect_refs_and_calls(st[3], refs, calls)
        elif k == "unroll":
            _walk_expr(st[2], refs, calls); _walk_expr(st[3], refs, calls); _collect_refs_and_calls(st[4], refs, calls)
        elif k in ("for", "pipeline_for"):
            _walk_expr(st[4], refs, calls); _walk_expr(st[5], refs, calls)
            _collect_refs_and_calls([st[6]], refs, calls); _collect_refs_and_calls(st[7], refs, calls)


def _check_dead_locals(stmts, owner, file, macro_params=()):
    refs = set()
    _collect_refs_and_calls(stmts, refs, set())

    def walk(block):
        terminated = False
        for st in block:
            ln, k = st[-1], st[0]
            if terminated: raise Skl2CompileError("E_DEAD_CODE", f"unreachable statement after return in {owner}", file, ln)
            if k == "return": terminated = True
            elif k == "shared" and st[2] not in refs: raise Skl2CompileError("E_DEAD_CODE", f"unused shared array {st[2]!r} in {owner}", file, ln)
            elif k in ("decl", "reg_array") and st[3] not in refs and st[3] not in macro_params:
                raise Skl2CompileError("E_DEAD_CODE", f"unused local register {st[3]!r} in {owner}", file, ln)
            elif k in ("if", "if_constexpr"):
                if walk(st[2]) and (walk(st[3]) if st[3] else False): terminated = True
            elif k in ("unroll", "for", "pipeline_for"):
                walk(st[4 if k == "unroll" else 7])
        return terminated

    walk(stmts)


def _check_pdl(kernel, macros):
    all_calls = set()
    _collect_refs_and_calls(kernel["body"], set(), all_calls)
    if not kernel.get("attrs", {}).get("pdl") and not (all_calls & (PDL_WAIT_CALLS | PDL_RELEASE_CALLS)):
        return
    file, name = kernel["file"], kernel["name"]

    def step(stmts, acq, rel):
        for st in stmts:
            ln, k = st[-1], st[0]
            if k == "expr" and st[1][0] == "call" and st[1][1] in macros:
                acq, rel, exited = step(macros[st[1][1]]["body"], acq, rel)
                if exited: return acq, rel, True
                continue
            calls = []
            if k == "expr": _walk_expr(st[1], calls=calls)
            elif k == "decl" and st[4] is not None: _walk_expr(st[4], calls=calls)
            elif k == "assign": _walk_expr(st[3], calls=calls)
            for c in calls:
                if c in PDL_WAIT_CALLS: acq = True
                elif c in PDL_RELEASE_CALLS: rel = True
                elif c in STORE_CALLS and not acq:
                    raise Skl2CompileError("E_PDL_HAZARD", f"PDL kernel {name} performs global write {c}() before acqbulk()/pdl_wait()", file, ln)
            if k == "return":
                if not rel: raise Skl2CompileError("E_PDL_HAZARD", f"PDL kernel {name} returns without calling preexit()/pdl_release()", file, ln)
                return acq, rel, True
            if k in ("if", "if_constexpr"):
                a1, r1, e1 = step(st[2], acq, rel)
                if st[3]:
                    a2, r2, e2 = step(st[3], acq, rel)
                    if e1 and e2: return a1 and a2, r1 and r2, True
                    acq, rel = (a2, r2) if e1 else ((a1, r1) if e2 else (a1 and a2, r1 and r2))
                elif not e1:
                    acq, rel = acq and a1, rel and r1
            elif k == "unroll":
                acq, rel, exited = step(st[4], acq, rel)
                if exited: return acq, rel, True
            elif k in ("for", "pipeline_for"):
                step(st[7], acq, rel)
        return acq, rel, False

    _, rel_end, exited = step(kernel["body"], False, False)
    if not exited and not rel_end:
        raise Skl2CompileError("E_PDL_HAZARD", f"PDL kernel {name} exits without calling preexit()/pdl_release()", file, kernel["line"])


def check_anti_slop(unit, allow_unused_lib=False):
    dup_items, all_calls = [], set()
    used_templates = {inst["template"] for inst in unit["instantiations"]}
    for mname, m in unit["macros"].items():
        ccn, depth = _ccn_and_depth(m["body"])
        if ccn + 1 > MAX_CCN or depth > MAX_NESTING:
            raise Skl2CompileError("E_COMPLEXITY", f"macro {mname} has CCN={ccn + 1} (max {MAX_CCN}) and nesting={depth} (max {MAX_NESTING})", m["file"], m["line"])
        _check_dead_locals(m["body"], f"macro {mname}", m["file"], set(m["params"]))
        dup_items.append((f"macro {mname}", m["file"], m["body"]))
        _collect_refs_and_calls(m["body"], set(), all_calls)
    for k in list(unit["templates"].values()) + list(unit["kernels"]):
        name, sc = k["name"], _count_stmts(k["body"])
        if sc > MAX_KERNEL_STMTS:
            raise Skl2CompileError("E_KERNEL_TOO_LONG", f"kernel {name} has {sc} statements (max {MAX_KERNEL_STMTS}); compose with macro/codec/epilogue", k["file"], k["line"])
        ccn, depth = _ccn_and_depth(k["body"])
        if ccn + 1 > MAX_CCN or depth > MAX_NESTING:
            raise Skl2CompileError("E_COMPLEXITY", f"kernel {name} has CCN={ccn + 1} (max {MAX_CCN}) and nesting={depth} (max {MAX_NESTING})", k["file"], k["line"])
        _check_dead_locals(k["body"], f"kernel {name}", k["file"])
        _check_pdl(k, unit["macros"])
        _collect_refs_and_calls(k["body"], set(), all_calls)
        dup_items.append((f"kernel {name}", k["file"], k["body"]))
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
