import copy
import hashlib
import json
import os
import re
import sys

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKLC_DIR = os.path.join(TOOLS_DIR, "sklc")
if SKLC_DIR not in sys.path:
    sys.path.append(SKLC_DIR)

from codegen import compile_program
from cubin_elf import build_cubin
from skl2_frontend import Skl2CompileError, eval_meta_expr, try_eval_meta
from skl2_lint import check_anti_slop

COMPILER_VERSION = "skl2c-1.0"
CORE_CONCEPTS = ("linear", "moe", "attention", "recurrent", "norm_rope", "route_topk", "head", "allreduce")


def resolve_consts(unit):
    env = {}
    for sec in ("codecs", "epilogues"):
        for k, v in unit[sec].items():
            env[k] = {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
            env[k]["name"] = k
    for name, (expr, file, line) in unit["consts"].items():
        try: env[name] = eval_meta_expr(expr, env)
        except Exception as e: raise Skl2CompileError("E_CONST", f"cannot evaluate const {name}: {e}", file, line)
    return env


def _sub_expr(expr, env, rename, reg_arrays=None):
    if not isinstance(expr, tuple): return expr
    k = expr[0]
    if k in ("int", "float", "str"): return expr
    if k == "ident":
        nm = expr[1]
        if nm in rename:
            rep = rename[nm]
            return rep if isinstance(rep, tuple) else ("ident", rep)
        if nm in env and isinstance(env[nm], (int, float)):
            return ("float" if isinstance(env[nm], float) else "int", env[nm])
        return expr
    if k == "field":
        v = try_eval_meta(expr, env)
        if isinstance(v, (int, float)): return ("float" if isinstance(v, float) else "int", v)
        if isinstance(v, str): return ("str", v)
        return ("field", _sub_expr(expr[1], env, rename, reg_arrays), expr[2])
    if k == "un": out = ("un", expr[1], _sub_expr(expr[2], env, rename, reg_arrays))
    elif k == "bin": out = ("bin", expr[1], _sub_expr(expr[2], env, rename, reg_arrays), _sub_expr(expr[3], env, rename, reg_arrays))
    elif k == "ternary":
        c = _sub_expr(expr[1], env, rename, reg_arrays)
        cv = try_eval_meta(c, env)
        if cv is not None: return _sub_expr(expr[2] if cv else expr[3], env, rename, reg_arrays)
        return ("ternary", c, _sub_expr(expr[2], env, rename, reg_arrays), _sub_expr(expr[3], env, rename, reg_arrays))
    elif k == "call": out = ("call", expr[1], [_sub_expr(a, env, rename, reg_arrays) for a in expr[2]])
    elif k == "index":
        base, idx = _sub_expr(expr[1], env, rename, reg_arrays), _sub_expr(expr[2], env, rename, reg_arrays)
        if reg_arrays and base[0] == "ident" and base[1] in reg_arrays:
            iv = try_eval_meta(idx, env)
            if iv is None: raise Skl2CompileError("E_CONST", f"register array {base[1]} index must be compile-time constant")
            return ("ident", f"{base[1]}__{int(iv)}")
        out = ("index", base, idx)
    else: return expr
    v = try_eval_meta(out, env)
    return ("float" if isinstance(v, float) else "int", v) if isinstance(v, (int, float)) else out


class Expander:
    def __init__(self, macros, env):
        self.macros, self.env, self.uid, self.reg_arrays = macros, env, 0, set()

    def _ren_name(self, rename, nm):
        v = rename.get(nm, nm)
        return v[1] if isinstance(v, tuple) else v

    def expand_stmts(self, stmts, rename):
        out, ra = [], self.reg_arrays
        for st in stmts:
            k = st[0]
            if k == "expr" and st[1][0] == "call" and st[1][1] in self.macros:
                m = self.macros[st[1][1]]
                call_args = [_sub_expr(a, self.env, rename, ra) for a in st[1][2]]
                if len(call_args) != len(m["params"]):
                    raise Skl2CompileError("E_MACRO_ARITY", f"macro {m['name']} expects {len(m['params'])} args, got {len(call_args)}", m["file"], st[-1])
                self.uid += 1
                sub_ren = dict(rename)
                for p, a in zip(m["params"], call_args): sub_ren[p] = a
                m_locals = set()
                self._collect_locals(m["body"], m_locals)
                for loc in m_locals:
                    if loc not in m["params"]: sub_ren[loc] = f"__m{self.uid}_{loc}"
                out.extend(self.expand_stmts(m["body"], sub_ren))
            elif k == "shared":
                out.append(("shared", st[1], self._ren_name(rename, st[2]), _sub_expr(st[3], self.env, rename, ra)))
            elif k == "reg_array":
                nm = self._ren_name(rename, st[3])
                n = int(eval_meta_expr(_sub_expr(st[4], self.env, rename, ra), self.env))
                self.reg_arrays.add(nm)
                out.extend(("decl", st[1], st[2], f"{nm}__{i}", None) for i in range(n))
            elif k == "decl":
                init = _sub_expr(st[4], self.env, rename, ra) if st[4] is not None else None
                out.append(("decl", st[1], st[2], self._ren_name(rename, st[3]), init))
            elif k == "return": out.append(("return",))
            elif k == "assign": out.append(("assign", st[1], _sub_expr(st[2], self.env, rename, ra), _sub_expr(st[3], self.env, rename, ra)))
            elif k == "expr": out.append(("expr", _sub_expr(st[1], self.env, rename, ra)))
            elif k in ("if", "if_constexpr"):
                cond = _sub_expr(st[1], self.env, rename, ra)
                cv = try_eval_meta(cond, self.env)
                if cv is not None: out.extend(self.expand_stmts(st[2] if cv else st[3], rename))
                elif k == "if_constexpr": raise Skl2CompileError("E_CONST", "if constexpr condition is not compile-time constant", "<lower>", st[-1])
                else: out.extend(self._lower_cond_if(cond, self.expand_stmts(st[2], rename), self.expand_stmts(st[3], rename)))
            elif k == "unroll":
                var, s_e, e_e = st[1], _sub_expr(st[2], self.env, rename, ra), _sub_expr(st[3], self.env, rename, ra)
                s_v, e_v = try_eval_meta(s_e, self.env), try_eval_meta(e_e, self.env)
                if s_v is not None and e_v is not None and (self._has_macro_or_constexpr(st[4]) or self.reg_arrays):
                    had, old = var in self.env, self.env.get(var)
                    for iv in range(int(s_v), int(e_v)):
                        self.env[var] = iv
                        ib = self.expand_stmts(st[4], rename)
                        out.append(("if", ("int", 1), ib, [])) if any(x[0] == "decl" for x in ib) else out.extend(ib)
                    if had: self.env[var] = old
                    else: self.env.pop(var, None)
                else:
                    sub_ren = dict(rename); sub_ren.pop(var, None)
                    out.append(("unroll", var, s_e, e_e, self.expand_stmts(st[4], sub_ren)))
            elif k in ("for", "pipeline_for"):
                var, vname = st[3], self._ren_name(rename, st[3])
                sub_ren = dict(rename); sub_ren[var] = vname
                step = st[6]
                out.append((k, st[1], st[2], vname, _sub_expr(st[4], self.env, rename, ra), _sub_expr(st[5], self.env, sub_ren, ra), ("assign", step[1], _sub_expr(step[2], self.env, sub_ren, ra), _sub_expr(step[3], self.env, sub_ren, ra)), self.expand_stmts(st[7], sub_ren)))
        return out

    def _lower_cond_if(self, cond, t_b, e_b):
        if cond[0] == "bin" and cond[1] == "&&" and not e_b:
            return self._lower_cond_if(cond[2], self._lower_cond_if(cond[3], t_b, []), [])
        if cond[0] == "bin" and cond[1] == "||" and not e_b and t_b and t_b[-1][0] == "return" and len(t_b) <= 3:
            return self._lower_cond_if(cond[2], t_b, []) + self._lower_cond_if(cond[3], t_b, [])
        return [("if", cond, t_b, e_b)]

    def _collect_locals(self, stmts, out):
        for st in stmts:
            k = st[0]
            if k in ("decl", "reg_array"): out.add(st[3])
            elif k == "shared": out.add(st[2])
            elif k in ("if", "if_constexpr"): self._collect_locals(st[2], out); self._collect_locals(st[3], out)
            elif k in ("unroll", "for", "pipeline_for"):
                if k != "unroll": out.add(st[3])
                self._collect_locals(st[4 if k == "unroll" else 7], out)

    def _has_macro_or_constexpr(self, stmts):
        for st in stmts:
            k = st[0]
            if k == "if_constexpr" or (k == "expr" and st[1][0] == "call" and st[1][1] in self.macros): return True
            if k in ("if", "if_constexpr") and (self._has_macro_or_constexpr(st[2]) or self._has_macro_or_constexpr(st[3])): return True
            if k in ("unroll", "for", "pipeline_for") and self._has_macro_or_constexpr(st[4 if k == "unroll" else 7]): return True
        return False


def _lower_kernel(kdef, name, env, macros):
    params = []
    for p in kdef["params"]:
        if p.get("cond") is not None and not eval_meta_expr(p["cond"], env): continue
        if p["type"] == "blob":
            params.append({"name": p["name"], "type": "blob", "is_ptr": False, "blob_bytes": int(eval_meta_expr(p["blob_bytes"], env)), "blob_align": int(eval_meta_expr(p["blob_align"], env))})
        else:
            params.append({"name": p["name"], "type": p["type"], "is_ptr": p["is_ptr"]})
    return {"name": name, "block": int(eval_meta_expr(kdef["block"], env)), "regs": int(eval_meta_expr(kdef["regs"], env)), "params": params, "body": Expander(macros, env).expand_stmts(kdef["body"], {})}


def lower_unit(unit, only_set=None, renames=None, allow_unused_lib=False):
    check_anti_slop(unit, allow_unused_lib=allow_unused_lib)
    base_env, lowered = resolve_consts(unit), []
    for inst in unit["instantiations"]:
        tname = inst["template"]
        if tname not in unit["templates"]: raise Skl2CompileError("E_TEMPLATE", f"unknown template {tname!r}", inst["file"], inst["line"])
        tpl = unit["templates"][tname]
        if len(inst["args"]) != len(tpl["tparams"]):
            raise Skl2CompileError("E_TEMPLATE", f"template {tname} expects {len(tpl['tparams'])} args, got {len(inst['args'])}", inst["file"], inst["line"])
        kname = inst["name"]
        if only_set and kname not in only_set: continue
        env = dict(base_env)
        for tp, arg_e in zip(tpl["tparams"], inst["args"]): env[tp] = eval_meta_expr(arg_e, base_env)
        lowered.append(_lower_kernel(tpl, renames.get(kname, kname) if renames else kname, env, unit["macros"]))
    for k in unit["kernels"]:
        if only_set and k["name"] not in only_set: continue
        lowered.append(_lower_kernel(k, renames.get(k["name"], k["name"]) if renames else k["name"], dict(base_env), unit["macros"]))
    return {"consts": {k: v for k, v in base_env.items() if isinstance(v, (int, float))}, "kernels": lowered}


def compile_unit(unit, arch="sm121", only_set=None, renames=None, allow_unused_lib=False):
    prog = lower_unit(unit, only_set=only_set, renames=renames, allow_unused_lib=allow_unused_lib)
    if not prog["kernels"]: raise Skl2CompileError("E_EMPTY", "no kernels selected to compile")
    try: images = compile_program(prog, strict_reg_budget=True)
    except RuntimeError as e:
        if str(e).startswith("E_REG_SPILL"): raise Skl2CompileError("E_REG_SPILL", str(e).split(": ", 1)[1]) from e
        raise
    return images, build_cubin(images, arch=arch)


def _ver_key(v):
    return (1, int(v[1:]), v) if (isinstance(v, str) and v.startswith("v") and v[1:].isdigit()) else (0, 0, str(v))


def resolve_version(table, name, version="latest", kind="primitive"):
    if name not in table: raise Skl2CompileError("E_REF", f"unknown {kind} {name!r}")
    vers = table[name]
    if version == "latest":
        chosen = max(vers.keys(), key=_ver_key)
        return chosen, vers[chosen]
    if version not in vers:
        raise Skl2CompileError("E_VERSION", f"{kind} {name}@{version} not found (available: {', '.join(sorted(vers, key=_ver_key))})")
    return version, vers[version]


def select_primitive_variant(unit, pname, version="latest", context=None):
    ver, obj = resolve_version(unit["primitives"], pname, version, kind="primitive")
    spec, env, chosen = obj["spec"], dict(context or {}), None
    pol = spec.get("policy")
    if pol:
        chosen = next((r["target"] for r in pol["rules"] if try_eval_meta(r["cond"], env)), pol.get("default"))
    variants = spec.get("variants", {})
    if chosen is None and variants: chosen = next(iter(variants))
    return {"primitive": pname, "version": ver, "variant": chosen, "variant_spec": variants.get(chosen, {})}


def _strip_meta(obj):
    if isinstance(obj, dict): return {k: _strip_meta(v) for k, v in sorted(obj.items()) if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)): return [_strip_meta(v) for v in obj]
    return obj


def _canon_sha256(obj):
    return hashlib.sha256(json.dumps(_strip_meta(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_lockfile(unit, model_names=None):
    plan = resolve_models(unit, model_names=model_names) if unit["models"] else None
    prims = {p: {"version": v, "sha256": _canon_sha256(o["spec"])} for p in sorted(unit["primitives"]) for v, o in [resolve_version(unit["primitives"], p, "latest")]}
    models_lock = {
        m: {"version": v, "extends": f"{o['extends']}@{o['extends_version']}" if o["extends"] else None, "sha256": _canon_sha256(o["spec"])}
        for m in sorted(unit["models"]) for v, o in [resolve_version(unit["models"], m, "latest", "model")]
    }
    out = {
        "schema": "skl2.lock.v1", "compiler": COMPILER_VERSION, "primitives": prims, "models": models_lock,
        "codecs": {k: _canon_sha256(v) for k, v in sorted(unit["codecs"].items())},
        "epilogues": {k: _canon_sha256(v) for k, v in sorted(unit["epilogues"].items())},
    }
    if plan:
        out["resolved_models"] = {m: {k: info[k] for k in ("inheritance", "resolved_primitives", "kernels")} for m, info in sorted(plan["models"].items())}
    return out


def verify_lockfile(unit, lock_data):
    cur = build_lockfile(unit)
    for sec in ("primitives", "models", "codecs", "epilogues"):
        for k, exp_val in lock_data.get(sec, {}).items():
            actual = cur.get(sec, {})
            if k not in actual: raise Skl2CompileError("E_LOCK_MISMATCH", f"lockfile entry {sec}.{k} missing from source unit")
            if actual[k] != exp_val: raise Skl2CompileError("E_LOCK_MISMATCH", f"lockfile mismatch for {sec}.{k}: expected {exp_val}, got {actual[k]}")


def _deep_merge(base, delta):
    res = copy.deepcopy(base)
    for k, v in delta.items():
        if k == "overrides": res = _deep_merge(res, v)
        elif k == "fusions": res["fusions"] = list(res.get("fusions", [])) + list(v)
        elif k == "uses":
            by_ref = {u["ref"]: u["version"] for u in res.get("uses", [])}
            by_ref.update({u["ref"]: u["version"] for u in v})
            res["uses"] = [{"ref": r, "version": ver} for r, ver in by_ref.items()]
        elif isinstance(v, dict) and isinstance(res.get(k), dict): res[k] = _deep_merge(res[k], v)
        else: res[k] = copy.deepcopy(v)
    return res


def _count_leaf_fields(d):
    return sum(len(v) if k in ("fusions", "uses") else (_count_leaf_fields(v) if isinstance(v, dict) else 1) for k, v in d.items())


def resolve_models(unit, model_names=None):
    models = unit["models"]
    targets = list(model_names) if model_names else [n for n in models if not n.startswith("Reference")]
    resolved, kernel_registry = {}, {}

    def resolve_one(name, ver="latest", stack=()):
        if name in stack: raise Skl2CompileError("E_CYCLE", f"cyclic model inheritance: {' -> '.join(stack + (name,))}")
        actual_v, m = resolve_version(models, name, ver, kind="model")
        p_spec, p_chain = resolve_one(m["extends"], m["extends_version"], stack + (name,)) if m["extends"] else ({}, [])
        return _deep_merge(p_spec, m["spec"]), [f"{name}@{actual_v}"] + p_chain

    for name in targets:
        m_name, m_ver = name.split("@", 1) if "@" in name else (name, "latest")
        spec, chain = resolve_one(m_name, m_ver)
        actual_v, m = resolve_version(models, m_name, m_ver, kind="model")
        codec, act, norm, mixer, ffn = spec.get("weight_codec", "BF16"), spec.get("act_dtype", "bf16"), spec.get("norm", "rmsnorm"), spec.get("mixer", "gqa"), spec.get("ffn", "swiglu")
        fusions = spec.get("fusions", [])
        fused_tags = {rhs for _, rhs in fusions}
        resolved_prims, ar_ctx = {}, {"msg_bytes": int(spec.get("hidden", 4096)) * 2, "ranks": int(spec.get("tp_degree", 16)), "topology": spec.get("topology", "hybrid_mode0")}
        for u in list(unit.get("uses", [])) + list(spec.get("uses", [])):
            if u["ref"] in unit["primitives"]:
                sel = select_primitive_variant(unit, u["ref"], u["version"], context=ar_ctx)
                resolved_prims[u["ref"]] = f"{sel['version']}:{sel['variant']}"
        tp = spec.get("tp_collective")
        if not tp and "std.allreduce" in unit["primitives"]:
            sel = select_primitive_variant(unit, "std.allreduce", "latest", context=ar_ctx)
            tp = sel["variant"]
            resolved_prims.setdefault("std.allreduce", f"{sel['version']}:{sel['variant']}")
        tp = tp or "hybrid_inline_128"
        kernels = [
            f"linear.skinny_gemv.{codec.lower()}_{act}", f"linear.mma_gemm.{codec.lower()}_{act}",
            f"norm_rope.fused_residual_{norm}_{act}" if "fused_residual_rmsnorm" in fused_tags else f"norm_rope.{norm}_{act}",
        ]
        if spec.get("rope", True): kernels.append(f"norm_rope.rope_{spec.get('rope_mode', 'neox')}_{act}")
        if mixer in ("mla", "mla_sparse"):
            kernels.append(f"attention.mla_{act}")
            if mixer == "mla_sparse" or spec.get("sparse_indexer"): kernels.append("route_topk.sparse_indexer_topk")
        elif mixer in ("hybrid_gdn", "hybrid_kda"):
            kernels.extend([f"recurrent.{mixer.split('_')[1]}_decode_{act}", "recurrent.causal_conv1d_swish", f"attention.gqa_{act}"])
        else: kernels.append(f"attention.{mixer}_{act}")
        if spec.get("moe_experts", 0):
            kernels.extend([f"route_topk.moe_router_e{spec['moe_experts']}_k{spec.get('moe_topk', 8)}", f"moe.grouped_linear_{spec.get('moe_codec', codec).lower()}_{act}"])
        if "swiglu_linear" in fused_tags or ffn == "swiglu": kernels.append(f"epilogue.swiglu_{act}")
        elif ffn == "situ": kernels.append(f"epilogue.situ_{act}")
        kernels.extend([f"head.certified_{spec.get('head_codec', 'fp8').lower()}", f"allreduce.{tp}_fused_{norm}" if "ar_rmsnorm" in fused_tags else f"allreduce.{tp}"])
        for k in kernels: kernel_registry.setdefault(k, []).append(m_name)
        resolved[m_name] = {
            "version": actual_v, "inheritance": chain, "resolved_fields": _count_leaf_fields(spec),
            "diff_fields": _count_leaf_fields(m["spec"]), "resolved_primitives": resolved_prims,
            "spec": spec, "fusions": [f"{'+'.join(lhs)}->{rhs}" for lhs, rhs in fusions], "kernels": sorted(set(kernels)),
        }
    total_slots = sum(len(v["kernels"]) for v in resolved.values())
    unique_kernels = sorted(kernel_registry)
    return {
        "models": resolved, "unique_kernels": unique_kernels,
        "shared_kernels": {k: v for k, v in sorted(kernel_registry.items()) if len(v) > 1},
        "dedup_ratio": round(1.0 - (len(unique_kernels) / max(total_slots, 1)), 4),
    }


C99_PARAM_TYPES = {"ptr": "SparkDeviceAddress", "u64": "uint64_t", "u32": "uint32_t", "i32": "int32_t", "f32": "float", "u16": "uint16_t", "u8": "uint8_t", "b8": "const void *"}


def _cid(s): return re.sub(r"[^A-Za-z0-9_]", "_", str(s))


def _policy_c_expr(e, params, tags):
    k = e[0]
    if k == "int": return f"{int(e[1])}u"
    if k == "str":
        t = f"SKL2_TAG_{_cid(e[1]).upper()}"
        if t not in tags: tags.append(t)
        return t
    if k == "ident":
        if e[1] not in params: params.append(e[1])
        return e[1]
    if k == "un": return f"({e[1]}{_policy_c_expr(e[2], params, tags)})"
    if k == "bin": return f"({_policy_c_expr(e[2], params, tags)} {e[1]} {_policy_c_expr(e[3], params, tags)})"
    return "0u"


def _emit_policy_c99(unit, model_plan, img_cids):
    if not unit or not unit.get("primitives"): return []
    pinned = {pn: pv.split(":", 1)[0] for pn, pv in next(iter(model_plan["models"].values())).get("resolved_primitives", {}).items()} if (model_plan and len(model_plan.get("models", {})) == 1) else {}
    tags, out = ["SKL2_TAG_NONE"], []
    for pn in sorted(unit["primitives"]):
        spec = resolve_version(unit["primitives"], pn, pinned.get(pn, "latest"))[1]["spec"]
        variants, pol = spec.get("variants", {}), spec.get("policy")
        if not variants or not pol: continue
        con = _cid(spec.get("concept", pn.split(".")[-1]))
        vnames = list(variants.keys())
        vmap = {vn: f"SKL2_VAR_{con.upper()}_{_cid(vn).upper()}" for vn in vnames}
        out.append(f"typedef enum Skl2Variant_{con} {{ " + ", ".join(f"{vmap[vn]} = {i}" for i, vn in enumerate(vnames)) + f" }} Skl2Variant_{con};")
        params, rules = [], []
        for r in pol.get("rules", []): rules.append((_policy_c_expr(r["cond"], params, tags), vmap[r["target"]]))
        sig, dflt = ", ".join(f"uint32_t {p}" for p in params) or "void", vmap[pol.get("default") or vnames[0]]
        out.append(f"SKL2_INLINE Skl2Variant_{con} Skl2Select_{con}({sig}) {{ {' '.join(f'if {c} return {t};' if c.startswith('(') else f'if ({c}) return {t};' for c, t in rules)} return {dflt}; }}")
        kcases = [f"case {vmap[vn]}: return SKL2_ORD_{_cid(vs['kernel'])};" for vn, vs in variants.items() if isinstance(vs, dict) and _cid(vs.get("kernel", "")) in img_cids]
        if kcases: out.append(f"SKL2_INLINE Skl2KernelOrdinal Skl2Ordinal_{con}(Skl2Variant_{con} v) {{ switch (v) {{ {' '.join(kcases)} default: return SKL2_ORD_COUNT; }} }}")
    return [f"typedef enum Skl2PolicyTag {{ " + ", ".join(f"{t} = {i}" for i, t in enumerate(tags)) + " } Skl2PolicyTag;"] + out


def emit_c_header(images, model_plan=None, arch="sm121", unit=None):
    guard, mdefs = "LAUNCH", []
    if model_plan and len(model_plan.get("models", {})) == 1:
        mname, minfo = next(iter(model_plan["models"].items()))
        guard = _cid(mname).upper()
        for fk, fv in sorted(minfo.get("spec", {}).items()):
            mk = _cid(fk).upper()
            if isinstance(fv, bool): mdefs.append(f"#define SKL2_{guard}_{mk} {1 if fv else 0}u")
            elif isinstance(fv, int): mdefs.append(f"#define SKL2_{guard}_{mk} {fv}u")
            elif isinstance(fv, float): mdefs.append(f"#define SKL2_{guard}_{mk} {fv:.9g}f")
    cids = [_cid(k.name) for k in images]
    lines = [
        f"#ifndef SKL2_{guard}_H", f"#define SKL2_{guard}_H", "#include <stdint.h>", "#include <string.h>", '#include "sparkpipe/spark_device.h"',
        "#ifdef __cplusplus", 'extern "C" {', "#endif",
        "#if defined(__GNUC__) || defined(__clang__)", "#define SKL2_INLINE static inline __attribute__((unused))", "#else", "#define SKL2_INLINE static inline", "#endif",
        f'#define SKL2_{guard}_ARCH "{arch}"', *mdefs,
        "typedef enum Skl2KernelOrdinal { " + ", ".join(f"SKL2_ORD_{cid} = {i}" for i, cid in enumerate(cids)) + f", SKL2_ORD_COUNT = {len(cids)} }} Skl2KernelOrdinal;",
        f"typedef struct Skl2Module {{ SparkDeviceLibrary *library; SparkDeviceKernel *kernels[{max(1, len(cids))}]; }} Skl2Module;",
        f"static const char *const kSkl2KernelNames[{max(1, len(cids))}] = {{ " + ", ".join(f'"{k.name}"' for k in images) + " };",
        *_emit_policy_c99(unit, model_plan, set(cids)),
    ]
    layout_cases, launch_fns = [], []
    for k, cid in zip(images, cids):
        layout, pbytes = k.param_layout()
        entries = ", ".join(f"{{ {poff}u, {pb}u, {pa}u }}" for _, _, pa, pb, poff in layout) or "{ 0u, 0u, 1u }"
        lines.extend([f"static const SparkDeviceParameter kSkl2Layout_{cid}[{max(1, len(layout))}] = {{ {entries} }};", f"#define SKL2_PARAM_BYTES_{cid} {pbytes}u", f"#define SKL2_BLOCK_THREADS_{cid} {k.max_threads}u", f"#define SKL2_STATIC_SMEM_{cid} {k.smem_bytes}u"])
        layout_cases.append(f"\tcase SKL2_ORD_{cid}: *count = {len(layout)}u; *bytes = {pbytes}u; return kSkl2Layout_{cid};")
        p_sigs, p_copies = [], []
        for idx, (pname, pkind, _, pb, poff) in enumerate(layout):
            cname, ct = f"p_{_cid(pname)}_{idx}", C99_PARAM_TYPES[pkind]
            p_sigs.append(f"{ct}{cname}" if ct.endswith("*") else f"{ct} {cname}")
            p_copies.append(f"\tmemcpy(buf + {poff}u, {cname if pkind == 'b8' else '&' + cname}, {pb}u);")
        extra = (", " + ", ".join(p_sigs)) if p_sigs else ""
        launch_fns.append(
            f"SKL2_INLINE SparkStatus Skl2Launch_{cid}(SparkDeviceQueue *queue, const Skl2Module *mod, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t shared_bytes, uint32_t dependent{extra}) {{\n"
            f"\tuint8_t buf[{max(1, pbytes)}]; SparkDeviceLaunch launch; memset(buf, 0, sizeof(buf));\n" + "".join(c + "\n" for c in p_copies) +
            f"\tmemset(&launch, 0, sizeof(launch)); launch.groups.x = gx; launch.groups.y = gy; launch.groups.z = gz;\n"
            f"\tlaunch.threads.x = SKL2_BLOCK_THREADS_{cid}; launch.threads.y = 1u; launch.threads.z = 1u;\n"
            f"\tlaunch.shared_bytes = shared_bytes; launch.constant_bytes = {pbytes}u; launch.constants = {'buf' if pbytes > 0 else '0'}; launch.dependent = dependent;\n"
            f"\treturn SparkDeviceKernelLaunch(queue, mod->kernels[SKL2_ORD_{cid}], &launch); }}"
        )
    lines.extend([
        "SKL2_INLINE void Skl2ModuleUnload(Skl2Module *mod) { uint32_t i; if (mod == 0) return; for (i = 0u; i < (uint32_t)SKL2_ORD_COUNT; ++i) { if (mod->kernels[i] != 0) SparkDeviceKernelDestroy(mod->kernels[i]); mod->kernels[i] = 0; } if (mod->library != 0) SparkDeviceLibraryUnload(mod->library); mod->library = 0; }",
        "SKL2_INLINE SparkStatus Skl2ModuleLoad(Skl2Module *mod, const char *cubin_path, const char *owner) { SparkStatus st; uint32_t i; if (mod == 0 || cubin_path == 0) return SPARK_STATUS_INVALID_ARGUMENT; memset(mod, 0, sizeof(*mod)); st = SparkDeviceLibraryLoad(&mod->library, cubin_path, owner); if (st != SPARK_STATUS_OK) return st; for (i = 0u; i < (uint32_t)SKL2_ORD_COUNT; ++i) { st = SparkDeviceKernelCreate(&mod->kernels[i], mod->library, kSkl2KernelNames[i]); if (st != SPARK_STATUS_OK) { Skl2ModuleUnload(mod); return st; } } return SPARK_STATUS_OK; }",
        "SKL2_INLINE const SparkDeviceParameter *Skl2KernelLayout(Skl2KernelOrdinal ord, uint32_t *count, uint32_t *bytes) { switch (ord) {",
        *layout_cases, "\tdefault: *count = 0u; *bytes = 0u; return 0; } }",
        *launch_fns, "#ifdef __cplusplus", "}", "#endif", "#endif",
    ])
    return "\n".join(lines) + "\n"


def emit_embedded_c_header(images, cubin, manifest, prefix="Skl2Embedded"):
    up = _cid(prefix).upper()
    mb = manifest.encode("utf-8") + b"\x00"
    def fmt_bytes(data):
        return ["\t" + ",".join(f"0x{b:02x}" for b in data[i:i + 24]) + "," for i in range(0, len(data), 24)]
    cids = [_cid(k.name) for k in images]
    u = "__attribute__((unused)) "
    lines = [
        f"#ifndef {up}_EMBEDDED_H", f"#define {up}_EMBEDDED_H", "#include <stdint.h>",
        f"enum {prefix}Id {{ " + ", ".join(f"{up}_{c} = {i}" for i, c in enumerate(cids)) + f", {up}_COUNT = {len(cids)} }};",
        f"static const char *const {u}k{prefix}Names[{max(1, len(cids))}] = {{ " + ", ".join(f'"{k.name}"' for k in images) + " };",
        f"static const uint8_t {u}k{prefix}Cubin[{len(cubin)}] = {{", *fmt_bytes(cubin), "};",
        f"static const uint8_t {u}k{prefix}Manifest[{len(mb)}] = {{", *fmt_bytes(mb), "};",
        "#endif",
    ]
    return "\n".join(lines) + "\n"
