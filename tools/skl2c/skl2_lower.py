import copy
import hashlib
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKLC_DIR = os.path.join(TOOLS_DIR, "sklc")
if SKLC_DIR not in sys.path:
    sys.path.append(SKLC_DIR)

from codegen import compile_program
from cubin_elf import build_cubin, build_manifest
from skl2_frontend import Skl2CompileError, eval_meta_expr, try_eval_meta
from skl2_lint import check_anti_slop

COMPILER_VERSION = "skl2c-1.0"
CORE_CONCEPTS = ("linear", "moe", "attention", "recurrent", "norm_rope", "route_topk", "head", "allreduce")


def resolve_consts(unit):
    env = {}
    for k, v in unit["codecs"].items():
        env[k] = {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
        env[k]["name"] = k
    for k, v in unit["epilogues"].items():
        env[k] = {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
        env[k]["name"] = k
    for name, (expr, file, line) in unit["consts"].items():
        try:
            env[name] = eval_meta_expr(expr, env)
        except Exception as e:
            raise Skl2CompileError("E_CONST", f"cannot evaluate const {name}: {e}", file, line)
    return env


def _sub_expr(expr, env, rename):
    if not isinstance(expr, tuple):
        return expr
    k = expr[0]
    if k in ("int", "float", "str"):
        return expr
    if k == "ident":
        nm = expr[1]
        if nm in rename:
            rep = rename[nm]
            return rep if isinstance(rep, tuple) else ("ident", rep)
        if nm in env and isinstance(env[nm], (int, float)):
            return ("float", env[nm]) if isinstance(env[nm], float) else ("int", env[nm])
        return expr
    if k == "field":
        v = try_eval_meta(expr, env)
        if isinstance(v, (int, float)):
            return ("float", v) if isinstance(v, float) else ("int", v)
        if isinstance(v, str):
            return ("str", v)
        return ("field", _sub_expr(expr[1], env, rename), expr[2])
    if k == "un":
        out = ("un", expr[1], _sub_expr(expr[2], env, rename))
    elif k == "bin":
        out = ("bin", expr[1], _sub_expr(expr[2], env, rename), _sub_expr(expr[3], env, rename))
    elif k == "ternary":
        c = _sub_expr(expr[1], env, rename)
        cv = try_eval_meta(c, env)
        if cv is not None:
            return _sub_expr(expr[2] if cv else expr[3], env, rename)
        return ("ternary", c, _sub_expr(expr[2], env, rename), _sub_expr(expr[3], env, rename))
    elif k == "call":
        out = ("call", expr[1], [_sub_expr(a, env, rename) for a in expr[2]])
    elif k == "index":
        out = ("index", _sub_expr(expr[1], env, rename), _sub_expr(expr[2], env, rename))
    else:
        return expr
    v = try_eval_meta(out, env)
    if isinstance(v, (int, float)):
        return ("float", v) if isinstance(v, float) else ("int", v)
    return out


class Expander:
    def __init__(self, macros, env):
        self.macros = macros
        self.env = env
        self.uid = 0

    def expand_stmts(self, stmts, rename):
        out = []
        for st in stmts:
            k = st[0]
            if k == "expr" and st[1][0] == "call" and st[1][1] in self.macros:
                m = self.macros[st[1][1]]
                call_args = [_sub_expr(a, self.env, rename) for a in st[1][2]]
                if len(call_args) != len(m["params"]):
                    raise Skl2CompileError(
                        "E_MACRO_ARITY",
                        f"macro {m['name']} expects {len(m['params'])} args, got {len(call_args)}",
                        m["file"],
                        st[-1],
                    )
                self.uid += 1
                sub_ren = dict(rename)
                for p, a in zip(m["params"], call_args):
                    sub_ren[p] = a
                m_locals = set()
                self._collect_locals(m["body"], m_locals)
                for loc in m_locals:
                    if loc not in m["params"]:
                        sub_ren[loc] = f"__m{self.uid}_{loc}"
                out.extend(self.expand_stmts(m["body"], sub_ren))
                continue
            if k == "shared":
                nm = rename.get(st[2], st[2])
                if isinstance(nm, tuple):
                    nm = nm[1]
                out.append(("shared", st[1], nm, _sub_expr(st[3], self.env, rename)))
            elif k == "decl":
                nm = rename.get(st[3], st[3])
                if isinstance(nm, tuple):
                    nm = nm[1]
                init = _sub_expr(st[4], self.env, rename) if st[4] is not None else None
                out.append(("decl", st[1], st[2], nm, init))
            elif k == "return":
                out.append(("return",))
            elif k == "assign":
                out.append(("assign", st[1], _sub_expr(st[2], self.env, rename), _sub_expr(st[3], self.env, rename)))
            elif k == "expr":
                out.append(("expr", _sub_expr(st[1], self.env, rename)))
            elif k in ("if", "if_constexpr"):
                cond = _sub_expr(st[1], self.env, rename)
                cv = try_eval_meta(cond, self.env)
                if cv is not None:
                    out.extend(self.expand_stmts(st[2] if cv else st[3], rename))
                elif k == "if_constexpr":
                    raise Skl2CompileError("E_CONST", f"if constexpr condition is not compile-time constant", "<lower>", st[-1])
                else:
                    t_b = self.expand_stmts(st[2], rename)
                    e_b = self.expand_stmts(st[3], rename)
                    out.extend(self._lower_cond_if(cond, t_b, e_b))
            elif k == "unroll":
                var = st[1]
                s_e = _sub_expr(st[2], self.env, rename)
                e_e = _sub_expr(st[3], self.env, rename)
                s_v = try_eval_meta(s_e, self.env)
                e_v = try_eval_meta(e_e, self.env)
                if s_v is not None and e_v is not None and self._has_macro_or_constexpr(st[4]):
                    old = self.env.get(var, None)
                    had = var in self.env
                    for iv in range(int(s_v), int(e_v)):
                        self.env[var] = iv
                        out.extend(self.expand_stmts(st[4], rename))
                    if had:
                        self.env[var] = old
                    else:
                        self.env.pop(var, None)
                else:
                    sub_ren = dict(rename)
                    sub_ren.pop(var, None)
                    out.append(("unroll", var, s_e, e_e, self.expand_stmts(st[4], sub_ren)))
            elif k in ("for", "pipeline_for"):
                var = st[3]
                vname = rename.get(var, var)
                if isinstance(vname, tuple):
                    vname = vname[1]
                sub_ren = dict(rename)
                sub_ren[var] = vname
                step = st[6]
                step_l = ("assign", step[1], _sub_expr(step[2], self.env, sub_ren), _sub_expr(step[3], self.env, sub_ren))
                out.append((
                    k, st[1], st[2], vname,
                    _sub_expr(st[4], self.env, rename),
                    _sub_expr(st[5], self.env, sub_ren),
                    step_l,
                    self.expand_stmts(st[7], sub_ren),
                ))
        return out

    def _lower_cond_if(self, cond, t_b, e_b):
        if cond[0] == "bin" and cond[1] == "&&" and not e_b:
            return self._lower_cond_if(cond[2], self._lower_cond_if(cond[3], t_b, []), [])
        if cond[0] == "bin" and cond[1] == "||" and not e_b and len(t_b) == 1 and t_b[0][0] == "return":
            return self._lower_cond_if(cond[2], t_b, []) + self._lower_cond_if(cond[3], t_b, [])
        return [("if", cond, t_b, e_b)]

    def _collect_locals(self, stmts, out):
        for st in stmts:
            k = st[0]
            if k == "decl":
                out.add(st[3])
            elif k == "shared":
                out.add(st[2])
            elif k in ("if", "if_constexpr"):
                self._collect_locals(st[2], out)
                self._collect_locals(st[3], out)
            elif k == "unroll":
                self._collect_locals(st[4], out)
            elif k in ("for", "pipeline_for"):
                out.add(st[3])
                self._collect_locals(st[7], out)

    def _has_macro_or_constexpr(self, stmts):
        for st in stmts:
            k = st[0]
            if k == "if_constexpr":
                return True
            if k == "expr" and st[1][0] == "call" and st[1][1] in self.macros:
                return True
            if k in ("if", "if_constexpr") and (self._has_macro_or_constexpr(st[2]) or self._has_macro_or_constexpr(st[3])):
                return True
            if k == "unroll" and self._has_macro_or_constexpr(st[4]):
                return True
            if k in ("for", "pipeline_for") and self._has_macro_or_constexpr(st[7]):
                return True
        return False


def _lower_kernel(kdef, name, env, macros):
    block = int(eval_meta_expr(kdef["block"], env))
    regs = int(eval_meta_expr(kdef["regs"], env))
    params = []
    for p in kdef["params"]:
        if p.get("cond") is not None and not eval_meta_expr(p["cond"], env):
            continue
        if p["type"] == "blob":
            params.append({
                "name": p["name"], "type": "blob", "is_ptr": False,
                "blob_bytes": int(eval_meta_expr(p["blob_bytes"], env)),
                "blob_align": int(eval_meta_expr(p["blob_align"], env)),
            })
        else:
            params.append({"name": p["name"], "type": p["type"], "is_ptr": p["is_ptr"]})
    body = Expander(macros, env).expand_stmts(kdef["body"], {})
    return {"name": name, "block": block, "regs": regs, "params": params, "body": body}


def lower_unit(unit, only_set=None, renames=None, allow_unused_lib=False):
    check_anti_slop(unit, allow_unused_lib=allow_unused_lib)
    base_env = resolve_consts(unit)
    lowered_kernels = []
    for inst in unit["instantiations"]:
        tname = inst["template"]
        if tname not in unit["templates"]:
            raise Skl2CompileError("E_TEMPLATE", f"unknown template {tname!r}", inst["file"], inst["line"])
        tpl = unit["templates"][tname]
        if len(inst["args"]) != len(tpl["tparams"]):
            raise Skl2CompileError(
                "E_TEMPLATE",
                f"template {tname} expects {len(tpl['tparams'])} args, got {len(inst['args'])}",
                inst["file"],
                inst["line"],
            )
        kname = inst["name"]
        if only_set and kname not in only_set:
            continue
        env = dict(base_env)
        for tp, arg_e in zip(tpl["tparams"], inst["args"]):
            env[tp] = eval_meta_expr(arg_e, base_env)
        out_name = renames.get(kname, kname) if renames else kname
        lowered_kernels.append(_lower_kernel(tpl, out_name, env, unit["macros"]))

    for k in unit["kernels"]:
        if only_set and k["name"] not in only_set:
            continue
        out_name = renames.get(k["name"], k["name"]) if renames else k["name"]
        lowered_kernels.append(_lower_kernel(k, out_name, dict(base_env), unit["macros"]))

    num_consts = {k: v for k, v in base_env.items() if isinstance(v, (int, float))}
    return {"consts": num_consts, "kernels": lowered_kernels}


def compile_unit(unit, arch="sm121", only_set=None, renames=None, allow_unused_lib=False):
    prog = lower_unit(unit, only_set=only_set, renames=renames, allow_unused_lib=allow_unused_lib)
    if not prog["kernels"]:
        raise Skl2CompileError("E_EMPTY", "no kernels selected to compile")
    try:
        images = compile_program(prog, strict_reg_budget=True)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("E_REG_SPILL"):
            raise Skl2CompileError("E_REG_SPILL", msg.split(": ", 1)[1]) from e
        raise
    cubin = build_cubin(images, arch=arch)
    return images, cubin


def _ver_key(v):
    if isinstance(v, str) and v.startswith("v") and v[1:].isdigit():
        return (1, int(v[1:]), v)
    return (0, 0, str(v))


def resolve_version(table, name, version="latest", kind="primitive"):
    if name not in table:
        raise Skl2CompileError("E_REF", f"unknown {kind} {name!r}")
    vers = table[name]
    if version == "latest":
        chosen = max(vers.keys(), key=_ver_key)
        return chosen, vers[chosen]
    if version not in vers:
        avail = ", ".join(sorted(vers.keys(), key=_ver_key))
        raise Skl2CompileError("E_VERSION", f"{kind} {name}@{version} not found (available: {avail})")
    return version, vers[version]


def select_primitive_variant(unit, pname, version="latest", context=None):
    ver, obj = resolve_version(unit["primitives"], pname, version, kind="primitive")
    spec = obj["spec"]
    env = dict(context or {})
    pol = spec.get("policy")
    chosen = None
    if pol:
        for rule in pol["rules"]:
            val = try_eval_meta(rule["cond"], env)
            if val:
                chosen = rule["target"]
                break
        if chosen is None:
            chosen = pol.get("default")
    variants = spec.get("variants", {})
    if chosen is None and variants:
        chosen = next(iter(variants))
    return {
        "primitive": pname,
        "version": ver,
        "variant": chosen,
        "variant_spec": variants.get(chosen, {}),
    }


def _strip_meta(obj):
    if isinstance(obj, dict):
        return {k: _strip_meta(v) for k, v in sorted(obj.items()) if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_meta(v) for v in obj]
    if isinstance(obj, tuple):
        return [_strip_meta(v) for v in obj]
    return obj


def _canon_sha256(obj):
    blob = json.dumps(_strip_meta(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_lockfile(unit, model_names=None):
    plan = resolve_models(unit, model_names=model_names) if unit["models"] else None
    prims = {}
    for pname in sorted(unit["primitives"].keys()):
        ver, obj = resolve_version(unit["primitives"], pname, "latest", kind="primitive")
        prims[pname] = {"version": ver, "sha256": _canon_sha256(obj["spec"])}
    models_lock = {}
    for mname in sorted(unit["models"].keys()):
        ver, obj = resolve_version(unit["models"], mname, "latest", kind="model")
        models_lock[mname] = {
            "version": ver,
            "extends": f"{obj['extends']}@{obj['extends_version']}" if obj["extends"] else None,
            "sha256": _canon_sha256(obj["spec"]),
        }
    codecs_lock = {k: _canon_sha256(v) for k, v in sorted(unit["codecs"].items())}
    epilogues_lock = {k: _canon_sha256(v) for k, v in sorted(unit["epilogues"].items())}
    out = {
        "schema": "skl2.lock.v1",
        "compiler": COMPILER_VERSION,
        "primitives": prims,
        "models": models_lock,
        "codecs": codecs_lock,
        "epilogues": epilogues_lock,
    }
    if plan:
        out["resolved_models"] = {
            m: {
                "inheritance": info["inheritance"],
                "resolved_primitives": info["resolved_primitives"],
                "kernels": info["kernels"],
            }
            for m, info in sorted(plan["models"].items())
        }
    return out


def verify_lockfile(unit, lock_data):
    cur = build_lockfile(unit)
    for sec in ("primitives", "models", "codecs", "epilogues"):
        expected = lock_data.get(sec, {})
        actual = cur.get(sec, {})
        for k, exp_val in expected.items():
            if k not in actual:
                raise Skl2CompileError("E_LOCK_MISMATCH", f"lockfile entry {sec}.{k} missing from source unit")
            if actual[k] != exp_val:
                raise Skl2CompileError(
                    "E_LOCK_MISMATCH",
                    f"lockfile mismatch for {sec}.{k}: expected {exp_val}, got {actual[k]}",
                )


def _deep_merge(base, delta):
    res = copy.deepcopy(base)
    for k, v in delta.items():
        if k == "overrides":
            res = _deep_merge(res, v)
        elif k == "fusions":
            res["fusions"] = list(res.get("fusions", [])) + list(v)
        elif k == "uses":
            by_ref = {u["ref"]: u["version"] for u in res.get("uses", [])}
            for u in v:
                by_ref[u["ref"]] = u["version"]
            res["uses"] = [{"ref": r, "version": ver} for r, ver in by_ref.items()]
        elif isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = _deep_merge(res[k], v)
        else:
            res[k] = copy.deepcopy(v)
    return res


def _count_leaf_fields(d):
    n = 0
    for k, v in d.items():
        if k in ("fusions", "uses"):
            n += len(v)
        elif isinstance(v, dict):
            n += _count_leaf_fields(v)
        else:
            n += 1
    return n


def resolve_models(unit, model_names=None):
    models = unit["models"]
    targets = list(model_names) if model_names else [
        n for n in models.keys() if not n.startswith("Reference")
    ]
    resolved = {}
    kernel_registry = {}

    def resolve_one(name, ver="latest", stack=()):
        if name in stack:
            raise Skl2CompileError("E_CYCLE", f"cyclic model inheritance: {' -> '.join(stack + (name,))}")
        actual_v, m = resolve_version(models, name, ver, kind="model")
        if m["extends"]:
            p_spec, p_chain = resolve_one(m["extends"], m["extends_version"], stack + (name,))
        else:
            p_spec, p_chain = {}, []
        return _deep_merge(p_spec, m["spec"]), [f"{name}@{actual_v}"] + p_chain

    for name in targets:
        m_ver = "latest"
        m_name = name
        if "@" in name:
            m_name, m_ver = name.split("@", 1)
        spec, chain = resolve_one(m_name, m_ver)
        actual_v, m = resolve_version(models, m_name, m_ver, kind="model")
        codec = spec.get("weight_codec", "BF16")
        act = spec.get("act_dtype", "bf16")
        norm = spec.get("norm", "rmsnorm")
        mixer = spec.get("mixer", "gqa")
        ffn = spec.get("ffn", "swiglu")
        fusions = spec.get("fusions", [])
        fused_tags = {rhs for _, rhs in fusions}

        resolved_prims = {}
        all_uses = list(unit.get("uses", [])) + list(spec.get("uses", []))
        ar_ctx = {
            "msg_bytes": int(spec.get("hidden", 4096)) * 2,
            "ranks": int(spec.get("tp_degree", 16)),
            "topology": spec.get("topology", "hybrid_mode0"),
        }
        for u in all_uses:
            if u["ref"] in unit["primitives"]:
                sel = select_primitive_variant(unit, u["ref"], u["version"], context=ar_ctx)
                resolved_prims[u["ref"]] = f"{sel['version']}:{sel['variant']}"
        tp = spec.get("tp_collective")
        if not tp and "std.allreduce" in unit["primitives"]:
            sel = select_primitive_variant(unit, "std.allreduce", "latest", context=ar_ctx)
            tp = sel["variant"]
            resolved_prims.setdefault("std.allreduce", f"{sel['version']}:{sel['variant']}")
        tp = tp or "hybrid_inline_128"

        kernels = []
        kernels.append(f"linear.skinny_gemv.{codec.lower()}_{act}")
        kernels.append(f"linear.mma_gemm.{codec.lower()}_{act}")
        if "fused_residual_rmsnorm" in fused_tags:
            kernels.append(f"norm_rope.fused_residual_{norm}_{act}")
        else:
            kernels.append(f"norm_rope.{norm}_{act}")
        if spec.get("rope", True):
            kernels.append(f"norm_rope.rope_{spec.get('rope_mode', 'neox')}_{act}")
        if mixer in ("mla", "mla_sparse"):
            kernels.append(f"attention.mla_{act}")
            if mixer == "mla_sparse" or spec.get("sparse_indexer"):
                kernels.append("route_topk.sparse_indexer_topk")
        elif mixer in ("hybrid_gdn", "hybrid_kda"):
            kernels.append(f"recurrent.{mixer.split('_')[1]}_decode_{act}")
            kernels.append("recurrent.causal_conv1d_swish")
            kernels.append(f"attention.gqa_{act}")
        else:
            kernels.append(f"attention.{mixer}_{act}")
        if spec.get("moe_experts", 0):
            kernels.append(f"route_topk.moe_router_e{spec['moe_experts']}_k{spec.get('moe_topk', 8)}")
            moe_codec = spec.get("moe_codec", codec).lower()
            kernels.append(f"moe.grouped_linear_{moe_codec}_{act}")
        if "swiglu_linear" in fused_tags or ffn == "swiglu":
            kernels.append(f"epilogue.swiglu_{act}")
        elif ffn == "situ":
            kernels.append(f"epilogue.situ_{act}")
        kernels.append(f"head.certified_{spec.get('head_codec', 'fp8').lower()}")
        if "ar_rmsnorm" in fused_tags:
            kernels.append(f"allreduce.{tp}_fused_{norm}")
        else:
            kernels.append(f"allreduce.{tp}")

        for k in kernels:
            kernel_registry.setdefault(k, []).append(m_name)

        total_fields = _count_leaf_fields(spec)
        diff_fields = _count_leaf_fields(m["spec"])
        resolved[m_name] = {
            "version": actual_v,
            "inheritance": chain,
            "resolved_fields": total_fields,
            "diff_fields": diff_fields,
            "resolved_primitives": resolved_prims,
            "spec": spec,
            "fusions": [f"{'+'.join(lhs)}->{rhs}" for lhs, rhs in fusions],
            "kernels": sorted(set(kernels)),
        }

    total_slots = sum(len(v["kernels"]) for v in resolved.values())
    unique_kernels = sorted(kernel_registry.keys())
    return {
        "models": resolved,
        "unique_kernels": unique_kernels,
        "shared_kernels": {k: v for k, v in sorted(kernel_registry.items()) if len(v) > 1},
        "dedup_ratio": round(1.0 - (len(unique_kernels) / max(total_slots, 1)), 4),
    }
