import os
import sys

_SKL2C_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skl2c")
if _SKL2C_DIR not in sys.path:
    sys.path.append(_SKL2C_DIR)

import isa_sm121 as isa
from cubin_elf import KernelImage
from skl2_frontend import try_eval_meta as try_eval_const

TYPE_REGS = {"u8": 1, "u16": 1, "u32": 1, "i32": 1, "f32": 1, "u64": 2, "u32x2": 2, "f32x2": 2, "u32x4": 4, "f32x4": 4, "u32x8": 8}
TYPE_ALIGN = {"u8": 1, "u16": 1, "u32": 1, "i32": 1, "f32": 1, "u64": 2, "u32x2": 2, "f32x2": 2, "u32x4": 4, "f32x4": 4, "u32x8": 4}
PARAM_SPEC = {"u8": ("u8", 1, 1), "u16": ("u16", 2, 2), "u32": ("u32", 4, 4), "i32": ("i32", 4, 4), "f32": ("f32", 4, 4), "u64": ("u64", 8, 8)}
LOAD_CALLS = {
    "ldg256_stream": ("u32x8", "256_ef"), "ldg256_persist": ("u32x8", "256_el"), "ldg256_default": ("u32x8", "256_def"),
    "ldg128_stream": ("u32x4", "128_ef"), "ldg128_persist": ("u32x4", "128_const"),
    "ldg64_stream": ("u32x2", "64_ef"), "ldg64_persist": ("u32x2", "64_const"),
    "ldg32_stream": ("u32", "32_ef"), "ldg32_persist": ("u32", "32_const"),
    "ldg16_stream": ("u32", "16_ef"), "ldg16_persist": ("u32", "16_const"),
    "ldg8_stream": ("u32", "8_ef"), "ldg8_persist": ("u32", "8_const"),
}
LDG_SPEC = {
    "128_ef": (isa.LDG_W_128, isa.CACHE_EF), "128_const": (isa.LDG_W_128, isa.CACHE_CONSTANT),
    "64_ef": (isa.LDG_W_64, isa.CACHE_EF), "64_const": (isa.LDG_W_64, isa.CACHE_CONSTANT),
    "32_ef": (isa.LDG_W_32, isa.CACHE_EF), "32_const": (isa.LDG_W_32, isa.CACHE_CONSTANT),
    "16_ef": (isa.LDG_W_U16, isa.CACHE_EF), "16_const": (isa.LDG_W_U16, isa.CACHE_CONSTANT),
    "8_ef": (isa.LDG_W_U8, isa.CACHE_EF), "8_const": (isa.LDG_W_U8, isa.CACHE_CONSTANT),
}
STG_SPEC = {
    "stg128": (isa.STG_W_128, isa.STG_CACHE_DEFAULT), "stg128_stream": (isa.STG_W_128, isa.STG_CACHE_EF), "stg128_sys": (isa.STG_W_128, isa.STG_CACHE_STRONG_SYS),
    "stg64": (isa.STG_W_64, isa.STG_CACHE_DEFAULT), "stg32": (isa.STG_W_32, isa.STG_CACHE_DEFAULT),
    "stg32_stream": (isa.STG_W_32, isa.STG_CACHE_EF), "stg32_sys": (isa.STG_W_32, isa.STG_CACHE_STRONG_SYS),
    "stg16": (isa.STG_W_U16, isa.STG_CACHE_DEFAULT), "stg8": (isa.STG_W_U8, isa.STG_CACHE_DEFAULT),
}
STORE_CALLS = set(STG_SPEC) | {"stg256", "sts32", "dyn_sts32"}
SR_CALLS = ("ctaid_x", "ctaid_y", "ctaid_z", "tid_x", "lane_id", "ntid_x", "ntid_y", "grid_x", "grid_y", "warp_id")
SR_S2R = {"ctaid_x": isa.SR_CTAID_X, "ctaid_y": isa.SR_CTAID_Y, "ctaid_z": isa.SR_CTAID_Z, "tid_x": isa.SR_TID_X, "lane_id": isa.SR_LANEID}
SR_LDC = {"ntid_x": isa.C0_NTID_X, "ntid_y": isa.C0_NTID_Y, "grid_x": isa.C0_NCTAID_X, "grid_y": isa.C0_NCTAID_Y}
CMP_MAP = {"<": isa.CMP_LT, "==": isa.CMP_EQ, "<=": isa.CMP_LE, ">": isa.CMP_GT, "!=": isa.CMP_NE, ">=": isa.CMP_GE}
FIELD_IDX = {"x": 0, "y": 1, "z": 2, "w": 3}
PACK_TYPES = {"pack_u32x2": "u32x2", "pack_u32x4": "u32x4", "pack_f32x4": "f32x4", "pack_u32x8": "u32x8"}
SHFL_ENC = {"shfl_xor": isa.encode_shfl_bfly, "shfl_down": isa.encode_shfl_down, "shfl_up": isa.encode_shfl_up}


class Val:
    __slots__ = ("vtype", "regs", "is_temp")
    def __init__(self, vtype, regs, is_temp=False):
        self.vtype, self.regs, self.is_temp = vtype, tuple(regs), is_temp
    @property
    def r0(self):
        return self.regs[0]


class Op:
    def __init__(
        self, encoder, reads=(), writes=(), ureads=(), uwrites=(), preads=(), pwrites=(),
        var_lat=False, is_branch=False, is_exit=False, is_bar=False, is_store=False,
        target_label=None, label=None, fixed_lat=5, pipe="gen", mem_addr_reads=(),
        forced_wr=None, forced_wait=0, batch_id=None,
    ):
        self.encoder = encoder
        self.reads = {r for r in reads if r != isa.RZ}
        self.writes = {r for r in writes if r != isa.RZ}
        self.ureads = {u for u in ureads if u != isa.URZ}
        self.uwrites = {u for u in uwrites if u != isa.URZ}
        self.preads = {p for p in preads if p != isa.PT}
        self.pwrites = {p for p in pwrites if p != isa.PT}
        self.var_lat, self.is_branch, self.is_exit, self.is_bar, self.is_store = var_lat, is_branch, is_exit, is_bar, is_store
        self.target_label, self.label, self.fixed_lat, self.pipe = target_label, label, fixed_lat, pipe
        self.mem_addr_reads = {r for r in mem_addr_reads if r != isa.RZ}
        self.forced_wr, self.forced_wait, self.batch_id = forced_wr, forced_wait, batch_id
        self.wr, self.rd, self.wait, self.stall = 7, 7, 0, 1


class RegAlloc:
    def __init__(self, max_regs=248):
        self.max_regs, self.used, self.peak, self.tick, self.free_tick = max_regs, set(), 0, 0, {}

    def alloc(self, count, align=1, prefer_lru=False, avoid=()):
        if prefer_lru and self.peak >= count:
            avoid_set = set(avoid)
            best, best_key, r = None, None, 0
            while r + count <= self.peak:
                if r % align == 0 and all((r + k) not in self.used for k in range(count)):
                    key = (1 if any((r + k) in avoid_set for k in range(count)) else 0, max(self.free_tick.get(r + k, 0) for k in range(count)), r)
                    if best_key is None or key < best_key:
                        best_key, best = key, r
                r += align
            if best is not None and (best_key[0] == 0 or self.peak + count > self.max_regs):
                self.used.update(range(best, best + count))
                return tuple(range(best, best + count))
        r = 0
        while r + count <= self.max_regs:
            if r % align == 0 and all((r + k) not in self.used for k in range(count)):
                if avoid and any((r + k) in avoid for k in range(count)):
                    r += align
                    continue
                self.used.update(range(r, r + count))
                if r + count > self.peak:
                    self.peak = r + count
                return tuple(range(r, r + count))
            r += align
        if avoid:
            return self.alloc(count, align=align, prefer_lru=False, avoid=())
        raise RuntimeError(f"register allocator exhausted (need {count} aligned {align}, peak {self.peak})")

    def free(self, regs):
        self.tick += 1
        for r in regs:
            self.used.discard(r)
            self.free_tick[r] = self.tick


class KernelCompiler:
    def __init__(self, kdef, consts, strict_reg_budget=False):
        self.kdef, self.consts, self.strict_reg_budget = kdef, dict(consts), strict_reg_budget
        self.alloc, self.ops, self.scopes, self.smem, self.smem_bytes = RegAlloc(), [], [{}], {}, 0
        self.barriers = self.bssy_depth = self.label_counter = self.batch_counter = 0
        self.ur_desc = 4
        self.cached_sr, self.param_offsets, self.param_meta, self.param_loaded = {}, {}, [], {}
        self.temp_stack, self.recent_mem_addr, self.assigned_vars = [], set(), set()
        off = 0
        for p in kdef["params"]:
            if p["type"] == "blob":
                pkind, palign, pbytes, vtype = "b8", int(p["blob_align"]), int(p["blob_bytes"]), "blob"
            elif p["is_ptr"]:
                pkind, palign, pbytes, vtype = "ptr", 8, 8, "u64"
            else:
                pkind, palign, pbytes = PARAM_SPEC[p["type"]]
                vtype = p["type"]
            off = (off + palign - 1) // palign * palign
            self.param_offsets[p["name"]] = (isa.C0_PARAM_BASE + off, vtype, pbytes)
            self.param_meta.append((p["name"], pkind, palign, pbytes))
            off += pbytes

    def new_label(self, prefix="L"):
        self.label_counter += 1
        return f"{prefix}_{self.label_counter}"

    def emit(self, op):
        self.ops.append(op)
        return op

    def emit_label(self, name):
        self.ops.append(Op(None, label=name))

    def all_live_regs(self):
        live = set()
        for frame in self.scopes:
            for v in frame.values(): live.update(v.regs)
        for v in self.param_loaded.values(): live.update(v.regs)
        for v in self.cached_sr.values(): live.update(v.regs)
        for frame in self.temp_stack:
            for v in frame: live.update(v.regs)
        return live

    def free_if_unreferenced(self, regs):
        live = self.all_live_regs()
        dead = [r for r in regs if r not in live]
        if dead:
            self.alloc.free(dead)

    def alloc_val(self, vtype, temp=False, avoid=()):
        v = Val(vtype, self.alloc.alloc(TYPE_REGS[vtype], TYPE_ALIGN[vtype], prefer_lru=temp, avoid=avoid), is_temp=temp)
        if temp:
            self.temp_stack[-1].append(v)
        return v

    def dst_for(self, vtype, hint_dst, any_scalar=False, non_f32=False):
        if hint_dst is not None:
            if any_scalar and len(hint_dst.regs) == 1: return hint_dst
            if non_f32 and len(hint_dst.regs) == 1 and hint_dst.vtype != "f32": return hint_dst
            if not any_scalar and not non_f32 and hint_dst.vtype == vtype: return hint_dst
        return self.alloc_val(vtype, temp=True)

    def push_temps(self): self.temp_stack.append([])
    def pop_temps(self, keep=None):
        keep_regs = set(keep.regs) if keep is not None else set()
        popped = self.temp_stack.pop()
        if not popped: return
        live = self.all_live_regs() | keep_regs
        for v in popped:
            dead = [r for r in v.regs if r not in live]
            if dead: self.alloc.free(dead)

    def push_scope(self): self.scopes.append({})
    def pop_scope(self):
        for v in self.scopes.pop().values(): self.free_if_unreferenced(v.regs)

    def lookup_var(self, name):
        for frame in reversed(self.scopes):
            if name in frame: return frame[name]
        return None

    def bind_var(self, name, val): self.scopes[-1][name] = val

    def get_param(self, name):
        if name in self.param_loaded: return self.param_loaded[name]
        c_off, vtype, pbytes = self.param_offsets[name]
        v = self.alloc_val(vtype, temp=False)
        width = 64 if pbytes == 8 else 32
        self.emit(Op(lambda wr, rd, wait, stall, r=v.r0, o=c_off, w=width: isa.encode_ldc(r, o, width=w, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        self.param_loaded[name] = v
        return v

    def get_sr(self, key):
        if key in self.cached_sr: return self.cached_sr[key]
        if key in SR_S2R:
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0, s=SR_S2R[key]: isa.encode_s2r(r, s, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key in SR_LDC:
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0, o=SR_LDC[key]: isa.encode_ldc(r, o, width=32, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "warp_id":
            tid = self.get_sr("tid_x")
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, d=v.r0, s=tid.r0: isa.encode_shf_r_u32_hi_i(d, 5, s, stall=stall, wait=wait), reads=(tid.r0,), writes=v.regs))
        else:
            raise ValueError(f"unknown sr: {key}")
        self.cached_sr[key] = v
        return v

    def collect_uses(self, node, used_idents, used_srs):
        if isinstance(node, list):
            for x in node: self.collect_uses(x, used_idents, used_srs)
            return
        if not isinstance(node, tuple) or not node: return
        tag = node[0]
        if tag == "ident":
            used_idents.add(node[1])
            return
        if tag == "call" and node[1] in SR_CALLS:
            used_srs.add(node[1])
            if node[1] == "warp_id": used_srs.add("tid_x")
        for item in node[1:]: self.collect_uses(item, used_idents, used_srs)

    def collect_assigned(self, node, assigned):
        if isinstance(node, list):
            for x in node: self.collect_assigned(x, assigned)
            return
        if not isinstance(node, tuple) or not node: return
        tag = node[0]
        if tag == "assign":
            lhs = node[2]
            while isinstance(lhs, tuple) and lhs[0] in ("index", "field"): lhs = lhs[1]
            if isinstance(lhs, tuple) and lhs[0] == "ident": assigned.add(lhs[1])
        elif tag in ("for", "pipeline_for"):
            assigned.add(node[3])
        for item in node[1:]: self.collect_assigned(item, assigned)

    def compile(self):
        self.emit(Op(lambda wr, rd, wait, stall, u=self.ur_desc: isa.encode_ldcu(u, isa.C0_GLOBAL_DESC, width=64, wr=wr, stall=stall, wait=wait), uwrites=(self.ur_desc, self.ur_desc + 1), var_lat=True))
        body = self.kdef["body"]
        self.assigned_vars = set()
        self.collect_assigned(body, self.assigned_vars)
        first_param, last_param, first_sr, last_sr, last_var = {}, {}, {}, {}, {}
        for idx, st in enumerate(body):
            ui, us = set(), set()
            self.collect_uses(st, ui, us)
            for name in ui:
                if name in self.param_offsets:
                    first_param.setdefault(name, idx)
                    last_param[name] = idx
                else:
                    last_var[name] = idx
            for sr in us:
                first_sr.setdefault(sr, idx)
                last_sr[sr] = idx
        for idx, st in enumerate(body):
            for p in self.kdef["params"]:
                if p["type"] != "blob" and first_param.get(p["name"]) == idx: self.get_param(p["name"])
            for sr in SR_CALLS:
                if first_sr.get(sr) == idx: self.get_sr(sr)
            self.compile_stmt(st)
            for pname in list(self.param_loaded):
                if last_param.get(pname, -1) == idx: self.free_if_unreferenced(self.param_loaded.pop(pname).regs)
            for sr in list(self.cached_sr):
                if last_sr.get(sr, -1) == idx: self.free_if_unreferenced(self.cached_sr.pop(sr).regs)
            for vname in list(self.scopes[0]):
                if last_var.get(vname, -1) <= idx: self.free_if_unreferenced(self.scopes[0].pop(vname).regs)
        self.emit(Op(lambda wr, rd, wait, stall: isa.encode_exit(stall=stall, wait=wait), is_exit=True))
        insns = self.schedule_and_encode()
        max_r = max((r for op in self.ops for r in (op.reads | op.writes) if r != isa.RZ), default=-1)
        needed_regs = (max(max(max_r + 1, 0) + 2, 16) + 1) & ~1
        if self.strict_reg_budget and needed_regs > self.kdef["regs"]:
            raise RuntimeError(f"E_REG_SPILL: kernel {self.kdef['name']} requires {needed_regs} registers (peak R{max_r}), exceeding declared budget {self.kdef['regs']}")
        reg_count = (max(self.kdef["regs"], needed_regs) + 1) & ~1
        return KernelImage(name=self.kdef["name"], insns=insns, params=self.param_meta, registers=reg_count, max_threads=self.kdef["block"], barriers=self.barriers, smem_bytes=self.smem_bytes, stack_bytes=0)

    def compile_stmts(self, stmts):
        for st in stmts: self.compile_stmt(st)

    def compile_scoped_stmts(self, stmts):
        last_local = {}
        for idx, st in enumerate(stmts):
            ui, us = set(), set()
            self.collect_uses(st, ui, us)
            for name in ui: last_local[name] = idx
        for idx, st in enumerate(stmts):
            self.compile_stmt(st)
            frame = self.scopes[-1]
            for vname in list(frame):
                if last_local.get(vname, -1) <= idx: self.free_if_unreferenced(frame.pop(vname).regs)

    def compile_stmt(self, st):
        kind = st[0]
        if kind == "shared":
            _, vtype, name, size_expr = st
            count, elem_bytes = int(try_eval_const(size_expr, self.consts)), 4 * TYPE_REGS[vtype]
            align = max(4, elem_bytes)
            off = (self.smem_bytes + align - 1) // align * align
            self.smem[name] = (off, vtype, elem_bytes, count)
            self.smem_bytes = off + count * elem_bytes
        elif kind == "return":
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_exit(stall=stall, wait=wait), is_exit=True))
        elif kind == "unroll":
            _, var, start_e, end_e, body = st
            s_val, e_val = int(try_eval_const(start_e, self.consts)), int(try_eval_const(end_e, self.consts))
            had_old, old_val = var in self.consts, self.consts.get(var)
            for idx in range(s_val, e_val):
                self.consts[var] = idx
                self.push_scope()
                self.compile_stmts(body)
                self.pop_scope()
            if had_old: self.consts[var] = old_val
            else: self.consts.pop(var, None)
        elif kind == "decl":
            _, _, vtype, name, init = st
            if init is not None and name not in self.assigned_vars and init[0] == "call" and init[1] in SR_CALLS and vtype == "u32":
                self.bind_var(name, self.get_sr(init[1]))
                return
            v = self.alloc_val(vtype, temp=False)
            self.bind_var(name, v)
            if init is not None:
                self.push_temps()
                self.eval_into(init, v)
                self.pop_temps(keep=v)
        elif kind == "assign":
            self.push_temps()
            self.compile_assign(st[1], st[2], st[3])
            self.pop_temps()
        elif kind == "expr":
            self.push_temps()
            self.eval_expr(st[1])
            self.pop_temps()
        elif kind == "if":
            _, cond, then_body, else_body = st
            cval = try_eval_const(cond, self.consts)
            if cval is not None:
                self.push_scope()
                self.compile_stmts(then_body if cval else else_body)
                self.pop_scope()
            else:
                self.compile_runtime_if(cond, then_body, else_body)
        elif kind in ("for", "pipeline_for"):
            self.compile_for(st, pipelined=(kind == "pipeline_for"))
        else:
            raise ValueError(f"unsupported stmt: {kind}")

    def compile_runtime_if(self, cond, then_body, else_body):
        if len(then_body) == 1 and then_body[0][0] == "return" and not else_body:
            self.push_temps()
            pred = self.eval_cond(cond, pred=0)
            self.emit(Op(lambda wr, rd, wait, stall, p=pred: isa.encode_exit(pred=p, neg=False, stall=stall, wait=wait), preads=(pred,), is_exit=True))
            self.pop_temps()
            return
        if not else_body and len(then_body) <= 8 and all(st[0] == "expr" and st[1][0] == "call" and st[1][1] in STORE_CALLS for st in then_body):
            self.push_temps()
            pred = self.eval_cond(cond, pred=0)
            self.pop_temps()
            for st in then_body:
                self.push_temps()
                self.emit_store(st[1][1], st[1][2], pred=pred)
                self.pop_temps()
            return
        lbl_else, lbl_end, lbl_reconv = self.new_label("ELSE"), self.new_label("ENDIF"), self.new_label("RECONV")
        conv_bar = self.bssy_depth
        self.bssy_depth += 1
        self.push_temps()
        self.emit(Op(lambda delta, wr, rd, wait, stall, b=conv_bar: isa.encode_bssy(delta, bar=b, stall=stall, wait=wait), is_branch=True, target_label=lbl_reconv))
        pred = self.eval_cond(cond, pred=0)
        self.emit(Op(lambda delta, wr, rd, wait, stall, p=pred: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait), preads=(pred,), is_branch=True, target_label=lbl_else if else_body else lbl_end))
        self.pop_temps()
        self.push_scope()
        self.compile_stmts(then_body)
        self.pop_scope()
        if else_body:
            self.emit(Op(lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait), is_branch=True, target_label=lbl_end))
            self.emit_label(lbl_else)
            self.push_scope()
            self.compile_stmts(else_body)
            self.pop_scope()
        self.emit_label(lbl_end)
        self.emit(Op(lambda wr, rd, wait, stall, b=conv_bar: isa.encode_bsync(bar=b, stall=stall, wait=wait), is_bar=True))
        self.emit_label(lbl_reconv)
        self.bssy_depth -= 1

    def _emit_smem_store(self, base_off, elem_bytes, idx_e, val_e, pred=isa.PT):
        idx_c = try_eval_const(idx_e, self.consts)
        val = self.eval_expr(val_e)
        if idx_c is not None:
            off = base_off + int(idx_c) * elem_bytes
            self.emit(Op(lambda wr, rd, wait, stall, rv=val.r0, o=off, p=pred: isa.encode_sts(isa.RZ, rv, offset=o, pred=p, neg=False, rd_bar=rd, stall=stall, wait=wait), reads=(val.r0,), preads=(pred,), mem_addr_reads=(val.r0,), is_store=True))
        else:
            idx_v = self.eval_expr(idx_e)
            raddr = self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, d=raddr.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(d, a, eb, stall=stall, wait=wait), reads=(idx_v.r0,), writes=(raddr.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, ra=raddr.r0, rv=val.r0, o=base_off, p=pred: isa.encode_sts(ra, rv, offset=o, pred=p, neg=False, rd_bar=rd, stall=stall, wait=wait), reads=(raddr.r0, val.r0), preads=(pred,), mem_addr_reads=(raddr.r0, val.r0), is_store=True))

    def _emit_smem_load(self, base_off, elem_bytes, idx_e, dst, is_128=False):
        enc = isa.encode_lds128 if is_128 else isa.encode_lds
        idx_c = try_eval_const(idx_e, self.consts)
        if idx_c is not None:
            off = base_off + int(idx_c) * elem_bytes
            self.emit(Op(lambda wr, rd, wait, stall, r=dst.r0, o=off, fn=enc: fn(r, isa.RZ, offset=o, wr=wr, rd_bar=rd, stall=stall, wait=wait), writes=dst.regs, var_lat=True))
        else:
            idx_v = self.eval_expr(idx_e)
            raddr = self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=raddr.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(r, a, eb, stall=stall, wait=wait), reads=(idx_v.r0,), writes=(raddr.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=dst.r0, ra=raddr.r0, o=base_off, fn=enc: fn(r, ra, offset=o, wr=wr, rd_bar=rd, stall=stall, wait=wait), reads=(raddr.r0,), mem_addr_reads=(raddr.r0,), writes=dst.regs, var_lat=True))
        return dst

    def emit_store(self, name, args, pred=isa.PT):
        if name == "dyn_sts32":
            self._emit_smem_store((self.smem_bytes + 15) & ~15, 4, args[0], args[1], pred=pred)
            return
        if name == "sts32":
            base_off, _, elem_bytes, _ = self.smem[args[0][1]]
            self._emit_smem_store(base_off, elem_bytes, args[1], args[2], pred=pred)
            return
        ur = self.ur_desc
        if name == "stg256":
            ptr, val = self.eval_expr(args[0]), self.eval_expr(args[1])
            addr_reads = (ptr.regs[0], ptr.regs[1])
            self.recent_mem_addr = set(addr_reads)
            self.emit(Op(lambda wr, rd, wait, stall, pa=ptr.regs[0], vlo=val.regs[0], vhi=val.regs[4], u=ur, p=pred: isa.encode_stg256(pa, vlo, vhi, u, pred=p, neg=False, rd=rd, stall=stall, wait=wait), reads=addr_reads + val.regs, preads=(pred,), mem_addr_reads=addr_reads + val.regs, ureads=(ur, ur + 1), is_store=True))
            return
        ptr, off = self.eval_addr_offset(args[0])
        val = self.eval_expr(args[1])
        addr_reads = (ptr.regs[0], ptr.regs[1])
        self.recent_mem_addr = set(addr_reads)
        wbits, cbits = STG_SPEC[name]
        self.emit(Op(lambda wr, rd, wait, stall, pa=ptr.regs[0], rv=val.regs[0], u=ur, wb=wbits, cb=cbits, o=off, p=pred: isa.encode_stg(pa, rv, u, width_bits=wb, cache=cb, offset=o, pred=p, neg=False, rd=rd, stall=stall, wait=wait), reads=addr_reads + val.regs, preads=(pred,), mem_addr_reads=addr_reads + val.regs, ureads=(ur, ur + 1), is_store=True))

    def can_pipeline_loop(self, var, cond, step, body):
        if cond[0] != "bin" or cond[1] != "<" or cond[2] != ("ident", var): return False
        if step[0] != "assign" or step[1] != "+=" or step[2] != ("ident", var): return False
        loads = 0
        for st in body:
            if st[0] == "decl" and st[4] is not None and st[4][0] == "call" and st[4][1] in LOAD_CALLS: loads += 1
            else: break
        return 1 <= loads < len(body)

    def compile_for(self, st, pipelined=False):
        _, _, vtype, var, init_e, cond_e, step_st, body = st
        if pipelined and self.can_pipeline_loop(var, cond_e, step_st, body):
            self.compile_pipelined_for(vtype, var, init_e, cond_e, step_st, body)
            return
        self.push_scope()
        ivar = self.alloc_val(vtype, temp=False)
        self.bind_var(var, ivar)
        self.push_temps()
        self.eval_into(init_e, ivar)
        self.pop_temps(keep=ivar)
        lbl_head, lbl_end = self.new_label("LOOP"), self.new_label("ENDLOOP")
        self.push_temps()
        pred = self.eval_cond(cond_e, pred=0)
        self.emit(Op(lambda delta, wr, rd, wait, stall, p=pred: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait), preads=(pred,), is_branch=True, target_label=lbl_end))
        self.pop_temps()
        self.emit_label(lbl_head)
        self.push_scope()
        self.compile_scoped_stmts(body)
        self.pop_scope()
        self.push_temps()
        self.compile_stmt(step_st)
        pred2 = self.eval_cond(cond_e, pred=0)
        self.emit(Op(lambda delta, wr, rd, wait, stall, p=pred2: isa.encode_bra(delta, pred=p, neg=False, stall=stall, wait=wait), preads=(pred2,), is_branch=True, target_label=lbl_head))
        self.pop_temps()
        self.emit_label(lbl_end)
        self.pop_scope()

    def compile_pipelined_for(self, vtype, var, init_e, cond_e, step_st, body):
        load_stmts = []
        for st in body:
            if st[0] == "decl" and st[4] is not None and st[4][0] == "call" and st[4][1] in LOAD_CALLS: load_stmts.append(st)
            else: break
        comp_stmts = body[len(load_stmts):]
        self.push_scope()
        ivar, inext = self.alloc_val(vtype, temp=False), self.alloc_val(vtype, temp=False)
        self.bind_var(var, ivar)
        self.push_temps()
        self.eval_into(init_e, ivar)
        self.pop_temps(keep=ivar)
        limit_expr, step_expr = cond_e[3], step_st[3]
        bank0, bank1 = {}, {}
        for _, _, ltype, lname, _ in load_stmts:
            bank0[lname] = self.alloc_val(ltype, temp=False)
            bank1[lname] = self.alloc_val(ltype, temp=False)
        lbl_loop, lbl_epi0, lbl_epi1, lbl_end = self.new_label("PLOOP"), self.new_label("PEPI0"), self.new_label("PEPI1"), self.new_label("PEND")
        self.push_temps()
        p_init = self.eval_cond(("bin", "<", ("ident", var), limit_expr), pred=0)
        self.emit(Op(lambda delta, wr, rd, wait, stall, p=p_init: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait), preads=(p_init,), is_branch=True, target_label=lbl_end))
        self.pop_temps()

        def emit_bank_loads(idx_val, bank, bar_id):
            self.batch_counter += 1
            bid, saved = self.batch_counter, self.scopes[-1][var]
            self.scopes[-1][var] = idx_val
            self.push_temps()
            for _, _, _, lname, lexpr in load_stmts:
                self.emit_load_call(lexpr[1], lexpr[2], bank[lname], forced_wr=bar_id, batch_id=bid)
            self.pop_temps()
            self.scopes[-1][var] = saved

        def emit_bank_compute(bank, bar_id):
            self.push_scope()
            self.scopes[-1].update(bank)
            self.push_scope()
            start_idx = len(self.ops)
            self.compile_stmts(comp_stmts)
            bank_regs = {r for lval in bank.values() for r in lval.regs}
            for op in self.ops[start_idx:]:
                if op.label is None and (op.reads & bank_regs):
                    op.forced_wait |= 1 << bar_id
                    break
            self.pop_scope()
            self.scopes.pop()

        def emit_step(dst_v, src_v):
            self.push_temps()
            s_const = try_eval_const(step_expr, self.consts)
            if s_const is not None:
                self.emit(Op(lambda wr, rd, wait, stall, d=dst_v.r0, a=src_v.r0, im=int(s_const): isa.encode_iadd3_i(d, a, im, stall=stall, wait=wait), reads=(src_v.r0,), writes=(dst_v.r0,)))
            else:
                sv = self.eval_expr(step_expr)
                self.emit(Op(lambda wr, rd, wait, stall, d=dst_v.r0, a=src_v.r0, b=sv.r0: isa.encode_iadd3_r(d, a, b, stall=stall, wait=wait), reads=(src_v.r0, sv.r0), writes=(dst_v.r0,)))
            self.pop_temps()

        emit_bank_loads(ivar, bank0, bar_id=4)
        emit_step(inext, ivar)
        self.emit_label(lbl_loop)
        for cur_bank, nxt_bank, cur_bar, nxt_bar, epi_lbl in ((bank0, bank1, 4, 5, lbl_epi0), (bank1, bank0, 5, 4, lbl_epi1)):
            self.push_temps()
            p_chk = self.eval_cmp_val(inext, limit_expr, "<", pred=0)
            self.emit(Op(lambda delta, wr, rd, wait, stall, p=p_chk: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait), preads=(p_chk,), is_branch=True, target_label=epi_lbl))
            self.pop_temps()
            emit_bank_loads(inext, nxt_bank, bar_id=nxt_bar)
            emit_bank_compute(cur_bank, bar_id=cur_bar)
            self.emit(Op(lambda wr, rd, wait, stall, d=ivar.r0, s=inext.r0: isa.encode_mov_r(d, s, stall=stall, wait=wait), reads=(inext.r0,), writes=(ivar.r0,)))
            emit_step(inext, ivar)
        self.emit(Op(lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait), is_branch=True, target_label=lbl_loop))
        self.emit_label(lbl_epi0)
        emit_bank_compute(bank0, bar_id=4)
        self.emit(Op(lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait), is_branch=True, target_label=lbl_end))
        self.emit_label(lbl_epi1)
        emit_bank_compute(bank1, bar_id=5)
        self.emit_label(lbl_end)
        for v in (*bank0.values(), *bank1.values(), inext): self.alloc.free(v.regs)
        self.pop_scope()

    def compile_assign(self, op, lhs, rhs):
        dst = self.eval_lvalue(lhs)
        self.eval_into(rhs if op == "=" else ("bin", op[0], lhs, rhs), dst)

    def eval_lvalue(self, expr):
        if expr[0] == "ident":
            v = self.lookup_var(expr[1])
            if v is None: raise ValueError(f"undefined variable: {expr[1]}")
            return v
        if expr[0] in ("index", "field"):
            base = self.eval_lvalue(expr[1])
            idx = int(try_eval_const(expr[2], self.consts)) if expr[0] == "index" else FIELD_IDX[expr[2]]
            return Val("f32" if base.vtype.startswith("f32") else "u32", (base.regs[idx],))
        raise ValueError(f"invalid lvalue: {expr}")

    def eval_cmp_val(self, lhs_v, rhs_expr, op_str, pred=0):
        cmp_op, is_float, is_signed = CMP_MAP[op_str], lhs_v.vtype == "f32", lhs_v.vtype == "i32"
        rc = try_eval_const(rhs_expr, self.consts)
        if rc is not None:
            if is_float:
                self.emit(Op(lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, f=float(rc), c=cmp_op: isa.encode_fsetp_i(p, a, f, cmp_op=c, stall=stall, wait=wait), reads=(lhs_v.r0,), pwrites=(pred,), fixed_lat=13))
            else:
                self.emit(Op(lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, im=int(rc) & 0xFFFFFFFF, c=cmp_op, u=not is_signed: isa.encode_isetp_i(p, a, im, cmp_op=c, unsigned=u, stall=stall, wait=wait), reads=(lhs_v.r0,), pwrites=(pred,), fixed_lat=13))
            return pred
        rhs_v = self.eval_expr(rhs_expr)
        if is_float or rhs_v.vtype == "f32":
            self.emit(Op(lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, b=rhs_v.r0, c=cmp_op: isa.encode_fsetp_r(p, a, b, cmp_op=c, stall=stall, wait=wait), reads=(lhs_v.r0, rhs_v.r0), pwrites=(pred,), fixed_lat=13))
        else:
            self.emit(Op(lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, b=rhs_v.r0, c=cmp_op, u=not is_signed: isa.encode_isetp_r(p, a, b, cmp_op=c, unsigned=u, stall=stall, wait=wait), reads=(lhs_v.r0, rhs_v.r0), pwrites=(pred,), fixed_lat=13))
        return pred

    def eval_cond(self, expr, pred=0):
        if expr[0] == "bin" and expr[1] in CMP_MAP:
            return self.eval_cmp_val(self.eval_expr(expr[2]), expr[3], expr[1], pred=pred)
        v = self.eval_expr(expr)
        self.emit(Op(lambda wr, rd, wait, stall, p=pred, a=v.r0: isa.encode_isetp_i(p, a, 0, cmp_op=isa.CMP_NE, unsigned=True, stall=stall, wait=wait), reads=(v.r0,), pwrites=(pred,), fixed_lat=13))
        return pred

    def eval_into(self, expr, dst):
        if expr[0] == "call" and expr[1] in LOAD_CALLS:
            self.emit_load_call(expr[1], expr[2], dst)
            return dst
        if expr[0] == "call" and expr[1] == "fma":
            a = self.eval_expr(expr[2][0])
            b_const, c_const = try_eval_const(expr[2][1], self.consts), try_eval_const(expr[2][2], self.consts)
            if c_const is not None and float(c_const) == 0.0:
                if b_const is not None:
                    self.emit(Op(lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, f=float(b_const): isa.encode_fmul_i(d, ra, f, stall=stall, wait=wait), reads=(a.r0,), writes=(dst.r0,), pipe="fp32"))
                else:
                    b = self.eval_expr(expr[2][1])
                    self.emit(Op(lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, rb=b.r0: isa.encode_fmul_r(d, ra, rb, stall=stall, wait=wait), reads=(a.r0, b.r0), writes=(dst.r0,), pipe="fp32"))
                return dst
            c = self.eval_expr(expr[2][2])
            if b_const is not None:
                self.emit(Op(lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, f=float(b_const), rc=c.r0: isa.encode_ffma_i(d, ra, f, rc, stall=stall, wait=wait), reads=(a.r0, c.r0), writes=(dst.r0,), pipe="fp32"))
            else:
                b = self.eval_expr(expr[2][1])
                self.emit(Op(lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, rb=b.r0, rc=c.r0: isa.encode_ffma_r(d, ra, rb, rc, stall=stall, wait=wait), reads=(a.r0, b.r0, c.r0), writes=(dst.r0,), pipe="fp32"))
            return dst
        val = self.eval_expr(expr, hint_dst=dst)
        if val.regs != dst.regs:
            for d_r, s_r in zip(dst.regs, val.regs):
                self.emit(Op(lambda wr, rd, wait, stall, d=d_r, s=s_r: isa.encode_mov_r(d, s, stall=stall, wait=wait), reads=(s_r,), writes=(d_r,)))
        return dst

    def eval_addr_offset(self, expr):
        if expr[0] == "call" and expr[1] == "ptr_add":
            a = expr[2]
            idx_c, str_c = try_eval_const(a[1], self.consts), try_eval_const(a[2], self.consts)
            if idx_c is not None and str_c is not None:
                byte_off = int(idx_c) * int(str_c)
                if -0x800000 <= byte_off < 0x800000:
                    return self.eval_expr(a[0]), byte_off
        return self.eval_expr(expr), 0

    def emit_load_call(self, fn_name, args, dst, forced_wr=None, batch_id=None):
        _, mode = LOAD_CALLS[fn_name]
        ur = self.ur_desc
        if mode.startswith("256_"):
            ptr = self.eval_expr(args[0])
            addr_reads = (ptr.regs[0], ptr.regs[1])
            self.recent_mem_addr = set(addr_reads)
            pol = {"256_ef": "ef", "256_el": "el", "256_def": "default"}[mode]
            self.emit(Op(lambda wr, rd, wait, stall, rlo=dst.regs[0], rhi=dst.regs[4], pa=ptr.regs[0], u=ur, p=pol: isa.encode_ldg256(rlo, rhi, pa, u, policy=p, wr=wr, rd_bar=rd, stall=stall, wait=wait), reads=addr_reads, mem_addr_reads=addr_reads, ureads=(ur, ur + 1), writes=dst.regs, var_lat=True, forced_wr=forced_wr, batch_id=batch_id))
            return
        ptr, off = self.eval_addr_offset(args[0])
        addr_reads = (ptr.regs[0], ptr.regs[1])
        self.recent_mem_addr = set(addr_reads)
        wbits, cbits = LDG_SPEC[mode]
        self.emit(Op(lambda wr, rd, wait, stall, r=dst.regs[0], pa=ptr.regs[0], u=ur, wb=wbits, cb=cbits, o=off: isa.encode_ldg(r, pa, u, width_bits=wb, cache=cb, offset=o, wr=wr, rd_bar=rd, stall=stall, wait=wait), reads=addr_reads, mem_addr_reads=addr_reads, ureads=(ur, ur + 1), writes=dst.regs, var_lat=True, forced_wr=forced_wr, batch_id=batch_id))

    def eval_expr(self, expr, hint_dst=None):
        cval = try_eval_const(expr, self.consts)
        if cval is not None:
            if isinstance(cval, float):
                d = self.dst_for("f32", hint_dst)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, im=isa.f32_bits(cval): isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(d.r0,)))
                return d
            d = self.dst_for("u32", hint_dst, non_f32=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, im=int(cval) & 0xFFFFFFFF: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(d.r0,)))
            return d
        kind = expr[0]
        if kind == "ident":
            v = self.lookup_var(expr[1])
            if v is not None: return v
            if expr[1] in self.param_offsets: return self.get_param(expr[1])
            raise ValueError(f"undefined identifier: {expr[1]}")
        if kind in ("index", "field"):
            base = self.eval_expr(expr[1])
            idx = int(try_eval_const(expr[2], self.consts)) if kind == "index" else FIELD_IDX[expr[2]]
            return Val("f32" if base.vtype.startswith("f32") else "u32", (base.regs[idx],))
        if kind == "ternary":
            _, cond, t_e, f_e = expr
            cv = try_eval_const(cond, self.consts)
            if cv is not None: return self.eval_expr(t_e if cv else f_e, hint_dst=hint_dst)
            pred, tv, fc = self.eval_cond(cond, pred=1), self.eval_expr(t_e), try_eval_const(f_e, self.consts)
            d = self.dst_for(tv.vtype, hint_dst)
            if fc is not None and len(d.regs) == 1:
                imm = isa.f32_bits(fc) if tv.vtype == "f32" else (int(fc) & 0xFFFFFFFF)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=tv.r0, im=imm, p=pred: isa.encode_sel_i(r, a, im, p, stall=stall, wait=wait), reads=(tv.r0,), preads=(pred,), writes=(d.r0,)))
                return d
            fv = self.eval_expr(f_e)
            for dr, tr, fr in zip(d.regs, tv.regs, fv.regs):
                self.emit(Op(lambda wr, rd, wait, stall, r=dr, a=tr, b=fr, p=pred: isa.encode_sel_r(r, a, b, p, stall=stall, wait=wait), reads=(tr, fr), preads=(pred,), writes=(dr,)))
            return d
        if kind == "un":
            op, v = expr[1], self.eval_expr(expr[2])
            d = self.dst_for(v.vtype, hint_dst)
            if op == "-":
                if v.vtype == "f32":
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_lop3_i(r, a, 0x80000000, lut=0x3C, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,)))
                else:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, b=v.r0: isa.encode_iadd3_r(r, isa.RZ, b, neg_b=True, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,)))
            elif op == "~":
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_lop3_i(r, a, 0, lut=0x0F, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,)))
            else:
                raise ValueError(f"unsupported unary op: {op}")
            return d
        if kind == "bin": return self.eval_bin(expr[1], expr[2], expr[3], hint_dst=hint_dst)
        if kind == "call": return self.eval_call(expr[1], expr[2], hint_dst=hint_dst)
        raise ValueError(f"unsupported expr: {expr}")

    def eval_bin(self, op, lhs_e, rhs_e, hint_dst=None):
        lhs, rc = self.eval_expr(lhs_e), try_eval_const(rhs_e, self.consts)
        if lhs.vtype == "f32":
            d = self.dst_for("f32", hint_dst)
            if op == "/":
                rhs, inv = self.eval_expr(rhs_e), self.alloc_val("f32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=inv.r0, b=rhs.r0: isa.encode_mufu(r, b, op=isa.MUFU_RCP, wr=wr, stall=stall, wait=wait), reads=(rhs.r0,), writes=(inv.r0,), var_lat=True))
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=inv.r0: isa.encode_fmul_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, inv.r0), writes=(d.r0,)))
                return d
            if op not in ("+", "-", "*"): raise ValueError(f"unsupported f32 op: {op}")
            if rc is not None:
                fv = -float(rc) if op == "-" else float(rc)
                fn = isa.encode_fmul_i if op == "*" else isa.encode_fadd_i
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, f=fv, enc=fn: enc(r, a, f, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,), pipe="fp32"))
                return d
            rhs = self.eval_expr(rhs_e)
            if op == "-":
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_fadd_r(r, a, b, neg_b=True, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,), pipe="fp32"))
            else:
                fn = isa.encode_fmul_r if op == "*" else isa.encode_fadd_r
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0, enc=fn: enc(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,), pipe="fp32"))
            return d
        d = self.dst_for("u32", hint_dst, non_f32=True)
        if op in ("+", "-", "*"):
            if rc is not None:
                im = (-int(rc) if op == "-" else int(rc)) & 0xFFFFFFFF
                fn = isa.encode_imad_i if op == "*" else isa.encode_iadd3_i
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=im, enc=fn: enc(r, a, i, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            if op == "-":
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_iadd3_r(r, a, b, neg_b=True, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            else:
                fn = isa.encode_imad_r if op == "*" else isa.encode_iadd3_r
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0, enc=fn: enc(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op in ("<<", ">>"):
            if rc is not None:
                sh = int(rc) & 31
                if sh == 0: return lhs
                if op == "<<":
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, s=sh: isa.encode_shf_l_i(r, a, s, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                else:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, s=sh, c=lhs.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            if op == "<<":
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_shf_l_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            else:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, b=rhs.r0, c=lhs.r0: isa.encode_shf_r_u32_hi_r(r, b, c, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op in ("&", "|", "^"):
            lut = {"&": 0xC0, "|": 0xFC, "^": 0x3C}[op]
            if rc is not None:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=int(rc) & 0xFFFFFFFF, l=lut: isa.encode_lop3_i(r, a, i, lut=l, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
            else:
                rhs = self.eval_expr(rhs_e)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0, l=lut: isa.encode_lop3_r(r, a, b, lut=l, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        raise ValueError(f"unsupported u32 op: {op}")

    def _emit_hadd2_quad(self, d, h0, h1):
        for i, (hr, hi_h) in enumerate(((h0.r0, False), (h0.r0, True), (h1.r0, False), (h1.r0, True))):
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[i], rb=hr, hh=hi_h: isa.encode_hadd2_f32_half(r, rb, hi_half=hh, stall=stall, wait=wait), reads=(hr,), writes=(d.regs[i],), pipe="fp32"))

    def eval_call(self, name, args, hint_dst=None):
        if name in SR_CALLS: return self.get_sr(name)
        if name in ("pdl_wait", "acqbulk"):
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_acqbulk(stall=stall, wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))
        if name in ("pdl_release", "preexit"):
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_preexit(stall=max(stall, 8), wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))
        if name in ("membar_sys", "membar_all_sys"):
            mb = isa.MEMBAR_SC_SYS if name == "membar_sys" else isa.MEMBAR_ALL_SYS
            self.emit(Op(lambda wr, rd, wait, stall, m=mb: isa.encode_membar(m, stall=max(stall, 6), wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))
        if name == "ldgsts128":
            base_off, _, elem_bytes, _ = self.smem[args[0][1]]
            idx_c = try_eval_const(args[1], self.consts)
            ptr, goff = self.eval_addr_offset(args[2])
            ur, rsm = self.ur_desc, self.alloc_val("u32", temp=True)
            if idx_c is not None:
                self.emit(Op(lambda wr, rd, wait, stall, r=rsm.r0, im=(base_off + int(idx_c) * elem_bytes) & 0xFFFFFFFF: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(rsm.r0,)))
            else:
                idx_v = self.eval_expr(args[1])
                self.emit(Op(lambda wr, rd, wait, stall, r=rsm.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(r, a, eb, stall=stall, wait=wait), reads=(idx_v.r0,), writes=(rsm.r0,)))
                if base_off != 0:
                    self.emit(Op(lambda wr, rd, wait, stall, r=rsm.r0, a=rsm.r0, bo=base_off: isa.encode_iadd3_i(r, a, bo, stall=stall, wait=wait), reads=(rsm.r0,), writes=(rsm.r0,)))
            addr_reads = (ptr.regs[0], ptr.regs[1], rsm.r0)
            self.recent_mem_addr = set(addr_reads)
            self.emit(Op(lambda wr, rd, wait, stall, sm=rsm.r0, pa=ptr.regs[0], u=ur, o=goff: isa.encode_ldgsts128(sm, pa, u, offset=o, rd_bar=rd, stall=stall, wait=wait), reads=addr_reads, mem_addr_reads=addr_reads, ureads=(ur, ur + 1), is_store=True))
            return Val("u32", (isa.RZ,))
        if name == "ldgdepbar":
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_ldgdepbar(wr=0, stall=stall, wait=wait), is_bar=True, forced_wr=0))
            return Val("u32", (isa.RZ,))
        if name == "depbar_le":
            cnt = int(try_eval_const(args[0], self.consts))
            self.emit(Op(lambda wr, rd, wait, stall, c=cnt: isa.encode_depbar_le(count=c, stall=max(stall, 7), wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))
        if name == "hmma16816_bf16":
            a, b, c = self.eval_expr(args[0]), self.eval_expr(args[1]), self.eval_expr(args[2])
            d = self.dst_for("f32x4", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, dr=d.r0, ar=a.r0, br=b.r0, cr=c.r0: isa.encode_hmma_16816_f32_bf16(dr, ar, br, cr, stall=stall, wait=wait), reads=a.regs + b.regs + c.regs, writes=d.regs, fixed_lat=14, pipe="mma"))
            return d
        if name == "barrier":
            self.barriers = max(self.barriers, 1)
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_bar_sync(0, stall=stall, wait=wait), is_bar=True))
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=max(stall, 1), yield_=True, wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))
        if name == "prefetch_l2": return Val("u32", (isa.RZ,))
        if name == "ptr_add":
            base = self.eval_expr(args[0])
            idx_c, str_c = try_eval_const(args[1], self.consts), try_eval_const(args[2], self.consts)
            if idx_c is not None and str_c is not None:
                byte_off = int(idx_c) * int(str_c)
                if byte_off == 0: return base
                d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u64") else self.alloc_val("u64", temp=True, avoid=self.recent_mem_addr)
                tmp = self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=tmp.r0, im=byte_off & 0xFFFFFFFF: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(tmp.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=tmp.r0, c=base.r0: isa.encode_imad_wide_i(r, a, 1, c, stall=stall, wait=wait), reads=(tmp.r0, base.regs[0], base.regs[1]), writes=d.regs))
                return d
            idx_v = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u64") else self.alloc_val("u64", temp=True, avoid=self.recent_mem_addr)
            if str_c is not None:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=idx_v.r0, im=int(str_c) & 0xFFFFFFFF, c=base.r0: isa.encode_imad_wide_i(r, a, im, c, stall=stall, wait=wait), reads=(idx_v.r0, base.regs[0], base.regs[1]), writes=d.regs))
            else:
                str_v = self.eval_expr(args[2])
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=idx_v.r0, b=str_v.r0, c=base.r0: isa.encode_imad_wide_r(r, a, b, c, stall=stall, wait=wait), reads=(idx_v.r0, str_v.r0, base.regs[0], base.regs[1]), writes=d.regs))
            return d
        if name in LOAD_CALLS:
            vtype, _ = LOAD_CALLS[name]
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == TYPE_REGS[vtype]) else self.alloc_val(vtype, temp=True)
            self.emit_load_call(name, args, d)
            return d
        if name in STORE_CALLS:
            self.emit_store(name, args)
            return Val("u32", (isa.RZ,))
        if name in ("param_u32", "param_f32", "param_u64"):
            pname = args[0][1]
            c_off, _, pbytes = self.param_offsets[pname]
            boff, width = int(try_eval_const(args[1], self.consts)), 64 if name == "param_u64" else 32
            if boff < 0 or boff + width // 8 > pbytes or (boff % (width // 8)) != 0:
                raise ValueError(f"{name} offset {boff} out of bounds or misaligned for {pname} ({pbytes}B)")
            d = self.dst_for({"param_u32": "u32", "param_f32": "f32", "param_u64": "u64"}[name], hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, o=c_off + boff, w=width: isa.encode_ldc(r, o, width=w, wr=wr, stall=stall, wait=wait), writes=d.regs, var_lat=True))
            return d
        if name in ("dyn_lds32", "dyn_lds32_f32"):
            vtype = "f32" if name == "dyn_lds32_f32" else "u32"
            return self._emit_smem_load((self.smem_bytes + 15) & ~15, 4, args[0], self.dst_for(vtype, hint_dst, any_scalar=True))
        if name in ("lds128", "lds32"):
            base_off, vtype, elem_bytes, _ = self.smem[args[0][1]]
            if name == "lds128":
                return self._emit_smem_load(base_off, elem_bytes, args[1], self.dst_for("f32x4" if vtype == "f32" else "u32x4", hint_dst), is_128=True)
            return self._emit_smem_load(base_off, elem_bytes, args[1], self.dst_for(vtype, hint_dst, any_scalar=True))
        if name == "fma":
            d, a, bc, c = self.dst_for("f32", hint_dst), self.eval_expr(args[0]), try_eval_const(args[1], self.consts), self.eval_expr(args[2])
            if bc is not None:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, f=float(bc), rc=c.r0: isa.encode_ffma_i(r, ra, f, rc, stall=stall, wait=wait), reads=(a.r0, c.r0), writes=(d.r0,), pipe="fp32"))
            else:
                b = self.eval_expr(args[1])
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, rb=b.r0, rc=c.r0: isa.encode_ffma_r(r, ra, rb, rc, stall=stall, wait=wait), reads=(a.r0, b.r0, c.r0), writes=(d.r0,), pipe="fp32"))
            return d
        if name in ("maxf", "minf"):
            a, b, d = self.eval_expr(args[0]), self.eval_expr(args[1]), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, rb=b.r0, m=(name == "maxf"): isa.encode_fmnmx_r(r, ra, rb, is_max=m, stall=stall, wait=wait), reads=(a.r0, b.r0), writes=(d.r0,)))
            return d
        if name == "fabsf":
            a, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_lop3_i(r, ra, 0x7FFFFFFF, lut=0xC0, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), pipe="lop3"))
            return d
        if name in ("rcp", "rsqrt", "exp2_ftz"):
            a, d, op = self.eval_expr(args[0]), self.dst_for("f32", hint_dst), {"rcp": isa.MUFU_RCP, "rsqrt": isa.MUFU_RSQ, "exp2_ftz": isa.MUFU_EX2}[name]
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=a.r0, o=op: isa.encode_mufu(r, rb, op=o, wr=wr, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), var_lat=True))
            return d
        if name == "expf_fast":
            a, scaled, d = self.eval_expr(args[0]), self.alloc_val("f32", temp=True), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=scaled.r0, ra=a.r0: isa.encode_fmul_i(r, ra, 1.4426950408889634, stall=stall, wait=wait), reads=(a.r0,), writes=(scaled.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=scaled.r0: isa.encode_mufu(r, rb, op=isa.MUFU_EX2, wr=wr, stall=stall, wait=wait), reads=(scaled.r0,), writes=(d.r0,), var_lat=True))
            return d
        if name == "silu":
            x = self.eval_expr(args[0])
            t0, t1, t2, t3, d = self.alloc_val("f32", temp=True), self.alloc_val("f32", temp=True), self.alloc_val("f32", temp=True), self.alloc_val("f32", temp=True), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=t0.r0, ra=x.r0: isa.encode_fmul_i(r, ra, -1.4426950408889634, stall=stall, wait=wait), reads=(x.r0,), writes=(t0.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t1.r0, rb=t0.r0: isa.encode_mufu(r, rb, op=isa.MUFU_EX2, wr=wr, stall=stall, wait=wait), reads=(t0.r0,), writes=(t1.r0,), var_lat=True))
            self.emit(Op(lambda wr, rd, wait, stall, r=t2.r0, ra=t1.r0: isa.encode_fadd_i(r, ra, 1.0, stall=stall, wait=wait), reads=(t1.r0,), writes=(t2.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t3.r0, rb=t2.r0: isa.encode_mufu(r, rb, op=isa.MUFU_RCP, wr=wr, stall=stall, wait=wait), reads=(t2.r0,), writes=(t3.r0,), var_lat=True))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=x.r0, rb=t3.r0: isa.encode_fmul_r(r, ra, rb, stall=stall, wait=wait), reads=(x.r0, t3.r0), writes=(d.r0,)))
            return d
        if name == "u32_to_f32":
            a, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=a.r0: isa.encode_i2fp_u32(r, rb, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d
        if name in ("u32_as_f32", "f32_as_u32"):
            return Val("f32" if name == "u32_as_f32" else "u32", self.eval_expr(args[0], hint_dst=hint_dst).regs)
        if name in ("bf16_lo", "bf16_to_f32"):
            a, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_shf_l_i(r, ra, 16, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d
        if name == "bf16_hi":
            a, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_lop3_i(r, ra, 0xFFFF0000, lut=0xC0, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), pipe="lop3"))
            return d
        if name == "f32_to_bf16":
            a, d = self.eval_expr(args[0]), self.dst_for("u32", hint_dst, any_scalar=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rlo=a.r0: isa.encode_f2fp_bf16_pack(r, isa.RZ, rlo, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d
        if name in ("pack_bf16x2", "cvt_f32x2_e4m3x2", "pack_u16x2"):
            lo, hi, d = self.eval_expr(args[0]), self.eval_expr(args[1]), self.dst_for("u32", hint_dst, any_scalar=True)
            if name == "pack_u16x2":
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lo.r0, c=hi.r0: isa.encode_prmt_i(r, a, 0x5410, c, stall=stall, wait=wait), reads=(lo.r0, hi.r0), writes=(d.r0,)))
            else:
                fn = isa.encode_f2fp_bf16_pack if name == "pack_bf16x2" else isa.encode_f2fp_e4m3_pack
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rhi=hi.r0, rlo=lo.r0, enc=fn: enc(r, rhi, rlo, stall=stall, wait=wait), reads=(hi.r0, lo.r0), writes=(d.r0,)))
            return d
        if name == "cvt_e2m1x2_f32x2":
            b, d, fp16x2 = self.eval_expr(args[0]), self.dst_for("f32x2", hint_dst), self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=fp16x2.r0, rb=b.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b.r0,), writes=(fp16x2.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[0], rb=fp16x2.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=False, stall=stall, wait=wait), reads=(fp16x2.r0,), writes=(d.regs[0],), pipe="fp32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[1], rb=fp16x2.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=True, stall=stall, wait=wait), reads=(fp16x2.r0,), writes=(d.regs[1],), pipe="fp32"))
            return d
        if name == "cvt_e2m1x4_f32x4":
            w, sh, d = self.eval_expr(args[0]), int(try_eval_const(args[1], self.consts)), self.dst_for("f32x4", hint_dst)
            h0, h1 = self.alloc_val("u32", temp=True), self.alloc_val("u32", temp=True)
            if sh == 0:
                b1 = self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=b1.r0, c=w.r0: isa.encode_shf_r_u32_hi_i(r, 8, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b1.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h0.r0, rb=w.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(w.r0,), writes=(h0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h1.r0, rb=b1.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b1.r0,), writes=(h1.r0,)))
            else:
                b0, b1 = self.alloc_val("u32", temp=True), self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=b0.r0, s=sh, c=w.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=b1.r0, s=sh + 8, c=w.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b1.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h0.r0, rb=b0.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b0.r0,), writes=(h0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h1.r0, rb=b1.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b1.r0,), writes=(h1.r0,)))
            self._emit_hadd2_quad(d, h0, h1)
            return d
        if name == "cvt_e4m3x4_f32x4":
            w, d = self.eval_expr(args[0]), self.dst_for("f32x4", hint_dst)
            h0, h1 = self.alloc_val("u32", temp=True), self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=h0.r0, rb=w.r0: isa.encode_f2fp_e4m3_unpack(r, rb, hi_half=False, stall=stall, wait=wait), reads=(w.r0,), writes=(h0.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=h1.r0, rb=w.r0: isa.encode_f2fp_e4m3_unpack(r, rb, hi_half=True, stall=stall, wait=wait), reads=(w.r0,), writes=(h1.r0,)))
            self._emit_hadd2_quad(d, h0, h1)
            return d
        if name == "e4m3_to_f32":
            b, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            sign, absb, exp, mant, sub_s = (self.alloc_val(t, temp=True) for t in ("u32", "u32", "u32", "u32", "f32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=sign.r0, a=b.r0: isa.encode_lop3_i(r, a, 0x80, lut=0xC0, stall=stall, wait=wait), reads=(b.r0,), writes=(sign.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sign.r0, a=sign.r0: isa.encode_shf_l_i(r, a, 24, stall=stall, wait=wait), reads=(sign.r0,), writes=(sign.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=absb.r0, a=b.r0: isa.encode_lop3_i(r, a, 0x7F, lut=0xC0, stall=stall, wait=wait), reads=(b.r0,), writes=(absb.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, c=absb.r0: isa.encode_shf_r_u32_hi_i(r, 3, c, stall=stall, wait=wait), reads=(absb.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=mant.r0, a=absb.r0: isa.encode_lop3_i(r, a, 0x07, lut=0xC0, stall=stall, wait=wait), reads=(absb.r0,), writes=(mant.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sub_s.r0, rb=mant.r0: isa.encode_i2fp_u32(r, rb, stall=stall, wait=wait), reads=(mant.r0,), writes=(sub_s.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sub_s.r0, ra=sub_s.r0: isa.encode_fmul_i(r, ra, 1.0 / 512.0, stall=stall, wait=wait), reads=(sub_s.r0,), writes=(sub_s.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=exp.r0: isa.encode_isetp_i(2, a, 0, cmp_op=isa.CMP_NE, unsigned=True, stall=stall, wait=wait), reads=(exp.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0: isa.encode_iadd3_i(r, a, 120, stall=stall, wait=wait), reads=(exp.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0: isa.encode_shf_l_i(r, a, 23, stall=stall, wait=wait), reads=(exp.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=mant.r0, a=mant.r0: isa.encode_shf_l_i(r, a, 20, stall=stall, wait=wait), reads=(mant.r0,), writes=(mant.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0, c=mant.r0: isa.encode_lop3_r(r, a, c, lut=0xFC, stall=stall, wait=wait), reads=(exp.r0, mant.r0), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0, c=sub_s.r0: isa.encode_sel_r(r, a, c, 2, stall=stall, wait=wait), reads=(exp.r0, sub_s.r0), preads=(2,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=absb.r0: isa.encode_isetp_i(2, a, 0x7F, cmp_op=isa.CMP_EQ, unsigned=True, stall=stall, wait=wait), reads=(absb.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0: isa.encode_sel_i(r, a, 0x7FC00000, 2, neg_sp=True, stall=stall, wait=wait), reads=(exp.r0,), preads=(2,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=exp.r0, c=sign.r0: isa.encode_lop3_r(r, a, c, lut=0xFC, stall=stall, wait=wait), reads=(exp.r0, sign.r0), writes=(d.r0,)))
            return d
        if name == "ue4m3_to_f32":
            b, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            exp, mant, e_adj, e_sh, m_sh, norm_bits, sub_f, sub_s = (self.alloc_val(t, temp=True) for t in ("u32", "u32", "u32", "u32", "u32", "u32", "f32", "f32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, c=b.r0: isa.encode_shf_r_u32_hi_i(r, 3, c, stall=stall, wait=wait), reads=(b.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, a=exp.r0: isa.encode_lop3_i(r, a, 0x0F, lut=0xC0, stall=stall, wait=wait), reads=(exp.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=mant.r0, a=b.r0: isa.encode_lop3_i(r, a, 0x07, lut=0xC0, stall=stall, wait=wait), reads=(b.r0,), writes=(mant.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=e_adj.r0, a=exp.r0: isa.encode_iadd3_i(r, a, 120, stall=stall, wait=wait), reads=(exp.r0,), writes=(e_adj.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=e_sh.r0, a=e_adj.r0: isa.encode_shf_l_i(r, a, 23, stall=stall, wait=wait), reads=(e_adj.r0,), writes=(e_sh.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=m_sh.r0, a=mant.r0: isa.encode_shf_l_i(r, a, 20, stall=stall, wait=wait), reads=(mant.r0,), writes=(m_sh.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=norm_bits.r0, a=e_sh.r0, c=m_sh.r0: isa.encode_lop3_r(r, a, c, lut=0xFC, stall=stall, wait=wait), reads=(e_sh.r0, m_sh.r0), writes=(norm_bits.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sub_f.r0, rb=mant.r0: isa.encode_i2fp_u32(r, rb, stall=stall, wait=wait), reads=(mant.r0,), writes=(sub_f.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sub_s.r0, ra=sub_f.r0: isa.encode_fmul_i(r, ra, 1.0 / 512.0, stall=stall, wait=wait), reads=(sub_f.r0,), writes=(sub_s.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=exp.r0: isa.encode_isetp_i(2, a, 0, cmp_op=isa.CMP_NE, unsigned=True, stall=stall, wait=wait), reads=(exp.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=norm_bits.r0, c=sub_s.r0: isa.encode_sel_r(r, a, c, 2, stall=stall, wait=wait), reads=(norm_bits.r0, sub_s.r0), preads=(2,), writes=(d.r0,)))
            return d
        if name == "ue8m0_to_f32":
            b, d = self.eval_expr(args[0]), self.dst_for("f32", hint_dst)
            eb, sh, t1 = (self.alloc_val("u32", temp=True) for _ in range(3))
            self.emit(Op(lambda wr, rd, wait, stall, r=eb.r0, a=b.r0: isa.encode_lop3_i(r, a, 0xFF, lut=0xC0, stall=stall, wait=wait), reads=(b.r0,), writes=(eb.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sh.r0, a=eb.r0: isa.encode_shf_l_i(r, a, 23, stall=stall, wait=wait), reads=(eb.r0,), writes=(sh.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=eb.r0: isa.encode_isetp_i(2, a, 0, cmp_op=isa.CMP_EQ, unsigned=True, stall=stall, wait=wait), reads=(eb.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t1.r0, a=sh.r0: isa.encode_sel_i(r, a, 0x00400000, 2, neg_sp=True, stall=stall, wait=wait), reads=(sh.r0,), preads=(2,), writes=(t1.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=eb.r0: isa.encode_isetp_i(2, a, 255, cmp_op=isa.CMP_EQ, unsigned=True, stall=stall, wait=wait), reads=(eb.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=t1.r0: isa.encode_sel_i(r, a, 0, 2, neg_sp=True, stall=stall, wait=wait), reads=(t1.r0,), preads=(2,), writes=(d.r0,)))
            return d
        if name == "pow2_ceil_e8m0_bits":
            amax, d = self.eval_expr(args[0]), self.dst_for("u32", hint_dst, any_scalar=True)
            sc, b_add, exp, cl = (self.alloc_val(t, temp=True) for t in ("f32", "u32", "u32", "u32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=sc.r0, ra=amax.r0: isa.encode_fmul_i(r, ra, 1.0 / 448.0, stall=stall, wait=wait), reads=(amax.r0,), writes=(sc.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=b_add.r0, ra=sc.r0: isa.encode_iadd3_i(r, ra, 0x007FFFFF, stall=stall, wait=wait), reads=(sc.r0,), writes=(b_add.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, c=b_add.r0: isa.encode_shf_r_u32_hi_i(r, 23, c, stall=stall, wait=wait), reads=(b_add.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=exp.r0: isa.encode_isetp_i(2, a, 254, cmp_op=isa.CMP_GT, unsigned=True, stall=stall, wait=wait), reads=(exp.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=cl.r0, a=exp.r0: isa.encode_sel_i(r, a, 254, 2, neg_sp=True, stall=stall, wait=wait), reads=(exp.r0,), preads=(2,), writes=(cl.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=amax.r0: isa.encode_fsetp_i(2, a, 0.0, cmp_op=isa.CMP_GT, stall=stall, wait=wait), reads=(amax.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=cl.r0: isa.encode_sel_i(r, a, 0, 2, neg_sp=False, stall=stall, wait=wait), reads=(cl.r0,), preads=(2,), writes=(d.r0,)))
            return d
        if name in PACK_TYPES:
            vals, d = [self.eval_expr(a) for a in args], self.dst_for(PACK_TYPES[name], hint_dst)
            for dr, v in zip(d.regs, vals):
                self.emit(Op(lambda wr, rd, wait, stall, r=dr, s=v.r0: isa.encode_mov_r(r, s, stall=stall, wait=wait), reads=(v.r0,), writes=(dr,)))
            return d
        if name in SHFL_ENC:
            v, arg1, enc = self.eval_expr(args[0]), int(try_eval_const(args[1], self.consts)), SHFL_ENC[name]
            d = self.dst_for(v.vtype, hint_dst)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0, m=arg1, fn=enc: fn(r, a, m, wr=wr, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,), var_lat=True))
            return d
        if name == "shfl_idx":
            v, lane_c = self.eval_expr(args[0]), try_eval_const(args[1], self.consts)
            d = self.dst_for(v.vtype, hint_dst)
            if lane_c is not None and int(lane_c) == 0:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_shfl_idx(r, a, isa.RZ, wr=wr, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,), var_lat=True))
            else:
                lv = self.eval_expr(args[1])
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0, l=lv.r0: isa.encode_shfl_idx(r, a, l, wr=wr, stall=stall, wait=wait), reads=(v.r0, lv.r0), writes=(d.r0,), var_lat=True))
            return d
        if name in ("warp_sum", "warp_max", "oct_sum", "oct_max", "quad_max", "quad_sum"):
            d = self.dst_for("f32", hint_dst)
            v = self.eval_expr(args[0], hint_dst=d)
            if d.r0 != v.r0:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, s=v.r0: isa.encode_mov_r(r, s, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,)))
            tmp = self.alloc_val("f32", temp=True)
            masks = (16, 8, 4, 2, 1) if name.startswith("warp_") else ((4, 2, 1) if name.startswith("oct_") else (2, 1))
            is_max = name.endswith("_max")
            for m in masks:
                self.emit(Op(lambda wr, rd, wait, stall, r=tmp.r0, a=d.r0, mk=m: isa.encode_shfl_bfly(r, a, mk, wr=wr, stall=stall, wait=wait), reads=(d.r0,), writes=(tmp.r0,), var_lat=True))
                if is_max:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=d.r0, b=tmp.r0: isa.encode_fmnmx_r(r, a, b, is_max=True, stall=stall, wait=wait), reads=(d.r0, tmp.r0), writes=(d.r0,)))
                else:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=d.r0, b=tmp.r0: isa.encode_fadd_r(r, a, b, stall=stall, wait=wait), reads=(d.r0, tmp.r0), writes=(d.r0,), pipe="fp32"))
            return d
        raise ValueError(f"unknown builtin: {name}")

    def list_schedule_block(self, block):
        n = len(block)
        if n <= 2: return block
        succ, indeg = [{} for _ in range(n)], [0] * n
        for i in range(n):
            oi = block[i]
            for j in range(i + 1, n):
                oj, lat = block[j], 0
                if (oi.writes & oj.reads) or (oi.uwrites & oj.ureads):
                    if oi.var_lat: lat = 20 if oi.mem_addr_reads else 14
                    elif oi.pipe == "fp32" and oj.pipe == "fp32": lat = 4
                    elif oj.mem_addr_reads & oi.writes: lat = oi.fixed_lat + 1
                    else: lat = oi.fixed_lat
                if oi.pwrites & oj.preads: lat = max(lat, 13 if oj.mem_addr_reads else 5)
                if (oi.reads & oj.writes) or (oi.ureads & oj.uwrites) or (oi.preads & oj.pwrites):
                    lat = max(lat, 6 if ((oi.mem_addr_reads & oj.writes) or (oi.var_lat and (oi.reads & oj.writes))) else 1)
                if (oi.writes & oj.writes) or (oi.uwrites & oj.uwrites) or (oi.pwrites & oj.pwrites): lat = max(lat, 1)
                if (oi.is_store and (oj.is_store or oj.var_lat)) or (oi.var_lat and oj.is_store): lat = max(lat, 1)
                if oi.forced_wr is not None and oj.forced_wr is not None and oi.forced_wr != oj.forced_wr: lat = max(lat, 1)
                if lat > 0:
                    succ[i][j] = lat
                    indeg[j] += 1
        height = [1] * n
        for i in range(n - 1, -1, -1):
            height[i] = max((lat + height[j] for j, lat in succ[i].items()), default=1)
        ready, earliest, cur_cycle, out = [i for i in range(n) if indeg[i] == 0], [0] * n, 0, []
        while ready:
            best_pos = min(range(len(ready)), key=lambda pos: (max(0, earliest[ready[pos]] - cur_cycle), 0 if (block[ready[pos]].var_lat and not block[ready[pos]].is_store) else 1, -height[ready[pos]], ready[pos]))
            chosen = ready.pop(best_pos)
            out.append(block[chosen])
            issue_cyc = max(cur_cycle, earliest[chosen])
            for j, lat in succ[chosen].items():
                if issue_cyc + lat > earliest[j]: earliest[j] = issue_cyc + lat
                indeg[j] -= 1
                if indeg[j] == 0: ready.append(j)
            cur_cycle = issue_cyc + 1
        return out

    def reorder_ops(self):
        out, block = [], []
        for op in self.ops:
            if op.label is not None or op.is_branch or op.is_exit or op.is_bar or op.forced_wait != 0:
                if block:
                    out.extend(self.list_schedule_block(block))
                    block = []
                out.append(op)
            else:
                block.append(op)
        if block: out.extend(self.list_schedule_block(block))
        self.ops = out

    def schedule_and_encode(self):
        self.reorder_ops()
        real_ops, reg_bar, ureg_bar, rd_bar, bar_batch, bar_count = [], {}, {}, {}, {}, [0] * 6
        has_forced = any(op.forced_wr is not None and op.forced_wr >= 4 for op in self.ops)
        reserve_sb0 = any(op.forced_wr == 0 for op in self.ops)
        dyn_bars = 4 if has_forced else 6
        rd_slot, wr_bars, min_wr, bar_cap = dyn_bars - 1, dyn_bars - 1, 1 if reserve_sb0 else 0, 28
        next_bar = min_wr

        def clear_bars(mask):
            nonlocal reg_bar, ureg_bar, rd_bar
            reg_bar = {k: v for k, v in reg_bar.items() if not (mask & (1 << v))}
            ureg_bar = {k: v for k, v in ureg_bar.items() if not (mask & (1 << v))}
            rd_bar = {k: v for k, v in rd_bar.items() if not (mask & (1 << v))}
            for b in range(6):
                if mask & (1 << b):
                    bar_batch.pop(b, None)
                    bar_count[b] = 0

        for op in self.ops:
            all_b = set(reg_bar.values()) | set(ureg_bar.values()) | set(rd_bar.values())
            if op.label is not None:
                pending_dyn = sum(1 << b for b in all_b if not has_forced or b < 4)
                if pending_dyn:
                    prev = real_ops[-1] if real_ops else None
                    if prev is not None and prev.label is None and not prev.preads and not prev.var_lat and not prev.mem_addr_reads and not prev.is_branch:
                        prev.wait |= pending_dyn
                    else:
                        nop = Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=stall, wait=wait), forced_wait=pending_dyn)
                        nop.wait = pending_dyn
                        real_ops.append(nop)
                clear_bars(0x0F if has_forced else 0x3F)
                real_ops.append(op)
                continue
            wait_mask = op.forced_wait
            for r in op.reads | op.writes:
                if r in reg_bar: wait_mask |= 1 << reg_bar[r]
            for r in op.writes:
                if r in rd_bar: wait_mask |= 1 << rd_bar[r]
            for u in op.ureads | op.uwrites:
                if u in ureg_bar: wait_mask |= 1 << ureg_bar[u]
            if op.is_bar or (op.is_exit and not op.preads):
                for b in all_b: wait_mask |= 1 << b
            elif op.is_branch:
                pending_b = sum(1 << b for b in all_b if not has_forced or b < 4)
                if not op.preads:
                    wait_mask |= pending_b
                elif pending_b:
                    nop = Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=stall, wait=wait), forced_wait=pending_b)
                    nop.wait = pending_b
                    real_ops.append(nop)
                    clear_bars(pending_b)
            if op.var_lat:
                if op.forced_wr is not None:
                    b = op.forced_wr
                    if not (op.batch_id is not None and bar_batch.get(b) == op.batch_id) and b in all_b:
                        wait_mask |= 1 << b
                else:
                    b, span = None, wr_bars - min_wr
                    for step in range(span):
                        cand = min_wr + ((next_bar - min_wr + step) % span)
                        if bar_count[cand] == 0 and not (wait_mask & (1 << cand)):
                            b, next_bar = cand, min_wr + ((cand - min_wr + 1) % span)
                            break
                    if b is None:
                        for cand in range(wr_bars - 1, min_wr - 1, -1):
                            if bar_count[cand] < bar_cap and not (wait_mask & (1 << cand)):
                                b = cand
                                break
                    if b is None:
                        b, next_bar = next_bar, min_wr + ((next_bar - min_wr + 1) % span)
                        wait_mask |= 1 << b
                op.wr = b
            if op.mem_addr_reads:
                if bar_count[rd_slot] >= bar_cap and not (wait_mask & (1 << rd_slot)):
                    wait_mask |= 1 << rd_slot
                op.rd = rd_slot
            self_conflict = ((1 << op.wr) if op.var_lat and (wait_mask & (1 << op.wr)) else 0) | ((1 << op.rd) if op.mem_addr_reads and (wait_mask & (1 << op.rd)) else 0)
            if self_conflict:
                nop = Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=stall, wait=wait), forced_wait=self_conflict)
                nop.wait = self_conflict
                real_ops.append(nop)
                wait_mask &= ~self_conflict
                clear_bars(self_conflict)
            op.wait = wait_mask
            if wait_mask and not op.preads: clear_bars(wait_mask)
            if op.var_lat:
                for r in op.writes: reg_bar[r] = op.wr
                for u in op.uwrites: ureg_bar[u] = op.wr
                bar_batch[op.wr] = op.batch_id
                bar_count[op.wr] += 1
            if op.mem_addr_reads:
                for r in op.mem_addr_reads: rd_bar[r] = op.rd
                bar_count[op.rd] += 1
            real_ops.append(op)

        ready_r, war_ready_r, ready_p, bar_issue_cycle, cur_cycle, last_real = {}, {}, {}, {}, 0, None
        for op in real_ops:
            if op.label is not None:
                if last_real is not None and last_real.stall < 6: last_real.stall = 6
                ready_r.clear(); war_ready_r.clear(); ready_p.clear()
                cur_cycle += 6
                last_real = None
                continue
            need = cur_cycle
            for r in op.reads:
                if r in ready_r:
                    p_cyc, p_pipe, p_lat = ready_r[r]
                    need = max(need, p_cyc + (4 if p_pipe == "fp32" and op.pipe == "fp32" else p_lat))
            for r in op.writes:
                if r in war_ready_r: need = max(need, war_ready_r[r])
            for r in op.mem_addr_reads:
                if r in ready_r: need = max(need, ready_r[r][0] + ready_r[r][2] + 1)
            for p in op.preads:
                if p in ready_p:
                    req = ready_p[p]
                    need = max(need, req + 8 if (op.is_branch or op.is_exit or op.mem_addr_reads) else req)
            if op.wait:
                for b in range(6):
                    if (op.wait & (1 << b)) and b in bar_issue_cycle: need = max(need, bar_issue_cycle[b] + 5)
            gap = need - cur_cycle
            if gap > 0 and last_real is not None:
                add = min(gap, 15 - last_real.stall)
                last_real.stall += add
                cur_cycle += add
            op.stall = 5 if (op.is_exit or op.is_branch or op.is_bar) else 1
            for r in op.writes: ready_r[r] = (cur_cycle, op.pipe, op.fixed_lat)
            for r in op.mem_addr_reads: war_ready_r[r] = max(war_ready_r.get(r, 0), cur_cycle + 6)
            if op.var_lat:
                for r in op.reads: war_ready_r[r] = max(war_ready_r.get(r, 0), cur_cycle + 6)
            for p in op.pwrites: ready_p[p] = cur_cycle + 5
            if op.var_lat: bar_issue_cycle[op.wr] = cur_cycle
            if op.mem_addr_reads: bar_issue_cycle[op.rd] = cur_cycle
            cur_cycle += op.stall
            last_real = op

        label_pc, pc = {}, 0
        for op in real_ops:
            if op.label is not None: label_pc[op.label] = pc
            else: pc += 16
        insns, pc = [], 0
        for op in real_ops:
            if op.label is not None: continue
            insns.append(op.encoder(label_pc[op.target_label] - (pc + 16), op.wr, op.rd, op.wait, op.stall) if op.is_branch else op.encoder(op.wr, op.rd, op.wait, op.stall))
            pc += 16
        return insns


def compile_program(prog, strict_reg_budget=False):
    consts = prog["consts"]
    return [KernelCompiler(kdef, consts, strict_reg_budget=strict_reg_budget).compile() for kdef in prog["kernels"]]
