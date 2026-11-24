import isa_sm121 as isa
from cubin_elf import KernelImage
from frontend import try_eval_const

TYPE_REGS = {
    "u8": 1,
    "u16": 1,
    "u32": 1,
    "i32": 1,
    "f32": 1,
    "u64": 2,
    "u32x2": 2,
    "f32x2": 2,
    "u32x4": 4,
    "f32x4": 4,
    "u32x8": 8,
}

TYPE_ALIGN = {
    "u8": 1,
    "u16": 1,
    "u32": 1,
    "i32": 1,
    "f32": 1,
    "u64": 2,
    "u32x2": 2,
    "f32x2": 2,
    "u32x4": 4,
    "f32x4": 4,
    "u32x8": 4,
}

PARAM_SPEC = {
    "u8": ("u8", 1, 1),
    "u16": ("u16", 2, 2),
    "u32": ("u32", 4, 4),
    "i32": ("i32", 4, 4),
    "f32": ("f32", 4, 4),
    "u64": ("u64", 8, 8),
}

LOAD_CALLS = {
    "ldg256_stream": ("u32x8", "256_ef"),
    "ldg256_persist": ("u32x8", "256_el"),
    "ldg256_default": ("u32x8", "256_def"),
    "ldg128_stream": ("u32x4", "128_ef"),
    "ldg128_persist": ("u32x4", "128_const"),
    "ldg64_stream": ("u32x2", "64_ef"),
    "ldg64_persist": ("u32x2", "64_const"),
    "ldg32_stream": ("u32", "32_ef"),
    "ldg32_persist": ("u32", "32_const"),
    "ldg16_stream": ("u32", "16_ef"),
    "ldg16_persist": ("u32", "16_const"),
    "ldg8_stream": ("u32", "8_ef"),
    "ldg8_persist": ("u32", "8_const"),
}


class Val:
    def __init__(self, vtype, regs, is_temp=False):
        self.vtype = vtype
        self.regs = tuple(regs)
        self.is_temp = is_temp

    @property
    def r0(self):
        return self.regs[0]


class Op:
    def __init__(
        self,
        encoder,
        reads=(),
        writes=(),
        ureads=(),
        uwrites=(),
        preads=(),
        pwrites=(),
        var_lat=False,
        is_branch=False,
        is_exit=False,
        is_bar=False,
        target_label=None,
        label=None,
        fixed_lat=5,
        pipe="gen",
        mem_addr_reads=(),
        forced_wr=None,
        forced_wait=0,
        batch_id=None,
    ):
        self.encoder = encoder
        self.reads = set(r for r in reads if r != isa.RZ)
        self.writes = set(r for r in writes if r != isa.RZ)
        self.ureads = set(u for u in ureads if u != isa.URZ)
        self.uwrites = set(u for u in uwrites if u != isa.URZ)
        self.preads = set(p for p in preads if p != isa.PT)
        self.pwrites = set(p for p in pwrites if p != isa.PT)
        self.var_lat = var_lat
        self.is_branch = is_branch
        self.is_exit = is_exit
        self.is_bar = is_bar
        self.target_label = target_label
        self.label = label
        self.fixed_lat = fixed_lat
        self.pipe = pipe
        self.mem_addr_reads = set(r for r in mem_addr_reads if r != isa.RZ)
        self.forced_wr = forced_wr
        self.forced_wait = forced_wait
        self.batch_id = batch_id
        self.wr = 7
        self.rd = 7
        self.wait = 0
        self.stall = 1


class RegAlloc:
    def __init__(self, max_regs=248):
        self.max_regs = max_regs
        self.used = set()
        self.peak = 0

    def alloc(self, count, align=1):
        r = 0
        while r + count <= self.max_regs:
            if r % align == 0 and all((r + k) not in self.used for k in range(count)):
                for k in range(count):
                    self.used.add(r + k)
                if r + count > self.peak:
                    self.peak = r + count
                return tuple(range(r, r + count))
            r += align
        raise RuntimeError(f"register allocator exhausted (need {count} aligned {align}, peak {self.peak})")

    def free(self, regs):
        for r in regs:
            self.used.discard(r)


class KernelCompiler:
    def __init__(self, kdef, consts):
        self.kdef = kdef
        self.consts = dict(consts)
        self.alloc = RegAlloc()
        self.ops = []
        self.scopes = [{}]
        self.smem = {}
        self.smem_bytes = 0
        self.barriers = 0
        self.label_counter = 0
        self.batch_counter = 0
        self.ur_desc = 4
        self.cached_sr = {}
        self.param_offsets = {}
        self.param_meta = []
        self.param_loaded = {}
        self.temp_stack = []

        off = 0
        for p in kdef["params"]:
            if p["is_ptr"]:
                pkind, palign, pbytes = "ptr", 8, 8
                vtype = "u64"
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

    def alloc_val(self, vtype, temp=False):
        regs = self.alloc.alloc(TYPE_REGS[vtype], TYPE_ALIGN[vtype])
        v = Val(vtype, regs, is_temp=temp)
        if temp:
            self.temp_stack[-1].append(v)
        return v

    def push_temps(self):
        self.temp_stack.append([])

    def pop_temps(self, keep=None):
        keep_regs = set(keep.regs) if keep is not None else set()
        for v in self.temp_stack.pop():
            if not any(r in keep_regs for r in v.regs):
                self.alloc.free(v.regs)

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        frame = self.scopes.pop()
        for v in frame.values():
            self.alloc.free(v.regs)

    def lookup_var(self, name):
        for frame in reversed(self.scopes):
            if name in frame:
                return frame[name]
        return None

    def bind_var(self, name, val):
        self.scopes[-1][name] = val

    def get_param(self, name):
        if name in self.param_loaded:
            return self.param_loaded[name]
        c_off, vtype, pbytes = self.param_offsets[name]
        v = self.alloc_val(vtype, temp=False)
        width = 64 if pbytes == 8 else 32
        self.emit(Op(
            lambda wr, rd, wait, stall, r=v.r0, o=c_off, w=width: isa.encode_ldc(r, o, width=w, wr=wr, stall=stall, wait=wait),
            writes=v.regs,
            var_lat=True,
        ))
        self.param_loaded[name] = v
        return v

    def get_sr(self, key):
        if key in self.cached_sr:
            return self.cached_sr[key]
        if key == "ctaid_x":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_s2r(r, isa.SR_CTAID_X, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "ctaid_y":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_s2r(r, isa.SR_CTAID_Y, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "tid_x":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_s2r(r, isa.SR_TID_X, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "lane_id":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_s2r(r, isa.SR_LANEID, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "ntid_x":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_ldc(r, isa.C0_NTID_X, width=32, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "grid_x":
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, r=v.r0: isa.encode_ldc(r, isa.C0_NCTAID_X, width=32, wr=wr, stall=stall, wait=wait), writes=v.regs, var_lat=True))
        elif key == "warp_id":
            tid = self.get_sr("tid_x")
            v = self.alloc_val("u32", temp=False)
            self.emit(Op(lambda wr, rd, wait, stall, d=v.r0, s=tid.r0: isa.encode_shf_r_u32_hi_i(d, 5, s, stall=stall, wait=wait), reads=(tid.r0,), writes=v.regs))
        else:
            raise ValueError(f"unknown sr: {key}")
        self.cached_sr[key] = v
        return v

    def collect_uses(self, node, used_params, used_srs):
        if isinstance(node, list):
            for x in node:
                self.collect_uses(x, used_params, used_srs)
            return
        if not isinstance(node, tuple) or not node:
            return
        tag = node[0]
        if tag == "ident":
            if node[1] in self.param_offsets:
                used_params.add(node[1])
            return
        if tag == "call" and node[1] in ("ctaid_x", "ctaid_y", "tid_x", "lane_id", "ntid_x", "grid_x", "warp_id"):
            used_srs.add(node[1])
        for item in node[1:]:
            self.collect_uses(item, used_params, used_srs)

    def compile(self):
        self.emit(Op(
            lambda wr, rd, wait, stall, u=self.ur_desc: isa.encode_ldcu(u, isa.C0_GLOBAL_DESC, width=64, wr=wr, stall=stall, wait=wait),
            uwrites=(self.ur_desc, self.ur_desc + 1),
            var_lat=True,
        ))
        used_params = set()
        used_srs = set()
        self.collect_uses(self.kdef["body"], used_params, used_srs)
        for p in self.kdef["params"]:
            if p["name"] in used_params:
                self.get_param(p["name"])
        for sr in ("ctaid_x", "ctaid_y", "tid_x", "lane_id", "ntid_x", "grid_x", "warp_id"):
            if sr in used_srs:
                self.get_sr(sr)
        self.compile_stmts(self.kdef["body"])
        self.emit(Op(
            lambda wr, rd, wait, stall: isa.encode_exit(stall=stall, wait=wait),
            is_exit=True,
        ))
        insns = self.schedule_and_encode()
        reg_count = max(self.kdef["regs"], self.alloc.peak + 4, 16)
        reg_count = ((reg_count + 7) // 8) * 8
        return KernelImage(
            name=self.kdef["name"],
            insns=insns,
            params=self.param_meta,
            registers=reg_count,
            max_threads=self.kdef["block"],
            barriers=self.barriers,
            smem_bytes=self.smem_bytes,
            stack_bytes=0,
        )

    def compile_stmts(self, stmts):
        for st in stmts:
            self.compile_stmt(st)

    def compile_stmt(self, st):
        kind = st[0]
        if kind == "shared":
            _, vtype, name, size_expr = st
            count = int(try_eval_const(size_expr, self.consts))
            elem_bytes = 4 * TYPE_REGS[vtype]
            align = max(4, elem_bytes)
            off = (self.smem_bytes + align - 1) // align * align
            self.smem[name] = (off, vtype, elem_bytes, count)
            self.smem_bytes = off + count * elem_bytes
            return
        if kind == "return":
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_exit(stall=stall, wait=wait), is_exit=True))
            return
        if kind == "unroll":
            _, var, start_e, end_e, body = st
            s_val = int(try_eval_const(start_e, self.consts))
            e_val = int(try_eval_const(end_e, self.consts))
            had_old = var in self.consts
            old_val = self.consts.get(var)
            for idx in range(s_val, e_val):
                self.consts[var] = idx
                self.push_scope()
                self.compile_stmts(body)
                self.pop_scope()
            if had_old:
                self.consts[var] = old_val
            else:
                self.consts.pop(var, None)
            return
        if kind == "decl":
            _, _, vtype, name, init = st
            v = self.alloc_val(vtype, temp=False)
            self.bind_var(name, v)
            if init is not None:
                self.push_temps()
                self.eval_into(init, v)
                self.pop_temps(keep=v)
            return
        if kind == "assign":
            _, op, lhs, rhs = st
            self.push_temps()
            self.compile_assign(op, lhs, rhs)
            self.pop_temps()
            return
        if kind == "expr":
            self.push_temps()
            self.eval_expr(st[1])
            self.pop_temps()
            return
        if kind == "if":
            _, cond, then_body, else_body = st
            cval = try_eval_const(cond, self.consts)
            if cval is not None:
                self.push_scope()
                self.compile_stmts(then_body if cval else else_body)
                self.pop_scope()
                return
            self.compile_runtime_if(cond, then_body, else_body)
            return
        if kind == "for":
            self.compile_for(st, pipelined=False)
            return
        if kind == "pipeline_for":
            self.compile_for(st, pipelined=True)
            return
        raise ValueError(f"unsupported stmt: {kind}")

    def compile_runtime_if(self, cond, then_body, else_body):
        if len(then_body) == 1 and then_body[0][0] == "return" and not else_body:
            self.push_temps()
            pred = self.eval_cond(cond, pred=0)
            self.emit(Op(
                lambda wr, rd, wait, stall, p=pred: isa.encode_exit(pred=p, neg=False, stall=stall, wait=wait),
                preads=(pred,),
                is_exit=True,
            ))
            self.pop_temps()
            return
        if not else_body and all(
            st[0] == "expr" and st[1][0] == "call" and st[1][1] in ("stg256", "stg128", "stg64", "stg32", "stg16", "stg8", "sts32")
            for st in then_body
        ):
            self.push_temps()
            pred = self.eval_cond(cond, pred=0)
            for st in then_body:
                self.emit_pred_store(st[1][1], st[1][2], pred)
            self.pop_temps()
            return
        lbl_else = self.new_label("ELSE")
        lbl_end = self.new_label("ENDIF")
        lbl_reconv = self.new_label("RECONV")
        self.push_temps()
        self.emit(Op(
            lambda delta, wr, rd, wait, stall: isa.encode_bssy(delta, bar=0, stall=stall, wait=wait),
            is_branch=True,
            target_label=lbl_reconv,
        ))
        pred = self.eval_cond(cond, pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=pred: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait),
            preads=(pred,),
            is_branch=True,
            target_label=lbl_else if else_body else lbl_end,
        ))
        self.pop_temps()
        self.push_scope()
        self.compile_stmts(then_body)
        self.pop_scope()
        if else_body:
            self.emit(Op(
                lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait),
                is_branch=True,
                target_label=lbl_end,
            ))
            self.emit_label(lbl_else)
            self.push_scope()
            self.compile_stmts(else_body)
            self.pop_scope()
        self.emit_label(lbl_end)
        self.emit(Op(
            lambda wr, rd, wait, stall: isa.encode_bsync(bar=0, stall=stall, wait=wait),
            is_bar=True,
        ))
        self.emit_label(lbl_reconv)

    def emit_pred_store(self, name, args, pred):
        if name == "sts32":
            sname = args[0][1]
            base_off, _, elem_bytes, _ = self.smem[sname]
            idx_c = try_eval_const(args[1], self.consts)
            val = self.eval_expr(args[2])
            if idx_c is not None:
                off = base_off + int(idx_c) * elem_bytes
                self.emit(Op(
                    lambda wr, rd, wait, stall, rv=val.r0, o=off, p=pred: isa.encode_sts(isa.RZ, rv, offset=o, pred=p, neg=False, rd_bar=rd, stall=stall, wait=wait),
                    reads=(val.r0,),
                    preads=(pred,),
                    mem_addr_reads=(val.r0,),
                ))
            else:
                idx_v = self.eval_expr(args[1])
                raddr = self.alloc_val("u32", temp=True)
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=raddr.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(d, a, eb, stall=stall, wait=wait),
                    reads=(idx_v.r0,),
                    writes=(raddr.r0,),
                ))
                self.emit(Op(
                    lambda wr, rd, wait, stall, ra=raddr.r0, rv=val.r0, o=base_off, p=pred: isa.encode_sts(ra, rv, offset=o, pred=p, neg=False, rd_bar=rd, stall=stall, wait=wait),
                    reads=(raddr.r0, val.r0),
                    preads=(pred,),
                    mem_addr_reads=(raddr.r0, val.r0),
                ))
            return
        ptr = self.eval_expr(args[0])
        val = self.eval_expr(args[1])
        ur = self.ur_desc
        addr_reads = (ptr.regs[0], ptr.regs[1])
        if name == "stg256":
            self.emit(Op(
                lambda wr, rd, wait, stall, pa=ptr.regs[0], vlo=val.regs[0], vhi=val.regs[4], u=ur, p=pred: isa.encode_stg256(pa, vlo, vhi, u, pred=p, neg=False, rd=rd, stall=stall, wait=wait),
                reads=addr_reads + val.regs,
                preads=(pred,),
                mem_addr_reads=addr_reads + val.regs,
                ureads=(ur, ur + 1),
            ))
            return
        wbits = {
            "stg128": isa.STG_W_128,
            "stg64": isa.STG_W_64,
            "stg32": isa.STG_W_32,
            "stg16": isa.STG_W_U16,
            "stg8": isa.STG_W_U8,
        }[name]
        self.emit(Op(
            lambda wr, rd, wait, stall, pa=ptr.regs[0], rv=val.regs[0], u=ur, wb=wbits, p=pred: isa.encode_stg(pa, rv, u, width_bits=wb, pred=p, neg=False, rd=rd, stall=stall, wait=wait),
            reads=addr_reads + val.regs,
            preads=(pred,),
            mem_addr_reads=addr_reads + val.regs,
            ureads=(ur, ur + 1),
        ))

    def can_pipeline_loop(self, var, cond, step, body):
        if cond[0] != "bin" or cond[1] != "<" or cond[2] != ("ident", var):
            return False
        if step[0] != "assign" or step[1] != "+=" or step[2] != ("ident", var):
            return False
        loads = []
        for st in body:
            if st[0] == "decl" and st[4] is not None and st[4][0] == "call" and st[4][1] in LOAD_CALLS:
                loads.append(st)
            else:
                break
        return len(loads) >= 1 and len(loads) < len(body)

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
        lbl_head = self.new_label("LOOP")
        lbl_end = self.new_label("ENDLOOP")
        self.push_temps()
        pred = self.eval_cond(cond_e, pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=pred: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait),
            preads=(pred,),
            is_branch=True,
            target_label=lbl_end,
        ))
        self.pop_temps()
        self.emit_label(lbl_head)
        self.push_scope()
        self.compile_stmts(body)
        self.pop_scope()
        self.push_temps()
        self.compile_stmt(step_st)
        pred2 = self.eval_cond(cond_e, pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=pred2: isa.encode_bra(delta, pred=p, neg=False, stall=stall, wait=wait),
            preads=(pred2,),
            is_branch=True,
            target_label=lbl_head,
        ))
        self.pop_temps()
        self.emit_label(lbl_end)
        self.pop_scope()

    def compile_pipelined_for(self, vtype, var, init_e, cond_e, step_st, body):
        load_stmts = []
        for st in body:
            if st[0] == "decl" and st[4] is not None and st[4][0] == "call" and st[4][1] in LOAD_CALLS:
                load_stmts.append(st)
            else:
                break
        comp_stmts = body[len(load_stmts):]

        self.push_scope()
        ivar = self.alloc_val(vtype, temp=False)
        inext = self.alloc_val(vtype, temp=False)
        self.bind_var(var, ivar)
        self.push_temps()
        self.eval_into(init_e, ivar)
        self.pop_temps(keep=ivar)

        limit_expr = cond_e[3]
        step_expr = step_st[3]
        bank0 = {}
        bank1 = {}
        for _, _, ltype, lname, _ in load_stmts:
            bank0[lname] = self.alloc_val(ltype, temp=False)
            bank1[lname] = self.alloc_val(ltype, temp=False)

        lbl_loop = self.new_label("PLOOP")
        lbl_epi0 = self.new_label("PEPI0")
        lbl_epi1 = self.new_label("PEPI1")
        lbl_end = self.new_label("PEND")

        self.push_temps()
        p_init = self.eval_cond(("bin", "<", ("ident", var), limit_expr), pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=p_init: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait),
            preads=(p_init,),
            is_branch=True,
            target_label=lbl_end,
        ))
        self.pop_temps()

        def emit_bank_loads(idx_val, bank, bar_id):
            self.batch_counter += 1
            bid = self.batch_counter
            saved = self.scopes[-1][var]
            self.scopes[-1][var] = idx_val
            self.push_temps()
            for _, _, _, lname, lexpr in load_stmts:
                self.emit_load_call(lexpr[1], lexpr[2], bank[lname], forced_wr=bar_id, batch_id=bid)
            self.pop_temps()
            self.scopes[-1][var] = saved

        def emit_bank_compute(bank, bar_id):
            self.push_scope()
            for lname, lval in bank.items():
                self.scopes[-1][lname] = lval
            start_idx = len(self.ops)
            self.compile_stmts(comp_stmts)
            bank_regs = set()
            for lval in bank.values():
                bank_regs.update(lval.regs)
            for op in self.ops[start_idx:]:
                if op.label is None and (op.reads & bank_regs):
                    op.forced_wait |= 1 << bar_id
                    break
            self.scopes.pop()

        def emit_step(dst_v, src_v):
            self.push_temps()
            s_const = try_eval_const(step_expr, self.consts)
            if s_const is not None:
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=dst_v.r0, a=src_v.r0, im=int(s_const): isa.encode_iadd3_i(d, a, im, stall=stall, wait=wait),
                    reads=(src_v.r0,),
                    writes=(dst_v.r0,),
                ))
            else:
                sv = self.eval_expr(step_expr)
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=dst_v.r0, a=src_v.r0, b=sv.r0: isa.encode_iadd3_r(d, a, b, stall=stall, wait=wait),
                    reads=(src_v.r0, sv.r0),
                    writes=(dst_v.r0,),
                ))
            self.pop_temps()

        emit_bank_loads(ivar, bank0, bar_id=4)
        emit_step(inext, ivar)
        self.emit_label(lbl_loop)

        self.push_temps()
        p0 = self.eval_cmp_val(inext, limit_expr, "<", pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=p0: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait),
            preads=(p0,),
            is_branch=True,
            target_label=lbl_epi0,
        ))
        self.pop_temps()

        emit_bank_loads(inext, bank1, bar_id=5)
        emit_bank_compute(bank0, bar_id=4)
        self.emit(Op(
            lambda wr, rd, wait, stall, d=ivar.r0, s=inext.r0: isa.encode_mov_r(d, s, stall=stall, wait=wait),
            reads=(inext.r0,),
            writes=(ivar.r0,),
        ))
        emit_step(inext, ivar)

        self.push_temps()
        p1 = self.eval_cmp_val(inext, limit_expr, "<", pred=0)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall, p=p1: isa.encode_bra(delta, pred=p, neg=True, stall=stall, wait=wait),
            preads=(p1,),
            is_branch=True,
            target_label=lbl_epi1,
        ))
        self.pop_temps()

        emit_bank_loads(inext, bank0, bar_id=4)
        emit_bank_compute(bank1, bar_id=5)
        self.emit(Op(
            lambda wr, rd, wait, stall, d=ivar.r0, s=inext.r0: isa.encode_mov_r(d, s, stall=stall, wait=wait),
            reads=(inext.r0,),
            writes=(ivar.r0,),
        ))
        emit_step(inext, ivar)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait),
            is_branch=True,
            target_label=lbl_loop,
        ))

        self.emit_label(lbl_epi0)
        emit_bank_compute(bank0, bar_id=4)
        self.emit(Op(
            lambda delta, wr, rd, wait, stall: isa.encode_bra(delta, stall=stall, wait=wait),
            is_branch=True,
            target_label=lbl_end,
        ))

        self.emit_label(lbl_epi1)
        emit_bank_compute(bank1, bar_id=5)

        self.emit_label(lbl_end)
        for v in bank0.values():
            self.alloc.free(v.regs)
        for v in bank1.values():
            self.alloc.free(v.regs)
        self.alloc.free(inext.regs)
        self.pop_scope()

    def compile_assign(self, op, lhs, rhs):
        dst = self.eval_lvalue(lhs)
        if op == "=":
            self.eval_into(rhs, dst)
            return
        bin_op = op[0]
        self.eval_into(("bin", bin_op, lhs, rhs), dst)

    def eval_lvalue(self, expr):
        if expr[0] == "ident":
            v = self.lookup_var(expr[1])
            if v is None:
                raise ValueError(f"undefined variable: {expr[1]}")
            return v
        if expr[0] == "index":
            base = self.eval_lvalue(expr[1])
            idx = int(try_eval_const(expr[2], self.consts))
            elem_type = "f32" if base.vtype.startswith("f32") else "u32"
            return Val(elem_type, (base.regs[idx],))
        if expr[0] == "field":
            base = self.eval_lvalue(expr[1])
            idx = {"x": 0, "y": 1, "z": 2, "w": 3}[expr[2]]
            elem_type = "f32" if base.vtype.startswith("f32") else "u32"
            return Val(elem_type, (base.regs[idx],))
        raise ValueError(f"invalid lvalue: {expr}")

    def eval_cmp_val(self, lhs_v, rhs_expr, op_str, pred=0):
        cmp_map = {
            "<": isa.CMP_LT,
            "==": isa.CMP_EQ,
            "<=": isa.CMP_LE,
            ">": isa.CMP_GT,
            "!=": isa.CMP_NE,
            ">=": isa.CMP_GE,
        }
        cmp_op = cmp_map[op_str]
        is_float = lhs_v.vtype == "f32"
        is_signed = lhs_v.vtype == "i32"
        rc = try_eval_const(rhs_expr, self.consts)
        if rc is not None:
            if is_float:
                fv = float(rc)
                self.emit(Op(
                    lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, f=fv, c=cmp_op: isa.encode_fsetp_i(p, a, f, cmp_op=c, stall=stall, wait=wait),
                    reads=(lhs_v.r0,),
                    pwrites=(pred,),
                    fixed_lat=13,
                ))
            else:
                iv = int(rc) & 0xFFFFFFFF
                self.emit(Op(
                    lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, im=iv, c=cmp_op, u=not is_signed: isa.encode_isetp_i(p, a, im, cmp_op=c, unsigned=u, stall=stall, wait=wait),
                    reads=(lhs_v.r0,),
                    pwrites=(pred,),
                    fixed_lat=13,
                ))
            return pred
        rhs_v = self.eval_expr(rhs_expr)
        if is_float or rhs_v.vtype == "f32":
            self.emit(Op(
                lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, b=rhs_v.r0, c=cmp_op: isa.encode_fsetp_r(p, a, b, cmp_op=c, stall=stall, wait=wait),
                reads=(lhs_v.r0, rhs_v.r0),
                pwrites=(pred,),
                fixed_lat=13,
            ))
        else:
            self.emit(Op(
                lambda wr, rd, wait, stall, p=pred, a=lhs_v.r0, b=rhs_v.r0, c=cmp_op, u=not is_signed: isa.encode_isetp_r(p, a, b, cmp_op=c, unsigned=u, stall=stall, wait=wait),
                reads=(lhs_v.r0, rhs_v.r0),
                pwrites=(pred,),
                fixed_lat=13,
            ))
        return pred

    def eval_cond(self, expr, pred=0):
        if expr[0] == "bin" and expr[1] in ("<", "<=", ">", ">=", "==", "!="):
            lhs_v = self.eval_expr(expr[2])
            return self.eval_cmp_val(lhs_v, expr[3], expr[1], pred=pred)
        v = self.eval_expr(expr)
        self.emit(Op(
            lambda wr, rd, wait, stall, p=pred, a=v.r0: isa.encode_isetp_i(p, a, 0, cmp_op=isa.CMP_NE, unsigned=True, stall=stall, wait=wait),
            reads=(v.r0,),
            pwrites=(pred,),
            fixed_lat=13,
        ))
        return pred

    def eval_into(self, expr, dst):
        if expr[0] == "call" and expr[1] in LOAD_CALLS:
            self.emit_load_call(expr[1], expr[2], dst)
            return dst
        if expr[0] == "call" and expr[1] == "fma":
            a = self.eval_expr(expr[2][0])
            b_const = try_eval_const(expr[2][1], self.consts)
            c = self.eval_expr(expr[2][2])
            if b_const is not None:
                fv = float(b_const)
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, f=fv, rc=c.r0: isa.encode_ffma_i(d, ra, f, rc, stall=stall, wait=wait),
                    reads=(a.r0, c.r0),
                    writes=(dst.r0,),
                    pipe="fp32",
                ))
            else:
                b = self.eval_expr(expr[2][1])
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=dst.r0, ra=a.r0, rb=b.r0, rc=c.r0: isa.encode_ffma_r(d, ra, rb, rc, stall=stall, wait=wait),
                    reads=(a.r0, b.r0, c.r0),
                    writes=(dst.r0,),
                    pipe="fp32",
                ))
            return dst
        val = self.eval_expr(expr, hint_dst=dst)
        if val.regs != dst.regs:
            for d_r, s_r in zip(dst.regs, val.regs):
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=d_r, s=s_r: isa.encode_mov_r(d, s, stall=stall, wait=wait),
                    reads=(s_r,),
                    writes=(d_r,),
                ))
        return dst

    def emit_load_call(self, fn_name, args, dst, forced_wr=None, batch_id=None):
        ptr = self.eval_expr(args[0])
        _, mode = LOAD_CALLS[fn_name]
        ur = self.ur_desc
        addr_reads = (ptr.regs[0], ptr.regs[1])
        if mode.startswith("256_"):
            pol = {"256_ef": "ef", "256_el": "el", "256_def": "default"}[mode]
            self.emit(Op(
                lambda wr, rd, wait, stall, rlo=dst.regs[0], rhi=dst.regs[4], pa=ptr.regs[0], u=ur, p=pol: isa.encode_ldg256(rlo, rhi, pa, u, policy=p, wr=wr, rd_bar=rd, stall=stall, wait=wait),
                reads=addr_reads,
                mem_addr_reads=addr_reads,
                ureads=(ur, ur + 1),
                writes=dst.regs,
                var_lat=True,
                forced_wr=forced_wr,
                batch_id=batch_id,
            ))
            return
        w_map = {
            "128_ef": (isa.LDG_W_128, isa.CACHE_EF),
            "128_const": (isa.LDG_W_128, isa.CACHE_CONSTANT),
            "64_ef": (isa.LDG_W_64, isa.CACHE_EF),
            "64_const": (isa.LDG_W_64, isa.CACHE_CONSTANT),
            "32_ef": (isa.LDG_W_32, isa.CACHE_EF),
            "32_const": (isa.LDG_W_32, isa.CACHE_CONSTANT),
            "16_ef": (isa.LDG_W_U16, isa.CACHE_EF),
            "16_const": (isa.LDG_W_U16, isa.CACHE_CONSTANT),
            "8_ef": (isa.LDG_W_U8, isa.CACHE_EF),
            "8_const": (isa.LDG_W_U8, isa.CACHE_CONSTANT),
        }
        wbits, cbits = w_map[mode]
        self.emit(Op(
            lambda wr, rd, wait, stall, r=dst.regs[0], pa=ptr.regs[0], u=ur, wb=wbits, cb=cbits: isa.encode_ldg(r, pa, u, width_bits=wb, cache=cb, wr=wr, rd_bar=rd, stall=stall, wait=wait),
            reads=addr_reads,
            mem_addr_reads=addr_reads,
            ureads=(ur, ur + 1),
            writes=dst.regs,
            var_lat=True,
            forced_wr=forced_wr,
            batch_id=batch_id,
        ))

    def eval_expr(self, expr, hint_dst=None):
        cval = try_eval_const(expr, self.consts)
        if cval is not None:
            if isinstance(cval, float):
                d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
                bits = isa.f32_bits(cval)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, im=bits: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(d.r0,)))
                return d
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1 and hint_dst.vtype != "f32") else self.alloc_val("u32", temp=True)
            iv = int(cval) & 0xFFFFFFFF
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, im=iv: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(d.r0,)))
            return d

        kind = expr[0]
        if kind == "ident":
            name = expr[1]
            v = self.lookup_var(name)
            if v is not None:
                return v
            if name in self.param_offsets:
                return self.get_param(name)
            raise ValueError(f"undefined identifier: {name}")

        if kind == "index":
            base = self.eval_expr(expr[1])
            idx = int(try_eval_const(expr[2], self.consts))
            elem_type = "f32" if base.vtype.startswith("f32") else "u32"
            return Val(elem_type, (base.regs[idx],))

        if kind == "field":
            base = self.eval_expr(expr[1])
            idx = {"x": 0, "y": 1, "z": 2, "w": 3}[expr[2]]
            elem_type = "f32" if base.vtype.startswith("f32") else "u32"
            return Val(elem_type, (base.regs[idx],))

        if kind == "ternary":
            _, cond, t_e, f_e = expr
            cv = try_eval_const(cond, self.consts)
            if cv is not None:
                return self.eval_expr(t_e if cv else f_e, hint_dst=hint_dst)
            pred = self.eval_cond(cond, pred=1)
            tv = self.eval_expr(t_e)
            fc = try_eval_const(f_e, self.consts)
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == tv.vtype) else self.alloc_val(tv.vtype, temp=True)
            if fc is not None and len(d.regs) == 1:
                imm = isa.f32_bits(fc) if tv.vtype == "f32" else (int(fc) & 0xFFFFFFFF)
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, a=tv.r0, im=imm, p=pred: isa.encode_sel_i(r, a, im, p, stall=stall, wait=wait),
                    reads=(tv.r0,),
                    preads=(pred,),
                    writes=(d.r0,),
                ))
                return d
            fv = self.eval_expr(f_e)
            for dr, tr, fr in zip(d.regs, tv.regs, fv.regs):
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=dr, a=tr, b=fr, p=pred: isa.encode_sel_r(r, a, b, p, stall=stall, wait=wait),
                    reads=(tr, fr),
                    preads=(pred,),
                    writes=(dr,),
                ))
            return d

        if kind == "un":
            op, sub = expr[1], expr[2]
            v = self.eval_expr(sub)
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == v.vtype) else self.alloc_val(v.vtype, temp=True)
            if op == "-":
                if v.vtype == "f32":
                    self.emit(Op(
                        lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_lop3_i(r, a, 0x80000000, lut=0x3C, stall=stall, wait=wait),
                        reads=(v.r0,),
                        writes=(d.r0,),
                    ))
                else:
                    self.emit(Op(
                        lambda wr, rd, wait, stall, r=d.r0, b=v.r0: isa.encode_iadd3_r(r, isa.RZ, b, neg_b=True, stall=stall, wait=wait),
                        reads=(v.r0,),
                        writes=(d.r0,),
                    ))
                return d
            if op == "~":
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_lop3_i(r, a, 0, lut=0x0F, stall=stall, wait=wait),
                    reads=(v.r0,),
                    writes=(d.r0,),
                ))
                return d
            raise ValueError(f"unsupported unary op: {op}")

        if kind == "bin":
            return self.eval_bin(expr[1], expr[2], expr[3], hint_dst=hint_dst)

        if kind == "call":
            return self.eval_call(expr[1], expr[2], hint_dst=hint_dst)

        raise ValueError(f"unsupported expr: {expr}")

    def eval_bin(self, op, lhs_e, rhs_e, hint_dst=None):
        lhs = self.eval_expr(lhs_e)
        rc = try_eval_const(rhs_e, self.consts)

        if lhs.vtype == "f32":
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            if op == "+":
                if rc is not None:
                    fv = float(rc)
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, f=fv: isa.encode_fadd_i(r, a, f, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,), pipe="fp32"))
                    return d
                rhs = self.eval_expr(rhs_e)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_fadd_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,), pipe="fp32"))
                return d
            if op == "-":
                if rc is not None:
                    fv = -float(rc)
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, f=fv: isa.encode_fadd_i(r, a, f, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,), pipe="fp32"))
                    return d
                rhs = self.eval_expr(rhs_e)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_fadd_r(r, a, b, neg_b=True, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,), pipe="fp32"))
                return d
            if op == "*":
                if rc is not None:
                    fv = float(rc)
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, f=fv: isa.encode_fmul_i(r, a, f, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,), pipe="fp32"))
                    return d
                rhs = self.eval_expr(rhs_e)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_fmul_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,), pipe="fp32"))
                return d
            if op == "/":
                rhs = self.eval_expr(rhs_e)
                inv = self.alloc_val("f32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=inv.r0, b=rhs.r0: isa.encode_mufu(r, b, op=isa.MUFU_RCP, wr=wr, stall=stall, wait=wait), reads=(rhs.r0,), writes=(inv.r0,), var_lat=True))
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=inv.r0: isa.encode_fmul_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, inv.r0), writes=(d.r0,)))
                return d
            raise ValueError(f"unsupported f32 op: {op}")

        d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1 and hint_dst.vtype != "f32") else self.alloc_val("u32", temp=True)
        if op == "+":
            if rc is not None:
                im = int(rc) & 0xFFFFFFFF
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=im: isa.encode_iadd3_i(r, a, i, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_iadd3_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op == "-":
            if rc is not None:
                im = (-int(rc)) & 0xFFFFFFFF
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=im: isa.encode_iadd3_i(r, a, i, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_iadd3_r(r, a, b, neg_b=True, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op == "*":
            if rc is not None:
                im = int(rc) & 0xFFFFFFFF
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=im: isa.encode_imad_i(r, a, i, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_imad_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op == "<<":
            if rc is not None:
                sh = int(rc) & 31
                if sh == 0:
                    return lhs
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, s=sh: isa.encode_shf_l_i(r, a, s, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0: isa.encode_shf_l_r(r, a, b, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op == ">>":
            if rc is not None:
                sh = int(rc) & 31
                if sh == 0:
                    return lhs
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, s=sh, c=lhs.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, b=rhs.r0, c=lhs.r0: isa.encode_shf_r_u32_hi_r(r, b, c, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        if op in ("&", "|", "^"):
            lut = {"&": 0xC0, "|": 0xFC, "^": 0x3C}[op]
            if rc is not None:
                im = int(rc) & 0xFFFFFFFF
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, i=im, l=lut: isa.encode_lop3_i(r, a, i, lut=l, stall=stall, wait=wait), reads=(lhs.r0,), writes=(d.r0,)))
                return d
            rhs = self.eval_expr(rhs_e)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lhs.r0, b=rhs.r0, l=lut: isa.encode_lop3_r(r, a, b, lut=l, stall=stall, wait=wait), reads=(lhs.r0, rhs.r0), writes=(d.r0,)))
            return d
        raise ValueError(f"unsupported u32 op: {op}")

    def eval_call(self, name, args, hint_dst=None):
        if name in ("ctaid_x", "ctaid_y", "tid_x", "lane_id", "ntid_x", "grid_x", "warp_id"):
            return self.get_sr(name)

        if name == "pdl_wait":
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_acqbulk(stall=stall, wait=wait)))
            return Val("u32", (isa.RZ,))

        if name == "pdl_release":
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_preexit(stall=max(stall, 8), wait=wait)))
            return Val("u32", (isa.RZ,))

        if name == "barrier":
            self.barriers = max(self.barriers, 1)
            self.emit(Op(lambda wr, rd, wait, stall: isa.encode_bar_sync(0, stall=stall, wait=wait), is_bar=True))
            return Val("u32", (isa.RZ,))

        if name == "prefetch_l2":
            return Val("u32", (isa.RZ,))

        if name == "ptr_add":
            base = self.eval_expr(args[0])
            idx_c = try_eval_const(args[1], self.consts)
            str_c = try_eval_const(args[2], self.consts)
            if idx_c is not None and str_c is not None:
                byte_off = int(idx_c) * int(str_c)
                if byte_off == 0:
                    return base
                d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u64") else self.alloc_val("u64", temp=True)
                tmp = self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=tmp.r0, im=byte_off & 0xFFFFFFFF: isa.encode_mov_i(r, im, stall=stall, wait=wait), writes=(tmp.r0,)))
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, a=tmp.r0, c=base.r0: isa.encode_imad_wide_i(r, a, 1, c, stall=stall, wait=wait),
                    reads=(tmp.r0, base.regs[0], base.regs[1]),
                    writes=d.regs,
                ))
                return d
            idx_v = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u64") else self.alloc_val("u64", temp=True)
            if str_c is not None:
                sv = int(str_c) & 0xFFFFFFFF
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, a=idx_v.r0, im=sv, c=base.r0: isa.encode_imad_wide_i(r, a, im, c, stall=stall, wait=wait),
                    reads=(idx_v.r0, base.regs[0], base.regs[1]),
                    writes=d.regs,
                ))
            else:
                str_v = self.eval_expr(args[2])
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, a=idx_v.r0, b=str_v.r0, c=base.r0: isa.encode_imad_wide_r(r, a, b, c, stall=stall, wait=wait),
                    reads=(idx_v.r0, str_v.r0, base.regs[0], base.regs[1]),
                    writes=d.regs,
                ))
            return d

        if name in LOAD_CALLS:
            vtype, _ = LOAD_CALLS[name]
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == TYPE_REGS[vtype]) else self.alloc_val(vtype, temp=True)
            self.emit_load_call(name, args, d)
            return d

        if name in ("stg256", "stg128", "stg64", "stg32", "stg16", "stg8"):
            ptr = self.eval_expr(args[0])
            val = self.eval_expr(args[1])
            ur = self.ur_desc
            addr_reads = (ptr.regs[0], ptr.regs[1])
            if name == "stg256":
                self.emit(Op(
                    lambda wr, rd, wait, stall, pa=ptr.regs[0], vlo=val.regs[0], vhi=val.regs[4], u=ur: isa.encode_stg256(pa, vlo, vhi, u, rd=rd, stall=stall, wait=wait),
                    reads=addr_reads + val.regs,
                    mem_addr_reads=addr_reads + val.regs,
                    ureads=(ur, ur + 1),
                ))
                return Val("u32", (isa.RZ,))
            wbits = {
                "stg128": isa.STG_W_128,
                "stg64": isa.STG_W_64,
                "stg32": isa.STG_W_32,
                "stg16": isa.STG_W_U16,
                "stg8": isa.STG_W_U8,
            }[name]
            self.emit(Op(
                lambda wr, rd, wait, stall, pa=ptr.regs[0], rv=val.regs[0], u=ur, wb=wbits: isa.encode_stg(pa, rv, u, width_bits=wb, rd=rd, stall=stall, wait=wait),
                reads=addr_reads + val.regs,
                mem_addr_reads=addr_reads + val.regs,
                ureads=(ur, ur + 1),
            ))
            return Val("u32", (isa.RZ,))

        if name == "sts32":
            sname = args[0][1]
            base_off, _, elem_bytes, _ = self.smem[sname]
            idx_c = try_eval_const(args[1], self.consts)
            val = self.eval_expr(args[2])
            if idx_c is not None:
                off = base_off + int(idx_c) * elem_bytes
                self.emit(Op(
                    lambda wr, rd, wait, stall, rv=val.r0, o=off: isa.encode_sts(isa.RZ, rv, offset=o, rd_bar=rd, stall=stall, wait=wait),
                    reads=(val.r0,),
                    mem_addr_reads=(val.r0,),
                ))
            else:
                idx_v = self.eval_expr(args[1])
                raddr = self.alloc_val("u32", temp=True)
                self.emit(Op(
                    lambda wr, rd, wait, stall, d=raddr.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(d, a, eb, stall=stall, wait=wait),
                    reads=(idx_v.r0,),
                    writes=(raddr.r0,),
                ))
                self.emit(Op(
                    lambda wr, rd, wait, stall, ra=raddr.r0, rv=val.r0, o=base_off: isa.encode_sts(ra, rv, offset=o, rd_bar=rd, stall=stall, wait=wait),
                    reads=(raddr.r0, val.r0),
                    mem_addr_reads=(raddr.r0, val.r0),
                ))
            return Val("u32", (isa.RZ,))

        if name == "lds32":
            sname = args[0][1]
            base_off, vtype, elem_bytes, _ = self.smem[sname]
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val(vtype, temp=True)
            idx_c = try_eval_const(args[1], self.consts)
            if idx_c is not None:
                off = base_off + int(idx_c) * elem_bytes
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, o=off: isa.encode_lds(r, isa.RZ, offset=o, wr=wr, rd_bar=rd, stall=stall, wait=wait),
                    writes=(d.r0,),
                    var_lat=True,
                ))
            else:
                idx_v = self.eval_expr(args[1])
                raddr = self.alloc_val("u32", temp=True)
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=raddr.r0, a=idx_v.r0, eb=elem_bytes: isa.encode_imad_i(r, a, eb, stall=stall, wait=wait),
                    reads=(idx_v.r0,),
                    writes=(raddr.r0,),
                ))
                self.emit(Op(
                    lambda wr, rd, wait, stall, r=d.r0, ra=raddr.r0, o=base_off: isa.encode_lds(r, ra, offset=o, wr=wr, rd_bar=rd, stall=stall, wait=wait),
                    reads=(raddr.r0,),
                    mem_addr_reads=(raddr.r0,),
                    writes=(d.r0,),
                    var_lat=True,
                ))
            return d

        if name == "fma":
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            a = self.eval_expr(args[0])
            bc = try_eval_const(args[1], self.consts)
            c = self.eval_expr(args[2])
            if bc is not None:
                fv = float(bc)
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, f=fv, rc=c.r0: isa.encode_ffma_i(r, ra, f, rc, stall=stall, wait=wait), reads=(a.r0, c.r0), writes=(d.r0,), pipe="fp32"))
            else:
                b = self.eval_expr(args[1])
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, rb=b.r0, rc=c.r0: isa.encode_ffma_r(r, ra, rb, rc, stall=stall, wait=wait), reads=(a.r0, b.r0, c.r0), writes=(d.r0,), pipe="fp32"))
            return d

        if name in ("maxf", "minf"):
            a = self.eval_expr(args[0])
            b = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            is_max = name == "maxf"
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0, rb=b.r0, m=is_max: isa.encode_fmnmx_r(r, ra, rb, is_max=m, stall=stall, wait=wait), reads=(a.r0, b.r0), writes=(d.r0,)))
            return d

        if name == "fabsf":
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_lop3_i(r, ra, 0x7FFFFFFF, lut=0xC0, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), pipe="lop3"))
            return d

        if name in ("rcp", "rsqrt", "exp2_ftz"):
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            op = {"rcp": isa.MUFU_RCP, "rsqrt": isa.MUFU_RSQ, "exp2_ftz": isa.MUFU_EX2}[name]
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=a.r0, o=op: isa.encode_mufu(r, rb, op=o, wr=wr, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), var_lat=True))
            return d

        if name == "expf_fast":
            a = self.eval_expr(args[0])
            scaled = self.alloc_val("f32", temp=True)
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=scaled.r0, ra=a.r0: isa.encode_fmul_i(r, ra, 1.4426950408889634, stall=stall, wait=wait), reads=(a.r0,), writes=(scaled.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=scaled.r0: isa.encode_mufu(r, rb, op=isa.MUFU_EX2, wr=wr, stall=stall, wait=wait), reads=(scaled.r0,), writes=(d.r0,), var_lat=True))
            return d

        if name == "silu":
            x = self.eval_expr(args[0])
            t0 = self.alloc_val("f32", temp=True)
            t1 = self.alloc_val("f32", temp=True)
            t2 = self.alloc_val("f32", temp=True)
            t3 = self.alloc_val("f32", temp=True)
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=t0.r0, ra=x.r0: isa.encode_fmul_i(r, ra, -1.4426950408889634, stall=stall, wait=wait), reads=(x.r0,), writes=(t0.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t1.r0, rb=t0.r0: isa.encode_mufu(r, rb, op=isa.MUFU_EX2, wr=wr, stall=stall, wait=wait), reads=(t0.r0,), writes=(t1.r0,), var_lat=True))
            self.emit(Op(lambda wr, rd, wait, stall, r=t2.r0, ra=t1.r0: isa.encode_fadd_i(r, ra, 1.0, stall=stall, wait=wait), reads=(t1.r0,), writes=(t2.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t3.r0, rb=t2.r0: isa.encode_mufu(r, rb, op=isa.MUFU_RCP, wr=wr, stall=stall, wait=wait), reads=(t2.r0,), writes=(t3.r0,), var_lat=True))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=x.r0, rb=t3.r0: isa.encode_fmul_r(r, ra, rb, stall=stall, wait=wait), reads=(x.r0, t3.r0), writes=(d.r0,)))
            return d

        if name == "u32_to_f32":
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rb=a.r0: isa.encode_i2fp_u32(r, rb, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d

        if name == "u32_as_f32":
            a = self.eval_expr(args[0])
            return Val("f32", a.regs)

        if name == "f32_as_u32":
            a = self.eval_expr(args[0])
            return Val("u32", a.regs)

        if name in ("bf16_lo", "bf16_to_f32"):
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_shf_l_i(r, ra, 16, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d

        if name == "bf16_hi":
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, ra=a.r0: isa.encode_lop3_i(r, ra, 0xFFFF0000, lut=0xC0, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,), pipe="lop3"))
            return d

        if name == "f32_to_bf16":
            a = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rlo=a.r0: isa.encode_f2fp_bf16_pack(r, isa.RZ, rlo, stall=stall, wait=wait), reads=(a.r0,), writes=(d.r0,)))
            return d

        if name == "pack_bf16x2":
            lo = self.eval_expr(args[0])
            hi = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rhi=hi.r0, rlo=lo.r0: isa.encode_f2fp_bf16_pack(r, rhi, rlo, stall=stall, wait=wait), reads=(hi.r0, lo.r0), writes=(d.r0,)))
            return d

        if name == "cvt_f32x2_e4m3x2":
            lo = self.eval_expr(args[0])
            hi = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, rhi=hi.r0, rlo=lo.r0: isa.encode_f2fp_e4m3_pack(r, rhi, rlo, stall=stall, wait=wait), reads=(hi.r0, lo.r0), writes=(d.r0,)))
            return d

        if name == "pack_u16x2":
            lo = self.eval_expr(args[0])
            hi = self.eval_expr(args[1])
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=lo.r0, c=hi.r0: isa.encode_prmt_i(r, a, 0x5410, c, stall=stall, wait=wait), reads=(lo.r0, hi.r0), writes=(d.r0,)))
            return d

        if name == "cvt_e2m1x2_f32x2":
            b = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32x2") else self.alloc_val("f32x2", temp=True)
            fp16x2 = self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=fp16x2.r0, rb=b.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b.r0,), writes=(fp16x2.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[0], rb=fp16x2.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=False, stall=stall, wait=wait), reads=(fp16x2.r0,), writes=(d.regs[0],), pipe="fp32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[1], rb=fp16x2.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=True, stall=stall, wait=wait), reads=(fp16x2.r0,), writes=(d.regs[1],), pipe="fp32"))
            return d

        if name == "cvt_e2m1x4_f32x4":
            w = self.eval_expr(args[0])
            sh = int(try_eval_const(args[1], self.consts))
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32x4") else self.alloc_val("f32x4", temp=True)
            h0 = self.alloc_val("u32", temp=True)
            h1 = self.alloc_val("u32", temp=True)
            if sh == 0:
                b1 = self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=b1.r0, c=w.r0: isa.encode_shf_r_u32_hi_i(r, 8, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b1.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h0.r0, rb=w.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(w.r0,), writes=(h0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h1.r0, rb=b1.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b1.r0,), writes=(h1.r0,)))
            else:
                b0 = self.alloc_val("u32", temp=True)
                b1 = self.alloc_val("u32", temp=True)
                self.emit(Op(lambda wr, rd, wait, stall, r=b0.r0, s=sh, c=w.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=b1.r0, s=sh + 8, c=w.r0: isa.encode_shf_r_u32_hi_i(r, s, c, stall=stall, wait=wait), reads=(w.r0,), writes=(b1.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h0.r0, rb=b0.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b0.r0,), writes=(h0.r0,)))
                self.emit(Op(lambda wr, rd, wait, stall, r=h1.r0, rb=b1.r0: isa.encode_f2fp_e2m1_unpack(r, rb, stall=stall, wait=wait), reads=(b1.r0,), writes=(h1.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[0], rb=h0.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=False, stall=stall, wait=wait), reads=(h0.r0,), writes=(d.regs[0],), pipe="fp32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[1], rb=h0.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=True, stall=stall, wait=wait), reads=(h0.r0,), writes=(d.regs[1],), pipe="fp32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[2], rb=h1.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=False, stall=stall, wait=wait), reads=(h1.r0,), writes=(d.regs[2],), pipe="fp32"))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.regs[3], rb=h1.r0: isa.encode_hadd2_f32_half(r, rb, hi_half=True, stall=stall, wait=wait), reads=(h1.r0,), writes=(d.regs[3],), pipe="fp32"))
            return d

        if name == "ue4m3_to_f32":
            b = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            exp = self.alloc_val("u32", temp=True)
            mant = self.alloc_val("u32", temp=True)
            e_adj = self.alloc_val("u32", temp=True)
            e_sh = self.alloc_val("u32", temp=True)
            m_sh = self.alloc_val("u32", temp=True)
            norm_bits = self.alloc_val("u32", temp=True)
            sub_f = self.alloc_val("f32", temp=True)
            sub_s = self.alloc_val("f32", temp=True)
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
            b = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            eb = self.alloc_val("u32", temp=True)
            sh = self.alloc_val("u32", temp=True)
            t1 = self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=eb.r0, a=b.r0: isa.encode_lop3_i(r, a, 0xFF, lut=0xC0, stall=stall, wait=wait), reads=(b.r0,), writes=(eb.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=sh.r0, a=eb.r0: isa.encode_shf_l_i(r, a, 23, stall=stall, wait=wait), reads=(eb.r0,), writes=(sh.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=eb.r0: isa.encode_isetp_i(2, a, 0, cmp_op=isa.CMP_EQ, unsigned=True, stall=stall, wait=wait), reads=(eb.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=t1.r0, a=sh.r0: isa.encode_sel_i(r, a, 0x00400000, 2, neg_sp=True, stall=stall, wait=wait), reads=(sh.r0,), preads=(2,), writes=(t1.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=eb.r0: isa.encode_isetp_i(2, a, 255, cmp_op=isa.CMP_EQ, unsigned=True, stall=stall, wait=wait), reads=(eb.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=t1.r0: isa.encode_sel_i(r, a, 0, 2, neg_sp=True, stall=stall, wait=wait), reads=(t1.r0,), preads=(2,), writes=(d.r0,)))
            return d

        if name == "pow2_ceil_e8m0_bits":
            amax = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and len(hint_dst.regs) == 1) else self.alloc_val("u32", temp=True)
            sc = self.alloc_val("f32", temp=True)
            b_add = self.alloc_val("u32", temp=True)
            exp = self.alloc_val("u32", temp=True)
            cl = self.alloc_val("u32", temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=sc.r0, ra=amax.r0: isa.encode_fmul_i(r, ra, 1.0 / 448.0, stall=stall, wait=wait), reads=(amax.r0,), writes=(sc.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=b_add.r0, ra=sc.r0: isa.encode_iadd3_i(r, ra, 0x007FFFFF, stall=stall, wait=wait), reads=(sc.r0,), writes=(b_add.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=exp.r0, c=b_add.r0: isa.encode_shf_r_u32_hi_i(r, 23, c, stall=stall, wait=wait), reads=(b_add.r0,), writes=(exp.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=exp.r0: isa.encode_isetp_i(2, a, 254, cmp_op=isa.CMP_GT, unsigned=True, stall=stall, wait=wait), reads=(exp.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=cl.r0, a=exp.r0: isa.encode_sel_i(r, a, 254, 2, neg_sp=True, stall=stall, wait=wait), reads=(exp.r0,), preads=(2,), writes=(cl.r0,)))
            self.emit(Op(lambda wr, rd, wait, stall, a=amax.r0: isa.encode_fsetp_i(2, a, 0.0, cmp_op=isa.CMP_GT, stall=stall, wait=wait), reads=(amax.r0,), pwrites=(2,)))
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=cl.r0: isa.encode_sel_i(r, a, 0, 2, neg_sp=False, stall=stall, wait=wait), reads=(cl.r0,), preads=(2,), writes=(d.r0,)))
            return d

        if name == "pack_u32x4":
            vals = [self.eval_expr(a) for a in args]
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u32x4") else self.alloc_val("u32x4", temp=True)
            for dr, v in zip(d.regs, vals):
                self.emit(Op(lambda wr, rd, wait, stall, r=dr, s=v.r0: isa.encode_mov_r(r, s, stall=stall, wait=wait), reads=(v.r0,), writes=(dr,)))
            return d

        if name == "pack_u32x8":
            vals = [self.eval_expr(a) for a in args]
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "u32x8") else self.alloc_val("u32x8", temp=True)
            for dr, v in zip(d.regs, vals):
                self.emit(Op(lambda wr, rd, wait, stall, r=dr, s=v.r0: isa.encode_mov_r(r, s, stall=stall, wait=wait), reads=(v.r0,), writes=(dr,)))
            return d

        if name == "shfl_xor":
            v = self.eval_expr(args[0])
            mask = int(try_eval_const(args[1], self.consts))
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == v.vtype) else self.alloc_val(v.vtype, temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0, m=mask: isa.encode_shfl_bfly(r, a, m, wr=wr, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,), var_lat=True))
            return d

        if name == "shfl_down":
            v = self.eval_expr(args[0])
            delta = int(try_eval_const(args[1], self.consts))
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == v.vtype) else self.alloc_val(v.vtype, temp=True)
            self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0, dl=delta: isa.encode_shfl_down(r, a, dl, wr=wr, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,), var_lat=True))
            return d

        if name == "shfl_idx":
            v = self.eval_expr(args[0])
            lane_c = try_eval_const(args[1], self.consts)
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == v.vtype) else self.alloc_val(v.vtype, temp=True)
            if lane_c is not None and int(lane_c) == 0:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0: isa.encode_shfl_idx(r, a, isa.RZ, wr=wr, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,), var_lat=True))
            else:
                lv = self.eval_expr(args[1])
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=v.r0, l=lv.r0: isa.encode_shfl_idx(r, a, l, wr=wr, stall=stall, wait=wait), reads=(v.r0, lv.r0), writes=(d.r0,), var_lat=True))
            return d

        if name in ("warp_sum", "warp_max", "oct_sum", "oct_max", "quad_max", "quad_sum"):
            v = self.eval_expr(args[0])
            d = hint_dst if (hint_dst is not None and hint_dst.vtype == "f32") else self.alloc_val("f32", temp=True)
            if d.r0 != v.r0:
                self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, s=v.r0: isa.encode_mov_r(r, s, stall=stall, wait=wait), reads=(v.r0,), writes=(d.r0,)))
            tmp = self.alloc_val("f32", temp=True)
            if name.startswith("warp_"):
                masks = (16, 8, 4, 2, 1)
            elif name.startswith("oct_"):
                masks = (4, 2, 1)
            else:
                masks = (2, 1)
            is_max = name.endswith("_max")
            for m in masks:
                self.emit(Op(lambda wr, rd, wait, stall, r=tmp.r0, a=d.r0, mk=m: isa.encode_shfl_bfly(r, a, mk, wr=wr, stall=stall, wait=wait), reads=(d.r0,), writes=(tmp.r0,), var_lat=True))
                if is_max:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=d.r0, b=tmp.r0: isa.encode_fmnmx_r(r, a, b, is_max=True, stall=stall, wait=wait), reads=(d.r0, tmp.r0), writes=(d.r0,)))
                else:
                    self.emit(Op(lambda wr, rd, wait, stall, r=d.r0, a=d.r0, b=tmp.r0: isa.encode_fadd_r(r, a, b, stall=stall, wait=wait), reads=(d.r0, tmp.r0), writes=(d.r0,), pipe="fp32"))
            return d

        raise ValueError(f"unknown builtin: {name}")

    def schedule_and_encode(self):
        real_ops = []
        reg_bar = {}
        ureg_bar = {}
        rd_bar = {}
        bar_batch = {}
        next_bar = 0
        has_forced = any(op.forced_wr is not None for op in self.ops)
        dyn_bars = 4 if has_forced else 6

        for op in self.ops:
            if op.label is not None:
                pending_dyn = 0
                for b in set(reg_bar.values()) | set(ureg_bar.values()) | set(rd_bar.values()):
                    if not has_forced or b < 4:
                        pending_dyn |= 1 << b
                if pending_dyn:
                    prev = real_ops[-1] if real_ops else None
                    if prev is not None and prev.label is None and not prev.preads and not prev.var_lat and not prev.mem_addr_reads and not prev.is_branch:
                        prev.wait |= pending_dyn
                    else:
                        nop = Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=stall, wait=wait), forced_wait=pending_dyn)
                        nop.wait = pending_dyn
                        real_ops.append(nop)
                if has_forced:
                    reg_bar = {k: v for k, v in reg_bar.items() if v >= 4}
                    ureg_bar = {k: v for k, v in ureg_bar.items() if v >= 4}
                    rd_bar = {k: v for k, v in rd_bar.items() if v >= 4}
                else:
                    reg_bar.clear()
                    ureg_bar.clear()
                    rd_bar.clear()
                real_ops.append(op)
                continue

            wait_mask = op.forced_wait
            for r in op.reads | op.writes:
                if r in reg_bar:
                    wait_mask |= 1 << reg_bar[r]
            for r in op.writes:
                if r in rd_bar:
                    wait_mask |= 1 << rd_bar[r]
            for u in op.ureads | op.uwrites:
                if u in ureg_bar:
                    wait_mask |= 1 << ureg_bar[u]
            if op.is_bar or (op.is_exit and not op.preads):
                for b in set(reg_bar.values()) | set(ureg_bar.values()) | set(rd_bar.values()):
                    wait_mask |= 1 << b
            elif op.is_branch:
                pending_b = 0
                for b in set(reg_bar.values()) | set(ureg_bar.values()) | set(rd_bar.values()):
                    if not has_forced or b < 4:
                        pending_b |= 1 << b
                if not op.preads:
                    wait_mask |= pending_b
                elif pending_b:
                    nop = Op(lambda wr, rd, wait, stall: isa.encode_nop(stall=stall, wait=wait), forced_wait=pending_b)
                    nop.wait = pending_b
                    real_ops.append(nop)
                    reg_bar = {k: v for k, v in reg_bar.items() if not (pending_b & (1 << v))}
                    ureg_bar = {k: v for k, v in ureg_bar.items() if not (pending_b & (1 << v))}
                    rd_bar = {k: v for k, v in rd_bar.items() if not (pending_b & (1 << v))}
                    for b in range(6):
                        if pending_b & (1 << b):
                            bar_batch.pop(b, None)

            if op.var_lat:
                if op.forced_wr is not None:
                    b = op.forced_wr
                    same_batch = op.batch_id is not None and bar_batch.get(b) == op.batch_id
                    if not same_batch and (
                        any(v == b for v in reg_bar.values())
                        or any(v == b for v in ureg_bar.values())
                        or any(v == b for v in rd_bar.values())
                    ):
                        wait_mask |= 1 << b
                else:
                    b = next_bar
                    next_bar = (next_bar + 1) % dyn_bars
                    if (
                        any(v == b for v in reg_bar.values())
                        or any(v == b for v in ureg_bar.values())
                        or any(v == b for v in rd_bar.values())
                    ):
                        wait_mask |= 1 << b
                op.wr = b

            if op.mem_addr_reads:
                rb = next_bar
                next_bar = (next_bar + 1) % dyn_bars
                if rb == op.wr:
                    rb = next_bar
                    next_bar = (next_bar + 1) % dyn_bars
                if (
                    any(v == rb for v in reg_bar.values())
                    or any(v == rb for v in ureg_bar.values())
                    or any(v == rb for v in rd_bar.values())
                ):
                    wait_mask |= 1 << rb
                op.rd = rb

            op.wait = wait_mask
            if wait_mask and not op.preads:
                reg_bar = {k: v for k, v in reg_bar.items() if not (wait_mask & (1 << v))}
                ureg_bar = {k: v for k, v in ureg_bar.items() if not (wait_mask & (1 << v))}
                rd_bar = {k: v for k, v in rd_bar.items() if not (wait_mask & (1 << v))}
                for b in range(6):
                    if wait_mask & (1 << b):
                        bar_batch.pop(b, None)

            if op.var_lat:
                b = op.wr
                for r in op.writes:
                    reg_bar[r] = b
                for u in op.uwrites:
                    ureg_bar[u] = b
                bar_batch[b] = op.batch_id
            if op.mem_addr_reads:
                rb = op.rd
                for r in op.mem_addr_reads:
                    rd_bar[r] = rb
            real_ops.append(op)

        ready_r = {}
        war_ready_r = {}
        ready_p = {}
        bar_issue_cycle = {}
        cur_cycle = 0
        last_real = None

        for op in real_ops:
            if op.label is not None:
                if last_real is not None and last_real.stall < 6:
                    last_real.stall = 6
                ready_r.clear()
                war_ready_r.clear()
                ready_p.clear()
                cur_cycle += 6
                last_real = None
                continue

            need = cur_cycle
            for r in op.reads:
                if r in ready_r:
                    p_cyc, p_pipe, p_lat = ready_r[r]
                    if p_pipe == "fp32" and op.pipe == "fp32":
                        lat = 4
                    else:
                        lat = p_lat
                    if p_cyc + lat > need:
                        need = p_cyc + lat
            for r in op.writes:
                if r in war_ready_r and war_ready_r[r] > need:
                    need = war_ready_r[r]
            for r in op.mem_addr_reads:
                if r in ready_r:
                    p_cyc, _, p_lat = ready_r[r]
                    if p_cyc + p_lat + 1 > need:
                        need = p_cyc + p_lat + 1
            for p in op.preads:
                if p in ready_p:
                    req = ready_p[p]
                    if (op.is_branch or op.is_exit or op.mem_addr_reads) and req + 8 > need:
                        need = req + 8
                    elif req > need:
                        need = req
            if op.wait:
                for b in range(6):
                    if (op.wait & (1 << b)) and b in bar_issue_cycle:
                        req_b = bar_issue_cycle[b] + 5
                        if req_b > need:
                            need = req_b

            gap = need - cur_cycle
            if gap > 0 and last_real is not None:
                add = min(gap, 15 - last_real.stall)
                last_real.stall += add
                cur_cycle += add

            op.stall = 1
            if op.is_exit or op.is_branch or op.is_bar:
                op.stall = 5

            for r in op.writes:
                ready_r[r] = (cur_cycle, op.pipe, op.fixed_lat)
            for r in op.mem_addr_reads:
                war_ready_r[r] = max(war_ready_r.get(r, 0), cur_cycle + 6)
            if op.var_lat:
                for r in op.reads:
                    war_ready_r[r] = max(war_ready_r.get(r, 0), cur_cycle + 6)
            for p in op.pwrites:
                ready_p[p] = cur_cycle + 5
            if op.var_lat:
                bar_issue_cycle[op.wr] = cur_cycle
            if op.mem_addr_reads:
                bar_issue_cycle[op.rd] = cur_cycle

            cur_cycle += op.stall
            last_real = op

        label_pc = {}
        pc = 0
        for op in real_ops:
            if op.label is not None:
                label_pc[op.label] = pc
            else:
                pc += 16

        insns = []
        pc = 0
        for op in real_ops:
            if op.label is not None:
                continue
            if op.is_branch:
                delta = label_pc[op.target_label] - (pc + 16)
                insns.append(op.encoder(delta, op.wr, op.rd, op.wait, op.stall))
            else:
                insns.append(op.encoder(op.wr, op.rd, op.wait, op.stall))
            pc += 16
        return insns


def compile_program(prog):
    consts = prog["consts"]
    return [KernelCompiler(kdef, consts).compile() for kdef in prog["kernels"]]
