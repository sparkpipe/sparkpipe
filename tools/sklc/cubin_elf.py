import hashlib
import struct

from isa_sm121 import C0_PARAM_BASE, encode_bra, encode_nop, pack_insn

EI_CLASS_64 = 2
EI_DATA_LE = 1
EI_OSABI_CUDA = 0x41
EI_ABIVERSION_BLACKWELL = 8
ET_EXEC = 2
EM_CUDA = 190

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_CUDA_INFO = 0x70000000
SHT_CUDA_COMPAT = 0x70000086

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHF_INFO_LINK = 0x40
SHF_NV_CUINFO = 0x01000040
SHF_NV_TKINFO = 0x02000000

PT_LOAD = 1
PT_PHDR = 6
PF_X = 1
PF_W = 2
PF_R = 4

ARCH_FLAGS = {
    "sm121": 0x06007902,
    "sm121a": 0x06007902,
    "sm120": 0x06007802,
    "sm120a": 0x06007802,
}

ARCH_CANONICAL = {
    "sm121": "sm_121a",
    "sm121a": "sm_121a",
    "sm120": "sm_120a",
    "sm120a": "sm_120a",
}


class StringTable:
    def __init__(self):
        self.buf = bytearray(b"\x00")
        self.map = {"": 0}

    def add(self, text):
        if text in self.map:
            return self.map[text]
        off = len(self.buf)
        self.buf.extend(text.encode("utf-8") + b"\x00")
        self.map[text] = off
        return off

    def data(self):
        return bytes(self.buf)


class KernelImage:
    def __init__(self, name, insns, params, registers, max_threads=256, barriers=0, smem_bytes=0, stack_bytes=0):
        self.name = name
        self.insns = list(insns)
        self.params = list(params)
        self.registers = max(16, int(registers))
        self.max_threads = int(max_threads)
        self.barriers = int(barriers)
        self.smem_bytes = int(smem_bytes)
        self.stack_bytes = int(stack_bytes)

    def param_layout(self):
        layout = []
        off = 0
        for pname, pkind, palign, pbytes in self.params:
            off = (off + palign - 1) // palign * palign
            layout.append((pname, pkind, palign, pbytes, off))
            off += pbytes
        return layout, off

    def code_bytes(self):
        exit_offsets = []
        body = bytearray()
        for idx, (lo, hi) in enumerate(self.insns):
            if (lo & 0xFFF) == 0x94D:
                exit_offsets.append(idx * 16)
            body.extend(pack_insn(lo, hi))
        body.extend(pack_insn(*encode_bra(-16, stall=0, yield_=True)))
        while len(body) % 128 != 0:
            body.extend(pack_insn(*encode_nop(stall=0, yield_=True)))
        return bytes(body), exit_offsets


def nvinfo_attr_val(attr, payload):
    return struct.pack("<BBH", 4, attr, len(payload)) + payload


def nvinfo_attr_hword(attr, val):
    return struct.pack("<BBH", 3, attr, val & 0xFFFF)


def nvinfo_attr_byte(attr, val):
    return struct.pack("<BBH", 2, attr, val & 0xFF)


def align_up(value, align):
    return value if align <= 1 else (value + align - 1) // align * align


def nvrm_short_name(name):
    if len(name) <= 64:
        return name
    base = name
    if name.startswith("_Z"):
        idx = 2
        if idx < len(name) and name[idx] == "N":
            idx += 1
            while idx < len(name) and name[idx] in "rVKRO":
                idx += 1
        while idx < len(name):
            if name[idx] == "L":
                idx += 1
                continue
            if not name[idx].isdigit():
                break
            end = idx
            while end < len(name) and name[end].isdigit():
                end += 1
            count = int(name[idx:end])
            if count == 0 or len(name) - end < count:
                break
            base = name[end:end + count]
            idx = end + count
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{base[:51]}_{digest}"


ARCH_SM_NUM = {
    "sm121": 121,
    "sm121a": 121,
    "sm120": 120,
    "sm120a": 120,
}


def build_tkinfo_note(canon_arch):
    tool = b"sklc\x00"
    ver = b"Cuda compilation tools, release 13.0, V13.0.88\x00"
    build = b"Build sklc-4.0\x00"
    opts = f"-arch {canon_arch} -m 64 \x00".encode("ascii")
    s1 = len(tool)
    s2 = s1 + len(ver)
    s3 = s2 + len(build)
    strings = b"\x00" + tool + ver + build + opts
    hdr = struct.pack("<IIIIII", 2, 0, 1, 1 + s1, 1 + s2, 1 + s3)
    desc = hdr + strings
    while len(desc) % 4 != 0:
        desc += b"\x00"
    return struct.pack("<III12s", 12, len(desc), 2000, b"NVIDIA Corp\x00") + desc


def build_cuinfo_note(sm_num):
    desc = struct.pack("<HHI", 2, sm_num, 130)
    return struct.pack("<III12s", 12, len(desc), 1000, b"NVIDIA Corp\x00") + desc


def build_cubin(kernels, arch="sm121"):
    if arch not in ARCH_FLAGS:
        raise ValueError(f"unsupported arch: {arch}")
    e_flags = ARCH_FLAGS[arch]
    canon_arch = ARCH_CANONICAL[arch]
    sm_num = ARCH_SM_NUM[arch]
    shstr = StringTable()
    strtab = StringTable()

    sections = []

    def add_section(name, stype, flags, data, link=0, info=0, align=1, entsize=0, nobits_size=0):
        idx = len(sections) + 1
        sections.append({
            "idx": idx,
            "name": name,
            "name_off": shstr.add(name),
            "type": stype,
            "flags": flags,
            "data": data,
            "size": nobits_size if stype == SHT_NOBITS else len(data),
            "link": link,
            "info": info,
            "align": align,
            "entsize": entsize,
            "offset": 0,
        })
        return idx

    shstr_idx = add_section(".shstrtab", SHT_STRTAB, 0, b"", align=1)
    strtab_idx = add_section(".strtab", SHT_STRTAB, 0, b"", align=1)
    symtab_idx = add_section(".symtab", SHT_SYMTAB, 0, b"", link=strtab_idx, info=1, align=8, entsize=24)
    tkinfo_idx = add_section(".note.nv.tkinfo", SHT_NOTE, SHF_NV_TKINFO, build_tkinfo_note(canon_arch), link=0, info=0, align=4)
    cuinfo_idx = add_section(".note.nv.cuinfo", SHT_NOTE, SHF_NV_CUINFO, build_cuinfo_note(sm_num), link=tkinfo_idx, info=0, align=4)
    nvinfo_idx = add_section(".nv.info", SHT_CUDA_INFO, 0, b"", link=symtab_idx, info=0, align=4)
    nvcompat_data = (
        nvinfo_attr_byte(0x09, 1)
        + nvinfo_attr_byte(0x02, 1)
        + nvinfo_attr_byte(0x05, 5)
        + nvinfo_attr_hword(0x07, 0x0101)
        + nvinfo_attr_byte(0x03, 0)
        + nvinfo_attr_byte(0x06, 1)
        + nvinfo_attr_val(0x0B, b"\x00" * 8)
    )
    nvcompat_idx = add_section(".nv.compat", SHT_CUDA_COMPAT, 0, nvcompat_data, link=0, info=0, align=4)
    sections[cuinfo_idx - 1]["info"] = nvcompat_idx

    sym_locals = [struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)]
    sym_globals = []
    kernel_meta = []

    prepared = []
    for k in kernels:
        code, exit_offsets = k.code_bytes()
        layout, param_bytes = k.param_layout()
        const0_size = C0_PARAM_BASE + param_bytes
        info_k_idx = add_section(
            f".nv.info.{k.name}",
            SHT_CUDA_INFO,
            SHF_INFO_LINK,
            b"",
            link=symtab_idx,
            info=0,
            align=4,
        )
        prepared.append({
            "kernel": k,
            "code": code,
            "code_len": len(code),
            "exit_offsets": exit_offsets,
            "layout": layout,
            "param_bytes": param_bytes,
            "const0_size": const0_size,
            "info_k_idx": info_k_idx,
            "text_idx": 0,
            "smem_idx": 0,
            "const0_idx": 0,
            "const0_sym": 0,
            "func_sym": 0,
        })

    for m in prepared:
        k = m["kernel"]
        text_idx = add_section(
            f".text.{k.name}",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_EXECINSTR,
            m["code"],
            link=symtab_idx,
            info=0,
            align=128,
        )
        m["text_idx"] = text_idx
        sections[m["info_k_idx"] - 1]["info"] = text_idx
        sym_locals.append(struct.pack("<IBBHQQ", strtab.add(f".text.{k.name}"), 0x03, 0, text_idx, 0, 0))

    for m in prepared:
        k = m["kernel"]
        if k.smem_bytes > 0:
            m["smem_idx"] = add_section(
                f".nv.shared.{k.name}",
                SHT_NOBITS,
                SHF_WRITE | SHF_ALLOC | SHF_INFO_LINK,
                b"",
                link=0,
                info=m["text_idx"],
                align=16,
                nobits_size=k.smem_bytes,
            )

    for m in prepared:
        k = m["kernel"]
        const0_idx = add_section(
            f".nv.constant0.{k.name}",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_INFO_LINK,
            b"\x00" * m["const0_size"],
            link=0,
            info=m["text_idx"],
            align=4,
        )
        m["const0_idx"] = const0_idx
        m["const0_sym"] = len(sym_locals)
        sym_locals.append(struct.pack("<IBBHQQ", strtab.add(f".nv.constant0.{k.name}"), 0x03, 0, const0_idx, 0, 0))
        kernel_meta.append(m)

    local_count = len(sym_locals)
    for m in kernel_meta:
        k = m["kernel"]
        func_sym = local_count + len(sym_globals)
        m["func_sym"] = func_sym
        sym_globals.append(struct.pack("<IBBHQQ", strtab.add(k.name), 0x12, 0x10, m["text_idx"], 0, m["code_len"]))
        bar_bits = (k.barriers & 0x7F) << 20
        sections[m["text_idx"] - 1]["info"] = func_sym | bar_bits

    nvinfo_global = bytearray()
    for m in kernel_meta:
        k = m["kernel"]
        fsym = m["func_sym"]
        nvinfo_global.extend(nvinfo_attr_val(0x2F, struct.pack("<II", fsym, k.registers)))
        nvinfo_global.extend(nvinfo_attr_val(0x11, struct.pack("<II", fsym, k.stack_bytes)))
        nvinfo_global.extend(nvinfo_attr_val(0x12, struct.pack("<II", fsym, k.stack_bytes)))

        info_k = bytearray()
        info_k.extend(nvinfo_attr_val(0x37, struct.pack("<I", 130)))
        for ordinal in range(len(m["layout"]) - 1, -1, -1):
            _, pkind, palign, pbytes, poff = m["layout"][ordinal]
            space = 5 if pkind == "ptr" else 0
            flags_word = ((pbytes & 0x3FFF) << 18) | (0x1F << 12) | ((space & 0xF) << 8)
            info_k.extend(nvinfo_attr_val(0x17, struct.pack("<IHHI", 0, ordinal, poff, flags_word)))
        info_k.extend(nvinfo_attr_hword(0x50, 0))
        info_k.extend(nvinfo_attr_hword(0x1B, 0x00FF))
        info_k.extend(nvinfo_attr_hword(0x5F, 0x0101))
        info_k.extend(nvinfo_attr_byte(0x4A, 0))
        if m["exit_offsets"]:
            info_k.extend(nvinfo_attr_val(0x1C, b"".join(struct.pack("<I", o) for o in m["exit_offsets"])))
        info_k.extend(nvinfo_attr_val(0x05, struct.pack("<III", k.max_threads, 1, 1)))
        info_k.extend(nvinfo_attr_hword(0x19, m["param_bytes"]))
        info_k.extend(nvinfo_attr_val(0x0A, struct.pack("<IHH", m["const0_sym"], C0_PARAM_BASE, m["param_bytes"])))
        info_k.extend(nvinfo_attr_val(0x36, struct.pack("<I", 0)))
        sections[m["info_k_idx"] - 1]["data"] = bytes(info_k)
        sections[m["info_k_idx"] - 1]["size"] = len(info_k)

    sections[nvinfo_idx - 1]["data"] = bytes(nvinfo_global)
    sections[nvinfo_idx - 1]["size"] = len(nvinfo_global)

    symtab_bytes = b"".join(sym_locals + sym_globals)
    sections[symtab_idx - 1]["data"] = symtab_bytes
    sections[symtab_idx - 1]["size"] = len(symtab_bytes)
    sections[symtab_idx - 1]["info"] = local_count

    strtab_bytes = strtab.data()
    sections[strtab_idx - 1]["data"] = strtab_bytes
    sections[strtab_idx - 1]["size"] = len(strtab_bytes)

    shstr_bytes = shstr.data()
    sections[shstr_idx - 1]["data"] = shstr_bytes
    sections[shstr_idx - 1]["size"] = len(shstr_bytes)

    blob = bytearray(64)
    for sec in sections:
        off = align_up(len(blob), sec["align"])
        if off > len(blob):
            blob.extend(b"\x00" * (off - len(blob)))
        sec["offset"] = off
        if sec["type"] != SHT_NOBITS:
            blob.extend(sec["data"])

    shoff = align_up(len(blob), 8)
    if shoff > len(blob):
        blob.extend(b"\x00" * (shoff - len(blob)))

    shdrs = bytearray()
    shdrs.extend(struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    for sec in sections:
        shdrs.extend(struct.pack(
            "<IIQQQQIIQQ",
            sec["name_off"],
            sec["type"],
            sec["flags"],
            0,
            sec["offset"],
            sec["size"],
            sec["link"],
            sec["info"],
            sec["align"],
            sec["entsize"],
        ))
    blob.extend(shdrs)

    phoff = align_up(len(blob), 8)
    if phoff > len(blob):
        blob.extend(b"\x00" * (phoff - len(blob)))

    text_secs = [s for s in sections if (s["flags"] & SHF_EXECINSTR) != 0]
    smem_secs = [s for s in sections if s["type"] == SHT_NOBITS]
    const_secs = [s for s in sections if (s["flags"] & SHF_ALLOC) != 0 and not (s["flags"] & SHF_EXECINSTR) and s["type"] != SHT_NOBITS]
    extra_ph = []
    if text_secs:
        t_start = text_secs[0]["offset"]
        t_end = text_secs[-1]["offset"] + text_secs[-1]["size"]
        extra_ph.append((PT_LOAD, PF_R | PF_X, t_start, t_end - t_start, t_end - t_start))
    if smem_secs:
        s_off = smem_secs[0]["offset"]
        s_mem = max(s["size"] for s in smem_secs)
        extra_ph.append((PT_LOAD, PF_R | PF_W, s_off, 0, s_mem))
    if const_secs:
        c_start = const_secs[0]["offset"]
        c_end = const_secs[-1]["offset"] + const_secs[-1]["size"]
        extra_ph.append((PT_LOAD, PF_R, c_start, c_end - c_start, c_end - c_start))

    phnum = 2 + len(extra_ph)
    ph_bytes = phnum * 56
    phdrs = bytearray()
    phdrs.extend(struct.pack("<IIQQQQQQ", PT_PHDR, PF_R, phoff, 0, 0, ph_bytes, ph_bytes, 8))
    phdrs.extend(struct.pack("<IIQQQQQQ", PT_LOAD, PF_R, phoff, 0, 0, ph_bytes, ph_bytes, 8))
    for ptype, pflags, poff, filesz, memsz in extra_ph:
        phdrs.extend(struct.pack("<IIQQQQQQ", ptype, pflags, poff, 0, 0, filesz, memsz, 8))
    blob.extend(phdrs)

    e_ident = b"\x7fELF" + bytes([EI_CLASS_64, EI_DATA_LE, 1, EI_OSABI_CUDA, EI_ABIVERSION_BLACKWELL]) + b"\x00" * 7
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident,
        ET_EXEC,
        EM_CUDA,
        1,
        0,
        phoff,
        shoff,
        e_flags,
        64,
        56,
        phnum,
        64,
        len(sections) + 1,
        shstr_idx,
    )
    blob[:64] = ehdr
    return bytes(blob)


def build_manifest(kernels, cubin_bytes, source_path, source_bytes, arch="sm121"):
    canon = ARCH_CANONICAL[arch]
    lines = [
        "nvcc sklc-4.0",
        f"flags --arch {canon}",
        f"source {source_path} {hashlib.sha256(source_bytes).hexdigest()}",
        f"cubin {hashlib.sha256(cubin_bytes).hexdigest()} {len(cubin_bytes)}",
    ]
    for k in sorted(kernels, key=lambda x: x.name):
        short = nvrm_short_name(k.name)
        lines.append(f"kernel {k.name} short {short} registers {k.registers} barriers {k.barriers} stack {k.stack_bytes} spill_stores 0 spill_loads 0 smem {k.smem_bytes}")
        layout, _ = k.param_layout()
        for idx, (_, pkind, palign, pbytes, poff) in enumerate(layout):
            lines.append(f"param {k.name} {idx} {pkind} align {palign} bytes {pbytes} offset {poff}")
        lines.append(f"bound {k.name} maxntid {k.max_threads} 1 1")
    return "\n".join(lines) + "\n"
