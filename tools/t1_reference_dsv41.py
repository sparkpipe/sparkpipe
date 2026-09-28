import json
import math
import os
import re
import sys
import time

import numpy as np

_DEBUG = bool(os.environ.get("T1_REF_DSV41_DEBUG"))
_NO_ENGRAM = bool(os.environ.get("T1_REF_DSV41_NO_ENGRAM"))
_NO_COMPRESS = bool(os.environ.get("T1_REF_DSV41_NO_COMPRESS"))
_NIBBLE_HIGH = bool(os.environ.get("T1_REF_DSV41_NIBBLE_HIGH"))

from t1_reference_common import (Safetensors, bf16_round_f32, bf16_to_f32,
                                 define_float, define_uint, f32_to_bf16_u16,
                                 rmsnorm, sigmoid)

PREFIX = "layers."
LINEAR_BLOCK_ROWS = 4096
ENGRAM_MAP_REPO_PATH = os.path.join("qualification", "t1_reference", "dsv41",
                                    "engram_token_map.npy")

_E4M3_F32 = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _s = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m / 8.0 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
    _E4M3_F32[_i] = _s * _v

_E4M3_POS = []
_E4M3_CODE = []
for _c in range(1, 0x7F):
    _e = (_c >> 3) & 0xF
    _m = _c & 0x7
    _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7) if _e else (_m / 8.0) * 2.0 ** -6
    _E4M3_POS.append(_v)
    _E4M3_CODE.append(_c)
_E4M3_POS = np.array(_E4M3_POS, dtype=np.float32)
_E4M3_CODE = np.array(_E4M3_CODE, dtype=np.int32)

_E2M1_F32 = np.zeros(16, dtype=np.float32)
for _i in range(16):
    _s = -1.0 if _i & 0x8 else 1.0
    _e = (_i >> 1) & 0x3
    _m = _i & 0x1
    _E2M1_F32[_i] = _s * ((1.0 + _m / 2.0) * 2.0 ** _e if _e else _m / 2.0)

_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                     dtype=np.float32)


def _grid_round_magnitude(a, grid, codes):
    out = np.zeros_like(a)
    nz = a > 0
    if not nz.any():
        return out
    v = a[nz]
    idx = np.searchsorted(grid, v)
    lo = np.clip(idx - 1, 0, grid.size - 1)
    hi = np.clip(idx, 0, grid.size - 1)
    dlo = v - grid[lo]
    dhi = grid[hi] - v
    pick_hi = dhi < dlo
    pick_hi = pick_hi | ((dhi == dlo) & ((codes[hi] & 1) == 1))
    out[nz] = np.where(pick_hi, grid[hi], grid[lo])
    return out


def _fp8_round_abs(a):
    return _grid_round_magnitude(a, _E4M3_POS, _E4M3_CODE)


def _fp4_round_abs(a):
    return _grid_round_magnitude(a, _E2M1_POS,
                                 np.arange(8, dtype=np.int32))


def _ceil_log2_pow2(t):
    m, e = np.frexp(t.astype(np.float32))
    return (e - (m == np.float32(0.5))).astype(np.int32)


def _exp2_rows(scale_int):
    return np.exp2(scale_int.astype(np.float32))


def _fp8_qdq_rows(x):
    n, d = x.shape
    b = x.reshape(n, d // 32, 32)
    amax = np.maximum(np.abs(b).max(-1, keepdims=True), 1e-4)
    scale = _exp2_rows(_ceil_log2_pow2(amax / np.float32(448.0)))
    q = np.clip(b / scale, -448.0, 448.0)
    deq = np.sign(q) * _fp8_round_abs(np.abs(q)) * scale
    return deq.reshape(n, d)


def _fp4_qdq_e8m0(x):
    n, d = x.shape
    b = x.reshape(n, d // 32, 32)
    amax = np.maximum(np.abs(b).max(-1, keepdims=True),
                      np.float32(6.0 * 2.0 ** -126))
    scale = _exp2_rows(_ceil_log2_pow2(amax / np.float32(6.0)))
    q = np.clip(b / scale, -6.0, 6.0)
    deq = np.sign(q) * _fp4_round_abs(np.abs(q)) * scale
    return deq.reshape(n, d)


def _fp4_qdq_e4m3_16(x):
    n, d = x.shape
    b = x.reshape(n, d // 16, 16)
    amax = np.maximum(np.abs(b).max(-1, keepdims=True),
                      np.float32(6.0 * 2.0 ** -9))
    scale = _fp8_round_abs(amax / np.float32(6.0))
    q = np.clip(b / scale, -6.0, 6.0)
    deq = np.sign(q) * _fp4_round_abs(np.abs(q)) * scale
    return deq.reshape(n, d)


def _softplus(x):
    return np.logaddexp(np.float32(0), x)


def _silu(x):
    return x * sigmoid(x)


def _softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _next_prime(start, seen):
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def _config_value(config, path):
    if "|" in path:
        for option in path.split("|"):
            try:
                return _config_value(config, option)
            except (KeyError, TypeError, IndexError):
                continue
        raise KeyError(path)
    node = config
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z0-9_]+)\[(\d+)\]", part)
        if match:
            node = node[match.group(1)][int(match.group(2))]
        else:
            node = node[part]
    return node


class Dsv41ConfigError(ValueError):
    pass


DEFINES_VS_CONFIG = [
    ("HIDDEN_DIMENSION", "hidden_size", "uint"),
    ("LAYER_COUNT", "num_hidden_layers", "uint"),
    ("OUTPUT_VOCAB_COUNT", "vocab_size", "uint"),
    ("RMS_NORM_EPSILON", "rms_norm_eps", "float"),
    ("ATTENTION_HEAD_COUNT", "num_attention_heads", "uint"),
    ("HEAD_DIMENSION", "head_dim", "uint"),
    ("QK_ROPE_HEAD_DIMENSION", "qk_rope_head_dim", "uint"),
    ("QUERY_LORA_RANK", "q_lora_rank", "uint"),
    ("OUTPUT_LORA_RANK", "o_lora_rank", "uint"),
    ("OUTPUT_GROUP_COUNT", "o_groups", "uint"),
    ("SLIDING_WINDOW_TOKENS", "sliding_window", "uint"),
    ("SWIGLU_LIMIT", "swiglu_limit", "float"),
    ("ROPE_THETA", "rope_theta", "float"),
    ("YARN_FACTOR", "rope_scaling.factor", "float"),
    ("YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS",
     "rope_scaling.original_max_position_embeddings", "uint"),
    ("YARN_BETA_FAST", "rope_scaling.beta_fast", "uint"),
    ("YARN_BETA_SLOW", "rope_scaling.beta_slow", "uint"),
    ("COMPRESS_ROPE_THETA", "compress_rope_theta", "float"),
    ("MOE_ROUTED_EXPERT_COUNT", "n_routed_experts", "uint"),
    ("MOE_EXPERTS_PER_TOKEN", "num_experts_per_tok", "uint"),
    ("MOE_INTERMEDIATE_DIMENSION", "moe_intermediate_size", "uint"),
    ("MOE_ROUTED_SCALING_FACTOR", "routed_scaling_factor", "float"),
    ("HC_MULT", "hc_mult", "uint"),
    ("HC_SINKHORN_ITERATIONS", "hc_sinkhorn_iters", "uint"),
    ("HC_EPS", "hc_eps", "float"),
    ("INDEX_HEAD_COUNT", "index_n_heads", "uint"),
    ("INDEX_HEAD_DIMENSION", "index_head_dim", "uint"),
    ("INDEX_TOP_K", "index_topk", "uint"),
    ("CANDIDATE_SOURCE_LAYER",
     "candidate_source_layer|candidate_source_layer_id", "uint"),
    ("CANDIDATE_TOPK_BLOCKS", "candidate_topk_blocks", "uint"),
    ("CANDIDATE_BLOCK_SIZE", "candidate_block_size", "uint"),
    ("KV_SOURCE_LAYER_0", "kv_source_layer_ids[0]", "uint"),
    ("KV_SOURCE_LAYER_1", "kv_source_layer_ids[1]", "uint"),
    ("KV_SOURCE_LAYER_2", "kv_source_layer_ids[2]", "uint"),
    ("KV_SOURCE_LAYER_3", "kv_source_layer_ids[3]", "uint"),
    ("INDEX_SOURCE_LAYER_0", "index_source_layer_ids[0]", "uint"),
    ("INDEX_SOURCE_LAYER_1", "index_source_layer_ids[1]", "uint"),
    ("INDEX_SOURCE_LAYER_2", "index_source_layer_ids[2]", "uint"),
    ("INDEX_SOURCE_LAYER_3", "index_source_layer_ids[3]", "uint"),
    ("INDEX_SOURCE_LAYER_4", "index_source_layer_ids[4]", "uint"),
    ("INDEX_SOURCE_LAYER_5", "index_source_layer_ids[5]", "uint"),
    ("INDEX_SOURCE_LAYER_6", "index_source_layer_ids[6]", "uint"),
    ("INDEX_SOURCE_LAYER_7", "index_source_layer_ids[7]", "uint"),
]

ENGRAM_DEFINES_VS_CONFIG = [
    ("ENGRAM_MODULE_COUNT", "engram_layer_ids#len", "uint"),
    ("ENGRAM_LAYER_0", "engram_layer_ids[0]", "uint"),
    ("ENGRAM_LAYER_1", "engram_layer_ids[1]", "uint"),
    ("ENGRAM_HEAD_COUNT", "engram_n_heads", "uint"),
    ("ENGRAM_HEAD_DIMENSION", "engram_head_dim", "uint"),
    ("ENGRAM_MAX_NGRAM_SIZE", "engram_max_ngram_size", "uint"),
    ("ENGRAM_VOCAB_SIZE", "engram_vocab_size", "uint"),
    ("ENGRAM_COMPRESSED_VOCAB_SIZE",
     "engram_compressed_vocab_size", "uint"),
    ("ENGRAM_PAD_TOKEN_ID", "engram_pad_token_id", "uint"),
]

DEFINES_RECORDED_ONLY = [
    ("KV_GLOBAL_SCALE_CHANNELS",
     "compressed-KV fp4 activation scale group width 16 per the official "
     "kernel"),
]

RATIO_HEADER_RE = re.compile(
    r"SPARK_DSV41_FLASH_MODEL_LAYER_COMPRESSION_RATIO\(layer_index\)"
    r"[\s\\]*\(\(uint32_t\[\]\)\{([^}]+)\}\[\(layer_index\)\]\)")


def cross_check(defines, config, engram_enabled):
    checks = list(DEFINES_VS_CONFIG)
    if engram_enabled:
        checks += ENGRAM_DEFINES_VS_CONFIG
    for dname, cpath, kind in checks:
        want = define_uint(defines, dname) if kind == "uint" \
            else define_float(defines, dname)
        try:
            if cpath.endswith("#len"):
                got = len(_config_value(config, cpath[:-4]))
            else:
                got = _config_value(config, cpath)
        except (KeyError, TypeError, IndexError):
            raise Dsv41ConfigError(
                f"config key {cpath} missing for SPARK_LLM_{dname}")
        if isinstance(got, list):
            got = got[0]
        if abs(float(want) - float(got)) > 0:
            raise Dsv41ConfigError(
                f"SPARK_LLM_{dname}={want} disagrees with config "
                f"{cpath}={got}")
    mismatches = []
    for dname, note in DEFINES_RECORDED_ONLY:
        if dname not in defines:
            continue
        mismatches.append({
            "define": f"SPARK_LLM_{dname}",
            "value": define_uint(defines, dname),
            "note": note,
        })
    return mismatches


def _parse_family_header(path, defines, ratios):
    text = open(path).read()
    problems = []
    checked = []
    findings = []
    simple = [
        ("SPARK_DSV41_FLASH_MODEL_HIDDEN_DIMENSION", "HIDDEN_DIMENSION",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_LAYER_COUNT", "LAYER_COUNT", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_OUTPUT_VOCAB_COUNT", "OUTPUT_VOCAB_COUNT",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_RMS_NORM_EPSILON", "RMS_NORM_EPSILON",
         "float"),
        ("SPARK_DSV41_FLASH_MODEL_EOS_TOKEN_ID", "END_OF_TEXT_TOKEN_ID",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_ATTENTION_HEAD_COUNT",
         "ATTENTION_HEAD_COUNT", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_HEAD_DIMENSION", "HEAD_DIMENSION", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_QK_ROPE_HEAD_DIMENSION",
         "QK_ROPE_HEAD_DIMENSION", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_QUERY_LORA_RANK", "QUERY_LORA_RANK",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_OUTPUT_LORA_RANK", "OUTPUT_LORA_RANK",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_OUTPUT_GROUP_COUNT", "OUTPUT_GROUP_COUNT",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_SLIDING_WINDOW", "SLIDING_WINDOW_TOKENS",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_ROUTED_EXPERT_COUNT",
         "MOE_ROUTED_EXPERT_COUNT", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_EXPERTS_PER_TOKEN",
         "MOE_EXPERTS_PER_TOKEN", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_MOE_INTERMEDIATE_DIMENSION",
         "MOE_INTERMEDIATE_DIMENSION", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_HC_MULT", "HC_MULT", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_COUNT", "INDEX_HEAD_COUNT",
         "uint"),
        ("SPARK_DSV41_FLASH_MODEL_INDEX_HEAD_DIMENSION",
         "INDEX_HEAD_DIMENSION", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_INDEX_TOP_K", "INDEX_TOP_K", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_CANDIDATE_SOURCE_LAYER",
         "CANDIDATE_SOURCE_LAYER", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_CANDIDATE_BLOCK_COUNT",
         "CANDIDATE_TOPK_BLOCKS", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_CANDIDATE_BLOCK_SIZE",
         "CANDIDATE_BLOCK_SIZE", "uint"),
        ("SPARK_DSV41_FLASH_MODEL_ENGRAM_MODULE_COUNT",
         "ENGRAM_MODULE_COUNT", "uint"),
    ]
    for macro, dname, kind in simple:
        if dname not in defines:
            continue
        match = re.search(rf"#define\s+{macro}\s+([0-9.eE+-]+)f?u?\s*$",
                          text, re.M)
        if match is None:
            problems.append(f"{macro} not found in family header")
            continue
        value = float(match.group(1))
        want = float(define_uint(defines, dname) if kind == "uint"
                     else define_float(defines, dname))
        checked.append(macro)
        if value != want:
            problems.append(
                f"{macro}={value} disagrees with SPARK_LLM_{dname}={want}")
    match = RATIO_HEADER_RE.search(text)
    if match is None:
        problems.append("compression ratio table not found in family header")
    else:
        table = [int(v) for v in
                 re.split(r"[\s\\,]+", match.group(1).replace("u", ""))
                 if v]
        if table != list(ratios):
            findings.append({
                "family_header_ratio_table": table,
                "checkpoint_compress_ratios": list(ratios),
                "note": "the family header compression table disagrees "
                        "with the checkpoint; the checkpoint and both "
                        "published configs agree, so the reference "
                        "follows the checkpoint",
            })
        else:
            checked.append("LAYER_COMPRESSION_RATIO")
    return checked, problems, findings


class Dsv41FlashEngine:
    def __init__(self, checkpoint_dir, defines, config):
        self.checkpoint_dir = checkpoint_dir
        self.st = Safetensors(checkpoint_dir)
        full = json.load(open(os.path.join(checkpoint_dir, "config.json")))
        text = full.get("text_config", full)
        engram_ids = list(text.get("engram_layer_ids") or [])
        self.mismatches = cross_check(defines, config, bool(engram_ids))
        self.hidden = int(text["hidden_size"])
        self.layers = int(text["num_hidden_layers"])
        self.vocab = int(text["vocab_size"])
        self.eps = float(text["rms_norm_eps"])
        self.heads = int(text["num_attention_heads"])
        self.head_dim = int(text["head_dim"])
        self.rope_dim = int(text["qk_rope_head_dim"])
        self.q_lora = int(text["q_lora_rank"])
        self.o_lora = int(text["o_lora_rank"])
        self.o_groups = int(text["o_groups"])
        self.window = int(text["sliding_window"])
        self.limit = float(text["swiglu_limit"])
        self.rope_theta = float(text["rope_theta"])
        self.yarn_factor = float(text["rope_scaling"]["factor"])
        self.yarn_original = int(
            text["rope_scaling"]["original_max_position_embeddings"])
        self.beta_fast = int(text["rope_scaling"]["beta_fast"])
        self.beta_slow = int(text["rope_scaling"]["beta_slow"])
        self.compress_theta = float(text["compress_rope_theta"])
        self.experts = int(text["n_routed_experts"])
        self.topk = int(text["num_experts_per_tok"])
        self.moe_inter = int(text["moe_intermediate_size"])
        self.route_scale = float(text["routed_scaling_factor"])
        self.hc_mult = int(text["hc_mult"])
        self.sinkhorn = int(text["hc_sinkhorn_iters"])
        self.hc_eps = float(text["hc_eps"])
        self.index_heads = int(text["index_n_heads"])
        self.index_dim = int(text["index_head_dim"])
        self.index_topk = int(text["index_topk"])
        self.candidate_source = int(text.get(
            "candidate_source_layer",
            text.get("candidate_source_layer_id", -1)))
        self.candidate_blocks = int(text["candidate_topk_blocks"])
        self.candidate_block = int(text["candidate_block_size"])
        self.eot = define_uint(defines, "END_OF_TEXT_TOKEN_ID")
        if int(full.get("eos_token_id", self.eot)) != self.eot:
            raise Dsv41ConfigError(
                f"checkpoint eos_token_id={full.get('eos_token_id')} "
                f"disagrees with SPARK_LLM_END_OF_TEXT_TOKEN_ID={self.eot}")
        self.mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.hc_dim = self.hc_mult * self.hidden
        self.heads_per_group = self.heads // self.o_groups
        if self.o_groups * self.heads_per_group != self.heads:
            raise Dsv41ConfigError("attention group geometry inconsistent")
        ratios = list(text["compress_ratios"])
        if len(ratios) < self.layers:
            raise Dsv41ConfigError(
                f"compress_ratios has {len(ratios)} entries for "
                f"{self.layers} layers")
        self.ratios = ratios[:self.layers]
        self.kv_sources = set(int(v) for v in text["kv_source_layer_ids"])
        self.index_sources = set(int(v) for v in text["index_source_layer_ids"])
        for layer in sorted(self.kv_sources | self.index_sources):
            if layer >= self.layers or self.ratios[layer] <= 0:
                raise Dsv41ConfigError(
                    f"source layer {layer} has compress ratio "
                    f"{self.ratios[layer] if layer < self.layers else 'na'}")
        header_path = os.environ.get("DSV41_FAMILY_HEADER")
        if header_path:
            checked, problems, findings = _parse_family_header(
                header_path, defines, self.ratios)
            if problems:
                raise Dsv41ConfigError("; ".join(problems))
            self.mismatches.append({
                "family_header": os.path.abspath(header_path),
                "checked": checked,
            })
            self.mismatches.extend(findings)
        self.full_freqs = self._freqs(self.rope_theta, 0)
        self.compress_freqs = self._freqs(self.compress_theta,
                                          self.yarn_original)
        self._engram_init(text, engram_ids)
        self._convert_u32 = np.empty(0, dtype=np.uint32)
        self._head_input = None

    def _freqs(self, theta, original):
        inv = theta ** (-np.arange(0, self.rope_dim, 2, dtype=np.float64)
                        / self.rope_dim)
        if original > 0:
            def corrected(rotations):
                return (self.rope_dim
                        * math.log(original / (rotations * 2 * math.pi))
                        / (2 * math.log(theta)))

            low = max(math.floor(corrected(self.beta_fast)), 0)
            high = min(math.ceil(corrected(self.beta_slow)),
                       self.rope_dim - 1)
            ramp = np.clip((np.arange(self.rope_dim // 2, dtype=np.float64)
                            - low) / max(high - low, 1e-3), 0.0, 1.0)
            smooth = 1.0 - ramp
            inv = inv / self.yarn_factor * (1.0 - smooth) + inv * smooth
        return inv.astype(np.float64)

    def _rotate(self, rows, table, position, inverse=False):
        angle = np.float64(position) * table
        cos = np.cos(angle)
        sin = np.sin(angle)
        if inverse:
            sin = -sin
        real = rows[..., 0::2].copy()
        imag = rows[..., 1::2].copy()
        rows[..., 0::2] = real * cos - imag * sin
        rows[..., 1::2] = imag * cos + real * sin
        return rows

    def _engram_init(self, text, layer_ids):
        self.engram_layers = layer_ids
        if not layer_ids:
            self.engram = False
            return
        count = len(layer_ids)
        ngram = int(text["engram_max_ngram_size"])
        heads = int(text["engram_n_heads"])
        bucket = int(text["engram_vocab_size"])
        primes = []
        seen = set()
        for _ in range(count):
            per_layer = []
            for _ in range(ngram - 1):
                row = []
                current = bucket - 1
                for _ in range(heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    row.append(current)
                per_layer.append(row)
            primes.append(per_layer)
        self.engram_primes = np.array(primes, dtype=np.int64)
        offsets = []
        for per_layer in primes:
            flat = np.array(per_layer, dtype=np.int64).reshape(-1)
            offsets.append(np.concatenate([[0], np.cumsum(flat)[:-1]]))
        self.engram_offsets = np.stack(offsets)
        bound = max(1, (np.iinfo(np.int64).max
                        // int(text["engram_compressed_vocab_size"])) // 2)
        multipliers = []
        for layer_id in layer_ids:
            rng = np.random.default_rng(10007 * layer_id)
            values = rng.integers(low=0, high=bound, size=(ngram,),
                                  dtype=np.int64)
            multipliers.append(values * 2 + 1)
        self.engram_multipliers = np.stack(multipliers)
        map_path = os.environ.get("SPARK_DSV41_TOKEN_MAP")
        if not map_path:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            map_path = os.path.join(root, ENGRAM_MAP_REPO_PATH)
        if not os.path.exists(map_path):
            raise Dsv41ConfigError(
                f"engram token map missing at {map_path}; build it from the "
                "checkpoint tokenizer with the compressed-vocab pipeline")
        token_map = np.load(map_path)
        if token_map.shape[0] != self.vocab:
            raise Dsv41ConfigError(
                f"engram token map has {token_map.shape[0]} rows for vocab "
                f"{self.vocab}")
        compressed = int(token_map.max()) + 1
        pinned = int(text["engram_compressed_vocab_size"])
        if compressed > pinned:
            raise Dsv41ConfigError(
                f"engram token map spans {compressed} ids beyond the pinned "
                f"compressed vocab {pinned}")
        self.engram_token_map = token_map
        self.engram_pad = int(token_map[int(text["engram_pad_token_id"])])
        self.engram_ngram = ngram
        self.engram_heads = heads
        self.engram_head_dim = int(text["engram_head_dim"])
        self.engram = True

    def _engram_hashes(self, caches, token_id):
        ids = caches.setdefault("engram_ids", [])
        ids.append(int(self.engram_token_map[token_id]))
        position = len(ids) - 1
        count = len(self.engram_layers)
        ngram = self.engram_ngram
        toks = []
        blocked = False
        for shift in range(ngram):
            source = ids[position - shift] if position - shift >= 0 else None
            if position < shift or source is None or source == -1:
                blocked = True
            toks.append(self.engram_pad if blocked else source)
        toks = np.array(toks, dtype=np.int64)
        products = toks[None, :] * self.engram_multipliers
        rolling = products[:, 0].copy()
        out = np.zeros((count, (ngram - 1) * self.engram_heads),
                       dtype=np.int64)
        for i in range(1, ngram):
            rolling = np.bitwise_xor(rolling, products[:, i])
            out[:, (i - 1) * self.engram_heads:i * self.engram_heads] = \
                rolling[:, None] % self.engram_primes[:, i - 1]
        return out + self.engram_offsets

    def _bf16_matvec(self, slab, count, in_dim, x):
        n = count * in_dim
        if n > self._convert_u32.size:
            self._convert_u32 = np.empty(n, dtype=np.uint32)
        words = self._convert_u32[:n]
        words[:] = slab.reshape(-1)
        words <<= 16
        return words.view(np.float32).reshape(count, in_dim) @ x

    def _scale_grid(self, name, rows):
        scale_name = name[:-len(".weight")] + ".scale"
        scale = self.st.pread(scale_name).astype(np.int32)
        row_blocks, col_blocks = scale.shape
        if rows % row_blocks:
            raise Dsv41ConfigError(
                f"{name} rows {rows} do not pack into scale grid "
                f"{scale.shape}")
        return scale, row_blocks, col_blocks

    def _fp8_block_rows(self, slab, start, rb_rows, scales, col_blocks):
        rows, cols = slab.shape
        srow = np.arange(start, start + rows) // rb_rows
        block = _E4M3_F32[slab].reshape(rows, col_blocks,
                                        cols // col_blocks) \
            * scales[srow][:, :, None]
        return bf16_to_f32(f32_to_bf16_u16(block.reshape(rows, cols)))

    def _matvec_chunked(self, x, name, rows, in_dim, dtype):
        out = np.empty(rows, dtype=np.float32)
        if dtype == "BF16":
            for start in range(0, rows, LINEAR_BLOCK_ROWS):
                count = min(LINEAR_BLOCK_ROWS, rows - start)
                slab = self.st.raw_rows(name, start, count)
                out[start:start + count] = self._bf16_matvec(slab, count,
                                                             in_dim, x)
            return out
        scale, row_blocks, col_blocks = self._scale_grid(name, rows)
        rb_rows = rows // row_blocks
        scales = _exp2_rows(scale - 127)
        step = max(rb_rows, LINEAR_BLOCK_ROWS - LINEAR_BLOCK_ROWS % rb_rows)
        for start in range(0, rows, step):
            count = min(step, rows - start)
            slab = self.st.raw_rows(name, start, count)
            out[start:start + count] = self._fp8_block_rows(
                slab, start, rb_rows, scales, col_blocks) @ x
        return out

    def _linear(self, x, name):
        entry = self.st.entry(name)
        rows, in_dim = entry["shape"]
        if in_dim != x.shape[0]:
            raise Dsv41ConfigError(
                f"{name} shape {entry['shape']} disagrees with activation "
                f"width {x.shape[0]}")
        if entry["dtype"] not in ("BF16", "U8", "F8_E4M3"):
            raise Dsv41ConfigError(
                f"unsupported weight dtype for {name}: {entry['dtype']}")
        return bf16_round_f32(self._matvec_chunked(x, name, rows, in_dim,
                                                   entry["dtype"]))

    def _fp32_projector(self, name):
        raw = self.st.pread(name)
        if raw.dtype != np.uint16:
            raise Dsv41ConfigError(f"{name} must be BF16")
        return bf16_to_f32(raw)

    def _norm_weight(self, name):
        raw = self.st.pread(name)
        if raw.dtype != np.uint16:
            raise Dsv41ConfigError(f"{name} must be BF16")
        return bf16_to_f32(raw.reshape(-1))

    def _mxfp4_matvec(self, x, name):
        entry = self.st.entry(name)
        rows, packed = entry["shape"]
        cols = packed * 2
        if cols != x.shape[0]:
            raise Dsv41ConfigError(
                f"{name} width {cols} disagrees with activation width "
                f"{x.shape[0]}")
        scale_name = name[:-len(".weight")] + ".scale"
        sentry = self.st.entry(scale_name)
        if sentry["shape"] != [rows, cols // 32]:
            raise Dsv41ConfigError(
                f"{scale_name} {sentry['shape']} does not pack "
                f"{rows}x{cols}")
        out = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, LINEAR_BLOCK_ROWS):
            count = min(LINEAR_BLOCK_ROWS, rows - start)
            payload = self.st.raw_rows(name, start, count)
            scale = self.st.raw_rows(scale_name, start, count)
            exp_rows = scale.astype(np.int32).astype(np.float32) - 127
            nib = np.empty((count, cols), dtype=np.int32)
            nib[:, 0::2] = payload & 0xF
            nib[:, 1::2] = payload >> 4
            w = _E2M1_F32[nib].reshape(count, cols // 32, 32)
            block = (w * np.exp2(exp_rows)[:, :, None]).reshape(count, cols)
            out[start:start + count] = bf16_to_f32(
                f32_to_bf16_u16(block)) @ x
        return out

    def _expert_linear(self, x, name):
        entry = self.st.entry(name)
        rows, in_dim = entry["shape"]
        if entry["dtype"] == "BF16":
            if in_dim != x.shape[0]:
                raise Dsv41ConfigError(
                    f"{name} shape {entry['shape']} disagrees with "
                    f"activation width {x.shape[0]}")
            return bf16_round_f32(self._matvec_chunked(x, name, rows, in_dim,
                                                       "BF16"))
        if entry["dtype"] not in ("U8", "I8"):
            raise Dsv41ConfigError(
                f"unsupported expert weight dtype for {name}: "
                f"{entry['dtype']}")
        return bf16_round_f32(self._mxfp4_matvec(x, name))

    def _hc_mixes(self, layer, streams, kind):
        p = f"{PREFIX}{layer}.hc_{kind}_"
        flat = streams.reshape(-1).astype(np.float32)
        rsqrt = 1.0 / np.sqrt(flat.dot(flat) / flat.size + self.eps)
        fn = self.st.pread(p + "fn")
        if fn.dtype != np.float32:
            raise Dsv41ConfigError(f"{p}fn must be F32")
        mixes = fn.reshape(self.mix_hc, self.hc_dim) @ flat * rsqrt
        scale = self.st.pread(p + "scale").reshape(3)
        base = self.st.pread(p + "base").reshape(self.mix_hc)
        m = self.hc_mult
        pre = sigmoid(mixes[:m] * scale[0] + base[:m]) + self.hc_eps
        post = 2.0 * sigmoid(mixes[m:2 * m] * scale[1] + base[m:2 * m])
        comb = (mixes[2 * m:].reshape(m, m) * scale[2]
                + base[2 * m:].reshape(m, m))
        comb = _softmax_last(comb) + self.hc_eps
        comb = comb / (comb.sum(axis=0, keepdims=True) + self.hc_eps)
        for _ in range(self.sinkhorn - 1):
            comb = comb / (comb.sum(axis=1, keepdims=True) + self.hc_eps)
            comb = comb / (comb.sum(axis=0, keepdims=True) + self.hc_eps)
        return pre, post, comb

    def _hc_pre(self, streams, pre_mix):
        return bf16_round_f32((pre_mix[:, None] * streams).sum(axis=0))

    def _hc_post(self, x, residual, post, comb):
        return bf16_round_f32(post[:, None] * x[None, :] + comb @ residual)

    def _compressor(self, layer, x_norm, position, cache):
        ratio = self.ratios[layer]
        p = f"{PREFIX}{layer}.attn.compressor."
        if ratio == 1:
            kv = self._linear(x_norm, p + "wkv.weight")
            return bf16_round_f32(rmsnorm(
                kv, self._norm_weight(p + "norm.weight"), self.eps))
        x = x_norm.astype(np.float32)
        kv = self._fp32_projector(p + "wkv.weight") @ x
        score = self._fp32_projector(p + "wgate.weight") @ x
        slot = position % ratio
        cache["kv_state"][slot] = kv
        cache["score_state"][slot] = score
        if (position + 1) % ratio:
            return None
        pooled = (cache["kv_state"]
                  * _softmax_last(cache["score_state"])).sum(axis=0)
        pooled = bf16_round_f32(pooled)
        return bf16_round_f32(rmsnorm(
            pooled, self._norm_weight(p + "norm.weight"), self.eps))

    def _select_candidate_blocks(self, logits, width):
        block = self.candidate_block
        padded = width + (-width % block)
        scores = np.full(padded, -np.inf, dtype=np.float32)
        scores[:width] = logits
        scores = scores.reshape(-1, block).max(axis=1)
        scores[(width - 1) // block] = np.inf
        keep = min(self.candidate_blocks, scores.size)
        order = np.argsort(-scores, kind="stable")[:keep]
        mask = np.zeros(scores.size, dtype=bool)
        mask[order[scores[order] > -np.inf]] = True
        return np.repeat(mask, block)[:width]

    def _indexer(self, layer, x_norm, qr, latent, position, offset,
                 compress_len, shared):
        if layer not in self.index_sources:
            idxs = shared.get("topk_idxs")
            if idxs is None:
                raise Dsv41ConfigError(
                    f"layer {layer} has no published index selection")
            return idxs
        p = f"{PREFIX}{layer}.attn.indexer."
        ratio = self.ratios[layer]
        table = self._layer_freqs(layer)
        if layer in self.kv_sources and latent is not None:
            k = self._linear(latent, p + "wk.weight")
            k = bf16_round_f32(rmsnorm(
                k, self._norm_weight(p + "k_norm.weight"), self.eps))
            tail = bf16_round_f32(self._rotate(
                k[-self.rope_dim:].copy(), table, position + 1 - ratio))
            k = np.concatenate([k[:-self.rope_dim], tail])
            k = bf16_round_f32(
                _fp4_qdq_e8m0(k.reshape(1, self.index_dim))[0])
            shared.setdefault("index_k", {}).setdefault(layer, []).append(k)
            shared["index_owner"] = layer
        rows = shared.setdefault("index_k", {}).get(
            shared.get("index_owner"), [])
        keys = np.stack(rows) if rows else np.zeros(
            (0, self.index_dim), dtype=np.float32)
        q = self._linear(qr, p + "wq_b.weight").reshape(
            self.index_heads, self.index_dim)
        q[:, -self.rope_dim:] = bf16_round_f32(self._rotate(
            q[:, -self.rope_dim:].copy(), table, position))
        q = bf16_round_f32(_fp4_qdq_e8m0(q))
        index_score = np.zeros(keys.shape[0], dtype=np.float32)
        if keys.shape[0]:
            scores = np.maximum(q @ keys.T, 0.0)
            weights = bf16_round_f32(self._linear(
                x_norm, p + "weights_proj.weight")
                * np.float32(self.index_dim ** -0.5
                             * self.index_heads ** -0.5))
            index_score = (scores * weights[:, None]).sum(axis=0)
        if layer == self.candidate_source:
            shared["candidates"] = self._select_candidate_blocks(
                index_score, compress_len)
        elif self.candidate_source < layer:
            if "candidates" not in shared:
                raise Dsv41ConfigError(
                    f"layer {layer} expects candidate blocks before running")
            index_score = np.where(shared["candidates"], index_score,
                                   -np.inf)
        topk = min(self.index_topk, keys.shape[0])
        if topk >= index_score.size:
            selected = np.arange(index_score.size)
        else:
            selected = np.argpartition(-index_score, topk - 1)[:topk]
        selected = np.sort(selected)
        return (selected + offset).astype(np.int32)

    def _layer_freqs(self, layer):
        if self.ratios[layer] > 0:
            return self.compress_freqs
        return self.full_freqs

    def _attention(self, layer, x_norm, position, cache, shared):
        if _DEBUG:
            t0 = time.perf_counter()
        p = f"{PREFIX}{layer}.attn."
        ratio = self.ratios[layer]
        table = self._layer_freqs(layer)
        q_latent = self._linear(x_norm, p + "wq_a.weight")
        qr = bf16_round_f32(rmsnorm(
            q_latent, self._norm_weight(p + "q_norm.weight"), self.eps))
        q = self._linear(qr, p + "wq_b.weight").reshape(
            self.heads, self.head_dim)
        q[:, -self.rope_dim:] = bf16_round_f32(self._rotate(
            q[:, -self.rope_dim:].copy(), table, position))
        kv = self._linear(x_norm, p + "wkv.weight")
        kv = bf16_round_f32(rmsnorm(
            kv, self._norm_weight(p + "kv_norm.weight"), self.eps))
        kv[-self.rope_dim:] = bf16_round_f32(self._rotate(
            kv[-self.rope_dim:].copy(), table, position))
        win = cache["window"]
        win[position % self.window] = bf16_round_f32(
            _fp8_qdq_rows(kv.reshape(1, -1))[0])
        if position == 0:
            window_idx = np.array([0], dtype=np.int64)
            window_rows = win[:1].copy()
        else:
            oldest = position % self.window + 1
            window_idx = np.concatenate([np.arange(oldest, self.window),
                                         np.arange(oldest)]).astype(np.int64)
            keep = window_idx <= position
            window_idx = window_idx[keep]
            window_rows = win[window_idx]
        offset = window_rows.shape[0]
        parts_rows = [window_rows]
        parts_idx = [window_idx]
        if ratio > 0 and _NO_COMPRESS:
            ratio = 0
        if ratio > 0:
            compress_len = (position + 1) // ratio
            latent = None
            if layer in self.kv_sources:
                latent = self._compressor(layer, x_norm, position, cache)
                shared.setdefault("compress_rows", {})
                shared["compress_rows"].setdefault(layer, [])
                shared["compress_owner"] = layer
            if _DEBUG:
                print(f"  pre-indexer {time.perf_counter() - t0:.3f}s",
                      flush=True)
            idxs = self._indexer(layer, x_norm, qr, latent, position,
                                 offset, compress_len, shared)
            if _DEBUG:
                print(f"  indexer {time.perf_counter() - t0:.3f}s",
                      flush=True)
            shared["topk_idxs"] = idxs
            if latent is not None:
                tail = bf16_round_f32(self._rotate(
                    latent[-self.rope_dim:].copy(), table,
                    position + 1 - ratio))
                latent = np.concatenate([latent[:-self.rope_dim], tail])
                latent = bf16_round_f32(
                    _fp4_qdq_e4m3_16(latent.reshape(1, -1))[0])
                shared["compress_rows"][layer].append(latent)
            rows = shared["compress_rows"][shared["compress_owner"]]
            if len(rows) != compress_len:
                raise Dsv41ConfigError(
                    f"layer {layer} position {position}: compress cache has "
                    f"{len(rows)} rows, expected {compress_len}")
            if compress_len:
                parts_rows.append(np.stack(rows))
                parts_idx.append(np.asarray(idxs, dtype=np.int64) - offset)
        keys = np.concatenate(parts_rows, axis=0)
        index = np.concatenate(parts_idx)
        if _DEBUG:
            print(f"  post-cache {time.perf_counter() - t0:.3f}s",
                  flush=True)
        sink_raw = self.st.pread(p + "attn_sink")
        if sink_raw.dtype != np.float32:
            raise Dsv41ConfigError(f"{p}attn_sink must be F32")
        sink = sink_raw.reshape(self.heads)
        valid = index >= 0
        full = np.full((self.heads, index.size), -np.inf, dtype=np.float32)
        full[:, valid] = (q @ keys[index[valid]].T) \
            * np.float32(self.head_dim ** -0.5)
        rows_max = np.maximum(full.max(axis=1), -1e30)
        weights = np.exp(full - rows_max[:, None])
        denom = weights.sum(axis=1) + np.exp(sink - rows_max)
        o = (weights @ keys[index[valid]]) / denom[:, None]
        o = bf16_round_f32(o)
        o[:, -self.rope_dim:] = bf16_round_f32(self._rotate(
            o[:, -self.rope_dim:].copy(), table, position, inverse=True))
        o = bf16_round_f32(o)
        grouped = o.reshape(self.o_groups, -1)
        projected = np.empty((self.o_groups, self.o_lora), dtype=np.float32)
        for g in range(self.o_groups):
            projected[g] = self._wo_group_matvec(grouped[g],
                                                 p + "wo_a.weight", g)
        out = bf16_round_f32(projected.reshape(-1))
        return self._linear(out, p + "wo_b.weight")

    def _wo_group_matvec(self, x, name, group):
        entry = self.st.entry(name)
        rows, in_dim = entry["shape"]
        if entry["dtype"] not in ("U8", "F8_E4M3"):
            raise Dsv41ConfigError(f"{name} must be fp8-packed")
        if in_dim != x.shape[0]:
            raise Dsv41ConfigError(
                f"{name} shape {entry['shape']} disagrees with activation "
                f"width {x.shape[0]}")
        scale, row_blocks, col_blocks = self._scale_grid(name, rows)
        rb_rows = rows // row_blocks
        scales = _exp2_rows(scale - 127)
        step = max(rb_rows, LINEAR_BLOCK_ROWS - LINEAR_BLOCK_ROWS % rb_rows)
        out = np.empty(self.o_lora, dtype=np.float32)
        first = group * self.o_lora
        for start in range(first, first + self.o_lora, step):
            count = min(step, first + self.o_lora - start)
            slab = self.st.raw_rows(name, start, count)
            out[start - first:start - first + count] = self._fp8_block_rows(
                slab, start, rb_rows, scales, col_blocks) @ x
        return out

    def _shared_expert(self, layer, x):
        p = f"{PREFIX}{layer}.ffn.shared_experts."
        gate = np.minimum(self._linear(x, p + "w1.weight"), self.limit)
        up = np.clip(self._linear(x, p + "w3.weight"), -self.limit,
                     self.limit)
        activated = bf16_round_f32(_silu(gate) * up)
        return self._linear(activated, p + "w2.weight")

    def _moe(self, layer, x, sink):
        if _DEBUG:
            tm = time.perf_counter()
        p = f"{PREFIX}{layer}.ffn."
        scores = self._fp32_projector(p + "gate.weight") @ x
        routed = np.sqrt(_softplus(scores))
        bias = self.st.pread(p + "gate.bias").reshape(self.experts)
        if bias.dtype != np.float32:
            raise Dsv41ConfigError(f"{p}gate.bias must be F32")
        order = np.argsort(-(routed + bias), kind="stable")[:self.topk]
        selected = np.sort(order)
        picked = routed[selected]
        weights = picked / (picked.sum() + 1e-20) * self.route_scale
        y = np.zeros(self.hidden, dtype=np.float32)
        for i in range(self.topk):
            expert_prefix = p + f"experts.{int(selected[i])}."
            gate = np.minimum(bf16_round_f32(self._expert_linear(
                x, expert_prefix + "w1.weight")), self.limit)
            up = np.clip(bf16_round_f32(self._expert_linear(
                x, expert_prefix + "w3.weight")), -self.limit, self.limit)
            activated = bf16_round_f32(_silu(gate) * up)
            y += weights[i] * bf16_round_f32(self._expert_linear(
                activated, expert_prefix + "w2.weight"))
        y += self._shared_expert(layer, x)
        if _DEBUG:
            print(f"  moe {time.perf_counter() - tm:.3f}s", flush=True)
        sink.append(selected.astype(np.int32))
        sink.append(weights.astype(np.float32))
        return bf16_round_f32(y)

    def _engram_row(self, layer, row_id):
        weight = self.st.raw_rows(
            f"{PREFIX}{layer}.engram.embed.weight", row_id, 1)
        scale = self.st.raw_rows(
            f"{PREFIX}{layer}.engram.embed.scale", row_id, 1)
        dim = self.engram_head_dim
        deq = _E4M3_F32[weight.reshape(dim)] \
            * np.repeat(np.exp2(scale.astype(np.int32).astype(np.float32)
                                - 127).reshape(-1), 32)
        return bf16_round_f32(deq)

    def _engram_apply(self, layer, streams, hash_row):
        p = f"{PREFIX}{layer}.engram."
        embed = np.stack([self._engram_row(layer, int(c))
                          for c in hash_row]).reshape(-1)
        kv = self._linear(embed, p + "wkv.weight")
        split = self.hc_mult * self.hidden
        if kv.size != split + self.hidden:
            raise Dsv41ConfigError(
                f"{p}wkv produced {kv.size} outputs for hc_mult "
                f"{self.hc_mult}")
        key = kv[:split].reshape(self.hc_mult, self.hidden)
        value = kv[split:]
        weight = bf16_to_f32(self.st.pread(p + "q_weight")) \
            * bf16_to_f32(self.st.pread(p + "k_weight"))
        h = streams.astype(np.float32)
        rstd = (1.0 / np.sqrt((h * h).mean(-1) + self.eps)) \
            * (1.0 / np.sqrt((key * key).mean(-1) + self.eps))
        dot = (h * weight * key).sum(-1) * rstd * np.float32(
            self.hidden ** -0.5)
        gate = sigmoid(np.copysign(
            np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot))
        return bf16_round_f32(h + gate[:, None] * value[None, :])

    def _forward_layer(self, layer, streams, pre_mix, position, cache,
                       shared):
        p = f"{PREFIX}{layer}."
        attn_pre, attn_post, attn_comb = self._hc_mixes(layer, streams,
                                                        "attn")
        x = self._hc_pre(streams, pre_mix)
        x = bf16_round_f32(rmsnorm(
            x, self._norm_weight(p + "attn_norm.weight"), self.eps))
        x = self._attention(layer, x, position, cache, shared)
        streams = self._hc_post(x, streams, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = self._hc_mixes(layer, streams, "ffn")
        x = self._hc_pre(streams, attn_pre)
        x = bf16_round_f32(rmsnorm(
            x, self._norm_weight(p + "ffn_norm.weight"), self.eps))
        sink = []
        x = self._moe(layer, x, sink)
        streams = self._hc_post(x, streams, ffn_post, ffn_comb)
        return streams, ffn_pre, sink

    def _init_position(self, position, caches):
        if _DEBUG:
            print(f"init position {position}", flush=True)
        if position > 0:
            return
        caches.clear()
        for layer in range(self.layers):
            cache = {"window": np.zeros((self.window, self.head_dim),
                                        dtype=np.float32)}
            ratio = self.ratios[layer]
            if layer in self.kv_sources:
                cache["kv_state"] = np.zeros((ratio, self.head_dim),
                                             dtype=np.float32)
                cache["score_state"] = np.zeros((ratio, self.head_dim),
                                                dtype=np.float32)
            caches[layer] = cache
        caches["shared"] = {}

    def decode_step(self, token_id, position, states, caches, capture,
                    capture_streams=None):
        if token_id < 0 or token_id >= self.vocab:
            raise ValueError(f"token {token_id} outside vocabulary "
                             f"{self.vocab}")
        self._init_position(position, caches)
        row = self.st.raw_rows("embed.weight", token_id, 1)
        if row.dtype != np.uint16:
            raise ValueError("reference embedding must be BF16")
        streams = np.repeat(bf16_to_f32(row[0])[None, :], self.hc_mult,
                            axis=0)
        pre_mix = np.zeros(self.hc_mult, dtype=np.float32)
        pre_mix[0] = 1.0
        hashes = self._engram_hashes(caches, token_id) if self.engram \
            else None
        shared = caches["shared"]
        for layer in range(self.layers):
            mark = time.perf_counter() if _DEBUG else None
            if self.engram and layer in self.engram_layers and not _NO_ENGRAM:
                hash_index = self.engram_layers.index(layer)
                streams = self._engram_apply(layer, streams,
                                             hashes[hash_index])
            streams, pre_mix, sink = self._forward_layer(
                layer, streams, pre_mix, position, caches[layer], shared)
            if _DEBUG:
                print(f"pos {position} layer {layer} "
                      f"{time.perf_counter() - mark:.3f}s", flush=True)
            if sink:
                capture[(position, layer)] = sink
            if capture_streams is not None:
                capture_streams(layer, streams)
            if not np.isfinite(streams).all():
                raise ValueError(
                    f"nonfinite reference state at layer {layer}")
        self._head_input = self._hc_pre(streams, pre_mix)
        return streams

    def logits(self, streams, chunk=2048):
        if self._head_input is None:
            raise ValueError("logits called before decode_step")
        norm = bf16_round_f32(rmsnorm(
            self._head_input, self._norm_weight("norm.weight"), self.eps))
        entry = self.st.entry("head.weight")
        if entry["dtype"] != "BF16":
            raise ValueError("reference lm head must be BF16")
        rows_total = entry["shape"][0]
        best = -np.inf
        best_token = -1
        for start in range(0, rows_total, chunk):
            count = min(chunk, rows_total - start)
            head = self.st.raw_rows("head.weight", start, count)
            scores = bf16_to_f32(head) @ norm
            i = int(np.argmax(scores))
            if float(scores[i]) > best:
                best = float(scores[i])
                best_token = start + i
        return best_token, best


def f32_to_bf16_u16(x):
    from t1_reference_common import f32_to_bf16_u16
    return f32_to_bf16_u16(x)


ENGINE_CLASS = Dsv41FlashEngine
