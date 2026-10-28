from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = """        value = bf16(rms(value_raw, None, self.epsilon))
        return self.linear(self.attend(query, key, value, positions, 0 if full else self.window, 1.0), p + "o_proj.weight")"""
new = """        value = bf16(rms(value_raw, None, self.epsilon))
        attended = self.attend(query, key, value, positions, 0 if full else self.window, 1.0)
        if layer == DUMP_LAYER:
            tag = "abcd"[[FLOOR[x] for x in "abcd"].index((REVERSED, CHUNK, QK_SPLIT))]
            for name, array in (("q", query), ("k", key), ("v", value), (tag + ".out", attended)):
                np.ascontiguousarray(array, dtype=np.float32).tofile(f"{DUMP_PREFIX}.{name}")
        return self.linear(attended, p + "o_proj.weight")"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("REVERSED = False\nCHUNK = 1024\n", "REVERSED = False\nCHUNK = 1024\nDUMP_LAYER = int(__import__('os').environ.get('DUMP_LAYER', '-1'))\nDUMP_PREFIX = __import__('os').environ.get('DUMP_PREFIX', '')\n", 1)
p.write_text(s)
print("ok")
