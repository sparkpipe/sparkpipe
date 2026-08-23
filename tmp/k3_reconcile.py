import json, struct, os

ALIGN = 128

def reconcile(path):
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        magic, ver, mlen = struct.unpack("<IIQ", f.read(16))
        assert magic == 0x4B33504B, hex(magic)
        man = json.loads(f.read(mlen))
    base = 16 + mlen
    base += (-base) % ALIGN
    inv = sum(t["bytes"] for t in man["tensors"].values())
    pad_total = len(man["tensors"]) * (ALIGN - 1)
    cfg = man["config"]
    tk = man["format"]["mxfp4_interleave"]["tile_k"]
    name = os.path.basename(path)
    print(name)
    print("  fmt v%d tile_k=%d tensors=%d tp_degree=%s layers=%s first_layer=%s" % (
        ver, tk, len(man["tensors"]), cfg.get("tp_degree", "-"),
        cfg.get("layers", "-"), cfg.get("first_layer", "-")))
    print("  file_size      = %d" % sz)
    print("  inventory      = %d (sum of manifest tensor bytes)" % inv)
    print("  payload_receipt= %d (file minus header+manifest)" % (sz - base))
    print("  ratio receipt/inv = %.6f  (inter-tensor pad slack <= %d)" % (
        (sz - base) / inv, pad_total))
    print("  manifest_len   = %d" % mlen)

for p in [
    "/home/spark0/sparkdata/k3.mxfp4.tp16/tp16_slice_l0-3.pack",
    "/home/spark0/sparkdata/k3.mxfp4.tp16/tp16_slice_l0-3.rank00.pack",
    "/home/spark0/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage0.rank00.pack",
]:
    if os.path.exists(p):
        reconcile(p)
    else:
        print("MISSING", p)
