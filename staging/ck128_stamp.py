import mmap
import os
import struct
import sys

MASK = (1 << 64) - 1
C1 = 0x87C37B91114253D5
C2 = 0x4CF5AD432745937F


def rotl64(x, r):
    return ((x << r) | (x >> (64 - r))) & MASK


def fmix(k):
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & MASK
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & MASK
    k ^= k >> 33
    return k


class Murmur3_x64_128:
    def __init__(self):
        self.h1 = 0
        self.h2 = 0
        self.total = 0
        self.tail = b""

    def _block(self, k1, k2):
        k1 = (k1 * C1) & MASK
        k1 = rotl64(k1, 31)
        k1 = (k1 * C2) & MASK
        self.h1 ^= k1
        self.h1 = rotl64(self.h1, 27)
        self.h1 = (self.h1 + self.h2) & MASK
        self.h1 = (self.h1 * 5 + 0x52DCE729) & MASK
        k2 = (k2 * C2) & MASK
        k2 = rotl64(k2, 33)
        k2 = (k2 * C1) & MASK
        self.h2 ^= k2
        self.h2 = rotl64(self.h2, 31)
        self.h2 = (self.h2 + self.h1) & MASK
        self.h2 = (self.h2 * 5 + 0x38495AB5) & MASK

    def update(self, data):
        self.total += len(data)
        if self.tail:
            want = 16 - len(self.tail)
            take = data[:want]
            self.tail += take
            data = data[len(take):]
            if len(self.tail) == 16:
                k1, k2 = struct.unpack("<QQ", self.tail)
                self._block(k1, k2)
                self.tail = b""
        nblocks = len(data) // 16
        for i in range(nblocks):
            k1, k2 = struct.unpack_from("<QQ", data, i * 16)
            self._block(k1, k2)
        rest = data[nblocks * 16:]
        self.tail += rest
        if len(self.tail) >= 16:
            k1, k2 = struct.unpack("<QQ", self.tail[:16])
            self._block(k1, k2)
            self.tail = self.tail[16:]

    def digest(self):
        h1, h2 = self.h1, self.h2
        tb = len(self.tail)
        k1 = 0
        k2 = 0
        if tb > 8:
            k2 = int.from_bytes(self.tail[8:], "little")
            k2 = (k2 * C2) & MASK
            k2 = rotl64(k2, 33)
            k2 = (k2 * C1) & MASK
            h2 ^= k2
        if tb > 0:
            k1 = int.from_bytes(self.tail[:8], "little")
            k1 = (k1 * C1) & MASK
            k1 = rotl64(k1, 31)
            k1 = (k1 * C2) & MASK
            h1 ^= k1
        h1 ^= self.total
        h2 ^= self.total
        h1 = (h1 + h2) & MASK
        h2 = (h2 + h1) & MASK
        h1 = fmix(h1)
        h2 = fmix(h2)
        h1 = (h1 + h2) & MASK
        h2 = (h2 + h1) & MASK
        return struct.pack("<QQ", h1, h2).hex()


def main():
    if len(sys.argv) != 2:
        print("usage: ck128_stamp.py <pack-path>", file=sys.stderr)
        return 2
    pack = sys.argv[1]
    hasher = Murmur3_x64_128()
    with open(pack, "rb") as f:
        with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
            step = 64 * 1024 * 1024
            pos = 0
            size = m.size()
            while pos < size:
                hasher.update(m[pos:pos + step])
                pos += step
    digest = hasher.digest()
    out = pack + ".ck128"
    with open(out, "w") as fo:
        fo.write(digest + "\n")
    print(f"ck128 {digest} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
