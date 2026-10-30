import re

MASK = (1 << 64) - 1
ROUNDS = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
OFFSETS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]
P = 2 ** 256 - 2 ** 32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
SIGNATURE = re.compile(r"^0x[0-9a-fA-F]{130}$")


def _rotate(value, shift):
    return ((value << shift) | (value >> (64 - shift))) & MASK if shift else value


def _permute(lanes):
    for constant in ROUNDS:
        column = [lanes[x] ^ lanes[x + 5] ^ lanes[x + 10] ^ lanes[x + 15] ^ lanes[x + 20] for x in range(5)]
        mix = [column[(x - 1) % 5] ^ _rotate(column[(x + 1) % 5], 1) for x in range(5)]
        lanes = [lanes[i] ^ mix[i % 5] for i in range(25)]
        moved = [0] * 25
        for x in range(5):
            for y in range(5):
                moved[y + 5 * ((2 * x + 3 * y) % 5)] = _rotate(lanes[x + 5 * y], OFFSETS[x][y])
        lanes = [moved[i] ^ (~moved[(i % 5 + 1) % 5 + 5 * (i // 5)] & MASK & moved[(i % 5 + 2) % 5 + 5 * (i // 5)]) for i in range(25)]
        lanes[0] ^= constant
    return lanes


def _sponge(data, pad, rate=136, size=32):
    block = bytearray(data)
    block.append(pad)
    while len(block) % rate:
        block.append(0)
    block[-1] |= 0x80
    lanes = [0] * 25
    for start in range(0, len(block), rate):
        for i in range(rate // 8):
            lanes[i] ^= int.from_bytes(block[start + 8 * i:start + 8 * i + 8], "little")
        lanes = _permute(lanes)
    return b"".join(lane.to_bytes(8, "little") for lane in lanes)[:size]


def keccak256(data):
    return _sponge(data, 0x01)


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0]:
        if (a[1] + b[1]) % P == 0:
            return None
        slope = 3 * a[0] * a[0] * pow(2 * a[1], -1, P) % P
    else:
        slope = (b[1] - a[1]) * pow(b[0] - a[0], -1, P) % P
    x = (slope * slope - a[0] - b[0]) % P
    return x, (slope * (a[0] - x) - a[1]) % P


def multiply(point, scalar):
    result = None
    while scalar:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def address_of(point):
    return "0x" + keccak256(point[0].to_bytes(32, "big") + point[1].to_bytes(32, "big"))[-20:].hex()


def personal_digest(text):
    body = text.encode("utf-8")
    return keccak256(b"\x19Ethereum Signed Message:\n" + str(len(body)).encode() + body)


def recover(digest, signature):
    if not SIGNATURE.match(signature or ""):
        return None
    raw = bytes.fromhex(signature[2:])
    r, s, v = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:64], "big"), raw[64]
    parity = v - 27 if v >= 27 else v
    if parity not in (0, 1) or not 0 < r < N or not 0 < s <= N // 2:
        return None
    square = (pow(r, 3, P) + 7) % P
    y = pow(square, (P + 1) // 4, P)
    if y * y % P != square:
        return None
    if y & 1 != parity:
        y = P - y
    z = int.from_bytes(digest, "big") % N
    inverse = pow(r, -1, N)
    point = _add(multiply((r, y), s * inverse % N), multiply(G, (N - z) * inverse % N))
    return address_of(point) if point else None


def personal_recover(text, signature):
    return recover(personal_digest(text), signature)
