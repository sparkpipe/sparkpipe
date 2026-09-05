import ctypes

lib = ctypes.CDLL("/tmp/probe_dso.so")
lib.SparkHiddenTransportGetInterface.restype = ctypes.c_void_p
ptr = lib.SparkHiddenTransportGetInterface()
ptrs = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint64))

names = [
    "initialize", "destroy", "post_receive", "send", "poll",
    "post_receive_batch", "send_batch", "get_poll_descriptors",
    "register_persistent_receive", "persistent_remote_credit_ready",
    "reserve_persistent_send", "cancel_persistent_send",
    "activate_persistent_receive", "cancel_persistent_receive",
    "send_persistent", "release_persistent_receive",
    "send_fixed", "set_fixed_remote", "set_fixed_local",
]
slots = {}
for i, n in enumerate(names):
    slots[n] = ptrs[2 + i]

dlsym = ctypes.CDLL(None).dlsym
candidates = [
    "SparkHiddenTransportPersistentRingSend",
    "SparkHiddenTransportPersistentRingPostReceive",
    "SparkHiddenTransportPersistentRingPoll",
    "SparkHiddenTransportSendFixed",
]
resolved = {}
for c in candidates:
    try:
        resolved[c] = dlsym(lib._handle, c.encode())
    except Exception:
        resolved[c] = 0

print("send_fixed slot =", hex(slots["send_fixed"]))
for n in names:
    for c, a in resolved.items():
        if a and slots[n] == a:
            print("MATCH: slot", n, "==", c)
for c, a in resolved.items():
    print("dlsym", c, hex(a))
