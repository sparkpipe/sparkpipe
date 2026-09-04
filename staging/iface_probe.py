import ctypes
import os

lib = ctypes.CDLL(os.path.expanduser("~/probe_dso.so"))
lib.SparkHiddenTransportGetInterface.restype = ctypes.c_void_p
ptr = lib.SparkHiddenTransportGetInterface()
words = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
print("abi", words[0], "desc", words[1], "caps", hex(words[2]))
ptrs = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint64))
names = [
    "init", "destroy", "post_recv", "send", "poll", "post_batch",
    "send_batch", "get_poll", "reg_pers", "ready", "reserve",
    "cancel_send", "activate", "cancel_recv", "send_pers", "release",
    "send_fixed", "set_fixed_remote", "set_fixed_local",
]
base = 2
for i, n in enumerate(names):
    print(n, hex(ptrs[base + i]))
