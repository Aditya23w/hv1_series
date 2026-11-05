#!/usr/bin/env python3
"""Vertex-cluster decimation for STLs above MuJoCo's 200k-face limit."""
import struct, sys, numpy as np

LIMIT = 190000  # stay under MuJoCo's 200000 hard cap

def read_stl(path):
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    raw = np.frombuffer(d[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(raw[:, 12:48].tobytes(), dtype="<f4").reshape(n, 3, 3).astype(np.float64)

def write_stl(path, tri):
    n = len(tri)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    buf = bytearray(b"decimated for mujoco".ljust(80, b"\0") + struct.pack("<I", n))
    rec = np.zeros((n, 50), dtype=np.uint8)
    rec[:, 0:12]  = np.frombuffer(nrm.astype("<f4").tobytes(), dtype=np.uint8).reshape(n, 12)
    rec[:, 12:48] = np.frombuffer(tri.astype("<f4").tobytes(), dtype=np.uint8).reshape(n, 36)
    buf += rec.tobytes()
    open(path, "wb").write(bytes(buf))

def cluster(tri, cell):
    v = tri.reshape(-1, 3)
    key = np.floor(v / cell).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    # representative = centroid of each cell (smoother than snapping to grid corner)
    rep = np.zeros((len(cnt), 3))
    np.add.at(rep, inv, v)
    rep /= cnt[:, None]
    idx = inv.reshape(-1, 3)
    keep = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx = idx[keep]
    # drop duplicate faces
    _, u = np.unique(np.sort(idx, axis=1), axis=0, return_index=True)
    idx = idx[np.sort(u)]
    return rep[idx]

src, dst = sys.argv[1], sys.argv[2]
tri = read_stl(src)
ext = np.ptp(tri.reshape(-1, 3), axis=0)  # np.ptp, not ndarray.ptp (removed in numpy 2)
cell = ext.max() / 2000.0
for _ in range(30):
    out = cluster(tri, cell)
    if len(out) <= LIMIT:
        break
    cell *= 1.25
write_stl(dst, out)
print(f"{src}: {len(tri)} -> {len(out)} faces  (cell {cell*1000:.2f} mm)")
print("  bbox src", np.round(tri.reshape(-1,3).min(0),4), np.round(tri.reshape(-1,3).max(0),4))
print("  bbox dst", np.round(out.reshape(-1,3).min(0),4), np.round(out.reshape(-1,3).max(0),4))
