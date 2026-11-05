"""v11i pose table (step 1): V11I_POSE_TABLE and its GPU lookup.

  python mjlab_hv1v2/tests/test_pose_table2d.py      (pytest is not installed)
"""

from __future__ import annotations

import torch

from mjlab_hv1v2.hv1v2_constants import (
  HOME_KEYFRAME,
  V11H_HEIGHT_RANGE,
  V11H_HEIGHT_TABLE,
  V11I_HEIGHT_RANGE,
  V11I_POSE_TABLE,
  V11I_WAIST_PITCH_RANGE,
)
from mjlab_hv1v2.mdp_v11h import HeightPoseTable
from mjlab_hv1v2.mdp_v11i import PoseTable2D

ACTION_ORDER = tuple(
  f"{s}_{j}_joint" for s in ("right", "left")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))


def _tab():
  return PoseTable2D(ACTION_ORDER, "cpu")


def test_shape_and_axes():
  t = V11I_POSE_TABLE
  nz, nw = len(t["pelvis_z"]), len(t["waist"])
  for key in ("q", "tau_static"):
    assert len(t[key]) == nz and all(len(r) == nw for r in t[key])
    assert all(len(c) == 12 for r in t[key] for c in r)
  z, w = t["pelvis_z"], t["waist"]
  assert all(a < b for a, b in zip(z, z[1:])) and all(a < b for a, b in zip(w, w[1:]))
  assert z[0] <= V11I_HEIGHT_RANGE[0] and z[-1] == V11I_HEIGHT_RANGE[1]
  assert w[0] <= V11I_WAIST_PITCH_RANGE[0] and w[-1] >= V11I_WAIST_PITCH_RANGE[1]


def test_home_cell_is_home_keyframe():
  tab = _tab()
  q, _ = tab(torch.tensor([V11I_HEIGHT_RANGE[1], 1.2]), torch.tensor([0.0, 0.0]))
  home = torch.tensor([HOME_KEYFRAME.joint_pos[n] for n in ACTION_ORDER])
  assert torch.allclose(q, home.expand(2, -1), atol=1e-6)


def test_waist_zero_equals_v11h():
  """At w = 0 the 2-D table reproduces v11h's 1-D table over v11h's range."""
  t2, t1 = _tab(), HeightPoseTable(ACTION_ORDER, "cpu")
  h = torch.linspace(V11H_HEIGHT_RANGE[0], V11H_HEIGHT_RANGE[1], 200)
  h = torch.cat([h, torch.tensor(V11H_HEIGHT_TABLE["pelvis_z"], dtype=torch.float32)])
  h = h[h >= V11H_HEIGHT_TABLE["pelvis_z"][0]]
  q2, tau2 = t2(h, torch.zeros_like(h))
  q1, tau1 = t1(h)
  assert (q2 - q1).abs().max() < 1e-6, (q2 - q1).abs().max()
  assert (tau2 - tau1).abs().max() < 1e-4, (tau2 - tau1).abs().max()


def test_cells_hit_exactly_and_bilinear_midpoints():
  tab = _tab()
  Z, W = torch.meshgrid(tab.z, tab.w, indexing="ij")
  q, tau = tab(Z.reshape(-1), W.reshape(-1))
  assert torch.allclose(q, tab.q.reshape(-1, 12), atol=1e-6)
  assert torch.allclose(tau, tab.tau.reshape(-1, 12), atol=1e-4)
  zm, wm = 0.5 * (tab.z[:-1] + tab.z[1:]), 0.5 * (tab.w[:-1] + tab.w[1:])
  Zm, Wm = torch.meshgrid(zm, wm, indexing="ij")
  qm, _ = tab(Zm.reshape(-1), Wm.reshape(-1))
  ref = 0.25 * (tab.q[:-1, :-1] + tab.q[1:, :-1] + tab.q[:-1, 1:] + tab.q[1:, 1:])
  assert torch.allclose(qm, ref.reshape(-1, 12), atol=1e-5)


def test_clamps_outside_the_grid():
  tab = _tab()
  q_lo, _ = tab(torch.tensor([0.50]), torch.tensor([-1.0]))
  assert torch.allclose(q_lo[0], tab.q[0, 0], atol=1e-6)
  q_hi, _ = tab(torch.tensor([1.50]), torch.tensor([1.0]))
  assert torch.allclose(q_hi[0], tab.q[-1, -1], atol=1e-6)


def test_monotone_and_lean_relieves_knee():
  """Lower = more knee flexion and torque at every waist; more forward lean =
  less knee torque at every height. Worst corner inside the plan's limits."""
  tab = _tab()
  k = ACTION_ORDER.index("left_knee_joint")
  for j in range(len(tab.w)):
    assert bool(torch.all(tab.q[1:, j, k] <= tab.q[:-1, j, k] + 1e-3))
    assert bool(torch.all(tab.tau[1:, j, k].abs() <= tab.tau[:-1, j, k].abs() + 1e-3))
  # lean: compare solved (non-v11h) columns only, the v11h column is a
  # different solver objective (weak CoM) and may sit off-trend by ~1 N.m
  for i in range(len(tab.z)):
    t = tab.tau[i, :, k].abs()
    assert bool(torch.all(t[2:] <= t[1:-1] + 1e-3)) and t[0] >= t[2]
  worst = float(tab.tau[0, 0, k].abs())
  assert worst <= 78.0, worst


def test_left_right_mirror():
  tab = _tab()
  Z, W = torch.meshgrid(tab.z, tab.w, indexing="ij")
  q, tau = tab(Z.reshape(-1), W.reshape(-1))
  for j, sign in (("hip_pitch", 1), ("knee", 1), ("ankle_pitch", 1),
                  ("hip_roll", -1), ("hip_yaw", -1), ("ankle_roll", -1)):
    r, l = ACTION_ORDER.index(f"right_{j}_joint"), ACTION_ORDER.index(f"left_{j}_joint")
    assert torch.allclose(q[:, r], sign * q[:, l], atol=3e-3), j
    assert torch.allclose(tau[:, r], sign * tau[:, l], atol=0.3), j


if __name__ == "__main__":
  import sys
  fails = 0
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      try:
        fn(); print(f"PASS {name}")
      except AssertionError as e:
        fails += 1; print(f"FAIL {name}: {e}")
  sys.exit(1 if fails else 0)
