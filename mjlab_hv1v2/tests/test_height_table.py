"""v11h height table (step 1): the table and its GPU lookup.

  python -m pytest mjlab_hv1v2/tests/test_height_table.py
"""

from __future__ import annotations

import torch

from mjlab_hv1v2.hv1v2_constants import (
  HOME_KEYFRAME,
  V11H_HEIGHT_RANGE,
  V11H_HEIGHT_TABLE,
  V11H_HOME_Z,
)
from mjlab_hv1v2.mdp_v11h import HeightPoseTable

# Action-term order (right leg first), not the table's.
ACTION_ORDER = tuple(
  f"{s}_{j}_joint" for s in ("right", "left")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))


def _tab():
  return HeightPoseTable(ACTION_ORDER, "cpu")


def test_shape_and_order():
  t = V11H_HEIGHT_TABLE
  n = len(t["pelvis_z"])
  assert len(t["q"]) == n and len(t["tau_static"]) == n
  assert all(len(r) == 12 for r in t["q"] + t["tau_static"])
  z = t["pelvis_z"]
  assert all(a < b for a, b in zip(z, z[1:]))
  # command range lies inside the table (no extrapolation)
  assert z[0] <= V11H_HEIGHT_RANGE[0] and V11H_HEIGHT_RANGE[1] >= z[-1]


def test_home_row_is_home_keyframe():
  tab = _tab()
  q, tau = tab(torch.tensor([0.980, V11H_HOME_Z, 1.2]))   # above the top clamps to HOME
  home = torch.tensor([HOME_KEYFRAME.joint_pos[n] for n in ACTION_ORDER])
  assert torch.allclose(q, home.expand(3, -1), atol=1e-6)
  assert torch.allclose(tau, tab.tau_home.expand(3, -1))


def test_rows_hit_exactly_and_interpolate():
  tab = _tab()
  z = tab.z
  q, _ = tab(z)
  assert torch.allclose(q, tab.q, atol=1e-6)
  mid = 0.5 * (z[:-1] + z[1:])
  qm, tm = tab(mid)
  assert torch.allclose(qm, 0.5 * (tab.q[:-1] + tab.q[1:]), atol=1e-5)
  assert torch.allclose(tm, 0.5 * (tab.tau[:-1] + tab.tau[1:]), atol=1e-4)


def test_monotone_crouch():
  """Lower pelvis = more knee flexion and more knee torque, both legs."""
  tab = _tab()
  h = torch.linspace(V11H_HEIGHT_RANGE[0], V11H_HEIGHT_RANGE[1], 50)
  q, tau = tab(h)
  for side in ("right", "left"):
    k = ACTION_ORDER.index(f"{side}_knee_joint")
    assert bool(torch.all(q[1:, k] <= q[:-1, k] + 1e-6))
    assert bool(torch.all(tau[1:, k].abs() <= tau[:-1, k].abs() + 1e-4))
  # bottom of the range: knee ~1.0 rad, inside the continuous rating
  qb, tb = tab(torch.tensor([V11H_HEIGHT_RANGE[0]]))
  k = ACTION_ORDER.index("right_knee_joint")
  assert 0.9 < float(qb[0, k]) < 1.0 and abs(float(tb[0, k])) < 85.0


def test_left_right_mirror():
  """Mirror pairs: pitch/knee equal, roll/yaw opposite (pose and torque)."""
  tab = _tab()
  q, tau = tab(torch.linspace(0.89, 0.98, 10))
  for j, sign in (("hip_pitch", 1), ("knee", 1), ("ankle_pitch", 1),
                  ("hip_roll", -1), ("hip_yaw", -1), ("ankle_roll", -1)):
    r = ACTION_ORDER.index(f"right_{j}_joint"); l = ACTION_ORDER.index(f"left_{j}_joint")
    assert torch.allclose(q[:, r], sign * q[:, l], atol=2e-3), j
    assert torch.allclose(tau[:, r], sign * tau[:, l], atol=0.2), j
