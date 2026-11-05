"""v11i (2026-11-05): v11h + a deeper height range (0.86 m) + operator waist
pitch (-0.09 .. +0.44 rad, <= 0.5 rad/s), fine-tuned from v11h model_26999.

Plan: ~/.claude/plans/now-i-have-a-robust-lecun.md

  step 1  PoseTable2D: (pelvis height, waist pitch) -> reference leg pose and
          static leg torque, bilinear on the GPU, over V11I_POSE_TABLE.
"""

from __future__ import annotations

from typing import Sequence

import torch

from mjlab_hv1v2.hv1v2_constants import V11I_POSE_TABLE


class PoseTable2D:
  """(pelvis height h, waist pitch w) -> (reference leg pose, static leg torque).

    tab = PoseTable2D(term.target_names, device)
    q_ref, tau = tab(h, w)       # h, w: (N,) -> (N, J), (N, J)

  Both inputs are clamped to the table's axes, then interpolated bilinearly
  between the four bracketing cells (torch.searchsorted per axis; neither axis
  is evenly spaced). `joint_names` picks and orders the columns, as in
  mdp_v11h.HeightPoseTable. At w = 0 this equals HeightPoseTable exactly for
  h >= 0.88885 m (the waist-0 column IS V11H_HEIGHT_TABLE), up to float rounding
  of the interpolation weights -- see tests/test_pose_table2d.py.
  """

  def __init__(self, joint_names: Sequence[str], device: str | torch.device,
               table: dict = V11I_POSE_TABLE):
    names = list(table["joint_names"])
    missing = [n for n in joint_names if n not in names]
    assert not missing, f"not in the v11i pose table: {missing}"
    cols = [names.index(n) for n in joint_names]
    z = torch.tensor(table["pelvis_z"], dtype=torch.float32, device=device)
    w = torch.tensor(table["waist"], dtype=torch.float32, device=device)
    assert bool(torch.all(z[1:] > z[:-1])), "pelvis_z must be strictly ascending"
    assert bool(torch.all(w[1:] > w[:-1])), "waist must be strictly ascending"
    self.joint_names = tuple(joint_names)
    self.z, self.w = z, w
    # (nz, nw, J)
    self.q = torch.tensor(table["q"], dtype=torch.float32, device=device)[..., cols]
    self.tau = torch.tensor(table["tau_static"], dtype=torch.float32, device=device)[..., cols]
    self.z_min, self.z_max = float(z[0]), float(z[-1])
    self.w_min, self.w_max = float(w[0]), float(w[-1])
    self.w0 = int(torch.nonzero(w == 0.0)[0])
    self.q_home = self.q[-1, self.w0]
    self.tau_home = self.tau[-1, self.w0]

  @staticmethod
  def _bracket(axis: torch.Tensor, x: torch.Tensor, lo: float, hi: float):
    x = torch.clamp(x, lo, hi)
    i1 = torch.searchsorted(axis, x.contiguous(), right=False).clamp(1, len(axis) - 1)
    i0 = i1 - 1
    t = ((x - axis[i0]) / (axis[i1] - axis[i0])).clamp(0.0, 1.0)
    return i0, i1, t

  def __call__(self, h: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zi0, zi1, tz = self._bracket(self.z, h, self.z_min, self.z_max)
    wi0, wi1, tw = self._bracket(self.w, w, self.w_min, self.w_max)
    tz, tw = tz.unsqueeze(-1), tw.unsqueeze(-1)

    def lerp2(A):
      a = A[zi0, wi0] * (1 - tw) + A[zi0, wi1] * tw
      b = A[zi1, wi0] * (1 - tw) + A[zi1, wi1] * tw
      return a * (1 - tz) + b * tz

    return lerp2(self.q), lerp2(self.tau)
