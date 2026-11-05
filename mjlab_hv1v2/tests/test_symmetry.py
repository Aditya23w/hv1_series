"""v11 mirror maps (src/mjlab_hv1v2/symmetry.py). Run before training:

    python -m pytest mjlab_hv1v2/tests/test_symmetry.py -q

1. FK reflection: mirror a random configuration with the derived map; every
   body's CoM must be the y-reflection of its partner's. 1 mm on legs, pelvis,
   torso, shoulders/elbows; 10 mm on the wrist links and the head (the
   model's own asymmetry there, measured at q = 0: <= 9 mm). A sign error
   shows up as centimetres. An involution test alone cannot catch it.
2. Involution M(M(x)) = x on the full v11 actor layout and on actions.
3. The mirrored HOME_KEYFRAME equals HOME_KEYFRAME (1e-3 rad).
4. v11h (838 = v11 + height_cmd last): the first 837 columns mirror exactly as
   v11's, the height column maps to itself unchanged, involution holds.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import torch

from mjlab_hv1v2.hv1v2_constants import HOME_KEYFRAME, HV1V2_XML, V11_UB_OBS_JOINTS
from mjlab_hv1v2.symmetry import ActorMirror, joint_mirror, partner

S = np.diag([1.0, -1.0, 1.0])
LOOSE = re.compile(r"(wrist_(pitch|yaw)_link|wrist_roll_link|head_.*_link)$")


def _model():
  m = mujoco.MjModel.from_xml_path(str(HV1V2_XML))
  return m, mujoco.MjData(m)


def _hinges(m):
  return [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def _set(m, d, names, q):
  d.qpos[:] = m.qpos0
  for n, v in zip(names, q):
    d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
  mujoco.mj_kinematics(m, d)
  return d.xipos.copy()


def test_fk_reflection():
  m, d = _model()
  names = _hinges(m)
  perm, sign = joint_mirror(names)
  rng = np.random.default_rng(0)
  bodies = [m.body(i).name for i in range(1, m.nbody)]
  for _ in range(50):
    lo, hi = m.jnt_range[[m.joint(n).id for n in names]].T
    q = rng.uniform(lo, hi) * 0.6
    qm = np.array([sign[i] * q[perm[i]] for i in range(len(q))])
    a = _set(m, d, names, q)
    b = _set(m, d, names, qm)
    for n in bodies:
      i, j = m.body(n).id, m.body(partner(n)).id
      err = np.linalg.norm(S @ a[i] - b[j])
      tol = 0.010 if LOOSE.search(n) else 0.001
      assert err < tol, f"{n}: mirrored CoM off by {1000 * err:.1f} mm (tol {1000 * tol:.0f})"


def _layout():
  m, _ = _model()
  joints = tuple(_hinges(m))
  legs = tuple(n for n in joints if any(s in n for s in ("hip", "knee", "ankle")))
  terms = [("angvel", 3, 10), ("gravity", 3, 10), (joints, 31, 10), (joints, 31, 10),
           (legs, 12, 10), ("cmd", 3, 0), (V11_UB_OBS_JOINTS, 17, 0), (V11_UB_OBS_JOINTS, 17, 0)]
  return terms, legs


def test_involution():
  terms, legs = _layout()
  om = ActorMirror(terms, "cpu")
  am = ActorMirror([(legs, 12, 0)], "cpu")
  assert om.width == 837
  x = torch.randn(64, 837)
  assert torch.allclose(om(om(x)), x)
  a = torch.randn(64, 12)
  assert torch.allclose(am(am(a)), a)
  # not trivially the identity
  assert not torch.allclose(om(x), x)


def test_anchor_symmetric():
  m, _ = _model()
  names = _hinges(m)
  jp = HOME_KEYFRAME.joint_pos
  q = []
  for n in names:
    hit = next((v for k, v in jp.items() if k == n or (any(c in k for c in "*$") and re.match(k, n))), 0.0)
    q.append(hit)
  q = np.array(q)
  perm, sign = joint_mirror(names)
  qm = np.array([sign[i] * q[perm[i]] for i in range(len(q))])
  bad = {n: (q[i], qm[i]) for i, n in enumerate(names) if abs(q[i] - qm[i]) > 1e-3}
  assert not bad, f"anchor not mirror-symmetric: {bad}"


def test_v11h_height_column():
  terms, _ = _layout()
  m11 = ActorMirror(terms, "cpu")
  mh = ActorMirror(terms + [("scalar", 1, 0)], "cpu")
  assert mh.width == 838
  assert torch.equal(mh.perm[:837], m11.perm) and torch.equal(mh.sign[:837], m11.sign)
  assert int(mh.perm[837]) == 837 and float(mh.sign[837]) == 1.0
  x = torch.randn(64, 838)
  assert torch.allclose(mh(mh(x)), x)
  assert torch.equal(mh(x)[:, 837], x[:, 837])
  assert torch.allclose(mh(x)[:, :837], m11(x[:, :837]))


if __name__ == "__main__":   # pytest is not installed in hv1v2_mjlab
  for _name in ("test_fk_reflection", "test_involution", "test_anchor_symmetric",
                "test_v11h_height_column"):
    globals()[_name]()
    print("PASS", _name)
