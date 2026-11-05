#!/usr/bin/env python3
"""Does walk_hv1v1_v5.mnn produce the same actions as policy.onnx?

This is the only check that matters after a format conversion. A converted graph
that loads, has the right input/output names and the right shapes can still be
numerically wrong -- a fused operator, a different default epsilon, a transposed
Gemm. So compare them on the SAME observations, and on REAL ones: sim_obs.npz
holds 800 frames recorded off an actual sim2sim rollout across 8 commands
(stand, forward 0.5/1.0/1.3, back 0.8, strafe, both turns), not random noise,
because random vectors never visit the part of the input space the robot uses.

  python verify_mnn.py              # ONNX vs MNN on sim_obs.npz
  python verify_mnn.py --random     # also sweep random inputs, as a stress test

Needs onnxruntime and MNN. Run it anywhere the two files are; it reads nothing
else from the project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# What a deployed action error actually costs, so the thresholds below mean
# something physical rather than being round numbers.
#   action -> joint target is  target = anchor + action * action_scale
#   the largest action_scale here is 0.32 rad (hips, knees)
# so 1e-3 of action is 3.2e-4 rad = 0.018 deg at the worst joint. Anything at or
# below that is far inside the robot's own encoder resolution and backlash.
TOL_GOOD = 1e-3
TOL_FAIL = 1e-2


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--random", action="store_true")
  args = ap.parse_args()

  import onnxruntime as ort
  import MNN

  cfg = json.loads((HERE / "deploy.json").read_text())
  obs_dim = cfg["policy"]["obs_dim"]
  act_dim = cfg["policy"]["action_dim"]
  scales = np.array([cfg["action"]["scale"][j["name"]] for j in cfg["joints"]])

  sess = ort.InferenceSession(str(HERE / "policy.onnx"),
                              providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name

  interp = MNN.Interpreter(str(HERE / "walk_hv1v1_v5.mnn"))
  session = interp.createSession({})
  mnn_in = interp.getSessionInput(session)
  mnn_out = interp.getSessionOutput(session)
  print(f"onnx in  : {in_name} {sess.get_inputs()[0].shape}")
  print(f"mnn  in  : {mnn_in.getShape()}   out {mnn_out.getShape()}")

  def run_mnn(x: np.ndarray) -> np.ndarray:
    tmp = MNN.Tensor((1, obs_dim), MNN.Halide_Type_Float,
                     x.astype(np.float32).reshape(1, obs_dim), MNN.Tensor_DimensionType_Caffe)
    mnn_in.copyFrom(tmp)
    interp.runSession(session)
    out = MNN.Tensor((1, act_dim), MNN.Halide_Type_Float,
                     np.zeros((1, act_dim), np.float32), MNN.Tensor_DimensionType_Caffe)
    mnn_out.copyToHostTensor(out)
    return np.array(out.getNumpyData()).reshape(act_dim).copy()

  def compare(X: np.ndarray, label: str) -> float:
    a = np.concatenate([sess.run(None, {in_name: X[i:i + 1]})[0] for i in range(len(X))])
    b = np.stack([run_mnn(X[i]) for i in range(len(X))])
    d = np.abs(a - b)
    # The number that matters on the robot: action error * action_scale, in rad.
    rad = (d * scales).max()
    print(f"\n{label}  ({len(X)} frames)")
    print(f"   max |onnx - mnn|      {d.max():.3e}")
    print(f"   mean |onnx - mnn|     {d.mean():.3e}")
    print(f"   worst joint-angle err {rad:.3e} rad  ({np.degrees(rad):.5f} deg)")
    return float(d.max())

  worst = 0.0
  npz = HERE / "sim_obs.npz"
  if npz.exists():
    d = np.load(npz)
    worst = max(worst, compare(d["obs"].astype(np.float32), "REAL sim2sim observations"))
    # Cross-check: the recorded ONNX actions must also still reproduce.
    a = np.concatenate([sess.run(None, {in_name: d["obs"][i:i + 1].astype(np.float32)})[0]
                        for i in range(len(d["obs"]))])
    print(f"   onnx reproduces recorded actions to {np.abs(a - d['act']).max():.3e}")
  else:
    print("\nsim_obs.npz missing -- falling back to random input only")
    args.random = True

  if args.random:
    rng = np.random.default_rng(0)
    worst = max(worst, compare(rng.normal(size=(256, obs_dim)).astype(np.float32),
                               "random N(0,1) stress test"))

  ok = worst <= TOL_FAIL
  grade = "EXCELLENT" if worst <= TOL_GOOD else ("ACCEPTABLE" if ok else "FAILED")
  print(f"\n  worst disagreement {worst:.3e}  ->  {grade}")
  print(f"  (<= {TOL_GOOD:g} is below the robot's encoder resolution; "
        f"> {TOL_FAIL:g} must not be deployed)")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
