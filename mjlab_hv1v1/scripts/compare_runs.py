#!/usr/bin/env python3
"""Compare two training runs on metrics that are actually comparable.

WHY THIS EXISTS. Three HV1V1 runs were killed on bad cross-run comparisons, two
of them because of arithmetic that looks harmless:

  * ``Episode_Reward/<term>`` is NOT a rate. reward_manager.py:127 accumulates
    ``raw * weight * step_dt`` over the episode's ACTUAL steps, and :109 divides
    the episode sum by ``max_episode_length_s`` (a NOMINAL 20.0 s). So the logged
    value scales with the ACHIEVED episode length. A run that survives longer
    reads higher on every term with no behavioural difference -- and lower on the
    ones it has learned to stop paying. HV1V1 ran 853-step episodes against
    HV1V2's 580 at the same iteration, which confounded the comparison in both
    directions.
  * ``Metrics/twist/error_vel_xy`` accumulates per step and divides by
    ``resampling_time_range[1] / step_dt`` (velocity_command.py:64-75). Shortening
    resampling from 8.0 s to 6.0 s inflates it 1.33x for identical physical
    error. That artifact alone killed one run.

So: never eyeball these across runs. This script does the correction, reads every
constant from each run's OWN params/env.yaml rather than from current code, and
REFUSES to print a comparison it cannot make honest.

  python mjlab_hv1v1/scripts/compare_runs.py \
      --a mjlab_hv1v2/logs/rsl_rl/hv1v2_velocity/2026-09-25_00-09-30_v4 \
      --b mjlab_hv1v1/logs/rsl_rl/hv1v1_velocity/..._v4run \
      --iters 1000,3000,5000

Read-only: it opens TensorBoard event files and YAML, and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Physical quantities. Already means over envs and feet, independent of episode
# length and of every weight -- so they need no correction and are the metrics to
# trust. peak_height_mean and landing_force_mean are per-LANDING means, so they
# are conditioned on stepping happening at all: a robot that stops stepping makes
# them stale rather than zero.
RAW_METRICS = (
  "Metrics/air_time_mean",
  "Metrics/peak_height_mean",
  "Metrics/slip_velocity_mean",
  "Metrics/landing_force_mean",
  "Metrics/angular_momentum_mean",
)

TRAIN_METRICS = ("Train/mean_episode_length", "Train/mean_reward")


class _Tolerant(yaml.SafeLoader):
  """env.yaml carries python object tags for classes that may not import.

  Based on SafeLoader, NOT UnsafeLoader: UnsafeLoader registers a constructor for
  the longer prefix ``python/object/apply:`` which wins over a catch-all on
  ``python/``, and then fails on any class it cannot import (e.g. a viewer enum
  that moved). SafeLoader has no python constructors, so the catch-all applies.
  """


def _as_container(loader, suffix, node):
  del suffix
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return loader.construct_scalar(node)


for _prefix in ("tag:yaml.org,2002:python/", "!"):
  _Tolerant.add_multi_constructor(_prefix, _as_container)


class Run:
  def __init__(self, path: Path):
    self.path = path
    env_yaml = path / "params" / "env.yaml"
    if not env_yaml.exists():
      sys.exit(f"{path}: no params/env.yaml -- cannot normalise without its config")
    cfg = yaml.load(env_yaml.read_text(), Loader=_Tolerant)
    self.episode_length_s = float(cfg["episode_length_s"])
    self.step_dt = float(cfg["sim"]["mujoco"]["timestep"]) * int(cfg["decimation"])
    twist = cfg["commands"]["twist"]
    self.resample_max = float(twist["resampling_time_range"][1])
    self.rewards = cfg["rewards"]

    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    self._tags = set(ea.Tags()["scalars"])
    self._series = {t: {e.step: e.value for e in ea.Scalars(t)} for t in self._tags}

  @property
  def name(self) -> str:
    return self.path.name

  def weight(self, term: str) -> float | None:
    entry = self.rewards.get(term)
    return None if entry is None else float(entry["weight"])

  def params(self, term: str) -> dict:
    entry = self.rewards.get(term) or {}
    return {
      k: v for k, v in (entry.get("params") or {}).items() if isinstance(v, (int, float))
    }

  def at(self, tag: str, it: int) -> float | None:
    """Value at the logged step nearest `it`, or None if that is too far away."""
    series = self._series.get(tag)
    if not series:
      return None
    step = min(series, key=lambda s: abs(s - it))
    # Refuse a match further than one save interval away -- silently comparing
    # iteration 1000 against iteration 4000 is exactly the kind of error this
    # script exists to prevent.
    if abs(step - it) > max(200, 0.1 * it):
      return None
    return series[step]

  def reward_terms(self) -> set[str]:
    return {t.split("/", 1)[1] for t in self._tags if t.startswith("Episode_Reward/")}

  def per_step_raw(self, term: str, it: int) -> float | None:
    """Undo mjlab's logging so the number means "raw term value per step".

    logged = sum_over_achieved_steps(raw * weight * step_dt) / episode_length_s
    =>  raw_per_step = logged * episode_length_s / (ep_len_steps * step_dt * weight)
    """
    logged = self.at(f"Episode_Reward/{term}", it)
    ep_len = self.at("Train/mean_episode_length", it)
    w = self.weight(term)
    if logged is None or ep_len is None or not w or ep_len <= 0:
      return None
    return logged * self.episode_length_s / (ep_len * self.step_dt * w)

  def per_step_error_vel(self, it: int, tag: str) -> float | None:
    """error_vel_xy / error_vel_yaw as a genuine per-step error."""
    logged = self.at(tag, it)
    ep_len = self.at("Train/mean_episode_length", it)
    if logged is None or ep_len is None or ep_len <= 0:
      return None
    return logged * (self.resample_max / self.step_dt) / ep_len


def _fmt(v: float | None, width: int = 12, prec: int = 4) -> str:
  return " " * (width - 1) + "-" if v is None else f"{v:{width}.{prec}f}"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--a", required=True, help="reference run directory (e.g. HV1V2 v4)")
  ap.add_argument("--b", required=True, help="run under test")
  ap.add_argument("--iters", default="1000,3000,5000")
  ap.add_argument("--terms", default="", help="comma-separated reward terms; default = all shared")
  args = ap.parse_args()

  a, b = Run(Path(args.a)), Run(Path(args.b))
  iters = [int(x) for x in args.iters.split(",") if x.strip()]

  print(f"A = {a.name}\nB = {b.name}\n")

  # --- config differences that change what a metric MEANS -------------------
  print("CONFIG (anything flagged here makes the raw logged values incomparable)")
  for label, va, vb in (
    ("episode_length_s", a.episode_length_s, b.episode_length_s),
    ("step_dt", a.step_dt, b.step_dt),
    ("resampling_time_range[1]", a.resample_max, b.resample_max),
  ):
    flag = "" if va == vb else "   <- DIFFERS, corrected below"
    print(f"   {label:26s} A {va:<10g} B {vb:<10g}{flag}")

  terms = (
    [t.strip() for t in args.terms.split(",") if t.strip()]
    or sorted(a.reward_terms() & b.reward_terms())
  )
  differing = [
    (t, a.weight(t), b.weight(t)) for t in terms if a.weight(t) != b.weight(t)
  ]
  if differing:
    print("\n   reward weights differ (raw Episode_Reward is meaningless across these):")
    for t, wa, wb in differing:
      print(f"      {t:26s} A {wa}  B {wb}")
  diff_params = [
    (t, a.params(t), b.params(t)) for t in terms if a.params(t) != b.params(t)
  ]
  if diff_params:
    print("\n   reward params differ (the term measures a different thing in each run):")
    for t, pa, pb in diff_params:
      only = {k: (pa.get(k), pb.get(k)) for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
      print(f"      {t:26s} {only}")

  # --- physical metrics: no correction needed ------------------------------
  print("\nRAW PHYSICAL METRICS (no normalisation needed -- trust these)")
  for tag in RAW_METRICS + TRAIN_METRICS:
    if tag not in a._tags and tag not in b._tags:
      continue
    short = tag.split("/")[-1]
    row = "".join(f"{_fmt(a.at(tag, it), 11)}{_fmt(b.at(tag, it), 11)}" for it in iters)
    print(f"   {short:22s}{row}")
  print(f"   {'':22s}" + "".join(f"{'A@'+str(it):>11s}{'B@'+str(it):>11s}" for it in iters))

  # --- normalised reward terms ---------------------------------------------
  print("\nREWARD TERMS, normalised to RAW VALUE PER STEP")
  print("   (logged * episode_length_s / (mean_episode_length * step_dt * weight);")
  print("    so these are comparable even when weights and episode lengths differ)")
  print(f"   {'term':26s}" + "".join(f"{'A@'+str(it):>11s}{'B@'+str(it):>11s}" for it in iters))
  for t in terms:
    row = "".join(
      f"{_fmt(a.per_step_raw(t, it), 11)}{_fmt(b.per_step_raw(t, it), 11)}" for it in iters
    )
    mark = "  (w differs)" if a.weight(t) != b.weight(t) else ""
    print(f"   {t:26s}{row}{mark}")

  # --- command tracking ----------------------------------------------------
  print("\nCOMMAND TRACKING, per-step error (divisor and episode length corrected)")
  for tag in ("Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw"):
    short = tag.split("/")[-1]
    row = "".join(
      f"{_fmt(a.per_step_error_vel(it, tag), 11)}{_fmt(b.per_step_error_vel(it, tag), 11)}"
      for it in iters
    )
    print(f"   {short:26s}{row}")
  print(
    "\n   NOTE: per-step velocity error is a weak signal while neither policy walks."
    "\n   For a real tracking comparison, pin a fixed command on matched-iteration"
    "\n   checkpoints and read root_link_lin_vel_b instead."
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
