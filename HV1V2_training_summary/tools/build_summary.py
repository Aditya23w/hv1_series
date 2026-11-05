"""Regenerate 03_training/runs/* from the mjlab run directories.

For every run listed in RUNS it writes, into 03_training/runs/<version>/:
  env.yaml, agent.yaml      verbatim copies of the run's params/
  config_summary.md         rewards, observations, actions, DR events, commands,
                            terminations, curriculum, PPO, read from env.yaml
  curves.csv                every TensorBoard scalar, one column per tag
  curves.png                the key training curves

Run with the hv1v2_mjlab conda env (needs tensorboard, matplotlib, pyyaml):
  python tools/build_summary.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT.parent / "mjlab_hv1v2" / "logs" / "rsl_rl" / "hv1v2_velocity"
OUT = ROOT / "03_training" / "runs"

# version -> run directory. Smoke / check runs are left out.
RUNS = {
  "v1": "2026-09-24_20-25-01_v1",
  "v2": "2026-09-24_21-01-54_v2",
  "v3": "2026-09-24_21-21-00_v3",
  "v4": "2026-09-25_00-09-30_v4",
  "v5": "2026-10-05_15-19-51_v5",
  "v6": "2026-10-05_18-38-33_v6",
  "v7": "2026-10-06_17-58-29_v7",
  "v8": "2026-10-10_22-35-10_v8",
  "v9": "2026-10-20_23-37-27_v9",
  "v5ft": "2026-10-26_20-07-21_v5ft",
  "v10": "2026-10-26_23-21-21_v10",
  # v11: final stage (v11g) only. Its TensorBoard logs (and v11b-v11f) are on the
  # remote PC; only the checkpoint and params were copied here.
  "v11": "2026-11-02_13-24-36_v11g19k",
}

PLOTS = [
  ("Train/mean_reward", "mean episode reward"),
  ("Train/mean_episode_length", "mean episode length (steps)"),
  ("Episode_Termination/fell_over", "Episode_Termination/fell_over (as logged)"),
  ("Episode_Reward/track_linear_velocity", "track_linear_velocity"),
  ("Episode_Reward/track_angular_velocity", "track_angular_velocity"),
  ("Policy/mean_noise_std", "action noise std"),
  ("Metrics/torque_frac_of_continuous", "torque / continuous rating"),
  ("Episode_Reward/action_rate_l2", "action_rate_l2"),
  ("Loss/learning_rate", "learning rate"),
]


class _Loader(yaml.SafeLoader):
  pass


def _any_tag(loader, suffix, node):
  if suffix.startswith("name:"):
    return suffix[len("name:"):]
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _any_tag)


def load_yaml(p: Path):
  return yaml.load(p.read_text(), _Loader)


def short(f) -> str:
  return str(f).rsplit(".", 1)[-1] if f else ""


def _clean(v):
  """Drop None / empty / internal fields so params read as what was set."""
  if isinstance(v, dict):
    out = {}
    for k, x in v.items():
      if str(k).startswith("_") or k in ("name", "preserve_order"):
        continue
      x = _clean(x)
      if x in (None, [], {}, ""):
        continue
      out[k] = x
    return out
  if isinstance(v, list):
    xs = [_clean(x) for x in v]
    return [] if all(x is None for x in xs) else xs
  return v


def fmt(v, n: int = 220) -> str:
  s = str(_clean(v)).replace("|", "\\|")
  return s if len(s) <= n else s[: n - 3] + "..."


def config_md(version: str, run: str, env: dict, agent: dict) -> str:
  L = [f"# {version} — `{run}`", ""]
  L += ["Generated from `params/env.yaml` and `params/agent.yaml` by "
        "`tools/build_summary.py`.", ""]

  sim = env.get("sim", {}).get("mujoco", {})
  L += ["## Simulation", "",
        f"- num_envs: {env['scene'].get('num_envs')}",
        f"- physics dt: {sim.get('timestep')} s, decimation {env.get('decimation')} "
        f"→ policy at {1 / (sim.get('timestep', 0.005) * env.get('decimation', 4)):.0f} Hz",
        f"- episode length: {env.get('episode_length_s')} s", ""]

  L += ["## Rewards", "", "| term | function | weight | params |", "|---|---|---|---|"]
  for k, v in env["rewards"].items():
    L.append(f"| `{k}` | {short(v.get('func'))} | {v.get('weight')} | {fmt(v.get('params', {}))} |")
  L.append("")

  for grp, g in env["observations"].items():
    L += [f"## Observations — {grp}", "",
          f"concatenate={g.get('concatenate_terms')}, corruption={g.get('enable_corruption')}, "
          f"history={g.get('history_length')}", "",
          "| term | function | noise | history | params |", "|---|---|---|---|---|"]
    for k, v in g.get("terms", {}).items():
      if not isinstance(v, dict):
        continue
      L.append(f"| `{k}` | {short(v.get('func'))} | {fmt(v.get('noise'))} | "
               f"{v.get('history_length')} | {fmt(v.get('params', {}))} |")
    L.append("")

  L += ["## Actions", ""]
  for k, v in env["actions"].items():
    L += [f"### `{k}`", "", "```", fmt(yaml.safe_dump(
      {kk: vv for kk, vv in v.items() if kk not in ("scale", "offset")},
      default_flow_style=False, width=120))[:4000], "```"]
    if isinstance(v.get("scale"), dict):
      L += ["", "scale: " + ", ".join(f"`{a}` {b}" for a, b in v["scale"].items())]
  L.append("")

  L += ["## Events (resets, domain randomisation, pushes)", "",
        "| event | function | mode | params |", "|---|---|---|---|"]
  for k, v in env["events"].items():
    L.append(f"| `{k}` | {short(v.get('func'))} | {v.get('mode')} "
             f"{v.get('interval_range_s') or ''} | {fmt(v.get('params', {}))} |")
  L.append("")

  L += ["## Commands", ""]
  for k, v in env["commands"].items():
    keep = {kk: vv for kk, vv in v.items() if kk not in ("viz", "debug_vis")}
    L += [f"### `{k}`", "", "```", yaml.safe_dump(keep, default_flow_style=None,
                                                    width=120)[:4000], "```", ""]

  L += ["## Terminations", ""]
  for k, v in env["terminations"].items():
    L.append(f"- `{k}`: {short(v.get('func'))} {fmt(v.get('params', {}))}"
             f"{' (time-out)' if v.get('time_out') else ''}")
  L.append("")

  if env.get("curriculum"):
    L += ["## Curriculum", ""]
    for k, v in env["curriculum"].items():
      L.append(f"- `{k}`: {short(v.get('func'))} {fmt(v.get('params', {}), 2000)}")
    L.append("")

  if env.get("metrics"):
    L += ["## Logged metrics", "", ", ".join(f"`{k}`" for k in env["metrics"]), ""]

  alg = agent.get("algorithm", {})
  L += ["## PPO (rsl_rl)", "",
        f"- actor {agent['actor']['hidden_dims']} {agent['actor']['activation']}, "
        f"critic {agent['critic']['hidden_dims']}, obs normalisation "
        f"{agent['actor'].get('obs_normalization')}",
        f"- init action std {agent['actor'].get('distribution_cfg', {}).get('init_std')}",
        f"- steps/env/iter {agent.get('num_steps_per_env')}, max_iterations "
        f"{agent.get('max_iterations')}, save every {agent.get('save_interval')}",
        f"- resume: {agent.get('resume')} from `{agent.get('load_run')}` / "
        f"`{agent.get('load_checkpoint')}`", "",
        "| param | value |", "|---|---|"]
  for k, v in alg.items():
    L.append(f"| {k} | {v} |")
  return "\n".join(L) + "\n"


def export_curves(run_dir: Path, out: Path) -> dict[str, list]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ev = sorted(run_dir.glob("events.out.tfevents.*"))
  if not ev:
    return {}
  acc = EventAccumulator(str(ev[0]), size_guidance={"scalars": 0})
  acc.Reload()
  tags = acc.Tags()["scalars"]
  series = {t: {e.step: e.value for e in acc.Scalars(t)} for t in tags}
  steps = sorted({s for d in series.values() for s in d})
  with open(out / "curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iteration", *tags])
    for s in steps:
      w.writerow([s, *(series[t].get(s, "") for t in tags)])
  return series


def _ema(ys, a: float = 0.02):
  out, m = [], None
  for y in ys:
    m = y if m is None else m + a * (y - m)
    out.append(m)
  return out


def plot_curves(series: dict, version: str, out: Path) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  avail = [(t, lbl) for t, lbl in PLOTS if t in series]
  if not avail:
    return
  n = len(avail)
  cols = 3
  rows = (n + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
  for ax, (t, lbl) in zip(axes.flat, avail):
    xs = sorted(series[t])
    ys = [series[t][x] for x in xs]
    ax.plot(xs, ys, lw=0.5, alpha=0.3, color="C0")
    ax.plot(xs, _ema(ys), lw=1.4, color="C0")
    ax.set_title(lbl, fontsize=10)
    ax.set_xlabel("iteration", fontsize=8)
    ax.grid(alpha=0.3)
  for ax in list(axes.flat)[n:]:
    ax.axis("off")
  fig.suptitle(f"HV1V2 {version} training", fontsize=13)
  fig.tight_layout()
  fig.savefig(out / "curves.png", dpi=110)
  plt.close(fig)


def plot_lineage(all_series: dict[str, dict]) -> None:
  """v1..v5 reward / episode length on one absolute-iteration axis."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(16, 4))
  for (t, lbl), ax in zip(PLOTS[:3], axes):
    for v in ("v1", "v2", "v3", "v4", "v5"):
      s = all_series.get(v, {}).get(t)
      if s:
        xs = sorted(s)
        ax.plot(xs, _ema([s[x] for x in xs]), lw=1.2, label=v)
    ax.set_title(lbl)
    ax.set_xlabel("iteration (as logged; v5 resumes v4 at 40000)")
    ax.grid(alpha=0.3)
    ax.legend()
  fig.suptitle("Full-body locomotion lineage v1 → v5")
  fig.tight_layout()
  fig.savefig(OUT.parent / "lineage_v1_v5.png", dpi=110)
  plt.close(fig)


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  all_series = {}
  for version, run in RUNS.items():
    rd = LOGS / run
    if not rd.exists():
      print(f"skip {version}: {rd} missing")
      continue
    od = OUT / version
    od.mkdir(exist_ok=True)
    for name in ("env.yaml", "agent.yaml"):
      shutil.copy2(rd / "params" / name, od / name)
    env = load_yaml(rd / "params" / "env.yaml")
    agent = load_yaml(rd / "params" / "agent.yaml")
    (od / "config_summary.md").write_text(config_md(version, run, env, agent))
    ckpts = sorted((p.name for p in rd.glob("model_*.pt")),
                   key=lambda s: int(s[6:-3]))
    (od / "checkpoints.txt").write_text(
      f"run dir: mjlab_hv1v2/logs/rsl_rl/hv1v2_velocity/{run}\n" + "\n".join(ckpts) + "\n")
    series = export_curves(rd, od)
    all_series[version] = series
    plot_curves(series, version, od)
    print(f"{version}: {len(series)} scalar tags, {len(ckpts)} checkpoints")
  plot_lineage(all_series)


if __name__ == "__main__":
  main()
