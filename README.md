# hv1_series

RL locomotion for the **HV1V2** humanoid (92.7 kg, 31 joints): training in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp + rsl_rl PPO), sim2sim
in MuJoCo, and deployment bundles for the `merai_sdk` runtime.

## Layout

| Path | What |
|---|---|
| `mjlab_hv1v2/` | HV1V2 tasks, rewards, curricula (`src/mjlab_hv1v2`), scripts, tests, deploy bundles (`exported/`) |
| `mjlab_hv1v1/` | earlier HV1V1 tasks (reference) |
| `urdf/`, `meshes/`, `mjcf/` | robot description; `mjcf/urdf2mjcf.py` builds the MJCF |
| `URDF_HV1V2_26092026/` | original CAD URDF export |
| `HV1V2_training_summary/` | per-version training notes, curves, policies |
| `ACTUATOR_DATASHEET.txt` | actuator ratings |

Not in this repo (see `.gitignore`): `mjlab_src/` (upstream mjlab), `merai_sdk/`
([own repo](https://github.com/Aditya23w/merai_sdk)), training logs and `.pt` checkpoints.

## Setup

```bash
conda create -n hv1v2_mjlab python=3.11 && conda activate hv1v2_mjlab
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx (sm_120)
git clone https://github.com/mujocolab/mjlab.git mjlab_src
git -C mjlab_src checkout c2e1e0640 && pip install -e mjlab_src
pip install -e mjlab_hv1v2
```

Versions: mjlab 1.6.0, mujoco / mujoco-warp 3.11.0, rsl-rl-lib 5.5.1. Details in `SETUP.md`.

## Use

```bash
cd mjlab_hv1v2
# train (latest task: v11h, variable pelvis height)
RUN=v11h_s43 ./scripts/train.sh Mjlab-Velocity-Flat-HV1V2-Decoupled-v11h
# evaluate
python scripts/eval_v11_gates.py --ckpt <model.pt> --task <task> [--height 0.92]
python scripts/push_sweep.py      --ckpt <model.pt> --task <task> --heights 0.979 0.89
# sim2sim (keyboard: arrows = height, PgUp = home)
python scripts/sim2sim.py --onnx exported/deploy_v11h27k/policy.onnx
# deploy bundle
python scripts/export_deploy.py --run <run_suffix> --task <task>
```

## Policies

| Bundle | Policy | Notes |
|---|---|---|
| `exported/deploy_v11h27k` | v11h `model_26999` | walk + stand, pelvis height 0.89–0.979 m (obs 838) |
| `exported/deploy_v11g19k` | v11g `model_19000` | fixed height, teleop upper body (obs 837) |

Each bundle has `policy.onnx`, `deploy.json` (joint order, gains, obs layout,
limits) and a README for the bundle. In progress: **v11i**, a 0.86 m height floor plus operator waist pitch.
