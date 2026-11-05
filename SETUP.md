# HV1V2 — mjlab training environment

## Environment

Conda env **`hv1v2_mjlab`** (`/home/aditya/miniconda3/envs/hv1v2_mjlab`), with mjlab
installed **editable** from the `mjlab_src/` checkout in this directory.

```bash
conda activate hv1v2_mjlab
```

| | |
|---|---|
| Python | 3.11.16 |
| mjlab | 1.6.0 (editable, `mjlab_src/` @ `c2e1e0640`) |
| mujoco / mujoco-warp | 3.11.0 / 3.11.0 |
| warp-lang | 1.17.0 |
| torch | 2.11.0+cu128 |
| rsl-rl-lib | 5.5.1 |
| GPU | RTX 5080 Laptop, sm_120, 16 GB |

Installed per mjlab's officially supported "Method 3 — classic pip / venv / conda"
path. `mjlab_src` is a normal git clone, so `git pull && pip install -e .` updates it.

### Why these versions

- **torch from the cu128 index.** The RTX 5080 is Blackwell (`sm_120`); the default
  PyPI torch wheel does not ship that arch. Verified `sm_120` is in
  `torch.cuda.get_arch_list()` and that Warp reports `arch 120`.
- **mujoco 3.11.0** is what `mjlab==1.6.0` pins, and it is the same version
  `mjcf/hv1v2.xml` was built and validated against.
- A separate env from the existing `unitree_rl_mjlab` (mjlab 1.2.0, mujoco 3.5.0),
  which is too old for this model.

## Verification actually run

- Warp initializes on `cuda:0`, `sm_120`, mempool enabled.
- `list-envs` reports all 12 built-in tasks.
- **Real training runs end to end** on `Mjlab-Velocity-Flat-Unitree-G1`:
  512 envs → 0.32 s/iter; 4096 envs → 0.63 s/iter. Checkpoints, tensorboard
  events and an exported `.onnx` are all written.
- `mjcf/hv1v2.xml` loads under this env's mujoco 3.11.0, steps 4 s clean
  (0 warnings), and parses as an `MjSpec` — 35 bodies, 31 actuators, 2 keyframes.

```bash
conda activate hv1v2_mjlab
list-envs
train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## HV1V2 in mjlab

Registered and training. See `mjlab_hv1v2/README.md`.

```bash
conda activate hv1v2_mjlab
list-envs | grep HV1V2
./mjlab_hv1v2/scripts/train.sh
```

| Task | Actor sees | Use |
|---|---|---|
| `Mjlab-Velocity-Flat-HV1V2` | IMU + encoders + actions + command | **primary** — deployable |
| `Mjlab-Velocity-Flat-HV1V2-State-Estimation` | + base linear velocity | baseline / ablation |
| `Mjlab-Velocity-Rough-HV1V2` | + terrain height scan | rough terrain |

`mjlab_hv1v2` is a standalone pip package that registers through mjlab's
`mjlab.tasks` entry point, so **`mjlab_src/` is still a pristine clone** and can be
`git pull`ed. The robot's plant comes from `ACTUATOR_DATASHEET.txt`; the MJCF the
tasks use is `mjcf/urdf2mjcf.py --target robot`, not the viewer model.
