#!/usr/bin/env python3
"""Generate MuJoCo models from urdf/HV1V2_URDF_NEW_FOOT_V1_220926.urdf.

Two targets, from one source of truth:

  --target viewer   mjcf/hv1v2.xml  -- standalone, for looking at the robot.
                    Floor, lights, skybox, position servos with placeholder
                    gains, and two keyframes.  This is what mjcf/view.py loads.

  --target robot    a training model for mjlab.  No floor, no lights, no
                    actuators, NO KEYFRAMES, and the sensor block the mjlab
                    velocity task requires.  Adds a pelvis IMU site, flat foot
                    collision pads and foot sites.  mjlab supplies terrain,
                    lights, actuators and the initial-state keyframe itself.

Do not hand-edit the generated XML; change this file and regenerate.
"""
import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Two DIFFERENT ROBOTS share this generator. They are the same body with
# different feet, and they must never be mixed:
#
#   hv1v2  the newer foot. Ankle 36.5% from the heel, 0.969 kg per foot.
#   hv1v1  the older foot. Ankle 22.0% from the heel, 1.782 kg per foot, and
#          a sole that reaches 52 mm further forward and 34 mm less far back.
#
# NOTE ON NAMING: hv1v1's export directory and URDF are called
# "URDF_HV1V2_26092026" even though it is the OLD-foot robot. The foot geometry
# is what identifies it, not the filename.
# ---------------------------------------------------------------------------
ROBOTS = {
    "hv1v2": dict(
        urdf=os.path.join(ROOT, "urdf/HV1V2_URDF_NEW_FOOT_V1_220926.urdf"),
        meshes=os.path.join(ROOT, "meshes"),
        viewer_out=os.path.join(ROOT, "mjcf/hv1v2.xml"),
        viewer_meshdir="../meshes",
        robot_out=os.path.join(
            ROOT, "mjlab_hv1v2/src/mjlab_hv1v2/assets/hv1v2/hv1v2_robot.xml"),
        patches={
            "left":  [dict(x=(-0.0959, -0.0131), y=(-0.0502, 0.0502)),
                      dict(x=(0.0811, 0.1671),   y=(-0.0585, 0.0585))],
            "right": [dict(x=(-0.0956, -0.0128), y=(-0.0502, 0.0502)),
                      dict(x=(0.0814, 0.1674),   y=(-0.0585, 0.0585))],
        },
        sole_z={"left": -0.0380, "right": -0.0380},
    ),
    "hv1v1": dict(
        urdf=os.path.join(ROOT, "URDF_HV1V2_26092026/urdf/URDF_HV1V2_26092026.urdf"),
        meshes=os.path.join(ROOT, "URDF_HV1V2_26092026/meshes"),
        viewer_out=os.path.join(ROOT, "mjcf/hv1v1.xml"),
        viewer_meshdir="../URDF_HV1V2_26092026/meshes",
        robot_out=os.path.join(
            ROOT, "mjlab_hv1v1/src/mjlab_hv1v1/assets/hv1v1/hv1v1_robot.xml"),
        patches={
            "left":  [dict(x=(-0.0620, 0.0320), y=(-0.0540, 0.0540)),
                      dict(x=(0.1140, 0.2195),  y=(-0.0629, 0.0629))],
            "right": [dict(x=(-0.0617, 0.0323), y=(-0.0540, 0.0540)),
                      dict(x=(0.1143, 0.2198),  y=(-0.0629, 0.0629))],
        },
        sole_z={"left": -0.0374, "right": -0.0374},
    ),
}

# Selected before the module-level URDF parse below, so --robot has to be read
# from argv here rather than in main().
def _pick_robot() -> str:
    import sys
    for i, a in enumerate(sys.argv):
        if a == "--robot" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--robot="):
            return a.split("=", 1)[1]
    return "hv1v2"


ROBOT = _pick_robot()
assert ROBOT in ROBOTS, f"unknown --robot {ROBOT!r}; choose from {sorted(ROBOTS)}"
CFG = ROBOTS[ROBOT]
URDF = CFG["urdf"]
MESHES = CFG["meshes"]

# Bodies that carry a site but no collision geometry.
NO_COLLIDE = {"pelvis_imu_link", "torso_imu_link"}
# The feet get flat pads instead of a convex hull of the whole foot mesh.
FOOT_BODIES = {"left_ankle_roll_link": "left", "right_ankle_roll_link": "right"}

# Sensors the mjlab velocity task looks up by name (velocity_env_cfg.py
# references them as robot/<name>; the prefix comes from MjSpec.attach).
REQUIRED_SENSORS = ("imu_lin_vel", "imu_ang_vel", "root_angmom")

r = ET.parse(URDF).getroot()


def fv(s):
    return [float(x) for x in s.split()]


def f(x, n=6):
    s = f"{x:.{n}g}"
    return "0" if s in ("-0", "-0.0") else s


def vec(v, n=6):
    return " ".join(f(x, n) for x in v)


# ---- links -----------------------------------------------------------------
links = {}
for l in r.findall("link"):
    name = l.get("name")
    d = {"name": name}
    i = l.find("inertial")
    if i is not None:
        it = i.find("inertia")
        d["inertial"] = dict(
            pos=fv(i.find("origin").get("xyz")),
            mass=float(i.find("mass").get("value")),
            # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
            full=[float(it.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")],
        )
    v = l.find("visual")
    if v is not None:
        d["mesh"] = os.path.basename(v.find("geometry/mesh").get("filename"))
        c = v.find("material/color")
        d["rgba"] = fv(c.get("rgba")) if c is not None else [0.8, 0.8, 0.8, 1]
    links[name] = d

# ---- joints ----------------------------------------------------------------
children, jinfo = {}, {}
child_links = set()
for j in r.findall("joint"):
    o, p, c = j.find("origin"), j.find("parent").get("link"), j.find("child").get("link")
    ax = j.find("axis")
    lim = j.find("limit")
    jinfo[c] = dict(
        name=j.get("name"), type=j.get("type"), parent=p,
        pos=fv(o.get("xyz")), rpy=fv(o.get("rpy")),
        axis=fv(ax.get("xyz")) if ax is not None else None,
        range=[float(lim.get("lower")), float(lim.get("upper"))] if lim is not None else None,
        effort=float(lim.get("effort")) if lim is not None else None,
        vel=float(lim.get("velocity")) if lim is not None else None,
    )
    children.setdefault(p, []).append(c)
    child_links.add(c)

root_link = [n for n in links if n not in child_links]
assert len(root_link) == 1, root_link
root_link = root_link[0]

act_joints = [ji for ji in jinfo.values() if ji["type"] != "fixed"]
JNAMES = [ji["name"] for ji in act_joints]
TOTAL_MASS = sum(d["inertial"]["mass"] for d in links.values() if "inertial" in d)

# ---- materials (dedup by colour) -------------------------------------------
mats, mat_of = {}, {}
for n, d in links.items():
    if "rgba" not in d:
        continue
    key = tuple(round(x, 4) for x in d["rgba"])
    if key not in mats:
        mats[key] = f"mat{len(mats)}"
    mat_of[n] = mats[key]


def mesh_file(name):
    """The STL the MJCF should reference (decimated copy where one exists)."""
    fn = links[name]["mesh"]
    alt = fn[:-4] + "_mjcf.STL"
    if os.path.exists(os.path.join(MESHES, alt)):
        return alt, f"  <!-- decimated copy of {fn} (>200k faces) -->"
    return fn, ""


# ============================================================================
# Emitter
# ============================================================================
IND = "  "


def build(target, meshdir, pads=None, poses=None):
    """Return the XML for one target as a list of lines."""
    robot = target == "robot"
    L = []

    def w(depth, s):
        L.append(IND * depth + s)

    def emit_body(name, depth):
        ji = jinfo.get(name)
        attrs = [f'name="{name}"']
        if ji:
            attrs.append(f'pos="{vec(ji["pos"])}"')
            if any(abs(x) > 1e-12 for x in ji["rpy"]):
                q = R.from_euler("xyz", ji["rpy"]).as_quat()  # x,y,z,w
                attrs.append(f'quat="{vec([q[3], q[0], q[1], q[2]], 9)}"')
        w(depth, f"<body {' '.join(attrs)}>")

        if name == root_link:
            w(depth + 1, '<freejoint name="floating_base"/>')
            if robot:
                # The IMU the state estimator reads. Co-located with
                # pelvis_imu_link, but on the pelvis so it survives any change
                # to the IMU link itself. gyro/velocimeter report in this frame.
                w(depth + 1, '<site name="imu_in_pelvis" pos="0 0 -0.02" size="0.01" group="4"/>')

        d = links[name]
        if "inertial" in d:
            i = d["inertial"]
            w(depth + 1, f'<inertial pos="{vec(i["pos"])}" mass="{f(i["mass"], 9)}" '
                         f'fullinertia="{vec(i["full"], 9)}"/>')

        if ji and ji["type"] != "fixed":
            a = [f'name="{ji["name"]}"', f'axis="{vec(ji["axis"], 3)}"']
            if ji["range"]:
                a.append(f'range="{vec(ji["range"])}"')
            if ji["effort"]:
                a.append(f'actuatorfrcrange="{f(-ji["effort"])} {f(ji["effort"])}"')
            w(depth + 1, f'<joint {" ".join(a)}/>')

        if name in NO_COLLIDE:
            w(depth + 1, f'<site name="{name}_site" pos="0 0 0" size="0.01" group="4"/>')

        vis_sfx, col_sfx = ("_visual", "_collision") if robot else ("_vis", "_col")
        if "mesh" in d:
            m = d["mesh"][:-4]
            w(depth + 1, f'<geom name="{name}{vis_sfx}" class="visual" mesh="{m}" '
                         f'material="{mat_of[name]}"/>')
            # The feet swap their convex hull for flat pads: a rounded hull
            # under a flat sole fights the pads for contact.
            skip_hull = robot and name in FOOT_BODIES
            if name not in NO_COLLIDE and not skip_hull:
                w(depth + 1, f'<geom name="{name}{col_sfx}" class="collision" mesh="{m}"/>')

        if robot and name in FOOT_BODIES and pads:
            side = FOOT_BODIES[name]
            p = pads[side]
            w(depth + 1, f'<site name="{side}_foot" pos="{vec(p["site"], 5)}" '
                         f'size="0.01" group="4" rgba="1 0 0 1"/>')
            for k, (pos, size) in enumerate(p["pads"], start=1):
                w(depth + 1, f'<geom name="{side}_foot{k}_collision" class="collision" '
                             f'type="box" size="{vec(size, 5)}" pos="{vec(pos, 5)}"/>')

        for c in children.get(name, []):
            emit_body(c, depth + 1)
        w(depth, "</body>")

    # ---- header ------------------------------------------------------------
    w(0, f'<mujoco model="{ROBOT.upper()}">')
    w(1, '<!-- Generated by mjcf/urdf2mjcf.py from')
    w(1, '     urdf/HV1V2_URDF_NEW_FOOT_V1_220926.urdf (SolidWorks export).')
    w(1, f'     {len(links)} links / {len(act_joints)} actuated hinge joints, '
         f'total mass {TOTAL_MASS:.2f} kg.')
    if robot:
        w(1, '     TARGET=robot: training model for mjlab. No floor, lights,')
        w(1, '     actuators or keyframes - mjlab supplies all four. Do not add')
        w(1, '     a keyframe here: Scene._add_entities takes spec.keys[0] and')
        w(1, '     would use it instead of the configured initial state. -->')
    else:
        w(1, '     TARGET=viewer: standalone model for mjcf/view.py. The actuator')
        w(1, '     gains and joint defaults here are PLACEHOLDERS, not the real')
        w(1, '     plant; the training model gets those from mjlab config. -->')
    w(1, f'<compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>')
    if not robot:
        # mjlab drives timestep/integrator through MujocoCfg and warns that any
        # non-default <option> field on an attached entity is ignored.
        w(1, '<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>')
        w(1, '<size memory="20M"/>')
        L.append("")
        w(1, "<visual>")
        w(2, '<global offwidth="1920" offheight="1080" azimuth="140" elevation="-15"/>')
        w(2, '<quality shadowsize="4096" offsamples="8"/>')
        w(2, '<headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>')
        w(2, '<map znear="0.01" zfar="30"/>')
        w(2, '<rgba haze="0.15 0.18 0.22 1"/>')
        w(1, "</visual>")
    L.append("")

    # ---- defaults ----------------------------------------------------------
    w(1, "<default>")
    if robot:
        w(2, '<!-- No <joint> defaults on purpose. armature / frictionloss /')
        w(2, '     damping come from the mjlab actuator config, and a joint that')
        w(2, '     no actuator regex matches must stay at zero so the mismatch')
        w(2, '     is detectable rather than silently inheriting a placeholder. -->')
    else:
        w(2, '<!-- URDF carries no joint dynamics; these are viewer-friendly placeholders. -->')
        w(2, '<joint damping="1" armature="0.01" frictionloss="0.1"/>')
    w(2, '<geom solref="0.005 1"/>')
    w(2, '<default class="visual">')
    w(3, '<geom type="mesh" group="2" contype="0" conaffinity="0" density="0"/>')
    w(2, "</default>")
    w(2, '<default class="collision">')
    w(3, '<geom type="mesh" group="3" density="0" friction="0.8 0.02 0.01" rgba="0.9 0.4 0.2 0.4"/>')
    w(2, "</default>")
    w(1, "</default>")
    L.append("")

    # ---- assets ------------------------------------------------------------
    w(1, "<asset>")
    if not robot:
        w(2, '<texture type="skybox" builtin="gradient" rgb1="0.24 0.3 0.4" rgb2="0.05 0.06 0.08" width="512" height="3072"/>')
        w(2, '<texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.28" rgb2="0.31 0.33 0.35" width="512" height="512"/>')
        w(2, '<material name="grid" texture="grid" texrepeat="6 6" texuniform="true" reflectance="0.05"/>')
    for key, mname in mats.items():
        w(2, f'<material name="{mname}" rgba="{vec(key, 5)}" specular="0.3" shininess="0.3"/>')
    for n in sorted(links):
        if "mesh" not in links[n]:
            continue
        fn, note = mesh_file(n)
        w(2, f'<mesh name="{links[n]["mesh"][:-4]}" file="{fn}" maxhullvert="64"/>{note}')
    w(1, "</asset>")
    L.append("")

    # ---- worldbody ---------------------------------------------------------
    w(1, "<worldbody>")
    if not robot:
        w(2, '<light name="spot" pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>')
        w(2, '<light name="fill" pos="2 -2 2.5" dir="-0.6 0.6 -0.7" diffuse="0.35 0.35 0.35"/>')
        w(2, '<geom name="floor" type="plane" size="0 0 0.05" material="grid" condim="3" friction="0.8 0.02 0.01"/>')
    emit_body(root_link, 2)
    w(1, "</worldbody>")
    L.append("")

    # ---- contact -----------------------------------------------------------
    # MuJoCo already skips parent-child pairs; grandparent pairs overlap too
    # because collision uses convex hulls of the full link meshes.
    w(1, "<contact>")
    for c, ji in jinfo.items():
        gp = jinfo.get(ji["parent"], {}).get("parent")
        if gp and "mesh" in links[c] and "mesh" in links[gp]:
            w(2, f'<exclude body1="{gp}" body2="{c}"/>')
    # The torso hull fills in the concave waist, so it clips the pelvis once the
    # waist joints move off zero.
    w(2, '<exclude body1="pelvis" body2="torso_link"/>')
    w(1, "</contact>")
    L.append("")

    # ---- actuators (viewer only) -------------------------------------------
    if not robot:
        w(1, "<actuator>")
        w(2, '<!-- Position servos. The URDF gives no gains, so kp/kv are scaled off each')
        w(2, '     joint\'s effort limit - stiff enough to hold the keyframes under gravity.')
        w(2, '     forcerange is the URDF effort limit verbatim. -->')
        for ji in act_joints:
            lo, hi = ji["range"]
            e = ji["effort"]
            kp, kv = 40.0 * e, 1.2 * e
            w(2, f'<position name="{ji["name"]}" joint="{ji["name"]}" '
                 f'kp="{f(kp)}" kv="{f(kv)}" '
                 f'ctrlrange="{f(lo)} {f(hi)}" forcerange="{f(-e)} {f(e)}"/>')
        w(1, "</actuator>")
        L.append("")

    # ---- sensors -----------------------------------------------------------
    w(1, "<sensor>")
    if robot:
        w(2, '<!-- Exactly the three the mjlab velocity task resolves by name,')
        w(2, '     plus the accelerometer for a hardware-matched IMU triple. -->')
        w(2, '<gyro name="imu_ang_vel" site="imu_in_pelvis"/>')
        w(2, '<velocimeter name="imu_lin_vel" site="imu_in_pelvis"/>')
        w(2, '<accelerometer name="imu_lin_acc" site="imu_in_pelvis"/>')
        w(2, f'<subtreeangmom name="root_angmom" body="{root_link}"/>')
    else:
        for b, jn in (("pelvis_imu_link", "pelvis"), ("torso_imu_link", "torso")):
            w(2, f'<framequat name="{jn}_imu_quat" objtype="site" objname="{b}_site"/>')
            w(2, f'<gyro name="{jn}_imu_gyro" site="{b}_site"/>')
            w(2, f'<accelerometer name="{jn}_imu_acc" site="{b}_site"/>')
    w(1, "</sensor>")

    # ---- keyframes (viewer only) -------------------------------------------
    if not robot:
        L.append("")
        w(1, "<keyframe>")
        for pname, pose in poses.items():
            w(2, f'<key name="{pname}" qpos="{vec(qpos_of(pose, 0), 9)}" '
                 f'ctrl="{vec([pose.get(n, 0.0) for n in JNAMES], 9)}"/>')
        w(1, "</keyframe>")

    w(0, "</mujoco>")
    return L


def qpos_of(pose, z):
    return [0, 0, z, 1, 0, 0, 0] + [pose.get(n, 0.0) for n in JNAMES]


# ============================================================================
# Foot pads, measured from the real sole geometry
# ============================================================================
# The sole is NOT a flat plate. It has two contact patches - heel and toe -
# separated by an arch that sits 6-8 mm proud of the sole plane. Tiling the
# bounding box would fill that arch in and merge two patches into one, which
# would widen the support polygon and change how the foot rolls. So the pads
# are fitted to the measured patches instead.
#
# Regression guard: patches measured once, in the ankle_roll body frame.
EXPECTED_PATCHES = CFG["patches"]
EXPECTED_SOLE_Z = CFG["sole_z"]
SOLE_SLAB = 0.002      # vertices within this of the lowest count as "sole"
PATCH_GAP = 0.030      # x-gap that separates one contact patch from the next
PAD_HALF_Z = 0.006     # pad half-thickness


def measure_feet(viewer_xml_path):
    """Contact patches per foot, in the ankle_roll body frame, from the mesh."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(viewer_xml_path)
    out = {}
    for body, side in FOOT_BODIES.items():
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_col")
        assert g >= 0, f"no collision geom on {body}"
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        # geom frame -> body frame
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        loc = v @ Rg.reshape(3, 3).T + m.geom_pos[g]
        z = float(loc[:, 2].min())
        sole = loc[loc[:, 2] < z + SOLE_SLAB]

        # split the contact vertices into patches along x
        order = np.argsort(sole[:, 0])
        xs = sole[order, 0]
        cuts = [0] + list(np.where(np.diff(xs) > PATCH_GAP)[0] + 1) + [len(xs)]
        patches = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel = sole[order[a:b]]
            patches.append(dict(x=(float(sel[:, 0].min()), float(sel[:, 0].max())),
                                y=(float(sel[:, 1].min()), float(sel[:, 1].max()))))

        exp = EXPECTED_PATCHES[side]
        assert len(patches) == len(exp), (
            f"{side} foot: found {len(patches)} contact patches, expected {len(exp)}. "
            "The sole geometry changed; re-check the pads before training.")
        assert abs(z - EXPECTED_SOLE_Z[side]) < 1e-3, (
            f"{side} sole plane moved: {z:.4f} vs {EXPECTED_SOLE_Z[side]:.4f}")
        for i, (got, want) in enumerate(zip(patches, exp)):
            for k in ("x", "y"):
                for j in (0, 1):
                    assert abs(got[k][j] - want[k][j]) < 1e-3, (
                        f"{side} foot patch {i+1} {k}{j} moved: "
                        f"{got[k][j]:.4f} vs expected {want[k][j]:.4f}. "
                        "The sole geometry changed; re-check the pads before training.")

        # One pad per (patch, y-half): heel-y, heel+y, toe-y, toe+y = foot1..4.
        # Splitting in y lets the pair represent roll about the foot's long axis.
        pads = []
        for p in patches:
            x0, x1 = p["x"]
            y0, y1 = p["y"]
            hx, hy = (x1 - x0) / 2, (y1 - y0) / 4
            for cy in (y0 + hy, y1 - hy):
                pads.append((((x0 + x1) / 2, cy, z + PAD_HALF_Z), (hx, hy, PAD_HALF_Z)))

        xlo = min(p["x"][0] for p in patches)
        xhi = max(p["x"][1] for p in patches)
        out[side] = dict(pads=pads, site=((xlo + xhi) / 2, 0.0, z),
                         patches=patches, z=z)
    return out


# ============================================================================
# main
# ============================================================================
POSES = {
    # exactly the URDF zero pose (forearms point forward)
    "stand": {},
    # same, with the elbows folded so the arms hang naturally at the sides
    "home": {"left_elbow_joint": 1.5708, "right_elbow_joint": 1.5708,
             "left_shoulder_roll_joint": 0.12, "right_shoulder_roll_joint": -0.12},
}


def solve_keyframe_heights(path, L):
    """Set each keyframe's pelvis z so the soles just touch the floor."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    for k, (pname, pose) in enumerate(POSES.items()):
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        lo = np.inf
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            lo = min(lo, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        z = round(-lo, 5)
        old = f'<key name="{pname}" qpos="{vec(qpos_of(pose, 0), 9)}"'
        new = f'<key name="{pname}" qpos="{vec(qpos_of(pose, z), 9)}"'
        i = next(i for i, x in enumerate(L) if old in x)
        L[i] = L[i].replace(old, new)
        print(f"  keyframe {pname}: pelvis z = {z}")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="hv1v2",
                    help="which robot to generate (read at import time too)")
    ap.add_argument("--target", choices=("viewer", "robot"), default="viewer")
    ap.add_argument("--out", default=None, help="output path (default depends on target)")
    ap.add_argument("--meshdir", default=None, help="compiler meshdir (default depends on target)")
    ap.add_argument("--copy-meshes", action="store_true",
                    help="symlink the referenced STLs next to the output")
    args = ap.parse_args()

    if args.target == "viewer":
        out = args.out or CFG["viewer_out"]
        meshdir = args.meshdir or CFG["viewer_meshdir"]
        L = build("viewer", meshdir, poses=POSES)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write("\n".join(L) + "\n")
        L = solve_keyframe_heights(out, L)
        open(out, "w").write("\n".join(L) + "\n")
        print(f"wrote {out}  ({len(L)} lines)")
        return

    # robot target: measure the feet off a throwaway viewer model, because the
    # robot model deliberately has no foot hull to measure.
    out = args.out or CFG["robot_out"]
    meshdir = args.meshdir or "meshes"
    tmpdir = os.path.join(ROOT, "mjcf")
    fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="_measure_", dir=tmpdir)
    os.close(fd)
    try:
        open(tmp, "w").write("\n".join(build("viewer", CFG["viewer_meshdir"], poses=POSES)) + "\n")
        pads = measure_feet(tmp)
    finally:
        os.unlink(tmp)

    for side, p in pads.items():
        names = ("heel", "toe")
        print(f"  {side} sole plane z {p['z']:+.4f}, {len(p['pads'])} pads:")
        for nm, q in zip(names, p["patches"]):
            print(f"    {nm:4s} x {q['x'][0]:+.4f}..{q['x'][1]:+.4f}  "
                  f"y {q['y'][0]:+.4f}..{q['y'][1]:+.4f}  "
                  f"({1000*(q['x'][1]-q['x'][0]):.0f} x {1000*(q['y'][1]-q['y'][0]):.0f} mm)")

    L = build("robot", meshdir, pads=pads)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")

    if args.copy_meshes:
        mdir = os.path.join(os.path.dirname(out), meshdir)
        os.makedirs(mdir, exist_ok=True)
        n = 0
        for name in links:
            if "mesh" not in links[name]:
                continue
            fn, _ = mesh_file(name)
            dst = os.path.join(mdir, fn)
            if not os.path.exists(dst):
                os.symlink(os.path.join(MESHES, fn), dst)
                n += 1
        print(f"  linked {n} new meshes into {mdir}")

    print(f"wrote {out}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
