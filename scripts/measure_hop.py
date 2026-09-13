#!/usr/bin/env python3
"""Phase 1 measurement harness for the forward-hop env (CPU, no GPU needed).

AGENTS.md step 2 — "verify physics assumptions in sim BEFORE training" — was
skipped when the hop env was authored, because the authoring sandbox had no
CUDA device to run mjlab's MuJoCo-Warp step. That was a false constraint: the
measurements the hop actually needs are kinematic and ballistic, and CPU
MuJoCo runs them fine. This script is that missing step.

Four subcommands, in the order the plan uses them:

  settle   is the standing spawn a stable equilibrium? (tilt, not just z —
           a settle test that only records z reports fallen states as
           "resting fine")
  heights  exact-kinematics trunk heights: STAND_Z for THIS model, the crouch
           floor, and the extension travel available between them
  pushoff  what takeoff velocity a HAND-DESIGNED OPEN-LOOP push-off produces.
           A hop is ballistic once both feet leave, so air time, apex and
           takeoff velocity are one number wearing three hats (T = 2*v_z0/g).
           Read the ceiling caveat below before quoting this number.
  ranges   derive the mid-air spawn constants from a target air time

CEILING CAVEAT — READ BEFORE QUOTING `pushoff` (added 2026-09-13)
-----------------------------------------------------------------
`pushoff` was originally written as "THE GATING MEASUREMENT", on the theory
that if it could not reach TARGET_AIR_TIME then the target was impossible.
**That interpretation is retracted.** It sweeps hand-designed open-loop
profiles (crouch depth, counter-movement, ramp rate, toe-off, proximal-to-distal
sequencing) with the head and hip roll/yaw frozen at HOME, and in its
trunk-constrained variant it pins the trunk upright so it cannot topple. Its
best result is ~0.385 m/s of takeoff velocity, i.e. ~0.078 s of flight.

A trained policy beats it by 63%: the verified community jump policy
`ThomasBurgess2000/microduck-max-height-jump` reaches **0.628 m/s and 140 ms**
of air time on this robot, reproduced exactly this session. So `pushoff`
measures what a scripted extension can do, NOT what the robot can do. Treat its
output as a floor on capability and a diagnostic of the push geometry — never
as a physical ceiling, and never as grounds for declaring a target impossible.

What the push-off sweep DID establish, and which still stands: neither actuator
limit binds during the push (torque peaks at 0.43 Nm against a 1.068 Nm clamp,
joint speed at 7.3 rad/s against 22.4 rad/s no-load), so the binding constraint
on a hop is BALANCE DURING THE PUSH, not actuator power. Commanded open-loop,
the robot rotates about its toe instead of rising — tilt goes 0.4 -> 95 deg with
both feet still in contact and trunk z FALLING.

Everything is measured against the SAME physics training uses: the BAM M6
XL330 voltage-controlled actuator from scripts/infer_policy.py (its
load_bam_model / load_mujoco_with_bam are imported, not reimplemented, so the
BAM constants cannot drift from the ones tests/test_infer_policy_bam.py locks
to training). Measuring against the XML PD actuators instead would report a
push-off the trained policy can never reproduce.
"""

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# scene.xml = robot_groundcontact.xml (exactly what MICRODUCK_STANDUP_ROBOT_CFG
# compiles, which is the model the hop env runs on) + a floor plane. The robot
# spec on its own has no ground, so it cannot be used for contact measurements.
SCENE_XML = REPO / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene.xml"

# Training-matched CPU sim config, mirroring scripts/infer_policy.py: 5 ms
# physics, decimation 4 -> 50 Hz control, which is the rate policies are
# trained and deployed at. Measuring push-off at a different control rate
# would measure a different actuator bandwidth.
SIM_TIMESTEP = 0.005
DECIMATION = 4
CONTROL_DT = SIM_TIMESTEP * DECIMATION

G = 9.81

# Servo joint i lives at qpos[7 + i] (qpos[0:7] is the trunk freejoint) and is
# driven by ctrl[i] — true on the walk/groundcontact models, where ctrl idx ==
# joint idx. Asserted in _build() so a model revision cannot break it silently.
QPOS_SERVO_OFFSET = 7

FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

# Leg pitch chain, per leg, in (hip_pitch, knee, ankle) order. The hop's
# push-off lives entirely in these six joints.
_LEFT_CHAIN = ("left_hip_pitch", "left_knee", "left_ankle")
_RIGHT_CHAIN = ("right_hip_pitch", "right_knee", "right_ankle")


def _infer_policy():
    """Import scripts/infer_policy.py as a module.

    Same importlib pattern tests/test_infer_policy_bam.py uses — the file is a
    script, not an installed module, so it cannot simply be imported by name.
    """
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _home_ctrl(actuator_names):
    """Resolve HOME_FRAME's regex->angle map onto the actuator order.

    HOME_FRAME is the ground truth for the standing pose (the STAND2 frame the
    keyframe in scene.xml also carries); resolving it here rather than reading
    the keyframe means this script measures the pose the ENV spawns, not a
    scene file that could drift from it.

    Imported as a MODULE, not `from ... import HOME_FRAME`: importing the
    symbol trips a circular-import warning through the mjlab task registry.
    """
    from mjlab_microduck.robot import microduck_constants as mc

    out = np.zeros(len(actuator_names), dtype=float)
    for pattern, value in mc.HOME_FRAME.joint_pos.items():
        rx = re.compile(pattern)
        for i, name in enumerate(actuator_names):
            if rx.fullmatch(name) or rx.match(name):
                out[i] = float(value)
    return out


def _build(vin, vin_drop_gain=0.0):
    """Compile the scene with BAM actuators. Built once per voltage and reused."""
    ip = _infer_policy()
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, ip.BAM_MAX_CURRENT)
    model, data, bam_ctrl, names = ip.load_mujoco_with_bam(
        str(SCENE_XML), bam_model, SIM_TIMESTEP, vin_drop_gain, ip.BAM_VIN_MIN
    )
    assert model.nu == 14, f"expected 14 actuators, got {model.nu}"
    for i, name in enumerate(names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"actuator {name} has no joint of the same name"
        assert model.jnt_qposadr[jid] == QPOS_SERVO_OFFSET + i, (
            f"ctrl idx {i} ({name}) maps to qpos {model.jnt_qposadr[jid]}, "
            f"expected {QPOS_SERVO_OFFSET + i} — the ctrl-idx == joint-idx "
            f"assumption this script relies on no longer holds for this model"
        )
    return model, data, bam_ctrl, names


def _geom_ids(model):
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS
    )
    assert floor >= 0 and all(g >= 0 for g in feet)
    return floor, feet


def _robot_collision_geoms(model, floor_gid):
    return [
        g
        for g in range(model.ngeom)
        if g != floor_gid and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def _contacts(model, data, floor_gid, feet_gids):
    """(left_foot_down, right_foot_down, any_nonfoot_touching_floor).

    The third value is what makes "airborne" trustworthy. Testing only that
    both FEET are off the floor calls a robot lying on its trunk airborne —
    which is precisely the false-air-time exploit (butt-bounce) the env's own
    clean-time guard exists to close, and it silently produced 1.19 s "hops"
    in an earlier version of this script.
    """
    left = right = nonfoot = False
    for i in range(data.ncon):
        c = data.contact[i]
        if floor_gid not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == floor_gid else c.geom1
        if other == feet_gids[0]:
            left = True
        elif other == feet_gids[1]:
            right = True
        else:
            nonfoot = True
    return left, right, nonfoot


def _trunk_tilt(model, data):
    """Angle between the trunk's body-z axis and world up, in radians."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    zz = data.xmat[bid].reshape(3, 3)[2, 2]
    return math.acos(float(np.clip(zz, -1.0, 1.0)))


def _euler_quat(roll, pitch, yaw):
    """Mirrors the quaternion construction in mdp.reset_hop_state."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _min_vertex_z(model, data, geom_ids):
    """Lowest world-z over the actual mesh vertices of the given geoms.

    Exact kinematics, not a bounding-radius estimate: geom_rbound on a foot
    mesh overstates the drop by centimetres, which is exactly the scale of
    error AGENTS.md flags as having cost days on STAND_Z.
    """
    verts_all = np.asarray(model.mesh_vert).reshape(-1, 3)
    lo = math.inf
    for g in geom_ids:
        pos = data.geom_xpos[g]
        mat = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            did = model.geom_dataid[g]
            adr = model.mesh_vertadr[did]
            num = model.mesh_vertnum[did]
            world = verts_all[adr : adr + num] @ mat.T + pos
            lo = min(lo, float(world[:, 2].min()))
        else:
            lo = min(lo, float(pos[2] - model.geom_rbound[g]))
    return lo


def _leg_pose(home, names, p, q):
    """Leg pitch chain parameterised so the sole stays parallel to the trunk.

    The chain is (hip_pitch, knee, ankle) and their signed sum is the sole's
    pitch relative to the trunk. Setting left = (-p, -q, +(p+q)) and right the
    mirror makes that sum zero, so the foot stays flat on the floor at every
    depth and (p, q) sweeps a 2-D family of flat-footed poses.

    Direction, established by the `heights` sweep and NOT assumed: LARGER p and
    q are EXTENSION (taller). Max height is p~0.80, q~0.60 at z~0.1408 m;
    HOME sits at p~0.458, q~0.005 and z~0.1172 m, i.e. HOME is already a
    ~24 mm crouched stand, not the top of the stroke. An earlier version of
    this script assumed p=0 was a straight leg and swept q~0, which holds the
    knee straight and tilts the whole leg like a compass instead of squatting —
    it reported an 8 mm stroke where the real one is 97 mm.

    Head and hip yaw/roll stay at HOME: the hop is sagittal, and letting them
    move would confound the push-off with a balance strategy.
    """
    ctrl = home.copy()
    idx = {n: i for i, n in enumerate(names)}
    for (hp, kn, an), sign in ((_LEFT_CHAIN, -1.0), (_RIGHT_CHAIN, +1.0)):
        ctrl[idx[hp]] = sign * p
        ctrl[idx[kn]] = sign * q
        ctrl[idx[an]] = -sign * (p + q)
    return ctrl


def _joint_pose_ok(model, names, ctrl):
    """Does this pose respect every servo's joint limit?"""
    for i, n in enumerate(names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = model.jnt_range[jid]
        if model.jnt_limited[jid] and not (lo - 1e-9 <= ctrl[i] <= hi + 1e-9):
            return False
    return True


def _height_grid(model, data, names, home, floor_gid, robot_geoms,
                 p_range=(-1.2, 1.4), q_range=(-1.4, 1.4), step=0.05):
    """Kinematic trunk height over the feasible (p, q) family. Built once."""
    grid = {}
    for p in np.arange(p_range[0], p_range[1] + 1e-9, step):
        for q in np.arange(q_range[0], q_range[1] + 1e-9, step):
            if abs(p + q) > 1.5707:
                continue
            ctrl = _leg_pose(home, names, float(p), float(q))
            if not _joint_pose_ok(model, names, ctrl):
                continue
            grid[(round(float(p), 3), round(float(q), 3))] = _kinematic_trunk_z(
                model, data, ctrl, floor_gid, robot_geoms
            )
    return grid


def _pose_for_height(grid, target_z, toward):
    """Feasible (p, q) whose kinematic height is closest to target_z.

    Ties are broken toward `toward` (the extension pose), so the crouch chosen
    for a push-off lies on a coordinated path to the extension rather than in
    some kinematically-equivalent but contorted corner of the family.
    """
    pe, qe = toward
    best = min(
        grid.items(),
        key=lambda kv: (round(abs(kv[1] - target_z), 4),
                        (kv[0][0] - pe) ** 2 + (kv[0][1] - qe) ** 2),
    )
    return best[0], best[1]


def _kinematic_trunk_z(model, data, joint_ctrl, floor_gid, robot_geoms):
    """Trunk z when this joint pose is set down on the floor (no gravity, no load)."""
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, 0.5)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[QPOS_SERVO_OFFSET :] = joint_ctrl
    mujoco.mj_kinematics(model, data)
    return 0.5 - _min_vertex_z(model, data, robot_geoms)


def _settle_hold(model, data, bam_ctrl, target, steps):
    """Hold one ctrl target for `steps` control steps."""
    for _ in range(steps):
        bam_ctrl.q_target[:] = target
        for _ in range(DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)


# ── settle ────────────────────────────────────────────────────────────────────


def cmd_settle(args):
    """Is the hop's standing spawn a stable equilibrium?

    The hop spawns half its episodes standing (z in [0.11, 0.12], tilt +/-5 deg,
    HOME joints + N(0, 0.08) joint noise) and expects the policy to hop from
    there. If that state is not an equilibrium, those episodes start with a
    recovery the reward function never asked for, and every liftoff
    measurement taken from it is contaminated.
    """
    model, data, bam_ctrl, names = _build(args.vin, args.vin_drop_gain)
    floor_gid, feet_gids = _geom_ids(model)
    home = _home_ctrl(names)
    rng = np.random.default_rng(args.seed)
    steps = int(round(args.hold_s / CONTROL_DT))

    final_z, final_tilt, feet_down = [], [], []
    for _ in range(args.trials):
        mujoco.mj_resetData(model, data)
        z0 = rng.uniform(args.z_min, args.z_max)
        pitch = rng.uniform(-args.tilt_max, args.tilt_max)
        roll = rng.uniform(-args.tilt_max, args.tilt_max)
        yaw = rng.uniform(-math.pi, math.pi)
        data.qpos[0:3] = (0.0, 0.0, z0)
        data.qpos[3:7] = _euler_quat(roll, pitch, yaw)
        data.qpos[QPOS_SERVO_OFFSET :] = home + rng.normal(0.0, args.joint_noise, model.nu)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        bam_ctrl.reset(data.qpos)

        _settle_hold(model, data, bam_ctrl, home, steps)

        left, right, _ = _contacts(model, data, floor_gid, feet_gids)
        final_z.append(float(data.qpos[2]))
        final_tilt.append(_trunk_tilt(model, data))
        feet_down.append(left and right)

    final_z = np.array(final_z)
    tilt_deg = np.degrees(np.array(final_tilt))
    upright = tilt_deg <= args.upright_tilt_deg
    both_feet = np.array(feet_down)
    ok = upright & both_feet

    print(f"\n=== settle: {args.trials} noisy standing spawns, {args.hold_s:.1f} s "
          f"holding HOME ctrl @ {args.vin:.1f} V ===")
    print(f"spawn: z in [{args.z_min}, {args.z_max}] m, tilt +/-{math.degrees(args.tilt_max):.1f} deg, "
          f"joint noise sigma={args.joint_noise}")
    print(f"upright (tilt <= {args.upright_tilt_deg:.0f} deg):  "
          f"{upright.sum()}/{args.trials}  ({100*upright.mean():.1f}%)")
    print(f"both feet on the floor:       {both_feet.sum()}/{args.trials}  "
          f"({100*both_feet.mean():.1f}%)")
    print(f"upright AND both feet down:   {ok.sum()}/{args.trials}  ({100*ok.mean():.1f}%)")
    print(f"final tilt  deg: mean {tilt_deg.mean():6.2f}  sd {tilt_deg.std():5.2f}  "
          f"min {tilt_deg.min():5.2f}  max {tilt_deg.max():6.2f}")
    print(f"final trunk z  m: mean {final_z.mean():.4f}  sd {final_z.std():.4f}  "
          f"min {final_z.min():.4f}  max {final_z.max():.4f}")
    if ok.mean() >= 0.9:
        print("\nVERDICT: standing spawn IS a stable equilibrium.")
    else:
        print("\nVERDICT: the standing spawn is NOT a passive equilibrium.")
        print("Read this carefully before 'fixing' the spawn: the same open-loop")
        print("hold topples from EVERY pose in the flat-footed family (HOME is")
        print("merely the most stable, ~0.9 s to 10 deg vs 0.16-0.28 s elsewhere),")
        print("and it topples under the stock XML PD actuators too, slightly")
        print("FASTER than under BAM. So this is not a BAM artifact and not a bad")
        print("spawn height — this robot has no passively stable standing pose and")
        print("is balanced actively by the policy. What it does mean is that hop")
        print("episodes begin with the robot already falling forward, so the")
        print("policy must arrest a topple before it can hop, and any open-loop")
        print("measurement taken from a free-settled state is contaminated.")
    print(f"\nUnder-load settled height (use as the sag cross-check on STAND_Z): "
          f"{final_z[ok].mean() if ok.any() else float('nan'):.4f} m")
    return 0


# ── heights ───────────────────────────────────────────────────────────────────


def cmd_heights(args):
    """Exact-kinematics trunk heights, and the extension stroke available.

    STAND_Z = 0.115 in the hop cfg was INHERITED from the standup/roulade envs,
    not measured for this model. AGENTS.md: measure target heights off the
    actual robot in sim, never carry them across model revisions.

    Sweeps the full 2-D flat-footed pose family rather than one slice — the
    slice this script originally swept held the knee near-straight and
    under-reported the stroke by an order of magnitude.
    """
    model, data, bam_ctrl, names = _build(args.vin, args.vin_drop_gain)
    floor_gid, _ = _geom_ids(model)
    robot_geoms = _robot_collision_geoms(model, floor_gid)
    home = _home_ctrl(names)
    idx = {n: i for i, n in enumerate(names)}
    home_p = -home[idx["left_hip_pitch"]]
    home_q = -home[idx["left_knee"]]
    z_home = _kinematic_trunk_z(model, data, home, floor_gid, robot_geoms)

    grid = _height_grid(model, data, names, home, floor_gid, robot_geoms, step=args.step)
    (p_ext, q_ext), z_ext = max(grid.items(), key=lambda kv: kv[1])
    (p_cr, q_cr), z_cr = min(grid.items(), key=lambda kv: kv[1])

    print(f"\n=== heights: exact kinematics (mesh vertices) on the compiled model ===")
    print(f"collidable robot geoms: {len(robot_geoms)}   total mass: "
          f"{model.body_mass.sum():.6f} kg   weight: {model.body_mass.sum()*G:.4f} N")
    print(f"feasible flat-footed poses swept: {len(grid)} (step {args.step} rad)")

    print(f"\nHOME / STAND2 pose (p={home_p:.4f}, q={home_q:.4f}):")
    print(f"  kinematic trunk z .............. {z_home:.4f} m")
    print(f"  hop cfg STAND_Z = 0.1150 m ..... delta {1000*(z_home-0.115):+.1f} mm")
    print(f"\nfull extension (p={p_ext:+.2f}, q={q_ext:+.2f}): z = {z_ext:.4f} m")
    print(f"deepest crouch (p={p_cr:+.2f}, q={q_cr:+.2f}): z = {z_cr:.4f} m")
    print(f"\ntotal kinematic stroke ............ {1000*(z_ext - z_cr):.1f} mm")
    print(f"stroke available ABOVE HOME ....... {1000*(z_ext - z_home):.1f} mm")
    print(f"HOME is {1000*(z_ext - z_home):.1f} mm below full extension — it is "
          f"already a crouched stand, not the top of the stroke.")
    print(f"\nCross-check: docs/superpowers/specs/2026-08-04-roller-standup-design.md")
    print(f"measured 'debout 0.1407' on this robot; this sweep gives {z_ext:.4f} m.")

    print(f"\nheight vs extension along the HOME->max-extension direction:")
    print(f"  {'z [m]':>9} {'p':>7} {'q':>7} {'vs HOME [mm]':>13}")
    for target in np.arange(round(z_cr, 3), z_ext + 1e-9, args.report_step):
        (pp, qq), zz = _pose_for_height(grid, float(target), (p_ext, q_ext))
        print(f"  {zz:9.4f} {pp:7.2f} {qq:7.2f} {1000*(zz-z_home):+13.1f}")

    print("\nBallistic context — the stroke must be long enough to accelerate the")
    print("CoM to the takeoff speed each air time needs (constant-a approximation,")
    print("and the acceleration is what the legs must produce ON TOP of holding")
    print("the robot's own weight):")
    print(f"  {'T [s]':>6} {'v_z0 [m/s]':>11} {'apex rise [mm]':>15} "
          f"{'a over 50mm':>13} {'total leg force':>17}")
    for T in (0.08, 0.10, 0.12, 0.15, 0.20):
        v = G * T / 2
        stroke = args.assumed_stroke
        a = v * v / (2 * stroke)
        print(f"  {T:6.2f} {v:11.3f} {1000*v*v/(2*G):15.1f} {a:9.2f} m/s2 "
              f"{(a + G)/G:14.2f} x body weight")
    return 0


# ── pushoff ───────────────────────────────────────────────────────────────────


def _pushoff_trial(model, data, bam_ctrl, names, floor_gid, feet_gids,
                   crouch, extend, z_crouch, ramp_steps, settle_steps, flight_s,
                   dip=None, dip_steps=0):
    """One push-off attempt, measured as a capability upper bound.

    The robot is placed EXACTLY in the crouch pose, at rest, feet on the floor,
    and the extension is commanded immediately (settle_steps defaults to 0).
    That is deliberate: this robot has no passively stable pose (see `settle`),
    so any free-settle window before the push just measures a topple with a leg
    extension on top of it. Starting from the exact pose at zero velocity asks
    the one question that gates the whole task — CAN the legs produce this
    thrust — under best-case initial conditions. Whether a policy can get
    itself into the crouch is a separate, later question.

    Airborne means NO robot geom touches the floor, and the takeoff velocity is
    only recorded if the trunk is actually moving UP at that instant.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, z_crouch + 0.0005)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[QPOS_SERVO_OFFSET :] = crouch
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    bam_ctrl.reset(data.qpos)
    if settle_steps:
        _settle_hold(model, data, bam_ctrl, crouch, settle_steps)

    if dip is None:
        dip = crouch
    z_start = float(data.qpos[2])
    tilt_start = math.degrees(_trunk_tilt(model, data))

    air = 0.0
    max_air = 0.0
    apex = z_start
    v_peak = 0.0
    v_takeoff = 0.0
    took_off = False
    nonfoot_touch_on_ground = False
    n_steps = int(round(flight_s / CONTROL_DT))

    for k in range(n_steps):
        if k < dip_steps:
            # Counter-movement: drive DOWN first. A real jump loads the legs on
            # the way down and reverses; a pure ramp from rest cannot. Swept
            # because it is the obvious way a naive extension under-reports
            # what the actuators can do.
            bam_ctrl.q_target[:] = dip
            continue
        j = k - dip_steps
        a = 1.0 if ramp_steps <= 1 else min(1.0, (j + 1) / ramp_steps)
        bam_ctrl.q_target[:] = crouch + a * (extend - crouch)
        for _ in range(DECIMATION):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            l, r, nf = _contacts(model, data, floor_gid, feet_gids)
            vz = float(data.qvel[2])
            v_peak = max(v_peak, vz)
            airborne = (not l) and (not r) and (not nf)
            if nf:
                nonfoot_touch_on_ground = True
            if airborne:
                if air == 0.0 and vz > 0.0:
                    # Only a rising trunk counts as a takeoff; a descending one
                    # with nothing touching is a robot mid-topple.
                    v_takeoff = vz
                    took_off = True
                air += SIM_TIMESTEP
                apex = max(apex, float(data.qpos[2]))
            else:
                max_air = max(max_air, air)
                air = 0.0
    max_air = max(max_air, air)

    return {
        "z_start": z_start,
        "tilt_start_deg": tilt_start,
        "air_time": max_air if took_off else 0.0,
        "v_takeoff": v_takeoff,
        "v_peak": v_peak,
        "apex": apex,
        "apex_rise": apex - z_start,
        "took_off": took_off,
        "nonfoot_touch": nonfoot_touch_on_ground,
        "final_tilt_deg": math.degrees(_trunk_tilt(model, data)),
    }


def cmd_pushoff(args):
    """What air time a hand-designed open-loop push-off produces.

    NOT a capability ceiling — see the CEILING CAVEAT in the module docstring.
    A trained policy reaches 0.628 m/s / 140 ms on this robot, against this
    sweep's best of ~0.385 m/s / ~0.078 s. Use this to diagnose the push
    geometry and as a floor, never to declare a target impossible.

    TARGET_AIR_TIME = 0.15 s demands v_z0 = 0.736 m/s and ~0.20 J of vertical
    kinetic energy on 0.7372 kg.

    Swept at both ends of the training voltage DR range: a hop that only works
    at 8.2 V will not transfer to a robot on a sagging battery.

    The settle window before the push is deliberately SHORT. This robot has no
    passively stable standing pose (see `settle`): held open-loop it pitches
    forward through 10 deg in 0.2-0.9 s depending on depth, so a long settle
    would measure a topple with a leg extension on top of it rather than a
    push-off. Each trial reports the tilt at push-start so a contaminated
    trial is visible rather than silently averaged in.
    """
    print("\n=== pushoff: maximal coordinated leg extension under BAM ===")
    results = []
    for vin in args.vin_sweep:
        model, data, bam_ctrl, names = _build(vin, args.vin_drop_gain)
        floor_gid, feet_gids = _geom_ids(model)
        robot_geoms = _robot_collision_geoms(model, floor_gid)
        home = _home_ctrl(names)
        grid = _height_grid(model, data, names, home, floor_gid, robot_geoms, step=0.05)
        (p_ext, q_ext), z_ext = max(grid.items(), key=lambda kv: kv[1])
        extend = _leg_pose(home, names, p_ext, q_ext)
        settle_steps = int(round(args.settle_s / CONTROL_DT))

        # Crouch along a COORDINATED path: hold the hip at the extension value
        # and drive the knee down. That is how a jump actually works, and it
        # avoids the contorted equal-height poses a nearest-height search over
        # the whole 2-D family returns (q = -1.25 hyperextends the knee).
        crouches = []
        for q in np.arange(q_ext, -1.45, -args.crouch_step):
            ctrl = _leg_pose(home, names, p_ext, float(q))
            if abs(p_ext + q) > 1.5707 or not _joint_pose_ok(model, names, ctrl):
                continue
            z = _kinematic_trunk_z(model, data, ctrl, floor_gid, robot_geoms)
            crouches.append((float(q), z, ctrl))

        print(f"\n--- vin = {vin:.2f} V --- extension p={p_ext:+.2f} q={q_ext:+.2f} "
              f"(kinematic z {z_ext:.4f} m), settle {args.settle_s:.2f} s")
        print(f"  (rows with a real flight, or v_peak > {args.report_v} m/s)")
        print(f"  {'q_crouch':>9} {'crouch z':>9} {'stroke':>8} {'dip':>4} {'ramp':>5} "
              f"{'air [s]':>8} {'v_z0':>7} {'v_peak':>7} {'rise [mm]':>10}")
        for q_c, z_c, crouch in crouches:
          for dip_steps in args.dip_sweep:
            dip = _leg_pose(home, names, p_ext, max(q_c - args.dip_depth, -1.4))
            for ramp in args.ramp_sweep:
                r = _pushoff_trial(
                    model, data, bam_ctrl, names, floor_gid, feet_gids,
                    crouch, extend, z_c, ramp, settle_steps, args.flight_s,
                    dip=dip, dip_steps=dip_steps,
                )
                r.update(vin=vin, z_crouch=z_c, q_c=q_c, ramp=ramp, dip_steps=dip_steps,
                         stroke=z_ext - z_c)
                results.append(r)
                if r["air_time"] > 0 or r["v_peak"] > args.report_v:
                    print(f"  {q_c:9.2f} {z_c:9.4f} {1000*(z_ext-z_c):7.1f}mm "
                          f"{dip_steps:4d} {ramp:5d} {r['air_time']:8.4f} "
                          f"{r['v_takeoff']:7.3f} {r['v_peak']:7.3f} "
                          f"{1000*r['apex_rise']:10.1f}")

    print("\n=== summary ===")
    # air_time is only non-zero when the trunk was RISING with no robot geom
    # touching the floor, so it is already the trustworthy measure. Filtering
    # additionally on "no non-foot contact anywhere in the trial" would discard
    # genuine flights whose deep-crouch START rests on the body.
    for vin in args.vin_sweep:
        sub = [r for r in results if r["vin"] == vin]
        flights = [r for r in sub if r["air_time"] > 0]
        print(f"\nvin {vin:.2f} V:  {len(flights)}/{len(sub)} profiles produced a real flight")
        if flights:
            b = max(flights, key=lambda r: r["air_time"])
            print(f"  best air time: {b['air_time']:.4f} s  (crouch z={b['z_crouch']:.4f}, "
                  f"stroke {1000*b['stroke']:.1f} mm, dip={b['dip_steps']}, ramp={b['ramp']}, "
                  f"v_z0={b['v_takeoff']:.3f} m/s)")
            print(f"  ballistic check: 2*v_z0/g = {2*b['v_takeoff']/G:.4f} s vs "
                  f"measured {b['air_time']:.4f} s")
        vp = max((r["v_peak"] for r in sub), default=0.0)
        print(f"  peak upward trunk velocity over all profiles: {vp:.3f} m/s "
              f"-> a perfect takeoff at that speed would give {2*vp/G:.4f} s of flight")

    t_max = max((r["air_time"] for r in results), default=0.0)
    v_peak_all = max((r["v_peak"] for r in results), default=0.0)
    lo_v = min(args.vin_sweep)
    t_max_low = max((r["air_time"] for r in results if r["vin"] == lo_v), default=0.0)
    v_req = G * 0.15 / 2

    print(f"\nT_max (measured flight, any voltage) ...... {t_max:.4f} s")
    print(f"T_max (measured flight, at {lo_v:.1f} V) ......... {t_max_low:.4f} s")
    print(f"ceiling implied by peak upward velocity ... {2*v_peak_all/G:.4f} s "
          f"(v_peak {v_peak_all:.3f} m/s)")
    print(f"hop cfg TARGET_AIR_TIME ................... 0.1500 s "
          f"(needs v_z0 = {v_req:.3f} m/s)")
    print(f"hop cfg HOP_MIN_AIR_TIME .................. 0.0600 s "
          f"(needs v_z0 = {G*0.06/2:.3f} m/s)")
    if v_peak_all > 0:
        print(f"shortfall of THIS SWEEP vs TARGET_AIR_TIME  "
              f"{(v_req/v_peak_all)**2:.2f}x in energy")
    print(f"reference: a TRAINED policy reaches ....... 0.6275 m/s / 0.1400 s")

    print("\nCEILING CAVEAT — this is NOT a capability ceiling and must not be")
    print("used to call a target impossible. It sweeps hand-designed open-loop")
    print("profiles (crouch depth, counter-movement, ramp rate, toe-off,")
    print("proximal-to-distal sequencing) with the head and hip roll/yaw frozen.")
    print("A trained policy beats it by ~63% in takeoff velocity: the verified")
    print("microduck-max-height-jump policy reaches 0.628 m/s and 140 ms of air")
    print("time on this robot. Treat these numbers as a FLOOR on capability and")
    print("as a diagnostic of the push geometry.")
    print("\nWhat this sweep does establish: neither actuator limit binds during")
    print("the push (torque ~0.43 of 1.068 Nm, joint speed ~7.3 of 22.4 rad/s")
    print("no-load), so the binding constraint on a hop is BALANCE DURING THE")
    print("PUSH, not actuator power. Commanded open-loop the robot rotates about")
    print("its toe instead of rising.")
    print("\nDo NOT size TARGET_AIR_TIME down from these numbers — size it")
    print("against a policy-achievable figure (~0.14 s measured).")
    return 0


def cmd_ranges(args):
    """Derive the mid-air spawn constants from a target air time.

    Defect #5 in the plan: the shipped ranges (z 0.14-0.18 m while descending
    at 0.5-1.5 m/s) back-solve to hops of 0.175-0.383 s, 1.2x to 2.6x the
    0.15 s target — so half of all training experience has been teaching
    recovery from landings the target hop never produces. These formulas make
    the spawn the descent of the hop actually being trained.
    """
    T = args.air_time
    spread = args.spread
    T_hi = spread * T
    stand_z = args.stand_z

    apex_rise_hi = G * T_hi * T_hi / 8.0
    vz_touchdown_hi = G * T_hi / 2.0
    vx_max = args.forward_dist / T

    print(f"\n=== ranges: mid-air spawn derived from TARGET_AIR_TIME = {T:.4f} s ===")
    print(f"STAND_Z = {stand_z:.4f} m, spread = +/-{100*(spread-1):.0f}% of T, "
          f"TARGET_FORWARD_DIST = {args.forward_dist:.3f} m\n")
    print(f"  {'air time T':>11} {'v_z0 [m/s]':>11} {'apex rise [mm]':>15} {'apex z [m]':>11}")
    for t in (T / spread, T, T_hi):
        v = G * t / 2
        print(f"  {t:11.4f} {v:11.3f} {1000*v*v/(2*G):15.1f} {stand_z + v*v/(2*G):11.4f}")

    print(f"\nDerived constants for microduck_hop_env_cfg.py:")
    print(f"  TARGET_AIR_TIME   = {T:.3f}")
    print(f"  HOP_MIN_AIR_TIME  = {0.4*T:.3f}")
    print(f"  MIDAIR_Z_MIN      = {stand_z + 0.003:.3f}")
    print(f"  MIDAIR_Z_MAX      = {stand_z + apex_rise_hi:.3f}")
    print(f"  MIDAIR_VZ_RANGE   = ({-vz_touchdown_hi:.2f}, -0.20)")
    print(f"  MIDAIR_VX_RANGE   = (0.0, {vx_max:.2f})")
    print(f"\n(Phase 2 applies these; this subcommand only derives them.)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vin", type=float, default=8.2,
                    help="BAM battery voltage [V] (training DR range is 6.5-8.2)")
    ap.add_argument("--vin-drop-gain", type=float, default=0.0,
                    help="load-dependent voltage sag gain (training DR 0.0-0.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("settle", help="is the standing spawn a stable equilibrium?")
    s.add_argument("--trials", type=int, default=32)
    s.add_argument("--hold-s", type=float, default=3.0)
    s.add_argument("--z-min", type=float, default=0.11)
    s.add_argument("--z-max", type=float, default=0.12)
    s.add_argument("--tilt-max", type=float, default=math.radians(5.0))
    s.add_argument("--joint-noise", type=float, default=0.08)
    s.add_argument("--upright-tilt-deg", type=float, default=15.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_settle)

    h = sub.add_parser("heights", help="exact-kinematics trunk heights + extension travel")
    h.add_argument("--step", type=float, default=0.05, help="(p,q) grid step [rad]")
    h.add_argument("--report-step", type=float, default=0.005,
                   help="height rows to print [m]")
    h.add_argument("--assumed-stroke", type=float, default=0.05,
                   help="stroke [m] used for the force-requirement table")
    h.set_defaults(func=cmd_heights)

    p = sub.add_parser("pushoff", help="THE GATING MEASUREMENT: achievable air time")
    p.add_argument("--vin-sweep", type=float, nargs="+", default=[6.5, 8.2])
    p.add_argument("--crouch-step", type=float, default=0.2,
                   help="knee-angle step [rad] between crouch depths")
    p.add_argument("--ramp-sweep", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--dip-sweep", type=int, nargs="+", default=[0, 4, 8],
                   help="counter-movement duration [control steps]; 0 = none")
    p.add_argument("--dip-depth", type=float, default=0.4,
                   help="extra knee flexion [rad] during the counter-movement")
    p.add_argument("--report-v", type=float, default=0.25,
                   help="only print rows above this peak upward velocity [m/s]")
    p.add_argument("--settle-s", type=float, default=0.0)
    p.add_argument("--flight-s", type=float, default=1.2)
    p.set_defaults(func=cmd_pushoff)

    r = sub.add_parser("ranges", help="derive mid-air spawn constants from an air time")
    r.add_argument("--air-time", type=float, required=True)
    r.add_argument("--stand-z", type=float, default=0.115)
    r.add_argument("--forward-dist", type=float, default=0.05)
    r.add_argument("--spread", type=float, default=1.33)
    r.set_defaults(func=cmd_ranges)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
