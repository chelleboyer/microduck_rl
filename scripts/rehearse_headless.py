#!/usr/bin/env python3
"""Headless deployment rehearsal: run an ONNX policy through infer_policy.py's sim path.

`scripts/infer_policy.py` is the CPU MuJoCo deployment rehearsal (plain MuJoCo,
NOT the mjlab/warp training sim; BAM M6 actuators as in training), but its
main() always opens `mujoco.viewer.launch_passive` and a raw-tty TerminalInput,
so it cannot run in a container with no display / no tty. This script drives
EXACTLY the same simulation path, importing the pieces from infer_policy.py:

  - scene selection  : `--scene` else infer_policy.MICRODUCK_XML (main()'s default)
  - actuators        : load_bam_model + load_mujoco_with_bam (or --no-bam: XML position actuators)
  - policy           : infer_policy.PolicyInference(...) built with the arguments
                       main() passes for `--walking <onnx> --new-cmd-obs` (61-D obs)
  - initial pose     : main()'s freejoint/default-pose setup (trunk z = 0.125)
  - control loop     : policy.infer() + policy.apply_action(), decimation 4,
                       bam_ctrl.update() before every mujoco.mj_step (dt = 0.005)

What is dropped relative to main(): the viewer, keyboard, real-time sleeping,
CSV/record/debug/odometry extras. All command slots are zero (twist 0 via
set_vel_cmd(0, 0, 0); head/body slots are 0 because nothing presses a key).

Metrics are computed AFTER each control step (i.e. at the 50 Hz observation
instant, like a training contact sensor read at step end):
  - flight   : a run of >= 3 consecutive control steps (0.06 s) in which NO
               robot geom touches the `floor` geom
  - dirty    : a flight whose landing is followed, before the next flight
               starts (or the rollout ends), by any NON-foot robot geom touching
               the floor. Foot geoms = `left_foot_collision` / `right_foot_collision`
               (the only geoms on bodies ankle_left / ankle_right; the hop env's
               feet_ground sensor uses the same two names).

Video (--video OUT.mp4): offscreen mujoco.Renderer 640x480, tracking camera on
trunk_base, one frame per control step (50 fps), written with mediapy (ffmpeg from imageio-ffmpeg if no system ffmpeg). There is
no EGL in the Claude Code container; this combination is verified to work:

    cd /home/user/microduck_rl && MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x1024x24" \\
        uv run python scripts/rehearse_headless.py --policy policy.onnx --video out.mp4

Without --video no GL context is created, so plain `uv run python ...` works.
"""

import argparse
import math
import os
import sys

import numpy as np
import mujoco

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import infer_policy  # noqa: E402  (the CPU deployment rehearsal this script mirrors)

FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
MIN_FLIGHT_STEPS = 3          # 3 control steps = 0.06 s
UPRIGHT_COS = 0.9
DECIMATION = 4                # main(): `decimation = 4`
SIM_DT = 0.005                # main(): load_mujoco_with_bam(xml_path, bam_model, 0.005, ...)


def _default_scene():
    # main() resolves MICRODUCK_XML relative to the cwd (it is a relative path);
    # resolve it against the repo root so this works from any cwd.
    return os.path.join(_REPO_ROOT, infer_policy.MICRODUCK_XML)


def build_sim(policy_path, scene=None, no_bam=False, vin=7.4, vin_drop_gain=0.1,
              kp_fw=infer_policy.BAM_KP_FW, current_limit=0.0, delay=None):
    """Build (model, data, bam_ctrl, policy) exactly as infer_policy.main() does."""
    # Mirrors main() "# Parse delay arguments" (default: --delay absent -> no delay).
    delay_min_lag = delay_max_lag = 0
    if delay is not None:
        if len(delay) == 0:
            delay_min_lag, delay_max_lag = 1, 2
        elif len(delay) == 1:
            delay_min_lag = delay_max_lag = delay[0]
        elif len(delay) == 2:
            delay_min_lag, delay_max_lag = delay
        else:
            raise ValueError("--delay accepts 0, 1, or 2 arguments")

    # Mirrors main() "if args.scene: ... else: xml_path = MICRODUCK_XML" (no
    # roller / kick flags here) and the BAM / --no-bam model load that follows.
    xml_path = scene or _default_scene()
    print(f"Loading MuJoCo model from: {xml_path}")
    bam_ctrl = None
    if not no_bam:
        bam_model = infer_policy.load_bam_model(kp_fw, vin, current_limit)
        vdg = vin_drop_gain if vin_drop_gain > 0 else None
        model, data, bam_ctrl, _ = infer_policy.load_mujoco_with_bam(
            xml_path, bam_model, SIM_DT, vdg, infer_policy.BAM_VIN_MIN)
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.timestep = SIM_DT
        data = mujoco.MjData(model)
        print("Legacy MuJoCo position actuators (--no-bam): NOT the actuator the policy was trained with")
        # Mirrors main()'s "(--no-bam only) XL330 firmware current limit" block.
        if current_limit and current_limit > 0:
            from bam.model import load_model
            kt = load_model(motor_name="xl330", model="m6").kt.value
            model.actuator_forcerange[:, 0] = -kt * current_limit
            model.actuator_forcerange[:, 1] = kt * current_limit
            model.actuator_forcelimited[:] = 1

    # Mirrors main() "# Initialize policy": the arguments main() passes for
    # `--walking <onnx> --new-cmd-obs` with every other flag at its default.
    policy = infer_policy.PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=policy_path,
        action_scale=1.0,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=None,
        switch_threshold=0.05,
        use_projected_gravity=True,      # main(): not args.raw_accelerometer
        ground_pick_onnx_path=None,
        ground_pick_period=4.0,
        sit_onnx_path=None,
        new_cmd_obs=True,
        slope_onnx_path=None,
        sitstand_onnx_path=None,
        kick_left_onnx_path=None,
        kick_right_onnx_path=None,
        roulade_onnx_path=None,
        kick_duration=3.0,
        roulade_duration=2.0,
    )
    policy.set_vel_cmd(0.0, 0.0, 0.0)   # main(): policy.set_vel_cmd(args.lin_vel_x, ...) at defaults

    # Mirrors main() "# Set initial position to default pose" (non-roller branch).
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.125
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    if bam_ctrl is not None:
        bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return model, data, bam_ctrl, policy, xml_path


def _floor_contact_state(model, data, floor_id, robot_geom, foot_geom):
    """Return (any_robot_geom_on_floor, any_nonfoot_robot_geom_on_floor)."""
    any_robot = any_nonfoot = False
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        if g1 == floor_id:
            other = g2
        elif g2 == floor_id:
            other = g1
        else:
            continue
        if c.dist > 0:        # within margin but not touching
            continue
        if robot_geom[other]:
            any_robot = True
            if not foot_geom[other]:
                any_nonfoot = True
    return any_robot, any_nonfoot


# Training-matched delays (--match-training-delays). infer_policy.py's main()
# has none of these, so a policy trained with them sees a different sensor/
# actuator timing in the default rehearsal. Values mirror the training cfgs:
#   actuator: microduck_constants.py BAM cfg delay_min_lag=3 / delay_max_lag=6
#             PHYSICS substeps (5 ms each) — one lag per episode here;
#   joint_vel: microduck_hop_env_cfg.py delay_min_lag = delay_max_lag = 1 control step;
#   base_ang_vel / projected gravity: 0..1 control step, resampled every 64 steps.
TRAIN_ACT_DELAY_SUBSTEPS = (3, 6)
TRAIN_IMU_DELAY_STEPS = (0, 1)
TRAIN_IMU_DELAY_PERIOD = 64
OBS_ANG_VEL, OBS_GRAVITY = slice(0, 3), slice(3, 6)


def _delay_observations(policy, rng, n_joints):
    """Wrap policy.get_observations with training's per-term observation delays."""
    qvel = slice(6 + n_joints, 6 + 2 * n_joints)
    orig = policy.get_observations
    hist = []
    state = {"calls": 0, "imu_lag": 0}

    def delayed():
        raw = orig()
        hist.append(raw.copy())
        del hist[:-3]
        if state["calls"] % TRAIN_IMU_DELAY_PERIOD == 0:
            state["imu_lag"] = int(rng.integers(TRAIN_IMU_DELAY_STEPS[0], TRAIN_IMU_DELAY_STEPS[1] + 1))
        state["calls"] += 1
        out = raw.copy()
        prev = hist[-2] if len(hist) >= 2 else raw           # joint_vel: always 1 step late
        out[qvel] = prev[qvel]
        imu = hist[-1 - state["imu_lag"]] if len(hist) > state["imu_lag"] else raw
        out[OBS_ANG_VEL] = imu[OBS_ANG_VEL]
        out[OBS_GRAVITY] = imu[OBS_GRAVITY]
        return out

    policy.get_observations = delayed


def run_rehearsal(policy_path, seconds=6.0, scene=None, no_bam=False, video=None,
                  delay=None, match_training_delays=False, act_delay_substeps=None,
                  seed=None, return_traces=False, **bam_kwargs):
    model, data, bam_ctrl, policy, xml_path = build_sim(
        policy_path, scene=scene, no_bam=no_bam, delay=delay, **bam_kwargs)

    obs_size = int(policy.get_observations().size)
    act_lag = 0
    if match_training_delays:
        if delay:
            raise ValueError("--delay (control-step action lag) and --match-training-delays are exclusive")
        rng = np.random.default_rng(seed)
        act_lag = int(act_delay_substeps if act_delay_substeps is not None
                      else rng.integers(TRAIN_ACT_DELAY_SUBSTEPS[0], TRAIN_ACT_DELAY_SUBSTEPS[1] + 1))
        _delay_observations(policy, rng, policy.n_joints)
    pending = []      # (apply_at_substep, target positions) — substep-level actuator delay
    substep = 0
    trunk_id = policy.trunk_base_id
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = int(model.jnt_qposadr[freejoint_id])
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError(f"{xml_path} has no geom named 'floor'")
    # Robot geoms = every geom on a body under trunk_base (the robot's root).
    robot_geom = np.array([model.body_rootid[model.geom_bodyid[g]] == trunk_id
                           for g in range(model.ngeom)])
    foot_geom = np.zeros(model.ngeom, dtype=bool)
    for name in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"foot geom {name!r} missing from {xml_path}")
        foot_geom[gid] = True

    p0 = data.xpos[trunk_id].copy()
    x_axis = data.xmat[trunk_id].reshape(3, 3)[:, 0]
    heading = np.array([x_axis[0], x_axis[1], 0.0])
    heading /= np.linalg.norm(heading)

    renderer = cam = None
    frames = []
    if video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = trunk_id
        cam.distance = 0.7
        cam.azimuth = 120.0
        cam.elevation = -15.0

    control_dt = DECIMATION * model.opt.timestep
    n_steps = int(round(seconds / control_dt))
    heights, cos_tilts, airborne, nonfoot = [], [], [], []
    pitch, roll, vfwd, nf_bodies = [], [], [], []
    nan_step = None

    for step in range(n_steps):
        # Mirrors main()'s loop body with actual_dt = control_dt (no wall clock).
        policy.update_ground_pick_phase(control_dt)
        policy.update_behavior(control_dt)
        action = policy.infer()
        if not np.all(np.isfinite(action)):
            nan_step = step
            break
        if act_lag == 0:
            policy.apply_action(action)
        else:
            # Same target as apply_action() (new_cmd_obs: no head offset), but
            # it reaches the actuator act_lag PHYSICS substeps after issue.
            pending.append((substep + act_lag, policy.default_pose + action * policy.action_scale))
        for _ in range(DECIMATION):
            while pending and pending[0][0] <= substep:
                policy.set_position_targets(pending.pop(0)[1])
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)
            substep += 1
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            nan_step = step
            break

        heights.append(float(data.qpos[qpos_adr + 2]))
        cos_tilts.append(float(data.xmat[trunk_id][8]))   # body z-axis . world z
        on_floor, nf = _floor_contact_state(model, data, floor_id, robot_geom, foot_geom)
        airborne.append(not on_floor)
        nonfoot.append(nf)
        if return_traces:
            R = data.xmat[trunk_id].reshape(3, 3)
            # Body x = forward: pitch = nose-down positive, roll = right-side-down positive.
            pitch.append(float(math.degrees(math.asin(np.clip(-R[2, 0], -1, 1)))))
            roll.append(float(math.degrees(math.atan2(R[2, 1], R[2, 2]))))
            vfwd.append(float(np.dot(data.qvel[:3], heading)))
            touching = set()
            for c in data.contact[:data.ncon]:
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if o == floor_id and robot_geom[g] and not foot_geom[g]:
                        touching.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                       int(model.geom_bodyid[g])))
            nf_bodies.append(sorted(touching))

        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    # Flights: runs of airborne control steps >= MIN_FLIGHT_STEPS.
    flights = []   # (start, end_exclusive)
    s = None
    for i, a in enumerate(airborne + [False]):
        if a and s is None:
            s = i
        elif not a and s is not None:
            if i - s >= MIN_FLIGHT_STEPS:
                flights.append((s, i))
            s = None
    n_steps_done = len(airborne)
    dirty = 0
    landed = 0
    for j, (fs, fe) in enumerate(flights):
        if fe >= n_steps_done:
            continue   # still airborne at rollout end: no landing
        landed += 1
        window_end = flights[j + 1][0] if j + 1 < len(flights) else n_steps_done
        if any(nonfoot[fe:window_end]):
            dirty += 1

    p_end = data.xpos[trunk_id].copy()
    fwd = float(np.dot(p_end - p0, heading))
    sim_s = n_steps_done * control_dt
    h = np.array(heights) if heights else np.array([np.nan])
    ct = np.array(cos_tilts) if cos_tilts else np.array([np.nan])
    summary = dict(
        scene=xml_path, bam=not no_bam, obs_size=obs_size,
        sim_seconds=sim_s, control_steps=n_steps_done, nan=nan_step is not None,
        nan_step=nan_step,
        trunk_z_min=float(h.min()), trunk_z_median=float(np.median(h)), trunk_z_max=float(h.max()),
        max_tilt_deg=float(math.degrees(math.acos(np.clip(ct.min(), -1, 1)))),
        airborne_steps=int(sum(airborne)),
        flights=len(flights), landings=landed, dirty_landings=dirty,
        flight_durations_s=[(fe - fs) * control_dt for fs, fe in flights],
        forward_disp_m=fwd,
        forward_speed_mps=fwd / sim_s if sim_s > 0 else float("nan"),
        ended_upright=bool(ct[-1] >= UPRIGHT_COS), final_cos_tilt=float(ct[-1]),
        nonfoot_steps=int(sum(nonfoot)),
        training_delays=bool(match_training_delays), act_delay_substeps=act_lag,
    )
    if return_traces:
        summary["traces"] = dict(
            dt=control_dt, z=heights, cos_tilt=cos_tilts, pitch_deg=pitch, roll_deg=roll,
            v_forward=vfwd, airborne=airborne, nonfoot=nonfoot, nonfoot_bodies=nf_bodies,
            flights=flights)

    if renderer is not None:
        renderer.close()
        import mediapy
        try:   # no system ffmpeg in the container: use imageio-ffmpeg's bundled binary
            import imageio_ffmpeg
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pass
        mediapy.write_video(video, frames, fps=int(round(1.0 / control_dt)))
        summary["video"] = video
        summary["video_frames"] = len(frames)
    return summary


def print_summary(s):
    print("\n" + "=" * 72)
    print("Headless rehearsal summary")
    print("=" * 72)
    print(f"scene              : {s['scene']}")
    print(f"actuators          : {'BAM M6' if s['bam'] else 'XML position (--no-bam)'}")
    print(f"obs size           : {s['obs_size']}")
    print(f"sim seconds        : {s['sim_seconds']:.2f}   control steps: {s['control_steps']}")
    print(f"NaN                : {'YES at step ' + str(s['nan_step']) if s['nan'] else 'no'}")
    print(f"trunk z [m]        : min {s['trunk_z_min']:.4f}  median {s['trunk_z_median']:.4f}  "
          f"max {s['trunk_z_max']:.4f}")
    print(f"max tilt           : {s['max_tilt_deg']:.1f} deg")
    print(f"airborne steps     : {s['airborne_steps']}  (no robot geom on floor)")
    print(f"flights (>=0.06 s) : {s['flights']}  ({s['flights'] / max(s['sim_seconds'], 1e-9):.2f}/s), "
          f"landings {s['landings']}, with non-foot floor contact before next flight: "
          f"{s['dirty_landings']}")
    if s["flight_durations_s"]:
        d = np.array(s["flight_durations_s"])
        print(f"flight durations   : mean {d.mean():.3f} s, max {d.max():.3f} s")
    print(f"non-foot contact   : {s['nonfoot_steps']} control steps")
    print(f"forward disp       : {s['forward_disp_m']:+.3f} m along initial heading "
          f"({s['forward_speed_mps']:+.3f} m/s)")
    print(f"ended upright      : {s['ended_upright']} (final cos tilt {s['final_cos_tilt']:.3f})")
    if "video" in s:
        print(f"video              : {s['video']} ({s['video_frames']} frames)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", required=True, help="ONNX policy (61-D obs, new command layout)")
    p.add_argument("--scene", default=None,
                   help="Scene XML (default: infer_policy.MICRODUCK_XML = scene.xml, the "
                        "groundcontact model = MICRODUCK_STANDUP_ROBOT_CFG, the hop env's robot)")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--video", default=None, help="Write an mp4 (needs a GL context, see docstring)")
    p.add_argument("--no-bam", action="store_true", help="As infer_policy.py --no-bam")
    p.add_argument("--delay", type=int, nargs="*", default=None,
                   help="As infer_policy.py --delay (control-step action lag); default off, like main()")
    p.add_argument("--vin", type=float, default=7.4)
    p.add_argument("--vin-drop-gain", type=float, default=0.1)
    p.add_argument("--kp-fw", type=float, default=infer_policy.BAM_KP_FW)
    p.add_argument("--current-limit", type=float, default=0.0)
    p.add_argument("--match-training-delays", action="store_true",
                   help="Apply training's actuator delay (3-6 physics substeps) and observation "
                        "delays (joint_vel 1 step, IMU 0-1 step). Off by default, like main().")
    p.add_argument("--act-delay-substeps", type=int, default=None,
                   help="With --match-training-delays: fix the actuator lag instead of sampling 3..6")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1,
                   help="Run N rollouts (seeds seed..seed+N-1) and print an aggregate; no video")
    a = p.parse_args()
    kw = dict(seconds=a.seconds, scene=a.scene, no_bam=a.no_bam, delay=a.delay, vin=a.vin,
              vin_drop_gain=a.vin_drop_gain, kp_fw=a.kp_fw, current_limit=a.current_limit,
              match_training_delays=a.match_training_delays,
              act_delay_substeps=a.act_delay_substeps)
    if a.episodes > 1:
        runs = [run_rehearsal(a.policy, seed=a.seed + i, **kw) for i in range(a.episodes)]
        print_aggregate(runs)
        return 1 if any(r["nan"] for r in runs) else 0
    s = run_rehearsal(a.policy, video=a.video, seed=a.seed, **kw)
    print_summary(s)
    return 1 if s["nan"] else 0


def aggregate(runs):
    """Pool N rollouts: rates over total sim time, landing cleanliness over all landings."""
    t = sum(r["sim_seconds"] for r in runs)
    landings = sum(r["landings"] for r in runs)
    durs = [d for r in runs for d in r["flight_durations_s"]]
    return dict(
        rollouts=len(runs), sim_seconds=t, nan_rollouts=sum(r["nan"] for r in runs),
        flights_per_s=sum(r["flights"] for r in runs) / t if t else float("nan"),
        feet_only_landing_frac=(landings - sum(r["dirty_landings"] for r in runs)) / landings
        if landings else float("nan"),
        flight_s_median=float(np.median(durs)) if durs else float("nan"),
        forward_speed_mps=sum(r["forward_disp_m"] for r in runs) / t if t else float("nan"),
        ended_upright_frac=sum(r["ended_upright"] for r in runs) / len(runs),
        max_tilt_deg_worst=max(r["max_tilt_deg"] for r in runs),
        act_delay_substeps=sorted({r["act_delay_substeps"] for r in runs}),
    )


def print_aggregate(runs):
    print("\n" + "=" * 72)
    print(f"Headless rehearsal aggregate ({len(runs)} rollouts, "
          f"{'training delays' if runs[0]['training_delays'] else 'no delays'}, "
          f"{'BAM M6' if runs[0]['bam'] else 'XML position'})")
    print("=" * 72)
    for k, v in aggregate(runs).items():
        print(f"{k:<24s}: {v:.4g}" if isinstance(v, float) else f"{k:<24s}: {v}")


if __name__ == "__main__":
    sys.exit(main())
