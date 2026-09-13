import math

import pytest

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    CROUCH_OVERRIDES,
    CROUCH_Z_RANGE,
    FOOT_BODIES,
    HOP_MIN_AIR_TIME,
    MIDAIR_VX_RANGE,
    NONFOOT_BODY_PATTERN,
    STAND_Z,
    TARGET_AIR_TIME,
    TARGET_LAUNCH_VZ,
    TARGET_FORWARD_DIST,
    make_microduck_hop_env_cfg,
)


def test_hop_cfg_builds():
    cfg = make_microduck_hop_env_cfg()
    assert "hop_air_time" in cfg.rewards
    assert "hop_landing_composite" in cfg.rewards
    assert "hop_unweighting" in cfg.rewards
    assert "hop_forward_progress" in cfg.rewards


def test_hop_cfg_play_variant_builds():
    cfg = make_microduck_hop_env_cfg(play=True)
    assert "hop_air_time" in cfg.rewards


def test_hop_reward_signs():
    """Penalty-style / self-negating terms must obey the sign convention:
    every Episode_Reward/<term> must be <= 0 in wandb (AGENTS.md)."""
    cfg = make_microduck_hop_env_cfg()
    r = cfg.rewards
    # self-negating (returns <= 0) -> POSITIVE weight
    assert r["hop_stand_tax"].weight > 0
    assert r["gentle_landing"].weight >= 0
    # ordinary cost (returns >= 0) -> NEGATIVE weight
    assert r["action_rate_l2"].weight < 0
    assert r["self_collisions"].weight < 0
    # ordinary reward -> POSITIVE weight
    assert r["hop_air_time"].weight > 0
    assert r["hop_landing_composite"].weight > 0
    assert r["hop_upright_after_landing"].weight > 0
    assert r["hop_height_after_landing"].weight > 0
    assert r["hop_unweighting"].weight > 0
    assert r["hop_forward_progress"].weight > 0
    # ordinary cost (returns >= 0), ramped from 0 by curriculum -> NEGATIVE weight
    assert r["hop_no_crawl"].weight <= 0


def test_hop_gate_params_consistent():
    cfg = make_microduck_hop_env_cfg()
    r = cfg.rewards
    for name in (
        "hop_landing_composite",
        "hop_upright_after_landing",
        "hop_height_after_landing",
        "hop_stand_tax",
    ):
        assert r[name].params["min_air_time"] == HOP_MIN_AIR_TIME
    assert r["hop_air_time"].params["target_air_time"] == TARGET_AIR_TIME
    assert TARGET_AIR_TIME > HOP_MIN_AIR_TIME  # gate must open before full credit


def test_hop_landing_target_height_matches_standing():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["hop_landing_composite"].params["target_height"] == STAND_Z
    assert cfg.rewards["hop_stand_tax"].params["target_height"] == STAND_Z


def test_hop_no_fell_over_termination():
    """Landing badly and recovering is in-scope for this task, same as
    roulade/velstand — falling must not end the episode early."""
    cfg = make_microduck_hop_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_hop_command_slots_zero_padded():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6


def test_hop_spawn_mix_curriculum_never_zeroes_midair():
    cfg = make_microduck_hop_env_cfg()
    stages = cfg.curriculum["hop_spawn_mix"].params["param_stages"]
    for stage in stages:
        assert stage["params"]["midair_prob"] > 0.0


def test_hop_forward_target_positive_and_gate_params_consistent():
    cfg = make_microduck_hop_env_cfg()
    assert TARGET_FORWARD_DIST > 0.0
    assert cfg.rewards["hop_forward_progress"].params["target_distance"] == TARGET_FORWARD_DIST


def test_hop_midair_spawn_carries_forward_momentum():
    """Reverse-curriculum landings must practice recovering under forward
    speed, not a dead stop, or the trained recovery won't match a real
    forward hop's landing (see midair_vx_range docstring in mdp.py)."""
    cfg = make_microduck_hop_env_cfg()
    assert cfg.events["set_hop_state"].params["midair_vx_range"] == MIDAIR_VX_RANGE
    assert MIDAIR_VX_RANGE[1] > 0.0


def test_hop_ground_sensor_registered():
    """Both ground-cheat fixes (butt-bounce taint, no-crawl penalty) read
    this sensor; without it nothing distinguishes a leg-driven liftoff from
    a ground push-off, and both satisfy the feet-only air-time gate."""
    cfg = make_microduck_hop_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "nonfoot_ground_contact" in sensor_names
    sensor = next(s for s in cfg.scene.sensors if s.name == "nonfoot_ground_contact")
    assert sensor.primary.mode == "body"
    assert sensor.primary.pattern == NONFOOT_BODY_PATTERN


def test_ground_sensor_covers_every_nonfoot_collision_body():
    """THE regression test for the inert-sensor bug (2026-09-12).

    The first version of this sensor matched `trunk_base` alone. A CPU drop
    test showed a belly-down robot rests on `hip_l`/`hip_l_2`/`jaw_soft` and
    never on `trunk_base`, so the sensor never fired: the butt-bounce taint
    and the no-crawl penalty were both silently dead code, and
    `Episode_Reward/hop_no_crawl` read exactly 0.0000 for a whole training
    run while the video showed the robot prone throughout.

    The invariant that catches it: EVERY body owning a geom that can collide
    with the floor must either be a foot, or be matched by the sensor's
    pattern. Checked against the compiled model, so a model revision that
    adds a collidable body fails here instead of silently going unwatched.
    """
    import re

    import mujoco
    from mjlab.entity import Entity

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

    model = Entity(MICRODUCK_STANDUP_ROBOT_CFG).spec.compile()
    rx = re.compile(NONFOOT_BODY_PATTERN)

    # The floor is a plain contype=1/conaffinity=1 plane, so a geom can hit
    # it iff it shares a bit with that. This is what excludes the trunk's
    # contype=2 self-collision-only geom, which never touches the ground.
    unwatched = set()
    for g in range(model.ngeom):
        hits_floor = (model.geom_contype[g] & 1) or (model.geom_conaffinity[g] & 1)
        if not hits_floor:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if body in FOOT_BODIES:
            continue  # feet are supposed to touch the ground
        if not rx.match(body):
            unwatched.add(body)

    assert not unwatched, (
        f"bodies can hit the floor but the ground sensor does not watch them: "
        f"{sorted(unwatched)} — the taint gate and no-crawl penalty will be "
        f"blind to any cheat that loads these parts."
    )


def test_ground_sensor_does_not_watch_the_feet():
    """Feet touching the ground is the whole point of landing — if the
    pattern matched them, the taint would fire on every normal landing and
    zero out all hop credit."""
    import re

    rx = re.compile(NONFOOT_BODY_PATTERN)
    for foot in FOOT_BODIES:
        assert not rx.match(foot), f"{foot} must not be watched by the ground sensor"


def test_hop_no_crawl_ramps_in_after_torque_and_landing_polish():
    """Anti-worm tax must ramp in strictly later than the other curriculum
    polish terms — a no-armed robot needs its trunk-drag recovery option
    free until some landing strategy has already been found, or it may
    never discover how to get up from a bad landing at all."""
    cfg = make_microduck_hop_env_cfg()
    no_crawl_stages = cfg.curriculum["no_crawl_weight"].params["weight_stages"]
    torque_stages = cfg.curriculum["torque_rate_weight"].params["weight_stages"]
    assert no_crawl_stages[0]["weight"] == 0.0
    first_nonzero_step = next(s["step"] for s in no_crawl_stages if s["weight"] != 0.0)
    last_torque_step = torque_stages[-1]["step"]
    assert first_nonzero_step > last_torque_step
    # every non-initial stage must be negative (ordinary cost -> negative weight)
    for stage in no_crawl_stages[1:]:
        assert stage["weight"] < 0


def test_hop_rough_variant_not_offered():
    """No Rough task id is registered for hop yet (flat only, unlike
    roulade's siblings) -- this test documents that scope choice."""
    import inspect

    sig = inspect.signature(make_microduck_hop_env_cfg)
    assert "rough" not in sig.parameters


def test_hop_midair_spawn_opens_the_landing_gate():
    """The reverse-curriculum spawn must land on a FULLY OPEN gate.

    Regression for a bug that silently disabled half the training signal:
    reset_hop_state seeded the air-time frontier at exactly
    gate_min_air_time, which is the smoothstep's ZERO point, so
    hop_landing_composite / _upright / _height / _stand_tax all evaluated to
    0 for every mid-air episode. The fall itself cannot rescue this — from
    MIDAIR_Z_MIN..MAX against STAND_Z there is only ~0.02-0.08 s of air
    before touchdown.
    """
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_hop_env_cfg()
    seed_factor = microduck_mdp._HOP_GATE_FULL_OPEN
    gate_min = cfg.events["set_hop_state"].params["gate_min_air_time"]
    assert gate_min == HOP_MIN_AIR_TIME

    env = SimpleNamespace(num_envs=1, device="cpu")
    microduck_mdp._hop_state(env)
    env._hop_max_air_time = torch.tensor([gate_min * seed_factor])
    gate = microduck_mdp._hop_completion_gate(env, gate_min)
    assert gate.item() == 1.0, f"mid-air spawn gate is {gate.item()}, not fully open"

    # ...and a standing spawn, which must EARN the gate, still starts shut.
    env._hop_max_air_time = torch.tensor([0.0])
    assert microduck_mdp._hop_completion_gate(env, gate_min).item() == 0.0


def _stub_env(num_envs=1, step_dt=0.02):
    """Minimal env stub for the hop bookkeeping helpers (no sensors, no sim)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        num_envs=num_envs, device="cpu", step_dt=step_dt,
        common_step_counter=0, scene=SimpleNamespace(sensors={}),
    )


def test_hop_clean_clock_blocks_a_butt_bounce_but_not_a_standing_hop():
    """Credit must be attributed per FLIGHT, not per episode.

    Regression for the sticky taint: one topple used to pin the air-time
    frontier at 0 for the rest of the episode, leaving only penalties live,
    so the argmax of the remaining ~2.5 s was "do nothing".
    """
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    W = microduck_mdp._HOP_CLEAN_LIFTOFF_S
    env = _stub_env()
    microduck_mdp._hop_state(env)

    # Just off the ground after a trunk push: not eligible, no budget.
    env._hop_clean_time = torch.tensor([0.02])
    assert not bool(microduck_mdp._hop_is_clean(env).item())
    assert microduck_mdp._hop_clean_air_budget(env).item() == 0.0

    # Standing cleanly for a while: eligible, and the budget does not bind a
    # hop of TARGET_AIR_TIME.
    env._hop_clean_time = torch.tensor([1.0])
    assert bool(microduck_mdp._hop_is_clean(env).item())
    assert microduck_mdp._hop_clean_air_budget(env).item() > TARGET_AIR_TIME

    # The clock recovers: a robot that gets back up re-earns eligibility,
    # which the sticky flag could never do.
    env._hop_clean_time = torch.tensor([0.0])
    for _ in range(int(W / env.step_dt) + 1):
        env.common_step_counter += 1
        microduck_mdp._update_hop_clean_time(env)
    assert bool(microduck_mdp._hop_is_clean(env).item())


def test_hop_clean_budget_caps_a_prone_feet_up_jackpot():
    """A robot prone on its back accrues real foot air-time the whole time.

    Eligibility alone would let it cash seconds of that as one jackpot the
    moment it broke nonfoot contact; the budget caps creditable air time to
    the interval that could actually contain a flight.
    """
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env = _stub_env()
    microduck_mdp._hop_state(env)
    W = microduck_mdp._HOP_CLEAN_LIFTOFF_S

    both_feet_air = torch.tensor([3.0])  # prone on its back for 3 s
    env._hop_clean_time = torch.tensor([W + 0.02])  # just became clean
    credited = torch.minimum(both_feet_air, microduck_mdp._hop_clean_air_budget(env))
    assert credited.item() < 0.03, f"prone robot cashed {credited.item():.3f}s of air time"
    assert microduck_mdp._hop_completion_gate(env, HOP_MIN_AIR_TIME).sum().item() == 0.0


class _FakeScene(dict):
    def __init__(self, robot, sensors):
        super().__init__({"robot": robot})
        self.sensors = sensors


def _forward_env(step_dt=0.02):
    """Env stub wired far enough to drive _update_hop_forward_accum."""
    import torch
    from types import SimpleNamespace

    found = torch.zeros(1, 2)
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros(1, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # yaw 0 -> +x
    ))
    sensors = {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=found))}
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=step_dt, common_step_counter=0,
        scene=_FakeScene(robot, sensors),
    )
    return env, robot, sensors


def test_hop_forward_credit_starts_at_liftoff_not_at_spawn():
    """Ground travel must not be cashed the instant both feet leave.

    Regression for the walk-then-stumble exploit: the launch reference used
    to be recorded at reset, so 8 cm of feet-only lunging accrued silently
    and paid out in full on the first airborne frames.
    """
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _forward_env()
    microduck_mdp._hop_state(env)
    env._hop_clean_time = torch.tensor([1.0])  # clean: only the feet have touched

    # Phase 1: walk forward 20 cm with a foot always down. No credit, and
    # crucially no launch reference laid down at the spawn point.
    sensors["feet_ground_contact"].data.found = torch.tensor([[1.0, 0.0]])
    for i in range(10):
        env.common_step_counter += 1
        robot.data.root_link_pos_w = torch.tensor([[0.02 * (i + 1), 0.0, 0.115]])
        microduck_mdp._update_hop_forward_accum(env)
    assert env._hop_max_forward_dist.item() == 0.0

    # Phase 2: both feet leave, and it travels 1 cm per step through the air.
    # The launch frame is latched on the FIRST airborne sample, so n airborne
    # samples credit (n-1) increments — displacement is measured from liftoff.
    sensors["feet_ground_contact"].data.found = torch.tensor([[0.0, 0.0]])
    for i in range(3):
        env.common_step_counter += 1
        robot.data.root_link_pos_w = torch.tensor([[0.20 + 0.01 * (i + 1), 0.0, 0.13]])
        microduck_mdp._update_hop_forward_accum(env)

    credited = env._hop_max_forward_dist.item()
    assert credited == pytest.approx(0.02, abs=1e-6), (
        f"credited {credited:.3f} m; the 0.20 m walked on the ground must not count"
    )
    assert credited < TARGET_FORWARD_DIST, "a stumble must not saturate the forward target"


def test_hop_forward_credit_blocked_while_not_clean():
    """A butt-bounce's airborne frames earn no forward distance."""
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _forward_env()
    microduck_mdp._hop_state(env)
    env._hop_clean_time = torch.tensor([0.0])  # trunk was just on the ground

    sensors["feet_ground_contact"].data.found = torch.tensor([[0.0, 0.0]])
    for i in range(5):
        env.common_step_counter += 1
        robot.data.root_link_pos_w = torch.tensor([[0.02 * (i + 1), 0.0, 0.13]])
        microduck_mdp._update_hop_forward_accum(env)
    assert env._hop_max_forward_dist.item() == 0.0


# ── AMENDMENT 1 ports: crouch spawn, launch velocity, airborne attitude, metrics ──


def test_hop_crouch_spawn_bucket_configured():
    """The push-off half of the maneuver must get reverse-curriculum support.

    Before this bucket existed, both spawn types practised LANDING (mid-air
    directly; standing only after a liftoff the policy could not yet do), so
    the measured-hard half — the push, where the robot rotates about its toe
    instead of rising — got no on-policy data at its frontier at all.
    """
    cfg = make_microduck_hop_env_cfg()
    p = cfg.events["set_hop_state"].params
    assert p["crouch_prob"] > 0.0
    assert p["crouch_overrides"] == CROUCH_OVERRIDES
    assert (p["crouch_z_min"], p["crouch_z_max"]) == CROUCH_Z_RANGE
    # All three buckets present and normalisable.
    assert p["standing_prob"] + p["crouch_prob"] + p["midair_prob"] == pytest.approx(1.0)


def test_hop_crouch_pose_is_a_real_crouch_and_within_joint_limits():
    """Joint indices resolve on the ACTUAL model, and the pose is reachable.

    The crouch angles came from a policy trained on a different robot model,
    so both the index mapping and the joint limits are checked here rather
    than assumed. A silently out-of-limit target would be clamped by MuJoCo
    into a pose nobody chose.
    """
    import mujoco
    from mjlab.entity import Entity

    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

    entity = Entity(MICRODUCK_STANDUP_ROBOT_CFG)
    model = entity.spec.compile()
    servo_ids, servo_names = entity.find_joints(r"^(?!passive_).*")

    expected = {
        2: "left_hip_pitch", 3: "left_knee", 4: "left_ankle",
        11: "right_hip_pitch", 12: "right_knee", 13: "right_ankle",
    }
    for idx, name in expected.items():
        assert servo_names[idx] == name, (
            f"crouch override index {idx} is {servo_names[idx]}, not {name} — "
            f"the canonical 14-servo layout has changed"
        )
    assert set(CROUCH_OVERRIDES) == set(expected), (
        "the crouch should drive exactly the two leg pitch chains"
    )

    # Resolve the model joint id BY NAME. `servo_ids` are entity-local
    # indices and the model's joint array is offset by the trunk freejoint,
    # so indexing jnt_range with a servo id reads the wrong joint's limits
    # (it reports left_hip_pitch as +/-0.384 rad, which is left_hip_roll's).
    for idx, angle in CROUCH_OVERRIDES.items():
        name = servo_names[idx]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"{name} is not a joint on the compiled model"
        lo, hi = model.jnt_range[jid]
        assert lo <= angle <= hi, (
            f"crouch target {angle} for {name} is outside its joint limit "
            f"[{lo}, {hi}] — MuJoCo would clamp it into a pose nobody chose"
        )

    # Left and right pitch chains must be exact mirrors, or the crouch is a
    # twist and the symmetry mirror-loss is being fed an asymmetric spawn.
    for left, right in ((2, 11), (3, 12), (4, 13)):
        assert CROUCH_OVERRIDES[left] == -CROUCH_OVERRIDES[right]


def test_hop_crouch_height_is_below_standing():
    """MEASURED 0.0658 m resting height for this pose on this model.

    The band sits just above it so joint noise can't spawn the robot inside
    the floor, and well below STAND_Z so the bucket is a loaded crouch and
    not a second standing spawn.
    """
    assert CROUCH_Z_RANGE[0] < CROUCH_Z_RANGE[1] < STAND_Z
    assert CROUCH_Z_RANGE[0] >= 0.0658, "spawning below the resting height penetrates the floor"


def test_hop_spawn_mix_curriculum_keeps_every_bucket_alive():
    """Neither assist may go to zero, and standing must grow monotonically.

    Mid-air keeps recovery practised; crouch keeps the push frontier
    on-policy. Standing is the only bucket that matches deployment, so it
    has to take over — but never by abandoning either frontier.
    """
    cfg = make_microduck_hop_env_cfg()
    stages = cfg.curriculum["hop_spawn_mix"].params["param_stages"]
    standing = []
    for stage in stages:
        sp = stage["params"]
        assert sp["midair_prob"] > 0.0
        assert sp["crouch_prob"] > 0.0
        assert sp["standing_prob"] + sp["crouch_prob"] + sp["midair_prob"] == pytest.approx(1.0)
        standing.append(sp["standing_prob"])
    assert standing == sorted(standing)


def _airborne_env(step_dt=0.02):
    """Env stub wired for the airborne predicate and the launch-velocity term."""
    import torch
    from types import SimpleNamespace

    feet_found = torch.zeros(1, 2)
    nonfoot_found = torch.zeros(1, 1)
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.zeros(1, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_link_lin_vel_w=torch.zeros(1, 3),
    ))
    sensors = {
        "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=feet_found)),
        "nonfoot_ground_contact": SimpleNamespace(data=SimpleNamespace(found=nonfoot_found)),
    }
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=step_dt, common_step_counter=0,
        scene=_FakeScene(robot, sensors),
    )
    return env, robot, sensors


def test_hop_airborne_predicate_rejects_a_prone_robot():
    """"Airborne" must mean NO geom touching, not "both feet off".

    A robot lying on its trunk with its feet in the air passes a feet-only
    test — that is the butt-bounce, and it fooled both an earlier version of
    this env and the CPU measurement harness (which reported 1.19 s "hops").
    """
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _airborne_env()
    feet = sensors["feet_ground_contact"].data.found
    nonfoot = sensors["nonfoot_ground_contact"].data.found

    feet[:] = 0.0   # both feet off the ground
    nonfoot[:] = 1.0  # ...but the trunk is on it
    assert not bool(microduck_mdp._hop_airborne_now(env).item())

    nonfoot[:] = 0.0  # genuinely airborne
    assert bool(microduck_mdp._hop_airborne_now(env).item())

    feet[0, 0] = 1.0  # one foot back down
    assert not bool(microduck_mdp._hop_airborne_now(env).item())


def test_hop_airborne_tilt_penalty_is_a_cost_and_inert_on_the_ground():
    """Returns >= 0 (so it takes a NEGATIVE weight) and cannot tax a
    grounded or prone robot — which is why it is safe live from step 0."""
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _airborne_env()
    # 90 deg of pitch: heavily tilted.
    robot.data.root_link_quat_w = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])

    sensors["feet_ground_contact"].data.found[:] = 1.0
    assert microduck_mdp.hop_airborne_tilt_penalty(env).item() == 0.0

    sensors["feet_ground_contact"].data.found[:] = 0.0
    sensors["nonfoot_ground_contact"].data.found[:] = 1.0  # prone
    assert microduck_mdp.hop_airborne_tilt_penalty(env).item() == 0.0

    sensors["nonfoot_ground_contact"].data.found[:] = 0.0  # airborne and tilted
    assert microduck_mdp.hop_airborne_tilt_penalty(env).item() > 0.0


def test_hop_lateral_drift_penalty_leaves_the_forward_axis_free():
    """THE divergence from the jump policy.

    jump_horizontal_drift_cost penalises all horizontal speed, which is
    correct for a vertical jump and would fight hop_forward_progress
    directly. Only the component across the launch heading may be charged.
    """
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _airborne_env()
    microduck_mdp._hop_state(env)
    env._hop_launch_heading = torch.tensor([[1.0, 0.0]])  # launched along +x
    sensors["feet_ground_contact"].data.found[:] = 0.0
    sensors["nonfoot_ground_contact"].data.found[:] = 0.0

    robot.data.root_link_lin_vel_w = torch.tensor([[0.5, 0.0, 0.0]])  # pure forward
    assert microduck_mdp.hop_lateral_drift_penalty(env).item() == pytest.approx(0.0)

    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.5, 0.0]])  # pure sideways
    assert microduck_mdp.hop_lateral_drift_penalty(env).item() == pytest.approx(0.5)

    # Sign-agnostic: drifting the other way costs the same.
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, -0.5, 0.0]])
    assert microduck_mdp.hop_lateral_drift_penalty(env).item() == pytest.approx(0.5)


def test_hop_launch_velocity_frontier_only_advances_while_loaded():
    """The term must measure a PUSH, not a fall.

    Once both feet are off, upward velocity can only decrease, so an
    airborne robot must not be able to extend the frontier — otherwise a
    robot flung upward by anything at all would collect launch credit.
    """
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _airborne_env()
    microduck_mdp._hop_state(env)
    env._hop_clean_time = torch.tensor([1.0])  # long since clean -> eligible

    sensors["feet_ground_contact"].data.found[:] = 1.0  # loaded
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.3]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_launch_velocity_progress(env, target_velocity=0.6).item() > 0.0
    assert env._hop_max_launch_vz.item() == pytest.approx(0.3)

    # Airborne, and implausibly fast: the frontier must not move.
    sensors["feet_ground_contact"].data.found[:] = 0.0
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 5.0]])
    env.common_step_counter += 1
    microduck_mdp.hop_launch_velocity_progress(env, target_velocity=0.6)
    assert env._hop_max_launch_vz.item() == pytest.approx(0.3)


def test_hop_launch_velocity_is_blocked_while_not_clean():
    """A trunk-assisted launch earns nothing here, same as every other hop term."""
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot, sensors = _airborne_env()
    microduck_mdp._hop_state(env)
    env._hop_clean_time = torch.tensor([0.01])  # just bounced off the trunk

    sensors["feet_ground_contact"].data.found[:] = 1.0
    robot.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.5]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_launch_velocity_progress(env, target_velocity=0.6).item() == 0.0
    assert env._hop_max_launch_vz.item() == 0.0


def test_hop_metrics_report_physical_outcomes_and_never_pay():
    """cfg.metrics exists, reports episode verdicts, and feeds no reward.

    Episode_Reward/<term> logs the WEIGHTED value, so a term parked at
    weight 0 by a curriculum reads 0.0000 whatever the robot does — that is
    how the inert-sensor bug hid for a full run. These make run 3 readable
    either way.
    """
    cfg = make_microduck_hop_env_cfg()
    for name in (
        "valid_takeoff_rate",
        "max_air_time_s",
        "max_launch_velocity_mps",
        "max_com_rise_mm",
        "max_foot_rise_mm",
        "stable_landing_rate",
    ):
        assert name in cfg.metrics, f"missing metric {name}"
        # Episode verdicts, not per-step averages.
        assert cfg.metrics[name].reduce == "last"

    metric_funcs = {m.func for m in cfg.metrics.values()}
    reward_funcs = {r.func for r in cfg.rewards.values()}
    assert not (metric_funcs & reward_funcs), (
        "a metric function is also wired as a reward — metrics must never "
        "change what is optimised"
    )


def test_hop_new_reward_signs_follow_the_convention():
    """Every Episode_Reward/<penalty> must read <= 0 in wandb (AGENTS.md)."""
    cfg = make_microduck_hop_env_cfg()
    r = cfg.rewards
    # ordinary costs (return >= 0) -> NEGATIVE weight
    assert r["hop_airborne_tilt"].weight < 0
    assert r["hop_lateral_drift"].weight < 0
    # ordinary reward -> POSITIVE weight
    assert r["hop_launch_velocity"].weight > 0
    assert r["hop_launch_velocity"].params["target_velocity"] == TARGET_LAUNCH_VZ


def _metric_env(step_dt=0.02):
    """Env stub wired for the metric rise accumulators (trunk + foot sites)."""
    import torch
    from types import SimpleNamespace

    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.10]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        site_pos_w=torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.01]]]),
    ))
    env = SimpleNamespace(
        num_envs=1, device="cpu", step_dt=step_dt, common_step_counter=0,
        scene=_FakeScene(robot, {}),
    )
    env.scene.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
    return env, robot


def test_hop_rise_metrics_measure_from_the_episode_start_height():
    """Rises are offset-free by construction, and both references latch once.

    Measuring a rise rather than an absolute height is what lets these work
    with no model change: the left_foot/right_foot sites sit at the ankle
    frame, ~14 mm above and ~24 mm behind the sole, so their height above
    the floor is not a sole clearance — but the CHANGE in that height is
    exactly a rise.
    """
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot = _metric_env()
    microduck_mdp._hop_state(env)
    env._hop_ground_start = torch.tensor([True])
    sites = SimpleNamespace(site_ids=[0, 1])

    # First step latches the references; nothing has risen yet.
    env.common_step_counter += 1
    assert microduck_mdp.hop_metric_com_rise_mm(env, foot_site_cfg=sites).item() == 0.0
    assert microduck_mdp.hop_metric_foot_rise_mm(env, foot_site_cfg=sites).item() == 0.0

    # Rise 30 mm of trunk, 25 mm of foot.
    robot.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.13]])
    robot.data.site_pos_w = torch.tensor([[[0.0, 0.0, 0.035], [0.0, 0.0, 0.035]]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_metric_com_rise_mm(env, foot_site_cfg=sites).item() == pytest.approx(30.0)
    assert microduck_mdp.hop_metric_foot_rise_mm(env, foot_site_cfg=sites).item() == pytest.approx(25.0)

    # Frontiers are max-so-far: coming back down does not erase them.
    robot.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.09]])
    robot.data.site_pos_w = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_metric_com_rise_mm(env, foot_site_cfg=sites).item() == pytest.approx(30.0)


def test_hop_foot_rise_is_bilateral():
    """One foot up while the other stays down must score 0.

    Same both-feet-at-once requirement the air-time gate enforces, measured
    geometrically instead of through contacts.
    """
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot = _metric_env()
    microduck_mdp._hop_state(env)
    env._hop_ground_start = torch.tensor([True])
    sites = SimpleNamespace(site_ids=[0, 1])

    env.common_step_counter += 1
    microduck_mdp.hop_metric_foot_rise_mm(env, foot_site_cfg=sites)

    # Left foot lifts 40 mm, right foot stays put.
    robot.data.site_pos_w = torch.tensor([[[0.0, 0.0, 0.05], [0.0, 0.0, 0.01]]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_metric_foot_rise_mm(env, foot_site_cfg=sites).item() == 0.0


def test_hop_rise_metrics_zeroed_for_midair_spawns():
    """A mid-air spawn starts high and falls — reporting a rise it never
    made would inflate exactly the metric used to judge whether a hop
    happened at all."""
    import torch
    from types import SimpleNamespace
    from mjlab_microduck.tasks import mdp as microduck_mdp

    env, robot = _metric_env()
    microduck_mdp._hop_state(env)
    env._hop_ground_start = torch.tensor([False])  # spawned airborne
    sites = SimpleNamespace(site_ids=[0, 1])

    env.common_step_counter += 1
    microduck_mdp.hop_metric_com_rise_mm(env, foot_site_cfg=sites)
    robot.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.20]])
    env.common_step_counter += 1
    assert microduck_mdp.hop_metric_com_rise_mm(env, foot_site_cfg=sites).item() == 0.0
