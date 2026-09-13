import math

import pytest

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    FOOT_BODIES,
    HOP_MIN_AIR_TIME,
    MIDAIR_VX_RANGE,
    NONFOOT_BODY_PATTERN,
    STAND_Z,
    TARGET_AIR_TIME,
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
