import math

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    HOP_MIN_AIR_TIME,
    MIDAIR_VX_RANGE,
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


def test_hop_rough_variant_not_offered():
    """No Rough task id is registered for hop yet (flat only, unlike
    roulade's siblings) -- this test documents that scope choice."""
    import inspect

    sig = inspect.signature(make_microduck_hop_env_cfg)
    assert "rough" not in sig.parameters
