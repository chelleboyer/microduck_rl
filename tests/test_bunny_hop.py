"""Mjlab-BunnyHop: the perpetual forward bunny hop.

Run 6's continuous hop fell in 55 of 59 distinct delay-matched 10 s rehearsal
rollouts: the 3 s episodic hop task had no fall termination and never lived
past 3 s. BunnyHop = long episodes + a fall termination + per-hop landing
judgement.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.mdp import _HOP_GATE_FULL_OPEN, _hop_state
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    BUNNY_EPISODE_LENGTH_S,
    BUNNY_FALL_TILT_DEG,
    BUNNY_STANDING_Z_MAX,
    DIRTY_LANDING_SCALE_STAGES,
    FORWARD_WEIGHT_STAGES,
    HOP_MIN_AIR_TIME,
    MIDAIR_VX_RANGE,
    MicroduckBunnyHopRlCfg,
    make_microduck_hop_env_cfg,
)

LANDING_TERMS = ("hop_landing_composite", "hop_upright_after_landing", "hop_height_after_landing")


def test_bunny_hop_is_a_long_episode_gait_that_ends_on_a_fall():
    cfg = make_microduck_hop_env_cfg(perpetual=True)
    assert cfg.episode_length_s == BUNNY_EPISODE_LENGTH_S >= 3 * 3.0
    fell = cfg.terminations["fell"]
    assert fell.params["limit_angle"] == pytest.approx(math.radians(BUNNY_FALL_TILT_DEG))
    assert not fell.time_out
    # the handover height the rehearsal starts at (trunk z 0.123-0.125) is covered
    assert cfg.events["set_hop_state"].params["standing_z_max"] == BUNNY_STANDING_Z_MAX >= 0.125


def test_bunny_hop_judges_each_landing_on_its_own():
    cfg = make_microduck_hop_env_cfg(perpetual=True)
    for name in LANDING_TERMS:
        assert cfg.rewards[name].params["rearm_on_flight"] is True
        assert cfg.rewards[name].params["dirty_landing_scale"] == DIRTY_LANDING_SCALE_STAGES[-1][1]
    assert cfg.metrics["clean_landing_rate"].params["rearm_on_flight"] is True


def test_bunny_hop_continues_run6_so_its_finished_curricula_are_collapsed():
    cfg = make_microduck_hop_env_cfg(perpetual=True)
    assert "forward_gate" not in cfg.curriculum and "dirty_landing_scale" not in cfg.curriculum
    assert cfg.rewards["hop_forward_progress"].weight == FORWARD_WEIGHT_STAGES[-1]
    assert cfg.events["set_hop_state"].params["midair_vx_range"] == MIDAIR_VX_RANGE


def test_the_episodic_hop_is_untouched_and_the_modes_are_exclusive():
    plain = make_microduck_hop_env_cfg()
    assert "fell" not in plain.terminations and plain.episode_length_s == 3.0
    for name in LANDING_TERMS:
        assert "rearm_on_flight" not in plain.rewards[name].params
    with pytest.raises(AssertionError):
        make_microduck_hop_env_cfg(once=True, perpetual=True)
    assert MicroduckBunnyHopRlCfg.experiment_name == "microduck_hop_bunny"


class _Env:
    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.common_step_counter = 0
        self.feet = SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 2)))
        self.nonfoot = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 13)))
        self.scene = SimpleNamespace(sensors={"feet_ground_contact": self.feet,
                                              "nonfoot_ground_contact": self.nonfoot})
        _hop_state(self)
        self._hop_max_air_time[:] = HOP_MIN_AIR_TIME * _HOP_GATE_FULL_OPEN


def _step(env, airborne=False, nonfoot=False, rearm=True) -> float:
    env.common_step_counter += 1
    env.feet.data.found = torch.zeros(1, 2) if airborne else torch.ones(1, 2)
    env.nonfoot.data.found = torch.full((1, 13), float(nonfoot))
    return float(microduck_mdp._hop_clean_landing(env, True, HOP_MIN_AIR_TIME, 0.0, rearm)[0])


def test_a_new_flight_rearms_the_landing_latch():
    env = _Env()
    assert _step(env) == 1.0                              # clean stance
    assert _step(env, nonfoot=True) == 0.0                # jaw brushes: dirty
    assert _step(env) == 0.0                              # still this landing
    assert _step(env, airborne=True) == 1.0               # next hop: fresh chance
    assert _step(env) == 1.0                              # clean landing pays


def test_without_rearm_the_latch_still_lasts_the_episode():
    env = _Env()
    _step(env, nonfoot=True, rearm=False)
    assert _step(env, airborne=True, rearm=False) == 0.0
    assert _step(env, rearm=False) == 0.0
