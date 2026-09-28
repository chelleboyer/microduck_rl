"""Mjlab-HopOnce: the two-foot forward hop, then stand (no second flight).

Run 6 (W&B 2qbd5bto) learned a continuous bunny hop because re-hopping was
free; HopOnce charges every airborne step after the first touchdown.
"""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.mdp import _HOP_GATE_FULL_OPEN, _hop_state
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    DIRTY_LANDING_SCALE_STAGES,
    FORWARD_WEIGHT_STAGES,
    HOP_MIN_AIR_TIME,
    MIDAIR_VX_RANGE,
    ONCE_EXTRA_FLIGHT_WEIGHT,
    MicroduckHopOnceRlCfg,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)

LANDING_TERMS = ("hop_landing_composite", "hop_upright_after_landing", "hop_height_after_landing")


def test_once_adds_the_cost_and_metric_and_nothing_else_changes_for_the_plain_hop():
    plain = make_microduck_hop_env_cfg()
    once = make_microduck_hop_env_cfg(once=True)
    assert "hop_extra_flight" not in plain.rewards and "extra_flight_rate" not in plain.metrics
    assert once.rewards["hop_extra_flight"].weight == ONCE_EXTRA_FLIGHT_WEIGHT < 0
    assert once.metrics["extra_flight_rate"].reduce == "last"
    assert set(once.rewards) - set(plain.rewards) == {"hop_extra_flight"}


def test_once_collapses_the_source_curricula_to_their_final_stage():
    # It warm-starts from run 6 (AGENTS.md: collapse the SOURCE's curricula).
    cfg = make_microduck_hop_env_cfg(once=True)
    assert "forward_gate" not in cfg.curriculum and "dirty_landing_scale" not in cfg.curriculum
    assert cfg.rewards["hop_forward_progress"].weight == FORWARD_WEIGHT_STAGES[-1]
    assert cfg.events["set_hop_state"].params["midair_vx_range"] == MIDAIR_VX_RANGE
    for name in LANDING_TERMS:
        assert cfg.rewards[name].params["dirty_landing_scale"] == DIRTY_LANDING_SCALE_STAGES[-1][1]


def test_once_has_its_own_experiment_and_keeps_the_mirror_loss():
    assert MicroduckHopOnceRlCfg.experiment_name == "microduck_hop_once"
    assert MicroduckHopRlCfg.experiment_name == "microduck_hop"
    assert MicroduckHopOnceRlCfg.algorithm.symmetry_cfg is not None


def test_the_extra_flight_cost_outweighs_standing_through_it():
    # landing composite pays up to weight 4 x score 1 per second of standing
    plain = make_microduck_hop_env_cfg()
    assert abs(ONCE_EXTRA_FLIGHT_WEIGHT) >= plain.rewards["hop_landing_composite"].weight


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


def _step(env, airborne: bool, gate_open: bool = True) -> float:
    env.common_step_counter += 1
    if gate_open:
        env._hop_max_air_time[:] = HOP_MIN_AIR_TIME * _HOP_GATE_FULL_OPEN
    env.feet.data.found = torch.zeros(1, 2) if airborne else torch.ones(1, 2)
    return float(microduck_mdp.hop_extra_flight_cost(env, HOP_MIN_AIR_TIME)[0])


@pytest.fixture(autouse=True)
def _no_accum(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_update_hop_accum", lambda env: None)


def test_the_first_hop_is_free_and_a_second_flight_is_charged_every_step():
    env = _Env()
    assert _step(env, airborne=False, gate_open=False) == 0.0     # standing, before the hop
    assert _step(env, airborne=True, gate_open=False) == 0.0      # the hop itself
    assert _step(env, airborne=True) == 0.0                       # gate opens mid-flight
    assert _step(env, airborne=False) == 0.0                      # first touchdown
    assert not bool(env._hop_extra_flight[0])
    assert [_step(env, airborne=True) for _ in range(3)] == [1.0, 1.0, 1.0]   # hops again
    assert _step(env, airborne=False) == 0.0
    assert bool(env._hop_extra_flight[0])
    assert float(microduck_mdp.hop_metric_extra_flight(env, HOP_MIN_AIR_TIME)[0]) == 1.0


def test_a_midair_spawn_is_judged_from_its_first_touchdown():
    env = _Env()
    assert _step(env, airborne=True) == 0.0                       # spawned in the air, gate pre-seeded
    assert _step(env, airborne=False) == 0.0                      # touchdown = the landing
    assert _step(env, airborne=True) == 1.0                       # any flight after it costs


def test_the_once_latch_is_step_guarded():
    env = _Env()
    _step(env, airborne=False)                                    # landed
    env.common_step_counter += 1
    env.feet.data.found = torch.zeros(1, 2)
    microduck_mdp._update_hop_landed_once(env, HOP_MIN_AIR_TIME)  # flags the flight
    env.feet.data.found = torch.ones(1, 2)
    env._hop_extra_flight[:] = False
    microduck_mdp._update_hop_landed_once(env, HOP_MIN_AIR_TIME)  # same step: no-op
    assert not bool(env._hop_extra_flight[0])


# ── v2: take-off terms pay only until the first landing ──────────────────────

from mjlab_microduck.tasks.microduck_hop_env_cfg import ONCE_FIRST_FLIGHT_ONLY_TERMS  # noqa: E402


def test_once_pays_take_off_only_for_the_first_flight():
    once = make_microduck_hop_env_cfg(once=True)
    plain = make_microduck_hop_env_cfg()
    bunny = make_microduck_hop_env_cfg(perpetual=True)
    assert set(ONCE_FIRST_FLIGHT_ONLY_TERMS) == {
        "hop_unweighting", "hop_launch_velocity", "hop_air_time", "hop_forward_progress"}
    for name in ONCE_FIRST_FLIGHT_ONLY_TERMS:
        assert once.rewards[name].params["first_flight_only"] is True, name
        # the hop and the bunny hop must keep paying every flight
        assert "first_flight_only" not in plain.rewards[name].params, name
        assert "first_flight_only" not in bunny.rewards[name].params, name


def test_the_first_landing_zeroes_take_off_pay_but_not_before():
    env = _Env()
    assert microduck_mdp._hop_before_first_landing(env, False) == 1.0   # disabled: bit-identical
    _step(env, airborne=True)                                          # the hop, gate open
    assert float(microduck_mdp._hop_before_first_landing(env, True)[0]) == 1.0
    _step(env, airborne=False)                                         # first touchdown
    assert float(microduck_mdp._hop_before_first_landing(env, True)[0]) == 0.0
    _step(env, airborne=True)                                          # a re-hop earns nothing
    assert float(microduck_mdp._hop_before_first_landing(env, True)[0]) == 0.0
