"""Hopscotch one-foot landings on the hop env (landing="left" / "right").

The two-foot task (Mjlab-Hop-Flat-MicroDuck) must be untouched by the
landing parameter; the one-foot tasks gate the landing terms on
mdp.hop_landing_stance_factor, whose latch is checked here on a fake env.
"""

from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.mdp import _HOP_GATE_FULL_OPEN, _hop_state, hop_landing_stance_factor
from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    _LEG_JOINTS,
    _SUPPORT_LEG_JOINTS,
    HOP_MIN_AIR_TIME,
    MicroduckHopLeftRlCfg,
    MicroduckHopRightRlCfg,
    MicroduckHopRlCfg,
    make_microduck_hop_env_cfg,
)

GATED = ("hop_landing_composite", "hop_upright_after_landing", "hop_height_after_landing")
AIR, BOTH, LEFT, RIGHT = (0, 0), (1, 1), (1, 0), (0, 1)


# ── cfg ─────────────────────────────────────────────────────────────────────

def test_the_two_foot_task_is_unchanged_by_the_landing_parameter():
    default = make_microduck_hop_env_cfg()
    both = make_microduck_hop_env_cfg(landing="both")
    assert list(default.rewards) == list(both.rewards)
    for name in default.rewards:
        assert default.rewards[name].params == both.rewards[name].params, name
    for name in GATED:
        assert "stance" not in default.rewards[name].params, name
    assert "stance" not in default.metrics["stable_landing_rate"].params
    assert default.rewards["hop_landing_composite"].params["joint_indices"] == _LEG_JOINTS
    assert MicroduckHopRlCfg.experiment_name == "microduck_hop"


@pytest.mark.parametrize("landing", ["left", "right"])
def test_one_foot_tasks_gate_the_landing_on_the_named_foot(landing):
    cfg = make_microduck_hop_env_cfg(landing=landing)
    both = make_microduck_hop_env_cfg()
    assert list(cfg.rewards) == list(both.rewards)       # same terms, only the gating differs
    for name in GATED:
        assert cfg.rewards[name].params["stance"] == landing, name
    assert cfg.metrics["stable_landing_rate"].params["stance"] == landing
    # The other leg is meant to be up: the pose target is the support leg only.
    assert cfg.rewards["hop_landing_composite"].params["joint_indices"] == _SUPPORT_LEG_JOINTS[landing]
    # The take-off side is the two-foot hop's, untouched.
    for name in ("hop_unweighting", "hop_launch_velocity", "hop_air_time", "hop_forward_progress"):
        assert cfg.rewards[name].params == both.rewards[name].params, name


def test_mirror_loss_only_for_the_two_foot_hop():
    assert MicroduckHopRlCfg.algorithm.symmetry_cfg is not None
    assert MicroduckHopLeftRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHopRightRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHopLeftRlCfg.experiment_name == "microduck_hop_left"
    assert MicroduckHopRightRlCfg.experiment_name == "microduck_hop_right"


def test_support_legs_are_the_named_legs():
    assert _SUPPORT_LEG_JOINTS["left"] == [0, 1, 2, 3, 4]
    assert _SUPPORT_LEG_JOINTS["right"] == [9, 10, 11, 12, 13]


def test_landing_names_are_checked():
    with pytest.raises(AssertionError):
        make_microduck_hop_env_cfg(landing="sideways")


# ── the stance latch, on a fake env ─────────────────────────────────────────

class _Env:
    def __init__(self, n: int = 1):
        self.num_envs = n
        self.device = "cpu"
        self.common_step_counter = 0
        self.feet = SimpleNamespace(data=SimpleNamespace(found=torch.ones(n, 2)))
        self.scene = SimpleNamespace(sensors={"feet_ground_contact": self.feet})
        _hop_state(self)

    def open_gate(self):
        """As after a qualifying flight (or a mid-air spawn's pre-seed)."""
        self._hop_max_air_time[:] = HOP_MIN_AIR_TIME * _HOP_GATE_FULL_OPEN


def _step(env: _Env, feet, stance: str) -> float:
    env.common_step_counter += 1
    env.feet.data.found = torch.tensor([feet], dtype=torch.float32)
    return float(hop_landing_stance_factor(env, stance, HOP_MIN_AIR_TIME)[0])


def test_nothing_latches_before_a_flight():
    env = _Env()
    assert _step(env, LEFT, "left") == 0.0          # standing on the left foot is not a landing
    assert not bool(env._hop_td_latched[0])


@pytest.mark.parametrize("stance,good,bad", [("left", LEFT, RIGHT), ("right", RIGHT, LEFT)])
def test_a_clean_one_foot_landing_pays_while_held(stance, good, bad):
    env = _Env()
    env.open_gate()
    assert _step(env, AIR, stance) == 0.0           # still in the air: nothing latched
    assert not bool(env._hop_td_latched[0])
    assert _step(env, good, stance) == 1.0          # touchdown on the named foot
    assert bool(env._hop_td_latched[0]) and bool(env._hop_td_clean[0])
    assert _step(env, good, stance) == 1.0          # holding it
    assert _step(env, BOTH, stance) == 0.0          # other foot down: no pay this step
    assert _step(env, good, stance) == 1.0          # back on one foot: the latch still holds


@pytest.mark.parametrize("stance,wrong", [("left", RIGHT), ("left", BOTH),
                                          ("right", LEFT), ("right", BOTH)])
def test_landing_on_the_wrong_feet_latches_the_episode_at_zero(stance, wrong):
    env = _Env()
    env.open_gate()
    _step(env, AIR, stance)
    assert _step(env, wrong, stance) == 0.0
    good = LEFT if stance == "left" else RIGHT
    assert _step(env, good, stance) == 0.0          # shuffling onto the stance earns nothing


def test_the_latch_is_step_guarded():
    env = _Env()
    env.open_gate()
    env.common_step_counter += 1
    env.feet.data.found = torch.tensor([RIGHT], dtype=torch.float32)
    hop_landing_stance_factor(env, "left", HOP_MIN_AIR_TIME)      # latches messy this step
    env.feet.data.found = torch.tensor([LEFT], dtype=torch.float32)
    v = float(hop_landing_stance_factor(env, "left", HOP_MIN_AIR_TIME)[0])   # same step
    assert v == 0.0 and not bool(env._hop_td_clean[0])


def test_two_foot_task_multiplier_is_one():
    assert microduck_mdp._hop_stance(_Env(), None, HOP_MIN_AIR_TIME) == 1.0


def test_stance_names_are_checked():
    with pytest.raises(AssertionError):
        hop_landing_stance_factor(_Env(), "both", HOP_MIN_AIR_TIME)
