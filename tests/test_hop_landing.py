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
        # 13 non-foot bodies, as on the compiled groundcontact model.
        self.nonfoot = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(n, 13)))
        self.scene = SimpleNamespace(sensors={"feet_ground_contact": self.feet,
                                              "nonfoot_ground_contact": self.nonfoot})
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


# ── the clean-landing gate (run 3's dive-and-face-plant) ────────────────────

@pytest.mark.parametrize("landing", ["both", "left", "right"])
def test_every_landing_variant_requires_a_clean_landing(landing):
    cfg = make_microduck_hop_env_cfg(landing=landing)
    for name in GATED:
        assert cfg.rewards[name].params["require_clean_landing"] is True, name
    # Take-off terms are paid at liftoff and must not be touched by the gate:
    # a hop that lands badly has to keep beating not hopping at all.
    for name in ("hop_unweighting", "hop_launch_velocity", "hop_air_time", "hop_forward_progress"):
        assert "require_clean_landing" not in cfg.rewards[name].params, name
    assert "clean_landing_rate" in cfg.metrics
    assert cfg.metrics["clean_landing_rate"].reduce == "last"
    if landing == "both":
        assert "stance" not in cfg.metrics["clean_landing_rate"].params
    else:
        assert cfg.metrics["clean_landing_rate"].params["stance"] == landing


def _clean(env: _Env, nonfoot_touching: bool) -> float:
    env.common_step_counter += 1
    env.nonfoot.data.found = torch.zeros(1, 13)
    if nonfoot_touching:
        env.nonfoot.data.found[0, 1] = 1.0          # e.g. trunk_base / jaw_soft
    return float(microduck_mdp._hop_clean_landing(env, True, HOP_MIN_AIR_TIME))


def test_contact_before_any_flight_does_not_latch():
    # A pre-hop fall is the clean-time clock's business, not this latch's —
    # latching here would be the sticky taint that made "do nothing" win.
    env = _Env()
    assert _clean(env, True) == 1.0
    assert not bool(env._hop_landing_dirty[0])


def test_a_feet_only_landing_keeps_paying():
    env = _Env()
    env.open_gate()
    for _ in range(5):
        assert _clean(env, False) == 1.0


def test_a_dirty_landing_zeroes_the_rest_of_the_episode():
    env = _Env()
    env.open_gate()
    assert _clean(env, False) == 1.0                # flight
    assert _clean(env, True) == 0.0                 # chest / jaw hits the floor
    assert _clean(env, False) == 0.0                # standing back up earns nothing
    assert _clean(env, False) == 0.0


def test_the_clean_landing_latch_is_step_guarded():
    env = _Env()
    env.open_gate()
    env.common_step_counter += 1
    env.nonfoot.data.found = torch.ones(1, 13)
    microduck_mdp._update_hop_landing_clean(env, HOP_MIN_AIR_TIME)       # latches this step
    env.nonfoot.data.found = torch.zeros(1, 13)
    microduck_mdp._update_hop_landing_clean(env, HOP_MIN_AIR_TIME)       # same step: no-op
    assert bool(env._hop_landing_dirty[0])


def test_the_gate_off_is_bit_identical():
    env = _Env()
    env.open_gate()
    env.nonfoot.data.found = torch.ones(1, 13)
    assert microduck_mdp._hop_clean_landing(env, False, HOP_MIN_AIR_TIME) == 1.0
    assert not bool(env._hop_landing_dirty[0])      # not even evaluated


# ── run 5: a slope toward the clean landing (run 4's latch had none) ─────────

from mjlab_microduck.tasks.microduck_hop_env_cfg import (  # noqa: E402
    DIRTY_LANDING_SCALE_STAGES,
    LANDING_CONTACT_COST_WEIGHT,
)


def test_dirty_landings_keep_partial_pay_that_steps_down_to_zero():
    cfg = make_microduck_hop_env_cfg()
    for name in GATED:
        assert cfg.rewards[name].params["dirty_landing_scale"] == DIRTY_LANDING_SCALE_STAGES[0][1]
    stages = cfg.curriculum["dirty_landing_scale"].params["param_stages"]
    assert set(cfg.curriculum["dirty_landing_scale"].params["reward_names"]) == set(GATED)
    scales = [s["params"]["dirty_landing_scale"] for s in stages]
    steps = [s["step"] for s in stages]
    # Run 4: scale 0 from step 0 left no gradient after liftoff.
    assert scales[0] > 0.0
    assert scales == sorted(scales, reverse=True) and scales[-1] == 0.0
    assert steps == sorted(steps) and steps[0] == 0
    assert all(s % 24 == 0 for s in steps)            # env steps = iterations x 24


def test_landing_contact_cost_is_sized_so_hopping_still_beats_not_hopping():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["hop_landing_contact"].weight == LANDING_CONTACT_COST_WEIGHT < 0
    # Run 3's logged liftoff pay was ~3.3/episode (air 1.6 + forward 1.65); the
    # worst case costs ~0.83*|w| (lying down for the rest of the episode).
    assert 0.83 * abs(LANDING_CONTACT_COST_WEIGHT) < 3.3 / 1.5


def test_a_dirty_landing_keeps_the_scheduled_fraction():
    env = _Env()
    env.open_gate()
    env.common_step_counter += 1
    env.nonfoot.data.found = torch.ones(1, 13)
    assert float(microduck_mdp._hop_clean_landing(env, True, HOP_MIN_AIR_TIME, 0.5)[0]) == 0.5
    env.common_step_counter += 1
    env.nonfoot.data.found = torch.zeros(1, 13)
    assert float(microduck_mdp._hop_clean_landing(env, True, HOP_MIN_AIR_TIME, 0.25)[0]) == 0.25


def _contact_cost(env: _Env, touching: bool) -> float:
    env.common_step_counter += 1
    env.nonfoot.data.found = torch.full((1, 13), float(touching))
    return float(microduck_mdp.hop_landing_contact_cost(env, HOP_MIN_AIR_TIME)[0])


def test_contact_cost_is_dense_after_liftoff_and_silent_before(monkeypatch):
    monkeypatch.setattr(microduck_mdp, "_update_hop_accum", lambda env: None)
    env = _Env()
    assert _contact_cost(env, True) == 0.0            # a pre-hop fall is not charged
    env.open_gate()
    assert _contact_cost(env, False) == 0.0           # flight / feet-only landing: free
    assert [_contact_cost(env, True) for _ in range(3)] == [1.0, 1.0, 1.0]   # every step
    assert _contact_cost(env, False) == 0.0           # back on its feet: stops charging


def test_reward_param_curriculum_moves_every_named_term_together():
    terms = {n: SimpleNamespace(params={"dirty_landing_scale": 0.5}) for n in GATED}
    env = SimpleNamespace(common_step_counter=0,
                          reward_manager=SimpleNamespace(get_term_cfg=terms.__getitem__))
    stages = [{"step": it * 24, "params": {"dirty_landing_scale": s}} for it, s in DIRTY_LANDING_SCALE_STAGES]
    for it, scale in DIRTY_LANDING_SCALE_STAGES:
        env.common_step_counter = it * 24
        v = microduck_mdp.reward_param_curriculum(env, None, list(GATED), stages)
        assert float(v) == scale
        assert all(t.params["dirty_landing_scale"] == scale for t in terms.values())


# ── run 6: vertical first, forward only on measured clean landings ───────────

from mjlab_microduck.tasks.microduck_hop_env_cfg import (  # noqa: E402
    FORWARD_GATE_CLEAN_THRESHOLD,
    FORWARD_GATE_MIN_DWELL_ITERS,
    FORWARD_WEIGHT_STAGES,
    MIDAIR_VX_RANGE,
)

_FWD = SimpleNamespace(weight=0.0)
_SPAWN = SimpleNamespace(params={"midair_vx_range": (0.0, 0.0)})


def _gate_env(n: int = 8) -> _Env:
    env = _Env(n)
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: _FWD)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: _SPAWN)
    return env


def _gate(env, clean: bool, flew: bool = True, dwell=FORWARD_GATE_MIN_DWELL_ITERS * 24, alpha=1e-4):
    env._hop_max_air_time[:] = HOP_MIN_AIR_TIME * (2.0 if flew else 0.0)
    env._hop_landing_dirty[:] = not clean
    return float(microduck_mdp.hop_forward_gate_curriculum(
        env, torch.arange(env.num_envs), "hop_forward_progress", list(FORWARD_WEIGHT_STAGES),
        "set_hop_state", MIDAIR_VX_RANGE, FORWARD_GATE_CLEAN_THRESHOLD, dwell,
        HOP_MIN_AIR_TIME, alpha))


def test_forward_objective_starts_closed_in_the_cfg():
    cfg = make_microduck_hop_env_cfg()
    assert cfg.rewards["hop_forward_progress"].weight == FORWARD_WEIGHT_STAGES[0] == 0.0
    assert cfg.events["set_hop_state"].params["midair_vx_range"] == (0.0, 0.0)
    p = cfg.curriculum["forward_gate"].params
    assert p["weight_stages"][-1] == 5.0                  # the objective run 3 trained with
    assert p["midair_vx_range"] == MIDAIR_VX_RANGE
    assert p["event_name"] in cfg.events and p["reward_name"] in cfg.rewards


def test_dirty_landings_and_no_flight_never_open_the_gate():
    env = _gate_env()
    for _ in range(2000):
        env.common_step_counter += 24
        assert _gate(env, clean=False) == 0.0             # the dive
        assert _gate(env, clean=True, flew=False) == 0.0  # never took off
    assert env._hop_clean_ema == 0.0 and _SPAWN.params["midair_vx_range"] == (0.0, 0.0)


def test_clean_landings_open_one_stage_per_dwell_and_it_never_goes_back():
    env = _gate_env()
    dwell = FORWARD_GATE_MIN_DWELL_ITERS * 24
    env.common_step_counter = dwell                       # first stage needs no extra wait
    w = _gate(env, clean=True, alpha=0.5)                 # EMA jumps past the threshold
    assert w == FORWARD_WEIGHT_STAGES[1]
    assert _gate(env, clean=True, alpha=0.5) == FORWARD_WEIGHT_STAGES[1]   # dwell not yet over
    env.common_step_counter += dwell
    assert _gate(env, clean=True, alpha=0.5) == FORWARD_WEIGHT_STAGES[2]
    frac = FORWARD_WEIGHT_STAGES[2] / FORWARD_WEIGHT_STAGES[-1]
    assert _SPAWN.params["midair_vx_range"] == pytest.approx(
        (MIDAIR_VX_RANGE[0] * frac, MIDAIR_VX_RANGE[1] * frac))
    for _ in range(50):                                   # landings get worse again
        env.common_step_counter += dwell
        assert _gate(env, clean=False, alpha=0.5) == FORWARD_WEIGHT_STAGES[2]
    assert _FWD.weight == FORWARD_WEIGHT_STAGES[2]
