"""scripts/eval_hop.py's AC #4 predicate, tested standalone (CPU-only, no
checkpoint, no env). AC #4 (.claude/plans/microduck-forward-hop.md): an
episode is accepted iff real simultaneous double-foot flight occurred, no
non-foot body ever touched the ground, and at landing + 0.5s the trunk is
high enough, upright enough, and both feet are in contact. Each of these is
tested individually: a fully-passing episode accepts, and flipping exactly
one condition at a time rejects.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def eh():
    spec = importlib.util.spec_from_file_location("eval_hop", REPO / "scripts" / "eval_hop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def good_episode(eh):
    return eh.HopEpisodeObservation(
        peak_air_time_s=eh.HOP_MIN_AIR_TIME + 0.02,
        non_foot_contact_ever=False,
        landing_trunk_z_m=eh.AC4_MIN_TRUNK_Z_M + 0.01,
        landing_tilt_deg=eh.AC4_MAX_TILT_DEG - 5.0,
        landing_both_feet_contact=True,
    )


def test_hop_episode_accepted_when_every_condition_holds(eh, good_episode):
    assert eh.hop_episode_accepted(good_episode) is True


def test_hop_episode_rejects_short_air_time(eh, good_episode):
    bad = eh.HopEpisodeObservation(
        **{**vars(good_episode), "peak_air_time_s": eh.HOP_MIN_AIR_TIME - 0.01}
    )
    assert eh.hop_episode_accepted(bad) is False


def test_hop_episode_rejects_non_foot_contact(eh, good_episode):
    bad = eh.HopEpisodeObservation(**{**vars(good_episode), "non_foot_contact_ever": True})
    assert eh.hop_episode_accepted(bad) is False


def test_hop_episode_rejects_low_landing_height(eh, good_episode):
    bad = eh.HopEpisodeObservation(
        **{**vars(good_episode), "landing_trunk_z_m": eh.AC4_MIN_TRUNK_Z_M - 0.01}
    )
    assert eh.hop_episode_accepted(bad) is False


def test_hop_episode_rejects_excess_tilt(eh, good_episode):
    bad = eh.HopEpisodeObservation(
        **{**vars(good_episode), "landing_tilt_deg": eh.AC4_MAX_TILT_DEG + 0.01}
    )
    assert eh.hop_episode_accepted(bad) is False


def test_hop_episode_rejects_missing_both_feet_contact(eh, good_episode):
    bad = eh.HopEpisodeObservation(
        **{**vars(good_episode), "landing_both_feet_contact": False}
    )
    assert eh.hop_episode_accepted(bad) is False


def test_hop_episode_rejects_episode_that_never_reached_a_landing_snapshot(eh, good_episode):
    """A robot that never lifts off, or one that lifts off but never re-settles
    long enough for the +0.5s snapshot, must reject rather than crash on None."""
    bad = eh.HopEpisodeObservation(
        **{
            **vars(good_episode),
            "landing_trunk_z_m": None,
            "landing_tilt_deg": None,
            "landing_both_feet_contact": None,
        }
    )
    assert eh.hop_episode_accepted(bad) is False


def test_classify_end_state_never_lifted(eh):
    obs = eh.HopEpisodeObservation(
        peak_air_time_s=0.0,
        non_foot_contact_ever=False,
        landing_trunk_z_m=None,
        landing_tilt_deg=None,
        landing_both_feet_contact=None,
    )
    assert eh.classify_end_state(obs) == "never-lifted"


def test_classify_end_state_standing(eh):
    obs = eh.HopEpisodeObservation(
        peak_air_time_s=eh.HOP_MIN_AIR_TIME + 0.02,
        non_foot_contact_ever=False,
        landing_trunk_z_m=eh.AC4_MIN_TRUNK_Z_M,
        landing_tilt_deg=0.0,
        landing_both_feet_contact=True,
        final_trunk_z_m=eh.STAND_Z,
        final_upright=1.0,
        final_gravity_body_x=0.0,
        final_gravity_body_y=0.0,
    )
    assert eh.classify_end_state(obs) == "standing"


@pytest.mark.parametrize(
    "gx, gy, expected",
    [
        (1.0, 0.0, "prone-front"),
        (-1.0, 0.0, "prone-back"),
        (0.0, 1.0, "side"),
    ],
)
def test_classify_end_state_fallen_orientations(eh, gx, gy, expected):
    obs = eh.HopEpisodeObservation(
        peak_air_time_s=eh.HOP_MIN_AIR_TIME + 0.02,
        non_foot_contact_ever=True,
        landing_trunk_z_m=None,
        landing_tilt_deg=None,
        landing_both_feet_contact=None,
        final_trunk_z_m=0.03,
        final_upright=-1.0,
        final_gravity_body_x=gx,
        final_gravity_body_y=gy,
    )
    assert eh.classify_end_state(obs) == expected


def test_eval_hop_pulls_thresholds_from_hop_env_cfg_not_pasted_literals(eh):
    from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_MIN_AIR_TIME, STAND_Z

    assert eh.HOP_MIN_AIR_TIME == HOP_MIN_AIR_TIME
    assert eh.STAND_Z == STAND_Z


def test_eval_hop_forces_spawn_via_get_term_cfg_not_cfg_events(eh):
    """AGENTS.md: env.cfg.events[...] is a silent no-op (managers deepcopy
    their cfg at construction) — the spawn-forcing must go through
    env.event_manager.get_term_cfg. Lock this at the source level so the
    silent-no-op form cannot be reintroduced."""
    import inspect

    source = inspect.getsource(eh.run_battery)
    assert "event_manager.get_term_cfg" in source
    assert "env.cfg.events[" not in source


def test_eval_hop_sets_auto_reset_false(eh):
    """A landing + 0.5s snapshot can only be read off the SAME episode's
    terminal state, and mjlab's auto-reset (the default) overwrites that
    state the instant the episode ends. Lock this at the source level: no
    existing test touches run_battery's env construction, so a refactor that
    drops this unlabeled-looking line would silently corrupt every landing
    snapshot for episodes that end before max_episode_length, and nothing
    else would catch it."""
    import inspect

    source = inspect.getsource(eh.run_battery)
    assert "auto_reset = False" in source


def test_eval_hop_manually_resets_terminated_envs_within_a_wave(eh):
    """mjlab's auto_reset=False contract requires the caller to call
    env.reset(env_ids=...) for every env that terminates before the wave's
    next step(), or that step() raises for the whole batch. The hop cfg's
    sole non-timeout termination (nan_state) is an async per-env event, so a
    512-episode wave is expected to hit it occasionally. Lock the handshake
    at the source level."""
    import inspect

    source = inspect.getsource(eh._run_wave)
    assert "env.reset(env_ids=" in source
    assert "reset_buf" in source


def test_eval_hop_inference_mode_spans_the_whole_multi_wave_loop(eh):
    """torch promotes any tensor reassigned in-place inside inference_mode to
    an inference tensor (e.g. the BAM actuator's per-step delay buffer). If
    inference_mode were scoped to only _run_wave's step loop, the very next
    wave's wrapped.reset() call in run_battery's while loop -- which sits
    outside that scope -- would crash on the first in-place write to a
    tensor the prior wave had just promoted: an unconditional failure on any
    episodes > num_envs run, independent of any NaN termination. Lock that
    the span wraps run_battery's while loop (both wrapped.reset() and
    _run_wave) rather than living inside _run_wave alone."""
    import inspect

    battery_source = inspect.getsource(eh.run_battery)
    assert "with torch.inference_mode():" in battery_source
    # The span must open before the while loop, not inside it: the
    # inference_mode line must precede the `while remaining > 0` line, and
    # `_run_wave` must not open a second, independently-scoped span.
    assert battery_source.index("with torch.inference_mode():") < battery_source.index(
        "while remaining > 0"
    )
    wave_source = inspect.getsource(eh._run_wave)
    assert "with torch.inference_mode():" not in wave_source


def test_eval_hop_reports_peak_forward_displacement(eh):
    """The plan's task description for this script requires recording peak
    clean forward displacement from liftoff alongside air time and non-foot
    contact. Prove the field exists and that report() surfaces it."""
    import io
    from contextlib import redirect_stdout

    obs = eh.HopEpisodeObservation(
        peak_air_time_s=eh.HOP_MIN_AIR_TIME + 0.02,
        non_foot_contact_ever=False,
        landing_trunk_z_m=eh.AC4_MIN_TRUNK_Z_M + 0.01,
        landing_tilt_deg=eh.AC4_MAX_TILT_DEG - 5.0,
        landing_both_feet_contact=True,
        peak_forward_dist_m=0.123,
    )
    assert obs.peak_forward_dist_m == 0.123

    buf = io.StringIO()
    with redirect_stdout(buf):
        eh.report([obs])
    assert "forward" in buf.getvalue().lower()


def test_eval_hop_run_wave_accumulates_forward_displacement(eh):
    """_run_wave must call _update_hop_forward_accum and read the frontier
    via _hop_forward_state, not re-derive forward displacement itself."""
    import inspect

    source = inspect.getsource(eh._run_wave)
    assert "_update_hop_forward_accum(env)" in source
    assert "_hop_forward_state(env)" in source
