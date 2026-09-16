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


def test_eval_hop_clears_local_bookkeeping_on_mid_wave_reset(eh):
    """A mid-wave env.reset(env_ids=...) (the manual-reset handshake) only
    clears the MDP-level accumulators (reset_hop_state zeroes
    _hop_max_air_time/_hop_max_forward_dist for those env_ids). _run_wave's
    own per-episode locals -- non_foot_contact_ever and the landing_* latches
    -- must be cleared too, or a later episode segment in the same wave slot
    inherits an earlier, different episode's state and the reported
    HopEpisodeObservation splices two episodes together. Lock at the source
    level that every local bookkeeping tensor is reset for reset_ids right
    after the manual reset call."""
    import inspect

    source = inspect.getsource(eh._run_wave)
    reset_call_index = source.index("env.reset(env_ids=")
    cleared_after_reset = [
        "was_airborne[reset_ids] = False",
        "flight_start_step[reset_ids] = -1",
        "non_foot_contact_ever[reset_ids] = False",
        "landing_step[reset_ids] = -1",
        "landing_z[reset_ids] = float(\"nan\")",
        "landing_tilt_deg[reset_ids] = float(\"nan\")",
        "landing_both_feet[reset_ids] = False",
        "landing_recorded[reset_ids] = False",
    ]
    for line in cleared_after_reset:
        assert line in source, line
        assert source.index(line) > reset_call_index, line


def test_eval_hop_run_wave_accumulates_forward_displacement(eh):
    """_run_wave must call _update_hop_forward_accum and read the frontier
    via _hop_forward_state, not re-derive forward displacement itself."""
    import inspect

    source = inspect.getsource(eh._run_wave)
    assert "_update_hop_forward_accum(env)" in source
    assert "_hop_forward_state(env)" in source


# ── Wave lifecycle ───────────────────────────────────────────────────────────
#
# The tests above exercise the AC #4 predicate in isolation. These drive
# `_run_wave`'s real loop — step, terminate, reset, report — against a stub env
# that reproduces the one mjlab behaviour that makes the reporting fragile:
# `reset()` zeroes the MDP accumulators (`reset_hop_state`) and respawns the
# robot, and mjlab's `time_out` term fires for EVERY env on the wave's final
# step. Reading metrics after the loop therefore reports the replacement spawn
# rather than the episode that was measured, which reads as 0% acceptance for
# any policy — a real failure and a reporting bug are indistinguishable in the
# output, so only a lifecycle test can tell them apart.


class _StubData:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubSensor:
    def __init__(self, found):
        self.data = _StubData(found=found)


class _StubScene:
    def __init__(self, robot, sensors, env_origins):
        self._robot = robot
        self.sensors = sensors
        self.terrain = _StubData(env_origins=env_origins)

    def __getitem__(self, key):
        assert key == "robot"
        return self._robot


class _StubEnv:
    """Minimal mjlab-shaped env: accumulators that reset() clears, and a robot
    pose that reset() moves back to spawn."""

    SPAWN_Z = 0.117
    TERMINAL_Z = 0.052  # a fallen robot, distinguishable from the spawn pose

    def __init__(self, torch, num_envs, max_episode_length, step_dt):
        self._torch = torch
        self.device = "cpu"
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.step_dt = step_dt

        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool)
        self.acc_air_time = torch.zeros(num_envs)
        self.acc_forward = torch.zeros(num_envs)
        self.reset_calls = []

        self.root_pos = torch.zeros(num_envs, 3)
        self.root_pos[:, 2] = self.SPAWN_Z
        self.root_quat = torch.zeros(num_envs, 4)
        self.root_quat[:, 0] = 1.0  # upright
        self.gravity_b = torch.zeros(num_envs, 3)
        self.gravity_b[:, 2] = -1.0

        robot = _StubData(
            data=_StubData(
                root_link_pos_w=self.root_pos,
                root_link_quat_w=self.root_quat,
                projected_gravity_b=self.gravity_b,
            )
        )
        self.scene = _StubScene(
            robot,
            {
                "nonfoot_ground_contact": _StubSensor(torch.zeros(num_envs, 3)),
                "feet_ground_contact": _StubSensor(torch.ones(num_envs, 2)),
            },
            torch.zeros(num_envs, 3),
        )

    def reset(self, env_ids=None):
        """Mirrors reset_hop_state + respawn: clears accumulators, moves the
        robot back to the spawn pose."""
        self.reset_calls.append(list(env_ids.tolist()))
        self.acc_air_time[env_ids] = 0.0
        self.acc_forward[env_ids] = 0.0
        self.root_pos[env_ids, 2] = self.SPAWN_Z


class _StubWrapped:
    """Drives the scripted episode and fires reset_buf exactly where mjlab
    would: on `nan_state` mid-wave, and on `time_out` for every env at the end."""

    def __init__(self, torch, env, airborne_steps, nan_state_at):
        self._torch = torch
        self.env = env
        self.airborne_steps = airborne_steps
        self.nan_state_at = nan_state_at
        self.step_idx = -1

    def get_observations(self):
        return self._torch.zeros(self.env.num_envs, 1)

    def step(self, actions):
        torch = self._torch
        env = self.env
        self.step_idx += 1
        step = self.step_idx

        airborne = torch.tensor(
            [step in self.airborne_steps.get(i, ()) for i in range(env.num_envs)]
        )
        env.airborne_now = airborne
        env.acc_air_time += airborne.float() * env.step_dt
        env.acc_forward += airborne.float() * 0.01
        # Airborne or fallen, the trunk is never back at spawn height mid-episode.
        env.root_pos[:, 2] = env.TERMINAL_Z

        env.reset_buf = torch.zeros(env.num_envs, dtype=torch.bool)
        for i, at in self.nan_state_at.items():
            if step == at:
                env.reset_buf[i] = True
        if step == env.max_episode_length - 1:  # mjlab time_out, every env
            env.reset_buf[:] = True
        return get_observations_stub(torch, env), None, None, None


def get_observations_stub(torch, env):
    return torch.zeros(env.num_envs, 1)


class _StubMdp:
    """The three mdp entry points `_run_wave` uses, backed by the stub env's
    own accumulators so reset() clearing them is observable."""

    @staticmethod
    def _hop_airborne_now(env):
        return env.airborne_now

    @staticmethod
    def _update_hop_forward_accum(env):
        return None

    @staticmethod
    def hop_metric_max_air_time(env):
        return env.acc_air_time

    @staticmethod
    def _hop_forward_state(env):
        return (None, None, env.acc_forward)


@pytest.fixture
def wave(eh, monkeypatch):
    """Run one wave over a scripted episode and hand back the results."""
    torch = pytest.importorskip("torch")

    def _run(airborne_steps, nan_state_at=None, num_envs=1, max_steps=14, step_dt=0.1):
        env = _StubEnv(torch, num_envs, max_steps, step_dt)
        wrapped = _StubWrapped(torch, env, airborne_steps, nan_state_at or {})
        monkeypatch.setattr(eh, "microduck_mdp", _StubMdp)
        results = eh._run_wave(env, wrapped, lambda obs: obs, num_envs)
        return env, results

    return _run


def test_final_episode_metrics_survive_the_end_of_wave_reset(eh, wave):
    """THE REGRESSION. mjlab's time_out fires for every env on the wave's last
    step; the reset it triggers zeroes the accumulators and respawns the robot.
    Metrics must be snapshotted before that reset, or every episode reports
    zeros and the battery returns 0% acceptance for any policy."""
    # Airborne steps 2-3, landing at step 4, +0.5s snapshot due at step 9.
    env, results = wave(airborne_steps={0: (2, 3)})

    assert env.reset_calls, "time_out must have reset the env on the final step"
    assert env.reset_calls[-1] == [0]
    assert float(env.acc_air_time[0]) == 0.0, "reset must have cleared the accumulator"

    episode = results[0]
    assert episode.peak_air_time_s == pytest.approx(0.2, abs=1e-6), (
        "peak air time was read after the end-of-wave reset had zeroed it"
    )
    assert episode.peak_forward_dist_m > 0.0
    assert episode.landing_trunk_z_m is not None, (
        "the landing+0.5s snapshot was cleared by the end-of-wave reset"
    )
    assert episode.landing_both_feet_contact is True
    assert episode.final_trunk_z_m == pytest.approx(_StubEnv.TERMINAL_Z, abs=1e-6), (
        "end-state was read from the post-reset spawn pose, not the terminal state"
    )


def test_wave_reports_the_terminated_episode_not_its_truncated_replacement(eh, wave):
    """A mid-wave nan_state ends the episode being measured. The replacement
    segment that runs in the same slot for the rest of the wave is not a whole
    episode and must not be reported as one."""
    env, results = wave(airborne_steps={0: (2, 3, 10, 11)}, nan_state_at={0: 6})

    assert env.reset_calls[0] == [0], "nan_state must reset mid-wave"
    episode = results[0]
    # 0.2s from the pre-nan_state flight, not 0.2s from the replacement's
    # steps 10-11, and not the sum of both.
    assert episode.peak_air_time_s == pytest.approx(0.2, abs=1e-6)


def test_wave_snapshots_each_env_independently(eh, wave):
    """Slots terminate at different steps; each must report its own episode."""
    env, results = wave(
        airborne_steps={0: (2, 3), 1: (2, 3, 4)}, nan_state_at={0: 6}, num_envs=2
    )

    assert results[0].peak_air_time_s == pytest.approx(0.2, abs=1e-6)
    assert results[1].peak_air_time_s == pytest.approx(0.3, abs=1e-6)
