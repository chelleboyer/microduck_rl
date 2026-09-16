#!/usr/bin/env python3
"""Headless acceptance battery for the forward-hop checkpoint (AC #4).

Runs N episodes (default 512) of a trained ``Mjlab-Hop-Flat-MicroDuck`` (or
``-Backlash-`` variant) checkpoint, all from forced standing spawns, and
reports the AC #4 acceptance rate:

  .claude/plans/microduck-forward-hop.md, AC #4 — THE BAR:

    An episode is accepted iff: real simultaneous double-foot flight occurred
    (peak air time >= HOP_MIN_AIR_TIME); no non-foot body touched the ground
    at any point; and at landing + 0.5 s the trunk z >= 0.10 m, tilt <= 20 deg,
    and both feet are in contact.

Also reports an end-state cluster breakdown (standing / prone-front /
prone-back / side / never-lifted) over the same episodes.

Spawn state is forced to standing-only through
``env.event_manager.get_term_cfg("set_hop_state")`` — AGENTS.md: writing
``env.cfg.events[...]`` instead is a silent no-op, because managers deepcopy
their cfg at construction.

Needs a CUDA device for a real 512-episode run at full speed; building and
smoke-checking the script (``--help``, the predicate unit tests) is CPU-only.
Running this against a real checkpoint is a separate, human-launched act —
this script never submits an HF Job.
"""

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_MIN_AIR_TIME, STAND_Z

# AC #4's own literal thresholds (not measured constants, so they live here
# rather than in microduck_hop_env_cfg.py — they describe what counts as an
# acceptable landing, not a property of the robot).
AC4_MIN_TRUNK_Z_M = 0.10
AC4_MAX_TILT_DEG = 20.0
AC4_LANDING_DELAY_S = 0.5

END_STATE_CLUSTERS = ("standing", "prone-front", "prone-back", "side", "never-lifted")


@dataclass(frozen=True)
class HopEpisodeObservation:
    """What one episode did, in exactly the shape AC #4's predicate needs."""

    peak_air_time_s: float
    non_foot_contact_ever: bool
    landing_trunk_z_m: float | None
    landing_tilt_deg: float | None
    landing_both_feet_contact: bool | None
    # Recorded per the plan's task description, not part of the AC #4 accept/
    # reject predicate itself.
    peak_forward_dist_m: float = 0.0
    # End-state-only fields (not part of the AC #4 predicate).
    final_trunk_z_m: float = 0.0
    final_upright: float = 0.0
    final_gravity_body_x: float = 0.0
    final_gravity_body_y: float = 0.0


def hop_episode_accepted(
    obs: HopEpisodeObservation,
    min_air_time_s: float = HOP_MIN_AIR_TIME,
    min_trunk_z_m: float = AC4_MIN_TRUNK_Z_M,
    max_tilt_deg: float = AC4_MAX_TILT_DEG,
) -> bool:
    """AC #4's acceptance predicate, exactly as worded in the forward-hop plan.

    An episode that never produced a landing + 0.5s snapshot (never lifted
    off, or timed out before reaching it) has ``None`` landing fields and is
    rejected by that alone.
    """
    if obs.peak_air_time_s < min_air_time_s:
        return False
    if obs.non_foot_contact_ever:
        return False
    if (
        obs.landing_trunk_z_m is None
        or obs.landing_tilt_deg is None
        or obs.landing_both_feet_contact is None
    ):
        return False
    if obs.landing_trunk_z_m < min_trunk_z_m:
        return False
    if obs.landing_tilt_deg > max_tilt_deg:
        return False
    if not obs.landing_both_feet_contact:
        return False
    return True


def classify_end_state(
    obs: HopEpisodeObservation,
    min_air_time_s: float = HOP_MIN_AIR_TIME,
    stand_z_m: float = STAND_Z,
    upright_min: float = 0.9397,  # cos(20 deg), matches AC4_MAX_TILT_DEG
    stand_z_tol_m: float = 0.02,
) -> str:
    """Bucket the episode's FINAL frame for the end-state cluster breakdown.

    Front/back/side is read off ``projected_gravity_b`` (which local axis is
    currently "down"): a robot fallen chest-down has gravity aligned with its
    local +x (forward, the same axis reset_hop_state's sampled yaw points
    down-heading), back-down aligns it with local -x, and a side fall puts it
    on local +/-y. This is a coarse diagnostic for the report, not part of
    the AC #4 acceptance predicate — verify surprising clusters against video
    per AGENTS.md ("watch the video AND check which geom/axis touches").
    """
    if obs.peak_air_time_s < min_air_time_s:
        return "never-lifted"
    if obs.final_upright >= upright_min and abs(obs.final_trunk_z_m - stand_z_m) <= stand_z_tol_m:
        return "standing"
    if abs(obs.final_gravity_body_x) >= abs(obs.final_gravity_body_y):
        return "prone-front" if obs.final_gravity_body_x > 0 else "prone-back"
    return "side"


@dataclass(frozen=True)
class EvalHopConfig:
    wandb_run_path: str | None = None
    """W&B run to pull the checkpoint from, e.g. 'entity/project/run_id'."""
    checkpoint: int | None = None
    """Select a checkpoint by iteration number (e.g. 1000). Requires wandb_run_path
    unless checkpoint_file is also given."""
    checkpoint_file: str | None = None
    """Local .pt file. Takes precedence over wandb_run_path/checkpoint."""
    episodes: int = 512
    num_envs: int | None = None
    """Parallel envs per wave. Defaults to min(episodes, 512); episodes > num_envs
    runs multiple sequential waves."""
    standing_prob: float = 1.0
    crouch_prob: float = 0.0
    midair_prob: float = 0.0
    device: str | None = None


def _resolve_checkpoint(agent_cfg, cfg: EvalHopConfig) -> Path:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        return resume_path
    if cfg.checkpoint is not None:
        checkpoint_filename = f"model_{cfg.checkpoint}.pt"
        if cfg.wandb_run_path is not None:
            import wandb

            api = wandb.Api()
            wandb_run = api.run(str(cfg.wandb_run_path))
            run_id = cfg.wandb_run_path.split("/")[-1]
            download_dir = log_root_path / "wandb_checkpoints" / run_id
            resume_path = download_dir / checkpoint_filename
            if not resume_path.exists():
                available = [f.name for f in wandb_run.files() if "model" in f.name]
                if checkpoint_filename not in available:
                    raise FileNotFoundError(
                        f"Checkpoint '{checkpoint_filename}' not found in wandb run. "
                        f"Available: {sorted(available)}"
                    )
                wandb_run.file(checkpoint_filename).download(str(download_dir), replace=True)
            return resume_path
        return get_checkpoint_path(log_root_path, checkpoint=re.escape(checkpoint_filename))
    if cfg.wandb_run_path is None:
        raise ValueError(
            "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
    resume_path, _ = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
    return resume_path


def _run_wave(env: ManagerBasedRlEnv, wrapped: RslRlVecEnvWrapper, policy, wave_n: int) -> list[HopEpisodeObservation]:
    """Step every env in this wave for exactly one fixed-length episode.

    ``env.cfg.auto_reset`` must be False: a landing + 0.5s snapshot can only
    be read off the SAME episode's terminal state, and mjlab's auto-reset
    overwrites that state the instant the episode ends.

    Because auto-reset is off, mjlab requires the caller to manually reset
    any env that terminates before the wave's last step (the hop cfg's sole
    non-timeout termination is ``nan_state``, an async per-env event) — the
    next ``wrapped.step()`` otherwise raises for the whole batch. This loop
    does that after each step, before the next one is taken.

    Must be called from inside the caller's own ``torch.inference_mode()``
    span (``run_battery`` opens one around its whole multi-wave loop). A
    span scoped to just this function would promote tensors reassigned
    in-place during stepping/reset (e.g. the BAM actuator's per-step delay
    buffer) to inference tensors, which then fail on the very next
    ``wrapped.reset()`` the caller issues between waves, outside this
    function's span.
    """
    device = env.device
    num_envs = env.num_envs
    landing_delay_steps = round(AC4_LANDING_DELAY_S / env.step_dt)
    max_steps = env.max_episode_length

    was_airborne = torch.zeros(num_envs, dtype=torch.bool, device=device)
    flight_start_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    non_foot_contact_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landing_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    landing_z = torch.full((num_envs,), float("nan"), device=device)
    landing_tilt_deg = torch.full((num_envs,), float("nan"), device=device)
    landing_both_feet = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landing_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs = wrapped.get_observations()
    asset = env.scene["robot"]
    for step in range(max_steps):
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)

        airborne_now = microduck_mdp._hop_airborne_now(env)
        microduck_mdp._update_hop_forward_accum(env)
        liftoff = airborne_now & ~was_airborne
        flight_start_step = torch.where(
            liftoff, torch.full_like(flight_start_step, step), flight_start_step
        )
        landed = was_airborne & ~airborne_now
        flight_duration_s = (step - flight_start_step).float() * env.step_dt
        qualifying_landing = landed & (flight_duration_s >= HOP_MIN_AIR_TIME)
        # Re-latch on every qualifying landing (not just the episode's
        # first): a precursor bounce below HOP_MIN_AIR_TIME must not pin
        # the +0.5s snapshot to the wrong flight when a later, real hop
        # is the one that actually qualifies.
        landing_step = torch.where(
            qualifying_landing, torch.full_like(landing_step, step), landing_step
        )
        landing_recorded = landing_recorded & ~qualifying_landing
        was_airborne = airborne_now

        nf = env.scene.sensors["nonfoot_ground_contact"].data.found
        nf_touch = torch.nan_to_num(nf, nan=0.0).reshape(nf.shape[0], -1).any(dim=-1)
        non_foot_contact_ever |= nf_touch

        due_now = (
            (landing_step >= 0)
            & (step - landing_step == landing_delay_steps)
            & ~landing_recorded
        )
        if bool(due_now.any()):
            z = torch.nan_to_num(
                asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
            )
            quat = asset.data.root_link_quat_w
            upright = torch.nan_to_num(
                1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2)), nan=-1.0
            )
            tilt_deg = torch.rad2deg(torch.arccos(upright.clamp(-1.0, 1.0)))
            found = env.scene.sensors["feet_ground_contact"].data.found
            found = torch.nan_to_num(found, nan=0.0).reshape(found.shape[0], -1)[:, :2]
            both_feet = found[:, 0].bool() & found[:, 1].bool()

            landing_z = torch.where(due_now, z, landing_z)
            landing_tilt_deg = torch.where(due_now, tilt_deg, landing_tilt_deg)
            landing_both_feet = torch.where(due_now, both_feet, landing_both_feet)
            landing_recorded = landing_recorded | due_now

        # mjlab contract for auto_reset=False (manager_based_rl_env.py): any
        # env whose reset_buf fired must be reset before the next step()
        # call, or that call raises for the whole batch. The hop cfg's sole
        # non-timeout termination (nan_state) is async and per-env, so this
        # fires occasionally within a wave, not only on the final step.
        reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            env.reset(env_ids=reset_ids)
            obs = wrapped.get_observations()

    peak_air_time = microduck_mdp.hop_metric_max_air_time(env)
    peak_forward_dist = microduck_mdp._hop_forward_state(env)[2]
    final_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    final_quat = asset.data.root_link_quat_w
    final_upright = torch.nan_to_num(
        1.0 - 2.0 * (final_quat[:, 1].pow(2) + final_quat[:, 2].pow(2)), nan=-1.0
    )
    gravity_b = asset.data.projected_gravity_b

    results = []
    for i in range(wave_n):
        recorded = bool(landing_recorded[i])
        results.append(
            HopEpisodeObservation(
                peak_air_time_s=float(peak_air_time[i]),
                non_foot_contact_ever=bool(non_foot_contact_ever[i]),
                landing_trunk_z_m=float(landing_z[i]) if recorded else None,
                landing_tilt_deg=float(landing_tilt_deg[i]) if recorded else None,
                landing_both_feet_contact=bool(landing_both_feet[i]) if recorded else None,
                peak_forward_dist_m=float(peak_forward_dist[i]),
                final_trunk_z_m=float(final_z[i]),
                final_upright=float(final_upright[i]),
                final_gravity_body_x=float(gravity_b[i, 0]),
                final_gravity_body_y=float(gravity_b[i, 1]),
            )
        )
    return results


def run_battery(task_id: str, cfg: EvalHopConfig) -> list[HopEpisodeObservation]:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    resume_path = _resolve_checkpoint(agent_cfg, cfg)
    print(f"[INFO]: Loading checkpoint: {resume_path}")

    num_envs = cfg.num_envs or min(cfg.episodes, 512)
    env_cfg.scene.num_envs = num_envs
    env_cfg.auto_reset = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Writing directly to the env's cfg here would be a silent no-op — the
    # event manager deepcopies its cfg at construction (AGENTS.md).
    spawn_term = env.event_manager.get_term_cfg("set_hop_state")
    spawn_term.params["standing_prob"] = cfg.standing_prob
    spawn_term.params["crouch_prob"] = cfg.crouch_prob
    spawn_term.params["midair_prob"] = cfg.midair_prob

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    results: list[HopEpisodeObservation] = []
    remaining = cfg.episodes
    # One continuous inference_mode span across every wave, including the
    # inter-wave wrapped.reset() calls: torch promotes any tensor REASSIGNED
    # inside inference_mode to an inference tensor (e.g. the BAM actuator's
    # DelayBuffer._current_lags, reassigned every step via torch.where), and
    # an in-place write to an inference tensor from OUTSIDE inference_mode
    # raises. Scoping inference_mode to only _run_wave's step loop left the
    # very next wave's wrapped.reset() outside it, writing in-place to a
    # tensor _run_wave had just promoted — an unconditional crash on any
    # episodes > num_envs run, independent of any NaN termination.
    with torch.inference_mode():
        while remaining > 0:
            wave_n = min(num_envs, remaining)
            wrapped.reset()
            results.extend(_run_wave(env, wrapped, policy, wave_n))
            remaining -= wave_n

    env.close()
    return results


def report(results: list[HopEpisodeObservation]) -> None:
    n = len(results)
    accepted = [hop_episode_accepted(r) for r in results]
    acceptance_rate = sum(accepted) / n if n else 0.0

    clusters = {name: 0 for name in END_STATE_CLUSTERS}
    for r in results:
        clusters[classify_end_state(r)] += 1

    mean_forward_dist = sum(r.peak_forward_dist_m for r in results) / n if n else 0.0

    print("=" * 60)
    print(f"  Episodes:        {n}")
    print(f"  Accepted (AC#4): {sum(accepted)} ({acceptance_rate:.1%})")
    print(f"  Mean peak forward displacement: {mean_forward_dist:.3f} m")
    print("  End-state clusters:")
    for name in END_STATE_CLUSTERS:
        count = clusters[name]
        print(f"    {name:<14s} {count:4d} ({count / n:.1%})" if n else f"    {name:<14s} 0")
    print("=" * 60)


def main():
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    args = tyro.cli(
        EvalHopConfig,
        args=remaining_args,
        default=EvalHopConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    )
    del remaining_args

    results = run_battery(chosen_task, args)
    report(results)


if __name__ == "__main__":
    main()
