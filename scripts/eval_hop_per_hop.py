#!/usr/bin/env python3
"""Per-hop eval battery for hop checkpoints that hop more than once per episode.

`scripts/eval_hop.py` scores ONE hop then a stand (AC #4): its landing + 0.5 s
snapshot re-arms on every landing, so on a policy that keeps hopping the
snapshot almost never completes. Run 6 (W&B 2qbd5bto, 2026-09-28) hops ~10
times per 3 s episode and scored 3.9% there as an artifact of that.

This battery segments each episode into individual hops and scores each one:

  * a HOP is a run of steps with the WHOLE robot off the ground (no foot, no
    non-foot contact — mdp._hop_airborne_now) lasting >= HOP_MIN_AIR_TIME;
  * its LANDING is CLEAN when no non-foot body touches the ground from that
    touchdown until the next liftoff (or the episode's end);
  * its FORWARD distance is the trunk's displacement from liftoff to
    touchdown, projected on the trunk heading at liftoff.

Reports per-hop clean fraction, hops per second, air time and forward
distance per hop, forward speed, and end-state clusters. Spawn type is forced
through the live event manager (AGENTS.md: env.cfg writes are no-ops).
CPU works (slowly); nothing here submits a job.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class Hop:
    liftoff_step: int
    landing_step: int
    air_time_s: float
    forward_m: float
    clean: bool


def segment_hops(
    airborne: np.ndarray,
    nonfoot: np.ndarray,
    fwd_pos: np.ndarray,
    dt: float,
    min_air_time: float,
) -> list[Hop]:
    """Split ONE episode's per-step traces into scored hops.

    airborne: bool[T]      whole robot off the ground this step
    nonfoot:  bool[T]      any non-foot body touching the ground this step
    fwd_pos:  float[T, 3]  trunk (x, y, yaw); each hop's forward distance is
                           projected on the yaw of the last grounded step
                           before its liftoff.

    A flight still in progress when the episode ends is not a hop (it has no
    landing to score), and neither is one shorter than ``min_air_time``.
    """
    T = len(airborne)
    hops: list[Hop] = []
    t = 0
    while t < T:
        if not airborne[t]:
            t += 1
            continue
        start = t
        while t < T and airborne[t]:
            t += 1
        end = t  # first grounded step (or T if the episode ended in the air)
        air = (end - start) * dt
        if end >= T or air < min_air_time:
            continue
        nxt = end
        while nxt < T and not airborne[nxt]:
            nxt += 1
        x0, y0, yaw0 = fwd_pos[start - 1] if start > 0 else fwd_pos[start]
        x1, y1, _ = fwd_pos[end]
        forward = (x1 - x0) * np.cos(yaw0) + (y1 - y0) * np.sin(yaw0)
        clean = not bool(nonfoot[end:nxt].any())
        hops.append(Hop(start, end, float(air), float(forward), clean))
    return hops


@dataclass(frozen=True)
class EpisodeSummary:
    hops: list
    ended_upright: bool
    duration_s: float


def summarize(episodes: list[EpisodeSummary]) -> dict:
    hops = [h for e in episodes for h in e.hops]
    n_ep = len(episodes)
    total_t = sum(e.duration_s for e in episodes)
    if not hops:
        return {"episodes": n_ep, "hops": 0}
    air = np.array([h.air_time_s for h in hops])
    fwd = np.array([h.forward_m for h in hops])
    clean = np.array([h.clean for h in hops])
    first_clean = [e.hops[0].clean for e in episodes if e.hops]
    return {
        "episodes": n_ep,
        "hops": len(hops),
        "hops_per_s": len(hops) / total_t,
        "episodes_with_a_hop": sum(1 for e in episodes if e.hops) / n_ep,
        "clean_landing_frac_per_hop": float(clean.mean()),
        "first_hop_clean_frac": float(np.mean(first_clean)) if first_clean else 0.0,
        "all_hops_clean_episode_frac": sum(1 for e in episodes if e.hops and all(h.clean for h in e.hops)) / n_ep,
        "air_time_s_median": float(np.median(air)),
        "air_time_s_p10_p90": (float(np.percentile(air, 10)), float(np.percentile(air, 90))),
        "forward_m_per_hop_median": float(np.median(fwd)),
        "forward_speed_mps": float(fwd.sum() / total_t),
        "ended_upright_frac": sum(e.ended_upright for e in episodes) / n_ep,
    }


@dataclass(frozen=True)
class Cfg:
    checkpoint_file: str
    episodes: int = 128
    spawn: str = "standing"  # standing | crouch | midair
    device: str = "cpu"
    upright_min: float = 0.9


def run(task_id: str, cfg: Cfg) -> list[EpisodeSummary]:
    import torch

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    from mjlab_microduck.tasks import mdp as M
    from mjlab_microduck.tasks.microduck_hop_env_cfg import HOP_MIN_AIR_TIME

    n = cfg.episodes
    env_cfg = load_env_cfg(task_id, play=True)
    # The hop_spawn_mix curriculum rewrites set_hop_state's spawn mix on every
    # reset (curricula run before reset events), which would override cfg.spawn
    # (same fix as eval_hop.drop_spawn_mix_curriculum).
    env_cfg.curriculum.pop("hop_spawn_mix", None)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = n
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env.event_manager.get_term_cfg("set_hop_state").params.update(
        standing_prob=float(cfg.spawn == "standing"),
        crouch_prob=float(cfg.spawn == "crouch"),
        midair_prob=float(cfg.spawn == "midair"),
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task_id) or OnPolicyRunner)(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(cfg.checkpoint_file, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)
    asset = env.scene["robot"]

    T = int(env.max_episode_length)
    airborne = np.zeros((n, T), bool)
    nonfoot = np.zeros((n, T), bool)
    pose = np.zeros((n, T, 3))
    alive = np.ones(n, bool)
    length = np.full(n, T)
    upright_end = np.zeros(n, bool)

    def _upright():
        q = asset.data.root_link_quat_w
        return (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).cpu().numpy()

    with torch.inference_mode():
        wrapped.reset()
        obs = wrapped.get_observations()
        for t in range(T):
            obs, _, dones, _ = wrapped.step(policy(obs))
            airborne[:, t] = M._hop_airborne_now(env).cpu().numpy()
            nf = env.scene.sensors["nonfoot_ground_contact"].data.found
            nonfoot[:, t] = torch.nan_to_num(nf, nan=0.0).reshape(n, -1).any(-1).cpu().numpy()
            q = asset.data.root_link_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            xy = asset.data.root_link_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2]
            pose[:, t, :2] = xy.cpu().numpy()
            pose[:, t, 2] = yaw.cpu().numpy()
            up = _upright()
            ended = dones.bool().cpu().numpy() & alive
            upright_end[ended] = up[ended] >= cfg.upright_min
            length[ended] = t + 1
            alive &= ~ended
            if not alive.any():
                break
            reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_ids.numel() > 0:
                env.reset(env_ids=reset_ids)
                obs = wrapped.get_observations()
        up = _upright()
        upright_end[alive] = up[alive] >= cfg.upright_min
    env.close()

    out = []
    for i in range(n):
        L = int(length[i])
        hops = segment_hops(airborne[i, :L], nonfoot[i, :L], pose[i, :L], env.step_dt, HOP_MIN_AIR_TIME)
        out.append(EpisodeSummary(hops, bool(upright_end[i]), L * env.step_dt))
    return out


def main():
    import tyro

    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    task, rest = tyro.cli(tyro.extras.literal_type_from_choices(list_tasks()),
                          add_help=False, return_unknown_args=True)
    cfg = tyro.cli(Cfg, args=rest, config=(tyro.conf.FlagConversionOff,))
    s = summarize(run(task, cfg))
    print("=" * 60)
    print(f"  {task}  {Path(cfg.checkpoint_file).name}  spawn={cfg.spawn}")
    for k, v in s.items():
        print(f"  {k:<30s} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
