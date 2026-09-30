"""Mjlab-HopOnce-JumpWarm cfg and the community-jump checkpoint converter (CPU only)."""

import importlib.util
from pathlib import Path

import torch

from mjlab_microduck.tasks.microduck_hop_env_cfg import (
    FORWARD_WEIGHT_STAGES,
    JUMPWARM_SPAWN_MIX,
    MicroduckHopOnceJumpWarmRlCfg,
    MicroduckHopOnceRlCfg,
    make_microduck_hop_env_cfg,
)

_spec = importlib.util.spec_from_file_location(
    "convert_jump_checkpoint", Path(__file__).resolve().parents[1] / "scripts" / "convert_jump_checkpoint.py")
cj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cj)


def test_jump_warm_keeps_the_forward_gate_and_dirty_slope_live():
    warm = make_microduck_hop_env_cfg(once=True, jump_warm=True)
    once = make_microduck_hop_env_cfg(once=True)
    assert "forward_gate" in warm.curriculum and "dirty_landing_scale" in warm.curriculum
    assert warm.rewards["hop_forward_progress"].weight == FORWARD_WEIGHT_STAGES[0] == 0.0
    assert warm.events["set_hop_state"].params["midair_vx_range"] == (0.0, 0.0)
    # plain HopOnce still continues run 6 with both collapsed
    assert "forward_gate" not in once.curriculum


def test_jump_warm_is_hoponce_v3_plus_nothing_else():
    warm = make_microduck_hop_env_cfg(once=True, jump_warm=True)
    once = make_microduck_hop_env_cfg(once=True)
    assert set(warm.rewards) == set(once.rewards)
    for name in ("hop_extra_flight", "hop_planted"):
        assert warm.rewards[name].weight == once.rewards[name].weight


def test_jump_warm_starts_standing_heavy_and_keeps_the_later_stages():
    warm = make_microduck_hop_env_cfg(once=True, jump_warm=True)
    once = make_microduck_hop_env_cfg(once=True)
    ws = warm.curriculum["hop_spawn_mix"].params["param_stages"]
    os_ = once.curriculum["hop_spawn_mix"].params["param_stages"]
    assert ws[0]["params"] == JUMPWARM_SPAWN_MIX
    assert abs(sum(JUMPWARM_SPAWN_MIX.values()) - 1.0) < 1e-9
    assert ws[1:] == os_[1:]
    assert os_[0]["params"] != JUMPWARM_SPAWN_MIX          # HopOnce itself unchanged


def test_jump_warm_has_its_own_experiment_and_v3_ppo():
    assert MicroduckHopOnceJumpWarmRlCfg.experiment_name == "microduck_hop_once_jumpwarm"
    a, b = MicroduckHopOnceJumpWarmRlCfg.algorithm, MicroduckHopOnceRlCfg.algorithm
    assert (a.entropy_coef, a.desired_kl) == (b.entropy_coef, b.desired_kl)


def _fake_jump_ckpt():
    def mlp(i, o):
        return {"mlp.0.weight": torch.randn(512, i), "mlp.0.bias": torch.randn(512),
                "mlp.2.weight": torch.randn(256, 512), "mlp.2.bias": torch.randn(256),
                "mlp.4.weight": torch.randn(128, 256), "mlp.4.bias": torch.randn(128),
                "mlp.6.weight": torch.randn(o, 128), "mlp.6.bias": torch.randn(o)}

    def norm(n):
        return {"obs_normalizer._mean": torch.zeros(1, n), "obs_normalizer._var": torch.ones(1, n),
                "obs_normalizer._std": torch.ones(1, n), "obs_normalizer.count": torch.tensor(5)}

    actor = {**norm(61), "distribution.raw_std_param": torch.full((14,), -3.0), **mlp(61, 14)}
    critic = {**norm(74), **mlp(74, 1)}
    return {"actor_state_dict": actor, "critic_state_dict": critic,
            "optimizer_state_dict": [], "iter": 34995, "infos": []}


def test_converter_remaps_the_bounded_std_and_keeps_the_mean_path():
    src = _fake_jump_ckpt()
    out = cj.convert(src, init_std=0.3)
    a = out["actor_state_dict"]
    assert "distribution.raw_std_param" not in a
    assert torch.equal(a["distribution.std_param"], torch.full((14,), 0.3))
    for k, v in src["actor_state_dict"].items():
        if k != "distribution.raw_std_param":
            assert torch.equal(a[k], v), k
    assert out["critic_state_dict"].keys() == src["critic_state_dict"].keys()
    assert out["iter"] == 0 and out["infos"]["source_iter"] == 34995
    std = cj.jump_std(torch.tensor(-3.0))
    assert abs(out["infos"]["source_std_mean"] - float(std)) < 1e-6


def test_converter_emits_a_fresh_adam_state_our_optimizer_accepts():
    out = cj.convert(_fake_jump_ckpt(), lr=1e-3)
    n = 9 + 8        # actor: std + 4 Linear x (W, b); critic: 4 Linear x (W, b)
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(n)]
    opt = torch.optim.Adam(params, lr=5e-4)
    opt.load_state_dict(out["optimizer_state_dict"])        # raises on a group/param mismatch
    assert opt.state_dict()["state"] == {}
    assert opt.param_groups[0]["lr"] == 1e-3
