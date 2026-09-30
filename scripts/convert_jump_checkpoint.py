"""Convert the community jump checkpoint into a HopOnce-JumpWarm warm start.

Source: ThomasBurgess2000/microduck-max-height-jump, checkpoint/model_34995.pt
(Apache-2.0; verify its CHECKSUMS.sha256 first). Same actor (61 -> 512-256-128 -> 14)
and critic (74 -> 512-256-128 -> 1) shapes as ours, with obs normalizers, but the actor
uses a training-only BoundedGaussianDistribution:
std = min_std + (max_std - min_std) * sigmoid(raw_std_param). Ours is rsl_rl's Gaussian
with std_type="scalar", whose `distribution.std_param` IS the std. Only the mean path
(MLP + normalizer) carries the jump behaviour; the std is reset to `init_std`. Measured in
Mjlab-HopOnce-JumpWarm with stochastic actions, 128 standing spawns each (flight / clean
after landing / upright at 3 s): deterministic 87/51/58%, std 0.08 (the jump's own mean)
83/42/52%, std 0.15 80/35/41%, std 0.30 75/9/12%. At 0.30 the launch survives but clean
landings collapse, so the landing signal would be nearly invisible to PPO; 0.15 keeps it.

The checkpoint's optimizer state is empty, but rsl_rl's PPO.load always loads one, so the
output carries a FRESH Adam state: one param group over the actor's then the critic's
parameters, no moments, lr = our learning rate. Launch with MICRODUCK_WARM_START=1 so
the iteration and the step-based curricula restart at 0.

    uv run scripts/convert_jump_checkpoint.py SRC.pt OUT.pt [--init-std 0.15]
"""

from __future__ import annotations

import argparse

import torch

# BoundedGaussianDistribution bounds, from the jump's training/agent.yaml.
JUMP_MIN_STD = 0.05
JUMP_MAX_STD = 1.5


def jump_std(raw_std_param: torch.Tensor) -> torch.Tensor:
    return JUMP_MIN_STD + (JUMP_MAX_STD - JUMP_MIN_STD) * torch.sigmoid(raw_std_param)


def _n_params(state_dict: dict) -> int:
    """Parameters are everything but the EmpiricalNormalization buffers."""
    return sum(1 for k in state_dict if not k.startswith("obs_normalizer."))


def fresh_adam_state(n_params: int, lr: float) -> dict:
    return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1)) for _ in range(n_params)], lr=lr).state_dict()


def convert(ckpt: dict, init_std: float = 0.15, lr: float = 1.0e-3) -> dict:
    actor = dict(ckpt["actor_state_dict"])
    raw = actor.pop("distribution.raw_std_param")
    actor["distribution.std_param"] = torch.full_like(raw, float(init_std))
    return {
        "actor_state_dict": actor,
        "critic_state_dict": dict(ckpt["critic_state_dict"]),
        "optimizer_state_dict": fresh_adam_state(
            _n_params(actor) + _n_params(ckpt["critic_state_dict"]), lr),
        "iter": 0,
        "infos": {
            "source": "ThomasBurgess2000/microduck-max-height-jump checkpoint/model_34995.pt",
            "source_iter": int(ckpt.get("iter", -1)),
            "source_std_mean": float(jump_std(raw).mean()),
            "init_std": float(init_std),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--init-std", type=float, default=0.15)
    a = ap.parse_args()
    out = convert(torch.load(a.src, map_location="cpu", weights_only=False), a.init_std)
    torch.save(out, a.out)
    print(f"wrote {a.out}: {out['infos']}")


if __name__ == "__main__":
    main()
