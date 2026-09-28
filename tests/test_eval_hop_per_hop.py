"""The per-hop segmentation behind scripts/eval_hop_per_hop.py (pure, CPU-only)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_hop_per_hop import EpisodeSummary, segment_hops, summarize  # noqa: E402

DT = 0.02
MIN_AIR = 0.06  # 3 steps


def _trace(pattern: str, nonfoot_at=(), yaw=0.0, step_fwd=0.01):
    """'g' grounded, 'a' airborne; the trunk moves step_fwd per step along yaw."""
    T = len(pattern)
    air = np.array([c == "a" for c in pattern])
    nf = np.zeros(T, bool)
    nf[list(nonfoot_at)] = True
    s = np.arange(T) * step_fwd
    pose = np.stack([s * math.cos(yaw), s * math.sin(yaw), np.full(T, yaw)], axis=1)
    return air, nf, pose


def test_counts_each_qualifying_flight_and_ignores_short_ones():
    air, nf, pose = _trace("gg" + "aaaa" + "ggg" + "aa" + "ggg" + "aaaaa" + "gg")
    hops = segment_hops(air, nf, pose, DT, MIN_AIR)
    assert [h.liftoff_step for h in hops] == [2, 14]          # the 2-step flight is not a hop
    assert [round(h.air_time_s, 3) for h in hops] == [0.08, 0.1]


def test_flight_still_airborne_at_episode_end_is_not_scored():
    air, nf, pose = _trace("gg" + "aaaa" + "gg" + "aaaa")
    assert len(segment_hops(air, nf, pose, DT, MIN_AIR)) == 1


def test_landing_is_dirty_if_anything_but_feet_touches_before_the_next_liftoff():
    #            0123456789012345
    pattern = "gg" + "aaaa" + "gggg" + "aaaa" + "gg"
    air, nf, pose = _trace(pattern, nonfoot_at=(8,))            # chest after hop 1
    h = segment_hops(air, nf, pose, DT, MIN_AIR)
    assert [x.clean for x in h] == [False, True]
    air, nf, pose = _trace(pattern, nonfoot_at=(3,))            # contact DURING flight 1 is
    assert [x.clean for x in segment_hops(air, nf, pose, DT, MIN_AIR)] == [True, True]  # impossible by definition


def test_forward_is_projected_on_the_liftoff_heading():
    air, nf, pose = _trace("gg" + "aaaa" + "gg", yaw=math.pi / 2)   # moving along +y, facing +y
    (h,) = segment_hops(air, nf, pose, DT, MIN_AIR)
    assert h.forward_m == pytest.approx(0.05)                   # steps 1 -> 6 at 1 cm/step
    pose[:, 2] = 0.0                                            # same motion, facing +x: sideways
    (h,) = segment_hops(air, nf, pose, DT, MIN_AIR)
    assert h.forward_m == pytest.approx(0.0, abs=1e-9)


def test_summary_rates():
    air, nf, pose = _trace("gg" + "aaaa" + "gggg" + "aaaa" + "gg", nonfoot_at=(8,))
    ep1 = EpisodeSummary(segment_hops(air, nf, pose, DT, MIN_AIR), True, 16 * DT)
    ep2 = EpisodeSummary([], False, 16 * DT)
    s = summarize([ep1, ep2])
    assert s["hops"] == 2 and s["episodes_with_a_hop"] == 0.5
    assert s["clean_landing_frac_per_hop"] == 0.5
    assert s["first_hop_clean_frac"] == 0.0 and s["all_hops_clean_episode_frac"] == 0.0
    assert s["ended_upright_frac"] == 0.5
    assert s["hops_per_s"] == pytest.approx(2 / (32 * DT))
