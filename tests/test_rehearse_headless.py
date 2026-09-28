"""Smoke test for scripts/rehearse_headless.py (CPU, no rendering).

Runs the headless deployment rehearsal for 0.5 s on the run-6 hop ONNX when it
is present locally (it lives in a session scratchpad, not in the repo), and
checks the rollout is NaN-free and the policy sees the 61-D obs contract.
"""

import importlib.util
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "rehearse_headless.py")
_ONNX = os.environ.get(
    "MICRODUCK_REHEARSAL_ONNX",
    "/tmp/claude-0/-home-user-microduck-rl/963e0b91-2bfd-5325-b7af-8bdd20647a42/"
    "scratchpad/run6onnx/exported/policy.onnx",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("rehearse_headless", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.exists(_ONNX), reason=f"policy ONNX not found: {_ONNX}")
def test_rehearse_headless_half_second_no_nan():
    mod = _load_script()
    s = mod.run_rehearsal(_ONNX, seconds=0.5, video=None)
    assert s["obs_size"] == 61
    assert not s["nan"], f"NaN at control step {s['nan_step']}"
    assert s["control_steps"] == 25
    assert s["bam"] is True
