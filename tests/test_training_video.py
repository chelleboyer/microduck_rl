"""Every training run records video, and the video reaches W&B and the HF repo.

Run 3 of the hop (2026-09-28) logged stable_landing_rate 0.98 while every eval
episode face-planted; there was no training video to catch it. Three pieces
make video the default, each locked here:

  * train_hook.default_video_on appends `--video True` to `train` (and so to
    the argv `--hf-jobs` forwards to the job);
  * rsl_rl's Logger uploads each mp4 to W&B (key `video`) — locked below;
  * the HF Jobs uploader mirrors them into the checkpoint repo.
"""

import sys
from pathlib import Path

import pytest

from mjlab_microduck import train_hook

_TASK = "Mjlab-Hop-Flat-MicroDuck"


# ── the default flag ────────────────────────────────────────────────────────

def _argv_after(monkeypatch, argv, env=None):
    monkeypatch.setattr(sys, "argv", list(argv))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    train_hook.default_video_on()
    return sys.argv


def test_train_records_video_by_default(monkeypatch):
    argv = _argv_after(monkeypatch, ["train", _TASK, "--env.scene.num-envs", "4096"])
    assert argv[-2:] == ["--video", "True"]      # FlagConversionOff: a bare --video is not a flag


def test_the_flag_is_forwarded_to_hf_jobs(monkeypatch):
    seen = {}
    monkeypatch.setattr("mjlab_microduck.hf_jobs.submit", lambda a: seen.setdefault("argv", a) and 0)
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--hf-jobs"])
    train_hook.default_video_on()
    with pytest.raises(SystemExit):
        train_hook.maybe_submit_to_hf_jobs()
    assert seen["argv"][-2:] == ["--video", "True"]


@pytest.mark.parametrize("explicit", [["--video", "False"], ["--video", "True"], ["--video=False"]])
def test_an_explicit_choice_is_kept(monkeypatch, explicit):
    argv = _argv_after(monkeypatch, ["train", _TASK, *explicit])
    assert argv == ["train", _TASK, *explicit]


def test_video_length_is_not_mistaken_for_the_flag(monkeypatch):
    argv = _argv_after(monkeypatch, ["train", _TASK, "--video-length", "300"])
    assert argv[-2:] == ["--video", "True"]


def test_env_var_opts_out(monkeypatch):
    argv = _argv_after(monkeypatch, ["train", _TASK], env={"MICRODUCK_NO_VIDEO": "1"})
    assert argv == ["train", _TASK]


@pytest.mark.parametrize("argv", [["/x/.venv/bin/play", _TASK], ["train", "--help"], ["train"]])
def test_other_commands_and_help_are_untouched(monkeypatch, argv):
    assert _argv_after(monkeypatch, argv) == argv


# ── W&B upload is rsl_rl's own ──────────────────────────────────────────────

def test_rsl_rl_still_uploads_training_videos_to_wandb():
    """We rely on rsl_rl's Logger.log uploading every *.mp4 under the run dir
    (WandbSummaryWriter.save_video, once per filename). A wrapper that did the
    same thing doubled every clip in W&B (run 8gnv2koa, 2026-09-28) — if this
    ever fails after an rsl_rl upgrade, the videos silently stop reaching W&B."""
    import inspect

    from rsl_rl.utils import logger, wandb_utils

    assert "save_video" in inspect.getsource(logger.Logger.log)
    assert hasattr(wandb_utils.WandbSummaryWriter, "save_video")


# ── HF Jobs uploader ────────────────────────────────────────────────────────

def test_uploader_mirrors_training_videos():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "hf" / "uploader.py").read_text()
    assert 'root.glob("**/videos/**/*.mp4")' in src
