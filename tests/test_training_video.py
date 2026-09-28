"""Every training run records video, and the video reaches W&B and the HF repo.

Run 3 of the hop (2026-09-28) logged stable_landing_rate 0.98 while every eval
episode face-planted; there was no training video to catch it. Three pieces
make video the default, each locked here:

  * train_hook.default_video_on appends `--video True` to `train` (and so to
    the argv `--hf-jobs` forwards to the job);
  * MicroduckOnPolicyRunner pushes each finished mp4 to W&B (mjlab/rsl_rl
    only write them to disk);
  * the HF Jobs uploader mirrors them into the checkpoint repo.
"""

import sys
import types
from pathlib import Path

import pytest

from mjlab_microduck import train_hook
from mjlab_microduck.tasks import _log_training_videos_to_wandb

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


# ── W&B push from the runner's logger ───────────────────────────────────────

class _Logger:
    def __init__(self, log_dir, logger_type="wandb"):
        self.log_dir = str(log_dir)
        self.logger_type = logger_type
        self.calls = 0

    def log(self, it, *args, **kwargs):
        self.calls += 1


@pytest.fixture
def fake_wandb(monkeypatch):
    logged = []
    mod = types.SimpleNamespace(
        run=object(),
        Video=lambda path, format, caption: ("video", Path(path).name, format, caption),
        log=lambda data, step: logged.append((step, data)),
    )
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod, logged


def test_each_new_video_is_logged_once_at_its_iteration(tmp_path, fake_wandb):
    _, logged = fake_wandb
    videos = tmp_path / "videos" / "train"
    videos.mkdir(parents=True)
    logger = _Logger(tmp_path)
    _log_training_videos_to_wandb(logger)

    logger.log(it=0)                                  # nothing recorded yet
    assert logged == []
    (videos / "rl-video-step-0.mp4").write_bytes(b"x")
    logger.log(it=1)
    logger.log(it=2)                                  # same file: not re-logged
    (videos / "rl-video-step-2000.mp4").write_bytes(b"x")
    logger.log(84, 0, 1000)                           # positional `it`, as rsl_rl may call it
    assert logger.calls == 4                          # the original log still runs every time
    assert [(s, d["Video/train"][1]) for s, d in logged] == [
        (1, "rl-video-step-0.mp4"), (84, "rl-video-step-2000.mp4")]


def test_non_wandb_loggers_and_upload_errors_never_break_training(tmp_path, fake_wandb, capsys):
    mod, logged = fake_wandb
    videos = tmp_path / "videos" / "train"
    videos.mkdir(parents=True)
    (videos / "rl-video-step-0.mp4").write_bytes(b"x")

    tb = _Logger(tmp_path, logger_type="tensorboard")
    _log_training_videos_to_wandb(tb)
    tb.log(it=1)
    assert logged == []

    def boom(data, step):
        raise RuntimeError("network down")
    mod.log = boom
    wb = _Logger(tmp_path)
    _log_training_videos_to_wandb(wb)
    wb.log(it=1)                                      # must not raise
    assert "could not log rl-video-step-0.mp4" in capsys.readouterr().out


# ── HF Jobs uploader ────────────────────────────────────────────────────────

def test_uploader_mirrors_training_videos():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "hf" / "uploader.py").read_text()
    assert 'root.glob("**/videos/**/*.mp4")' in src
