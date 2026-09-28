"""Keep `train <task> ... --hf-jobs` working, whoever owns the `train` script.

The flag used to live in a `train` console script of our own, declared in
`[project.scripts]` and documented as "shadowing" mjlab's. It shadows nothing:
mjlab 1.3.0 declares `train` too, two distributions declaring the SAME script
name is last-writer-wins at install time, and mjlab won — `uv sync` left
`mjlab.scripts.train:main` in `.venv/bin/train`, so our wrapper was never
invoked and `uv run train ... --hf-jobs` died on tyro's
`Unrecognized options: --hf-jobs` (2026-08-31). Nothing warns about it: the
install succeeds and the flag silently disappears.

So the flag is not implemented in a console script at all any more. It is
intercepted here, from the `mjlab.tasks` plugin entry point: mjlab's own
`mjlab/__init__.py` calls `_import_registered_packages()` at module scope,
which imports `mjlab_microduck.tasks` — and mjlab's `train` reaches that while
executing `from mjlab.scripts.train import main`, i.e. before its two-stage
tyro parse ever sees argv. That path is mjlab's own, so no install order can
take it away from us.

`uv run scripts/hf/train_hf.py <task> ...` calls `submit()` directly and stays
the escape hatch if this interception ever stops firing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FLAG = "--hf-jobs"

#: Set on the job's environment by ``hf_jobs.submit`` — inside the job,
#: ``uv run train`` must always mean "train locally".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"


def _invoked_as_train() -> bool:
    """True when argv[0] is mjlab's trainer (console script or `-m`).

    `play --hf-jobs` must NOT submit a training job; let that command's own
    parser reject the flag instead.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


#: Opt-out for the video default below, e.g. a CPU box with no EGL/GL at all.
_NO_VIDEO_ENV = "MICRODUCK_NO_VIDEO"


def default_video_on() -> None:
    """Make `train` record videos unless the caller said otherwise.

    mjlab's TrainConfig defaults `video=False`, and it lives in site-packages,
    so the default is flipped here, on the same import path as --hf-jobs.
    Every run should leave something to WATCH: AGENTS.md — "Sim metrics can
    pass while the video fails the human eye." Run 3 of the hop (2026-09-28)
    logged stable_landing_rate 0.98 while every eval episode face-planted,
    and there was no training video to catch it.

    Runs BEFORE maybe_submit_to_hf_jobs, so the flag rides along in the argv
    that `--hf-jobs` forwards to the job; inside the job the flag is already
    present and this is a no-op. Opt out with an explicit `--video False` or
    MICRODUCK_NO_VIDEO=1. Syntax is `--video True`: mjlab parses with
    tyro.conf.FlagConversionOff, so a bare `--video` is not a flag.
    """
    if not _invoked_as_train() or os.environ.get(_NO_VIDEO_ENV) == "1":
        return
    args = sys.argv[1:]
    if any(a == "--video" or a.startswith("--video=") for a in args):
        return
    if len(args) == 0 or args[0].startswith("-"):
        return  # `train --help` or no task: leave mjlab's own CLI errors alone
    sys.argv[1:] = [*args, "--video", "True"]


def maybe_submit_to_hf_jobs() -> None:
    """Consume `--hf-jobs` and exit the process; a no-op without the flag.

    Called at import time of `mjlab_microduck.tasks`, so it runs inside mjlab's
    plugin loader. `SystemExit` is a `BaseException`, so it propagates through
    the loader's `except Exception` and out of `import mjlab` — the local
    trainer never starts.
    """
    if _FLAG not in sys.argv[1:]:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return

    from mjlab_microduck.hf_jobs import submit

    sys.exit(submit([a for a in sys.argv[1:] if a != _FLAG]))
