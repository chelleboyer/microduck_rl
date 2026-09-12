# Colab training (viability test)

Alternative to [`--hf-jobs`](../hf/README.md) for running training on a GPU,
using Colab compute you've already paid for instead of Hugging Face Jobs
billing. Not integrated into `train`/`hf_jobs.py` — this is a manual notebook
runbook, not a submission path, because Colab has no equivalent of HF Jobs'
`HfApi.run_job`: no headless submission, no persistent container, no
auto-resubmit on preemption.

## Status: unproven

Nothing here has been run yet. Upload `train_on_colab.ipynb` directly (File >
Upload notebook) rather than opening it from GitHub — the notebook itself
lives on `feat/colab-training-viability`, but it clones `feat/hop-env-training`
(§2), which is where the actual hop-task fixes are. Work through it top to
bottom. The smoke-test cell (§6, a stock task, not hop) is the actual
viability question — if MuJoCo Warp + CUDA torch don't come up cleanly on
whatever GPU tier Colab hands you, this path isn't viable and the notebook
stops being useful past that point. §7's default run targets
`Mjlab-Hop-Flat-MicroDuck` with both reward-hack fixes in place.

## Why this might not be worth it even if it works

- No `--detach`: the run dies the moment the tab/runtime disconnects. HF
  Jobs' background container is what actually makes multi-hour training
  practical; Colab's paid GPU-hours don't include that.
- No `uploader.py` equivalent pushing checkpoints in the background — this
  notebook relies on `train`'s own wandb logging instead, so a run that dies
  mid-way is only as recoverable as its last wandb checkpoint.
- Manual, one run at a time, one human watching a tab — fine for a quick
  test, not a replacement for submitting several runs and walking away.

## Secrets

Needs `GITHUB_TOKEN` (repo-scope PAT, this repo is private) and
`WANDB_API_KEY`. Set both as Colab secrets (key icon, left sidebar) rather
than pasting into a cell — the notebook checks `google.colab.userdata`
first and falls back to an interactive prompt so it also works outside
Colab.
