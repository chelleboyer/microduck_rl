# Runtime scenarios

What works on `develop` today, verified on 2026-09-14 rather than assumed. Every journey
here is CPU-only and free — the expensive check is a training run, and that is not the
factory's to run.

A journey describes what the repository **does today**. The hop env is not here because
it lives on `feat/hop-env-training` and is not on `develop`; it joins this file when it
merges.

## The task registry resolves

1. Run `uv run list-envs`.
2. A table of registered tasks prints, and it includes the MicroDuck families — velocity,
   velstand, standup, spin, ball-kick, swizzle — each with its `-Backlash-` twin where one
   is registered.
3. The command exits 0.

**What would make this fail:** a task module that raises on import, a registration that
silently drops a task so the table is short, or an entry whose id does not match the
module that registers it. A task that cannot be listed cannot be trained.

## The regression suite is green without a GPU

1. Run `uv run --with pytest pytest tests/ -q`.
2. Every test passes, and the count is reported — 225 passed, 1 skipped on `develop` at
   `cb70b79`; 237 passed on `feat/hop-env-training`.
3. No test requires CUDA, a display, or a network call to a paid service.

**What would make this fail:** a suite that collects zero tests and exits 0, a test that
needs a GPU and silently skips into a green result, or a cfg-invariant test that passes
because the invariant it checks was deleted alongside it.

## The task package imports on a machine with no GPU

1. Run `uv run python -c "import mjlab_microduck.tasks; print('TASKS_IMPORT_OK')"`.
2. The mdp patch banner prints — NaN-safe reward/advantage, the ONNX `passive_*` filter,
   the warm-start curriculum restart — and then `TASKS_IMPORT_OK`.
3. The command exits 0.

**What would make this fail:** an import that reaches for `torch.cuda` at module scope, a
patch that stops applying while its banner still prints, or a new dependency that only
resolves on the training host. This machine has no CUDA device, so an import that needs one
is a change nobody here can test before paying for it.
