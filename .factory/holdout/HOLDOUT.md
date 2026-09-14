# Holdout scenarios

Independent composed scenarios for microduck_rl. They combine what the journeys check
separately, and they target the failure modes this repository has actually had — the ones
recorded in `AGENTS.md` and in the hop plan's defect list.

**On isolation:** this file sits in the builder's checkout, so it is readable by the thing
it judges. That makes it a floor, not a hidden holdout. Real isolation means holding these
outside the builder's accessible checkout and giving them only to the verification
environment. The directory name does not do it.

## A registered task is a constructible task

1. List the registry with `uv run list-envs` and take every MicroDuck task id.
2. For each, the module that registers it imports and its cfg constructs on CPU.
3. Every id in the table resolves. No id lists but fails to build; no module builds a cfg
   that is registered under a different id than the one printed.

**What this is really checking:** that the table is not a hardcoded list drifting away from
what the code registers. A registry that lists a task nobody can construct costs a GPU
booking to discover.

## A reward term at weight zero is still visibly a reward term

1. Take a task cfg with a `cfg.metrics` block.
2. Set a reward term's weight to zero.
3. The cfg still constructs, the term is still present in the reward manager, and the
   `cfg.metrics` series that reports the same physical quantity still reports it.

**What this is really checking:** the footgun `AGENTS.md` names outright — `Episode_Reward/<term>`
logs the WEIGHTED value, so a term at weight 0 reads 0 regardless of behavior. That is how
the inert-sensor bug hid for a full run. A metrics block that goes quiet when a weight goes
to zero has reproduced the bug it exists to prevent.

## Physical constants agree with the model they describe

1. For each task cfg that carries a standing-height constant, read it.
2. Compile the model that cfg actually uses and settle the robot from its HOME frame.
3. The constant and the settled height agree within 1 mm, or the cfg carries a comment
   naming the measurement and the discrepancy.

**What this is really checking:** the live example. The hop cfg inherited `STAND_Z = 0.115`
while measurement said 0.1172 — 2.2 mm, scoring three separate reward terms. A constant
that no longer matches its model is not a rounding error, it is a silent mis-specification
of the objective.

## Nothing in the test path can spend money

1. Run the full check path: `python harness/ci.py`.
2. No Hugging Face Job is submitted, no `--hf-jobs` flag is passed, no GPU is provisioned,
   and no network call is made to a paid API.
3. `python .factory/policy.py selftest` passes, and a proposed invocation carrying
   `--hf-jobs` is refused.

**What this is really checking:** that the cheap gate stays cheap. A check that quietly
became a training run would be discovered on a bill.
