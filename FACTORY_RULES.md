# Project workflow guidance

Keep project constraints and irreversible-action requirements in this file and the
native project guidance. Shared workflow gates own authorization. Factory does not
interpret this document as a private merge policy.

Preserve scope, tests, security invariants and secrets. Report failed or unavailable
checks truthfully. Ordinary static/unit success does not replace required runtime
or holdout verification. Each required assertion needs observable evidence.

Specify this application's protected paths, required validation coverage and
publication constraints in the shared workflow's supported inputs. Preserving old
configuration during upgrade does not prove those rules remain enforced.

See [migration and integration requirements](factory/MIGRATION.md). No autonomy
level or locally written receipt permits bypassing an Archon gate.

---

## This repository's constraints

**`AGENTS.md` wins on every engineering question.** It is the conventions file (reached
through `CLAUDE.md`), it is far more specific than anything here, and it was written from
failures that cost real debugging weeks. Read its "Invariants — do not break these",
"Reward design", "Curricula" and "Sim2real footguns" sections before changing behavior
code. This file covers only process.

**Protected paths.** `MISSION.md`, `FACTORY_RULES.md`, `AGENTS.md`, `CLAUDE.md`,
`harness/`, `.factory/`, `.github/`, and any `.env`. Also every MJCF under
`src/mjlab_microduck/robot/microduck/`: those are Onshape exports, and hand-editing a
generated model is how a physics change enters disguised as a fix.

## The two gates

**Money.** The factory never launches training. Not a full run, and not the 64-env smoke
test either — a smoke test is a training run, and this machine has no CUDA device, so
every run is a Hugging Face Job that bills. The factory prepares the run, argues for it,
and writes the exact command in the PR body. A person types it.

```bash
python .factory/policy.py check-command "<the command>"
python .factory/policy.py selftest              # POLICY_SELFTEST_PASSED cases=15
```

**Merge.** The factory never merges and never approves. `archon-merge-queue` and
`archon-deploy` are never invoked; `archon-lifecycle` only in full preview with every
gating input passed explicitly. Same rules, same reasoning, as the duckfactory
installation.

Neither guard is a sandbox. Nothing stops a person typing the command. They exist so that
breaching a rule is a deliberate act rather than a forgotten default.

## Evidence rules

**A physical claim carries its measurement.** Any change that asserts the robot does
something names the command that produced the number. `STAND_Z` is the cautionary example:
a cfg inherited 0.115, measurement said 0.1172, and three reward terms scored against the
wrong reference until someone measured.

**A reward term added on a hypothesis says so.** "This should help the policy discover
liftoff" is a hypothesis and belongs in the PR body as one. It becomes a finding when a
run shows it.

**Counts are evidence, not thresholds.** `harness/ci.py` fails a suite that discovers zero
tests, and reports the count otherwise. `.factory/locks/floor.json` carries no
`unit_tests` key — a count floor punishes consolidating a suite and punishes tests that
skip under some conditions.

**A run's result is read before the next run is proposed.** The factory may analyze logs
and metrics and propose the next iteration. It may not propose a fourth run while the
third is unexamined.

## Scope

One issue per PR. A bug noticed while fixing another becomes a new issue. Tests are never
weakened to make a build pass; a test believed wrong is called out in the PR body and
defended. Never hardcode joint indices — resolve through `_servo_joint_ids`, per
`AGENTS.md`, or the backlash model silently goes wrong.
