# Mission

<!--
  Owner: humans only. Protected; the factory cannot edit it.

  This file says WHAT this repository is for and what it must never become.
  `AGENTS.md` is the conventions file -- how the code is written, the invariants,
  reward-design rules, curricula, sim2real footguns. It already exists, it is far more
  specific than anything written here, and it WINS on every engineering question.
  This file adds only what AGENTS.md does not cover: product scope, and the two gates
  that keep an autonomous agent from spending money or merging its own work.
-->

**Derived from:** `AGENTS.md` and `README.md` (the repository's own documentation)
**Last reconciled with them:** 2026-09-14

## What microduck_rl is

A reinforcement-learning research repository that trains behaviors for the Microduck
robot in MuJoCo, via mjlab. A behavior is an environment config plus its reward stack,
its MDP functions, its curricula and its tests; a trained behavior becomes an ONNX policy
published to the Hub, which the robot's daemon loads. The work is the environment design
and the reward design -- the policy is the output.

Two assumptions shape everything below. Training runs on **rented GPUs and costs money**;
there is no local CUDA device, so every run goes through Hugging Face Jobs. And this
repository is a **fork of `pollen-robotics/microduck_rl`** that tracks upstream: work
lands here, and upstream is a source, not a destination.

## Who it is for

- The person training Microduck behaviors, who needs the environment, the reward stack
  and the measurement harnesses to be correct before spending money on a run.

microduck_rl is not a robot runtime and not a deployment target. Policies leave through
the Hub; nothing here talks to hardware.

## Core capabilities (in scope)

The factory may accept issues in these areas.

**Environments and rewards**
- New task env cfg modules, and changes to existing ones
- MDP functions in `tasks/mdp.py` -- rewards, events, observations, commands, curricula
- Termination and reset design
- `cfg.metrics` blocks that log physical outcomes independently of reward weights

**Measurement and diagnosis**
- CPU measurement harnesses in `scripts/` that establish constants instead of guessing them
- Analysis of a completed run's logs and metrics, and proposals for the next iteration

**Correctness infrastructure**
- Tests in `tests/` -- cfg invariants and MDP-function regressions, CPU-only
- Export and publish path correctness

**Documentation**
- `AGENTS.md`, plans and reports under `.claude/`, keeping recorded state true

## Out of scope -- the factory must never do this

**Spending**
- Launching, queueing or scheduling any training run
- Anything that passes `--hf-jobs`, or otherwise submits a Hugging Face Job
- Provisioning or resizing GPU hardware, or changing the job flavor to spend more

**The robot's physics**
- Changing mass, inertia, friction, damping, gear, armature or joint limits in any MJCF
  to make a behavior easier to learn. Physics describes the robot; it is not a tuning knob
- Adding colliding or massed geometry to a model in the name of measurement. Sites are
  massless and non-colliding and that is exactly why they are the measurement tool

**Upstream**
- Pushing to `pollen-robotics/microduck_rl`
- Rewriting history on any shared branch

**Deployment**
- Publishing a policy to the Hub, or touching anything the robot daemon loads
- Talking to hardware, over any channel

**Scope drift**
- Rewriting the training stack -- swapping mjlab, the RL algorithm, or the actuator model
- Adding a second simulator

## Hard invariants -- not tunable by any issue

1. **Training is human-launched, always.** The factory may prepare a run, argue for it,
   and write down exactly what to type. A person types it. This is a money gate, and it
   does not relax as confidence grows. `.factory/policy.json` enforces it.
2. **A smoke test precedes every long run.** 64 envs, 5 iterations, per `AGENTS.md`:
   "A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents. Never
   launch a long run without one."
3. **Measured constants are measured.** A physical constant in a cfg either carries a
   measurement and its method, or it is a guess wearing a decimal point. `STAND_Z` is the
   standing example: the cfg inherited 0.115 and measurement said 0.1172.
4. **The factory never merges and never approves.** It takes work as far as a reviewed
   pull request and stops.
5. **The factory cannot modify governance files.** `MISSION.md`, `FACTORY_RULES.md` and
   `AGENTS.md` (via `CLAUDE.md`) are the constitution.
6. **The factory cannot modify its own judge.** `harness/`, `.factory/locks/` and
   `.factory/holdout/`. Adding an assertion is always welcome; removing or loosening one
   is a human decision.

## Allowed evolutions

- Adopting a lint gate. There is none today: `ruff check .` reports 452 findings under
  default rules, so this is triage work, and it is in scope when an issue is filed for it.
- Porting mechanisms from verified external policies, with the verification named.
- Promoting a CPU measurement harness into a test.

## Definition of done

**Gate 1 -- the checks pass.** `python harness/ci.py` prints `CHECKS_OK`, with
`UNIT_PASSED tests=N` for N greater than zero. CPU only; no GPU, no spending.

**Gate 2 -- physical claims carry evidence.** Any change asserting the robot does
something states how that was measured, with the command that produces the number. A
reward term added because it "should help" is a hypothesis, and says so.

**Gate 3 -- the runtime journeys pass.** `harness/END-TO-END.md`. They are deliberately
cheap, because the expensive check is a training run and that is not the factory's to run.

## Open questions -- decisions nobody has made yet

The factory may propose an answer, build against it, and record what it assumed; the
merge is then held for a human.

- **Q1** Does this fork ever contribute back upstream, or does it diverge permanently?
- **Q2** Should a lint gate be adopted, and at what rule selection given 452 findings?
- **Q3** What belongs in the harness as a journey versus staying a `scripts/` harness?

**Except these, which stop the factory** -- they spend money or change shared state:

- Anything that would launch a training run
- Anything that would push to upstream or rewrite shared history

## What the factory does NOT own -- permanently human

- **Whether a run is worth its cost.** The judgement that this reward stack has earned a
  GPU. The factory can make the case; it cannot make the call.
- **Whether a policy is good.** Watching the robot move and deciding it looks right.
- **What behavior to teach next.**
- **Whether this is worth merging.** Per hard invariant 4.

The factory owns the layer whose correctness can be asserted without a GPU: env cfgs,
MDP functions, tests, measurement harnesses, and the honesty of recorded state.
