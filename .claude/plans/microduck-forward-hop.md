# Feature: Microduck forward hop — attempt 1, runs 3+

The following plan should be complete, but it's important that you validate documentation and
codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files.

## Feature Description

An episodic hop policy for Microduck: from standing, drive both feet off the ground at the same
instant, travel forward through the airborne phase, and land on two feet and recover to standing.
Deployed like sit/standup/roulade — policy switch means the hop starts immediately; no phase clock,
no reference motion. Shares the 61D observation contract with every other policy so the runtime can
hot-swap it.

The env (`src/mjlab_microduck/tasks/microduck_hop_env_cfg.py`) already exists and has had two
training runs, both of which produced exploits rather than hops (butt-bounce, then worming). On
2026-09-13 a review found four defects that each removed a large fraction of the training signal;
all four are fixed and the suite is green at 221 tests. **This plan covers everything from that
state to a policy that meets the acceptance bar.**

## User Story

As a Microduck operator
I want to trigger a forward hop and have the robot land on two feet
So that the robot has a dynamic traversal move alongside walking, standup and the roulade

## Problem Statement

Two runs produced no hop. The proximate causes were not reward-tuning subtleties — they were
structural defects that made the reward function measure something other than hopping:

1. **The reverse curriculum paid nothing.** `reset_hop_state` seeded mid-air spawns at exactly
   `gate_min_air_time`, which is `_hop_completion_gate`'s *zero* point, so all four
   landing/recovery terms evaluated to 0.0000 for every mid-air episode.
2. **One topple killed the rest of the episode.** The butt-bounce taint was sticky per-episode, so
   after any non-foot ground contact the air-time frontier was pinned at 0, the completion gate
   could never open, and the remaining ~2.5 s of a 3.0 s episode had reward
   `-0.1*action_rate - 0.2*self_collisions` — whose argmax is "do nothing".
3. **Forward credit paid for walking.** `hop_forward_progress` measured displacement from the
   *spawn point* and merely sampled it on airborne steps, so "lunge 8 cm on the feet, then stumble
   through four airborne frames" collected the forward reward *and* opened the landing gate.
4. **`UNWEIGHT_FORCE_N` was a guess** (8.0 N vs a measured 7.23 N).

All four are fixed. A fifth defect is **not** fixed and is the first task below:

5. **The mid-air spawn ranges are physically inconsistent with the target hop.** A hop is ballistic
   once both feet leave, so air time and apex are rigidly linked. `MIDAIR_Z` 0.14–0.18 m combined
   with `MIDAIR_VZ` −1.5…−0.5 m/s implies apex heights of 0.153–0.295 m, i.e. hops of **0.175 s to
   0.383 s** of air time — 1.2× to 2.6× `TARGET_AIR_TIME = 0.15 s`. The reverse curriculum has been
   teaching recovery from landings the target hop never produces.

Underlying all five: this is the only env in the repo built without a design doc, and its constants
carry `UNVERIFIED` in the docstring. AGENTS.md step 2 ("verify physics assumptions in sim BEFORE
training") was skipped because the authoring sandbox had no GPU — but the measurements that matter
here are **kinematic and ballistic, and run fine on CPU**.

## Solution Statement

Three moves, in order:

1. **Measure before running.** Derive every constant from the compiled model and from ballistics,
   the way `docs/superpowers/specs/2026-08-04-roller-standup-design.md` does in its
   `## Constantes mesurées` section. Critically, establish whether the XL330 legs can produce the
   takeoff velocity `TARGET_AIR_TIME` demands at all — if they can't, the target is impossible and
   no amount of reward tuning will find it.
2. **Reduce run 3 to one question.** Discover liftoff-and-land *in place*: `hop_forward_progress`
   introduced at weight 0 and ramped in by curriculum once the skill exists — the same pattern the
   file already uses for `gentle_landing` and `hop_no_crawl`, and the AGENTS.md rule that an
   attempt-tax (or here, a second simultaneous objective) during discovery makes "do nothing" win.
   A run 3 failure then means "liftoff is hard", not "one of two things is hard".
3. **Score every run against a measurable bar** with a headless eval battery, because none exists
   today and the acceptance criterion cannot otherwise be checked.

## Out of Scope / Non-Goals

- **Not included: one-foot landing.** v1 lands on two feet. Explicitly a later step.
- **Not included: hop height or distance commanded from the twist slots.** The hop is a fixed
  maneuver triggered by policy switch. Command slots stay zero-padded for 61D parity.
- **Not included: chained / repeated hopping.** One hop per episode.
- **Not included: real-robot deployment.** This plan ends at a checkpoint meeting the sim bar plus
  an ONNX export rehearsal. `uv run publish` and hardware bring-up are a follow-up.
- **Not changing:** the 61D obs layout, the BAM actuator stack, the DR ranges (they are matched to
  standup/roulade for sim2real parity and changing them here would break that parity), or the
  `-Backlash-` variant policy.
- **Not re-litigating:** forward vs in-place as the *product* target. Forward remains the goal; this
  plan only sequences it after liftoff.

## Feature Metadata

**Feature Type**: Bug Fix + Enhancement (env exists, does not train)
**Estimated Complexity**: High — the failure mode is reward specification, and each run costs hours
**Primary Systems Affected**: `tasks/mdp.py` (Hop section), `tasks/microduck_hop_env_cfg.py`,
`tests/test_hop_cfg.py`, new `scripts/measure_hop.py` and `scripts/eval_hop.py`
**Dependencies**: mjlab, MuJoCo Warp (training, GPU), CPU MuJoCo + `bam` (measurement), rsl_rl, wandb

**Compute constraint — load-bearing:** there is **no local CUDA device**
(`torch.cuda.device_count() == 0`, no `nvidia-smi`). Every training run, including the mandatory
64-env/5-iteration smoke test, must go through `--hf-jobs`. Phase 1 measurement is CPU-only and
runs locally.

## Related Work

**Implements**: forward-hop PRD, `docs/ideas/hop-behavior.md` in the `pollen-robotics/microduck` repo
(product direction overridden to *forward* hop by the project owner; see hop cfg header)
**Branch**: `feat/hop-env-training`

**Back-references**:

- `docs/superpowers/specs/2026-08-04-roller-standup-design.md` — Why: the house spec format, and
  the `## Constantes mesurées` discipline this plan's Phase 1 imitates.
- `src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py` — Why: the hop is a structural copy of
  the roulade (progress frontier → completion gate → landing composite + bootstrap layers +
  stand-tax, with a mid-maneuver reverse curriculum). Its cfg docstring encodes a five-run lesson
  arc; the hop is on run 2 of the same shape.

**Forward-references**: (none yet)

**Note on location**: this plan lives in `.claude/plans/` per the user's explicit choice. The seven
existing envs use `docs/superpowers/specs/<date>-<name>-design.md` +
`docs/superpowers/plans/<date>-<name>.md`, in French. This is a deliberate divergence, not an
oversight.

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING

- `AGENTS.md` (whole file) — Why: the reward-design and curriculum rules below are all quoted from
  it. In particular "Reward design", "Curricula" and "Training ops & reading a run".
- `src/mjlab_microduck/tasks/mdp.py` **§ Hop**, from the `# ── Hop` banner (~line 7191) to
  `hop_no_crawl_penalty` — Why: the entire mechanism. Read `_update_hop_clean_time`,
  `_hop_clean_air_budget`, `_hop_completion_gate` and `_update_hop_forward_accum` docstrings in
  full; they record exactly which exploit each guard closes and why.
- `src/mjlab_microduck/tasks/microduck_hop_env_cfg.py` (whole file) — Why: the cfg under change.
  The header docstring is a running log of course corrections; append to it, don't rewrite it.
- `src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py` (whole file) — Why: the template. Every
  hop reward has a roulade counterpart; when unsure what shape something should take, mirror it.
- `src/mjlab_microduck/tasks/mdp.py` — `reward_weight` and `event_param_curriculum` — Why: the two
  curriculum mechanisms. `reward_weight` is a **step function, not an interpolation**.
- `scripts/infer_policy.py` lines 28–100 (`load_bam_model`, `load_mujoco_with_bam`) — Why: the
  **CPU BAM harness**. Phase 1 measurement must reuse this, not the XML PD actuators, or the
  measured push-off will not reflect training physics.
- `tests/test_hop_cfg.py` (whole file) — Why: the test conventions, including the stub-env pattern
  (`_stub_env`, `_forward_env`, `_FakeScene`) added on 2026-09-13 for testing mdp bookkeeping
  without a GPU.
- `tests/test_roller_standup_cfg.py` — Why: the pattern for asserting joint indices resolve against
  the *actual compiled model*.

### New Files to Create

- `scripts/measure_hop.py` — CPU MuJoCo + BAM measurement harness producing every constant below.
- `scripts/eval_hop.py` — headless checkpoint eval battery; per-spawn-type, reports the acceptance
  rate and an end-state cluster breakdown.
- `tests/test_hop_measurements.py` — locks the measured constants against the compiled model so a
  model revision can't silently invalidate them.

### Relevant Documentation

- [MuJoCo computation / actuators](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation)
  — Why: confirms how `ctrl` maps to force for the voltage-controlled BAM path in Task 1.3.
- [rsl_rl PPO runner](https://github.com/leggedrobotics/rsl_rl) — Why: `--agent.load-checkpoint` /
  `--agent.resume` semantics for the resume-based ladder in Phase 3.
- AGENTS.md is the primary reference and outranks any external source on this repo's conventions.

### Patterns to Follow

**Penalty sign convention** (AGENTS.md, "bit four envs"): mjlab-base cost functions return ≥ 0 and
take a **negative** weight; self-negating microduck functions (`*_penalty`, `*_tax`, returning ≤ 0)
take a **POSITIVE** weight. In the hop cfg today: `hop_no_crawl` is a cost → `-0.5`;
`gentle_landing` and `hop_stand_tax` are self-negating → `+0.002` / `+5.0`. The infallible check on
every run: **every `Episode_Reward/<penalty>` in wandb must be ≤ 0.**

**Curriculum**: `microduck_mdp.reward_weight` for weight schedules, `event_param_curriculum` for
event ranges. Steps are env steps — `iteration × 24` (`NUM_STEPS_PER_ENV = 24`). Mutate term cfgs
via the managers (`env.event_manager.get_term_cfg(...)`), never `env.cfg.events[...]` — managers
deepcopy at init so writes to `env.cfg` are silent no-ops.

**Joint indices**: never hardcode. Use `_servo_joint_ids` / `_servo_joint_pos` in mdp.py. The hop
runs on the walk model where ctrl idx == joint idx, but the helpers are identity there and correct
everywhere else.

**Step-guarded accumulators**: any per-episode state that *integrates* must guard on
`env.common_step_counter` so multiple reward terms reading it in one control step don't
double-advance it (`_update_hop_clean_time`, `_update_hop_accum`, `_update_hop_forward_accum`).
Idempotent updates (a max, an OR) don't need the guard.

---

## IMPLEMENTATION PLAN

### Phase 1: Measure (CPU, local, no GPU)

The AGENTS.md step-2 work that was skipped. Everything here runs locally. **Gating**: Task 1.3's
result can invalidate `TARGET_AIR_TIME` and force a target change before any run.

### Phase 2: Re-derive constants and reduce run 3 to one question

**Depends on:** Phase 1 (constants come from its output).

Set the mid-air spawn ranges from measured ballistics, gate the forward objective behind a
curriculum, and add the eval battery.

### Phase 3: Run 3 — discover liftoff in place

**Depends on:** Phase 2. Smoke test, then a 1000-iteration probe before committing 6000.

### Phase 4: Run 4+ — turn on forward, then polish

**Depends on:** Phase 3 clearing its bar. One variable set per run.

---

## STEP-BY-STEP TASKS

Execute in order. Each task is atomic and independently testable.

### CREATE `scripts/measure_hop.py`

- **IMPLEMENT**: CPU MuJoCo measurement harness with subcommands `settle`, `heights`, `pushoff`,
  `ranges`. Loads the standup robot spec via
  `MICRODUCK_STANDUP_ROBOT_CFG.spec_fn().compile()` and drives it with the **BAM** actuators from
  `scripts/infer_policy.py`, not the XML PD.
- **PATTERN**: `scripts/infer_policy.py:48` `load_bam_model`, `:58` `load_mujoco_with_bam`.
  Argument parsing mirrors `scripts/infer_policy.py`'s argparse block.
- **IMPORTS**: `mujoco`, `numpy`, `from bam.model import load_model`,
  `from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG`
- **GOTCHA**: importing `microduck_constants` directly triggers a circular-import warning through
  the mjlab task registry; import the module (`from mjlab_microduck.robot import
  microduck_constants as mc`) rather than the symbol, as this session verified.
- **GOTCHA**: `EntityCfg` has no `.spec` attribute — it is `.spec_fn()`, which returns the spec to
  compile.
- **VALIDATE**: `uv run scripts/measure_hop.py --help`
- **SATISFIES**: AC #1

### ADD `settle` subcommand to `scripts/measure_hop.py`

- **IMPLEMENT**: Hold HOME ctrl for 3 s from 32 noisy initial states (joint noise σ=0.08 rad, tilt
  ±5°, z ∈ [0.11, 0.12] — the current `set_hop_state` standing bucket). Record final trunk z **and
  final tilt angle**. Report the fraction that end upright (tilt ≤ 15°) and the mean/σ of settled z.
- **GOTCHA**: AGENTS.md — "a settle test that only records z reports fallen states as 'resting
  fine'". Tilt is mandatory, not optional.
- **DECISION POINT**: if the standing spawn is not a stable equilibrium, the standing bucket is
  starting episodes in a state the robot must first *recover* from, which corrupts every liftoff
  measurement. Fix the spawn before continuing.
- **VALIDATE**: `uv run scripts/measure_hop.py settle` — prints upright fraction and settled z
- **SATISFIES**: AC #1

### ADD `heights` subcommand to `scripts/measure_hop.py`

- **IMPLEMENT**: Exact-kinematics trunk heights: minimum vertex of the collidable geoms with the
  trunk brought to contact, for the HOME/STAND pose and for a maximal crouch. Confirms or replaces
  `STAND_Z = 0.115`, which the hop cfg **inherited from another env rather than measuring**, and
  yields the crouch depth Task 1.3 needs.
- **PATTERN**: `docs/superpowers/specs/2026-08-04-roller-standup-design.md` `## Constantes mesurées`
  — that env measured `debout 0.1407, ventre 0.0752, dos 0.0475` this way and cross-checked against
  the under-load value, finding ~2 mm of sag.
- **GOTCHA**: AGENTS.md — "Measure target heights off the actual robot in sim, never carry them
  across model revisions. A 5 mm-wrong STAND_Z once turned the goal into an impossible target."
- **VALIDATE**: `uv run scripts/measure_hop.py heights`
- **SATISFIES**: AC #1

### ADD `pushoff` subcommand to `scripts/measure_hop.py` — THE GATING MEASUREMENT

- **IMPLEMENT**: From the measured crouch, command a maximal coordinated leg extension (ankle, knee,
  hip_pitch to their extension limits) under BAM and measure **actual takeoff vertical velocity,
  peak apex, and simultaneous both-feet air time**. Sweep a few extension profiles; report the best
  achievable air time `T_max`.
- **WHY THIS IS GATING**: a hop is ballistic once airborne, so `T = 2·v_z0/g`. The current
  `TARGET_AIR_TIME = 0.15 s` demands `v_z0 = 0.736 m/s`, an apex rise of 27.6 mm, and roughly 0.20 J
  of vertical kinetic energy on 0.7372 kg. Whether 14 XL330s in this leg geometry can deliver that
  is **unknown and never checked**. If `T_max < 0.15 s`, `TARGET_AIR_TIME` is an impossible target
  and two runs of "the policy never hops" are explained without any reference to rewards.
- **GOTCHA**: measure under BAM with training's voltage DR range (`BAM_VIN_RANGE = (6.5, 8.2)`),
  and at the **low** end too — a hop that only works at 8.2 V will not transfer.
- **VALIDATE**: `uv run scripts/measure_hop.py pushoff` — prints `T_max`, `v_z0`, apex, per-voltage
- **SATISFIES**: AC #1, AC #2

### UPDATE `TARGET_AIR_TIME` / `HOP_MIN_AIR_TIME` in `microduck_hop_env_cfg.py`

- **IMPLEMENT**: Set `TARGET_AIR_TIME` to a value the `pushoff` measurement shows is reachable —
  AGENTS.md's Gaussian-std logic applied to a target: aim at what the robot can actually do, with
  headroom, not at an aspiration. Suggested rule: `TARGET_AIR_TIME = 0.75 · T_max`, and
  `HOP_MIN_AIR_TIME = 0.4 · TARGET_AIR_TIME`. Record the measured `T_max` in the comment.
- **GOTCHA**: `HOP_MIN_AIR_TIME` feeds both the gate and the mid-air seed
  (`gate_min_air_time × mdp._HOP_GATE_FULL_OPEN`). Changing it moves both — that coupling is
  intentional and locked by `test_hop_midair_spawn_opens_the_landing_gate`.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q`
- **SATISFIES**: AC #2

### UPDATE `MIDAIR_Z_MIN/MAX`, `MIDAIR_VZ_RANGE`, `MIDAIR_VX_RANGE` — defect #5

- **IMPLEMENT**: Derive the mid-air spawn from the ballistics of the *target* hop, so the reverse
  curriculum practices the landing the robot is actually being taught to produce. For a hop of air
  time `T`, apex rise is `g·T²/8` above `STAND_Z` and touchdown speed is `g·T/2`. Spawning at a
  random point in the descent of hops spanning `T ∈ [0.67·T_target, 1.33·T_target]` gives:
  - `MIDAIR_Z ∈ [STAND_Z + small, STAND_Z + g·(1.33·T)²/8]`
  - `MIDAIR_VZ ∈ [−g·(1.33·T)/2, −0.2]`
  - `MIDAIR_VX ∈ (0.0, TARGET_FORWARD_DIST / T_target)`

  For `T_target = 0.15 s` and `TARGET_FORWARD_DIST = 0.05 m` that is `MIDAIR_Z ≈ [0.118, 0.155]`,
  `MIDAIR_VZ ≈ (−1.0, −0.2)`, `MIDAIR_VX ≈ (0.0, 0.35)`. **Recompute from the measured `T`.**
- **GOTCHA**: the current values (`z` 0.14–0.18, `vz` −1.5…−0.5) imply apexes of 0.153–0.295 m, i.e.
  hops of **0.175–0.383 s** — 1.2× to 2.6× the 0.15 s target. Verified analytically this session.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q`
- **SATISFIES**: AC #2

### CREATE `tests/test_hop_measurements.py`

- **IMPLEMENT**: Lock the derived constants to the model and to each other:
  (a) `STAND_Z` is within 3 mm of the kinematic standing height of the compiled model;
  (b) `MIDAIR_Z_MAX` does not exceed the apex of a `1.4 × TARGET_AIR_TIME` hop;
  (c) `abs(MIDAIR_VZ_RANGE[0])` does not exceed the touchdown speed of a `1.4 × TARGET_AIR_TIME`
  hop; (d) `MIDAIR_VX_RANGE[1] · TARGET_AIR_TIME` is within 1.5× of `TARGET_FORWARD_DIST`.
- **PATTERN**: `tests/test_hop_cfg.py::test_ground_sensor_covers_every_nonfoot_collision_body`
  asserts against the compiled model — mirror that setup.
- **WHY**: makes defect #5 a class of bug that cannot silently recur, the same way the ground-sensor
  test made the inert-sensor bug non-recurrable.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_measurements.py -q`
- **SATISFIES**: AC #2, AC #7

### UPDATE `hop_forward_progress` weight + add `forward_weight` curriculum

- **IMPLEMENT**: Set `cfg.rewards["hop_forward_progress"].weight = 0.0` and add a
  `cfg.curriculum["forward_weight"]` `reward_weight` term with stages
  `[{step: 0, weight: 0.0}, {step: 1500*24, weight: 2.5}, {step: 2500*24, weight: 5.0}]`.
  Keep the term registered at weight 0 from step 0 — do not delete it.
- **WHY**: run 3 must answer one question. AGENTS.md: "any attempt-tax active while a hard skill is
  being explored makes 'do nothing' win", and "Phase-align every stage with what the policy has
  actually learned."
- **GOTCHA**: `Episode_Reward/hop_forward_progress` logs the **weighted** value, so it reads 0.0
  while the weight is 0 regardless of behavior. Interpret it against the schedule, not in isolation.
  This is exactly how the inert-sensor bug hid for a full run.
- **GOTCHA**: the stage steps here are placeholders — Phase 4 re-times them against where run 3
  actually discovers liftoff. Do not treat them as tuned.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q`
- **SATISFIES**: AC #3

### ADD test: forward objective is gated behind liftoff

- **IMPLEMENT**: Assert `cfg.rewards["hop_forward_progress"].weight == 0.0` at construction, that a
  `forward_weight` curriculum exists, that its first stage is weight 0, that its final weight is
  > 0, and that it starts no earlier than the `action_rate_weight` ramp.
- **PATTERN**: `tests/test_hop_cfg.py::test_hop_no_crawl_ramps_in_after_torque_and_landing_polish`
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q`
- **SATISFIES**: AC #3, AC #7

### CREATE `scripts/eval_hop.py`

- **IMPLEMENT**: Headless battery over a checkpoint. Runs N episodes (default 512) split by spawn
  type, and for each records: peak simultaneous air time, peak clean forward displacement from
  liftoff, whether any non-foot body touched the ground, and at `landing + 0.5 s` the trunk z, tilt,
  and both-feet contact. Reports the **acceptance rate** (AC #4's predicate) plus an end-state
  cluster breakdown (standing / prone-front / prone-back / side / never-lifted).
- **PATTERN**: `scripts/play_latest.py` and `scripts/wandb_utils.py` for checkpoint resolution from
  `--wandb-run-path`; `scripts/export.py` for the task-id → cfg path.
- **GOTCHA**: AGENTS.md — to force spawn states you must go through
  `env.event_manager.get_term_cfg("set_hop_state")`, **not** `env.cfg.events[...]`, which is a
  silent no-op because managers deepcopy their cfg at init. This has bitten eval scripts before.
- **GOTCHA**: needs a CUDA device → runs via `--hf-jobs` or wherever the checkpoint was trained.
- **WHY**: AC #4 cannot be checked without it, and AGENTS.md's "Measure before theorizing" requires
  it before any reward change in Phase 4.
- **VALIDATE**: `uv run scripts/eval_hop.py --help`
- **SATISFIES**: AC #4

### UPDATE the header docstring of `microduck_hop_env_cfg.py`

- **IMPLEMENT**: Append a "Course correction 4" block recording the measured constants with their
  derivations, and replace the file-level `UNVERIFIED, run 1` paragraph — the numbers are now
  measured, and leaving that paragraph in place would misrepresent the file's state.
- **PATTERN**: the existing course-correction blocks in the same docstring.
- **VALIDATE**: `uv run --with pytest pytest tests/test_hop_cfg.py -q`
- **SATISFIES**: AC #6

### RUN the smoke test — MANDATORY GATE

- **IMPLEMENT**: `uv run train Mjlab-Hop-MicroDuck --env.scene.num-envs 64
  --agent.max_iterations 5 --hf-jobs`
- **WHY**: AGENTS.md — "A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents.
  Never launch a long run without one." This env has never had one; it is where tensor-shape and
  physics-assumption bugs surface.
- **CHECK**: builds; steps NaN-free; obs is 61D; every reward term computes; ONNX exports.
- **VALIDATE**: job completes 5 iterations with no NaN and a finite mean reward
- **SATISFIES**: AC #5

### RUN 3 — probe, 1000 iterations at 4096 envs

- **IMPLEMENT**: `uv run train Mjlab-Hop-MicroDuck --env.scene.num-envs 4096
  --agent.max_iterations 1000 --hf-jobs`
- **WHY 1000, NOT 6000**: AGENTS.md budgets simple episodic tricks at ≈1000 iterations at 4096 envs.
  The question run 3 asks — "is liftoff discoverable at all under fixed rewards and measured
  constants" — is answered well before 6000, and a probe keeps the 6000-iteration commitment for a
  run whose answer is already yes.
- **WATCH IN WANDB, per AGENTS.md "reading a run"**:
  - `Episode_Reward/hop_unweighting` rising in the first ~200 iters — the discovery precursor. Flat
    at zero means liftoff is not being explored at all; stop and revisit `pushoff`.
  - `Episode_Reward/hop_air_time` becoming non-zero — **the main task term**. Total reward can rise
    purely on regularizers while the trick never happens.
  - `Episode_Reward/hop_landing_composite` non-zero from iteration 0 — this is the direct check that
    the gate-seed fix works. **If it reads 0.0000, the mid-air bucket is still dead; stop the run.**
  - Every `Episode_Reward/<penalty>` ≤ 0 — `hop_no_crawl`, `gentle_landing`, `hop_stand_tax`,
    `action_rate_l2`, `self_collisions`, `joint_torque_rate_l2`.
  - `Episode_Reward/hop_forward_progress` **should read exactly 0.0000** until iteration 1500 —
    it is weight-gated. Non-zero before then means the curriculum is misconfigured.
  - Any metric stepping **down** exactly at a curriculum boundary → pacing is wrong; stretch the
    stage or move it later, never earlier.
- **VALIDATE**: `uv run scripts/eval_hop.py --wandb-run-path <...> --checkpoint 1000`
- **SATISFIES**: AC #4, AC #5

### EVALUATE run 3 and branch

- **IMPLEMENT**: Run the battery. Watch the video as well as the metrics — AGENTS.md: "Sim metrics
  can pass while the video fails the human eye — watch the video AND check which geom/axis touches."
  Report what rollouts actually show ("lifts off but face-plants 1 in 3"), not "it works".
- **BRANCH A — liftoff discovered, acceptance ≥ 70%**: proceed to Phase 4 forward ramp.
- **BRANCH B — liftoff discovered, acceptance < 70%**: extend to 4000–6000 iterations by resume
  (`--agent.load-checkpoint model_1000.pt --agent.resume True`) before changing any reward.
- **BRANCH C — no liftoff at all**: do **not** tune rewards first. Re-check `pushoff`: if `T_max` is
  marginal the target is physically wrong. Only then consider raising `hop_unweighting`'s weight —
  it is currently 1.0 against a completed-hop stack worth several hundred, so the discovery gradient
  is ~0.4% of the payoff.
- **SATISFIES**: AC #4

### RUN 4 — enable the forward objective

- **IMPLEMENT**: Re-time the `forward_weight` stages to begin ~500 iterations after run 3's observed
  liftoff-discovery point, then run 4000–6000 iterations.
- **GOTCHA**: **one variable set per run.** Do not co-mingle the forward ramp with reward-weight
  tuning or the `max_paid_rate` change below, or run 4's outcome is unattributable.
- **VALIDATE**: `uv run scripts/eval_hop.py ... ` with the forward-distance criterion enabled
- **SATISFIES**: AC #4

### RUN 5 (conditional) — de-spike the air-time jackpot

- **IMPLEMENT**: Only if run 4's video shows violent or uncontrolled liftoff. Set
  `hop_air_time`'s `max_paid_rate` to ~0.5 so the frontier pays out over roughly twice the flight
  duration instead of all at once.
- **WHY**: `max_paid_rate = 1.0` is a **no-op** for a duration frontier — air time can only advance
  at 1× real time, so the cap never binds. The full `TARGET_AIR_TIME` payout (weight 6.0 × 50
  normalized units = ~300 reward) currently lands inside the ~8 steps of flight, ~37/step against a
  landing composite of at most 4/step. AGENTS.md: "any 'reach X' reward must be rate-limited or
  slewed... Arriving early at a goal state that then pays per-step is a jackpot that buys arbitrary
  violence." The roulade's equivalent cap (`max_paid_rate = 3.0` rad/s) *does* bind.
- **GOTCHA**: this changes total reward mass, so compare against run 4 on eval-battery acceptance
  rate, not on mean reward.
- **SATISFIES**: AC #4

---

## TESTING STRATEGY

### Unit Tests

CPU-only, no GPU, following `tests/test_hop_cfg.py`. Cover: measured constants against the compiled
model; ballistic self-consistency of the mid-air ranges; forward objective gated at weight 0 with a
ramp; reward-term signs; gate/seed coupling.

The stub-env pattern added on 2026-09-13 (`_stub_env`, `_forward_env`, `_FakeScene`) lets the mdp
bookkeeping helpers be tested without a simulator — extend it rather than inventing a new harness.

### Integration Tests

The 64-env/5-iteration smoke test is the integration test. It is the only thing that exercises the
Warp path, the sensors resolving against the real model, and the ONNX export together.

### Edge Cases

- Mid-air spawn lands on the completion gate's zero point (regression, covered).
- Robot prone on its back with both feet in the air accruing false air time (regression, covered).
- Butt-bounce: non-foot contact milliseconds before liftoff (regression, covered).
- Walk-then-stumble: feet-only ground travel cashed on the first airborne frame (regression,
  covered).
- Robot topples early then recovers — must be able to earn a hop afterwards (the sticky-taint fix;
  covered by the clean-clock recovery assertion).
- A model revision adding a collidable body — the ground sensor's negative lookahead plus
  `test_ground_sensor_covers_every_nonfoot_collision_body` covers it.

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run python -c "import mjlab_microduck.tasks.microduck_hop_env_cfg as m; m.make_microduck_hop_env_cfg()"
```

### Level 2: Unit Tests

```bash
uv run --with pytest pytest tests/test_hop_cfg.py tests/test_hop_measurements.py -q
```

### Level 3: Full Suite (zero regressions)

```bash
uv run --with pytest pytest tests/ -q     # baseline at plan time: 221 passed, 1 skipped
```

### Level 4: Measurement + Smoke

```bash
uv run scripts/measure_hop.py settle
uv run scripts/measure_hop.py heights
uv run scripts/measure_hop.py pushoff
uv run train Mjlab-Hop-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --hf-jobs
```

### Level 5: Training + Eval

```bash
uv run train Mjlab-Hop-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 1000 --hf-jobs
uv run scripts/eval_hop.py --wandb-run-path <entity/mjlab_microduck/run_id> --checkpoint 1000
uv run scripts/export.py Mjlab-Hop-MicroDuck --wandb-run-path <...>
uv run scripts/infer_policy.py --walking out.onnx        # CPU deployment rehearsal
```

---

## ACCEPTANCE CRITERIA

- [ ] **AC #1** — Every constant in `microduck_hop_env_cfg.py` marked `UNVERIFIED` has a measured or
      analytically derived value with its derivation recorded in the file. No `UNVERIFIED` remains.
- [ ] **AC #2** — The mid-air spawn ranges are ballistically consistent with `TARGET_AIR_TIME`
      (within 1.4×), locked by `tests/test_hop_measurements.py`.
- [ ] **AC #3** — `hop_forward_progress` is registered at weight 0 with a curriculum ramp, so run 3
      discovers liftoff alone.
- [ ] **AC #4 — THE BAR** — `scripts/eval_hop.py` reports **≥ 70% acceptance over 512 episodes from
      standing spawns**, where an episode is accepted iff: real simultaneous double-foot flight
      occurred (peak air time ≥ `HOP_MIN_AIR_TIME`); **no non-foot body touched the ground at any
      point**; and at landing + 0.5 s the trunk z ≥ 0.10 m, tilt ≤ 20°, and **both feet in contact**.
- [ ] **AC #5** — The 64-env/5-iteration smoke test passes: builds, NaN-free, obs is 61D, every
      reward term computes, ONNX exports.
- [ ] **AC #6** — Every `Episode_Reward/<penalty>` in the accepted run is ≤ 0 across all iterations.
- [ ] **AC #7** — Full suite green with no regressions; new behavior covered by tests.
- [ ] **AC #8** — The accepted checkpoint exports to ONNX via `scripts/export.py` (normalizer baked
      in) and runs in `scripts/infer_policy.py` under CPU BAM without NaN.

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Each task validation passed immediately
- [ ] All validation commands executed successfully
- [ ] Full test suite passes
- [ ] Smoke test green before any long run
- [ ] Eval battery run on the actual checkpoint, and the **video watched**
- [ ] Acceptance criteria all met
- [ ] Outcome reported as what rollouts actually show, not "it works"

---

## OPEN QUESTIONS / ASSUMPTIONS

- **Assumed** — `TARGET_AIR_TIME = 0.15 s` is physically achievable. This is the plan's single
  biggest risk and Task `pushoff` exists to kill it. If `T_max < 0.15 s`, two failed runs are
  explained by an impossible target and the whole reward discussion was downstream of a physics
  error. Confirm before Phase 3.
- **Assumed** — `STAND_Z = 0.115` transfers from the standup/roulade envs. It was inherited, not
  measured, for this env. Task `heights` checks it. AGENTS.md flags a 5 mm error here as having
  cost days.
- **Assumed** — the standing spawn (`z ∈ [0.11, 0.12]`, tilt ±5°, HOME joints) is a stable
  equilibrium. Never verified. Task `settle` checks it. Note `standing_z_min = 0.11` sits 5 mm
  *below* `STAND_Z`, which may spawn the robot slightly compressed or interpenetrating.
- **Assumed** — 3.0 s is enough for settle + hop + land + recover + hold. Derived from the roulade's
  5 s minus rotation time, not measured. If run 3 shows episodes timing out mid-recovery, revisit.
- **Open** — `hop_unweighting`'s weight of 1.0 gives a discovery gradient worth ~0.4% of a completed
  hop's payoff. Left alone for run 3 so the run stays attributable; revisit only under Branch C.
- **Open** — whether symmetry mirror-loss (`ENABLE_SYMMETRY = True`) helps or hurts here. A forward
  hop is sagittally symmetric so it should help, but it is untested for this env and no run has
  isolated it. Not a run-3 variable.
- **Confirmed this session** — no local CUDA device; all training via `--hf-jobs`.
- **Divergence, user-approved** — this plan lives in `.claude/plans/` rather than the repo's
  `docs/superpowers/{specs,plans}/` convention, and is in English rather than the existing docs'
  French.

## NOTES (open canvas)

### Why the ballistic check matters more than it looks

Once both feet leave the ground the robot is a projectile — no actuator can change the trajectory of
its centre of mass. So air time, apex and takeoff velocity are one number wearing three hats:

| air time T | takeoff v_z | apex rise | apex trunk z |
|---|---|---|---|
| 0.10 s | 0.491 m/s | 12.3 mm | 0.1273 m |
| **0.15 s** | **0.736 m/s** | **27.6 mm** | **0.1426 m** |
| 0.20 s | 0.981 m/s | 49.1 mm | 0.1641 m |
| 0.25 s | 1.226 m/s | 76.6 mm | 0.1916 m |

The mid-air spawn range (`z` 0.14–0.18 m while *still descending* at 0.5–1.5 m/s) back-solves to
apexes of 0.153–0.295 m — hops of 0.175 s to 0.383 s. So the reverse curriculum, which is 50% of all
experience, has been teaching recovery from landings 1.2× to 2.6× more energetic than the hop the
reward function asks for. Even with the gate-seed bug fixed it would have been training the wrong
skill. This is the kind of thing a `## Constantes mesurées` section catches for free, which is the
argument for Phase 1 in one paragraph.

### On "others are making much progress"

The seven envs that progressed all have a spec and a plan written before training. The hop has
neither, and its constants carry `UNVERIFIED`. Three of the four bugs fixed on 2026-09-13, plus
defect #5, are all the same failure: a number that was guessed, never measured, and then reasoned
about as though it were known. The roulade cfg docstring describes a five-run lesson arc — the hop
is on run 2 of an equally hard maneuver, with a worse starting position because it skipped the
design step, not because the tuning was worse.

### Rejected alternatives

- **Clear the taint on "recovered to standing"** (upright + feet-down + low velocity thresholds)
  instead of the decaying clean-time clock. Rejected: three new thresholds to tune, and it needs a
  definition of "recovered" that the decay gets for free — standing back up *necessarily* produces a
  stretch of feet-only contact.
- **Terminate the episode on non-foot ground contact.** Rejected: landing badly and recovering is
  explicitly part of the task (same reasoning as roulade/velstand), and it would delete the recovery
  training the mid-air bucket exists to provide.
- **Keep the forward objective live in run 3 and just fix the launch latch.** Rejected: it leaves
  two skills being discovered at once, so a failure is uninterpretable. The forward reward is worth
  ~250 of ~550 hop-stack points — not a small perturbation.
- **Raise `hop_unweighting` now to strengthen discovery.** Rejected for run 3: it is a plausible fix
  for Branch C, but changing it alongside four bug fixes and a scope reduction would make run 3
  unattributable. Held as a named contingency instead.

## AMENDMENTS

<!-- Append-only. Newest at the bottom. -->
