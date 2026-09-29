# Implementation Report — Microduck forward hop, AMENDMENT 1 (four ports)

**Plan**: `.claude/plans/microduck-forward-hop.md` (`AMENDMENT 1` section)
**Branch**: `feat/hop-env-training`
**Status**: COMPLETE — all four ports implemented and CPU-validated. Stopped
before the smoke test by explicit instruction (it needs `--hf-jobs` and costs
money).

## Summary

Ported the four mechanisms `AMENDMENT 1` identified from the verified community
jump policy `ThomasBurgess2000/microduck-max-height-jump`, each re-expressed
against the hop's own state machine rather than copied: a crouch spawn bucket, a
dense launch-velocity reward, airborne attitude penalties, and a `cfg.metrics`
block of direct physical outcomes. All four target the measured failure — the
push-off is a **balance** problem, not a power problem — which is why the two
earlier runs produced exploits instead of hops.

One constant had to be measured rather than ported: the jump's crouch pose is
specified by magnitude only, and its signs are model-dependent. Measured against
this compiled model, the correct signs put the trunk at **0.0658 m**; the
sign-flipped variant rests at 0.1078 m, barely below HOME's 0.1172 m and not a
crouch at all.

## Tasks completed

- ADD crouch spawn bucket → `src/mjlab_microduck/tasks/mdp.py` (`reset_hop_state`, UPDATE)
  and `microduck_hop_env_cfg.py` (`CROUCH_OVERRIDES`, `CROUCH_Z_RANGE`, spawn probs, UPDATE)
- ADD `hop_launch_velocity_progress` → `mdp.py` (UPDATE); wired at weight 3.0
- ADD `hop_airborne_tilt_penalty` + `hop_lateral_drift_penalty` → `mdp.py` (UPDATE);
  wired at −0.4 / −0.5
- ADD `cfg.metrics` block (6 terms) → `mdp.py` (metric functions) + cfg (UPDATE)
- ADD `_hop_airborne_now` shared airborne predicate → `mdp.py` (UPDATE)
- UPDATE `hop_spawn_mix` curriculum from two buckets to three
- UPDATE cfg header docstring with "Course correction 4" (the file's running-log convention)

## Tests added

`tests/test_hop_cfg.py`, 14 new cases (31 → 35 in this file; suite 221 → 235):

- `test_hop_crouch_spawn_bucket_configured` — bucket wired, probabilities normalise
- `test_hop_crouch_pose_is_a_real_crouch_and_within_joint_limits` — indices resolve to the
  expected joint NAMES on the compiled model, angles inside `jnt_range`, chains exact mirrors
- `test_hop_crouch_height_is_below_standing` — band above the measured 0.0658 m rest height
- `test_hop_spawn_mix_curriculum_keeps_every_bucket_alive` — no bucket hits zero; standing monotone
- `test_hop_airborne_predicate_rejects_a_prone_robot` — the butt-bounce regression
- `test_hop_airborne_tilt_penalty_is_a_cost_and_inert_on_the_ground`
- `test_hop_lateral_drift_penalty_leaves_the_forward_axis_free` — the divergence-from-jump regression
- `test_hop_launch_velocity_frontier_only_advances_while_loaded`
- `test_hop_launch_velocity_is_blocked_while_not_clean`
- `test_hop_metrics_report_physical_outcomes_and_never_pay`
- `test_hop_new_reward_signs_follow_the_convention`
- `test_hop_rise_metrics_measure_from_the_episode_start_height`
- `test_hop_foot_rise_is_bilateral`
- `test_hop_rise_metrics_zeroed_for_midair_spawns`

## Validation results

- `uv run --with pytest pytest tests/` → **235 passed, 1 skipped** (was 221 + 1)
- `uv run --with pytest pytest tests/test_hop_cfg.py -q` → **35 passed** (the plan's per-task VALIDATE)
- ruff: no new error *categories*. The repo has a large pre-existing count
  (85 in `mdp.py` at HEAD); the 4 new ones are all `UP045`
  (`Optional[X]` vs `X | None`), matching 47 existing uses in the same file, and
  the 8 in the test file are function-local import ordering, matching 7 existing.
  Deliberately left consistent with the surrounding code rather than mixed.
- **NOT run**: the 64-env / 5-iteration smoke test. Stopped here by instruction.

## Deviations from the plan

1. **Crouch height is `(0.066, 0.070)`, not the plan's `(0.068, 0.071)`.** Measured:
   the ported pose rests at 0.0658 m on this model. The jump trains on the
   all-collisions model, and AGENTS.md forbids carrying a target height across
   models ("a 5 mm-wrong STAND_Z once turned the goal into an impossible target
   for days"). The band sits just above the measured rest height so joint noise
   cannot spawn the robot inside the floor.
2. **Crouch signs measured, not ported.** The plan gives magnitudes
   (`0.4188, 1.3776, 0.9588`); the sign convention is per-joint on this model
   (HOME has `left_hip_pitch` negative but `left_ankle` positive). Both variants
   were evaluated kinematically and the one producing an actual crouch was kept.
3. **Launch-velocity weight 3.0, not the jump's 2.0.** AGENTS.md: compare reward
   *mass*, not weights, when moving a term between envs. This stack's positive
   terms sum to ~23.5; the plan's own instruction was "a real fraction of the hop
   stack, not a rounding error".
4. **Airborne penalties are live from step 0, not curriculum-ramped.** Both are
   gated on being genuinely airborne, so they are unreachable until a liftoff
   exists and cannot act as an attempt-tax during discovery — the specific reason
   the AGENTS.md "introduce taxes after the skill exists" rule exists. For the
   same reason neither can block a fallen robot's recovery.
5. **Sole clearance is reported as a foot *rise*, and no model change was made.**
   The plan offered adding the jump's `passive_*_sole_probe_*` sites or computing
   min sole-vertex height. Both were declined: the robot XMLs are generated by
   onshape-to-robot, and `additional.xml` injects at the top level only, so
   in-body sites would be hand-edits lost on the next re-export. Measuring the
   rise from the episode's own start height removes the site-to-sole offset
   entirely (the sites sit at the ankle frame, ~14 mm above and ~24 mm behind the
   sole) and needs nothing added to the model. Consequence: `max_foot_rise_mm` is
   **not** directly comparable to the jump's absolute 31.67 mm sole clearance —
   compare trends, not that number.

## Issues encountered

- **A test caught a real indexing trap.** `Entity.find_joints` returns
  entity-local indices, but `model.jnt_range` is model-indexed and offset by the
  trunk freejoint — reading limits with a servo id reports `left_hip_pitch` as
  ±0.384 rad (actually `left_hip_roll`'s). The test now resolves by name. All six
  leg pitch joints are ±1.5708, so every crouch angle is in range.
- **Metric reference latching was fragile and is now hardened.** The step guard
  means the *first* metric to call `_update_hop_metric_accum` each step is the one
  whose arguments take effect. With a single shared latch flag, a future metric
  that omitted `foot_site_cfg` would consume the episode's latch and leave the
  foot reference holding the *previous* episode's height — a silently wrong
  measurement. The trunk and foot references now carry separate flags.
- `MetricsManager` does resolve `SceneEntityCfg` params (verified in
  `manager_base._resolve_common_term_cfg`), so the foot sites populate and the
  metric cannot be silently inert.

## What is NOT done (deliberately out of this scope)

- The **smoke test** (64 envs / 5 iters, `--hf-jobs`) — the next required step.
- Plan defect **#5**, the still-open mid-air spawn ranges (`MIDAIR_Z` 0.14–0.18 /
  `MIDAIR_VZ` −1.5…−0.5 imply 0.175–0.383 s hops against a 0.15 s target).
- Applying the measured `STAND_Z = 0.1172` (cfg still carries the inherited 0.115).
- AC #3's weight-gating of `hop_forward_progress` to 0 during discovery.

Items 2–4 are Phase 2 in the plan. The cfg docstring now flags the first two as
measured-but-not-yet-applied so the staleness is visible rather than silent.
