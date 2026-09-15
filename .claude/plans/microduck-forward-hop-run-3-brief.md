# Hop run 3 — prepared brief (not executed)

Prepared for issue #5. **This brief does not launch anything.** Both commands below are
refused by `.factory/policy.py check-command` (MISSION.md hard invariant 1) — verified this
session, see "Verification" at the end. The factory prepares; a person runs the commands from
this file.

Reference: `.claude/plans/microduck-forward-hop.md`, "AMENDMENT 1" and "Amended run-3
watchlist" (L629-805).

## Status this brief assumes

- Both of run 3's named blockers are closed on this branch: mid-air spawn ballistics
  (`499739d`, `b3b9a2f`) and the `STAND_Z` reconciliation (`c92380b`, `67b225a`).
- AMENDMENT 1's four ports (`17f5e63`) are in `microduck_hop_env_cfg.py`: the crouch spawn
  bucket (`SPAWN_CROUCH_PROB`, L281), `hop_launch_velocity` (L426-430), `hop_airborne_tilt`
  (L520-523), and the six-term `cfg.metrics` block (L545-572).
- **Deviation from the plan, found while preparing this brief:** AC #3 — "`hop_forward_progress`
  registered at weight 0 with a curriculum ramp" — is **not implemented**. The term is
  registered at a fixed weight of 5.0 from step 0 (`microduck_hop_env_cfg.py:445-449`); no
  curriculum entry gates it (checked every `cfg.curriculum[...]` block, L796-897 — only
  `hop_spawn_mix`, `com_range`, `head_com_range`, `action_rate_weight`, `torque_rate_weight`,
  `gentle_landing_weight`, `no_crawl_weight` exist), and `tests/test_hop_cfg.py:55` only asserts
  `r["hop_forward_progress"].weight > 0`. This matches `AGENTS.md`'s own status line — "the rest
  of the plan's Phase 2 (AC #3: reducing run 3 to one question) is still open" — so it is not a
  regression, but it does mean the plan's original framing of run 3 ("is liftoff discoverable
  ALONE") no longer describes the run as configured. See "What question run 3 answers" below.

## 1. Smoke-test command

```
uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --hf-jobs
```

Source: plan L507 (Level 4 validation), matching `AGENTS.md`: "A 5-iteration smoke test at 64
envs catches ~95% of config errors for cents. Never launch a long run without one."

## 2. Full-run command

```
uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 1000 --hf-jobs
```

**Why 1000, not 6000** (plan L391-394): `AGENTS.md` budgets simple episodic tricks at ≈1000
iterations at 4096 envs. The question this run asks (see below) is answered well before 6000,
and a probe keeps the 6000-iteration commitment for a run whose answer is already yes.

## 3. Smoke-test pass conditions — ALL must hold before the full run is worth submitting

- Builds, steps NaN-free, obs is 61D, every reward term computes, ONNX exports (plan L383,
  `AGENTS.md`'s standard smoke gate).
- Every `Episode_Reward/<penalty>` reads **≤ 0**: self-negating terms `hop_no_crawl`,
  `gentle_landing`, `hop_stand_tax` and ordinary negative-weight terms `action_rate_l2`,
  `joint_torque_rate_l2`, `self_collisions`, `hop_airborne_tilt`, `hop_lateral_drift`
  (`microduck_hop_env_cfg.py:475-534`). A positive reading here is the `AGENTS.md` sign-convention
  bug, not a training-outcome question — fix it before spending anything on the full run.
- All six `cfg.metrics` series populate — not stuck at exactly 0 for every one of the 5
  iterations: `valid_takeoff_rate`, `max_air_time_s`, `max_launch_velocity_mps`,
  `max_com_rise_mm`, `max_foot_rise_mm`, `stable_landing_rate` (`microduck_hop_env_cfg.py:545-572`).
  Caveat, load-bearing for reading this correctly: `valid_takeoff_rate` alone is expected to read
  **≈`SPAWN_MIDAIR_PROB` = 0.30** even under 5 iterations of a near-random policy, because
  mid-air spawns pre-seed the gate past the threshold (`mdp.py:8344-8347`, `hop_metric_valid_takeoff`
  docstring). A flat **0.30** here is not failure; a flat **0.0** is — it means the metric wiring
  itself is broken, not that no hop happened.
- No NaN-guard termination fires.

## 4. Ordered watchlist for run 3 (reading order, per the "Amended run-3 watchlist", plan L800-805)

1. **`valid_takeoff_rate`** — read against the current `hop_spawn_mix` stage, not in isolation:
   at step 0 the mix is 35% standing / 35% crouch / 30% mid-air (`SPAWN_STANDING_PROB` /
   `SPAWN_CROUCH_PROB` / `SPAWN_MIDAIR_PROB`, L280-282), and mid-air spawns score this metric 1.0
   without a real takeoff. **Pass**: the rate rises measurably above the mid-air floor (~0.30 at
   step 0, ~0.25 after the L806 stage, ~0.20 after the L809 stage) as training progresses — real
   liftoffs from standing/crouch spawns are happening. **Stop-indicator**: pinned at exactly the
   mid-air-floor value for the whole run — no ground-spawn liftoff is occurring at all.
2. **`max_com_rise_mm`** — zeroed for mid-air spawns by construction (`mdp.py:8378-8382`), so this
   is the clean ground-liftoff signal `valid_takeoff_rate` can't give alone. **Pass**: rises above
   0 and trends upward, ideally toward the ballistic apex rise for `TARGET_AIR_TIME` (27.6 mm per
   the plan's ballistic table, L587). **Stop-indicator**: stays at 0 through iteration 200 —
   mirrors `hop_unweighting` staying flat at zero (see item 6): the push-off is not being found.
3. **`max_air_time_s`** — **Pass**: climbs past `HOP_MIN_AIR_TIME` (0.06 s, the landing-gate
   threshold) toward `TARGET_AIR_TIME` (0.15 s). **Stop-indicator**: stays below 0.06 s
   indefinitely once `max_com_rise_mm` is already rising — a partial unweight that never clears
   the landing gate.
4. **`max_launch_velocity_mps`** — **Pass**: climbs toward `TARGET_LAUNCH_VZ` (0.60 m/s,
   `microduck_hop_env_cfg.py:289`); the verified reference jump policy reaches 0.628 m/s, so
   values in that neighborhood mean the launch-velocity shaping is working as ported. **Stop
   indicator**: stays near 0 while `hop_unweighting` (item 6) is rising — the precursor is
   present but never converts to real upward speed.
5. **`max_foot_rise_mm`, `stable_landing_rate`** — the remaining two of the six `cfg.metrics`.
   Read alongside items 1-4 once liftoff is real; they judge the landing/recovery half
   specifically (`stable_landing_rate` requires trunk height + upright + only-feet-touching
   simultaneously, `mdp.py:8415-8430`) and are not gating on their own for this probe.
6. **`Episode_Reward/*` series** — `hop_unweighting` rising in the first ~200 iterations (the
   discovery precursor; flat at zero means liftoff isn't being explored at all — plan L396-397).
   `hop_air_time` becoming non-zero (the main dense task term; total reward can rise purely on
   regularizers while it stays zero). `hop_landing_composite` non-zero from iteration 0 (direct
   regression check on the mid-air gate-seed fix — if it reads 0.0000, that fix has broken).
   Every `Episode_Reward/<penalty>` still ≤ 0 throughout, not just at smoke-test time. Any metric
   in this list stepping **down** exactly at a curriculum boundary (iteration 1500 or 3000, the
   `hop_spawn_mix` / `action_rate_weight` / etc. stage steps, L806-897) means the pacing is wrong
   — stretch the stage, don't advance it further.

## 5. Stop rule

**Primary — kill at iteration 200 if no discovery**: by iteration 200, `valid_takeoff_rate` is
still pinned at its mid-air-spawn floor AND `max_com_rise_mm`/`hop_unweighting` are still flat at
zero. Per `AGENTS.md`, a rising `hop_unweighting` in the first ~200 iterations is the discovery
precursor; flat at zero this long means liftoff is not being explored at all, and no later
iteration budget fixes that — stop and re-check `pushoff` / Branch C in the plan (L419-422),
don't keep spending on the same run.

**Immediate, regardless of iteration** — kill on first occurrence of any of:
- any `Episode_Reward/<penalty>` reads **> 0** (sign-convention regression);
- `hop_landing_composite` reads exactly 0.0000 from iteration 0 (the mid-air gate-seed fix has
  regressed);
- the NaN-guard termination fires.

A metric stepping down exactly at a curriculum boundary is **not** a kill condition by itself —
per `AGENTS.md`, that means stretch the stage, which is a config change for the next run, not a
reason to abort the current one mid-flight.

## What question run 3 answers, and what a bad result would mean

As configured today (AC #3 still open, see "Status this brief assumes"), run 3 asks: **is
liftoff — and, since `hop_forward_progress` is live at a fixed weight of 5.0 rather than
weight-gated, some forward displacement alongside it — discoverable at all, under the full fixed
reward stack and the amendment's measured constants, within 1000 iterations at 4096 envs?** This
is a narrower claim than the plan's original "liftoff alone" framing, because the forward term is
not currently held at zero while liftoff is being found.

- If `valid_takeoff_rate` never rises above its mid-air floor and `max_com_rise_mm` stays at 0 —
  the four AMENDMENT 1 ports are not enough to escape the toe-rotation failure Phase 1 measured;
  the reward stack (or the crouch/launch-velocity shaping specifically) is still wrong, not the
  target.
- If liftoff is discovered but `hop_landing_composite` / `stable_landing_rate` stay at 0 — the
  push-off half now works but the landing/recovery half is still broken; investigate that half
  specifically before touching push-off shaping again.
- If any penalty reads positive — the reward stack has a sign-convention bug independent of
  whether the maneuver is ever found; fix it and rerun the smoke test before spending on a full
  run.

## Verification (run this session, not part of any launch)

```
$ python3 .factory/policy.py check-command "uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --hf-jobs"
POLICY_REFUSED ... violations=2 (--hf-jobs, uv run train — MISSION.md hard invariant 1)

$ python3 .factory/policy.py check-command "uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 1000 --hf-jobs"
POLICY_REFUSED ... violations=2 (--hf-jobs, uv run train — MISSION.md hard invariant 1)

$ python3 harness/ci.py
HARNESS_START mode=ordinary
STATIC_SKIPPED no 'static' command in harness.config.json
UNIT_PASSED tests=266
RUNTIME_NOT_RUN: shared workflow verification is separate
CHECKS_OK mode=ordinary
```

No Hugging Face Job was submitted in the course of preparing this brief.
