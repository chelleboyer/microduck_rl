# Warm-starting the hop from the community jump: Step 2 (GPU run)

Date: 2026-09-29. Follows `jump-warmstart-step1.md`.

## Pre-run check: v3 on true standing spawns

The `scripts/eval_hop.py` fix (`e0b1691`) means earlier traces had used the mixed spawn
population. Re-traced on standing spawns only (128 episodes), HopOnce v3 `model_3549` has:

- **no non-foot contact** in any episode;
- at landing + 0.5 s, 95 with both feet down, 24 mid-shuffle-step and 9 tapping in place;
- 125/128 with both feet down at the end.

Its only misses are recovery shuffles.

## Setup

- **Task:** `Mjlab-HopOnce-JumpWarm-Flat-MicroDuck` (`make_microduck_hop_env_cfg(once=True,
  jump_warm=True)`, commit `6f92a56`). It uses HopOnce v3's rewards and PPO settings. The forward
  gate and the dirty-landing slope stay live instead of being collapsed to run 6's final stage.
  The spawn mix starts at 70/15/15 (standing/crouch/midair).
- **Warm start:** `scripts/convert_jump_checkpoint.py` converted `model_34995.pt` (MLP and both
  normalizers kept; std reset to 0.15; fresh Adam state). It is uploaded as `model_0.pt` to W&B
  run `iktraxfw` and loaded with `MICRODUCK_WARM_START=1`.
- **Why std 0.15, not the planned 0.3:** a stochastic-rollout sweep on 128 standing spawns gave
  flight / clean after landing / upright at 3 s of 87/51/58% deterministic, 80/35/41% at std
  0.15, and 75/9/12% at 0.3. At 0.3 the landing signal would have been nearly invisible to PPO
  (the run-4 failure mode).
- **Smoke tests:** CPU (32 envs, 3 iterations) and HF (job `6abb9cfc…`, 64 envs, 5 iterations)
  both clean.
- **Run:** HF job `6abb9e0cc617607c354d4e3b`, W&B `zsj5w7gc`, 4096 envs, 1000 iterations,
  57 min. Checkpoints (every 50 iterations) are in
  `chelleboyer/mjlab-hoponce-jumpwarm-flat-microduck-20260929-111623`.

## Training curve

No NaNs, and every penalty stayed ≤ 0 throughout.

| iter | valid_takeoff | clean_landing | extra_flight | forward gate | action std |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.78 | 0.50 | 0.32 | 0 | 0.15 |
| 100 | 0.88 | 0.72 | 0.25 | 0 | 0.15 |
| 150 | 0.89 | 0.78 | 0.21 | 1.5 | 0.15 |
| 250 | 0.90 | 0.76 | 0.21 | 3.0 | 0.16 |
| 350 | 0.92 | 0.72 | 0.21 | 5.0 | 0.17 |
| 500 | 0.95 | 0.76 | 0.28 | 5.0 | 0.18 |
| 750 | 0.94 | 0.72 | 0.31 | 5.0 | 0.19 |
| 999 | 0.98 | 0.74–0.79 | 0.33–0.39 | 5.0 | 0.19 |

- **Forward gate:** it reached its last stage by iteration 350 with no step-down in clean
  landings at the stage boundaries, unlike run 6.
- **Re-hops in training:** `extra_flight_rate` crept up from 0.19 to 0.39. That metric covers
  all spawn types and stochastic actions; deterministic standing evals show 127/128 episodes with
  exactly one flight.

## Evaluation (fixed batteries, 128 episodes each, deterministic)

### Standing spawns

| | v3 `model_3549` | JumpWarm `model_500` | JumpWarm `model_999` |
|---|---:|---:|---:|
| AC #4 accepted | 89 (70%) | **116 (91%)** | 108 (84%) |
| AC #4 without the both-feet criterion | 126 (98%) | 124 (97%) | 116 (91%) |
| Exactly one flight | 119 | **127** | **127** |
| Clean landing, per hop | 100% | 99% | 95% |
| Air time, median | 0.16 s | 0.14 s | 0.16 s |
| Forward per hop, median | **7.5 cm** | 4.8 cm | 6.7 cm |
| Upright at 3 s | 100% | 97% | 100% |
| Tilt at landing + 0.5 s, p50/p90 | 2.7°/6.1° | 2.6°/8.5° | 5.9°/10.3° |

Both-feet snapshot misses (`one_foot` alone) are 37 for v3 and 8 for either JumpWarm
checkpoint. Non-foot contact is 1 for v3, 4 for `model_500` and 12 for `model_999`.

### Crouch spawns

| | v3 `model_3549` | JumpWarm `model_500` | JumpWarm `model_999` |
|---|---:|---:|---:|
| AC #4 accepted | 0 | 0 | 0 |
| First hop lands clean | 4% | 32% | 13% |
| End states (AC #4 battery) | standing 124 | **side 92**, never-lifted 31 | standing 117 |
| Upright at 3 s (per-hop) | 100% | 0% | 99% |

The crouch failures are not a spawn artifact. Crouch spawns sit at trunk z 67–70 mm with no
non-foot contact at t = 0 in 128/128 episodes (`model_999`); the first contact comes at a
median of step 18 (0.36 s), during or after the hop. The crouch bucket was 15% of training.

## Reading

- **Standing, which is how the policy is entered on the robot:** the warm start clearly beats
  v3 on AC #4 (91% / 84% against 70%) and on one-hop discipline (127 against 119 of 128). The
  cost is some forward distance (4.8–6.7 cm against 7.5 cm) and, for `model_999`, a slightly
  looser landing: more non-foot brushes and more tilt at the snapshot.
- **Crouch:** everything fails AC #4. `model_500` falls over from crouch; `model_999` gets back
  up, as v3 does.
- **Candidate:** `model_999` for balance (84% standing AC #4, and it recovers from crouch), or
  `model_500` if standing is all that matters. The checkpoints between them (550–950) are
  unevaluated and may do better on both.
