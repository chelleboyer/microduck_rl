# Implementation Report — Microduck forward hop, Phase 1 (measurement)

**Plan**: `.claude/plans/microduck-forward-hop.md`
**Branch**: `feat/hop-env-training`
**Status**: PARTIAL — Phase 1 complete, deliberately stopped before Phase 2

## Summary

Built `scripts/measure_hop.py`, the CPU measurement harness the hop env never
had, and ran the four Phase 1 measurements against the same BAM M6 physics
training uses. The gating measurement answers the plan's single biggest risk
in the negative: **`TARGET_AIR_TIME = 0.15 s` is roughly 4x beyond what these
legs can deliver**, so the hop's central target has been asking for a maneuver
the robot cannot perform. Two secondary findings — a 23.6 mm error in the
direction of `STAND_Z`'s meaning, and the absence of any passively stable
standing pose — also bear on the env. No env code was changed.

## Tasks completed

- CREATE `scripts/measure_hop.py` (CREATE) — 4 subcommands, BAM harness reused
  from `scripts/infer_policy.py` via importlib
- ADD `settle` subcommand → `scripts/measure_hop.py` (UPDATE)
- ADD `heights` subcommand → `scripts/measure_hop.py` (UPDATE)
- ADD `pushoff` subcommand — the gating measurement → `scripts/measure_hop.py` (UPDATE)
- ADD `ranges` subcommand → `scripts/measure_hop.py` (UPDATE)

## Measured results

### `settle` — the standing spawn is not a passive equilibrium

32 noisy standing spawns, 3 s holding HOME ctrl: **0/32 upright**, mean final
tilt 79.7 deg, mean final z 0.047 m. Pure forward pitch (roll stays ~0), with
the ankle back-driven to -0.09 rad. Tilt passes 10 deg at ~0.9 s and the robot
is down by ~1.4 s.

This is **not** a bad spawn height and **not** a BAM artifact:

- every spawn height from 0.110 to 0.119 m behaves identically;
- the stock XML PD actuators topple too, and *faster* (83 deg by 1.0 s vs 1.4 s);
- every pose in the flat-footed family topples, HOME merely slowest (0.9 s to
  10 deg, vs 0.16-0.28 s elsewhere) — HOME/STAND2 is a genuine local optimum.

The robot is an actively balanced biped with no passively stable standing pose.
The consequence for the hop is real: episodes begin with the robot already
falling forward, so the policy must arrest a topple before it can hop.

### `heights` — STAND_Z means something different than the cfg assumes

| quantity | value |
|---|---|
| HOME / STAND2 kinematic trunk z | **0.1172 m** |
| hop cfg `STAND_Z` | 0.1150 m (-2.2 mm) |
| full extension (p=+0.80, q=+0.60) | **0.1408 m** |
| deepest crouch | 0.0441 m |
| total kinematic stroke | 96.8 mm |
| stroke available above HOME | **23.6 mm** |

Method cross-check: `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
independently measured `debout 0.1407` on this robot; this sweep gives 0.1408 m.

The load-bearing finding is that **HOME is already a 23.6 mm crouched stand**,
not the top of the stroke. A hop launched from HOME has only 23.6 mm of leg
extension left before the legs are straight.

### `pushoff` — THE GATING MEASUREMENT

**Two earlier versions of this measurement were wrong and are retracted.** The
first counted a fallen robot as airborne; the second measured topples as
push-offs (commanded open-loop, the robot rotates about its toe — tilt 0.4 ->
95 deg with both feet still in contact and trunk z FALLING). Any "4.2x energy
shortfall" figure from those is void.

The trustworthy measurement constrains the trunk to a vertical slide, so it
physically cannot topple. That isolates the gating question — how much vertical
velocity can these legs generate — from the balance problem:

| | 6.5 V | 8.2 V | needed for 0.15 s |
|---|---|---|---|
| takeoff velocity | 0.311 m/s | **0.385 m/s** | 0.736 m/s |
| ballistic air time | 0.064 s | **0.078 s** | 0.150 s |
| apex rise | 26.6 mm | **30.1 mm** | 27.6 mm |

**The robot can hop.** With balance handled it leaves the ground, rises ~30 mm,
and gets ~0.03 s of measured flight against a ~0.078 s ballistic ceiling.

`TARGET_AIR_TIME = 0.15 s` still looks roughly 2x the achievable takeoff
velocity (~4x in energy) even under *idealised* balance, and a real policy must
also balance, so it can only do worse than this bound.

Toe-off and proximal-to-distal sequencing were both tested (3312 profiles,
trading total leg extension against ankle reserve — the ankle sits at 1.40 rad
of a 1.5708 limit at max extension, leaving only 0.171 rad of toe-off room).
Neither changed the result materially. A head/neck swing did not help either.

Neither actuator limit binds during the push: torque peaks at 0.43 Nm against
the 1.068 Nm forcerange clamp, and joint speed at 7.3 rad/s against a 22.4 rad/s
no-load speed (32%). So the constraint is geometric/control, not raw actuator
power — which is one reason a trained policy may beat these hand-designed
profiles.

**Scope of this bound, stated precisely:** it bounds hand-designed open-loop
profiles with the trunk pinned upright. It is NOT a hard physical ceiling. A
policy has freedoms this sweep did not use — dynamic hip_roll/yaw and head
motion (frozen at HOME here), launching from a moving rather than a resting
state, and a learned torque profile at 50 Hz instead of a linear ramp.

### `ranges` — derived spawn constants (NOT applied; Phase 2 applies them)

Sized off the low-voltage velocity ceiling (0.0691 s) by the plan's
`0.75 x T_max` rule, with `STAND_Z = 0.1172`:

```
TARGET_AIR_TIME   = 0.052       MIDAIR_Z_MIN    = 0.120
HOP_MIN_AIR_TIME  = 0.021       MIDAIR_Z_MAX    = 0.123
MIDAIR_VZ_RANGE   = (-0.34, -0.20)
MIDAIR_VX_RANGE   = (0.0, 0.96)
```

Note for Phase 2: `TARGET_FORWARD_DIST = 0.05 m` is not compatible with an air
time this short — 50 mm of travel in 0.052 s demands ~0.96 m/s of horizontal
launch speed. The forward target needs re-deriving alongside the air time.

## Tests added

None. `tests/test_hop_measurements.py` is a **Phase 2** task in the plan (it
locks the *re-derived* constants, which Phase 2 sets) and was correctly out of
this scope.

## Validation results

- Level 1 — `make_microduck_hop_env_cfg()` constructs: **PASS**
- Level 2 — `uv run scripts/measure_hop.py {--help,settle,heights,pushoff,ranges}`: **PASS**
- Level 3 — full suite: **221 passed, 1 skipped** — identical to the plan's
  baseline, no regressions (no env code was touched)

## Deviations from the plan

1. **Scene rather than bare robot spec.** The plan says load
   `MICRODUCK_STANDUP_ROBOT_CFG.spec_fn().compile()`. That spec has no floor,
   so no contact, settle or push-off measurement is possible from it. Used
   `scene.xml` instead, which includes exactly that robot file plus a floor
   plane; `_build()` asserts the ctrl-idx == joint-idx mapping so a model
   revision cannot silently invalidate the script.

2. **`pushoff` starts from the exact crouch pose at zero velocity, with no
   free-settle window** (`--settle-s` defaults to 0). The plan implied a
   settle. Because no pose is passively stable, any settle window measures a
   topple with a leg extension on top of it — the first run of this sweep
   showed trials starting at 21-30 deg of tilt. Starting from the exact pose
   asks the capability question under best-case conditions, which is what a
   gating upper bound should do.

3. **Counter-movement added to the sweep**, not in the plan. A single ramp from
   rest understates what the actuators can do, and the conclusion is strong
   enough that it needed the strongest push-off available.

4. **`ranges` is derivation-only** and writes nothing, to respect the Phase 1
   stop line.

## Issues encountered

Three defects in my own harness, all found and fixed before any number was
reported — recording them because they are the same class of bug the env
itself was bitten by:

1. **Air time counted a fallen robot.** Testing only "both feet off the floor"
   reported 0.83-1.19 s "hops" with *negative* takeoff velocity — a robot lying
   on its trunk. This is the butt-bounce exploit reproduced inside the
   measurement tool. Airborne now means no robot geom touches the floor AND the
   trunk is rising.
2. **The first `heights` sweep under-reported the stroke by 12x** (8 mm vs the
   real 96.8 mm). It held the knee near-straight and swept a leg *tilt* rather
   than a squat, and it assumed p=0 was full extension when extension is
   actually p~+0.80. Replaced with a 2-D sweep over the whole flat-footed
   family — which is what then matched the roller-standup spec to 0.1 mm.
3. **Nearest-height crouch selection picked contorted poses** (q=-1.25
   hyperextends the knee backwards). Crouches now follow a coordinated
   knee-drive path from the extension pose.

## What this means for the plan — SUPERSEDED, see below

*(This section originally concluded that `TARGET_AIR_TIME = 0.15 s` was out of
reach and asked whether a ~0.05 s hop was still the product. That conclusion is
retracted. It is left here, corrected in place, because the reasoning error is
worth keeping: a measurement was treated as a physical ceiling when it was only
a bound on the profiles I happened to script.)*

**The plan's biggest risk did NOT materialise.** A community policy,
`ThomasBurgess2000/microduck-max-height-jump`, reaches **0.628 m/s of launch
velocity and 140 ms of air time** on this robot — verified this session by
reproducing its own evaluation unmodified on HF Jobs (job
`6aa6c3b621047bf1b038461f`), which matched every published metric exactly. Its
model diff adds only massless, non-colliding sites and changes no physics
parameter.

So `TARGET_AIR_TIME = 0.15 s` is approximately right, not impossible, and the
two failed hop runs are explained by the reward defects already identified.

**Do not apply the `ranges` output derived at `T = 0.052 s`** — it would write
constants sized to a ~3 mm hop into the cfg. Re-derive against ~0.14 s.

What Phase 1 established that still stands:

- `STAND_Z` measures **0.1172 m** kinematically (cfg says 0.115, delta +2.2 mm),
  with the method cross-checked to 0.1 mm against the roller-standup spec;
- full extension is **0.1408 m**; HOME is already a ~24 mm crouched stand;
- the standing pose is not a passive equilibrium under BAM *or* XML PD — but a
  policy holds it at 0.49 deg of tilt, so this is an actively balanced biped;
- **the binding constraint on a hop is balance during the push, not actuator
  power**: torque peaked at 0.43 Nm of a 1.068 Nm clamp and joint speed at
  7.3 rad/s of 22.4 rad/s no-load. Commanded open-loop the robot rotates about
  its toe instead of rising.

That last point is the durable result of Phase 1, and it is what
`AMENDMENT 1` in the plan acts on: a crouch spawn bucket, a launch-velocity
reward, an airborne attitude penalty, and a direct-metrics block.
