# Warm-starting the hop from the community jump: Step 1 (CPU only)

Date: 2026-09-29. Source: `ThomasBurgess2000/microduck-max-height-jump` @ `7e5dc60`
(Apache-2.0). Every file passed `sha256sum -c CHECKSUMS.sha256`.

## Correction first: our eval batteries were not measuring the spawn they claimed

`scripts/eval_hop.py` and `scripts/eval_hop_per_hop.py` forced the spawn type through
`event_manager.get_term_cfg("set_hop_state")`. The `hop_spawn_mix` curriculum is an
`event_param_curriculum` that rewrites those same probabilities on every reset, and curricula
run before reset events. So every "standing" or "crouch" battery sampled the step-0 mix instead:
35% standing, 35% crouch and 30% mid-air. A forced-standing reset spawned trunk z between 66 and
166 mm (24/64 crouched, 17/64 mid-air); after the fix it spawns between 110 and 119 mm.

The fix is `e0b1691`: both scripts drop `hop_spawn_mix` before building the env, and tests cover it.

Every HopOnce number measured with these batteries before this note came from the mixed
population. That includes the AGENTS.md figures (v2 `model_3250` 44%/41%, v3 `model_3549` 55%,
the foot trace) and the scratch count-hops and foot-trace scripts, which used the same forcing.
AGENTS.md was left unedited as instructed; its HopOnce numbers need this correction.

## 1a. The jump ONNX in our HopOnce env

Harness: our `Mjlab-HopOnce-Flat-MicroDuck` play env (training DR, BAM, delays, obs noise), with
the jump's contract on the command slots. `twist = [1,0,0]` once both feet have been down for 4
consecutive steps, and back to 0 at the first touchdown after a whole-robot flight or 0.75 s after
arming. Head and body command slots are 0. Episodes are 3 s, 128 per battery, with forced spawns
(after the fix). Scored with `eval_hop.py` (AC #4) and `eval_hop_per_hop.py`, the same code paths
used for our checkpoints.

### Standing spawns

| | Jump ONNX (one-shot trigger) | HopOnce v3 `model_3549` |
|---|---:|---:|
| AC #4 accepted | 68/128 (53%) | **89/128 (70%)** |
| AC #4 without the both-feet-at-+0.5 s criterion | 69/128 (54%) | **126/128 (98%)** |
| Exactly one qualifying flight | 120/128 | 119/128 |
| Episodes with no flight | 5 | 0 |
| Clean (feet-only) landing, per hop | 91% | **100%** |
| Air time, median (p10–p90) | 0.10 s (0.06–0.14) | **0.16 s (0.14–0.18)** |
| Forward per hop, median | 2.1 cm | **7.5 cm** |
| Upright at 3 s | 59% | **100%** |
| AC #4 end states | standing 69, side 21, prone-back 14, never-lifted 23, prone-front 1 | standing 128 |

The jump's AC #4 misses are mostly falls after landing: 26 episodes with non-foot contact,
18 with low height and tilt, plus 11 no-flight episodes. This matches its own README, which says
it "lands but doesn't settle"; in our 3 s episodes about 40% of landings end on the back or side.
v3's only systematic miss is the both-feet snapshot, which the earlier foot trace attributes to
recovery shuffles. That trace was on the mixed-spawn population, so it needs re-running on
standing spawns.

### Crouch spawns

Neither policy is usable from our crouch bucket.

- **The jump never triggers from it:** 123/128 episodes. Our crouch pose doesn't keep both feet
  down for 4 steps. This is not the jump's own foot-flat crouch (`CROUCH_OVERRIDES`,
  z 0.068–0.071 m). It never lifts in 122/128, and 0% pass AC #4.
- **v3 fails AC #4 from crouch:** 0% accepted, with non-foot contact in every episode. 124/128
  still end standing, but only 4% of first hops land clean. Whether the non-foot contact starts
  at spawn was not checked.

### The flag does not gate the launch

Per-hop battery, standing spawns, with each flag contract:

| Flag contract | Episodes with a hop | Hops per episode | Upright at 3 s |
|---|---:|---|---:|
| One-shot (1 after 4 grounded steps, 0 at touchdown or 0.75 s) | 96% | {0: 5, 1: 120, 2: 3} | 59% |
| Held at 0 ("settle") | **89%** | {0: 14, 1: 109, 2: 5} | 66% |
| 1 from step 0, 0 at touchdown or 0.75 s | 11% | {0: 114, 1: 14} | 85% |
| Held at 1 | 2% | {0: 126, 1: 2} | 87% |

From a fresh standing spawn the jump launches whether or not it is asked. Held at 1, which its
README says is outside the contract, it almost never launches. The flag reads more like "0 = go,
1 = hold before you are supported" than as a launch request. A deployment implication for the
published ONNX: hot-swapping it in with the settle command may still produce a jump.

There is also a wait problem. In a 40-step check, 22/128 standing spawns never armed the trigger:
with the flag at 0 the policy did not hold both feet down for 4 steps in a row (feet chattering,
22–37° of tilt). Our standing spawns reach up to 119 mm with joint noise; the jump trained on
112–115 mm.

## 1b. The checkpoint in our HopOnce actor

- **Same network:** `checkpoint/model_34995.pt` is actor 61→512→256→128→14 and critic
  74→512→256→128→1 with ELU, and both carry obs normalizers, exactly our shapes.
- **Actor loads with `strict=True`** after one remap. `distribution.raw_std_param` (the
  BoundedGaussian logit) becomes our scalar `distribution.std_param` via
  `std = 0.05 + 1.45 * sigmoid(raw)`, the bounds from its `agent.yaml`. The result is a per-joint
  std of 0.05–0.21 (mean ≈ 0.08), against our init_std of 1.0.
- **Critic loads with `strict=True` unchanged.**
- **The loaded actor reproduces the published ONNX:** max |Δaction| = 9.5e-7 over 40 steps × 64
  envs of live HopOnce observations. The ONNX results above are the checkpoint's behaviour.
- The optimizer state is empty in the checkpoint, so it will start fresh.

**Which critic obs differ: none.** Actor and critic term lists, order, functions, parameters,
noise, scale, clip and delays are identical. The only textual differences are list-vs-tuple
spellings of the same joint regex. Corruption is on for the actor and off for the critic in both.
What does differ is:

- **Value scale:** the critic's value head was fit to the jump's reward stack, not ours.
- **Normalizer statistics:** the twist-x slot saw a 0/1 flag in the jump; HopOnce samples tiny
  near-zero values.

**Robot model: identical physics.** At the jump's base `d424a0c`, `get_standup_spec` loaded
`robot_allcollisions.xml`; today it loads `robot_groundcontact.xml`. Compiled, the jump-era
all-collisions model and today's groundcontact model have the same 11 collision geoms on the same
8 bodies. Masses, inertias, body positions, joint ranges, armature, damping and friction are also
identical. (The current `robot_allcollisions.xml` has 70 collision geoms; it is the renamed file
that matches.)

Artifacts (session scratchpad, not in the repo): `jump_eval.py` (the harness above),
`jump_load.py` (load, remap and self-test), and `jump_as_hoponce.pt` (the converted checkpoint).
