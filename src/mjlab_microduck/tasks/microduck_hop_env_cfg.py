"""Microduck hop task — attempt 1, run 1.

Episodic policy: robot starts standing, both feet leave the ground at the
same instant, travels FORWARD during the airborne phase, and lands upright
and recovers to standing. Triggered at deployment like sit/standup/roulade
(policy switch = hop starts immediately; no phase clock, no reference
motion).

Scope note: the product direction is a forward hop, not the in-place hop the
PRD's v1 non-goals describe (a real-time call from the project owner,
superseding that draft) — hop_forward_progress below is the training-side
answer. Landing on two feet stays the v1 target; one-foot landing is an
explicitly later step, not attempted here.

Course correction 2026-09-12 (mid-run-1): the first forward-hop run's
dominant strategy at ~1300/6000 iterations was a butt-bounce — trunk hits
the ground, rebounds, both feet come off for a moment, some of that
rebound happens to carry the robot forward. Every reward term up to that
point only checked the FEET, so this satisfied every gate as well as a
real leg-driven hop would have. nonfoot_ground_cfg + the clean-time clock
in mdp.py (_update_hop_clean_time) close it: hop credit requires the robot
to have been free of non-foot ground contact for _HOP_CLEAN_LIFTOFF_S, and
creditable air time is capped by how long it has been clean, so a
trunk-assisted liftoff earns nothing.

Course correction 2 (same day, run-2): fixing the butt-bounce revealed a
SECOND exploit rather than a clean hop — "worming", where the policy
drags/undulates its trunk along the ground to sweep repeatedly close to
the standing target instead of settling into it once. hop_no_crawl (mdp.
hop_no_crawl_penalty) closes this by taxing horizontal trunk velocity
while the trunk is in ground contact, without penalizing the contact
itself (the robot has no arms — bracing/rocking on the trunk is a
legitimate way to get upright after a bad landing). Ramped in via
curriculum, held back further than the other polish terms specifically
because blocking a no-armed robot's only recovery option too early could
prevent it from ever discovering how to get up at all.

Product requirements: ../../../docs/ideas/hop-behavior.md in the `microduck`
repo. This file is the training-side implementation of that PRD; nothing in
that document should be read as a training-time decision, and nothing here
should be read as re-scoping the product requirements.

Design (see the "Hop" section of mdp.py for the full mechanism):
  • The state this task tracks is per-episode SIMULTANEOUS air time —
    min(air_time_left, air_time_right), which is exactly zero whenever
    either foot is in contact. This is the hard gate that makes a hop a hop
    (both feet at once) rather than a step or a single-leg push, for free,
    with no separate asymmetry penalty.
  • hop_unweighting_bonus is dense discovery shaping toward the precursor
    (unweight both feet, rise) — a pure gate has no gradient until the
    policy stumbles onto a real double-foot liftoff by chance.
  • Landing/recovery reuses the roulade landing-composite pattern exactly
    (standing_composite_score × completion gate, plus upright/height
    bootstrap layers and a stand-tax so "land in a heap" isn't free), gated
    on having hopped instead of on rotation.
  • Reverse curriculum via mid-air spawns (the roulade lesson: "the second
    half is learnable on its own" — here, landing-and-recovery is trained
    directly from an airborne, falling spawn state, without requiring
    liftoff to already work).

Course correction 3 (2026-09-13, pre-run-3): three structural bugs, each of
which silently removed most of the training signal. See the linked mdp.py
docstrings for the full reasoning.
  • reset_hop_state seeded the mid-air bucket's air-time frontier at exactly
    gate_min_air_time, which is _hop_completion_gate's ZERO point — so all
    four landing/recovery terms paid 0 for every mid-air episode. Both sites
    now derive from mdp._HOP_GATE_FULL_OPEN.
  • The butt-bounce taint was STICKY for the episode, so one topple pinned
    the air-time frontier at 0 and left a reward function
    (-action_rate -self_collisions) whose argmax is "do nothing" for the
    remaining ~2.5 s. Replaced by the decaying clean-time clock, which
    attributes credit per flight instead of per episode.
  • hop_forward_progress measured displacement from the SPAWN point, so
    "lunge forward on the feet, then stumble through four airborne frames"
    collected the forward reward AND opened the landing gate. The launch
    frame is now latched at liftoff.

UNVERIFIED, run 1: every numeric constant below (EPISODE_LENGTH_S, air-time
targets, mid-air spawn ranges, force_norm) is a plausible guess, not a sim
measurement — this sandbox has no GPU to run mjlab's MuJoCo-Warp step, so
AGENTS.md step 2 ("verify physics assumptions in sim BEFORE training") could
not be done. The mandatory next step before any real training run is the
64-env / 5-iteration smoke test, which will need to run somewhere with a
CUDA device (locally or via --hf-jobs) and will very likely surface
tensor-shape or physics-assumption bugs this review could not catch.

DR / obs / regularisers mirror the standup/roulade envs (sim2real parity).
Discovery-difficulty for the liftoff itself is unverified, so — unlike
roulade's active-from-step-0 impact shaping — this file defaults to the
general AGENTS.md rule: motion-blockers and impact/smoothness taxes ramp in
via curriculum AFTER the skill exists, not before.
"""

import math
from copy import deepcopy

# Symmetry — a hop is sagittal / left-right symmetric like the roulade.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to roulade/standup for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-hop is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the roulade/standup envs) ─────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: standing settle + a short airborne event (UNVERIFIED target
# ~0.15 s) + landing + recovery + hold. Shorter than roulade's 5 s since a
# hop has no rotation to get through.
EPISODE_LENGTH_S = 3.0

# Measured standing trunk height, reused from the roulade/standup envs
# (same robot model, same standing pose) — not re-measured here.
STAND_Z = 0.115

# Every robot body EXCEPT the two ankles (which own left/right_foot_collision,
# the only geoms allowed to touch the ground). Used by nonfoot_ground_cfg —
# see the long comment there for why matching trunk_base alone was wrong.
NONFOOT_BODY_PATTERN = r"^(?!ankle_).*"
FOOT_BODIES = ("ankle_left", "ankle_right")

# ── Hop targets (UNVERIFIED — see file header) ───────────────────────────────
TARGET_AIR_TIME    = 0.15   # s of simultaneous air time that earns full progress credit
HOP_MIN_AIR_TIME   = 0.06   # s that opens the landing/recovery gate
UNWEIGHT_FORCE_N   = 7.23   # MEASURED: compiled model total mass 0.7372 kg * 9.81
TARGET_FORWARD_DIST = 0.05  # m of forward travel FROM LIFTOFF that earns full credit.
                             # Was 0.08 while this was (wrongly) measured from the spawn
                             # point; from-liftoff is a strictly harder target, and 0.05 is
                             # what TARGET_AIR_TIME buys at a modest ~0.33 m/s horizontal
                             # launch speed. Still a guess — but a deliberately reachable
                             # one, so the term saturates instead of paying for violence.
                             # Raise it once a run has produced a hop worth measuring.

# ── Mid-air spawn (reverse curriculum) ───────────────────────────────────────
MIDAIR_Z_MIN  = 0.14
MIDAIR_Z_MAX  = 0.18
MIDAIR_VZ_RANGE = (-1.5, -0.5)   # falling, UNVERIFIED — measure a real hop's descent speed
MIDAIR_VX_RANGE = (0.0, 0.6)     # forward speed at landing, UNVERIFIED — a forward hop lands
                                  # with real horizontal momentum; without this the
                                  # landing/recovery half of the reverse curriculum only ever
                                  # practices a dead-stop landing and won't transfer

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_hop_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck hop environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT, RIGHT order
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # "Any part of the robot EXCEPT a foot is touching the ground."
    #
    # Watches every non-ankle body, NOT just trunk_base. The first version of
    # this sensor matched `pattern="trunk_base"` alone and was completely
    # inert: measured with a CPU MuJoCo drop test (robot settled belly-down,
    # trunk z=0.035 against the 0.115 standing target), the floor contacts
    # are `hip_l`×3, `hip_l_2`×3, `jaw_soft`×1 and the two feet — trunk_base
    # appears nowhere, because the torso shell never reaches the floor. Both
    # ground-cheat fixes therefore never once fired, and
    # `Episode_Reward/hop_no_crawl` read exactly 0.0000 for a whole run while
    # the video plainly showed the robot prone. Watch the bodies that
    # actually bear weight, not the one the behaviour is named after.
    #
    # A negative lookahead rather than an explicit body list, so a future
    # model revision that adds a collidable body is covered without anyone
    # remembering to update this — and
    # test_ground_sensor_covers_every_nonfoot_collision_body asserts exactly
    # that against the compiled model, so an inert sensor cannot recur
    # silently.
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(
            mode="body", pattern=NONFOOT_BODY_PATTERN, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, nonfoot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: hop task set ──────────────────────────────────────────────────
    # Dense discovery shaping toward the precursor (unweight, rise) — see
    # mdp.hop_unweighting_bonus docstring for why the gated terms alone
    # aren't enough to find the maneuver in the first place.
    cfg.rewards["hop_unweighting"] = RewardTermCfg(
        func=microduck_mdp.hop_unweighting_bonus,
        weight=1.0,
        params={"force_norm": UNWEIGHT_FORCE_N},
    )

    # The one dense task signal once liftoff is real: paid increments of the
    # simultaneous-air-time frontier, capped so a longer hop isn't worth
    # arbitrarily more than a controlled one.
    cfg.rewards["hop_air_time"] = RewardTermCfg(
        func=microduck_mdp.hop_air_time_progress,
        weight=6.0,
        params={"target_air_time": TARGET_AIR_TIME, "max_paid_rate": 1.0},
    )

    # The forward-hop objective. Gated the same way hop_air_time is (genuine
    # simultaneous double-foot flight, this instant) — see
    # mdp.hop_forward_progress / _update_hop_forward_accum docstrings for why
    # a shuffle or walk-forward cannot farm this.
    cfg.rewards["hop_forward_progress"] = RewardTermCfg(
        func=microduck_mdp.hop_forward_progress,
        weight=5.0,
        params={"target_distance": TARGET_FORWARD_DIST, "max_paid_rate": 1.0},
    )

    # Completion-gated standing annuity — the dominant attractor, mirroring
    # roulade_landing_composite exactly.
    cfg.rewards["hop_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.hop_landing_composite,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.04,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "joint_indices": _LEG_JOINTS,
            "min_air_time":  HOP_MIN_AIR_TIME,
        },
    )
    cfg.rewards["hop_upright_after_landing"] = RewardTermCfg(
        func=microduck_mdp.hop_upright_after_landing,
        weight=1.5,
        params={"min_air_time": HOP_MIN_AIR_TIME},
    )
    cfg.rewards["hop_height_after_landing"] = RewardTermCfg(
        func=microduck_mdp.hop_height_after_landing,
        weight=1.0,
        params={"target_height": STAND_Z, "std": 0.04, "min_air_time": HOP_MIN_AIR_TIME},
    )
    cfg.rewards["hop_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.hop_stand_tax,
        weight=5.0,
        params={"target_height": STAND_Z, "min_air_time": HOP_MIN_AIR_TIME},
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards.pop("soft_landing", None)

    # Landing impact — self-negating |a_z| → POSITIVE weight (penalty sign
    # convention). Introduced at 0, ramped by curriculum (general AGENTS.md
    # rule; see file header on why this differs from roulade's step-0 choice).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Closes the worming exploit (see mdp.hop_no_crawl_penalty docstring):
    # ordinary cost (returns >= 0) -> NEGATIVE weight. Introduced at 0,
    # ramped by curriculum — same reasoning as gentle_landing above, but
    # more cautious here: the robot has no arms, so an attempt-tax active
    # too early could block the only recovery method a fallen robot has.
    cfg.rewards["hop_no_crawl"] = RewardTermCfg(
        func=microduck_mdp.hop_no_crawl_penalty,
        weight=0.0,
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.2,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright would oppose the push-off/flight phase; landing
    # uprightness is handled by the completion-gated terms above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / standup / roulade) ────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for both head (4) and body (6) — a hop
    # has no head/body-pose command, but the 61D obs layout parity with
    # every other policy is kept so the runtime stack works unchanged.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Landing badly and recovering IS part of the task (same reasoning as
    # roulade/velstand) — keep only the NaN guard + timeout.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    cfg.events["set_hop_state"] = EventTermCfg(
        func=microduck_mdp.reset_hop_state,
        mode="reset",
        params={
            "standing_prob":     0.5,
            "midair_prob":       0.5,
            "standing_z_min":    0.11,
            "standing_z_max":    0.12,
            "standing_tilt_max": math.radians(5.0),
            "midair_z_min":      MIDAIR_Z_MIN,
            "midair_z_max":      MIDAIR_Z_MAX,
            "midair_vz_range":   MIDAIR_VZ_RANGE,
            "midair_vx_range":   MIDAIR_VX_RANGE,
            "joint_noise_std":   0.08,
            "gate_min_air_time": HOP_MIN_AIR_TIME,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix: heavy mid-air early (landing/recovery is
    # learnable from day 0), shift toward standing starts as liftoff gets
    # discovered. Mid-air never goes to zero — it keeps recovery practiced.
    cfg.curriculum["hop_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_hop_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.50, "midair_prob": 0.50}},
                {"step": 1500 * 24,  "params": {"standing_prob": 0.65, "midair_prob": 0.35}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.80, "midair_prob": 0.20}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )

    # Smoothness/impact polish — introduced only after liftoff exists
    # (standup timing lesson: an attempt-tax active during discovery
    # prevents the maneuver from being found at all).
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": -5e-4},
                {"step": 2500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )
    # Anti-worm polish — same "introduced only after a landing strategy
    # exists" reasoning as the two curricula above, held back one stage
    # further: with no arms, a robot that hasn't yet found ANY way to get
    # upright from a bad landing needs its trunk-drag option free, or it
    # may never discover recovery at all. Ramps in only once the standard
    # 2500*24 stage (torque/impact polish) has already landed.
    cfg.curriculum["no_crawl_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "hop_no_crawl",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 3000 * 24,  "weight": -0.5},
                {"step": 4000 * 24,  "weight": -1.5},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckHopRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_hop",
    run_name="microduck_hop",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
