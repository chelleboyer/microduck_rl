# Factory operations

This application consumes a pinned complete Archon SDLC source. See
[factory/MIGRATION.md](factory/MIGRATION.md) for source installation and data mapping.

Shared workflow gates own decisions. The old autonomy dial and local receipts
cannot authorize merges. Inspect native run identity and status before responding
to a declared gate. Standing intake and unattended operation require separate
producer integration and live verification.

Project runtime scenarios: `harness/END-TO-END.md`.
Holdout scenarios: `.factory/holdout/HOLDOUT.md`.
Ordinary checks: `python harness/ci.py`.

Record this application's tested source SHA, candidate, native run IDs, coverage,
fresh-environment evidence and unresolved integration limits here after testing.

---

## This installation (microduck_rl, 2026-09-14)

| | |
|---|---|
| Pinned source | `f61c9c9e959fce6e6366ff5c02000783a75e6a4a`, 18 workflows |
| Installed on | `factory/install`, branched from `develop` at `cb70b79` |
| Ordinary checks | `python harness/ci.py` → `STATIC_SKIPPED`, `UNIT_PASSED tests=225`, `CHECKS_OK` |
| Conventions | `AGENTS.md`, via `CLAUDE.md`. It wins on every engineering question |
| Gates | money (`.factory/policy.py check-command`) and merge, both human |

## Known limits

Named rather than quietly absent.

| Limit | Consequence | Status |
|---|---|---|
| No lint gate exists | `ruff check .` reports 452 findings under default rules on `develop` (tasks 106, tests 34, scripts 49), and `AGENTS.md` names no lint command. `static` is empty, so the harness prints `STATIC_SKIPPED` every run rather than implying a check that is not happening | **open** — adopting one is triage work and its own ticket |
| The expensive check is the one the factory cannot run | A green gate here means the config is well-formed and the CPU tests pass. Whether the policy *learns* is answered only by a GPU run a human launches. The gap between `CHECKS_OK` and "the behavior works" is wider in this repository than in most | **inherent** |
| No browser or runtime driver | `harness/runtime.inputs.json` is unwired. The journeys in `END-TO-END.md` are CLI journeys and are run by hand today | open |
| Journeys describe `develop`, not the hop branch | The hop env and `scripts/measure_hop.py` live on `feat/hop-env-training`. They join the journeys when that branch merges | expected |
| The factory holds a token that can merge, and a shell that could type a train command | `.factory/policy.py` refuses both, and `MISSION.md` forbids them, but neither is a sandbox. The guarantee is a human | **accepted, permanently** |
| Upstream changed the all-collisions model | `name_servo_collision_geoms` (upstream `cb70b79`) names servo housing geoms, and `FULL_COLLISION` keys condim off names — so those geoms moved from default to condim 1, frictionless. The hop path uses the groundcontact model and is unaffected, but the reference policy `ThomasBurgess2000/microduck-max-height-jump` trains on all-collisions, so its verified numbers predate this change | **open** — matters when porting from it |
