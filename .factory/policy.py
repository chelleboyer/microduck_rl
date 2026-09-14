#!/usr/bin/env python3
"""Refuse factory invocations that would merge, approve or deploy.

    python .factory/policy.py check run archon-lifecycle --input merge_mode=auto
    python .factory/policy.py check-command "uv run train Mjlab-Hop-Flat-MicroDuck --hf-jobs"
    python .factory/policy.py check-schedule
    python .factory/policy.py selftest

WHAT THIS IS AND IS NOT. It is a check you can put in front of an invocation and
in front of the scheduler's config, so the rule that the factory never merges is
something a machine can fail on rather than a sentence in a document. It is NOT
a sandbox: nothing stops a person, or a workflow, from calling the Archon CLI
directly. The guarantee that a human merges comes from a human merging.

Two facts about the pinned pack drive every rule here:

  archon-merge-queue ends in `merge-approved-prs`, which runs gh with the
  repository owner's token. `auto` has no human node. `approve` has one -- and
  then the factory still performs the merge, so what the human authorized was a
  machine merging, not a merge.

  archon-lifecycle composes merge-queue with merge_mode defaulting to `approve`.
  The default is one input away from an unattended self-merge, which is why
  `approve` is refused here alongside `auto`.

Exit 0 allows, exit 1 refuses and names the rule. A refusal prints the workflow
and the input that caused it, because a guard that says only "denied" gets
switched off.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICY = HERE / "policy.json"
SCHEDULE = HERE / "schedule.json"


def load() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def parse_inputs(argv: list[str]) -> dict[str, str]:
    """Pull `--input key=value` pairs out of an argv, in both spellings."""
    found: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        pair = None
        if token == "--input" and i + 1 < len(argv):
            pair = argv[i + 1]
            i += 1
        elif token.startswith("--input="):
            pair = token[len("--input="):]
        if pair and "=" in pair:
            key, _, value = pair.partition("=")
            found[key.strip()] = value.strip()
        i += 1
    return found


# Flags that take a separate value token. Everything else is boolean, and
# assuming otherwise is how `run --detach archon-merge-queue` hid a forbidden
# workflow behind a flag -- caught by this file's own selftest, which is the
# entire argument for the selftest existing.
VALUE_FLAGS = {"--input", "--adopt", "--config", "--model", "--branch",
               "--workflow-source", "--reason", "--comment", "--cwd"}


def workflow_of(argv: list[str]) -> str | None:
    """The first bare token after `run`. Flags and their values are not it."""
    if "run" not in argv:
        return None
    rest = argv[argv.index("run") + 1:]
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith("-"):
            if token in VALUE_FLAGS and i + 1 < len(rest):
                i += 1
        else:
            return token
        i += 1
    return None


def command_violations(command: str, policy: dict) -> list[str]:
    """Refuse a shell command that would start a training run.

    There is no local CUDA device here, so `uv run train` means Hugging Face Jobs
    and a real bill -- including the 64-env smoke test, which is a training run
    like any other. MISSION.md hard invariant 1: the factory prepares a run and
    writes down what to type; a person types it.
    """
    found = []
    for pattern in policy["forbidden_command_patterns"]:
        if pattern in command:
            found.append(
                f"the command contains {pattern!r}, which starts or publishes work that "
                f"costs money or leaves this repository. Prepare it and hand it to a "
                f"person to run -- MISSION.md hard invariant 1."
            )
    return found


def violations(argv: list[str], policy: dict) -> list[str]:
    found: list[str] = []
    workflow = workflow_of(argv)
    inputs = parse_inputs(argv)

    if workflow and workflow in policy["forbidden_workflows"]:
        found.append(
            f"{workflow} is on forbidden_workflows: it merges or deploys, and this "
            f"installation keeps both as human acts. Standard path is "
            f"{policy['standard_path']}."
        )

    for key, banned in policy["forbidden_input_values"].items():
        if key in inputs and inputs[key] in banned:
            found.append(
                f"--input {key}={inputs[key]} is refused. Allowed elsewhere; not here. "
                f"`approve` is refused as well as `auto` because in approve mode the "
                f"factory still runs the merge."
            )

    required = policy["required_inputs"].get(workflow or "", {})
    for key, value in required.items():
        actual = inputs.get(key)
        if actual is None:
            found.append(
                f"{workflow} requires --input {key}={value!r} to be passed explicitly. "
                f"The pack's default is not relied on: defaults change when the pin moves."
            )
        elif actual != value:
            found.append(f"{workflow} requires --input {key}={value!r}, got {actual!r}.")

    return found


def report(found: list[str], subject: str) -> int:
    if found:
        print(f"POLICY_REFUSED subject={subject} violations={len(found)}")
        for line in found:
            print(f"  - {line}")
        return 1
    print(f"POLICY_OK subject={subject}")
    return 0


def check_schedule(policy: dict) -> int:
    if not SCHEDULE.exists():
        print("POLICY_OK subject=schedule (no .factory/schedule.json; nothing is scheduled)")
        return 0
    data = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    argv = ["run", data.get("workflow", "")]
    for key, value in (data.get("inputs") or {}).items():
        argv += ["--input", f"{key}={value}"]
    return report(violations(argv, policy), "schedule")


CASES: list[tuple[list[str], bool, str]] = [
    (["run", "archon-ship", "--input", "target=https://x/issues/1"], True,
     "the standard path is allowed"),
    (["run", "archon-merge-queue", "--input", "prs=[]"], False,
     "the merge workflow is refused outright"),
    (["run", "archon-deploy"], False, "deploy is refused outright"),
    (["run", "archon-lifecycle", "--input", "merge_mode=auto"], False,
     "auto is refused"),
    (["run", "archon-lifecycle", "--input", "merge_mode=approve"], False,
     "approve is refused too -- the factory would still run the merge"),
    (["run", "archon-lifecycle"], False,
     "lifecycle without explicit preview inputs is refused, not defaulted"),
    (["run", "archon-lifecycle", "--input", "merge_mode=preview",
      "--input", "discoveries_mode=preview", "--input", "deploy=",
      "--input", "health=", "--input", "identity="], True,
     "lifecycle in full preview is allowed"),
    (["run", "--detach", "archon-merge-queue"], False,
     "a flag before the workflow name does not hide it"),
    (["run", "archon-backlog", "--input=publication=auto"], False,
     "the --input=k=v spelling is parsed too"),
]

COMMAND_CASES: list[tuple[str, bool, str]] = [
    ("uv run --with pytest pytest tests/ -q", True, "the test suite is free and allowed"),
    ("uv run list-envs", True, "listing the registry is free and allowed"),
    ("python harness/ci.py", True, "the gate is free and allowed"),
    ("uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs", False,
     "a full training run is refused"),
    ("uv run train Mjlab-Hop-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5",
     False, "the smoke test is refused too -- it is a training run"),
    ("uv run publish --task Mjlab-Hop-Flat-MicroDuck --repo x/y", False,
     "publishing a policy to the Hub is refused"),
]


def selftest() -> int:
    policy = load()
    failures = 0
    for argv, should_pass, why in CASES:
        passed = not violations(argv, policy)
        if passed != should_pass:
            print(f"SELFTEST_FAILED {why}: expected {'allow' if should_pass else 'refuse'}")
            failures += 1
    for command, should_pass, why in COMMAND_CASES:
        passed = not command_violations(command, policy)
        if passed != should_pass:
            print(f"SELFTEST_FAILED {why}: expected {'allow' if should_pass else 'refuse'}")
            failures += 1
    total = len(CASES) + len(COMMAND_CASES)
    if failures:
        print(f"POLICY_SELFTEST_FAILED cases={total} failures={failures}")
        return 1
    # A count, so a suite that silently stopped running is not mistaken for a pass.
    print(f"POLICY_SELFTEST_PASSED cases={total} (workflow={len(CASES)} command={len(COMMAND_CASES)})")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    mode, rest = argv[0], argv[1:]
    if mode == "selftest":
        return selftest()
    policy = load()
    if mode == "check-schedule":
        return check_schedule(policy)
    if mode == "check":
        return report(violations(rest, policy), " ".join(rest) or "(empty)")
    if mode == "check-command":
        command = " ".join(rest)
        return report(command_violations(command, policy), command or "(empty)")
    print(f"unknown mode {mode!r}; expected check, check-command, check-schedule or selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
