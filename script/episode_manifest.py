"""
Episode manifest for fixed-episode evaluation.

Decouples expert screening from policy evaluation so that different policies
can be evaluated on identical physical episodes (see issue #8):

- Screening phase (one-off, script/screen_episodes.py): run the expert check
  once, save the accepted seeds together with their episode_info and
  _saved_dynamic_motion_info (dynamic start position, trajectory params, RNG
  state) into a manifest file. Rejected seeds are logged with reasons.
- Evaluation phase (script/eval_policy.py --overrides --episode_manifest <path>):
  load the manifest and skip online expert re-planning entirely.

The default evaluation protocol (online expert filtering, RoboTwin-compatible)
is unchanged; the manifest mode is strictly opt-in.
"""

import json
import os
import pickle
import subprocess
from copy import deepcopy
from datetime import datetime

MANIFEST_FORMAT_VERSION = 1

# Rejection reasons recorded during screening
REJECT_UNSTABLE = "unstable"
REJECT_EXCEPTION = "exception"
REJECT_EXPERT_CHECK_FAIL = "expert_check_fail"
REJECT_DYNAMIC_VALIDATION_FAIL = "dynamic_validation_fail"


def _get_git_commit():
    try:
        return (subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        return None


def default_manifest_path(task_name, task_config, cli_seed):
    return os.path.join("eval_manifest", task_name, task_config, f"seed{cli_seed}.pkl")


class EpisodeManifest:
    """
    Canonical list of expert-accepted episodes for one (task, task_config) pair.

    Attributes:
        metadata: dict describing how the manifest was generated.
        entries: list of {'seed', 'episode_info', 'dynamic_motion_info'}.
        rejected: list of {'seed', 'reason'} for rejected candidate seeds.
    """

    def __init__(self, metadata=None, entries=None, rejected=None):
        self.metadata = metadata or {}
        self.entries = entries or []
        self.rejected = rejected or []

    @classmethod
    def from_args(cls, args, cli_seed, st_seed, test_num):
        metadata = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "embodiment": args.get("embodiment"),
            "cli_seed": cli_seed,
            "st_seed": st_seed,
            "test_num": test_num,
            "use_dynamic": args.get("use_dynamic", False),
            "dynamic_level": args.get("dynamic_level"),
            "dynamic_coefficient": args.get("dynamic_coefficient"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _get_git_commit(),
        }
        return cls(metadata=metadata)

    def add_entry(self, seed, episode_info, dynamic_motion_info):
        self.entries.append({
            "seed": seed,
            "episode_info": deepcopy(episode_info),
            "dynamic_motion_info": deepcopy(dynamic_motion_info),
        })

    def add_rejected(self, seed, reason):
        self.rejected.append({"seed": seed, "reason": reason})

    @property
    def seeds(self):
        return [entry["seed"] for entry in self.entries]

    def save(self, path):
        """Save pickle payload plus a human-readable JSON summary alongside."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "entries": self.entries,
            "rejected": self.rejected,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        summary = {
            "metadata": self.metadata,
            "accepted_seeds": self.seeds,
            "rejected": self.rejected,
        }
        summary_path = os.path.splitext(path)[0] + ".json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        version = payload.get("metadata", {}).get("format_version")
        if version != MANIFEST_FORMAT_VERSION:
            raise ValueError(f"Unsupported manifest format version: {version} "
                             f"(expected {MANIFEST_FORMAT_VERSION})")
        return cls(
            metadata=payload["metadata"],
            entries=payload["entries"],
            rejected=payload.get("rejected", []),
        )

    def validate_against(self, args):
        """Raise if the manifest was generated under a different protocol."""
        checks = {
            "task_name": args["task_name"],
            "task_config": args["task_config"],
            "embodiment": args.get("embodiment"),
            "use_dynamic": args.get("use_dynamic", False),
            "dynamic_level": args.get("dynamic_level"),
            "dynamic_coefficient": args.get("dynamic_coefficient"),
        }
        mismatches = []
        for key, current in checks.items():
            recorded = self.metadata.get(key)
            if recorded != current:
                mismatches.append(f"{key}: manifest={recorded!r}, current={current!r}")
        if mismatches:
            raise ValueError("Episode manifest does not match the current evaluation config:\n  " +
                             "\n  ".join(mismatches))


def _expert_rollout(TASK_ENV, args, now_id, now_seed):
    """
    Run one expert rollout for a candidate seed.

    Returns (accepted, episode_info, dynamic_motion_info, reject_reason).
    Mirrors the expert-check block of the online evaluation loop.
    """
    from envs.utils.create_actor import UnStableError

    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()
    except UnStableError:
        TASK_ENV.close_env()
        return False, None, None, REJECT_UNSTABLE
    except Exception:
        TASK_ENV.close_env()
        return False, None, None, REJECT_EXCEPTION

    if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
        return False, None, None, REJECT_EXPERT_CHECK_FAIL

    dynamic_motion_info = getattr(TASK_ENV, "_saved_dynamic_motion_info", None)
    return True, episode_info, dynamic_motion_info, None


def _validate_dynamic_entry(TASK_ENV, args, now_id, now_seed, dynamic_motion_info):
    """
    Pre-validate that a dynamic episode can be initialized for evaluation
    (init_dynamic_motion_for_eval may reject trajectories whose extension
    crosses prohibited areas). Guarantees manifest entries never get skipped
    at evaluation time.
    """
    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV._saved_dynamic_motion_info = deepcopy(dynamic_motion_info)
        ok = TASK_ENV.init_dynamic_motion_for_eval()
    except Exception:
        ok = False
    TASK_ENV.close_env()
    TASK_ENV.release_episode_resources()
    return ok


def screen_episodes(TASK_ENV, args, cli_seed, st_seed, test_num, clear_cache_freq=None):
    """
    Run expert screening once and collect exactly test_num accepted episodes.

    Uses the same candidate-seed scanning order as the online protocol
    (starting from st_seed), so a manifest built from cli_seed=N covers the
    same seed range an online evaluation with seed=N would explore.
    """
    manifest = EpisodeManifest.from_args(args, cli_seed, st_seed, test_num)

    args = dict(args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    if clear_cache_freq is None:
        clear_cache_freq = args.get("clear_cache_freq", 20)

    use_dynamic = args.get("use_dynamic", False)
    now_seed = st_seed
    now_id = 0

    while len(manifest.entries) < test_num:
        accepted, episode_info, dynamic_motion_info, reason = _expert_rollout(
            TASK_ENV, args, now_id, now_seed)

        if accepted and use_dynamic:
            if not _validate_dynamic_entry(TASK_ENV, args, now_id, now_seed, dynamic_motion_info):
                accepted, reason = False, REJECT_DYNAMIC_VALIDATION_FAIL

        if accepted:
            manifest.add_entry(now_seed, episode_info, dynamic_motion_info)
            now_id += 1
            print(f"\033[92m[screen] seed {now_seed} accepted "
                  f"({len(manifest.entries)}/{test_num})\033[0m")
        else:
            manifest.add_rejected(now_seed, reason)
            print(f"\033[91m[screen] seed {now_seed} rejected ({reason})\033[0m")

        if (len(manifest.entries) + len(manifest.rejected)) % clear_cache_freq == 0:
            TASK_ENV.close_env(clear_cache=True)

        now_seed += 1

    return manifest


def online_episode_provider(TASK_ENV, args, st_seed):
    """
    Default RoboTwin-compatible episode source: scan candidate seeds and run
    the expert check online, yielding each accepted episode. Behavior matches
    the original inline loop in eval_policy.py.
    """
    now_seed = st_seed
    now_id = 0
    while True:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        accepted, episode_info, dynamic_motion_info, _ = _expert_rollout(
            TASK_ENV, args, now_id, now_seed)

        args["render_freq"] = render_freq

        if not accepted:
            now_seed += 1
            continue

        yield {
            "seed": now_seed,
            "episode_info": episode_info,
            "dynamic_motion_info": dynamic_motion_info,
        }
        now_id += 1
        now_seed += 1


def manifest_episode_provider(manifest):
    """Fixed episode source: yield the manifest entries in order."""
    for entry in manifest.entries:
        yield {
            "seed": entry["seed"],
            "episode_info": entry["episode_info"],
            "dynamic_motion_info": deepcopy(entry["dynamic_motion_info"]),
        }
