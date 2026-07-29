"""
One-off expert screening for fixed-episode evaluation.

Runs the expert check once for a (task, task_config, seed) triple and saves a
canonical episode manifest: the accepted seeds together with their episode
info and dynamic motion info (start position, trajectory params, RNG state),
plus the rejected candidate seeds with reasons.

The manifest can then be passed to eval_policy.py so that every policy is
evaluated on identical physical episodes:

    python script/screen_episodes.py --task_name beat_block_hammer \
        --task_config demo_clean_dynamic --seed 0

    python script/eval_policy.py --config <deploy_policy.yml> \
        --overrides --episode_manifest eval_manifest/beat_block_hammer/demo_clean_dynamic/seed0.pkl
"""

import sys
import os

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import argparse

from eval_policy import build_task_args, class_decorator
from episode_manifest import screen_episodes, default_manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Screen expert-solvable episodes into a manifest")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="CLI seed, same semantics as eval_policy.py")
    parser.add_argument("--test_num", type=int, default=100, help="number of accepted episodes to collect")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="output path (default: eval_manifest/<task>/<config>/seed<seed>.pkl)")
    return parser.parse_args()


def main(usr_args):
    task_name = usr_args.task_name
    task_config = usr_args.task_config
    seed = usr_args.seed
    test_num = usr_args.test_num

    args = build_task_args(task_name, task_config)
    args["eval_video_log"] = False
    args["render_freq"] = 0

    print("============= Screening Config =============")
    print(f"Task Name: {task_name}")
    print(f"Task Config: {task_config}")
    print(f"CLI Seed: {seed} | Episodes: {test_num}")
    print("Use Dynamic: " + str(args.get("use_dynamic", False)))
    if args.get("use_dynamic", False):
        print(" - Dynamic Level: " + str(args.get("dynamic_level", "N/A")))
        print(" - Dynamic Coefficient: " + str(args.get("dynamic_coefficient", "N/A")))
    print("============================================")

    TASK_ENV = class_decorator(task_name)

    st_seed = 100000 * (1 + seed)
    manifest = screen_episodes(TASK_ENV, args, seed, st_seed, test_num)

    manifest_path = usr_args.manifest_path or default_manifest_path(task_name, task_config, seed)
    summary_path = manifest.save(manifest_path)

    reasons = {}
    for item in manifest.rejected:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1

    print("\n============= Screening Result =============")
    print(f"Accepted: {len(manifest.entries)} episodes "
          f"(seed range {manifest.seeds[0]} - {manifest.seeds[-1]})")
    print(f"Rejected: {len(manifest.rejected)} candidates {reasons}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Summary saved to:  {summary_path}")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    main(parse_args())
