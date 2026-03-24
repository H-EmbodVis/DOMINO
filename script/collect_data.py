import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from copy import deepcopy
from envs import *
import yaml
import importlib
import json
import traceback
import os
import shutil
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[96mUse Dynamic:\033[0m " + str(args.get("use_dynamic", False)))
    if args.get("use_dynamic", False):
        print(" - Dynamic Level: " + str(args.get("dynamic_level", "N/A")))
        print(" - Dynamic Coefficient: " + str(args.get("dynamic_coefficient", "N/A")))
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        save_failed_cases = args.get("save_failed_cases", False)
        
        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                is_success = TASK_ENV.plan_success and TASK_ENV.check_success()

                if is_success:
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                elif save_failed_cases:
                    print(f"simulate data episode {suc_num} \033[91mFAIL\033[0m! (seed = {epid}) - \033[93mSAVING ANYWAY\033[0m")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                    fail_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True
        base_collect_args = deepcopy(args)

        clear_cache_freq = args["clear_cache_freq"]
        
        check_render_success = args.get("check_render_success", False)
        if check_render_success:
            print("\033[93m[Render Success Check: ENABLED]\033[0m")
        else:
            print("\033[93m[Render Success Check: DISABLED]\033[0m")

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        original_seed_list = seed_list.copy()
        successful_renders = []
        failed_episode_indices = []
        successful_episode_count = st_idx
        episode_idx = st_idx

        # Phase 1: Render existing seeds, skip failed ones
        while episode_idx < len(seed_list) and successful_episode_count < args["episode_num"]:
            current_seed = seed_list[episode_idx]
            print(f"\033[34mTask name: {args['task_name']} | Episode: {successful_episode_count} | Seed: {current_seed} (traj: episode{episode_idx})\033[0m")

            try:
                render_args = deepcopy(args)
                TASK_ENV.setup_demo(now_ep_num=successful_episode_count, seed=current_seed, **render_args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                render_args["left_joint_path"] = traj_data["left_joint_path"]
                render_args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(render_args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                
                if check_render_success:
                    render_success = TASK_ENV.check_success()
                else:
                    render_success = True
                
                if render_success:
                    info_db[f"episode_{successful_episode_count}"] = info

                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump(info_db, file, ensure_ascii=False, indent=4)

                    TASK_ENV.close_env(clear_cache=((successful_episode_count + 1) % clear_cache_freq == 0))
                    TASK_ENV.merge_pkl_to_hdf5_video()
                    TASK_ENV.remove_data_cache()
                    
                    if check_render_success:
                        print(f"\033[92mEpisode {successful_episode_count} render SUCCESS (seed={current_seed})\033[0m")
                    else:
                        print(f"\033[92mEpisode {successful_episode_count} render COMPLETE (seed={current_seed})\033[0m")
                    
                    successful_renders.append((episode_idx, current_seed))
                    successful_episode_count += 1
                else:
                    print(f"\033[91mEpisode render FAILED for seed {current_seed} - skipping\033[0m")
                    failed_episode_indices.append(episode_idx)
                    
                    TASK_ENV.close_env(clear_cache=True)
                    try:
                        TASK_ENV.remove_data_cache()
                    except:
                        pass
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"\033[91mException during render for seed {current_seed}: {e}\033[0m")
                failed_episode_indices.append(episode_idx)
                
                TASK_ENV.close_env(clear_cache=True)
                try:
                    TASK_ENV.remove_data_cache()
                except:
                    pass
                time.sleep(0.5)
            
            episode_idx += 1

        # Phase 2: If there are not enough successful ones, generate new seeds to supplement
        if successful_episode_count < args["episode_num"] and failed_episode_indices:
            print(f"\033[93m[Need {args['episode_num'] - successful_episode_count} more episodes, regenerating seeds...]\033[0m")
            
            try:
                TASK_ENV.close_env(clear_cache=True)
            except:
                pass
            time.sleep(0.5)
            
            regen_args_template = deepcopy(base_collect_args)
            
            next_new_seed = max(seed_list) + 1 if seed_list else 0
            new_traj_idx = len(seed_list)
            regen_attempts = 0
            
            while successful_episode_count < args["episode_num"]:
                try:
                    args_for_regen = deepcopy(regen_args_template)
                    args_for_regen["need_plan"] = True
                    args_for_regen["save_data"] = False
                    args_for_regen["left_joint_path"] = []
                    args_for_regen["right_joint_path"] = []
                    
                    TASK_ENV.setup_demo(now_ep_num=0, seed=next_new_seed, **args_for_regen)
                    TASK_ENV.play_once()
                    
                    is_success = TASK_ENV.plan_success and TASK_ENV.check_success()
                    
                    if is_success:
                        print(f"\033[92mNew seed {next_new_seed} planning succeeded\033[0m")
                        TASK_ENV.save_traj_data(new_traj_idx)
                        TASK_ENV.close_env(clear_cache=True)
                        time.sleep(0.3)
                        
                        render_args = deepcopy(base_collect_args)
                        TASK_ENV.setup_demo(now_ep_num=successful_episode_count, seed=next_new_seed, **render_args)
                        traj_data = TASK_ENV.load_tran_data(new_traj_idx)
                        render_args["left_joint_path"] = traj_data["left_joint_path"]
                        render_args["right_joint_path"] = traj_data["right_joint_path"]
                        TASK_ENV.set_path_lst(render_args)
                        
                        info_file_path = os.path.join(args["save_path"], "scene_info.json")
                        if not os.path.exists(info_file_path):
                            with open(info_file_path, "w", encoding="utf-8") as file:
                                json.dump({}, file, ensure_ascii=False)
                        with open(info_file_path, "r", encoding="utf-8") as file:
                            info_db = json.load(file)
                        
                        info = TASK_ENV.play_once()
                        render_success = TASK_ENV.check_success() if check_render_success else True
                        
                        if render_success:
                            info_db[f"episode_{successful_episode_count}"] = info
                            with open(info_file_path, "w", encoding="utf-8") as file:
                                json.dump(info_db, file, ensure_ascii=False, indent=4)
                            
                            TASK_ENV.close_env(clear_cache=True)
                            TASK_ENV.merge_pkl_to_hdf5_video()
                            TASK_ENV.remove_data_cache()
                            
                            print(f"\033[92mNew episode {successful_episode_count} render SUCCESS (seed={next_new_seed})\033[0m")
                            successful_renders.append((new_traj_idx, next_new_seed))
                            successful_episode_count += 1
                            new_traj_idx += 1
                        else:
                            print(f"\r\033[K\033[91mNew seed {next_new_seed} render failed\033[0m")
                            TASK_ENV.close_env(clear_cache=True)
                            try:
                                TASK_ENV.remove_data_cache()
                            except:
                                pass
                        
                    else:
                        print(f"New seed {next_new_seed} planning failed")
                        TASK_ENV.close_env(clear_cache=True)
                        time.sleep(0.3)
                except Exception as e:
                    print(f"Error with seed {next_new_seed}: {e}")
                    TASK_ENV.close_env(clear_cache=True)
                    time.sleep(0.5)
                
                next_new_seed += 1
                regen_attempts += 1

        final_pairs = []
        existing_seed_count = min(st_idx, len(seed_list))
        final_pairs.extend((idx, seed_list[idx]) for idx in range(existing_seed_count))
        used_indices = {idx for idx in range(existing_seed_count)}
        for traj_idx, seed in successful_renders:
            if traj_idx in used_indices:
                continue
            final_pairs.append((traj_idx, seed))
            used_indices.add(traj_idx)

        traj_dir = os.path.join(args["save_path"], "_traj_data")
        if os.path.isdir(traj_dir):
            available_pairs = []
            for traj_idx, seed in final_pairs:
                traj_path = os.path.join(traj_dir, f"episode{traj_idx}.pkl")
                if os.path.exists(traj_path):
                    available_pairs.append((traj_idx, seed))
                else:
                    print(f"\033[93mWarning: Missing traj file for seed {seed} (episode{traj_idx}) - dropping\033[0m")
            final_pairs = available_pairs

            temp_dir = os.path.join(traj_dir, "_tmp_reorder")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            for new_idx, (old_idx, _) in enumerate(final_pairs):
                src = os.path.join(traj_dir, f"episode{old_idx}.pkl")
                dst = os.path.join(temp_dir, f"episode{new_idx}.pkl")
                if os.path.exists(src):
                    shutil.move(src, dst)

            for name in os.listdir(traj_dir):
                if name == os.path.basename(temp_dir):
                    continue
                if name.endswith(".pkl"):
                    os.remove(os.path.join(traj_dir, name))

            for name in os.listdir(temp_dir):
                shutil.move(os.path.join(temp_dir, name), os.path.join(traj_dir, name))
            shutil.rmtree(temp_dir)

        final_seed_list = [seed for (_, seed) in final_pairs]
        if original_seed_list != final_seed_list:
            archive_path = os.path.join(args["save_path"], "seed_archive.txt")
            with open(archive_path, "a") as file:
                file.write("original: " + " ".join(str(s) for s in original_seed_list) + "\n")
                file.write("adjusted: " + " ".join(str(s) for s in final_seed_list) + "\n\n")
        with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
            for sed in final_seed_list:
                file.write("%s " % sed)
        
        print(f"\n\033[92mData collection complete: {successful_episode_count} episodes\033[0m")
        if failed_episode_indices:
            print(f"\033[93mNote: {len(failed_episode_indices)} episodes failed during render\033[0m")

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
