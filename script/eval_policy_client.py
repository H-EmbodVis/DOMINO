import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque
import base64
import numpy as np
from typing import Any

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

from generate_episode_instructions import *
from eval_metrics import EvalMetricsTracker, AggregatedMetrics, EpisodeMetrics

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)
            
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    
    return json.loads(json_str, object_hook=object_hook)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    def __init__(self, host='localhost', port=9999, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5
        
        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')
            
            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)
            
            # Receive and deserialize response
            response = self._recv_response()
            return response
            
        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")
        
        size = int.from_bytes(len_data, 'big')
        
        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        
        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        return response['res']

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
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

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100
    topk = 1

    # model = get_model(usr_args)
    model = ModelClient(port=port)
    st_seed, suc_num, aggregated_metrics = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   policy_conda_env=policy_conda_env)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    # Save basic result
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    # Save detailed metrics report
    metrics_report_path = os.path.join(save_dir, f"_metrics_report.txt")
    with open(metrics_report_path, "w") as file:
        file.write(aggregated_metrics.to_detailed_report())
    
    # Save metrics as JSON for programmatic access
    metrics_json_path = os.path.join(save_dir, f"_metrics.json")
    with open(metrics_json_path, "w") as file:
        summary = aggregated_metrics.get_summary()
        # Convert nan to null for JSON compatibility
        for k, v in summary.items():
            if isinstance(v, float) and np.isnan(v):
                summary[k] = None
        json.dump(summary, file, indent=2)
    
    # Save per-episode detailed results
    episodes_json_path = os.path.join(save_dir, f"_episodes_detail.json")
    with open(episodes_json_path, "w") as file:
        episodes_data = aggregated_metrics.get_all_episodes()
        json.dump(episodes_data, file, indent=2)

    print(f"Data has been saved to {file_path}")
    print(f"Metrics report saved to {metrics_report_path}")
    print(f"Per-episode details saved to {episodes_json_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    
    # Initialize aggregated metrics collector
    aggregated_metrics = AggregatedMetrics()

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        # Save dynamic motion info from Expert phase
        saved_dynamic_motion_info = getattr(TASK_ENV, '_saved_dynamic_motion_info', None)

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        
        # Restore dynamic motion info
        if saved_dynamic_motion_info is not None:
            TASK_ENV._saved_dynamic_motion_info = saved_dynamic_motion_info
        
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        # Initialize dynamic object motion for policy evaluation
        if args.get("use_dynamic", False):
            dynamic_init_success = TASK_ENV.init_dynamic_motion_for_eval()
            if not dynamic_init_success:
                print(f"Error: Failed to initialize dynamic motion for seed {now_seed}, skipping...")
                TASK_ENV.close_env()
                TASK_ENV.release_episode_resources()
                now_seed += 1
                succ_seed -= 1
                continue

        # Check object lifting and contact-stop configurations
        check_z_threshold = None
        target_actor_check = None
        start_z = None
        has_lifted = False
        stop_on_contact = True

        # Try to get config
        dynamic_config = getattr(TASK_ENV, 'get_dynamic_motion_config', lambda: None)()
        if dynamic_config and 'check_z_threshold' in dynamic_config:
            check_z_threshold = dynamic_config['check_z_threshold']
            if 'check_z_actor' in dynamic_config:
                target_actor_check = dynamic_config['check_z_actor']
            else:
                target_actor_check = dynamic_config['target_actor']
            start_z = target_actor_check.get_pose().p[2]
        if dynamic_config and 'stop_on_contact' in dynamic_config:
            stop_on_contact = dynamic_config['stop_on_contact']

        # Initialize metrics tracker for this episode
        metrics_tracker = EvalMetricsTracker(TASK_ENV, args)
        metrics_tracker.on_episode_start()
        TASK_ENV._metrics_tracker = metrics_tracker

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        fail_reason = None
        model.call(func_name='reset_model')
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            
            if check_z_threshold is not None and not has_lifted:
                curr_z = target_actor_check.get_pose().p[2]
                if curr_z - start_z > check_z_threshold:
                    has_lifted = True

            if TASK_ENV.eval_success:
                if check_z_threshold is not None and not has_lifted:
                    succ = False
                    fail_reason = "object_not_lifted"
                    break
                succ = True
                break
            
            # Check if dynamic object is out of bounds (lost from view)
            out_of_bounds = False
            if args.get("use_dynamic", False):
                # Stop motion if gripper contacts dynamic object
                if TASK_ENV.check_gripper_contact_dynamic_object() and stop_on_contact:
                    TASK_ENV.stop_dynamic_object_motion()
                
                # Only check boundaries if object hasn't been grasped
                if not getattr(TASK_ENV, '_dynamic_object_stopped', False):
                    if not TASK_ENV.check_dynamic_object_boundary():
                        TASK_ENV.eval_fail = True
                        fail_reason = "out_of_bounds"
                        out_of_bounds = True
            
            if out_of_bounds:
                metrics_tracker.record_out_of_bounds()
                break
            if TASK_ENV.eval_fail:
                fail_reason = "eval_fail"
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        elif fail_reason == "out_of_bounds":
            print("\033[91mFail! (Object out of bounds)\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Collect episode metrics
        episode_metrics = metrics_tracker.get_episode_metrics(succ, fail_reason, seed=now_seed)
        aggregated_metrics.add_episode(episode_metrics)
        TASK_ENV._metrics_tracker = None
        
        # Print episode metrics summary
        print(f"  MS: \033[96m{episode_metrics.manipulation_score:.1f}\033[0m | "
              f"RC: \033[96m{episode_metrics.route_completion:.1f}%\033[0m")
        # print(f"  MS: \033[96m{episode_metrics.manipulation_score:.1f}\033[0m | "
        #       f"RC: \033[96m{episode_metrics.route_completion:.1f}%\033[0m | "
        #       f"Eff: \033[96m{episode_metrics.efficiency:.1f}\033[0m | "
        #       f"Comfort: \033[96m{episode_metrics.comfort_score:.1f}\033[0m")
        if episode_metrics.penalty_events:
            penalty_str = ", ".join([f"{p.event_type}({p.penalty_factor})" for p in episode_metrics.penalty_events])
            print(f"  Penalties: \033[93m{penalty_str}\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((TASK_ENV.test_num + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.release_episode_resources()

        TASK_ENV.test_num += 1

        # Print running statistics
        summary = aggregated_metrics.get_summary()
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m | "
            f"Avg MS: \033[95m{summary['manipulation_score_mean']:.1f}\033[0m | "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, TASK_ENV.suc, aggregated_metrics


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config['port'] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
