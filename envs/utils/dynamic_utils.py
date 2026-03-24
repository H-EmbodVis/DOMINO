"""
Dynamic motion helper functions.
Used to support state saving, restoration, and execution counting for dynamic tasks.
"""
import numpy as np
import random
import torch
from typing import Dict, List, Any, Callable, Optional
import sapien


class RandomStateManager:
    """
    Random state manager.
    
    Used to maintain random state consistency between the planning and rendering phases of dynamic tasks.
    Supports state saving/restoration for numpy, Python random, and torch.
    """
    
    @staticmethod
    def save_state() -> Dict[str, Any]:
        """
        Save the state of all current random generators (serializable format).
        
        Returns:
            Dictionary containing all random states, which can be used for later restoration or serialization to a file.
        """
        # Convert torch state to bytes for pickle serialization
        torch_state = torch.get_rng_state()
        torch_state_bytes = torch_state.numpy().tobytes()
        
        torch_cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
            torch_cuda_states = [s.numpy().tobytes() for s in cuda_states]
        
        return {
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch': torch_state_bytes,
            'torch_cuda': torch_cuda_states,
        }
    
    @staticmethod
    def restore_state(state: Dict[str, Any]) -> None:
        """
        Restore random generators to the saved state.
        
        Args:
            state: State dictionary returned by save_state().
        """
        if state is None:
            return
            
        if 'numpy' in state and state['numpy'] is not None:
            np.random.set_state(state['numpy'])
            
        if 'python' in state and state['python'] is not None:
            random.setstate(state['python'])
            
        if 'torch' in state and state['torch'] is not None:
            # Restore torch state from bytes
            torch_state = state['torch']
            if isinstance(torch_state, bytes):
                torch_state = torch.from_numpy(
                    np.frombuffer(torch_state, dtype=np.uint8).copy()
                )
            torch.set_rng_state(torch_state)
            
        if 'torch_cuda' in state and state['torch_cuda'] is not None and torch.cuda.is_available():
            cuda_states = state['torch_cuda']
            if isinstance(cuda_states, list) and len(cuda_states) > 0:
                if isinstance(cuda_states[0], bytes):
                    cuda_states = [
                        torch.from_numpy(np.frombuffer(s, dtype=np.uint8).copy())
                        for s in cuda_states
                    ]
                torch.cuda.set_rng_state_all(cuda_states)
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set seed for all random generators.
        
        Args:
            seed: Random seed.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class DynamicMotionHelper:
    """Dynamic motion helper class, providing state management and execution counting functions."""
    
    @staticmethod
    def save_robot_state(robot) -> Dict[str, Any]:
        """
        Save robot state (joint positions, velocities, gripper states, etc.).
        
        Args:
            robot: Robot object.
            
        Returns:
            Dictionary containing the complete state of the robot.
        """
        initial_left_qpos = robot.left_entity.get_qpos().copy()
        initial_right_qpos = robot.right_entity.get_qpos().copy()
        
        initial_left_gripper_val = robot.get_left_gripper_val()
        initial_right_gripper_val = robot.get_right_gripper_val()
        
        initial_left_gripper_targets = [
            joint_info[0].get_drive_target()[0]
            for joint_info in robot.left_gripper
            if joint_info[0] is not None
        ]
        
        initial_right_gripper_targets = [
            joint_info[0].get_drive_target()[0]
            for joint_info in robot.right_gripper
            if joint_info[0] is not None
        ]
        
        return {
            'left_qpos': initial_left_qpos,
            'right_qpos': initial_right_qpos,
            'left_gripper_val': initial_left_gripper_val,
            'right_gripper_val': initial_right_gripper_val,
            'left_gripper_targets': initial_left_gripper_targets,
            'right_gripper_targets': initial_right_gripper_targets,
        }
    
    @staticmethod
    def save_actors_state(actors: List[Any], get_component_func: Callable) -> Dict[str, Any]:
        """
        Save pose and velocity states of multiple actors.
        
        Args:
            actors: List of Actor objects.
            get_component_func: Function to get rigid body dynamics component.
            
        Returns:
            Dictionary containing states of all actors.
        """
        actors_state = {}
        
        for i, actor in enumerate(actors):
            component = get_component_func(actor)
            if component is None:
                continue
                
            actor_key = f'actor_{i}'
            actors_state[actor_key] = {
                'actor': actor,
                'pose': actor.get_pose(),
                'component': component,
                'linear_velocity': component.get_linear_velocity().copy(),
                'angular_velocity': component.get_angular_velocity().copy(),
            }
        
        return actors_state
    
    @staticmethod
    def restore_robot_state(robot, state: Dict[str, Any], stabilization_steps: int = 0, scene=None):
        """
        Restore robot state.
        
        Args:
            robot: Robot object.
            state: Saved state dictionary.
            stabilization_steps: Number of stabilization steps.
            scene: Scene object (used to execute stabilization steps).
        """
        robot.left_entity.set_qpos(state['left_qpos'])
        robot.left_entity.set_qvel(np.zeros_like(state['left_qpos']))
        robot.right_entity.set_qpos(state['right_qpos'])
        robot.right_entity.set_qvel(np.zeros_like(state['right_qpos']))
        
        for joint in robot.left_arm_joints:
            joint_idx = robot.left_entity.get_active_joints().index(joint)
            joint.set_drive_target(state['left_qpos'][joint_idx])
            joint.set_drive_velocity_target(0.0)
        
        for joint in robot.right_arm_joints:
            joint_idx = robot.right_entity.get_active_joints().index(joint)
            joint.set_drive_target(state['right_qpos'][joint_idx])
            joint.set_drive_velocity_target(0.0)
        
        robot.left_gripper_val = state['left_gripper_val']
        robot.right_gripper_val = state['right_gripper_val']
        
        for idx, joint_info in enumerate(robot.left_gripper):
            if joint_info[0] is not None and idx < len(state['left_gripper_targets']):
                joint_info[0].set_drive_target(state['left_gripper_targets'][idx])
                joint_info[0].set_drive_velocity_target(0.0)
        
        for idx, joint_info in enumerate(robot.right_gripper):
            if joint_info[0] is not None and idx < len(state['right_gripper_targets']):
                joint_info[0].set_drive_target(state['right_gripper_targets'][idx])
                joint_info[0].set_drive_velocity_target(0.0)
        
        # Execute stabilization steps
        if scene is not None and stabilization_steps > 0:
            for _ in range(stabilization_steps):
                scene.step()
    
    @staticmethod
    def restore_actors_state(actors_state: Dict[str, Any], stabilization_steps: int = 0, scene=None):
        """
        Restore actors state.
        
        Args:
            actors_state: Saved actors state dictionary.
            stabilization_steps: Number of stabilization steps.
            scene: Scene object.
        """
        for actor_key, actor_data in actors_state.items():
            actor = actor_data['actor']
            component = actor_data['component']
            
            actor.actor.set_pose(actor_data['pose'])
            component.set_linear_velocity(actor_data['linear_velocity'])
            component.set_angular_velocity(actor_data['angular_velocity'])
        
        # Execute stabilization steps
        if scene is not None and stabilization_steps > 0:
            for _ in range(stabilization_steps):
                scene.step()


class StepCounter:
    """Step counter, used to count execution steps during dry run."""
    
    def __init__(self, scene):
        self.scene = scene
        self.counter = 0
        self.original_step = None
    
    def __enter__(self):
        """Enter context manager, start counting."""
        self.counter = 0
        self.original_step = self.scene.step
        
        def counting_step():
            self.counter += 1
            self.original_step()
        
        self.scene.step = counting_step
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, restore original step function."""
        self.scene.step = self.original_step
        return False
    
    def get_count(self) -> int:
        """Get count result."""
        return self.counter

