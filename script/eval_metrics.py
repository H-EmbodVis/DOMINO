"""
Evaluation Metrics Module for Dynamic Manipulation Tasks.

Metrics:
1. Manipulation Score (MS): Route completion with penalty factors
2. Efficiency (E_eff): Steps efficiency score
3. Comfort Score (C_comf): Motion smoothness based on jerk
4. Penalty Events: Collision, joint limits, target loss
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque


@dataclass
class PenaltyEvent:
    """Single penalty event record."""
    event_type: str      # collision, joint_limit, target_loss
    timestep: int
    penalty_factor: float
    details: str = ""


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    success: bool = False
    steps_taken: int = 0
    max_steps: int = 0
    
    # Seed used for this episode
    seed: Optional[int] = None
    
    # Route Completion (0-100)
    route_completion: float = 0.0
    
    # Manipulation Score
    manipulation_score: float = 0.0
    
    # Efficiency (0-100 if success, 0 if fail)
    # efficiency: float = 0.0
    
    # Comfort/Smoothness (0-100)
    # comfort_score: float = 0.0
    # avg_jerk: float = 0.0
    
    # Penalty events
    penalty_events: List[PenaltyEvent] = field(default_factory=list)
    total_penalty_factor: float = 1.0
    
    # Failure reason
    fail_reason: Optional[str] = None


class EvalMetricsTracker:
    """
    Tracks and computes evaluation metrics during closed-loop policy evaluation.
    
    Penalty Factors:
    - Out of Bounds: 0.5
    - Unexpected Collision with Clutter: 0.8
    - Joint Limit: 0.8 (disabled in simulation)
    
    Note: Joint limit detection is disabled in simulation because TOPP trajectory
    smoothing and PD control prevent large joint jumps. This issue only occurs
    on real robots with direct joint position commands.
    """
    
    PENALTY_OUT_OF_BOUNDS = 0.5  # Target exits workspace
    PENALTY_COLLISION = 0.8
    
    # Jerk normalization parameters for comfort score
    # Based on empirical analysis of robot manipulation:
    # - Policy step frequency: ~10 Hz (each take_action step)
    # - Typical EE velocity: 0.05-0.2 m/s
    # - Typical jerk (with dt=1 policy step): 0.001-0.1 m/step^3
    # Using exponential decay: comfort = 100 * exp(-jerk / JERK_SCALE)
    # JERK_SCALE chosen so that "good" motion (jerk ~0.01) gives ~60-80 score
    # and "poor" motion (jerk ~0.1) gives ~20-40 score
    JERK_SCALE = 0.02  # Scale parameter for exponential decay
    
    def __init__(self, env, args: dict):
        """
        Args:
            env: Task environment instance
            args: Configuration dictionary
        """
        self.env = env
        self.args = args
        self.use_dynamic = args.get("use_dynamic", False)
        
        # Reset state
        self.reset()
        
    def reset(self):
        """Reset tracker for new episode."""
        self.step_count = 0
        self.penalty_events: List[PenaltyEvent] = []
        
        # End effector trajectory tracking
        self.ee_positions_left: List[np.ndarray] = []
        self.ee_positions_right: List[np.ndarray] = []
        self.ee_velocities_left: List[np.ndarray] = []
        self.ee_velocities_right: List[np.ndarray] = []
        self.ee_accelerations_left: List[np.ndarray] = []
        self.ee_accelerations_right: List[np.ndarray] = []
        
        # Initial positions for route completion
        self.initial_ee_left: Optional[np.ndarray] = None
        self.initial_ee_right: Optional[np.ndarray] = None
        
        # Dynamic motion tracking
        self.dynamic_start_step: Optional[int] = None
        self.dynamic_end_step: Optional[int] = None
        self.intercept_position: Optional[np.ndarray] = None
        self.dynamic_target_actor: Optional[Any] = None
        self.dynamic_target_positions: List[np.ndarray] = []
        self.initial_target_position: Optional[np.ndarray] = None
        self.out_of_bounds_target_position: Optional[np.ndarray] = None
        
        # Collision tracking
        self.previous_contacts: set = set()
        self.penalized_clutter_objects: set = set()  # Only penalize each clutter once
        
    def on_episode_start(self):
        """Called when episode starts, after environment setup."""
        self.reset()
        
        # Record initial EE positions
        try:
            left_pose = self.env.robot.get_left_ee_pose()
            right_pose = self.env.robot.get_right_ee_pose()
            self.initial_ee_left = np.array(left_pose[:3])
            self.initial_ee_right = np.array(right_pose[:3])
        except Exception:
            pass
        
        # Get intercept position for dynamic tasks
        if self.use_dynamic:
            saved_info = getattr(self.env, '_saved_dynamic_motion_info', None)
            if saved_info:
                config = self.env.get_dynamic_motion_config()
                if config:
                    self.intercept_position = config.get('end_position')
            self.dynamic_target_actor = self._get_dynamic_target_actor()
            self.initial_target_position = self._get_dynamic_target_position()
        
        # Build set of task-related actor names (should not be counted as collision)
        self._build_task_actor_names()
                    
    def on_step(self, out_of_bounds: bool = False) -> None:
        """
        Called after each policy step. Records metrics and penalty events.
        
        Args:
            out_of_bounds: Whether target went out of bounds (from env check)
        """
        self.step_count += 1
        
        # Track EE trajectory
        self._track_ee_trajectory()
        if self.use_dynamic:
            target_pos = self._get_dynamic_target_position()
            if target_pos is not None:
                self.dynamic_target_positions.append(target_pos)
        
        # Record out_of_bounds as penalty
        if out_of_bounds:
            self._add_penalty(PenaltyEvent(
                event_type="out_of_bounds",
                timestep=self.step_count,
                penalty_factor=self.PENALTY_OUT_OF_BOUNDS,
                details="Target object left workspace/view"
            ))
        
        # Check unexpected collisions with clutter objects
        collision_info = self._check_clutter_collision()
        if collision_info:
            self._add_penalty(PenaltyEvent(
                event_type="collision",
                timestep=self.step_count,
                penalty_factor=self.PENALTY_COLLISION,
                details=collision_info
            ))
        
        # Note: Joint limit detection is disabled in simulation.
        # In simulation, TOPP trajectory smoothing and PD control prevent
        # large joint jumps ("big loop" movements). This issue only occurs
        # on real robots with direct joint position commands.
    
    def _track_ee_trajectory(self):
        """Record end effector positions and compute derivatives."""
        try:
            left_pose = self.env.robot.get_left_ee_pose()
            right_pose = self.env.robot.get_right_ee_pose()
            
            left_pos = np.array(left_pose[:3])
            right_pos = np.array(right_pose[:3])
            
            self.ee_positions_left.append(left_pos)
            self.ee_positions_right.append(right_pos)
            
            # Compute velocity (first derivative)
            if len(self.ee_positions_left) >= 2:
                dt = 1.0  # Normalized timestep
                vel_left = (self.ee_positions_left[-1] - self.ee_positions_left[-2]) / dt
                vel_right = (self.ee_positions_right[-1] - self.ee_positions_right[-2]) / dt
                self.ee_velocities_left.append(vel_left)
                self.ee_velocities_right.append(vel_right)
                
                # Compute acceleration (second derivative)
                if len(self.ee_velocities_left) >= 2:
                    acc_left = (self.ee_velocities_left[-1] - self.ee_velocities_left[-2]) / dt
                    acc_right = (self.ee_velocities_right[-1] - self.ee_velocities_right[-2]) / dt
                    self.ee_accelerations_left.append(acc_left)
                    self.ee_accelerations_right.append(acc_right)
                    
        except Exception:
            pass
    
    def _build_task_actor_names(self):
        """
        Build a set of all task-related actor names that should NOT be counted as collision.
        
        This includes:
        - All actors defined in the task (e.g., hammer, block, cup, coaster, etc.)
        - Table and ground
        - Robot gripper links
        
        Only collisions with clutter objects should be penalized.
        """
        self.task_actor_names: set = {'table', 'ground'}
        
        # Add robot gripper names
        try:
            self.task_actor_names.update(self.env.robot.gripper_name)
        except Exception:
            pass
        
        # Add robot link names
        self.robot_link_names: set = set()
        try:
            for link in self.env.robot.left_entity.get_links():
                self.robot_link_names.add(link.name)
            for link in self.env.robot.right_entity.get_links():
                self.robot_link_names.add(link.name)
        except Exception:
            pass
        
        # Collect all task-related actors from the environment
        # These are typically stored as instance attributes on the task class
        try:
            # Get all actors in the scene
            for actor in self.env.scene.get_all_actors():
                actor_name = actor.name
                # Skip clutter objects (they have unique names with timestamps or indices)
                # Clutter objects are tracked in _clutter_actor_refs
                clutter_refs = getattr(self.env, '_clutter_actor_refs', {})
                if actor_name in clutter_refs:
                    continue
                # Skip robot links
                if actor_name in self.robot_link_names:
                    continue
                # This is a task actor, add to allowed set
                self.task_actor_names.add(actor_name)
        except Exception:
            pass
        
        # Build clutter object names set
        self.clutter_object_names: set = set()
        try:
            clutter_refs = getattr(self.env, '_clutter_actor_refs', {})
            self.clutter_object_names = set(clutter_refs.keys())
        except Exception:
            pass
    
    def _check_clutter_collision(self) -> Optional[str]:
        """
        Check for collisions between robot and clutter objects.
        
        Only clutter objects (random objects placed for domain randomization)
        should trigger collision penalties. Task-related objects (hammer, block,
        cup, etc.) are part of the manipulation task and contacts with them
        are expected.
        
        Each clutter object is only penalized once per episode.
        
        Returns:
            Collision info string if new clutter collision detected, None otherwise.
        """
        # If no clutter objects exist, no collision check needed
        if not self.clutter_object_names:
            return None
        
        try:
            contacts = self.env.scene.get_contacts()
            
            for contact in contacts:
                body0_name = contact.bodies[0].entity.name
                body1_name = contact.bodies[1].entity.name
                
                # Check if robot is involved
                robot_involved = (body0_name in self.robot_link_names or 
                                  body1_name in self.robot_link_names)
                if not robot_involved:
                    continue
                
                # Get the other object (non-robot)
                other_name = body1_name if body0_name in self.robot_link_names else body0_name
                
                # Skip if not a clutter object
                if other_name not in self.clutter_object_names:
                    continue
                
                # Skip if this clutter object has already been penalized
                if other_name in self.penalized_clutter_objects:
                    continue
                
                # New clutter collision detected - mark as penalized
                self.penalized_clutter_objects.add(other_name)
                robot_part = body0_name if body0_name in self.robot_link_names else body1_name
                return f"Clutter collision: {robot_part} <-> {other_name}"
            
        except Exception:
            pass
            
        return None
    
    def _add_penalty(self, event: PenaltyEvent):
        """Add a penalty event."""
        self.penalty_events.append(event)

    def record_out_of_bounds(self) -> None:
        """Record out-of-bounds penalty without advancing step count."""
        if self.use_dynamic:
            self.out_of_bounds_target_position = self._get_dynamic_target_position()
        self._add_penalty(PenaltyEvent(
            event_type="out_of_bounds",
            timestep=self.step_count,
            penalty_factor=self.PENALTY_OUT_OF_BOUNDS,
            details="Target object left workspace/view"
        ))

    def _get_dynamic_target_actor(self) -> Optional[Any]:
        """Get dynamic target actor reference for eval, if available."""
        target_actor = getattr(self.env, "_eval_target_actor", None)
        if target_actor is not None:
            return target_actor
        try:
            config = self.env.get_dynamic_motion_config()
            if config:
                return config.get("target_actor")
        except Exception:
            pass
        return None

    def _get_dynamic_target_position(self) -> Optional[np.ndarray]:
        """Get current dynamic target position (xyz)."""
        actor = self.dynamic_target_actor
        if actor is None:
            actor = self._get_dynamic_target_actor()
            self.dynamic_target_actor = actor
        if actor is None:
            return None
        try:
            pose = actor.get_pose()
            return np.array(pose.p)
        except Exception:
            try:
                pose = actor.actor.get_pose()
                return np.array(pose.p)
            except Exception:
                return None
    
    def compute_route_completion(self, success: bool) -> float:
        """
        Compute route completion score (0-100).
        For dynamic tasks: measures progress toward the final target position
        based on each arm's initial and final EE positions.
        """
        if success:
            return 100.0
        
        if not self.use_dynamic:
            # For static tasks, use simple success-based completion
            return 100.0 if success else 0.0
        
        try:
            # Resolve final target position
            final_target = self.out_of_bounds_target_position
            if final_target is None and self.dynamic_target_positions:
                final_target = self.dynamic_target_positions[-1]

            if final_target is None:
                return 0.0
            
            def compute_progress(initial_ee: Optional[np.ndarray], final_ee: Optional[np.ndarray]) -> Optional[float]:
                if initial_ee is None or final_ee is None:
                    return None
                initial_dist = np.linalg.norm(initial_ee - final_target)
                final_dist = np.linalg.norm(final_ee - final_target)
                if initial_dist < 0.01:
                    return 1.0 if final_dist < 0.05 else 0.5
                progress = (initial_dist - final_dist) / initial_dist
                return float(np.clip(progress, 0, 1))
            
            initial_left = self.initial_ee_left if self.initial_ee_left is not None else (
                self.ee_positions_left[0] if self.ee_positions_left else None
            )
            initial_right = self.initial_ee_right if self.initial_ee_right is not None else (
                self.ee_positions_right[0] if self.ee_positions_right else None
            )
            final_left = self.ee_positions_left[-1] if self.ee_positions_left else self.initial_ee_left
            final_right = self.ee_positions_right[-1] if self.ee_positions_right else self.initial_ee_right
            
            progress_left = compute_progress(initial_left, final_left)
            progress_right = compute_progress(initial_right, final_right)
            
            progress_vals = [p for p in [progress_left, progress_right] if p is not None]
            if not progress_vals:
                return 0.0
            
            return max(progress_vals) * 100.0
            
        except Exception:
            return 0.0
    
    def compute_jerk_score(self) -> Tuple[float, float]:
        """
        Compute comfort score based on jerk (derivative of acceleration).
        
        Jerk is computed as the rate of change of acceleration.
        Requires at least 4 position samples to compute 1 jerk value:
        positions(N) -> velocities(N-1) -> accelerations(N-2) -> jerks(N-3)
        
        Uses exponential decay formula for more robust scoring:
        comfort = 100 * exp(-avg_jerk / JERK_SCALE)
        
        This ensures:
        - Zero jerk -> 100 score (perfect smoothness)
        - Moderate jerk (~JERK_SCALE) -> ~37 score
        - High jerk (>> JERK_SCALE) -> approaches 0 asymptotically
        
        Returns:
            Tuple[float, float]: (comfort_score 0-100, average_jerk)
        """
        # Need at least 2 acceleration samples to compute 1 jerk
        if len(self.ee_accelerations_left) < 2 or len(self.ee_accelerations_right) < 2:
            # Not enough data - check if we have positions and compute directly
            if len(self.ee_positions_left) < 4:
                return 100.0, 0.0
        
        try:
            jerks = []
            
            # Compute jerk from accelerations if available
            if len(self.ee_accelerations_left) >= 2:
                for i in range(1, len(self.ee_accelerations_left)):
                    jerk_left = self.ee_accelerations_left[i] - self.ee_accelerations_left[i-1]
                    jerks.append(np.linalg.norm(jerk_left))
                for i in range(1, len(self.ee_accelerations_right)):
                    jerk_right = self.ee_accelerations_right[i] - self.ee_accelerations_right[i-1]
                    jerks.append(np.linalg.norm(jerk_right))
            else:
                # Fallback: compute jerk directly from positions using finite differences
                # jerk ≈ (p[i+3] - 3*p[i+2] + 3*p[i+1] - p[i]) for third derivative
                for positions in [self.ee_positions_left, self.ee_positions_right]:
                    if len(positions) >= 4:
                        for i in range(len(positions) - 3):
                            jerk = (positions[i+3] - 3*positions[i+2] + 
                                   3*positions[i+1] - positions[i])
                            jerks.append(np.linalg.norm(jerk))
            
            if not jerks:
                return 100.0, 0.0
            
            avg_jerk = np.mean(jerks)
            
            # Comfort score using exponential decay for smooth, bounded scoring
            # exp(-x/scale) gives values from 1 (x=0) to ~0 (x >> scale)
            # This avoids the cliff-edge behavior of linear clamping
            comfort = np.exp(-avg_jerk / self.JERK_SCALE) * 100.0
            
            return comfort, avg_jerk
            
        except Exception as e:
            return 100.0, 0.0
    
    def compute_efficiency(self, success: bool, steps: int, max_steps: int) -> float:
        """
        Compute efficiency score.
        E_eff = (1 - T_success / T_max) * 100 if success, else 0
        """
        if not success:
            return 0.0
        
        if max_steps <= 0:
            return 0.0
        
        return (1 - steps / max_steps) * 100.0
    
    def compute_total_penalty_factor(self) -> float:
        """Compute combined penalty factor from all events."""
        factor = 1.0
        for event in self.penalty_events:
            factor *= event.penalty_factor
        return factor
    
    def get_episode_metrics(self, success: bool, fail_reason: Optional[str] = None, seed: Optional[int] = None) -> EpisodeMetrics:
        """
        Compute and return all metrics for the episode.
        
        Args:
            success: Whether the task was successful
            fail_reason: Reason for failure if applicable
            seed: Random seed used for this episode
        """
        max_steps = getattr(self.env, 'step_lim', 1000)
        
        # Route completion
        route_completion = self.compute_route_completion(success)
        
        # Penalty factor
        total_penalty = self.compute_total_penalty_factor()
        
        # Manipulation score
        manipulation_score = route_completion * total_penalty
        
        # Efficiency
        # efficiency = self.compute_efficiency(success, self.step_count, max_steps)
        
        # Comfort score
        # comfort_score, avg_jerk = self.compute_jerk_score()
        
        return EpisodeMetrics(
            success=success,
            steps_taken=self.step_count,
            max_steps=max_steps,
            seed=seed,
            route_completion=route_completion,
            manipulation_score=manipulation_score,
            # efficiency=efficiency,
            # comfort_score=comfort_score,
            # avg_jerk=avg_jerk,
            penalty_events=self.penalty_events.copy(),
            total_penalty_factor=total_penalty,
            fail_reason=fail_reason
        )


class AggregatedMetrics:
    """Aggregates metrics across multiple episodes."""
    
    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
    
    def add_episode(self, metrics: EpisodeMetrics):
        """Add episode metrics."""
        self.episodes.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary statistics."""
        if not self.episodes:
            return {}
        
        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e.success)
        
        # Average metrics (only for valid values)
        ms_scores = [e.manipulation_score for e in self.episodes]
        rc_scores = [e.route_completion for e in self.episodes]
        # comfort_scores = [e.comfort_score for e in self.episodes]
        
        # Efficiency only for successful episodes
        # eff_scores = [e.efficiency for e in self.episodes if e.success]
        
        # Steps for successful episodes
        # success_steps = [e.steps_taken for e in self.episodes if e.success]
        
        # Penalty counts
        collision_count = sum(
            1 for e in self.episodes 
            for p in e.penalty_events if p.event_type == "collision"
        )
        out_of_bounds_count = sum(
            1 for e in self.episodes 
            for p in e.penalty_events if p.event_type == "out_of_bounds"
        )
        
        return {
            "total_episodes": n,
            "success_count": successes,
            "success_rate": successes / n * 100 if n > 0 else 0,
            
            "manipulation_score_mean": np.mean(ms_scores) if ms_scores else 0,
            "manipulation_score_std": np.std(ms_scores) if ms_scores else 0,
            
            "route_completion_mean": np.mean(rc_scores) if rc_scores else 0,
            "route_completion_std": np.std(rc_scores) if rc_scores else 0,
            
            # "efficiency_mean": np.mean(eff_scores) if eff_scores else float('nan'),
            # "efficiency_std": np.std(eff_scores) if eff_scores else float('nan'),
            
            # "comfort_score_mean": np.mean(comfort_scores) if comfort_scores else 0,
            # "comfort_score_std": np.std(comfort_scores) if comfort_scores else 0,
            
            # "avg_steps_success": np.mean(success_steps) if success_steps else float('nan'),
            
            "penalty_clutter_collision_total": collision_count,
            # "penalty_joint_limit_total": joint_limit_count,
            "penalty_out_of_bounds_total": out_of_bounds_count,
        }
    
    def to_csv_row(self) -> str:
        """Generate CSV format summary."""
        summary = self.get_summary()
        headers = list(summary.keys())
        values = [str(summary[k]) for k in headers]
        return ",".join(headers) + "\n" + ",".join(values)
    
    def to_detailed_report(self) -> str:
        """Generate detailed text report."""
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "EVALUATION METRICS REPORT",
            "=" * 60,
            "",
            f"Total Episodes: {summary['total_episodes']}",
            f"Success Rate: {summary['success_rate']:.1f}% ({summary['success_count']}/{summary['total_episodes']})",
            "",
            "--- Manipulation Score ---",
            f"  Mean: {summary['manipulation_score_mean']:.2f}",
            f"  Std:  {summary['manipulation_score_std']:.2f}",
            "",
            "--- Route Completion ---",
            f"  Mean: {summary['route_completion_mean']:.2f}%",
            f"  Std:  {summary['route_completion_std']:.2f}",
            "",
            # "--- Efficiency (Success Only) ---",
            # f"  Mean: {summary['efficiency_mean']:.2f}" if not np.isnan(summary['efficiency_mean']) else "  Mean: N/A",
            # f"  Avg Steps: {summary['avg_steps_success']:.1f}" if not np.isnan(summary['avg_steps_success']) else "  Avg Steps: N/A",
            # "",
            # "--- Comfort Score (Smoothness) ---",
            # f"  Mean: {summary['comfort_score_mean']:.2f}",
            # f"  Std:  {summary['comfort_score_std']:.2f}",
            # "",
            "--- Penalty Events ---",
            f"  Clutter Collisions: {summary['penalty_clutter_collision_total']}",
            f"  Out of Bounds:      {summary['penalty_out_of_bounds_total']}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """
        Get detailed metrics for all episodes as a list of dictionaries.
        Useful for saving per-episode results to JSON or CSV.
        
        Returns:
            List of dictionaries, each containing all metrics for one episode
        """
        episodes_data = []
        for i, episode in enumerate(self.episodes):
            episode_dict = {
                "episode_id": i + 1,  # 1-indexed for readability
                "seed": episode.seed,
                "success": episode.success,
                "steps_taken": episode.steps_taken,
                "max_steps": episode.max_steps,
                "route_completion": episode.route_completion,
                "manipulation_score": episode.manipulation_score,
                # "efficiency": episode.efficiency if episode.success else None,  # None for failed episodes
                # "comfort_score": episode.comfort_score,
                # "avg_jerk": episode.avg_jerk,
                "total_penalty_factor": episode.total_penalty_factor,
                "fail_reason": episode.fail_reason,
                "penalty_events": [
                    {
                        "event_type": p.event_type,
                        "timestep": p.timestep,
                        "penalty_factor": p.penalty_factor,
                        "details": p.details
                    }
                    for p in episode.penalty_events
                ]
            }
            episodes_data.append(episode_dict)
        return episodes_data
