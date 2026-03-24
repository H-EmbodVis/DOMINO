import numpy as np
import sapien
from typing import Tuple, Optional, Callable


class TrajectoryGenerator:
    """Generate dynamic trajectories with different complexity levels."""
    
    def __init__(
        self,
        table_bounds: Tuple[float, float, float, float] = (-0.35, 0.35, -0.15, 0.25),
        workspace_z: float = 0.76,
        timestep: float = 1/250,
        max_retries: int = 1000,
    ):
        """
        Args:
            table_bounds: (x_min, x_max, y_min, y_max) workspace boundaries
            workspace_z: Height of the workspace plane
            timestep: Simulation timestep
            max_retries: Maximum number of retries for trajectory generation
        """
        self.x_min, self.x_max, self.y_min, self.y_max = table_bounds
        self.workspace_z = workspace_z
        self.timestep = timestep
        self.max_retries = max_retries
        
        # Define quadrants for visibility constraint
        self.quadrant_edges = {
            'left_top': {'x_edge': self.x_min, 'y_edge': self.y_max},
            'right_top': {'x_edge': self.x_max, 'y_edge': self.y_max},
            'left_bottom': {'x_edge': self.x_min, 'y_edge': self.y_min},
            'right_bottom': {'x_edge': self.x_max, 'y_edge': self.y_min},
        }
    
    def _get_quadrant(self, x: float, y: float) -> str:
        """Determine which quadrant a point belongs to."""
        x_center = (self.x_min + self.x_max) / 2
        y_center = (self.y_min + self.y_max) / 2
        
        if x < x_center and y >= y_center:
            return 'left_top'
        elif x >= x_center and y >= y_center:
            return 'right_top'
        elif x < x_center and y < y_center:
            return 'left_bottom'
        else:
            return 'right_bottom'
    
    def _check_trajectory_visibility(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        trajectory_func: Optional[Callable] = None,
        num_samples: int = 20,
    ) -> bool:
        """
        Check if trajectory maintains good visibility based on quadrant rules.
        The trajectory should not exit through forbidden edges.
        """
        end_quadrant = self._get_quadrant(end_pos[0], end_pos[1])
        edges = self.quadrant_edges[end_quadrant]
        
        # Sample points along trajectory
        if trajectory_func is None:
            # Linear trajectory
            t_values = np.linspace(0, 1, num_samples)
            trajectory_points = [start_pos + t * (end_pos - start_pos) for t in t_values]
        else:
            # Custom trajectory function
            trajectory_points = trajectory_func(num_samples)
        
        # Check each point
        for point in trajectory_points:
            # Check if exits through forbidden edges
            if end_quadrant == 'left_top':
                if point[0] < edges['x_edge'] or point[1] > edges['y_edge']:
                    return False
            elif end_quadrant == 'right_top':
                if point[0] > edges['x_edge'] or point[1] > edges['y_edge']:
                    return False
            elif end_quadrant == 'left_bottom':
                if point[0] < edges['x_edge'] or point[1] < edges['y_edge']:
                    return False
            elif end_quadrant == 'right_bottom':
                if point[0] > edges['x_edge'] or point[1] < edges['y_edge']:
                    return False
            
            # Check if point is within bounds
            if not (self.x_min <= point[0] <= self.x_max and 
                    self.y_min <= point[1] <= self.y_max):
                return False
        
        return True
    
    def _sample_velocity_magnitude(self, dynamic_coefficient: float) -> float:
        """Sample velocity magnitude from half-sided normal distribution."""
        # Use half-normal distribution centered at dynamic_coefficient
        sigma = dynamic_coefficient / 3.0  # ~99.7% within coefficient
        velocity_mag = np.abs(np.random.normal(dynamic_coefficient * 0.8, sigma))
        # Clamp to reasonable range
        return np.clip(velocity_mag, 0.01, dynamic_coefficient)
    
    def generate_level1_trajectory(
        self,
        end_position: np.ndarray,
        dynamic_coefficient: float,
        total_duration: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Level 1: Constant velocity motion.
        
        Returns:
            start_position: Initial position for the object
            velocity: Constant velocity vector
            success: Whether generation succeeded
        """
        for attempt in range(self.max_retries):
            # Sample velocity magnitude
            velocity_mag = self._sample_velocity_magnitude(dynamic_coefficient)
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            velocity_2d = velocity_mag * np.array([np.cos(angle), np.sin(angle)])
            velocity = np.array([velocity_2d[0], velocity_2d[1], 0.0])
            
            # Calculate start position
            displacement = velocity * total_duration
            start_position = end_position - displacement
            start_position[2] = self.workspace_z
            
            # Check visibility constraint
            if self._check_trajectory_visibility(start_position, end_position):
                return start_position, velocity, True
        
        return None, None, False
    
    def generate_level2_trajectory(
        self,
        end_position: np.ndarray,
        dynamic_coefficient: float,
        total_duration: float,
    ) -> Tuple[np.ndarray, Callable, bool, Optional[dict]]:
        """
        Level 2: High-order polynomial trajectory with variable velocity.
        
        Returns:
            start_position: Initial position for the object
            trajectory_function: Function that returns position at time t
            success: Whether generation succeeded
            poly_coeffs: Dictionary containing polynomial coefficients for serialization
        """
        for attempt in range(self.max_retries):
            # Generate random control points for smooth curve
            num_control_points = np.random.randint(3, 6)
            
            # Start with random position
            start_x = np.random.uniform(self.x_min, self.x_max)
            start_y = np.random.uniform(self.y_min, self.y_max)
            start_position = np.array([start_x, start_y, self.workspace_z])
            
            # Generate intermediate control points
            control_points_x = [start_x]
            control_points_y = [start_y]
            
            for i in range(1, num_control_points - 1):
                # Interpolate with some randomness
                t = i / (num_control_points - 1)
                x = start_x + t * (end_position[0] - start_x) + np.random.uniform(-0.1, 0.1)
                y = start_y + t * (end_position[1] - start_y) + np.random.uniform(-0.1, 0.1)
                control_points_x.append(np.clip(x, self.x_min, self.x_max))
                control_points_y.append(np.clip(y, self.y_min, self.y_max))
            
            control_points_x.append(end_position[0])
            control_points_y.append(end_position[1])
            
            # Fit polynomial
            t_control = np.linspace(0, 1, num_control_points)
            poly_degree = min(num_control_points - 1, 5)
            coeffs_x = np.polyfit(t_control, control_points_x, poly_degree)
            coeffs_y = np.polyfit(t_control, control_points_y, poly_degree)
            
            poly_x = np.poly1d(coeffs_x)
            poly_y = np.poly1d(coeffs_y)
            
            # Create trajectory function
            def get_trajectory_points(num_samples):
                t_samples = np.linspace(0, 1, num_samples)
                points = []
                for t in t_samples:
                    x = poly_x(t)
                    y = poly_y(t)
                    points.append(np.array([x, y, self.workspace_z]))
                return points
            
            # Check visibility and velocity constraints
            trajectory_points = get_trajectory_points(50)
            
            # Calculate total arc length of trajectory (sum of all segment distances)
            total_arc_length = 0
            for i in range(1, len(trajectory_points)):
                segment_distance = np.linalg.norm(
                    (trajectory_points[i] - trajectory_points[i-1])[:2]
                )
                total_arc_length += segment_distance
            
            # Average velocity = total arc length / total duration
            average_velocity = total_arc_length / total_duration
            
            # Check if average velocity is within limits and trajectory is visible
            if (average_velocity <= dynamic_coefficient and 
                self._check_trajectory_visibility(
                    start_position, end_position, get_trajectory_points
                )):
                
                # Create time-parameterized functions
                def create_trajectory_func(poly_x_inner, poly_y_inner, duration):
                    def trajectory_at_time(t):
                        normalized_t = np.clip(t / duration, 0, 1)
                        x = float(poly_x_inner(normalized_t))
                        y = float(poly_y_inner(normalized_t))
                        return np.array([x, y, self.workspace_z])
                    return trajectory_at_time
                
                trajectory_func = create_trajectory_func(poly_x, poly_y, total_duration)
                
                # Return polynomial coefficients for serialization
                poly_coeffs = {
                    'poly_x_coeffs': coeffs_x.tolist(),
                    'poly_y_coeffs': coeffs_y.tolist(),
                }
                
                return start_position, trajectory_func, True, poly_coeffs
        
        return None, None, False, None
    
    def generate_level3_trajectory(
        self,
        end_position: np.ndarray,
        dynamic_coefficient: float,
        total_duration: float,
    ) -> Tuple[np.ndarray, list, bool]:
        """
        Level 3: Unpredictable motion with sudden transitions.
        Randomly divides time into 2-3 segments, each using Level 1 (20%) or Level 2 (80%) motion.
        
        Returns:
            start_position: Initial position for the object
            segment_trajectories: List of trajectory segments with transition info
            success: Whether generation succeeded
        """
        for attempt in range(self.max_retries):
            # Randomly divide total duration into 2-3 segments
            num_segments = np.random.randint(2, 4)
            
            # Generate random time splits
            split_ratios = np.random.dirichlet(np.ones(num_segments))
            segment_durations = split_ratios * total_duration
            
            # Start from random position
            start_x = np.random.uniform(self.x_min, self.x_max)
            start_y = np.random.uniform(self.y_min, self.y_max)
            current_position = np.array([start_x, start_y, self.workspace_z])
            start_position = current_position.copy()
            
            segment_trajectories = []
            trajectory_points = [current_position.copy()]
            
            # Generate each segment
            for seg_idx in range(num_segments):
                seg_duration = segment_durations[seg_idx]
                
                # Determine if this is the last segment
                is_last_segment = (seg_idx == num_segments - 1)
                
                if is_last_segment:
                    # Last segment should end at target
                    target_pos = end_position
                else:
                    # Intermediate target: random position in workspace
                    target_x = np.random.uniform(self.x_min, self.x_max)
                    target_y = np.random.uniform(self.y_min, self.y_max)
                    target_pos = np.array([target_x, target_y, self.workspace_z])
                
                # 20% chance Level 1 (constant velocity), 80% chance Level 2 (polynomial)
                use_level1 = np.random.random() < 0.2
                
                if use_level1:
                    # Level 1: Constant velocity segment
                    velocity = (target_pos - current_position) / seg_duration
                    velocity_mag = np.linalg.norm(velocity[:2])
                    
                    # Adjust velocity to respect dynamic coefficient
                    if velocity_mag > dynamic_coefficient:
                        velocity = velocity * (dynamic_coefficient / velocity_mag)
                        # Recalculate actual end position
                        actual_end_pos = current_position + velocity * seg_duration
                    else:
                        actual_end_pos = target_pos
                    
                    segment_trajectories.append({
                        'type': 'velocity',
                        'start_pos': current_position.copy(),
                        'velocity': velocity,
                        'duration': seg_duration,
                        'end_pos': actual_end_pos,
                    })
                    
                    # Simulate trajectory
                    num_steps = int(seg_duration / self.timestep)
                    for _ in range(num_steps):
                        current_position = current_position + velocity * self.timestep
                        trajectory_points.append(current_position.copy())
                    
                    current_position = actual_end_pos.copy()
                    
                else:
                    # Level 2: Polynomial trajectory segment
                    num_control_points = np.random.randint(3, 5)
                    
                    control_points_x = [current_position[0]]
                    control_points_y = [current_position[1]]
                    
                    # Generate intermediate control points
                    for i in range(1, num_control_points - 1):
                        t = i / (num_control_points - 1)
                        x = current_position[0] + t * (target_pos[0] - current_position[0])
                        y = current_position[1] + t * (target_pos[1] - current_position[1])
                        # Add randomness
                        x += np.random.uniform(-0.08, 0.08)
                        y += np.random.uniform(-0.08, 0.08)
                        control_points_x.append(np.clip(x, self.x_min, self.x_max))
                        control_points_y.append(np.clip(y, self.y_min, self.y_max))
                    
                    control_points_x.append(target_pos[0])
                    control_points_y.append(target_pos[1])
                    
                    # Fit polynomial
                    t_control = np.linspace(0, 1, num_control_points)
                    poly_degree = min(num_control_points - 1, 5)
                    coeffs_x = np.polyfit(t_control, control_points_x, poly_degree)
                    coeffs_y = np.polyfit(t_control, control_points_y, poly_degree)
                    
                    poly_x = np.poly1d(coeffs_x)
                    poly_y = np.poly1d(coeffs_y)
                    
                    segment_trajectories.append({
                        'type': 'polynomial',
                        'start_pos': current_position.copy(),
                        'poly_x': coeffs_x,
                        'poly_y': coeffs_y,
                        'duration': seg_duration,
                        'end_pos': target_pos,
                    })
                    
                    # Simulate trajectory
                    num_steps = int(seg_duration / self.timestep)
                    for step in range(num_steps):
                        t = step / num_steps
                        x = float(poly_x(t))
                        y = float(poly_y(t))
                        current_position = np.array([x, y, self.workspace_z])
                        trajectory_points.append(current_position.copy())
                    
                    current_position = target_pos.copy()
            
            # Check if trajectory is valid
            final_pos = trajectory_points[-1]
            distance_to_target = np.linalg.norm(final_pos[:2] - end_position[:2])
            
            # Check all points are within bounds
            all_in_bounds = all(
                self.x_min <= pt[0] <= self.x_max and 
                self.y_min <= pt[1] <= self.y_max
                for pt in trajectory_points
            )
            
            # Calculate total arc length and average velocity for the entire trajectory
            total_arc_length = 0
            for i in range(1, len(trajectory_points)):
                segment_distance = np.linalg.norm(
                    (trajectory_points[i] - trajectory_points[i-1])[:2]
                )
                total_arc_length += segment_distance
            
            # Average velocity = total arc length / total duration
            average_velocity = total_arc_length / total_duration
            
            # Check visibility
            def get_sampled_trajectory(n):
                indices = np.linspace(0, len(trajectory_points)-1, n, dtype=int)
                return [trajectory_points[i] for i in indices]
            
            if (distance_to_target < 0.1 and 
                all_in_bounds and
                average_velocity <= dynamic_coefficient and
                self._check_trajectory_visibility(
                    start_position, end_position, get_sampled_trajectory
                )):
                return start_position, segment_trajectories, True
        
        return None, None, False

