import numpy as np
from pathlib import Path
import sapien.core as sapien
import transforms3d as t3d
from typing import Literal


def pause(task, till_close=False, show_point=False):
    if show_point:
        for point in Point.points:
            point.update()
    task.viewer.paused = True
    while task.viewer.paused:
        task.viewer.render()
    if till_close:
        while not task.viewer.closed:
            for point in Point.points:
                point.update()
            task.scene.step()
            task.scene.update_render()
            task.viewer.render()


import time
from functools import wraps


def timer(func):

    @wraps(func)
    def decorated(*args, **kwargs):
        name = func.__name__
        start_time = time.perf_counter()
        ret = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"Timer '{name}': {elapsed:.4f} seconds")
        with open("timer.log", "a", encoding="utf-8") as f:
            f.write(f"Timer '{name}': {elapsed:.4f} seconds\n")
        return ret

    return decorated


timer_dict = {}


def local_timer(name: str):
    if name in timer_dict:
        elapsed = time.perf_counter() - timer_dict[name]
        print(f"Local Timer '{name}': {elapsed:.4f} seconds")
        with open("timer.log", "a", encoding="utf-8") as f:
            f.write(f"Local Timer '{name}': {elapsed:.4f} seconds\n")
        del timer_dict[name]
    else:
        timer_dict[name] = time.perf_counter()


class Point:
    points: list["Point"] = []
    """Point under a specific base coordinate system"""

    def __init__(
        self,
        scene: sapien.Scene,
        base: sapien.Entity,
        base_scale: float,
        init_mat: np.ndarray,
        base_pose_mat: np.ndarray = None,
        scaled: bool = True,
        follow: sapien.Entity = None,
        name: str = "point",
        size: float = 0.05,
        eular_round_to: int = 0.01,
    ):
        self.name = name
        self.scene = scene
        self.base = base
        if base_pose_mat is not None:
            self.base_pose_mat = np.array(base_pose_mat)
        else:
            self.base_pose_mat = base.get_pose().to_transformation_matrix()
        self.follow = follow
        self.base_scale = base_scale
        self.eular_round_to = eular_round_to

        self.mat = np.array(init_mat)
        if not scaled:
            self.mat[:3, 3] *= self.base_scale

        self.pose = self.trans_base(
            self.base.get_pose().to_transformation_matrix(),
            self.base_pose_mat,
            self.mat,
        )
        self.mat = self.word2base(self.pose.to_transformation_matrix()).to_transformation_matrix()
        self.base_pose_mat = self.base.get_pose().to_transformation_matrix()

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(filename="./assets/objects/cube/textured.obj", scale=[size, size, size])
        self.point = builder.build(name=name)
        self.point.set_pose(self.pose)
        Point.points.append(self)

    def __del__(self):
        Point.points.remove(self)

    def get_pose(self) -> sapien.Pose:
        return self.pose

    @staticmethod
    def pose2list(pose: sapien.Pose) -> list:
        return pose.p.tolist() + pose.q.tolist()

    @staticmethod
    def round_eular(eular, round_to: int = 1) -> np.ndarray:
        unit = round_to / 180 * np.pi
        return np.round(np.array(eular) / unit) * unit

    @staticmethod
    def trans_mat(to_mat: np.ndarray, from_mat: np.ndarray, scale: float = 1.0):
        to_rot = to_mat[:3, :3]
        from_rot = from_mat[:3, :3]
        rot_mat = to_rot @ from_rot.T

        trans_mat = (to_mat[:3, 3] - from_mat[:3, 3]) / scale

        result = np.eye(4)
        result[:3, :3] = rot_mat
        result[:3, 3] = trans_mat
        result = np.where(np.abs(result) < 1e-5, 0, result)
        return result

    @staticmethod
    def trans_pose(to_pose: sapien.Pose, from_pose: sapien.Pose, scale: float = 1.0):
        return Point.trans_mat(
            to_pose.to_transformation_matrix(),
            from_pose.to_transformation_matrix(),
            scale,
        )

    @staticmethod
    def trans_base(
        now_base_mat: np.ndarray,
        init_base_mat: np.ndarray,
        init_pose_mat: np.ndarray,
        scale: float = 1.0,
    ):
        now_base_mat = np.array(now_base_mat)
        init_base_mat = np.array(init_base_mat)
        init_pose_mat = np.array(init_pose_mat)
        init_pose_mat[:3, 3] *= scale

        now_pose_mat = np.eye(4)
        base_trans_mat = Point.trans_mat(now_base_mat, init_base_mat)
        now_pose_mat[:3, :3] = (base_trans_mat[:3, :3] @ init_pose_mat[:3, :3] @ base_trans_mat[:3, :3].T)
        now_pose_mat[:3, 3] = base_trans_mat[:3, :3] @ init_pose_mat[:3, 3]

        # Convert to world coordinates
        p = now_pose_mat[:3, 3] + now_base_mat[:3, 3]
        q_mat = now_pose_mat[:3, :3] @ now_base_mat[:3, :3]
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))

    def get_output_mat(self):
        opt_mat = self.mat.copy()
        opt_mat[:3, 3] /= self.base_scale
        return opt_mat

    def base2world(self, entity_mat, scale=1.0) -> sapien.Pose:
        """Convert matrix from base coordinate system to world coordinate system"""
        entity_mat = np.array(entity_mat)
        base_mat = self.base.get_pose().to_transformation_matrix()
        p = entity_mat[:3, 3] * scale + base_mat[:3, 3]
        q_mat = entity_mat[:3, :3] @ base_mat[:3, :3]
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))

    def word2base(self, entity_mat, scale=1.0) -> sapien.Pose:
        """Convert matrix from world coordinate system to base coordinate system"""
        entity_mat = np.array(entity_mat)
        base_mat = self.base.get_pose().to_transformation_matrix()
        p = entity_mat[:3, 3] - base_mat[:3, 3]
        q_mat = entity_mat[:3, :3] @ base_mat[:3, :3].T
        return sapien.Pose(p, t3d.quaternions.mat2quat(q_mat))

    def set_pose(self, new_pose: sapien.Pose):
        """Update point position"""
        self.pose = new_pose
        self.point.set_pose(self.pose)
        self.mat = self.word2base(new_pose.to_transformation_matrix()).to_transformation_matrix()

    def update(self, force_output: bool = False, flexible: bool = False):
        new_mat = np.eye(4)
        if self.follow is not None:
            new_mat = self.trans_mat(
                self.follow.get_pose().to_transformation_matrix(),
                self.base.get_pose().to_transformation_matrix(),
            )
        elif flexible:
            new_mat = self.trans_mat(
                self.point.get_pose().to_transformation_matrix(),
                self.base.get_pose().to_transformation_matrix(),
            )
        else:
            new_mat = self.word2base(
                self.trans_base(
                    self.base.get_pose().to_transformation_matrix(),
                    self.base_pose_mat,
                    self.mat,
                ).to_transformation_matrix()).to_transformation_matrix()

        new_mat[:3, :3] = t3d.euler.euler2mat(
            *self.round_eular(t3d.euler.mat2euler(new_mat[:3, :3]), self.eular_round_to))
        self.pose = self.base2world(new_mat)
        self.point.set_pose(self.pose)

        if not np.allclose(new_mat, self.mat, atol=1e-3) or force_output:
            self.mat = new_mat
            self.base_pose_mat = self.base.get_pose().to_transformation_matrix()


def rotate_cone(new_pt: np.ndarray, origin: np.ndarray, z_dir: np.ndarray = [0, 0, 1]):
    x = origin - new_pt
    x = x / np.linalg.norm(x)
    bx_ = np.array(z_dir).reshape(3)
    z = bx_ - np.dot(x, bx_) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _tolist(pose: sapien.Pose | list | np.ndarray) -> list:
    if isinstance(pose, list):
        return pose
    elif isinstance(pose, sapien.Pose):
        return pose.p.tolist() + pose.q.tolist()
    else:
        return pose.tolist()


def _toPose(pose: sapien.Pose | list | np.ndarray) -> sapien.Pose:
    if isinstance(pose, list):
        assert len(pose) == 7 or len(pose) == 3
        if len(pose) == 3:
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    elif isinstance(pose, np.ndarray):
        assert pose.shape == (7, ) or pose.shape == (3, )
        if pose.shape == (3, ):
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    else:
        return pose


def rotate_along_axis(
    target_pose,
    center_pose,
    axis,
    theta: float = np.pi / 2,
    axis_type: Literal["center", "target", "world"] = "center",
    towards=None,
    camera_face=None,
) -> list:
    """
    Rotate around a specified axis by a given angle with center as origin.
    Can specify rotation direction via 'towards' (direction where dot product of center->target vector and towards vector is positive).

    target_pose: target point (e.g., pre-grasp point directly above object)
    center_pose: center point (e.g., object position)
    axis: rotation axis
    theta: rotation angle (radians)
    axis_type: type of rotation axis ('center': relative to center_pose, 'target': relative to target_pose, 'world': world coordinate system), default 'center'
    towards: rotation direction (optional), determines rotation direction if specified
    camera_face: camera facing direction (optional), restricts dot product of camera vector and this vector to be positive; if None, rotates without considering camera facing
    Returns: list, first 3 elements are coordinates, last 4 are quaternions
    """
    target_pose, center_pose = _toPose(target_pose), _toPose(center_pose)
    if theta == 0:
        return target_pose.p.tolist() + target_pose.q.tolist()
    rotate_mat = t3d.axangles.axangle2mat(axis, theta)

    target_mat = target_pose.to_transformation_matrix()
    center_mat = center_pose.to_transformation_matrix()
    if axis_type == "center":
        world_axis = (center_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    elif axis_type == "target":
        world_axis = (target_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    else:
        world_axis = np.array(axis).reshape(3)

    rotate_mat = t3d.axangles.axangle2mat(world_axis, theta)
    p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p
    if towards is not None:
        towards = np.dot(p - center_pose.p, np.array(towards).reshape(3))
        if towards < 0:
            rotate_mat = t3d.axangles.axangle2mat(world_axis, -theta)
            p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p

    if camera_face is None:
        q = t3d.quaternions.mat2quat(rotate_mat @ target_mat[:3, :3])
    else:
        q = t3d.quaternions.mat2quat(rotate_cone(p, center_pose.p, camera_face))
    return p.tolist() + q.tolist()


def rotate2rob(target_pose, rob_pose, box_pose, theta: float = 0.5) -> list:
    """
    Offset towards the specified rob_pose
    """
    target_pose, rob_pose, box_pose = (
        _toPose(target_pose),
        _toPose(rob_pose),
        _toPose(box_pose),
    )

    target_mat = target_pose.to_transformation_matrix()
    v1 = (target_mat[:3, :3] @ np.array([[1, 0, 0]]).T).reshape(3)
    v2 = box_pose.p - rob_pose.p
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    return rotate_along_axis(
        target_pose=target_pose,
        center_pose=box_pose,
        axis=axis,
        theta=angle * theta,
        axis_type="world",
        towards=-v2,
    )


def choose_dirct(block_mat, base_pose: sapien.Pose):
    pts = block_mat[:3, :3] @ np.array([[1, -1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]])
    dirts = np.sum(np.power(pts - base_pose.p.reshape(3, 1), 2), axis=0)
    return pts[:, np.argmin(dirts)] + block_mat[:3, 3]


def add_robot_visual_box(task, pose: sapien.Pose | list, name: str = "box"):
    box_path = Path("./assets/objects/cube/textured.obj")
    if not box_path.exists():
        print("[WARNNING] cube not exists!")
        return

    pose = _toPose(pose)
    scene: sapien.Scene = task.scene
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_visual_from_file(
        filename=str(box_path),
        scale=[
            0.04,
        ] * 3,
    )
    builder.set_name(name)
    builder.set_initial_pose(pose)
    return builder.build()


def cal_quat_dis(quat1, quat2):
    qmult = t3d.quaternions.qmult
    qinv = t3d.quaternions.qinverse
    qnorm = t3d.quaternions.qnorm
    delta_quat = qmult(qinv(quat1), quat2)
    return 2 * np.arccos(np.fabs((delta_quat / qnorm(delta_quat))[0])) / np.pi


def get_align_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from v1 to v2
    """
    v1 = np.array(v1).reshape(3)
    v2 = np.array(v2).reshape(3)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    else:
        return t3d.axangles.axangle2mat(axis, angle)


def generate_rotate_vectors(
    axis: Literal["x", "y", "z"] | np.ndarray | list,
    angle: np.ndarray | list | float,
    base: np.ndarray | sapien.Pose | list = None,
    vector: np.ndarray | list = [1, 0, 0],
) -> np.ndarray:
    """
    Get the rotation matrix from base to axis
    """
    if base is None:
        base = np.eye(4)
    else:
        base = _toPose(base).to_transformation_matrix()

    if isinstance(axis, str):
        if axis == "x":
            axis = np.array([1, 0, 0])
        elif axis == "y":
            axis = np.array([0, 1, 0])
        elif axis == "z":
            axis = np.array([0, 0, 1])
        else:
            raise ValueError("axis must be x, y or z")
    else:
        axis = np.array(axis).reshape(3)

    axis = (base[:3, :3] @ axis.reshape(3, 1)).reshape(3)
    vector = (base[:3, :3] @ np.array(vector).reshape(3, 1)).reshape(3)

    vector = np.array(vector).reshape((3, 1))
    angle = np.array(angle).flatten()
    rotate_mat = np.zeros((3, angle.shape[0]))
    for idx, a in enumerate(angle):
        rotate_mat[:, idx] = (t3d.axangles.axangle2mat(axis, a) @ vector).reshape(3)
    return rotate_mat


def get_product_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get the projection vector of v2 on v1
    """
    v1 = np.array(v1).reshape(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array(v2).reshape(3)
    return np.dot(v1, v2) * v1


def get_place_pose(
    actor_pose: np.ndarray | sapien.Pose | list,
    target_pose: np.ndarray | sapien.Pose | list,
    constrain: Literal["free", "align"] = "free",
    align_axis: list[np.ndarray] | np.ndarray | list = None,
    actor_axis: np.ndarray | list = [1, 0, 0],
    actor_axis_type: Literal["actor", "world"] = "actor",
    z_transform: bool = True,
) -> list:
    """
    Get the pose where the object should be placed
    Considerations:
        1. 3D coordinates match the given coordinates
        2. Object orientation is reasonable
            - Object z-axis matches the given coordinate z-axis
            - Satisfies certain constraints on the xy plane
                - No constraint (directly use the projection of object's current x,y on xOy plane)
                - Object's x-axis aligns with the given x-axis
                - Select the direction where the dot product of object's x-axis and the given world axis unit vector set is minimized

    actor_pose: current pose of the object
    target_pose: pose where the object should be placed
    constrain: constraint type for the object
        - free: no constraint
        - align: direction where dot product of object's x-axis and given world axis vector set is minimized
    align_axis: given world axis vector set, if None, defaults to x-axis of target_pose
    actor_axis: actor axis for dot product calculation, defaults to x-axis
    actor_axis_type: type of actor_axis, defaults to local coordinate system
        - actor: local coordinate system of actor_pose
        - world: world coordinate system
    """
    actor_pose_mat = _toPose(actor_pose).to_transformation_matrix()
    target_pose_mat = _toPose(target_pose).to_transformation_matrix()

    # Align the 3D coordinates of the object with the given coordinates
    actor_pose_mat[:3, 3] = target_pose_mat[:3, 3]

    target_x = target_pose_mat[:3, 0]
    target_y = target_pose_mat[:3, 1]
    target_z = target_pose_mat[:3, 2]

    # Align the z-axis of the object with the z-axis of the given coordinates
    actor2world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
    if z_transform:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, :3] @ actor2world[:3, 2], target_z)
    else:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, 2], target_z)
    actor_pose_mat[:3, :3] = z_align_matrix @ actor_pose_mat[:3, :3]

    if constrain == "align":
        if align_axis is None:
            align_axis = np.array(target_pose_mat[:3, :3] @ np.array([[1, 0, 0]]).T)
        elif isinstance(align_axis, list):
            align_axis = np.array(align_axis).reshape((-1, 3)).T
        else:
            align_axis = np.array(align_axis).reshape((3, -1))
        align_axis = align_axis / np.linalg.norm(align_axis, axis=0)

        if actor_axis_type == "actor":
            actor_axis = actor_pose_mat[:3, :3] @ np.array(actor_axis).reshape(3, 1)
        elif actor_axis_type == "world":
            actor_axis = np.array(actor_axis)
        closest_axis_id = np.argmax(actor_axis.reshape(3) @ align_axis)
        align_axis = align_axis[:, closest_axis_id]

        actor_axis_xOy = get_product_vector(target_x, actor_axis) + get_product_vector(target_y, actor_axis)
        align_axis_xOy = get_product_vector(target_x, align_axis) + get_product_vector(target_y, align_axis)
        align_mat_xOy = get_align_matrix(actor_axis_xOy, align_axis_xOy)
        actor_pose_mat[:3, :3] = align_mat_xOy @ actor_pose_mat[:3, :3]

    return (actor_pose_mat[:3, 3].tolist() + t3d.quaternions.mat2quat(actor_pose_mat[:3, :3]).tolist())


def get_face_prod(q, local_axis, target_axis):
    """
    get product of local_axis (under q world) and target_axis
    """
    q_mat = t3d.quaternions.quat2mat(q)
    face = q_mat @ np.array(local_axis).reshape(3, 1)
    face_prod = np.dot(face.reshape(3), np.array(target_axis))
    return face_prod
