"""Edge-side control runtime helpers for RoboNeuron execution."""

from __future__ import annotations

import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Protocol

import numpy as np
from ikpy.chain import Chain

from roboneuron_core.kernel import (
    ActionChunk,
    ActuationCommand,
    MotionIntent,
    NormalizedCartesianVelocityConfig,
    motion_intent_from_eef_delta,
    motion_intent_from_raw_step,
)


class MotionResolver(Protocol):
    def resolve(self, intent: MotionIntent, joint_positions: dict[str, float]) -> ActuationCommand:
        """Resolve a canonical motion intent into an actuation command."""


class ChunkScheduler:
    """Simple step scheduler for chunked VLA outputs."""

    def __init__(self) -> None:
        self._pending: deque[MotionIntent] = deque()
        self._step_duration_sec = 0.1
        self._next_dispatch_at = 0.0

    def load(self, intents: list[MotionIntent], *, step_duration_sec: float, now: float | None = None) -> None:
        if step_duration_sec <= 0:
            raise ValueError("Chunk step duration must be positive.")
        current_time = now if now is not None else time.monotonic()
        preserve_next_dispatch_at = bool(self._pending) and self._next_dispatch_at > current_time
        self._pending = deque(intents)
        self._step_duration_sec = step_duration_sec
        self._next_dispatch_at = self._next_dispatch_at if preserve_next_dispatch_at else current_time

    def dispatch_ready(self, *, now: float | None = None) -> MotionIntent | None:
        if not self._pending:
            return None

        current_time = now if now is not None else time.monotonic()
        if current_time < self._next_dispatch_at:
            return None

        intent = self._pending.popleft()
        self._next_dispatch_at = current_time + self._step_duration_sec
        return intent

    def clear(self) -> None:
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def step_duration_sec(self) -> float:
        return self._step_duration_sec


class ControlRuntime:
    """Bridge raw control inputs to a specific motion resolver."""

    def __init__(
        self,
        resolver: MotionResolver,
        *,
        normalized_velocity_config: NormalizedCartesianVelocityConfig | None = None,
    ) -> None:
        self.resolver = resolver
        self.normalized_velocity_config = normalized_velocity_config or NormalizedCartesianVelocityConfig()
        self.scheduler = ChunkScheduler()

    def resolve_eef_delta(self, action: list[float], joint_positions: dict[str, float]) -> ActuationCommand:
        intent = motion_intent_from_eef_delta(action)
        return self.resolver.resolve(intent, joint_positions)

    def queue_action_chunk(self, chunk: ActionChunk, *, now: float | None = None) -> None:
        intents = [
            motion_intent_from_raw_step(step, normalized_velocity_config=self.normalized_velocity_config)
            for step in chunk.steps
        ]
        self.scheduler.load(intents, step_duration_sec=chunk.step_duration_sec, now=now)

    def dispatch_ready(
        self,
        joint_positions: dict[str, float],
        *,
        now: float | None = None,
    ) -> ActuationCommand | None:
        intent = self.scheduler.dispatch_ready(now=now)
        if intent is None:
            return None
        return self.resolver.resolve(intent, joint_positions)

    def clear_action_chunk(self) -> None:
        self.scheduler.clear()


class URDFKinematicsResolver:
    """Resolve canonical Cartesian deltas into joint-space commands."""

    def __init__(
        self,
        urdf_path: str,
        *,
        gripper_open_position: float = 0.04,
        gripper_closed_position: float = 0.0,
    ) -> None:
        self.urdf_path = urdf_path
        self.gripper_open_position = gripper_open_position
        self.gripper_closed_position = gripper_closed_position

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        link_parent_map: dict[str, str] = {}
        link_names: set[str] = set()
        detected_gripper_joints: list[str] = []

        for joint in root.findall("joint"):
            name = joint.get("name") or ""
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            link_names.add(parent)
            link_names.add(child)
            link_parent_map[child] = parent
            if joint.get("type") == "prismatic" or "finger" in name.lower():
                detected_gripper_joints.append(name)

        self.base_link_name = list(link_names - set(link_parent_map.keys()))[0]

        with open(urdf_path, encoding="utf-8") as handle:
            xml_str = re.sub(r"<visual>.*?</visual>", "", handle.read(), flags=re.DOTALL)
            xml_str = re.sub(r"<collision>.*?</collision>", "", xml_str, flags=re.DOTALL)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False, encoding="utf-8") as tmp:
            tmp.write(xml_str)
            clean_urdf_path = tmp.name

        self.chain = Chain.from_urdf_file(clean_urdf_path, base_elements=[self.base_link_name])
        self.gripper_joints = detected_gripper_joints
        self.ik_mask: list[bool] = []
        self.active_joint_names: list[str] = []

        for link in self.chain.links:
            if link.joint_type == "fixed" or link.name == self.base_link_name:
                self.ik_mask.append(False)
            else:
                self.ik_mask.append(True)
                self.active_joint_names.append(link.name)

    def current_end_effector_pose(self, joint_positions: dict[str, float]) -> np.ndarray:
        current_ik_q = self._build_seed(joint_positions)
        return self.chain.forward_kinematics(current_ik_q)

    def resolve(self, intent: MotionIntent, joint_positions: dict[str, float]) -> ActuationCommand:
        if intent.mode != "cartesian_delta":
            raise ValueError(f"Unsupported motion intent mode: {intent.mode}")

        current_ik_q = self._build_seed(joint_positions)
        current_pose = self.current_end_effector_pose(joint_positions)
        target_pose = self._apply_cartesian_delta(current_pose, intent.arm, frame=intent.frame)

        target_ik_q = self.chain.inverse_kinematics_frame(
            target_pose,
            initial_position=current_ik_q,
            orientation_mode="all",
        )

        active_positions = [
            float(target_ik_q[index])
            for index in range(len(self.chain.links))
            if index < len(self.ik_mask) and self.ik_mask[index]
        ]
        gripper_positions = self._resolve_gripper_positions(intent.gripper_open_fraction, joint_positions)

        return ActuationCommand(
            joint_names=self.active_joint_names + self.gripper_joints,
            positions=active_positions + gripper_positions,
            gripper_open_fraction=intent.gripper_open_fraction,
        )

    def _build_seed(self, joint_positions: dict[str, float]) -> list[float]:
        seed = [0.0] * len(self.chain.links)
        for index, link in enumerate(self.chain.links):
            min_limit, max_limit = link.bounds
            position = joint_positions.get(link.name, 0.0)
            if min_limit is not None and max_limit is not None:
                position = float(np.clip(position, min_limit, max_limit))
            seed[index] = float(position)
        return seed

    def _resolve_gripper_positions(
        self,
        open_fraction: float | None,
        joint_positions: dict[str, float],
    ) -> list[float]:
        if not self.gripper_joints:
            return []

        if open_fraction is None:
            return [float(joint_positions.get(name, self.gripper_closed_position)) for name in self.gripper_joints]

        open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
        target_position = self.gripper_closed_position + (
            self.gripper_open_position - self.gripper_closed_position
        ) * open_fraction
        return [target_position] * len(self.gripper_joints)

    @staticmethod
    def _apply_cartesian_delta(current_pose: np.ndarray, delta: np.ndarray, *, frame: str) -> np.ndarray:
        dx, dy, dz, droll, dpitch, dyaw = np.asarray(delta, dtype=np.float64).reshape(6)

        cx, sx = np.cos(droll), np.sin(droll)
        cy, sy = np.cos(dpitch), np.sin(dpitch)
        cz, sz = np.cos(dyaw), np.sin(dyaw)
        rotation_matrix = (
            np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        )

        delta_pose = np.eye(4)
        delta_pose[:3, 3] = [dx, dy, dz]
        delta_pose[:3, :3] = rotation_matrix

        if frame in {"base", "world"}:
            return delta_pose @ current_pose
        return current_pose @ delta_pose


__all__ = [
    "ActionChunk",
    "ActuationCommand",
    "ChunkScheduler",
    "ControlRuntime",
    "URDFKinematicsResolver",
]
