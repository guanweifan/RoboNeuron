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
    motion_intents_from_action_chunk,
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
        raw_action_dispatch_period_sec: float = 0.1,
    ) -> None:
        if raw_action_dispatch_period_sec <= 0:
            raise ValueError("raw_action_dispatch_period_sec must be positive.")
        self.resolver = resolver
        self.normalized_velocity_config = normalized_velocity_config or NormalizedCartesianVelocityConfig()
        self.raw_action_dispatch_period_sec = float(raw_action_dispatch_period_sec)
        self.scheduler = ChunkScheduler()

    def resolve_intent(
        self,
        intent: MotionIntent,
        joint_positions: dict[str, float],
    ) -> ActuationCommand:
        return self.resolver.resolve(intent, joint_positions)

    def resolve_eef_delta(self, action: list[float], joint_positions: dict[str, float]) -> ActuationCommand:
        intent = motion_intent_from_eef_delta(action)
        return self.resolve_intent(intent, joint_positions)

    def queue_intents(
        self,
        intents: list[MotionIntent] | tuple[MotionIntent, ...],
        *,
        step_duration_sec: float,
        now: float | None = None,
    ) -> None:
        queued_intents, effective_step_duration_sec = self._resample_intents_for_dispatch(
            list(intents),
            step_duration_sec=step_duration_sec,
        )
        self.scheduler.load(
            queued_intents,
            step_duration_sec=effective_step_duration_sec,
            now=now,
        )

    def queue_action_chunk(self, chunk: ActionChunk, *, now: float | None = None) -> None:
        intents = motion_intents_from_action_chunk(
            chunk,
            normalized_velocity_config=self.normalized_velocity_config,
        )
        self.queue_intents(intents, step_duration_sec=chunk.step_duration_sec, now=now)

    def dispatch_ready(
        self,
        joint_positions: dict[str, float],
        *,
        now: float | None = None,
    ) -> ActuationCommand | None:
        intent = self.dispatch_ready_intent(now=now)
        if intent is None:
            return None
        return self.resolve_intent(intent, joint_positions)

    def dispatch_ready_intent(self, *, now: float | None = None) -> MotionIntent | None:
        return self.scheduler.dispatch_ready(now=now)

    def clear_action_chunk(self) -> None:
        self.scheduler.clear()

    def _resample_intents_for_dispatch(
        self,
        intents: list[MotionIntent],
        *,
        step_duration_sec: float,
    ) -> tuple[list[MotionIntent], float]:
        if not intents:
            return [], step_duration_sec
        if step_duration_sec <= self.raw_action_dispatch_period_sec:
            return intents, step_duration_sec

        substeps = max(1, int(np.ceil(step_duration_sec / self.raw_action_dispatch_period_sec)))
        if substeps <= 1:
            return intents, step_duration_sec

        effective_step_duration_sec = step_duration_sec / substeps
        resampled: list[MotionIntent] = []
        for intent in intents:
            resampled.extend(
                self._subdivide_intent(
                    intent,
                    substeps=substeps,
                    source_step_duration_sec=step_duration_sec,
                )
            )
        return resampled, effective_step_duration_sec

    @staticmethod
    def _subdivide_intent(
        intent: MotionIntent,
        *,
        substeps: int,
        source_step_duration_sec: float,
    ) -> list[MotionIntent]:
        if substeps <= 1 or intent.mode != "cartesian_delta":
            return [intent]

        total_delta = np.asarray(intent.arm, dtype=np.float64)
        nominal_delta = total_delta / substeps
        dispatched_delta = np.zeros_like(total_delta)
        subdivided: list[MotionIntent] = []

        for index in range(substeps):
            if index == substeps - 1:
                arm_delta = total_delta - dispatched_delta
            else:
                arm_delta = nominal_delta.copy()
                dispatched_delta += arm_delta

            metadata = dict(intent.metadata)
            metadata.update(
                {
                    "edge_substep_index": index,
                    "edge_substeps": substeps,
                    "edge_source_step_duration_sec": float(source_step_duration_sec),
                }
            )
            subdivided.append(
                MotionIntent(
                    mode=intent.mode,
                    arm=arm_delta,
                    gripper_open_fraction=intent.gripper_open_fraction,
                    frame=intent.frame,
                    metadata=metadata,
                )
            )

        return subdivided


class URDFKinematicsResolver:
    """Resolve small Cartesian deltas into joint-space commands via local differential IK."""

    def __init__(
        self,
        urdf_path: str,
        *,
        gripper_open_position: float = 0.04,
        gripper_closed_position: float = 0.0,
        max_joint_delta: float = 0.2,
        numerical_jacobian_epsilon: float = 1e-6,
        damping: float = 1e-3,
    ) -> None:
        self.urdf_path = urdf_path
        self.gripper_open_position = gripper_open_position
        self.gripper_closed_position = gripper_closed_position
        self.max_joint_delta = float(max_joint_delta)
        self.numerical_jacobian_epsilon = float(numerical_jacobian_epsilon)
        self.damping = float(damping)

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
        self.active_joint_indices: list[int] = []
        self.active_joint_limits: list[tuple[float | None, float | None]] = []

        for index, link in enumerate(self.chain.links):
            if link.joint_type == "fixed" or link.name == self.base_link_name:
                self.ik_mask.append(False)
            else:
                self.ik_mask.append(True)
                self.active_joint_names.append(link.name)
                self.active_joint_indices.append(index)
                self.active_joint_limits.append(link.bounds)

    def current_end_effector_pose(self, joint_positions: dict[str, float]) -> np.ndarray:
        current_ik_q = self._build_seed(joint_positions)
        return self.chain.forward_kinematics(current_ik_q)

    def resolve(self, intent: MotionIntent, joint_positions: dict[str, float]) -> ActuationCommand:
        if intent.mode != "cartesian_delta":
            raise ValueError(f"Unsupported motion intent mode: {intent.mode}")

        current_ik_q = np.asarray(self._build_seed(joint_positions), dtype=np.float64)
        base_cartesian_delta = self._cartesian_delta_in_base(current_ik_q, intent.arm, frame=intent.frame)
        joint_delta = self._cartesian_delta_to_joint_delta(base_cartesian_delta, current_ik_q)
        active_positions = self._apply_joint_delta(current_ik_q, joint_delta)
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

    def _cartesian_delta_in_base(
        self,
        current_ik_q: np.ndarray,
        delta: np.ndarray,
        *,
        frame: str,
    ) -> np.ndarray:
        cartesian_delta = np.asarray(delta, dtype=np.float64).reshape(6)
        if frame in {"base", "world"}:
            return cartesian_delta

        current_pose = self.chain.forward_kinematics(current_ik_q)
        rotation = current_pose[:3, :3]
        linear_delta = rotation @ cartesian_delta[:3]
        angular_delta = rotation @ cartesian_delta[3:6]
        return np.concatenate([linear_delta, angular_delta])

    def _cartesian_delta_to_joint_delta(
        self,
        cartesian_delta: np.ndarray,
        current_ik_q: np.ndarray,
    ) -> np.ndarray:
        if not self.active_joint_indices:
            return np.zeros(0, dtype=np.float64)

        jacobian = self._numerical_jacobian(current_ik_q)
        damping_matrix = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
        joint_delta = jacobian.T @ np.linalg.solve(damping_matrix, cartesian_delta)
        return self._limit_joint_delta(joint_delta)

    def _numerical_jacobian(self, current_ik_q: np.ndarray) -> np.ndarray:
        base_pose = self.chain.forward_kinematics(current_ik_q)
        base_rotation = base_pose[:3, :3]
        jacobian = np.zeros((6, len(self.active_joint_indices)), dtype=np.float64)

        for column, joint_index in enumerate(self.active_joint_indices):
            perturbed = current_ik_q.copy()
            perturbed[joint_index] += self.numerical_jacobian_epsilon
            perturbed_pose = self.chain.forward_kinematics(perturbed)

            jacobian[:3, column] = (
                perturbed_pose[:3, 3] - base_pose[:3, 3]
            ) / self.numerical_jacobian_epsilon
            rotation_delta = perturbed_pose[:3, :3] @ base_rotation.T
            jacobian[3:, column] = (
                self._rotation_matrix_to_rotvec(rotation_delta) / self.numerical_jacobian_epsilon
            )

        return jacobian

    def _limit_joint_delta(self, joint_delta: np.ndarray) -> np.ndarray:
        if joint_delta.size == 0:
            return joint_delta

        limits = np.full_like(joint_delta, self.max_joint_delta, dtype=np.float64)
        max_ratio = float(np.max(np.abs(joint_delta) / limits))
        if max_ratio > 1.0:
            joint_delta = joint_delta / max_ratio
        return joint_delta

    def _apply_joint_delta(self, current_ik_q: np.ndarray, joint_delta: np.ndarray) -> list[float]:
        active_positions: list[float] = []
        for offset, joint_index in enumerate(self.active_joint_indices):
            min_limit, max_limit = self.active_joint_limits[offset]
            position = float(current_ik_q[joint_index] + joint_delta[offset])
            if min_limit is not None and max_limit is not None:
                position = float(np.clip(position, min_limit, max_limit))
            active_positions.append(position)
        return active_positions

    @staticmethod
    def _rotation_matrix_to_rotvec(rotation_matrix: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        skew_vector = 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        sin_angle = float(np.linalg.norm(skew_vector))
        cos_angle = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))

        if sin_angle < 1e-9:
            return skew_vector

        angle = float(np.arctan2(sin_angle, cos_angle))
        return skew_vector * (angle / sin_angle)


__all__ = [
    "ActionChunk",
    "ActuationCommand",
    "ChunkScheduler",
    "ControlRuntime",
    "URDFKinematicsResolver",
]
