import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Import base class
from .base import AdapterWrapper

logger = logging.getLogger("wrappers.calvin")

CALVIN_ROOT = os.environ.get("CALVIN_ROOT", "/home/xihuasen/calvin")
CALVIN_MODELS_PATH = str(Path(CALVIN_ROOT) / "calvin_models")

if CALVIN_MODELS_PATH not in sys.path:
    sys.path.insert(0, CALVIN_MODELS_PATH)

try:
    # Rename to avoid name collisions
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_env.envs.play_table_env import get_env as get_calvin_env_instance
    _CALVIN_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on optional runtime
    get_sequences = None  # type: ignore[assignment]
    get_calvin_env_instance = None  # type: ignore[assignment]
    _CALVIN_IMPORT_ERROR = exc


class CalvinAdapterWrapper(AdapterWrapper):
    """
    Adapter wrapper for the CALVIN benchmark.
    """
    def __init__(self,
                 dataset_path: str | Path,
                 task_id: int = 0,
                 show_gui: bool = False,
                 step_substeps: int = 1,
                 visual_resolution: int = 200, 
                 auto_init_env: bool = True,
                 load_sequences: bool = False,
                 **kwargs):

        super().__init__(**kwargs)
        self.dataset_path = str(dataset_path)
        self.task_id = task_id
        self.show_gui = show_gui
        self.step_substeps = step_substeps
        self.visual_resolution = visual_resolution
        self.auto_init_env = auto_init_env
        self.load_sequences = load_sequences

        self.env = None
        self.task_sequence = None
        self.current_task_info = {}
        self.raw_obs = None
        self.current_step_idx = 0

        if hasattr(self, "_initialized") and self._initialized:
            logger.info("CALVIN adapter already initialized, skipping duplicate initialization.")
            return

        if self.auto_init_env:
            self.create_simulation_environment()
            logger.info("CALVIN adapter initialized with real environment.")
        else:
            logger.info("CALVIN adapter init in deferred mode (auto_init_env=False).")

    
    def create_simulation_environment(self):
        """
        Creates the CALVIN environment and loads the task sequence.
        """
        if _CALVIN_IMPORT_ERROR is not None:
            raise RuntimeError(
                "CALVIN dependencies are missing. Install CALVIN before using CalvinAdapterWrapper."
            ) from _CALVIN_IMPORT_ERROR

        self.task_sequence = None
        self.current_task_info = {"language": ""}

        if self.load_sequences:
            sequences = self._get_sequences()
            if self.task_id >= len(sequences):
                raise ValueError(f"Task ID {self.task_id} out of range (max {len(sequences)-1})")
            self.task_sequence = sequences[self.task_id]


        obs_space = {
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": [],
            "state_obs": ["robot_obs"],
            "actions": ["rel_actions"],
            "language": ["language"],  
        }
        try:
            self.env = get_calvin_env_instance(self.dataset_path, show_gui=self.show_gui, obs_space=obs_space)
        except TypeError:
            self.env = get_calvin_env_instance(self.dataset_path, show_gui=self.show_gui)
        


        self.reset()
        self._initialized = True  
        seq_len = len(self.task_sequence) if self.task_sequence is not None else 0
        logger.info(f"CALVIN initialized. Sequence length: {seq_len} (load_sequences={self.load_sequences})")


    def reset(self) -> dict[str, Any]:
        """Standard reset method."""

        if self.env is None:
            self.create_simulation_environment()

        ret = self.env.reset()
        self.raw_obs = ret[0] if isinstance(ret, tuple) else ret
        self.current_step_idx = 0

   
        self._update_current_instruction()

        return self.obtain_observation()

    def _get_sequences(self):
        """Load task sequences with compatibility across calvin-agent versions."""
        if get_sequences is None:
            raise RuntimeError("CALVIN sequence loader is unavailable.")
        try:
            return get_sequences(n_tasks=1, n_sequences_per_task=1, path=self.dataset_path)
        except TypeError:
            pass

        try:
            sig = inspect.signature(get_sequences)
            kwargs = {}
            params = sig.parameters
            if "n_tasks" in params:
                kwargs["n_tasks"] = 1
            elif "num_tasks" in params:
                kwargs["num_tasks"] = 1
            if "n_sequences_per_task" in params:
                kwargs["n_sequences_per_task"] = 1
            elif "n_seq_per_task" in params:
                kwargs["n_seq_per_task"] = 1
            elif "num_sequences" in params:
                kwargs["num_sequences"] = 1
            if "path" in params:
                kwargs["path"] = self.dataset_path
            elif "dataset_path" in params:
                kwargs["dataset_path"] = self.dataset_path
            elif "data_path" in params:
                kwargs["data_path"] = self.dataset_path
            if kwargs:
                return get_sequences(**kwargs)
        except Exception:
            pass

        try:
            return get_sequences(1, 1, self.dataset_path)
        except TypeError:
            try:
                return get_sequences(self.dataset_path, 1, 1)
            except TypeError:
                return get_sequences(self.dataset_path)

    def _update_current_instruction(self):
        """Helper to get language instruction from sequence."""
        self.current_task_info = {"language": ""}
        if not self.task_sequence:
            return

        seq = self.task_sequence
        idx = self.current_step_idx

        # Common case: list of dicts or strings
        if isinstance(seq, (list, tuple)):
            if len(seq) == 2 and isinstance(seq[1], (list, tuple)):
                # Some CALVIN versions return (state_dict, task_list)
                tasks = seq[1]
                if tasks:
                    self.current_task_info = {
                        "language": tasks[min(idx, len(tasks) - 1)]
                    }
                return

            if len(seq) > 0:
                element = seq[min(idx, len(seq) - 1)]
                if isinstance(element, dict):
                    self.current_task_info = {
                        "language": element.get("language", "unknown task")
                    }
                elif isinstance(element, str):
                    self.current_task_info = {"language": element}
                elif isinstance(element, (list, tuple)):
                    for item in element:
                        if isinstance(item, str):
                            self.current_task_info = {"language": item}
                            break
                return

        if isinstance(seq, dict):
            self.current_task_info = {"language": seq.get("language", "unknown task")}

    def obtain_observation(self) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("env is None. Did you forget to create_simulation_environment() or set env?")
        if self.raw_obs is None:
            raise RuntimeError("raw_obs is None ...")

        obs = self.raw_obs

        # ---- rgb extraction: support nested or flat ----
        rgb_static = obs["rgb_obs"]["rgb_static"]
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]


        visual_observation: dict[str, np.ndarray] = {}

        if rgb_static is not None:
            img = np.asarray(rgb_static)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            visual_observation["agentview_image"] = img

        if rgb_gripper is not None:
            img = np.asarray(rgb_gripper)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            visual_observation["eye_in_hand_image"] = img

        # ---- robot_obs extraction: nested or flat ----
        robot_obs = obs.get("robot_obs")
        if robot_obs is None:
            robot_obs = np.zeros(15, dtype=np.float32)
        robot_obs = np.asarray(robot_obs).reshape(-1)

        # Your previous slicing assumption
        joint_pos = robot_obs[7:14] if robot_obs.shape[0] >= 14 else np.zeros(7, dtype=np.float32)
        gripper_width = robot_obs[6] if robot_obs.shape[0] > 6 else 0.0

        robot_state = np.zeros(8, dtype=np.float32)
        robot_state[:7] = joint_pos
        robot_state[7] = gripper_width

        instruction = self.current_task_info.get("language", "")
        return {
            "visual_observation": visual_observation,
            "robot_state": robot_state,
            "instruction": instruction
        }

    def step(self, action: np.ndarray | list, **kwargs) -> dict[str, Any]:

        action_np = np.asarray(action, dtype=np.float32).reshape(-1)

        if action_np.shape != (7,):
            raise ValueError(
                "CALVIN expects 7D relative EEF actions (dx, dy, dz, droll, dpitch, dyaw, gripper). "
                f"Got shape {action_np.shape}. If you have 8D joint actions, an IK mapping is required."
            )

        gripper_action = float(action_np[-1])
        if 0.0 <= gripper_action <= 1.0:
            # Treat [0,1] as open/close probability
            gripper_action = -1.0 if gripper_action < 0.5 else 1.0
        elif -1.0 <= gripper_action <= 1.0:
            # Collapse continuous [-1,1] to discrete sign
            gripper_action = -1.0 if gripper_action <= 0.0 else 1.0
        else:
            raise ValueError(
                f"Gripper action must be in [0,1] or [-1,1], got {gripper_action}."
            )

        action_to_env = action_np.copy()
        action_to_env[-1] = gripper_action

        # Execution
        total_reward = 0
        done = False
        info = {}

        for _ in range(self.step_substeps):
            ret = self.env.step(action_to_env)
            if isinstance(ret, tuple) and len(ret) == 4:
                self.raw_obs, r, d, i = ret
            elif isinstance(ret, tuple) and len(ret) == 5:
                self.raw_obs, r, terminated, truncated, i = ret
                d = bool(terminated or truncated)
            else:
                raise RuntimeError(f"Unexpected env.step return: {type(ret)} len={len(ret) if isinstance(ret, tuple) else 'NA'}")
            
            total_reward += r
            done = done or d
            info.update(i)

        self.current_step_idx += 1
        self._update_current_instruction()

        return {
            "observation": self.obtain_observation(),
            "reward": total_reward,
            "done": done,
            "task_info": info
        }
        
        
    def update_environment(self, action):
        """
        Backward-compatible alias for older simulation harnesses.
        Libero-style callers use update_environment(action).
        """
        return self.step(action)
