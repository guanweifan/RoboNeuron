# wrappers/libero.py
import os
import logging
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import numpy as np

# Import base class
from .base import AdapterWrapper

# Import Libero library components
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    # Retain the informative error for setup issues
    raise ImportError("Libero library not installed. Please refer to its documentation for installation.")


logger = logging.getLogger("wrappers.libero")


def get_libero_env(task, resolution: int = 256):
    """Initializes and returns the LIBERO environment and task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # Seed is crucial for reproducible object placement
    return env, task_description


class LiberoAdapterWrapper(AdapterWrapper):
    """
    Adapter wrapper for the LIBERO benchmark environments.

    Manages environment creation, state retrieval, and action execution.
    """
    def __init__(self, 
                 libero_suite_name: str = 'libero_spatial', 
                 task_id: int = 0,
                 visual_resolution: int = 768,
                 step_substeps: int = 10,
                 **kwargs):
        """
        Args:
            libero_suite_name (str): The name of the LIBERO benchmark suite.
            task_id (int): Index of the task within the suite.
            visual_resolution (int): Resolution (H/W) for camera images.
            step_substeps (int): Number of environment steps per policy action.
            **kwargs: Base adapter configurations.
        """
        super().__init__(**kwargs)
        self.libero_suite_name = libero_suite_name
        self.task_id = task_id
        self.visual_resolution = visual_resolution
        self.step_substeps = step_substeps
        
        # Internal state variables
        self.env: Optional[OffScreenRenderEnv] = None
        self.task_description: Optional[str] = None
        self.observation: Optional[Dict[str, Any]] = None
        self.agentview_images: List[np.ndarray] = []

    def create_simulation_environment(self):
        """
        Creates the Libero simulation environment instance and performs initial reset.
        """
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.libero_suite_name]()
        
        task = task_suite.get_task(self.task_id)
        # self.initial_states = task_suite.get_task_init_states(self.task_id)
        
        self.env, self.task_description = get_libero_env(task, resolution=self.visual_resolution)
        self.env.reset()
        self.observation, self.reward, self.task_status, self.task_info = None, None, None, None
        self.agentview_images = []

        # Initialize to the first initial state by stepping (10 times for stability)
        initial_action = [0, 0, 0, 0, 0, 0, -1] # Example action to ensure environment is active
        for _ in range(self.step_substeps):
             self.observation, self.reward, self.task_status, self.task_info = self.env.step(initial_action)

    def obtain_observation(self) -> Dict[str, Any]:
        """
        Gathers the current state and formats it into a standardized dictionary.
        
        Returns:
            Dict[str, Any]: Standardized observation containing:
                            - 'visual_observation': Dict of processed images.
                            - 'robot_state': np.ndarray of joint position and gripper position (8D).
                            - 'instruction': str of the task description.
        """
        if self.observation is None:
            raise RuntimeError("Environment not initialized or observation not read.")
            
        # 1. Visual observation processing
        visual_observation: Dict[str, np.ndarray] = {}
        
        # Libero observations often require rotation/flipping
        agentview_image = self.observation["agentview_image"][::-1, ::-1]
        eye_in_hand_image = self.observation["robot0_eye_in_hand_image"][::-1, ::-1]
        
        visual_observation["agentview_image"] = agentview_image
        visual_observation["eye_in_hand_image"] = eye_in_hand_image
        self.agentview_images.append(agentview_image)
        
        # 2. Robot state (Proprioception)
        # 7 joint positions + 1 gripper position (8D)
        robot_state = np.zeros(8, dtype=np.float32)
        robot_state[:7] = self.observation["robot0_joint_pos"]
        robot_state[7] = self.observation["robot0_gripper_qpos"][0]
        
        # 3. Text instruction
        instruction = self.task_description
        
        return {
            "visual_observation": visual_observation,
            "robot_state": robot_state,
            "instruction": instruction
        }

    def step(self, 
             action: Union[np.ndarray, list], 
             use_eepose_delta: bool = False) -> Dict[str, Any]:
        """
        Applies a control command to the environment.

        Args:
            action (Union[np.ndarray, list]): The policy action. Assumed to be 7D 
                                              (6 joint + 1 gripper state) delta in 
                                              joint space by default.
            use_eepose_delta (bool): If True, treats the first 6 action dimensions 
                                     as end-effector pose delta (dx, dy, dz, dr, dp, dy).

        Returns:
            Dict[str, Any]: Observation, reward, status, and info from the environment step.
        """
        action_np = np.array(action)
        
        if use_eepose_delta:
            # TODO: Implementation required here to translate eepose delta to joint delta 
            # or to use the environment's internal eepose control mode if available.
            logger.warning("End-effector delta (eepose_delta) mode is requested but not yet implemented.")
            # For now, treat it as joint delta if action is 7D
            assert action_np.shape == (7,), "Action shape must be (7,) for eepose delta/joint delta."
        else:
            assert action_np.shape == (7,), "Action shape must be (7,) for joint delta."
            
        # --- Gripper Action Transformation (from [0, 1] to Libero's [-1, 1]) ---
        # Action is assumed to be normalized joint delta + gripper state [0, 1]
        
        # 1. Transform gripper state [0, 1] (open/close) to [-1, 1]
        gripper_action_libero = 2.0 * action_np[-1] - 1.0
        
        # 2. Map to discrete action (-1: open, 1: close)
        gripper_action_libero = np.sign(gripper_action_libero)
        
        # 3. Libero convention: -1 is open, 1 is close. 
        # Final action: [6 joint deltas, 1 gripper state]
        action_to_env = action_np.copy()
        action_to_env[-1] = gripper_action_libero * -1.0
        # ------------------------------------------------------------------------

        # Step the environment multiple times to reach the desired delta
        for _ in range(self.step_substeps):
            self.observation, self.reward, self.task_status, self.task_info = self.env.step(action_to_env.tolist())
            
        return {
            "observation": self.observation, 
            "reward": self.reward, 
            "task_status": self.task_status, 
            "task_info": self.task_info
        }
