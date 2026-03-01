# Pi0 wrapper implementation.

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import ModelWrapper  # Shared base class with OpenVLAWrapper

logger = logging.getLogger("wrappers.pi0")


def _import_openpi_modules():
    """Import OpenPI modules lazily so registry import stays robust."""
    try:
        from openpi.policies import policy_config as policy_config_mod
        from openpi.shared import download as download_mod
        from openpi.training import config as config_mod
    except ImportError as exc:
        raise RuntimeError(
            "Pi0Wrapper requires OpenPI dependencies. Install openpi before using model 'pi0'."
        ) from exc
    return config_mod, policy_config_mod, download_mod


class Pi0Wrapper(ModelWrapper):
    """
    Wrapper for OpenPI pi0 / pi05 policies.

    - load(): create policy with config_name + checkpoint_dir following OpenPI examples
    - predict_action(): build observation from PIL.Image + text instruction,
      call policy.infer(observation)["actions"], and return action_chunk
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        # Name passed to openpi.training.config.get_config, e.g. "pi0_aloha_sim" / "pi05_droid"
        config_name: str = "pi05_droid",
        # Default prompt used when instruction is missing
        default_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_path:
                - If it starts with 'gs://...', use openpi.shared.download.maybe_download
                - If it is local, use it directly as checkpoint_dir
            config_name:
                OpenPI training config name such as "pi05_droid" or "pi0_aloha_sim"
            default_prompt:
                Fallback prompt when predict_action gets no instruction or an empty one
        """
        super().__init__(model_path, **kwargs)
        self.config_name = config_name
        self.default_prompt = default_prompt

        self._policy = None  # type: ignore[assignment]

    # -------------------- Model loading -------------------- #
    def load(self) -> None:
        config_mod, policy_config_mod, download_mod = _import_openpi_modules()
        cfg = config_mod.get_config(self.config_name)

        # model_path can be gs:// or a local path
        mp = str(self.model_path)
        if mp.startswith("gs://"):
            logger.info("[Pi0Wrapper] Downloading checkpoint from %s", mp)
            checkpoint_dir = download_mod.maybe_download(mp)
        else:
            checkpoint_dir = mp

        logger.info(
            "[Pi0Wrapper] Creating trained policy (config=%s, checkpoint_dir=%s)",
            self.config_name,
            checkpoint_dir,
        )
        self._policy = policy_config_mod.create_trained_policy(cfg, checkpoint_dir)

        logger.info("[Pi0Wrapper] Policy created. Metadata: %s", getattr(self._policy, "metadata", {}))

    # -------------------- Build observation -------------------- #
    def _make_dummy_observation(
        self,
        image: Image.Image,
        instruction: str | None,
    ) -> dict:
        """
        For tests and compatibility with legacy image+instruction-only callers.
        In production, construct a full observation upstream.
        """
        img_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prompt = (instruction or "").strip() or (self.default_prompt or "")

        joint_pos = np.zeros((7,), dtype=np.float32)
        gripper_pos = np.zeros((1,), dtype=np.float32)

        return {
            "observation/exterior_image_1_left": img_np,
            "observation/wrist_image_left": img_np,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_pos,
            "prompt": prompt,
        }

    # -------------------- Inference API -------------------- #
    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        **kwargs: Any,
    ):
        """
        Unified inference interface:

        - Input: one PIL.Image + text instruction
        - Internal: build observation dict and call self._policy.infer(...)
        - Output: ["actions"] from the OpenPI policy (action_chunk)

        Extra **kwargs (for example accel_method / accel_config) are ignored
        for compatibility with higher-level generic call signatures.
        """
        if self._policy is None:
            raise RuntimeError("Pi0Wrapper.load() must be called before predict_action().")

        obs = self._make_dummy_observation(image, instruction)
        result = self._policy.infer(obs)

        if "actions" not in result:
            raise KeyError("openpi policy did not return 'actions' in infer() output")

        actions = result["actions"]
        return actions
