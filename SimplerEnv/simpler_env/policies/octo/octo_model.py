from collections import deque
from typing import Optional, Sequence
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
import tensorflow as tf
from transforms3d.euler import euler2axangle
from tqdm import tqdm
from simpler_env.utils.action.action_ensemble import ActionEnsembler
import imageio

def rescale_actions(actions, dataset_id, safety_margin=1e-5, dataset_statistics=None):
    """
    rescale xyz, and rotation actions to be within -1 and 1, then clip actions to stay within safety margin
    """
    if "bridge" in dataset_id:
        ACT_MIN = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.])
        ACT_MAX = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.])
    elif "fractal" in dataset_id:
        ACT_MIN = np.array([-2.02045202, -5.49789953, -2.03166342, -1.56991792, -1.56989217, -1.57041943,  0.        ])
        ACT_MAX = np.array([ 2.99845934, 22.09052849,  2.75075245,  1.57063651,  1.53210866, 1.56915224,  1.        ])
    else:
        assert dataset_statistics is not None
        ACT_MIN = dataset_statistics["min"]
        ACT_MAX = dataset_statistics["max"]
        
    mask = np.array([True, True, True, True, True, True, True])
    actions = np.where(
        mask,
        np.clip((actions - ACT_MIN) / (ACT_MAX - ACT_MIN) * 2 - 1, -1 + safety_margin, 1 - safety_margin),
        np.clip(actions, -1 + safety_margin, 1 - safety_margin),
    )
    return np.array(actions)

def unnormalize_action(action, unnormalization_statistics):
    mask = unnormalization_statistics.get(
        "mask", np.ones_like(unnormalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = np.where(
        mask,
        (action * unnormalization_statistics["std"])
        + unnormalization_statistics["mean"],
        action,
    )
    return action

class OctoInference:
    def __init__(
        self,
        model_type: str = "octo-base",
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        exec_horizon: int = 1,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        sticky_step: int = 1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            dataset_id = "bridge_dataset"
            action_ensemble = True
            action_ensemble_temp = 0.0
        elif policy_setup == "google_robot":
            dataset_id = "fractal20220817_data"
            action_ensemble = True
            action_ensemble_temp = 0.0
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup
        self.sticky_gripper_num_repeat = sticky_step

        if model_type in ["octo-base", "octo-small", "octo-base-1.5", "octo-small-1.5"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_statistics = self.model.dataset_statistics[dataset_id]["action"]
        else:
            raise NotImplementedError(f"{model_type} not supported yet.")

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.num_samples = None
        self.use_vgps = False
        self.dataset_id = dataset_id

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        if self.use_vgps:
            self.task_value = self.model.create_tasks(texts=[task_description for _ in range(self.num_samples)])
            self.pbar = tqdm(total=self.max_episode_steps)
        
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.is_gripper_closed = False

    def get_action(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
        """
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]
        self.rng, key = jax.random.split(self.rng)

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key)
            
        norm_raw_actions = norm_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)
        assert norm_raw_actions.shape == (self.pred_action_horizon, 7)

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None]  # [1, 7]

        raw_actions = unnormalize_action(norm_raw_actions, self.action_statistics)

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        return raw_action


    def init_vgps(self, num_samples, get_values, critic_text_processor, action_temp, max_episode_steps):
        self.num_samples = num_samples
        self.get_values = get_values
        self.critic_text_processor = critic_text_processor
        self.action_temp = action_temp
        self.use_vgps = True
        self.max_episode_steps = max_episode_steps
        self.pbar = tqdm(total=self.max_episode_steps)

    def get_vgps_action(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
        """
        prompt_embed_critic = self.critic_text_processor.encode(self.task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        images, pad_mask = images[None], pad_mask[None]

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]

        pad_key = "timestep_pad_mask" if "-1.5" in self.model_type else "pad_mask"
        input_observation = {"image_primary": images, pad_key: pad_mask}
        norm_raw_actions = self.model.sample_actions(input_observation, self.task, timestep_pad_mask=pad_mask, rng=key, sample_shape=(self.num_samples,))
        
        assert norm_raw_actions.shape == (self.num_samples, 1, self.pred_action_horizon, 7)
        # we first unnormalize the actions with mean/std used for training Octo policy
        critic_actions = unnormalize_action(norm_raw_actions[:, 0, 0, :], self.action_statistics)
        # then normalize it for the critic with min/max used for training the critic
        critic_actions = rescale_actions(critic_actions, dataset_id = self.dataset_id, dataset_statistics = self.action_statistics)
        assert critic_actions.shape == (self.num_samples, 7)
       
        values = self.get_values(
            observations = {"image": np.repeat(images[-1][-1][None], self.num_samples, axis=0)},
            goals = {"language": prompt_embed_critic},
            actions = critic_actions
        )
        assert values.shape == (self.num_samples,)

        self.pbar.set_description(f"Values: max={values.max():.2f}, min={values.min():.2f}, mean={values.mean():.2f}")
        self.pbar.update(1)

        if self.action_temp > 0:
            self.rng, key = jax.random.split(self.rng)
            action_index = jax.random.categorical(key, values / self.action_temp)
            norm_raw_actions = norm_raw_actions[action_index]
        else:
            action_index = np.argmax(values)
            norm_raw_actions = norm_raw_actions[action_index]

        norm_raw_actions = norm_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)
            norm_raw_actions = norm_raw_actions[None]  # [1, 7]

        raw_actions = unnormalize_action(norm_raw_actions, self.action_statistics)
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return raw_action


    def step(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if self.use_vgps:
            raw_action = self.get_vgps_action(image)
        else:
            raw_action = self.get_action(image)

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
        # sticky gripper logic
        elif self.policy_setup == "widowx_bridge":
            
            if (raw_action["open_gripper"].item() < 0.5) != self.is_gripper_closed:
                self.gripper_action_repeat += 1
            else:
                self.gripper_action_repeat = 0

            if self.gripper_action_repeat >= self.sticky_gripper_num_repeat:
                self.is_gripper_closed = not self.is_gripper_closed
                self.gripper_action_repeat = 0


            gripper_action = -1.0 if self.is_gripper_closed else 1.0
            action["gripper"] = (
                np.array([gripper_action])
            )  # binarize gripper action to 1 (open) and -1 (close)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action