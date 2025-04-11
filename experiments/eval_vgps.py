import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import os
import numpy as np
from absl import app, flags, logging
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders
import wandb
import jax
import jax.numpy as jnp
import imageio
from simpler_env.policies.octo.octo_model import OctoInference
from flax.training import checkpoints
import tensorflow as tf
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"


FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("model_name", "octo-small", "Model name.")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 20, "Number of evaluation episodes.")
flags.DEFINE_boolean("use_vgps", None, "Use V-GPS or not", required=True)
flags.DEFINE_string("vgps_checkpoint", "", "vgps checkpoint name.")
flags.DEFINE_string("vgps_wandb", "", "vgps wandb run name.")
flags.DEFINE_integer("num_samples", 10, "Number of action samples.")
flags.DEFINE_float("action_temp", 1.0, "action softmax temperature. The beta value in the paper.")

def load_vgps_checkpoint(path, wandb_run_name):
    # check path
    assert os.path.exists(path), f"Checkpoint path {path} does not exist"

    """
    You can either specify wandb_run_name to load the exact configuration from Weights & Biases or use the pretrained_checkpoint.yaml file if you are using the provided pre-trained checkpoints.
    """

    if wandb_run_name == "":
        # load from experiments/configs/pretrained_checkpoints.yaml
        import yaml
        with open("experiments/configs/pretrained_checkpoint.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        # load information from wandb
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])
    example_actions = np.zeros((1, 7), dtype=np.float32)
    example_obs = {
        "image": np.zeros((1, 256, 256, 3), dtype=np.uint8)
    }
    example_batch = {
        "observations": example_obs,
        "goals": {
            "language": np.zeros(
                (
                    1,
                    512,
                ),
                dtype=np.float32,
            ),
        },
        "actions": example_actions,
    }

    # create agent from wandb config
    agent = agents[config["agent"]].create(
            rng=jax.random.PRNGKey(0),
            encoder_def=encoder_def,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            **config["agent_kwargs"],
    )
    # load text processor
    critic_text_processor = text_processors[config["text_processor"]]()

    agent = checkpoints.restore_checkpoint(path, agent)

    def get_values(observations, goals, actions):
        print("observation:", observations)
        print("goals: ", goals)
        print("actions: ", actions)
        values = agent.get_q_values(observations, goals, actions)
        return values

    return get_values, critic_text_processor

def main(_):
    logging.set_verbosity(logging.ERROR)
    tf.config.set_visible_devices([], 'GPU')
    print(FLAGS.flag_values_dict())
    
    if 'env' in locals():
        print("Closing existing env")
        env.close()
        del env
    
    env = simpler_env.make(FLAGS.task_name)
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print("Instruction", instruction)

    if "google" in FLAGS.task_name:
        policy_setup = "google_robot"
        STICKY_GRIPPER_NUM_STEPS = 15
    else:
        policy_setup = "widowx_bridge"
        STICKY_GRIPPER_NUM_STEPS = 3

    # @title Select your model and environment
    tf.config.set_visible_devices([], 'GPU')
    model = OctoInference(model_type=FLAGS.model_name, policy_setup=policy_setup, init_rng=FLAGS.seed, sticky_step=STICKY_GRIPPER_NUM_STEPS)
    if FLAGS.use_vgps:
        assert FLAGS.vgps_checkpoint != ""
        get_values, critic_text_processor = load_vgps_checkpoint(FLAGS.vgps_checkpoint, FLAGS.vgps_wandb)
        model.init_vgps(FLAGS.num_samples, get_values, critic_text_processor, action_temp=FLAGS.action_temp, max_episode_steps=env._max_episode_steps)
 

    #@title Run inference
    successes = []
    episode_stats_dict = None
    
    for i in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        model.reset(instruction)
        print(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        # while not (predicted_terminated or truncated):
        while not (success or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            # update image observation
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            timestep += 1

        episode_stats = info.get("episode_stats", {})
        if episode_stats_dict is None:
            episode_stats_dict = {}
            for key, value in episode_stats.items():
                episode_stats_dict[key] = [value]
        else:
            for key, value in episode_stats.items():
                episode_stats_dict[key].append(value)

        if success:
            successes.append(1)
        else:
            successes.append(0)
        print(f"Episode {i}, success: {success}")
        if "consecutive_grasp" in episode_stats_dict:
            print(f"Success Rate: grasp -- {sum(episode_stats_dict['consecutive_grasp'])} / {i+1} | success -- {sum(successes)} / {i + 1}")
        else:
            print(f"Success Rate: success -- {sum(successes)} / {i + 1}")

        # save the video
        base_folder = f"logs/{FLAGS.model_name}_VGPS_{FLAGS.use_vgps}"
        video_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}/{FLAGS.task_name}")
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
        video_path = os.path.join(video_folder, f"{i}_success{success}.mp4")
        imageio.mimsave(video_path, images, fps=10)


    log_message = f"model: {FLAGS.model_name}\nuse_vgps: {FLAGS.use_vgps}\ntask_name: {FLAGS.task_name}\nseed: {FLAGS.seed}\nsuccess_rate: {sum(successes) / len(successes)}"
    if "consecutive_grasp" in episode_stats_dict:
        log_message += f"\nSuccess Rate: grasp -- {sum(episode_stats_dict['consecutive_grasp'])} / {len(successes)} | success -- {sum(successes)} / {len(successes)}"
    else:
        log_message += f"\nSuccess Rate: success -- {sum(successes)} / {len(successes)}"

    print(log_message)
    log_file = os.path.join(video_folder, "log.txt")

    with open(log_file, "w") as f:
        f.write(log_message)
    
    log_file_all = os.path.join(base_folder, f"log_{FLAGS.task_name}.txt")
    with open(log_file_all, "a") as f:
        f.write(f"seed: {FLAGS.seed}, success -- {sum(successes)} / {len(successes)}, success rate: {sum(successes) / len(successes)}\n")
        
if __name__ == "__main__":
    app.run(main)