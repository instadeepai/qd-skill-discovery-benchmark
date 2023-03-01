import os
from dataclasses import dataclass
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from jax.flatten_util import ravel_pytree
from qdax.core.neuroevolution.networks.diayn_networks import make_diayn_networks
from qdax.utils.metrics import CSVLogger
from qdbenchmark import environments
from qdbenchmark.hierarchical.hierarchical_ppo import train as train_ppo

from brax.training.distribution import NormalTanhDistribution


@dataclass
class ExperimentConfig:
    episode_length: int

    # PPO config
    env_batch_size: int
    num_h_steps: int
    num_steps: int
    normalize_observations: bool
    seed: int
    learning_rate: float
    entropy_cost: float
    discount: float

    # Skills config
    algorithm_name: str
    env_name: str
    policy_path: str

    omit_obs: int
    log_frequency: int


@hydra.main(config_path="config", config_name="halfcheetah_hurdles_sd")
def train(config: ExperimentConfig):
    env_fn = environments.create_fn(env_name=config.env_name)

    dummy_env = environments.create(env_name=config.env_name)
    action_size = dummy_env.action_size
    observation_size = dummy_env.observation_size

    print("Action size: ", action_size)
    print("Observation size: ", observation_size)

    if config.algorithm_name not in config.policy_path:
        print("Algorithm name not in policy path")

    policies_params = jnp.load(config.policy_path)

    dummy_obs = jnp.zeros((1, observation_size - config.omit_obs + 5))
    skill_policy_model, _, _ = make_diayn_networks(
        action_size=action_size, num_skills=5
    )
    dummy_policy_params = skill_policy_model.init(jax.random.PRNGKey(0), dummy_obs)
    _, recons_fn = jax.flatten_util.ravel_pytree(dummy_policy_params)
    skill_policy_params = recons_fn(policies_params)

    def make_skill_inference_fn(skill_policy_model, skill_policy_params, action_size):
        parametric_action_distribution = NormalTanhDistribution(event_size=action_size)
        sample_action_fn = parametric_action_distribution.sample

        def skill_inference_fn(
            obs: jnp.ndarray,
            random_key: jnp.ndarray,
            skill: jnp.ndarray,
            deterministic: bool = False,
        ):
            skill = jax.nn.one_hot(skill, 5)
            obs = obs[..., config.omit_obs :]
            obs = jnp.concatenate([obs, skill], axis=-1)
            actions_logits = skill_policy_model.apply(skill_policy_params, obs)

            if not deterministic:
                random_key, key_sample = jax.random.split(random_key)
                actions = sample_action_fn(actions_logits, key_sample)

            else:
                # The first half of parameters is for mean
                # and the second half for variance
                actions = jax.nn.tanh(
                    actions_logits[..., : actions_logits.shape[-1] // 2]
                )
            return actions

        return skill_inference_fn

    num_skills = 5
    skill_inference_fn = make_skill_inference_fn(
        skill_policy_model=skill_policy_model,
        skill_policy_params=skill_policy_params,
        action_size=action_size,
    )

    csv_logger = CSVLogger("metrics.csv", header=["metric_name", "step", "value"])
    progress_fn = make_progress_fn(csv_logger)
    (inference, params, metrics) = train_ppo(
        environment_fn=env_fn,
        num_timesteps=config.num_steps,
        num_h_steps=config.num_h_steps,
        episode_length=config.episode_length,
        num_skills=num_skills,
        skill_inference_fn=skill_inference_fn,
        action_repeat=1,
        num_envs=config.env_batch_size,
        max_devices_per_host=None,
        num_eval_envs=256,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        discounting=config.discount,
        seed=config.seed,
        unroll_length=20,
        batch_size=512,
        num_minibatches=32,
        num_update_epochs=8,
        log_frequency=config.log_frequency,
        normalize_observations=config.normalize_observations,
        reward_scaling=1.0,
        progress_fn=progress_fn,
    )

    h_policy_params = params[1]
    h_policy_params_flatten, _ = ravel_pytree(h_policy_params)

    save_dir = "policy"

    os.makedirs(save_dir, exist_ok=True)

    jnp.save(
        os.path.join(
            save_dir,
            f"hierarchical-policy-{config.num_steps:.0f}.npy",
        ),
        h_policy_params_flatten,
    )

    normalizer_params = params[0]
    normalizer_params_flatten, _ = ravel_pytree(normalizer_params)

    jnp.save(
        os.path.join(
            save_dir,
            f"normalizer-{config.num_steps:.0f}.npy",
        ),
        normalizer_params_flatten,
    )


def make_progress_fn(csv_logger: CSVLogger):
    def progress(num_steps: Any, metrics: Any) -> Any:
        metric = {
            "metric_name": "eval_mean_return",
            "value": float(metrics["eval/episode_reward"]),
            "step": int(num_steps),
        }

        csv_logger.log(metric)

        metric = {
            "metric_name": "max_fitness",
            "value": float(metrics["max_fitness"]),
            "step": int(num_steps),
        }
        csv_logger.log(metric)

        metric = {
            "metric_name": "average_episode_length",
            "value": float(metrics["eval/avg_episode_length"]),
            "step": int(num_steps),
        }
        csv_logger.log(metric)

        metric = {
            "metric_name": "environment_steps",
            "value": int(num_steps),
            "step": int(num_steps),
        }

        for key in list(metrics.keys()):
            if "_loss" in key:
                metric = {
                    "metric_name": key,
                    "value": float(metrics[key]),
                    "step": int(num_steps),
                }
                csv_logger.log(metric)

        for key in list(metrics.keys()):
            metric = {
                "metric_name": key,
                "value": float(metrics[key]),
                "step": int(num_steps),
            }
            csv_logger.log(metric)

    return progress


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
