import os
from dataclasses import dataclass
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger
from qdbenchmark import environments
from qdbenchmark.hierarchical.hierarchical_ppo import train as train_ppo


@dataclass
class ExperimentConfig:
    episode_length: int

    env_batch_size: int
    num_h_steps: int
    num_steps: int
    normalize_observations: bool
    seed: int

    num_sub_centroids: int
    learning_rate: float
    entropy_cost: float

    algorithm_name: str
    env_name: str
    repertoire_path: str
    discount: float

    omit_obs: int
    log_frequency: int


@hydra.main(config_path="config/", config_name="halfcheetah_hurdles_qd")
def train(config: ExperimentConfig):
    env_fn = environments.create_fn(env_name=config.env_name)

    dummy_env = environments.create(env_name=config.env_name)
    action_size = dummy_env.action_size
    observation_size = dummy_env.observation_size

    print("Action size: ", action_size)
    print("Observation size: ", observation_size)

    if config.algorithm_name not in config.repertoire_path:
        print("Algorithm name not in repertoire path")

    sub_centroids = None

    genotypes = jnp.load(os.path.join(config.repertoire_path, "genotypes.npy"))
    fitnesses = jnp.load(os.path.join(config.repertoire_path, "fitnesses.npy"))
    centroids = jnp.load(os.path.join(config.repertoire_path, "centroids.npy"))
    descriptors = jnp.load(os.path.join(config.repertoire_path, "descriptors.npy"))

    policy_layer_sizes = (
        256,
        256,
        action_size,
    )
    print(policy_layer_sizes)

    policy_network_repertoire = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    dummy_batch = jnp.zeros(
        observation_size - config.omit_obs,
    )

    init_variables_repertoire = policy_network_repertoire.init(
        jax.random.PRNGKey(0), dummy_batch
    )

    _, recons_fn_repertoire = jax.flatten_util.ravel_pytree(init_variables_repertoire)
    genotypes = jax.vmap(recons_fn_repertoire)(genotypes)

    if config.num_sub_centroids > 0:
        num_sub_centroids = config.num_sub_centroids
        sub_centroids = compute_cvt_centroids(
            num_descriptors=centroids.shape[-1],
            num_init_cvt_samples=10000,
            num_centroids=num_sub_centroids,
            minval=centroids.min(axis=0),
            maxval=centroids.max(axis=0),
        )

        sub_repertoire = MapElitesRepertoire(
            centroids=sub_centroids,
            genotypes=jax.tree_map(
                lambda x: jnp.zeros_like(x)[:num_sub_centroids], genotypes
            ),
            fitnesses=jnp.ones(num_sub_centroids) * -jnp.inf,
            descriptors=jnp.zeros_like(descriptors[:num_sub_centroids]),
        )

        sub_repertoire = sub_repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
        )

        fitnesses = sub_repertoire.fitnesses
        genotypes = sub_repertoire.genotypes

    policy_params_repertoire = jax.tree_map(
        lambda x: x[fitnesses != -jnp.inf], genotypes
    )

    num_skills = jnp.sum(fitnesses != -jnp.inf)

    def make_skill_inference_fn(skill_policy_model, skill_policy_params, action_size):
        def skill_inference_fn(
            obs: jnp.ndarray,
            random_key: jnp.ndarray,
            skill: jnp.ndarray,
            deterministic: bool = False,
        ):
            obs = obs[..., config.omit_obs :]
            policy_params = jax.tree_map(lambda x: x[skill], skill_policy_params)
            actions = jax.vmap(skill_policy_model.apply)(policy_params, obs)

            return actions

        return skill_inference_fn

    skill_inference_fn = make_skill_inference_fn(
        skill_policy_model=policy_network_repertoire,
        skill_policy_params=policy_params_repertoire,
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

    if sub_centroids is not None:
        save_dir = centroids
        os.makedirs(save_dir, exist_ok=True)
        jnp.save(
            os.path.join(
                save_dir,
                f"centroids_{config.num_sub_centroids}.npy",
            ),
            sub_centroids,
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
