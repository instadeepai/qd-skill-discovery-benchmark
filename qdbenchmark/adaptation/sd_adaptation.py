import functools
import logging
import time
from dataclasses import dataclass

import hydra
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdbenchmark import environments
from qdbenchmark.adaptation.constants import ADAPTATION_CONSTANTS
from qdbenchmark.adaptation.utils import play_step_fn_skill_discovery


@dataclass
class ExperimentConfig:
    policy_path: str
    run_config_path: str
    num_init_state: int

    env_name: str
    algorithm_name: str

    adaptation_name: str
    adaptation_idx: int


@hydra.main(config_path="config", config_name="sd")
def run_adaptation_skill_discovery(config: ExperimentConfig) -> None:
    # setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(
        logging.INFO
    )  # only print WARN-level logs to screen
    logger = logging.getLogger(f"{__name__}")

    random_key = jax.random.PRNGKey(0)

    run_config = OmegaConf.load(config.run_config_path)

    adaptation_constants = ADAPTATION_CONSTANTS[config.adaptation_name]
    adaptation_constants_env = adaptation_constants[run_config.env_name]

    env_kwargs = {}
    env_kwargs[config.adaptation_name] = jax.tree_map(
        lambda x: x[config.adaptation_idx], adaptation_constants_env
    )

    if config.algorithm_name not in config.run_config_path:
        logger.warning("Algorithm name not in run_config_path")

    assert (
        run_config.env_name == config.env_name
    ), "Must pass a config of the selected environment"

    print(env_kwargs)
    eval_env = environments.create(
        env_name=run_config.env_name,
        batch_size=1,
        episode_length=run_config.episode_length,
        auto_reset=True,
        eval_metrics=True,
        **env_kwargs,
    )

    multi_idx_tuples = [
        (x, y)
        for x in range(run_config.num_skills)
        for y in range(config.num_init_state)
    ]
    multi_idx = pd.MultiIndex.from_tuples(
        multi_idx_tuples, names=["policy_number", "init_state"]
    )

    df = pd.DataFrame(index=multi_idx, columns=["fitnesses", "descriptors"])

    policy_layer_sizes = tuple(run_config.hidden_layer_sizes) + (
        2 * eval_env.action_size,
    )
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    fake_batch = jnp.zeros(eval_env.observation_size + run_config.num_skills)
    random_key, subkey = jax.random.split(random_key)
    init_variables = policy_network.init(subkey, fake_batch)

    flat, recons_fn = jax.flatten_util.ravel_pytree(init_variables)
    flat_policy = jnp.load(config.policy_path)

    num_evaluations = run_config.num_skills * config.num_init_state

    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]

    play_step_fn = functools.partial(
        play_step_fn_skill_discovery,
        env=eval_env,
        policy_network=policy_network,
        deterministic=True,
    )

    random_key, subkey = jax.random.split(random_key)

    scoring_fn, random_key = create_brax_scoring_fn(
        env=eval_env,
        policy_network=policy_network,
        bd_extraction_fn=bd_extraction_fn,
        random_key=subkey,
        episode_length=run_config.episode_length,
        deterministic=True,
        play_step_fn=play_step_fn,
    )

    recons_fn(flat_policy)

    # Reconstruct policy
    repeated_policy = jnp.repeat(flat_policy[None], num_evaluations, axis=0)
    print(repeated_policy.shape)

    policies = jax.vmap(recons_fn)(repeated_policy)
    t0 = time.time()

    skills = jnp.eye(run_config.num_skills)
    repeated_skills = jnp.repeat(skills, config.num_init_state, axis=0)

    policies_params = {"policy_params": policies, "skills": repeated_skills}

    random_key, subkey = jax.random.split(random_key)

    eval_fitnesses, descriptors, extra_scores, random_key = scoring_fn(
        policies_params, random_key
    )

    logger.info(
        f"Time to evaluate {run_config.num_skills} policies on {config.num_init_state}\
          random initial states: {time.time()-t0:.2f}s"
    )
    # Logging
    descriptors = np.array(descriptors)

    descriptors = [str(tuple(descriptor[0])) for descriptor in descriptors]
    df.loc[:, "descriptors"] = descriptors
    df.loc[:, "fitnesses"] = eval_fitnesses
    df.to_csv("results.csv")

    return


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    run_adaptation_skill_discovery()
