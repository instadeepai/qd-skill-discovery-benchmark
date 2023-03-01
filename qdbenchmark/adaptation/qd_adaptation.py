import logging
import os
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


@dataclass
class ExperimentConfig:
    repertoire_path: str
    run_config_path: str
    num_init_state: int

    env_name: str
    algorithm_name: str

    adaptation_name: str
    adaptation_idx: int


@hydra.main(config_path="config", config_name="qd")
def run_adaptation_qd(config: ExperimentConfig) -> None:
    # setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(
        logging.INFO
    )  # only print WARN-level logs to screen
    logger = logging.getLogger(f"{__name__}")

    random_key = jax.random.PRNGKey(0)

    run_config = OmegaConf.load(config.run_config_path)

    if config.algorithm_name not in config.run_config_path:
        logger.warning("Algorithm name not in run_config_path")

    assert (
        run_config.env_name == config.env_name
    ), "Must pass a config of the selected environment"

    adaptation_constants = ADAPTATION_CONSTANTS[config.adaptation_name]
    adaptation_constants_env = adaptation_constants[run_config.env_name]

    env_kwargs = {}
    env_kwargs[config.adaptation_name] = jax.tree_map(
        lambda x: x[config.adaptation_idx], adaptation_constants_env
    )

    print(env_kwargs)
    eval_env = environments.create(
        env_name=run_config.env_name,
        batch_size=1,
        episode_length=run_config.episode_length,
        auto_reset=True,
        eval_metrics=True,
        **env_kwargs,
    )

    # Init policy network

    policy_layer_sizes = tuple(run_config.policy_hidden_layer_sizes) + (
        eval_env.action_size,
    )

    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    fake_batch = jnp.zeros(eval_env.observation_size)

    random_key, subkey = jax.random.split(random_key)
    init_variables = policy_network.init(subkey, fake_batch)

    flat, recons_fn = jax.flatten_util.ravel_pytree(init_variables)

    bd_extraction_fn = environments.behavior_descriptor_extractor[run_config.env_name]

    random_key, subkey = jax.random.split(random_key)
    scoring_fn, random_key = create_brax_scoring_fn(
        env=eval_env,
        policy_network=policy_network,
        bd_extraction_fn=bd_extraction_fn,
        random_key=subkey,
        episode_length=run_config.episode_length,
        deterministic=True,
    )

    t0 = time.time()
    genotypes = jnp.load(os.path.join(config.repertoire_path, "genotypes.npy"))
    fitnesses = jnp.load(os.path.join(config.repertoire_path, "fitnesses.npy"))

    policies = jax.tree_map(
        lambda x: x[fitnesses != -jnp.inf], jax.vmap(recons_fn)(genotypes)
    )

    scoring_fn = jax.jit(scoring_fn)
    num_policies = jax.tree_util.tree_leaves(policies)[0].shape[0]

    multi_idx_tuples = [
        (x, y) for x in range(config.num_init_state) for y in range(num_policies)
    ]
    multi_idx = pd.MultiIndex.from_tuples(
        multi_idx_tuples, names=["init_state", "policy_number"]
    )

    df = pd.DataFrame(index=multi_idx, columns=["fitnesses", "descriptors"])
    t0 = time.time()
    logger.info("Filling repertoire with random states")
    for k in range(config.num_init_state):
        random_key, subkey = jax.random.split(random_key)
        eval_fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            policies, random_key
        )
        if (k + 1) % 10 == 0:
            logger.info(
                f"Time to evaluate policies on 10 random \
                    initial states: {time.time()-t0:.2f}s"
            )
            t0 = time.time()

        # Logging
        descriptors = np.array(descriptors)

        descriptors = [str(tuple(descriptor[0])) for descriptor in descriptors]
        df.loc[(k,), "descriptors"] = descriptors
        df.loc[(k,), "fitnesses"] = eval_fitnesses
    df.to_csv("results.csv")
    return


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    run_adaptation_qd()
