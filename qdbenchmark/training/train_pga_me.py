import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hydra.core.config_store import ConfigStore
from qdax import environments
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdax.types import RNGKey
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from qdbenchmark.utils.logging import LoggingConfig
from qdbenchmark.utils.metrics import log_accumulated_metrics


@dataclass
class ExperimentConfig:
    """Configuration for this experiment script"""

    env_name: str
    seed: int
    num_iterations: Optional[int]
    time_limit: Optional[int]

    episode_length: int
    buffer_size: int
    env_batch_size: int
    policy_hidden_layer_sizes: Tuple[int, ...]

    # TD3 params
    batch_size: int
    soft_tau_update: float
    critic_hidden_layer_sizes: Tuple[int, ...]
    critic_learning_rate: float
    greedy_learning_rate: float
    policy_learning_rate: float
    discount: float
    noise_clip: float
    policy_noise: float
    reward_scaling: float

    single_init_state: bool

    num_centroids: int
    num_init_cvt_samples: int
    proportion_mutation_ga: float
    num_critic_training_steps: int
    num_pg_training_steps: int
    iso_sigma: float
    line_sigma: float
    fitness_range: Tuple[float, ...]

    logging: LoggingConfig


@hydra.main(config_path="config/pga_me", config_name="pointmaze")
def train(config: ExperimentConfig) -> None:
    """Launches and monitors training."""

    assert (
        config.logging.save_checkpoints_period % config.logging.log_period == 0
    ), f"{config.logging.save_checkpoints_period} not a multiple/\
       of {config.logging.log_period}"

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    output_dir = os.getcwd()

    _repertoire_dir = os.path.join(output_dir, "checkpoints", "repertoire")
    _repertoire_img_dir = os.path.join(output_dir, "images")
    os.makedirs(_repertoire_img_dir, exist_ok=True)
    os.makedirs(_repertoire_dir, exist_ok=True)

    csv_logger = CSVLogger("metrics.csv", header=["metric_name", "step", "value"])

    # Init environment
    env = environments.create(
        config.env_name, episode_length=config.episode_length, auto_reset=True
    )

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of policies
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.env_batch_size)
    fake_batch = jnp.zeros(shape=(config.env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    assert (
        config.env_name in environments.behavior_descriptor_extractor.keys()
    ), "Please register the bd extractor needed"

    random_key, subkey = jax.random.split(random_key)
    if config.single_init_state:
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=config.env_batch_size, axis=0
        )
    else:
        keys = jax.random.split(subkey, num=config.env_batch_size)

    reward_offset = environments.reward_offset[config.env_name]

    # Prepare the scoring function
    random_key, subkey = jax.random.split(random_key)
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    scoring_fn, random_key = create_brax_scoring_fn(
        env=env,
        policy_network=policy_network,
        bd_extraction_fn=bd_extraction_fn,
        random_key=random_key,
        episode_length=config.episode_length,
        deterministic=False,
    )

    # Define emitter
    variation_fn = partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )
    # create the config
    pgame_config = PGAMEConfig(
        env_batch_size=config.env_batch_size,
        batch_size=config.batch_size,
        soft_tau_update=config.soft_tau_update,
        critic_hidden_layer_size=config.critic_hidden_layer_sizes,
        critic_learning_rate=config.critic_learning_rate,
        greedy_learning_rate=config.greedy_learning_rate,
        policy_learning_rate=config.policy_learning_rate,
        discount=config.discount,
        noise_clip=config.noise_clip,
        policy_noise=config.policy_noise,
        reward_scaling=config.reward_scaling,
        replay_buffer_size=config.buffer_size,
        proportion_mutation_ga=config.proportion_mutation_ga,
        num_critic_training_steps=config.num_critic_training_steps,
        num_pg_training_steps=config.num_pg_training_steps,
    )

    # Initialize the PGAME emitter
    pg_emitter = PGAMEEmitter(
        config=pgame_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
    )

    metrics_fn = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.episode_length,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=pg_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=minval,
        maxval=maxval,
        random_key=random_key,
    )

    duration = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {duration:.2f}s")

    # Prepare scan over map_elites update to perform several iterations at a time
    @jax.jit
    def update_scan_fn(
        carry: Tuple[MapElitesRepertoire, EmitterState, RNGKey], unused: Any
    ) -> Tuple[Tuple[MapElitesRepertoire, EmitterState, RNGKey], Any]:
        # iterate over repertoire
        repertoire, emitter_state, random_key = carry
        (
            repertoire,
            emitter_state,
            metrics,
            random_key,
        ) = map_elites.update(
            repertoire,
            emitter_state,
            random_key,
        )

        return (repertoire, emitter_state, random_key), metrics

    # Init algorithm
    logger.warning("--- Algorithm initialisation ---")
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")

    current_step = 0
    total_training_time = 0.0
    time_duration = 0.0
    num_iterations = 0
    # Main loop

    if config.time_limit is not None:
        num_loop_iterations = jnp.inf
    elif config.num_iterations is not None:
        num_loop_iterations = config.num_iterations // config.logging.log_period + 1
    else:
        raise NotImplementedError(
            "Either a time limit or a number of iterations must\
be specified"
        )

    current_step = 0
    total_training_time = 0.0
    time_duration = 0.0
    num_iterations = 0

    # Main loop
    iteration = 0
    while iteration < num_loop_iterations:
        if iteration > 0:
            start_time = time.time()
            (
                repertoire,
                emitter_state,
                random_key,
            ), metrics = jax.lax.scan(
                update_scan_fn,
                (repertoire, emitter_state, random_key),
                (),
                length=config.logging.log_period,
            )

            # Subsample the number of logged metrics as it can take a while to
            # log every single point
            metrics = jax.tree_map(
                lambda x: x[:: config.logging.metrics_subsample], metrics
            )
            time_duration = time.time() - start_time
            total_training_time += time_duration
            num_iterations = iteration * config.logging.log_period

        else:
            metrics = metrics_fn(repertoire)
            metrics = jax.tree_map(lambda x: x[None], metrics)

        logger.warning("-" * 60)
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_loop_iterations-1} ---"
        )
        logger.warning(
            f"--- Iteration number : {iteration * config.logging.log_period}"
            f" out of {(num_loop_iterations-1) * config.logging.log_period}---"
        )

        last_step = current_step
        current_step = (
            iteration
            * config.env_batch_size
            * config.episode_length
            * config.logging.log_period
        )
        logger.warning(f"--- Current number of steps in env: {current_step:.2f}")
        logger.warning(f"--- Time duration (batch of iteration): {time_duration:.2f}")
        logger.warning(
            f"--- Time duration (Sum of iterations): {total_training_time:.2f}"
        )

        if env.behavior_descriptor_length == 2:
            fig, ax = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=minval,
                maxval=maxval,
                repertoire_descriptors=repertoire.descriptors,
                ax=None,
                vmin=config.fitness_range[0],
                vmax=config.fitness_range[1],
            )
            fig.savefig(os.path.join(_repertoire_img_dir, f"repertoire_{current_step}"))
            plt.close(fig)

        if num_iterations % config.logging.save_checkpoints_period == 0:
            repertoire.save(path=os.path.join(_repertoire_dir, ""))

        # Logging part
        metric = {
            "metric_name": "environment_steps",
            "value": int(current_step),
            "step": int(current_step),
        }
        csv_logger.log(metric)
        metric = {
            "metric_name": "iteration_duration",
            "value": float(time_duration),
            "step": int(current_step),
        }
        csv_logger.log(metric)
        log_accumulated_metrics(
            metrics=metrics,
            metric_logger=csv_logger,
            current_step=current_step,
            last_step=last_step,
        )

        iteration += 1
        if config.time_limit is not None:
            if time.time() - init_time > config.time_limit:
                logger.warning("Exiting due to time limit")
                break
    duration = time.time() - init_time

    logger.warning("--- Final metrics ---")
    logger.warning(f"Duration: {duration:.2f}s")
    logger.warning(f"Training duration: {total_training_time:.2f}s")
    logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
